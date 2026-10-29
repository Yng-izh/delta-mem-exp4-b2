# latent_memory.py — 潜层记忆（Latent Memory）：压缩器 / 桥接器 / 槽位载荷库 / 潜层读出
#
# 目标：让"记忆"以**潜向量**而不是明文证据块进入冻结的生成模型。
#   历史文本 --encode--> token 级隐状态 (h0, h18) --LatentCompressor--> Z ∈ R^{m×d}
#   Z 存进 LatentSlotBank（按稳定槽位 ID） --LatentBridge--> soft tokens --> generate
#
# 为什么单独一个模块（而不是塞进 v9_slotpool.py）：
#   1. v9_slotpool 是**寻址层**（非重叠切分 / IDF-MaxSim / L18 稠密 / 时间约束 / CR 双阈值
#      覆写 / 多跳图搜索），这些逻辑不需要为了 Latent Memory 重写；
#   2. 神经网络部分要能单独训练、单独保存 checkpoint、单独做消融；
#   3. 读出接口必须**在签名上**就禁止退回明文 RAG（见 generate_latent_answer）。
#
# 三个独立组件：
#   LatentCompressor  h0/h18 + padding mask -> Z [B, m, d]（低秩融合 + m 个可学习 query +
#                     cross-attention + 输出投影回 hidden size；内部宽度 inner_dim=512）
#   LatentBridge      Z -> soft tokens（LayerNorm -> Linear -> SiLU -> Linear）
#   LatentSlotBank    稳定槽位 ID -> 潜载荷 [k, m, d]（k = 该槽位被分成的块数）
#
# ⚠️ 没有训练的压缩器/桥接器，即使张量维度完全正确，冻结 Qwen 也没有理由理解这些连续向量。
#    训练脚本见 train_latent_memory.py（阶段 A 单事实读出 + 阶段 B Teacher-Student 蒸馏）。
#
# ⚠️ 本模块**不**声称"已经能恢复事实"。能力边界：
#    - 检索/寻址仍然由 v9 的明文索引完成（encode 时看得到原文，读出时看不到）；
#    - 读出侧只接受潜载荷，因此 answer_in_context 这类"明文是否进 prompt"的诊断在
#      latent 模式下不适用（见 run_ar_v7.py 的 readout.latent_slots / 独立报告）；
#    - 压缩是有损的：m 个槽位向量能否承载实体/数字/日期，必须靠训练 + 评测回答，
#      不能靠"维度对上了"来推断。

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_SCHEMA = 1


# ----------------------------- 环境参数 -----------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None else str(v).strip()


# 初始配置（数值是起点，不是结论：训练时长/显存需要实测 profiling）
LATENT_M               = _env_int("LATENT_M", 4)                # 每个单元的潜槽位数 m
LATENT_LAYER           = _env_int("LATENT_LAYER", 18)           # 取第几层隐状态（0 层恒为 embedding）
LATENT_INNER_DIM       = _env_int("LATENT_INNER_DIM", 512)      # 压缩器内部宽度
LATENT_N_HEADS         = _env_int("LATENT_N_HEADS", 8)          # cross-attention 头数
LATENT_BOTTLENECK      = _env_int("LATENT_BOTTLENECK", 512)     # Bridge 瓶颈宽度
LATENT_MAX_UNIT_TOKENS = _env_int("LATENT_MAX_UNIT_TOKENS", 64) # 单块最多 token 数（超出要再分块，不截断）
LATENT_READ_BUDGET     = _env_int("LATENT_READ_BUDGET", 32)     # 读出预算（软 token 数，= K × m）
# 软 token 的每向量范数目标：必须对齐真实 token embedding 的尺度，否则软 token 会变成
# attention sink 把真实 token 淹没（实测未约束时到 96~159，而 Qwen3-4B embedding 是 1.05）。
LATENT_TARGET_NORM     = _env_float("LATENT_TARGET_NORM", 1.0)
LATENT_CKPT            = _env_str("LATENT_CKPT", "")            # 压缩器+桥接器 checkpoint
# FST-Mem：每个 (实体,属性) 事实键保留多少条**历史**版本（当前状态不计入）。
# =0 表示"硬覆写，历史全丢"，是做对照的合法取值，不是错误。
LATENT_HIST_MAX        = _env_int("LATENT_HIST_MAX", 4)
# >0 时只编码前 N 个槽位（smoke / 大池调试用）；0 = 不限制。
LATENT_BANK_MAX = int(os.environ.get("LATENT_BANK_MAX", "0"))
LATENT_DTYPE           = _env_str("LATENT_DTYPE", "bf16").lower()

_DTYPES = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
           "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
           "fp32": torch.float32, "float32": torch.float32, "float": torch.float32}


def latent_dtype() -> torch.dtype:
    """载荷存储精度（LATENT_DTYPE）。未知取值直接报错，不静默退回 fp32。"""
    if LATENT_DTYPE not in _DTYPES:
        raise ValueError(f"LATENT_DTYPE={LATENT_DTYPE!r} 非法；可选 bf16 | fp16 | fp32")
    return _DTYPES[LATENT_DTYPE]


def latent_config() -> Dict[str, Any]:
    """进 manifest / checkpoint 的配置快照（只含影响结果的项）。"""
    return {
        "schema": LATENT_SCHEMA,
        "n_slots": int(LATENT_M),
        "layer": int(LATENT_LAYER),
        "inner_dim": int(LATENT_INNER_DIM),
        "num_heads": int(LATENT_N_HEADS),
        "bottleneck": int(LATENT_BOTTLENECK),
        "max_unit_tokens": int(LATENT_MAX_UNIT_TOKENS),
        "read_budget": int(LATENT_READ_BUDGET),
        "bank_max": int(LATENT_BANK_MAX),
        "hist_max": int(LATENT_HIST_MAX),
        "dtype": LATENT_DTYPE,
    }


def latent_budget_slots(token_budget: Optional[int] = None,
                        tokens_per_slot: Optional[int] = None) -> int:
    """潜层读出预算 -> 最多能读多少个槽位（K = budget // m，向下取整）。

    明文预算（AR_V9_CTX_TOKEN_BUDGET=2500 token）和潜层预算**不是一回事**：
    潜层预算的单位是软 token 数，K 个槽位占 K×m 个位置。
    """
    b = LATENT_READ_BUDGET if token_budget is None else int(token_budget)
    m = LATENT_M if tokens_per_slot is None else int(tokens_per_slot)
    if b <= 0 or m <= 0:
        return 0
    return max(1, b // m)


def describe() -> str:
    """一行配置摘要（日志/dry-run 用）。"""
    cfg = latent_config()
    k = latent_budget_slots()
    return (f"latent: m={cfg['n_slots']} layer={cfg['layer']} "
            f"inner={cfg['inner_dim']} heads={cfg['num_heads']} "
            f"bottleneck={cfg['bottleneck']} max_unit_tokens={cfg['max_unit_tokens']} "
            f"read_budget={cfg['read_budget']} -> K<={k} slots dtype={cfg['dtype']} "
            f"ckpt={LATENT_CKPT or '<unset>'}")


# ----------------------------- 精度 / 设备小工具 -----------------------------
def _model_device(model: Any, default: str = "cpu") -> torch.device:
    try:
        p = next(model.parameters())
        return p.device
    except Exception:
        return torch.device(default)


def _model_hidden_size(model: Any) -> int:
    cfg = getattr(model, "config", None)
    hs = getattr(cfg, "hidden_size", None) if cfg is not None else None
    if hs:
        return int(hs)
    try:
        return int(model.get_input_embeddings().weight.shape[1])
    except Exception as e:                       # pragma: no cover - 模型形态异常
        raise ValueError(f"无法确定模型 hidden_size: {e}")


@contextmanager
def delta_write_disabled(model: Any):
    """编码期禁止 Delta 写入：记忆编码不得改动在线状态（否则同一 context 内前后不一致）。

    **必须恢复调用前的状态**，不能"用完置 False 就完事"（旧实现在 finally 里
    `fn(model, False)`，于是第一次进这个上下文之后写入被**永久关闭**：后续 chunk
    一个字节都写不进在线状态，跑分照样出数、只是全错，日志里毫无痕迹）。

    这里直接改模块属性 `write_enabled` 而不调 `set_write_enabled()`：后者在
    `enabled=True` 时会顺手清掉 `read_context_mask`、在 `False` 时会清掉
    `write_message_ids/write_sentence_ids` —— 恢复快照时把这些旁路状态一起清掉，
    同样是静默错误。
    """
    mods = []
    if model is not None:
        try:
            from deltamem.core.delta import iter_delta_mem_modules as _iter
            mods = [m for _, m in _iter(model)]
        except Exception:
            mods = []
    if not mods:
        yield False
        return
    prev = [(m, bool(getattr(m, "write_enabled", True))) for m in mods]
    try:
        for m, _p in prev:
            m.write_enabled = False
        yield True
    finally:
        for m, p in prev:
            m.write_enabled = p


def _safe_torch_load(path: str) -> Any:
    """torch.load 兼容层：2.6+ 默认 weights_only=True，旧格式需要显式关闭。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:                            # 老 torch 没有 weights_only 参数
        return torch.load(path, map_location="cpu")
    except Exception:
        # 含非张量对象的 checkpoint（例如带 numpy/自定义类）：显式关闭 weights_only
        return torch.load(path, map_location="cpu", weights_only=False)


# ----------------------------- 1) 压缩器 -----------------------------
class LatentCompressor(nn.Module):
    """把历史文本压成 m 个潜向量：Z ∈ R^{B×m×d}。

    结构（最小可用版，刻意不用 d×d 多头注意力 + 多个大 MLP）：
        x = input_proj(h0) + hidden_proj(h18)          # 低秩特征融合（内部宽度 inner_dim）
        q = m 个可学习 query
        z = cross_attn(q, x, x, key_padding_mask=~mask)
        Z = LayerNorm(output_proj(z))                  # 投影回模型 hidden size

    空输入 / 全 padding 行：PyTorch 的 MHA 在"整行被 mask"时 softmax 分母为 0 -> NaN。
    这里把空行的第 0 个位置临时解除 mask 保证数值有限，随后把该行输出**整体置零**，
    并在 info 里记 n_empty_rows（下游据此决定是否跳过，而不是拿到 NaN 静默传播）。
    """

    def __init__(self, hidden_size: int, n_slots: int = LATENT_M,
                 inner_dim: int = LATENT_INNER_DIM, num_heads: int = LATENT_N_HEADS):
        super().__init__()
        if int(hidden_size) <= 0:
            raise ValueError(f"hidden_size 必须为正，得到 {hidden_size}")
        if int(n_slots) <= 0:
            raise ValueError(f"n_slots 必须为正，得到 {n_slots}")
        if int(inner_dim) <= 0:
            raise ValueError(f"inner_dim 必须为正，得到 {inner_dim}")
        if int(num_heads) <= 0 or int(inner_dim) % int(num_heads) != 0:
            raise ValueError(f"inner_dim={inner_dim} 必须能被 num_heads={num_heads} 整除")
        self.hidden_size = int(hidden_size)
        self.n_slots = int(n_slots)
        self.inner_dim = int(inner_dim)
        self.num_heads = int(num_heads)
        self.input_proj = nn.Linear(self.hidden_size, self.inner_dim)
        self.hidden_proj = nn.Linear(self.hidden_size, self.inner_dim)
        self.queries = nn.Parameter(torch.randn(self.n_slots, self.inner_dim) * 0.02)
        self.attn = nn.MultiheadAttention(self.inner_dim, num_heads=self.num_heads,
                                          batch_first=True)
        self.output_proj = nn.Linear(self.inner_dim, self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, h0: torch.Tensor, h18: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z, _info = self.forward_with_info(h0, h18, attention_mask)
        return z

    def forward_with_info(self, h0: torch.Tensor, h18: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None
                          ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        h0 = _as_3d(h0, "h0")
        h18 = _as_3d(h18, "h18")
        if tuple(h0.shape) != tuple(h18.shape):
            raise ValueError(f"h0/h18 形状不一致：{tuple(h0.shape)} vs {tuple(h18.shape)}")
        B, L, d = h0.shape
        if d != self.hidden_size:
            raise ValueError(f"h0/h18 的 hidden={d} 与压缩器 hidden_size={self.hidden_size} 不一致")
        if L <= 0:
            raise ValueError("h0/h18 的序列长度为 0（空输入必须由调用方给出至少 1 个 padding 位）")

        if attention_mask is None:
            mask = torch.ones((B, L), dtype=torch.bool, device=h0.device)
        else:
            mask = attention_mask
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if tuple(mask.shape) != (B, L):
                raise ValueError(f"attention_mask 形状 {tuple(mask.shape)} != {(B, L)}")
            mask = mask.bool().to(h0.device)

        empty = ~mask.any(dim=1)                       # 整行都是 padding
        safe_mask = mask.clone()
        if bool(empty.any()):
            safe_mask[empty, 0] = True                 # 避免全 mask 行 -> NaN

        p = self.queries
        # 掩码必须跟到**参数设备**，不能跟 h0.device：
        # encode_latent_units 为了省显存把 h0/h18 留在 CPU（返回 [N,T,d] 的 CPU 张量），
        # 而 h0/h18 在这里才被搬到 p.device。若 mask 跟着 h0 留在 CPU，
        # GPU 上 SDPA 会报 "attn_bias is on cpu, different from other tensors on cuda"。
        # CPU 单测永远发现不了这个问题（两者都是 CPU）。
        dev_p = p.device
        safe_mask = safe_mask.to(dev_p)
        empty = empty.to(dev_p)
        x = (self.input_proj(h0.to(device=dev_p, dtype=p.dtype))
             + self.hidden_proj(h18.to(device=dev_p, dtype=p.dtype)))
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        z, _w = self.attn(q, x, x, key_padding_mask=~safe_mask, need_weights=False)
        z = self.norm(self.output_proj(z))
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        if bool(empty.any()):
            z = z.masked_fill(empty.view(B, 1, 1), 0.0)
        info = {"n_rows": int(B), "seq_len": int(L),
                "n_tokens": [int(v) for v in mask.sum(dim=1).tolist()],
                "n_empty_rows": int(empty.sum().item())}
        return z, info


def _as_3d(t: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.dim() != 3:
        raise ValueError(f"{name} 必须是 [B,L,d] 或 [L,d]，得到 {tuple(t.shape)}")
    return t


# ----------------------------- 2) 桥接器 -----------------------------
class LatentBridge(nn.Module):
    """Z -> soft tokens：LayerNorm -> Linear -> SiLU -> Linear -> **范数对齐**（输入层注入，不改 Qwen attention）。

    为什么要显式对齐范数（这是实测踩到的坑，不是洁癖）
    ---------------------------------------------------
    soft token 是直接拼进 `inputs_embeds` 的，模型对它的唯一"合法感"来自它和真实 token
    embedding 处在同一个尺度上。Qwen3-4B 真实 token embedding 每向量范数约 **1.05**；
    而上面这条未约束的 Linear 输出范数实测到了 **96~159**（差两个量级）。

    后果不是"略微不准"，而是**软 token 变成 attention sink**：它抢走几乎全部注意力权重，
    真实 token 被淹没。实测反事实很干净 —— 训练 200 条、8 epoch、loss 从 12.6 降到 2.8，
    但同一批 200 条只答对 1~3 条；把 m 从 4 加到 16（软 token 16→64 个）**毫无改善**；
    而把记忆原文的 embedding 直接当软 token 注入（oracle）则 4/4 全对。

    所以这里把输出方向归一化、范数交给一个受限的可学习标量（初值=真实 embedding 范数，
    夹在 [0.25, 4] 倍以内，防止训练把尺度重新推到爆炸区）。
    """

    NORM_MIN = 0.25
    NORM_MAX = 4.0

    def __init__(self, hidden_size: int, bottleneck: int = LATENT_BOTTLENECK,
                 target_norm: float = LATENT_TARGET_NORM):
        super().__init__()
        if int(hidden_size) <= 0 or int(bottleneck) <= 0:
            raise ValueError(f"hidden_size/bottleneck 必须为正：{hidden_size}/{bottleneck}")
        if not (float(target_norm) > 0):
            raise ValueError(f"target_norm 必须为正，得到 {target_norm}")
        self.hidden_size = int(hidden_size)
        self.bottleneck = int(bottleneck)
        self.net = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.bottleneck),
            nn.SiLU(),
            nn.Linear(self.bottleneck, self.hidden_size),
        )
        # 可学习但受限的尺度；初值取真实 token embedding 的每向量范数
        self.out_norm = nn.Parameter(torch.tensor(float(target_norm)))

    def scale(self) -> torch.Tensor:
        """当前生效的范数（已夹在 [NORM_MIN, NORM_MAX] 内）。"""
        return self.out_norm.clamp(self.NORM_MIN, self.NORM_MAX)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = _as_3d(z, "z")
        if z.shape[-1] != self.hidden_size:
            raise ValueError(f"z 的最后一维 {z.shape[-1]} != hidden_size={self.hidden_size}")
        p = next(self.parameters())
        out = self.net(z.to(device=p.device, dtype=p.dtype))
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        # 方向保留、范数对齐到真实 embedding 尺度（见类文档里的实测数据）
        n = out.norm(dim=-1, keepdim=True)
        unit = out / n.clamp_min(1e-6)
        return unit * self.scale().to(device=out.device, dtype=out.dtype)


# ----------------------------- 2.5) 潜空间内容寻址（Router） -----------------------------
LATENT_D_KEY = _env_int("LATENT_D_KEY", 256)


def stack_payloads(payloads: Sequence[torch.Tensor]) -> torch.Tensor:
    """把 encode() 返回的 per-slot payload 列表堆成 [K, m, d]。

    每个 payload 形状 [k, m, d]（k = 该单元被切成几块）。块轴取均值：一个槽位被切成
    多块时，它的"寻址特征"是这几块的平均，而不是其中某一块 —— 否则同一个槽位在
    "是否超长"两种情况下会得到不同的键，路由结果随无关因素抖动。
    """
    if not payloads:
        raise ValueError("stack_payloads 收到空列表")
    zs = [p if p.dim() == 3 else p.unsqueeze(0) for p in payloads]
    m_set = {int(z.shape[1]) for z in zs}
    if len(m_set) != 1:
        raise ValueError(f"payload 的 m 不一致：{sorted(m_set)}，无法堆叠成 [K,m,d]")
    return torch.stack([z.mean(dim=0) for z in zs], dim=0)


class LatentRouter(nn.Module):
    """无位置偏置的潜空间内容寻址：query 与槽位键内积 -> 选槽 -> 只送出那一个槽。

    要解决的问题（实测证据，干净进程重载 ckpt、训练同款 prompt、Delta 写入关闭）：
        同一条事实 "Taron Velstrik works at Velcor."
            单槽输入            -> 15/16
            4 槽 target_only    -> 15/16   （内容完全相同，只是外面套了 4 槽的壳）
            4 槽 full（目标首位）->  1/16
            4 槽 target_end     ->  0/16
    即：读出完全由"槽位排在软 token 序列的第几位"决定，与内容无关。把 K 个槽位的
    K*m 个软 token 裸拼进 inputs_embeds，Qwen 只能按位置去注意，于是把"第 1 个槽"
    当成答案源。槽数变、顺序变，结论就崩。

    Router 把"选哪个槽"从"软 token 排第几"里解耦：
        s_k = <q, k_in_k> / tau              # 集合打分，与 K 和顺序无关
        Z_read = softmax(s)·Z   (训练，可导)
        Z_read = Z_{argmax_k s_k} (推理，只送 1 个槽 = m 个软 token)
    这样无论外面给 1 个还是 4 个槽、目标在第 1 位还是最后一位，bridge 看到的
    永远是它最熟悉的"单槽 m 个软 token"，位置先验被彻底去掉。

    两个键分工（对应单跳 / 多跳）：
        k_in  ：主语 + 关系  -> 与问题匹配（本类的主用途）
        k_out ：客体值       -> 多跳时用 q' = norm(q + k_out) 接力找下一跳
    """

    def __init__(self, d_model: int, n_slots: int = LATENT_M, d_key: int = LATENT_D_KEY,
                 init_temp: float = 0.07):
        super().__init__()
        if int(d_model) <= 0 or int(n_slots) <= 0 or int(d_key) <= 0:
            raise ValueError(f"d_model/n_slots/d_key 必须为正：{d_model}/{n_slots}/{d_key}")
        self.d_model = int(d_model)
        self.n_slots = int(n_slots)          # m：单个槽位的潜向量个数
        self.d_key = int(d_key)
        self.slot_flat = self.n_slots * self.d_model
        # 问题特征 = L0 平均嵌入 ‖ L18 平均隐状态（拼接，共 2*d_model）
        self.q_net = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_key), nn.SiLU(),
            nn.Linear(self.d_key, self.d_key))
        self.k_in_net = nn.Sequential(
            nn.Linear(self.slot_flat, self.d_key), nn.SiLU(),
            nn.Linear(self.d_key, self.d_key))
        self.k_out_net = nn.Sequential(
            nn.Linear(self.slot_flat, self.d_key), nn.SiLU(),
            nn.Linear(self.d_key, self.d_key))
        self.log_temp = nn.Parameter(torch.tensor(float(math.log(init_temp))))

    def encode_query(self, q_feat: torch.Tensor) -> torch.Tensor:
        """q_feat: [B, 2*d_model]（L0 mean ‖ L18 mean）-> 归一化查询向量 [B, d_key]。"""
        q_feat = torch.as_tensor(q_feat)
        if q_feat.dim() == 1:
            q_feat = q_feat.unsqueeze(0)
        if int(q_feat.shape[-1]) != self.d_model * 2:
            raise ValueError(f"q_feat 最后一维 {q_feat.shape[-1]} != {self.d_model * 2}")
        p = next(self.parameters())
        return torch.nn.functional.normalize(
            self.q_net(q_feat.to(device=p.device, dtype=p.dtype)), dim=-1)

    def encode_keys(self, slot_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """slot_z: [B, K, m, d] -> (k_in [B,K,d_key], k_out [B,K,d_key])，均已归一化。"""
        slot_z = torch.as_tensor(slot_z)
        if slot_z.dim() == 3:
            slot_z = slot_z.unsqueeze(0)
        if slot_z.dim() != 4:
            raise ValueError(f"slot_z 必须是 [B,K,m,d] 或 [K,m,d]，得到 {tuple(slot_z.shape)}")
        if int(slot_z.shape[-2]) != self.n_slots or int(slot_z.shape[-1]) != self.d_model:
            raise ValueError(f"slot_z 尾部应是 [m={self.n_slots}, d={self.d_model}]，"
                             f"得到 {tuple(slot_z.shape[-2:])}")
        p = next(self.parameters())
        flat = slot_z.to(device=p.device, dtype=p.dtype).flatten(-2)   # [B,K,m*d]
        k_in = torch.nn.functional.normalize(self.k_in_net(flat), dim=-1)
        k_out = torch.nn.functional.normalize(self.k_out_net(flat), dim=-1)
        return k_in, k_out

    def scores(self, q_vec: torch.Tensor, k_in: torch.Tensor) -> torch.Tensor:
        """集合打分 s_k = <q, k_in_k> / tau，形状 [B, K]（与 K、与顺序无关）。"""
        tau = self.log_temp.exp().clamp_min(1e-4)
        return torch.einsum("bd,bkd->bk", q_vec, k_in.to(q_vec.dtype)) / tau

    def forward(self, q_feat: torch.Tensor, slot_z: torch.Tensor,
                hard: bool = False, target_idx: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (z_read [B,m,d], 路由 logits [B,K], k_in, k_out)。

        **前向一律硬选 argmax**（训练与推理同一条路径），差别只在梯度：

        hard=False（训练）：直通估计 z_read = z_hard + (z_soft - z_soft.detach())
            —— 前向数值与推理**逐位相同**，梯度仍能从 z_soft 回传到 logits。
        hard=True（推理）：纯 z_hard，不走任何聚合。

        为什么不能用软门控聚合做训练前向（原先的错）：
            实测 4 槽 K=4、τ=0.07 下 ||z_soft|| = 61.9 而 ||z_hard|| = 100.2，
            ||z_soft - z_hard|| = 54.6% of ||z_hard||，权重 w=[0.18,0.05,0.23,0.54]
            并不接近 one-hot。也就是说 bridge 在训练时看到的是"4 个事实按权重
            混合后的平均向量"，推理时看到的是"一条干净的事实"——两个分布。
            bridge 的 norm-align 只抹平范数，抹不平方向，train/infer 依旧错位。
            更糟的是训练初期 logits≈0、w 接近均匀，z_read≈全体事实的均值，
            QA 损失根本无法下降，梯度信号全是噪声。
            硬选前向 + STE 梯度同时满足两件事：分布一致、路由仍可学
            （况且路由本身还有 loss_addr 的显式 CE 监督）。
        """
        q_vec = self.encode_query(q_feat)
        k_in, k_out = self.encode_keys(slot_z)
        logits = self.scores(q_vec, k_in)
        z = slot_z if slot_z.dim() == 4 else slot_z.unsqueeze(0)
        z = z.to(device=q_vec.device, dtype=q_vec.dtype)
        idx = logits.argmax(dim=-1)                                   # [B]
        z_hard = z[torch.arange(z.shape[0], device=z.device), idx]     # [B,m,d]
        if hard or not self.training:
            return z_hard, logits, k_in, k_out
        w = torch.softmax(logits, dim=-1)                             # [B,K]
        # 载荷必须 detach：软期望的唯一用途是把**路由分数**的梯度送回去。
        # 不 detach 时梯度会经由软期望那一路回传到**全部候选槽位**的载荷上，
        # 未被选中事实的压缩器表示也被当前答案的梯度拖走（FST 侧早已这么修，
        # Router 侧漏了 —— 同一个病，只是从数值挪到了梯度）。
        z_soft = torch.einsum("bk,bkmd->bmd", w, z.detach())
        # 前向取 z_hard，梯度走 z_soft（STE）
        z_read = z_hard + (z_soft - z_soft.detach())
        return z_read, logits, k_in, k_out

    def forward_hops(self, q_feat: torch.Tensor, slot_z: torch.Tensor,
                     n_hops: int = 2, hard: bool = False) -> Dict[str, Any]:
        """多跳前向：第 1 跳选槽 -> q' = norm(q + k_out[选中槽]) -> 第 2 跳再选槽。

        修的是一个**架构级缺陷**，不是调参问题：
            旧训练路径只调 `router(q_feat, slot_z)`，拿回的 `k_out` 直接被丢掉，
            只有第 1 跳的 `loss_addr` 进图；推理侧 `chain()` 却用
            `q' = norm(q + k_out[idx])` 接力找下一跳 —— 于是 `k_out_net`
            在正常训练里**没有任何梯度**，第二跳实际由一个随机投影在寻址。
            HotpotQA 的现象正对得上：第一跳寻址在涨（read_acc 35%→50%）、
            答案准确率不动。单跳监督永远学不会"第二跳该读哪条事实"。

        返回每一跳的 logits（训练侧逐跳算 CE）与载荷（推理侧用同一套拼接）。
        """
        q_vec = self.encode_query(q_feat)
        k_in, k_out = self.encode_keys(slot_z)
        z = slot_z if slot_z.dim() == 4 else slot_z.unsqueeze(0)
        z = z.to(device=q_vec.device, dtype=q_vec.dtype)
        n = max(1, min(int(n_hops), int(z.shape[1])))
        # 用 Python set 记录已选槽，**不能**用张量就地写 used[idx]=True：
        # 那个 bool 张量被上一跳的 masked_fill 引用进了计算图，就地改它会触发
        #   RuntimeError: one of the variables needed for gradient computation
        #   has been modified by an inplace operation: [CPUBoolType [1, 8]] version 2
        # （这正是 10k 正训一 backward 就会炸的地方，冒烟先撞出来了）。
        used_idx: set = set()
        z_list: List[torch.Tensor] = []
        logits_list: List[torch.Tensor] = []
        picks: List[int] = []
        for _ in range(n):
            logits = self.scores(q_vec, k_in)                          # [1,K]
            if used_idx:
                mask = torch.zeros_like(logits, dtype=torch.bool)
                mask[0, sorted(used_idx)] = True
                logits = logits.masked_fill(mask, float("-inf"))
            idx = logits.argmax(dim=-1)                                # [1]
            used_idx.add(int(idx[0].item()))
            z_hard = z[0, idx]                                         # [1,m,d]
            if not (hard or not self.training):
                w = torch.softmax(logits, dim=-1)
                # einsum "bk,bkmd->bmd" 出来已经是 [1,m,d]（b 维保留），
                # 旧代码又 unsqueeze(0) 成 [1,1,m,d]；与 [1,m,d] 的 z_hard 相加时
                # 广播成四维，一路传到 bridge(z) 才炸：
                #   ValueError: z 必须是 [B,L,d] 或 [L,d]，得到 (1, 1, 4, 64)
                # 单跳路径（同文件 497 行）没有这次 unsqueeze —— 两处必须一致。
                z_soft = torch.einsum("bk,bkmd->bmd", w, z.detach())
                z_hard = z_hard + (z_soft - z_soft.detach())
            z_list.append(z_hard)
            logits_list.append(logits)
            picks.append(int(idx[0].item()))
            # 查询更新**留在计算图里**：这是 k_out_net 唯一能拿到梯度的通路
            q_vec = torch.nn.functional.normalize(
                q_vec + k_out[:, int(idx[0].item())], dim=-1)
        return {"z_list": z_list, "logits_list": logits_list, "picks": picks,
                "k_in": k_in, "k_out": k_out}

    @torch.no_grad()
    def chain(self, q_feat: torch.Tensor, slot_z: torch.Tensor, hops: int = 2
              ) -> List[int]:
        """多跳推理：直接复用训练用的 forward_hops（同一套打分与查询更新规则）。

        旧实现自己维护了一份循环，与训练侧根本不是同一条路径 ——
        推理用 k_out、训练不监督 k_out，两份代码让这个错误一直看不出来。
        """
        return self.forward_hops(q_feat, slot_z, n_hops=hops, hard=True)["picks"]


def compose_routed_soft(model: Any, tokenizer: Any, bridge: "LatentBridge",
                        z_list: Sequence[torch.Tensor],
                        dev: Any = None) -> torch.Tensor:
    """若干槽位载荷 -> 一段 soft 序列（每槽过一次 bridge，槽间插原生 "\\n"）。

    **训练与推理必须调同一个函数**：旧实现里训练是 `bridge(z_read)` 一段、
    推理是 `bridge(z1) + "\\n" + bridge(z2)`，两跳样本的 train/infer 输入分布不同，
    而且训练侧第 2 跳的载荷根本没进图。
    """
    if not z_list:
        raise ValueError("compose_routed_soft 收到空载荷列表")
    dev = _model_device(model) if dev is None else dev
    emb = model.get_input_embeddings()
    nl = emb(_tokenize_no_special(tokenizer, "\n", dev))
    pieces = []
    for j, zj in enumerate(z_list):
        if not torch.is_tensor(zj):
            raise TypeError(f"compose_routed_soft 第 {j} 个载荷不是张量：{type(zj)}")
        if zj.dim() == 4:
            # 单跳/多跳都要交 [B,m,d]。四维几乎一定是"索引时保留了多余维度"，
            # 静默 squeeze 会掩盖上游 bug，这里直接报出来。
            raise ValueError(f"compose_routed_soft 第 {j} 个载荷是四维 {tuple(zj.shape)}，"
                             f"应为 [B,m,d]")
        if j:
            pieces.append(nl)
        s = bridge(zj)
        if s.dim() == 2:
            s = s.unsqueeze(0)
        pieces.append(s.reshape(1, -1, s.shape[-1]))
    return torch.cat(pieces, dim=1)


def extract_query_feature(model: Any, tokenizer: Any, question: str,
                          system_prompt: str = "", layer: int = LATENT_LAYER,
                          device: Any = None) -> torch.Tensor:
    """问题 -> [1, 2d] 查询特征（L0 平均嵌入 ‖ L18 平均隐状态）。

    为什么两层都要：L0 是词形（实体名靠它区分），L18 是语境化语义（关系靠它区分）。
    只用 L0 时同关系不同主语几乎不可分；只用 L18 时未见实体的词形会被磨平。
    """
    if model is None or tokenizer is None:
        raise ValueError("extract_query_feature 需要 model 与 tokenizer")
    dev = torch.device(device) if device is not None else _model_device(model)
    full, n_sys_chars = build_chat_parts(tokenizer, system_prompt, question)
    # 只用问题本身（system 是常量，放进键里只会给所有样本加同一个偏移）
    q_text = full[n_sys_chars:] if n_sys_chars else full
    ids = _tokenize_no_special(tokenizer, q_text or question, dev)
    with torch.no_grad(), delta_write_disabled(model):
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    output_hidden_states=True, use_cache=False)
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise RuntimeError("模型没有返回 hidden_states（需要 output_hidden_states=True）")
    if int(layer) >= len(hs):
        raise ValueError(f"layer={layer} 超出 hidden_states 层数 {len(hs)}")
    h0 = hs[0].mean(dim=1)                                  # [1, d]
    hl = hs[int(layer)].mean(dim=1)                         # [1, d]
    p = model.get_input_embeddings().weight
    return torch.cat([h0, hl], dim=-1).to(dtype=p.dtype)


# ---------------- 2.6) FST-Mem：事实键（实体—属性级稳定寻址） ----------------
# FST-Mem = Factorized State-Trajectory Memory（状态—轨迹解耦的抗干扰潜层记忆）。
#
# 与 2.5 的 LatentRouter 的分工（不是替代关系）：
#   LatentRouter    : 在"整条事实"粒度上做内容寻址 —— 解决"K 个槽里选哪一个"。
#   LatentFactKey   : 在"(实体,属性)"粒度上给出**稳定身份键** —— 解决"张三的雇主"
#                     和"张三的居住地"是两件不同的事，更新其中一件不能动另一件。
# 两者都只是"选哪条事实"，都不解决"同一条事实的多个时间版本怎么共存"。
FST_KEY_SEP = "||"


def _norm_key_part(v: Any) -> str:
    """实体/属性名的规范化：小写、去标点、压空白、剔除分隔符本身。"""
    t = str(v if v is not None else "").strip().lower()
    t = t.replace(FST_KEY_SEP, " ")
    for ch in ".,!?;:\"'()[]{}":
        t = t.replace(ch, " ")
    return " ".join(t.split())


def fact_key(entity: Any, relation: Any) -> str:
    """(实体, 属性) -> 稳定事实键 'zhang san||employer'。

    键必须在"值变了"之后**保持不变** —— 这正是更新局部性的前提：新值写回同一个
    键，旧值下沉进该键的历史，其他键一个字节都不动。
    """
    e = _norm_key_part(entity)
    r = _norm_key_part(relation)
    if not e or not r:
        raise ValueError(f"事实键要求 entity 与 relation 都非空：{entity!r}/{relation!r}")
    return f"{e}{FST_KEY_SEP}{r}"


def split_fact_key(key: Any) -> Tuple[str, str]:
    k = str(key)
    e, _, r = k.partition(FST_KEY_SEP)
    if not e or not r:
        raise ValueError(f"事实键格式非法（应为 'entity{FST_KEY_SEP}relation'）：{k!r}")
    return e, r


def extract_text_feature(model: Any, tokenizer: Any, text: str,
                         layer: int = LATENT_LAYER, device: Any = None) -> torch.Tensor:
    """任意短文本 -> [1, d_model] 平均隐状态（事实键的实体/属性、时间条件共用）。

    与 extract_query_feature 的区别：那个要 2d（L0‖L18，为了同时保住词形和语境），
    这里要 d（要拼进 k_{e,r} 的联合映射里，实体名/属性名/年份都是短串，单层够用）。
    空串返回零向量（不是报错）："没有时间条件"是合法输入。
    """
    if model is None or tokenizer is None:
        raise ValueError("extract_text_feature 需要 model 与 tokenizer")
    dev = torch.device(device) if device is not None else _model_device(model)
    p = model.get_input_embeddings().weight
    t = str(text if text is not None else "").strip()
    if not t:
        return torch.zeros((1, int(p.shape[-1])), dtype=p.dtype, device=dev)
    ids = _tokenize_no_special(tokenizer, t, dev)
    with torch.no_grad(), delta_write_disabled(model):
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    output_hidden_states=True, use_cache=False)
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise RuntimeError("模型没有返回 hidden_states（需要 output_hidden_states=True）")
    if int(layer) >= len(hs):
        raise ValueError(f"layer={layer} 超出 hidden_states 层数 {len(hs)}")
    return hs[int(layer)].mean(dim=1).to(dtype=p.dtype)


class LatentFactKeyEncoder(nn.Module):
    """k_{e,r} = f(entity, relation)：实体向量与属性向量**联合**构造的稳定寻址键。

    为什么不能只用整条事实的键（LatentRouter 的 k_in 就是整条事实）：
        "Zhang works at Alibaba." 与 "Zhang lives in Beijing." 两条事实的整句键
        都会随**值**变化而变化。于是"张三换雇主"这件事，在整句键里表现为
        "一个键消失了、另一个键出现了"，而不是"同一个键的值变了" —— 系统没有
        任何"这是同一个属性"的先验，更新局域性无从谈起。
        拆成 (实体, 属性) 之后，键与值无关，值变了键不动。

    键由三部分拼成（留交互项）：
        h_e = e_net(实体特征), h_r = r_net(属性特征)
        k   = normalize(out([h_e, h_r, h_e ⊙ h_r]))
    交互项不是装饰：有些属性名本身很泛（"employer"/"residence" 都只是名词），
    真正区分它们的是与实体的搭配；纯拼接无法表达"这个实体在这个属性上"。
    """

    def __init__(self, d_model: int, d_key: int = LATENT_D_KEY, init_temp: float = 0.07):
        super().__init__()
        if int(d_model) <= 0 or int(d_key) <= 0:
            raise ValueError(f"d_model/d_key 必须为正：{d_model}/{d_key}")
        self.d_model = int(d_model)
        self.d_key = int(d_key)
        self.e_net = nn.Sequential(nn.Linear(self.d_model, self.d_key), nn.SiLU(),
                                   nn.Linear(self.d_key, self.d_key))
        self.r_net = nn.Sequential(nn.Linear(self.d_model, self.d_key), nn.SiLU(),
                                   nn.Linear(self.d_key, self.d_key))
        self.out = nn.Sequential(nn.Linear(self.d_key * 3, self.d_key), nn.SiLU(),
                                 nn.Linear(self.d_key, self.d_key))
        self.q_net = nn.Sequential(nn.Linear(self.d_model * 2, self.d_key), nn.SiLU(),
                                   nn.Linear(self.d_key, self.d_key))
        self.log_temp = nn.Parameter(torch.tensor(float(math.log(init_temp))))

    def _as_b(self, t: torch.Tensor, name: str) -> torch.Tensor:
        t = torch.as_tensor(t)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.dim() != 2 or int(t.shape[-1]) != self.d_model:
            raise ValueError(f"{name} 必须是 [B,{self.d_model}]，得到 {tuple(t.shape)}")
        p = next(self.parameters())
        return t.to(device=p.device, dtype=p.dtype)

    def encode_keys(self, e_feat: torch.Tensor, r_feat: torch.Tensor) -> torch.Tensor:
        """[G,d] × [G,d] -> 归一化键 [G, d_key]。"""
        e = self._as_b(e_feat, "e_feat")
        r = self._as_b(r_feat, "r_feat")
        if int(e.shape[0]) != int(r.shape[0]):
            raise ValueError(f"实体/属性特征行数不一致：{int(e.shape[0])} vs {int(r.shape[0])}")
        he = self.e_net(e)
        hr = self.r_net(r)
        joint = torch.cat([he, hr, he * hr], dim=-1)
        return torch.nn.functional.normalize(self.out(joint), dim=-1)

    def encode_query(self, q_feat: torch.Tensor) -> torch.Tensor:
        """[B, 2d] -> 归一化查询 [B, d_key]（与 LatentRouter 同一套特征口径）。"""
        q = torch.as_tensor(q_feat)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        if int(q.shape[-1]) != self.d_model * 2:
            raise ValueError(f"q_feat 最后一维应是 {self.d_model * 2}，得到 {int(q.shape[-1])}")
        p = next(self.parameters())
        return torch.nn.functional.normalize(
            self.q_net(q.to(device=p.device, dtype=p.dtype)), dim=-1)

    def scores(self, q_vec: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """集合打分 [B,G]：与 G 和排列顺序无关（纯内积 + 温度）。"""
        tau = self.log_temp.exp().clamp_min(1e-4)
        return torch.einsum("bd,gd->bg", q_vec, keys.to(q_vec.dtype)) / tau


class TemporalStateSelector(nn.Module):
    """机制三的第一步之外那一步：在**同一个事实键内部**按时间条件选版本。

    两层寻址必须分开，这是 FST-Mem 与"给槽位加个时间戳"的差别：
        若把时间塞进同一个打分式（s = <q, k(e,r,t)>），那么"哪个属性"和
        "哪个时间版本"会被同一个 softmax 竞争 —— 提问"2023 年张三在哪工作"时，
        2023 年的**居住地**和 2025 年的**雇主**会跟正确答案一起抢概率质量。
        事实级先选(实体,属性)，版本级再选时间，两个竞争被隔离开。

    版本键 = 版本的潜载荷 [m,d] 与时间特征拼接后过 MLP：
        k_ver = normalize(ver_net([z_v.flatten(), t_v]))
    查询键 = normalize(q_net([q_feat, t_q]))，t_q 是问题里的时间条件（没有就零向量）。
    """

    def __init__(self, d_model: int, n_slots: int = LATENT_M,
                 d_key: int = LATENT_D_KEY, init_temp: float = 0.07):
        super().__init__()
        if int(d_model) <= 0 or int(n_slots) <= 0 or int(d_key) <= 0:
            raise ValueError(f"d_model/n_slots/d_key 必须为正：{d_model}/{n_slots}/{d_key}")
        self.d_model = int(d_model)
        self.n_slots = int(n_slots)
        self.d_key = int(d_key)
        self.slot_flat = self.n_slots * self.d_model
        self.ver_net = nn.Sequential(nn.Linear(self.slot_flat + self.d_model, self.d_key),
                                     nn.SiLU(), nn.Linear(self.d_key, self.d_key))
        self.q_net = nn.Sequential(nn.Linear(self.d_model * 3, self.d_key), nn.SiLU(),
                                   nn.Linear(self.d_key, self.d_key))
        self.log_temp = nn.Parameter(torch.tensor(float(math.log(init_temp))))

    def encode_versions(self, ver_z: torch.Tensor, ver_t_feat: torch.Tensor) -> torch.Tensor:
        """ver_z [...,V,m,d] + ver_t_feat [...,V,d] -> 归一化版本键 [...,V,d_key]。"""
        p = next(self.parameters())
        z = torch.as_tensor(ver_z)
        if z.dim() != 4:
            raise ValueError(f"ver_z 必须是 [G,V,m,d]，得到 {tuple(z.shape)}")
        if int(z.shape[-2]) != self.n_slots or int(z.shape[-1]) != self.d_model:
            raise ValueError(f"ver_z 尾部应是 [m={self.n_slots}, d={self.d_model}]，"
                             f"得到 {tuple(z.shape[-2:])}")
        t = torch.as_tensor(ver_t_feat)
        if t.dim() == 2:
            t = t.unsqueeze(0)
        if t.dim() != 3 or int(t.shape[-1]) != self.d_model \
                or int(t.shape[1]) != int(z.shape[1]):
            raise ValueError(f"ver_t_feat 必须是 [G,V,{self.d_model}]，得到 {tuple(t.shape)}")
        z = z.to(device=p.device, dtype=p.dtype)
        t = t.to(device=p.device, dtype=p.dtype)
        flat = z.flatten(-2)                                        # [G,V,m*d]
        return torch.nn.functional.normalize(
            self.ver_net(torch.cat([flat, t], dim=-1)), dim=-1)

    def encode_query(self, q_feat: torch.Tensor, tq_feat: Optional[torch.Tensor] = None
                     ) -> torch.Tensor:
        """([B,2d], [B,d]) -> 归一化时间条件查询 [B, d_key]（tq 缺省为零向量）。"""
        q = torch.as_tensor(q_feat)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        if int(q.shape[-1]) != self.d_model * 2:
            raise ValueError(f"q_feat 最后一维应是 {self.d_model * 2}，得到 {int(q.shape[-1])}")
        if tq_feat is None:
            tq = torch.zeros((int(q.shape[0]), self.d_model), dtype=q.dtype, device=q.device)
        else:
            tq = torch.as_tensor(tq_feat)
            if tq.dim() == 1:
                tq = tq.unsqueeze(0)
            if int(tq.shape[-1]) != self.d_model:
                raise ValueError(f"tq_feat 最后一维应是 {self.d_model}，得到 {int(tq.shape[-1])}")
            if int(tq.shape[0]) == 1 and int(q.shape[0]) > 1:
                tq = tq.expand(int(q.shape[0]), -1)
        p = next(self.parameters())
        x = torch.cat([q, tq.to(device=q.device, dtype=q.dtype)], dim=-1)
        return torch.nn.functional.normalize(
            self.q_net(x.to(device=p.device, dtype=p.dtype)), dim=-1)

    def scores(self, q_vec: torch.Tensor, k_ver: torch.Tensor) -> torch.Tensor:
        """[1,d_key] × [V,d_key] -> [1,V]。"""
        tau = self.log_temp.exp().clamp_min(1e-4)
        return torch.einsum("bd,vd->bv", q_vec, k_ver.to(q_vec.dtype)) / tau


def _payload_md(p: torch.Tensor) -> torch.Tensor:
    """任意 per-slot 载荷（[m,d] 或 [k,m,d]）-> 规范 [m,d]。

    走 stack_payloads 而不是自己 cat/reshape：多块槽位必须按既有口径取块均值，
    否则同一个槽位在"是否超长"两种情况下会得到两个不同的版本表示（静默错位）。
    """
    z = torch.as_tensor(p)
    if z.dim() == 2:
        return z
    return stack_payloads([z])[0]


def pad_version_stack(per_group: Sequence[Sequence[torch.Tensor]],
                      dtype: Optional[torch.dtype] = None,
                      device: Any = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """每组事实的版本载荷 -> (ver_z [G,V,m,d] 零填充, ver_mask [G,V] bool)。

    G = 事实组数，V = 组内最大版本数。填充位用 mask 屏蔽（打分里 -1e4），
    绝不靠"填充向量恰好得分低"——随机初始化时那只是运气。
    """
    groups = [list(g) for g in per_group]
    if not groups:
        raise ValueError("pad_version_stack 收到空分组")
    mds = [[_payload_md(p) for p in g] for g in groups]
    for g in mds:
        if not g:
            raise ValueError("存在没有任何版本的实体—属性组（事实键建了但没写入值）")
    ms = {int(x.shape[0]) for g in mds for x in g}
    ds = {int(x.shape[-1]) for g in mds for x in g}
    if len(ms) != 1 or len(ds) != 1:
        raise ValueError(f"版本载荷的 [m,d] 不一致：m={sorted(ms)} d={sorted(ds)}")
    m, d = ms.pop(), ds.pop()
    G = len(mds)
    V = max(len(g) for g in mds)
    z = torch.zeros((G, V, m, d), dtype=dtype or mds[0][0].dtype, device=device)
    mask = torch.zeros((G, V), dtype=torch.bool, device=device)
    for gi, g in enumerate(mds):
        for vi, x in enumerate(g):
            z[gi, vi] = x.to(device=device, dtype=z.dtype)
            mask[gi, vi] = True
    return z, mask


class FstRouter(nn.Module):
    """FST-Mem 的两级寻址：**先选事实键，再选时间版本**。

        s_grp[g] = <q, k_{e_g,r_g}> / τ              # 事实级（与时间无关）
        s_ver[v] = <q_t, k_ver_v> / τ                # 版本级（组内，带时间条件）
        Z_read   = z[g*, v*]                         # 只送这一个版本的 m 个软 token

    前向一律 argmax（训练/推理同一条路径），梯度走 STE —— 与 LatentRouter 同一条
    纪律：软加权做前向会让 bridge 在训练时看到"多版本混合向量"、推理时看到
    "一条干净事实"，两个分布，指标必然虚高后崩（2.5 里实测过，见 audit_soft_hard.py）。
    """

    def __init__(self, d_model: int, n_slots: int = LATENT_M,
                 d_key: int = LATENT_D_KEY, init_temp: float = 0.07):
        super().__init__()
        self.d_model = int(d_model)
        self.n_slots = int(n_slots)
        self.d_key = int(d_key)
        self.key = LatentFactKeyEncoder(self.d_model, self.d_key, init_temp)
        self.state = TemporalStateSelector(self.d_model, self.n_slots,
                                           self.d_key, init_temp)

    def pick_version(self, g: int, lg_ver_all: torch.Tensor, ver_mask: Any = None,
                     ver_policy: str = "learned", ver_times: Any = None,
                     query_time: Any = None) -> int:
        """组内版本读出：**显式时间规则优先**，learned 只作消融对照。

        为什么不再默认交给分类器：状态库已经把"当前版本"这件事结构性地写清楚了
        （`list_versions` 最后一个就是当前状态），CR 的判定又是"最新值算对"。
        旧实现把这个已知事实重新交给一个 softmax 重新学，实测 ver 0.525 反而输给
        "恒取最新"的 0.885 —— 那不是训练不足，是把结构信息丢了。
        只有问题里带明确时间条件、且该条件能与某个版本时间对上时，才需要按时间选。
        """
        row = lg_ver_all[int(g)]
        ok = None
        if ver_mask is not None:
            ok = torch.as_tensor(ver_mask).to(row.device)[int(g)].bool()
        policy = str(ver_policy or "current")
        if policy == "learned":
            r = row if ok is None else row.masked_fill(~ok, float("-inf"))
            return int(r.argmax().item())
        avail = ([i for i, b in enumerate(ok.tolist()) if b] if ok is not None
                 else list(range(int(row.numel()))))
        if not avail:
            return 0
        q = _norm_time(query_time)
        if policy == "time" and q and ver_times is not None:
            ts = list(ver_times[int(g)])
            best, best_sc = None, 0
            for i in avail:                       # 旧 -> 新，同分时更新的胜出
                if i >= len(ts):
                    continue
                sc = _time_match_score(q, ts[i])
                if sc >= best_sc and sc > 0:
                    best, best_sc = i, sc
            if best is not None:
                return int(best)
        # 没有时间条件 / 时间对不上 -> 当前状态（组内最后写入且仍在库里的版本）
        return int(avail[-1])

    def forward(self, q_feat: torch.Tensor, e_feat: torch.Tensor, r_feat: torch.Tensor,
                ver_z: torch.Tensor, ver_t_feat: torch.Tensor,
                tq_feat: Optional[torch.Tensor] = None,
                ver_mask: Optional[torch.Tensor] = None, hard: bool = False,
                k_groups: int = 1, ver_policy: str = "learned",
                ver_times: Any = None, query_time: Any = None
                ) -> Dict[str, Any]:
        """返回 dict：z_read [1,m,d]、z_reads List[[1,m,d]]、logits_group [1,G]、
        logits_ver [1,V]、groups / versions（本次读出选了哪些键与版本）。

        ver_z 是一次性传入的 [G,V,m,d]（G 个事实组、每组最多 V 个版本），
        logits_ver     : **被选中那一组**内部的版本打分（跨组版本不互相竞争）。
        logits_ver_all : [G,V] 每一组各自的版本打分（掩码位 = -1e4）。

        为什么要同时给 all：版本级监督必须打在**目标组**那一行上。
        只拿 logits_ver（被选中组）去算 CE，一旦路由初期选错组，监督就会
        指向另一组的版本布局（甚至指向被掩码的填充位），loss 直接爆到 1e4 量级 ——
        实测踩过：loss_ver 从 0 跳到 9990。这不是数值问题，是监督对象错了。

        k_groups>1：一次读出多个键（多跳/多事实题必须能同时取多条证据）。
        旧接口硬编码"只送 1 个版本的 m 个软 token"，即使键选对了，证据也不足以
        回答两跳题 —— 这不是训练问题，是读出带宽不够。
        """
        q_vec = self.key.encode_query(q_feat)
        k_grp = self.key.encode_keys(e_feat, r_feat)                # [G,dk]
        lg_grp = self.key.scores(q_vec, k_grp)                      # [1,G]
        n_g = max(1, min(int(k_groups), int(lg_grp.shape[-1])))
        groups: List[int] = []
        lg_tmp = lg_grp.clone()
        for _ in range(n_g):                       # 贪心取互不重复的键
            g = int(lg_tmp[0].argmax().item())
            groups.append(g)
            lg_tmp[0, g] = float("-inf")

        k_ver = self.state.encode_versions(ver_z, ver_t_feat)       # [G,V,dk]
        qt_vec = self.state.encode_query(q_feat, tq_feat)           # [1,dk]
        lg_ver_all = self.state.scores(qt_vec, k_ver.reshape(
            -1, k_ver.shape[-1]))                                   # [G*V] -> 下面还原
        lg_ver_all = lg_ver_all.reshape(int(k_ver.shape[0]), int(k_ver.shape[1]))
        if ver_mask is not None:
            mask = torch.as_tensor(ver_mask).to(device=lg_ver_all.device)
            if mask.dim() != 2 or int(mask.shape[0]) != int(ver_z.shape[0]):
                raise ValueError(f"ver_mask 必须是 [G,V]，得到 {tuple(mask.shape)}")
            lg_ver_all = lg_ver_all.masked_fill(~mask, -1e4)
        versions = [self.pick_version(g, lg_ver_all, ver_mask, ver_policy,
                                      ver_times, query_time) for g in groups]
        # 读出用**被选中组**的版本打分；监督用 tgt_group 那一行（见 logits_ver_all）
        g0 = int(groups[0])
        lg_all = lg_ver_all[g0:g0 + 1]                              # [1,V]

        z_all = torch.as_tensor(ver_z)
        z_reads = [_payload_md(z_all[g, v]).unsqueeze(0) for g, v in zip(groups, versions)]
        out = {"z_read": z_reads[0], "z_reads": z_reads,
               "logits_group": lg_grp, "logits_ver": lg_all,
               "logits_ver_all": lg_ver_all,
               "group": g0, "version": int(versions[0]),
               "groups": groups, "versions": [int(x) for x in versions],
               "ver_policy": str(ver_policy or "current"), "hard": True}
        if hard or not self.training:
            return out
        # STE：前向 z_hard，梯度走版本级的软加权期望。
        # 两个容易写错、且**不会报错只会悄悄更差**的点：
        #  a) 载荷必须 detach：否则梯度会同时从 z_hard（one-hot 那一路）和软期望那一路
        #     回传到全部 G×V 条载荷上，压缩器被迫同时满足所有事实——与 2.5 节实测过的
        #     "软加权前向"同一个病，只是从数值挪到了梯度。载荷只经由 z_hard 拿梯度。
        #  b) 版本分布必须**按组各算一次**：把选中组的 wv 套到所有组上，等于假设
        #     "别的组里第 v 个版本也重要"，向量本身就不再是任何一致的期望量。
        # 组级不再做软期望：组的选择本来就有 lg_grp 上的显式 CE 监督，
        # 多取一组时再摊一次全局软期望只会让两组的梯度互相污染。
        wv_all = torch.softmax(lg_ver_all, dim=-1)                  # [G,V]
        zf = z_all.flatten(-2).detach()                             # [G,V,m*d]
        for j, g in enumerate(groups):
            z_soft = torch.einsum("v,vd->d", wv_all[g],
                                  zf[g].to(device=wv_all.device, dtype=wv_all.dtype)
                                  ).reshape(1, self.n_slots, self.d_model)
            # STE 要求前向数值与推理逐位一致：转回载荷精度/设备再做"梯度直通"，
            # 否则 z_hard(bf16) + Δ(float32) 会被提升成 float32，bridge 的 bf16 权重报错。
            z_soft = z_soft.to(device=z_reads[j].device, dtype=z_reads[j].dtype)
            z_reads[j] = z_reads[j] + (z_soft - z_soft.detach())
        out["z_read"] = z_reads[0]
        out["z_reads"] = z_reads
        out["hard"] = False
        return out


# ----------------------------- 3) 槽位载荷库 -----------------------------
def _slot_sort_key(slot_id: str):
    """'epi:12' -> ('epi', 12)；非数字后缀退化为字符串排序（不抛错）。"""
    s = str(slot_id)
    if ":" in s:
        k, _, rest = s.partition(":")
        try:
            return (k, int(rest), "")
        except ValueError:
            return (k, -1, rest)
    return ("", -1, s)


class LatentSlotBank:
    """稳定槽位 ID -> 潜载荷 [k, m, d]。

    - **并行**存在：texts / tokv / words / metas 一个都不删，它们仍然承担
      检索、结构化分析与诊断；本库只额外挂一份潜载荷。
    - 一个槽位可能被切成 k 块（单元长于 LATENT_MAX_UNIT_TOKENS 时**再分块**，绝不截断），
      因此载荷形状是 [k, m, d]；gather() 按槽位顺序把它们摊平成 [Σk, m, d]，
      k=1 时就是规范里的 [K, m, d]。
    - append 重复 ID / overwrite 缺失 ID / gather 缺失 ID 一律**报错**，不静默补零：
      潜载荷错位比明文错位更难发现。
    - detach：存储路径默认 detach + CPU + bf16；**训练路径必须 detach=False**，
      否则梯度断在压缩器输出上（见 train_latent_memory.py）。
    """

    def __init__(self, n_slots: Optional[int] = None, dim: Optional[int] = None,
                 dtype: Optional[torch.dtype] = None):
        self.payloads: Dict[str, torch.Tensor] = {}
        self.versions: Dict[str, int] = {}
        self.n_slots: Optional[int] = None if n_slots is None else int(n_slots)
        self.dim: Optional[int] = None if dim is None else int(dim)
        self.dtype = dtype if dtype is not None else latent_dtype()
        self.n_append = 0
        self.n_overwrite = 0
        self.n_drop = 0
        self.n_shifted = 0

    # ---- 基本容器协议 ----
    def __len__(self) -> int:
        return len(self.payloads)

    def __contains__(self, slot_id: Any) -> bool:
        return str(slot_id) in self.payloads

    def slot_ids(self, namespace: Optional[str] = None) -> List[str]:
        ids = sorted(self.payloads, key=_slot_sort_key)
        if namespace is None:
            return ids
        pre = f"{str(namespace).rstrip(':')}:"
        return [s for s in ids if s.startswith(pre)]

    def n_chunks(self, slot_id: Any) -> int:
        z = self.payloads.get(str(slot_id))
        return 0 if z is None else int(z.shape[0])

    def n_vectors(self) -> int:
        """已存软向量总数（Σ k×m）——读出预算核对用。"""
        return int(sum(int(z.shape[0]) * int(z.shape[1]) for z in self.payloads.values()))

    def n_tokens_stored(self) -> int:
        """所有槽位的软 token 数（Σ k×m）。"""
        return self.n_vectors()

    # ---- 写入 ----
    def _normalize(self, z: torch.Tensor, name: str = "z") -> torch.Tensor:
        if not torch.is_tensor(z):
            z = torch.as_tensor(z)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        if z.dim() != 3:
            raise ValueError(f"{name} 必须是 [m,d] 或 [k,m,d]，得到 {tuple(z.shape)}")
        k, m, d = (int(v) for v in z.shape)
        if k <= 0 or m <= 0 or d <= 0:
            raise ValueError(f"{name} 含空维度：{tuple(z.shape)}")
        if self.n_slots is None:
            self.n_slots = m
        elif m != self.n_slots:
            raise ValueError(f"{name} 的 m={m} 与库内 n_slots={self.n_slots} 不一致")
        if self.dim is None:
            self.dim = d
        elif d != self.dim:
            raise ValueError(f"{name} 的 d={d} 与库内 dim={self.dim} 不一致")
        return z

    def _store_tensor(self, z: torch.Tensor, detach: bool) -> torch.Tensor:
        if detach:
            z = z.detach()
        return z.to(device="cpu", dtype=self.dtype)

    def append(self, slot_id: Any, z: torch.Tensor, detach: bool = True) -> str:
        sid = str(slot_id)
        if sid in self.payloads:
            raise ValueError(f"duplicate slot_id: {sid}")
        z = self._normalize(z)
        self.payloads[sid] = self._store_tensor(z, detach)
        self.versions[sid] = 1
        self.n_append += 1
        return sid

    def overwrite(self, slot_id: Any, z: torch.Tensor, detach: bool = True) -> str:
        sid = str(slot_id)
        if sid not in self.payloads:
            raise KeyError(f"overwrite 不存在的槽位：{sid}")
        z = self._normalize(z)
        self.payloads[sid] = self._store_tensor(z, detach)
        self.versions[sid] = int(self.versions.get(sid, 1)) + 1
        self.n_overwrite += 1
        return sid

    def drop(self, slot_id: Any) -> None:
        sid = str(slot_id)
        if sid not in self.payloads:
            raise KeyError(f"drop 不存在的槽位：{sid}")
        self.payloads.pop(sid, None)
        self.versions.pop(sid, None)
        self.n_drop += 1

    def shift_index(self, namespace: str, drop_index: int) -> int:
        """淘汰第 drop_index 个槽位后，把 ns:i (i>drop_index) 重编号成 ns:i-1。

        原位覆写（_replace）保持下标稳定，但**淘汰**会移动列表下标（StatePool._drop /
        CRStatePool._evict）。潜载荷以 ID 为键，因此淘汰时必须同步映射，否则后续
        'state:5' 会指向另一条事实的载荷 —— 这类错位在指标上只会表现为"莫名其妙变差"。
        返回被移动的槽位数。
        """
        pre = f"{str(namespace).rstrip(':')}:"
        j = int(drop_index)
        moved = 0
        idxs = []
        for s in list(self.payloads.keys()):
            if s.startswith(pre):
                try:
                    idxs.append(int(s[len(pre):]))
                except ValueError:
                    continue
        for i in sorted(idxs, reverse=True):
            if i < j:
                continue
            old = f"{pre}{i}"
            if old not in self.payloads:
                continue
            if i == j:
                self.payloads.pop(old, None)
                self.versions.pop(old, None)
                continue
            new = f"{pre}{i - 1}"
            if new in self.payloads:
                raise RuntimeError(f"shift_index 目标冲突：{new} 已存在（库内下标不连续？）")
            self.payloads[new] = self.payloads.pop(old)
            self.versions[new] = self.versions.pop(old, 1)
            moved += 1
        self.n_shifted += moved
        return moved

    # ---- 读取 ----
    def payload(self, slot_id: Any, device: Any = None) -> torch.Tensor:
        sid = str(slot_id)
        if sid not in self.payloads:
            raise KeyError(f"潜载荷缺失：{sid}")
        z = self.payloads[sid]
        return z if device is None else z.to(device=device)

    def gather(self, slot_ids: Sequence[Any], device: Any = "cpu") -> torch.Tensor:
        """按给定顺序取出载荷并摊平：返回 [Σk, m, d]（k=1 时即 [K, m, d]）。"""
        ids = [str(s) for s in (slot_ids or [])]
        if not ids:
            m = self.n_slots or 0
            d = self.dim or 0
            return torch.zeros((0, m, d), dtype=self.dtype)
        missing = [s for s in ids if s not in self.payloads]
        if missing:
            raise KeyError(f"潜载荷缺失 {len(missing)} 个槽位（前 5 个：{missing[:5]}）")
        out = torch.cat([self.payloads[s] for s in ids], dim=0)
        return out.to(device=device)

    def stats(self) -> Dict[str, Any]:
        return {"n_slots": len(self.payloads), "n_soft_vectors": self.n_vectors(),
                "n_slots_per_ns": {ns: len(self.slot_ids(ns))
                                   for ns in sorted({s.partition(':')[0] for s in self.payloads})},
                "n_append": self.n_append, "n_overwrite": self.n_overwrite,
                "n_drop": self.n_drop, "n_shifted": self.n_shifted,
                "n_slots_cfg": self.n_slots, "dim": self.dim,
                "dtype": str(self.dtype).replace("torch.", "")}


# ------------- 3.5) FST-Mem：状态—轨迹解耦的槽位库 -------------
def _norm_time(t: Any) -> str:
    """时间条件规范化：'2023年' / '2023-05' / '2023' 统一成小写无标点短串。"""
    s = str(t if t is not None else "").strip().lower()
    for ch in ".,;:":
        s = s.replace(ch, " ")
    return " ".join(s.split())


def _time_match_score(query: str, cand: str) -> int:
    """时间匹配等级：2 = 完全相同；1 = 一方是另一方前缀（'2023' vs '2023-05'）；0 = 不匹配。"""
    q, c = _norm_time(query), _norm_time(cand)
    if not q or not c:
        return 0
    if q == c:
        return 2
    if c.startswith(q) or q.startswith(c):
        return 1
    return 0


def _as_year(t: Any) -> Optional[int]:
    """尽量从时间串里取出年份（'2023' / '2023-05' / '2023 spring' 都返回 2023）。"""
    s = _norm_time(t)
    digits = ""
    for ch in s:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    if len(digits) >= 4:
        try:
            return int(digits[:4])
        except ValueError:
            return None
    return None


class LatentStateTrajectoryBank(LatentSlotBank):
    """每个 (实体,属性) 事实键 = **一个当前状态 + 有限条历史版本**，彼此物理隔离。

    要解决的问题（研究问题 2、3）：
        硬覆写：只留最新值 -> "2023 年在哪工作"永远答不出（历史丢了）。
        全部追加：新旧事实同时进读出 -> 事实竞争，当前/历史互相干扰，
                  而且"更新"退化成"多了一条看起来矛盾的记忆"。
        本库：旧值**下沉到同一个键的历史**，新值占据当前位；读出时先选键、再选版本，
              只有被选中的那一个版本进 bridge。更新 ≠ 删除，也不等于并发注入。

    更新局域性由数据结构保证，不靠训练"学会别串"：
        write(zhang||employer, z2) 只碰 zhang||employer 这一个键的槽位，
        索引到 zhang||residence 的载荷一字节不变 —— 见 tests/test_fst_mem.py 的断言。

    容量：hist_max 条历史。超限**淘汰最旧的那条**（FIFO）并真正释放槽位，
    绝不"留着但读不到"——那种设计会让显存随对话长度线性增长，等于没做压缩。
    历史容量与保真度的取舍见 KNOWN_ISSUES.md（历史压缩存在容量上限）。
    """

    def __init__(self, n_slots: Optional[int] = None, dim: Optional[int] = None,
                 dtype: Optional[torch.dtype] = None, hist_max: Optional[int] = None):
        super().__init__(n_slots=n_slots, dim=dim, dtype=dtype)
        self.hist_max = int(LATENT_HIST_MAX if hist_max is None else hist_max)
        if self.hist_max < 0:
            raise ValueError(f"hist_max 不能为负：{self.hist_max}")
        # key -> {"cur": slot_id, "cur_time": str, "hist": [(slot_id, time), ...], "n_write": int}
        self.groups: Dict[str, Dict[str, Any]] = {}
        self.n_evicted = 0
        # slot_id -> 该版本的**客体值原文**（多跳接力用：上一跳选中的值 -> 下一跳的实体）
        self.slot_values: Dict[str, str] = {}

    # ---- 键 ----
    def group_keys(self) -> List[str]:
        """所有事实键，按**首次写入顺序**（不是排序，时间轨迹依赖这个顺序）。"""
        return list(self.groups.keys())

    def group_text(self, key: Any) -> Tuple[str, str]:
        """该事实键对应的 (实体原文, 属性原文)。

        优先返回 write() 时记下的**原始文本**（训练与读出必须用同一份字符串构造
        键特征）；没记过（手工建库/旧数据）才退化到从键反推的规范化文本。
        """
        g = self.groups.get(str(key))
        if g is None:
            raise KeyError(f"未知事实键：{key}")
        e, r = split_fact_key(key)
        if g.get("ent"):
            e = str(g["ent"])
        if g.get("rel"):
            r = str(g["rel"])
        return e, r

    def has_group(self, key: Any) -> bool:
        return str(key) in self.groups

    def _slot_id(self, key: str, n: int) -> str:
        return f"{key}@{int(n)}"

    # ---- 写入（状态转移） ----
    def write(self, key: Any, z: torch.Tensor, time: Any = None,
              detach: bool = True, entity: Any = None, relation: Any = None,
              value: Any = None) -> str:
        """写一个新版本：旧当前值下沉进历史，新值成为当前值。返回新槽位 ID。

        entity/relation 是**原始文本**（保留大小写）。它必须存在这里而不是由
        `split_fact_key(key)` 反推：键是规范化（小写、去标点）过的，反推出来的文本
        与训练时喂进编码器的文本不同，同一实体在训练/读出两条路上会得到不同的
        键特征 —— 不报错、指标看不出，只是"评估时路由器面对没见过的输入"。

        value 是该版本的**客体值原文**。库里只有"键 + 时间 + 载荷"时，多跳的第二跳
        无从下手（键是静态身份键，没有"客体值"这一路）；把值随版本一起存下来，
        第二跳就能用"上一跳的值 = 下一跳的实体"做确定性接力。
        """
        key = str(key)
        split_fact_key(key)                    # 格式校验：别把整句事实当键写进来
        z = self._normalize(z)
        t = _norm_time(time)
        v = "" if value is None else str(value)
        g = self.groups.get(key)
        if g is None:
            sid = self._slot_id(key, 0)
            self.append(sid, z, detach=detach)
            self.slot_values[sid] = v
            self.groups[key] = {"cur": sid, "cur_time": t, "hist": [],
                                "n_write": 1,
                                "ent": (None if entity is None else str(entity)),
                                "rel": (None if relation is None else str(relation))}
            return sid
        sid = self._slot_id(key, int(g["n_write"]))
        old = g["cur"]
        self.append(sid, z, detach=detach)     # 先 append 再改指针：append 抛错则状态不变
        self.slot_values[sid] = v
        g["hist"].append((old, g["cur_time"]))
        g["cur"] = sid
        g["cur_time"] = t
        g["n_write"] = int(g["n_write"]) + 1
        self._evict_history(key)
        return sid

    def value_at(self, key: Any, index: int) -> str:
        """该键第 index 个版本（旧 -> 新，最后一个 = 当前）的客体值原文。"""
        vers = self.list_versions(key)
        i = int(index)
        if not (0 <= i < len(vers)):
            raise IndexError(f"版本下标越界：{i}/{len(vers)}（key={key!r}）")
        return str(self.slot_values.get(vers[i]["slot_id"], ""))

    def _evict_history(self, key: str) -> int:
        """历史超过 hist_max 时淘汰最旧的，并真正释放其槽位。返回淘汰条数。"""
        g = self.groups.get(key)
        if g is None:
            raise KeyError(f"_evict_history 收到未知事实键：{key}")
        n = 0
        while len(g["hist"]) > self.hist_max:
            sid, _t = g["hist"].pop(0)
            if sid in self.payloads:
                self.drop(sid)
            self.slot_values.pop(sid, None)     # 淘汰即真释放：值文本也不许留着
            n += 1
        self.n_evicted += n
        return n

    # ---- 读取 ----
    # 注意：**不能**叫 versions —— 父类 LatentSlotBank 的 self.versions 是
    # {slot_id: 覆写次数} 字典，同名会把它覆盖成方法，overwrite/drop/shift_index 立刻崩。
    def list_versions(self, key: Any) -> List[Dict[str, Any]]:
        """该事实键的全部版本：历史（旧->新）在前，当前状态在**最后**。"""
        key = str(key)
        g = self.groups.get(key)
        if g is None:
            raise KeyError(f"未知事实键：{key}")
        out = [{"slot_id": sid, "time": t, "current": False}
               for sid, t in g["hist"]]
        out.append({"slot_id": g["cur"], "time": g["cur_time"], "current": True})
        return out

    def count_versions(self, key: Any) -> int:
        return len(self.list_versions(key))

    def read(self, key: Any, time: Any = None) -> str:
        """按时间条件选槽位 ID。time 为空 -> 当前状态。

        匹配顺序：完全相等 > 前缀（'2023' vs '2023-05'）> 不晚于查询年份的最新版本
        > 当前状态。最后一级兜底是**当前**而不是"最旧"：宁可答最新的值，也不要
        答一个用户没问过的旧值（错得可解释）。
        """
        vers = self.list_versions(key)
        cur = vers[-1]["slot_id"]
        q = _norm_time(time)
        if not q:
            return cur
        best_sid, best_sc = None, 0
        for v in vers:                       # 旧 -> 新；同分时后来的（更新的）胜出
            sc = _time_match_score(q, v["time"])
            if sc >= best_sc and sc > 0:
                best_sid, best_sc = v["slot_id"], sc
        if best_sid is not None:
            return best_sid
        qy = _as_year(q)
        if qy is not None:
            cands = []
            for v in vers:
                vy = _as_year(v["time"])
                if vy is not None and vy <= qy:
                    cands.append((vy, v["slot_id"]))
            if cands:
                cands.sort(key=lambda x: x[0])
                return cands[-1][1]
        return cur

    def payload_at(self, key: Any, time: Any = None, device: Any = None) -> torch.Tensor:
        return self.payload(self.read(key, time), device=device)

    def gather_versions(self, key: Any, device: Any = "cpu"
                        ) -> Tuple[torch.Tensor, List[str], List[bool]]:
        """-> (ver_z [V,m,d], times [V], is_current [V])，顺序为 旧->新（当前在最后）。"""
        vers = self.list_versions(key)
        zs = [_payload_md(self.payload(v["slot_id"])) for v in vers]
        return (torch.stack(zs, dim=0).to(device=device),
                [str(v["time"]) for v in vers],
                [bool(v["current"]) for v in vers])

    def gather_all_versions(self, keys: Optional[Sequence[Any]] = None
                            ) -> List[List[torch.Tensor]]:
        """按 group_keys() 顺序给出每组的版本载荷列表（可直接喂 pad_version_stack）。"""
        ks = self.group_keys() if keys is None else [str(k) for k in keys]
        return [[self.payload(v["slot_id"]) for v in self.list_versions(k)] for k in ks]

    # ---- 统计 ----
    def stats(self) -> Dict[str, Any]:
        st = super().stats()
        st.update({
            "fst_hist_max": int(self.hist_max),
            "fst_n_groups": len(self.groups),
            "fst_n_versions": int(sum(len(g["hist"]) + 1 for g in self.groups.values())),
            "fst_n_evicted": int(self.n_evicted),
            "fst_n_writes": int(sum(int(g["n_write"]) for g in self.groups.values())),
            "fst_hist_lens": {k: len(g["hist"]) for k, g in self.groups.items()},
        })
        return st


# ----------------------------- 编码：文本 -> token 级特征 -----------------------------
def _split_chunks(n_tokens: int, width: int) -> List[Tuple[int, int]]:
    """把 [0, n_tokens) 切成不重叠的 width 宽窗口（**不截断**，长单元分成多块）。"""
    w = int(width)
    if w <= 0:
        raise ValueError(f"width 必须为正，得到 {width}")
    n = int(n_tokens)
    if n <= 0:
        return [(0, 0)]
    return [(s, min(s + w, n)) for s in range(0, n, w)]


def encode_latent_units(model: Any, tokenizer: Any, texts: Sequence[str],
                        layer: int = LATENT_LAYER,
                        max_unit_tokens: int = LATENT_MAX_UNIT_TOKENS,
                        parent_ids: Optional[Sequence[str]] = None,
                        device: Any = None, batch_size: int = 8) -> Dict[str, Any]:
    """文本 -> 压缩器所需的 **token 级**特征（h0 / h18 / padding mask + 分块账本）。

    为什么不能复用 `_layer18_provider()`：那个接口返回的是经过 mean-pool 的 [N, d]，
    token 轴已经被压掉了，拿它做 token-level cross-attention 压缩是无意义的
    （等于对着一句向量再注意一次）。这里保留 token 轴。

    分块（重要）：单元长于 max_unit_tokens 时必须**再分块**并记录父槽位 ID，
    不能简单截断前 max_unit_tokens 个 token —— 否则答案可能正好落在被丢掉的后半段，
    指标下降却查不出原因。返回里 parents / chunk_index / n_chunks / span 就是这本账。

    冻结与写入：整段在 torch.no_grad() + delta_write_disabled() 下执行，
    模型不更新、Delta 在线状态不变（编码不能改动记忆本身）。
    """
    texts = ["" if t is None else str(t) for t in (texts or [])]
    if parent_ids is None:
        parent_ids = [str(i) for i in range(len(texts))]
    parent_ids = [str(p) for p in parent_ids]
    if len(parent_ids) != len(texts):
        raise ValueError(f"parent_ids 长度 {len(parent_ids)} != texts 长度 {len(texts)}")
    if int(max_unit_tokens) <= 0:
        raise ValueError(f"max_unit_tokens 必须为正，得到 {max_unit_tokens}")

    dev = torch.device(device) if device is not None else _model_device(model)
    bs = max(1, int(batch_size))
    h0_out: List[torch.Tensor] = []
    h18_out: List[torch.Tensor] = []
    mask_out: List[torch.Tensor] = []
    parents: List[str] = []
    chunk_index: List[int] = []
    n_chunks: List[int] = []
    span: List[Tuple[int, int]] = []
    n_tokens: List[int] = []
    chunk_text: List[Optional[str]] = []
    n_empty_units = 0

    with torch.no_grad(), delta_write_disabled(model):
        for s in range(0, len(texts), bs):
            batch = texts[s:s + bs]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=False)
            ids = enc["input_ids"].to(dev)
            am = enc["attention_mask"].to(dev)
            out = model(input_ids=ids, attention_mask=am, output_hidden_states=True,
                        use_cache=False)
            hs = getattr(out, "hidden_states", None)
            if hs is None:
                raise RuntimeError("模型没有返回 hidden_states（需要 output_hidden_states=True）")
            if int(layer) >= len(hs):
                raise ValueError(f"LATENT_LAYER={layer} 超出 hidden_states 层数 {len(hs)}")
            h0, hl = hs[0], hs[int(layer)]
            d = int(h0.shape[-1])
            for r in range(len(batch)):
                L = int(am[r].sum().item())
                pid = parent_ids[s + r]
                spans = _split_chunks(L, max_unit_tokens)
                for k, (a, b) in enumerate(spans):
                    if b <= a:                        # 空单元：保留 1 个全 padding 位
                        h0_out.append(torch.zeros((1, d), dtype=h0.dtype))
                        h18_out.append(torch.zeros((1, d), dtype=hl.dtype))
                        mask_out.append(torch.zeros((1,), dtype=torch.long))
                        chunk_text.append(None)
                        n_empty_units += 1
                    else:
                        h0_out.append(h0[r, a:b].to("cpu"))
                        h18_out.append(hl[r, a:b].to("cpu"))
                        mask_out.append(torch.ones((b - a,), dtype=torch.long))
                        chunk_text.append(_decode_ids(tokenizer, ids[r, a:b]))
                    parents.append(pid)
                    chunk_index.append(int(k))
                    n_chunks.append(len(spans))
                    span.append((int(a), int(b)))
                    n_tokens.append(int(max(0, b - a)))

    T = max((int(t.shape[0]) for t in h0_out), default=1)
    d = int(h0_out[0].shape[-1]) if h0_out else 0
    Nc = len(h0_out)
    h0_b = torch.zeros((Nc, T, d), dtype=h0_out[0].dtype if Nc else torch.float32)
    h18_b = torch.zeros((Nc, T, d), dtype=h18_out[0].dtype if Nc else torch.float32)
    m_b = torch.zeros((Nc, T), dtype=torch.long)
    for i in range(Nc):
        L = int(h0_out[i].shape[0])
        h0_b[i, :L] = h0_out[i]
        h18_b[i, :L] = h18_out[i]
        m_b[i, :L] = mask_out[i]
    return {"h0": h0_b, "h18": h18_b, "mask": m_b,
            "parents": parents, "chunk_index": chunk_index, "n_chunks": n_chunks,
            "span": span, "n_tokens": n_tokens, "chunk_text": chunk_text,
            "n_units": len(texts), "n_chunks_total": Nc, "n_empty_units": n_empty_units,
            "max_unit_tokens": int(max_unit_tokens), "layer": int(layer),
            "pad_len": int(T)}


def _decode_ids(tokenizer: Any, ids: torch.Tensor) -> Optional[str]:
    fn = getattr(tokenizer, "decode", None)
    if fn is None:
        return None
    try:
        return fn(ids.tolist(), skip_special_tokens=True)
    except Exception:
        return None


class LatentEncoder:
    """冻结模型 + 压缩器 = 文本 -> 潜载荷（按父槽位分组，长单元多块）。"""

    def __init__(self, model: Any, tokenizer: Any, compressor: LatentCompressor,
                 layer: int = LATENT_LAYER, max_unit_tokens: int = LATENT_MAX_UNIT_TOKENS,
                 device: Any = None, batch_size: int = 8):
        if model is None or tokenizer is None:
            raise ValueError("LatentEncoder 需要 model 与 tokenizer")
        if not isinstance(compressor, LatentCompressor):
            raise TypeError("compressor 必须是 LatentCompressor")
        self.model = model
        self.tokenizer = tokenizer
        self.compressor = compressor
        self.layer = int(layer)
        self.max_unit_tokens = int(max_unit_tokens)
        self.device = torch.device(device) if device is not None else _model_device(model)
        self.batch_size = max(1, int(batch_size))

    def features(self, texts: Sequence[str],
                 parent_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        return encode_latent_units(self.model, self.tokenizer, texts,
                                   layer=self.layer,
                                   max_unit_tokens=self.max_unit_tokens,
                                   parent_ids=parent_ids,
                                   device=self.device, batch_size=self.batch_size)

    def encode(self, texts: Sequence[str], parent_ids: Optional[Sequence[str]] = None,
               grad: bool = False) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """返回 (per-parent payload 列表, info)。

        每个 payload 形状 [k, m, d]（k = 该单元被分成的块数，1 表示未分块）。
        grad=False（默认）：存储路径，全程 no_grad，输出可直接进 LatentSlotBank。
        grad=True：训练路径，压缩器前向保留梯度（**不要**把它放进 torch.no_grad）。
        """
        texts = ["" if t is None else str(t) for t in (texts or [])]
        if not texts:
            return [], {"n_units": 0, "n_chunks": 0, "n_tokens_in": 0,
                        "n_soft_tokens": 0, "compression_ratio": None}
        feats = self.features(texts, parent_ids=parent_ids)
        if grad:
            z, cinfo = self.compressor.forward_with_info(feats["h0"], feats["h18"],
                                                        feats["mask"])
        else:
            with torch.no_grad():
                z, cinfo = self.compressor.forward_with_info(feats["h0"], feats["h18"],
                                                             feats["mask"])
        # 按父槽位分组（保持输入顺序，块内按 chunk_index）
        order: List[str] = []
        groups: Dict[str, List[int]] = {}
        for i, pid in enumerate(feats["parents"]):
            if pid not in groups:
                groups[pid] = []
                order.append(pid)
            groups[pid].append(i)
        payloads: List[torch.Tensor] = []
        for pid in order:
            rows = sorted(groups[pid], key=lambda i: feats["chunk_index"][i])
            payloads.append(z[rows])                 # [k, m, d]
        n_soft = int(sum(int(p.shape[0]) * int(p.shape[1]) for p in payloads))
        n_in = int(sum(feats["n_tokens"]))
        info = {"n_units": len(texts), "n_parents": len(order),
                "n_chunks": int(feats["n_chunks_total"]),
                "n_empty_units": int(feats["n_empty_units"]),
                "n_tokens_in": n_in, "n_soft_tokens": n_soft,
                "compression_ratio": (round(n_soft / n_in, 6) if n_in else None),
                "n_slots": int(z.shape[1]), "dim": int(z.shape[-1]),
                "n_empty_rows": int(cinfo.get("n_empty_rows", 0))}
        return payloads, info


# ----------------------------- checkpoint -----------------------------
class LatentReader:
    """已训练好的压缩器 + 桥接器 + 编码器（推理期只读）。"""

    def __init__(self, compressor: LatentCompressor, bridge: LatentBridge,
                 encoder: LatentEncoder, config: Dict[str, Any],
                 checkpoint: str, fingerprint: Dict[str, Any],
                 router: Optional["LatentRouter"] = None,
                 fst_router: Optional["FstRouter"] = None):
        self.compressor = compressor
        self.bridge = bridge
        self.encoder = encoder
        self.router = router
        self.fst_router = fst_router
        self.config = dict(config or {})
        self.checkpoint = checkpoint
        self.fingerprint = dict(fingerprint or {})

    def checkpoint_info(self) -> Dict[str, Any]:
        """进 manifest 的 checkpoint 身份（路径 + 大小 + sha256 前 16 位 + 配置）。"""
        info = dict(self.fingerprint)
        info["config"] = dict(self.config)
        return info

    def n_slots(self) -> int:
        return int(self.config.get("n_slots", self.compressor.n_slots))

    def __repr__(self) -> str:
        return (f"LatentReader(ckpt={os.path.basename(self.checkpoint) or '<unset>'}, "
                f"m={self.n_slots()}, layer={self.config.get('layer')}, "
                f"inner={self.config.get('inner_dim')})")


def checkpoint_fingerprint(path: str) -> Dict[str, Any]:
    """checkpoint 身份指纹（路径 + 字节数 + 全文 sha256 前 16 位），进 manifest。

    路径 + 字节数 + 全文 sha256 前 16 位。（旧 run_manifest 模块已随废除方案移除，
    指纹逻辑就地内联，口径与原来一致。）
    """
    if not path or not os.path.isfile(path):
        return {"checkpoint": path or "", "exists": False}
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"checkpoint": path, "exists": True, "bytes": os.path.getsize(path),
            "sha256_16": h.hexdigest()[:16]}


def load_latent_reader(checkpoint: str, model: Any, tokenizer: Any,
                       device: Any = None, strict: bool = True) -> LatentReader:
    """加载 checkpoint。**缺失/不匹配一律报错**，绝不静默初始化随机权重。

    随机初始化的压缩器 + 桥接器在维度上完全合法，但冻结 Qwen 没有任何理由读懂它，
    指标会接近"没有记忆"——那种结果一旦被当成"方法无效"就是误判。
    """
    if not checkpoint:
        raise FileNotFoundError(
            "AR_READOUT_MODE=latent 需要 LATENT_CKPT（训练脚本 train_latent_memory.py 的产物）；"
            "未训练就启用潜层读出只能得到无意义的随机投影。")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"LATENT_CKPT 不存在：{checkpoint}")
    blob = _safe_torch_load(checkpoint)
    if not isinstance(blob, dict):
        raise ValueError(f"checkpoint 结构非法（期望 dict，得到 {type(blob).__name__}）")
    cfg = dict(blob.get("config") or {})
    for key in ("n_slots", "inner_dim", "bottleneck"):
        if key not in cfg:
            raise KeyError(f"checkpoint 缺少 config[{key!r}]；请用本仓库的 "
                           f"train_latent_memory.py 重新保存")
    hs = _model_hidden_size(model)
    if int(cfg.get("hidden_size", hs)) != hs:
        raise ValueError(f"checkpoint hidden_size={cfg.get('hidden_size')} "
                         f"与当前模型 hidden_size={hs} 不一致")
    dev = torch.device(device) if device is not None else _model_device(model)
    comp = LatentCompressor(hs, n_slots=int(cfg["n_slots"]),
                            inner_dim=int(cfg["inner_dim"]),
                            num_heads=int(cfg.get("num_heads", LATENT_N_HEADS)))
    bridge = LatentBridge(hs, bottleneck=int(cfg["bottleneck"]),
                          target_norm=float(cfg.get("target_norm", LATENT_TARGET_NORM)))
    comp.load_state_dict(blob["compressor"], strict=strict)
    bridge.load_state_dict(blob["bridge"], strict=strict)
    comp.to(dev).eval()
    bridge.to(dev).eval()
    comp.requires_grad_(False)
    bridge.requires_grad_(False)
    enc = LatentEncoder(model, tokenizer, comp,
                        layer=int(cfg.get("layer", LATENT_LAYER)),
                        max_unit_tokens=int(cfg.get("max_unit_tokens",
                                                    LATENT_MAX_UNIT_TOKENS)),
                        device=dev)
    router = None
    if bool(cfg.get("has_router", False)):
        # 有 Router 的 ckpt 必须把 Router 一起恢复：缺了它，多槽输入会退回
        # "K 个槽裸拼"的老路径（位置敏感、槽数依赖），指标会莫名其妙地掉回基线。
        blob_router = blob.get("router")
        if not blob_router:
            raise ValueError(f"config.has_router=True 但 checkpoint 里没有 router 权重：{checkpoint}")
        router = LatentRouter(hs, n_slots=int(cfg["n_slots"]),
                              d_key=int(cfg.get("d_key", LATENT_D_KEY)))
        router.load_state_dict(blob_router, strict=strict)
        router.to(dev).eval()
        router.requires_grad_(False)
    fst_router = None
    if bool(cfg.get("has_fst", False)):
        # 同理：FST ckpt 缺了两级 Router 就只剩 comp+bridge，读出会退回"全版本裸拼"，
        # 而全版本裸拼正是本方法要解决的那种输入形态。宁可报错，不给静默退化留路。
        blob_fst = blob.get("fst_router")
        if not blob_fst:
            raise ValueError(f"config.has_fst=True 但 checkpoint 里没有 fst_router 权重："
                             f"{checkpoint}")
        fst_router = FstRouter(hs, n_slots=int(cfg["n_slots"]),
                               d_key=int(cfg.get("d_key", LATENT_D_KEY)))
        fst_router.load_state_dict(blob_fst, strict=strict)
        fst_router.to(dev).eval()
        fst_router.requires_grad_(False)
    return LatentReader(comp, bridge, enc, cfg, checkpoint,
                        checkpoint_fingerprint(checkpoint), router=router,
                        fst_router=fst_router)


def save_latent_checkpoint(path: str, compressor: LatentCompressor, bridge: LatentBridge,
                           config: Dict[str, Any], meta: Optional[Dict[str, Any]] = None,
                           router: Optional["LatentRouter"] = None,
                           fst_router: Optional["FstRouter"] = None) -> str:
    """保存压缩器 + 桥接器(+可选 Router / FST 两级 Router) + 配置。

    推理侧 load_latent_reader 严格对齐：has_router/has_fst 为 True 而权重缺失一律报错。
    """
    if not isinstance(compressor, LatentCompressor) or not isinstance(bridge, LatentBridge):
        raise TypeError("save_latent_checkpoint 需要 LatentCompressor + LatentBridge")
    if int(compressor.hidden_size) != int(bridge.hidden_size):
        raise ValueError("压缩器与桥接器的 hidden_size 不一致")
    cfg = dict(config or {})
    cfg.setdefault("schema", LATENT_SCHEMA)
    cfg["hidden_size"] = int(compressor.hidden_size)
    cfg["n_slots"] = int(compressor.n_slots)
    cfg["inner_dim"] = int(compressor.inner_dim)
    cfg["num_heads"] = int(compressor.num_heads)
    cfg["bottleneck"] = int(bridge.bottleneck)
    if router is not None:
        cfg["has_router"] = True
        cfg["d_key"] = int(router.d_key)
        if int(router.d_model) != int(compressor.hidden_size):
            raise ValueError("router.d_model != compressor.hidden_size")
        if int(router.n_slots) != int(compressor.n_slots):
            raise ValueError("router.n_slots != compressor.n_slots")
    else:
        cfg["has_router"] = False
    if fst_router is not None:
        cfg["has_fst"] = True
        cfg["d_key"] = int(fst_router.d_key)
        if int(fst_router.d_model) != int(compressor.hidden_size):
            raise ValueError("fst_router.d_model != compressor.hidden_size")
        if int(fst_router.n_slots) != int(compressor.n_slots):
            raise ValueError("fst_router.n_slots != compressor.n_slots")
    else:
        cfg["has_fst"] = False
    blob = {"schema": LATENT_SCHEMA, "config": cfg,
            "compressor": compressor.state_dict(),
            "bridge": bridge.state_dict(),
            "router": (router.state_dict() if router is not None else None),
            "fst_router": (fst_router.state_dict() if fst_router is not None else None),
            "meta": dict(meta or {})}
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)
    return path


# ----------------------------- 4) 潜层读出 -----------------------------
def build_chat_parts(tokenizer: Any, system_prompt: str, question: str) -> Tuple[str, int]:
    """渲染完整 chat 文本，并返回 system 段（含结束符）的字符长度。

    软 token 的插入点必须**可验证**：这里要求"只渲染 system 消息"的结果是完整渲染的前缀。
    前缀关系不成立时直接报错，而不是猜一个位置 —— 插错位置会让模型看到错乱的对话结构，
    指标下降却看不出原因。
    """
    sysp = (system_prompt or "").strip()
    msgs: List[Dict[str, str]] = []
    if sysp:
        msgs.append({"role": "system", "content": sysp})
    msgs.append({"role": "user", "content": question})
    full = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if not sysp:
        return full, 0
    sys_only = tokenizer.apply_chat_template([{"role": "system", "content": sysp}],
                                             add_generation_prompt=False, tokenize=False)
    if not str(full).startswith(str(sys_only)):
        raise RuntimeError(
            "chat template 不满足『system 段是完整渲染的前缀』，无法确定潜 token 的插入点。"
            "请显式指定插入位置（不要靠猜测），否则软 token 会落在对话结构中间。")
    return full, len(sys_only)


def _tokenize_no_special(tokenizer: Any, text: str, device: Any) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"].to(device)


def _emit_latent(model: Any, tokenizer: Any, soft: Optional[torch.Tensor], question: str,
                 system_prompt: str, max_new_tokens: int = 32,
                 pad_token_id: Optional[int] = None,
                 extra_meta: Optional[Dict[str, Any]] = None
                 ) -> Tuple[str, Dict[str, Any]]:
    """把**已经算好的** soft tokens（[1,T,d] 或 None）插进 system 段之后并生成。

    单独立出来是为了让两条读出路径（2.5 的单级 Router 与 FST 的两级 Router）
    共用同一套插入/掩码/解码逻辑：拷贝一份就是两份会漂移的实现，而"软 token 插在
    哪里""返回序列含不含 prompt 前缀"这两件事一旦写错，指标会以很难察觉的方式变差。
    """
    dev = _model_device(model)
    emb_layer = model.get_input_embeddings()
    full, n_sys_chars = build_chat_parts(tokenizer, system_prompt, question)
    ids_full = _tokenize_no_special(tokenizer, full, dev)
    n_sys = 0
    if n_sys_chars:
        ids_pre = _tokenize_no_special(tokenizer, full[:n_sys_chars], dev)
        n_sys = int(ids_pre.shape[1])
        if ids_full[0, :n_sys].tolist() != ids_pre[0].tolist():
            raise RuntimeError(
                "system 段单独分词与整段分词在边界上不一致（BPE 边界跨界），"
                "无法保证软 token 插在 system 之后；请改用带显式分隔符的模板。")

    pre = emb_layer(ids_full[:, :n_sys])
    post = emb_layer(ids_full[:, n_sys:])
    parts = [pre]
    n_soft = 0
    if soft is not None:
        if soft.dim() == 2:
            soft = soft.unsqueeze(0)
        if soft.dim() != 3:
            raise ValueError(f"soft 必须是 [1,T,d] 或 [T,d]，得到 {tuple(soft.shape)}")
        n_soft = int(soft.shape[1])
        parts.append(soft.to(device=dev, dtype=pre.dtype))
    parts.append(post)
    input_embeds = torch.cat(parts, dim=1)
    attention_mask = torch.ones(input_embeds.shape[:2], dtype=torch.long, device=dev)
    n_in = int(input_embeds.shape[1])

    pad = pad_token_id
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None) or getattr(tokenizer, "pad_token_id", None) or 0
    # 生成期必须关掉 Delta 写入：upstream 的 write_enabled 默认 True，`model.generate`
    # 会边读 prompt 边更新在线状态、再把 Delta 读出加回 q/o 注意力。而训练侧的记忆
    # 编码 / 问题特征 / student 前向全都在 delta_write_disabled 下进行 —— 两侧不一致时，
    # adapter 等于在评测时额外开了一套训练时从未使用的记忆系统，指标会以"全方法一起
    # 退化、输出塌成固定 token"的形式坏掉，且看不出是这里的问题。
    # 只读策略下没有预先写入的状态，Delta 读出应为零，生成行为回到纯主干。
    with delta_write_disabled(model), torch.no_grad():
        out = model.generate(inputs_embeds=input_embeds, attention_mask=attention_mask,
                             max_new_tokens=int(max_new_tokens), do_sample=False,
                             use_cache=True, pad_token_id=pad)
    if not torch.is_tensor(out):
        raise RuntimeError(f"generate() 返回了非张量：{type(out).__name__}")
    seq = out[0]
    if int(seq.shape[0]) > n_in:
        # 返回序列比输入长：prompt 前缀被一起返回了，必须切掉 n_in 再解码。
        pred = tokenizer.decode(seq[n_in:].tolist(), skip_special_tokens=True)
        decode_mode = "full_sequence_returned"
        n_out = int(seq.shape[0]) - n_in
    else:
        # 只返回新 token：序列里没有 prompt 前缀，整段就是答案。
        pred = tokenizer.decode(seq.tolist(), skip_special_tokens=True)
        decode_mode = "new_tokens_only"
        n_out = int(seq.shape[0])
    meta = {"n_in": n_in, "n_out": n_out, "n_soft": n_soft,
            "n_sys_tokens": int(n_sys), "insert_at": int(n_sys),
            "decode_mode": decode_mode, "max_new_tokens": int(max_new_tokens)}
    meta.update(dict(extra_meta or {}))
    return pred, meta


def generate_latent_answer(model: Any, tokenizer: Any, bridge: LatentBridge,
                           payloads: Optional[torch.Tensor], question: str,
                           system_prompt: str, max_new_tokens: int = 32,
                           pad_token_id: Optional[int] = None,
                           router: Optional["LatentRouter"] = None,
                           q_feat: Optional[torch.Tensor] = None,
                           layer: int = LATENT_LAYER, hops: int = 1
                           ) -> Tuple[str, Dict[str, Any]]:
    """潜层读出：只有 system / question 是正常 token，历史证据全部走 soft tokens。

    签名里**没有** block / evidence_text 参数：接口层面就不给"意外退回明文 RAG"留口子。

    调用关系（与设计一致）：
        soft = bridge(payloads).reshape(1, K*m, d)
        input_embeds = cat([system_embeds, soft, query_embeds], dim=1)

    两个必须处理的现实问题：
      1. 位置连续性 / attention mask：软 token 全部参与注意（mask=1），且插在 system 段之后；
      2. 输出解码：Qwen + Delta-Mem 改过的 generate() 对 inputs_embeds 的返回序列**不保证**
         只含新 token。这里两种形态都处理，并把实际形态写进 meta["decode_mode"]，
         不做"一定只返回新 token"的假设（否则会把 prompt 尾部当答案）。

    router（可选）：多槽输入时先做集合内容寻址，只把被选中那一个槽送进 bridge。
      有无 router 的差别就是"位置敏感"与"位置无关"的差别（实测同一事实：
      4 槽裸拼 target_end = 0/16；路由后应逐位等于单槽结果）。
      hops>1 时按 k_out 链式选出多跳槽位，槽位之间插 Qwen 原生 "\n" embedding 隔离。
    """
    if tokenizer is None or model is None:
        raise ValueError("generate_latent_answer 需要 model 与 tokenizer")
    if payloads is not None and not torch.is_tensor(payloads):
        # encode() 返回的是 per-slot payload 列表（每项 [k,m,d]，k=该单元被切的块数）。
        # **必须**走 stack_payloads：直接 torch.as_tensor(list_of_3d_tensors) 会在
        # 形状不一致时报 "only one element tensors can be converted to scalars"，
        # 而 cat 掉再 reshape 又会丢掉槽边界（把"块"当成"槽"）。
        if isinstance(payloads, (list, tuple)):
            payloads = stack_payloads(list(payloads))
        else:
            payloads = torch.as_tensor(payloads)
        if payloads.dim() == 2:
            payloads = payloads.unsqueeze(0)
    has_payload = bool(payloads is not None and payloads.numel() > 0)
    if has_payload and bridge is None:
        raise ValueError("给了潜载荷但没有 bridge：无法把 Z 映射成 soft tokens")

    dev = _model_device(model)
    emb_layer = model.get_input_embeddings()
    soft: Optional[torch.Tensor] = None
    routed: Optional[List[int]] = None
    if has_payload and router is not None:
        slot_z = payloads if payloads.dim() == 4 else payloads.unsqueeze(0)   # [1,K,m,d]
        slot_z = slot_z.to(dev)
        qf = q_feat
        if qf is None:
            qf = extract_query_feature(model, tokenizer, question, system_prompt,
                                       layer=layer, device=dev)
        if int(hops) > 1:
            # 多跳走 forward_hops：与训练侧**同一个函数**，保证 k_out 的用法一致
            out = router.forward_hops(qf, slot_z, n_hops=int(hops), hard=True)
        else:
            _z, logits, _ki, _ko = router(qf, slot_z, hard=True)
            out = {"picks": [int(logits[0].argmax().item())],
                   "z_list": [slot_z[:, int(logits[0].argmax().item())]]}
        routed = list(out["picks"])
        soft = compose_routed_soft(model, tokenizer, bridge, out["z_list"], dev=dev)
    elif has_payload:
        # bridge 不做内容寻址：整库所有槽位**全给**，K*m 个软 token 一起注入，
        # 这是 bridge 的设计语义（"压缩后整段送去读"），不是遗漏了截断。
        # payloads 在评测/训练侧统一是 [1,K,m,d]，必须先归一化到 3 维再进 bridge：
        # 直接把 4 维喂给 bridge 会在 _as_3d 里抛
        # "z 必须是 [B,L,d] 或 [L,d]，得到 (1,20,4,2560)" —— bridge 评测实际就死在这。
        pz = payloads
        if pz.dim() == 4:
            if int(pz.shape[0]) != 1:
                raise ValueError(
                    f"bridge 只支持 batch=1 的潜载荷，得到 {tuple(pz.shape)}")
            pz = pz[0]
        elif pz.dim() == 2:
            pz = pz.unsqueeze(0)
        soft = bridge(pz.to(dev))                    # [K, m, d] -> [K, m, hidden]
        if soft.dim() == 3:
            soft = soft.reshape(1, -1, soft.shape[-1])
    pred, meta = _emit_latent(model, tokenizer, soft, question, system_prompt,
                              max_new_tokens=max_new_tokens, pad_token_id=pad_token_id,
                              extra_meta={
                                  "has_payload": has_payload, "routed": routed,
                                  "hops": int(hops),
                                  "n_slots_in": (int(payloads.shape[0])
                                                 if has_payload else 0)})
    return pred, meta


def generate_latent_answer_fst(model: Any, tokenizer: Any, bridge: LatentBridge,
                               bank: "LatentStateTrajectoryBank", question: str,
                               system_prompt: str, fst_router: "FstRouter",
                               max_new_tokens: int = 32,
                               pad_token_id: Optional[int] = None,
                               layer: int = LATENT_LAYER,
                               query_time: Any = None,
                               device: Any = None,
                               hops: int = 1,
                               ver_policy: str = "learned",
                               feat_cache: Optional[Dict[Any, torch.Tensor]] = None
                               ) -> Tuple[str, Dict[str, Any]]:
    """FST-Mem 的两级寻址读出：**先选 (实体,属性)，再按时间条件选版本**。

    与 generate_latent_answer 的差别只有一点，但是关键的一点：
        单级 Router 把当前值和历史值平铺成 K 个槽一起打分，时间条件只能混在查询里
        影响"选哪个槽"；而"2023 年的张三-雇主"与"张三-居住地"会同时在同一个
        softmax 里竞争。这里两级打分互不竞争：组内选版本时，其他属性根本不在候选集里。

    返回 meta 额外字段（评估寻址准确率要用，不能只报答案对错）：
        group_keys / group_picked / n_groups / ver_times / ver_picked / ver_time /
        ver_is_current / query_time / n_versions / n_slots_in（= n_groups）
    """
    if model is None or tokenizer is None:
        raise ValueError("generate_latent_answer_fst 需要 model 与 tokenizer")
    if not isinstance(bank, LatentStateTrajectoryBank):
        raise TypeError("generate_latent_answer_fst 需要 LatentStateTrajectoryBank")
    if int(hops) < 1:
        raise ValueError(f"hops 必须 >= 1，得到 {hops}")
    dev = torch.device(device) if device is not None else _model_device(model)
    keys = bank.group_keys()
    if not keys:
        raise ValueError("bank 里没有任何事实键：先 write() 再读出")
    if bridge is None:
        raise ValueError("generate_latent_answer_fst 需要 bridge")

    def _tf(text: str) -> torch.Tensor:
        # 缓存键必须带 layer/device：同一段文本在不同层/不同设备上的特征不同，
        # 只按文本命中会把另一套配置的向量拿来打分（路由悄悄错位，指标看不出）。
        k = ("txt", str(text), int(layer), str(dev))
        if feat_cache is not None and k in feat_cache:
            return feat_cache[k]
        v = extract_text_feature(model, tokenizer, text, layer=layer, device=dev)
        if feat_cache is not None:
            feat_cache[k] = v.detach()
        return v

    # 实体/属性文本取 bank 里存的**原始文本**，与训练侧完全同源（见 write 的说明）
    ents_rels = [bank.group_text(k) for k in keys]
    e_feat = torch.cat([_tf(e) for e, _r in ents_rels], dim=0)            # [G,d]
    r_feat = torch.cat([_tf(r) for _e, r in ents_rels], dim=0)
    per_group = bank.gather_all_versions(keys)                            # List[List[payload]]
    ver_z, ver_mask = pad_version_stack(per_group, dtype=bank.dtype, device="cpu")
    dev_z = torch.device(dev)
    ver_z = ver_z.to(dev_z)
    ver_mask = ver_mask.to(dev_z)
    times = [[v["time"] for v in bank.list_versions(k)] for k in keys]
    V = int(ver_z.shape[1])
    ver_t_feat = torch.zeros((len(keys), V, int(ver_z.shape[-1])),
                             dtype=ver_z.dtype, device=dev_z)
    for gi, ts in enumerate(times):
        for vi, t in enumerate(ts):
            ver_t_feat[gi, vi] = _tf(t).to(device=dev_z, dtype=ver_t_feat.dtype)[0]

    q_feat = extract_query_feature(model, tokenizer, question, system_prompt,
                                   layer=layer, device=dev)
    # 查询时间条件也用**规范化后**的文本取特征：训练侧先 _norm_time 再取特征，
    # 这里若把原串（'2021,' / ' May 2021'）直接送进去，两侧时间向量就不是同一分布。
    tq_feat = _tf(_norm_time(query_time)) if _norm_time(query_time) else None
    n_read = max(1, min(int(hops), len(keys)))
    out = fst_router(q_feat, e_feat, r_feat, ver_z, ver_t_feat, tq_feat=tq_feat,
                     ver_mask=ver_mask, hard=True, k_groups=n_read,
                     ver_policy=ver_policy, ver_times=times, query_time=query_time)
    picked: List[Tuple[int, int]] = list(zip(out["groups"], out["versions"]))
    # 第 2..n 跳：用上一跳选中版本的**客体值**去找"以它为实体"的事实键。
    # 这里刻意**不用**可学习的 k_out —— FST 的键是静态身份键，没有"客体值"这一路，
    # 硬塞一个 chain 网络就回到"推理用未训练网络"的老问题上（Router 侧的旧 bug）。
    # 值 -> 实体是数据里就有的确定性链接，不该交给学习器重新猜。
    while len(picked) < n_read:
        last_v = _norm_key_part(bank.value_at(keys[picked[-1][0]], picked[-1][1]))
        if not last_v:
            break
        taken = {g for g, _v in picked}
        cands = [i for i, k in enumerate(keys)
                 if i not in taken and _norm_key_part(bank.group_text(k)[0])
                 and _norm_key_part(bank.group_text(k)[0]) in last_v]
        if not cands:
            break
        sub = out["logits_group"][0][cands]
        g2 = int(cands[int(sub.argmax().item())])
        v2 = int(fst_router.pick_version(g2, out["logits_ver_all"], ver_mask,
                                         ver_policy, times, query_time))
        picked.append((g2, v2))
    z_list = [_payload_md(ver_z[g, v]).unsqueeze(0) for g, v in picked]
    soft = compose_routed_soft(model, tokenizer, bridge, z_list, dev=dev)
    g, v = picked[0]
    pred, meta = _emit_latent(model, tokenizer, soft, question, system_prompt,
                              max_new_tokens=max_new_tokens, pad_token_id=pad_token_id,
                              extra_meta={
                                  "has_payload": True, "routed": [int(x[0]) for x in picked],
                                  "hops": len(picked),
                                  "n_slots_in": len(keys),
                                  "group_keys": keys, "group_picked": int(g),
                                  "groups_picked": [int(x[0]) for x in picked],
                                  "versions_picked": [int(x[1]) for x in picked],
                                  "chain_values": [bank.value_at(keys[x[0]], x[1])
                                                   for x in picked],
                                  "n_groups": len(keys),
                                  "ver_times": times[g], "ver_picked": int(v),
                                  "ver_time": times[g][v],
                                  "ver_is_current": bool(v == len(times[g]) - 1),
                                  "query_time": _norm_time(query_time),
                                  "ver_policy": str(ver_policy),
                                  "n_versions": len(times[g])})
    return pred, meta


# ----------------------------- 预算选择（纯函数，可单测） -----------------------------
def select_latent_cr_paths(slot_ids: Sequence[int], paths: Sequence[Dict[str, Any]],
                           token_budget: Optional[int] = None,
                           tokens_per_slot: Optional[int] = None
                           ) -> Tuple[List[int], Dict[str, Any]]:
    """CR 潜层读出：按预算优先保留**完整**多跳路径。

    为什么不能 `selected[:K]`：多跳链的第二跳往往排在后面，简单截断会只留第一跳、
    把回答需要的第二跳删掉，指标下降还查不出原因。因此以"路径"为不可分割单位：
      - 先按给定顺序（2 跳在前、3 跳在后，分支内按链分降序）整条整条地装；
      - 装不下的整条跳过并记账（n_paths_skipped），不做半条路径；
      - 一条完整路径都装不下时，**保留优先级最高的那一条**并标 overflow=True
        （宁可超预算一点，也不要"有检索结果但读出为空"）。
    """
    tps = LATENT_M if tokens_per_slot is None else int(tokens_per_slot)
    budget = LATENT_READ_BUDGET if token_budget is None else int(token_budget)
    tps = max(1, tps)
    # 与 latent_budget_slots 同一口径：给了预算就至少保留 1 个槽位（宁可轻微超预算，
    # 也不要"有检索结果但读出为空"），真正装不下的情况由下面的 overflow 分支处理。
    max_slots = latent_budget_slots(budget, tps)
    info: Dict[str, Any] = {"budget": int(budget), "tokens_per_slot": int(tps),
                            "max_slots": int(max_slots), "n_paths_in": len(paths or []),
                            "n_paths_kept": 0, "n_paths_skipped": [],
                            "overflow": False, "kept_paths": [],
                            "n_kept": 0, "n_dropped": 0, "dropped": []}
    if max_slots <= 0:
        info["disabled"] = True
        info["n_dropped"] = len(set(int(i) for i in (slot_ids or [])))
        info["dropped"] = sorted({int(i) for i in (slot_ids or [])})[:200]
        return [], info

    kept: List[int] = []
    seen = set()

    def _add(i: int) -> bool:
        if i in seen:
            return False
        seen.add(i)
        kept.append(i)
        return True

    for pi, p in enumerate(paths or []):
        facts: List[int] = []
        for f in (p.get("facts") or []):
            if f is None:
                continue
            facts.append(int(f))
        need = len([f for f in facts if f not in seen])
        if need == 0:
            continue
        if len(kept) + need > max_slots:
            if not kept:
                for f in facts:
                    _add(f)
                info["overflow"] = True
                info["n_paths_kept"] += 1
                info["kept_paths"].append({"path": pi, "path_len": p.get("path_len"),
                                           "facts": facts, "overflow": True})
                break
            info["n_paths_skipped"].append({"path": pi, "path_len": p.get("path_len"),
                                            "need": need, "facts": facts})
            continue
        for f in facts:
            _add(f)
        info["n_paths_kept"] += 1
        info["kept_paths"].append({"path": pi, "path_len": p.get("path_len"),
                                   "score": p.get("score"), "facts": facts})
    # 补位只允许用"不属于任何被跳过路径"的槽位：否则整条路径虽然被跳过，
    # 它的第二跳又会被补位单独塞回来，等价于把一条多跳路径切成半条。
    blocked = {f for s in info["n_paths_skipped"] for f in s["facts"]}
    for i in slot_ids or []:
        if len(kept) >= max_slots:
            break
        if int(i) in blocked:
            continue
        _add(int(i))
    dropped = sorted({int(i) for i in (slot_ids or [])} - seen)
    info["n_kept"] = len(kept)
    info["n_dropped"] = len(dropped)
    info["dropped"] = dropped[:200]
    return kept, info


# ----------------------------- 池 -> 载荷库 -----------------------------
def build_latent_bank(epi: Any = None, state_pool: Any = None, encoder: Any = None,
                      namespaces: Optional[Dict[str, str]] = None,
                      max_slots: Optional[int] = None) -> Tuple[LatentSlotBank, Dict[str, Any]]:
    """为已建好的池子构造潜载荷库（每个槽位的**原始内容**编码一次，读出只用载荷）。

    命名空间区分 'epi:0' / 'state:0' / 'cr:0'：AR 的 Episodic 与 State 池都用从 0 开始的
    局部下标，不加命名空间必然串味。

    max_slots>0 时只编码前 max_slots 个槽位（用于 smoke），被跳过的数量记在 stats 里。
    """
    if encoder is None:
        raise ValueError("build_latent_bank 需要 encoder（LatentEncoder）")
    max_slots = LATENT_BANK_MAX if max_slots is None else int(max_slots)
    ns_map = {"epi": "epi", "state": "state"}
    ns_map.update({str(k): str(v) for k, v in (namespaces or {}).items()})
    bank = LatentSlotBank()
    stats: Dict[str, Any] = {"per_pool": {}, "n_units": 0, "n_chunks": 0,
                             "n_tokens_in": 0, "n_soft_tokens": 0, "n_skipped": 0}
    for pool, key in ((epi, "epi"), (state_pool, "state")):
        if pool is None or len(pool) == 0:
            continue
        ns = ns_map.get(key, key)
        texts = list(pool.texts)
        ids = list(range(len(texts)))
        if max_slots and len(ids) > max_slots:
            stats["n_skipped"] += len(ids) - max_slots
            ids = ids[:max_slots]
        parents = [f"{ns}:{i}" for i in ids]
        payloads, info = encoder.encode([texts[i] for i in ids], parent_ids=parents)
        for sid, z in zip(parents, payloads):
            bank.append(sid, z)
        stats["per_pool"][ns] = {"n_slots": len(payloads), **info}
        stats["n_units"] += int(info.get("n_units", 0))
        stats["n_chunks"] += int(info.get("n_chunks", 0))
        stats["n_tokens_in"] += int(info.get("n_tokens_in", 0))
        stats["n_soft_tokens"] += int(info.get("n_soft_tokens", 0))
    n_in = stats["n_tokens_in"]
    stats["compression_ratio"] = (round(stats["n_soft_tokens"] / n_in, 6) if n_in else None)
    stats.update(bank.stats())
    return bank, stats


def attach_bank_to_pools(bank: LatentSlotBank, encoder: Any, epi: Any = None,
                         state_pool: Any = None, cr_pool: Any = None) -> Dict[str, int]:
    """把载荷库挂到池上，使 _store / _replace / _drop 之后能同步潜载荷（P1）。

    挂接本身不改变检索：MaxSim / 时间打分 / CR 双阈值覆写一行都不动。
    返回 {namespace: 已编码槽位数}。
    """
    out: Dict[str, int] = {}
    for pool, ns in ((epi, "epi"), (state_pool, "state"), (cr_pool, "cr")):
        if pool is None:
            continue
        out[ns] = int(pool.attach_latent_bank(bank, namespace=ns, encoder=encoder))
    return out


def latent_readout_record(reader: Optional[LatentReader], slot_ids: Sequence[Any] = (),
                          n_vectors: int = 0, extra: Optional[Dict[str, Any]] = None
                          ) -> Dict[str, Any]:
    """逐题 readout 字段：显式声明这是潜层读出，而不是明文块。"""
    rec = {"mode": "latent", "latent_slots": True,
           "n_slots": len(list(slot_ids or [])), "n_vectors": int(n_vectors),
           "checkpoint": (reader.checkpoint if reader is not None else None),
           "checkpoint_id": ((reader.fingerprint or {}).get("sha256_16")
                             if reader is not None else None),
           "note": ("历史证据只以潜向量（soft tokens）进入生成上下文；"
                    "answer_in_context 这类明文匹配诊断在 latent 模式下不适用")}
    if extra:
        rec.update(extra)
    return rec


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
