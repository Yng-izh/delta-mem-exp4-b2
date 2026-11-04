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
# 压缩器内部宽度。v3 是 512：两个方向（h0->m=4 个槽）的 cross-attention 把整段
# 事实塞进 512 维瓶颈，是 P06 量化出的信息损失来源之一，v4 默认加宽到 1024。
# runs/run_train_v3.sh 显式传 --inner-dim 512，所以旧结果仍可复现。
LATENT_INNER_DIM       = _env_int("LATENT_INNER_DIM", 1024)
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
# ---- v4 手术一：h0 符号直通（Lexical Skip-Connection） ----
# 压缩器把 h0/h18 一起投到 inner_dim 再还原，等于给"人名/地名/数字"加了一道低通滤波：
# 金标证据命中的 28 题里仍有 20 题答错（条件 EM 28.6% vs oracle 明文 55.4%）。h0 本身
# 就在冻结模型输入嵌入所在的流形上，所以把交叉注意力的权重直接拿去聚合**未降维的**
# h0，再从 bridge 输出端加回来，可以做到"注意力头挑中的那几个 token 零失真直达生成端"。
LATENT_LEX_SKIP        = _env_int("LATENT_LEX_SKIP", 1)         # 1=启用直通，0=退回纯语义分支
# 直通分支的**选通温度**。0.05 ≈ 近似 argmax：让每个槽抓到一个**具体实体 token**
# 的原生 h0，而不是把 30 个 token 平均成一团"模糊面团"（2026-11-03 归因：
# 同样 33 token 预算下，原生 h0 = 32.8% EM，压缩软向量只有 17.2%）。
LATENT_LEX_TAU         = _env_float("LATENT_LEX_TAU", 0.05)
# 门控 α 初值。**不能是 0**（2026-11-04 定位到的死锁）：
#   末层零初始化 ⇒ res = unit(0)·scale = 0 ⇒ e = lex + α·res = lex。
#   α=0 时 ∂e/∂W_last ∝ α = 0，而 ∂e/∂α ∝ res = 0，于是语义分支与门控**同时**
#   拿不到梯度，只能靠"另一条不走 α 的重建路径"启动 —— 而那条路径正是与 QA 冲突的
#   旧写法。改成 0.05 后：res 仍然严格为 0（输出逐位等于 z_lex，oracle_embed 起点不变），
#   但 ∂e/∂W_last ∝ 0.05 ≠ 0，语义分支从第 0 步就能学；res 一旦非零，α 也开始动。
LATENT_LEX_GATE_INIT   = _env_float("LATENT_LEX_GATE_INIT", 0.05)
# α 的**硬上限**（tanh 之后的截断）。残差锚定形式的唯一风险是"残差越训越大，
# 把 z_lex 这条原生词向量拉离词表流形"：v5 实测 α 已从 0 涨到 tanh(0.2705)=0.264、
# 同时 bridge 末层权重范数 13.29、out_norm 从 1.05 涨到 1.36，读出端 EM 反而比 v3 低
# （11.8% vs 17.2%）。这里给残差一个可解释的上限：α<=0.5 时 e = lex + α·res 与 lex 的
# 夹角不超过 ~30°，方向仍由原生词向量主导；需要更强残差时再显式调这个环境变量。
LATENT_ALPHA_MAX       = _env_float("LATENT_ALPHA_MAX", 0.5)
# ---- v4 手术二：HybridLatentRouter 的 h0 Token-MaxSim 底盘 ----
LATENT_ADDR_GAMMA      = _env_float("LATENT_ADDR_GAMMA", 0.5)   # γ 初值：MaxSim 占比（可学习）
LATENT_MS_TAU          = _env_float("LATENT_MS_TAU", 0.2)       # MaxSim 的固定逆温度
LATENT_KEY_TOKENS      = _env_int("LATENT_KEY_TOKENS", 64)      # 每个槽位留多少 token 做寻址键
# 标题锚点增益：单元文本以 "[anchor] ..." 开头，第 1 个有效 token 就是段落标题，
# 而它往往正是下一跳要找的桥接实体（HotpotQA 的两跳就是靠标题串起来的）。
LATENT_ANCHOR_BOOST    = _env_float("LATENT_ANCHOR_BOOST", 1.3)
# 标题区覆盖多少个 token：标题会被切成多个 subword，只给第 0 位加成等于没加成
LATENT_ANCHOR_TOKENS   = _env_int("LATENT_ANCHOR_TOKENS", 3)
# 第二跳词级接力取几个桥接 token
LATENT_RELAY_TOKENS    = _env_int("LATENT_RELAY_TOKENS", 4)
# 第 2 跳出键的新信息抑制强度 λ：k_out 要聚焦"问题里没出现过的新实体"，
# 因此对与问题高度相似的 token 按 λ·sim 扣分。0 = 关闭（回到旧行为）。
LATENT_NOVELTY_LAMBDA  = _env_float("LATENT_NOVELTY_LAMBDA", 1.0)
# ---- v4 手术三：槽位间块对角隔离 ----
# 1=尝试在读出侧对不同槽位的软 token 加块对角掩码（需要 4D attention mask，
# 只有 eager 注意力实现支持；不支持时自动降级为原生隔离栅栏并在 meta 里写明）。
# 默认**开启**槽位块对角隔离：v4 训练全程 slot_isolation=0，4 个槽位在同一个
# 自注意力里互相串扰，注入 token 从 22 涨到 58 而召回率没跟上（2026-11-03 实测）。
# 只在 eager 后端真正生效；SDPA 下会自动降级为围栏并在 meta 里写明。
LATENT_SLOT_ISOLATION  = _env_int("LATENT_SLOT_ISOLATION", 1)
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
        "lex_skip": int(LATENT_LEX_SKIP),
        "key_tokens": int(LATENT_KEY_TOKENS),
        "addr_gamma": float(LATENT_ADDR_GAMMA),
        "ms_tau": float(LATENT_MS_TAU),
        "slot_isolation": int(LATENT_SLOT_ISOLATION),
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
                 inner_dim: int = LATENT_INNER_DIM, num_heads: int = LATENT_N_HEADS,
                 lex_skip: bool = bool(LATENT_LEX_SKIP),
                 lex_tau: float = LATENT_LEX_TAU):
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
        # h0 符号直通（v4）：见文件头 LATENT_LEX_SKIP 的说明。关闭时 z_lex=None，
        # 下游 bridge 行为与 v3 逐位一致（旧 checkpoint 因此可以原样加载）。
        self.lex_skip = bool(lex_skip)
        self.lex_tau = float(lex_tau)   # 直通分支选通温度（见 LATENT_LEX_TAU）
        self.input_proj = nn.Linear(self.hidden_size, self.inner_dim)
        self.hidden_proj = nn.Linear(self.hidden_size, self.inner_dim)
        self.queries = nn.Parameter(torch.randn(self.n_slots, self.inner_dim) * 0.02)
        self.attn = nn.MultiheadAttention(self.inner_dim, num_heads=self.num_heads,
                                          batch_first=True)
        self.output_proj = nn.Linear(self.inner_dim, self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, h0: torch.Tensor, h18: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_sem, _lex, _info = self.forward_split(h0, h18, attention_mask)
        return z_sem

    def forward_split(self, h0: torch.Tensor, h18: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """返回 (z_sem [B,m,d], z_lex [B,m,d] 或 None, info)。

        z_lex = A · h0，其中 A 是**同一个**交叉注意力头对单元 token 的（去 padding、
        按 token 轴重新归一化的）权重，h0 是**未经降维**的第 0 层词嵌入/embedding。

        为什么这条路必须绕开 inner_dim 瓶颈：h0 已经处在冻结 Qwen 的输入嵌入流形上，
        它和 `inputs_embeds` 是同一个空间。走一遍 input_proj -> 512 -> output_proj
        再回到 d_model，等价于对"专有名词的词形"做了一次低通滤波 —— HotpotQA 的
        人名/地名/年份正好活在这个被滤掉的高频段里。直通分支不做任何投影。
        """
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
        # lex_skip 打开时必须同时取出注意力权重（need_weights=True）：这条路径多付一次
        # 权重回传，但省掉"再跑一次注意力"的整段开销，而且**与语义分支共用同一份
        # 注意力**，所以"挑哪些 token"和"压哪些 token"不会各说各话。
        z, w = self.attn(q, x, x, key_padding_mask=~safe_mask,
                         need_weights=bool(self.lex_skip),
                         average_attn_weights=True)
        z = self.norm(self.output_proj(z))
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        if bool(empty.any()):
            z = z.masked_fill(empty.view(B, 1, 1), 0.0)

        z_lex: Optional[torch.Tensor] = None
        lex_pick: Optional[List[List[int]]] = None
        if self.lex_skip:
            if w is None:
                raise RuntimeError("lex_skip=True 但注意力权重为 None（need_weights 未生效）")
            if tuple(w.shape)[1:] != (self.n_slots, L):
                # batch_first=True 下权重应是 [B, m, L]；不同 torch 版本偶有 [B, L, m]，
                # 静默广播会得到"用错轴聚合"的载荷（形状合法、语义全错），因此显式校正。
                if tuple(w.shape)[1:] == (L, self.n_slots):
                    w = w.transpose(1, 2)
                else:
                    raise ValueError(f"注意力权重形状异常：{tuple(w.shape)}，"
                                     f"期望 [B,{self.n_slots},{L}]")
            # 被 mask 掉的位置权重必须清零再归一化：否则 padding token 会分走一部分
            # 聚合质量，而 h0 在 padding 位上是 0 向量 —— 表现为"直通分支整体变暗"。
            # ---- 尖锐选通（v4 手术二）----
            # 旧写法直接复用语义分支的 softmax 权重：那是**平滑平均**，把 30 个 token
            # 混成一个不对应词表任何实体的"面团"向量。实测同样 33 token 预算，
            # 原生 h0 能到 32.8% EM，而平滑平均出来的潜向量只有 17.2%。
            # 现在对同一份 logits 施极低温度 τ_lex≈0.05，并**逐行互斥**：
            # 第 r 行挑走一个 token 后，后面的行不能再选它 —— m 个槽分别锚到
            # m 个不同的高显著 token（实体/数字），这正是冻结模型能直接读懂的东西。
            w = w.to(dtype=torch.float32).masked_fill(~safe_mask.unsqueeze(1), 0.0)
            logits = torch.log(w.clamp_min(1e-9))
            logits = logits.masked_fill(~safe_mask.unsqueeze(1), -1e4)
            taken = torch.zeros((B, L), dtype=torch.bool, device=logits.device)
            rows = []
            picked_cols = []
            for _r in range(self.n_slots):
                row = logits[:, _r, :].masked_fill(taken, -1e4)
                p_r = torch.softmax(row / max(1e-3, float(self.lex_tau)), dim=-1)
                rows.append(p_r)
                pick = p_r.argmax(dim=-1, keepdim=True)          # [B,1]
                picked_cols.append(pick.squeeze(1))
                taken = taken.scatter(1, pick, True)
            # 选出的是哪几个 token（列号）：符号保真诊断必须能打印"压缩器锚在哪个词上"，
            # 否则"实体有没有进潜层"只能靠猜（见 tools/diag_symbol_fidelity.py）。
            lex_pick = torch.stack(picked_cols, dim=1).tolist()   # [B,m] 列号
            w = torch.stack(rows, dim=1)                          # [B,m,L]
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            wd = h0.dtype if h0.dtype.is_floating_point else p.dtype
            z_lex = torch.einsum("bml,bld->bmd",
                                 w.to(dtype=wd),
                                 h0.to(device=w.device, dtype=wd))
            z_lex = torch.nan_to_num(z_lex, nan=0.0, posinf=0.0, neginf=0.0)
            if bool(empty.any()):
                z_lex = z_lex.masked_fill(empty.view(B, 1, 1), 0.0)

        info = {"n_rows": int(B), "seq_len": int(L),
                "n_tokens": [int(v) for v in mask.sum(dim=1).tolist()],
                "n_empty_rows": int(empty.sum().item()),
                "lex_skip": bool(self.lex_skip),
                "lex_pick": lex_pick}
        return z, z_lex, info


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
                 target_norm: float = LATENT_TARGET_NORM,
                 lex_skip: bool = bool(LATENT_LEX_SKIP),
                 lex_gate_init: float = LATENT_LEX_GATE_INIT,
                 alpha_max: float = LATENT_ALPHA_MAX):
        super().__init__()
        if int(hidden_size) <= 0 or int(bottleneck) <= 0:
            raise ValueError(f"hidden_size/bottleneck 必须为正：{hidden_size}/{bottleneck}")
        if not (float(target_norm) > 0):
            raise ValueError(f"target_norm 必须为正，得到 {target_norm}")
        if not (0.0 < float(alpha_max) <= 1.0):
            raise ValueError(f"alpha_max 必须在 (0,1] 内，得到 {alpha_max}")
        self.hidden_size = int(hidden_size)
        self.bottleneck = int(bottleneck)
        self.alpha_max = float(alpha_max)
        self.net = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.bottleneck),
            nn.SiLU(),
            nn.Linear(self.bottleneck, self.hidden_size),
        )
        # 最后一层零初始化：配合 α 初值 ≈0，保证第 0 步 E_soft 严格等于 z_lex
        # （残差从"什么都没有"开始长，而不是一开始就往原生词向量上叠噪声）。
        # 注意零初始化**必须**搭配非零 α 初值（见 LATENT_LEX_GATE_INIT 的注释）：
        # α 也恰好为 0 时语义分支与门控同时失去梯度，会死锁在"永远只有 z_lex"。
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # 可学习但受限的尺度；初值取真实 token embedding 的每向量范数
        self.out_norm = nn.Parameter(torch.tensor(float(target_norm)))
        # ---- v4：h0 直通的**方向**门控 ----
        # 为什么是"方向混合"而不是"范数相加"：这个类存在的唯一理由是让软 token 的
        # 每向量范数与真实 embedding 对齐（1.05）；直接 `out + α·LN(z_lex)` 会把范数
        # 推到 60+，软 token 立刻退化成 attention sink（本类文档里那组实测数据）。
        # 因此在单位球面上做门控插值：方向由语义/符号两支按 α 竞争，范数仍由受限标量
        # 给出。α→0 时结果与 v3 逐位相同（unit(out)·scale，归一化是幂等的）。
        self.lex_skip = bool(lex_skip)
        # 诊断开关：True 时 forward 会把"残差有多强 / 输出范数多大"记进 self.last_stats。
        # 默认关闭 —— 这些统计要做 .item()，每步都同步会把 4B 前向的空隙填满。
        # 训练脚本在 --diag-every 的那一步打开它，用来回答"α 是不是在把读出拖离词表"。
        self.collect_stats = False
        self.last_stats: Dict[str, float] = {}
        # 门控参数**只在开启直通时注册**。v2/v3 的 ckpt 里根本没有 `lex_gate`，
        # 无条件注册会让 `load_state_dict(strict=True)` 报
        # `Missing key(s) in state_dict: "lex_gate"`，于是所有历史权重都读不进来
        # （2026-11-02 实测：v3 ckpt 加载直接 RuntimeError，评测一行都跑不了）。
        # 旧 ckpt 的 lex_skip=False，本就不该有这份参数；v4 产物照常带。
        if self.lex_skip:
            self.lex_gate = nn.Parameter(torch.tensor(float(lex_gate_init)))

    def scale(self) -> torch.Tensor:
        """当前生效的范数（已夹在 [NORM_MIN, NORM_MAX] 内）。"""
        return self.out_norm.clamp(self.NORM_MIN, self.NORM_MAX)

    def alpha(self) -> torch.Tensor:
        """h0 直通的**残差**系数 α（tanh，初值很小，上限 alpha_max）。

        初值必须**非零但很小**（默认 0.05）：末层零初始化让 res 严格为 0，于是第 0 步
        输出逐元素等于 z_lex（零训练就有 oracle_embed 的起点），但若 α 也恰好为 0，
        ∂e/∂W_last ∝ α 与 ∂e/∂α ∝ res 会同时为零 —— 语义分支与门控一起死锁。
        这是把"残差从零开始长"和"梯度必须能流进语义分支"同时满足的唯一取值方式。

        上限是**约束**而不是装饰：v5 的 α 一路涨到 0.264 而读出 EM 反而退到 11.8%，
        说明残差在偏离词表流形。clamp 之后超过上限的梯度为 0（硬约束），
        残差强度只能通过调整 alpha_max 放开，不能靠训练自己越界。
        """
        if not self.lex_skip:
            return torch.zeros((), device=self.out_norm.device, dtype=self.out_norm.dtype)
        return torch.tanh(self.lex_gate).clamp(-self.alpha_max, self.alpha_max)

    def forward(self, z: torch.Tensor, z_lex: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Z [B,m,d]（+ 可选 z_lex [B,m,d]）-> soft tokens [B,m,d]，范数对齐真实 embedding。"""
        z = _as_3d(z, "z")
        if z.shape[-1] != self.hidden_size:
            raise ValueError(f"z 的最后一维 {z.shape[-1]} != hidden_size={self.hidden_size}")
        p = next(self.parameters())
        out = self.net(z.to(device=p.device, dtype=p.dtype))
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        # 方向保留、范数对齐到真实 embedding 尺度（见类文档里的实测数据）
        #
        # **分母必须夹在 1.0，不能夹在 1e-6**：末层零初始化（见 __init__）使第 0 步
        # ‖out‖≈0，而 d(out/‖out‖)/dout ≈ 1/‖out‖ —— 夹在 1e-6 等于给末层乘上 1e6
        # 的雅可比。v5 实测初始 bridge 分支梯度范数 1.85e10，grad_clip=1.0 之后
        # 实际更新方向几乎完全由这一条病态路径决定（裁剪前/后的比值就是证据）。
        # 夹在 1.0 的含义：‖out‖>=1 时逐位等价于原来的单位化；‖out‖<1 时不再放大，
        # 残差范数自然小于 target_norm（正是"从零开始长"想要的行为）。
        n = out.norm(dim=-1, keepdim=True)
        unit = out / n.clamp_min(1.0)
        res = unit * self.scale().to(device=unit.device, dtype=unit.dtype)
        if z_lex is not None and self.lex_skip:
            # ---- 残差锚定（v4 手术二）----
            # 旧写法把 Bridge(z_sem) 与 z_lex 在单位球面上按 α 混合方向再缩放范数。
            # 那是错的：z_lex 已经是**词表里的真实 embedding**，混合会把它拉离词表方向；
            # 而冻结 Qwen 恰恰只认这种离散、尖锐的词向量（同预算下 32.8% vs 17.2%）。
            # 新写法：z_lex 作主干，Bridge(z_sem) 只作**零初始化残差**补充被截断的谓词
            # 与上下文关系。α=0 时输出逐元素等于 z_lex，天然继承 oracle_embed 的起点。
            lex = _as_3d(z_lex, "z_lex")
            if tuple(lex.shape) != tuple(z.shape):
                if int(lex.numel()) == int(z.numel()):
                    lex = lex.reshape(tuple(z.shape))
                else:
                    raise ValueError(f"z_lex 形状 {tuple(lex.shape)} 必须与 z {tuple(z.shape)} 一致")
            lex = lex.to(device=res.device, dtype=res.dtype)
            ln = lex.norm(dim=-1, keepdim=True)
            valid = ln > 1e-6                      # 空槽（全零直通向量）不参与
            a = self.alpha().to(device=res.device, dtype=res.dtype)
            e = lex + a * res
            e = torch.where(valid, e, res)          # 空槽退回纯语义残差
            tgt = self.scale().to(device=res.device, dtype=res.dtype)
            en = e.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            fac = (tgt / en).clamp(self.NORM_MIN, self.NORM_MAX)
            if self.collect_stats:
                with torch.no_grad():
                    _ln = float(ln[valid].mean()) if bool(valid.any()) else 0.0
                    _rn = float((a * res).norm(dim=-1).mean())
                    self.last_stats = {
                        "alpha": float(a), "scale": float(tgt),
                        "lex_norm": _ln, "res_norm": _rn,
                        "res_ratio": (_rn / _ln) if _ln > 0 else 0.0,
                        "e_norm": float(e.norm(dim=-1).mean()),
                        "net_out_norm": float(n.mean()),
                        "w_last": float(self.net[-1].weight.norm()),
                    }
            return e * fac
        if self.collect_stats:
            with torch.no_grad():
                self.last_stats = {"alpha": 0.0, "scale": float(self.scale()),
                                   "lex_norm": 0.0, "res_norm": 0.0, "res_ratio": 0.0,
                                   "e_norm": float(res.norm(dim=-1).mean()),
                                   "net_out_norm": float(n.mean()),
                                   "w_last": float(self.net[-1].weight.norm())}
        return res


def relay_bridge_tokens(h0_slot: torch.Tensor, mask_slot: Any = None,
                        q_tok: Optional[torch.Tensor] = None,
                        k: int = 4,
                        h18_slot: Optional[torch.Tensor] = None) -> torch.Tensor:
    """从某个槽位的 token 里挑出**新信息**（与原问题最不相似）的 k 个，用于第二跳接力。

    为什么是"最不相似"：HotpotQA 的两跳结构是"第一条证据句尾出现一个新实体 ->
    第二条证据正好在讲这个实体"。所以第二跳要沿着**原问题里没有出现过**的词去找。
    旧实现直接 `sim.topk()` 取了**最相似**的 k 个 —— 方向完全反了：它把问题里已有的
    词又塞回查询，第二跳于是继续围着第一跳打转，这正是"首跳 93%、完整召回 24%"的机制。

    两个必须处理的坑（只把 topk 换成 largest=False 是不够的）：
      1. padding 位的相似度是 -1e4，反向选择会优先选中 padding —— 先用 mask 排除；
      2. **标点与功能词的相似度同样偏低**，单纯取"最不相似"会把它们选进来。本层拿不到
         token 文本，因此用 h18 偏离度当"信息量"代理：score = (1 - sim) × salience，
         只有在有效 token 里同时"不像问题"且"有内容"的才入选。这是**近似**处理，
         不是文本级标点过滤（要精确过滤需要把 token 文本传进来）。

    返回 [n, d]（n <= k），设备/精度跟随 h0_slot。
    """
    h = torch.as_tensor(h0_slot)
    if h.dim() != 2:
        raise ValueError(f"relay_bridge_tokens 需要 [T,d]，得到 {tuple(h.shape)}")
    T = int(h.shape[0])
    if T == 0:
        return h[:0]
    valid = torch.ones((T,), dtype=torch.bool, device=h.device)
    if mask_slot is not None:
        m = torch.as_tensor(mask_slot).to(h.device)
        if m.numel() == T:
            valid = m.bool()
    if q_tok is None or not bool(valid.any()):
        return h[valid][: max(1, int(k))]
    qt = torch.as_tensor(q_tok).to(device=h.device, dtype=torch.float32)
    if qt.dim() == 3:
        qt = qt[0]
    hf = h.to(torch.float32)
    mu = hf[valid].mean(dim=0, keepdim=True)
    hn = torch.nn.functional.normalize(hf - mu, dim=-1)
    qn = torch.nn.functional.normalize(qt - mu, dim=-1)
    sim = (hn @ qn.t()).max(dim=-1).values                     # [T] 与问题的最大相似度
    sim = sim.masked_fill(~valid, 1e4)                          # pad 排到最后
    n_valid = int(valid.sum().item())
    kk = max(1, min(int(k), n_valid))
    # 显著性（信息量）代理：h18 偏离该槽位均值方向的程度。
    sal = None
    if h18_slot is not None:
        h18 = torch.as_tensor(h18_slot).to(device=h.device, dtype=torch.float32)
        if h18.dim() == 2 and int(h18.shape[0]) == T:
            mu18 = h18[valid].mean(dim=0, keepdim=True)
            sal = (h18 - mu18).norm(dim=-1)
            s_max = float(sal[valid].max().clamp_min(1e-6))
            sal = (sal / s_max).clamp(0.0, 1.0)
    # score 越大越该被选：不像问题（1-sim）× 有内容（salience）
    score = (1.0 - sim.clamp(-1.0, 1.0))
    if sal is not None:
        score = score * (0.25 + sal)          # 0.25 下限：显著性低也不至于直接归零
    score = score.masked_fill(~valid, -1e4)
    score = score.masked_fill(sim >= 0.9, -1e4)   # 与问题近乎相同的不要
    order = score.argsort(descending=True)
    pick = order[:kk]
    pick = pick[score[pick] > -1e3]
    if int(pick.numel()) == 0:
        pick = order[:kk]
    return h[pick]


# ----------------------------- 2.5) 潜空间内容寻址（Router） -----------------------------
LATENT_D_KEY = _env_int("LATENT_D_KEY", 256)


def pool_slot_keys(payloads: Sequence[torch.Tensor]) -> torch.Tensor:
    """per-slot payload 列表 -> **单元级寻址键** [K, m, d]（块轴 attention pooling）。

    这是"键-载荷解耦"的第一半。旧实现让 `stack_payloads` 的块轴**均值**同时充当
    寻址键和读出载荷：一条被 `--max-unit-tokens` 切成 C 块的句子，均值既把多块证据
    压成"平均语义"去寻址，又把长句内容在读出前抹平。两件事用一个池化，等于用一个
    通道承担两种粒度 —— 这是设计缺陷，不是调参问题。

    这里的池化**无新增参数**（用块均值当查询做 softmax 加权），因此不改动任何
    checkpoint 的参数集合；要换成可学习查询另开一轮，别偷偷加参数让旧 ckpt 静默错位。
    """
    if not payloads:
        raise ValueError("pool_slot_keys 收到空列表")
    outs = []
    for p in payloads:
        z = p if torch.is_tensor(p) else torch.as_tensor(p)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        if z.dim() != 3:
            raise ValueError(f"payload 必须是 [k,m,d]，得到 {tuple(z.shape)}")
        q = z.mean(dim=0)                                   # [m,d] 块轴查询
        s = (z * q.unsqueeze(0)).sum(-1).sum(-1) / (float(z.shape[-1]) ** 0.5)
        w = torch.softmax(s.float(), dim=0).to(z.dtype)      # [k]
        outs.append(torch.einsum("k,kmd->md", w, z))
    return torch.stack(outs, dim=0)


def pack_slot_reads(payloads: Sequence[torch.Tensor]) -> list:
    """per-slot payload 列表 -> **块级读出载荷**，每个槽保留全部 C_i*m 个向量 [C_i*m, d]。

    这是"键-载荷解耦"的第二半，也是评测侧的真 bug 修复：`stack_payloads` 的均值
    会把"哪个专有名词落在哪一块"直接抹掉，于是 `--max-unit-tokens 32` 切出来的
    多块长句在评测时等于只剩一个平均向量。训练侧（`unit_encode_text` 多块 cat）
    保留块，评测侧求均值 —— 两侧载荷分布不一致，读出的信息量凭空少一半。
    """
    reads = []
    for p in payloads:
        z = p if torch.is_tensor(p) else torch.as_tensor(p)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        reads.append(z.reshape(-1, z.shape[-1]))
    return reads


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

    def encode_keys(self, slot_z: torch.Tensor,
                    kfeat: Optional[Dict[str, Any]] = None,
                    q_tok: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """slot_z: [B, K, m, d] -> (k_in [B,K,d_key], k_out [B,K,d_key])，均已归一化。

        q_tok 只在 HybridLatentRouter 里用（第 2 跳的新信息抑制 + 词级接力）；
        基类为了调用点统一也接受它，但直接忽略。

        kfeat 见 scores 的说明：基类不用，HybridLatentRouter 用它把键的输入从
        "压缩器潜载荷"换成"冻结模型的 h0/h18 特征"。
        """
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

    def scores(self, q_vec: torch.Tensor, k_in: torch.Tensor,
               q_tok: Optional[torch.Tensor] = None,
               kfeat: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """集合打分 s_k = <q, k_in_k> / tau，形状 [B, K]（与 K、与顺序无关）。

        q_tok / kfeat 是给 HybridLatentRouter 的 h0 Token-MaxSim 底盘留的口子；
        基类（纯稠密内积）用不到，接收后忽略 —— 这样上层调用点只有一份，
        不会出现"v3 走一条、v4 走另一条"的双实现漂移。
        """
        tau = self.log_temp.exp().clamp_min(1e-4)
        return torch.einsum("bd,bkd->bk", q_vec, k_in.to(q_vec.dtype)) / tau

    def forward(self, q_feat: torch.Tensor, slot_z: torch.Tensor,
                hard: bool = False, target_idx: Optional[torch.Tensor] = None,
                kfeat: Optional[Dict[str, Any]] = None,
                q_tok: Optional[torch.Tensor] = None
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
        k_in, k_out = self.encode_keys(slot_z, kfeat=kfeat)
        logits = self.scores(q_vec, k_in, q_tok=q_tok, kfeat=kfeat)
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

    def addr_loss_set(self, q_feat: torch.Tensor, slot_z: torch.Tensor,
                      gold_idx: Sequence[int], n_hops: int = 2,
                      kfeat: Optional[Dict[str, Any]] = None,
                      q_tok: Optional[torch.Tensor] = None):
        """无序第一跳 + 双向金标条件第二跳的寻址损失（**从数学上消除 loss_addr=inf**）。

        旧实现的死法是确定的：它把"金标证据链按 evidence 顺序"当成跳序，第 1 跳用
        argmax 硬选、第 2 跳用 `-inf` 屏蔽第 1 跳已选槽。只要 argmax 先落在 g2 上，
        第 2 跳的目标 g2 自己就被屏蔽 —— logit=-inf → CE=inf → 梯度无意义。
        10k 训练里 ~63% 的步数是 inf，寻址头整条没被训练，下游读对率 6%（随机水平）
        就是它的直接后果。HotpotQA 的 supporting_facts 本身也**不含推理顺序**
        （comparison 题无序、bridge 题按文档位置排），按顺序监督从一开始就是错的。

        现在：第 1 跳对金标集合做负对数边缘化（顺序无关）；第 2 跳显式构造两条
        金标引导分支，掩码只屏蔽"本分支用掉的那一跳"，目标槽永远不可能落在掩码里。
        槽位内容对寻址损失 `detach()`：寻址梯度只更新打分头，不把压缩器内容拖走。
        """
        # q_tok 必须传进去：HybridLatentRouter.encode_keys 只有拿到它才会算"与问题的
        # 相似度"并给 k_out 做新信息抑制。旧写法不传，于是这套抑制在训练期从未被优化，
        # 推理期却全程生效 —— 训练与推理跑的是两套不同的键（2026-11-03 定位）。
        k_in, k_out = self.encode_keys(slot_z.detach() if torch.is_tensor(slot_z) else slot_z,
                                       kfeat=kfeat, q_tok=q_tok)
        q_vec = self.encode_query(q_feat)
        K = int(slot_z.shape[0]) if slot_z.dim() == 3 else int(slot_z.shape[1])
        golds = [int(g) for g in gold_idx if 0 <= int(g) < K]
        if not golds:
            return None, {"why": "no_gold"}
        logits1 = self.scores(q_vec, k_in, q_tok=q_tok, kfeat=kfeat)   # [1,K]
        dev = logits1.device
        if int(n_hops) <= 1 or len(golds) < 2:
            tgt = torch.tensor([golds[0]], device=dev)
            return F.cross_entropy(logits1, tgt), {"hop1": float(
                F.cross_entropy(logits1, tgt).detach()), "golds": golds}
        gset = torch.tensor(golds, device=dev, dtype=torch.long)
        # **必须**先 log_softmax 再 logsumexp：直接对原始 logits 做 logsumexp 得到的是
        # "金标 logit 的负 log-sum-exp"，不是 -log P(G|q)。那样第一跳只要把金标 logit
        # 一起抬高就能降损失，根本不必学会把金标与干扰槽分开，损失甚至能一路变成大负数；
        # 再叠上可学习温度（log_temp），这个目标会被彻底玩坏。集合边缘化必须落在概率上。
        logp1 = F.log_softmax(logits1, dim=-1)
        l_hop1 = -torch.logsumexp(logp1[0, gset], dim=0)         # -log Σ_{g∈G} P(g|q)
        terms = []
        for a in golds:
            for b in golds:
                if a == b:
                    continue
                mask = torch.zeros_like(logits1, dtype=torch.bool)
                mask[0, a] = True                                # 只屏蔽本分支用掉的那一跳
                q2 = torch.nn.functional.normalize(q_vec + k_out[:, a], dim=-1)
                # 第二跳的查询 token 与推理**完全一致**：原问题 token + 本分支选中槽里
                # "与原问题最不相似"的桥接 token。旧实现直接用原问题的 q_tok，等于训练
                # 一个不含接力的第二跳、却去评测带接力的第二跳 —— 这是召回平台化的第二病根。
                qtok2 = q_tok
                if q_tok is not None and kfeat is not None:
                    try:
                        _mk = kfeat.get("mask")
                        _relay = relay_bridge_tokens(
                            kfeat["h0"][int(a)],
                            (torch.as_tensor(_mk)[int(a)] if _mk is not None else None),
                            q_tok, k=int(LATENT_RELAY_TOKENS),
                            h18_slot=(kfeat["h18"][int(a)] if "h18" in kfeat else None))
                        if int(_relay.numel()) > 0:
                            _relay = _relay.to(device=logits1.device).unsqueeze(0)
                            _base = torch.as_tensor(q_tok).to(
                                device=_relay.device, dtype=_relay.dtype)
                            qtok2 = torch.cat([_base, _relay], dim=1)
                    except Exception as _e:
                        print(f"[relay] 训练期接力跳过：{_e}", flush=True)
                logits2 = self.scores(q2, k_in, q_tok=qtok2, kfeat=kfeat).masked_fill(
                    mask, float("-inf"))
                terms.append(F.cross_entropy(logits2, torch.tensor([b], device=dev)))
        l_hop2 = torch.stack(terms).mean()
        loss = l_hop1 + l_hop2
        return loss, {"hop1": float(l_hop1.detach()), "hop2": float(l_hop2.detach()),
                      "golds": golds, "n_branch": len(terms)}

    def forward_hops(self, q_feat: torch.Tensor, slot_z: torch.Tensor,
                     n_hops: int = 2, hard: bool = False,
                     reads: Optional[Sequence[torch.Tensor]] = None,
                     force_first: Optional[int] = None,
                     kfeat: Optional[Dict[str, Any]] = None,
                     q_tok: Optional[torch.Tensor] = None,
                     topk_per_hop: int = 1,
                     lex_reads: Optional[Sequence[torch.Tensor]] = None) -> Dict[str, Any]:
        """多跳前向：第 1 跳选槽 -> q' = norm(q + k_out[选中槽]) -> 第 2 跳再选槽。

        修的是一个**架构级缺陷**，不是调参问题：
            旧训练路径只调 `router(q_feat, slot_z)`，拿回的 `k_out` 直接被丢掉，
            只有第 1 跳的 `loss_addr` 进图；推理侧 `chain()` 却用
            `q' = norm(q + k_out[idx])` 接力找下一跳 —— 于是 `k_out_net`
            在正常训练里**没有任何梯度**，第二跳实际由一个随机投影在寻址。
            HotpotQA 的现象正对得上：第一跳寻址在涨（read_acc 35%→50%）、
            答案准确率不动。单跳监督永远学不会"第二跳该读哪条事实"。

        topk_per_hop（v4 手术）：每跳取 top-b 个槽。`b=1` 时与本文件此前版本
        **逐位等价**；`b=2, n_hops=2` 时一次读出 4 个槽 —— 这是覆盖 HotpotQA 里
        161/500 条"金标证据 >2 句"的题的唯一物理手段（2 槽的读出带宽装不下 3 句话）。
        4 槽 = 16 个软 token（m=4），仍远小于 BM25-top4 的 151 个明文 token。
        """
        q_vec0 = self.encode_query(q_feat)
        # q_tok 一起传进去：HybridLatentRouter 用它给 k_out 做新信息抑制（见 encode_keys）
        k_in, k_out = self.encode_keys(slot_z, kfeat=kfeat, q_tok=q_tok)
        z = slot_z if slot_z.dim() == 4 else slot_z.unsqueeze(0)
        z = z.to(device=q_vec0.device, dtype=q_vec0.dtype)
        K = int(z.shape[1])
        n = max(1, min(int(n_hops), K))
        b = max(1, min(int(topk_per_hop), K))
        max_picks = max(1, min(int(n) * int(b), K))
        # 用 Python set 记录已选槽，**不能**用张量就地写 used[idx]=True：
        # 那个 bool 张量被上一跳的 masked_fill 引用进了计算图，就地改它会触发
        #   RuntimeError: one of the variables needed for gradient computation
        #   has been modified by an inplace operation: [CPUBoolType [1, 8]] version 2
        # （这正是 10k 正训一 backward 就会炸的地方，冒烟先撞出来了）。
        used_idx: set = set()
        z_list: List[torch.Tensor] = []
        lex_list: List[torch.Tensor] = []
        logits_list: List[Any] = []
        picks: List[int] = []
        # 分支状态：(查询向量 [1,dk], 该分支已选路径)。b=1 时分支数恒为 1，退化为原实现。
        # 每条 beam 必须**各自**维护查询 token：topk_per_hop=2 时第 1 跳会选 A、B 两个
        # 起点，A 的新实体和 B 的新实体完全不同。旧实现只从 take[0] 取一次桥接词给所有
        # 分支共用 —— B 分支于是拿着 A 的新实体去找 B 的下一跳，这一路必然错。
        branches: List[Tuple[torch.Tensor, Tuple[int, ...], Optional[torch.Tensor]]] = [
            (q_vec0, (), q_tok)]
        for hop in range(n):
            if len(picks) >= max_picks:
                break
            # 1) 每个分支各自打分，收集候选 (logit, 槽号, 分支号)
            cands: List[Tuple[float, int, int]] = []
            gold_first = (hop == 0 and force_first is not None
                          and 0 <= int(force_first) < K)
            for bi, (qv, _path, qtok_b) in enumerate(branches):
                if gold_first:
                    cands.append((float("inf"), int(force_first), bi))
                    break
                logits = self.scores(qv, k_in, q_tok=qtok_b, kfeat=kfeat)     # [1,K]
                if used_idx:
                    mask = torch.zeros_like(logits, dtype=torch.bool)
                    mask[0, sorted(used_idx)] = True
                    logits = logits.masked_fill(mask, float("-inf"))
                vals, idxs = logits[0].topk(min(int(b) + len(used_idx), K))
                for v, i in zip(vals.tolist(), idxs.tolist()):
                    if i in used_idx:
                        continue
                    cands.append((float(v), int(i), bi))
            if not cands:
                break
            # 2) 全局取"至多 b 个**新**槽"：同一个槽被多个分支选中时只读一次
            cands.sort(key=lambda x: (-x[0], x[1]))
            take: List[Tuple[int, int]] = []
            for _v, i, bi in cands:
                if i in used_idx:
                    continue
                used_idx.add(i)
                take.append((i, bi))
                if len(take) >= b or (len(picks) + len(take)) >= max_picks:
                    break
            if not take:
                break
            # 3) 逐槽建图：STE 的软期望必须用"该槽被选中那一刻"的 logits
            new_branches: List[Tuple[torch.Tensor, Tuple[int, ...],
                                     Optional[torch.Tensor]]] = []
            for i, bi in take:
                qv, path, qtok_b = branches[bi]
                idx = torch.tensor([int(i)], device=q_vec0.device)
                z_hard = z[0, idx]                                       # [1,m,d]
                logits_here = None
                if not (hard or not self.training):
                    # 重算这一分支的 logits 只为取软权重（K 很小，代价可忽略）。
                    # 千万不要复用一个已被 masked_fill 就地改过的张量：那会让
                    # 计算图里的旧引用失效（见上面 used_idx 的注释）。
                    lg = self.scores(qv, k_in, q_tok=qtok_b, kfeat=kfeat)
                    if path:
                        m2 = torch.zeros_like(lg, dtype=torch.bool)
                        m2[0, sorted(set(path))] = True
                        lg = lg.masked_fill(m2, float("-inf"))
                    w = torch.softmax(lg, dim=-1)
                    # einsum "bk,bkmd->bmd" 出来已经是 [1,m,d]（b 维保留），
                    # 旧代码又 unsqueeze(0) 成 [1,1,m,d]；与 [1,m,d] 的 z_hard 相加时
                    # 广播成四维，一路传到 bridge(z) 才炸：
                    #   ValueError: z 必须是 [B,L,d] 或 [L,d]，得到 (1, 1, 4, 64)
                    # 单跳路径（同文件 497 行）没有这次 unsqueeze —— 两处必须一致。
                    z_soft = torch.einsum("bk,bkmd->bmd", w, z.detach())
                    z_hard = z_hard + (z_soft - z_soft.detach())
                    logits_here = lg
                logits_list.append(logits_here)
                # 读出载荷优先用**块级保真**版本（reads[idx]），只有拿不到时才退回块均值载荷。
                # 键（打分）走的是单元级池化键，载荷走的是该槽全部 C_i*m 个块 —— 这正是
                # "键-载荷解耦"在推理路径上的落点。
                if reads is not None and int(i) < len(reads):
                    zr = reads[int(i)]
                    z_list.append(zr if zr.dim() == 3 else zr.unsqueeze(0))
                else:
                    z_list.append(z_hard)
                if lex_reads is not None and int(i) < len(lex_reads):
                    lr = lex_reads[int(i)]
                    lex_list.append(lr if lr.dim() == 3 else lr.unsqueeze(0))
                picks.append(int(i))
                # 该分支自己的词级接力：从**它自己**选中的槽里取"与问题最不相似"的
                # 若干 token 的 h0，拼进它自己的查询 token 集合（下一跳用）。
                qtok_new = qtok_b
                if hop == 0 and kfeat is not None and qtok_b is not None:
                    try:
                        _h0 = torch.as_tensor(kfeat["h0"])
                        _mk = kfeat.get("mask")
                        _h18 = torch.as_tensor(kfeat["h18"]) if "h18" in kfeat else None
                        _relay = relay_bridge_tokens(
                            _h0[int(i)],
                            (torch.as_tensor(_mk)[int(i)] if _mk is not None else None),
                            qtok_b, k=int(LATENT_RELAY_TOKENS),
                            h18_slot=(_h18[int(i)] if _h18 is not None else None))
                        if int(_relay.numel()) > 0:
                            _relay = _relay.to(device=q_vec0.device).unsqueeze(0)
                            _base = torch.as_tensor(qtok_b).to(
                                device=_relay.device, dtype=_relay.dtype)
                            qtok_new = torch.cat([_base, _relay], dim=1)
                    except Exception as _e:                       # 接力失败不致命，但要留痕
                        print(f"[relay] 分支 {bi} 词级接力跳过：{_e}", flush=True)
                # 查询更新**留在计算图里**：这是 k_out_net 唯一能拿到梯度的通路
                new_branches.append((torch.nn.functional.normalize(
                    qv + k_out[:, int(i)], dim=-1), tuple(path) + (int(i),), qtok_new))
            # 分支数上限 = b：b=2、3 跳时若不截断会指数膨胀（每跳 ×b）
            branches = new_branches[:b]
        return {"z_list": z_list, "lex_list": (lex_list or None),
                "logits_list": logits_list, "picks": picks,
                "k_in": k_in, "k_out": k_out,
                "topk_per_hop": int(b), "n_hops": int(n)}

    @torch.no_grad()
    def chain(self, q_feat: torch.Tensor, slot_z: torch.Tensor, hops: int = 2,
              kfeat: Optional[Dict[str, Any]] = None,
              q_tok: Optional[torch.Tensor] = None,
              topk_per_hop: int = 1) -> List[int]:
        """多跳推理：直接复用训练用的 forward_hops（同一套打分与查询更新规则）。

        旧实现自己维护了一份循环，与训练侧根本不是同一条路径 ——
        推理用 k_out、训练不监督 k_out，两份代码让这个错误一直看不出来。
        """
        return self.forward_hops(q_feat, slot_z, n_hops=hops, hard=True, kfeat=kfeat,
                                 q_tok=q_tok, topk_per_hop=topk_per_hop)["picks"]


class HybridLatentRouter(LatentRouter):
    """混合双塔寻址器（v4）：**寻址特征与压缩通路解耦**。

    为什么必须解耦（实测证据）
    --------------------------
    v3 的 k_in/k_out 是用"压缩器输出的 m×d 潜向量"摊平后投影出来的。于是寻址头的
    输入分布随压缩器一起漂移：压缩器还在学的时候，键每天都在变；而把整句 4 个向量
    压成**一个** k_out，等于要求"句首主语"和"句尾桥接实体"共用一个向量 ——
    多跳的 q' = norm(q + k_out) 于是退化成噪声叠加。v3 在 500 题上的完整证据召回
    5.6%、比 v2 的 6.0% 还低，而 BM25-top4 不训练就有 36.0%，说明问题不在"没学好"，
    而在"学的东西里有损且不稳定"。

    两个分支
    --------
    1. 稀疏精确分支（**免训练**）：问题 token 与单元 token 在归一化 h0 上的词级
       MaxSim。h0 是词形嵌入，专有名词/数字在这里几乎不受上下文平滑影响。这一项
       让 Router 在**第 0 步**就有 ≈BM25 的底盘召回，而不是从 5.6% 的盲选开始。
    2. 稠密多跳分支（可学习）：cat(h0,h18) 的 token 特征经两组注意力池化出
       k_in（聚焦句首/锚点）与 k_out（聚焦句尾/专名），再做内积。
    最终 s = γ·MaxSim/τ_ms + (1-γ)·<q,k_in>/τ，γ 可学习但被 sigmoid 夹在 (0,1)。

    键的输入是**冻结模型特征**，与压缩器无关：即使压缩器今天还没收敛，寻址也不会
    跟着一起坏。这也是"两个通路各管一件事"的落地方式。
    """

    def __init__(self, d_model: int, n_slots: int = LATENT_M, d_key: int = LATENT_D_KEY,
                 init_temp: float = 0.07, gamma: float = LATENT_ADDR_GAMMA,
                 ms_tau: float = LATENT_MS_TAU,
                 anchor_boost: float = LATENT_ANCHOR_BOOST,
                 anchor_tokens: int = LATENT_ANCHOR_TOKENS,
                 novelty_lambda: float = LATENT_NOVELTY_LAMBDA):
        super().__init__(d_model, n_slots, d_key, init_temp)
        if not (0.0 < float(gamma) < 1.0):
            raise ValueError(f"gamma 必须落在 (0,1)，得到 {gamma}")
        self.ms_tau = float(ms_tau)
        d2 = int(d_model) * 2
        # —— 稠密多跳分支：两组独立的注意力池化头（入键 / 出键）——
        self.pool_in = nn.Sequential(nn.Linear(d2, self.d_key), nn.SiLU(),
                                     nn.Linear(self.d_key, 1))
        self.pool_out = nn.Sequential(nn.Linear(d2, self.d_key), nn.SiLU(),
                                      nn.Linear(self.d_key, 1))
        self.proj_in = nn.Linear(d2, self.d_key)
        self.proj_out = nn.Linear(d2, self.d_key)
        # 位置先验（两个标量）：入键压句首的锚点/标题，出键压句尾的桥接实体。
        # 用**学习到的偏置**而不是写死"取第 0 个 token"：单元文本总以 "[anchor] " 开头，
        # 但分块后第 0 块以外的块并不以锚点开头，硬取第 0 位会在长单元上取错位置。
        self.bias_in = nn.Parameter(torch.zeros(2))
        self.bias_out = nn.Parameter(torch.zeros(2))
        # —— 稀疏精确分支：γ（MaxSim 占比）——
        self.gamma_logit = nn.Parameter(torch.tensor(float(math.log(gamma / (1 - gamma)))))
        self.anchor_boost = float(anchor_boost)
        self.anchor_tokens = int(anchor_tokens)
        self.novelty_lambda = float(novelty_lambda)

    # ---- 打分 ----
    def gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.gamma_logit)

    @staticmethod
    def maxsim(q_tok: torch.Tensor, k_tok: torch.Tensor, k_mask: Any) -> torch.Tensor:
        """词级 MaxSim：mean_{q token}( max_{unit token}( cos ) )，返回 [B,K]。

        "max 而不是 mean"是关键：一句事实里只有一两个 token 与问题里的实体对应，
        平均池化会把它们稀释掉（这正是 v3 稠密内积丢掉稀有实体匹配的机制）。
        """
        q = torch.as_tensor(q_tok)
        k = torch.as_tensor(k_tok)
        if q.dim() == 3:
            q = q[0]
        if k.dim() == 2:
            k = k.unsqueeze(0)
        K, T, _d = (int(v) for v in k.shape)
        # kfeat 可能是缓存在 CPU 上的（省显存），q_tok 在 GPU —— 必须显式对齐设备，
        # 否则 einsum 直接 "Expected all tensors to be on the same device"。
        # 统一在 float32 上算相似度（bf16 下 normalize 再点积的误差会淹没 MaxSim 的区分度）。
        dev = q.device
        qf = q.to(device=dev, dtype=torch.float32)
        kf = k.to(device=dev, dtype=torch.float32)
        m_b = None
        if k_mask is not None:
            m_b = torch.as_tensor(k_mask).to(device=dev).bool()
            if m_b.dim() == 1:
                m_b = m_b.unsqueeze(0)
            if tuple(m_b.shape) != (K, T):
                m_b = m_b.expand(K, T)
        # ---- 去均值（v4 手术三）----
        # Qwen 的词嵌入空间是**各向异性**的：不减去候选 token 的均值方向，
        # what/the/of 这类功能词与任何 token 的余弦都偏高，把稀有实体（人名/年份）
        # 的匹配分淹没。减去当前样本候选 token 的均值，等于把"公共方向"去掉，
        # 只留下区分度。这只用样本内的统计量，不引入外部资源。
        if m_b is not None and bool(m_b.any()):
            mu = kf[m_b].mean(dim=0, keepdim=True)                  # [1,d]
        else:
            mu = kf.reshape(-1, kf.shape[-1]).mean(dim=0, keepdim=True)
        qf = qf - mu
        kf = kf - mu
        qn = torch.nn.functional.normalize(qf, dim=-1)
        kn = torch.nn.functional.normalize(kf, dim=-1)
        # 注意下标：查询侧是 q、槽内 token 侧是 s。旧写法 "td,ktd->tkt" 把 t 同时
        # 用作"查询 token"和"槽内 token"，输出里 t 出现两次 → einsum 直接 RuntimeError
        # （这条分支只在 hybrid router + MaxSim 打开时走到，单测才第一次执行到）。
        sim = torch.einsum("qd,ksd->qks", qn, kn)                          # [Tq,K,T]
        if k_mask is not None:
            m = torch.as_tensor(k_mask).to(sim.device).bool()
            if m.dim() == 1:
                m = m.unsqueeze(0)
            sim = sim.masked_fill(~m.unsqueeze(0), float("-inf"))
        best = sim.max(dim=-1).values                                      # [Tq,K]
        best = torch.nan_to_num(best, neginf=-1.0)
        # ---- IDF 加权（v4 手术三）----
        # 每个问题 token 按"在多少个候选单元里能找到它"降权：df 大 = 到处都有的
        # 功能词，权重趋近下限；df=1 = 只在一个单元出现的专有名词，权重最高。
        # 这是免训练的（不需要语料，直接在当前候选集内算），和 BM25 的 idf 同源。
        df = (best > 0.5).to(torch.float32).sum(dim=1)                     # [Tq]
        w_idf = torch.log((float(K) + 1.0) / (df + 1.0)) + 1.0
        w_idf = w_idf / w_idf.mean().clamp_min(1e-6)
        return (best * w_idf.unsqueeze(1)).mean(dim=0).unsqueeze(0)        # [1,K]

    def encode_keys(self, slot_z: torch.Tensor,
                    kfeat: Optional[Dict[str, Any]] = None,
                    q_tok: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """优先用**冻结特征** kfeat 出键；拿不到时才退回潜载荷（与基类同路径）。

        q_tok（v4 手术三）：给定问题 token 时，出键 k_out 会做**新信息抑制** ——
        把与该问题高度相似的 token 压下去，让 k_out 聚焦"第 1 跳槽位里新出现的
        桥接实体"。这正是第 2 跳要找的东西（HotpotQA 的第 2 跳段落标题通常就是
        第 1 跳句尾的那个人名/地名）。
        """
        if kfeat is None:
            return super().encode_keys(slot_z)
        h0 = torch.as_tensor(kfeat["h0"])
        h18 = torch.as_tensor(kfeat["h18"])
        if h0.dim() != 3 or h18.dim() != 3:
            raise ValueError(f"kfeat 的 h0/h18 必须是 [K,T,d]，得到 {tuple(h0.shape)}/"
                             f"{tuple(h18.shape)}")
        if tuple(h0.shape) != tuple(h18.shape):
            raise ValueError("kfeat 的 h0/h18 形状不一致")
        K, T, d = (int(v) for v in h0.shape)
        if d != self.d_model:
            raise ValueError(f"kfeat 的 hidden={d} 与 router.d_model={self.d_model} 不一致")
        p = next(self.parameters())
        m = kfeat.get("mask")
        mask = (torch.ones((K, T), dtype=torch.bool, device=p.device) if m is None
                else torch.as_tensor(m).to(p.device).bool())
        x = torch.cat([h0.to(device=p.device, dtype=p.dtype),
                       h18.to(device=p.device, dtype=p.dtype)], dim=-1)     # [K,T,2d]
        # 位置先验：第 1 个有效 token / 最后一个有效 token
        ar = torch.arange(T, device=p.device).unsqueeze(0)
        first = mask & (ar == 0)
        last_pos = (mask.to(torch.long) * ar).max(dim=1, keepdim=True).values
        last = mask & (ar == last_pos)
        neg_inf = torch.finfo(x.dtype).min
        # 标题锚点增益（免训练）：第 1 个有效 token 是 "[anchor] ..." 的标题，
        # 两跳链正是靠标题串起来的，给它一个固定倍数（默认 1.3，即 log 空间 +0.26）。
        _anchor_bonus = math.log(max(1.0, float(self.anchor_boost)))
        # 逐 token 与问题的相似度（用于出键的新信息抑制）。在去均值后的 h0 上算，
        # 与 MaxSim 同一条度量，避免"寻址用一种相似度、出键用另一种"。
        tok_qsim = None
        if q_tok is not None:
            qt = torch.as_tensor(q_tok).to(device=p.device, dtype=torch.float32)
            if qt.dim() == 3:
                qt = qt[0]
            h0f = h0.to(device=p.device, dtype=torch.float32)
            mu = (h0f[mask].mean(dim=0, keepdim=True) if bool(mask.any())
                  else h0f.reshape(-1, h0f.shape[-1]).mean(dim=0, keepdim=True))
            hn = torch.nn.functional.normalize(h0f - mu, dim=-1)
            qn_ = torch.nn.functional.normalize(qt - mu, dim=-1)
            tok_qsim = torch.einsum("ksd,qd->ksq", hn, qn_).max(dim=-1).values  # [K,T]

        # 标题区：单元文本以 "[anchor] ..." 开头，标题会被切成多个 subword，
        # 只给第 0 位加成 ≈ 没加成（这正是"anchor_boost 名义 1.3、实际没生效"的原因）。
        # 这里给**前 anchor_tokens 个有效 token** 一起加成。
        _rank = torch.cumsum(mask.to(torch.long), dim=1) - 1
        head = mask & (_rank < max(1, int(self.anchor_tokens)))

        def _pool(net: nn.Sequential, proj: nn.Linear, bias: nn.Parameter,
                  novelty: bool = False):
            a = net(x).squeeze(-1)                                          # [K,T]
            a = a + bias[0] * first.to(a.dtype) + bias[1] * last.to(a.dtype)
            a = a + _anchor_bonus * head.to(a.dtype)                        # 标题区
            if novelty and tok_qsim is not None and float(self.novelty_lambda) > 0:
                # 与问题越像的 token 越该被压：第 2 跳要的是**新**实体
                a = a - float(self.novelty_lambda) * tok_qsim.to(a.dtype)
            a = a.masked_fill(~mask, neg_inf)
            w = torch.softmax(a.float(), dim=-1).to(x.dtype)                # [K,T]
            z = torch.einsum("kt,ktd->kd", w, x)
            return torch.nn.functional.normalize(proj(z), dim=-1)

        k_in = _pool(self.pool_in, self.proj_in, self.bias_in)
        k_out = _pool(self.pool_out, self.proj_out, self.bias_out, novelty=True)
        return k_in.unsqueeze(0), k_out.unsqueeze(0)

    def scores(self, q_vec: torch.Tensor, k_in: torch.Tensor,
               q_tok: Optional[torch.Tensor] = None,
               kfeat: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        dense = super().scores(q_vec, k_in)
        if q_tok is None or kfeat is None:
            return dense
        ms = self.maxsim(q_tok, kfeat["h0"], kfeat.get("mask")).to(dense.dtype)
        g = self.gamma().to(dense.dtype)
        # MaxSim ∈ [-1,1]，要变成 logit 必须给逆温度；用固定 τ_ms 而不是可学习的 log_temp，
        # 因为 log_temp 已被稠密分支用来调温度 —— 两者共用一个参数会让"混合比例"
        # 和"尖锐度"纠缠在一起，任何一方都学不动。
        return g * (ms / float(self.ms_tau)) + (1.0 - g) * dense


def build_block_diag_mask(seg_ids: Sequence[int], dtype: Any = torch.float32,
                          device: Any = None) -> torch.Tensor:
    """段编号 -> 块对角**加性**注意力掩码 [1,1,L,L]（0 / finfo.min）。

    语义：不同槽位的软 token 之间**互不可见**；同一槽位内部、以及问题/答案对所有
    槽位可见。返回的是"已取反"的加性掩码（0 表示可见），与 HF eager 实现的
    `_update_causal_mask` 对 4D 输入的约定一致。

    注意（必须写清楚，不许含糊）：SDPA/FlashAttention 后端**不接受** 4D 掩码，
    所以这个函数只在 eager 后端可用；`_emit_latent` 会先探测后端，不可用时降级为
    "原生隔离栅栏"（每槽前后插 [Fact j] 标记）并在 meta 里写明降级原因。
    """
    ids = [int(v) for v in (seg_ids or [])]
    L = len(ids)
    if L <= 0:
        raise ValueError("build_block_diag_mask 需要非空的段编号")
    dev = torch.device(device) if device is not None else None
    seg = torch.tensor(ids, dtype=torch.long, device=dev)
    same = seg.unsqueeze(0) == seg.unsqueeze(1)                 # [L,L]
    # ---- 共享段 0 必须双向全见，否则这个掩码会把整个读出打死 ----
    # 段 0 = system / 围栏 / 问题（非槽位 token）。若只写 `causal & same`，
    # 问题（段 0）就**看不到任何槽位**（槽位是 1..K），槽位也看不到 system ——
    # 语义上等价于"记忆被整段屏蔽"，模型只能凭参数知识答，指标会掉回 no-memory
    # 水平，而这在日志里表现为"隔离一开就掉点"，很容易被误读成"隔离有害"。
    # 正确关系：seg_j == 0（共享键）对所有人可见；seg_i == 0（共享查询）可见所有
    # 因果允许的键；只有**两个不同槽位之间**才互不可见。
    causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=dev))
    shared_row = (seg == 0).unsqueeze(1).expand(L, L)           # 行 i 是共享段
    shared_col = (seg == 0).unsqueeze(0).expand(L, L)           # 列 j 是共享段
    allowed = causal & (same | shared_row | shared_col)
    finfo = torch.finfo(dtype)
    mask = torch.zeros((1, 1, L, L), dtype=dtype, device=dev)
    mask = mask.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), finfo.min)
    return mask


def compose_routed_soft(model: Any, tokenizer: Any, bridge: "LatentBridge",
                        z_list: Sequence[torch.Tensor],
                        dev: Any = None,
                        lex_list: Optional[Sequence[torch.Tensor]] = None,
                        isolate: bool = False,
                        return_segs: bool = False):
    """若干槽位载荷 -> 一段 soft 序列（每槽过一次 bridge，槽间插原生 "\\n"）。

    **训练与推理必须调同一个函数**：旧实现里训练是 `bridge(z_read)` 一段、
    推理是 `bridge(z1) + "\\n" + bridge(z2)`，两跳样本的 train/infer 输入分布不同，
    而且训练侧第 2 跳的载荷根本没进图。

    lex_list（v4）：每槽的 h0 符号直通向量，逐槽交给 bridge 做门控方向混合。
    isolate（v4）：槽位隔离。返回 (soft, seg_ids)（return_segs=True 时），
        seg_ids 是逐 token 的段号，供 `build_block_diag_mask` 构造块对角掩码。
    """
    if not z_list:
        raise ValueError("compose_routed_soft 收到空载荷列表")
    dev = _model_device(model) if dev is None else dev
    emb = model.get_input_embeddings()
    pieces = []
    segs: List[int] = []
    # 段号约定：0 = 原生标记（对所有槽位可见的公共前缀），j+1 = 第 j 个槽位。
    # 用**单调递增**的段号，块对角掩码里的"同段可见"才等价于"同槽可见"。
    for j, zj in enumerate(z_list):
        if not torch.is_tensor(zj):
            raise TypeError(f"compose_routed_soft 第 {j} 个载荷不是张量：{type(zj)}")
        if zj.dim() == 4:
            # 单跳/多跳都要交 [B,m,d]。四维几乎一定是"索引时保留了多余维度"，
            # 静默 squeeze 会掩盖上游 bug，这里直接报出来。
            raise ValueError(f"compose_routed_soft 第 {j} 个载荷是四维 {tuple(zj.shape)}，"
                             f"应为 [B,m,d]")
        # 槽位边界锚点：每段软 token 前插一段**原生词表**的离散嵌入。
        # 纯软向量裸拼时，相邻槽的自注意力会互相串扰（跨槽内容被混着读），
        # 而原生 token 会把"这是另一条事实"这件事显式写进序列，同时给出槽序号。
        # isolate=True 时用**加倍的围栏**（前后各一段），因为块对角掩码在 SDPA 后端
        # 拿不到时，这条围栏是唯一还能起作用的隔离手段。
        fence = (f"\n[Fact {j + 1}]: " if not isolate
                 else f"\n-- Fact {j + 1} (memory slot, isolated) --\n")
        anchor = emb(_tokenize_no_special(tokenizer, fence, dev))
        pieces.append(anchor)
        segs.extend([0] * int(anchor.shape[1]))
        zl = None
        if lex_list is not None and j < len(lex_list) and lex_list[j] is not None:
            zl = lex_list[j]
            # 两种打包口径必须对齐：读出载荷有时是 [k*m, d]（块已展平，见 pack_slot_reads），
            # 直通载荷是 [k, m, d]。元素数与顺序都一致（k 为外层），reshape 是恒等映射；
            # 不 reshape 就会在 bridge 的形状校验里炸掉（实测 2/4/2560 vs 1/8/2560）。
            if tuple(zl.shape) != tuple(zj.shape):
                if int(zl.numel()) != int(zj.numel()):
                    raise ValueError(f"第 {j} 槽直通载荷元素数 {int(zl.numel())} 与语义载荷 "
                                     f"{int(zj.numel())} 不一致，无法对齐")
                zl = zl.reshape(tuple(zj.shape))
        s = bridge(zj, zl)
        if s.dim() == 2:
            s = s.unsqueeze(0)
        s = s.reshape(1, -1, s.shape[-1])
        pieces.append(s)
        segs.extend([j + 1] * int(s.shape[1]))
        if isolate:
            tail = emb(_tokenize_no_special(tokenizer, f"\n-- end Fact {j + 1} --\n", dev))
            pieces.append(tail)
            segs.extend([0] * int(tail.shape[1]))
    soft = torch.cat(pieces, dim=1)
    if return_segs:
        return soft, segs
    return soft


def extract_query_feature_ex(model: Any, tokenizer: Any, question: str,
                             system_prompt: str = "", layer: int = LATENT_LAYER,
                             device: Any = None) -> Dict[str, Any]:
    """问题 -> {"pooled": [1,2d], "h0_tok": [1,T,d], "h18_tok": [1,T,d]}。

    为什么不能只给 pooled：HybridLatentRouter 的稀疏分支要在**词级**上算 MaxSim
    （问题里的实体 token 对记忆里的实体 token），mean-pool 之后 token 轴没了，
    这种匹配就退化成向量内积，正是 v3 丢掉稀有实体匹配的那个病。
    """
    if model is None or tokenizer is None:
        raise ValueError("extract_query_feature_ex 需要 model 与 tokenizer")
    dev = torch.device(device) if device is not None else _model_device(model)
    full, n_sys_chars = build_chat_parts(tokenizer, system_prompt, question)
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
    h0_tok = hs[0]
    hl_tok = hs[int(layer)]
    p = model.get_input_embeddings().weight
    return {"pooled": torch.cat([h0_tok.mean(dim=1), hl_tok.mean(dim=1)], dim=-1).to(dtype=p.dtype),
            "h0_tok": h0_tok.detach().to(dtype=p.dtype),
            "h18_tok": hl_tok.detach().to(dtype=p.dtype)}


def slot_token_features(encoder: Any, texts: Sequence[str],
                        key_tokens: int = LATENT_KEY_TOKENS) -> Dict[str, Any]:
    """单元文本 -> 逐槽 token 级冻结特征 {"h0":[K,T,d], "h18":[K,T,d], "mask":[K,T]}。

    与 `LatentEncoder.encode` 共用同一次冻结前向（`features()`），不额外付编码成本；
    多块单元沿 token 轴拼接（截到 key_tokens），因为寻址要的是"这句话里有哪些词"，
    与压缩器怎么分块无关。
    """
    texts = ["" if t is None else str(t) for t in (texts or [])]
    if not texts:
        return {"h0": torch.zeros((0, 1, 0)), "h18": torch.zeros((0, 1, 0)),
                "mask": torch.zeros((0, 1), dtype=torch.long), "n_units": 0}
    feats = encoder.features(texts, parent_ids=[str(i) for i in range(len(texts))])
    groups: Dict[str, List[int]] = {}
    order: List[str] = []
    for i, pid in enumerate(feats["parents"]):
        if pid not in groups:
            groups[pid] = []
            order.append(pid)
        groups[pid].append(i)
    cap = max(1, int(key_tokens))
    per_slot: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for pid in order:
        rows = sorted(groups[pid], key=lambda i: feats["chunk_index"][i])
        h0 = torch.cat([feats["h0"][i][: int(feats["n_tokens"][i]) or 1] for i in rows], dim=0)
        h18 = torch.cat([feats["h18"][i][: int(feats["n_tokens"][i]) or 1] for i in rows], dim=0)
        n = int(h0.shape[0])
        h0 = h0[:cap]
        h18 = h18[:cap]
        m = torch.zeros((cap,), dtype=torch.long)
        m[: int(h0.shape[0])] = 1
        if int(h0.shape[0]) < cap:
            pad = torch.zeros((cap - int(h0.shape[0]), int(h0.shape[-1])), dtype=h0.dtype)
            h0 = torch.cat([h0, pad], dim=0)
            h18 = torch.cat([h18, pad], dim=0)
        per_slot.append((h0, h18, m))
        del n
    return {"h0": torch.stack([t[0] for t in per_slot], dim=0),
            "h18": torch.stack([t[1] for t in per_slot], dim=0),
            "mask": torch.stack([t[2] for t in per_slot], dim=0),
            "n_units": len(per_slot), "key_tokens": int(cap)}


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


# 只收"明确指向过去"的短语，**不收** was/were/had 这类普通过去时：
# HotpotQA/MAB 的问题里 "Where was X born?" 随处可见，把它当回溯意图会让
# 版本策略在绝大多数样本上白跑一遍 time 分支（结果虽仍回落到 current，但
# 日志和消融会被这层噪声搅浑）。
_PAST_MARKERS = ("ago", "previously", "formerly", "used to", "in the past",
                 "earlier", "at that time", "back then", "old value",
                 "before the", "before that", "prior to")


def _query_has_time_intent(query_text: Any, query_time: Any = None) -> bool:
    """问题里是否表达了**回溯历史**的意图（决定版本策略走 current 还是 time）。

    依据：MAB-CR 与 FST 合成集里，只有带明确时间条件的题才需要读历史队列；
    其余题的目标值是"当前状态"，此时确定性取最新版的准确率是 0.885，
    而 learned 版本分类器只有 0.525。把这条结构信息交还给规则，不是偷懒 ——
    是把"已知的确定性"从"要学的随机变量"里拿回来。
    """
    if _norm_time(query_time):
        return True
    q = _norm_time(query_text)
    if not q:
        return False
    if _as_year(q) is not None:                 # 问题里出现年份
        return True
    return any(mk in q for mk in _PAST_MARKERS)


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
                     query_time: Any = None, query_text: Any = None) -> int:
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
        if policy == "hybrid":
            # 混合版本策略（v4）：先看**问题本身**有没有回溯意图，再决定走规则还是学习。
            # 依据是实测：纯 learned 版本选择 0.525，恒选最新 0.885 —— 把"最新值算对"
            # 这个已知结构重新交给一个 softmax 去学，是白丢 36 个点。所以默认确定性
            # 走 current，只有问题里出现时间标记（年份/过去时/ago/before/previous…）
            # 或给了 as_of 时，才启用时间规则/学习打分。
            policy = "time" if _query_has_time_intent(query_text, query_time) else "current"
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
                ver_times: Any = None, query_time: Any = None,
                q_vec: Optional[torch.Tensor] = None,
                used_groups: Optional[Sequence[int]] = None,
                query_text: Any = None
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
        # q_vec 非空 = 多跳接力的当前查询（**潜空间**里更新，不碰任何明文值）。
        # 这是 v4 修掉 P03/P07 的落点：旧代码的接力循环用 `slot_values` 明文字符串
        # 去猜"下一跳是不是以它为实体"，既回到明文依赖，又因为 k_groups=n 提前取满
        # 而永远不执行（死分支）。
        q_vec = (self.key.encode_query(q_feat) if q_vec is None
                 else torch.nn.functional.normalize(torch.as_tensor(
                     q_vec, device=next(self.parameters()).device,
                     dtype=next(self.parameters()).dtype), dim=-1))
        k_grp = self.key.encode_keys(e_feat, r_feat)                # [G,dk]
        lg_grp = self.key.scores(q_vec, k_grp)                      # [1,G]
        if used_groups:
            m0 = torch.zeros_like(lg_grp, dtype=torch.bool)
            for gi in used_groups:
                if 0 <= int(gi) < int(lg_grp.shape[-1]):
                    m0[0, int(gi)] = True
            lg_grp = lg_grp.masked_fill(m0, float("-inf"))
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
                                      ver_times, query_time, query_text) for g in groups]
        # 读出用**被选中组**的版本打分；监督用 tgt_group 那一行（见 logits_ver_all）
        g0 = int(groups[0])
        lg_all = lg_ver_all[g0:g0 + 1]                              # [1,V]

        z_all = torch.as_tensor(ver_z)
        z_reads = [_payload_md(z_all[g, v]).unsqueeze(0) for g, v in zip(groups, versions)]
        out = {"z_read": z_reads[0], "z_reads": z_reads,
               "logits_group": lg_grp, "logits_ver": lg_all,
               "logits_ver_all": lg_ver_all,
               # 版本键 [G,V,dk]：多跳接力的**潜层出键**。返回它，接力侧就不用再
               # 重算一遍 encode_versions（FST 每问本来就要付 11s 的编码账）。
               "k_ver": k_ver, "q_vec": q_vec,
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
        # v4：h0 符号直通载荷（与 payloads 同键同形状）。None 表示该槽位没有直通载荷
        # （例如 v3 checkpoint 或 lex_skip=0 的编码器），读出侧 bridge 会自动忽略。
        self.lex: Dict[str, torch.Tensor] = {}
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

    def append(self, slot_id: Any, z: torch.Tensor, detach: bool = True,
               lex: Optional[torch.Tensor] = None) -> str:
        sid = str(slot_id)
        if sid in self.payloads:
            raise ValueError(f"duplicate slot_id: {sid}")
        z = self._normalize(z)
        self.payloads[sid] = self._store_tensor(z, detach)
        if lex is not None:
            lx = self._normalize(lex, name="lex")
            if tuple(lx.shape) != tuple(z.shape):
                raise ValueError(f"lex 形状 {tuple(lx.shape)} 必须与载荷 {tuple(z.shape)} 一致")
            self.lex[sid] = self._store_tensor(lx, detach)
        self.versions[sid] = 1
        self.n_append += 1
        return sid

    def overwrite(self, slot_id: Any, z: torch.Tensor, detach: bool = True,
                  lex: Optional[torch.Tensor] = None) -> str:
        sid = str(slot_id)
        if sid not in self.payloads:
            raise KeyError(f"overwrite 不存在的槽位：{sid}")
        z = self._normalize(z)
        self.payloads[sid] = self._store_tensor(z, detach)
        if lex is not None:
            lx = self._normalize(lex, name="lex")
            if tuple(lx.shape) != tuple(z.shape):
                raise ValueError(f"lex 形状 {tuple(lx.shape)} 必须与载荷 {tuple(z.shape)} 一致")
            self.lex[sid] = self._store_tensor(lx, detach)
        self.versions[sid] = int(self.versions.get(sid, 1)) + 1
        self.n_overwrite += 1
        return sid

    def drop(self, slot_id: Any) -> None:
        sid = str(slot_id)
        if sid not in self.payloads:
            raise KeyError(f"drop 不存在的槽位：{sid}")
        self.payloads.pop(sid, None)
        self.lex.pop(sid, None)
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
                self.lex.pop(old, None)
                self.versions.pop(old, None)
                continue
            new = f"{pre}{i - 1}"
            if new in self.payloads:
                raise RuntimeError(f"shift_index 目标冲突：{new} 已存在（库内下标不连续？）")
            self.payloads[new] = self.payloads.pop(old)
            if old in self.lex:
                self.lex[new] = self.lex.pop(old)
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

    def payload_lex(self, slot_id: Any, device: Any = None) -> Optional[torch.Tensor]:
        """该槽位的 h0 符号直通载荷（没有则 None —— 不补零冒充）。"""
        z = self.lex.get(str(slot_id))
        if z is None:
            return None
        return z if device is None else z.to(device=device)

    def gather_lex(self, slot_ids: Sequence[Any], device: Any = "cpu"
                   ) -> List[Optional[torch.Tensor]]:
        """按给定顺序返回直通载荷列表（缺失位是 None，与 gather 的严格报错不同：
        直通分支是可选增强，缺它时 bridge 的行为必须与 v3 逐位一致，而不是报错。"""
        return [self.payload_lex(s, device=device) for s in (slot_ids or [])]

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
                "n_slots_with_lex": len(self.lex),
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
              value: Any = None, lex: Optional[torch.Tensor] = None) -> str:
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
            self.append(sid, z, detach=detach, lex=lex)
            self.slot_values[sid] = v
            self.groups[key] = {"cur": sid, "cur_time": t, "hist": [],
                                "n_write": 1,
                                "ent": (None if entity is None else str(entity)),
                                "rel": (None if relation is None else str(relation))}
            return sid
        sid = self._slot_id(key, int(g["n_write"]))
        old = g["cur"]
        self.append(sid, z, detach=detach, lex=lex)   # 先 append 再改指针：append 抛错则状态不变
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

    def payload_lex_at(self, key: Any, time: Any = None,
                       device: Any = None) -> Optional[torch.Tensor]:
        """该键被选中的版本的 h0 直通载荷（没有则 None）。"""
        return self.payload_lex(self.read(key, time), device=device)

    def lex_for(self, key: Any, index: int, device: Any = None) -> Optional[torch.Tensor]:
        """该键第 index 个版本（旧->新）的直通载荷，供 FST 多跳读出按版本取用。"""
        vers = self.list_versions(key)
        i = int(index)
        if not (0 <= i < len(vers)):
            raise IndexError(f"版本下标越界：{i}/{len(vers)}（key={key!r}）")
        return self.payload_lex(vers[i]["slot_id"], device=device)

    def gather_versions(self, key: Any, device: Any = "cpu"
                        ) -> Tuple[torch.Tensor, List[str], List[bool]]:
        """-> (ver_z [V,m,d], times [V], is_current [V])，顺序为 旧->新（当前在最后）。"""
        vers = self.list_versions(key)
        zs = [_payload_md(self.payload(v["slot_id"])) for v in vers]
        return (torch.stack(zs, dim=0).to(device=device),
                [str(v["time"]) for v in vers],
                [bool(v["current"]) for v in vers])

    def gather_all_versions_lex(self, keys: Optional[Sequence[Any]] = None
                                ) -> List[List[Optional[torch.Tensor]]]:
        """与 gather_all_versions 同构的直通载荷版本列表（缺失位是 None）。"""
        ks = self.group_keys() if keys is None else [str(k) for k in keys]
        return [[self.payload_lex(v["slot_id"]) for v in self.list_versions(k)] for k in ks]

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
    chunk_ids: List[List[int]] = []      # 每块的 token id：符号保真诊断要按列号回查"哪个词"
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
                        chunk_ids.append([])
                        n_empty_units += 1
                    else:
                        h0_out.append(h0[r, a:b].to("cpu"))
                        h18_out.append(hl[r, a:b].to("cpu"))
                        mask_out.append(torch.ones((b - a,), dtype=torch.long))
                        chunk_text.append(_decode_ids(tokenizer, ids[r, a:b]))
                        chunk_ids.append([int(v) for v in ids[r, a:b].tolist()])
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
            "chunk_ids": chunk_ids,
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
        """返回 (per-parent payload 列表, info)。兼容旧接口，见 encode_full。"""
        payloads, _lex, _kf, info = self.encode_full(texts, parent_ids=parent_ids,
                                                     grad=grad, want_lex=False,
                                                     want_kfeat=False)
        return payloads, info

    def encode_full(self, texts: Sequence[str],
                    parent_ids: Optional[Sequence[str]] = None,
                    grad: bool = False, want_lex: bool = True,
                    want_kfeat: bool = True
                    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]],
                               Optional[Dict[str, Any]], Dict[str, Any]]:
        """返回 (per-parent 语义载荷, per-parent 直通载荷, 逐槽 token 特征, info)。

        每个 payload 形状 [k, m, d]（k = 该单元被分成的块数，1 表示未分块）。
        grad=False（默认）：存储路径，全程 no_grad，输出可直接进 LatentSlotBank。
        grad=True：训练路径，压缩器前向保留梯度（**不要**把它放进 torch.no_grad）。

        三份输出对应 v4 的两处解耦：
          - 语义载荷 -> 读出（bridge 的语义分支）
          - 直通载荷 -> 读出（bridge 的 h0 符号分支，绕过 inner_dim 瓶颈）
          - token 特征 -> 寻址（HybridLatentRouter 的键，来自冻结模型，与压缩器无关）
        """
        texts = ["" if t is None else str(t) for t in (texts or [])]
        if not texts:
            return [], None, None, {"n_units": 0, "n_chunks": 0, "n_tokens_in": 0,
                                    "n_soft_tokens": 0, "compression_ratio": None}
        feats = self.features(texts, parent_ids=parent_ids)
        kw = {"want_lex": bool(want_lex and getattr(self.compressor, "lex_skip", False))}
        if grad:
            z, z_lex, cinfo = self.compressor.forward_split(feats["h0"], feats["h18"],
                                                           feats["mask"])
        else:
            with torch.no_grad():
                z, z_lex, cinfo = self.compressor.forward_split(feats["h0"], feats["h18"],
                                                               feats["mask"])
        if not kw["want_lex"]:
            z_lex = None
        # 按父槽位分组（保持输入顺序，块内按 chunk_index）
        order: List[str] = []
        groups: Dict[str, List[int]] = {}
        for i, pid in enumerate(feats["parents"]):
            if pid not in groups:
                groups[pid] = []
                order.append(pid)
            groups[pid].append(i)
        payloads: List[torch.Tensor] = []
        lex_payloads: Optional[List[torch.Tensor]] = [] if z_lex is not None else None
        for pid in order:
            rows = sorted(groups[pid], key=lambda i: feats["chunk_index"][i])
            payloads.append(z[rows])                 # [k, m, d]
            if lex_payloads is not None:
                lex_payloads.append(z_lex[rows])
        n_soft = int(sum(int(p.shape[0]) * int(p.shape[1]) for p in payloads))
        n_in = int(sum(feats["n_tokens"]))
        info = {"n_units": len(texts), "n_parents": len(order),
                "n_chunks": int(feats["n_chunks_total"]),
                "n_empty_units": int(feats["n_empty_units"]),
                "n_tokens_in": n_in, "n_soft_tokens": n_soft,
                "compression_ratio": (round(n_soft / n_in, 6) if n_in else None),
                "n_slots": int(z.shape[1]), "dim": int(z.shape[-1]),
                "n_empty_rows": int(cinfo.get("n_empty_rows", 0)),
                "lex_skip": bool(z_lex is not None)}
        kfeat = None
        if want_kfeat:
            kfeat = self._kfeat_from_feats(feats, order, groups)
        return payloads, lex_payloads, kfeat, info

    @staticmethod
    def _kfeat_from_feats(feats: Dict[str, Any], order: Sequence[str],
                          groups: Dict[str, List[int]]) -> Dict[str, Any]:
        """把 encode_full 里已经算好的 token 特征整理成逐槽 [K,T,d]（不再跑模型）。"""
        cap = max(1, int(LATENT_KEY_TOKENS))
        h0s, h18s, ms = [], [], []
        for pid in order:
            rows = sorted(groups[pid], key=lambda i: feats["chunk_index"][i])
            h0 = torch.cat([feats["h0"][i][: max(1, int(feats["n_tokens"][i]))]
                            for i in rows], dim=0)
            h18 = torch.cat([feats["h18"][i][: max(1, int(feats["n_tokens"][i]))]
                             for i in rows], dim=0)
            h0, h18 = h0[:cap], h18[:cap]
            m = torch.zeros((cap,), dtype=torch.long)
            m[: int(h0.shape[0])] = 1
            if int(h0.shape[0]) < cap:
                pad = torch.zeros((cap - int(h0.shape[0]), int(h0.shape[-1])),
                                  dtype=h0.dtype)
                h0 = torch.cat([h0, pad], dim=0)
                h18 = torch.cat([h18, pad], dim=0)
            h0s.append(h0)
            h18s.append(h18)
            ms.append(m)
        return {"h0": torch.stack(h0s, dim=0), "h18": torch.stack(h18s, dim=0),
                "mask": torch.stack(ms, dim=0), "n_units": len(order),
                "key_tokens": int(cap)}


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
    lex_skip = bool(cfg.get("lex_skip", False))
    comp = LatentCompressor(hs, n_slots=int(cfg["n_slots"]),
                            inner_dim=int(cfg["inner_dim"]),
                            num_heads=int(cfg.get("num_heads", LATENT_N_HEADS)),
                            lex_skip=lex_skip)
    bridge = LatentBridge(hs, bottleneck=int(cfg["bottleneck"]),
                          target_norm=float(cfg.get("target_norm", LATENT_TARGET_NORM)),
                          lex_skip=bool(cfg.get("bridge_lex_skip", lex_skip)),
                          alpha_max=float(cfg.get("bridge_alpha_max",
                                                  LATENT_ALPHA_MAX)))
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
        rkind = str(cfg.get("router_kind", "latent"))
        rcls = HybridLatentRouter if rkind == "hybrid" else LatentRouter
        # ms_tau / anchor_boost / anchor_tokens / novelty_lambda 都是**普通属性**，
        # 不进 state_dict；不按 ckpt 恢复就会静默退回环境变量默认值，
        # 表现为"同一个权重换个进程跑出来的召回不一样"。
        _hkw: Dict[str, Any] = {}
        if rcls is HybridLatentRouter:
            _hkw = {"gamma": float(cfg.get("addr_gamma", LATENT_ADDR_GAMMA)),
                    "ms_tau": float(cfg.get("router_ms_tau", LATENT_MS_TAU)),
                    "anchor_boost": float(cfg.get("router_anchor_boost",
                                                  LATENT_ANCHOR_BOOST)),
                    "anchor_tokens": int(cfg.get("router_anchor_tokens",
                                                 LATENT_ANCHOR_TOKENS)),
                    "novelty_lambda": float(cfg.get("router_novelty_lambda",
                                                    LATENT_NOVELTY_LAMBDA))}
        router = rcls(hs, n_slots=int(cfg["n_slots"]),
                      d_key=int(cfg.get("d_key", LATENT_D_KEY)), **_hkw)
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
    cfg["lex_skip"] = bool(getattr(compressor, "lex_skip", False))
    cfg["bridge_lex_skip"] = bool(getattr(bridge, "lex_skip", False))
    # 非参数约束也必须落盘：alpha_max 决定"残差能长到多强"，不存进 ckpt 的话
    # 同一个权重换个进程/换台机器读出来就是另一套读出行为（与 router 的 ms_tau 同类）。
    cfg["bridge_alpha_max"] = float(getattr(bridge, "alpha_max", LATENT_ALPHA_MAX))
    if bool(cfg["lex_skip"]) != bool(cfg["bridge_lex_skip"]):
        raise ValueError(f"压缩器 lex_skip={cfg['lex_skip']} 与 bridge lex_skip="
                         f"{cfg['bridge_lex_skip']} 不一致：直通载荷生成端与消费端必须同时开或同时关")
    if router is not None:
        cfg["has_router"] = True
        cfg["d_key"] = int(router.d_key)
        # router_kind 决定 load 时用哪个类重建：Hybrid 比基类多了注意力池化头与 γ，
        # 类选错时 strict=True 会在「unexpected keys」上直接报错（这是想要的行为，
        # 好过悄悄退回基类、让 MaxSim 底盘消失）。
        cfg["router_kind"] = ("hybrid" if isinstance(router, HybridLatentRouter)
                              else "latent")
        # 非参数超参必须显式存：它们不进 state_dict，只能靠 config 还原
        cfg["router_ms_tau"] = float(getattr(router, "ms_tau", LATENT_MS_TAU))
        cfg["router_anchor_boost"] = float(getattr(router, "anchor_boost",
                                                   LATENT_ANCHOR_BOOST))
        cfg["router_anchor_tokens"] = int(getattr(router, "anchor_tokens",
                                                  LATENT_ANCHOR_TOKENS))
        cfg["router_novelty_lambda"] = float(getattr(router, "novelty_lambda",
                                                     LATENT_NOVELTY_LAMBDA))
        if int(router.d_model) != int(compressor.hidden_size):
            raise ValueError("router.d_model != compressor.hidden_size")
        if int(router.n_slots) != int(compressor.n_slots):
            raise ValueError("router.n_slots != compressor.n_slots")
    else:
        cfg["has_router"] = False
        cfg.setdefault("router_kind", "none")
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


def _prefill_isolated_then_generate(model: Any, input_embeds: torch.Tensor,
                                    block_mask: torch.Tensor,
                                    attention_mask: torch.Tensor,
                                    max_new_tokens: int, pad_token_id: int,
                                    dev: Any) -> Optional[torch.Tensor]:
    """用块对角掩码做 prefill（n_in-1 个 token），再从最后一个 token 接着贪心解码。

    返回**新生成**的 token 张量 [n_out]（与 generate 只返回新 token 的口径一致）；
    任何环节不兼容就抛异常，由调用方退回普通路径（不在这里吞错，吞了会看不出降级）。
    """
    n_in = int(input_embeds.shape[1])
    if n_in < 2:
        return None
    pre_emb = input_embeds[:, :-1]
    m = block_mask[:, :, : n_in - 1, : n_in - 1]
    with delta_write_disabled(model), torch.no_grad():
        out = model(inputs_embeds=pre_emb, attention_mask=m, use_cache=True,
                    output_hidden_states=False)
        pkv = getattr(out, "past_key_values", None)
        if pkv is None:
            raise RuntimeError("prefill 未返回 past_key_values（use_cache 未生效）")
        gen = model.generate(inputs_embeds=input_embeds[:, -1:], attention_mask=attention_mask,
                             past_key_values=pkv, max_new_tokens=int(max_new_tokens),
                             do_sample=False, use_cache=True, pad_token_id=pad_token_id)
    if not torch.is_tensor(gen):
        raise RuntimeError(f"隔离路径 generate() 返回了非张量：{type(gen).__name__}")
    return gen[0]


def _attn_impl(model: Any) -> str:
    cfg = getattr(model, "config", None)
    return str(getattr(cfg, "_attn_implementation", "") or "").lower()


def _emit_latent(model: Any, tokenizer: Any, soft: Optional[torch.Tensor], question: str,
                 system_prompt: str, max_new_tokens: int = 32,
                 pad_token_id: Optional[int] = None,
                 extra_meta: Optional[Dict[str, Any]] = None,
                 seg_ids: Optional[Sequence[int]] = None,
                 isolate: bool = False
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

    # ---- 槽位间块对角隔离（v4）----
    # 只对"软 token 段"生效：不同槽位的软 token 互不可见，同一槽位内部与
    # 问题/答案侧照常全见。实现上是一张 4D 加性掩码 —— 而 SDPA/FA2 后端不接受 4D
    # 掩码（HF 的 _update_causal_mask 在 sdpa 下会直接用自建的因果掩码，把我们的
    # 掩码**静默丢掉**）。因此这里先探测后端：不支持就**明确降级**为围栏隔离
    # （compose_routed_soft(isolate=True) 的 [-- Fact j --] 前后围栏），
    # 并把实际生效的模式写进 meta，绝不假装隔离已经生效。
    # 为什么不在 generate 里直接塞 4D mask：`GenerationMixin` 每个解码步都会对
    # attention_mask 做 `cat([mask, ones(1)], dim=-1)` 来延长它，4D 张量在这里必然
    # 形状不匹配。所以块对角隔离只在 **prefill** 阶段生效（prefill 一次拿 past_key_values，
    # 再让 generate 从最后一个 prompt token 接着解码），并且全程 try/except：
    # 一旦上游包装器/past_key_values 形态不兼容，就**退回**等价的普通路径并写明原因，
    # 绝不让"隔离"变成一次静默的崩溃或静默的失效。
    slot_isolation = "off"
    block_mask: Optional[torch.Tensor] = None
    if isolate and seg_ids is not None and soft is not None:
        impl = _attn_impl(model)
        n_pre = int(pre.shape[1])
        n_soft_tok = int(soft.shape[1])
        n_post = int(post.shape[1])
        full_segs = [0] * n_pre + [int(v) for v in seg_ids][:n_soft_tok] + [0] * n_post
        if len(full_segs) != n_in:
            raise ValueError(f"seg_ids 长度 {len(full_segs)} != 输入长度 {n_in}")
        if impl in ("eager", "flex_attention"):
            try:
                block_mask = build_block_diag_mask(full_segs, dtype=input_embeds.dtype,
                                                   device=dev)
                slot_isolation = f"block_diag:{impl}"
            except Exception as e:                       # pragma: no cover - 后端差异
                slot_isolation = f"fence_only:mask_error:{type(e).__name__}"
        else:
            slot_isolation = f"fence_only:attn_impl={impl or 'unknown'}"
    elif isolate and soft is not None:
        slot_isolation = "fence_only:no_seg_ids"


    pad = pad_token_id
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None) or getattr(tokenizer, "pad_token_id", None) or 0
    # 生成期必须关掉 Delta 写入：upstream 的 write_enabled 默认 True，`model.generate`
    # 会边读 prompt 边更新在线状态、再把 Delta 读出加回 q/o 注意力。而训练侧的记忆
    # 编码 / 问题特征 / student 前向全都在 delta_write_disabled 下进行 —— 两侧不一致时，
    # adapter 等于在评测时额外开了一套训练时从未使用的记忆系统，指标会以"全方法一起
    # 退化、输出塌成固定 token"的形式坏掉，且看不出是这里的问题。
    # 只读策略下没有预先写入的状态，Delta 读出应为零，生成行为回到纯主干。
    seq = None
    if block_mask is not None and n_in > 1:
        try:
            seq = _prefill_isolated_then_generate(
                model, input_embeds, block_mask, attention_mask,
                int(max_new_tokens), pad, dev)
        except Exception as e:                            # pragma: no cover - 上游差异
            slot_isolation = f"fallback:{type(e).__name__}:{str(e)[:80]}"
            seq = None
    if seq is None:
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
            "slot_isolation": slot_isolation,
            "decode_mode": decode_mode, "max_new_tokens": int(max_new_tokens)}
    meta.update(dict(extra_meta or {}))
    return pred, meta


def generate_latent_answer(model: Any, tokenizer: Any, bridge: LatentBridge,
                           payloads: Optional[torch.Tensor], question: str,
                           system_prompt: str, max_new_tokens: int = 32,
                           pad_token_id: Optional[int] = None,
                           router: Optional["LatentRouter"] = None,
                           q_feat: Optional[torch.Tensor] = None,
                           layer: int = LATENT_LAYER, hops: int = 1,
                           reads: Optional[Sequence[torch.Tensor]] = None,
                           kfeat: Optional[Dict[str, Any]] = None,
                           q_tok: Optional[torch.Tensor] = None,
                           lex_reads: Optional[Sequence[torch.Tensor]] = None,
                           topk_per_hop: int = 1,
                           isolate: int = LATENT_SLOT_ISOLATION
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
    seg_ids: Optional[List[int]] = None
    routed: Optional[List[int]] = None
    if has_payload and router is not None:
        slot_z = payloads if payloads.dim() == 4 else payloads.unsqueeze(0)   # [1,K,m,d]
        slot_z = slot_z.to(dev)
        qf = q_feat
        if qf is None:
            qf = extract_query_feature(model, tokenizer, question, system_prompt,
                                       layer=layer, device=dev)
        # v4：单跳也走 forward_hops —— 因为它现在同时支持 topk_per_hop（4 槽读出）
        # 与 h0 MaxSim 底盘。分成两条路径就会重演"训练一个函数、推理另一个函数"
        # 的老毛病（本文件 header 里记着那次代价）。
        out = router.forward_hops(qf, slot_z, n_hops=max(1, int(hops)), hard=True,
                                  reads=reads, kfeat=kfeat, q_tok=q_tok,
                                  topk_per_hop=int(topk_per_hop),
                                  lex_reads=lex_reads)
        routed = list(out["picks"])
        soft = compose_routed_soft(model, tokenizer, bridge, out["z_list"], dev=dev,
                                   lex_list=out.get("lex_list"),
                                   isolate=bool(isolate), return_segs=True)
        soft, seg_ids = soft
    elif has_payload:
        # bridge 不做内容寻址：整库所有槽位**全给**，K*m 个软 token 一起注入，
        # 这是 bridge 的设计语义（"压缩后整段送去读"），不是遗漏了截断。
        # payloads 在评测/训练侧统一是 [1,K,m,d]，必须先归一化到 3 维再进 bridge：
        # 直接把 4 维喂给 bridge 会在 _as_3d 里抛
        # "z 必须是 [B,L,d] 或 [L,d]，得到 (1,20,4,2560)" —— bridge 评测实际就死在这。
        if reads is not None and len(reads) > 0:
            # 全槽读出：把每个槽的**全部块**按槽序拼成一段（段间仍由
            # compose 的思路插入边界锚点在路由路径里完成；这里是无路由全给路径，
            # 直接按块序拼接），而不是只给块均值。
            pz = torch.cat([r if r.dim() == 2 else r.reshape(-1, r.shape[-1])
                            for r in reads], dim=0)
        else:
            pz = payloads
            if pz.dim() == 4:
                if int(pz.shape[0]) != 1:
                    raise ValueError(
                        f"bridge 只支持 batch=1 的潜载荷，得到 {tuple(pz.shape)}")
                pz = pz[0]
            elif pz.dim() == 2:
                pz = pz.unsqueeze(0)
        plz = None
        if lex_reads is not None and len(lex_reads) > 0:
            rows = [r if r is not None else torch.zeros_like(rr)
                    for r, rr in zip(lex_reads, reads or lex_reads)]
            plz = torch.cat([r if r.dim() == 2 else r.reshape(-1, r.shape[-1])
                             for r in rows], dim=0)
        soft = bridge(pz.to(dev), None if plz is None else plz.to(dev))
        if soft.dim() == 3:
            soft = soft.reshape(1, -1, soft.shape[-1])
        # 全槽路径没有槽边界信息，隔离按"每个软向量一段"（最保守）：
        # 逐段隔离比"假装不隔离"更接近意图，且不需要额外声明。
        if bool(isolate):
            seg_ids = list(range(1, int(soft.shape[1]) + 1))
    pred, meta = _emit_latent(model, tokenizer, soft, question, system_prompt,
                              max_new_tokens=max_new_tokens, pad_token_id=pad_token_id,
                              seg_ids=seg_ids, isolate=bool(isolate),
                              extra_meta={
                                  "has_payload": has_payload, "routed": routed,
                                  "hops": int(hops), "topk_per_hop": int(topk_per_hop),
                                  "lex_skip": bool(lex_reads is not None),
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
                               ver_policy: str = "hybrid",
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
    # ---- 第 1 跳：只取 **1** 个事实组 ----
    # 这里是 P03 的修复点。旧实现一上来就 `k_groups=n_read`，一次性把 n 个组都取回来，
    # 于是后面 `while len(picked) < n_read` 的条件**恒为 False** —— FST 的核心创新
    # "多跳状态链式接力"是一段从未执行过的死代码，实际退化成单次静态 Top-K。
    out = fst_router(q_feat, e_feat, r_feat, ver_z, ver_t_feat, tq_feat=tq_feat,
                     ver_mask=ver_mask, hard=True, k_groups=1,
                     ver_policy=ver_policy, ver_times=times, query_time=query_time,
                     query_text=question)
    picked: List[Tuple[int, int]] = list(zip(out["groups"], out["versions"]))
    chain_log: List[Dict[str, Any]] = [{"hop": 1, "group": int(picked[0][0]),
                                        "version": int(picked[0][1]),
                                        "via": "query"}]
    # ---- 第 2..n 跳：**在潜空间里**接力 ----
    # 用上一跳选中版本的潜层出键 k_{g,v}^out 更新查询向量，再对全部事实键重新打分
    # （已选组屏蔽）。这与旧实现有本质差别：
    #   - 旧：拿 `slot_values` 的**明文字符串**去匹配"谁的实体出现在上一跳的值里"，
    #     一旦解析模板漏了 relation、或值的写法与实体不一致，接力就静默断掉；
    #     而且明文参与决策，等于把"纯潜层读出"的立论让掉一半。
    #   - 新：q ← norm(q + k_{g,v}^out)，全程只有向量，没有字符串。
    q_vec_chain = out.get("q_vec")
    used_groups = [int(picked[0][0])]
    while len(picked) < n_read:
        g_prev, v_prev = picked[-1]
        k_ver = out.get("k_ver")
        if k_ver is None or q_vec_chain is None:
            break
        k_out = k_ver[int(g_prev), int(v_prev)]
        q_vec_chain = torch.nn.functional.normalize(q_vec_chain + k_out, dim=-1)
        out2 = fst_router(q_feat, e_feat, r_feat, ver_z, ver_t_feat, tq_feat=tq_feat,
                          ver_mask=ver_mask, hard=True, k_groups=1,
                          ver_policy=ver_policy, ver_times=times, query_time=query_time,
                          q_vec=q_vec_chain, used_groups=used_groups,
                          query_text=question)
        g2 = int(out2["groups"][0])
        if g2 in used_groups:                  # 屏蔽失效（全被 mask 时的 -inf 兜底）
            break
        v2 = int(out2["versions"][0])
        picked.append((g2, v2))
        used_groups.append(g2)
        chain_log.append({"hop": len(picked), "group": g2, "version": v2,
                          "via": "latent_k_out"})
    z_list = [_payload_md(ver_z[g, v]).unsqueeze(0) for g, v in picked]
    # 直通载荷按 (事实键, 版本) 从库里取：库与载荷同生同灭，因此不需要重新编码。
    lex_list = []
    for g, v in picked:
        try:
            lex_list.append(bank.lex_for(keys[g], v, device=dev))
        except (KeyError, IndexError):
            lex_list.append(None)
    if all(x is None for x in lex_list):
        lex_list = None
    soft = compose_routed_soft(model, tokenizer, bridge, z_list, dev=dev,
                              lex_list=lex_list, isolate=True, return_segs=True)
    soft, seg_ids = soft
    g, v = picked[0]
    pred, meta = _emit_latent(model, tokenizer, soft, question, system_prompt,
                              max_new_tokens=max_new_tokens, pad_token_id=pad_token_id,
                              seg_ids=seg_ids, isolate=True,
                              extra_meta={
                                  "has_payload": True, "routed": [int(x[0]) for x in picked],
                                  "lex_skip": bool(lex_list is not None),
                                  "hops": len(picked),
                                  "n_slots_in": len(keys),
                                  "group_keys": keys, "group_picked": int(g),
                                  "groups_picked": [int(x[0]) for x in picked],
                                  "versions_picked": [int(x[1]) for x in picked],
                                  # chain_values 只作**诊断**保留（人力核对接力对不对），
                                  # 决策已经不依赖它了；chain_log 才是接力的执行轨迹。
                                  "chain_values": [bank.value_at(keys[x[0]], x[1])
                                                   for x in picked],
                                  "chain_log": chain_log,
                                  "chain_via": [c["via"] for c in chain_log],
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
