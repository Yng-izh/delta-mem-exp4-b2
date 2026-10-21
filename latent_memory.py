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
LATENT_CKPT            = _env_str("LATENT_CKPT", "")            # 压缩器+桥接器 checkpoint
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
    """编码期禁止 Delta 写入：记忆编码不得改动在线状态（否则同一 context 内前后不一致）。"""
    fn = None
    try:
        from deltamem.core.delta import set_delta_mem_write_enabled as fn  # noqa: F811
    except Exception:
        fn = None
    if fn is None or model is None:
        yield False
        return
    try:
        fn(model, False)
    except Exception:
        yield False
        return
    try:
        yield True
    finally:
        try:
            fn(model, False)
        except Exception:
            pass


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
    """Z -> soft tokens：LayerNorm -> Linear -> SiLU -> Linear（输入层注入，不改 Qwen 内部 attention）。"""

    def __init__(self, hidden_size: int, bottleneck: int = LATENT_BOTTLENECK):
        super().__init__()
        if int(hidden_size) <= 0 or int(bottleneck) <= 0:
            raise ValueError(f"hidden_size/bottleneck 必须为正：{hidden_size}/{bottleneck}")
        self.hidden_size = int(hidden_size)
        self.bottleneck = int(bottleneck)
        self.net = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.bottleneck),
            nn.SiLU(),
            nn.Linear(self.bottleneck, self.hidden_size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = _as_3d(z, "z")
        if z.shape[-1] != self.hidden_size:
            raise ValueError(f"z 的最后一维 {z.shape[-1]} != hidden_size={self.hidden_size}")
        p = next(self.parameters())
        out = self.net(z.to(device=p.device, dtype=p.dtype))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


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

    - **并行**存在：v9_slotpool 的 texts / tokv / words / metas 一个都不删，它们仍然承担
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
                 checkpoint: str, fingerprint: Dict[str, Any]):
        self.compressor = compressor
        self.bridge = bridge
        self.encoder = encoder
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

    优先复用 run_manifest.file_fingerprint（同一个口径，避免两处哈希算法漂移）；
    run_manifest 不可用时退回本地实现。
    """
    try:
        import run_manifest
        return run_manifest.file_fingerprint(path, label="checkpoint")
    except Exception:
        pass
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
    bridge = LatentBridge(hs, bottleneck=int(cfg["bottleneck"]))
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
    return LatentReader(comp, bridge, enc, cfg, checkpoint,
                        checkpoint_fingerprint(checkpoint))


def save_latent_checkpoint(path: str, compressor: LatentCompressor, bridge: LatentBridge,
                           config: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> str:
    """保存压缩器 + 桥接器 + 配置（推理侧 load_latent_reader 严格对齐这份 config）。"""
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
    blob = {"schema": LATENT_SCHEMA, "config": cfg,
            "compressor": compressor.state_dict(),
            "bridge": bridge.state_dict(),
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


def generate_latent_answer(model: Any, tokenizer: Any, bridge: LatentBridge,
                           payloads: Optional[torch.Tensor], question: str,
                           system_prompt: str, max_new_tokens: int = 32,
                           pad_token_id: Optional[int] = None
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
    """
    if tokenizer is None or model is None:
        raise ValueError("generate_latent_answer 需要 model 与 tokenizer")
    if payloads is not None and (not torch.is_tensor(payloads)):
        payloads = torch.as_tensor(payloads)
    has_payload = bool(payloads is not None and payloads.numel() > 0)
    if has_payload and bridge is None:
        raise ValueError("给了潜载荷但没有 bridge：无法把 Z 映射成 soft tokens")

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
    if has_payload:
        soft = bridge(payloads.to(dev))              # [K, m, d] -> [K*m, d]
        if soft.dim() == 3:
            soft = soft.reshape(1, -1, soft.shape[-1])
        n_soft = int(soft.shape[1])
        parts.append(soft.to(dtype=pre.dtype))
    parts.append(post)
    input_embeds = torch.cat(parts, dim=1)
    attention_mask = torch.ones(input_embeds.shape[:2], dtype=torch.long, device=dev)
    n_in = int(input_embeds.shape[1])

    pad = pad_token_id
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None) or getattr(tokenizer, "pad_token_id", None) or 0
    with torch.no_grad():
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
            "decode_mode": decode_mode, "max_new_tokens": int(max_new_tokens),
            "has_payload": has_payload}
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

    挂接本身不改变检索：v9_slotpool 的 MaxSim / 时间打分 / CR 双阈值覆写一行都不动。
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
