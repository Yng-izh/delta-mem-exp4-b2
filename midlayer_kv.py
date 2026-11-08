#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""midlayer_kv.py — 把证据的**逐层真实 KV** 插到逻辑位置 S+E+T 中间，由原生注意力读。

为什么改成"中间插入"（2026-11-07 实验 D）
-----------------------------------------
实验 C 的注入把记忆整段放在动态 token **前面**，等价于"前缀拼接"。那只能回答"证据在
prompt 最前面"这一种情况，而且问题侧前 18 层完全看不到证据 —— 但更要紧的是：它无法
做等价性验证，因为你没法把"只藏起 Evidence、保留 System 指令"这件事表达出来。

现在按 Token 把完整 prompt 切成三段：

    S  证据之前的固定 prompt（system 指令 + "Memory:\\n" 头）
    E  金标证据本体
    T  证据之后的一切（消息结束符 + 问题 + assistant 生成前缀）

离线：把 `S+E` 跑一遍冻结主干，只取出 **E 那一段**的逐层 KV 存下来。
在线：只喂 `S+T`（E 的 token 根本不进残差流），把 E 的 KV 插回它原本的位置。

因果掩码在明文里本来就保证 E 看不到 T，所以 `[S,E]` 的离线编码与完整前向中 E 的计算
**逐位相同**。于是在**全 36 层**注入时，这套接口在数学上必须复现明文前向 —— 这就是
D0 的等价性门槛（`runs/exp_d_kv_recovery.sh` 的第一关）。

四条必须钉死的规则
------------------
1. **位置不能压缩**：T 的 position_ids 要保留 E 占掉的那段（`ns+ne ..`），否则 RoPE 的
   相对偏移全错。E 的 K 按 `ns .. ns+ne-1` 施加 RoPE。
2. **掩码按逻辑序**：S 只能看自己前面；T 能看全部 S、全部 E、以及自己之前的 T。
   把 E 当成"前缀谁都可见"会让 S 看到本该在它后面的内容。
3. **先过 `input_layernorm` 再投影**：`hidden_states[l]` 是残差流，第 l 层的
   `k_proj/v_proj` 消费的是 `input_layernorm(残差流)`。漏这一步会把注意力输出放大
   两个量级、残差流被冲掉，而且**给真实 KV 时同样崩**（实验 C 第一版的乱码就是这个）。
4. **S/E/T 必须来自同一次分词**：拼接后要与完整 prompt 的 token id 逐位相等，
   否则 BPE 边界变化会让"等价性"测试本身失效（`split_prompt_tokens` 里强制校验）。

离线编码协议（D1 的三个臂）
---------------------------
| encoding | 离线编码输入 | 检验什么 |
|---|---|---|
| `full_prefix` | `S+E` | 严格对齐，用于 D0 等价性 |
| `bare` | 只有 `E` | 丢掉前缀上下文的代价 |
| `anchor` | 固定短锚点 + `E` | 简单固定前缀是否够 |

三者共用同一套"取 E 段 → 按逻辑位置重施 RoPE → 投影成各层 K/V"的代码；差别只在
离线那一次前向喂了什么。`anchor` 与 `bare` 重投影 RoPE 是**必须**的：它们离线时用的
是别的绝对位置，直接沿用会与 `S+E+T` 的逻辑位置不一致。
"""
from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import latent_memory as LM

# 证据在 prompt 里所在的段落标题（S 的一部分）
MID_KV_EVIDENCE_HEADER = "Memory:\n"
# `anchor` 编码用的固定短前缀：不含问题、不含任何题目信息，可提前写入并复用
MID_KV_ANCHOR_TEXT = "Memory:\n- "
# 注入起点默认值（D2 会扫 0/6/12/18）
MID_KV_START_LAYER = LM._env_int("MID_KV_START_LAYER", 0)
# 一次最多带多少记忆 token（防显存；只截尾，不参与方法定义）
MID_KV_MAX_MEM_TOKENS = LM._env_int("MID_KV_MAX_MEM_TOKENS", 320)
# 离线编码协议
MID_KV_ENCODINGS = ("full_prefix", "bare", "anchor")


# ---------------------------------------------------------------------
# 0) 模型结构探测（duck-typing，不 import transformers 的模型类）
# ---------------------------------------------------------------------
_LAYER_PATHS = ("model.layers", "model.model.layers", "transformer.h", "layers",
                "model.decoder.layers")


def decoder_layers(model: Any) -> List[nn.Module]:
    for path in _LAYER_PATHS:
        obj = model
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and len(obj) > 0:
            return list(obj)
    raise ValueError("找不到解码器层（试过 %s）" % ", ".join(_LAYER_PATHS))


def attn_of(layer: nn.Module) -> nn.Module:
    for name in ("self_attn", "attention", "attn", "self_attention"):
        m = getattr(layer, name, None)
        if isinstance(m, nn.Module):
            return m
    raise ValueError(f"层 {type(layer).__name__} 里找不到注意力模块")


def input_norm_of(layer: nn.Module) -> nn.Module:
    """注意力**之前**的归一化。残差流必须过它才能喂 `k_proj/v_proj`。"""
    for name in ("input_layernorm", "ln_1", "norm1", "attention_norm", "ln_attn"):
        m = getattr(layer, name, None)
        if isinstance(m, nn.Module):
            return m
    raise ValueError(f"层 {type(layer).__name__} 里找不到输入归一化；"
                     f"跳过它会静默产出错误的 K/V，所以直接报错")


def _cfg_of(attn: nn.Module) -> Any:
    return getattr(attn, "config", None)


def head_dim_of(attn: nn.Module) -> int:
    hd = getattr(attn, "head_dim", None)
    if isinstance(hd, int) and hd > 0:
        return int(hd)
    cfg = _cfg_of(attn)
    hd = getattr(cfg, "head_dim", None) if cfg is not None else None
    if isinstance(hd, int) and hd > 0:
        return int(hd)
    n = num_heads_of(attn)
    out = int(getattr(attn.q_proj, "out_features", 0))
    if n > 0 and out > 0:
        return out // n
    raise ValueError("无法确定 head_dim")


def num_heads_of(attn: nn.Module) -> int:
    for name in ("num_heads", "num_attention_heads"):
        v = getattr(attn, name, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    cfg = _cfg_of(attn)
    v = getattr(cfg, "num_attention_heads", None) if cfg is not None else None
    if isinstance(v, int) and v > 0:
        return int(v)
    hd = getattr(attn, "head_dim", None)
    if not isinstance(hd, int) or hd <= 0:
        hd = getattr(cfg, "head_dim", None) if cfg is not None else None
    out = int(getattr(attn.q_proj, "out_features", 0))
    if isinstance(hd, int) and hd > 0 and out > 0:
        return out // hd
    raise ValueError("无法确定 query 头数")


def num_kv_heads_of(attn: nn.Module) -> int:
    hd = head_dim_of(attn)
    out = int(getattr(attn.k_proj, "out_features", 0))
    if out > 0 and hd > 0:
        return out // hd
    raise ValueError("无法确定 kv 头数")


def rope_module(model: Any) -> nn.Module:
    for path in ("model.rotary_emb", "model.model.rotary_emb", "rotary_emb"):
        obj = model
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and isinstance(obj, nn.Module):
            return obj
    first = attn_of(decoder_layers(model)[0])
    r = getattr(first, "rotary_emb", None)
    if isinstance(r, nn.Module):
        return r
    raise ValueError("找不到 RoPE 模块（需要能被调用为 rotary_emb(x, position_ids)）")


def _rope_cos_sin(rope: nn.Module, dtype: torch.dtype, device: Any,
                  positions: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """按位置取 (cos, sin)，统一成 [1,1,N,D]。假输入只用来拿 dtype/device。"""
    pos = torch.as_tensor(positions, dtype=torch.long, device=device).reshape(1, -1)
    probe = torch.zeros((1, 1, int(pos.shape[1])), dtype=dtype, device=device)
    cos, sin = rope(probe, pos)
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def rot_positions(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """半分裂 RoPE（Qwen 系），x: [B,H,N,D]。奇数 head_dim 直接报错。"""
    d = int(x.shape[-1])
    if d % 2 != 0:
        raise ValueError(f"RoPE 需要偶数 head_dim，得到 {d}")
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# ---------------------------------------------------------------------
# 1) S / E / T 的 Token 级切分（必须可验证）
# ---------------------------------------------------------------------
class SplitError(RuntimeError):
    """S/E/T 无法在 token 边界上严格切开。宁可停下，也不要带着错位继续跑。"""


def _ids_of(tokenizer: Any, text: str) -> List[int]:
    return tokenizer(str(text), return_tensors="pt",
                     add_special_tokens=False)["input_ids"][0].tolist()


def split_prompt_tokens(tokenizer: Any, system_prompt: str, evidence_text: str,
                        question: str,
                        header: str = MID_KV_EVIDENCE_HEADER) -> Dict[str, Any]:
    """把完整 prompt 切成 S / E / T，并保证 `ids_S+ids_E+ids_T == ids_full`（逐位）。

    为什么要这个保证：后续"证据离线编码 + 在线中间插入"要宣称与明文等价，前提是
    三段的 token 序列与明文完全一致。分开分词在边界上可能被 BPE 合并（"\\n" 与 "- "
    粘成一个 token 之类），那样"等价性"测的就不是模型，而是分词器的脾气。

    两种切法：`exact`（分段单独分词后拼接等于整体分词）与 `matched`（整体分词里存在
    与 E 完全一致的一段，直接按位置切）。都不成立就抛 `SplitError`。
    """
    ev = str(evidence_text).strip("\n")
    if not ev.strip():
        raise SplitError("证据文本为空")
    sysp = (system_prompt or "").strip()
    sys_content = f"{sysp}\n\n{header}{ev}" if sysp else f"{header}{ev}"
    msgs = [{"role": "system", "content": sys_content},
            {"role": "user", "content": question}]
    full_text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                             tokenize=False)
    c0 = str(full_text).find(ev)
    if c0 < 0:
        raise SplitError("chat 模板没有原样保留证据文本（可能做了转义/strip）")
    ids_full = _ids_of(tokenizer, full_text)
    ids_pre = _ids_of(tokenizer, str(full_text)[:c0])
    ids_ev = _ids_of(tokenizer, ev)
    ids_post = _ids_of(tokenizer, str(full_text)[c0 + len(ev):])
    method = "exact"
    if ids_pre + ids_ev + ids_post != ids_full:
        method = "matched"
        ns = _find_sub(ids_full, ids_ev)
        if ns < 0:
            raise SplitError(
                "证据段无法在 token 边界上定位：分段分词 %d 个 token、整体分词 %d 个，"
                "且整体序列里找不到证据段（BPE 边界被合并）。"
                % (len(ids_pre) + len(ids_ev) + len(ids_post), len(ids_full)))
    else:
        ns = len(ids_pre)
    ne = len(ids_ev)
    ids_S = list(ids_full[:ns])
    ids_E = list(ids_full[ns:ns + ne])
    ids_T = list(ids_full[ns + ne:])
    if ids_S + ids_E + ids_T != ids_full:           # 构造保证，但别靠"应该"
        raise SplitError("S/E/T 拼接后与完整 prompt 不一致")
    if ne <= 0 or len(ids_T) <= 0:
        raise SplitError(f"切分退化：ne={ne} nt={len(ids_T)}")
    return {"full_text": str(full_text), "ids_S": ids_S, "ids_E": ids_E, "ids_T": ids_T,
            "ids_full": ids_full, "ns": ns, "ne": ne, "nt": len(ids_T),
            "evidence_text": ev, "split_method": method}


def _find_sub(hay: Sequence[int], needle: Sequence[int]) -> int:
    n, m = len(hay), len(needle)
    if m <= 0 or m > n:
        return -1
    for i in range(n - m + 1):
        if list(hay[i:i + m]) == list(needle):
            return i
    return -1


# ---------------------------------------------------------------------
# 2) 记忆的 K/V：取 E 段的隐藏状态 -> 各层自己的 W_k/W_v（按逻辑位置施 RoPE）
# ---------------------------------------------------------------------
def project_kv(layer: nn.Module, h: torch.Tensor, positions: Any,
               rope: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """残差流 h [1,N,d] -> 该层 (k, v) [1,Hkv,N,Dh]，k 已按 `positions` 施 RoPE。

    `h` 必须与这一层在真实前向里看到的分布一致（即已过该层 `input_layernorm`）；
    `positions` 是 E 在**完整 prompt** 里的逻辑位置，不能是离线编码时的位置。
    """
    attn = attn_of(layer)
    if h.dim() != 3:
        raise ValueError(f"h 必须是 [1,N,d]，得到 {tuple(h.shape)}")
    w = attn.k_proj.weight
    x = input_norm_of(layer)(h.to(device=w.device, dtype=w.dtype))
    n = int(x.shape[1])
    hd = head_dim_of(attn)
    nkv = num_kv_heads_of(attn)
    k = attn.k_proj(x).view(1, n, nkv, hd).transpose(1, 2)
    v = attn.v_proj(x).view(1, n, nkv, hd).transpose(1, 2)
    kn = getattr(attn, "k_norm", None)
    if isinstance(kn, nn.Module) and any(True for _ in kn.parameters()):
        k = kn(k)
    cos, sin = _rope_cos_sin(rope, k.dtype, k.device, positions)
    k = rot_positions(k, cos, sin)
    return k.contiguous(), v.contiguous()


@torch.no_grad()
def capture_span_hidden(model: Any, ids: Sequence[int], span_start: int,
                        span_len: int, layers: Sequence[int]) -> Dict[int, torch.Tensor]:
    """编码 `ids`，返回**指定层输入**上、`[span_start, span_start+span_len)` 那一段。

    `hidden_states[l]` 是第 l 层的输入（第 0 项是 embedding），口径与
    `output_hidden_states` 的定义一致。
    """
    dev = LM._model_device(model)
    t = torch.tensor([list(int(i) for i in ids)], dtype=torch.long, device=dev)
    with LM.delta_write_disabled(model):
        out = model(input_ids=t, use_cache=False, output_hidden_states=True)
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise RuntimeError("主干没有返回 hidden_states")
    need = sorted(set(int(l) for l in layers))
    if need and need[-1] >= len(hs):
        raise ValueError(f"需要的层 {need[-1]} 超出 hidden_states 长度 {len(hs)}")
    return {l: hs[l][:, int(span_start):int(span_start) + int(span_len)].detach()
            for l in need}


def build_evidence_kv(model: Any, tokenizer: Any, split: Dict[str, Any],
                      encoding: str, pos_start: int, n_tokens: int,
                      layers: Sequence[int], rope: Optional[nn.Module] = None,
                      pad_repeat: bool = False) -> "MemoryKV":
    """按某种离线编码协议，把 E 段变成 `layers` 各层的 K/V。

    `pos_start`/`n_tokens` 是**在线逻辑位置**的显式参数：`wrong` 反事实要用别的题目的
    隐藏状态、但插在本题的位置上，所以这两个量不能从 `split` 里偷。
    """
    enc = str(encoding)
    if enc not in MID_KV_ENCODINGS:
        raise ValueError(f"未知编码协议 {enc!r}，可选 {MID_KV_ENCODINGS}")
    rope = rope_module(model) if rope is None else rope
    if enc == "full_prefix":
        ids = list(split["ids_S"]) + list(split["ids_E"])
        span_start = len(split["ids_S"])
    elif enc == "bare":
        ids = list(split["ids_E"])
        span_start = 0
    else:                                            # anchor
        a = _ids_of(tokenizer, MID_KV_ANCHOR_TEXT)
        ids = a + list(split["ids_E"])
        span_start = len(a)
    span_len = min(int(split["ne"]), int(n_tokens))
    if span_len <= 0:
        raise ValueError("E 段长度为 0")
    hs = capture_span_hidden(model, ids, span_start, span_len, layers)
    if span_len < int(n_tokens):
        if not pad_repeat:
            raise ValueError(f"证据段只有 {span_len} 个 token，不足要求的 {n_tokens}")
        # wrong 反事实专用：别的题目的证据可能更短，而它必须占满**本题** E 的位置区间，
        # 否则插进去的 key 位置与 T 的逻辑位置对不上（RoPE 偏移会错）。
        hs = {l: torch.cat([h, h[:, -1:, :].repeat(1, int(n_tokens) - span_len, 1)], dim=1)
              for l, h in hs.items()}
        span_len = int(n_tokens)
    layers_l = decoder_layers(model)
    positions = torch.arange(int(pos_start), int(pos_start) + span_len,
                             device=hs[min(hs)].device)
    per: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for l in sorted(hs):
        per[int(l)] = project_kv(layers_l[int(l)], hs[int(l)], positions, rope)
    return MemoryKV(per, n_tokens=span_len, insert_at=int(pos_start),
                    start_layer=min(per), encoding=enc, rope=rope,
                    hidden_size=int(hs[min(hs)].shape[-1]),
                    truncated=bool(span_len < int(split["ne"])))


class MemoryKV:
    """一段证据在若干层上的 (k, v)，以及它应当插回的逻辑位置。

    刻意不是 `nn.Module`：本阶段零训练，记忆必须是外部常量，不能进优化器。
    """

    def __init__(self, per_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                 n_tokens: int, insert_at: int, start_layer: int, encoding: str,
                 rope: Optional[nn.Module] = None, hidden_size: int = 0,
                 truncated: bool = False):
        self.per_layer = dict(per_layer)
        self.n = int(n_tokens)
        self.insert_at = int(insert_at)
        self.start_layer = int(start_layer)
        self.encoding = str(encoding)
        self.rope = rope
        self.hidden_size = int(hidden_size)
        self.truncated = bool(truncated)
        self._mask_cache: Dict[Any, torch.Tensor] = {}

    @property
    def layers(self) -> List[int]:
        return sorted(self.per_layer)

    def k(self, layer: int) -> torch.Tensor:
        return self.per_layer[int(layer)][0]

    def v(self, layer: int) -> torch.Tensor:
        return self.per_layer[int(layer)][1]

    def subset(self, start_layer: int) -> "MemoryKV":
        """同一份捕获复用到更晚的注入起点（D2 的 L6/L12/L18 不再重复编码）。"""
        sl = int(start_layer)
        keep = {l: kv for l, kv in self.per_layer.items() if l >= sl}
        if not keep:
            raise ValueError(f"subset({sl}) 为空：本体只有层 {self.layers[:3]}…{self.layers[-1:]}")
        return MemoryKV(keep, n_tokens=self.n, insert_at=self.insert_at,
                        start_layer=sl, encoding=self.encoding, rope=self.rope,
                        hidden_size=self.hidden_size, truncated=self.truncated)

    def insert_mask(self, n_s: int, n_t: int, dtype: torch.dtype, device: Any) -> torch.Tensor:
        key = (int(n_s), int(n_t), str(dtype), str(device))
        m = self._mask_cache.get(key)
        if m is None:
            m = build_insert_mask(int(n_s), self.n, int(n_t), dtype, device)
            self._mask_cache[key] = m
        return m

    def nbytes_kv(self) -> int:
        return int(sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                       for k, v in self.per_layer.values()))

    def describe(self) -> Dict[str, Any]:
        return {"encoding": self.encoding, "start_layer": self.start_layer,
                "n_mem": self.n, "insert_at": self.insert_at,
                "n_layers": len(self.per_layer), "truncated": self.truncated,
                "kv_MB": round(self.nbytes_kv() / 1e6, 2),
                "state_bytes_per_token": self.hidden_size * 2}


def build_insert_mask(n_s: int, n_mem: int, n_t: int, dtype: torch.dtype,
                      device: Any) -> torch.Tensor:
    """[1,1,n_s+n_t,n_s+n_mem+n_t] 加性掩码，列序 = S | E | T。

    S 只能看自己前面的 S；T 能看全部 S、全部 E、以及自己之前的 T。
    E 不做 query（它不在在线残差流里），所以掩码没有 E 的行。
    """
    if n_s < 0 or n_mem < 0 or n_t <= 0:
        raise ValueError(f"掩码尺寸非法：n_s={n_s} n_mem={n_mem} n_t={n_t}")
    n_q = n_s + n_t
    n_kv = n_s + n_mem + n_t
    allowed = torch.zeros((n_q, n_kv), dtype=torch.bool)
    if n_s:                                          # S 行：只看 S 的因果前缀
        allowed[:n_s, :n_s] = torch.tril(torch.ones((n_s, n_s), dtype=torch.bool))
    if n_t:                                          # T 行：S 全见 + E 全见 + T 因果
        allowed[n_s:, :n_s + n_mem] = True
        allowed[n_s:, n_s + n_mem:] = torch.tril(torch.ones((n_t, n_t), dtype=torch.bool))
    mask = torch.zeros((1, 1, n_q, n_kv), dtype=dtype, device=device)
    return mask.masked_fill(~allowed.unsqueeze(0).unsqueeze(0).to(device),
                            torch.finfo(dtype).min)


# ---------------------------------------------------------------------
# 3) 注入：把 E 的 K/V 插在 S 与 T 之间
# ---------------------------------------------------------------------
_ACTIVE: Dict[int, "MemoryKV"] = {}


def _question_kv(attn: nn.Module, hidden_states: torch.Tensor,
                 position_embeddings: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, q_len, _ = hidden_states.shape
    hd = head_dim_of(attn)
    n_q, n_kv = num_heads_of(attn), num_kv_heads_of(attn)
    q = attn.q_proj(hidden_states).view(bsz, q_len, n_q, hd).transpose(1, 2)
    k = attn.k_proj(hidden_states).view(bsz, q_len, n_kv, hd).transpose(1, 2)
    v = attn.v_proj(hidden_states).view(bsz, q_len, n_kv, hd).transpose(1, 2)
    qn, kn = getattr(attn, "q_norm", None), getattr(attn, "k_norm", None)
    if isinstance(qn, nn.Module) and any(True for _ in qn.parameters()):
        q = qn(q)
    if isinstance(kn, nn.Module) and any(True for _ in kn.parameters()):
        k = kn(k)
    if position_embeddings is None:
        raise RuntimeError("注入路径要求主干传入 position_embeddings")
    cos, sin = position_embeddings
    return rot_positions(q, cos, sin), rot_positions(k, cos, sin), v


def _midlayer_attn_forward(self, hidden_states, position_embeddings=None,
                           attention_mask=None, past_key_value=None,
                           past_key_values=None, cache_position=None,
                           position_ids=None, **kwargs):
    """装了记忆的层改走这里：K/V 序列 = S 段 + E 段（记忆）+ T 段。

    与原生 forward 的差别**只有**"K/V 中间多了一段 E"；q/k 投影、q_norm/k_norm、
    RoPE、o_proj、缩放全部照原层执行。`hidden_states` 里是 S+T（E 不在残差流里），
    所以按 `mem.insert_at` 把它切开再拼。
    """
    mem = _ACTIVE.get(id(self))
    if mem is None:
        raise RuntimeError("中间层 KV 包装器没有登记记忆（_ACTIVE 缺项）")
    if past_key_value is not None or past_key_values is not None:
        # transformers ≥5 的 decoder 传的是复数名，两种都拦：本路径逐步整段前向，
        # 悄悄把 cache 丢掉会退化成"第二轮起什么都看不到"的静默错误。
        raise RuntimeError("中间层 KV 注入不支持 KV cache（本路径逐步整段前向）")
    if int(hidden_states.shape[0]) != 1:
        raise ValueError(f"记忆 K/V 只对应单条样本（batch=1），收到 "
                         f"batch={int(hidden_states.shape[0])}")
    q, k, v = _question_kv(self, hidden_states, position_embeddings)
    layer = _layer_of_attn(self)
    k_mem = mem.k(layer).to(device=k.device, dtype=k.dtype)
    v_mem = mem.v(layer).to(device=v.device, dtype=v.dtype)
    ns = mem.insert_at
    n = int(k.shape[2])
    if ns > n:
        raise ValueError(f"插入位置 {ns} 超出当前序列长度 {n}")
    if k_mem.shape[2] != mem.n:
        raise ValueError(f"记忆长度 {k_mem.shape[2]} 与登记值 {mem.n} 不一致")
    K = torch.cat([k[:, :, :ns], k_mem, k[:, :, ns:]], dim=2)
    V = torch.cat([v[:, :, :ns], v_mem, v[:, :, ns:]], dim=2)
    n_rep = int(q.shape[1]) // int(K.shape[1])
    if n_rep > 1:                                    # GQA：SDPA 要求头数一致
        K = K.repeat_interleave(n_rep, dim=1)
        V = V.repeat_interleave(n_rep, dim=1)
    n_t = n - ns
    mask = mem.insert_mask(ns, n_t, k.dtype, k.device)
    out = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
    out = out.transpose(1, 2).reshape(int(hidden_states.shape[0]), n, -1)
    return self.o_proj(out.contiguous()), None


def _layer_of_attn(attn: nn.Module) -> int:
    l = getattr(attn, "_midlayer_layer", None)
    if l is None:
        raise RuntimeError("注意力模块缺少 _midlayer_layer 标记")
    return int(l)


def _install(attn: nn.Module, layer: int, mem: MemoryKV):
    had = "forward" in attn.__dict__
    prev = attn.__dict__.get("forward")
    prev_layer = attn.__dict__.get("_midlayer_layer")
    attn._midlayer_layer = int(layer)
    attn.forward = types.MethodType(_midlayer_attn_forward, attn)
    _ACTIVE[id(attn)] = mem
    return (attn, had, (prev, prev_layer))


def _restore(rec) -> None:
    attn, had, prev = rec
    _ACTIVE.pop(id(attn), None)
    prev_fwd, prev_layer = prev
    if had:
        attn.forward = prev_fwd
    else:
        try:
            del attn.forward
        except AttributeError:
            pass
    if prev_layer is None:
        try:
            del attn._midlayer_layer
        except AttributeError:
            pass
    else:
        attn._midlayer_layer = prev_layer


@contextmanager
def inject_memory_kv(model: Any, mem: Optional[MemoryKV]):
    """只在 `mem.per_layer` 覆盖到的层上装包装器；退出时**逐模块**恢复。"""
    if mem is None or int(mem.n) <= 0:
        yield False
        return
    layers = decoder_layers(model)
    recs = []
    try:
        for l in mem.layers:
            if l >= len(layers):
                raise ValueError(f"记忆层 {l} 超出主干层数 {len(layers)}")
            recs.append(_install(attn_of(layers[l]), l, mem))
        yield True
    finally:
        for rec in reversed(recs):
            _restore(rec)


# ---------------------------------------------------------------------
# 4) 生成：明文 / KV 注入 / 原生前缀 cache 三条路共用一套贪心与停止条件
# ---------------------------------------------------------------------
def _stop_ids(tokenizer: Any, stop_ids: Optional[Sequence[int]]) -> set:
    s = set(int(i) for i in (stop_ids or []) if i is not None)
    if not s:
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            s.add(int(eos))
    return s


@torch.no_grad()
def _greedy_loop(model: Any, tokenizer: Any, cur: torch.Tensor, pos: torch.Tensor,
                 next_pos: int, mem: Optional[MemoryKV], max_new_tokens: int,
                 stops: set) -> Tuple[List[int], Optional[torch.Tensor]]:
    """从已拼接好的 embeds 出发贪心解码（不走 cache）。

    `pos` 是当前序列的**逻辑位置**（KV 路径里 S 与 T 之间留着 E 的空档），
    `next_pos` 是新生成 token 的起始逻辑位置。
    """
    emb_layer = model.get_input_embeddings()
    dev = cur.device
    out_ids: List[int] = []
    first_logits: Optional[torch.Tensor] = None
    with inject_memory_kv(model, mem), LM.delta_write_disabled(model):
        for step in range(int(max_new_tokens)):
            o = model(inputs_embeds=cur, use_cache=False, position_ids=pos)
            logits = o.logits[0, -1, :].float()
            if first_logits is None:
                first_logits = logits.clone()
            nxt = int(logits.argmax().item())
            if nxt in stops:
                break
            out_ids.append(nxt)
            nid = torch.tensor([[nxt]], device=dev)
            cur = torch.cat([cur, emb_layer(nid)], dim=1)
            pos = torch.cat([pos, torch.tensor([[next_pos + step]], device=dev)], dim=1)
    return out_ids, first_logits


@torch.no_grad()
def text_generate(model: Any, tokenizer: Any, prompt_text: str,
                  max_new_tokens: int = 16, stop_ids: Optional[Sequence[int]] = None,
                  return_logits: bool = False):
    """明文参照：整段 prompt（含证据）一次性前向 + 贪心。与 KV 路径同一套解码。"""
    dev = LM._model_device(model)
    ids = LM._tokenize_no_special(tokenizer, prompt_text, dev)
    emb = model.get_input_embeddings()(ids)
    pos = torch.arange(int(ids.shape[1]), device=dev).unsqueeze(0)
    out_ids, first = _greedy_loop(model, tokenizer, emb, pos, int(ids.shape[1]),
                                  None, max_new_tokens, _stop_ids(tokenizer, stop_ids))
    pred = tokenizer.decode(out_ids, skip_special_tokens=True)
    meta = {"arm_kind": "text", "n_prompt": int(ids.shape[1]), "n_out": len(out_ids)}
    return (pred, meta, first) if return_logits else (pred, meta)


@torch.no_grad()
def midlayer_kv_generate(model: Any, tokenizer: Any, mem: Optional[MemoryKV],
                         split: Dict[str, Any], max_new_tokens: int = 16,
                         stop_ids: Optional[Sequence[int]] = None,
                         return_logits: bool = False):
    """只喂 S+T，E 的 K/V 插回逻辑位置。位置显式保留 E 占掉的空档。"""
    dev = LM._model_device(model)
    ids_in = list(split["ids_S"]) + list(split["ids_T"])
    ids_t = torch.tensor([ids_in], dtype=torch.long, device=dev)
    emb = model.get_input_embeddings()(ids_t)
    ns, ne, nt = int(split["ns"]), int(split["ne"]), int(split["nt"])
    pos = torch.cat([torch.arange(0, ns, device=dev),
                     torch.arange(ns + ne, ns + ne + nt, device=dev)]).unsqueeze(0)
    out_ids, first = _greedy_loop(model, tokenizer, emb, pos, ns + ne + nt, mem,
                                  max_new_tokens, _stop_ids(tokenizer, stop_ids))
    pred = tokenizer.decode(out_ids, skip_special_tokens=True)
    meta = {"arm_kind": "kv", "n_prompt": int(len(ids_in)), "n_out": len(out_ids),
            "n_s": ns, "n_t": nt, "position_gap": [ns, ns + ne]}
    if mem is not None:
        meta.update(mem.describe())
    return (pred, meta, first) if return_logits else (pred, meta)


@torch.no_grad()
def prefix_kv_generate(model: Any, tokenizer: Any, split: Dict[str, Any],
                       max_new_tokens: int = 16,
                       stop_ids: Optional[Sequence[int]] = None,
                       return_logits: bool = False):
    """D0-A：用**原生 KV cache** 把 `S+E` 做成前缀，再接 T 继续解码。

    这条路径不改注意力、不碰掩码，只验证"因果前缀拆成两次前向"是否与一次前向等价。
    它通过了，才说明后面"自定义注入"的那套位置/掩码逻辑有资格谈等价性。
    """
    dev = LM._model_device(model)
    emb_layer = model.get_input_embeddings()
    stops = _stop_ids(tokenizer, stop_ids)
    ids_pre = list(split["ids_S"]) + list(split["ids_E"])
    ids_t = list(split["ids_T"])
    with LM.delta_write_disabled(model):
        pre = torch.tensor([ids_pre], dtype=torch.long, device=dev)
        out = model(input_ids=pre, use_cache=True)
        cache = out.past_key_values
        t = torch.tensor([ids_t], dtype=torch.long, device=dev)
        out = model(input_ids=t, past_key_values=cache, use_cache=True,
                    cache_position=torch.arange(len(ids_pre),
                                                len(ids_pre) + len(ids_t), device=dev))
        cache = out.past_key_values
        first = out.logits[0, -1, :].float().clone()
        nxt = int(first.argmax().item())
        n_past = len(ids_pre) + len(ids_t)
        out_ids: List[int] = []
        while len(out_ids) < int(max_new_tokens):
            if nxt in stops:
                break
            out_ids.append(nxt)
            nid = torch.tensor([[nxt]], dtype=torch.long, device=dev)
            out = model(input_ids=nid, past_key_values=cache, use_cache=True,
                        cache_position=torch.arange(n_past, n_past + 1, device=dev))
            cache = out.past_key_values
            n_past += 1
            nxt = int(out.logits[0, -1, :].float().argmax().item())
    pred = tokenizer.decode(out_ids, skip_special_tokens=True)
    meta = {"arm_kind": "prefix_kv", "n_prompt": len(ids_pre) + len(ids_t),
            "n_out": len(out_ids), "n_s": split["ns"], "n_t": split["nt"],
            "cache": "native"}
    return (pred, meta, first) if return_logits else (pred, meta)


def describe() -> str:
    return (f"[midlayer_kv] start_layer={MID_KV_START_LAYER} "
            f"max_mem_tokens={MID_KV_MAX_MEM_TOKENS} encodings={MID_KV_ENCODINGS}")


if __name__ == "__main__":
    print(describe())
