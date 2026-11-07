#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""midlayer_kv.py — 同源中间层 KV 读出（Homologous Mid-Layer KV Readout）。

为什么换读出接口
----------------
实验 A 已经给出两条独立的失败证据：把冻结主干第 18 层的隐藏状态翻译回**第 0 层词向量**
（残差+零初始化 750 步 / LayerNorm+直投影+事实重建 1000 步），验证集 `full` 一直贴着
`none`（20.0 vs 17.5），而明文 `oracle_text` 稳定 55%。训练损失正常下降、验证集不动
= 学的是记忆化，不是可泛化的跨空间翻译。

结论不是"再练久一点"，而是这条通路本身不对：
  - 第 18 层隐藏状态每向量范数实测均值 262、最大 5984，目标 embedding 只有 1.087；
  - 记忆通往主干**只有**一个 ≤256 维的 Reader 输出瓶颈，事实要穿过它再被当作词向量读。

本模块把读出接口换成方案里的第二条路：**记忆不回输入层**，而是由**第 18~35 层自己的
原生注意力**直接读。主干权重、注意力投影全部冻结，只在外侧做"同一坐标系内的搬运"。

三条路径（全部零训练可跑）
--------------------------
| 记忆来源 (`source`) | 每个目标层 ℓ 的 K/V 从哪来 | 回答什么问题 |
|---|---|---|
| `oracle_upper` | 完整证据在**第 ℓ 层输入**的真实隐藏状态 | 只在高层注入记忆，够不够答题（接口天花板） |
| `shared_h18`   | **同一份** H¹⁸ 过各层自己的 `k_proj/v_proj` | 单层状态天然有多少可读性 |
| `none`         | 不注入 | 冻结模型的参数知识下界 |

`oracle_upper` 与 `shared_h18` 在 **ℓ = start_layer 那一层完全相同**（都用 H¹⁸），从下一层
开始分叉。所以两者的差就是"单层状态能否替代逐层状态"这件事本身，不含任何其他变量 ——
这是本模块最核心的对照，`tests/test_midlayer_kv.py` 里有硬断言。

为什么必须先在高层位置做 RoPE、且把问题放在记忆之后
--------------------------------------------------
第 ℓ 层的 `k_proj` 输出必须按**该 token 的位置**施加 RoPE，否则 key 与 query 的相对
位置关系是错的（RoPE 的全部信息都在这个相对偏移里）。所以：
  - 记忆 token 占位置 `0 .. N-1`，K 用这些位置做 RoPE；
  - 问题 token 占位置 `N .. N+n-1`，由 `position_ids` 显式指定；
  - 注意力掩码允许每个问题 token 看到**全部记忆** + 因果范围内的**问题**。
这正是"记忆当成 KV cache 前缀"的语义。第 18 层残差状态虽然还没经过第 18 层的 RoPE，
但**已经**经过前 18 层含位置信息的注意力 —— 所以"存的是 Pre-RoPE 状态"不等于"与位置
无关"，多段拼接场景要单独验（见 EXPERIMENTS.md「已知风险」）。

存储口径
--------
每个 token 存 `d=2560` 个 bf16（5 KB），**远小于**逐层 KV cache（36 层 × K+V），
但远大于原始 token id。真正的压缩要靠后续 `N -> M`，本轮不涉及。

实验二（训练残差 Adapter）的脚手架
----------------------------------
`ResidualKVAdapter` 实现
    H~^(ℓ) = H^(18) + γ_ℓ · Δ_ℓ(LN(H^(18)), q)
`γ_ℓ` 初始化为 0，于是**第 0 步严格等于 `shared_h18`** —— 训练从"已被零训练实验验证过的
起点"长出来，而不是从随机初始化开始赌。配套 `state_loss()` 是逐层逐 token 的密集监督：

    L_state = Σ_ℓ ‖ LN(H~^(ℓ)) − LN(H_teacher^(ℓ)) ‖²

按方案，**实验二必须在实验一（本模块的零训练对照）通过后才开跑**，所以这里只提供
可测的模块与损失，不提供训练脚本。
"""
from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import latent_memory as LM

# 记忆从第几层开始注入（第 18 层 = 隐藏状态 H¹⁸ 产出的那一层）
MID_KV_START_LAYER = LM._env_int("MID_KV_START_LAYER", 18)
# 一次最多带多少记忆 token（只防显存，不参与方法定义）
MID_KV_MAX_MEM_TOKENS = LM._env_int("MID_KV_MAX_MEM_TOKENS", 320)
# 记忆的 K/V 来源：逐层真实状态 vs 单层复用
MID_KV_SOURCES = ("oracle_upper", "shared_h18")
# Adapter 的瓶颈宽度（实验二用；实验一零训练，不涉及）
MID_KV_BOTTLENECK = LM._env_int("MID_KV_BOTTLENECK", 256)


# ---------------------------------------------------------------------
# 0) 模型结构探测（全 duck-typing：不 import transformers，也不写死 Qwen3 类名）
# ---------------------------------------------------------------------
_LAYER_PATHS = ("model.layers", "model.model.layers", "transformer.h", "layers",
                "model.decoder.layers")


def decoder_layers(model: Any) -> List[nn.Module]:
    """主干解码器层列表。路径按 transformers 的实际命名逐个试，全失败就报错。"""
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
    raise ValueError("找不到主干的解码器层（试过 %s）；本模块只支持标准 HF 结构"
                     % ", ".join(_LAYER_PATHS))


def attn_of(layer: nn.Module) -> nn.Module:
    """从解码器层里取注意力模块（自注意力）。"""
    for name in ("self_attn", "attention", "attn", "self_attention"):
        m = getattr(layer, name, None)
        if isinstance(m, nn.Module):
            return m
    raise ValueError(f"层 {type(layer).__name__} 里找不到注意力模块")


def input_norm_of(layer: nn.Module) -> nn.Module:
    """解码器层的**注意力前**归一化（Qwen/Llama 叫 `input_layernorm`）。

    这个模块不是可选的装饰：`hidden_states[l]` 是第 l 层的**残差流**，而第 l 层的
    `k_proj/v_proj` 消费的是 `input_layernorm(残差流)`。把残差流直接喂进投影，得到的
    K/V 与这一层训练时见过的分布差一个归一化（实测残差流每向量范数约 262，归一化后
    是 O(√d)）—— 表现不是"读不出来"，而是**输出直接崩成乱码**，且给真实逐层 KV 时
    同样崩，极易被误读成"高层读不了记忆"。
    """
    for name in ("input_layernorm", "ln_1", "norm1", "attention_norm", "ln_attn"):
        m = getattr(layer, name, None)
        if isinstance(m, nn.Module):
            return m
    raise ValueError(f"层 {type(layer).__name__} 里找不到注意力前归一化"
                     f"（试过 input_layernorm/ln_1/norm1/attention_norm/ln_attn）；"
                     f"跳过它会静默产出错误的 K/V，所以这里直接报错")


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
    if n <= 0 or out <= 0:
        raise ValueError("无法确定 head_dim")
    return out // n


def num_heads_of(attn: nn.Module) -> int:
    """query 头数。"""
    for name in ("num_heads", "num_attention_heads"):
        v = getattr(attn, name, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    cfg = _cfg_of(attn)
    for name in ("num_attention_heads",):
        v = getattr(cfg, name, None) if cfg is not None else None
        if isinstance(v, int) and v > 0:
            return int(v)
    # 最后一条：从 q_proj 的真实输出宽度反推。写成显式两步 ——
    # `a or b if cond else c` 的优先级是 `(a or b) if cond else c`，看着对、改一下就错。
    hd = getattr(attn, "head_dim", None)
    if not isinstance(hd, int) or hd <= 0:
        hd = getattr(cfg, "head_dim", None) if cfg is not None else None
    out = int(getattr(attn.q_proj, "out_features", 0))
    if isinstance(hd, int) and hd > 0 and out > 0:
        return out // hd
    raise ValueError("无法确定 query 头数")


def num_kv_heads_of(attn: nn.Module) -> int:
    """key/value 头数（GQA 下小于 query 头数）。从 k_proj 的真实输出形状推，最可靠。"""
    hd = head_dim_of(attn)
    out = int(getattr(attn.k_proj, "out_features", 0))
    if out > 0 and hd > 0:
        return out // hd
    for name in ("num_key_value_heads", "num_kv_heads"):
        v = getattr(attn, name, None)
        if isinstance(v, int) and v > 0:
            return int(v)
        cfg = _cfg_of(attn)
        v = getattr(cfg, name, None) if cfg is not None else None
        if isinstance(v, int) and v > 0:
            return int(v)
    return num_heads_of(attn)


def rope_module(model: Any) -> nn.Module:
    """RoPE 模块。新版本挂在 model 上（`model.model.rotary_emb`），旧版本挂在每层注意力上。"""
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
    raise ValueError("找不到 RoPE 模块；本模块需要它能被调用为 rotary_emb(x, position_ids)")


def _rope_cos_sin(rope: nn.Module, dtype: torch.dtype, device: Any,
                  positions: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """按位置取 (cos, sin)，形状统一成 [1,1,N,D]（可直接乘 [B,H,N,D]）。

    假输入只需要 dtype/device —— 它的内容不参与计算（RoPE 只依赖 position_ids）。
    精度跟随主干 dtype：cos/sin 在 bf16 下有截断，但问题 token 那边也是同样的截断，
    两边口径一致比"记忆侧用 fp32 更准"更重要。
    """
    pos = torch.as_tensor(positions, dtype=torch.long, device=device).reshape(1, -1)
    probe = torch.zeros((1, 1, int(pos.shape[1])), dtype=dtype, device=device)
    cos, sin = rope(probe, pos)
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def rot_positions(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对最后一维做半分裂 RoPE（Qwen 系默认 rope，不是 interleaved）。

    x: [B,H,N,D]；cos/sin: [B,1,N,D]（或可广播形态）。D 为奇数时直接报错 ——
    RoPE 的半分裂写法在奇数维上没有定义，静默取整会让 key 悄悄错位。
    """
    d = int(x.shape[-1])
    if d % 2 != 0:
        raise ValueError(f"RoPE 需要偶数 head_dim，得到 {d}")
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


# ---------------------------------------------------------------------
# 1) 记忆的 K/V：把一个隐藏状态按各层自己的 W_k/W_v 投影成该层的 key/value
# ---------------------------------------------------------------------
def project_kv(layer: nn.Module, h: torch.Tensor, positions: Any,
               rope: Optional[nn.Module] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """残差流 h [1,N,d] -> 该层 (k, v) [1,Hkv,N,Dh]，k 已按 `positions` 施加 RoPE。

    四个必须照做、漏掉就会静默错的细节：
      - **先过该层的 `input_layernorm`**。传进来的是残差流，而 `k_proj/v_proj` 消费的是
        归一化之后的量。漏这一步不会报错，只会让 K/V 全错、输出崩成乱码。
      - **k 要过 k_norm**（Qwen3 在 `k_proj` 后接一个 head_dim 维 RMSNorm），漏掉等于把
        未归一化的 key 送进注意力，分数尺度整体失真；v 不归一化。
      - **k 要按记忆位置做 RoPE**，位置用调用方给的 `positions`（记忆占 0..N-1）。
      - 投影矩阵必须用**目标层自己的** —— 这正是"各层原生长什么样就读成什么样"，
        不做任何跨层映射（跨层映射是实验二 Adapter 的事）。
    """
    attn = attn_of(layer)
    if h.dim() != 3:
        raise ValueError(f"h 必须是 [1,N,d]，得到 {tuple(h.shape)}")
    p = attn.k_proj.weight
    h = input_norm_of(layer)(h.to(device=p.device, dtype=p.dtype))
    n = int(h.shape[1])
    hd = head_dim_of(attn)
    nkv = num_kv_heads_of(attn)
    if int(attn.k_proj.out_features) != nkv * hd:
        raise ValueError(f"k_proj 输出 {int(attn.k_proj.out_features)} != nkv*hd={nkv * hd}")
    k = attn.k_proj(h).view(1, n, nkv, hd).transpose(1, 2)
    v = attn.v_proj(h).view(1, n, nkv, hd).transpose(1, 2)
    kn = getattr(attn, "k_norm", None)
    if isinstance(kn, nn.Module) and any(True for _ in kn.parameters()):
        k = kn(k)
    if rope is not None:
        cos, sin = _rope_cos_sin(rope, k.dtype, k.device, positions)
        k = rot_positions(k, cos, sin)
    return k.contiguous(), v.contiguous()


def capture_evidence_hidden(model: Any, tokenizer: Any, text: str,
                            start_layer: int = MID_KV_START_LAYER,
                            max_tokens: int = MID_KV_MAX_MEM_TOKENS,
                            upper: bool = True) -> Dict[str, Any]:
    """编码证据原文，返回**各层输入**的隐藏状态。

    口径：`hidden_states[l]` 是"第 l 层的输入"（output_hidden_states 的第一项是 embedding，
    所以 `hidden_states[l]` = 前 l 层（0..l-1）的输出）。注意它是**残差流**，
    第 l 层的 `k_proj/v_proj` 消费的是 `input_layernorm(它)` —— 这一步在 `project_kv` 里做。

    `upper=True` 时才保存 start_layer 之上的每一层（`oracle_upper` 需要）；
    只做 `shared_h18` 时传 False，省一半显存。

    为什么整段一起编码而不是逐单元编码再拼：`oracle_upper` 要复现"这段证据进了明文
    prompt 之后在第 l 层的真实 K/V"，逐单元编码会切断跨句注意力，得到的就不是那个量。
    """
    dev = LM._model_device(model)
    ids = LM._tokenize_no_special(tokenizer, str(text), dev)
    if int(max_tokens) > 0 and int(ids.shape[1]) > int(max_tokens):
        ids = ids[:, : int(max_tokens)]
    n = int(ids.shape[1])
    if n <= 0:
        raise ValueError("证据文本分词后为空，无法建立记忆")
    n_layers = len(decoder_layers(model))
    if int(start_layer) >= n_layers:
        raise ValueError(f"start_layer={start_layer} 超出主干层数 {n_layers}")
    # 编码期必须关掉 Delta 写入：挂了 adapter 时 `write_enabled` 默认为 True，
    # 离线编码证据会顺手改动在线状态，同一次评测里前后样本看到的上下文就不一样了。
    with LM.delta_write_disabled(model), torch.no_grad():
        out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise RuntimeError("主干没有返回 hidden_states（需要 output_hidden_states=True）")
    if len(hs) < n_layers:
        raise RuntimeError(f"hidden_states 只有 {len(hs)} 层，主干有 {n_layers} 层")
    layers = list(range(int(start_layer), n_layers)) if upper else [int(start_layer)]
    return {"h": {l: hs[l].detach() for l in layers},
            "n": n, "n_layers": n_layers, "start_layer": int(start_layer),
            "text_tokens": n}


def build_memory_kv(model: Any, cap: Dict[str, Any], source: str,
                    start_layer: Optional[int] = None) -> "MemoryKV":
    """把捕获到的隐藏状态投影成各目标层的 K/V。

    source=`shared_h18`：所有层都用**同一份** H^start（单层复用）。
    source=`oracle_upper`：第 l 层用第 l 层自己的真实输入状态（逐层真实）。
    """
    source = str(source)
    if source not in MID_KV_SOURCES:
        raise ValueError(f"未知的记忆来源 {source!r}，可选 {MID_KV_SOURCES}")
    sl = int(cap["start_layer"] if start_layer is None else start_layer)
    n_layers = int(cap["n_layers"])
    rope = rope_module(model)
    layers = decoder_layers(model)
    h_shared = cap["h"][sl]
    per: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for l in range(sl, n_layers):
        h = h_shared if source == "shared_h18" else cap["h"][l]
        per[l] = project_kv(layers[l], h, positions=torch.arange(
            int(h.shape[1]), device=h.device), rope=rope)
    return MemoryKV(per, n_tokens=int(cap["n"]), start_layer=sl, source=source,
                    rope=rope, hidden_size=int(h_shared.shape[-1]))


class MemoryKV:
    """一段记忆在若干层上的 (k, v)。**只存在，不修改主干**。

    刻意不做成 nn.Module：实验一是零训练对照，记忆必须是"外部常量"；
    做成 Module 会让它意外进优化器，把"接口是否可读"变成"能不能训出来"。
    """

    def __init__(self, per_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                 n_tokens: int, start_layer: int, source: str,
                 rope: Optional[nn.Module] = None, hidden_size: int = 0):
        self.per_layer = dict(per_layer)
        self.n = int(n_tokens)
        self.start_layer = int(start_layer)
        self.source = str(source)
        self.rope = rope
        self.hidden_size = int(hidden_size)
        self._mask_cache: Dict[Any, torch.Tensor] = {}

    # ---- 访问 ----
    @property
    def layers(self) -> List[int]:
        return sorted(self.per_layer)

    def k(self, layer: int) -> torch.Tensor:
        return self.per_layer[int(layer)][0]

    def v(self, layer: int) -> torch.Tensor:
        return self.per_layer[int(layer)][1]

    # ---- 记账：存储成本必须能被审计，不能只写"5KB/token"这种口头结论 ----
    def nbytes_kv(self) -> int:
        tot = 0
        for k, v in self.per_layer.values():
            tot += k.numel() * k.element_size() + v.numel() * v.element_size()
        return int(tot)

    def bytes_per_token(self) -> float:
        return float(self.nbytes_kv()) / max(1, self.n)

    def state_bytes_per_token(self) -> float:
        """单层状态（H^start）本身的存储：bf16 下 2560 维 = 5120 B。"""
        return float(self.hidden_size * 2)

    def to(self, device: Any) -> "MemoryKV":
        dev = torch.device(device)
        self.per_layer = {l: (k.to(dev), v.to(dev))
                          for l, (k, v) in self.per_layer.items()}
        self._mask_cache.clear()
        return self

    def mask(self, q_len: int, dtype: torch.dtype, device: Any) -> torch.Tensor:
        # device 也进 key：同一份记忆在不同设备上再跑时会拿到上一个设备建的掩码，
        # 而这种错误只在 SDPA 内部以设备不匹配的形式冒出来，极难回指到这里。
        key = (int(q_len), str(dtype), str(device))
        m = self._mask_cache.get(key)
        if m is None:
            m = build_prefix_causal_mask(int(q_len), self.n, dtype, device)
            self._mask_cache[key] = m
        return m

    def describe(self) -> Dict[str, Any]:
        return {"source": self.source, "start_layer": self.start_layer,
                "n_tokens": self.n, "n_layers": len(self.per_layer),
                "kv_MB": round(self.nbytes_kv() / 1e6, 2),
                "kv_bytes_per_token": round(self.bytes_per_token(), 1),
                "state_bytes_per_token": round(self.state_bytes_per_token(), 1)}


def build_prefix_causal_mask(q_len: int, n_mem: int, dtype: torch.dtype,
                             device: Any) -> torch.Tensor:
    """[1,1,q_len,n_mem+q_len] 加性掩码：记忆全可见 + 问题内部因果。

    记忆在左边（位置 0..n_mem-1），问题在右边。这正是"KV cache 前缀"的形状：
    每个问题 token 都能看到**整段记忆**（记忆是离线写入的、与当前问题无关），
    但看不到自己后面的问题 token。把记忆列也做成因果（只让问题最后一位看全部记忆）
    会退化成"只读最后几个词"，那不是本方法的语义。
    """
    if q_len <= 0 or n_mem < 0:
        raise ValueError(f"掩码尺寸非法：q_len={q_len} n_mem={n_mem}")
    allowed = torch.zeros((q_len, n_mem + q_len), dtype=torch.bool)
    if n_mem > 0:
        allowed[:, :n_mem] = True
    allowed[:, n_mem:] = torch.tril(torch.ones((q_len, q_len), dtype=torch.bool))
    mask = torch.zeros((1, 1, q_len, n_mem + q_len), dtype=dtype, device=device)
    return mask.masked_fill(~allowed.unsqueeze(0).unsqueeze(0).to(device),
                            torch.finfo(dtype).min)


# ---------------------------------------------------------------------
# 2) 注入：把记忆 K/V 挂到目标层的原生注意力上（不改权重、不用 cache）
# ---------------------------------------------------------------------
_ACTIVE: Dict[int, "MemoryKV"] = {}


def _question_kv(attn: nn.Module, hidden_states: torch.Tensor,
                 position_embeddings: Any) -> Tuple[torch.Tensor, torch.Tensor,
                                                    torch.Tensor]:
    """按目标层自己的投影算问题侧的 q/k/v，并施加位置编码。"""
    bsz, q_len, _ = hidden_states.shape
    hd = head_dim_of(attn)
    n_q = num_heads_of(attn)
    n_kv = num_kv_heads_of(attn)
    q = attn.q_proj(hidden_states).view(bsz, q_len, n_q, hd).transpose(1, 2)
    k = attn.k_proj(hidden_states).view(bsz, q_len, n_kv, hd).transpose(1, 2)
    v = attn.v_proj(hidden_states).view(bsz, q_len, n_kv, hd).transpose(1, 2)
    qn = getattr(attn, "q_norm", None)
    if isinstance(qn, nn.Module) and any(True for _ in qn.parameters()):
        q = qn(q)
    kn = getattr(attn, "k_norm", None)
    if isinstance(kn, nn.Module) and any(True for _ in kn.parameters()):
        k = kn(k)
    if position_embeddings is None:
        raise RuntimeError("注入路径要求主干传入 position_embeddings"
                           "（Qwen 系 decoder layer 会传；否则本模块无法定位问题位置）")
    cos, sin = position_embeddings
    q = rot_positions(q, cos, sin)
    k = rot_positions(k, cos, sin)
    return q, k, v


def _midlayer_attn_forward(self, hidden_states, position_embeddings=None,
                           attention_mask=None, past_key_value=None,
                           cache_position=None, position_ids=None, **kwargs):
    """安装了记忆的那几层改走这里：问题 K/V 与记忆 K/V 拼成一条 K/V 序列。

    与原生 forward 的差别**只有**"K/V 序列左边多了一段记忆"，其余（q/k 投影、
    q_norm/k_norm、RoPE、o_proj、缩放）全部照原层执行 —— 这正是方案要的
    "用原生注意力读记忆"，而不是再学一个读出器。
    """
    entry = _ACTIVE.get(id(self))
    if entry is None:
        raise RuntimeError("中间层 KV 包装器没有登记记忆（_ACTIVE 缺项）")
    mem = entry
    if past_key_value is not None:
        # 本路径不做增量解码，也不接管 KV cache。悄悄把 cache 丢掉会让"第二轮之后
        # 什么都看不到"退化成静默错误，所以直接拒绝。
        raise RuntimeError("中间层 KV 注入不支持 past_key_value（本路径逐步整段前向）")
    if int(hidden_states.shape[0]) != 1:
        # 记忆 K/V 只对应一条样本。batch>1 时 cat 会因为 batch 维不匹配而炸，
        # 但报错信息指向 cat 而不是"这里只支持单条"，所以提前拦下来。
        raise ValueError(f"记忆 K/V 是单条样本的（batch=1），收到 batch="
                         f"{int(hidden_states.shape[0])}")
    q, k, v = _question_kv(self, hidden_states, position_embeddings)
    dev, dt = k.device, k.dtype
    k_mem = mem.k(_layer_of_attn(self)).to(device=dev, dtype=dt)
    v_mem = mem.v(_layer_of_attn(self)).to(device=dev, dtype=dt)
    K = torch.cat([k_mem, k], dim=2)
    V = torch.cat([v_mem, v], dim=2)
    n_rep = int(q.shape[1]) // int(K.shape[1]) if int(K.shape[1]) > 0 else 1
    if n_rep > 1:                                        # GQA：SDPA 要求 q/kv 头数一致
        K = K.repeat_interleave(n_rep, dim=1)
        V = V.repeat_interleave(n_rep, dim=1)
    q_len = int(q.shape[2])
    mask = mem.mask(q_len, dt, dev)
    out = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
    out = out.transpose(1, 2).reshape(int(hidden_states.shape[0]), q_len, -1)
    return self.o_proj(out.contiguous()), None


def _layer_of_attn(attn: nn.Module) -> int:
    l = getattr(attn, "_midlayer_layer", None)
    if l is None:
        raise RuntimeError("注意力模块缺少 _midlayer_layer 标记")
    return int(l)


def _install(attn: nn.Module, layer: int, mem: MemoryKV) -> Tuple[nn.Module, bool, Any]:
    had = "forward" in attn.__dict__
    prev = attn.__dict__.get("forward")
    prev_layer = attn.__dict__.get("_midlayer_layer")
    attn._midlayer_layer = int(layer)
    attn.forward = types.MethodType(_midlayer_attn_forward, attn)
    _ACTIVE[id(attn)] = mem
    return (attn, had, (prev, prev_layer))


def _restore(rec: Tuple[nn.Module, bool, Any]) -> None:
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
def inject_memory_kv(model: Any, mem: Optional[MemoryKV],
                     start_layer: Optional[int] = None):
    """在 `[start_layer, 末层]` 上挂记忆 K/V；退出时**逐模块恢复**原 forward。

    恢复必须逐模块做（不能"改一个类属性了事"）：主干是共享对象，一旦有模块留在
    被包装状态，后面所有不带记忆的前向都会莫名其妙地要求 `_ACTIVE` 里有登记，
    而这种错误在跑分时只表现为分数异常，很难定位到是残留的包装器。
    """
    if mem is None or int(mem.n) <= 0:
        yield False
        return
    sl = int(mem.start_layer if start_layer is None else start_layer)
    layers = decoder_layers(model)
    recs: List[Tuple[nn.Module, bool, Any]] = []
    try:
        for l in range(sl, len(layers)):
            recs.append(_install(attn_of(layers[l]), l, mem))
        yield True
    finally:
        for rec in reversed(recs):
            _restore(rec)


# ---------------------------------------------------------------------
# 3) 读出：问题单独过前 18 层，第 18 层起由原生注意力读记忆
# ---------------------------------------------------------------------
@torch.no_grad()
def midlayer_kv_generate(model: Any, tokenizer: Any, mem: Optional[MemoryKV],
                         question: str, system_prompt: str = "",
                         max_new_tokens: int = 16,
                         stop_ids: Optional[Sequence[int]] = None,
                         extra_meta: Optional[Dict[str, Any]] = None
                         ) -> Tuple[str, Dict[str, Any]]:
    """贪心解码，不走 KV cache。

    为什么不复用 `model.generate`：注入需要"问题位置从 N 开始"，而 `generate` 的
    cache 与 mask 书签会按"cache 里已有 N 条"重算，与"只有部分层有记忆"这个非标准
    形态冲突。整段重前向虽然 O(T²)，但语义只有一条：**每一层都在用同一份记忆**。
    一个 prompt 约 50 token、答案 16 token，代价可以接受。
    """
    dev = LM._model_device(model)
    text, _n_sys = LM.build_chat_parts(tokenizer, system_prompt, question)
    ids = LM._tokenize_no_special(tokenizer, text, dev)
    emb_layer = model.get_input_embeddings()
    cur = emb_layer(ids)
    n_prompt = int(ids.shape[1])
    stops = set(int(i) for i in (stop_ids or []) if i is not None)
    if not stops:
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            stops.add(int(eos))
    n_mem = int(mem.n) if mem is not None else 0
    out_ids: List[int] = []
    with inject_memory_kv(model, mem), LM.delta_write_disabled(model):
        for _ in range(int(max_new_tokens)):
            n = int(cur.shape[1])
            # 问题位置从 n_mem 开始：记忆占 0..n_mem-1，两边 RoPE 位置连续且不重叠
            pos = torch.arange(n_mem, n_mem + n, device=dev,
                               dtype=torch.long).unsqueeze(0)
            o = model(inputs_embeds=cur, use_cache=False, position_ids=pos)
            nxt = int(o.logits[0, -1].argmax(dim=-1).item())
            if nxt in stops:
                break
            out_ids.append(nxt)
            cur = torch.cat([cur, emb_layer(torch.tensor([[nxt]], device=dev))], dim=1)
    pred = tokenizer.decode(out_ids, skip_special_tokens=True)
    meta: Dict[str, Any] = {
        "n_prompt": n_prompt, "n_out": len(out_ids), "n_mem": n_mem,
        "max_new_tokens": int(max_new_tokens), "decode": "greedy_no_cache",
        "position_mode": "memory_prefix" if n_mem else "plain",
    }
    if mem is not None:
        meta.update(mem.describe())
    meta.update(dict(extra_meta or {}))
    return pred, meta


# ---------------------------------------------------------------------
# 4) 实验二脚手架：残差 Adapter（H^18 -> 各高层状态）+ 状态蒸馏损失
# ---------------------------------------------------------------------
class ResidualKVAdapter(nn.Module):
    """H~^(ℓ) = H^(18) + γ_ℓ · Δ_ℓ(LN(H^(18)), q)，γ 初始为 0。

    为什么残差在这里成立、在实验 A 里不成立：实验 A 要跨**表示空间**（第 18 层 ->
    词向量），残差项长不到能改方向的量级，上游梯度实测恰好为 0。这里输入输出同在
    第 18 层坐标系，γ 直接乘在被监督的量上，第 0 步就等于已验证的 `shared_h18`。
    所以这里**可以**用残差 + 零初始化，那是正确的归纳偏置，而不是偷懒。

    问题条件化（`q_vec` 非空时）用低秩门控，与 `QueryCondReader` 同一套写法：
    记忆是写入时生成的，同一段记忆要服务不同问题，适配必须看问题。
    """

    def __init__(self, hidden_size: int, start_layer: int, end_layer: int,
                 bottleneck: int = MID_KV_BOTTLENECK):
        super().__init__()
        if int(end_layer) < int(start_layer):
            raise ValueError(f"end_layer={end_layer} < start_layer={start_layer}")
        self.hidden_size = int(hidden_size)
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer)
        self.bottleneck = int(bottleneck)
        n = self.end_layer - self.start_layer + 1
        self.ln = nn.LayerNorm(self.hidden_size)
        self.down = nn.Linear(self.hidden_size, self.bottleneck, bias=False)
        self.q_proj = nn.Linear(self.hidden_size, self.bottleneck, bias=False)
        self.up = nn.ModuleList([nn.Linear(self.bottleneck, self.hidden_size)
                                 for _ in range(n)])
        # γ 零初始化：第 0 步严格是恒等映射（= shared_h18）
        self.gamma = nn.Parameter(torch.zeros(n))
        for u in self.up:
            nn.init.normal_(u.weight, std=0.02)
            nn.init.zeros_(u.bias)
        self.last_gamma: List[float] = []

    def forward(self, h18: torch.Tensor,
                q_vec: Optional[torch.Tensor] = None) -> Dict[int, torch.Tensor]:
        """h18 [1,N,d] -> {层号: H~^(ℓ) [1,N,d]}"""
        if h18.dim() != 3 or int(h18.shape[-1]) != self.hidden_size:
            raise ValueError(f"h18 必须是 [1,N,{self.hidden_size}]，得到 {tuple(h18.shape)}")
        x = self.ln(h18)
        f = self.down(x)                                  # [1,N,b]
        if q_vec is not None:
            q = self.q_proj(q_vec.reshape(1, -1)[:, : self.hidden_size])
            gate = torch.sigmoid((f * q.unsqueeze(1)).sum(dim=-1, keepdim=True)
                                 / (self.bottleneck ** 0.5))
            f = f * gate
        out: Dict[int, torch.Tensor] = {}
        with torch.no_grad():
            self.last_gamma = [float(g) for g in self.gamma.detach()]
        for i, l in enumerate(range(self.start_layer, self.end_layer + 1)):
            out[l] = h18 + self.gamma[i] * self.up[i](f)
        return out


def adapter_memory_kv(model: Any, adapter: ResidualKVAdapter,
                      h18: torch.Tensor, q_vec: Optional[torch.Tensor] = None
                      ) -> MemoryKV:
    """Adapter 输出的各层状态 -> MemoryKV（与零训练路径共用同一套投影）。"""
    states = adapter(h18, q_vec)
    rope = rope_module(model)
    layers = decoder_layers(model)
    per: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for l, h in states.items():
        per[int(l)] = project_kv(layers[int(l)], h,
                                 positions=torch.arange(int(h.shape[1]), device=h.device),
                                 rope=rope)
    return MemoryKV(per, n_tokens=int(h18.shape[1]), start_layer=adapter.start_layer,
                    source="adapter", rope=rope, hidden_size=int(h18.shape[-1]))


def state_loss(adapted: Dict[int, torch.Tensor],
               teacher: Dict[int, torch.Tensor]) -> torch.Tensor:
    """L_state = Σ_ℓ ‖LN(H~^(ℓ)) − LN(H_teacher^(ℓ))‖²（先层内归一，再按 token 与层平均）。

    先 LN 再比：两边的**尺度**本就不是我们想约束的东西（残差坐标系相同，差的是
    "第 ℓ 层该往哪个方向偏"），直接算欧氏距离会让损失被大范数位置主导。
    缺层直接报错而不是跳过 —— 静默跳过会让 loss 看起来很漂亮，实际只监督了一两层。
    """
    if not adapted:
        raise ValueError("state_loss 收到空的 adapted")
    missing = [l for l in adapted if int(l) not in {int(k) for k in teacher}]
    if missing:
        raise ValueError(f"teacher 缺少第 {sorted(missing)} 层的真实状态")
    terms = []
    for l in sorted(adapted):
        a, t = adapted[l], teacher[int(l)]
        if tuple(a.shape) != tuple(t.shape):
            raise ValueError(f"第 {l} 层形状不一致：{tuple(a.shape)} vs {tuple(t.shape)}")
        d = int(a.shape[-1])
        ln_a = F.layer_norm(a.float(), (d,))
        ln_t = F.layer_norm(t.float(), (d,))
        terms.append(((ln_a - ln_t) ** 2).mean())
    return torch.stack(terms).mean()


def kl_agree_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                  n_ans: int, tau: float = 1.0) -> torch.Tensor:
    """答案位置的分布对齐（方案里的 L_KL）。只用于实验二，实验一不需要。"""
    if int(n_ans) <= 0:
        raise ValueError("kl_agree_loss 需要 n_ans>0")
    s = student_logits[0, -(int(n_ans) + 1):-1, :]
    t = teacher_logits[0, -(int(n_ans) + 1):-1, :]
    if s.shape != t.shape:
        raise ValueError(f"学生/教师答案段形状不一致：{tuple(s.shape)} vs {tuple(t.shape)}")
    return F.kl_div(F.log_softmax(s.float() / tau, dim=-1),
                    F.softmax(t.float() / tau, dim=-1),
                    reduction="batchmean") * (tau ** 2)


def describe() -> str:
    return (f"[midlayer_kv] start_layer={MID_KV_START_LAYER} "
            f"max_mem_tokens={MID_KV_MAX_MEM_TOKENS} sources={MID_KV_SOURCES}")


if __name__ == "__main__":
    print(describe())
