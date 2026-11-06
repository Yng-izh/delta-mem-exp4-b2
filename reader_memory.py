#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""reader_memory.py — 实验 A/B：**完整隐藏状态记忆** + 轻量 Query-conditioned Reader。

为什么要换一套接口（而不是继续改 Compressor/Bridge）
----------------------------------------------------
前几轮的证据链已经很硬：注入接口没问题（整句原话 56% ≈ 明文 53.5%），符号保真是
必要条件（+15 点，p=0.0003），而"压缩出来的连续关系向量"净贡献为零、还占位置亏分
（实验 2：4+4 的 24.5% 对 8 符号的 37.5%）。

问题不在"压缩得不够好"，而在**读的接口**：把记忆压成几个向量、直接塞进 prompt，
等于要求冻结模型自己去解压。所以先做实验 A —— **不压缩**，把整段证据的隐藏状态
原样当记忆，只训一个轻量的、看问题的 Reader，先证明"这条读取通路能把事实送进去"。

三个部件
--------
1. **记忆**：每个金标证据单元在冻结主干第 `layer` 层的**逐 token 隐藏状态**，
   按原文顺序拼成一整段 `[N, d]`。N 就是 token 数（不压缩，通常 100~200）。
2. **Reader**（`QueryCondReader`）：逐 token 的**问题条件化**变换
       gate_i = sigmoid( (W_h h_i)·(W_q q) / √b + β )
       out_i  = h_i + W_2( SiLU( W_1 [h_i ; gate_i·(W_h h_i)·(W_q q)] ) )
   然后按**整向量 L2** 对齐到真实 embedding 的每向量范数（≈1.087）。可训练参数约 1~2M。
   为什么是逐 token 变换而不是"再从记忆里挑几个"：实验 A 要验证的是"读取通路通不通"，
   一旦让 Reader 自己挑，就又变成压缩问题了，会把两个变量混在一起。
3. **注入**：整段变换后的向量按原文顺序插进 user 消息体内的 `Evidence:` 段，
   与明文 oracle 逐字同构（复用 `latent_memory._emit_latent`）。

三个对照（缺一不可）
--------------------
| 模式 | 记忆 | 回答的问题 |
|---|---|---|
| `none` | 不给 | 冻结模型靠参数知识能到多少（下界） |
| `shuffle` | 给，但**打乱 token 顺序** | Reader 到底有没有在用记忆内容（关键对照） |
| `full` | 给完整记忆 | 这条读取通路的上限 |
| `oracle_text` | 给明文原文 | 参照上界（与潜层无关） |

**打乱记忆这一条是整个实验的命门**：如果 `full` 明显高于 `shuffle`，说明 Reader 真的
在读内容；如果两者差不多，那 Reader 只是在给模型加一段"有信息量的噪声"，
后面的压缩实验就无从谈起。

实验 B 的接口
-------------
把记忆来源从"完整隐藏状态"换成"压缩器的 K 个潜槽"即可，**Reader 完全不变**：
    mem = compressor(units)          # [K, d]
    soft = reader(mem, q)            # [K, d]
于是"压缩比"只改变记忆长度，读取通路保持不变 —— 这正是实验 B 要的"固定 Reader"。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import latent_memory as LM
import train_latent_memory as T

# 记忆取第几层：与压缩器看的是同一层，方便两条路线直接对照
READER_LAYER = LM._env_int("READER_LAYER", 18)
# Reader 的瓶颈宽度。d=2560 时 1.3M 参数左右；太小表达不了问题条件化，太大就不是"轻量"
READER_BOTTLENECK = LM._env_int("READER_BOTTLENECK", 256)
# 完整记忆最多保留多少 token：只用于防显存爆，不参与方法定义
READER_MEM_MAX = LM._env_int("READER_MEM_MAX", 320)
# 记忆来源：full=完整隐藏状态；slotsK=压缩器 K 个潜槽（实验 B）
READER_MEM_MODES = ("full", "slots8", "slots4", "slots2")
# 打乱方式：tokens=同一段记忆里打乱 token 顺序；wrong=换成别的样本的记忆
READER_SHUFFLE_MODES = ("none", "tokens", "wrong")


class QueryCondReader(nn.Module):
    """逐 token 的问题条件化变换，输出直接可注入的 embedding 序列。

    为什么必须"条件化"而不是一个纯 MLP：记忆是写入时生成的（Query-Agnostic），
    同一段证据要服务不同的问题。如果 Reader 不看问题，它只能把记忆原样转一遍，
    那"训练"这件事就没有意义（实验 A 会退化成"给原词向量"）。条件化让同一段记忆
    在不同问题下呈现不同形态 —— 这是唯一能在**冻结主干 + 不压缩**的前提下
    让记忆变得"可读"的杠杆。
    """

    NORM_MIN = 0.25
    NORM_MAX = 4.0

    def __init__(self, hidden_size: int, bottleneck: int = READER_BOTTLENECK,
                 target_norm: float = 1.087, tau: float = 1.0):
        super().__init__()
        if int(hidden_size) <= 0 or int(bottleneck) <= 0:
            raise ValueError(f"hidden/bottleneck 必须为正：{hidden_size}/{bottleneck}")
        self.hidden_size = int(hidden_size)
        self.bottleneck = int(bottleneck)
        self.tau = float(tau)
        # 门控：问题向量与记忆 token 的内积 -> 每个 token 一个"该看多重"的标量。
        # 用低秩投影（d -> bottleneck）而不是 d×d，参数量从 6.5M 降到 1.3M。
        self.q_proj = nn.Linear(self.hidden_size, self.bottleneck, bias=False)
        self.h_proj = nn.Linear(self.hidden_size, self.bottleneck, bias=False)
        # 变换：输入是 [h_i ; 门控后的低秩特征]，输出回 hidden_size 的**残差**
        self.mlp1 = nn.Linear(self.hidden_size + self.bottleneck, self.bottleneck)
        self.mlp2 = nn.Linear(self.bottleneck, self.hidden_size)
        self.gate_bias = nn.Parameter(torch.zeros(1))
        # 末层零初始化：第 0 步输出逐位等于原始隐藏状态（"不改记忆"的起点），
        # 与 Bridge 的零初始化同一个理由 —— 让训练从"什么都不做"开始长。
        nn.init.zeros_(self.mlp2.weight)
        nn.init.zeros_(self.mlp2.bias)
        # 范数对齐：真实 token embedding 的每向量范数（可学习、有硬上下界）
        self.out_norm = nn.Parameter(torch.tensor(float(target_norm)))
        self.collect_stats = False
        self.last_stats: Dict[str, float] = {}

    def scale(self) -> torch.Tensor:
        return self.out_norm.clamp(self.NORM_MIN, self.NORM_MAX)

    def forward(self, mem_h: torch.Tensor, mem_mask: Optional[torch.Tensor] = None,
                q_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """mem_h [B,N,d] + q_vec [B,d] -> [B,N,d]（padding 位置输出全零）。

        q_vec 为 None 时退化成"不看问题"的 Reader（留给消融，不作为主方法）。
        """
        if mem_h.dim() != 3:
            raise ValueError(f"mem_h 必须是 [B,N,d]，得到 {tuple(mem_h.shape)}")
        if int(mem_h.shape[-1]) != self.hidden_size:
            raise ValueError(f"mem_h 最后一维 {int(mem_h.shape[-1])} != hidden={self.hidden_size}")
        B, N, d = int(mem_h.shape[0]), int(mem_h.shape[1]), int(mem_h.shape[2])
        p = next(self.parameters())
        h = mem_h.to(device=p.device, dtype=p.dtype)
        hb = self.h_proj(h)                                   # [B,N,b]
        if q_vec is None:
            gate = torch.ones((B, N, 1), device=p.device, dtype=p.dtype)
            feat = hb
        else:
            q = q_vec.to(device=p.device, dtype=p.dtype).reshape(B, -1)[:, :d]
            qb = self.q_proj(q).unsqueeze(1)                  # [B,1,b]
            # 除以 √b 是标准缩放：不除的话点积随瓶颈宽度线性增长，sigmoid 会饱和成 0/1
            gate = torch.sigmoid((hb * qb).sum(dim=-1, keepdim=True)
                                 / (self.bottleneck ** 0.5) + self.gate_bias)
            feat = hb * gate
        delta = self.mlp2(F.silu(self.mlp1(torch.cat([h, feat], dim=-1))))
        out = h + delta
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        # 整向量 L2 对齐（**不是** RMSNorm×常数：那会把范数推到 ≈target·√d）
        n = out.norm(dim=-1, keepdim=True)
        unit = out / n.clamp_min(1.0)
        out = unit * self.scale().to(device=out.device, dtype=out.dtype)
        if mem_mask is not None:
            m = mem_mask.to(device=p.device).to(dtype=out.dtype).unsqueeze(-1)
            out = out * m
        if self.collect_stats:
            with torch.no_grad():
                self.last_stats = {
                    "scale": float(self.scale()),
                    "out_norm": float(out.norm(dim=-1).mean()),
                    "gate_mean": float(gate.mean()),
                    "delta_ratio": float(delta.norm(dim=-1).mean()
                                         / n.clamp_min(1e-6).mean()),
                    "w_last": float(self.mlp2.weight.norm()),
                }
        return out


# ---------------------------------------------------------------------
# 记忆构建
# ---------------------------------------------------------------------
@torch.no_grad()
def build_full_memory(model, tokenizer, item, unit_idx: Sequence[int],
                      layer: int = READER_LAYER,
                      max_tokens: int = READER_MEM_MAX) -> Dict[str, Any]:
    """金标证据单元的**逐 token 隐藏状态**，按原文顺序拼成 [N, d]。

    为什么取第 layer 层而不是最后一层：压缩器（以及阶段一的寻址键）用的就是这一层，
    两条路线必须看同一份特征，否则"完整记忆 vs 压缩记忆"的对照里混进了层数这个变量。
    """
    dev = LM._model_device(model)
    texts = [T.unit_encode_text(item, int(i)) for i in unit_idx]
    if not texts:
        return {"h": torch.zeros((1, 0, int(LM._model_hidden_size(model)))),
                "mask": torch.zeros((1, 0), dtype=torch.long), "n": 0, "texts": []}
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
    ids = enc["input_ids"].to(dev)
    am = enc["attention_mask"].to(dev)
    out = model(input_ids=ids, attention_mask=am, output_hidden_states=True, use_cache=False)
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise RuntimeError("模型没有返回 hidden_states（需要 output_hidden_states=True）")
    if int(layer) >= len(hs):
        raise ValueError(f"layer={layer} 超出 hidden_states 层数 {len(hs)}")
    h = hs[int(layer)].detach()
    # 按单元顺序拼成一整段（去掉 padding），再截到 max_tokens
    rows: List[torch.Tensor] = []
    for r in range(len(texts)):
        n = int(am[r].sum().item())
        if n > 0:
            rows.append(h[r, :n])
    if not rows:
        return {"h": torch.zeros((1, 0, int(h.shape[-1]))),
                "mask": torch.zeros((1, 0), dtype=torch.long), "n": 0, "texts": texts}
    cat = torch.cat(rows, dim=0)
    if int(max_tokens) > 0 and int(cat.shape[0]) > int(max_tokens):
        cat = cat[: int(max_tokens)]
    N = int(cat.shape[0])
    return {"h": cat.unsqueeze(0), "mask": torch.ones((1, N), dtype=torch.long,
                                                      device=cat.device),
            "n": N, "texts": texts}


@torch.no_grad()
def build_slot_memory(model, tokenizer, compressor, item, unit_idx: Sequence[int],
                      max_tokens: int = READER_MEM_MAX,
                      layer: int = READER_LAYER) -> Dict[str, Any]:
    """实验 B 的记忆来源：压缩器把每个单元压成 K 个潜槽，再拼成 [K*n_units, d]。

    与实验 A 的接口**完全一致**（都是 [N, d] 的隐藏状态序列），所以 Reader 一行不用改。
    这也是"固定 Reader、只换记忆长度"这个要求的实现方式。
    """
    dev = LM._model_device(model)
    texts = [T.unit_encode_text(item, int(i)) for i in unit_idx]
    if not texts:
        d = int(getattr(compressor, "hidden_size", 0))
        return {"h": torch.zeros((1, 0, d)), "mask": torch.zeros((1, 0), dtype=torch.long),
                "n": 0, "texts": []}
    feats = LM.encode_latent_units(model, tokenizer, texts, layer=int(layer),
                                   max_unit_tokens=int(getattr(compressor, "max_unit_tokens",
                                                               LM.LATENT_MAX_UNIT_TOKENS)),
                                   parent_ids=[str(i) for i in range(len(texts))],
                                   device=dev)
    # 压缩器 forward 返回 z_sem [块数, K, d]：每个单元可能切成多块，块按原序排列，
    # 所以 reshape 之后就是"按原文顺序铺开的 K×块数 个潜槽"。
    z = compressor(feats["h0"].to(dev), feats["h18"].to(dev), feats["mask"].to(dev))
    slots = z.reshape(-1, int(z.shape[-1]))          # [块数*K, d]
    if int(max_tokens) > 0 and int(slots.shape[0]) > int(max_tokens):
        slots = slots[: int(max_tokens)]
    N = int(slots.shape[0])
    return {"h": slots.unsqueeze(0),
            "mask": torch.ones((1, N), dtype=torch.long, device=slots.device),
            "n": N, "texts": texts}


def shuffle_memory(mem: Dict[str, Any], mode: str = "tokens",
                   rng: Optional[torch.Generator] = None) -> Dict[str, Any]:
    """打乱记忆（实验 A 的关键对照）。

    `tokens`：把同一段记忆里的 token 顺序随机重排 —— 内容还在，但**顺序与绑定关系没了**。
      这是最严格的一条：如果 Reader 真的在用"哪个词挨着哪个词"，打乱就该掉分；
      如果它只是在做词袋式的加权，打乱几乎不掉分 —— 那"关系绑定"这个卖点就不成立。
    `wrong`：换成**别的样本**的记忆（由调用方负责替换），这里只做校验。

    注意：`h` 是 [1,N,d]，`mask` 全 1 时重排才等价于"只换顺序"；有 padding 时只重排
    有效位，避免把 padding 的零向量搬到前面（那会变成"多给了一堆空位"，是另一个变量）。
    """
    mode = str(mode or "none").lower()
    if mode in ("none", ""):
        return mem
    h, m = mem["h"], mem["mask"]
    N = int(h.shape[1])
    if N <= 1:
        return mem
    g = rng if rng is not None else torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g).to(h.device)
    if mode == "tokens":
        return {**mem, "h": h[:, perm, :], "mask": m[:, perm]}
    if mode == "wrong":
        # 真正的替换在调用方做（需要另一条样本的记忆）；这里只记一笔，不改内容
        return mem
    raise ValueError(f"未知的打乱方式 {mode!r}，可选 {READER_SHUFFLE_MODES}")


# ---------------------------------------------------------------------
# 推理：记忆 -> Reader -> 注入 -> 生成
# ---------------------------------------------------------------------
@torch.no_grad()
def reader_generate(model, tokenizer, reader: Optional[QueryCondReader],
                    mem: Optional[Dict[str, Any]], question: str,
                    system_prompt: str = "", q_vec: Optional[torch.Tensor] = None,
                    max_new_tokens: int = 16, inject_mode: str = "user",
                    fence_style: str = "bracket", header: Optional[str] = None,
                    extra_meta: Optional[Dict[str, Any]] = None
                    ) -> Tuple[str, Dict[str, Any]]:
    """把记忆过一遍 Reader，注入 prompt 并生成。mem=None 即"无记忆"对照。

    复用 `latent_memory._emit_latent`：注入位置、围栏、解码口径与前面所有实验逐位相同，
    否则"Reader 有没有用"会和"prompt 模板变了"混在一起。
    """
    soft = None
    if mem is not None and int(mem.get("n") or 0) > 0 and reader is not None:
        soft = reader(mem["h"], mem["mask"], q_vec)
    meta_extra = dict(extra_meta or {})
    meta_extra.setdefault("has_payload", soft is not None)
    if mem is not None:
        meta_extra.setdefault("mem_tokens", int(mem.get("n") or 0))
    pred, meta = LM._emit_latent(
        model, tokenizer, soft, question, system_prompt,
        max_new_tokens=int(max_new_tokens), inject_mode=inject_mode,
        fence_style=fence_style, header=header, extra_meta=meta_extra)
    return pred, meta


# ---------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------
def reader_config(reader: Optional[QueryCondReader], compressor=None,
                  extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"schema": "reader_v1", "layer": int(READER_LAYER)}
    if reader is not None:
        cfg.update({"reader_hidden": int(reader.hidden_size),
                    "reader_bottleneck": int(reader.bottleneck),
                    "reader_tau": float(reader.tau),
                    "reader_target_norm": float(reader.out_norm.detach().cpu().item()),
                    "mem_mode": str(getattr(reader, "_mem_mode", "full"))})
    if compressor is not None:
        # 构造参数必须**全部**落盘：只存 n_slots 的话，inner_dim/heads/bottleneck
        # 用默认值重建，strict load 立刻报形状不符（这个坑 E1 时期踩过一次）。
        cfg["compressor_hidden"] = int(compressor.hidden_size)
        cfg["compressor_n_slots"] = int(compressor.n_slots)
        cfg["compressor_inner_dim"] = int(compressor.inner_dim)
        cfg["compressor_num_heads"] = int(compressor.num_heads)
        cfg["compressor_lex_skip"] = bool(getattr(compressor, "lex_skip", False))
        cfg["compressor_lex_token_mode"] = str(getattr(compressor, "lex_token_mode", "mean"))
    if extra:
        cfg.update(extra)
    return cfg


def save_reader_checkpoint(path: str, reader: Optional[QueryCondReader],
                           compressor=None, config: Optional[Dict[str, Any]] = None,
                           stats: Optional[Dict[str, Any]] = None) -> str:
    """只存 Reader（+ 实验 B 的压缩器）。主干永远冻结，不入库。"""
    blob: Dict[str, Any] = {"config": dict(config or reader_config(reader, compressor)),
                            "stats": dict(stats or {})}
    if reader is not None:
        blob["reader"] = reader.state_dict()
    if compressor is not None:
        blob["compressor"] = compressor.state_dict()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(blob, path)
    return path


def load_reader_checkpoint(path: str, model=None, device: Any = None):
    """返回 (reader, compressor, config)。结构按 ckpt 里的配置重建，strict 加载。"""
    blob = LM._safe_torch_load(path)
    if not isinstance(blob, dict) or "config" not in blob:
        raise ValueError(f"{path} 不是 reader checkpoint（缺 config）")
    cfg = dict(blob["config"])
    dev = torch.device(device) if device is not None else (
        LM._model_device(model) if model is not None else torch.device("cpu"))
    reader = None
    if "reader" in blob:
        reader = QueryCondReader(int(cfg["reader_hidden"]),
                                 bottleneck=int(cfg["reader_bottleneck"]),
                                 target_norm=float(cfg["reader_target_norm"]),
                                 tau=float(cfg.get("reader_tau", 1.0))).to(dev)
        reader.load_state_dict(blob["reader"], strict=True)
        reader.eval()
        reader._mem_mode = str(cfg.get("mem_mode", "full"))
    comp = None
    if "compressor" in blob:
        comp = LM.LatentCompressor(
            int(cfg["compressor_hidden"]), n_slots=int(cfg["compressor_n_slots"]),
            inner_dim=int(cfg.get("compressor_inner_dim", LM.LATENT_INNER_DIM)),
            num_heads=int(cfg.get("compressor_num_heads", LM.LATENT_N_HEADS)),
            lex_skip=bool(cfg.get("compressor_lex_skip", False)),
            lex_token_mode=str(cfg.get("compressor_lex_token_mode", "mean"))).to(dev)
        comp.load_state_dict(blob["compressor"], strict=True)
        comp.eval()
    return reader, comp, cfg
