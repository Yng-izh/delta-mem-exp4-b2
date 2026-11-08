#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fadkv_codec.py — FAD-KV：分层 K/V 解耦 + 功能蒸馏的跨层 KV 压缩器（实验 F）。

阶段一：**通道压缩，序列长度不变**
--------------------------------
把 L6~35 的 K/V 按"层段 × K/V"切成 3 组（每段交错覆盖深浅层），每组一个独立的线性
编解码器：

    z_g = ((U[:, cols_g] − μ_g) / σ_g) · W_enc_g          z_g ∈ R^{d_g}
    Û[:, cols_g] = μ_g + σ_g ⊙ (z_g · W_dec_g)

总潜码 Σ d_g ≈ 1024 维 → 2 KiB/token（bf16），即 60 倍。
`W_enc_g` 初始化为该组的分组 PCA 基底、`W_dec_g` 为其转置，所以**第 0 步就等于 PCA**
（`GroupedLinearCodec.from_grouped`），不存在"从随机权重开始训练"。

阶段二：**双粒度序列压缩**
------------------------
    Na 个事实锚点：逐个保留，每个 1536 维
    Nb 个背景 token：连续 4 个池化成一个槽，每槽 384 维

平均存储 = 0.3×1536 + 0.7×(384/4) = 528 维 ≈ 1.03 KiB/token，序列保留率
= 0.3 + 0.7/4 = 47.5%。锚点选择**不看问题**（默认按 KV 范数排序，也可给内容词标记），
所以写入仍然可以提前做。

为什么 K/V 要分开、要分层段
--------------------------
Key 决定"谁被查中"、Value 决定"读出什么"，两者的低秩结构不同；第 6 层和第 34 层学到的
东西也不是一类。共用一套基底等于强迫它们挤进同一个子空间 —— 实验 E 的全局 PCA-1024 只有
23.0%，而 2048 维能到 31.0%，正是"预算被摊薄"的表现。

`memory_kv_from_uv` 的位置约定：池化槽的位置取它覆盖的**第一个**原始位置；序列变短不影响
插入掩码（掩码只认 S | 记忆 | T 三段，不假设记忆长度等于 E 段长度）。
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import kv_lowrank as LR


# ---------------------------------------------------------------------
# 阶段一：分组线性编解码器（可训练，PCA 初始化）
# ---------------------------------------------------------------------
class GroupedLinearCodec(nn.Module):
    """按组切片的线性编解码。参数就是每组一个 `W_enc (D_g×d_g)` 与 `W_dec (d_g×D_g)`。"""

    @staticmethod
    def _key(name: str) -> str:
        """组名 -> 合法模块属性名。`nn.Module` 不允许 buffer/参数名含 "."，
        而分组名是 `seg0.k` 这种，直接用会在 register_buffer 处 KeyError。"""
        return str(name).replace(".", "__")

    def __init__(self, layout: Dict[str, Any], groups: Dict[str, Dict[str, Any]],
                 latent_dims: Dict[str, int], n_segments: int = 3):
        super().__init__()
        self.layout = dict(layout)
        self.n_segments = int(n_segments)
        self.groups = {n: {"cols": list(g["cols"]), "layers": list(g["layers"])}
                       for n, g in groups.items()}
        self.native_dim = int(layout["dim"])
        self.latent_dims = {n: int(latent_dims[n]) for n in self.groups}
        for n, g in self.groups.items():
            d_nat, d_lat = len(g["cols"]), self.latent_dims[n]
            self.register_buffer(f"mean__{self._key(n)}", torch.zeros(d_nat))
            self.register_buffer(f"scale__{self._key(n)}", torch.ones(d_nat))
            self.register_parameter(f"enc__{self._key(n)}", nn.Parameter(torch.zeros(d_nat, d_lat)))
            self.register_parameter(f"dec__{self._key(n)}", nn.Parameter(torch.zeros(d_lat, d_nat)))
        self.group_order = list(self.groups)

    # ---- 构造：从分组 PCA 直接搬过来，第 0 步 == PCA ----
    @staticmethod
    def from_grouped(gr: LR.GroupedLowRankKV) -> "GroupedLinearCodec":
        codec = GroupedLinearCodec(gr.layout, gr.groups,
                                   {n: p.dim for n, p in gr.parts.items()},
                                   n_segments=gr.n_segments)
        with torch.no_grad():
            for n, p in gr.parts.items():
                getattr(codec, f"mean__{codec._key(n)}").copy_(p.mean.float())
                getattr(codec, f"scale__{codec._key(n)}").copy_(p.scale.float())
                # W_enc = 基底（D×d），W_dec = 基底转置（d×D）
                getattr(codec, f"enc__{codec._key(n)}").copy_(p.basis.float())
                getattr(codec, f"dec__{codec._key(n)}").copy_(p.basis.float().T)
        return codec

    @property
    def latent_dim(self) -> int:
        return int(sum(self.latent_dims.values()))

    def _slice(self, U: torch.Tensor, n: str) -> torch.Tensor:
        return U[:, torch.as_tensor(self.groups[n]["cols"], device=U.device)]

    def encode(self, U: torch.Tensor) -> torch.Tensor:
        if U.dim() != 2 or int(U.shape[1]) != self.native_dim:
            raise ValueError(f"U 必须是 [N,{self.native_dim}]，得到 {tuple(U.shape)}")
        out = []
        for n in self.group_order:
            x = (self._slice(U, n).float() - getattr(self, f"mean__{self._key(n)}")) \
                / getattr(self, f"scale__{self._key(n)}")
            out.append(x @ getattr(self, f"enc__{self._key(n)}"))
        return torch.cat(out, dim=-1)

    def decode(self, Z: torch.Tensor) -> torch.Tensor:
        if Z.dim() != 2 or int(Z.shape[1]) != self.latent_dim:
            raise ValueError(f"Z 必须是 [N,{self.latent_dim}]，得到 {tuple(Z.shape)}")
        out = torch.zeros((int(Z.shape[0]), self.native_dim), dtype=torch.float32,
                          device=Z.device)
        off = 0
        for n in self.group_order:
            d = self.latent_dims[n]
            x = Z[:, off:off + d] @ getattr(self, f"dec__{self._key(n)}")
            x = x * getattr(self, f"scale__{self._key(n)}") + getattr(self, f"mean__{self._key(n)}")
            out[:, torch.as_tensor(self.groups[n]["cols"], device=Z.device)] = x
            off += d
        return out

    def roundtrip(self, U: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(U))

    def kv_loss(self, U: torch.Tensor) -> torch.Tensor:
        """逐组归一化 MSE（各组量纲不同，直接求全空间 MSE 会被大组主导）。"""
        total = U.new_zeros(())
        for n in self.group_order:
            tgt = self._slice(U, n).float()
            pred = self.decode(self.encode(U))
            pred = pred[:, torch.as_tensor(self.groups[n]["cols"], device=U.device)]
            total = total + ((pred - tgt) ** 2).mean() / tgt.var().clamp_min(1e-6)
        return total / max(1, len(self.group_order))

    def rel_error(self, U: torch.Tensor) -> float:
        d = self.roundtrip(U)
        return float((d - U.float()).norm() / U.float().norm().clamp_min(1e-6))

    def describe(self) -> Dict[str, Any]:
        return {"n_groups": len(self.groups), "native_dim": self.native_dim,
                "latent_dim": self.latent_dim,
                "per_group": dict(self.latent_dims),
                "bytes_per_token": self.latent_dim * 2,
                "ratio": round(self.native_dim / max(1, self.latent_dim), 2)}


# ---------------------------------------------------------------------
# 阶段二：双粒度序列压缩
# ---------------------------------------------------------------------
def select_anchors(U: torch.Tensor, ratio: float,
                   token_flags: Optional[Sequence[bool]] = None) -> List[int]:
    """挑出"事实锚点"：默认按 token 的 KV 范数排序，Query-Agnostic。

    为什么不用问题：写入必须能在问题出现之前完成。给了 `token_flags`（数字/实体/关系词）
    时先把这些位置保下来，再按范数补齐 —— 那也只是证据自身的属性，与未来问题无关。
    """
    n = int(U.shape[0])
    k = max(1, min(n, int(round(float(ratio) * n))))
    score = U.float().norm(dim=-1)
    pri = torch.zeros(n, dtype=torch.bool)
    if token_flags is not None:
        f = torch.as_tensor([bool(x) for x in token_flags][:n])
        if int(f.numel()) < n:
            f = torch.cat([f, torch.zeros(n - int(f.numel()), dtype=torch.bool)])
        pri = f
    # 先按"是否内容词"、再按范数排序，取前 k
    order = sorted(range(n), key=lambda i: (bool(pri[i]), float(score[i])), reverse=True)
    return sorted(order[:k])


def pool_groups(idx: Sequence[int], pool: int) -> List[List[int]]:
    """把背景下标按连续 `pool` 个一槽切开（末尾不足一槽的单独成槽，不丢）。"""
    ids = sorted(int(i) for i in idx)
    p = max(1, int(pool))
    out: List[List[int]] = []
    for s in range(0, len(ids), p):
        out.append(ids[s:s + p])
    return out


class DualGranularityCodec(nn.Module):
    """锚点逐个保留（高维）+ 背景池化（低维）。两套 codec 各自分组、各自 PCA 初始化。"""

    def __init__(self, anchor_codec: GroupedLinearCodec, bg_codec: GroupedLinearCodec,
                 anchor_ratio: float = 0.3, pool: int = 4):
        super().__init__()
        if int(anchor_codec.native_dim) != int(bg_codec.native_dim):
            raise ValueError("两套 codec 的 native_dim 必须一致")
        self.anchor_codec = anchor_codec
        self.bg_codec = bg_codec
        self.anchor_ratio = float(anchor_ratio)
        self.pool = int(pool)

    def compress(self, U: torch.Tensor,
                 token_flags: Optional[Sequence[bool]] = None) -> Dict[str, Any]:
        """U [N, D] -> {锚点潜码 [Na, da], 背景槽潜码 [Nb, db], 下标, 位置}"""
        n = int(U.shape[0])
        idx_a = select_anchors(U, self.anchor_ratio, token_flags)
        mask = torch.zeros(n, dtype=torch.bool)
        mask[torch.as_tensor(idx_a)] = True
        idx_b = [i for i in range(n) if not bool(mask[i])]
        chunks = pool_groups(idx_b, self.pool)
        # 背景槽的原生表示 = 被合并那几个 token 的均值（池化发生在 U 空间）
        U_b = torch.stack([U[torch.as_tensor(c)].float().mean(dim=0) for c in chunks]) \
            if chunks else U.new_zeros((0, int(U.shape[1])))
        z_a = self.anchor_codec.encode(U[torch.as_tensor(idx_a)]) if idx_a else \
            U.new_zeros((0, self.anchor_codec.latent_dim))
        z_b = self.bg_codec.encode(U_b) if chunks else U.new_zeros((0, self.bg_codec.latent_dim))
        pos_a = list(idx_a)
        pos_b = [int(c[0]) for c in chunks]           # 槽位置 = 覆盖的第一个原始位置
        return {"z_anchor": z_a, "z_bg": z_b, "idx_anchor": idx_a, "chunks_bg": chunks,
                "pos": pos_a + pos_b, "n": n,
                "n_anchor": len(idx_a), "n_slot": len(chunks)}

    def decompress(self, packed: Dict[str, Any]) -> Tuple[torch.Tensor, List[int]]:
        """潜码 -> 压缩后的 KV 序列 [Na+Nb, D] 与它在原逻辑位置上的坐标。"""
        parts: List[torch.Tensor] = []
        if int(packed["n_anchor"]) > 0:
            parts.append(self.anchor_codec.decode(packed["z_anchor"]))
        if int(packed["n_slot"]) > 0:
            parts.append(self.bg_codec.decode(packed["z_bg"]))
        if not parts:
            raise ValueError("压缩结果为空")
        return torch.cat(parts, dim=0), list(packed["pos"])

    def roundtrip(self, U: torch.Tensor, token_flags: Optional[Sequence[bool]] = None):
        return self.decompress(self.compress(U, token_flags))

    def storage(self) -> Dict[str, Any]:
        da, db = self.anchor_codec.latent_dim, self.bg_codec.latent_dim
        a = self.anchor_ratio
        per_token = a * da + (1.0 - a) * (db / max(1, self.pool))
        seq_keep = a + (1.0 - a) / max(1, self.pool)
        return {"anchor_dim": da, "bg_dim": db, "anchor_ratio": a, "pool": self.pool,
                "latent_dims_per_token": round(float(per_token), 1),
                "bytes_per_token": int(round(per_token * 2)),
                "kib_per_token": round(per_token * 2 / 1024, 3),
                "seq_keep": round(float(seq_keep), 4)}


# ---------------------------------------------------------------------
# 存取
# ---------------------------------------------------------------------
def save_codec(path: str, codec: nn.Module, config: Optional[Dict[str, Any]] = None,
               stats: Optional[Dict[str, Any]] = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    blob: Dict[str, Any] = {"config": dict(config or {}), "stats": dict(stats or {})}
    if isinstance(codec, DualGranularityCodec):
        blob["kind"] = "fadkv_stage2"
        blob["anchor"] = codec.anchor_codec.state_dict()
        blob["bg"] = codec.bg_codec.state_dict()
        blob["hparams"] = {"anchor_ratio": codec.anchor_ratio, "pool": codec.pool}
        blob["meta_anchor"] = _codec_meta(codec.anchor_codec)
        blob["meta_bg"] = _codec_meta(codec.bg_codec)
    else:
        blob["kind"] = "fadkv_stage1"
        blob["codec"] = codec.state_dict()
        blob["meta"] = _codec_meta(codec)
    torch.save(blob, path)
    return path


def _codec_meta(codec: GroupedLinearCodec) -> Dict[str, Any]:
    return {"layout": codec.layout, "groups": codec.groups,
            "latent_dims": codec.latent_dims, "n_segments": codec.n_segments}


def _codec_from_meta(meta: Dict[str, Any]) -> GroupedLinearCodec:
    return GroupedLinearCodec(meta["layout"], meta["groups"], meta["latent_dims"],
                              n_segments=int(meta["n_segments"]))


def load_codec(path: str, device: Any = "cpu"):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "kind" not in blob:
        raise ValueError(f"{path} 不是 fadkv codec ckpt")
    if blob["kind"] == "fadkv_stage2":
        codec = DualGranularityCodec(_codec_from_meta(blob["meta_anchor"]),
                                     _codec_from_meta(blob["meta_bg"]),
                                     anchor_ratio=blob["hparams"]["anchor_ratio"],
                                     pool=blob["hparams"]["pool"])
        codec.anchor_codec.load_state_dict(blob["anchor"], strict=True)
        codec.bg_codec.load_state_dict(blob["bg"], strict=True)
    else:
        codec = _codec_from_meta(blob["meta"])
        codec.load_state_dict(blob["codec"], strict=True)
    return codec.to(torch.device(device)), dict(blob.get("config") or {})


def describe() -> str:
    return "[fadkv] 阶段一=分组线性编解码(PCA 初始化, 2 KiB/token)；阶段二=锚点+池化(约 1.03 KiB)"


if __name__ == "__main__":
    print(describe())
