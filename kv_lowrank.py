#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""kv_lowrank.py — 跨层真实的 Pre-RoPE K/V 的**线性低秩压缩**（实验 E）。

压缩对象
--------
一段证据的每个 token，在第 `layers`（默认 6~35）层上有自己的 K 和 V：

    U_i = [K_i^(6), V_i^(6), K_i^(7), V_i^(7), …, K_i^(35), V_i^(35)] ∈ R^61440

K 取在 `k_norm` 之后、RoPE 之前（压缩器不该去拟合"已经被某个绝对位置旋转过"的表示，
恢复时按在线逻辑位置重施 RoPE），V 是原生投影输出。30 层 × 2 × (8 头 × 128) = 61440。

压缩怎么做
----------
逐维标准化后做一次**全局共享**的线性投影（所有 token、所有证据共用一套基）：

    z_i = Bᵀ · ((U_i − μ) / σ)          z ∈ R^k,  k = 512 / 1024 / 2048
    Û_i = μ + σ ⊙ (B · z_i)             Û ∈ R^61440

`μ/σ` 是**逐维**（因而也就是逐层、逐 K/V）的固定统计量 —— 不做这一步，数值幅度大的层
会单独主导主成分方向，PCA 就变成"给某几层做压缩"。

为什么用流式拟合而不是直接算协方差
----------------------------------
61440² 是不可能的（3.7e9 元素）。这里用随机化低秩（Halko 等人的 range finder），
**三遍**扫数据：

  1. 逐维均值/标准差（一遍）
  2. Y = Σ X̂ᵀ(X̂Ω)，Ω 是 [D, k+oversample] 随机高斯；`qr(Y)` 得到值域的一组正交基 Q
  3. B_small = Σ (X̂Q)ᵀ(X̂Q)，`eigh` 取特征值最大的 k 个方向，B = Q·V_k

三遍都要重算 U（每遍重新跑一次冻结主干），代价是几分钟；换来的是不需要把 61440 维的
样本矩阵一次性放进显存/内存（2 万 token × 61440 × 2B ≈ 2.4 GB，且协方差不可行）。

> 只拟合一个最高维（2048）的基底，512/1024 直接取前 k 列。主成分子空间是嵌套的，
> 分开拟合三套反而会让三档之间不可比。
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch


class LowRankKV:
    """一套共享的线性编解码器：`z = Bᵀ x̂`，`x̂ = μ + σ ⊙ (B z)`。"""

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor, basis: torch.Tensor,
                 layout: Dict[str, Any], explained: Optional[torch.Tensor] = None,
                 meta: Optional[Dict[str, Any]] = None):
        if mean.dim() != 1 or scale.dim() != 1 or basis.dim() != 2:
            raise ValueError("mean/scale 必须是一维、basis 必须是二维")
        d = int(mean.shape[0])
        if int(scale.shape[0]) != d or int(basis.shape[0]) != d:
            raise ValueError(f"形状不一致：mean={d} scale={int(scale.shape[0])} "
                             f"basis={tuple(basis.shape)}")
        if int(basis.shape[1]) > d:
            raise ValueError(f"维数 {int(basis.shape[1])} 超过原始维度 {d}")
        self.mean = mean
        self.scale = scale
        self.basis = basis
        self.layout = dict(layout)
        self.explained = explained
        self.meta = dict(meta or {})

    @property
    def dim(self) -> int:
        return int(self.basis.shape[1])

    @property
    def native_dim(self) -> int:
        return int(self.basis.shape[0])

    def encode(self, U: torch.Tensor) -> torch.Tensor:
        """U [N, D] -> z [N, k]"""
        x = self._norm(U)
        return x @ self.basis

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z [N, k] -> Û [N, D]"""
        if z.dim() != 2 or int(z.shape[1]) != self.dim:
            raise ValueError(f"z 必须是 [N,{self.dim}]，得到 {tuple(z.shape)}")
        return self.mean.to(z.device) + self.scale.to(z.device) * (z @ self.basis.to(z.device).T)

    def roundtrip(self, U: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(U))

    def rel_error(self, U: torch.Tensor) -> float:
        """整体相对误差 ‖Û−U‖/‖U‖（一个数，用于和 QA 一起记账）。"""
        d = self.decode(self.encode(U))
        return float((d.float() - U.float()).norm() / U.float().norm().clamp_min(1e-6))

    def _norm(self, U: torch.Tensor) -> torch.Tensor:
        if U.dim() != 2 or int(U.shape[1]) != self.native_dim:
            raise ValueError(f"U 必须是 [N,{self.native_dim}]，得到 {tuple(U.shape)}")
        return (U.to(self.mean.device, torch.float32) - self.mean) / self.scale

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save({"schema": "kv_lowrank_v1", "mean": self.mean.cpu(),
                    "scale": self.scale.cpu(), "basis": self.basis.cpu(),
                    "explained": None if self.explained is None else self.explained.cpu(),
                    "layout": self.layout, "meta": self.meta}, path)
        return path

    @staticmethod
    def load(path: str, device: Any = "cpu", dim: Optional[int] = None) -> "LowRankKV":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or blob.get("schema") != "kv_lowrank_v1":
            raise ValueError(f"{path} 不是 kv_lowrank checkpoint")
        basis = blob["basis"]
        if dim is not None:
            k = int(dim)
            if k > int(basis.shape[1]):
                raise ValueError(f"请求 {k} 维，但基底只有 {int(basis.shape[1])} 维")
            lr = LowRankKV(blob["mean"], blob["scale"], basis[:, :k], blob["layout"],
                           blob.get("explained"), blob.get("meta"))
            return lr.to(device)
        return LowRankKV(blob["mean"], blob["scale"], basis, blob["layout"],
                         blob.get("explained"), blob.get("meta")).to(device)

    def to(self, device: Any) -> "LowRankKV":
        dev = torch.device(device)
        self.mean = self.mean.to(dev)
        self.scale = self.scale.to(dev)
        self.basis = self.basis.to(dev)
        return self

    def describe(self) -> Dict[str, Any]:
        return {"dim": self.dim, "native_dim": self.native_dim,
                "bytes_per_token": self.dim * 2,
                "native_bytes_per_token": self.native_dim * 2,
                "ratio": round(self.native_dim / max(1, self.dim), 2),
                "layers": self.layout.get("layers", [None])[0:1] +
                          self.layout.get("layers", [0, 0])[-1:]}


def fit_lowrank(batches: Callable[[], Iterable[torch.Tensor]], layout: Dict[str, Any],
                dim: int, oversample: int = 64, device: Any = "cuda",
                seed: int = 0, log: Optional[Callable[[str], None]] = None) -> LowRankKV:
    """流式拟合线性低秩基。`batches()` 每次调用返回**重新生成**的 U 批次迭代器。

    必须能在同一份数据上被调用三次（均值/标准差、随机投影、小矩阵特征分解）。
    """
    d = int(layout["dim"])
    k = int(dim)
    if k <= 0 or k > d:
        raise ValueError(f"dim={k} 非法（原始维度 {d}）")
    dev = torch.device(device)
    say = log or (lambda m: None)
    r = min(d, k + int(oversample))

    # ---- 第 1 遍：逐维均值与标准差 ----
    n_tok = 0
    s1 = torch.zeros(d, dtype=torch.float64)
    s2 = torch.zeros(d, dtype=torch.float64)
    for U in batches():
        x = U.float().cpu().double()
        n_tok += int(x.shape[0])
        s1 += x.sum(dim=0)
        s2 += (x * x).sum(dim=0)
    if n_tok <= 0:
        raise ValueError("拟合数据为空")
    mean = (s1 / n_tok)
    var = (s2 / n_tok) - mean * mean
    scale = var.clamp_min(0).sqrt().clamp_min(1e-6)
    say(f"[fit] tokens={n_tok} native_dim={d} target_dim={k} r={r}")

    # ---- 第 2 遍：随机投影累加 Y = Σ X̂ᵀ(X̂Ω) ----
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    omega = torch.randn((d, r), generator=g, dtype=torch.float32).to(dev)
    mean_d = mean.float().to(dev)
    scale_d = scale.float().to(dev)
    Y = torch.zeros((d, r), dtype=torch.float32, device=dev)
    for U in batches():
        xn = (U.to(dev, torch.float32) - mean_d) / scale_d
        Y += xn.T @ (xn @ omega)
    say("[fit] 随机投影累加完成，开始 QR")
    Q, _ = torch.linalg.qr(Y)
    del Y, omega
    torch.cuda.empty_cache() if dev.type == "cuda" else None

    # ---- 第 3 遍：小矩阵 B = Σ (X̂Q)ᵀ(X̂Q)，取前 k 个方向 ----
    B = torch.zeros((r, r), dtype=torch.float32, device=dev)
    for U in batches():
        xn = (U.to(dev, torch.float32) - mean_d) / scale_d
        xq = xn @ Q
        B += xq.T @ xq
    evals, evecs = torch.linalg.eigh(B)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    basis = Q @ evecs[:, :k]
    total = float(evals.clamp_min(0).sum())
    explained = (evals[:k].clamp_min(0) / max(total, 1e-12))
    say(f"[fit] 前 {k} 个主成分累计解释方差 {float(explained.sum()) * 100:.2f}%")
    return LowRankKV(mean.to(torch.float32), scale.to(torch.float32), basis,
                     layout=layout,
                     explained=explained,
                     meta={"n_tokens": int(n_tok), "oversample": int(oversample),
                           "seed": int(seed), "explained_ratio": float(explained.sum())})


# ---------------------------------------------------------------------
# 分组基底（实验 F 的初始化：分层段 × K/V 各一套）
# ---------------------------------------------------------------------
def group_slices(layout: Dict[str, Any], n_segments: int = 3) -> Dict[str, Dict[str, Any]]:
    """按"层段 × K/V"把跨层 K/V 长向量切成若干组，给出每组在 U 里的列区间。

    为什么分层段：第 6 层与第 34 层学到的东西不是一类；一套全局基会让它们互相抢主成分。
    为什么 K/V 分开：Key 决定"谁被查中"、Value 决定"读出什么"，两者的低秩结构不同，
    共用一个子空间等于强迫它们对齐。
    """
    ls = list(layout["layers"])
    n = int(n_segments)
    if n <= 0 or n > len(ls):
        raise ValueError(f"n_segments={n} 非法（层数 {len(ls)}）")
    chunks = [ls[i::n] for i in range(n)]              # 交错切分：每段都能覆盖深浅
    chunks = [sorted(c) for c in chunks if c]
    out: Dict[str, Dict[str, Any]] = {}
    for si, chunk in enumerate(chunks):
        for kind in ("k", "v"):
            cols: List[int] = []
            for l in chunk:
                a, b = layout["per_layer"][l][kind]
                cols.extend(range(a, b))
            if not cols:
                continue
            out[f"seg{si}.{kind}"] = {"cols": cols, "layers": list(chunk),
                                     "dim": len(cols)}
    if not out:
        raise ValueError("分组为空")
    return out


def split_latent_budget(groups: Dict[str, Dict[str, Any]], latent_dim: int,
                        kind_weights: Optional[Dict[str, float]] = None) -> Dict[str, int]:
    """把总潜码预算按组摊开；不足以每组一维时直接报错（不静默改动预算）。

    `kind_weights` 给 `{"k": w_k, "v": w_v}` 时按 K/V 分成两块再在块内均分。
    为什么需要：实测（`tools/diag_kv_by_type.py`）Value 的相对重建误差是 Key 的 1.6 倍，
    吃掉 67% 的总误差，而两者的预算是对半分的 —— 平均摊在这里是错的方向。
    """
    names = list(groups)
    total = int(latent_dim)
    if total < len(names):
        raise ValueError(f"潜码预算 {total} 维不够 {len(names)} 组各分一维")
    if not kind_weights:
        base = total // len(names)
        rem = total - base * len(names)
        out = {}
        for i, n in enumerate(names):
            d = base + (1 if i < rem else 0)
            out[n] = min(d, int(groups[n]["dim"]))
        return out

    def kind_of(n: str) -> str:
        return str(n).rsplit(".", 1)[-1]

    wk = float(kind_weights.get("k", 0.5))
    wv = float(kind_weights.get("v", 0.5))
    tot_w = wk + wv
    if tot_w <= 0:
        raise ValueError("kind_weights 之和必须为正")
    pools = {"k": [n for n in names if kind_of(n) == "k"],
             "v": [n for n in names if kind_of(n) == "v"]}
    other = [n for n in names if kind_of(n) not in ("k", "v")]
    budget = {"k": int(round(total * wk / tot_w)), "v": 0}
    budget["v"] = total - budget["k"]
    out: Dict[str, int] = {}
    for kind, pool in pools.items():
        if not pool:
            budget["k" if kind == "v" else "v"] += budget[kind]
            continue
        b = budget[kind] // len(pool)
        rem = budget[kind] - b * len(pool)
        for i, n in enumerate(pool):
            out[n] = min(b + (1 if i < rem else 0), int(groups[n]["dim"]))
    if other:
        b = total // max(1, len(names))
        for n in other:
            out[n] = min(b, int(groups[n]["dim"]))
    return out


class GroupedLowRankKV:
    """若干组 `LowRankKV` 的组合体：每组分自己的一段列、各自的标准化和基底。

    编解码都在**完整的 U 空间**上工作（输入 [N, D]、输出 [N, D]），内部按组切片，
    所以对上层调用方与 `LowRankKV` 完全同构。
    """

    def __init__(self, parts: Dict[str, LowRankKV], layout: Dict[str, Any],
                 n_segments: int, groups: Dict[str, Dict[str, Any]],
                 meta: Optional[Dict[str, Any]] = None):
        self.parts = dict(parts)
        self.layout = dict(layout)
        self.n_segments = int(n_segments)
        self.groups = dict(groups)
        self.meta = dict(meta or {})

    @property
    def native_dim(self) -> int:
        return int(self.layout["dim"])

    @property
    def latent_dim(self) -> int:
        return int(sum(p.dim for p in self.parts.values()))

    def encode(self, U: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.parts[n].encode(U[:, g["cols"]])
                          for n, g in self.groups.items()], dim=-1)

    def decode(self, Z: torch.Tensor) -> torch.Tensor:
        if int(Z.shape[-1]) != self.latent_dim:
            raise ValueError(f"潜码宽度 {int(Z.shape[-1])} != {self.latent_dim}")
        out = None
        off = 0
        for n, g in self.groups.items():
            p = self.parts[n]
            seg = Z[:, off:off + p.dim]
            off += p.dim
            piece = p.decode(seg)
            if out is None:
                out = U_like = torch.zeros((int(Z.shape[0]), self.native_dim),
                                           dtype=piece.dtype, device=piece.device)
            out[:, g["cols"]] = piece
        if out is None:
            raise ValueError("解码为空")
        return out

    def roundtrip(self, U: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(U))

    def rel_error(self, U: torch.Tensor) -> float:
        d = self.decode(self.encode(U))
        return float((d.float() - U.float()).norm() / U.float().norm().clamp_min(1e-6))

    def group_errors(self, U: torch.Tensor) -> Dict[str, float]:
        d = self.decode(self.encode(U))
        out = {}
        for n, g in self.groups.items():
            a, b = d[:, g["cols"]].float(), U[:, g["cols"]].float()
            out[n] = float((a - b).norm() / b.norm().clamp_min(1e-6))
        return out

    def to(self, device: Any) -> "GroupedLowRankKV":
        for p in self.parts.values():
            p.to(device)
        return self

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save({"schema": "kv_lowrank_grouped_v1",
                    "parts": {n: {"mean": p.mean.cpu(), "scale": p.scale.cpu(),
                                  "basis": p.basis.cpu(),
                                  "explained": None if p.explained is None else p.explained.cpu()}
                              for n, p in self.parts.items()},
                    "layout": self.layout, "n_segments": self.n_segments,
                    "groups": self.groups, "meta": self.meta}, path)
        return path

    @staticmethod
    def load(path: str, device: Any = "cpu") -> "GroupedLowRankKV":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or blob.get("schema") != "kv_lowrank_grouped_v1":
            raise ValueError(f"{path} 不是分组基底 ckpt")
        parts = {n: LowRankKV(b["mean"], b["scale"], b["basis"], {},
                              b.get("explained"), {})
                 for n, b in blob["parts"].items()}
        return GroupedLowRankKV(parts, blob["layout"], blob["n_segments"],
                                blob["groups"], blob.get("meta")).to(device)

    def describe(self) -> Dict[str, Any]:
        return {"n_groups": len(self.parts), "native_dim": self.native_dim,
                "latent_dim": self.latent_dim,
                "per_group": {n: int(p.dim) for n, p in self.parts.items()},
                "bytes_per_token": self.latent_dim * 2,
                "ratio": round(self.native_dim / max(1, self.latent_dim), 2)}


def fit_grouped(batches: Callable[[], Iterable[torch.Tensor]], layout: Dict[str, Any],
                latent_dim: int, n_segments: int = 3, oversample: int = 64,
                device: Any = "cuda", seed: int = 0,
                kind_weights: Optional[Dict[str, float]] = None,
                log: Optional[Callable[[str], None]] = None) -> GroupedLowRankKV:
    """每组独立拟合一套低秩基（每组自己的标准化与主成分），总潜码 ≈ `latent_dim`。

    组内维度按 `split_latent_budget` 摊分后仍可能被该组原始维度截断（不会超），
    最终 `latent_dim` 以实际值为准，并写进 meta。
    """
    say = log or (lambda m: None)
    groups = group_slices(layout, n_segments)
    budget = split_latent_budget(groups, latent_dim, kind_weights=kind_weights)
    parts: Dict[str, LowRankKV] = {}
    for i, (name, g) in enumerate(groups.items()):
        cols = g["cols"]

        def sub_batches(_cols=cols):
            for U in batches():
                yield U[:, _cols]

        sub_layout = {"dim": len(cols), "layers": g["layers"],
                      "nkv": layout["nkv"], "hd": layout["hd"]}
        # 过采样下限：保留秩 k 只占随机子空间 r 的 1/3 以内。每组原生维度比全局小一个
        # 数量级，套用同一个 +64 会让 r≈1.37k（用了 73% 的子空间），挑出来的"主成分"
        # 只是对真子空间的粗糙近似——实测分组零训练 5.5%，比全局 23.5% 还差。
        over = max(int(oversample), 2 * int(budget[name]))
        say(f"[fit-grouped] {name}: native={len(cols)} -> latent={budget[name]} "
            f"r={min(len(cols), int(budget[name]) + over)} "
            f"(layers {g['layers'][0]}..{g['layers'][-1]}, n={len(g['layers'])})")
        parts[name] = fit_lowrank(sub_batches, sub_layout, dim=budget[name],
                                  oversample=over, device=device,
                                  seed=int(seed) + i, log=say)
    return GroupedLowRankKV(parts, layout, n_segments, groups,
                            {"latent_dim_requested": int(latent_dim),
                             "oversample": int(oversample)})


def describe() -> str:
    return "[kv_lowrank] 流式随机化低秩（3 遍扫描）：逐维标准化 + 全局/分组共享线性基"


if __name__ == "__main__":
    print(describe())
