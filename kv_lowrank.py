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


def describe() -> str:
    return "[kv_lowrank] 流式随机化低秩（3 遍扫描）：逐维标准化 + 全局共享线性基"


if __name__ == "__main__":
    print(describe())
