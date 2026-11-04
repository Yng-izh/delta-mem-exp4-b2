#!/usr/bin/env python3
"""两份 eval_latent_table.py 的产出逐题配对比较（同一批题、同一条判定）。

为什么必须有这个工具：整轮实验的结论全是"同一批 200 题上 A 变 B"，而两个**独立**
百分比相减会把配对信息丢掉 —— 200 题里有 130 题两边都错，那部分对方差毫无贡献。
这里用 McNemar 精确二项：只看"一边对一边错"的题，再看两个方向的题目数是否对称。

用法：
    python tools/cmp_readout.py A.json B.json
    python tools/cmp_readout.py A.json B.json --label-a 基线 --label-b 实验
    python tools/cmp_readout.py A.json B.json C.json      # 多份：依次与第一份比

判定口径：用行里的 `ok` 字段（= 归一化整串相等），不用 summary 里的 acc ——
summary 是各自平均，配对检验必须逐题对齐。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


def load(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not os.path.isfile(path):
        raise SystemExit(f"找不到 {path}")
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    if not isinstance(blob, dict) or "rows" not in blob:
        raise SystemExit(f"{path} 不是 eval_latent_table 的产出（缺 rows）")
    return dict(blob.get("summary") or {}), list(blob["rows"])


def _key(row: Dict[str, Any]) -> Any:
    """逐题对齐的键：优先 idx（同一份数据的行号）。"""
    return row.get("idx")


def _ok(row: Dict[str, Any]) -> bool:
    v = row.get("ok")
    if v is None:                      # 兼容老产出：用 em>=1 当整串相等
        return float(row.get("em") or 0.0) >= 1.0 - 1e-9
    return bool(v)


def mcnemar(b: int, c: int) -> float:
    """双侧精确二项检验（b/c = 两个方向的"只有一边对"题目数）。"""
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, p))


def compare(sa: Dict[str, Any], ra: List[Dict[str, Any]],
            sb: Dict[str, Any], rb: List[Dict[str, Any]],
            la: str, lb: str) -> int:
    ka = [_key(r) for r in ra]
    kb = [_key(r) for r in rb]
    if ka != kb:
        # idx 不一致时退回"按顺序对齐"，但必须显式说出来：静默对齐会做出假的配对
        if len(ra) != len(rb):
            print(f"[错误] 两份文件题数不同：{len(ra)} vs {len(rb)}，无法配对")
            return 2
        print("[warn] 两份文件的 idx 序列不同，按**顺序**配对（请确认是同一批题）")
    n = min(len(ra), len(rb))
    oa = [_ok(r) for r in ra[:n]]
    ob = [_ok(r) for r in rb[:n]]
    ca, cb = sum(oa), sum(ob)
    only_a = sum(1 for x, y in zip(oa, ob) if x and not y)
    only_b = sum(1 for x, y in zip(oa, ob) if y and not x)
    p = mcnemar(only_a, only_b)

    def _g(k: str) -> Optional[float]:
        v = sa.get(k)
        return None if v is None else float(v)

    print(f"n={n}")
    print(f"{'指标':<12}{la:>14}{lb:>14}{'差值':>12}")
    print(f"{'acc/EM':<12}{ca / n:>14.4f}{cb / n:>14.4f}{(cb - ca) / n * 100:>+11.1f}pt")
    for k, name in (("f1", "f1"), ("mem_tokens", "注入 token"), ("prompt_tokens", "prompt token")):
        va, vb = _g(k), _g(k)
        va = _g(k)
        if va is None:
            continue
        vb = float(sb.get(k) or 0.0)
        print(f"{name:<12}{va:>14.3f}{vb:>14.3f}{vb - va:>+11.3f}")
    print(f"\n只 {la} 对：{only_a}   只 {lb} 对：{only_b}   "
          f"McNemar 精确 p = {p:.4f}")
    if p < 0.05:
        better = lb if only_b > only_a else la
        print(f"→ 差异显著（p<0.05），{better} 更好")
    else:
        print("→ 差异不显著（p>=0.05）：这批数据不足以支持'有变化'")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="eval 产出逐题配对比较")
    ap.add_argument("files", nargs="+", help="至少两份 json：第一份当基线")
    ap.add_argument("--label-a", default="")
    ap.add_argument("--label-b", default="")
    args = ap.parse_args(argv)
    if len(args.files) < 2:
        ap.error("至少给两份 json")
    sa, ra = load(args.files[0])
    la = args.label_a or os.path.basename(args.files[0])
    rc = 0
    for path in args.files[1:]:
        sb, rb = load(path)
        lb = args.label_b or os.path.basename(path)
        print(f"\n=== {la}  vs  {lb} ===")
        rc |= compare(sa, ra, sb, rb, la, lb)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
