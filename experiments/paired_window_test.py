#!/usr/bin/env python3
"""paired_window_test.py — EventQA 窗口扫描的**配对**统计检验。

背景：`eventqa_window_sweep.md` 早期把四组窗口的 SF 做**非配对**对比（各组 n 不等），
结论不可靠。本脚本按 (context_id, question_idx) 取交集做严格配对比较：

  1. 逐对 McNemar **精确**检验（只用不一致对 b/c，二项分布双尾）
  2. 配对差值 ΔSF 的 95% 置信区间（Wald，方差 (b+c)/n² − Δ²/n）
  3. 当前 n 下的最小可检测效应量 MDE（α=0.05 双尾、power=0.80，
     MDE ≈ (z_{0.975}+z_{0.80})·sqrt(π̂_d/n)，π̂_d 为观测不一致率）

输入：服务器 `/share/exp4group/results/sweep/*.jsonl`（逐题记录，未入库）。
      先拷到本地目录，例如
        scp v100-2:/share/exp4group/results/sweep/{ar_v9_eventqa_65536_b*,ctl_*}.jsonl /tmp/sweep/
输出：`experiments/results/paired_window_test.json`

用法：
  python3 experiments/paired_window_test.py --raw /tmp/sweep \
      --out experiments/results/paired_window_test.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from itertools import combinations

WINDOWS = ["b0_a20", "b4_a16", "b10_a10", "b16_a4"]
CONTROLS = ["ctl_b0_a0", "ctl_b200_a200"]
FILES = {w: f"ar_v9_eventqa_65536_{w}.jsonl" for w in WINDOWS}
FILES.update({c: f"{c}.jsonl" for c in CONTROLS})

Z_ALPHA = 1.959963984540054  # 双侧 α=0.05
Z_POWER = 0.8416212335729143  # power=0.80


def load(path):
    """读逐题 JSONL，返回 {(ctx,qid): correct} 与去重信息。"""
    rows, dup = {}, 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (int(r["context_id"]), int(r["question_idx"]))
            if key in rows:
                dup += 1
            rows[key] = 1 if float(r.get("eventqa_recall") or 0) >= 0.5 else 0
    return rows, dup


def sf(rows, keys):
    keys = list(keys)
    return sum(rows[k] for k in keys) / len(keys) if keys else float("nan")


def wilson(k, n, z=Z_ALPHA):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def newcombe_paired(n11, n00, b01, b10, n):
    """Newcombe (1998) method 10：配对比例差的双侧 95% CI（小样本更稳）。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p1, p2 = (n11 + b01) / n, (n11 + b10) / n
    l1, u1 = wilson(n11 + b01, n)
    l2, u2 = wilson(n11 + b10, n)
    a, b, c, d = n11 + b01, b10 + n00, n11 + b10, b01 + n00
    den = math.sqrt(a * b * c * d)
    phi = (n11 * n00 - b01 * b10) / den if den else 0.0
    delta = p1 - p2
    lo = delta - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = delta + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return (lo, hi)


def binom_two_sided(k, n, p=0.5):
    """精确二项双尾 p 值（P(X<=k) 与 P(X>=k) 之和，截断 1）。"""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))
    upper = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))
    return min(1.0, 2 * min(tail, upper))


def mcnemar(a, b, keys):
    """返回 b00/b01/b10/b11 与 McNemar 精确 p（a=行, b=列）。"""
    b01 = sum(1 for k in keys if a[k] == 1 and b[k] == 0)  # a 对 b 错
    b10 = sum(1 for k in keys if a[k] == 0 and b[k] == 1)  # a 错 b 对
    n11 = sum(1 for k in keys if a[k] == 1 and b[k] == 1)
    n00 = sum(1 for k in keys if a[k] == 0 and b[k] == 0)
    n = len(keys)
    nd = b01 + b10
    p = binom_two_sided(min(b01, b10), nd) if nd else 1.0
    delta = (b01 - b10) / n if n else float("nan")
    var = (nd / n**2 - delta**2 / n) if n else float("nan")
    se = math.sqrt(var) if var and var > 0 else 0.0
    lo, hi = delta - Z_ALPHA * se, delta + Z_ALPHA * se
    pi_d = nd / n if n else float("nan")
    mde = (Z_ALPHA + Z_POWER) * math.sqrt(pi_d / n) if n and pi_d > 0 else float("nan")
    nlo, nhi = newcombe_paired(n11, n00, b01, b10, n)
    return {
        "n": n,
        "a_correct_b_wrong": b01,
        "a_wrong_b_correct": b10,
        "both_correct": n11,
        "both_wrong": n00,
        "discordant_rate": round(pi_d, 4),
        "delta_sf": round(delta, 4),
        "delta_pp": round(100 * delta, 2),
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
        "ci95_low_pp": round(100 * lo, 2),
        "ci95_high_pp": round(100 * hi, 2),
        "ci95_newcombe_low_pp": round(100 * nlo, 2),
        "ci95_newcombe_high_pp": round(100 * nhi, 2),
        "ci95_newcombe_excludes_0": bool(nlo > 0 or nhi < 0),
        "mcnemar_exact_p": round(p, 6),
        "mde_pp_80power": round(100 * mde, 2) if mde == mde else None,
        "observed_effect_below_mde": bool(mde == mde and abs(delta) < mde),
        "ci_method": "primary: Newcombe(1998) method 10 paired CI; secondary: Wald var=(b+c)/n^2-delta^2/n",
        "p_method": "McNemar exact two-sided binomial on discordant pairs",
    }


def holm(pvals):
    """Holm-Bonferroni 调整（同一数据集内 6 对比较）。"""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        prev = max(prev, adj)
        out[k] = round(prev, 6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/tmp/sweep")
    ap.add_argument("--out", default="experiments/results/paired_window_test.json")
    args = ap.parse_args()

    data, meta = {}, {}
    for name, fn in FILES.items():
        path = os.path.join(args.raw, fn)
        rows, dup = load(path)
        data[name] = rows
        ctx = {}
        for c, _ in rows:
            ctx[c] = ctx.get(c, 0) + 1
        meta[name] = {
            "file": fn,
            "rows": len(rows),
            "duplicate_keys_dropped": dup,
            "rows_per_context": {str(k): ctx[k] for k in sorted(ctx)},
            "sf_all_rows": round(sf(rows, rows), 4),
            "sf_ctx0": round(sf(rows, [k for k in rows if k[0] == 0]), 4),
            "mean_block_chunks": None,
        }

    # 平均块句数（保留原始聚合口径，用于与旧 JSON 对照）
    for name, fn in FILES.items():
        vals = []
        with open(os.path.join(args.raw, fn)) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("block_chunks") is not None:
                    vals.append(r["block_chunks"])
        meta[name]["mean_block_chunks"] = round(sum(vals) / len(vals), 2) if vals else None

    # ---- 交集 ----
    inter_all = set.intersection(*[set(data[w]) for w in WINDOWS])
    inter_ctx0 = {k for k in inter_all if k[0] == 0}
    ctx1 = {k for k in inter_all if k[0] == 1}

    def summarise(keys, label):
        keys = sorted(keys)
        pairs = {f"{a}_vs_{b}": mcnemar(data[a], data[b], keys) for a, b in combinations(WINDOWS, 2)}
        holm_p = holm({k: v["mcnemar_exact_p"] for k, v in pairs.items()})
        for k, v in pairs.items():
            v["p_holm_adjusted"] = holm_p[k]
            v["significant_raw_0.05"] = v["mcnemar_exact_p"] < 0.05
            v["significant_holm_0.05"] = holm_p[k] < 0.05
        return {
            "label": label,
            "n": len(keys),
            "n_ctx0": sum(1 for k in keys if k[0] == 0),
            "n_ctx1": sum(1 for k in keys if k[0] == 1),
            "sf": {w: round(sf(data[w], keys), 4) for w in WINDOWS},
            "n_pairs_tested": len(pairs),
            "pairs": pairs,
        }

    datasets = {
        "ctx0_n100": summarise(inter_ctx0, "ctx0 全 100 题（四组都完整，主口径）"),
        "ctx0_ctx1_intersection": summarise(
            inter_all, "ctx0 100 + ctx1 45 交集（四组共同完成部分）"
        ),
    }

    # ---- 对照（共同子集上可配对） ----
    controls = {}
    for c in CONTROLS:
        keys = set(data[c])
        if not keys:
            continue
        common = keys & set(data["b16_a4"])
        block = {
            "control": c,
            "control_rows": len(keys),
            "pairable_with_default_b16_a4": len(common) == len(keys),
            "common_keys_n": len(common),
            "note": "对照只跑了 ctx0 前若干题；这些题号默认组也跑过，故在共同子集上可做配对检验",
            "sf_control": round(sf(data[c], keys), 4),
            "sf_default_on_same_keys": round(sf(data["b16_a4"], common), 4)
            if common
            else None,
        }
        if common:
            block["mcnemar_control_vs_default"] = mcnemar(
                data[c], data["b16_a4"], sorted(common)
            )
            block["mcnemar_default_vs_control"] = mcnemar(
                data["b16_a4"], data[c], sorted(common)
            )
        controls[c] = block

    # ---- 与已入库聚合快照（2026-10-21_eventqa_sweep.json）对照 ----
    drift = {"note": "聚合快照 n 与原始 JSONL 现有行数比对；差异说明快照是在最后若干行写入前生成的", "windows": {}}
    agg_path = os.path.join(os.path.dirname(args.out), "2026-10-21_eventqa_sweep.json")
    if os.path.exists(agg_path):
        agg = json.load(open(agg_path))
        for w in WINDOWS:
            a = agg.get("windows", {}).get(w, {}).get("all_rows", {})
            drift["windows"][w] = {
                "aggregate_json_n": a.get("n"),
                "aggregate_json_sf": a.get("sf"),
                "raw_jsonl_n": meta[w]["rows"],
                "raw_jsonl_sf": meta[w]["sf_all_rows"],
                "delta_rows": (meta[w]["rows"] - a["n"]) if a.get("n") is not None else None,
                "delta_sf": round(meta[w]["sf_all_rows"] - a["sf"], 4) if a.get("sf") is not None else None,
            }
        for c in CONTROLS:
            a = agg.get("controls", {}).get(c, {}).get("partial", {})
            drift["windows"][c] = {
                "aggregate_json_n": a.get("n"),
                "aggregate_json_sf": a.get("sf"),
                "raw_jsonl_n": meta[c]["rows"],
                "raw_jsonl_sf": meta[c]["sf_all_rows"],
                "delta_rows": (meta[c]["rows"] - a["n"]) if a.get("n") is not None else None,
                "delta_sf": round(meta[c]["sf_all_rows"] - a["sf"], 4) if a.get("sf") is not None else None,
            }
    else:
        drift["error"] = f"aggregate snapshot not found: {agg_path}"

    out = {
        "generated_at": "2026-10-21T10:55:00+08:00",
        "purpose": "EventQA 窗口方向的配对检验：同 (context_id, question_idx) 交集 + McNemar 精确检验 + 配对差值 95% CI + MDE",
        "raw_data": {
            "server_path": "/share/exp4group/results/sweep/",
            "host": "v100-max2 (v100-2)",
            "committed": False,
            "copied_to": "/tmp/sweep/ (local scratch, not in repo)",
        },
        "correctness_metric": "eventqa_recall >= 0.5（与 sf / eventqa_recall 聚合口径一致，6 选 1）",
        "method": {
            "pairing": "按 (context_id, question_idx) 取交集，逐题配对；不同 context 的题不混在一起",
            "test": "McNemar 精确检验：双侧二项检验，n=不一致对数 (b+c)，p=0.5；同一数据集内 6 对比较用 Holm-Bonferroni 校正",
            "ci": "主口径 Newcombe(1998) method 10 配对比例差 CI（小样本更稳）；副口径 Wald，var=(b+c)/n^2 - Δ^2/n",
            "mde": "最小可检测效应量（α=0.05 双侧、power=0.80）：MDE=(1.96+0.8416)*sqrt(π_d/n)，π_d 为观测不一致率",
            "alpha": 0.05,
            "power": 0.8,
            "two_sided": True,
            "multiplicity": "Holm-Bonferroni over the 6 pairwise window comparisons within each dataset",
        },
        "sources": meta,
        "drift_vs_committed_aggregate": drift,
        "datasets": datasets,
        "controls": controls,
        "sample_size_planning": {
            "formula": "n ≈ π_d * ((z_{0.975}+z_{0.80}) / δ)^2（π_d=观测不一致率，δ=目标差值）",
            "pi_d_used": 0.12,
            "note": "π_d 取 ctx0 n=100 上 b0_a20 vs b16_a4 的观测不一致率（12/100）",
            "n_required_80power": {
                f"delta_{int(d*100)}pp": int(
                    math.ceil(0.12 * ((Z_ALPHA + Z_POWER) / d) ** 2)
                )
                for d in (0.05, 0.08, 0.10)
            },
            "current_n_ctx0": 100,
            "current_n_intersection": len(inter_all),
        },
        "verdict_rules": [
            "p >= 0.05（或 Holm 校正后 >= 0.05）或 CI 跨 0 → 该对差异不显著，不得声称优劣",
            "CI 完全在 0 一侧且 p < 0.05 且过 Holm → 方向有统计支持",
            "观测效应 |ΔSF| < MDE → 当前 n 不足以稳定判定该量级差异（power < 0.80）",
        ],
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"wrote {args.out}")

    for name, ds in datasets.items():
        print(f"\n== {name}  n={ds['n']} (ctx0={ds['n_ctx0']}, ctx1={ds['n_ctx1']})")
        print("   SF:", ds["sf"])
        for pair, r in ds["pairs"].items():
            print(
                f"   {pair}: Δ={r['delta_pp']:+.2f}pp CI[{r['ci95_low_pp']:+.2f},{r['ci95_high_pp']:+.2f}] "
                f"b={r['a_correct_b_wrong']} c={r['a_wrong_b_correct']} p={r['mcnemar_exact_p']:.4g} MDE={r['mde_pp_80power']}pp"
            )
    print("\n== controls")
    for c, b in controls.items():
        print(f"   {c}: n={b['common_keys_n']} sf={b['sf_control']} vs default {b['sf_default_on_same_keys']}")
        m = b.get("mcnemar_control_vs_default")
        if m:
            print(
                f"      Δ={m['delta_pp']:+.2f}pp CI[{m['ci95_low_pp']:+.2f},{m['ci95_high_pp']:+.2f}] "
                f"b={m['a_correct_b_wrong']} c={m['a_wrong_b_correct']} p={m['mcnemar_exact_p']:.4g} MDE={m['mde_pp_80power']}pp"
            )


if __name__ == "__main__":
    main()
