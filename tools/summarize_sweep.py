#!/usr/bin/env python3
"""summarize_sweep.py — 汇总 EventQA 窗口扫描的 jsonl（纯标准库，可在本机跑）。

读入若干结果文件，逐文件打印：
    n           题目数
    SF          substring_exact_match 均值
    anchor_acc  anchor_answer_hit 均值（锚点单元本身是否含答案）
    in_ctx      answer_in_context 均值（最终证据窗口是否覆盖答案）
    AR@1/3/10   answer_recall_at_k（答案串命中代理，非证据级召回）
    win         window.back/ahead（若记录里带）

用法：
    python3 tools/summarize_sweep.py results/ar_v9_eventqa_*.jsonl
    python3 tools/summarize_sweep.py --json results/*.jsonl
"""

import json
import sys


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _fmt(v):
    return "  n/a" if v is None else f"{v:.3f}"


def summarize(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    n = len(recs)
    wins = [r.get("window") for r in recs if r.get("window")]
    win = wins[0] if wins else {}
    ar = {}
    for k in ("1", "3", "10"):
        ar[k] = _mean([(r.get("answer_recall_at_k") or {}).get(k) for r in recs])
    return {
        "file": path,
        "n": n,
        "sf": _mean([r.get("substring_exact_match") for r in recs]),
        "anchor_acc": _mean([r.get("anchor_answer_hit") for r in recs]),
        "in_ctx": _mean([r.get("answer_in_context") for r in recs]),
        "ar1": ar["1"], "ar3": ar["3"], "ar10": ar["10"],
        "back": win.get("back"), "ahead": win.get("ahead"),
        "run_id": recs[0].get("run_id") if recs else None,
        "readout": (recs[0].get("readout") or {}).get("mode") if recs else None,
    }


def main(argv):
    as_json = "--json" in argv
    files = [a for a in argv if not a.startswith("--")]
    if not files:
        print(__doc__)
        return 2
    rows = [summarize(p) for p in files]
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    print(f"{'window':>10} {'n':>5} {'SF':>7} {'anchor':>7} {'in_ctx':>7} "
          f"{'AR@1':>6} {'AR@3':>6} {'AR@10':>6}  run_id / file")
    for r in rows:
        w = (f"{r['back']}/{r['ahead']}" if r["back"] is not None else "?")
        print(f"{w:>10} {r['n']:>5} {_fmt(r['sf']):>7} {_fmt(r['anchor_acc']):>7} "
              f"{_fmt(r['in_ctx']):>7} {_fmt(r['ar1']):>6} {_fmt(r['ar3']):>6} "
              f"{_fmt(r['ar10']):>6}  {r['run_id']} {r['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
