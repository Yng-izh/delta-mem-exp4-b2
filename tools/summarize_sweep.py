#!/usr/bin/env python3
"""summarize_sweep.py — 汇总 EventQA 窗口扫描的 jsonl（纯标准库，可在本机跑）。

读入若干结果文件，逐文件打印：
    n           题目数
    SF          substring_exact_match 均值
    anchor_acc  anchor_answer_hit 均值（锚点单元本身是否含答案）
    in_ctx      answer_in_context 均值（最终证据窗口是否覆盖答案）
    AR@1/3/10   answer_recall_at_k（答案串命中代理，非证据级召回）
    win         window.back/ahead（若记录里带）

联合归因四格表（与 ar_diag.summarize 同口径，直接复用 ar_diag.joint_table）：
    按 (answer_recall_at_k=1 与否) × (SF 对/错) 分四格，输出
        R1/SF1  R1/SF0  R0/SF1  R0/SF0   四格样本数
        RAcc|R1  Reader Acc | Recall=1（检索命中时读出答对的条件准确率）
        RFail    检索失败率 = (R0/SF1 + R0/SF0) / n
        RBW      读到但答错率 = R1/SF0 / n
    evidence_recall_at_k 为 None 时只给答案命中代理口径，并在输出里明确标注
    "无金标证据位置"，绝不把代理当成证据级召回。

用法：
    python3 tools/summarize_sweep.py results/ar_v9_eventqa_*.jsonl
    python3 tools/summarize_sweep.py --json results/*.jsonl
    python3 tools/summarize_sweep.py --joint-k 3 results/*.jsonl
"""

import json
import os
import sys

# 复用 ar_diag 的四格口径（同一份实现，避免两个脚本各写一套后漂移）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import ar_diag
except Exception as _e:                                    # 极端情况下退化为只报旧列
    ar_diag = None
    print(f"[warn] 无法导入 ar_diag（{_e}）；本次不输出联合归因四格表", file=sys.stderr)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _fmt(v):
    return "  n/a" if v is None else f"{v:.3f}"


def summarize(path, joint_k=1):
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
    joint = ar_diag.joint_table(recs, recall_k=joint_k) if ar_diag is not None else None
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
        "joint": joint,
    }


def main(argv):
    as_json = "--json" in argv
    joint_k = 1
    if "--joint-k" in argv:
        i = argv.index("--joint-k")
        try:
            joint_k = int(argv[i + 1])
            argv = argv[:i] + argv[i + 2:]
        except (IndexError, ValueError):
            print("[ERR] --joint-k 需要一个整数", file=sys.stderr)
            return 2
    files = [a for a in argv if not a.startswith("--")]
    if not files:
        print(__doc__)
        return 2
    rows = [summarize(p, joint_k=joint_k) for p in files]
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

    # ---- 联合归因四格表（Retrieval × Reader）----
    if ar_diag is None:
        return 0
    print()
    print(f"== 联合归因（retrieval@{joint_k} × substring_exact_match）==")
    print(f"{'file':<44} {'jn':>4} {'R1/SF1':>7} {'R1/SF0':>7} {'R0/SF1':>7} "
          f"{'R0/SF0':>7} {'RAcc|R1':>8} {'RFail':>7} {'RBW':>7}")
    any_evidence = False
    for r in rows:
        j = r["joint"] or {}
        cells = j.get("cells") or {}
        print(f"{os.path.basename(r['file'])[:44]:<44} {j.get('n', 0):>4} "
              f"{cells.get('recall1_sf1', {}).get('n', 0):>7} "
              f"{cells.get('recall1_sf0', {}).get('n', 0):>7} "
              f"{cells.get('recall0_sf1', {}).get('n', 0):>7} "
              f"{cells.get('recall0_sf0', {}).get('n', 0):>7} "
              f"{_fmt(j.get('reader_acc_given_recall1')):>8} "
              f"{_fmt(j.get('retrieval_failure_rate')):>7} "
              f"{_fmt(j.get('read_but_wrong_rate')):>7}")
        any_evidence = any_evidence or bool(j.get("evidence_based"))
    note = (rows[0]["joint"] or {}).get("evidence_note") if rows else None
    if note:
        print(f"[注] {note}")
    if not any_evidence:
        print("[注] 所有文件的 evidence_recall_at_k 均为 None：四格表只用答案命中代理，"
              "不是证据级召回")
    print("      RAcc|R1 = Reader Acc | Recall=1；RFail = 检索失败率；"
          "RBW = 读到但答错率（占四格样本）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
