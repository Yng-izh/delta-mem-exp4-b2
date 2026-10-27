#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/prep_mab.py — MemoryAgentBench 的 Conflict Resolution 转成现有训练/评测格式。

为什么只做 CR
-------------
用户定的主线问题（多事实抗干扰、动态更新与历史访问）在 MAB 里对应 Conflict
Resolution：同一个事实键在上下文里被反复改写，只有**最新**的值算对。这正是 FST-Mem
"状态—历史分离 + 局部更新"要证明的场景，也是唯一一个官方标注了"冲突覆盖"语义的
公开子任务。

数据事实（实测 /share/latent/MemoryAgentBench/data/Conflict_Resolution-*.parquet）
  - 8 个 context 组（factconsolidation_{sh,mh}_{6k,32k,64k,262k}），每组 100 问 = 800 问
  - context = `"Here is a list of facts:\n0. <fact>\n1. ..."`，**编号递增 = 越靠后越新**
  - metadata 里只有 source / qa_pair_ids，**没有**金标事实编号、没有时间戳、没有 question_types
    → 金标只能机械推导：事实文本里含答案值的那条；同一键的多条里取**编号最大**的
      那条（这就是"最新值"，正是 CR 的判定）。推导规则写在 `derive_gold`。

容量（必须写明，不许偷偷截断）
----------------------------
一条 6k context 有几百条事实，而我们的槽位记忆一条事实一个槽、每槽要付一次编码前向：
几百槽 × 每问 = 每问几十秒。所以按 `--cap` 限定每问可见事实数，选法是
"金标链 + 同键冲突版本 + BM25 相关事实"，**同键所有版本必须一起进**，否则
"用最新值覆盖旧值"这条被测能力会被选择策略本身消掉。cap 会写进每条样本的 meta，
报告里必须带出来。

用法
----
    python tools/prep_mab.py --cap 32 --out synth/mab_cr_cap32.jsonl
    python tools/prep_mab.py --cap 64 --limit-questions 50 --out synth/mab_cr_small.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

LETTERS = re.compile(r"[a-z0-9]+")
FACT_LINE = re.compile(r"^(\d+)\.\s*(.+)$")


def _norm(s):
    return " ".join(LETTERS.findall(str(s).lower()))


def _toks(s):
    return LETTERS.findall(str(s).lower())


# ---- 事实模板 → (subject, relation) -------------------------------------------------
# MAB-CR 的事实由 Wikidata 三元组按模板生成，模板集很小且稳定；
# 用显式模板表解析而不是 LLM 抽取：可复现、零成本、错了一眼能看出来。
TEMPLATES = [
    (re.compile(r"^The (.+) of (.+) is (.+)$"), 2, 1),
    (re.compile(r"^The (.+) of (.+) is located in the (.+) of (.+)$"), 2, 1),
    (re.compile(r"^(.+) is located in the (.+) of (.+)$"), 1, 2),
    (re.compile(r"^(.+) was born in the (.+) of (.+)$"), 1, "born_in_city"),
    (re.compile(r"^(.+) died in the (.+) of (.+)$"), 1, "died_in_city"),
    (re.compile(r"^(.+) worked in the (.+) of (.+)$"), 1, "worked_in_city"),
    (re.compile(r"^(.+) is a citizen of (.+)$"), 1, "citizenship"),
    (re.compile(r"^(.+) is married to (.+)$"), 1, "spouse"),
    (re.compile(r"^(.+) plays the (.+) of (.+)$"), 1, "plays_position"),
    (re.compile(r"^(.+) is located in the (.+)$"), 1, "located_in"),
    (re.compile(r"^(.+) was written by (.+)$"), 1, "author"),
    (re.compile(r"^(.+) is (.+)$"), 1, "is_a"),
]


def parse_fact(text):
    """把一条事实文本解析成 (entity, relation, value)。解析不出时 relation='other'。

    relation 是**事实键的一半**：同一 entity 同一 relation 的不同取值就是"冲突版本"。
    解析失败全部退化到 'other' 会把不同属性混进同一组 —— 那等于把更新做错，
    所以 `other` 的比例必须统计并在报告里写明。
    """
    t = str(text).strip().rstrip(".")
    for pat, ent_grp, rel_grp in TEMPLATES:
        m = pat.match(t)
        if not m:
            continue
        ent = m.group(ent_grp).strip()
        if isinstance(rel_grp, int):
            rel = "rel:" + _norm(m.group(rel_grp))
        else:
            rel = rel_grp
        val = m.groups()[-1].strip()
        return ent, rel, val
    return t, "other", t


def parse_context(ctx):
    """返回 [{'idx':int, 'text':str, 'entity':str, 'relation':str, 'value':str}]。"""
    facts = []
    for line in str(ctx).split("\n"):
        line = line.strip()
        if not line or line.lower().startswith("here is a list"):
            continue
        m = FACT_LINE.match(line)
        if not m:
            continue
        idx, text = int(m.group(1)), m.group(2).strip()
        ent, rel, val = parse_fact(text)
        facts.append({"idx": idx, "text": text, "entity": ent,
                      "relation": rel, "value": val})
    return facts


def key_of(f):
    return (_norm(f["entity"]), f["relation"])


def derive_gold(facts, answers):
    """金标 = 含答案值的事实里**编号最大**的那条（最新值）。

    返回 (gold_idx 列表, 命中的答案串)。答案值在文本里按词边界匹配，避免
    "London" 命中 "Londonderry"；允许 answers 是单层或多层列表。
    """
    cands = []
    for a in answers:
        if isinstance(a, (list, tuple)):
            cands.extend(str(x) for x in a)
        else:
            cands.append(str(a))
    cands = [c for c in cands if c.strip()]
    hits = []
    used = []
    for c in cands:
        c_n = _norm(c)
        if not c_n:
            continue
        hit = [f for f in facts
               if re.search(r"(?<![a-z0-9])" + re.escape(c_n) + r"(?![a-z0-9])",
                            _norm(f["text"]))]
        if hit:
            hits.append(max(f["idx"] for f in hit))
            used.append(c)
    if not hits:
        return [], None
    # 只保留"最新"的那条事实编号（CR 的判定就是最新值）
    newest = max(hits)
    return [newest], used[0]


def bm25_scores(facts, question):
    q = set(_toks(question))
    if not q:
        return {f["idx"]: 0.0 for f in facts}
    df = {}
    for f in facts:
        for w in set(_toks(f["text"])):
            df[w] = df.get(w, 0) + 1
    n = len(facts)
    out = {}
    for f in facts:
        d = _toks(f["text"])
        s = 0.0
        for w in q:
            if w in d:
                import math
                s += math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
        out[f["idx"]] = s / (1 + 0.02 * len(d))
    return out


def select_facts(facts, question, gold_idx, cap):
    """金标 + 同键全部版本 + BM25 相关事实，凑到 cap 条（保持原始编号顺序）。

    同键版本必须整组进入：只留最新那条，"抗干扰/覆盖旧值"就没有被考到；
    只留旧的那条则样本无解。两种都会被这里挡住。
    """
    by_idx = {f["idx"]: f for f in facts}
    keep = set()
    for i in gold_idx:
        if i not in by_idx:
            continue
        keep.add(i)
        k = key_of(by_idx[i])
        for f in facts:
            if key_of(f) == k:
                keep.add(f["idx"])
    if len(keep) > cap:
        raise ValueError(f"金标+同键版本已 {len(keep)} 条 > cap={cap}，请调大 cap")
    scores = bm25_scores(facts, question)
    rest = sorted((scores[i], -by_idx[i]["idx"]) for i in by_idx if i not in keep)
    for _s, negi in rest:
        if len(keep) >= cap:
            break
        keep.add(-negi)
    # 金标必须还在（理论上一定在，但选择逻辑改了就靠这条兜住）
    if not set(gold_idx) <= keep:
        raise ValueError("选择后金标丢失")
    return sorted(keep)


def build_item(facts, keep, question, answers, gold_idx, gold_val, meta, qid):
    sel = [f for f in facts if f["idx"] in keep]
    units = [f["text"] for f in sel]
    ev = [f["text"] for f in sel if f["idx"] in gold_idx]
    keys = [key_of(f) for f in sel]
    n_conflict = max([sum(1 for k2 in keys if k2 == k) for k in keys])
    return {
        "kind": "mab_cr",
        "memory": " ".join(units),
        "units": units,
        # 时间轴用事实编号：CR 的"新"就是编号更大（数据集没有时间戳）
        "unit_meta": [{"entity": f["entity"], "relation": f["relation"],
                       "time": f["idx"], "distractor": f["idx"] not in gold_idx}
                      for f in sel],
        "question": question,
        "answer": gold_val,
        "evidence": ev,
        "hops": 1,
        "as_of": max(keep),
        "meta": dict(meta, qid=qid, n_units=len(units), n_kept=len(keep),
                     n_gold=len(ev), max_versions=n_conflict,
                     gold_idx=gold_idx),
    }


def load_rows(path=None, task="cr"):
    if path is None:
        path = f"/share/latent/MemoryAgentBench/data/{'Conflict_Resolution' if task == 'cr' else 'Accurate_Retrieval'}-00000-of-00001.parquet"
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist(), path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--cap", type=int, default=32, help="每条样本最多可见事实数")
    ap.add_argument("--limit-questions", type=int, default=0, help="每组只取前 N 问")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-caps", default="", help="逗号分隔，如 16,32,64：额外产出抗干扰扫描集")
    args = ap.parse_args(argv)

    rows, path = load_rows(args.parquet)
    print(f"[in] {path} 组数={len(rows)}", flush=True)

    items, dropped, other_rels = [], 0, 0
    caps = [args.cap] + ([int(x) for x in args.all_caps.split(",") if x.strip()]
                         if args.all_caps else [])
    extra = {c: [] for c in caps[1:]}
    for r in rows:
        source = (r.get("metadata") or {}).get("source") or "?"
        facts = parse_context(r["context"])
        other_rels += sum(1 for f in facts if f["relation"] == "other")
        qs = r["questions"]
        if args.limit_questions:
            qs = qs[:args.limit_questions]
        for qi, q in enumerate(qs):
            ans = r["answers"][qi]
            gold_idx, gold_val = derive_gold(facts, ans)
            if not gold_idx:
                dropped += 1
                continue
            qid = f"{source}_q{qi}"
            for cap in caps:
                try:
                    keep = select_facts(facts, q, gold_idx, cap)
                except ValueError as e:
                    dropped += 1
                    print(f"[skip] {qid} cap={cap}: {e}", file=sys.stderr)
                    continue
                it = build_item(facts, keep, q, ans, gold_idx, gold_val,
                                {"source": source, "cap": cap,
                                 "n_facts_total": len(facts)}, qid)
                if cap == caps[0]:
                    items.append(it)
                else:
                    extra[cap].append(it)

    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    d = os.path.dirname(os.path.abspath(args.out)) or "."
    stem = os.path.splitext(os.path.basename(args.out))[0]
    for cap, its in extra.items():
        sub = os.path.join(d, f"{stem}_cap{cap}.jsonl")
        with open(sub, "w", encoding="utf-8") as f:
            for it in its:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[out] {len(its)} 条 -> {sub} (cap={cap})", flush=True)

    nf = sum(it["meta"]["n_facts_total"] for it in items) or 1
    nu = sum(it["meta"]["n_units"] for it in items) / max(1, len(items))
    n_conf = sum(1 for it in items if it["meta"]["max_versions"] > 1) / max(1, len(items))
    print(f"[out] {len(items)} 条 -> {args.out} | cap={args.cap} "
          f"平均可见事实={nu:.1f} 含冲突版本样本={n_conf:.1%} "
          f"解析失败关系={other_rels / nf:.1%} 丢弃={dropped}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
