#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/gen_latent_min16.py — 实验 1 的 16 条反事实配对样本（P0 可学习性验证）。

设计（为什么是反事实配对）：
    8 组 × 2 条。同组**问题完全相同**、记忆里目标事实的值不同：
        Taron works at Velcor.  -> Where does Taron work? -> Velcor
        Taron works at Nivora.  -> Where does Taron work? -> Nivora
    这样"直接背 Question→Answer"不可行：同一问题对应两个不同答案，
    模型必须真的从潜层记忆里读出"这条记忆里的值"才能两个都答对。

约束：
    - 虚构实体（音节合成，与评测集零重叠）
    - 答案 1–3 token（短实体名）
    - 每条样本 1 个事实槽位；无干扰事实、无多跳
    - 训练集 = 测试集 = 同 16 条（刻意设计的过拟合实验）

输出：synth/latent_min16.jsonl（字段与 gen_synth_qa.py 的产物一致，
      train_latent_memory.py 的 --data 直接消费）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from v7_slotpool import parse_fact_triple  # noqa: E402  与正式语料同一抽取器

# 8 组反事实配对：(subject, relation, [value_a, value_b], question 模板)
# subject 全部不同 —— 同一主语出现两次会让"值"本身有歧义，破坏配对设计。
# 值都取单音节词或 2 词以内短名：Qwen tokenizer 下 1-3 token（实测）。
PAIRS = [
    ("Taron Velstrik",   "works at",        ["Velcor", "Nivora"],            "Where does {s} work?"),
    ("Mira Quevald",     "was born in",     ["Ostrel", "Vandhe"],            "Where was {s} born?"),
    ("Dov Krasilm",      "studied at",      ["Haldwyn", "Perthia"],          "Where did {s} study?"),
    ("Yuna Frolmist",    "is married to",   ["Elian", "Sachen"],             "Who is {s} married to?"),
    ("Ravik Tolmstead",  "founded",         ["Brackline", "Ondrey"],         "What did {s} found?"),
    ("Isla Norvund",     "works as",        ["geologist", "archivist"],      "What does {s} work as?"),
    ("Piet Arlensk",     "lives in",        ["Kestmor", "Duvhal"],           "Where does {s} live?"),
    ("Sela Vriqtash",    "is employed by",  ["Fenwick", "Talquin"],          "Who is {s} employed by?"),
]

SENT_TEMPLATES = {
    "works at":       "{s} works at {v}.",
    "was born in":    "{s} was born in {v}.",
    "studied at":     "{s} studied at {v}.",
    "is married to":  "{s} is married to {v}.",
    "founded":        "{s} founded {v}.",
    "works as":       "{s} works as a {v}.",
    "lives in":       "{s} lives in {v}.",
    "is employed by": "{s} is employed by {v}.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "synth", "latent_min16.jsonl"))
    args = ap.parse_args()

    rows = []
    for s, rel, values, q_t in PAIRS:
        for v in values:
            sent = SENT_TEMPLATES[rel].format(s=s, v=v)
            triple = parse_fact_triple(sent)
            if triple is None:
                raise SystemExit(f"抽取失败（模板与 parse_fact_triple 不兼容）：{sent!r}")
            subj, rel_p, obj = triple
            rows.append({
                "kind": "single", "hops": 1,
                "memory": [sent],
                "question": q_t.format(s=s),
                "answer": v,
                "subject": subj, "relation": rel_p, "evidence": [sent],
            })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 自检：8 组 × 2 条；同组问题一致、答案不同；答案长度 1-3 token 交给用户侧验证
    assert len(rows) == 16, len(rows)
    by_q = {}
    for r in rows:
        by_q.setdefault(r["question"], []).append(r["answer"])
    assert len(by_q) == 8, f"配对组数应为 8，得到 {len(by_q)}"
    for q, answers in by_q.items():
        assert len(set(answers)) == 2, f"{q} 的两个答案应不同：{answers}"
        assert len(answers) == 2
    print(f"[gen] {args.out}: {len(rows)} 条 / {len(by_q)} 组反事实配对，全部通过 parse_fact_triple")


if __name__ == "__main__":
    main()
