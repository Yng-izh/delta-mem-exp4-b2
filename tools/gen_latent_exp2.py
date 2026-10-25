#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/gen_latent_exp2.py — 实验 2 数据：200 训练 + 64 验证，每条 4 槽（1 目标 + 3 干扰）。

设计（承接实验 1 的反事实配对）：
    - 1 个 (subject, relation) 组合 = 1 组反事实配对：问题相同、目标事实的值不同，
      模型必须真的读记忆才能两组都答对（纯背 Q→A 不可行）。
    - 记忆 = 目标事实 + 3 条干扰事实（其他主语/其他关系），共 4 个事实槽位。
    - 验证集实体与训练集**不重叠**（测新实体上的泛化）。
    - 答案 1-3 token（与实验 1 同一批短值池）。
"""
from __future__ import annotations
import argparse, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from v7_slotpool import parse_fact_triple  # noqa: E402

SENT = {
    "works at": "{s} works at {v}.",        "was born in": "{s} was born in {v}.",
    "studied at": "{s} studied at {v}.",    "is married to": "{s} is married to {v}.",
    "founded": "{s} founded {v}.",          "lives in": "{s} lives in {v}.",
}
Q = {
    "works at": "Where does {s} work?",     "was born in": "Where was {s} born?",
    "studied at": "Where did {s} study?",   "is married to": "Who is {s} married to?",
    "founded": "What did {s} found?",       "lives in": "Where does {s} live?",
}
VALUES = {   # 与实验 1 同口径：Qwen tokenizer 下 1-3 token
    "works at": ["Velcor", "Nivora", "Talquin", "Fenwick", "Ondrey", "Brackline"],
    "was born in": ["Ostrel", "Vandhe", "Kestmor", "Duvhal", "Perthia", "Haldwyn"],
    "studied at": ["Haldwyn", "Perthia", "Ostrel", "Vandhe", "Kestmor", "Duvhal"],
    "is married to": ["Elian", "Sachen", "Taron", "Mira", "Ravik", "Sela"],
    "founded": ["Brackline", "Ondrey", "Velcor", "Nivora", "Talquin", "Fenwick"],
    "lives in": ["Kestmor", "Duvhal", "Ostrel", "Vandhe", "Perthia", "Haldwyn"],
}
SYL1 = ["Tar", "Mir", "Dov", "Yun", "Rav", "Isl", "Pie", "Sel", "Oma", "Kel",
        "Bran", "Fen", "Lud", "Har", "Nor", "Vey", "Quin", "Jor", "Ald", "Wen"]
SYL2 = ["ik", "a", "und", "elm", "ish", "ora", "ent", "ald", "rik", "ina",
        "ost", "urd", "eli", "arn", "ovi", "esk", "ana", "olt", "ira", "yn"]


def gen_names(rng, n):
    names, seen = [], set()
    while len(names) < n:
        s = rng.choice(SYL1) + rng.choice(SYL2)
        if s not in seen:
            seen.add(s); names.append(s)
    return names


def make_split(rng, n_pairs, subjects, relations):
    """n_pairs 组反事实配对 -> 2*n_pairs 条样本。"""
    rows = []
    for subj in subjects:
        rel = rng.choice(relations)
        v_a, v_b = rng.sample(VALUES[rel], 2)
        # 3 条干扰事实：其他主语 + 随机关系（保证与问题无歧义），配对内两条共用同一组干扰
        dis = []
        while len(dis) < 3:
            s2 = rng.choice(subjects)
            r2 = rng.choice(relations)
            if s2 == subj and r2 == rel:
                continue
            f = SENT[r2].format(s=s2, v=rng.choice(VALUES[r2]))
            if f not in dis:
                dis.append(f)
        for v in (v_a, v_b):
            tgt = SENT[rel].format(s=subj, v=v)
            mem = [tgt] + dis
            rng.shuffle(mem)
            rows.append({"kind": "single", "hops": 1, "memory": mem, "units": mem,
                         "question": Q[rel].format(s=subj), "answer": v,
                         "subject": subj, "relation": rel, "evidence": [tgt]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", default=os.path.join(ROOT, "synth", "latent_exp2_train.jsonl"))
    ap.add_argument("--val-out", default=os.path.join(ROOT, "synth", "latent_exp2_val.jsonl"))
    ap.add_argument("--n-train-pairs", type=int, default=100)
    ap.add_argument("--n-val-pairs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    relations = list(SENT)
    # 实体不重叠：训练 100 主语、验证 32 主语
    sub_train = gen_names(rng, args.n_train_pairs)
    sub_val = gen_names(rng, args.n_val_pairs + 40)
    sub_val = [s for s in sub_val if s not in set(sub_train)][:args.n_val_pairs]
    assert len(sub_val) == args.n_val_pairs
    tr = make_split(rng, args.n_train_pairs, sub_train, relations)
    va = make_split(rng, args.n_val_pairs, sub_val, relations)
    for path, rows in ((args.train_out, tr), (args.val_out, va)):
        for r in rows:
            for sent in r["units"]:
                assert parse_fact_triple(sent) is not None, sent
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 自检：配对数、同组同问题不同答案、实体不重叠
    assert len(tr) == 2 * args.n_train_pairs and len(va) == 2 * args.n_val_pairs
    assert not (set(sub_train) & set(s["subject"] for s in va))
    for rows, n in ((tr, args.n_train_pairs), (va, args.n_val_pairs)):
        by_q = {}
        for r in rows:
            by_q.setdefault(r["question"], []).append(r["answer"])
        assert len(by_q) == n and all(len(set(v)) == 2 for v in by_q.values())
    print(f"[gen] train={len(tr)} ({args.n_train_pairs} 组) -> {args.train_out}")
    print(f"[gen] val={len(va)} ({args.n_val_pairs} 组) -> {args.val_out}；实体与训练集不重叠")


if __name__ == "__main__":
    main()
