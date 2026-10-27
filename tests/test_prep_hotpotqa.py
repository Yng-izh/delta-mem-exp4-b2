#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tests/test_prep_hotpotqa.py — HotpotQA 转换器的纯函数单测（不联网、不需要 datasets）。

为什么值得单独测：转换器里有两处"写错了不会报错"的地方
  1) `evidence` 必须能在 `units` 里逐条找到原文 —— 否则 `target_slot_index` 退回
     "按答案串找槽位"，而 HotpotQA 的答案串经常出现在干扰段落里 → loss_addr 标错槽位，
     训练损失照样下降。
  2) `order=shuffled` 必须只改顺序、**不改集合** —— 否则"抗干扰"实验里的顺序轴同时
     变了内容，结果无法归因。

用法：python tests/test_prep_hotpotqa.py
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import prep_hotpotqa as P

N_OK = 0
FAILS = []


def ok(name, cond, extra=""):
    global N_OK
    if cond:
        N_OK += 1
    else:
        FAILS.append(name)
        print(f"[FAIL] {name} {extra}")
    return bool(cond)


REC = {
    "id": "q1",
    "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
    "answer": "Arthur's Magazine",
    "type": "comparison",
    "level": "hard",
    "supporting_facts": {"title": ["Arthur's Magazine", "First for Women"],
                         "sent_id": [0, 0]},
    "context": {"title": ["Arthur's Magazine", "First for Women", "D1", "D2"],
                "sentences": [
                    ["Arthur's Magazine was an American literary periodical."],
                    ["First for Women is a women's magazine."],
                    ["Arthur's Magazine was started in 1844 in Philadelphia.", "Unrelated."],
                    ["Something about First for Women and 2011."],
                ]},
}


def test_build_item():
    it = P.build_item(REC, n_distractors=1, order="natural", rng=random.Random(0))
    ok("H1 units = 黄金段落全部句子 + 1 个干扰段落",
       it["meta"]["n_gold_paras"] == 2 and it["meta"]["n_distractor_paras"] == 1
       and it["meta"]["n_units"] == 4, extra=str(it["meta"]))
    ok("H2 evidence 是支持事实原文（不是整段）",
       it["evidence"] == ["Arthur's Magazine was an American literary periodical.",
                          "First for Women is a women's magazine."], extra=str(it["evidence"]))
    ok("H3 evidence 全部能在 units 里找到（槽位定位不会退回答案串）",
       all(e in it["units"] for e in it["evidence"]))
    ok("H4 memory 是 units 的拼接且 hops=2（comparison）",
       it["memory"] == " ".join(it["units"]) and it["hops"] == 2)
    ok("H5 答案出现在**干扰段落**里时 meta.leak=True",
       it["meta"]["leak"] is True, extra="干扰段落 D1 含 Arthur's Magazine 字样")
    ok("H5b 黄金段落别处的同句不算 leak（HotpotQA 答案天然取自黄金段落）",
       P.build_item(REC, n_distractors=0, order="natural",
                    rng=random.Random(3))["meta"]["leak"] is False)

    # H6 集合不变性：shuffled 只改顺序，不改集合
    a = P.build_item(REC, n_distractors=2, order="natural", rng=random.Random(1))
    b = P.build_item(REC, n_distractors=2, order="shuffled", rng=random.Random(1))
    ok("H6 shuffled 与 natural 的 units 集合相同",
       sorted(a["units"]) == sorted(b["units"]) and a["question"] == b["question"])
    ok("H7 shuffled 的顺序确实变了（否则顺序轴没有对照意义）",
       a["units"] != b["units"])

    # H8 干扰段落数可调到 0（= 只有黄金事实，抗干扰实验的下界）
    c = P.build_item(REC, n_distractors=0, order="natural", rng=random.Random(2))
    ok("H8 n_distractors=0 时只剩黄金段落的句子（2 段各 1 句）",
       c["meta"]["n_distractor_paras"] == 0 and c["meta"]["n_units"] == 2,
       extra=str(c["meta"]))

    # H9 支持事实标题不在 context 里必须报错，不许静默产出错标签
    bad = dict(REC, supporting_facts={"title": ["Nope"], "sent_id": [0]})
    try:
        P.build_item(bad, n_distractors=1, order="natural", rng=random.Random(0))
        raised = False
    except ValueError:
        raised = True
    ok("H9 supporting_facts 标题缺失报错", raised)

    # H10 shuffled 但没给 rng -> 报错（拒绝不可复现的随机）
    try:
        P.build_item(REC, n_distractors=1, order="shuffled", rng=None)
        raised = False
    except ValueError:
        raised = True
    ok("H10 shuffled 缺 rng 报错", raised)


def test_layout_compat():
    """转换结果必须能直接被训练侧的 load_synth_jsonl 吃下。"""
    import tempfile
    import train_latent_memory as T

    it = P.build_item(REC, n_distractors=1, order="natural", rng=random.Random(0))
    d = tempfile.mkdtemp(prefix="hp_")
    path = os.path.join(d, "x.jsonl")
    P.write_jsonl([it], path)
    items = T.load_synth_jsonl(path)
    ok("H11 load_synth_jsonl 能吃下转换结果", len(items) == 1, extra=str(items[:1]))
    got = items[0]
    ok("H12 units 逐条保留（一个事实一个槽位）",
       list(got["units"]) == it["units"], extra=str(got.get("units"))[:200])
    idx = T.target_slot_index(got["units"], got["answer"], evidence=got.get("evidence"))
    ok("H13 target_slot_index 用 evidence 命中黄金槽位",
       got["units"][idx] in it["evidence"], extra=f"idx={idx} unit={got['units'][idx]!r}")


def main():
    test_build_item()
    test_layout_compat()
    print("=" * 70)
    if FAILS:
        print(f"FAILED {len(FAILS)}/{N_OK + len(FAILS)}: {FAILS}")
        return 1
    print(f"ALL PASS ({N_OK} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
