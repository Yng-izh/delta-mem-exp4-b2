#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tests/test_prep_mab.py — MAB-CR 转换器纯逻辑单测（不需要 parquet、不联网）。

要守住的三个不变量（写错都**不会报错**，只会让 CR 实验失去意义）：
  M1 金标取"含答案值且编号最大"的事实 —— 旧的冲突值不许被当成金标；
  M2 同键的**所有**版本必须一起进上下文 —— 只留最新则"覆盖"没被考到，
     只留旧值则样本无解，两种都会让更新指标失真；
  M3 每个金标事实都必须出现在 units 里，且 unit_meta 与 units 逐条对齐。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import prep_mab as M

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


CTX = """Here is a list of facts:
0. Thomas Kyd was born in the city of London.
1. The chairperson of Fatah is Mahmoud Abbas.
2. Thomas Kyd was born in the city of Strasbourg.
3. Amy Winehouse died in the city of Camden Town.
4. Thomas Kyd was born in the city of Norwich.
5. Bengaluru is located in the continent of Asia.
6. Pedro Pierluisi worked in the city of Washington, D.C.."""


def test_parse():
    facts = M.parse_context(CTX)
    ok("M0 解析出 7 条事实", len(facts) == 7, extra=str(len(facts)))
    f0 = facts[0]
    ok("M0b entity/relation/value 解析正确",
       f0["entity"] == "Thomas Kyd" and f0["relation"] == "born_in_city"
       and f0["value"] == "London", extra=str(f0))
    ok("M0c 同键三条被识别为同一 (entity, relation)",
       len({M.key_of(f) for f in facts if f["entity"] == "Thomas Kyd"}) == 1)
    ok("M0d 'The chairperson of X is Y' 的键是 X",
       facts[1]["entity"] == "Fatah", extra=str(facts[1]))


def test_gold():
    facts = M.parse_context(CTX)
    gold, val = M.derive_gold(facts, [["Norwich"]])
    ok("M1 金标取编号最大的命中（最新值）", gold == [4], extra=str(gold))
    gold2, _ = M.derive_gold(facts, [["London"]])
    ok("M1b 答案若是旧值（London），金标= 含该值的最大编号（不是恒取最新）",
       gold2 == [0], extra=str(gold2))
    gold3, v3 = M.derive_gold(facts, [["Reykjavik"]])
    ok("M1c 答案不在任何事实里 -> 空（调用方丢弃，不许猜）", gold3 == [] and v3 is None)
    # 双层 list 的答案（官方是 list[list[str]]）
    gold4, _ = M.derive_gold(facts, [["Camden Town", "Camden"]])
    ok("M1d 答案候选取命中任一", gold4 == [3], extra=str(gold4))


def test_select():
    facts = M.parse_context(CTX)
    gold, _ = M.derive_gold(facts, [["Norwich"]])
    keep = M.select_facts(facts, "Where was Thomas Kyd born?", gold, cap=7)
    ok("M2 同键所有版本都在上下文里", {0, 2, 4} <= set(keep), extra=str(sorted(keep)))
    ok("M2b 金标在上下文里", set(gold) <= set(keep))
    keep2 = M.select_facts(facts, "Where was Thomas Kyd born?", gold, cap=4)
    ok("M2c cap 生效且仍含全部同键版本",
       len(keep2) == 4 and {0, 2, 4} <= set(keep2), extra=str(sorted(keep2)))
    ok("M2d 选择结果保持原始编号顺序", keep2 == sorted(keep2))
    try:
        M.select_facts(facts, "q", gold, cap=2)
        raised = False
    except ValueError:
        raised = True
    ok("M2e cap 小于同键版本数时报错（不许静默丢版本）", raised)


def test_item():
    facts = M.parse_context(CTX)
    gold, val = M.derive_gold(facts, [["Norwich"]])
    keep = M.select_facts(facts, "Where was Thomas Kyd born?", gold, cap=5)
    it = M.build_item(facts, keep, "Where was Thomas Kyd born?", [["Norwich"]],
                      gold, val, {"source": "t", "cap": 5,
                                  "n_facts_total": len(facts)}, "q0")
    ok("M3 unit_meta 与 units 逐条对齐",
       len(it["unit_meta"]) == len(it["units"]) == len(keep))
    ok("M3b evidence 是金标事实原文且都在 units 里",
       it["evidence"] == ["Thomas Kyd was born in the city of Norwich."]
       and all(e in it["units"] for e in it["evidence"]))
    ok("M3c meta 记录 max_versions（报告里要带出冲突强度）",
       it["meta"]["max_versions"] == 3, extra=str(it["meta"]))
    ok("M3d time 用事实编号（CR 无时间戳，越靠后越新）",
       [m["time"] for m in it["unit_meta"]] == sorted(m["time"] for m in it["unit_meta"]))
    ok("M3e answer 取的是最新值对应的答案串", it["answer"] == "Norwich")


def test_layout_compat():
    import tempfile
    import json
    import train_latent_memory as T
    facts = M.parse_context(CTX)
    gold, val = M.derive_gold(facts, [["Norwich"]])
    keep = M.select_facts(facts, "Where was Thomas Kyd born?", gold, cap=5)
    it = M.build_item(facts, keep, "Where was Thomas Kyd born?", [["Norwich"]],
                      gold, val, {"source": "t", "cap": 5,
                                  "n_facts_total": len(facts)}, "q0")
    d = tempfile.mkdtemp(prefix="mab_")
    p = os.path.join(d, "x.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
    items = T.load_synth_jsonl(p)
    ok("M4 load_synth_jsonl 能吃下", len(items) == 1)
    got = items[0]
    ok("M4b units 逐条保留（一条事实一个槽位）", got["units"] == it["units"])
    ok("M4c unit_meta 保留 entity/relation/time",
       got["unit_meta"][0]["relation"] and "time" in got["unit_meta"][0])
    idx = T.target_slot_index(got["units"], got["answer"], evidence=got.get("evidence"))
    ok("M4d target_slot_index 命中金标槽位",
       got["units"][idx] in it["evidence"], extra=f"idx={idx}")


def main():
    test_parse()
    test_gold()
    test_select()
    test_item()
    test_layout_compat()
    print("=" * 70)
    if FAILS:
        print(f"FAILED {len(FAILS)}/{N_OK + len(FAILS)}: {FAILS}")
        return 1
    print(f"ALL PASS ({N_OK} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
