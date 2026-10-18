"""v6 parse_fact_triple 6 例单测（纯文本，无 torch 依赖）"""
import sys
sys.path.insert(0, "/Users/yngizh/quant/deepseek/delta-mem-exp4")
import re, torch, torch.nn.functional as F  # noqa: F401  (module imports torch at top)
import v6_slotpool as v6

CASES = [
    # (fact, expected (subj, rel, obj) 三元组关键特征)
    ("The partner of Nicolae Ceaușescu is Elena Ceaușescu.",
     lambda t: t[0] == "Nicolae Ceaușescu" and t[1] == "partner of" and t[2] == "Elena Ceaușescu"),
    ("Elena Ceaușescu was educated at the University of Bucharest.",
     lambda t: t[0] == "Elena Ceaușescu" and t[1] == "was educated at" and "University of Bucharest" in t[2]),
    ("Mihai Eminescu was born in Ipotești.",
     lambda t: t[0] == "Mihai Eminescu" and t[1] == "was born in" and t[2] == "Ipotești"),
    ("Nicolae Ceaușescu died in Târgoviște on 25 December 1989.",
     lambda t: t[0] == "Nicolae Ceaușescu" and t[1].startswith("died") and "Târgoviște" in t[2]),
    ("Ion Creangă was born in Humulești and he wrote Childhood.",
     lambda t: t[1] == "wrote" and t[2] == "Childhood" and t[0].startswith("Ion Creangă")),
    ("Bucharest is the capital of Romania and it was developed by Carol I.",
     lambda t: "developed by" in t[1]),
]

fails = 0
for i, (fact, chk) in enumerate(CASES, 1):
    t = v6.parse_fact_triple(fact)
    ok = chk(t)
    print(f"case{i}: {'PASS' if ok else 'FAIL'}  fact={fact!r}\n   -> {t}")
    if not ok:
        fails += 1

# 禁退化裸系动词：谓词不得恰为 is/was/are/were（除非句子真的只有系动词）
for fact in ["Mihai Eminescu was born in Ipotești.", "Bucharest was developed by Carol I."]:
    t = v6.parse_fact_triple(fact)
    if t[1].lower() in v6.BARE_COPULA:
        print(f"BARE-COPULA-LEAK: {fact} -> {t}")
        fails += 1

print(f"\n{len(CASES) - fails}/{len(CASES)} passed" if fails else f"\nALL {len(CASES)} PASSED")
sys.exit(1 if fails else 0)
