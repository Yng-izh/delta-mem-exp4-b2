#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/gen_synth_qa.py — 从**独立于评测集**的事实文本自动构建单跳 / 多跳问答

为什么需要它
------------
`train_latent_memory.py` 需要一个"事实 -> 问答"的训练集，而**训练事实绝不能来自最终评测集**
（MemoryAgentBench / LongMemEval / EventQA 的切片）。否则潜层读出学到的是评测集本身，
指标再好也只是背答案。所以训练数据必须从另一处事实文本抽取。

流水线
------
    事实文本行 ──parse_fact_triple()──> (subject, relation, object)
                                        │
                    ┌───────────────────┴────────────────────┐
                    ▼                                        ▼
            单跳问答（主体）                        多跳问答（少量补充）
      Q: 由 relation 选模板，问 subject            f1: S --rel1--> M
      A: object                                   f2: M --rel2--> X  （obj(f1)==subj(f2) 归一化相等）
                                                  Q: 链式模板（curated 优先，否则通用）
                                                  A: X
    每条样本的 memory = 目标事实 + N 条干扰事实（同关系族优先），避免"唯一事实就是答案"。

三元组解析器直接复用 `v7_slotpool.parse_fact_triple` —— 与线上 CR 图检索**同一个解析器**。
换解析器会让训练分布与推理分布错位，那是最难查的一类 bug。

已知边界（故意不猜）
--------------------
- 关系不在 `REL_SPECS` 里 → **跳过**，不生成半通顺的问题；跳过原因计入 `--stats`。
- 主语是代词 / 客体过长 / 含分号 → 跳过（这些几乎都是解析噪声）。
- `was born in` 的客体按"4 位年份"判定问 when 还是 where。
- 多跳只接受 obj(f1) 与 subj(f2) **归一化后精确相等**（与 `v9_cr.linked()` 同一硬约束）。

用法
----
    # 看几条（不写文件）
    python tools/gen_synth_qa.py --facts tools/demo_facts.txt --n-single 6 --n-multi 3 --show 9

    # 只检查解析质量（哪些句子解析失败、为什么）
    python tools/gen_synth_qa.py --facts tools/demo_facts.txt --debug-parse

    # 批量产出训练集
    python tools/gen_synth_qa.py --facts my_facts.txt --out data/synth_train.jsonl \\
        --n-single 4000 --n-multi 600 --distractors 3 --seed 13
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from v7_slotpool import parse_fact_triple   # 与线上 CR 同一个解析器（纯文本，无模型）

# =====================================================================
# 1) 关系 -> 问句模板
# =====================================================================
# q      : 单跳问句模板（{S} = 主语原文）
# noun   : 关系名词短语，供多跳通用模板拼句（"the <noun> of <S>"）
# group  : 关系族，用于挑干扰事实（同族干扰才真的有迷惑性）
REL_SPECS = {
    "works at":                    {"q": "Where does {S} work?",                       "noun": "employer",      "group": "employ",  "proper": True},
    "worked at":                   {"q": "Where did {S} work?",                       "noun": "former employer", "group": "employ",  "proper": True},
    "works for":                   {"q": "Who does {S} work for?",                    "noun": "employer",      "group": "employ",  "proper": True},
    "worked for":                  {"q": "Who did {S} work for?",                     "noun": "former employer", "group": "employ",  "proper": True},
    "is located in":               {"q": "Where is {S} located?",                     "noun": "location",      "group": "locate",  "proper": True},
    "was located in":              {"q": "Where was {S} located?",                    "noun": "location",      "group": "locate",  "proper": True},
    "is located in the country of":   {"q": "Which country is {S} in?",               "noun": "country",       "group": "locate",  "proper": True},
    "is located in the continent of": {"q": "Which continent is {S} in?",             "noun": "continent",     "group": "locate",  "proper": True},
    "was educated at":             {"q": "Where was {S} educated?",                   "noun": "alma mater",    "group": "educate", "proper": True},
    "is educated at":              {"q": "Where is {S} educated?",                    "noun": "alma mater",    "group": "educate", "proper": True},
    "is married to":               {"q": "Who is {S} married to?",                    "noun": "spouse",        "group": "family",  "proper": True},
    "was married to":              {"q": "Who was {S} married to?",                   "noun": "former spouse", "group": "family",  "proper": True},
    "is the capital of":           {"q": "What is {S} the capital of?",               "noun": "capital of",    "group": "geo",     "proper": True},
    "is a citizen of":             {"q": "What country is {S} a citizen of?",         "noun": "citizenship",   "group": "geo",     "proper": True},
    "holds the position of":       {"q": "What position does {S} hold?",              "noun": "position",      "group": "role"},
    "is famous for":               {"q": "What is {S} famous for?",                   "noun": "claim to fame", "group": "fame"},
    "was founded by":              {"q": "Who founded {S}?",                          "noun": "founder",       "group": "found",   "proper": True},
    "is founded by":               {"q": "Who founded {S}?",                          "noun": "founder",       "group": "found",   "proper": True},
    "founded":                     {"q": "What did {S} found?",                       "noun": "founding",      "group": "found",   "proper": True},
    "was written by":              {"q": "Who wrote {S}?",                            "noun": "author",        "group": "write",   "proper": True},
    "is written by":               {"q": "Who wrote {S}?",                            "noun": "author",        "group": "write",   "proper": True},
    "wrote":                       {"q": "What did {S} write?",                       "noun": "work",          "group": "write",   "proper": True},
    "was directed by":             {"q": "Who directed {S}?",                         "noun": "director",      "group": "direct",  "proper": True},
    "is directed by":              {"q": "Who directed {S}?",                         "noun": "director",      "group": "direct",  "proper": True},
    "was created in":              {"q": "When was {S} created?",                     "noun": "creation year", "group": "time"},
    "was born in":                 {"q": "Where was {S} born?",                       "noun": "birthplace",    "group": "birth",   "proper": True},
    "was born on":                 {"q": "When was {S} born?",                        "noun": "birth date",    "group": "birth"},
    "is affiliated with":          {"q": "What is {S} affiliated with?",              "noun": "affiliation",   "group": "affil"},
}

# 多跳：链式问句（(rel1_group, rel2_group) -> 模板）。命中即用自然问法，
# 否则退回通用 "What is the <noun2> of the <noun1> of <S>?"（略生硬但语义无歧义）。
CHAIN_TEMPLATES = {
    ("employ", "locate"):     "Where is the organization where {S} works located?",
    ("employ", "found"):      "Who founded the organization where {S} works?",
    ("employ", "employ"):     "Where does the employer of {S} work?",
    ("educate", "locate"):    "Where is the institution where {S} was educated located?",
    ("educate", "found"):     "Who founded the institution where {S} was educated?",
    ("birth", "locate"):      "Where is the birthplace of {S} located?",
    ("birth", "geo"):         "Which entity has the birthplace of {S} as its capital?",
    ("birth", "employ"):      "Where does the birthplace of {S} work?",   # 罕见，仅占位
    ("found", "locate"):      "Where is the organization founded by {S} located?",
    ("found", "geo"):         "What country is the organization founded by {S} in?",
    ("write", "locate"):      "Where is the work written by {S} located?",  # 罕见
    ("family", "employ"):     "Where does the spouse of {S} work?",
    ("family", "birth"):      "Where was the spouse of {S} born?",
    ("family", "locate"):     "Where is the spouse of {S} located?",
    ("role", "employ"):       "Where does the person holding {S}'s position work?",
    ("fame", "direct"):       "Who directed the work that {S} is famous for?",
    ("direct", "locate"):     "Where is the work directed by {S} located?",
}

# 精确到第二跳**关系**的链式模板（优先于 group 级）：答案类型决定问法。
# 例："is located in the country of" 的答案是国家 -> 问 "In which country ..."，
# 而不是泛泛的 "Where is ... located?"（后者在答案是"国家"时略不精确）。
CHAIN_TEMPLATES_REL = {
    ("employ",  "is located in the country of"):     "In which country is the employer of {S}?",
    ("birth",   "is located in the country of"):     "In which country is the birthplace of {S}?",
    ("educate", "is located in the country of"):     "In which country is the institution where {S} was educated?",
    ("found",   "is located in the country of"):     "In which country is the organization founded by {S}?",
    ("employ",  "is located in the continent of"):   "On which continent is the employer of {S}?",
    ("birth",   "is located in the continent of"):   "On which continent is the birthplace of {S}?",
}

# 同一关系的多种问法：单条事实可产出多条样本（表面多样性），避免模型只学会一种句式。
PARAPHRASES = {'works at': ['Where does {S} work?', 'Which organization employs {S}?', 'Where is {S} employed?'], 'worked at': ['Where did {S} work?', 'Which organization employed {S}?'], 'works for': ['Who does {S} work for?', 'Which organization does {S} work for?'], 'worked for': ['Who did {S} work for?'], 'is located in': ['Where is {S} located?', 'In which place is {S}?'], 'was located in': ['Where was {S} located?'], 'is located in the country of': ['Which country is {S} in?', 'In which country is {S} located?'], 'is located in the continent of': ['Which continent is {S} in?'], 'was educated at': ['Where was {S} educated?', 'Which institution did {S} attend?'], 'is educated at': ['Where is {S} educated?'], 'is married to': ['Who is {S} married to?', 'Who is the spouse of {S}?'], 'was married to': ['Who was {S} married to?'], 'is the capital of': ['What is {S} the capital of?', '{S} is the capital of which entity?'], 'is a citizen of': ['What country is {S} a citizen of?', 'Of which country is {S} a citizen?'], 'holds the position of': ['What position does {S} hold?', 'Which position is held by {S}?'], 'is famous for': ['What is {S} famous for?', 'For what is {S} known?'], 'was founded by': ['Who founded {S}?', 'By whom was {S} founded?'], 'is founded by': ['Who founded {S}?'], 'founded': ['What did {S} found?', 'Which organization did {S} found?'], 'was written by': ['Who wrote {S}?'], 'is written by': ['Who wrote {S}?'], 'wrote': ['What did {S} write?', 'Which work did {S} write?'], 'was directed by': ['Who directed {S}?'], 'is directed by': ['Who directed {S}?'], 'was created in': ['When was {S} created?'], 'was born in': ['Where was {S} born?', 'In which place was {S} born?'], 'was born on': ['When was {S} born?'], 'is affiliated with': ['What is {S} affiliated with?']}
PARAPHRASES_YEAR = ['When was {S} born?', 'In which year was {S} born?']

YEAR_RE = re.compile(r"^(1[0-9]{3}|20[0-9]{2})$")
PRONOUNS = {"he", "she", "it", "they", "we", "i", "you", "his", "her", "its",
            "their", "him", "them", "this", "that", "these", "those"}
BAD_OBJ_CHARS = (";", " and ", " but ", " because ", " which ", " who ")
MAX_SUBJ_WORDS = 5
MAX_OBJ_WORDS = 6


def norm(s):
    """实体归一化：小写、去冠词/标点、压空格（与 v9_cr.linked 的口径一致）。"""
    t = str(s or "").lower().strip().strip(".,;:\"'()[]")
    t = re.sub(r"^(the|a|an)\s+", "", t)
    t = re.sub(r"[^a-z0-9\u00c0-\u024f\u4e00-\u9fff ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# =====================================================================
# 2) 抽取 + 过滤
# =====================================================================
def extract(fact):
    """事实文本 -> (triple 或 None, 跳过原因)。"""
    fact = (fact or "").strip()
    if not fact:
        return None, "empty"
    if not fact.endswith("."):
        fact = fact + "."
    try:
        subj, rel, obj = parse_fact_triple(fact)
    except Exception as e:                       # 解析器对畸形输入不该抛，兜住并记账
        return None, f"parse_error:{type(e).__name__}"
    subj, rel, obj = (str(subj or "").strip(), str(rel or "").strip(), str(obj or "").strip())
    if not subj or not rel or not obj:
        return None, "incomplete_triple"
    if subj == rel == obj:
        return None, "degenerate_triple"
    if norm(subj) == norm(obj):
        return None, "subj_equals_obj"
    if subj.split()[0].lower() in PRONOUNS:
        return None, "pronoun_subject"
    if len(subj.split()) > MAX_SUBJ_WORDS:
        return None, "subject_too_long"
    if len(obj.split()) > MAX_OBJ_WORDS:
        return None, "object_too_long"
    if any(b in f" {obj.lower()} " for b in BAD_OBJ_CHARS):
        return None, "object_has_clause"

    rel_key = re.sub(r"\s+", " ", rel.lower()).strip(" .")
    spec = REL_SPECS.get(rel_key)
    if spec is None:
        return None, f"unmapped_relation:{rel_key}"
    variant = None
    if rel_key == "was born in" and YEAR_RE.match(obj.strip(".")):
        # 同一关系、客体是年份 -> 问 when 而不是 where（否则问句与答案类型错配）
        spec = {"q": "When was {S} born?", "noun": "birth year", "group": "birth"}
        variant = "year"
    if spec.get("proper") and not has_proper_noun(obj):
        # 实体型关系的客体必须是专名。"Elena was educated at the university."
        # 解析没问题，但 "the university" 不identify任何实体，答案不可核验 -> 丢弃。
        return None, "object_not_specific"
    return {"text": fact, "subject": subj, "relation": rel_key, "object": obj,
            "group": spec["group"], "noun": spec["noun"], "q": spec["q"],
            "proper": bool(spec.get("proper")), "variant": variant}, None


def has_proper_noun(text):
    """客体里是否含专名（首字母大写的词）。用于判定答案是否"可核验"。"""
    return any(w[:1].isupper() for w in str(text or "").split() if w)


def read_facts(path):
    """读事实文本：.jsonl 取 "text"/"fact" 字段（可带 cluster）；其余按行。

    返回 [(text, cluster|None)]。cluster 用于按簇切 train/val：簇间实体不共享，
    所以按簇切分不会出现"训练见过同一个实体"的泄漏。
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    out.append((line, None))
                    continue
                out.append((rec.get("text") or rec.get("fact") or "", rec.get("cluster")))
            else:
                out.append((line, None))
    return [(t, c) for t, c in out if t]


def read_passages(path):
    """从段落文本里切句（--passages 用）。只做粗切句，解析过滤交给 extract。"""
    sents = []
    with open(path, encoding="utf-8") as f:
        for block in f.read().split("\n\n"):
            for s in re.split(r"(?<=[.!?])\s+", block.replace("\n", " ")):
                s = s.strip()
                if 4 <= len(s.split()) <= 40:
                    sents.append(s)
    return sents


# =====================================================================
# 3) 单跳 / 多跳 组装
# =====================================================================
def question_forms(f, limit=1):
    """同一三元组的多种问法（去重保序）。limit=1 时就是主模板。"""
    pool = (PARAPHRASES_YEAR if f.get("variant") == "year"
            else PARAPHRASES.get(f["relation"], [f["q"]]))
    out, seen = [], set()
    for tpl in pool:
        q = tpl.format(S=f["subject"])
        if q not in seen:
            seen.add(q)
            out.append(q)
        if len(out) >= max(1, limit):
            break
    return out


def build_single(facts, rng, distractors=3, variants=1):
    out = []
    for i, f in enumerate(facts):
        # 干扰事实：同关系族优先（才有迷惑性），不足则用其他事实补
        # 干扰事实不能自带答案：否则答案字符串同时出现在两行，问题有歧义
        # （实测未过滤时 6.71% 的样本答案出现在 !=1 条记忆里）。
        ans = norm(f["object"])
        same = [g for j, g in enumerate(facts)
                if j != i and g["group"] == f["group"] and norm(g["object"]) != ans]
        other = [g for j, g in enumerate(facts)
                 if j != i and g["group"] != f["group"] and norm(g["object"]) != ans]
        rng.shuffle(same)
        rng.shuffle(other)
        mem = [f["text"]] + [g["text"] for g in (same + other)[:max(0, distractors)]]
        for q in question_forms(f, variants):
            out.append({
                "kind": "single", "hops": 1,
                "memory": list(mem),
                "question": q,
                "answer": f["object"],
                "subject": f["subject"], "relation": f["relation"],
                "evidence": [f["text"]],
                "meta": {"group": f["group"], "noun": f["noun"],
                         "variant": f.get("variant"), "cluster": f.get("cluster"),
                         "n_distractors": len(mem) - 1},
            })
    return out


def components(facts):
    """按共享实体做连通分量（并查集）。并发时按分量切分，多跳链不会被切断。"""
    parent = list(range(len(facts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 只按**实体**连边：主语恒为实体；客体仅当关系是实体型时才算实体。
    # 反例（踩过）："chief engineer" 只有 12 个取值、"The Silent Fjord" 这类作品名会重复，
    # 若把它们也当实体连边，1200 个簇会被连成 1 个分量 -> 并发退化为单进程。
    first = {}
    for i, f in enumerate(facts):
        keys = [("s", norm(f["subject"]))]
        if f.get("proper"):
            keys.append(("o", norm(f["object"])))
        for key in keys:
            if key in first:
                union(first[key], i)
            else:
                first[key] = i
    groups = {}
    for i in range(len(facts)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _worker(payload):
    """子进程：在一个连通分量子集上生成单跳 + 多跳。"""
    idxs, facts, seed, distractors, variants, n_multi = payload
    sub = [facts[i] for i in idxs]
    rng = random.Random(seed)
    singles = build_single(sub, rng, distractors, variants)
    multis = build_multi(sub, rng, n_multi or None, distractors)
    return singles, multis


def stratify(records, key):
    """按 key 分桶后轮转取，保证前 N 条覆盖不同关系（否则语料顺序会让演示全是 works at）。"""
    buckets = {}
    for r in records:
        buckets.setdefault(key(r), []).append(r)
    out = []
    keys = sorted(buckets)
    while any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k]:
                out.append(buckets[k].pop(0))
    return out


def build_multi(facts, rng, limit=None, distractors=3):
    """obj(f1) 归一化 == subj(f2) 归一化 -> 2 跳链。"""
    by_subj = {}
    for j, f in enumerate(facts):
        by_subj.setdefault(norm(f["subject"]), []).append(j)
    chains, seen = [], set()
    for i, f1 in enumerate(facts):
        for j in by_subj.get(norm(f1["object"]), []):
            if j == i:
                continue
            f2 = facts[j]
            key = (norm(f1["subject"]), norm(f1["object"]), norm(f2["object"]), f2["relation"])
            if key in seen:
                continue
            tpl = (CHAIN_TEMPLATES_REL.get((f1["group"], f2["relation"]))
                   or CHAIN_TEMPLATES.get((f1["group"], f2["group"])))
            if tpl:
                q = tpl.format(S=f1["subject"])
            elif f2["relation"] == "is located in the country of":
                q = f"In which country is the {f1['noun']} of {f1['subject']}?"
            elif f2["relation"] == "is located in the continent of":
                q = f"On which continent is the {f1['noun']} of {f1['subject']}?"
            else:
                # 通用模板：语义无歧义，只是略生硬
                q = (f"What is the {f2['noun']} of the {f1['noun']} of {f1['subject']}?")
            ans2 = norm(f2["object"])
            others = [g["text"] for k, g in enumerate(facts)
                      if k not in (i, j) and norm(g["object"]) != ans2]
            rng.shuffle(others)
            seen.add(key)
            chains.append({
                "kind": "multi", "hops": 2,
                "memory": [f1["text"], f2["text"]] + others[:max(0, distractors)],
                "question": q,
                "answer": f2["object"],
                "bridge_entity": f1["object"],
                "subject": f1["subject"],
                "relations": [f1["relation"], f2["relation"]],
                "evidence": [f1["text"], f2["text"]],
                "meta": {"groups": [f1["group"], f2["group"]],
                         "answer_type": f2["relation"],
                         "clusters": [f1.get("cluster"), f2.get("cluster")],
                         "templated": bool(tpl),
                         "n_distractors": len(others[:max(0, distractors)])},
            })
    rng.shuffle(chains)
    return chains[:limit] if limit else chains


# =====================================================================
# 4) CLI
# =====================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="从事实文本构建单跳/多跳问答")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--facts", help="一行一条事实（或 jsonl 带 text/fact 字段）")
    src.add_argument("--passages", help="段落文本，内部切句后逐句解析")
    ap.add_argument("--out", help="输出 jsonl（缺省只打印统计/样本）")
    ap.add_argument("--n-single", type=int, default=0, help="单跳条数（0=全部）")
    ap.add_argument("--n-multi", type=int, default=0, help="多跳条数（0=全部；默认少量补充）")
    ap.add_argument("--multi-ratio", type=float, default=0.0,
                    help=">0 时按 单跳条数×该比例 决定多跳条数（覆盖 --n-multi）")
    ap.add_argument("--distractors", type=int, default=3, help="每条样本附带的干扰事实数")
    ap.add_argument("--variants", type=int, default=1,
                    help="同一三元组产出几种问法（表面多样性，默认 1）")
    ap.add_argument("--cluster-range", default="",
                    help="只取 cluster 在 [A,B) 的事实（如 0:1600 训练 / 1600:1800 验证）")
    ap.add_argument("--parallel", type=int, default=1,
                    help="并发进程数（按共享实体的连通分量切分，多跳链不会被切断）")
    ap.add_argument("--total", type=int, default=0,
                    help="目标样本总数（0=不限制）；多跳条数由 --multi-ratio 决定")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--show", type=int, default=0, help="打印前 N 条样本")
    ap.add_argument("--no-stratify", dest="stratify", action="store_false",
                    help="关闭分层轮转（默认开启：按关系轮转，避免演示全是同一个关系）")
    ap.add_argument("--debug-parse", action="store_true", help="只打印每条事实的解析结果与跳过原因")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    if args.passages:
        raw = [(t, None) for t in read_passages(args.passages)]
    else:
        raw = read_facts(args.facts)
    lo, hi = 0, 10 ** 9
    if args.cluster_range:
        a, _, b = args.cluster_range.partition(":")
        lo, hi = int(a or 0), int(b or 10 ** 9)
    facts, skipped, n_filtered = [], {}, 0
    for line, cluster in raw:
        if cluster is not None and not (lo <= int(cluster) < hi):
            n_filtered += 1
            continue
        tri, why = extract(line)
        if tri is None:
            skipped[why] = skipped.get(why, 0) + 1
        else:
            tri["cluster"] = cluster
            facts.append(tri)

    if args.debug_parse:
        print(f"[parse] 输入 {len(raw)} 条 -> 解析成功 {len(facts)} 条")
        for line in raw:
            tri, why = extract(line)
            if tri is None:
                print(f"  SKIP  {line[:78]:<78} <- {why}")
            else:
                print(f"  OK    ({tri['subject']}) --[{tri['relation']}]--> ({tri['object']})"
                      f"  q={tri['q'].format(S=tri['subject'])}")
        if skipped:
            print("[skip reasons]", json.dumps(skipped, ensure_ascii=False, sort_keys=True))
        return 0

    if not facts:
        print("[ERR] 没有可用的三元组（全部被过滤）")
        return 1

    # --- 生成（按连通分量并发：多跳链不会被切到两个进程）---
    comps = components(facts)
    workers = max(1, min(int(args.parallel), len(comps)))
    if workers > 1:
        shards = [[] for _ in range(workers)]
        for k, comp in enumerate(sorted(comps, key=len, reverse=True)):
            shards[k % workers].extend(comp)
        payloads = [(sh, facts, args.seed * 1000 + k, args.distractors,
                     args.variants, args.n_multi) for k, sh in enumerate(shards)]
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_worker, payloads)
        singles = [r for sp, _ in parts for r in sp]
        multis = [r for _, mm in parts for r in mm]
    else:
        singles = build_single(facts, rng, args.distractors, args.variants)
        multis = build_multi(facts, rng, args.n_multi or None, args.distractors)

    # --- 去重（同问句+同答案只留一条）---
    def dedup(recs):
        seen, out = set(), []
        for r in recs:
            key = (r["question"].lower().strip(), r["answer"].lower().strip())
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out, len(recs) - len(out)

    singles, dup_s = dedup(singles)
    multis, dup_m = dedup(multis)
    rng.shuffle(singles)
    rng.shuffle(multis)

    # --- 配比与截断 ---
    n_multi = int(args.n_multi)
    if args.multi_ratio > 0:
        base = args.total or args.n_single or len(singles)
        n_multi = max(1, int(round(base * args.multi_ratio)))
    # --total 模式下必须显式截断多跳（哪怕目标条数是 0），否则多跳会把配额吃光、
    # 单跳被切成 0 条 —— 这个 bug 在 --total 500 --multi-ratio 0 时暴露过。
    if args.total or n_multi:
        multis = multis[:max(0, n_multi)]
    if args.total:
        singles = singles[:max(0, args.total - len(multis))]
    elif args.n_single:
        singles = singles[:args.n_single]
    if args.stratify:
        singles = stratify(singles, key=lambda r: r["relation"])
        multis = stratify(multis, key=lambda r: tuple(r["meta"]["groups"]))

    stats = {
        "facts_in": len(raw), "facts_used": len(raw) - n_filtered,
        "filtered_by_cluster": n_filtered, "cluster_range": args.cluster_range or None,
        "parsed": len(facts), "skipped": skipped,
        "components": len(comps), "workers": workers,
        "single": len(singles), "multi": len(multis),
        "total": len(singles) + len(multis),
        "dedup_dropped": {"single": dup_s, "multi": dup_m},
        "multi_templated": sum(1 for m in multis if m["meta"]["templated"]),
        "relations_used": sorted({f["relation"] for f in facts}),
        "n_questions_per_relation": {},
    }
    for f in facts:
        stats["n_questions_per_relation"][f["relation"]] = \
            stats["n_questions_per_relation"].get(f["relation"], 0) + 1
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))

    records = singles + multis
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for i, r in enumerate(records):
                f.write(json.dumps(dict(r, id=f"synth-{i:05d}"), ensure_ascii=False) + "\n")
        print(f"[out] {args.out}: {len(records)} 条")
        with open(args.out + ".stats.json", "w", encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=2, sort_keys=True)

    for r in records[:args.show]:
        print("\n" + "-" * 68)
        print(f"[{r['kind']} hops={r['hops']}] Q: {r['question']}")
        print(f"A: {r['answer']}")
        for k, line in enumerate(r["memory"], 1):
            mark = "*" if line in r["evidence"] else " "
            print(f"  {mark} {k}. {line}")
        if r["kind"] == "multi":
            print(f"  bridge={r['bridge_entity']}  rels={r['relations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
