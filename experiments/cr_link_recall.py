#!/usr/bin/env python3
# cr_link_recall.py — t19：量化 CR 两处硬门控的**反向风险**（纯解析，CPU，不需要 GPU/模型）
#
# 背景：v9 CR 有两处"宁可不断、不可误连"的保守硬门控，各有反向风险：
#   A. 2 跳桥接硬约束（t1）：linked(f1,f2) 要求 _cr_norm(obj(f1)) == _cr_norm(subj(f2))。
#      反向风险 = 该成的链不成 → 2 跳召回损失。
#   B. 属性类别硬门控（t2）：同主语但 _cr_attr_key 不同一律不覆写。
#      反向风险 = 本该被新值覆写的旧值留下来（stale），与"覆写丢槽"是同一类 bug 的另一面。
#
# 数据：本机真实 CR 数据（Conflict_Resolution / FactConsolidation 的 fact pool + 金标问答）。
#   - fact pool：results/groupA_factconsolidation_mh_6k.jsonl 的 user_prompt 里那段
#     "Here is a list of facts: 0. ... 1. ..."（100 行共享同一 pool，脚本内断言只此一份）。
#   - 问句/金标：同一文件的 mh_6k 100 题（多跳）+ results/v6groupB2_factconsolidation_sh_6k.jsonl
#     的 sh_6k 100 题（单跳，同一 pool 家族：87% 金标出现在本 pool 文本中）。
#   - sh_262k / mh_262k 的 context（18333 条事实）本机没有落盘（/share 未挂载），
#     JSON 里明确标 unavailable，不编造数值。
#
# 用法：
#   python3 experiments/cr_link_recall.py                     # 打印摘要
#   python3 experiments/cr_link_recall.py --write             # 同时写 experiments/results/cr_link_recall.json
import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import torch
import torch.nn.functional as F

import v9_slotpool as v9
from v7_slotpool import parse_fact_triple
from v9_slotpool import _cr_attr_key, _cr_verb_core, _cr_norm

POOL_SRC = os.path.join(REPO, "results/groupA_factconsolidation_mh_6k.jsonl")
SH_SRC = os.path.join(REPO, "results/v6groupB2_factconsolidation_sh_6k.jsonl")
OUT_JSON = os.path.join(HERE, "results/cr_link_recall.json")

THRESH = 0.05          # 用户口径：失败率 > 5% 才动归一
MIN_JUDGED = 30        # A 的判定样本下限：少于这个数不下结论
MIN_COVERAGE = 0.30    # A 的覆盖率下限：judged / required

# ---------------------------------------------------------------------------
# 0. 数据加载
# ---------------------------------------------------------------------------
FACT_SPLIT_RE = re.compile(r"(?<![0-9])(\d+)\.\s")


def load_facts():
    """从真实 prompt 里切出 fact pool（serial 严格递增，0,1,2,...）。

    返回 [(serial, text)]；同时断言 100 行的 pool 完全一致（否则口径不可比）。
    """
    rows = [json.loads(l) for l in open(POOL_SRC, encoding="utf-8") if l.strip()]
    pools = set()
    for r in rows:
        up = r["user_prompt"]
        pools.add(up[: up.find("I have learned the facts")])
    assert len(pools) == 1, f"预期单一 fact pool，实际 {len(pools)} 个"
    part = pools.pop()

    cand = [(int(m.group(1)), m.end()) for m in FACT_SPLIT_RE.finditer(part)]
    starts, expect = [], 0
    for num, end in cand:                     # 只接受 0,1,2,... 连续序列
        if num == expect:
            starts.append((num, end))
            expect += 1
    facts = []
    for k, (num, st) in enumerate(starts):
        end = starts[k + 1][1] if k + 1 < len(starts) else len(part)
        txt = part[st:end].strip()
        txt = re.sub(r"\s*\d+\.\s*$", "", txt).strip()   # 去掉尾部串号
        if txt and not txt.startswith("<"):
            facts.append((num, txt))
    return facts


def load_questions():
    """-> {"mh_6k": [(question, [gold...])], "sh_6k": [...]}"""
    def _gold(a):
        if isinstance(a, str):
            try:
                a = json.loads(a.replace("'", '"'))
            except Exception:
                a = [a]
        if not isinstance(a, list):
            a = [a]
        return [str(x) for x in a if str(x).strip()]

    out = {}
    mh = []
    for line in open(POOL_SRC, encoding="utf-8"):
        r = json.loads(line)
        q = r["query"]
        i = q.find("Now Answer the Question:")
        if i >= 0:
            q = q[i + len("Now Answer the Question:"):]
        mh.append((q.strip(), _gold(r["answer"])))
    out["mh_6k"] = mh
    out["sh_6k"] = [(json.loads(l)["query"].strip(), _gold(json.loads(l)["answer"]))
                    for l in open(SH_SRC, encoding="utf-8") if l.strip()]
    return out


# ---------------------------------------------------------------------------
# 1. 确定性归一（分析口径，与实现无关；也用于"应该成链/应该覆写"的判定）
# ---------------------------------------------------------------------------
GRAN_TAIL_RE = re.compile(
    r"\b(?:the\s+)?(?:city|country|state|continent|county|town|village|province|"
    r"district|republic|kingdom)\s+of\s+", re.IGNORECASE)
INST_SUFFIX = {"university", "univ", "college", "inc", "incorporated", "corp",
               "corporation", "company", "co", "ltd", "plc", "group", "school"}


def _tokens(s):
    return [w for w in re.findall(r"[a-z0-9\u00c0-\u024f]+", (s or "").lower())
            if w not in v9.STOPWORDS]


def _sing(w):
    """极简单复数归一（只处理规则复数，避免误伤）。"""
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def loose_entity_norm(s):
    """实体归一（A/C 口径）：大小写/标点/停用词/冠词 + 粒度尾链 + 机构后缀 + 单复数。

    只做**确定性**规则，无模糊匹配、无 embedding 近似。
    """
    t = (s or "").lower()
    t = GRAN_TAIL_RE.sub(" ", t)                    # "the city of London" -> "London"
    toks = [_sing(w) for w in _tokens(t)]
    while len(toks) > 1 and toks[-1] in INST_SUFFIX:  # 尾部机构后缀
        toks = toks[:-1]
    return " ".join(toks)


def loose_attr_norm(text):
    """属性类别归一（B 口径）：动词核心 + 粒度尾链剥离 + 表语头词单复数 + 时态归一。"""
    subj, vp, det, head = v9._cr_parse_parts(text)
    core = _cr_verb_core(vp)
    core = GRAN_TAIL_RE.sub(" ", core).strip()
    core = " ".join(_sing(w) for w in _tokens(core))
    head = " ".join(_sing(w) for w in _tokens(head))
    return f"{core}|{head}"


def gran_tail_of(text):
    m = GRAN_TAIL_RE.search((text or "").lower())
    return m.group(0).strip() if m else ""


# ---------------------------------------------------------------------------
# 2. 语料解析
# ---------------------------------------------------------------------------
def build_index(facts):
    """每条事实的解析结果：主语/关系/宾语 + 三种归一形式。"""
    idx = []
    for serial, text in facts:
        subj, rel, obj = parse_fact_triple(text)
        idx.append({
            "serial": serial, "text": text,
            "subj": subj, "rel": rel, "obj": obj,
            "subj_n": _cr_norm(subj), "obj_n": _cr_norm(obj),
            "subj_loose": loose_entity_norm(subj), "obj_loose": loose_entity_norm(obj),
            "attr_key": _cr_attr_key(text), "attr_loose": loose_attr_norm(text),
        })
    return idx


# ---------------------------------------------------------------------------
# 3. A：2 跳桥接召回损失
# ---------------------------------------------------------------------------
def analyse_chain_recall(idx, questions, slice_name, max_samples=10):
    """统计"金标答案需要 2 跳链"的题目里，硬约束因归一化不等而丢链的比例。

    判定（写死，避免口径漂移）：
      answer_fact  = 宾语或原文含金标答案的事实（f2）
      needs_chain  = f2 的主语**不在**问句里（问句问的是别的实体 → 必须经桥接才能到 f2）
      bridge(f1)   = f1 != f2 且 loose_entity_norm(obj(f1)) == loose_entity_norm(subj(f2))
                     且 f1 的主语**在**问句里（桥接起点就是问句实体）
      strict_ok    = 至少一个 bridge 满足硬约束 _cr_norm(obj(f1)) == _cr_norm(subj(f2))
      norm_fail    = 有 bridge 但全都不满足硬约束        ← 归一化造成的丢链（本任务要量化的）
      no_bridge    = 根本没有 bridge 候选                ← 覆盖/解析问题，不计入失败率
    """
    stat = collections.Counter()
    samples = []
    for q, golds in questions:
        qn = _cr_norm(q)
        q_toks = set(_tokens(q))
        golds_l = [g.lower() for g in golds if g.strip()]
        if not golds_l:
            stat["q_no_gold"] += 1
            continue
        answer_facts = [f for f in idx
                        if any(g in f["text"].lower() or g in f["obj"].lower()
                               for g in golds_l)]
        if not answer_facts:
            stat["q_answer_not_in_pool"] += 1
            continue
        stat["q_with_answer_fact"] += 1
        for f2 in answer_facts:
            subj_hit = (f2["subj_n"] and f2["subj_n"] in qn) or \
                       (set(f2["subj_n"].split()) and set(f2["subj_n"].split()) <= q_toks)
            if subj_hit:
                stat["af_single_hop"] += 1        # 答案事实的主语就是问句实体：不需要链
                continue
            stat["af_needs_chain"] += 1
            bridges = [f1 for f1 in idx
                       if f1["serial"] != f2["serial"]
                       and f1["obj_loose"] and f1["obj_loose"] == f2["subj_loose"]
                       and ((f1["subj_n"] and f1["subj_n"] in qn)
                            or (set(f1["subj_n"].split()) and
                                set(f1["subj_n"].split()) <= q_toks))]
            if not bridges:
                stat["af_no_bridge"] += 1
                continue
            strict = [f1 for f1 in bridges
                      if f1["obj_n"] and f1["obj_n"] == f2["subj_n"]]
            if strict:
                stat["af_strict_ok"] += 1
                continue
            stat["af_norm_fail"] += 1
            if len(samples) < max_samples:
                b = bridges[0]
                samples.append({
                    "question": q,
                    "gold": golds,
                    "f1": b["text"], "f1_serial": b["serial"],
                    "f1_obj_surface": b["obj"], "f1_obj_norm": b["obj_n"],
                    "f1_obj_loose": b["obj_loose"],
                    "f2": f2["text"], "f2_serial": f2["serial"],
                    "f2_subj_surface": f2["subj"], "f2_subj_norm": f2["subj_n"],
                    "f2_subj_loose": f2["subj_loose"],
                    "why": "归一化后 obj(f1) != subj(f2)，硬约束不成链",
                })
    judged = stat["af_strict_ok"] + stat["af_norm_fail"]
    return {
        "slice": slice_name,
        "n_questions": len(questions),
        "n_questions_with_answer_fact": stat["q_with_answer_fact"],
        "n_questions_answer_not_in_pool": stat["q_answer_not_in_pool"],
        "n_answer_facts": stat["af_single_hop"] + stat["af_needs_chain"],
        "n_answer_facts_single_hop": stat["af_single_hop"],
        "n_chains_required": stat["af_needs_chain"],
        "n_chains_strict_ok": stat["af_strict_ok"],
        "n_chains_no_bridge_candidate": stat["af_no_bridge"],
        "n_chains_norm_fail": stat["af_norm_fail"],
        "n_judged": judged,
        "failure_rate": (stat["af_norm_fail"] / judged) if judged else None,
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# 4. B：属性类别过度拒绝导致的覆写丢失
# ---------------------------------------------------------------------------
class _StubTok:
    """词级 stub tokenizer（与 test_v9_cr.py 同款），只给 StatePool 做谓词余弦用。"""

    def __call__(self, texts, add_special_tokens=False, padding=False,
                 truncation=False, max_length=None, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        seqs = [[w for w in re.findall(r"[a-z0-9\u00c0-\u024f'-]+", t.lower())] or ["<e>"]
                for t in texts]
        L = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), L), dtype=torch.long)
        am = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            for j, w in enumerate(s):
                ids[i, j] = _wid(w)
            am[i, :len(s)] = 1
        return {"input_ids": ids, "attention_mask": am}


_VOCAB = {}


def _wid(w):
    if w not in _VOCAB:
        _VOCAB[w] = len(_VOCAB) + 10
    return _VOCAB[w]


def _lexical_emb(n=20000, dim=256):
    """词袋式确定性嵌入：同词集余弦≈1，词集不同则低 —— 只为让谓词余弦可算且可解释。

    真实跑分用 Qwen 嵌入；这里只需要"同模板谓词高相似、异模板低相似"的性质，
    用来观察 StatePool 端到端写入结果（属性门控本身与嵌入无关）。
    """
    g = torch.Generator().manual_seed(20261021)
    return F.normalize(torch.randn(n, dim, generator=g), dim=-1)


def analyse_overwrite(idx, questions, facts, max_samples=10):
    """B 组指标：应覆写却因属性门控被拒（stale）的比例。

    should_overwrite 对（金标口径，与实现无关）：
      同主语（_cr_norm 严格相等）+ 同属性类别（loose_attr_norm 相等）+ 值不同
      + serial 新者应胜。这是 B4 提议的确定性归一之后的"同一属性"。
    attr_rejected：当前实现 _cr_attr_key 不同 → 门控直接拒绝覆写 → 旧值存活（stale）。
    over_merge_risk：loose 判为同属性、但两边**粒度尾链不同**（city of vs country of）
      —— 这种合并是危险的（不同粒度是不同属性），用来判断归一是否安全。
    """
    groups = collections.defaultdict(list)
    for f in idx:
        groups[(f["subj_n"], f["attr_loose"])].append(f)

    pairs, attr_rejected, over_merge_risk, same_key = [], [], [], []
    for (s_norm, a_loose), fs in groups.items():
        if not s_norm or len(fs) < 2:
            continue
        fs = sorted(fs, key=lambda x: x["serial"])
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                old, new = fs[i], fs[j]
                if old["obj_n"] == new["obj_n"]:
                    continue                     # 值相同：不算冲突
                rec = {
                    "subject": new["subj"], "attr_loose": a_loose,
                    "old_serial": old["serial"], "new_serial": new["serial"],
                    "old_value": old["obj"], "new_value": new["obj"],
                    "old_text": old["text"], "new_text": new["text"],
                    "old_attr_key": old["attr_key"], "new_attr_key": new["attr_key"],
                    "old_tail": gran_tail_of(old["text"]), "new_tail": gran_tail_of(new["text"]),
                }
                pairs.append(rec)
                if old["attr_key"] == new["attr_key"]:
                    same_key.append(rec)
                else:
                    attr_rejected.append(rec)
                    if rec["old_tail"] != rec["new_tail"]:
                        over_merge_risk.append(rec)

    # 真实 StatePool 写入（端到端 stale 计数）
    v9.tok = _StubTok()
    v9.EMB = _lexical_emb()
    import v9_cr
    pool = v9_cr.CRStatePool()          # CR 路径：upsert + 双阈值覆写 + 属性门控
    for serial, text in facts:
        pool.upsert(text, serial)
    stats = pool.stats()
    pool_lower = [t.lower() for t in pool.texts]

    def _survives(val):
        v = (val or "").lower().strip().rstrip(".")
        return any(v and v in t for t in pool_lower)

    stale = [p for p in pairs
             if _survives(p["old_value"]) and _survives(p["new_value"])]

    # 槽级复核（**这才是真 stale**）：同一主语 + 同一 loose 属性类别下，池里是否并存
    # 多个"值不同"的活槽。上面 `stale` 用的是"值是否作为子串出现在池里"，会被别的
    # 事实里的同名字符串污染，只能当粗代理；决策用槽级口径。
    slot_groups = collections.defaultdict(list)
    for t in pool.texts:
        s_i, _r, o_i = parse_fact_triple(t)
        slot_groups[(_cr_norm(s_i), loose_attr_norm(t))].append((_cr_norm(o_i), t))
    dup_attr_groups = []
    for key, vs in slot_groups.items():
        objs = {o for o, _t in vs if o}
        if len(objs) > 1:
            dup_attr_groups.append({"key": list(key), "n_slots": len(vs),
                                    "values": sorted(objs),
                                    "texts": [t for _o, t in vs]})
    n_slot_groups = sum(1 for vs in slot_groups.values() if any(o for o, _ in vs))

    # 逐题口径：金标答案所在冲突组是否受门控影响（"比例"要落到题目上才有意义）
    q_with_conflict = q_at_risk = 0
    for q, golds in questions:
        gl = [g.lower() for g in golds if g.strip()]
        hit = [p for p in pairs
               if any(g in p["new_value"].lower() or g in p["new_text"].lower() for g in gl)]
        if not hit:
            continue
        q_with_conflict += 1
        if any(p in attr_rejected for p in hit):
            q_at_risk += 1

    n_pairs = len(pairs)
    return {
        "n_facts_written": len(facts),
        "pool_size_after_write": len(pool),
        "n_should_overwrite_pairs": n_pairs,
        "n_groups": len([g for g in groups.values() if len(g) > 1]),
        "n_attr_rejected_pairs": len(attr_rejected),
        "overwrite_miss_rate": (len(attr_rejected) / n_pairs) if n_pairs else None,
        "n_over_merge_risk_pairs": len(over_merge_risk),
        "over_merge_risk_rate": (len(over_merge_risk) / n_pairs) if n_pairs else None,
        "n_same_attr_key_pairs": len(same_key),
        "pool_stats": {k: stats.get(k) for k in
                       ("n_merge_checked", "n_merge_rejected", "n_attr_rejected",
                        "n_overwrites", "n_dup") if k in stats},
        "n_attr_rejected_gt0": 1 if stats.get("n_attr_rejected", 0) > 0 else 0,
        "n_attr_rejected_gt0_rate": 1.0 if stats.get("n_attr_rejected", 0) > 0 else 0.0,
        "n_stale_pairs": len(stale),
        "stale_rate": (len(stale) / n_pairs) if n_pairs else None,
        "stale_definition": "粗代理：old/new 两个值都作为子串出现在池文本里（会被别的事实污染）",
        "n_slot_groups": n_slot_groups,
        "n_duplicate_attr_groups": len(dup_attr_groups),
        "duplicate_attr_rate": (len(dup_attr_groups) / n_slot_groups) if n_slot_groups else None,
        "duplicate_attr_definition": "槽级：同主语 + 同 loose 属性类别下池中并存多个不同值的活槽 = 真 stale",
        "samples_duplicate_attr": dup_attr_groups[:max_samples],
        "n_questions_with_conflict_group": q_with_conflict,
        "n_questions_at_risk": q_at_risk,
        "question_at_risk_rate": (q_at_risk / q_with_conflict) if q_with_conflict else None,
        "samples_attr_rejected": attr_rejected[:max_samples],
        "samples_over_merge_risk": over_merge_risk[:max_samples],
        "samples_stale": stale[:max_samples],
    }


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="写 experiments/results/cr_link_recall.json")
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()

    facts = load_facts()
    questions = load_questions()
    idx = build_index(facts)

    chain = {name: analyse_chain_recall(idx, qs, name, args.samples)
             for name, qs in questions.items()}
    overwrite = analyse_overwrite(idx, questions["mh_6k"], facts, args.samples)

    # A：两个切片都看。但**判定样本很小**——大量题目在解析层就找不到桥接候选
    #    （no_bridge_candidate），只有一小部分链真正被判过。判定样本不足时不得下结论。
    a_mh = chain["mh_6k"]["failure_rate"]
    a_sh = chain["sh_6k"]["failure_rate"]

    def _cov(c):
        req = c.get("n_chains_required") or 0
        j = c.get("n_judged") or 0
        return (j / req) if req else None

    cov_mh, cov_sh = _cov(chain["mh_6k"]), _cov(chain["sh_6k"])
    judged = {"mh_6k": chain["mh_6k"].get("n_judged") or 0,
              "sh_6k": chain["sh_6k"].get("n_judged") or 0}
    a_rates = [r for r in (a_mh, a_sh) if r is not None]
    a_worst = max(a_rates) if a_rates else None
    a_underpowered = any(n < MIN_JUDGED for n in judged.values()) or any(
        (c is not None and c < MIN_COVERAGE) for c in (cov_mh, cov_sh))
    a_over = (not a_underpowered) and a_worst is not None and a_worst > THRESH

    # B：决策用**槽级**口径（真 stale）。pair 级 overwrite_miss_rate 是"属性键不同"的比例，
    #    实测为 0，但它看不见"同一属性因写入时序落到不同槽"的重复；粗代理 stale_rate 会被
    #    子串污染。三个数都报，决策只认 duplicate_attr_rate。
    b_pair = overwrite["overwrite_miss_rate"]
    b_slot = overwrite["duplicate_attr_rate"]
    b_over = b_slot is not None and b_slot > THRESH

    conclusion = {
        "threshold": THRESH,
        "A_failure_rate_mh_6k": a_mh,
        "A_failure_rate_sh_6k": a_sh,
        "A_judged_n": judged,
        "A_coverage": {"mh_6k": cov_mh, "sh_6k": cov_sh},
        "A_decision_rule": (
            f"判定样本需 ≥{MIN_JUDGED} 条且覆盖率 ≥{MIN_COVERAGE:.0%} 才下结论；"
            "否则记未定论（本轮 mh/sh 判定样本分别只有 "
            f"{judged['mh_6k']}/{judged['sh_6k']} 条，大量题目在解析层找不到桥接候选）"),
        "B_pair_level_attr_reject_rate": b_pair,
        "B_slot_level_duplicate_attr_rate": b_slot,
        "B_stale_substring_proxy_rate": overwrite["stale_rate"],
        "B_decision_rule": "用槽级 duplicate_attr_rate（真 stale）；pair 级与子串代理仅作参考",
        "A_exceeds_threshold": a_over,
        "A_underpowered": a_underpowered,
        "B_exceeds_threshold": b_over,
        "actions": [],
    }
    if a_underpowered:
        conclusion["actions"].append(
            f"A 未定论：判定样本 {judged['mh_6k']}(mh) / {judged['sh_6k']}(sh) 条、覆盖率 "
            f"{(cov_mh or 0):.0%}/{(cov_sh or 0):.0%}，最差切片失败率 {(a_worst or 0):.2%} 不足以支撑改动 "
            "→ 不实现别名归一；要提高覆盖率需换事实库更全的切片（sh/mh_262k 本机未落盘）")
    elif a_over:
        conclusion["actions"].append(
            f"A 超阈（最差切片 {a_worst:.2%}，样本充足）：应在 v9_cr 内加 _cr_link_norm（仅确定性规则："
            "大小写/标点/冠词/粒度尾链/机构后缀），硬约束本身不变 —— **本轮未实现**")
    else:
        conclusion["actions"].append(
            f"A 未超阈（最差切片 {a_worst:.2%} < {THRESH:.0%}）：按计划不实现别名归一")
    if b_over:
        conclusion["actions"].append(
            f"B 超阈（槽级重复 {b_slot:.2%}）：应在 v9_slotpool._cr_verb_core 内加确定性归一"
            "（粒度尾链/冠词/单复数），并补反例测试 —— **本轮未实现**")
    else:
        conclusion["actions"].append(
            f"B 未超阈（槽级重复 {b_slot:.2%} < {THRESH:.0%}）：按计划不动 _cr_verb_core")

    out = {
        "task": "t19",
        "generated_at": "2026-10-21",
        "data": {
            "pool_source": os.path.relpath(POOL_SRC, REPO),
            "questions": {k: os.path.relpath(v, REPO) for k, v in
                          (("mh_6k", POOL_SRC), ("sh_6k", SH_SRC))},
            "n_facts": len(facts),
            "n_questions": {k: len(v) for k, v in questions.items()},
            "unavailable_slices": {
                "sh_262k": "本机未落盘该 slice 的 context（18333 条事实，/share 未挂载），未测量",
                "mh_262k": "同上",
            },
            "notes": "纯解析统计，不需要 GPU/模型；事实与金标均来自真实 CR 跑分产物。",
        },
        "A_chain_recall": chain,
        "B_overwrite_miss": overwrite,
        "conclusion": conclusion,
    }

    print(f"[A] mh_6k 失败率 = {a_mh:.2%}" if a_mh is not None else "[A] mh_6k 无样本")
    print(f"[A] sh_6k 失败率 = {a_sh:.2%}" if a_sh is not None else "[A] sh_6k 无样本")
    print(f"[A] mh_6k 需链={chain['mh_6k']['n_chains_required']} "
          f"strict_ok={chain['mh_6k']['n_chains_strict_ok']} "
          f"norm_fail={chain['mh_6k']['n_chains_norm_fail']} "
          f"no_bridge={chain['mh_6k']['n_chains_no_bridge_candidate']}")
    print(f"[B] should_overwrite={overwrite['n_should_overwrite_pairs']} "
          f"pair_attr_rejected={overwrite['n_attr_rejected_pairs']} "
          f"pair_rate={b_pair:.2%} over_merge_risk={overwrite['over_merge_risk_rate']:.2%}"
          if b_pair is not None else "[B] 无样本")
    print(f"[B] pool_stats={overwrite['pool_stats']}")
    print(f"[B] 粗代理 stale={overwrite['n_stale_pairs']}（子串口径，仅参考） | "
          f"槽级重复组={overwrite['n_duplicate_attr_groups']}/{overwrite['n_slot_groups']} "
          f"= {(b_slot or 0):.2%}（**决策口径**）")
    print(f"[conclusion] {json.dumps(conclusion, ensure_ascii=False, indent=2)}")

    if args.write:
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[write] {OUT_JSON}")
    return out


if __name__ == "__main__":
    main()
