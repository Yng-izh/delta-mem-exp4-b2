# test_v9_cr.py — v9_cr / v9_slotpool CR 双阈值 CPU 单测（stub tokenizer，无模型/无 GPU）
#
# 覆盖 t5 四项：
#   1. StatePool._should_merge 双阈值（主语精确匹配 + 关系谓词余弦 > 0.80）
#   2. CRStatePool：Beam=6 ego-graph 第 1 跳
#   3. 双跳链（2 事实 2 关系）联合打分 Score_chain：(fact1, fact2) 硬约束 obj==subj、
#      两段关系分别与 query 两段关系匹配；3 事实链走独立 path_len=3 分支；路径渲染
#      只渲染参与评分的事实与边（事实行数 == 边数）
#   4. 反事实强约束 Prompt（OUTDATED / counterfactual / 最大编号 / unknown）
import os
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import v9_slotpool as v9
import v9_cr


class StubTok:
    """词级 stub tokenizer（与 test_v9_slotpool.py 同款）。"""

    def __call__(self, texts, add_special_tokens=False, padding=False,
                 truncation=False, max_length=None, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        seqs = []
        for t in texts:
            ws = v9.WORD_RE.findall(t.lower())
            if not ws:
                ws = ["<empty>"]
            if truncation and max_length:
                ws = ws[:max_length]
            seqs.append(ws)
        L = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), L), dtype=torch.long)
        am = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            for j, w in enumerate(s):
                ids[i, j] = wid(w)
            am[i, :len(s)] = 1
        return {"input_ids": ids, "attention_mask": am}


DIM = 64
torch.manual_seed(0)
EMB = F.normalize(torch.randn(10000, DIM), dim=-1)


def wid(w):
    return hash(w) % 9000 + 10


def embed(w, vec):
    EMB[wid(w)] = F.normalize(torch.as_tensor(vec, dtype=torch.float32), dim=0)


def basis(i):
    v = torch.zeros(DIM)
    v[i] = 1.0
    return v


v9_cr.set_encoder(StubTok(), EMB)


def ok(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"  ok - {name}")


def _texts(pool):
    return [pool.texts[i] for i in range(len(pool))]


# ---------------- 1. 双阈值覆写 ----------------
def test_state_pool_dual_threshold():
    embed("alice", basis(0))
    embed("born", basis(2))
    embed("educated", basis(3))
    embed("paris", basis(4))
    embed("lyon", basis(5))
    embed("oxford", basis(6))

    st = v9.StatePool()
    st.add_batch([("Alice was born in Paris.", {"start": 0})])
    st.add_batch([("Alice was born in Lyon.", {"start": 1})])
    ok("dual-threshold merges same subj+rel", len(st) == 1 and st.n_dup == 1)
    ok("overwrite keeps newest text", "Lyon" in st.texts[0])

    st.add_batch([("Bob was born in Paris.", {"start": 2})])
    ok("different subject -> new slot", len(st) == 2)

    st2 = v9.StatePool()
    st2.add_batch([("Alice was born in Paris.", {"start": 0})])
    st2.add_batch([("Alice was educated at Oxford.", {"start": 1})])
    ok("same subject different relation -> new slot", len(st2) == 2)

    ok("cr keys tracked", len(st2.cr_subj) == 2 and st2.cr_relk[0] is not None)
    ok("rel theta default 0.80", abs(v9.CR_REL_THETA - 0.80) < 1e-9)


def test_cr_pool_upsert_dual_threshold():
    embed("alice", basis(0))
    embed("alicia", F.normalize(0.95 * basis(0) + 0.31 * basis(1), dim=0))
    embed("born", basis(2))
    embed("educated", basis(3))
    embed("paris", basis(4))
    embed("lyon", basis(5))
    embed("oxford", basis(6))
    embed("cambridge", basis(7))

    pool = v9_cr.CRStatePool()
    ok("is a StatePool", isinstance(pool, v9.StatePool))
    ok("empty upsert", pool.upsert("   ", 0) is False)
    ok("new fact 1", pool.upsert("Alice was born in Paris.", 1) is False)
    # 主语相似但不精确相同 -> 不覆写（v8 mean-cos 会误合）
    ok("similar subject not exact -> new", pool.upsert("Alicia was educated at Cambridge.", 2) is False)
    ok("pool size 2", len(pool) == 2)
    # 主语精确 + 谓词相同 -> 覆写（不同宾语也不影响）
    ok("exact subj+rel overwrite", pool.upsert("Alice was born in Lyon.", 3) is True)
    ok("pool size stays 2", len(pool) == 2)
    ok("n_overwrites", pool.n_overwrites == 1)
    ok("newest value wins", any("Lyon" in t for t in _texts(pool)))
    ok("stale value gone", not any("Paris" in t for t in _texts(pool)))
    # 同主语不同谓词 -> 新槽（先看 argmax 槽主语是否精确）
    pool.upsert("Alice was educated at Oxford.", 4)
    ok("same subj diff rel -> new slot", len(pool) == 3)


# ---------------- 1b. CR 属性解析：不同属性不得撞成同一谓词 ----------------
def test_cr_copular_attributes_not_merged():
    """回归：'Alice is a doctor.' 与 'Alice is vegetarian.' 不得互相覆写。

    旧实现把两条都解析成 ('Alice','is')，双阈值必然通过 -> 后者吃掉前者，
    职业属性从证据块里彻底消失。
    """
    embed("alice", basis(0))
    embed("doctor", basis(1))
    embed("vegetarian", basis(2))
    pool = v9_cr.CRStatePool()
    ok("doctor is new slot", pool.upsert("Alice is a doctor.", 1) is False)
    ok("vegetarian is new slot", pool.upsert("Alice is vegetarian.", 2) is False)
    ok("both attributes kept", len(pool) == 2 and pool.n_overwrites == 0)
    ok("doctor fact kept", any("doctor" in t for t in _texts(pool)))
    ok("vegetarian fact kept", any("vegetarian" in t for t in _texts(pool)))
    ok("subject index lists both", pool._subj_index.get(v9._cr_norm("Alice")) == [0, 1])

    block = pool.retrieve_multihop_context("What is Alice?")
    ok("block keeps both attributes",
       "doctor" in block and "vegetarian" in block)

    # 同类属性仍要原位覆写（槽 id 稳定）
    ok("same attribute overwrites",
       pool.upsert("Alice is a doctor at City Hospital.", 3) is True)
    ok("slot count stable after overwrite", len(pool) == 2)
    ok("doctor slot refreshed in place", "City Hospital" in pool.texts[0])
    ok("vegetarian slot untouched", pool.texts[1] == "Alice is vegetarian.")
    ok("serial refreshed", pool.metas[0]["serial"] == 3)
    ok("graph keys stay aligned", len(pool.cr_E_subj) == len(pool) == len(pool.cr_attr))


# ---------------- 2. Beam=6 ego-graph 第 1 跳 ----------------
def test_beam6_ego_graph():
    facts = [
        "Alice is affiliated with religA.",
        "Alice is located in placeB.",
        "Alice is famous for thingC.",
        "Alice was born in cityD.",
        "Alice was educated at schoolE.",
        "Alice was created by makerF.",
        "Alice is married to spouseG.",
        "Alice is a citizen of nationH.",
    ]
    pool = v9_cr.CRStatePool()
    for i, f in enumerate(facts, 1):
        pool.upsert(f, i)
    ok("8 distinct attributes kept", len(pool) == 8)
    ok("default beam=6", pool.beam == 6 and v9_cr.CR_BEAM == 6)

    block = pool.retrieve_multihop_context("Alice")
    dbg = pool.last_dbg
    ok("hop1 keeps beam=6 seeds", len(dbg["seeds"]) == 6)
    ok("ego adds remaining 2 attributes", len(dbg["ego"]) == 2)
    ok("ego-graph selects all 8", dbg["n_selected"] == 8)
    lines = [ln for ln in block.splitlines() if ln.strip()]
    ok("block has 8 facts", len(lines) == 8)
    serials = [int(ln.split(".")[0]) for ln in lines]
    ok("block sorted by serial (recency)", serials == list(range(1, 9)))

    # 小 beam 时 ego 仍补全同主语属性（ego-graph 独立于 beam）
    pool.retrieve_multihop_context("Alice", beam=2)
    ok("beam override", len(pool.last_dbg["seeds"]) == 2)
    ok("ego still expands", pool.last_dbg["n_selected"] >= 2 + len(pool.last_dbg["ego"]))


def test_ego_forward_neighbor():
    embed("alice", basis(0))
    embed("springfield", basis(1))
    pool = v9_cr.CRStatePool()
    pool.upsert("Alice is affiliated with Springfield.", 1)
    pool.upsert("Springfield is located in France.", 2)
    block = pool.retrieve_multihop_context("Alice")
    ego = [j for _s, j in pool.last_dbg["ego"]]
    ok("forward neighbor enters ego", 1 in ego)
    ok("ego block contains forward fact", "France" in block)


# ---------------- 3. 双跳链（2 事实 2 关系）联合打分 ----------------
def _hop_pool():
    """Alice --born in--> Springfield --located in--> France --located in--> Europe。

    0: Alice was born in Springfield.     （fact1 候选）
    1: Springfield is located in France.  （fact2：与 0 拼成真 2 跳，答案 France）
    2: Springfield is famous for pizza.   （同 seed 的干扰出边）
    3: France is located in Europe.       （第三跳：只能进 path_len=3 分支）
    4: Pizza was created in Italy.        （与任何链都不连通的孤立边）
    """
    embed("alice", basis(0))
    embed("springfield", basis(1))
    embed("france", basis(3))
    embed("pizza", basis(5))
    embed("born", basis(6))
    embed("located", basis(7))
    embed("famous", basis(8))
    embed("created", basis(9))
    embed("europe", basis(10))
    embed("italy", basis(11))
    pool = v9_cr.CRStatePool()
    for i, f in enumerate(["Alice was born in Springfield.",
                           "Springfield is located in France.",
                           "Springfield is famous for pizza.",
                           "France is located in Europe.",
                           "Pizza was created in Italy."], 1):
        pool.upsert(f, i)
    return pool


Q2 = "Alice was born in Springfield and Springfield is located in France?"


def test_chain_two_fact_structure():
    """评分主体是 (fact1, fact2)：obj(fact1)==subj(fact2) 硬约束 + 两段关系匹配。"""
    pool = _hop_pool()
    ok("chain pool size", len(pool) == 5)
    E_q, _k, _ = v9_cr._enc(Q2)
    E_qrels = [v9_cr._enc("was born in")[0], v9_cr._enc("is located in")[0]]

    good = pool.chain_score(0, 1, E_q, E_qrels)      # Alice-born->Springfield-located->France
    bad = pool.chain_score(0, 2, E_q, E_qrels)       # 第二段关系与 query 第二段不匹配
    ok("two-fact chain scores", good is not None and bad is not None)
    ok("both-relations-matching chain scores higher", good > bad + 0.1)

    # 硬约束：obj(fact1) != subj(fact2) 直接不构成链
    ok("broken chain returns None", pool.chain_score(0, 4, E_q, E_qrels) is None)
    ok("reverse chain returns None", pool.chain_score(1, 0, E_q, E_qrels) is None)
    ok("self chain returns None", pool.chain_score(0, 0, E_q, E_qrels) is None)
    ok("hard constraint on normalised entities",
       pool.linked(0, 1) is True and pool.linked(1, 3) is True
       and pool.linked(0, 3) is False)

    # 单段 query 关系时两段都比它（旧口径不回归）
    E_qrel1 = v9_cr._enc("was born in")[0]
    ok("single query relation still scores",
       pool.chain_score(0, 1, E_q, [E_qrel1]) is not None)
    ok("joint weights default", abs(pool.w_pred - 0.30) < 1e-9)


def test_third_hop_does_not_pollute_two_hop():
    """第三跳存在与否不改变前两跳的分数（v9 早期把 3 事实链当双跳打分会污染）。"""
    E_q, _k, _ = v9_cr._enc(Q2)
    E_qrels = [v9_cr._enc("was born in")[0], v9_cr._enc("is located in")[0]]

    with_third = _hop_pool()
    without = v9_cr.CRStatePool()
    for i, f in enumerate(["Alice was born in Springfield.",
                           "Springfield is located in France.",
                           "Springfield is famous for pizza.",
                           "Pizza was created in Italy."], 1):
        without.upsert(f, i)
    ok("third-hop fact absent in slim pool", len(without) == 4)
    s_with = with_third.chain_score(0, 1, E_q, E_qrels)
    s_without = without.chain_score(0, 1, E_q, E_qrels)
    ok("two-hop score independent of third hop", abs(s_with - s_without) < 1e-9)


def test_two_hop_path_ranks_first():
    """回归：真 2 跳路径必须是 dbg/路径展示里的第一选择。"""
    pool = _hop_pool()
    block = pool.retrieve_multihop_context(Q2)
    dbg = pool.last_dbg
    ok("query has two relations", dbg["query_rels"] == ["was born in", "is located in"])
    ok("query relation parsed", dbg["query_rel"].lower() == "was born in")
    ok("two-hop chains scored", len(dbg["chains2"]) >= 2)
    ok("best two-hop chain is the real 2-hop", dbg["chains2"][0][1:] == (0, 1))
    ok("two-hop chains ordered descending",
       all(dbg["chains2"][i][0] >= dbg["chains2"][i + 1][0]
           for i in range(len(dbg["chains2"]) - 1)))
    ok("chains2 entries are 2-fact", all(len(c) == 3 for c in dbg["chains2"]))

    first = dbg["paths"][0]
    ok("first path is 2-fact", first["path_len"] == 2 and first["facts"] == [0, 1])
    ok("two-hop path facts == edges", len(first["facts"]) == len(first["edges"]) == 2)
    ok("answer of the two-hop path is the 2nd object", first["labels"][-1] == "France")
    ok("path view on by default", dbg["path_view"] is True and v9_cr.CR_PATH_VIEW is True)
    ok("block renders 2-hop path first",
       block.splitlines()[0].startswith("PATH 1") and "len=2" in block.splitlines()[0])
    ok("block contains the two-hop answer", "France" in block)
    ok("dbg distinguishes path_len", "path_len" in first
       and dbg["n_paths_by_len"][2] >= 1)


def test_three_fact_chain_independent_branch():
    """3 事实链走独立 path_len=3 分支：单独打分、单独渲染，不被 2 跳分支吞掉。"""
    pool = _hop_pool()
    E_q, _k, _ = v9_cr._enc(Q2)
    E_qrels = [v9_cr._enc("was born in")[0], v9_cr._enc("is located in")[0]]

    sc3 = pool.chain_score3(0, 1, 3, E_q, E_qrels)
    ok("three-fact chain scores", sc3 is not None)
    ok("three-fact chain needs all links",
       pool.chain_score3(0, 1, 4, E_q, E_qrels) is None
       and pool.chain_score3(0, 2, 3, E_q, E_qrels) is None)
    ok("three-fact chain rejects repeated facts",
       pool.chain_score3(0, 1, 0, E_q, E_qrels) is None)

    block = pool.retrieve_multihop_context(Q2)
    dbg = pool.last_dbg
    ok("three-hop branch populated", len(dbg["chains3"]) >= 1)
    ok("best three-hop chain is the 3-fact path", dbg["chains3"][0][1:] == (0, 1, 3))
    ok("three-hop entries are 3-fact", all(len(c) == 4 for c in dbg["chains3"]))
    ok("3-hop chains never leak into the 2-hop branch",
       all(len(c["facts"]) == 2 for c in dbg["chains"] if c["path_len"] == 2))
    ok("dbg path_len counts", dbg["n_paths_by_len"][2] >= 1
       and dbg["n_paths_by_len"][3] >= 1)

    lens = [p["path_len"] for p in dbg["paths"]]
    ok("paths keep 2-hop before 3-hop", lens == sorted(lens) and 3 in lens)
    three = [p for p in dbg["paths"] if p["path_len"] == 3][0]
    ok("three-hop path facts == edges", len(three["facts"]) == len(three["edges"]) == 3)
    ok("three-hop path entities chained",
       three["labels"] == ["Alice", "Springfield", "France", "Europe"])

    # 默认 CR_PATHS 下 3 跳分支也必须有位置（不被 2 跳分支吞掉），且单独渲染
    rendered = dbg["paths_rendered"]
    ok("three-hop path rendered by default budget",
       any(p["path_len"] == 3 for p in rendered))
    hdr = [ln for ln in block.splitlines() if ln.startswith("PATH ") and "len=3" in ln]
    ok("rendered three-hop header", len(hdr) == 1)
    idx = block.splitlines().index(hdr[0])
    fact_rows = 0
    for ln in block.splitlines()[idx + 1:]:
        if not re.match(r"\s*\d+\.\s", ln):
            break
        fact_rows += 1
    ok("three-hop fact rows == edges (3)", fact_rows == 3)
    ok("two-hop paths rendered too",
       sum(1 for p in rendered if p["path_len"] == 2) >= 1)
    ok("rendered paths keep 2-hop before 3-hop",
       [p["path_len"] for p in rendered] == sorted(p["path_len"] for p in rendered))
    ok("two-hop path still ranks first",
       rendered[0]["path_len"] == 2 and rendered[0]["facts"] == [0, 1])


def test_two_hop_chain_joint_score_legacy_shape():
    """2 跳问题（Alice --attended--> UnivB --located in--> CityC，答案 CityC）。

    旧实现把 3 事实链当"双跳"打分，这条 2 跳链根本排不到第一；新实现必须排第一。
    """
    embed("alice", basis(0))
    embed("universityb", basis(1))
    embed("cityc", basis(2))
    embed("countryd", basis(4))
    embed("educated", basis(12))
    embed("located", basis(7))
    embed("attended", basis(13))
    pool = v9_cr.CRStatePool()
    for i, f in enumerate(["Alice was educated at UniversityB.",
                           "UniversityB is located in CityC.",
                           "CityC is located in CountryD.",
                           "UniversityB is famous for its museum."], 1):
        pool.upsert(f, i)
    q = "Alice was educated at UniversityB, and UniversityB is located in CityC?"
    block = pool.retrieve_multihop_context(q)
    dbg = pool.last_dbg
    ok("legacy-shape pool size", len(pool) == 4)
    ok("legacy-shape two-hop chain first", dbg["chains2"][0][1:] == (0, 1))
    ok("legacy-shape answer city is 2nd object", dbg["paths"][0]["labels"][-1] == "CityC")
    ok("legacy-shape path is 2-fact", dbg["paths"][0]["path_len"] == 2)
    ok("legacy-shape answer in block", "CityC" in block)


# ---------------- 4. 反事实强约束 Prompt ----------------
def test_counterfactual_prompt():
    block = "1. Alice was born in Paris.\n2. Alice was born in Lyon."
    p = v9_cr.build_cr_prompt("Where was Alice born?", block)
    ok("prompt has facts", block in p)
    ok("prompt has question", "QUESTION: Where was Alice born?" in p)
    ok("prompt marks outdated", "OUTDATED" in p)
    ok("prompt does counterfactual check", "Counterfactual" in p)
    ok("prompt picks largest number", "LARGEST number" in p)
    ok("prompt bans superseded value", "superseded" in p)
    ok("prompt fallback unknown", "unknown" in p)
    ok("cf prompt default on", v9_cr.CR_CF_PROMPT is True)
    p2 = v9_cr.build_cr_prompt("Q?", "")
    ok("empty block tolerated", "(no facts retrieved)" in p2 and "Q?" in p2)
    ok("prompt deterministic", p == v9_cr.build_cr_prompt("Where was Alice born?", block))


# ---------------- runner 接线冒烟（模拟 CR_MODE=v9 调用序列，不加载模型） ----------------
def test_runner_wiring_smoke():
    import json
    v9_cr.set_encoder(StubTok(), EMB)
    pool = v9_cr.CRStatePool()
    for serial, fact in [(1, "Alice was born in Paris."),
                         (2, "Alice was born in Lyon."),
                         (3, "Alice was educated at Oxford.")]:
        pool.upsert(fact, serial)
    qtxt = "Where was Alice born?"
    block = pool.retrieve_multihop_context(qtxt)
    prompt = v9_cr.build_cr_prompt(qtxt, block)
    rec = {"question_idx": 0, "query": qtxt, "output": "Lyon", "cr_dbg": pool.last_dbg}
    dumped = json.dumps(rec)                       # JSONL 写入必须可序列化
    ok("wiring pool", len(pool) == 2)
    ok("wiring block", "Lyon" in block and "Oxford" in block)
    ok("wiring prompt", "FACTS:" in prompt and "Lyon" in prompt and "QUESTION:" in prompt)
    ok("wiring dbg serializable", json.loads(dumped)["cr_dbg"]["beam"] == 6)


# ---------------- 边界 ----------------
def test_empty_and_evict():
    pool = v9_cr.CRStatePool()
    ok("empty retrieve", pool.retrieve_multihop_context("q") == "")
    ok("empty dbg", pool.last_dbg.get("n_pool") == 0)

    small = v9_cr.CRStatePool(pool_max=3)
    for i, f in enumerate(["Bob was born in A.", "Carol was born in B.",
                           "Dave was born in C.", "Eve was born in D."], 1):
        small.upsert(f, i)
    ok("capacity enforced", len(small) <= 3)
    ok("eviction counted", small.n_evicts >= 1)
    ok("retrieval after eviction", small.retrieve_multihop_context("Bob") is not None)


# ---------------- 5. 原位覆写：槽 id / 图键 / 子句索引一致 ----------------
def test_inplace_replace_keeps_graph_keys():
    """P1：覆写不再 drop+append，槽 id 稳定且图键随槽原位更新。"""
    pool = v9_cr.CRStatePool()
    pool.upsert("Alice was born in Paris.", 1)
    pool.upsert("Bob was born in Rome.", 2)
    ok("two slots before overwrite", len(pool) == 2)
    ok("overwrite detected", pool.upsert("Alice was born in Lyon.", 3) is True)
    ok("slot count stable after overwrite", len(pool) == 2)
    ok("slot 0 updated in place", "Lyon" in pool.texts[0] and "Alice" in pool.texts[0])
    ok("slot 1 untouched", pool.texts[1] == "Bob was born in Rome.")
    ok("serial refreshed on same slot", pool.metas[0].get("serial") == 3)
    ok("graph keys follow slot", pool.cr_subj_norm[0] == v9._cr_norm("Alice")
       and pool.cr_obj_norm[0] == v9._cr_norm("Lyon"))
    ok("graph key lists stay aligned",
       len(pool.cr_E_subj) == len(pool) == len(pool.cr_k_obj) == len(pool.cr_n_subj))
    ok("subject index consistent", pool._subj_index.get(v9._cr_norm("Alice")) == [0])
    ok("overwrite counter", pool.n_overwrites == 1)
    block = pool.retrieve_multihop_context("Where was Alice born?")
    ok("retrieval sees new value", "Lyon" in block and "Paris" not in block)


def test_subject_index_survives_eviction():
    """索引唯一会伤精度的方式是"陈旧"：淘汰/覆写后必须指向活槽。

    主语倒排只做候选枚举、不参与打分；但一旦残留死槽，覆写就会打到错位置。
    这里逐条 upsert 检查索引与槽内容一致，并验证淘汰后覆写仍命中同主语槽。
    """
    pool = v9_cr.CRStatePool(pool_max=3)
    for i, t in enumerate(["Alice was born in Paris.", "Bob was born in Rome.",
                           "Carol was born in Lyon.", "Dave was born in Nice.",
                           "Eve was born in Oslo."], 1):
        pool.upsert(t, i)
        bad = [(k, j) for k, v in pool._subj_index.items() for j in v
               if j >= len(pool) or pool.cr_subj[j] != k]
        ok(f"index consistent after upsert {i}", bad == [])
        ok(f"index covers every slot {i}",
           all(pool._subj_index.get(k) for k in set(pool.cr_subj)))
    ok("capacity enforced", len(pool) <= 3)
    ok("evictions counted", pool.n_evicts >= 1)

    subj = pool.texts[0].split()[0]
    n_before = len(pool)
    pool.upsert(f"{subj} was born in Madrid.", 99)
    ok("overwrite after eviction reuses slot", len(pool) == n_before)
    ok("overwrite after eviction wrote new value", any("Madrid" in t for t in pool.texts))
    ok("index still consistent after post-eviction overwrite",
       all(j < len(pool) and pool.cr_subj[j] == k
           for k, v in pool._subj_index.items() for j in v))


# ---------------- 6. Hop1 矩阵初筛（P1 性能） ----------------
def test_hop1_matrix_prefilter():
    """P1：Hop1 先主语均值向量矩阵初筛，再逐 token MaxSim；大池截断、召回不丢。"""
    n = 120
    pool = v9_cr.CRStatePool()
    for i in range(n):
        pool.upsert(f"Entity{i} is affiliated with Org{i}.", i + 1)
    ok("prefilter pool size", len(pool) == n)

    E_q, _k, _ = v9_cr._enc("Entity7")
    cand = pool.hop1_candidates(E_q)
    ok("candidates capped at V9_CR_HOP1_M", len(cand) == v9_cr.CR_HOP1_M < n)
    ok("candidates deterministic", cand == pool.hop1_candidates(E_q))
    ok("gold subject survives prefilter", 7 in cand)
    ok("hop1 top1 is gold", pool.hop1_scores(E_q)[0][1] == 7)

    # 小池不截断：n <= M 时与全量扫描等价
    small = v9_cr.CRStatePool()
    for i in range(5):
        small.upsert(f"Entity{i} is affiliated with Org{i}.", i + 1)
    ok("small pool scans all",
       small.hop1_candidates(v9_cr._enc("Entity3")[0]) == [0, 1, 2, 3, 4])


# ---------------- 7. 路径式读出 / 平铺回退 ----------------
def test_path_view_and_flat_fallback():
    embed("alice", basis(0))
    embed("springfield", basis(1))
    embed("france", basis(3))
    embed("born", basis(6))
    embed("located", basis(7))

    pool = v9_cr.CRStatePool()
    pool.upsert("Alice was born in Springfield.", 1)
    pool.upsert("Springfield is located in France.", 2)
    pool.upsert("Springfield is famous for pizza.", 3)
    pool.upsert("France was born in Europe.", 4)     # 提供第二跳目标

    block = pool.retrieve_multihop_context("Alice was born in?")
    ok("path header rendered", block.startswith("PATH 1"))
    ok("path shows entity chain", "->" in block and "was born in" in block)
    ok("path keeps global serials", re.search(r"^1\. Alice", block, re.M) is not None)

    old = v9_cr.CR_PATH_VIEW
    v9_cr.CR_PATH_VIEW = False
    try:
        flat = pool.retrieve_multihop_context("Alice was born in?")
        ok("flat fallback has no path header", "PATH " not in flat)
        flat_serials = [int(m.group(1)) for m in
                        (re.match(r"\s*(\d+)\.\s", ln) for ln in flat.splitlines()) if m]
        ok("flat block sorted by serial", flat_serials == sorted(flat_serials))
    finally:
        v9_cr.CR_PATH_VIEW = old


if __name__ == "__main__":
    test_state_pool_dual_threshold()
    test_cr_pool_upsert_dual_threshold()
    test_cr_copular_attributes_not_merged()
    test_beam6_ego_graph()
    test_ego_forward_neighbor()
    test_chain_two_fact_structure()
    test_third_hop_does_not_pollute_two_hop()
    test_two_hop_path_ranks_first()
    test_three_fact_chain_independent_branch()
    test_two_hop_chain_joint_score_legacy_shape()
    test_counterfactual_prompt()
    test_runner_wiring_smoke()
    test_empty_and_evict()
    test_inplace_replace_keeps_graph_keys()
    test_subject_index_survives_eviction()
    test_hop1_matrix_prefilter()
    test_path_view_and_flat_fallback()
    print("ALL V9 CR CPU TESTS PASSED")
