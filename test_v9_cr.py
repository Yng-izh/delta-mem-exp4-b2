# test_v9_cr.py — v9_cr / v9_slotpool CR 双阈值 CPU 单测（stub tokenizer，无模型/无 GPU）
#
# 覆盖 t5 四项：
#   1. StatePool._should_merge 双阈值（主语精确匹配 + 关系谓词余弦 > 0.80）
#   2. CRStatePool：Beam=6 ego-graph 第 1 跳
#   3. 双跳链谓词联合打分 Score_chain（谓词匹配的链分数更高、排序更靠前）
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


# ---------------- 3. 双跳链谓词联合打分 ----------------
def test_chain_joint_score():
    embed("alice", basis(0))
    embed("springfield", basis(1))
    embed("france", basis(3))
    embed("pizza", basis(5))
    embed("born", basis(6))
    embed("located", basis(7))
    embed("famous", basis(8))
    embed("created", basis(9))

    pool = v9_cr.CRStatePool()
    pool.upsert("Alice was born in Springfield.", 1)
    pool.upsert("Springfield is located in France.", 2)
    pool.upsert("Springfield is famous for pizza.", 3)
    pool.upsert("France was born in Europe.", 4)
    pool.upsert("Pizza was created in Italy.", 5)
    ok("chain pool size", len(pool) == 5)

    E_q, _k, _ = v9_cr._enc("Alice was born in?")
    E_qrel, _kr, _ = v9_cr._enc("was born in")
    good = pool.chain_score(0, 1, 3, E_q, E_qrel)   # mid=located, tgt=born
    bad = pool.chain_score(0, 2, 4, E_q, E_qrel)    # mid=famous,  tgt=created
    ok("predicate-matching chain scores higher", good > bad + 0.1)
    ok("joint weights default", abs(pool.w_pred - 0.30) < 1e-9)

    block = pool.retrieve_multihop_context("Alice was born in?")
    dbg = pool.last_dbg
    ok("query relation parsed", dbg["query_rel"].lower() == "was born in")
    chains = [c for c in dbg["chains"] if c[2] is not None]
    ok("two-hop chains scored", len(chains) >= 2)
    ok("best two-hop chain is predicate-matching", chains[0][1] == 1 and chains[0][2] == 3)
    ok("chain order descending", chains[0][0] >= chains[-1][0])
    ok("target fact retrieved", "Europe" in block)
    ok("path view on by default", dbg["path_view"] is True and v9_cr.CR_PATH_VIEW is True)
    ok("best path is the predicate-matching chain",
       dbg["paths"][0]["facts"][:2] == [0, 1] and dbg["paths"][0]["target"] == 3)
    serials = [int(m.group(1)) for m in
               (re.match(r"\s*(\d+)\.\s", ln) for ln in block.splitlines()) if m]
    ok("every selected fact appears once", sorted(serials) == [1, 2, 3, 4, 5])


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
    test_beam6_ego_graph()
    test_ego_forward_neighbor()
    test_chain_joint_score()
    test_counterfactual_prompt()
    test_runner_wiring_smoke()
    test_empty_and_evict()
    test_inplace_replace_keeps_graph_keys()
    test_subject_index_survives_eviction()
    test_hop1_matrix_prefilter()
    test_path_view_and_flat_fallback()
    print("ALL V9 CR CPU TESTS PASSED")
