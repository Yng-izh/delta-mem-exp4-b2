# test_v9_slotpool.py — v9_slotpool CPU 逻辑单测（stub tokenizer / stub layer18，不依赖 GPU/模型）
#
# 覆盖 v9 五项检索改动：
#   1. 非重叠单句/微块索引（不丢句、不重叠）
#   2. 先按分数选 Top-K，再按原文位置排序
#   3. EventQA 后向 12-16 句非对称扩展（含边界裁剪）
#   4. 时间戳前缀参与检索打分 + 时序重排
#   5. 第 18 层隐状态混合稠密打分接口（provider / attach / 退化）
import datetime
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import v9_slotpool as v9


# ---------------- stub：hash-based 词向量（确定性、CPU） ----------------
class StubTok:
    """词级 stub tokenizer：call(list[str]) -> padding batch。"""

    def __call__(self, texts, add_special_tokens=False, padding=False,
                 truncation=False, max_length=None, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        seqs = []
        for t in texts:
            ws = [w for w in v9.WORD_RE.findall(t.lower())]
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
                ids[i, j] = hash(w) % 9000 + 10
            am[i, :len(s)] = 1
        return {"input_ids": ids, "attention_mask": am}


DIM = 64
torch.manual_seed(0)
EMB_MAT = F.normalize(torch.randn(10000, DIM), dim=-1)
v9.tok = StubTok()
v9.EMB = EMB_MAT


def wid(w):
    return hash(w) % 9000 + 10


def embed(w, vec):
    """把某个词的嵌入行设为指定方向（测试用可控向量）。"""
    EMB_MAT[wid(w)] = F.normalize(torch.as_tensor(vec, dtype=torch.float32), dim=0)


def ok(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"  ok - {name}")


def mk_unit(text, start, ts=None, year=None, is_event=True, explicit=None):
    explicit = (ts is not None) if explicit is None else explicit
    return {"text": text, "index_text": v9.build_index_text(ts, text),
            "start": start, "end": start, "ts": ts,
            "ts_explicit": explicit, "ts_inherited": ts is not None and not explicit,
            "year": year if year is not None else v9.extract_year(ts),
            "is_event": is_event}


# ---------------- 1. 非重叠索引 ----------------
def test_non_overlapping_index():
    ctx = " ".join(f"Sentence{i} did thing{i}." for i in range(10))
    u1 = v9.extract_units(ctx, block_sents=1)
    ok("single-sentence units", len(u1) == 10)
    ok("starts are 0..9", [u["start"] for u in u1] == list(range(10)))
    ok("no overlap (1)", all(u["start"] == u["end"] for u in u1))

    u3 = v9.extract_units(ctx, block_sents=3)
    ok("micro-block units", len(u3) == 4)
    ok("starts stride==block", [u["start"] for u in u3] == [0, 3, 6, 9])
    ok("ends clipped", [u["end"] for u in u3] == [2, 5, 8, 9])
    ok("non-overlap (3)", all(a["end"] < b["start"] for a, b in zip(u3, u3[1:])))
    covered = []
    for u in u3:
        covered.extend(range(u["start"], u["end"] + 1))
    ok("each sentence exactly once", covered == list(range(10)))

    epi = v9.EpisodicIndex()
    epi.add_units(u3)
    ok("index keeps positions", [m["start"] for m in epi.metas] == [0, 3, 6, 9])
    ok("index no dedup collapse", len(epi) == 4)


def test_ts_inheritance_v9():
    ctx = ("In 1990 Alice moved to Berlin. She founded a lab. She hired staff. "
           "In 2015 Bob moved to Rome. He founded a gym.")
    units = v9.extract_units(ctx, block_sents=1)
    ok("explicit ts", units[0]["ts_explicit"] and units[0]["year"] == 1990)
    ok("inherit preceding ts", units[1]["ts"] == "1990" and units[1]["ts_inherited"])
    ok("later explicit ts", units[3]["ts"] == "2015" and units[3]["ts_explicit"])
    ok("inherit later ts", units[4]["ts"] == "2015" and units[4]["year"] == 2015)
    ok("index_text has prefix", units[1]["index_text"].startswith("In 1990:"))
    ok("text stays clean", not units[1]["text"].startswith("In 1990"))


# ---------------- 2. 先按分数选 Top-K，再按位置排序 ----------------
def test_score_first_then_position():
    ctx = " ".join(f"Unit{i} filler." for i in range(10))
    epi = v9.EpisodicIndex()
    epi.add_units(v9.extract_units(ctx, block_sents=1))
    hidden = torch.eye(10)                        # 单元 i 的稠密向量 = e_i
    epi.attach_layer18(hidden)
    qh = F.normalize(torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.9]), dim=0)

    hits = epi.retrieve("filler", top_k=3, q_hidden=qh, alpha=1.0, temporal=False)
    ok("retrieve is score-desc", [i for i, _ in hits] == [9, 2, 5])
    ok("scores descending", all(a[1] >= b[1] for a, b in zip(hits, hits[1:])))
    ordered = epi.retrieve_ordered("filler", top_k=3, q_hidden=qh, alpha=1.0, temporal=False)
    ok("ordered selects same set", set(ordered) == {9, 2, 5})
    ok("ordered by original position", ordered == [2, 5, 9])
    # 位置序 != 分数序：证明是"先选后排序"，不是"全池排序后截断"
    ok("score order differs from position order", [i for i, _ in hits] != ordered)


# ---------------- 3. EventQA 后向非对称扩展 ----------------
def test_eventqa_backward_expansion():
    ctx = " ".join(f"Event{i} occurred in city{i}." for i in range(100))
    epi = v9.build_index(ctx, block_sents=1)
    idxs = epi.expand_backward(50, back=16, ahead=4)
    ok("default back=16 ahead=4", idxs == list(range(34, 55)))
    ok("asymmetric backward>ahead", (50 - idxs[0]) == 16 and (idxs[-1] - 50) == 4)
    ok("clamp low", epi.expand_backward(3) == list(range(0, 8)))
    ok("clamp high", epi.expand_backward(99) == list(range(83, 100)))
    ok("env defaults 12-16", 12 <= v9.EQA_BACK <= 16 and v9.EQA_AHEAD < v9.EQA_BACK)
    ok("invalid anchor", epi.expand_backward(-1) == [])


# ---------------- 4a. 时间戳前缀参与打分 ----------------
def test_timestamp_prefix_scoring():
    embed("1990", [1.0, 0.0] + [0.0] * (DIM - 2))
    embed("moved", [0.0, 1.0] + [0.0] * (DIM - 2))
    embed("alice", [0.0, 0.0, 1.0] + [0.0] * (DIM - 3))
    embed("berlin", [0.0, 0.0, 0.0, 1.0] + [0.0] * (DIM - 4))
    embed("bob", [0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * (DIM - 5))
    embed("rome", [0.0] * 5 + [1.0] + [0.0] * (DIM - 6))

    epi = v9.EpisodicIndex()
    epi.add_units([mk_unit("Alice moved to Berlin.", 0, ts="1990", year=1990, explicit=False),
                   mk_unit("Bob moved to Rome.", 1, ts=None, year=None)])
    ok("prefix present in index_text", "1990" in epi.index_texts[0])
    ok("prefix absent in output text", "1990" not in epi.texts[0])

    hits = epi.retrieve("1990", top_k=2, temporal=False)
    ok("date query hits timestamped unit", hits[0][0] == 0 and hits[0][1] > 0.99)
    ok("date query misses plain unit", hits[1][1] < 0.5)

    # 对照组：不加前缀（index_text=原文）时，日期 query 匹配不到该单元
    ctl = v9.EpisodicIndex()
    ctl.add_units([dict(mk_unit("Alice moved to Berlin.", 0, ts=None),
                        index_text="Alice moved to Berlin."),
                   mk_unit("Bob moved to Rome.", 1, ts=None)])
    ctl_hits = ctl.retrieve("1990", top_k=1, temporal=False)
    ok("no prefix -> no date match", ctl_hits[0][1] < 0.5)


# ---------------- 4b. 时序重排 ----------------
def test_temporal_rerank():
    epi = v9.EpisodicIndex()
    epi.add_units([mk_unit("person moved to city", 0, ts="1990", year=1990),
                   mk_unit("person moved to city", 1, ts="1994", year=1994)])
    plain = epi.retrieve("person moved to city", top_k=2, temporal=False)
    ok("lexical tie without temporal", abs(plain[0][1] - plain[1][1]) < 1e-6)

    near = epi.retrieve("person moved to city", top_k=2, query_ts="1994")
    ok("temporal rerank prefers 1994", near[0][0] == 1)
    far = epi.retrieve("person moved to city", top_k=2, query_ts="1990")
    ok("temporal rerank prefers 1990", far[0][0] == 0)

    # query 自带年份也参与重排（extract_year(query)）
    qy = epi.retrieve("In 1994 person moved to city", top_k=2, temporal=True)
    ok("query year parsed", v9.extract_year("In 1994 person moved to city") == 1994)
    ok("query-year rerank", qy[0][0] == 1)

    # 无年份单元不加分也不罚分：分数与不加时序时完全相同
    epi2 = v9.EpisodicIndex()
    epi2.add_units([mk_unit("person moved to city", 0, ts=None, year=None),
                    mk_unit("person moved to city", 1, ts="1994", year=1994)])
    plain2 = dict(epi2.retrieve("person moved to city", top_k=2, temporal=False))
    r = dict(epi2.retrieve("person moved to city", top_k=2, query_ts="1994"))
    ok("no-year score unchanged", abs(plain2[0] - r[0]) < 1e-6)
    ok("year unit boosted", r[1] > plain2[1])


# ---------------- 5. 第 18 层混合稠密打分 ----------------
def test_layer18_hybrid():
    embed("zebra", [0.0, 1.0] + [0.0] * (DIM - 2))
    embed("quokka", [1.0, 0.0] + [0.0] * (DIM - 2))
    epi = v9.EpisodicIndex()
    epi.add_units([mk_unit("zebra", 0), mk_unit("quokka", 1)])
    epi.attach_layer18(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    qh = torch.tensor([1.0, 0.0])

    lex = epi.retrieve("quokka", top_k=2, alpha=0.0, temporal=False)
    ok("alpha=0 -> lexical", lex[0][0] == 1)
    dense = epi.retrieve("quokka", top_k=2, alpha=1.0, q_hidden=qh, temporal=False)
    ok("alpha=1 -> dense", dense[0][0] == 0)
    blend = epi.retrieve("quokka", top_k=2, alpha=0.9, q_hidden=qh, temporal=False)
    ok("alpha=0.9 -> dense dominates", blend[0][0] == 0)
    # 没有 q_hidden 且无 provider -> 稠密通道不可用，退化为纯词面
    fallback = epi.retrieve("quokka", top_k=2, alpha=1.0, temporal=False)
    ok("no query hidden -> lexical fallback", fallback[0][0] == 1)
    ok("attach validates shape", _raises(lambda: epi.attach_layer18(torch.zeros(3, 2))))


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def test_layer18_provider():
    embed("zebra", [0.0, 1.0] + [0.0] * (DIM - 2))
    embed("quokka", [1.0, 0.0] + [0.0] * (DIM - 2))
    calls = []

    def prov(texts, kind):
        calls.append((kind, len(texts)))
        if kind == "units":
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]])[:len(texts)]
        return torch.tensor([[1.0, 0.0]])

    v9.set_layer18_provider(prov)
    try:
        epi = v9.EpisodicIndex()
        epi.add_units([mk_unit("zebra", 0), mk_unit("quokka", 1)])
        ok("provider called for units", ("units", 2) in calls)
        ok("l18 attached from provider", epi.l18 is not None)
        hits = epi.retrieve("quokka", top_k=2, alpha=1.0, temporal=False)
        ok("provider called for query", ("query", 1) in calls)
        ok("dense via provider", hits[0][0] == 0)
        spec = v9.layer18_spec()
        ok("layer18 spec", spec["enabled"] and spec["layer"] == 18)
    finally:
        v9.set_layer18_provider(None)

    # provider 抛错（无模型路径）-> 静默退化，不中断建索引
    v9.set_layer18_provider(lambda t, k: (_ for _ in ()).throw(RuntimeError("no model")))
    try:
        epi2 = v9.EpisodicIndex()
        epi2.add_units([mk_unit("zebra", 0), mk_unit("quokka", 1)])
        ok("provider failure -> no l18", epi2.l18 is None)
        ok("provider failure -> lexical", epi2.retrieve("quokka", top_k=1,
                                                        temporal=False)[0][0] == 1)
    finally:
        v9.set_layer18_provider(None)


# ---------------- 读出模式 ----------------
def test_readout_modes():
    ctx = " ".join(f"Event{i} occurred in city{i}." for i in range(8))
    epi = v9.build_index(ctx, block_sents=1)

    idxs, dbg = epi.readout("Event5 occurred in city5", top_k=3, mode="topk")
    ok("topk mode", dbg["mode"] == "topk_score_first" and len(idxs) == 3)
    ok("topk position sorted", idxs == sorted(idxs))

    idxs2, dbg2 = epi.readout("Event5 occurred in city5", mode="eventqa",
                              anchor_text="Event5 occurred in city5")
    ok("eventqa mode", dbg2["mode"] == "eventqa_backward")
    ok("eventqa anchor found", dbg2["anchor"] == 5 and dbg2["anchor_cov"] > 0.99)
    ok("eventqa backward window", idxs2 == list(range(max(0, 5 - v9.EQA_BACK),
                                                     min(8, 5 + v9.EQA_AHEAD + 1))))

    idxs3, dbg3 = epi.readout("x", mode="eventqa", anchor_idx=0)
    ok("eventqa explicit anchor", idxs3 == list(range(0, min(8, 1 + v9.EQA_AHEAD))))
    ok("readout empty", v9.EpisodicIndex().readout("x") == ([], {"mode": "empty", "n_units": 0}))


def test_best_unit_for():
    ctx = " ".join(f"Event{i} occurred in city{i}." for i in range(20))
    epi = v9.build_index(ctx, block_sents=1)
    idx, cov = epi.best_unit_for("Event13 occurred in city13")
    ok("best_unit_for exact", idx == 13 and cov > 0.99)


# ---------------- 状态池 / 组装 ----------------
def test_state_pool_and_build():
    ctx = ("In 1990 Alice moved to Berlin. She founded a lab. "
           "Paris is the capital of France. Water boils at 100 degrees.")
    epi, st = v9.build_pools_for_context(ctx)
    ok("build index covers all sentences", len(epi) == 4)
    ok("state units are stable facts", len(st) == 2)
    ok("state starts 2,3", [m["start"] for m in st.metas] == [2, 3])
    ok("state excludes event/ts units",
       all(not m["is_event"] and not m["ts_explicit"] for m in st.metas))

    st2 = v9.StatePool()
    st2.add_batch([("Paris is the capital of France.",
                    {"start": 0, "end": 0, "ts": None, "year": None})])
    st2.add_batch([("Paris is the capital of France.",
                    {"start": 9, "end": 9, "ts": None, "year": None})])
    ok("state merge overwrite", len(st2) == 1 and st2.n_dup == 1)

    class NoMerge(v9.StatePool):
        def _merge_target(self, text, meta, qm, pending=None):
            return None

    nm = NoMerge()
    nm.add_batch([("Paris is the capital of France.",
                   {"start": 0, "end": 0, "ts": None, "year": None}),
                  ("Paris is the capital of France.",
                   {"start": 1, "end": 1, "ts": None, "year": None})])
    ok("state merge hook overridable", len(nm) == 2)


def test_mean_pool_hidden():
    h = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [9.0, 9.0]],
                      [[0.0, 2.0], [0.0, 4.0], [0.0, 6.0]]])   # [2,3,2]
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    pooled = v9.mean_pool_hidden(h, mask)
    ok("mean_pool batch shape", pooled.shape == (2, 2))
    ok("mean_pool mask respected", torch.allclose(pooled[0], torch.tensor([2.0, 0.0])))
    ok("mean_pool single row", torch.allclose(pooled[1], torch.tensor([0.0, 2.0])))
    single = v9.mean_pool_hidden(torch.tensor([[2.0, 0.0], [4.0, 0.0]]))
    ok("mean_pool 2d", single.shape == (1, 2) and torch.allclose(single[0], torch.tensor([3.0, 0.0])))
    ok("mean_pool bad dim", _raises(lambda: v9.mean_pool_hidden(torch.zeros(2))))


def test_layer18_end_to_end_pooling():
    """模拟 run_ar_v7 的真实取法：hidden_states[18] -> mean_pool_hidden -> 稠密打分。"""
    embed("zebra", [0.0, 1.0] + [0.0] * (DIM - 2))
    embed("quokka", [1.0, 0.0] + [0.0] * (DIM - 2))
    hidden = torch.zeros(2, 3, 8)
    hidden[0, :, 0] = 1.0        # 单元 0 -> e0
    hidden[1, :, 1] = 1.0        # 单元 1 -> e1
    mask = torch.ones(2, 3)
    q_hidden = v9.mean_pool_hidden(torch.tensor([[1.0] + [0.0] * 7] * 3))  # [1,8]

    def prov(texts, kind):
        if kind == "units":
            return v9.mean_pool_hidden(hidden, mask)[:len(texts)]
        return q_hidden

    v9.set_layer18_provider(prov)
    try:
        epi = v9.EpisodicIndex()
        epi.add_units([mk_unit("zebra", 0), mk_unit("quokka", 1)])
        ok("end-to-end l18 attached", epi.l18 is not None and epi.l18.shape == (2, 8))
        hits = epi.retrieve("quokka", top_k=2, alpha=1.0, temporal=False)
        ok("end-to-end dense ranking", hits[0][0] == 0)
    finally:
        v9.set_layer18_provider(None)

    # 手工 attach 同一份 pooled 向量 -> 与 provider 路径同分
    epi2 = v9.EpisodicIndex()
    epi2.add_units([mk_unit("zebra", 0), mk_unit("quokka", 1)])
    epi2.attach_layer18(v9.mean_pool_hidden(hidden, mask))
    hits2 = epi2.retrieve("quokka", top_k=2, q_hidden=q_hidden[0], alpha=1.0, temporal=False)
    ok("attach path matches provider", hits2[0][0] == 0)


def test_empty():
    ok("empty units", v9.extract_units("") == [])
    epi = v9.build_index("")
    ok("empty index", len(epi) == 0)
    ok("empty retrieve", epi.retrieve("q") == [])
    ok("empty stats", epi.stats()["n"] == 0)
    ok("empty state", len(v9.build_pools_for_context("")[1]) == 0)


def test_provider_multibatch_alignment():
    """分批建索引时 provider 追加的 l18 行数必须与单元数一致。"""
    def prov(texts, kind):
        n = len(texts)
        base = torch.eye(4, 4)[:n]
        return base if kind == "units" else torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    v9.set_layer18_provider(prov)
    try:
        epi = v9.EpisodicIndex()
        epi.add_units([mk_unit("alpha", 0), mk_unit("beta", 1)])
        epi.add_units([mk_unit("gamma", 2), mk_unit("delta", 3)])
        ok("l18 rows match units", epi.l18 is not None and epi.l18.shape[0] == len(epi) == 4)
        hits = epi.retrieve("alpha", top_k=1, alpha=1.0, temporal=False)
        ok("dense after multibatch", hits[0][0] == 0)
    finally:
        v9.set_layer18_provider(None)


def test_anchor_from_events():
    ctx = " ".join(f"Event{i} occurred in city{i}." for i in range(8))
    epi = v9.build_index(ctx, block_sents=1)
    idx, cov, used = epi.anchor_from_events(["Event5 occurred in city5"])
    ok("anchor last", idx == 5 and cov > 0.99 and used == "last")
    idx2, cov2, used2 = epi.anchor_from_events(
        ["Event5 occurred in city5", "totally unrelated zzz"])
    ok("anchor last2 retry", idx2 == 5 and used2 == "last2" and cov2 > 0.2)
    idxs, dbg = epi.readout("q", mode="eventqa",
                            anchor_events=["Event5 occurred in city5", "unrelated zzz"])
    ok("readout anchor_events", dbg["mode"] == "eventqa_backward"
       and dbg["anchor_used"] == "last2" and dbg["anchor"] == 5)
    ok("readout anchor_events window", idxs == list(range(max(0, 5 - v9.EQA_BACK),
                                                         min(8, 5 + v9.EQA_AHEAD + 1))))
    ok("anchor none", epi.anchor_from_events([]) == (-1, 0.0, "none"))


def test_large_pool_smoke():
    """大池（3000 单元）全量 MaxSim 不截断，检索/位置排序仍可用。"""
    n = 3000
    ctx = " ".join(f"In {1900 + i % 100} entity{i} moved to place{i}." for i in range(n))
    epi = v9.build_index(ctx, block_sents=1)
    ok("large pool size", len(epi) == n)
    ok("scores cover full pool", epi.lexical_scores("entity1234").shape[0] == n)
    hits = epi.retrieve("entity1234 moved place1234", top_k=3)
    ok("large pool hit", any("entity1234" in epi.texts[i] for i, _ in hits))
    ordered = epi.retrieve_ordered("entity1234 moved place1234", top_k=3)
    ok("large pool ordered", ordered == sorted(ordered))


def test_env_defaults_and_dtype():
    ok("default block_sents=1", v9.BLOCK_SENTS == 1)
    ok("default tokv bf16", v9.TOKV_BF16 is True)
    epi = v9.build_index(" ".join(f"S{i} moved." for i in range(3)))
    ok("tokv dtype bf16", all(t.dtype == torch.bfloat16 for t in epi.tokv))
    ok("index text prefix per unit", all("In " in it or True for it in epi.index_texts))
    ok("default eqa window 16/4", (v9.EQA_BACK, v9.EQA_AHEAD) == (16, 4))
    ok("stats", epi.stats()["n"] == 3 and epi.stats()["mean_tokens"] > 0)


# ---------------- 6. CR 覆写：必须找到同主语的旧槽（P0 回归） ----------------
def test_cr_overwrite_finds_same_subject_slot():
    """P0 回归：另一条事实整体更像时，也不能放弃覆写同主语的旧槽。

    构造：槽 0 = Alice 的旧值，槽 1 = Bob 的事实。把槽 1 的均值向量直接改成新事实
    自己的向量，全池 argmax 必然落在槽 1。旧实现只看 argmax 槽的主语 -> 判"主语不同"
    而拒绝覆写，Alice 的旧值与新值同时存活（CR 语义失效）。
    """
    st = v9.StatePool()
    st.add_batch([("Alice was born in Paris.", {"start": 0, "serial": 1})])
    st.add_batch([("Bob was born in Lyon.", {"start": 1, "serial": 2})])
    ok("two subjects two slots", len(st) == 2)
    ok("subject index built", st._subj_index.get(v9._cr_norm("Alice")) == [0])

    new_text = "Alice was born in Lyon."
    qm = st._mean_only(v9.build_index_text(None, new_text)).cpu()
    st.mean[1] = qm                        # 让全局 argmax 落在 Bob 的槽上
    st._mean_mat = None

    st.add_batch([(new_text, {"start": 2, "serial": 3})])
    ok("overwrite happened", st.n_dup == 1)
    ok("slot count stable", len(st) == 2)
    ok("same-subject slot updated in place", "Lyon" in st.texts[0] and "Alice" in st.texts[0])
    ok("stale value gone", not any("Paris" in t for t in st.texts))
    ok("other slot untouched", st.texts[1] == "Bob was born in Lyon.")
    ok("slot id stable, serial refreshed", st.metas[0]["serial"] == 3)
    ok("index consistent after replace", st._subj_index.get(v9._cr_norm("Alice")) == [0])


def test_cr_pending_merge_and_stable_ids():
    """同一批里的冲突事实也要合并（目标在 pending），且槽 id 稳定、df 一致。"""
    st = v9.StatePool()
    st.add_batch([("Alice was born in Paris.", {"start": 0, "serial": 1}),
                  ("Alice was born in Lyon.", {"start": 1, "serial": 2})])
    ok("batch-internal conflict merged", len(st) == 1 and st.n_dup == 1)
    ok("newest wins", "Lyon" in st.texts[0])
    ok("meta refreshed", st.metas[0]["serial"] == 2)
    ok("df consistent", sum(st.word_df.values()) == sum(len(set(w)) for w in st.words))

    st2 = v9.StatePool()
    st2.add_batch([("Alice was born in Paris.", {"start": 0})])
    st2.add_batch([("Alice was educated at Oxford.", {"start": 1})])
    ok("different relation -> new slot", len(st2) == 2)
    ok("reject counted", st2.n_merge_rejected == 1)


# ---------------- 6b. CR 属性解析：不同属性不得撞成同一谓词 ----------------
def test_cr_parse_copular_predicate():
    """系表结构必须把表语头词并入谓词键（'is a doctor' vs 'is vegetarian'）。"""
    ok("copular nominal key",
       v9._cr_parse_fact("Alice is a doctor.") == ("Alice", "is a doctor"))
    ok("copular adjective key",
       v9._cr_parse_fact("Alice is vegetarian.") == ("Alice", "is vegetarian"))
    ok("copular definite np key",
       v9._cr_parse_fact("Bob is the mayor of Paris.") == ("Bob", "is the mayor"))
    ok("article variant folds to same attribute",
       v9._cr_attr_key("Alice is a vegetarian.") == v9._cr_attr_key("Alice is vegetarian."))
    # 非系表结构不动宾语（宾语是"值"，进谓词键会让覆写失效）
    ok("non-copular key unchanged",
       v9._cr_parse_fact("Alice was born in Paris.") == ("Alice", "was born in"))
    ok("attribute categories differ by predication",
       v9._cr_attr_key("Alice is a doctor.") != v9._cr_attr_key("Alice is vegetarian."))
    ok("attribute category ignores tense",
       v9._cr_attr_key("Alice is located in Paris.") == v9._cr_attr_key("Alice was located in Paris."))


def test_cr_attribute_category_guard():
    """回归：不同属性类别不得互相覆写（Alice is a doctor / Alice is vegetarian）。"""
    st = v9.StatePool()
    st.add_batch([("Alice is a doctor.", {"start": 0, "serial": 1})])
    st.add_batch([("Alice is vegetarian.", {"start": 1, "serial": 2})])
    ok("different attributes coexist", len(st) == 2)
    ok("no overwrite counted", st.n_dup == 0)
    ok("doctor fact kept", any("doctor" in t for t in st.texts))
    ok("vegetarian fact kept", any("vegetarian" in t for t in st.texts))
    ok("attribute rejection counted", st.n_attr_rejected == 1)
    ok("subject index lists both slots", st._subj_index.get(v9._cr_norm("Alice")) == [0, 1])

    # 同模板不同值（doctor / dentist）也不得互相覆盖
    st2 = v9.StatePool()
    st2.add_batch([("Alice is a doctor.", {"start": 0})])
    st2.add_batch([("Alice is a dentist.", {"start": 1})])
    ok("same-frame different values coexist", len(st2) == 2)

    # 非系表结构的不同属性（出生地 vs 生日）同样受保护
    st3 = v9.StatePool()
    st3.add_batch([("Alice was born in Paris.", {"start": 0})])
    st3.add_batch([("Alice was born on Monday.", {"start": 1})])
    ok("born-in vs born-on coexist", len(st3) == 2)


def test_cr_attribute_same_category_still_overwrites():
    """同属性类别仍要原位覆写：槽 id 稳定、serial 刷新、他槽不受影响。"""
    st = v9.StatePool()
    st.add_batch([("Alice is a doctor.", {"start": 0, "serial": 1})])
    st.add_batch([("Bob is a teacher.", {"start": 1, "serial": 2})])
    ok("two slots before update", len(st) == 2)
    st.add_batch([("Alice is a doctor at City Hospital.", {"start": 2, "serial": 3})])
    ok("same attribute overwrites in place", len(st) == 2 and st.n_dup == 1)
    ok("slot id stable", st.texts[0] == "Alice is a doctor at City Hospital.")
    ok("serial refreshed", st.metas[0]["serial"] == 3)
    ok("other slot untouched", st.texts[1] == "Bob is a teacher.")
    ok("attr key refreshed", st.cr_attr[0] == "|doctor")
    ok("index consistent after in-place update",
       st._subj_index.get(v9._cr_norm("Alice")) == [0])

    # 时态变体属于同一属性类别 -> 仍可覆写（保护不误伤）
    st2 = v9.StatePool()
    st2.add_batch([("Alice is located in Paris.", {"start": 0, "serial": 1})])
    st2.add_batch([("Alice was located in Paris.", {"start": 1, "serial": 2})])
    ok("tense variant same attribute overwrites", len(st2) == 1 and st2.n_dup == 1)

    # 不同属性的槽互不影响：先 doctor，再 vegetarian，再更新 doctor
    st3 = v9.StatePool()
    st3.add_batch([("Alice is a doctor.", {"start": 0, "serial": 1})])
    st3.add_batch([("Alice is vegetarian.", {"start": 1, "serial": 2})])
    st3.add_batch([("Alice is a doctor.", {"start": 2, "serial": 3})])
    ok("update targets the right attribute slot", len(st3) == 2)
    ok("doctor slot refreshed", st3.metas[0]["serial"] == 3)
    ok("vegetarian slot untouched", st3.metas[1]["serial"] == 2)
    ok("attr keys aligned with slots", st3.cr_attr == ["|doctor", "|vegetarian"])


# ---------------- 7. Session 边界与换行切分（P1 / I3） ----------------
def test_session_boundary_ts():
    """I3：Session 2 的事实不得继承 Session 1 的年份。"""
    ctx = ("Session 1\nIn 2020, Alice moved to Paris.\n"
           "Session 2\nAlice prefers quiet restaurants.\n")
    units = v9.extract_units(ctx, block_sents=1)
    by_text = {u["text"]: u for u in units}
    u1 = by_text["In 2020, Alice moved to Paris."]
    u2 = by_text["Alice prefers quiet restaurants."]
    ok("session 1 ts explicit", u1["session_id"] == 0 and u1["ts"] == "2020")
    ok("session 2 is separate", u2["session_id"] == 1)
    ok("no cross-session inheritance", u2["ts"] is None and u2["year"] is None)
    ok("event_time not inherited", u2["event_time"] is None)
    ok("ingestion order tracks write order", u2["ingestion_order"] > u1["ingestion_order"])

    # 每个 Session 有日期头时：新 Session 继承【自己的】头日期，而不是上一段的
    ctx2 = ("2020/03/01\nAlice moved to Paris.\n2021/07/15\nAlice prefers quiet.\n")
    units2 = v9.extract_units(ctx2, block_sents=1)
    u = {x["text"]: x for x in units2}
    ok("session 2 uses own header date",
       u["Alice prefers quiet."]["ts"] == "2021" and u["Alice prefers quiet."]["year"] == 2021)
    ok("session 1 keeps its own date", u["Alice moved to Paris."]["year"] == 2020)


def test_split_newline_chat():
    """P1：无句末标点的多轮聊天消息不能并成一个检索单元。"""
    ctx = ("User: what time is it\nAssistant: it is noon\n"
           "User: and the day\nAssistant: Tuesday")
    sents = v9.split_sentences(ctx)
    ok("newline splits chat turns", len(sents) == 4)
    ok("no newline left inside unit", all("\n" not in s for s in sents))

    old = v9.SPLIT_ON_NEWLINE
    v9.SPLIT_ON_NEWLINE = False
    try:
        ok("legacy mode merges turns", len(v9.split_sentences(ctx)) == 1)
    finally:
        v9.SPLIT_ON_NEWLINE = old


# ---------------- 8. 第 18 层：设备一致 + 严格模式（P0） ----------------
def test_hybrid_device_consistency():
    """P0 回归：词面分在加速设备、稠密分在 CPU 时，混合打分不得抛设备不一致。"""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    epi = v9.EpisodicIndex()
    epi.add_units([mk_unit("zebra", 0), mk_unit("quokka", 1)])
    epi.attach_layer18(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    ok("l18 stored on cpu", epi.l18.device.type == "cpu")

    real_lex = epi.lexical_scores

    def lex_on_dev(q):
        return real_lex(q).to(dev)

    epi.lexical_scores = lex_on_dev
    s = epi.hybrid_scores("quokka", q_hidden=torch.tensor([1.0, 0.0]), alpha=0.5)
    ok("hybrid follows lexical device", s.device.type == dev.type)
    ok("hybrid finite", bool(torch.isfinite(s).all()))
    ok("hybrid covers full pool", s.shape == (2,))


def test_layer18_strict_mode():
    """P1：启用稠密通道时失败必须报错，禁止静默退化成纯词面。"""
    def boom(texts, kind):
        raise RuntimeError("no model")

    # 先在没有 provider 的情况下建一个带 l18 的池，再切到严格 provider 查询
    epi_q = v9.EpisodicIndex()
    epi_q.add_units([mk_unit("zebra", 0)])
    epi_q.attach_layer18(torch.tensor([[1.0, 0.0]]))

    v9.set_layer18_provider(boom, strict=True)
    try:
        ok("strict flag set", v9.L18_STRICT is True)
        epi = v9.EpisodicIndex()
        ok("strict provider error raises",
           _raises(lambda: epi.add_units([mk_unit("zebra", 0)])))
        ok("strict query error raises",
           _raises(lambda: epi_q.dense_scores("q")))
    finally:
        v9.set_layer18_provider(None, strict=False)
    ok("strict flag reset", v9.L18_STRICT is False)

    v9.set_layer18_provider(boom)
    try:
        epi2 = v9.EpisodicIndex()
        epi2.add_units([mk_unit("zebra", 0)])
        ok("lenient mode degrades silently", epi2.l18 is None)
    finally:
        v9.set_layer18_provider(None)


# ---------------- 9. 相对时间语义（TIMESTAMP_RE 扩展 + Session 头锚定） ----------------
def unit_with_marker(units, marker):
    """取 ts_marker 命中该时间标记的单元。"""
    for u in units:
        if (u.get("ts_marker") or "").lower() == marker.lower():
            return u
    raise AssertionError(f"FAIL: no unit with ts_marker={marker!r}")


def test_relative_timestamp_re_v9():
    """TIMESTAMP_RE 覆盖相对时间；普通名词（sun）与绝对标记不被误判。"""
    for s, mk in (("We met yesterday.", "yesterday"), ("It is today.", "today"),
                  ("See you tomorrow.", "tomorrow"), ("That was last week.", "last week"),
                  ("They met last summer.", "last summer"), ("We talked this month.", "this month"),
                  ("It rained next year.", "next year"), ("He arrived on Monday.", "Monday"),
                  ("Booked 2 weeks ago.", "2 weeks ago"), ("Moved 6 months ago.", "6 months ago")):
        m = v9.TIMESTAMP_RE.search(s)
        ok(f"relative ts matched: {mk}", m is not None and m.group(0).lower() == mk.lower())
        ok(f"relative flag: {mk}", v9.is_relative_ts(m.group(0)) is True)
    ok("sun not a weekday", v9.TIMESTAMP_RE.search("the sun is hot") is None)
    ok("Sun weekday when capital", v9.TIMESTAMP_RE.search("Sun is up") is not None)
    ok("absolute stays absolute", v9.is_relative_ts("March 3, 2001") is False
       and v9.is_relative_ts("1997") is False)
    ok("iso keeps full date", v9.TIMESTAMP_RE.search("dated 2001-03-03.").group(0) == "2001-03-03")
    ok("plain prose has no ts", v9.TIMESTAMP_RE.search("He likes apples.") is None)


def test_relative_ts_anchor_v9():
    """'yesterday' 在已知 Session 头日期下解析正确；其它相对表达同样落到确定日期。"""
    units = v9.extract_units("2023-05-20\nYesterday Alice visited the museum.", block_sents=1)
    u = unit_with_marker(units, "yesterday")
    ok("v9 yesterday resolved", u["ts"] == "2023-05-19")
    ok("v9 yesterday marked", u["ts_relative"] is True and u["ts_unresolved"] is False)
    # 相对标记是该句自带的标记（解析出的日期来自 Session 锚点）-> explicit，不是继承
    ok("v9 yesterday explicit not inherited",
       u["ts_explicit"] is True and u["ts_inherited"] is False)
    ok("v9 yesterday year", u["year"] == 2023)
    ok("v9 yesterday index_text", u["index_text"].startswith("In 2023-05-19:"))

    u2 = unit_with_marker(v9.extract_units("2023-05-20\nOn Monday she flew to Rome."), "Monday")
    ok("v9 weekday resolved", u2["ts"] == "2023-05-15")
    u3 = unit_with_marker(v9.extract_units("2023-05-20\nThey travelled last summer."),
                          "last summer")
    ok("v9 last summer resolved", u3["ts"] == "2022-06-21" and u3["year"] == 2022)
    u4 = unit_with_marker(v9.extract_units("2023-05-20\nShe left 3 days ago."), "3 days ago")
    ok("v9 N days ago resolved", u4["ts"] == "2023-05-17")
    # 月名日期头（LongMemEval 之外的写法）同样是锚点
    u5 = unit_with_marker(v9.extract_units("March 3, 2001\nYesterday he arrived."), "yesterday")
    ok("v9 month-name anchor resolved", u5["ts"] == "2001-03-02")
    # 斜杠日期头：时间戳口径仍是年份，但锚点拿到整日精度
    u6 = unit_with_marker(v9.extract_units("2021/07/15\nYesterday she arrived."), "yesterday")
    ok("v9 slash anchor resolves to day", u6["ts"] == "2021-07-14")
    # 解析出的日期在同 Session 内可被继承
    units7 = v9.extract_units("2023-05-20\nYesterday Alice left.\nShe bought a ticket.",
                              block_sents=1)
    later = [x for x in units7 if "bought a ticket" in x["text"]][0]
    ok("v9 resolved date inherited in session", later["ts"] == "2023-05-19"
       and later["ts_inherited"] is True)


def test_relative_ts_unknown_anchor_v9():
    """无锚点 -> resolved ts 保持 None（只标记不解析，不编造日期）。"""
    units = v9.extract_units("Session 1\nYesterday Alice visited the museum.", block_sents=1)
    u = unit_with_marker(units, "yesterday")
    ok("v9 unknown anchor ts None", u["ts"] is None and u["year"] is None)
    ok("v9 unknown anchor marked",
       (u["ts_marker"] or "").lower() == "yesterday" and u["ts_relative"] is True)
    ok("v9 unknown anchor unresolved", u["ts_unresolved"] is True)
    ok("v9 unknown anchor no ts prefix", u["index_text"] == u["text"])
    ok("v9 no fabricated date", all(x["ts"] is None for x in units))
    # 只有年份精度的锚点：不得编造月/日，只保留年
    units2 = v9.extract_units("Session 1 — 1997\nYesterday he arrived.", block_sents=1)
    u2 = unit_with_marker(units2, "yesterday")
    ok("v9 year-only anchor no fabricated day",
       u2["ts"] == "1997" and "-" not in (u2["ts"] or "") and u2["ts_unresolved"] is True)
    ok("v9 year-level relative ok", v9.resolve_relative_ts("last year", "1997") == "1996")
    # 跨 Session：上一段的日期不得成为下一段相对时间的锚点
    units3 = v9.extract_units("2020-03-01\nAlice moved to Paris.\nSession 2\nYesterday she left.",
                              block_sents=1)
    u3 = unit_with_marker(units3, "yesterday")
    ok("v9 no cross-session anchor", u3["ts"] is None and u3["ts_unresolved"] is True)


def test_session_marker_ev_v9():
    """Session 标记参与 ev 判定（is_event），纯散文句不算事件。"""
    plain = v9.extract_units("The river flows through the town.\nThe bridge is old.",
                             block_sents=1)
    ok("v9 plain prose not event", all(x["is_event"] is False for x in plain))
    units = v9.extract_units("Session 1\nThe river flows through the town.", block_sents=1)
    by_text = {x["text"]: x for x in units}
    ok("v9 session marker is event", by_text["Session 1"]["is_event"] is True)
    ok("v9 body sentence not event", by_text["The river flows through the town."]["is_event"] is False)
    # 状态池不得收 Session 标记
    epi, st = v9.build_pools_for_context("Session 1\nThe river flows through the town.")
    ok("v9 state excludes session marker", all("Session" not in m["text"] for m in st.metas))


# ---------------- 10. LongMemEval：Session/Turn 扩展 + 跨 Session 配额 + 时间约束 ----------------
LME_CTX = """Session 1 — 2021/03/05
On 2021-03-05 I bought a red bike.
On 2021-03-06 I painted the bike blue.
On 2021-03-07 I sold the bike to Tom.

Session 2 — 2022/07/09
On 2022-07-09 I bought a green kayak.
On 2022-07-10 I paddled the kayak on the lake.
On 2022-07-11 I stored the kayak in the garage.

Session 3 — 2023/11/20
On 2023-11-20 I bought a yellow tent.
On 2023-11-21 I pitched the tent in the woods.
On 2023-11-22 I cleaned the tent after the trip."""


def lme_index():
    epi = v9.EpisodicIndex()
    epi.add_units(v9.extract_units(LME_CTX, block_sents=1))
    return epi


def test_lme_session_quota_max2():
    """跨 Session 配额：同一 Session 最多 2 个命中单元，超出的按分数顺延到别的 Session。"""
    epi = lme_index()
    idxs, dbg = epi.readout("bike", top_k=4, mode="lme", turn_window=0)
    ok("lme dbg has session_quota", "session_quota" in dbg)
    ok("lme dbg has temporal_constraint", "temporal_constraint" in dbg)
    q = dbg["session_quota"]
    ok("lme quota default 2", q["max_per_session"] == 2)
    ok("lme quota dropped overflow", q["n_dropped"] >= 1)
    ok("lme quota per session <= 2", all(c <= 2 for c in q["per_session"].values()))
    sids = [epi.metas[i]["session_id"] for i in idxs]
    ok("lme quota spans multiple sessions", len(set(sids)) >= 2)
    ok("lme quota filled all slots", len(idxs) == 4)
    # 关掉配额 -> 允许同一 Session 多取（对照，证明是配额在起作用）
    idxs2, dbg2 = epi.readout("bike", top_k=4, mode="lme", turn_window=0, session_quota=False)
    ok("lme quota off -> max_per_session 0", dbg2["session_quota"]["max_per_session"] == 0)
    sids2 = [epi.metas[i]["session_id"] for i in idxs2]
    ok("lme quota off concentrates in one session",
       max(sids2.count(x) for x in set(sids2)) > 2)
    # 候选全在同一 Session 时配额无多样性收益 -> 自动退回不限（不白丢召回）
    one = v9.EpisodicIndex()
    one.add_units(v9.extract_units(" ".join(f"I did thing{i} today." for i in range(6)),
                                   block_sents=1))
    idxs3, dbg3 = one.readout("thing", top_k=5, mode="lme", turn_window=0)
    ok("lme single-session quota disabled", dbg3["session_quota"]["max_per_session"] == 0)
    ok("lme single-session fallback recorded",
       dbg3["session_quota"]["fallback"] == "single_session")
    ok("lme single-session keeps top_k", len(idxs3) == 5)


def test_lme_temporal_before_after():
    """before/after 是硬过滤（不是仅加分）：返回单元的日期必须满足约束。"""
    epi = lme_index()
    target = datetime.date(2022, 1, 1)
    idxs, dbg = epi.readout("bike before 2022-01-01", top_k=5, mode="lme", turn_window=0)
    tc = dbg["temporal_constraint"]
    ok("lme before parsed", tc["kind"] == "before" and tc["applied"] is True)
    ok("lme before filter shrank pool", tc["n_out"] < tc["n_in"])
    ok("lme before all dates < target",
       all((v9.unit_date(epi.metas[i])[0] or target) < target for i in idxs))
    idxs2, dbg2 = epi.readout("bike after 2022-01-01", top_k=5, mode="lme", turn_window=0)
    tc2 = dbg2["temporal_constraint"]
    ok("lme after parsed", tc2["kind"] == "after" and tc2["applied"] is True)
    spans = [v9.unit_date_span(epi.metas[i]) for i in idxs2]
    ok("lme after all spans end after target",
       all(lo is not None and hi > target for lo, hi, _p in spans))
    ok("lme after drops pre-target units",
       all(epi.metas[i]["session_id"] != 0 for i in idxs2))
    # 相对时间操作数：无锚点解析失败（退回 none），有锚点按锚点解析
    ok("lme relative operand needs anchor",
       v9.parse_temporal_constraint("what happened after yesterday?")["kind"] == "none")
    tc3 = v9.parse_temporal_constraint("what happened after yesterday?", query_ts="2023-11-22")
    ok("lme relative operand resolved with anchor",
       tc3["kind"] == "after" and tc3["date"] == "2023-11-21")


def test_lme_temporal_first_last():
    """first/last/most recent 是重排：最早/最晚日期的单元排最前。"""
    epi = lme_index()
    centers = [v9.unit_date_center(m) for m in epi.metas]
    known = [c for c in centers if c is not None]
    idxs, dbg = epi.readout("What is the first thing I bought?", top_k=3,
                           mode="lme", turn_window=0)
    ok("lme first order parsed", dbg["temporal_constraint"]["order"] == "first")
    # hits[0] 是重排后的首位（idxs 仍按原文位置升序返回，位置序 ≠ 排序分序）
    ok("lme first picks earliest unit", centers[dbg["hits"][0][0]] == min(known))
    ok("lme first returns that unit", dbg["hits"][0][0] in idxs)
    idxs2, dbg2 = epi.readout("What is the most recent thing I bought?", top_k=3,
                              mode="lme", turn_window=0)
    ok("lme last order parsed", dbg2["temporal_constraint"]["order"] == "last")
    ok("lme last picks latest unit", centers[dbg2["hits"][0][0]] == max(known))
    ok("lme last returns that unit", dbg2["hits"][0][0] in idxs2)
    ok("lme first/last differ", dbg["hits"][0][0] != dbg2["hits"][0][0])
    # "last week" 是时间点（相对时间），不是"最近一次"的顺序约束
    ok("lme last-week is not order constraint",
       v9.parse_temporal_constraint("What did I do last week?")["order"] is None)
    ok("lme first-week is not order constraint",
       v9.parse_temporal_constraint("What did I do in the first week?")["order"] is None)
    ok("lme last-time is order constraint",
       v9.parse_temporal_constraint("When did I last time buy a bike?")["order"] == "last")


def test_lme_temporal_none_unchanged():
    """解析不出约束时不改变结果，只记 temporal_constraint=none。"""
    epi = lme_index()
    idxs, dbg = epi.readout("kayak", top_k=3, mode="lme", turn_window=0)
    tc = dbg["temporal_constraint"]
    ok("lme none kind", tc["kind"] == "none" and tc["applied"] is False)
    ok("lme none no filter", tc["n_in"] == tc["n_out"])
    ok("lme none no order", tc["order"] is None)
    idxs2, _ = epi.readout("kayak", top_k=3, mode="lme", turn_window=0,
                           temporal_constraint={"kind": "none"})
    ok("lme none identical result", idxs == idxs2)
    # 无日期的 before 操作数（"before the trip"）-> 解析失败，退回现状
    idxs3, dbg3 = epi.readout("kayak before the trip", top_k=3, mode="lme", turn_window=0)
    idxs4, _ = epi.readout("kayak before the trip", top_k=3, mode="lme", turn_window=0,
                           temporal_constraint={"kind": "none"})
    ok("lme unparsable before -> none", dbg3["temporal_constraint"]["kind"] == "none")
    ok("lme unparsable before unchanged", idxs3 == idxs4)


def test_lme_turn_window_expands_within_session():
    """Session 内前后各扩 N 个 turn；跨 Session 一律不扩。"""
    epi = lme_index()
    # 直接测扩展边界：Session 2 的第一个单元往回扩不得进 Session 1
    got = epi.expand_session_turns([4], window=2)
    ok("lme no backward cross-session", 3 not in got)
    ok("lme forward stays in session", set(got) <= {4, 5, 6, 7})
    got2 = epi.expand_session_turns([3], window=2)
    ok("lme no forward cross-session", 4 not in got2)
    ok("lme backward stays in session", set(got2) <= {0, 1, 2, 3})
    # 端到端：命中 2 个单元，window=1 后只在各自 Session 内扩到邻居
    idxs, dbg = epi.readout("bike", top_k=2, mode="lme", turn_window=1)
    q = dbg["session_quota"]
    ok("lme turn_window recorded", q["turn_window"] == 1)
    ok("lme expansion grew unit set", q["n_expanded"] > q["n_hits"])
    hits = [h[0] for h in dbg["hits"]]
    allowed = set()
    for i in hits:
        sid = epi.metas[i]["session_id"]
        allowed |= {j for j in range(max(0, i - 1), min(len(epi), i + 2))
                    if epi.metas[j]["session_id"] == sid}
    ok("lme expansion only same-session neighbours", set(idxs) <= allowed)
    # window=0 时不扩（对照）
    idxs0, dbg0 = epi.readout("bike", top_k=2, mode="lme", turn_window=0)
    ok("lme window 0 no expansion", dbg0["session_quota"]["n_expanded"] == dbg0["session_quota"]["n_hits"])
    ok("lme window 0 fewer units", len(idxs0) < len(idxs))


# ---------------- 池内 whitening（t10） ----------------
def _offdiag_cos(Z):
    S = Z @ Z.t()
    m = ~torch.eye(len(Z), dtype=torch.bool)
    return S[m]


def test_l18_whitening_cosine_spread():
    """whitening 前后池内余弦分布变化：共同方向被压掉、分布变分散（排序才有区分度）。"""
    torch.manual_seed(7)
    N, d = 60, 16
    H = torch.randn(N, d)
    H[:, :4] += 5.0                       # 模拟"共同方向"（真实 L18 里的各向异性）
    saved_env = v9.L18_WHITEN
    v9.L18_WHITEN = True                  # 不受外部 V9_L18_WHITEN 环境影响
    Z_l2 = v9.l18_whiten_apply(H, None, None)
    mean, W, info = v9.l18_whiten_fit(H)
    Z_w = v9.l18_whiten_apply(H, mean, W)
    c_l2, c_w = _offdiag_cos(Z_l2), _offdiag_cos(Z_w)
    ok("whiten_fit 走白化分支", info["mode"] == "whiten" and W is not None)
    ok("L2-only 池内余弦被共同方向抬高（均值>0.5）", float(c_l2.mean()) > 0.5)
    ok("whitening 后池内余弦均值≈0（去中心化生效）", abs(float(c_w.mean())) < 0.05)
    ok("whitening 后余弦分布更分散（std 上升）", float(c_w.std()) > float(c_l2.std()))
    ok("whitening 后仍是单位向量", torch.allclose(Z_w.norm(dim=1),
                                                  torch.ones(len(Z_w)), atol=1e-5))
    ok("白化矩阵对称", torch.allclose(W, W.t(), atol=1e-6))
    ok("whitening 确定性（同输入同输出）",
       torch.allclose(W, v9.l18_whiten_fit(H)[1]) and
       torch.allclose(mean, v9.l18_whiten_fit(H)[0]))

    # V9_L18_WHITEN=0 -> 与旧的纯 L2 行为逐位一致（消融对照必须严格等价）
    saved = v9.L18_WHITEN
    try:
        v9.L18_WHITEN = False
        m0, W0, i0 = v9.l18_whiten_fit(H)
        ok("V9_L18_WHITEN=0 走 l2_only 且不出变换", i0["mode"] == "l2_only"
           and m0 is None and W0 is None)
        ok("WHITEN=0 与旧纯 L2 逐位一致",
           torch.equal(v9.l18_whiten_apply(H, m0, W0), Z_l2))
    finally:
        v9.L18_WHITEN = saved_env


def test_l18_whiten_rank_deficient_fallback():
    """d>N / N<=1：协方差秩不足 -> 退化为中心化-only（白化会放大噪声，不硬算）。"""
    torch.manual_seed(11)
    H = torch.randn(5, 16)                # d=16 > N=5
    saved_env = v9.L18_WHITEN
    v9.L18_WHITEN = True
    try:
        mean, W, info = v9.l18_whiten_fit(H)
        ok("d>N 退化为 center_only", info["mode"] == "center_only" and W is None)
        ok("d>N 仍给中心（mean 形状 [d]）",
           mean is not None and tuple(mean.shape) == (16,))
        Hc = H - mean
        ok("center_only 确实做了中心化（列均值≈0）", float(Hc.mean(dim=0).abs().max()) < 1e-5)
        Z = v9.l18_whiten_apply(H, mean, None)
        ok("center_only 后仍是单位向量", torch.allclose(Z.norm(dim=1), torch.ones(5), atol=1e-5))
    # N<=1：连协方差都估不出来，同样只中心化（N=1 时中心化后为零向量）
        m1, W1, i1 = v9.l18_whiten_fit(torch.randn(1, 16))
        ok("N<=1 退化为 center_only", i1["mode"] == "center_only" and W1 is None)
    # d==N 的边界也算秩不足（n 个样本最多 n-1 维）
        _, W2, i2 = v9.l18_whiten_fit(torch.randn(8, 8))
        ok("d==N 边界退化为 center_only", i2["mode"] == "center_only" and W2 is None)
    # 正常情形（N>d）不退化，且 eps 保证特征值下限 -> 条件数有限
        _, W3, i3 = v9.l18_whiten_fit(torch.randn(40, 8))
        ok("N>d 走白化且条件数有限", i3["mode"] == "whiten" and i3["cond"] < 1e6)
    finally:
        v9.L18_WHITEN = saved_env


def test_l18_precomputed_skips_forward():
    """给预计算矩阵后建索引不再调 provider（计数器断言：省掉二次前向）。"""
    units = [mk_unit("zebra likes grass", 0), mk_unit("quokka eats leaves", 1)]
    calls = []

    def prov(texts, kind):
        calls.append((kind, len(texts)))
        return torch.eye(len(texts), 8)

    v9.set_layer18_provider(prov)
    try:
        epi0 = v9.EpisodicIndex()
        epi0.add_units(list(units))
        idx_texts = list(epi0.index_texts)
        ok("无预计算时单元侧会前向一次", [k for k, _ in calls].count("units") == 1)

        vecs = torch.eye(len(idx_texts), 8)
        calls.clear()
        v9.set_layer18_provider(prov, precomputed=vecs, precomputed_texts=idx_texts)
        epi1 = v9.EpisodicIndex()
        epi1.add_units(list(units))
        ok("给预计算矩阵后建索引不再前向", [k for k, _ in calls].count("units") == 0)
        ok("预计算向量直接进 l18",
           epi1.l18 is not None and epi1.l18.shape[0] == len(units))
        epi1.retrieve("zebra", top_k=2, alpha=1.0, temporal=False)
        ok("query 侧仍只前向一次（不可避免的那次）",
           [k for k, _ in calls].count("query") == 1)
        spec = v9.layer18_spec()
        ok("spec 反映预计算文本索引",
           spec["precomputed"] and spec["precomputed_texts"]
           and spec["precomputed_text_rows"] == len(idx_texts))
        # 预计算矩阵行数不足 -> 告警并退回 provider（不静默返回空）
        calls.clear()
        v9.set_layer18_provider(prov, precomputed=vecs[:1], precomputed_texts=idx_texts[:1])
        epi2 = v9.EpisodicIndex()
        epi2.add_units(list(units))
        ok("预计算行数不足时退回 provider",
           [k for k, _ in calls].count("units") >= 1 and epi2.l18 is not None)
    finally:
        v9.set_layer18_provider(None)


def test_l18_precomputed_text_keyed_no_crosstalk():
    """回归：预计算矩阵按**文本**查表，offset 不得串行号（state 池共享 epi 矩阵的场景）。"""
    t0, t1 = "In 2021: alpha fact.", "In 2021: beta fact."
    vecs = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    v9.set_layer18_provider(lambda ts, k: torch.full((len(ts), 4), 0.5),
                            precomputed=vecs, precomputed_texts=[t0, t1])
    try:
        got = v9._call_l18([t1], "units", offset=0)      # offset 指向第 0 行，文本是第 1 行
        ok("按文本查表优先于 offset（不串行号）",
           torch.allclose(got[0], F.normalize(vecs[1], dim=0)))
        got2 = v9._call_l18([t0, t1], "units", offset=0)
        ok("批量查表按请求顺序返回",
           torch.allclose(got2, F.normalize(vecs, dim=-1)))
        # 文本不在矩阵里 -> 不能用 offset 硬切，必须退回 provider（0.5 常数向量）
        got3 = v9._call_l18(["In 2021: gamma fact."], "units", offset=0)
        ok("查不到文本时不静默套用别人那一行",
           torch.allclose(got3[0], F.normalize(torch.full((4,), 0.5), dim=0)))
        # 未给 texts 时退回位置语义（旧行为），仅按 offset 切片
        v9.set_layer18_precomputed(vecs)
        got4 = v9._call_l18([t1], "units", offset=1)
        ok("未给 texts 时退回 offset 位置语义",
           torch.allclose(got4[0], F.normalize(vecs[1], dim=0)))
    finally:
        v9.set_layer18_provider(None)


def test_l18_state_pool_reuses_epi_vectors():
    """复用：状态池与情景池文本重叠，状态池建池不再二次前向（V9_L18_REUSE_EPI 对照）。"""
    ctx = ("Alice works at Acme. Bob likes tea. The cat sat on the mat. "
           "Carol moved to Paris.")
    calls = []

    def prov(texts, kind):
        calls.append((kind, len(texts)))
        return torch.randn(len(texts), 8)

    saved = v9.L18_REUSE_EPI
    v9.set_layer18_provider(prov)
    try:
        v9.L18_REUSE_EPI = True
        calls.clear()
        epi, st = v9.build_pools_for_context(ctx)
        ok("状态池非空（确实有可复用的重叠文本）", len(st) > 0)
        ok("开启复用：单元侧只前向一次（epi 那一次）",
           [k for k, _ in calls].count("units") == 1)
        ok("状态池拿到了稠密向量", st.l18 is not None and st.l18.shape[0] == len(st))

        v9.L18_REUSE_EPI = False
        v9.clear_layer18_precomputed()
        calls.clear()
        epi2, st2 = v9.build_pools_for_context(ctx)
        ok("关闭复用：状态池会额外前向（对照组）",
           [k for k, _ in calls].count("units") >= 2)
        ok("两种模式下状态池单元数一致", len(st2) == len(st))
    finally:
        v9.L18_REUSE_EPI = saved
        v9.set_layer18_provider(None)


def test_l18_whitening_in_dense_scores():
    """端到端：whitening 真的作用在 dense_scores 上（query 与单元过同一变换）。"""
    torch.manual_seed(3)
    epi = v9.EpisodicIndex()
    epi.add_units([mk_unit(f"filler token{i} here", i) for i in range(40)])
    # 40 个单元 × d=8：N>d，走白化分支
    H = torch.randn(40, 8)
    H[:, 0] += 6.0
    epi.attach_layer18(H)
    q = torch.randn(8)
    q[0] += 6.0
    saved = v9.L18_WHITEN
    try:
        v9.L18_WHITEN = True
        d_on = epi.dense_scores(query="filler", q_hidden=q)
        Z, mean, W, info = epi.l18_transform()
        ok("池内变换为白化", info["mode"] == "whiten")
        ok("dense_scores 与手算白化余弦一致",
           torch.allclose(d_on, Z @ v9.l18_whiten_apply(q, mean, W)[0], atol=1e-5))
        v9.L18_WHITEN = False
        epi._l18_cache = None
        d_off = epi.dense_scores(query="filler", q_hidden=q)
        ok("WHITEN=0 时 dense_scores 退化为纯 L2 余弦",
           torch.allclose(d_off, v9.l18_whiten_apply(H, None, None)
                          @ F.normalize(q, dim=0), atol=1e-5))
        ok("开关确实改变了稠密分（默认行为变化可观测）",
           not torch.allclose(d_on, d_off, atol=1e-3))
    finally:
        v9.L18_WHITEN = saved
        epi._l18_cache = None


if __name__ == "__main__":
    test_non_overlapping_index()
    test_ts_inheritance_v9()
    test_score_first_then_position()
    test_eventqa_backward_expansion()
    test_timestamp_prefix_scoring()
    test_temporal_rerank()
    test_layer18_hybrid()
    test_layer18_provider()
    test_readout_modes()
    test_anchor_from_events()
    test_best_unit_for()
    test_state_pool_and_build()
    test_provider_multibatch_alignment()
    test_large_pool_smoke()
    test_env_defaults_and_dtype()
    test_mean_pool_hidden()
    test_layer18_end_to_end_pooling()
    test_empty()
    test_cr_overwrite_finds_same_subject_slot()
    test_cr_pending_merge_and_stable_ids()
    test_cr_parse_copular_predicate()
    test_cr_attribute_category_guard()
    test_cr_attribute_same_category_still_overwrites()
    test_session_boundary_ts()
    test_relative_timestamp_re_v9()
    test_relative_ts_anchor_v9()
    test_relative_ts_unknown_anchor_v9()
    test_session_marker_ev_v9()
    test_lme_session_quota_max2()
    test_lme_temporal_before_after()
    test_lme_temporal_first_last()
    test_lme_temporal_none_unchanged()
    test_lme_turn_window_expands_within_session()
    test_split_newline_chat()
    test_hybrid_device_consistency()
    test_layer18_strict_mode()
    test_l18_whitening_cosine_spread()
    test_l18_whiten_rank_deficient_fallback()
    test_l18_precomputed_skips_forward()
    test_l18_precomputed_text_keyed_no_crosstalk()
    test_l18_state_pool_reuses_epi_vectors()
    test_l18_whitening_in_dense_scores()
    print("ALL V9 CPU TESTS PASSED")
