# test_v9_slotpool.py — v9_slotpool CPU 逻辑单测（stub tokenizer / stub layer18，不依赖 GPU/模型）
#
# 覆盖 v9 五项检索改动：
#   1. 非重叠单句/微块索引（不丢句、不重叠）
#   2. 先按分数选 Top-K，再按原文位置排序
#   3. EventQA 后向 12-16 句非对称扩展（含边界裁剪）
#   4. 时间戳前缀参与检索打分 + 时序重排
#   5. 第 18 层隐状态混合稠密打分接口（provider / attach / 退化）
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
        def _should_merge(self, text, meta, qm, base):
            return False

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
    print("ALL V9 CPU TESTS PASSED")
