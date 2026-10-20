# test_v8_slotpool.py — v8_slotpool CPU 逻辑单测（stub tokenizer，不依赖 GPU/模型）
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import v8_slotpool as v8


# ---------------- stub：hash-based 词向量（确定性、CPU） ----------------
class StubTok:
    """词级 stub tokenizer：call(list[str]) -> padding batch。"""

    def __call__(self, texts, add_special_tokens=False, padding=False,
                 truncation=False, max_length=None, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        seqs = []
        for t in texts:
            ws = [w for w in v8.WORD_RE.findall(t.lower())]
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
v8.tok = StubTok()
v8.EMB = EMB_MAT


def ok(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"  ok - {name}")


# ---------------- 1. 时间戳正则 ----------------
def test_timestamp_re():
    assert v8.TIMESTAMP_RE.search("In March 3, 2001 he moved.")
    assert v8.TIMESTAMP_RE.search("It happened in 1997.")
    assert v8.TIMESTAMP_RE.search("during the 1990s.")
    assert v8.TIMESTAMP_RE.search("on 12 January 2005.")
    assert v8.TIMESTAMP_RE.search("dated 2001-03-03.")
    assert not v8.TIMESTAMP_RE.search("He likes apples.")
    print("ok - timestamp_re")


# ---------------- 2. extract_adaptive_units：分流 + 滑窗 + 时间戳继承 ----------------
CTX = ("In March 3, 2001 John moved to Paris. He founded a startup there. "
       "He hired staff quickly. The startup failed later. In 2004 he returned to London. "
       "Paris is the capital of France. The Eiffel Tower is famous. "
       "Water boils at 100 degrees. Gold is a heavy metal.")

def test_extract():
    epi, st = v8.extract_adaptive_units(CTX)
    ok("episodic non-empty", len(epi) > 0)
    ok("state non-empty", len(st) > 0)
    # 滑窗宽度：窗口最多 WIN 句
    ok("win<=3", all(u["end"] - u["start"] + 1 <= v8.WIN for u in epi))
    # 时间戳继承：窗口 1（founded a startup）无显式标记 -> 继承 In March 3, 2001
    # 时间戳继承：句 0 显式 1990；句 1-2 事件句无标记 -> 窗口 (1..3) 继承 1990
    mini = "In 1990 Alice moved to Berlin. She founded a lab. She hired staff."
    epi2, _ = v8.extract_adaptive_units(mini)
    u1 = epi2[1] if len(epi2) > 1 else epi2[0]
    ok("ts inherit", u1["ts_inherited"] and "1990" in u1["ts"])
    # 状态池只有稳定陈述句
    ok("state stable", all("founded" not in u["text"] and "moved" not in u["text"]
                           and "returned" not in u["text"] for u in st))
    # 窗口 start 单调
    ok("starts ascending", all(a["start"] < b["start"] for a, b in zip(epi, epi[1:])))


# ---------------- 3. EpisodicFastPool：全池无截断 IDF-MaxSim ----------------
def make_context(n_sents=200, target_word="quantum"):
    sents = []
    for i in range(n_sents):
        if i == n_sents // 2:
            sents.append(f"In 1999 the {target_word} experiment succeeded in Geneva.")
        else:
            sents.append(f"Entity{i} did routine thing number {i} in city{i} in 19{90+i%10}.")
    return " ".join(sents)


def test_fullpool_no_truncation():
    # 强制 >3500 情景微块，验证检索扫描全池（不设 Top-2500 截断）
    big = make_context(n_sents=11000)
    epi, st = v8.build_pools_for_context(big)
    n = len(epi)
    ok("pool >3500 units", n > 3500)
    scores = epi.idf_maxsim_scores("quantum experiment succeeded Geneva")
    ok("scores cover full pool", scores.shape[0] == n)
    top = epi.retrieve("quantum experiment succeeded Geneva", top_k=3)
    ok("target unit in top3", any("quantum" in epi.texts[i] for i, _ in top))


def test_idf_weighting():
    # 含独有词（高 IDF）的单元应排在高频词单元前
    ctx = ("alpha beta common common common. gamma delta common common common. "
           "In 1990 the unique zebra crossed the river. epsilon zeta common common common.")
    epi, _ = v8.build_pools_for_context(ctx)
    hits = epi.retrieve("zebra crossed river", top_k=4)
    ok("unique unit top1", "zebra" in epi.texts[hits[0][0]])


def test_neighbor_expand():
    ctx = (" ".join(f"In 19{80+i} person{i} moved to place{i}." for i in range(30)))
    epi, _ = v8.build_pools_for_context(ctx)
    hits = epi.retrieve("person10 moved", top_k=2)
    idxs = epi.neighbor_expand(hits, extra=2)
    ok("neighbor expand >= hits", len(idxs) >= len(hits))
    starts = sorted(epi.metas[i]["start"] for i in idxs)
    hit_starts = {epi.metas[i]["start"] for i, _ in hits}
    hit_bounds = [(epi.metas[i]["start"], epi.metas[i]["end"]) for i, _ in hits]
    ok("neighbors adjacent", all(
        any(min(abs(s - st), abs(s - en)) <= v8.WIN for st, en in hit_bounds)
        for s in starts))


def test_state_pool_merge():
    st = v8.StatePool()
    st.add_batch([("Paris is the capital of France.", {"start": 0, "end": 0, "ts": None}),
                  ("The sky is blue.", {"start": 1, "end": 1, "ts": None})])
    n_before = len(st)
    # 近重复覆写（同 hash 词向量 -> mean cos = 1 > theta）
    st.add_batch([("Paris is the capital of France.", {"start": 9, "end": 9, "ts": None})])
    ok("state merge overwrite", len(st) == n_before)
    ok("state n_dup counted", st.n_dup >= 1)


def test_dedup():
    # 同一窗口文本（重复段+间隔）出现两次 -> 精确去重命中
    rep = ("A did X in 1990. B did Y in 1991. C did Z in 1992. "
           "D did P in 1993. E did Q in 1994. "
           "A did X in 1990. B did Y in 1991. C did Z in 1992.")
    epi, _ = v8.build_pools_for_context(rep)
    ok("exact dedup", epi.n_dup >= 1)


def test_empty_pool():
    epi, st = v8.build_pools_for_context("")
    ok("empty pools", len(epi) == 0 and len(st) == 0)
    ok("retrieve empty", epi.retrieve("q") == [])


# ---------------- 10. 回归：Bug1 cur_ts 第二轮污染 ----------------
def test_bug1_ts_no_second_pass_pollution():
    """时间标记之前的稳定陈述句不得继承【文档末尾】时间戳。"""
    ctx = ("Water boils at 100 degrees. Gold is a heavy metal. Ice is cold. "
           "Salt is salty. Iron is hard. "
           "In 2004 Alice moved to London. She founded a lab there. She hired staff.")
    epi, st = v8.extract_adaptive_units(ctx)
    ok("bug1 state exists", len(st) >= 1)
    ok("bug1 state starts are 0..2", [u["start"] for u in st] == [0, 1, 2])
    # 旧版：首轮 cur_ts 停在 2004，状态轮直接复用 -> 三条状态单元都带 "2004"
    ok("bug1 state ts not polluted", all(u["ts"] is None for u in st))
    ok("bug1 state ts_inherited False", all(u["ts_inherited"] is False for u in st))


def test_bug1_window_inherits_preceding_ts():
    """窗口无显式标记时继承【窗口起点之前最近】的时间戳，而非全篇末尾时间戳。"""
    ctx = ("In 1990 Alice moved to Berlin. She founded a lab. She hired staff. She sold the lab. "
           "In 2015 Bob moved to Rome. He founded a gym. He hired coaches.")
    epi, _ = v8.extract_adaptive_units(ctx)
    by_start = {u["start"]: u for u in epi}
    ok("bug1 window start=1 exists", 1 in by_start)
    w1 = by_start[1]
    # 旧版：cur_ts 已是全篇末尾 2015 -> w1["ts"] == "2015"
    ok("bug1 window ts inherits 1990", "1990" in (w1["ts"] or ""))
    ok("bug1 window ts not doc-last", "2015" not in (w1["ts"] or ""))
    ok("bug1 window ts_inherited True", w1["ts_inherited"] is True)
    # 窗口内显式标记优先，且不标记为继承
    ok("bug1 explicit ts in window", "2015" in (by_start[4]["ts"] or "")
       and by_start[4]["ts_inherited"] is False)


# ---------------- 11. 回归：Bug3 ev 时间触发判定 ----------------
def test_bug3_timestamp_triggers_episodic():
    """含显式时间标记但无过去时事件动词的句子必须进 episodic，不进 state。"""
    ctx = ("In 2001 the population of the town is 5000. The river flows through it. "
           "The bridge is old. The school is new.")
    epi, st = v8.extract_adaptive_units(ctx)
    # 旧版：无任何事件动词 -> epi 为空，四句全进 state
    ok("bug3 episodic non-empty", len(epi) == 1)
    ok("bug3 window start=0", epi[0]["start"] == 0 and epi[0]["end"] == 2)
    ok("bug3 window ts=2001", "2001" in (epi[0]["ts"] or ""))
    ok("bug3 window ts explicit", epi[0]["ts_inherited"] is False)
    ok("bug3 no state at ts sentence", all(u["start"] != 0 for u in st))
    ok("bug3 state only uncovered", [u["start"] for u in st] == [3])


def test_bug3_inherited_only_does_not_trigger():
    """仅"继承"到时间戳、窗口内无显式标记且无事件动词 -> 不触发 episodic（仍为 state）。"""
    ctx = ("In 2001 the population of the town is 5000. The river flows through it. "
           "The bridge is old. The school is new.")
    epi, st = v8.extract_adaptive_units(ctx)
    ok("bug3 inherited-only not episodic", all(u["start"] != 1 for u in epi))
    # 该句进入 state 时，继承的是它之前最近的时间戳（2001），不是 None
    s3 = [u for u in st if u["start"] == 3][0]
    ok("bug3 state inherits preceding ts", "2001" in (s3["ts"] or ""))
    ok("bug3 state ts_inherited", s3["ts_inherited"] is True)


def test_no_sentence_lost():
    """分流不丢句：每个句号索引至少被一个情景窗口或一个状态单元覆盖。"""
    for ctx in (CTX, ("In 2001 the population of the town is 5000. The river flows through it. "
                      "The bridge is old. The school is new.")):
        epi, st = v8.extract_adaptive_units(ctx)
        n = len(v8.split_sentences(ctx))
        covered = set()
        for u in epi:
            covered.update(range(u["start"], u["end"] + 1))
        covered.update(u["start"] for u in st)
        ok(f"no sentence lost (n={n})", covered == set(range(n)))


# ---------------- 12. 回归：Bug2 512 词 span 偏移 ----------------
def test_bug2_long_unit_span_offset():
    """>512 词单元不得越界吃进后续单元的词；后续单元 tokv/words/df 不得错位。"""
    long_text = " ".join(f"lw{i}" for i in range(600))
    short_text = "zebrafish zorblat quokka"
    epi = v8.EpisodicFastPool()
    epi.add_batch([(long_text, {"start": 0, "end": 0}),
                   (short_text, {"start": 1, "end": 1})])
    ok("bug2 two units", len(epi) == 2)
    # 旧版：spans[0]=(0,600) -> words[0] 长 515（含 short 的 3 个词），tokv[0] 长 515
    ok("bug2 long unit words truncated", len(epi.words[0]) == v8.UNIT_MAX_WORDS)
    ok("bug2 long unit last word", epi.words[0][-1] == "lw511")
    ok("bug2 long unit no bleed", all(not w.startswith("z") for w in epi.words[0][-8:]))
    ok("bug2 long unit tokv aligned", len(epi.tokv[0]) == len(epi.words[0]))
    # 旧版：spans[1]=(600,603) 越界 -> words[1]==["<empty>"]、tokv 为零向量
    ok("bug2 short unit words", epi.words[1] == ["zebrafish", "zorblat", "quokka"])
    ok("bug2 short unit tokv aligned", len(epi.tokv[1]) == 3)
    ok("bug2 short unit df", epi.word_df.get("zebrafish", 0) == 1)
    ok("bug2 total tokens", sum(t.shape[0] for t in epi.tokv) == v8.UNIT_MAX_WORDS + 3)
    # 检索：short 单元必须排第一（旧版其向量为零向量，无法命中）
    hits = epi.retrieve("zebrafish zorblat quokka", top_k=2)
    ok("bug2 short unit retrieved top1", hits and hits[0][0] == 1)
    # 长单元自身仍可被其末尾词命中（确认截断保留的是前 512 词）
    ok("bug2 long unit keeps first words", "lw511" in epi.words[0] and "lw512" not in epi.words[0])


def test_bug2_zero_token_unit_no_offset_drift():
    """零 token 兜底单元不得推进 token 偏移（否则后续单元整体错位）。"""
    epi = v8.EpisodicFastPool()
    real = v8.encode_words

    def fake_encode(words):
        """模拟异常 tokenizer：emptyunit 返回 0 token，其余 1 token。"""
        out, counts = [], []
        for w in words:
            if w == "emptyunit":
                counts.append(0)
                continue
            out.append(EMB_MAT[hash(w) % 9000 + 10].clone().unsqueeze(0))
            counts.append(1)
        flat = torch.cat(out, dim=0) if out else torch.zeros((0, DIM))
        return flat, counts

    v8.encode_words = fake_encode
    try:
        epi.add_batch([("emptyunit", {"start": 0, "end": 0}),
                       ("zebrafish zorblat", {"start": 1, "end": 1})])
    finally:
        v8.encode_words = real
    ok("bug2 fallback unit kept", len(epi) == 2)
    ok("bug2 fallback words", epi.words[0] == ["emptyunit"])
    ok("bug2 fallback tokv one row", len(epi.tokv[0]) == 1)
    # 旧版：兜底分支 off += 1 -> 后续单元 tokv 少一行
    ok("bug2 no drift after fallback", epi.words[1] == ["zebrafish", "zorblat"]
       and len(epi.tokv[1]) == 2)


def test_run_ar_import():
    """run_ar_v7.py 语法/结构验证：本地机器缺 transformers，无法导入；
    改为 AST 解析校验 v8 钩子存在（py_compile 已另行验证）。"""
    import ast
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "run_ar_v7.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    fns = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    ok("has build_pool_for_context_v8", "build_pool_for_context_v8" in fns)
    ok("has build_block_v8", "build_block_v8" in fns)
    ok("has v8_import", "v8_import" in fns)
    ok("MODE env hook", 'os.environ.get("AR_MODE", "v7")' in src)


if __name__ == "__main__":
    test_timestamp_re()
    test_extract()
    test_fullpool_no_truncation()
    test_idf_weighting()
    test_neighbor_expand()
    test_state_pool_merge()
    test_dedup()
    test_empty_pool()
    # v8.1 三个确定性 Bug 回归
    test_bug1_ts_no_second_pass_pollution()
    test_bug1_window_inherits_preceding_ts()
    test_bug3_timestamp_triggers_episodic()
    test_bug3_inherited_only_does_not_trigger()
    test_no_sentence_lost()
    test_bug2_long_unit_span_offset()
    test_bug2_zero_token_unit_no_offset_drift()
    test_run_ar_import()
    print("ALL V8 CPU TESTS PASSED")
