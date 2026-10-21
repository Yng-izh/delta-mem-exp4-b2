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


def unit_with_marker(units, marker):
    """取 ts_marker 命中该时间标记的单元（滑窗会重叠，按标记本身定位最可靠）。"""
    for u in units:
        if (u.get("ts_marker") or "").lower() == marker.lower():
            return u
    raise AssertionError(f"FAIL: no unit with ts_marker={marker!r}")


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


# ---------------- 13. v8.2 回归：相对时间正则覆盖 ----------------
def test_timestamp_re_relative():
    """TIMESTAMP_RE 必须覆盖相对时间，且不把普通名词当时间（sun/Sat 大小写）。"""
    for s in ("We met yesterday.", "It happens today.", "See you tomorrow.",
              "That was last week.", "They met last summer.", "We talked last month.",
              "It rained next year.", "He arrived on Monday.", "Back in 3 days ago.",
              "Booked 2 weeks ago.", "Moved 6 months ago.", "Born 30 years ago."):
        assert v8.TIMESTAMP_RE.search(s), f"FAIL: relative ts not matched: {s}"
        assert v8.is_relative_ts(v8.TIMESTAMP_RE.search(s).group(0)), f"FAIL: not relative: {s}"
    # 星期缩写只在首字母大写时算时间；全称星期名不歧义
    ok("abbrev weekday needs capital", v8.TIMESTAMP_RE.search("Sun is up") is not None
       and v8.TIMESTAMP_RE.search("the sun is hot") is None)
    # 绝对标记不得被误判成相对
    ok("absolute not relative", v8.is_relative_ts("In March 3, 2001 he moved.")
       is False and v8.is_relative_ts("1997") is False)
    # ISO 不再被年份分支吃掉（旧版 group(0) == "2001"）
    ok("iso keeps full date", v8.TIMESTAMP_RE.search("dated 2001-03-03.").group(0) == "2001-03-03")
    ok("plain prose has no ts", v8.TIMESTAMP_RE.search("He likes apples.") is None)


# ---------------- 14. v8.2 回归：Session 边界不继承时间（I3） ----------------
def test_v82_session_reset_ts():
    """跨 Session 的单元不得继承上一 Session 的日期。"""
    ctx = ("Session 1\nIn 2020, Alice moved to Paris. She founded a lab.\n"
           "Session 2\nAlice prefers quiet restaurants. She likes tea.")
    epi, st = v8.extract_adaptive_units(ctx)
    s1 = [u for u in epi if u["session_id"] == 0]
    s2 = [u for u in epi if u["session_id"] == 1]
    ok("v82 both sessions present", bool(s1) and bool(s2))
    ok("v82 session 1 keeps 2020", any("2020" in (u["ts"] or "") for u in s1))
    # 旧版：全篇单一 last_ts -> Session 2 的单元都带 2020
    ok("v82 session 2 no inherited ts",
       all(u["ts"] is None and u["ts_inherited"] is False for u in s2))
    ok("v82 state also reset",
       all(u["ts"] is None for u in st if u["session_id"] == 1))
    # 窗口不跨 Session：Session 1 的单元里不得出现 Session 2 的内容，反之亦然
    ok("v82 window not crossing sessions",
       all("quiet" not in u["text"] and "Session 2" not in u["text"] for u in s1)
       and all("Paris" not in u["text"] and "2020" not in u["text"] for u in s2))
    # 关掉 Session 重置时退回旧行为（同段 -> 继承 2020），证明差异确实来自 Session 边界
    old = v8.SESSION_RESET
    v8.SESSION_RESET = False
    try:
        epi_legacy, _ = v8.extract_adaptive_units(ctx)
        ok("v82 legacy single-block inherits", all(u["session_id"] == 0 for u in epi_legacy))
    finally:
        v8.SESSION_RESET = old


# ---------------- 15. v8.2 回归：相对时间落到确定日期（以 Session 头为锚） ----------------
def test_v82_relative_ts_with_anchor():
    """'yesterday' 在已知 Session 日期下必须解析成确定日期。"""
    epi, _ = v8.extract_adaptive_units("2023-05-20\nYesterday Alice visited the museum.")
    u = unit_with_marker(epi, "Yesterday")
    ok("v82 yesterday resolved", u["ts"] == "2023-05-19")
    ok("v82 yesterday not the anchor", u["ts"] != "2023-05-20")
    ok("v82 yesterday marked relative", u["ts_relative"] is True and u["ts_unresolved"] is False)
    ok("v82 yesterday not inherited", u["ts_inherited"] is False)

    # 星期名 / last summer / N days ago / last week 同样可解析
    epi2, _ = v8.extract_adaptive_units("2023-05-20\nOn Monday she flew to Rome.")
    u2 = unit_with_marker(epi2, "Monday")
    ok("v82 weekday resolved", u2["ts"] == "2023-05-15")
    epi3, _ = v8.extract_adaptive_units("2023-05-20\nThey travelled last summer.")
    u3 = unit_with_marker(epi3, "last summer")
    ok("v82 last summer resolved", u3["ts"] == "2022-06-21")
    epi4, _ = v8.extract_adaptive_units("2023-05-20\nShe left 3 days ago.")
    u4 = unit_with_marker(epi4, "3 days ago")
    ok("v82 N days ago resolved", u4["ts"] == "2023-05-17")
    # 月名日期头（LongMemEval 常见格式）
    epi5, _ = v8.extract_adaptive_units("March 3, 2001\nYesterday he arrived.")
    u5 = unit_with_marker(epi5, "Yesterday")
    ok("v82 month-name anchor resolved", u5["ts"] == "2001-03-02")
    # 解析出的日期可在同一 Session 内被后续单元继承
    epi6, _ = v8.extract_adaptive_units(
        "2023-05-20\nYesterday Alice visited the museum. She bought a ticket.")
    # 句 2 的窗口（start=2）没有自己的标记，只能继承句 1 解析出的日期
    later = [x for x in epi6 if x["start"] == 2][0]
    ok("v82 resolved date inherited in session", later["ts"] == "2023-05-19"
       and later["ts_inherited"] is True)


# ---------------- 16. v8.2 回归：无锚点时只标记不解析（不编造日期） ----------------
def test_v82_relative_ts_unknown_anchor():
    """没有 Session 头日期时 resolved ts 保持 None，但必须标记为相对时间。"""
    ctx = "Session 1\nYesterday Alice visited the museum."
    epi, st = v8.extract_adaptive_units(ctx)
    u = unit_with_marker(epi, "Yesterday")
    ok("v82 unknown anchor ts None", u["ts"] is None)
    ok("v82 unknown anchor marked", u["ts_marker"] == "Yesterday" and u["ts_relative"] is True)
    ok("v82 unknown anchor unresolved", u["ts_unresolved"] is True)
    ok("v82 no fabricated date", all(x["ts"] is None for x in epi + st))
    # 锚点精度不足（只有年份）时也不得编造月日
    epi2, _ = v8.extract_adaptive_units("Session 1 — 1997\nYesterday he arrived.")
    u2 = unit_with_marker(epi2, "Yesterday")
    # 锚点只有年份：不得编造月/日（只继承到年精度），同时标记为未解析
    ok("v82 year-only anchor no fabricated day",
       u2["ts"] == "1997" and "-" not in (u2["ts"] or "") and u2["ts_unresolved"] is True)
    # 但年份级相对时间可以解析
    ok("v82 year-level relative ok", v8.resolve_relative_ts("last year", "1997") == "1996")


# ---------------- 17. v8.2 回归：Session 标记纳入 ev 触发 ----------------
def test_v82_session_marker_triggers_episodic():
    """Session 标记本身是时间/结构边界，必须触发 episodic（不得落进状态池）。"""
    body = "The river flows through the town. The bridge is old."
    epi_plain, st_plain = v8.extract_adaptive_units(body)
    ok("v82 plain prose goes to state", len(epi_plain) == 0 and len(st_plain) == 2)
    epi, st = v8.extract_adaptive_units("Session 1\n" + body)
    ok("v82 session marker triggers episodic", len(epi) >= 1)
    ok("v82 session marker unit covers header", epi[0]["start"] == 0 and epi[0]["session_id"] == 0)
    ok("v82 session marker not in state",
       all("Session" not in x["text"] for x in st))
    ok("v82 session marker keeps ts None", epi[0]["ts"] is None)


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
    # v8.2 时间语义回归（Session 边界 + 相对时间 + Session 标记 ev）
    test_timestamp_re_relative()
    test_v82_session_reset_ts()
    test_v82_relative_ts_with_anchor()
    test_v82_relative_ts_unknown_anchor()
    test_v82_session_marker_triggers_episodic()
    print("ALL V8 CPU TESTS PASSED")
