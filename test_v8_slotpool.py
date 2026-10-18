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
    test_run_ar_import()
    print("ALL V8 CPU TESTS PASSED")
