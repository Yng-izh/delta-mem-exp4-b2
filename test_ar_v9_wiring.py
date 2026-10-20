# test_ar_v9_wiring.py — run_ar_v7.py 的 v9 接线 / 截断顺序 / 诊断字段 CPU 单测
#
# 不加载模型、不读 parquet、不初始化 CUDA：stub tokenizer + 随机小 EMB。
# 覆盖：
#   1. AR_MODE=v9 建池 + readout(topk) 先分数选 Top-K 再位置排序
#   2. EventQA 后向非对称扩展（anchor_events）与超配额时保留锚点邻域
#   3. 「先截断后排序」：v8 路径按分数优先裁候选，不再截掉高分尾部
#   4. ar_diag.record 逐题诊断字段（hits 分数降序 / answer_in_retrieved）
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ar_diag
import run_ar_v7 as R
import v9_slotpool as v9


class _Batch(dict):
    """同时支持 enc["input_ids"] 与 enc.input_ids（对齐 HF BatchEncoding）。"""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


class StubTok:
    """词级 stub tokenizer：str 或 list[str] -> padding batch。"""

    def __call__(self, texts, add_special_tokens=False, padding=False,
                 truncation=False, max_length=None, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        seqs = []
        for t in texts:
            ws = [w for w in v9.WORD_RE.findall(t.lower())] or ["<empty>"]
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
        return _Batch(input_ids=ids, attention_mask=am)


DIM = 32
torch.manual_seed(7)
EMB_MAT = F.normalize(torch.randn(10000, DIM), dim=-1)

R.tok = StubTok()
R.EMB = EMB_MAT
R.TOP_K = 2
R.V9_CTX_UNITS = 40
R.V9_TOPK_STATE = 0


def ok(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"  ok - {name}")


def fresh_v9(ctx, family="longmemeval"):
    """重置 v9 全局并建池（v9_import 会注入 R.tok / R.EMB）。"""
    v9.set_layer18_provider(None)
    R.FAMILY = family
    R.MODE = "v9"
    R.v9_slotpool = None
    return R.build_pool_for_context_v9(ctx)


# ---------------- 1. v9 topk：先分数选 Top-K 再位置排序 ----------------
def test_v9_topk_score_first_then_position():
    ctx = ("Alpha beta gamma. Delta epsilon zeta. Eta theta iota. "
           "Kappa lambda mu. Nu xi omicron. Pi rho sigma.")
    epi, st, _n = fresh_v9(ctx)
    ok("v9 units non-overlapping", len(epi) == 6)
    block, dbg, dctx = R.build_block_v9(epi, st, "epsilon kappa", {}, 0)

    ok("v9 dbg mode", dbg["mode"] == "topk_score_first")
    ok("v9 selected exactly Top-K", len(dctx["hits"]) == 2)
    scores = [s for _, s in dctx["hits"]]
    ok("v9 hits score-descending", scores == sorted(scores, reverse=True))
    ok("v9 used_ids position-ascending", dctx["used_ids"] == sorted(dctx["used_ids"]))
    ok("v9 used ids == dbg idxs", dctx["used_ids"] == dbg["idxs"])
    # 命中句 1（epsilon）与句 3（kappa）都必须被选中
    ok("v9 both matching sentences selected", set(dctx["used_ids"]) == {1, 3})
    ok("v9 block follows original order",
       block.split("\n") == [epi.texts[i] for i in dctx["used_ids"]])

    # 诊断字段：hits 顺序即 Recall@K 顺序
    d = ar_diag.record(answer="epsilon", query="epsilon kappa", flat=True, **dctx)
    ok("diag status ok", d["diag_status"] == "ok")
    ok("diag retrieved ids == hits order",
       d["retrieved_slot_ids"] == [i for i, _ in dctx["hits"]])
    ok("diag answer_in_retrieved", d["answer_in_retrieved"] is True)
    ok("diag answer_in_context", d["answer_in_context"] is True)
    ok("diag answer_hit_slot in Top-2", d["answer_hit_slot"] in (0, 1))
    ok("diag recall@3 hits", d["recall_at_k"]["3"] == 1)
    ok("diag recall@1 consistent with hit slot",
       d["recall_at_k"]["1"] == (1 if d["answer_hit_slot"] == 0 else 0))
    ok("diag used_slot_ids", d["used_slot_ids"] == dctx["used_ids"])

    # 逐题记录必须能直接写 JSONL（dbg 含 stats/hits，diag 平铺）
    import json
    rec = {"context_id": 0, "question_idx": 0, "retrieval": dbg, **d}
    line = json.dumps(rec, ensure_ascii=False)
    ok("diag record json-serializable", '"diag_status": "ok"' in line)


# ---------------- 2. EventQA 后向非对称扩展 ----------------
def test_v9_eventqa_backward():
    ctx = ("Alice moved to Paris in 1997. Bob painted the wall. "
           "Carol wrote a letter. Dave fixed the roof. Erin sang a song.")
    epi, st, _n = fresh_v9(ctx, family="eventqa")
    R.EQA_BACK, R.EQA_AHEAD = 1, 1
    meta = {"previous_events": ["1. Alice moved to Paris"]}
    block, dbg, dctx = R.build_block_v9(epi, st, "What happened next?", meta, 0)

    ok("eventqa mode", dbg["mode"] == "eventqa_backward")
    ok("eventqa anchor", dbg["anchor"] == 0)
    ok("eventqa asymmetric window", dctx["used_ids"] == [0, 1])
    ok("eventqa dbg back/ahead", (dbg["back"], dbg["ahead"]) == (1, 1))

    # 超配额：保留锚点邻域，而不是排序后砍尾部
    R.V9_CTX_UNITS = 2
    try:
        _b, dbg2, dctx2 = R.build_block_v9(epi, st, "What happened next?", meta, 0)
        ok("eventqa cap keeps anchor neighborhood", dctx2["used_ids"] == [0, 1])
    finally:
        R.V9_CTX_UNITS = 40
        R.EQA_BACK, R.EQA_AHEAD = 16, 4


# ---------------- 3. 先截断后排序 ----------------
def test_cap_before_sort():
    ok("cap keeps score order", R._cap_before_sort([9, 2, 5], 2) == [9, 2])
    ok("cap centered on anchor",
       R._cap_before_sort(list(range(10)), 4, anchor=5) == [3, 4, 5, 6])
    ok("cap dedupes", R._cap_before_sort([3, 3, 1, 2], 3) == [3, 1, 2])


class _FakeV8Epi:
    """v8 路径假池：命中按分数降序给出，邻域不扩展。"""

    def __init__(self, n=12):
        self.texts = [f"Sentence{i} word{i}." for i in range(n)]
        self.metas = [{"start": i, "end": i} for i in range(n)]

    def __len__(self):
        return len(self.texts)

    def retrieve(self, query, top_k=12):
        return [(8, 0.9), (2, 0.8), (11, 0.7), (0, 0.6), (5, 0.5), (3, 0.4)][:top_k]

    def neighbor_expand(self, hits, extra=0):
        return [i for i, _ in hits]

    def best_unit_for(self, text):
        return 0, 0.0


def test_v8_truncate_before_sort():
    R.FAMILY = "longmemeval"
    R.MODE = "v8"
    R.CTX_SENTS = 4
    R.TOP_K = 6
    try:
        epi = _FakeV8Epi()
        block, dbg, dctx = R.build_block_v8(epi, [], "word", {}, 0)
        # 分数序前 4 = [8,2,11,0]；按位置排序后拼块 = [0,2,8,11]
        ok("v8 cap before sort keeps high-score tail",
           dctx["used_ids"] == [0, 2, 8, 11])
        ok("v8 block order follows positions",
           block.split("\n") == [epi.texts[i] for i in [0, 2, 8, 11]])
        # 旧实现会先排序成 [0,2,3,5,8,11] 再截断 -> [0,2,3,5]，丢掉 8/11
        ok("v8 old truncation would have dropped 8/11",
           not set([0, 2, 8, 11]).issubset({0, 2, 3, 5}))
    finally:
        R.CTX_SENTS = 40
        R.TOP_K = 2


# ---------------- 4. v7 ChunkPool 诊断路径 ----------------
def test_v7_diag_path():
    R.FAMILY = "longmemeval"
    R.MODE = "v7"
    R.TOP_K = 2
    R.DEV = "cpu"
    try:
        p = R.ChunkPool()
        p.add_batch(["Alpha beta gamma.", "Delta epsilon zeta.", "Eta theta iota."])
        block, dbg, dctx = R.build_block(p, "epsilon", {}, 0)
        ok("v7 hits score-descending",
           [s for _, s in dctx["hits"]] == sorted([s for _, s in dctx["hits"]],
                                                  reverse=True))
        ok("v7 used ids position-ascending", dctx["used_ids"] == sorted(dctx["used_ids"]))
        ok("v7 block follows position order",
           block.split("\n") == [p.texts[i] for i in dctx["used_ids"]])
        d = ar_diag.record(answer="epsilon", query="epsilon", flat=True, **dctx)
        ok("v7 diag ok", d["diag_status"] == "ok" and d["answer_in_retrieved"] is True)
        ok("v7 pool has no metas -> starts None",
           all(s is None for s in d["retrieved_starts"]))
    finally:
        R.TOP_K = 2


# ---------------- 5. 诊断字段边界 ----------------
def test_diag_edge_cases():
    d = ar_diag.record(answer="Paris", query="q", hits=[], used_ids=[], flat=True)
    ok("empty hits -> no_hits", d["diag_status"] == "no_hits")
    ok("empty hits -> None answer_in_retrieved", d["answer_in_retrieved"] is None)
    rubric = "A long preference answer " * 20
    d2 = ar_diag.record(answer=rubric, query="q", hits=[(0, 1.0)], used_ids=[0],
                        texts=["some text"], flat=True)
    ok("rubric -> not applicable", d2["diag_status"] == "rubric_not_applicable")


def main():
    print("[1] v9 topk score-first")
    test_v9_topk_score_first_then_position()
    print("[2] v9 eventqa backward")
    test_v9_eventqa_backward()
    print("[3] truncate before sort")
    test_cap_before_sort()
    test_v8_truncate_before_sort()
    print("[4] v7 ChunkPool diag")
    test_v7_diag_path()
    print("[5] diag edge cases")
    test_diag_edge_cases()
    print("ALL PASS")


if __name__ == "__main__":
    main()
