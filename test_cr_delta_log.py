# test_cr_delta_log.py — t3：CR_DELTA_MODE（off|context|refresh）+ 逐题检索证据日志
#
# 覆盖：
#   1. CR_DELTA_MODE 解析/校验（非法值报错，不静默退回 off）
#   2. delta_plan：off=查询前清零；context=每题恢复写后快照；refresh=再按 Top-K 刷新
#   3. refresh_slots：分数序去重取 Top-K（off/context 不刷新）
#   4. 配置指纹随 CR_DELTA_MODE 变化（三档不共用结果文件），且不再有 keep_delta 死开关
#   5. 证据日志：block_fact_lines 只认事实行、CRStatePool.diag_ctx 输出分数降序候选，
#      喂 ar_diag.record 后 retrieved_*/used_* / 'cr:<idx>' 统一 ID / used_alignment 全对
#   6. runner 接线：--dry-run 真能跑通（不加载模型），且 dry-run 输出三档不同指纹
#
# 全部 CPU（stub tokenizer + 小嵌入矩阵），不需要模型/GPU/parquet。
import json
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import v9_slotpool as v9
import v9_cr
import ar_diag
import run_manifest

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


class StubTok:
    """词级 stub tokenizer（与 test_v9_cr.py 同款）。"""

    def __call__(self, texts, add_special_tokens=False, padding=False,
                 truncation=False, max_length=None, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        seqs = []
        for t in texts:
            ws = v9.WORD_RE.findall(t.lower()) or ["<empty>"]
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


v9_cr.set_encoder(StubTok(), EMB)


def ok(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"  ok - {name}")


def _hop_pool():
    """Alice --born in--> Springfield --located in--> France --located in--> Europe。"""
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
                           "Pizza was created in Italy."]):
        pool.upsert(f, i + 1)
    return pool


# ---------------- 1. CR_DELTA_MODE 解析 ----------------
def test_delta_mode_parse():
    ok("modes", v9_cr.CR_DELTA_MODES == ("off", "context", "refresh"))
    ok("default off", v9_cr.parse_delta_mode(None) == "off")
    ok("empty -> off", v9_cr.parse_delta_mode("") == "off")
    ok("strip/lower", v9_cr.parse_delta_mode("  CONTEXT ") == "context")
    ok("refresh", v9_cr.parse_delta_mode("refresh") == "refresh")
    raised = False
    try:
        v9_cr.parse_delta_mode("keep")
    except ValueError as e:
        raised = "CR_DELTA_MODE" in str(e)
    ok("invalid raises (不静默退回 off)", raised)


# ---------------- 2. delta_plan ----------------
def test_delta_plan():
    off = v9_cr.delta_plan("off", n_pool=5)
    ok("off resets before query", off["reset_before_query"] is True)
    ok("off no restore", off["restore_snapshot"] is False)
    ok("off no refresh", off["refresh_topk"] == 0)

    ctx = v9_cr.delta_plan("context", n_pool=5)
    ok("context no reset", ctx["reset_before_query"] is False)
    ok("context restores snapshot", ctx["restore_snapshot"] is True)
    ok("context no refresh", ctx["refresh_topk"] == 0)

    ref = v9_cr.delta_plan("refresh", n_pool=5)
    ok("refresh no reset", ref["reset_before_query"] is False)
    ok("refresh restores snapshot", ref["restore_snapshot"] is True)
    ok("refresh top-k from env default",
       ref["refresh_topk"] == v9_cr.CR_DELTA_REFRESH_TOPK > 0)
    ok("n_pool recorded", ref["n_pool"] == 5)
    raised = False
    try:
        v9_cr.delta_plan("bogus")
    except ValueError:
        raised = True
    ok("delta_plan rejects bogus", raised)


# ---------------- 3. refresh_slots ----------------
def test_refresh_slots():
    hits = [(3, 0.91), (1, 0.88), (3, 0.90), (7, 0.70)]
    ok("dedupe keeps first occurrence", v9_cr.refresh_slots(hits, 10) == [3, 1, 7])
    ok("top_k truncates", v9_cr.refresh_slots(hits, 2) == [3, 1])
    ok("top_k=0 no refresh", v9_cr.refresh_slots(hits, 0) == [])
    ok("top_k<0 no refresh", v9_cr.refresh_slots(hits, -1) == [])
    ok("empty hits", v9_cr.refresh_slots([], 5) == [])
    ok("plain ints accepted", v9_cr.refresh_slots([2, 0, 2], 5) == [2, 0])
    ok("dict hits accepted", v9_cr.refresh_slots([{"idx": 4, "score": .5}], 5) == [4])


# ---------------- 4. 配置指纹隔离 ----------------
def test_fingerprint_isolated():
    base = {"mode": "v9", "theta": 0.72, "delta_refresh_topk": 6}
    ids = {}
    for m in v9_cr.CR_DELTA_MODES:
        p = dict(base); p["delta_mode"] = m
        ids[m] = run_manifest.config_id(p, kind="cr", src="FactConsolidation")
    ok("three modes -> three config ids", len(set(ids.values())) == 3)
    p = dict(base); p["delta_mode"] = "context"
    ok("same config -> same id",
       run_manifest.config_id(p, kind="cr", src="FactConsolidation") == ids["context"])
    # 指纹里不许再出现只记录不生效的 keep_delta
    src = open(os.path.join(HERE, "run_group_b2_v7.py"), encoding="utf-8").read()
    # 死开关必须真的删掉：env 读取与指纹键都不许再出现（注释里提历史原因不算）
    ok("dead switch B2_KEEP_DELTA no longer read",
       'os.environ.get("B2_KEEP_DELTA"' not in src)
    ok("no keep_delta in fingerprint", '"keep_delta"' not in src)
    ok("delta_mode in fingerprint", '"delta_mode": CR_DELTA_MODE' in src)
    ok("diag logged per question", "ar_diag.record(answer=a" in src)
    ok("FINAL CR hit rate", "answer_hit@K" in src)
    ok("dry-run supported", 'DRY_RUN = "--dry-run" in sys.argv[1:]' in src)


# ---------------- 5. 证据日志 ----------------
def test_block_fact_lines():
    pool = _hop_pool()
    block = "\n".join([
        "PATH 1 (len=2, score=0.900, rel=0.50/0.40): A --[r]--> B",
        "1. Alice was born in Springfield.",
        "2. Springfield is located in France.",
        "OTHER SELECTED FACTS:",
        "5. Pizza was created in Italy.",
    ])
    lines = v9_cr.block_fact_lines(pool, block)
    ok("only fact lines kept", len(lines) == 3)
    ok("serial -> slot idx", [i for i, _ in lines] == [0, 1, 4])
    ok("line text kept verbatim", lines[0][1] == "1. Alice was born in Springfield.")
    # 没有 serial 元数据时按原文反查；反查不到就丢弃（不编造下标）
    bare = type("Bare", (), {"texts": ["A is B.", "C is D."], "metas": [{}, {}]})()
    got = v9_cr.block_fact_lines(bare, "7. C is D.\n8. 未知事实。")
    ok("fallback by text", [i for i, _ in got] == [1])
    ok("unknown dropped", len(got) == 1)


def test_diag_ctx_fields():
    pool = _hop_pool()
    q = "Where is the city Alice was born in located?"
    block = pool.retrieve_multihop_context(q)
    diag = pool.diag_ctx(block)
    hits = diag["hits"]
    ok("hits non-empty", len(hits) > 0)
    scores = [s for _, s in hits]
    ok("hits score-descending", scores == sorted(scores, reverse=True))
    ok("hits ids unique", len({i for i, _ in hits}) == len(hits))
    ok("hits cover used slots", set(diag["used_ids"]) <= {i for i, _ in hits})
    ok("used ids/texts aligned", len(diag["used_ids"]) == len(diag["used_texts"]) > 0)
    ok("pool kind is cr", diag["pool_kind"] == "cr")

    rec = ar_diag.record(answer="France", query=q, flat=True, **diag)
    ok("retrieved_slot_ids recorded", rec["retrieved_slot_ids"] == [i for i, _ in hits])
    ok("retrieved_scores recorded", rec["retrieved_scores"] == scores)
    ok("retrieved_texts aligned",
       len(rec["retrieved_texts"]) == len(hits)
       and all(rec["retrieved_texts"][k] == pool.texts[i] for k, (i, _) in enumerate(hits)))
    ok("unified ids prefixed cr:",
       all(u.startswith("cr:") for u in rec["retrieved_slot_uids"] + rec["used_slot_uids"]))
    ok("used_slot_ids are ints",
       all(isinstance(i, int) for i in rec["used_slot_ids"]))
    ok("used alignment ok", rec["used_alignment"] == "ok")
    ok("diag_status ok", rec["diag_status"] == "ok")
    ok("gold in retrieved", rec["answer_in_retrieved"] is True)
    ok("gold in context", rec["answer_in_context"] is True)
    ok("evidence recall not faked", rec["evidence_recall_at_k"] is None)
    # used_texts 就是真正拼进 prompt 的事实行（含重复的共享实体）
    ok("used_texts are prompt lines",
       all(t.split(". ", 1)[-1] in pool.texts for t in rec["used_texts"]))


def test_diag_ctx_empty_pool():
    pool = v9_cr.CRStatePool()
    block = pool.retrieve_multihop_context("anything?")
    diag = pool.diag_ctx(block)
    ok("empty pool no hits", diag["hits"] == [])
    ok("empty pool no used", diag["used_ids"] == [])
    rec = ar_diag.record(answer="x", query="q", flat=True, **diag)
    ok("empty pool status no_hits", rec["diag_status"] == "no_hits")
    ok("empty pool no fake recall", rec["answer_in_retrieved"] is None)


# ---------------- 5b. 查询循环字段 smoke（无模型，复刻 runner 的 rec 组装） ----------------
def test_rec_fields_smoke():
    """复刻 run_group_b2_v7 查询循环里的字段组装（不跑模型），固定 JSONL 契约。"""
    pool = _hop_pool()
    q, gold = "Where is the city Alice was born in located?", "France"
    block = pool.retrieve_multihop_context(q)
    diag = pool.diag_ctx(block)
    plan = v9_cr.delta_plan("refresh", n_pool=len(pool))
    ref_slots = v9_cr.refresh_slots(diag["hits"], plan["refresh_topk"])
    n_refresh = len([i for i in ref_slots if 0 <= i < len(pool.texts)])
    rec = {"question_idx": 0, "query": q, "answer": gold, "output": "France"}
    rec["cr_dbg"] = pool.last_dbg
    rec["delta"] = {"mode": plan["mode"],
                    "restore_snapshot": bool(plan["restore_snapshot"]),
                    "snapshot_layers": 0,
                    "refresh_topk": plan["refresh_topk"],
                    "refresh_slots": ref_slots, "n_refresh": n_refresh}
    rec.update(ar_diag.record(answer=gold, query=q, flat=True, **diag))

    for key in ("retrieved_slot_ids", "retrieved_scores", "retrieved_texts",
                "used_texts", "used_slot_ids", "retrieved_slot_uids",
                "used_slot_uids", "answer_in_retrieved", "answer_in_context",
                "answer_recall_at_k", "diag_status"):
        ok(f"rec has {key}", key in rec)
    ok("delta mode recorded", rec["delta"]["mode"] == "refresh")
    ok("delta refresh slots recorded", rec["delta"]["refresh_slots"] == ref_slots)
    ok("delta refresh count", rec["delta"]["n_refresh"] == n_refresh == len(ref_slots))
    ok("cr_dbg still present", "chains2" in rec["cr_dbg"])
    ok("rec json-serializable", isinstance(json.dumps(rec, ensure_ascii=False), str))


# ---------------- 6. runner --dry-run 端到端 ----------------
def _dry_run(mode):
    env = dict(os.environ)
    env.update({"CR_MODE": "v9", "CR_DELTA_MODE": mode, "HF_HUB_OFFLINE": "1"})
    p = subprocess.run([sys.executable, os.path.join(HERE, "run_group_b2_v7.py"),
                        "FactConsolidation", "--dry-run"],
                       cwd=HERE, env=env, capture_output=True, text=True, timeout=180)
    return p


def test_runner_dry_run():
    p = _dry_run("context")
    ok("dry-run exits 0", p.returncode == 0)
    out = p.stdout
    ok("dry-run reports mode", "CR_DELTA_MODE=context" in out)
    ok("dry-run prints manifest", "[dry-run] manifest" in out)
    ok("dry-run prints delta_plan", "delta_plan(context)" in out)
    ids = set()
    for m in v9_cr.CR_DELTA_MODES:
        tag = f"config_id(delta_mode={m:8s}) = "
        line = next((l for l in out.splitlines() if tag in l), None)
        ok(f"dry-run fingerprint line for {m}", line is not None)
        ids.add(line.rsplit("= ", 1)[1].strip())
    ok("dry-run three distinct fingerprints", len(ids) == 3)
    ok("dry-run no model load", "未加载模型" in out)

    bad = subprocess.run([sys.executable, os.path.join(HERE, "run_group_b2_v7.py"),
                          "FactConsolidation", "--dry-run"],
                         cwd=HERE, env={**os.environ, "CR_MODE": "v9",
                                        "CR_DELTA_MODE": "keep"},
                         capture_output=True, text=True, timeout=180)
    ok("invalid mode rejected", bad.returncode != 0
       and "CR_DELTA_MODE" in (bad.stdout + bad.stderr))


if __name__ == "__main__":
    test_delta_mode_parse()
    test_delta_plan()
    test_refresh_slots()
    test_fingerprint_isolated()
    test_block_fact_lines()
    test_diag_ctx_fields()
    test_diag_ctx_empty_pool()
    test_rec_fields_smoke()
    test_runner_dry_run()
    print("ALL CR DELTA/LOG TESTS PASSED")
