#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Accurate_Retrieval runner（v7 检索管线在"非冲突"任务族上的对照实验）

用法：
    /opt/conda/bin/python run_ar_v7.py eventqa_65536 3
    AR_MODE=plain /opt/conda/bin/python run_ar_v7.py longmemeval_s\\* 4

设计要点（为什么与 run_group_b2_v7.py 不同）：
  1. AR 没有"同一 (subj, rel) 的新值覆盖旧值"语义，证据必须全量保留。
     因此这里不复用 v7_slotpool.py 的覆写池，改为"分块证据池 ChunkPool"：
     唯一键去重、永不覆写、不做 FIFO 淘汰（容量随上下文长度弹性增长）。
  2. 检索单元从"事实三元组"改为"句子块"（默认 128 token）：
     - stage-1：块均值向量余弦，全池 GPU 矩阵乘（O(N·d)，N≈3k 也 <1ms）
     - stage-2：对 stage-1 的 Top-M 候选做 ColBERT MaxSim（query 逐 token 覆盖率）
       逐 token 向量常驻 CPU bf16（3k 块 ≈ 2GB），rerank 时只搬 M=64 块上 GPU
  3. AR_MODE 提供两档，同一脚本内形成对照：
     - plain：纯 top-K 块检索（≈ A chunked RAG 基线，无 rerank、无时序展开）
     - v7   ：MaxSim rerank + EventQA 时序窗口展开（定位"已发生事件"所在块，
              向后取窗口给"接下来发生什么"用）

支持的 source（parquet 内 metadata.source）：
    eventqa_65536 / eventqa_131072 / eventqa_full
    longmemeval_s* / longmemeval_s_-1_500（若存在）
  其它 source（ruler_*）需要 choice 专用指标，本脚本显式拒绝。

指标：直接调用 MemoryAgentBench 官方 post_process，与论文口径一致：
    - 全切片：substring_exact_match（SF）
    - eventqa：额外 eventqa_recall（6 选 1 全中）
"""
import sys, os, re, json, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/share/delta-Mem")
sys.path.insert(0, "/share/MemoryAgentBench")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltamem.core import HFDeltaMemConfig, attach_delta_mem
from deltamem.core.delta_impl import load_delta_mem_adapter
from deltamem.core.delta import reset_delta_mem_states, set_delta_mem_write_enabled
from deltamem.eval.official_memory_agent_bench_templates import get_template
from utils.eval_other_utils import chunk_text_into_sentences, post_process
import pyarrow.parquet as pq


# ----------------------------- 运行参数 -----------------------------
SRC   = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("AR_SRC", "")
GPU   = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("AR_GPU", "0")
MODE  = os.environ.get("AR_MODE", "v7")            # v7 | plain
CHUNK = int(os.environ.get("AR_CHUNK", "128"))     # 检索块 token 数
CHUNK_MAX_TOK = max(256, CHUNK * 2)                # 句子切分可能超出，硬截断
TOP_K       = int(os.environ.get("AR_TOP_K", "12"))        # 读出块数
RERANK_M    = int(os.environ.get("AR_RERANK_M", "64"))     # stage-2 候选数
EQA_WIN     = int(os.environ.get("AR_EQA_WIN", "4"))       # EventQA 锚点后取几块
CTX_MAX_TOK = int(os.environ.get("AR_CTX_MAX_TOK", "0"))   # 0=不截断，>0=保留尾部
DELTA_WRITE = os.environ.get("AR_DELTA_WRITE", "0") == "1"  # 是否照 CR 流程流式写 delta
LIMIT_ROWS  = int(os.environ.get("AR_LIMIT_ROWS", "0"))    # smoke：限制上下文数
LIMIT_Q     = int(os.environ.get("AR_LIMIT_Q", "0"))       # smoke：限制每题数
RESUME      = os.environ.get("AR_RESUME", "1") == "1"

MAX_NEW = {"eventqa": 40, "longmemeval": 50}

BASE    = "/share/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
ADAPTER = "/share/hf_home/hub/models--declare-lab--delta-mem_qwen3_4b-instruct/snapshots/6a1a990e770974811b8fb27dab73430807f76cb5"
PQ      = os.environ.get(
    "AR_PARQUET",
    "/share/latent/MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet",
)
OUT = os.environ.get(
    "AR_OUT",
    f"/share/exp4group/results/ar_{MODE}_{SRC.replace('*', 'star')}.jsonl",
)

def family_of(src):
    if src.startswith("eventqa"):
        return "eventqa"
    if src.startswith("longmemeval"):
        return "longmemeval"
    return None


FAMILY = family_of(SRC)
MAX_NEW_TOKENS = MAX_NEW.get(FAMILY, 50)
SYSTEM_TEMPLATE_NAME = "Long_context_agent_deltamem"


# ----------------------------- 模型（惰性初始化，便于 CPU 逻辑单测） -----------------------------
tok = None
model = None
EMB = None
DEV = None
_CLAIM = None   # 开机即占显存，避免被同机 GPU 调度器判为空闲抢走


def claim_gpu(mb=1024):
    """进程一起来就占住显存（>64MB 即不再算 idle），不与旁路调度器抢跑。"""
    global _CLAIM
    if DEV is not None or GPU is None:
        return
    g = int(GPU)
    dev = f"cuda:{g}"
    torch.cuda.init()
    torch.cuda.set_device(g)
    _CLAIM = torch.zeros(mb * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    print(f"[gpu] claimed {mb} MiB on {dev}", flush=True)


def init_model():
    global tok, model, EMB, DEV
    claim_gpu()
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map=f"cuda:{GPU}", attn_implementation="sdpa")
    cfg = HFDeltaMemConfig.from_pretrained(ADAPTER)
    attach_delta_mem(model, cfg)
    load_delta_mem_adapter(model, input_dir=ADAPTER)
    model.eval()
    EMB = model.get_input_embeddings().weight.detach()
    DEV = f"cuda:{GPU}"

sys_msg = mem_tpl = qry_tpl = None
DATASET_CFG = None


def init_templates():
    global sys_msg, mem_tpl, qry_tpl, DATASET_CFG
    sys_msg = get_template(SRC, "system", SYSTEM_TEMPLATE_NAME)
    mem_tpl = get_template(SRC, "memorize", SYSTEM_TEMPLATE_NAME)
    qry_tpl = get_template(SRC, "query", SYSTEM_TEMPLATE_NAME)
    DATASET_CFG = {"sub_dataset": SRC, "chunk_size": CHUNK, "debug": False}


def fmt(tpl, **kw):
    try:
        return tpl.format(time_stamp="", **kw)
    except KeyError:
        return tpl.format(**kw)


# ----------------------------- 证据池 -----------------------------
class ChunkPool:
    """分块证据池：只追加 + 文本去重，不覆写不淘汰。"""

    def __init__(self):
        self.texts = []
        self._seen = set()
        self.mean = []       # list[fp32 GPU (d,)]
        self.tokv = []       # list[cpu bf16 (L,d)]
        self.mean_mat = None
        self.n_dup = 0

    def __len__(self):
        return len(self.texts)

    def add_batch(self, chunks):
        chunks = [c for c in chunks if c and c.strip()]
        # 文本级去重（相同块在流内重复出现时只留一份）
        uniq, seen_local = [], set()
        for c in chunks:
            k = " ".join(c.split()).lower()
            if k in self._seen or k in seen_local:
                self.n_dup += 1
                continue
            seen_local.add(k)
            uniq.append(c)
        if not uniq:
            return
        enc = tok(uniq, return_tensors="pt", padding=True, truncation=True,
                  max_length=CHUNK_MAX_TOK, add_special_tokens=False)
        ids = enc.input_ids.to(EMB.device)
        am = enc.attention_mask.to(EMB.device)
        with torch.no_grad():
            ev = EMB[ids].float()                      # [B,L,d]
        w = am.float().unsqueeze(-1)
        m = (ev * w).sum(1) / w.sum(1).clamp(min=1e-6)
        m = F.normalize(m, dim=1)
        for b in range(len(uniq)):
            L = int(am[b].sum())
            self.texts.append(uniq[b])
            self._seen.add(" ".join(uniq[b].split()).lower())
            self.mean.append(m[b].to(DEV))
            self.tokv.append(F.normalize(ev[b, :L], dim=-1).to(torch.bfloat16).to("cpu"))
        self.mean_mat = None

    def _ensure_mat(self):
        if self.mean_mat is None and self.mean:
            self.mean_mat = torch.stack(self.mean)     # [N,d] fp32 GPU

    # ---- 查询侧编码 ----
    def encode_query(self, text):
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=64, add_special_tokens=False).input_ids[0].to(EMB.device)
        with torch.no_grad():
            qe = F.normalize(EMB[ids].float(), dim=-1)     # [Lq,d]
        qm = F.normalize(qe.mean(0), dim=0)
        return qe, qm

    def cov_maxsim(self, qe, idxs):
        """ColBERT 式 query 覆盖率：每个 query token 在块内取最大相似度后求均值。"""
        out = []
        for i in idxs:
            c = self.tokv[i].to(DEV).float()               # [L,d]
            out.append(float((c @ qe.T).max(0).values.mean()))
        return out

    def retrieve(self, query, top_k=TOP_K, rerank_m=RERANK_M, return_scores=False):
        self._ensure_mat()
        if len(self) == 0:
            return []
        qe, qm = self.encode_query(query)
        sims = self.mean_mat @ qm
        m = min(rerank_m, len(self))
        cand = sims.topk(m).indices.tolist()
        covs = self.cov_maxsim(qe, cand)
        order = sorted(zip(cand, covs), key=lambda x: -x[1])[:top_k]
        if return_scores:
            return order
        return [i for i, _ in order]

    def best_chunk_for(self, text, cand_m=32):
        """定位某段文本（如"已发生事件"）落在哪个块：mean 粗筛 + MaxSim 精排。
        返回 (chunk_idx, maxsim_coverage)。"""
        self._ensure_mat()
        qe, qm = self.encode_query(text)
        sims = self.mean_mat @ qm
        cand = sims.topk(min(cand_m, len(self))).indices.tolist()
        covs = self.cov_maxsim(qe, cand)
        best = max(zip(cand, covs), key=lambda x: x[1])
        return int(best[0]), float(best[1])


def build_pool_for_context(ctx):
    """把一个上下文切成句子块并灌入池；可选先按 token 截尾部。"""
    text = ctx
    if CTX_MAX_TOK > 0:
        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) > CTX_MAX_TOK:
            text = tok.decode(ids[-CTX_MAX_TOK:], skip_special_tokens=True)
    pool = ChunkPool()
    chunks = chunk_text_into_sentences(text, chunk_size=CHUNK)
    B = 24
    for i in range(0, len(chunks), B):
        pool.add_batch(chunks[i:i + B])
    return pool, len(tok(text, add_special_tokens=False).input_ids)


# ----------------------------- 读出块 -----------------------------
def parse_prev_events(meta, qi):
    evs = meta.get("previous_events")
    if not isinstance(evs, list) or qi >= len(evs) or not evs[qi]:
        return []
    out = []
    for line in str(evs[qi]).splitlines():
        m = re.match(r"\s*\d+\.\s*(.+)", line)
        if m:
            out.append(m.group(1).strip())
    return out


CORE_Q_RE = re.compile(r"(?:Now Answer the Question|Question)\s*[:：]\s*(.+)$", re.S)


def core_question(q):
    m = CORE_Q_RE.search(q)
    return m.group(1).strip() if m else q.strip()


def build_block(pool, q, meta, qi):
    """返回 (context_block, debug_info)。"""
    if len(pool) == 0:
        return "", {}

    if FAMILY == "eventqa":
        evs = parse_prev_events(meta, qi)
        if MODE == "v7" and evs:
            # 时序锚定：最后两个已发生事件 → 定位块 → 向后展开窗口
            anchor, cov = pool.best_chunk_for(evs[-1])
            used = "last"
            if cov < 0.5 and len(evs) > 1:        # 单事件锚不准 → 最后两事件拼接再锚一次
                a2, cov2 = pool.best_chunk_for(evs[-2] + " " + evs[-1])
                if cov2 > cov:
                    anchor, cov, used = a2, cov2, "last2"
            idxs = list(range(anchor, min(anchor + 1 + EQA_WIN, len(pool))))
            block = "\n".join(pool.texts[i] for i in idxs)
            return block, {"mode": "anchor", "anchor": anchor, "anchor_cov": round(cov, 4),
                           "anchor_used": used, "idxs": idxs}
        idxs = sorted(pool.retrieve(core_question(q), top_k=TOP_K))
        return "\n".join(pool.texts[i] for i in idxs), {"mode": "topk", "idxs": idxs}

    q_core = core_question(q)
    if MODE == "v7":
        idxs = sorted(pool.retrieve(q_core, top_k=TOP_K))
        return "\n".join(pool.texts[i] for i in idxs), {"mode": "maxsim", "idxs": idxs}
    # plain：纯均值向量 top-K，不 rerank
    pool._ensure_mat()
    _, qm = pool.encode_query(q_core)
    sims = pool.mean_mat @ qm
    idxs = sorted(sims.topk(min(TOP_K, len(pool))).indices.tolist())
    return "\n".join(pool.texts[i] for i in idxs), {"mode": "mean_topk", "idxs": idxs}


# ----------------------------- 主流程 -----------------------------
def main():
    global FAMILY, MAX_NEW_TOKENS
    if not SRC:
        raise SystemExit("usage: python run_ar_v7.py <eventqa_65536|longmemeval_s*> <GPU_ID>")
    FAMILY = family_of(SRC)
    if FAMILY is None:
        raise SystemExit(f"[ERR] 不支持的 source={SRC}；只支持 eventqa_* / longmemeval_*"
                         f"（ruler_* 需 choice 指标）")
    MAX_NEW_TOKENS = MAX_NEW[FAMILY]
    init_templates()
    claim_gpu()     # 先占显存，再慢慢加载权重
    init_model()
    rows_all = pq.read_table(PQ).to_pylist()
    rows = [x for x in rows_all if (x.get("metadata") or {}).get("source") == SRC]
    if not rows:
        raise SystemExit(f"[ERR] parquet 内没有 source={SRC} 的行")
    if LIMIT_ROWS > 0:
        rows = rows[:LIMIT_ROWS]
    print(f"[cfg] SRC={SRC} MODE={MODE} CHUNK={CHUNK} TOP_K={TOP_K} RERANK_M={RERANK_M} "
          f"EQA_WIN={EQA_WIN} CTX_MAX_TOK={CTX_MAX_TOK} rows={len(rows)} OUT={OUT}", flush=True)

    done = set()
    prev_sf = []
    if RESUME and os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r.get("context_id"), r.get("question_idx")))
                prev_sf.append(r.get("substring_exact_match", 0))
        print(f"[resume] 已完成 {len(done)} 题", flush=True)
    elif os.path.exists(OUT) and not RESUME:
        os.remove(OUT)

    results = []
    t_start = time.time()
    for ci, row in enumerate(rows):
        ctx = str(row.get("context", ""))
        qs = list(row.get("questions") or [])
        ans = list(row.get("answers") or [])
        meta = row.get("metadata") or {}
        if LIMIT_Q > 0:
            qs, ans = qs[:LIMIT_Q], ans[:LIMIT_Q]

        print(f"[ctx {ci+1}/{len(rows)}] chars={len(ctx)} qs={len(qs)} 构建证据池...", flush=True)
        pool, n_tok = build_pool_for_context(ctx)
        print(f"[ctx {ci+1}/{len(rows)}] tokens={n_tok} chunks={len(pool)} dup={pool.n_dup} "
              f"({time.time()-t_start:.0f}s)", flush=True)

        reset_delta_mem_states(model)
        set_delta_mem_write_enabled(model, False)

        for qi, q in enumerate(qs):
            if (ci, qi) in done:
                continue
            if qi >= len(ans):
                break
            a = ans[qi]
            try:
                qtxt = qry_tpl.format(question=q, label=(a[0] if isinstance(a, list) and a else a))
            except KeyError:
                qtxt = qry_tpl.format(question=q)

            try:
                block, dbg = build_block(pool, q, meta, qi)
            except Exception as e:                      # 单题失败不中断整轮
                print(f"[warn] retrieve fail q={qi}: {e}", flush=True)
                block, dbg = "", {"mode": "error", "err": str(e)}

            text = tok.apply_chat_template(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": f"{block}\n\n{qtxt}" if block else qtxt}],
                add_generation_prompt=True, tokenize=False)
            ids = tok(text, return_tensors="pt").input_ids.to(DEV)
            with torch.no_grad():
                out = model.generate(input_ids=ids, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False, use_cache=True,
                                     pad_token_id=tok.eos_token_id or 0)
            pred = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

            m, extra = post_process(
                {"output": pred, "input_len": ids.shape[1],
                 "output_len": out.shape[1] - ids.shape[1],
                 "memory_construction_time": 0, "query_time_len": 0},
                a, DATASET_CFG)
            qtypes = meta.get("question_types")
            rec = {
                "context_id": ci, "question_idx": qi,
                "qa_pair_id": ((meta.get("qa_pair_ids") or [])[qi]
                               if qi < len(meta.get("qa_pair_ids") or []) else None),
                "question_type": qtypes[qi] if isinstance(qtypes, list) and qi < len(qtypes) else None,
                "query": q, "answer": a, "output": pred,
                "retrieval": dbg, "block_chunks": len(dbg.get("idxs", []) or []),
                **{k: v for k, v in m.items()}, **extra,
            }
            results.append(rec)
            with open(OUT, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (len(results)) % 20 == 0:
                sf = sum(r.get("substring_exact_match", 0) for r in results) / len(results)
                err = sum(r.get("eventqa_recall", 0) for r in results) / len(results) if FAMILY == "eventqa" else 0.0
                print(f"[{ci+1}/{len(rows)}] {len(results)} 题 running SF={sf:.3f} "
                      f"eqa={err:.3f} ({time.time()-t_start:.0f}s)", flush=True)
        if DELTA_WRITE:
            set_delta_mem_write_enabled(model, True)
            for ch in chunk_text_into_sentences(ctx, chunk_size=512):
                txt = tok.apply_chat_template(
                    [{"role": "system", "content": sys_msg},
                     {"role": "user", "content": fmt(mem_tpl, context=ch)}],
                    add_generation_prompt=True, tokenize=False)
                ii = tok(txt, return_tensors="pt").input_ids.to(DEV)
                model(input_ids=ii, use_cache=True)
            set_delta_mem_write_enabled(model, False)

    # 汇总（含已有记录，便于断点续跑）
    all_recs = []
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                if line.strip():
                    all_recs.append(json.loads(line))
    n = len(all_recs)
    sf_all = sum(r.get("substring_exact_match", 0) for r in all_recs) / max(n, 1)
    eqa_all = (sum(r.get("eventqa_recall", 0) for r in all_recs) / max(n, 1)) if FAMILY == "eventqa" else 0.0
    print(f"FINAL ar_{MODE} {SRC}: SF={sf_all:.4f} eventqa_recall={eqa_all:.4f} n={n}", flush=True)

    by_type = {}
    for r in all_recs:
        t = r.get("question_type")
        if not t:
            continue
        by_type.setdefault(t, []).append(r.get("substring_exact_match", 0))
    if by_type:
        print("[by question_type] " + "  ".join(
            f"{t}={sum(v)/len(v):.3f}(n={len(v)})" for t, v in sorted(by_type.items())), flush=True)


if __name__ == "__main__":
    main()
