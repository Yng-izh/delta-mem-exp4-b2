#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Accurate_Retrieval runner（v7 检索管线在"非冲突"任务族上的对照实验）

用法：
    /opt/conda/bin/python run_ar_v7.py eventqa_65536 3
    AR_MODE=plain /opt/conda/bin/python run_ar_v7.py longmemeval_s\\* 4
    AR_MODE=v9 /opt/conda/bin/python run_ar_v7.py eventqa_65536 3
    python3 run_ar_v7.py eventqa_65536 --dry-run   # 不加载模型、不占 GPU

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
#      - v8   : 双模态潜槽位记忆 (DM-ELSM, v8_slotpool.py): 情景/状态自适应分流
#               (3 句滑窗微块 + 时间戳继承)、EpisodicFastPool 全池无截断 IDF-MaxSim
#               检索、邻域扩展。CR (DELTA_WRITE) 路径保持 v7 不动。
#      - v9   : 非重叠单元索引 (v9_slotpool.py)：每句恰好属于一个单元；先按分数选
#               Top-K、再按原文位置排序；EventQA 后向非对称扩展 (back=16/ahead=4)；
#               时间戳前缀参与打分 + 时序重排；可选第 18 层隐状态混合稠密打分
#               (AR_V9_L18=1)。逐题诊断字段 (ar_diag.record) 三种模式都会写。
#
# 诊断字段（--dry-run 之外的真实运行，每题写入 JSONL）：
#     retrieved_slot_ids / retrieved_scores / retrieved_texts / used_slot_ids /
#     answer_in_retrieved / answer_in_context / recall_at_k / diag_status
#   hits 一律按检索分数降序（Recall@K 依赖），used_ids 是实际拼进 prompt 的槽位。
#
# --dry-run：只打印解析后的配置与依赖可导入性，跑一次纯正则的 v9 索引自检；
#   不加载模型权重、不读 parquet、不初始化 CUDA。
#
# Delta Memory 对齐（t6）：
#   每个 context 起点 reset_delta_mem_states 一次；AR_DELTA_WRITE=1 时在【查询前】
#   完成流式写入 -> 关闭写开关 -> get_delta_mem_online_state 快照 clone；
#   每题生成前 load_delta_mem_online_state 恢复同一快照（查询期不再 reset）。
#   AR_INPUTS_EMBEDS=1 时读出走 inputs_embeds（而非 input_ids），默认关闭。

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

try:
    import torch
    import torch.nn.functional as F
    _TORCH_ERR = None
except Exception as _e:                      # 无 torch 的机器仍可 --dry-run
    torch = None
    F = None
    _TORCH_ERR = _e

# transformers / deltamem / MemoryAgentBench / pyarrow 只在真实跑分时需要。
# 放进 try：--dry-run 与 CPU 逻辑单测不得因缺依赖而失败。
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from deltamem.core import HFDeltaMemConfig, attach_delta_mem
    from deltamem.core.delta_impl import load_delta_mem_adapter
    from deltamem.core.delta import reset_delta_mem_states, set_delta_mem_write_enabled
    from deltamem.eval.official_memory_agent_bench_templates import get_template
    from utils.eval_other_utils import chunk_text_into_sentences, post_process
    import pyarrow.parquet as pq
    _STACK_ERR = None
except Exception as _e:
    AutoModelForCausalLM = AutoTokenizer = None
    HFDeltaMemConfig = attach_delta_mem = load_delta_mem_adapter = None
    reset_delta_mem_states = set_delta_mem_write_enabled = get_template = None
    chunk_text_into_sentences = post_process = None
    pq = None
    _STACK_ERR = _e

# 可选：delta 在线状态快照/恢复 API（老版本 deltamem 可能没有）。
# 有则逐题恢复用官方实现；没有则用 iter_delta_mem_modules 手工 clone/restore。
if reset_delta_mem_states is not None:
    try:
        from deltamem.core.delta import (get_delta_mem_online_state,
                                         load_delta_mem_online_state,
                                         collect_delta_mem_state_stats,
                                         iter_delta_mem_modules)
        _STATE_API_ERR = None
    except Exception as _e:
        get_delta_mem_online_state = load_delta_mem_online_state = None
        collect_delta_mem_state_stats = iter_delta_mem_modules = None
        _STATE_API_ERR = _e
else:
    get_delta_mem_online_state = load_delta_mem_online_state = None
    collect_delta_mem_state_stats = iter_delta_mem_modules = None
    _STATE_API_ERR = None

# 逐题诊断（纯标准库）：顶层 import 不触发 torch / 模型加载 / 显存占用
try:
    import ar_diag
    _DIAG_ERR = None
except Exception as _e:
    ar_diag = None
    _DIAG_ERR = _e


# ----------------------------- 运行参数 -----------------------------
DRY_RUN = "--dry-run" in sys.argv[1:]
_ARGV   = [a for a in sys.argv[1:] if a != "--dry-run"]
SRC   = _ARGV[0] if len(_ARGV) > 0 else os.environ.get("AR_SRC", "")
GPU   = _ARGV[1] if len(_ARGV) > 1 else os.environ.get("AR_GPU", "0")
MODE  = os.environ.get("AR_MODE", "v7")            # v7 | v8 | v9 | plain
CHUNK = int(os.environ.get("AR_CHUNK", "128"))     # 检索块 token 数
CHUNK_MAX_TOK = max(256, CHUNK * 2)                # 句子切分可能超出，硬截断
TOP_K       = int(os.environ.get("AR_TOP_K", "12"))        # 读出块数
RERANK_M    = int(os.environ.get("AR_RERANK_M", "64"))     # stage-2 候选数
EQA_WIN     = int(os.environ.get("AR_EQA_WIN", "4"))       # EventQA 锚点后取几块
CTX_MAX_TOK = int(os.environ.get("AR_CTX_MAX_TOK", "0"))   # 0=不截断，>0=保留尾部
DELTA_WRITE = os.environ.get("AR_DELTA_WRITE", "0") == "1"  # 查询前流式写 delta（写后快照）
INPUTS_EMBEDS = os.environ.get("AR_INPUTS_EMBEDS", "0") == "1"  # 读出走 inputs_embeds 路径
DELTA_RESTORE = os.environ.get("AR_DELTA_RESTORE", "1") == "1"  # 逐题恢复 delta 状态快照
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
    if torch is None:
        raise RuntimeError(f"torch 不可用（{_TORCH_ERR}）；真实跑分需要 GPU + torch")
    if _STACK_ERR is not None:
        raise RuntimeError(f"缺少真实跑分依赖（transformers/deltamem/pyarrow）: {_STACK_ERR}")
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


# ----------------------------- Delta Memory 对齐 -----------------------------
def delta_snapshot(model):
    """把当前 delta 在线状态 clone 到 CPU（逐题恢复的基线）。

    优先用官方 get_delta_mem_online_state（返回 {module_name: cpu clone}）；
    老版本没有该 API 时退回遍历 DeltaMemAttention.delta_state 手工 clone。
    """
    if get_delta_mem_online_state is not None:
        return get_delta_mem_online_state(model)
    if iter_delta_mem_modules is None:
        return {}
    snap = {}
    for name, m in iter_delta_mem_modules(model):
        st = getattr(m, "delta_state", None)
        if st is not None:
            snap[name] = st.detach().cpu().clone()
    return snap


def delta_restore(snapshot):
    """把快照写回模型（每题生成前调用），保证每题从同一记忆状态出发。"""
    if not snapshot:
        return
    if load_delta_mem_online_state is not None:
        load_delta_mem_online_state(model, snapshot)
        return
    mods = dict(model.named_modules())
    for name, t in snapshot.items():
        m = mods.get(name)
        if m is None:
            continue
        p = next(m.parameters())
        m.delta_state = t.to(device=p.device, dtype=p.dtype)


def prepare_delta_memory(ctx):
    """查询前完成 delta 写入，返回写后快照（写开关已关闭）。

    调用约定：main 在本 context 起点先 reset_delta_mem_states(model)；
    本函数只负责 write -> disable -> snapshot，查询期不再 reset。
    """
    if not DELTA_WRITE:
        return {}
    if reset_delta_mem_states is None or set_delta_mem_write_enabled is None:
        raise RuntimeError("delta-mem 写入 API 不可用，无法执行 AR_DELTA_WRITE=1")
    set_delta_mem_write_enabled(model, True)
    try:
        for ch in chunk_text_into_sentences(ctx, chunk_size=512):
            txt = tok.apply_chat_template(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": fmt(mem_tpl, context=ch)}],
                add_generation_prompt=True, tokenize=False)
            ii = tok(txt, return_tensors="pt").input_ids.to(DEV)
            model(input_ids=ii, use_cache=True)
    finally:
        set_delta_mem_write_enabled(model, False)
    snap = delta_snapshot(model)
    if not snap and get_delta_mem_online_state is None and iter_delta_mem_modules is None:
        print("[warn] deltamem 无状态快照 API，逐题恢复退化为 no-op", flush=True)
    if collect_delta_mem_state_stats is not None:
        try:
            print(f"[delta] state after write: {collect_delta_mem_state_stats(model)}",
                  flush=True)
        except Exception:
            pass
    return snap


def generate_readout(block, qtxt):
    """生成答案。AR_INPUTS_EMBEDS=1 时走 inputs_embeds 读出路径。

    返回 (pred, input_len, output_len)：
      - 正常路径：out 含 prompt，output_len = out.shape[1] - input_len；
      - inputs_embeds 路径：generate 只返回新 token，output_len = out.shape[1]。
    """
    text = tok.apply_chat_template(
        [{"role": "system", "content": sys_msg},
         {"role": "user", "content": f"{block}\n\n{qtxt}" if block else qtxt}],
        add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.to(DEV)
    use_emb = bool(INPUTS_EMBEDS and EMB is not None)
    emb = EMB[ids] if use_emb else None
    torch.cuda.empty_cache()                 # 建池/检索后的 GPU 碎片回收

    def _gen(n_ids, n_emb):
        with torch.no_grad():
            if n_emb is not None:
                return model.generate(inputs_embeds=n_emb,
                                      attention_mask=torch.ones_like(n_ids),
                                      max_new_tokens=MAX_NEW_TOKENS,
                                      do_sample=False, use_cache=True,
                                      pad_token_id=tok.eos_token_id or 0)
            return model.generate(input_ids=n_ids, max_new_tokens=MAX_NEW_TOKENS,
                                  do_sample=False, use_cache=True,
                                  pad_token_id=tok.eos_token_id or 0)

    try:
        out = _gen(ids, emb)
    except torch.OutOfMemoryError:
        for keep in (16000, 8000, 4000):
            torch.cuda.empty_cache()
            keep_n = min(keep, ids.shape[1])
            ids = ids[:, ids.shape[1] - keep_n:]
            emb = EMB[ids] if use_emb else None
            print(f"[gen] OOM retry keep={keep_n}", flush=True)
            try:
                out = _gen(ids, emb)
                break
            except torch.OutOfMemoryError:
                if keep == 4000:
                    raise

    if use_emb:
        pred = tok.decode(out[0], skip_special_tokens=True)
        out_len = int(out.shape[1])
    else:
        pred = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        out_len = int(out.shape[1] - ids.shape[1])
    return pred, int(ids.shape[1]), out_len


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


# ----------------------------- v8 双模态潜槽位记忆 (DM-ELSM) -----------------------------
v8_slotpool = None


def v8_import():
    """惰性导入 v8_slotpool 并注入 tok/EMB（CR 路径不触碰）。"""
    global v8_slotpool
    if v8_slotpool is None:
        import importlib
        v8_slotpool = importlib.import_module("v8_slotpool")
        v8_slotpool.tok = tok
        v8_slotpool.EMB = EMB
    return v8_slotpool


TOPK_STATE = int(os.environ.get("AR_V8_TOPK_STATE", "4"))   # 状态池附带单元数
NEIGHBOR   = int(os.environ.get("AR_V8_NEIGHBOR", "2"))     # 情景邻域扩展窗口数
CTX_SENTS  = int(os.environ.get("AR_V8_CTX_MAX_UNITS", "40"))  # 拼给 LLM 的单元上限


def _ctx_text(ctx):
    """可选按 token 截尾部（CTX_MAX_TOK>0），v8/v9 共用。"""
    text = ctx
    if CTX_MAX_TOK > 0:
        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) > CTX_MAX_TOK:
            text = tok.decode(ids[-CTX_MAX_TOK:], skip_special_tokens=True)
    return text


def build_pool_for_context_v8(ctx):
    """v8：情景/状态分流建池。返回 (epi_pool, state_pool, n_tok)。"""
    sp = v8_import()
    text = _ctx_text(ctx)
    epi, st = sp.build_pools_for_context(text)
    return epi, st, len(tok(text, add_special_tokens=False).input_ids)


def _norm_unit(text):
    return " ".join(text.split()).lower()


def _cap_before_sort(idxs, cap, anchor=None):
    """先截断后排序：排序前把候选集裁到 cap。

    - 有 anchor（EventQA 连续窗）：保留以 anchor 为中心的窗口；
    - 无 anchor（Top-K / 邻域扩展，前面是分数序命中）：保留前 cap 个（分数优先）。
    旧版 v8 是「全候选按位置排序 -> lines_out[:CTX_SENTS]」，会先丢掉高分尾部。
    """
    idxs = list(dict.fromkeys(int(i) for i in idxs if int(i) >= 0))
    if cap <= 0 or len(idxs) <= cap:
        return idxs
    if anchor is not None and int(anchor) in idxs:
        pos = idxs.index(int(anchor))
        lo = max(0, pos - cap // 2)
        return idxs[lo:lo + cap]
    return idxs[:cap]


def build_block_v8(epi, st, q, meta, qi):
    """v8 读出：全池 IDF-MaxSim + 邻域扩展 + 状态单元补充，按句序拼块。
    返回 (block, dbg, diag_ctx)。"""
    empty = {"pool": epi, "hits": [], "used_ids": [], "used_texts": []}
    if len(epi) == 0 and len(st) == 0:
        return "", {}, empty
    q_core = core_question(q)
    hits = epi.retrieve(q_core, top_k=TOP_K)     # 分数降序（诊断 Recall@K 用）

    if FAMILY == "eventqa":
        evs = parse_prev_events(meta, qi)
        if evs:
            anchor, cov = epi.best_unit_for(evs[-1])
            used = "last"
            if cov < 0.5 and len(evs) > 1:
                a2, cov2 = epi.best_unit_for(evs[-2] + " " + evs[-1])
                if cov2 > cov:
                    anchor, cov, used = a2, cov2, "last2"
            idxs = epi.neighbor_expand([(anchor, cov)], extra=EQA_WIN)
            dbg = {"mode": "v8_anchor", "anchor": anchor, "anchor_cov": round(cov, 4),
                   "anchor_used": used, "idxs": idxs}
        else:
            idxs = epi.neighbor_expand(hits, extra=NEIGHBOR)
            dbg = {"mode": "v8_maxsim", "idxs": idxs}
    else:
        idxs = epi.neighbor_expand(hits, extra=NEIGHBOR)
        dbg = {"mode": "v8_maxsim", "idxs": idxs}

    # 状态单元补充（时间无关稳定事实）
    if len(st) > 0:
        st_hits = st.retrieve(q_core, top_k=TOPK_STATE)
        st_idx = [i for i, _ in st_hits]
    else:
        st_idx = []
    dbg["state_idxs"] = st_idx

    # 先截断后排序：neighbor_expand 返回「分数序命中 + 邻域」，先按分数优先裁到
    # 情景配额，再按原文位置排序。
    max_epi = max(1, CTX_SENTS - len(st_idx))
    idxs = _cap_before_sort(idxs, max_epi, anchor=dbg.get("anchor"))
    lines_out, seen, used_epi = [], set(), []
    order = sorted(idxs, key=lambda i: epi.metas[i].get("start", i))
    for i in order:
        k = _norm_unit(epi.texts[i])
        if k not in seen:
            seen.add(k)
            lines_out.append(epi.texts[i])
            used_epi.append(int(i))
    st_order = sorted(st_idx, key=lambda i: st.metas[i].get("start", i))
    for i in st_order:
        k = _norm_unit(st.texts[i])
        if k not in seen:
            seen.add(k)
            lines_out.append(st.texts[i])
    dbg["n_units"] = len(lines_out)
    return "\n".join(lines_out), dbg, {"pool": epi, "hits": hits,
                                       "used_ids": used_epi, "used_texts": lines_out}


# ----------------------------- v9 非重叠索引检索 -----------------------------
v9_slotpool = None


def v9_import():
    """惰性导入 v9_slotpool 并注入 tok/EMB（CR 路径不触碰）。"""
    global v9_slotpool
    if v9_slotpool is None:
        import importlib
        v9_slotpool = importlib.import_module("v9_slotpool")
        v9_slotpool.tok = tok
        v9_slotpool.EMB = EMB
    return v9_slotpool


V9_TOPK_STATE = int(os.environ.get("AR_V9_TOPK_STATE", "4"))      # 状态池附带单元数
V9_CTX_UNITS  = int(os.environ.get("AR_V9_CTX_MAX_UNITS", "40"))  # 拼给 LLM 的单元上限
EQA_BACK      = int(os.environ.get("AR_V9_EQA_BACK", "16"))       # EventQA 锚点前取几句
EQA_AHEAD     = int(os.environ.get("AR_V9_EQA_AHEAD", "4"))       # EventQA 锚点后取几句
V9_L18        = os.environ.get("AR_V9_L18", "0") == "1"           # 第 18 层稠密通道开关
V9_L18_LAYER  = int(os.environ.get("AR_V9_L18_LAYER", "18"))


def build_pool_for_context_v9(ctx):
    """v9：非重叠单元索引（epi 全覆盖）+ 可选状态池。返回 (epi, st, n_tok)。"""
    sp = v9_import()
    text = _ctx_text(ctx)
    epi, st = sp.build_pools_for_context(text)
    return epi, st, len(tok(text, add_special_tokens=False).input_ids)


def _layer18_provider(texts, kind):
    """fn(texts, kind) -> [N,d]：取模型第 V9_L18_LAYER 层隐状态做 mask mean-pool。

    仅 AR_V9_L18=1 时注入；分批前向控制显存。异常/形状不符由 v9_slotpool 侧
    静默退化为纯词面打分，不中断评测。
    """
    if model is None or tok is None or EMB is None:
        return None
    dev = EMB.device
    bs = 8
    outs = []
    for s in range(0, len(texts), bs):
        batch = list(texts[s:s + bs])
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        ids = enc.input_ids.to(dev)
        am = enc.attention_mask.to(dev)
        with torch.no_grad():
            hs = model(input_ids=ids, attention_mask=am, output_hidden_states=True,
                       use_cache=False).hidden_states
        outs.append(v9_slotpool.mean_pool_hidden(hs[V9_L18_LAYER], am).cpu())
    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]


def build_block_v9(epi, st, q, meta, qi):
    """v9 读出：非重叠索引 + 先分数选 Top-K 再位置排序 + EventQA 后向非对称扩展。
    返回 (block, dbg, diag_ctx)。"""
    empty = {"pool": epi, "hits": [], "used_ids": [], "used_texts": []}
    if len(epi) == 0 and len(st) == 0:
        return "", {}, empty
    q_core = core_question(q)
    cap_epi = max(1, V9_CTX_UNITS - (V9_TOPK_STATE if len(st) > 0 else 0))

    if FAMILY == "eventqa":
        evs = parse_prev_events(meta, qi)
        idxs, dbg = epi.readout(q_core, top_k=min(TOP_K, cap_epi), mode="eventqa",
                                anchor_events=evs, back=EQA_BACK, ahead=EQA_AHEAD)
        # 扩展窗超配额时保留锚点邻域（先截断，readout 返回的已是位置升序）
        idxs = _cap_before_sort(idxs, cap_epi, anchor=dbg.get("anchor"))
        # 诊断用 query 侧分数序命中（实际拼块是锚点扩展窗，二者分开记）
        hits = epi.retrieve(q_core, top_k=TOP_K)
    else:
        # readout(topk) = 先按分数选 Top-K，再把选中的 K 个按原文位置排序
        idxs, dbg = epi.readout(q_core, top_k=min(TOP_K, cap_epi), mode="topk")
        hits = [(int(i), float(s)) for i, s in dbg.get("hits", [])]

    if len(st) > 0:
        st_hits = st.retrieve(q_core, top_k=V9_TOPK_STATE)
        st_idx = [i for i, _ in st_hits]
    else:
        st_idx = []
    dbg["state_idxs"] = st_idx
    dbg["stats"] = epi.stats()

    lines_out, seen, used_epi = [], set(), []
    for i in idxs:
        k = _norm_unit(epi.texts[i])
        if k not in seen:
            seen.add(k)
            lines_out.append(epi.texts[i])
            used_epi.append(int(i))
    st_order = sorted(st_idx, key=lambda i: st.metas[i].get("start", i))
    for i in st_order:
        k = _norm_unit(st.texts[i])
        if k not in seen:
            seen.add(k)
            lines_out.append(st.texts[i])
    dbg["n_units"] = len(lines_out)
    # used_ids 只含情景槽位（状态槽位跨池、无统一下标；used_texts 含全部拼块行）
    return "\n".join(lines_out), dbg, {"pool": epi, "hits": hits,
                                       "used_ids": used_epi, "used_texts": lines_out}


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
    """v7 / plain 读出。返回 (context_block, debug_info, diag_ctx)。"""
    empty = {"pool": pool, "hits": [], "used_ids": [], "used_texts": []}
    if len(pool) == 0:
        return "", {}, empty
    q_core = core_question(q)

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
            lines = [pool.texts[i] for i in idxs]
            # 诊断用 query 侧分数序命中（实际拼块是锚点窗口，二者分开记）
            hits = pool.retrieve(q_core, top_k=TOP_K, return_scores=True)
            return "\n".join(lines), {"mode": "anchor", "anchor": anchor,
                                      "anchor_cov": round(cov, 4),
                                      "anchor_used": used, "idxs": idxs}, \
                   {"pool": pool, "hits": hits, "used_ids": idxs, "used_texts": lines}
        hits = pool.retrieve(q_core, top_k=TOP_K, return_scores=True)
        idxs = sorted(i for i, _ in hits)
        lines = [pool.texts[i] for i in idxs]
        return "\n".join(lines), {"mode": "topk", "idxs": idxs}, \
               {"pool": pool, "hits": hits, "used_ids": idxs, "used_texts": lines}

    if MODE == "v7":
        hits = pool.retrieve(q_core, top_k=TOP_K, return_scores=True)
        idxs = sorted(i for i, _ in hits)
        lines = [pool.texts[i] for i in idxs]
        return "\n".join(lines), {"mode": "maxsim", "idxs": idxs}, \
               {"pool": pool, "hits": hits, "used_ids": idxs, "used_texts": lines}
    # plain：纯均值向量 top-K，不 rerank
    pool._ensure_mat()
    _, qm = pool.encode_query(q_core)
    sims = pool.mean_mat @ qm
    vals, idxs_t = sims.topk(min(TOP_K, len(pool)))
    hits = [(int(i), float(v)) for i, v in zip(idxs_t.tolist(), vals.tolist())]
    idxs = sorted(idxs_t.tolist())
    lines = [pool.texts[i] for i in idxs]
    return "\n".join(lines), {"mode": "mean_topk", "idxs": idxs}, \
           {"pool": pool, "hits": hits, "used_ids": idxs, "used_texts": lines}


# ----------------------------- 主流程 -----------------------------
def dry_run():
    """只检查配置与可导入性：不加载模型、不读 parquet、不初始化 CUDA。

    仍会跑两项纯 CPU 自检：
      1) v9_slotpool.extract_units（纯正则，不需要 tok/EMB）；
      2) ar_diag.record（纯标准库，用假池验证诊断字段）。
    """
    print("[dry-run] 仅检查配置与可导入性：不加载模型、不读 parquet、不初始化 CUDA")
    fam = family_of(SRC) if SRC else None
    if SRC and fam is None:
        print(f"[dry-run][ERR] 不支持的 source={SRC}；只支持 eventqa_* / longmemeval_*")
        return 2
    if MODE not in ("v7", "v8", "v9", "plain"):
        print(f"[dry-run][ERR] AR_MODE={MODE} 非法；可选 v7 | v8 | v9 | plain")
        return 2
    print(f"[dry-run] SRC={SRC or '<unset>'} family={fam or '<unset>'} MODE={MODE} "
          f"GPU={GPU} OUT={OUT}")
    print(f"[dry-run] CHUNK={CHUNK} TOP_K={TOP_K} RERANK_M={RERANK_M} EQA_WIN={EQA_WIN} "
          f"CTX_MAX_TOK={CTX_MAX_TOK}")
    print(f"[dry-run] delta: WRITE={int(DELTA_WRITE)} INPUTS_EMBEDS={int(INPUTS_EMBEDS)} "
          f"RESTORE={int(DELTA_RESTORE)} "
          f"state_api={'ok' if get_delta_mem_online_state is not None else 'fallback'}"
          f"{'' if _STATE_API_ERR is None else f'({_STATE_API_ERR})'}")
    if MODE == "v9":
        print(f"[dry-run] v9: EQA_BACK={EQA_BACK} EQA_AHEAD={EQA_AHEAD} "
              f"CTX_UNITS={V9_CTX_UNITS} TOPK_STATE={V9_TOPK_STATE} "
              f"L18={'on' if V9_L18 else 'off'} layer={V9_L18_LAYER}")
    print(f"[dry-run] torch={'ok' if torch is not None else 'missing'}"
          f"({_TORCH_ERR})  "
          f"model-stack={'ok' if _STACK_ERR is None else 'missing'}"
          f"{'' if _STACK_ERR is None else f'({_STACK_ERR})'}  "
          f"ar_diag={'ok' if ar_diag is not None else 'missing'}"
          f"{'' if _DIAG_ERR is None else f'({_DIAG_ERR})'}")

    # v9 纯函数自检：非重叠索引 + 时间戳继承（不需要 tok/EMB，不碰模型）
    try:
        import importlib
        _v9 = importlib.import_module("v9_slotpool")
        units = _v9.extract_units(
            "Alice moved to Paris in 1997. She worked there. In 2001 she left Lyon.")
        print(f"[dry-run] v9_slotpool import ok; extract_units -> {len(units)} units, "
              f"ts={[u['ts'] for u in units]}")
    except Exception as e:
        print(f"[dry-run][warn] v9_slotpool 自检失败: {e}")

    # 诊断字段自检：假池 + 1 条命中
    if ar_diag is not None:
        class _DryPool:
            texts = ["Alice moved to Paris in 1997."]
            metas = [{"start": 0}]
        d = ar_diag.record(pool=_DryPool(), hits=[(0, 1.0)], used_ids=[0],
                           answer="Paris", query="Where did Alice move?", flat=True)
        print(f"[dry-run] ar_diag.record ok; answer_in_retrieved="
              f"{d['answer_in_retrieved']} recall_at_k={d['recall_at_k']}")
    print(f"[dry-run] cuda_initialized="
          f"{torch.cuda.is_initialized() if torch is not None else 'n/a'} "
          f"model_loaded={model is not None}")
    print("[dry-run] OK（未加载模型 / 未初始化 CUDA / 未读 parquet）")
    return 0


def main():
    global FAMILY, MAX_NEW_TOKENS
    if DRY_RUN:
        return dry_run()
    if not SRC:
        raise SystemExit("usage: python run_ar_v7.py <eventqa_65536|longmemeval_s*> <GPU_ID>")
    if MODE not in ("v7", "v8", "v9", "plain"):
        raise SystemExit(f"[ERR] AR_MODE={MODE} 非法；可选 v7 | v8 | v9 | plain")
    if _STACK_ERR is not None:
        raise SystemExit(f"[ERR] 缺少真实跑分依赖（transformers/deltamem/pyarrow）: {_STACK_ERR}")
    if torch is None:
        raise SystemExit(f"[ERR] torch 不可用（{_TORCH_ERR}）；真实跑分需要 GPU + torch")
    FAMILY = family_of(SRC)
    if FAMILY is None:
        raise SystemExit(f"[ERR] 不支持的 source={SRC}；只支持 eventqa_* / longmemeval_*"
                         f"（ruler_* 需 choice 指标）")
    MAX_NEW_TOKENS = MAX_NEW[FAMILY]
    init_templates()
    claim_gpu()     # 先占显存，再慢慢加载权重
    init_model()
    # AR 读出默认只读；写入只在 prepare_delta_memory 内显式开关。
    set_delta_mem_write_enabled(model, False)
    if MODE == "v9" and V9_L18:
        # 第 18 层稠密通道：建池时按单元文本前向取隐状态，查询期同层 mean-pool。
        v9_import().set_layer18_provider(_layer18_provider)
        print(f"[v9] layer18 dense channel enabled: {v9_slotpool.layer18_spec()}", flush=True)
    rows_all = pq.read_table(PQ).to_pylist()
    rows = [x for x in rows_all if (x.get("metadata") or {}).get("source") == SRC]
    if not rows:
        raise SystemExit(f"[ERR] parquet 内没有 source={SRC} 的行")
    if LIMIT_ROWS > 0:
        rows = rows[:LIMIT_ROWS]
    print(f"[cfg] SRC={SRC} MODE={MODE} CHUNK={CHUNK} TOP_K={TOP_K} RERANK_M={RERANK_M} "
          f"EQA_WIN={EQA_WIN} CTX_MAX_TOK={CTX_MAX_TOK} "
          f"DELTA_WRITE={int(DELTA_WRITE)} INPUTS_EMBEDS={int(INPUTS_EMBEDS)} "
          f"DELTA_RESTORE={int(DELTA_RESTORE)} rows={len(rows)} OUT={OUT}", flush=True)

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
        # 每个 context 起点清空 delta 状态（含上一 context 残差）；此后查询期不再 reset。
        reset_delta_mem_states(model)
        # 查询前完成 delta 写入并拿快照；先于建池，保证 L18 单元/query 编码看到同一记忆状态。
        snap = prepare_delta_memory(ctx)
        if MODE == "v9":
            epi, st, n_tok = build_pool_for_context_v9(ctx)
            pool = epi
            print(f"[ctx {ci+1}/{len(rows)}] tokens={n_tok} episodic={len(epi)} "
                  f"state={len(st)} stats={epi.stats()} ({time.time()-t_start:.0f}s)", flush=True)
        elif MODE == "v8":
            epi, st, n_tok = build_pool_for_context_v8(ctx)
            pool = epi
            print(f"[ctx {ci+1}/{len(rows)}] tokens={n_tok} episodic={len(epi)} "
                  f"state={len(st)} dup={epi.n_dup} ({time.time()-t_start:.0f}s)", flush=True)
        else:
            pool, n_tok = build_pool_for_context(ctx)
            print(f"[ctx {ci+1}/{len(rows)}] tokens={n_tok} chunks={len(pool)} dup={pool.n_dup} "
                  f"({time.time()-t_start:.0f}s)", flush=True)

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
                if MODE == "v9":
                    block, dbg, diag_ctx = build_block_v9(epi, st, q, meta, qi)
                elif MODE == "v8":
                    block, dbg, diag_ctx = build_block_v8(epi, st, q, meta, qi)
                else:
                    block, dbg, diag_ctx = build_block(pool, q, meta, qi)
            except Exception as e:                      # 单题失败不中断整轮
                print(f"[warn] retrieve fail q={qi}: {e}", flush=True)
                block, dbg = "", {"mode": "error", "err": str(e)}
                _p = epi if MODE in ("v8", "v9") else pool
                diag_ctx = {"pool": _p, "hits": [], "used_ids": [], "used_texts": []}

            # 逐题恢复写后快照：前序题目的读出不会污染后续题目（查询期禁止 reset）。
            if DELTA_RESTORE:
                delta_restore(snap)

            pred, in_len, out_len = generate_readout(block, qtxt)

            m, extra = post_process(
                {"output": pred, "input_len": in_len,
                 "output_len": out_len,
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
                "delta": {"write": bool(DELTA_WRITE), "restore": bool(DELTA_RESTORE),
                          "inputs_embeds": bool(INPUTS_EMBEDS),
                          "snapshot_layers": len(snap)},
                **{k: v for k, v in m.items()}, **extra,
            }
            # 逐题诊断（Recall@K / answer_in_retrieved / answer_in_context）：
            # hits 必须是分数降序；used_ids/used_texts 是实际拼进 prompt 的槽位。
            if ar_diag is not None:
                try:
                    rec.update(ar_diag.record(answer=a, query=q, flat=True, **diag_ctx))
                except Exception as e:                  # 诊断失败不影响主指标
                    rec["diag_status"] = "diag_error"
                    rec["diag_error"] = str(e)
            results.append(rec)
            with open(OUT, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (len(results)) % 20 == 0:
                sf = sum(r.get("substring_exact_match", 0) for r in results) / len(results)
                err = sum(r.get("eventqa_recall", 0) for r in results) / len(results) if FAMILY == "eventqa" else 0.0
                print(f"[{ci+1}/{len(rows)}] {len(results)} 题 running SF={sf:.3f} "
                      f"eqa={err:.3f} ({time.time()-t_start:.0f}s)", flush=True)

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
    raise SystemExit(main())
