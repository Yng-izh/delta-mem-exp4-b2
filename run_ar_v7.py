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
#               拼块格式 '<serial>. [<date>] <原文>'（AR_V9_PROMPT_DATE=1 默认）：
#               日期取单元 event_time（显式或 Session 头继承），无时间则省略方括号；
#               serial 全局递增（"编号越大越新"不变）；=0 时逐字节等于旧输出。
#               拼块大小由 token 预算 AR_V9_CTX_TOKEN_BUDGET=2500 决定（按分数降序装填、
#               超预算即停、装完按位置输出）；被裁掉的槽位/token 记进
#               dropped_ids / dropped_tokens / dropped_reasons，不静默丢证据；
#               OOM 重试同样按证据优先级整行裁剪，逐次记录 keep_ids / dropped_ids。
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
#   写入开启但快照为空 -> 直接报错（除非 AR_ALLOW_EMPTY_DELTA_SNAP=1）：
#   空快照会让逐题恢复退化为 no-op，题目之间互相污染，指标不可解释。
#
# Delta 写入策略（AR_DELTA_WRITE_STRATEGY，只在 AR_DELTA_WRITE=1 时生效）：
#   all          现状：chunk_text_into_sentences(ctx, chunk_size=512) 全部分块盲写 —— 对照基线
#   filtered     只写状态/事实类单元：显式时间戳或事件动词的叙述句跳过、重复事实去重、
#                同主语+同属性的新旧值只写最终值（复用 v9 extract_units 的单元分类）
#   topk_refresh filtered + 按本 context 问题集的词面 IDF 粗排只刷新 Top-K 槽位
#                （写入先于建池，用不了建池后的检索器；retriever-driven refresh 是后续工作）
#   默认值决策：AR_DELTA_WRITE 仍默认 0（不写），三档策略写进配置指纹、互相可消融；
#   在拿到配对/显著证据之前，不把 filtered 设为默认 —— 先证明"过滤后的 Delta 更好"。
#   每题记录 delta.write_chunks / delta.skipped_chunks / delta.snapshot_layers /
#   delta.write_seconds / delta.skip_reasons，用于核对写入量是否与信息量脱钩。
#
# 读出模式（诚实命名，避免把明文嵌入说成 latent memory）：
#   AR_READOUT_MODE=prompt       默认：input_ids 走明文 prompt
#   AR_READOUT_MODE=text_embeds  用 EMB[ids] 走 inputs_embeds —— 这是【明文 token 的嵌入】，
#                                **没有**注入槽位内部状态或压缩 latent tokens；
#                                论文里不得当作已实现的 latent memory 方法
#   AR_READOUT_MODE=latent       真正的潜层读出（latent_memory.py）：
#                               历史证据先压成 Z∈R^{m×d}（LatentCompressor），再经
#                               LatentBridge 变成 soft tokens 插在 system 段之后；生成时
#                               **只有 system/question 是正常 token**，明文证据行一行都不进
#                               prompt（generate_latent_answer 的签名也不接受 block）。
#                               需要 AR_MODE=v9（复用 v9 的寻址层）+ 训练好的 LATENT_CKPT；
#                               缺 checkpoint 直接报错，绝不拿随机投影跑论文对比。
#                               预算单位变了：LATENT_READ_BUDGET 是**软 token 数**（K×m），
#                               不是 AR_V9_CTX_TOKEN_BUDGET 的 2500 个明文 token。
#                               逐题记录 readout.latent_slots=True + 注入槽位数/向量数/checkpoint；
#                               answer_in_context 在 latent 模式下不适用（没有明文进上下文），
#                               检索侧证据命中率仍由 answer_in_retrieved / recall_at_k 单独报告。
#   AR_INPUTS_EMBEDS=1 等价于 text_embeds（兼容旧命令）。
#
# 查询模板不用金标（AR_QTXT_NO_GOLD）：模板里的 {label} 若填成 gold，等于把答案塞进问题，
# 是一条潜在评测泄漏通道。latent 模式强制关闭（模板确实需要 {label} 时填空串并记
# query_label="empty(no_gold)"）；其它模式默认保持历史行为（=0），可显式置 1 做对照。
#
# 结果文件与 manifest（P0）：
#   输出默认带配置指纹 ar_{MODE}_{SRC}_{config_id}.jsonl，并写 <out>.manifest.json。
#   续跑前校验 manifest（配置 / 参数 / 代码 commit），不一致直接拒绝恢复，
#   避免 AR_DELTA_WRITE=0/1、AR_V9_L18=0/1 等消融写进同一个文件互相覆盖。
#   AR_OUT 可显式指定输出；AR_RESUME_MANIFEST=0 可关闭校验（不建议）。
#   AR_STRICT_RETRIEVAL=auto/1/0：auto 在稠密/潜层通道开启时让检索异常直接抛出，
#   否则 runner 的"单题失败不中断"会把设备/形状错误伪装成指标骤降。

支持的 source（parquet 内 metadata.source）：
    eventqa_65536 / eventqa_131072 / eventqa_full
    longmemeval_s* / longmemeval_s_-1_500（若存在）
  其它 source（ruler_*）需要 choice 专用指标，本脚本显式拒绝。

指标：直接调用 MemoryAgentBench 官方 post_process，与论文口径一致：
    - 全切片：substring_exact_match（SF）
    - eventqa：额外 eventqa_recall（6 选 1 全中）
"""
import sys, os, re, json, time, math, string

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

# 实验 manifest / 配置指纹 / 续跑校验（纯标准库）
try:
    import run_manifest
    _MANIFEST_ERR = None
except Exception as _e:
    run_manifest = None
    _MANIFEST_ERR = _e


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

# Delta 写入策略（只在 AR_DELTA_WRITE=1 时生效；默认仍是 0=不写，见文件头"Delta 写入策略"）
DELTA_WRITE_STRATEGIES = ("all", "filtered", "topk_refresh")
DELTA_WRITE_STRATEGY = os.environ.get("AR_DELTA_WRITE_STRATEGY", "all").strip().lower()
DELTA_WRITE_TOPK = int(os.environ.get("AR_DELTA_WRITE_TOPK", "8"))   # topk_refresh 的 K
DELTA_WRITE_CHUNK = int(os.environ.get("AR_DELTA_WRITE_CHUNK", "512"))  # all 的分块 token 数
if DELTA_WRITE_STRATEGY not in DELTA_WRITE_STRATEGIES:
    raise SystemExit(f"[ERR] AR_DELTA_WRITE_STRATEGY={DELTA_WRITE_STRATEGY!r} 非法；"
                     f"可选 {' | '.join(DELTA_WRITE_STRATEGIES)}")
# 最近一次 prepare_delta_memory 的写入统计（write_chunks/skipped_chunks/... 进每题 record）
DELTA_STATS = {}

INPUTS_EMBEDS = os.environ.get("AR_INPUTS_EMBEDS", "0") == "1"  # 兼容旧命令：读出走 inputs_embeds
DELTA_RESTORE = os.environ.get("AR_DELTA_RESTORE", "1") == "1"  # 逐题恢复 delta 状态快照
LIMIT_ROWS  = int(os.environ.get("AR_LIMIT_ROWS", "0"))    # smoke：限制上下文数
LIMIT_Q     = int(os.environ.get("AR_LIMIT_Q", "0"))       # smoke：限制每题数
RESUME      = os.environ.get("AR_RESUME", "1") == "1"

# 读出模式：prompt | text_embeds | latent
#   latent = 真正的潜层读出（latent_memory.py），需要 AR_MODE=v9 + LATENT_CKPT；
#   校验放在 main()/dry_run() 里（配置指纹要先能算出来，才能说清缺什么）。
READOUT_MODE = os.environ.get("AR_READOUT_MODE", "").strip().lower()
if not READOUT_MODE:
    READOUT_MODE = "text_embeds" if INPUTS_EMBEDS else "prompt"
if READOUT_MODE not in ("prompt", "text_embeds", "latent"):
    raise SystemExit(f"[ERR] AR_READOUT_MODE={READOUT_MODE!r} 非法；"
                     f"可选 prompt | text_embeds | latent")
INPUTS_EMBEDS = READOUT_MODE == "text_embeds"

# ---- 潜层读出配置（AR_READOUT_MODE=latent；其它模式完全不触碰，旧 config_id 不变）----
LATENT_CKPT          = os.environ.get("LATENT_CKPT", "").strip()
LATENT_M             = int(os.environ.get("LATENT_M", "4"))            # 每单元潜槽位数 m
LATENT_LAYER         = int(os.environ.get("LATENT_LAYER", "18"))       # 取第几层隐状态
LATENT_INNER_DIM     = int(os.environ.get("LATENT_INNER_DIM", "512"))
LATENT_MAX_UNIT_TOKENS = int(os.environ.get("LATENT_MAX_UNIT_TOKENS", "64"))
# 读出预算 = **软 token 数**（K×m）。32 / m=4 -> 最多 8 个槽位。
# 与 AR_V9_CTX_TOKEN_BUDGET（2500 个明文 token）不是同一个量，不能互相换算。
LATENT_READ_BUDGET   = int(os.environ.get("LATENT_READ_BUDGET", "32"))
LATENT_TOKENS_PER_SLOT = LATENT_M
# 与 latent_memory.latent_budget_slots 同一口径：预算 > 0 时至少给 1 个槽位，
# 否则 3 个软 token / m=4 会算出 K=0，"有预算却读出为空"。
LATENT_SLOTS_MAX = (max(1, LATENT_READ_BUDGET // LATENT_M)
                    if (LATENT_READ_BUDGET > 0 and LATENT_M > 0) else 0)
LATENT_BANK_MAX      = int(os.environ.get("LATENT_BANK_MAX", "0"))     # >0 只编码前 N 槽（smoke）

# 查询模板不用金标（潜在泄漏通道）：latent 模式强制开，其它模式默认保持历史行为。
_nogold_env = os.environ.get("AR_QTXT_NO_GOLD", "").strip().lower()
if READOUT_MODE == "latent":
    QTXT_NO_GOLD = True
elif _nogold_env:
    QTXT_NO_GOLD = _nogold_env in ("1", "true", "yes", "on")
else:
    QTXT_NO_GOLD = False

# 检索异常策略：auto = 稠密/潜层通道开启时必须报错（否则故障伪装成指标骤降）
STRICT_RETRIEVAL = os.environ.get("AR_STRICT_RETRIEVAL", "auto").strip().lower()

MAX_NEW = {"eventqa": 40, "longmemeval": 50}

BASE    = "/share/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
ADAPTER = "/share/hf_home/hub/models--declare-lab--delta-mem_qwen3_4b-instruct/snapshots/6a1a990e770974811b8fb27dab73430807f76cb5"
PQ      = os.environ.get(
    "AR_PARQUET",
    "/share/latent/MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet",
)

# 输出路径与 manifest：由 init_manifest() 惰性初始化（见文件末），
# 因为参数（V9_* 等）在下面才定义。未初始化前 OUT 为 None。
OUT = None
MANIFEST = None
RUN_ID = ""


def _manifest_params():
    """影响结果的全部配置（改这里必须同步 init_manifest 的调用方）。"""
    p = {
        "mode": MODE,
        "chunk": CHUNK,
        "top_k": TOP_K,
        "rerank_m": RERANK_M,
        "eqa_win": EQA_WIN,
        "ctx_max_tok": CTX_MAX_TOK,
        "delta_write": DELTA_WRITE,
        # 三档写入策略必须进指纹：all/filtered/topk_refresh 是互相对照的实验组，
        # 共用结果文件会让 AR_RESUME 把另一组的题目当成"已完成"直接跳过。
        "delta_write_strategy": DELTA_WRITE_STRATEGY,
        "delta_write_topk": DELTA_WRITE_TOPK,
        "delta_restore": DELTA_RESTORE,
        "readout_mode": READOUT_MODE,
        "strict_retrieval": STRICT_RETRIEVAL,
        "limit_rows": LIMIT_ROWS,
        "limit_q": LIMIT_Q,
    }
    if MODE == "v8":
        p["v8"] = {"topk_state": TOPK_STATE, "neighbor": NEIGHBOR, "ctx_units": CTX_SENTS}
    if MODE == "v9":
        # 索引/打分常量住在 v9_slotpool 模块里（只读导入，不注入 tok/EMB）
        import importlib
        sp = importlib.import_module("v9_slotpool")
        p["v9"] = {
            "block_sents": sp.BLOCK_SENTS, "unit_max_words": sp.UNIT_MAX_WORDS,
            "eqa_back": EQA_BACK, "eqa_ahead": EQA_AHEAD,
            "temporal_tau": sp.TEMPORAL_TAU, "temporal_bonus": sp.TEMPORAL_BONUS,
            "l18": V9_L18, "l18_layer": V9_L18_LAYER, "l18_alpha": sp.L18_ALPHA,
            # 池内 whitening 会改变稠密通道的余弦口径（默认开），必须进指纹：
            # 否则 WHITEN=0/1 两次跑会拿到同一个 config_id，续跑时把不同口径混进一个文件。
            "l18_whiten": sp.L18_WHITEN, "l18_whiten_eps": sp.L18_WHITEN_EPS,
            "l18_reuse_epi": sp.L18_REUSE_EPI,
            "topk_state": V9_TOPK_STATE, "ctx_units": V9_CTX_UNITS,
            # token 级证据预算（t6）：2500 是拼块大小的实际决定因素，必须进指纹
            "ctx_token_budget": V9_TOKEN_BUDGET,
            "session_reset": sp.SESSION_RESET, "split_newline": sp.SPLIT_ON_NEWLINE,
            "state_theta": sp.STATE_THETA, "cr_rel_theta": sp.CR_REL_THETA,
            # 时间进 Prompt 的开关也必须进指纹：DATE=0/1 是互相对照的读出实验组
            "prompt_date": V9_PROMPT_DATE, "prompt_serial": V9_PROMPT_SERIAL,
        }
    if READOUT_MODE == "latent":
        # 潜层读出的结果**完全由 checkpoint 决定**：路径 + 内容哈希 + 压缩结构 + 读出方式
        # 全部进指纹。否则同一路径换了新一代 checkpoint 后 config_id 不变，续跑会把两代
        # 模型的结果混进同一个文件。
        try:
            fp = (run_manifest.file_fingerprint(LATENT_CKPT, label="checkpoint")
                  if run_manifest is not None
                  else {"checkpoint": LATENT_CKPT, "exists": False})
        except Exception:
            fp = {"checkpoint": LATENT_CKPT, "exists": False}
        p["latent"] = {
            "readout": "latent",
            "checkpoint": fp,
            "n_slots": LATENT_M, "layer": LATENT_LAYER, "inner_dim": LATENT_INNER_DIM,
            "max_unit_tokens": LATENT_MAX_UNIT_TOKENS,
            "read_budget": LATENT_READ_BUDGET,
            "tokens_per_slot": LATENT_TOKENS_PER_SLOT,
            "max_slots": LATENT_SLOTS_MAX,
            "bank_max": LATENT_BANK_MAX,
            # 压缩率上限：最多 LATENT_MAX_UNIT_TOKENS 个 token -> m 个向量
            "compression_ratio_max": (round(LATENT_M / LATENT_MAX_UNIT_TOKENS, 6)
                                      if LATENT_MAX_UNIT_TOKENS > 0 else None),
            "qtxt_no_gold": QTXT_NO_GOLD,
        }
    if _nogold_env:
        # 显式设过 AR_QTXT_NO_GOLD 就必须进指纹（否则 0/1 两次跑会共用结果文件）
        p["qtxt_no_gold"] = QTXT_NO_GOLD
    return p


def init_manifest():
    """构建 manifest 并确定输出路径（幂等；main / dry_run 各调用一次）。"""
    global OUT, MANIFEST, RUN_ID
    if MANIFEST is not None:
        return MANIFEST
    if run_manifest is None:
        RUN_ID = "nomf"
        MANIFEST = {"schema": 0, "config_id": RUN_ID, "kind": "ar", "src": SRC,
                    "params": _manifest_params(), "git": {}, "env": {}}
    else:
        MANIFEST = run_manifest.build(kind="ar", src=SRC, params=_manifest_params(),
                                      extra={"mode": MODE, "argv": sys.argv[1:]})
        RUN_ID = MANIFEST["config_id"]
    OUT = os.environ.get("AR_OUT") or (
        run_manifest.default_out("/share/exp4group/results", f"ar_{MODE}", SRC, RUN_ID)
        if run_manifest is not None
        else f"/share/exp4group/results/ar_{MODE}_{SRC.replace('*', 'star')}.jsonl")
    return MANIFEST


def strict_retrieval_enabled():
    """是否把检索异常直接抛出（auto：稠密/潜层通道开启即严格）。"""
    if STRICT_RETRIEVAL in ("1", "true", "yes", "on"):
        return True
    if STRICT_RETRIEVAL in ("0", "false", "no", "off"):
        return False
    return bool(V9_L18 or DELTA_WRITE or INPUTS_EMBEDS or READOUT_MODE == "latent")

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


def build_qtxt(q, a):
    """构造查询文本，返回 (qtxt, query_label)。

    QTXT_NO_GOLD=True（latent 模式强制）时**不**把金标答案塞进查询模板：模板里的
    {label} 若被填成 gold 首字母，模型看问题就等于看答案，是一条潜在评测泄漏通道。
    模板确实需要 {label} 时用空串填充并返回 "empty(no_gold)"（既不泄漏也不崩），
    绝不偷偷把 gold 填回去。
    """
    label = None
    if not QTXT_NO_GOLD:
        label = (a[0] if isinstance(a, list) and a else a)
    kw = {"question": q}
    if label is not None:
        kw["label"] = label
    try:
        return qry_tpl.format(**kw), ("gold" if label is not None else "omitted")
    except (KeyError, IndexError):
        if label is None:
            try:
                return qry_tpl.format(question=q, label=""), "empty(no_gold)"
            except (KeyError, IndexError):
                return str(q), "raw_question"
        return qry_tpl.format(question=q), "omitted"


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


# ---------------------- Delta 有效写入：写入前过滤 / 去重 / 覆写合并 ----------------------
# 问题：旧实现把 chunk_text_into_sentences(ctx, chunk_size=512) 的**全部分块**前向写进
# Delta State —— 纯背景叙述、重复事实、后来被覆写掉的旧值全部进状态，写入量与信息量无关。
# 这里把"写什么"从"写多少 token"里拆出来，三档可消融（all 是对照基线）：
#   all          —— 现状（512 token 分块盲写），保留为对照，任何过滤收益都与它比；
#   filtered     —— 只写状态/事实类单元：显式时间戳或事件动词的叙述句（情景）跳过，
#                   完全重复的事实去重，同主语+同属性的新旧值只写**最终值**；
#   topk_refresh —— 在 filtered 之上，按本 context 问题集的词面 IDF 粗排只刷新 Top-K 槽位。
#                   ⚠️ 写入发生在建池之前（L18 单元/query 编码必须看到同一记忆状态），
#                   因此这里用不了建池后的检索器，"检索驱动刷新"当前是词面近似；
#                   真正的 retriever-driven refresh 需要重排写入/建池顺序，属后续工作。
#
# 过滤判据复用 v9_slotpool.extract_units 的单元切分与 is_event / ts_explicit 分类
# （v9 的单元级等价物：显式时间或事件动词 -> 情景，否则 -> 稳定陈述/状态）。
# 全部为纯标准库 + 纯正则，不需要模型前向，因此 CPU 单测与 --dry-run 都能跑。

_DELTA_STOP = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "but", "as", "is", "was", "are", "were", "be", "been", "being",
    "it", "its", "his", "her", "their", "he", "she", "they", "this", "that",
    "these", "those", "also", "very", "s", "t", "there", "here", "then",
}

# 谓词线索（只用于"同主语+同属性"的槽位键，不追求完整句法解析）
_FACT_VP_RE = re.compile(
    r"\b(?:is|was|are|were|has|had|have|works?|worked|lives?|lived|likes?|liked|"
    r"loves?|loved|owns?|owned|plays?|played|studies|studied|studying|moved|"
    r"joined|left|holds?|held|wants?|wanted|needs?|needed|prefers?|preferred|"
    r"uses?|used|drives?|drove|reads?|read|teaches|taught|manages|managed)\b",
    flags=re.IGNORECASE,
)
_COPULA = {"is", "was", "are", "were", "has", "had", "have"}
_PREPS = {"in", "at", "on", "to", "for", "with", "of", "from", "by", "about", "near", "into"}


def _delta_norm_text(text):
    """去重键：小写 + 去标点 + 压缩空白（同一事实的不同标点/大小写视为重复）。"""
    s = (text or "").lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def _delta_tokens(text):
    """词面 token（小写、去停用词），用于 topk_refresh 的粗排。"""
    ws = [w for w in re.findall(r"[a-z0-9']+", (text or "").lower()) if w not in _DELTA_STOP]
    return ws


def _delta_slot_key(text):
    """(主语归一化, 属性归一化) 槽位键：同主语 + 同属性 -> 同一槽，只保留最终值。

    系表结构必须把表语并进谓词键：裸系词会让 'Alice is a doctor.' 与
    'Alice is vegetarian.' 落进同一个槽，职业事实被后一句覆写掉（用户审计已实测）。
    动词谓词带紧跟的介词：'works at' 与 'works on' 是不同属性，不得互相覆写。
    无谓词线索时退化为整句归一化键（只能按完全重复去重）。
    """
    raw = (text or "").strip().rstrip(".!?;:")
    if not raw:
        return None
    m = _FACT_VP_RE.search(raw)
    if not m:
        k = _delta_norm_text(raw)
        return ("", k) if k else None
    subj = " ".join(w for w in _delta_norm_text(raw[:m.start()]).split()
                    if w not in _DELTA_STOP)
    verb = m.group(0).lower()
    rest = raw[m.end():]
    rest_words = [w for w in re.findall(r"[a-z0-9'\-]+", rest.lower())
                  if w not in _DELTA_STOP]
    if verb in _COPULA:
        # 表语头最多 2 词（'is doctor' / 'is vegetarian' 分开；多说一句也不误合并）
        pred = " ".join([verb] + rest_words[:2])
    else:
        first = re.findall(r"[a-z']+", rest.lower())
        prep = first[0] if first and first[0] in _PREPS else ""
        pred = f"{verb} {prep}".strip()
    return (subj, pred) if (subj or pred) else None


def _delta_units(ctx):
    """句级非重叠单元（含 is_event / ts_explicit 分类）。

    优先用 v9_slotpool.extract_units（与 v9 检索同一套切句/会话/时间语义，
    避免两处规则漂移）；v9_slotpool 不可导入时**报错**而不是猜一套替代规则，
    否则过滤语义会静默变成另一套东西。
    """
    sp = v9_import()
    units = sp.extract_units(ctx, block_sents=1)
    return units


def _filter_state_units(units):
    """状态/事实类过滤 + 去重 + 覆写合并。返回 (texts, reasons)。"""
    kept = []            # [(start, text, norm_key, slot_key)]
    reasons = {"episodic": 0, "duplicate": 0, "overwritten": 0}
    seen = {}            # 归一化全文 -> kept 下标
    slot_at = {}         # (主语, 属性) -> kept 下标
    for i, u in enumerate(units):
        text = (u.get("text") or "").strip()
        if not text:
            continue
        if u.get("is_event") or u.get("ts_explicit"):
            # 情景/叙述单元：显式时间戳或事件动词 -> 不是"当前状态"，不写
            reasons["episodic"] += 1
            continue
        nk = _delta_norm_text(text)
        if not nk:
            continue
        if nk in seen:
            reasons["duplicate"] += 1
            continue
        sk = _delta_slot_key(text)
        if sk is not None and sk in slot_at:
            # 同主语+同属性的新值：原位替换旧值 -> 状态里只有最终值
            j = slot_at[sk]
            seen.pop(kept[j][2], None)
            kept[j] = (kept[j][0], text, nk, sk)
            seen[nk] = j
            reasons["overwritten"] += 1
            continue
        seen[nk] = len(kept)
        if sk is not None:
            slot_at[sk] = len(kept)
        kept.append((u.get("start", i), text, nk, sk))
    kept.sort(key=lambda x: x[0])
    return [k[1] for k in kept], reasons


def _delta_lex_scores(texts, questions):
    """问题集 vs 单元的 IDF 词面重叠分（写入前粗排，不跑模型）。"""
    q_tokens = []
    for q in questions or []:
        q_tokens.extend(_delta_tokens(q))
    if not q_tokens or not texts:
        return None
    q_cnt = {}
    for w in q_tokens:
        q_cnt[w] = q_cnt.get(w, 0) + 1
    docs = [set(_delta_tokens(t)) for t in texts]
    df = {}
    for d in docs:
        for w in d:
            df[w] = df.get(w, 0) + 1
    n = max(len(docs), 1)
    scores = []
    for d in docs:
        s = 0.0
        for w, c in q_cnt.items():
            if w in d:
                s += c * math.log((n + 1.0) / (df[w] + 1.0) + 1.0)
        scores.append(s)
    return scores


def plan_delta_writes(ctx, strategy=None, questions=None, chunk_size=None, all_chunks=None,
                      topk=None):
    """决定本 context 要写进 Delta 的文本单元（纯标准库 + 纯正则，不碰模型）。

    返回 (texts, stats)。stats 字段：
      strategy / candidate_units / write_chunks / skipped_chunks /
      skip_reasons{episodic,duplicate,overwritten,topk} / unit_source / topk / chunk_size
    """
    strategy = (strategy or DELTA_WRITE_STRATEGY or "all").strip().lower()
    if strategy not in DELTA_WRITE_STRATEGIES:
        raise ValueError(f"AR_DELTA_WRITE_STRATEGY={strategy!r} 非法；"
                         f"可选 {' | '.join(DELTA_WRITE_STRATEGIES)}")
    csize = int(chunk_size or DELTA_WRITE_CHUNK)
    stats = {"strategy": strategy, "candidate_units": 0, "write_chunks": 0,
             "skipped_chunks": 0,
             "skip_reasons": {"episodic": 0, "duplicate": 0, "overwritten": 0, "topk": 0},
             "unit_source": "chunk512" if strategy == "all" else "v9",
             "topk": None, "chunk_size": csize}

    if strategy == "all":
        # 对照基线：与旧实现逐字一致（同样的分块、同样的顺序、全部写入）
        chunks = all_chunks
        if chunks is None:
            if chunk_text_into_sentences is None:
                raise RuntimeError("chunk_text_into_sentences 不可用（缺 MemoryAgentBench 依赖），"
                                   "无法执行 AR_DELTA_WRITE_STRATEGY=all")
            chunks = chunk_text_into_sentences(ctx, chunk_size=csize)
        texts = [t for t in (chunks or []) if t and t.strip()]
        stats.update(candidate_units=len(texts), write_chunks=len(texts), skipped_chunks=0)
        return texts, stats

    units = _delta_units(ctx)
    stats["candidate_units"] = len(units)
    texts, reasons = _filter_state_units(units)
    stats["skip_reasons"].update(reasons)

    if strategy == "topk_refresh":
        k = int(topk if topk is not None else DELTA_WRITE_TOPK)
        stats["topk"] = k
        scores = _delta_lex_scores(texts, questions)
        if scores is not None and 0 < k < len(texts):
            order = sorted(range(len(texts)), key=lambda i: (-scores[i], i))[:k]
            keep = set(order)
            stats["skip_reasons"]["topk"] = len(texts) - len(keep)
            texts = [t for i, t in enumerate(texts) if i in keep]

    stats["write_chunks"] = len(texts)
    stats["skipped_chunks"] = sum(stats["skip_reasons"].values())
    return texts, stats


def prepare_delta_memory(ctx, questions=None):
    """查询前完成 delta 写入，返回写后快照（写开关已关闭）。

    调用约定：main 在本 context 起点先 reset_delta_mem_states(model)；
    本函数只负责 plan -> write -> disable -> snapshot，查询期不再 reset。
    写什么由 AR_DELTA_WRITE_STRATEGY 决定（all | filtered | topk_refresh），
    questions 只给 topk_refresh 做写入前的 Top-K 槽位粗排。
    """
    if not DELTA_WRITE:
        DELTA_STATS.clear()
        DELTA_STATS.update({"strategy": DELTA_WRITE_STRATEGY, "write_chunks": 0,
                            "skipped_chunks": 0, "candidate_units": 0,
                            "write_seconds": 0.0, "snapshot_layers": 0,
                            "skip_reasons": {}})
        return {}
    if reset_delta_mem_states is None or set_delta_mem_write_enabled is None:
        raise RuntimeError("delta-mem 写入 API 不可用，无法执行 AR_DELTA_WRITE=1")
    texts, stats = plan_delta_writes(ctx, strategy=DELTA_WRITE_STRATEGY,
                                     questions=questions)
    if not texts:
        msg = (f"AR_DELTA_WRITE_STRATEGY={stats['strategy']} 过滤后没有任何可写单元"
               f"（候选 {stats['candidate_units']}，跳过 {stats['skipped_chunks']}）："
               f"空写入 -> 空快照 -> 逐题恢复退化为 no-op，Delta 实际未参与问答。")
        if os.environ.get("AR_ALLOW_EMPTY_DELTA_SNAP", "0") != "1":
            raise RuntimeError(msg + " 请放宽过滤/换策略，或显式设 AR_ALLOW_EMPTY_DELTA_SNAP=1")
        print(f"[warn] {msg}（AR_ALLOW_EMPTY_DELTA_SNAP=1，继续）", flush=True)
    t0 = time.time()
    set_delta_mem_write_enabled(model, True)
    try:
        for ch in texts:
            txt = tok.apply_chat_template(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": fmt(mem_tpl, context=ch)}],
                add_generation_prompt=True, tokenize=False)
            ii = tok(txt, return_tensors="pt").input_ids.to(DEV)
            model(input_ids=ii, use_cache=True)
    finally:
        set_delta_mem_write_enabled(model, False)
    stats["write_seconds"] = round(time.time() - t0, 3)
    print(f"[delta] strategy={stats['strategy']} write={stats['write_chunks']} "
          f"skipped={stats['skipped_chunks']} reasons={stats['skip_reasons']} "
          f"candidates={stats['candidate_units']} {stats['write_seconds']}s", flush=True)
    snap = delta_snapshot(model)
    stats["snapshot_layers"] = len(snap or {})
    DELTA_STATS.clear()
    DELTA_STATS.update(stats)
    if not snap:
        msg = ("delta 写入已开启，但写后快照为空：逐题恢复会退化为 no-op，"
               "题目之间会互相污染（指标不可解释）。请检查 deltamem 版本/快照 API。")
        if os.environ.get("AR_ALLOW_EMPTY_DELTA_SNAP", "0") == "1":
            print(f"[warn] {msg}（AR_ALLOW_EMPTY_DELTA_SNAP=1，继续）", flush=True)
        else:
            raise RuntimeError(msg)
    if collect_delta_mem_state_stats is not None:
        try:
            print(f"[delta] state after write: {collect_delta_mem_state_stats(model)}",
                  flush=True)
        except Exception:
            pass
    return snap


def generate_readout(block, qtxt, trim=None):
    """生成答案。READOUT_MODE=text_embeds 时用 EMB[ids] 走 inputs_embeds。

    ⚠️ text_embeds 只是把**同一段明文 prompt** 换成嵌入矩阵查表结果（EMB[ids]），
    没有注入槽位内部状态或压缩 latent tokens，因此**不是** latent memory 读出，
    不能作为"已实现潜槽位记忆"参与论文比较（真正的路径 B 仍是规划中）。

    OOM 重试（t6）：**按证据优先级裁剪**，从最低分单元开始丢，每次重试把
    keep_ids / dropped_ids / prompt_tokens 记进 trim["retries"]，绝不"保留最后 N token"
    地静默丢掉已选中的高分证据。trim 缺省（老调用方）时按行位置定优先级（越靠前越重要），
    仍然只按证据行整行丢、且照样记录，不做 token 尾部截断。

    trim 结构：{"lines": [行文本...], "prio": [优先级（大=重要）...], "ids": [槽位ID...]}
    返回 (pred, input_len, output_len)：
      - 正常路径：out 含 prompt，output_len = out.shape[1] - input_len；
      - inputs_embeds 路径：generate 只返回新 token，output_len = out.shape[1]。
    """
    if trim is not None and trim.get("lines"):
        lines = [str(x) for x in trim["lines"]]
        prio = [float(p) for p in (trim.get("prio") or [0.0] * len(lines))]
        lids = list(trim.get("ids") or [None] * len(lines))
        if not (len(prio) == len(lines) == len(lids)):
            raise ValueError("trim.lines/prio/ids 必须等长")
    else:
        lines = [l for l in str(block or "").split("\n") if l.strip()]
        # 兜底：没有槽位元数据时按行序定优先级（前面的行更重要），丢也是整行丢
        prio = [float(len(lines) - j) for j in range(len(lines))]
        lids = [None] * len(lines)
    use_emb = bool(READOUT_MODE == "text_embeds" and EMB is not None)

    def _render(ls):
        b = "\n".join(ls)
        t = tok.apply_chat_template(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": f"{b}\n\n{qtxt}" if b else qtxt}],
            add_generation_prompt=True, tokenize=False)
        return b, t

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

    retries = []

    def _flush_trim(raised=False):
        if trim is None:
            return
        trim["retries"] = list(retries)
        trim["final_keep_ids"] = list(lids)
        trim["final_dropped_ids"] = [d for r in retries for d in r["dropped_ids"]]
        trim["raised"] = bool(raised)

    torch.cuda.empty_cache()                 # 建池/检索后的 GPU 碎片回收
    while True:
        _block, text = _render(lines)
        ids = tok(text, return_tensors="pt").input_ids.to(DEV)
        emb = EMB[ids] if use_emb else None
        try:
            out = _gen(ids, emb)
            break
        except torch.OutOfMemoryError:
            if len(lines) <= 1:
                # 只剩最高分那一行还 OOM：如实抛出，不静默丢证据（但先落盘已做的裁剪）
                _flush_trim(raised=True)
                raise
            # 从最低优先级开始丢（同优先级按位置靠后的先丢），每次丢 1/4（至少 1 行）
            order = sorted(range(len(lines)), key=lambda j: (prio[j], -j))
            n_drop = max(1, len(lines) // 4)
            drop = set(order[:n_drop])
            keep_ids = [lids[j] for j in range(len(lines)) if j not in drop]
            retries.append({
                "attempt": len(retries) + 1,
                "prompt_tokens": int(ids.shape[1]),
                "kept_lines": len(keep_ids),
                "dropped_lines": len(drop),
                "dropped_ids": [lids[j] for j in sorted(drop)],
                "keep_ids": keep_ids,
            })
            print(f"[gen] OOM retry {len(retries)}: 按证据优先级丢 {len(drop)} 行"
                  f"（最低分起），保留 {len(keep_ids)} 行", flush=True)
            lines = [l for j, l in enumerate(lines) if j not in drop]
            prio = [p for j, p in enumerate(prio) if j not in drop]
            lids = [i for j, i in enumerate(lids) if j not in drop]
            torch.cuda.empty_cache()

    _flush_trim()

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


# ----------------------------- 时间信息进 Prompt（t5） -----------------------------
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_DATE_ISO_RE = re.compile(r"\b((?:19|20)\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_DATE_MDY_RE = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+((?:19|20)\d{2})\b",
    flags=re.IGNORECASE)
_DATE_DMY_RE = re.compile(
    r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+((?:19|20)\d{2})\b",
    flags=re.IGNORECASE)
_DATE_DECADE_RE = re.compile(r"\bthe\s+((?:19|20)\d{2})s\b", flags=re.IGNORECASE)
_DATE_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def format_event_date(ts):
    """单元时间 -> Prompt 里的日期串；**不伪造**。

    - 'YYYY-MM-DD' / 'YYYY/MM/DD' / 'YYYY.MM.DD' -> 'YYYY-MM-DD'
    - '12 January 2005' / 'January 12, 2005'      -> 'YYYY-MM-DD'
    - 只有年份 '1997'                             -> '1997'（不补月份/日）
    - 'the 1990s'                                 -> '1990s'
    - None / 解析不出                               -> None（调用方省略方括号）
    """
    if not ts:
        return None
    s = str(ts).strip()
    if not s:
        return None
    m = _DATE_ISO_RE.search(s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    m = _DATE_MDY_RE.search(s)
    if m:
        mo = _MONTHS[m.group(1)[:3].lower()]
        return f"{int(m.group(3)):04d}-{mo:02d}-{int(m.group(2)):02d}"
    m = _DATE_DMY_RE.search(s)
    if m:
        mo = _MONTHS[m.group(2)[:3].lower()]
        return f"{int(m.group(3)):04d}-{mo:02d}-{int(m.group(1)):02d}"
    m = _DATE_DECADE_RE.search(s)
    if m:
        return f"{m.group(1)}s"
    m = _DATE_YEAR_RE.search(s)
    if m:
        return m.group(1)                       # 只有年份：不编造月日
    return None


def v9_unit_serial(meta, idx=0):
    """全局递增编号（1-based）：= 单元在上下文中的写入顺序（ingestion order）。

    非重叠单元按位置单调递增，因此"编号越大越新"成立；编号**与日期无关**，
    加日期前缀不会改变编号顺序（反事实 Prompt 的规则不变）。
    """
    try:
        return int((meta or {}).get("start", idx)) + 1
    except Exception:
        return int(idx) + 1


def render_v9_prompt_lines(rows, with_date=None, with_serial=None):
    """[(serial, text, event_time)] -> Prompt 行（纯函数，单测/--dry-run 可直接调用）。

    格式 '<serial>. [<date>] <原文>'；无日期省略方括号，不伪造。
    with_date / with_serial 为 None 时取当前开关（AR_V9_PROMPT_DATE / _SERIAL）。
    with_date=False 且 with_serial=False 时逐字节等于旧输出（原文逐行）。
    """
    wd = V9_PROMPT_DATE if with_date is None else bool(with_date)
    ws = V9_PROMPT_SERIAL if with_serial is None else bool(with_serial)
    lines = []
    for serial, text, event_time in rows:
        parts = []
        if ws:
            parts.append(f"{int(serial)}.")
        if wd:
            d = format_event_date(event_time)
            if d:
                parts.append(f"[{d}]")
        parts.append(text)
        lines.append(" ".join(parts))
    return lines


# ---- token 级证据预算（t6）----
# 问题：AR_V9_CTX_MAX_UNITS=40 是按"单元个数"限流，与方案要求的 2500 token 预算不等价
# （单元长短差异大：40 个长句可能 2000+ token，40 个短句可能只有 300 token）。
# 现在拼块集合由 AR_V9_CTX_TOKEN_BUDGET（默认 2500）决定：候选按**分数降序**装填，
# 装到超预算就停（"停"而不是"跳过后继续"，保证块是分数序的前缀），装完再按原文位置
# 排序输出（I2 不变：编号越大越新、位置序）。AR_V9_CTX_MAX_UNITS 保留为**二级硬上限**
# （默认 0 = 关闭，即完全由 token 预算限流），用于消融"单元数上限"这一老口径。
# 预算只统计**证据行**（不含 system prompt / question / 生成），ctx_tokens 记录实际装入量。
V9_TOKEN_BUDGET = int(os.environ.get("AR_V9_CTX_TOKEN_BUDGET", "2500"))
# 被预算裁掉的槽位列表落盘上限（全池 Tail 可能上千条，全写会让 JSONL 爆炸；
# 计数另记 dropped_ids_total，是否截断另记 dropped_ids_truncated，不隐瞒）
V9_DROPPED_ID_CAP = int(os.environ.get("AR_V9_DROPPED_ID_CAP", "200"))

_PROMPT_TOK_CACHE = {}


def prompt_tokens(text):
    """一行证据在 prompt 里占的 token 数（带缓存，纯函数语义）。

    tok 缺失时按空白切词近似 —— 只发生在没有 tokenizer 的离线场景，且会同时体现在
    ctx_tokens 上，不用于任何"看起来更小"的对外数字。
    """
    s = "" if text is None else str(text)
    hit = _PROMPT_TOK_CACHE.get(s)
    if hit is not None:
        return hit
    n = None
    if tok is not None:
        try:
            ids = tok(s, add_special_tokens=False).input_ids
            n = int(ids.shape[-1]) if hasattr(ids, "shape") else len(ids)
        except Exception:
            n = None
    if n is None:
        n = len(s.split())
    if len(_PROMPT_TOK_CACHE) > 200000:
        _PROMPT_TOK_CACHE.clear()
    _PROMPT_TOK_CACHE[s] = n
    return n


def fill_token_budget(ranked, budget, text_of, already=0):
    """按优先级降序装填证据行，超预算即停。

    ranked  : 已按优先级（分数）降序的 id 列表；同分时调用方负责稳定次序。
    budget  : token 预算；<=0 表示不限制（回退旧行为）。
    text_of : id -> 该行最终拼进 prompt 的文本（含 serial/日期前缀），token 数即按它计。
    already : 已经占用的 token（状态池装填时传入情景行已用量）。

    返回 (kept_ids, dropped_ids, kept_tokens, dropped_tokens)。
    规则：第一条**永远保留**（最高分证据必在块内，避免预算过小时拼出空块）。
    """
    if budget <= 0:
        kept = list(ranked)
        return kept, [], sum(prompt_tokens(text_of(i)) for i in kept), 0
    kept, used = [], 0
    for i in ranked:
        n = prompt_tokens(text_of(i))
        if kept and already + used + n > budget:
            break
        kept.append(i)
        used += n
    dropped = list(ranked[len(kept):])
    dropped_tokens = sum(prompt_tokens(text_of(i)) for i in dropped)
    return kept, dropped, used, dropped_tokens


def _uid(kind, idx):
    """统一槽位 ID（与 ar_diag.slot_uid 同格式），供 OOM 重试记录用。"""
    try:
        return f"{kind}:{int(idx)}"
    except (TypeError, ValueError):
        return f"{kind}:{idx}"


def _v9_row(epi, i):
    """单元 -> 渲染行素材 (serial, 原文, event_time)。预算计数与最终渲染共用。"""
    m = epi.metas[i] if i < len(epi.metas) else {}
    return (v9_unit_serial(m, i), epi.texts[i], m.get("event_time"))


def build_block_v8(epi, st, q, meta, qi):
    """v8 读出：全池 IDF-MaxSim + 邻域扩展 + 状态单元补充，按句序拼块。
    返回 (block, dbg, diag_ctx)。"""
    empty = {"pool": epi, "hits": [], "used_ids": [], "used_texts": [],
             "state_pool": st, "used_state_ids": []}
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
    used_st = []
    for i in st_order:
        k = _norm_unit(st.texts[i])
        if k not in seen:
            seen.add(k)
            lines_out.append(st.texts[i])
            used_st.append(int(i))
    dbg["n_units"] = len(lines_out)
    # diag_ctx 必须带 state_pool/used_state_ids：ar_diag.record 按关键字展开 diag_ctx，
    # 缺了这两个键状态行槽位 ID 会落 None、used_alignment 退化成 texts_without_ids。
    return "\n".join(lines_out), dbg, {"pool": epi, "hits": hits,
                                       "used_ids": used_epi, "used_texts": lines_out,
                                       "state_pool": st, "used_state_ids": used_st}


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


latent_memory = None


def latent_import():
    """惰性导入 latent_memory（只有 AR_READOUT_MODE=latent 才需要 torch 侧神经网络）。"""
    global latent_memory
    if latent_memory is None:
        import importlib
        latent_memory = importlib.import_module("latent_memory")
    return latent_memory


V9_TOPK_STATE = int(os.environ.get("AR_V9_TOPK_STATE", "4"))      # 状态池附带单元数
# 拼给 LLM 的**单元数**上限：默认 0 = 关闭。t6 起拼块大小由 AR_V9_CTX_TOKEN_BUDGET
# （2500 token）决定；这个开关只保留作"按个数限流"老口径的消融（>0 时作为二级硬上限）。
V9_CTX_UNITS  = int(os.environ.get("AR_V9_CTX_MAX_UNITS", "0"))
EQA_BACK      = int(os.environ.get("AR_V9_EQA_BACK", "16"))       # EventQA 锚点前取几句
EQA_AHEAD     = int(os.environ.get("AR_V9_EQA_AHEAD", "4"))       # EventQA 锚点后取几句
V9_L18        = os.environ.get("AR_V9_L18", "0") == "1"           # 第 18 层稠密通道开关
V9_L18_LAYER  = int(os.environ.get("AR_V9_L18_LAYER", "18"))

# ---- 时间信息进生成 Prompt（t5）----
# 问题：extract_units 把 "In <ts>: <text>" 只写进 index_text（打分用），拼给 LLM 的
# 仍是 epi.texts 原文 -> 模型看不到继承来的 Session 日期，多会话时间推理必然失败。
# 现在 build_block_v9 输出统一格式 '<serial>. [<date>] <原文>'：
#   - date 取单元的 event_time（本句显式时间戳 或 Session 头日期继承），无时间则**省略方括号**
#     （不伪造日期）；只解析到年份时输出 [YYYY]，不补月份/日；
#   - serial 是**全局递增**编号（= 单元在上下文中的 1-based 写入顺序），"编号越大越新"的
#     反事实规则不变：日期前缀只加信息，不改变编号顺序。
# 开关：AR_V9_PROMPT_DATE=1（默认，新格式）| 0（旧格式，用于对照）；
#       AR_V9_PROMPT_SERIAL 不设时跟随 AR_V9_PROMPT_DATE（DATE=0 -> 与旧实现逐字节一致）；
#       显式设 AR_V9_PROMPT_SERIAL=1/0 可单独消融"编号前缀"（编号本身不带任何新信息，
#       只影响模型对新鲜度的判断，因此可单独开）。
V9_PROMPT_DATE = os.environ.get("AR_V9_PROMPT_DATE", "1") == "1"
_serial_env = os.environ.get("AR_V9_PROMPT_SERIAL", "").strip().lower()
if _serial_env in ("1", "true", "yes", "on"):
    V9_PROMPT_SERIAL = True
elif _serial_env in ("0", "false", "no", "off"):
    V9_PROMPT_SERIAL = False
else:
    V9_PROMPT_SERIAL = V9_PROMPT_DATE      # 未显式指定 -> 跟随 DATE（DATE=0 时逐字节等于旧输出）


def build_pool_for_context_v9(ctx):
    """v9：非重叠单元索引（epi 全覆盖）+ 可选状态池。返回 (epi, st, n_tok)。"""
    sp = v9_import()
    text = _ctx_text(ctx)
    epi, st = sp.build_pools_for_context(text)
    return epi, st, len(tok(text, add_special_tokens=False).input_ids)


def _layer18_provider(texts, kind):
    """fn(texts, kind) -> [N,d]：取模型第 V9_L18_LAYER 层隐状态做 mask mean-pool。

    仅 AR_V9_L18=1 时注入；分批前向控制显存。统一返回 **CPU fp32**：
    词面分在 GPU、稠密分在 CPU，由 v9_slotpool.hybrid_scores 在最后一步对齐设备
    （v9 早期版本直接相加 -> 设备不一致报错，且被 runner 的宽 except 吞成空证据块）。
    注入时 strict=True：provider 失败/形状不符直接报错，不再静默退化为纯词面。
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
    """v9 读出：非重叠索引 + 分数序装填（token 预算）+ 位置序输出 + EventQA 后向非对称扩展。

    拼块格式（t5）：'<serial>. [<date>] <原文>'
      - serial = 单元的全局递增编号（写入顺序），"编号越大越新"不变，日期不改变编号顺序；
      - date 取单元 event_time（显式或 Session 头继承），无时间省略方括号（不伪造）；
      - AR_V9_PROMPT_DATE=0（且未显式开 AR_V9_PROMPT_SERIAL）时逐字节等于旧输出。

    预算（t6）：
      - AR_V9_CTX_TOKEN_BUDGET（默认 2500）按**分数降序**装填证据行，超预算即停；
        装完再按原文位置排序输出（I2 不变）。
      - AR_V9_CTX_MAX_UNITS（默认 0=关闭）只是**二级单元数硬上限**，用于消融"按个数限流"
        的老口径；预算模式下不再由它决定块大小。
      - 被预算裁掉的槽位/token 数写进 dbg.dropped_ids / dbg.dropped_tokens，
        并给出 dropped_ids_total / dropped_ids_truncated（列表有上限，避免 JSONL 爆炸）。

    返回 (block, dbg, diag_ctx)。
    """
    empty = {"pool": epi, "hits": [], "used_ids": [], "used_texts": [],
             "state_pool": st, "used_state_ids": []}
    if len(epi) == 0 and len(st) == 0:
        return "", {}, empty
    q_core = core_question(q)
    budget = V9_TOKEN_BUDGET
    state_slots = V9_TOPK_STATE if len(st) > 0 else 0
    cap_epi = max(1, V9_CTX_UNITS - state_slots) if V9_CTX_UNITS > 0 else 0

    # 全池分数（降序）：预算装填的优先级 + 诊断 hits 的来源
    ranked_all = [(int(i), float(s)) for i, s in epi.retrieve(q_core, top_k=len(epi))]
    score_of = dict(ranked_all)
    cap_dropped = []        # 被二级单元数上限截掉的候选（单独记账，别让它凭空消失）

    if FAMILY == "eventqa":
        evs = parse_prev_events(meta, qi)
        idxs, dbg = epi.readout(q_core, top_k=min(TOP_K, cap_epi or TOP_K), mode="eventqa",
                                anchor_events=evs, back=EQA_BACK, ahead=EQA_AHEAD)
        # 扩展窗超配额时保留锚点邻域（先截断，readout 返回的已是位置升序）
        idxs = _cap_before_sort(idxs, cap_epi, anchor=dbg.get("anchor"))
        # 诊断用 query 侧分数序命中（实际拼块是锚点扩展窗，二者分开记）
        hits = ranked_all[:TOP_K]
        # 预算装填只在锚点窗口内进行（不越出窗口，也不改成全池 Top-N）
        ranked_epi = sorted((int(i) for i in idxs),
                            key=lambda i: (-score_of.get(i, float("-inf")), i)) \
            if budget > 0 else [int(i) for i in idxs]
    else:
        # readout(topk) = 先按分数选 Top-K，再把选中的 K 个按原文位置排序
        idxs, dbg = epi.readout(q_core, top_k=min(TOP_K, cap_epi or TOP_K), mode="topk")
        hits = [(int(i), float(s)) for i, s in dbg.get("hits", [])]
        if budget > 0:
            # 预算模式：候选 = 全池分数序（单元数上限若显式开启则再截断，截掉的要记账）
            ranked_epi = [i for i, _ in ranked_all]
            if cap_epi > 0 and len(ranked_epi) > cap_epi:
                cap_dropped = ranked_epi[cap_epi:]
                ranked_epi = ranked_epi[:cap_epi]
        else:
            ranked_epi = [int(i) for i in idxs]

    def _epi_line(i):
        return render_v9_prompt_lines([_v9_row(epi, i)])[0]

    kept_epi, dropped_epi, tok_epi, dtok_epi = fill_token_budget(
        ranked_epi, budget, _epi_line)
    dtok_cap = sum(prompt_tokens(_epi_line(i)) for i in cap_dropped)

    # 状态池：在剩余预算内按分数装填（状态行排在情景行之后，顺序约定不变）
    st_hits = st.retrieve(q_core, top_k=V9_TOPK_STATE) if len(st) > 0 else []
    st_ranked = [int(i) for i, _ in st_hits]
    st_score = {int(i): float(s) for i, s in st_hits}

    def _st_line(i):
        return render_v9_prompt_lines([_v9_row(st, i)])[0]

    if budget <= 0:
        kept_st, dropped_st, tok_st, dtok_st = list(st_ranked), [], 0, 0
    else:
        remaining = budget - tok_epi
        if remaining <= 0:
            # 预算已被情景行用光：状态行全部不装（如实记录，不偷偷塞）
            kept_st = []
            dropped_st = list(st_ranked)
            tok_st, dtok_st = 0, sum(prompt_tokens(_st_line(i)) for i in st_ranked)
        else:
            kept_st, dropped_st, tok_st, dtok_st = fill_token_budget(
                st_ranked, remaining, _st_line, already=tok_epi)
    dbg["state_idxs"] = kept_st      # 实际装入的状态槽位（被预算裁掉的见 dropped_state_ids）
    dbg["stats"] = epi.stats()

    rows, seen, used_epi, used_st, line_ids, line_prio = [], set(), [], [], [], []
    for i in sorted(kept_epi, key=lambda j: epi.metas[j].get("start", j)):
        k = _norm_unit(epi.texts[i])
        if k not in seen:
            seen.add(k)
            rows.append(_v9_row(epi, i))
            used_epi.append(int(i))
            line_ids.append(_uid("epi", i))
            line_prio.append(score_of.get(int(i), 0.0))
    # ⚠️ 状态补充行沿用"情景行在前、状态行在后"的既有约定（ar_diag._align_used 依赖此顺序
    # 对齐跨池 ID），因此整块的 serial 在最后几行可能回退——每行都带编号与日期，模型可自行
    # 判断新旧；若将来要严格单调，需同时改 ar_diag 的对齐约定。
    for i in sorted(kept_st, key=lambda j: st.metas[j].get("start", j)):
        k = _norm_unit(st.texts[i])
        if k not in seen:
            seen.add(k)
            rows.append(_v9_row(st, i))
            used_st.append(int(i))
            line_ids.append(_uid("state", i))
            line_prio.append(st_score.get(int(i), 0.0))
    lines_out = render_v9_prompt_lines(rows)
    dbg["n_units"] = len(lines_out)
    # 诊断：逐行 serial / 日期（None=该行没有时间，Prompt 里就没有方括号）
    dbg["serials"] = [r[0] for r in rows]
    dbg["unit_dates"] = [format_event_date(r[2]) if V9_PROMPT_DATE else None for r in rows]
    dbg["prompt_date"] = bool(V9_PROMPT_DATE)
    dbg["prompt_serial"] = bool(V9_PROMPT_SERIAL)
    # 预算账本：ctx_tokens 按**最终拼进 prompt 的行**重新统计（去重后才是真实占用）
    dbg["ctx_budget"] = budget
    dbg["ctx_tokens"] = sum(prompt_tokens(l) for l in lines_out)
    dropped_all = [int(i) for i in dropped_epi] + [int(i) for i in cap_dropped]
    dbg["dropped_ids"] = dropped_all[:V9_DROPPED_ID_CAP]
    dbg["dropped_ids_total"] = len(dropped_all)
    dbg["dropped_ids_truncated"] = len(dropped_all) > len(dbg["dropped_ids"])
    dbg["dropped_reasons"] = {"budget": len(dropped_epi), "unit_cap": len(cap_dropped),
                              "state_budget": len(dropped_st)}
    dbg["dropped_state_ids"] = [int(i) for i in dropped_st][:V9_DROPPED_ID_CAP]
    dbg["dropped_tokens"] = int(dtok_epi + dtok_st + dtok_cap)
    # OOM 重试要用：逐行槽位 ID + 证据优先级（分数越大越重要）
    dbg["line_ids"] = line_ids
    dbg["line_prio"] = [round(float(p), 6) for p in line_prio]
    # used_ids 只含情景槽位（状态槽位跨池、无统一下标；used_texts 含全部拼块行）
    return "\n".join(lines_out), dbg, {"pool": epi, "hits": hits,
                                       "used_ids": used_epi, "used_texts": lines_out,
                                       "state_pool": st, "used_state_ids": used_st}


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


def select_latent_slots(epi, st, q, meta, qi, latent_token_budget=None,
                        tokens_per_slot=None, topk_state=None):
    """潜层读出的**槽位选择**：复用既有检索寻址，但预算单位是槽位而不是明文 token。

    与 build_block_v9 的关键区别（必须分清，否则预算口径会算错）：
      - 不装填 2500 个明文 token，而是最多 K = LATENT_READ_BUDGET // m 个槽位
        （每个槽位贡献 m 个软 token）；
      - 返回**命名空间化**槽位 ID（'epi:3' / 'state:0'），交给 LatentSlotBank.gather；
      - 不返回任何明文行（dbg 只有 ID/分数/诊断），所以 latent 模式下 block 是空串。

    寻址规则与明文基线保持一致：
      - EventQA：事件锚点 + 后向非对称窗口；预算不足时**优先保留锚点**
        （它是"已发生事件"的定位点，丢掉锚点整题就失去时序参照）；
      - LongMemEval：复用 readout(mode="lme") 的时间约束 + 跨 Session 配额 + turn 扩展，
        只在预算不足时按检索分数裁，**不再做全池重排**（否则配额与约束会被覆盖）；
      - 其它：分数优先取 Top-K。
    返回 (slot_ids, dbg)。
    """
    lm = latent_import()
    budget = LATENT_READ_BUDGET if latent_token_budget is None else int(latent_token_budget)
    tps = LATENT_TOKENS_PER_SLOT if tokens_per_slot is None else int(tokens_per_slot)
    max_slots = lm.latent_budget_slots(budget, tps)
    n_state = (V9_TOPK_STATE if topk_state is None else int(topk_state))
    n_state = max(0, n_state) if (st is not None and len(st) > 0) else 0
    dbg = {"mode": "latent_select", "budget": int(budget),
           "tokens_per_slot": int(tps), "max_slots": int(max_slots),
           "hits": [], "epi_ids": [], "state_ids": [], "slot_ids": [],
           "dropped_epi_ids": [], "dropped_state_ids": [],
           "anchor": None, "anchor_used": None, "anchor_cov": None,
           "anchor_forced": False, "lme": None, "budget_used": 0,
           "pool": "epi"}
    if epi is None or len(epi) == 0 or max_slots <= 0:
        return [], dbg

    q_core = core_question(q)
    ranked_all = [(int(i), float(s)) for i, s in epi.retrieve(q_core, top_k=len(epi))]
    score_of = dict(ranked_all)
    epi_budget = max(0, max_slots - n_state)
    neg = float("-inf")

    if FAMILY == "eventqa":
        evs = parse_prev_events(meta, qi)
        idxs, rdbg = epi.readout(q_core, top_k=min(TOP_K, max(1, max_slots)),
                                 mode="eventqa", anchor_events=evs,
                                 back=EQA_BACK, ahead=EQA_AHEAD)
        anchor = rdbg.get("anchor")
        dbg["anchor"] = anchor
        dbg["anchor_used"] = rdbg.get("anchor_used")
        dbg["anchor_cov"] = rdbg.get("anchor_cov")
        window = [int(i) for i in idxs]
        ranked = sorted(window, key=lambda i: (-score_of.get(int(i), neg), int(i)))
        if anchor is not None and int(anchor) in ranked and ranked[0] != int(anchor):
            ranked.remove(int(anchor))
            ranked.insert(0, int(anchor))
            dbg["anchor_forced"] = True
        hits = ranked_all[:TOP_K]
    elif FAMILY == "longmemeval":
        idxs, rdbg = epi.readout(q_core, top_k=min(TOP_K, max(1, max_slots)), mode="lme")
        sel = [int(i) for i in idxs]
        # 预算不足时按检索分数裁，但保留 lme 的筛选结果集合（不回到全池重排）
        ranked = sorted(sel, key=lambda i: (-score_of.get(i, neg), i))
        dbg["lme"] = {"session_quota": rdbg.get("session_quota"),
                      "temporal_constraint": rdbg.get("temporal_constraint"),
                      "turn_window": (rdbg.get("session_quota") or {}).get("turn_window")}
        # 诊断用命中：lme 的 picked 已是分数降序（含配额顺延后的结果），比全池 Top-K 更准
        hits = [(int(i), float(s)) for i, s in rdbg.get("hits", [])] or ranked_all[:TOP_K]
    else:
        idxs, rdbg = epi.readout(q_core, top_k=min(TOP_K, max(1, max_slots)), mode="topk")
        ranked = [int(i) for i, _s in ranked_all]
        hits = [(int(i), float(s)) for i, s in rdbg.get("hits", [])]

    kept = ranked[:epi_budget] if epi_budget > 0 else []
    kept = sorted(kept, key=lambda i: epi.metas[i].get("start", i))
    dbg["dropped_epi_ids"] = [int(i) for i in ranked[len(kept):]][:V9_DROPPED_ID_CAP]
    dbg["dropped_epi_total"] = max(0, len(ranked) - len(kept))

    kept_state = []
    if n_state > 0 and len(kept) < max_slots:
        st_hits = st.retrieve(q_core, top_k=n_state)
        room = max_slots - len(kept)
        kept_state = [int(i) for i, _s in st_hits][:room]
        dbg["dropped_state_ids"] = [int(i) for i, _s in st_hits][room:][:V9_DROPPED_ID_CAP]
        dbg["state_scores"] = [[int(i), round(float(s), 4)] for i, s in st_hits]

    # 用池自己的命名空间生成槽位 ID（epi/state），不硬编码前缀：
    # 一旦 attach 时换了 namespace，硬编码会让 gather 静默取错载荷。
    _uid_e = getattr(epi, "slot_uid", lambda i: f"epi:{int(i)}")
    _uid_s = getattr(st, "slot_uid", lambda i: f"state:{int(i)}")
    slot_ids = [_uid_e(i) for i in kept] + [_uid_s(i) for i in kept_state]
    dbg["epi_ids"] = kept
    dbg["state_ids"] = kept_state
    dbg["slot_ids"] = list(slot_ids)
    dbg["hits"] = hits
    dbg["budget_used"] = len(slot_ids) * tps
    return slot_ids, dbg


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
def latent_dry_run():
    """latent 模式的 dry-run 自检：不加载模型、不读 parquet、不初始化 CUDA。

    检查四件事（缺一项潜层读出就跑不出有意义的结果）：
      1) AR_MODE=v9（复用 v9 的寻址层）与预算能算出可读槽位数；
      2) LATENT_CKPT 存在（没有训练过的压缩器/桥接器，潜层读出只是随机投影）；
      3) 载荷库的命名空间与 gather 形状（epi / state 都从 0 开始，不加命名空间必然串味）；
      4) 路径预算按整条保留（不是 selected[:K] 截断），且不同 m 的 config_id 不同。
    """
    try:
        lm = latent_import()
    except Exception as e:                     # torch / 依赖缺失也要给出可读错误
        print(f"[dry-run][ERR] latent_memory 不可导入：{e}")
        return 2
    print(f"[dry-run] latent: {lm.describe()}")
    if MODE != "v9":
        print(f"[dry-run][ERR] AR_READOUT_MODE=latent 需要 AR_MODE=v9（复用 v9 寻址层），"
              f"当前 AR_MODE={MODE}")
        return 2
    if LATENT_SLOTS_MAX <= 0:
        print(f"[dry-run][ERR] 预算算不出可读槽位数："
              f"LATENT_READ_BUDGET={LATENT_READ_BUDGET} / LATENT_M={LATENT_M}")
        return 2
    if not LATENT_CKPT:
        print("[dry-run][ERR] 未设置 LATENT_CKPT（train_latent_memory.py 的产物）："
              "未训练的压缩器/桥接器在维度上合法，但冻结 Qwen 没有理由读懂它")
        return 2
    if not os.path.isfile(LATENT_CKPT):
        print(f"[dry-run][ERR] LATENT_CKPT 不存在：{LATENT_CKPT}")
        return 2
    if run_manifest is not None:
        print(f"[dry-run] latent checkpoint: "
              f"{run_manifest.file_fingerprint(LATENT_CKPT, label='checkpoint')}")
    print(f"[dry-run] latent budget: {LATENT_READ_BUDGET} soft tokens / m={LATENT_M} "
          f"-> K<={LATENT_SLOTS_MAX} slots（state 池附加最多 {V9_TOPK_STATE}）")
    if torch is None:
        print(f"[dry-run][warn] torch 不可用（{_TORCH_ERR}），跳过载荷/路径预算自检")
        return 0
    # 自检 1：命名空间 + gather 形状（epi:0 与 state:0 必须互不覆盖）
    bank = lm.LatentSlotBank(n_slots=LATENT_M, dim=8, dtype=torch.float32)
    bank.append("epi:0", torch.zeros(LATENT_M, 8))
    bank.append("state:0", torch.ones(LATENT_M, 8))
    z = bank.gather(["epi:0", "state:0"])
    print(f"[dry-run] latent bank: ids={bank.slot_ids()} gather={tuple(z.shape)} "
          f"epi0_is_zero={bool(float(z[0].abs().sum()) == 0.0)} "
          f"state0_is_one={bool(float(z[1].abs().sum()) == float(LATENT_M * 8))}")
    # 自检 2：路径预算整条保留（不 selected[:K] 截断）
    kept, info = lm.select_latent_cr_paths(
        [0, 1, 2, 3],
        [{"path_len": 2, "facts": [0, 2]}, {"path_len": 2, "facts": [1, 3]}],
        LATENT_READ_BUDGET, LATENT_M)
    print(f"[dry-run] latent path budget: kept={kept} "
          f"paths_kept={info['n_paths_kept']} skipped={len(info['n_paths_skipped'])} "
          f"overflow={info['overflow']}")
    # 自检 3：指纹隔离（不同 m 必须不同 config_id，否则两组消融共用一个结果文件）
    if run_manifest is not None:
        base = _manifest_params()
        ids = set()
        for m in (LATENT_M, LATENT_M + 1):
            p = dict(base)
            p["latent"] = dict(base["latent"], n_slots=m)
            cid = run_manifest.config_id(p, kind="ar", src=SRC)
            ids.add(cid)
            print(f"[dry-run] config_id(latent m={m}) = {cid}")
        print(f"[dry-run] 指纹隔离: "
              f"{'ok' if len(ids) == 2 else 'FAIL（不同 m 撞了同一个 id）'}")
    return 0


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
    init_manifest()
    print(f"[dry-run] SRC={SRC or '<unset>'} family={fam or '<unset>'} MODE={MODE} "
          f"GPU={GPU} OUT={OUT}")
    print(f"[dry-run] manifest: "
          f"{(run_manifest.describe(MANIFEST) if run_manifest else 'unavailable')}"
          f"{'' if _MANIFEST_ERR is None else f' ({_MANIFEST_ERR})'}")
    print(f"[dry-run] CHUNK={CHUNK} TOP_K={TOP_K} RERANK_M={RERANK_M} EQA_WIN={EQA_WIN} "
          f"CTX_MAX_TOK={CTX_MAX_TOK}")
    print(f"[dry-run] readout: mode={READOUT_MODE} "
          f"latent_slots={READOUT_MODE == 'latent'} "
          f"qtxt_no_gold={int(QTXT_NO_GOLD)} "
          f"(text_embeds = EMB[ids] 明文 token 嵌入，不是槽位潜向量)")
    print(f"[dry-run] strict_retrieval={int(strict_retrieval_enabled())} "
          f"(raw={STRICT_RETRIEVAL})")
    print(f"[dry-run] delta: WRITE={int(DELTA_WRITE)} RESTORE={int(DELTA_RESTORE)} "
          f"strategy={DELTA_WRITE_STRATEGY} topk={DELTA_WRITE_TOPK} "
          f"state_api={'ok' if get_delta_mem_online_state is not None else 'fallback'}"
          f"{'' if _STATE_API_ERR is None else f'({_STATE_API_ERR})'}")
    if READOUT_MODE == "latent":
        rc = latent_dry_run()
        if rc != 0:
            return rc
    if MODE == "v9":
        print(f"[dry-run] v9: EQA_BACK={EQA_BACK} EQA_AHEAD={EQA_AHEAD} "
              f"CTX_UNITS={V9_CTX_UNITS or 'off'} "
              f"CTX_TOKEN_BUDGET={V9_TOKEN_BUDGET or 'off'} "
              f"TOPK_STATE={V9_TOPK_STATE} "
              f"L18={'on' if V9_L18 else 'off'} layer={V9_L18_LAYER} "
              f"prompt_date={'on' if V9_PROMPT_DATE else 'off'} "
              f"prompt_serial={'on' if V9_PROMPT_SERIAL else 'off'}")

    # v9 拼块格式自检（纯函数，不碰模型）：日期前缀 / 编号递增 / DATE=0 时逐字节等于旧输出
    _demo_rows = [(3, "Alice likes tea.", "2023-05-20"),
                  (4, "She moved to Paris.", None),
                  (9, "Bob visited her in 2024.", "2024")]
    print("[dry-run] v9 prompt lines (DATE=1 SERIAL=1):")
    for _ln in render_v9_prompt_lines(_demo_rows, with_date=True, with_serial=True):
        print(f"           {_ln}")
    _old = [r[1] for r in _demo_rows]
    if render_v9_prompt_lines(_demo_rows, with_date=False, with_serial=False) != _old:
        print("[dry-run][ERR] DATE=0/SERIAL=0 未能还原旧输出（应逐字节等于原文逐行）")
        return 2
    print(f"[dry-run] v9 prompt lines (DATE=0 SERIAL=0) == 旧输出: "
          f"{render_v9_prompt_lines(_demo_rows, with_date=False, with_serial=False)}")

    # token 预算自检（t6）：分数降序装填、超预算即停、最高分必在块内、dropped 可解释
    _brows = {i: render_v9_prompt_lines([(i + 1, f"Evidence line {i}.", None)])[0]
              for i in range(12)}
    _btok = {i: prompt_tokens(t) for i, t in _brows.items()}
    _budget = sum(_btok[i] for i in range(3))          # 恰好装 3 行
    _ranked = list(range(12))                          # 已按分数降序（0 最高）
    _kept, _dropped, _used, _dtok = fill_token_budget(
        _ranked, _budget, lambda i: _brows[i])
    print(f"[dry-run] v9 budget: budget={_budget} kept={_kept} used_tokens={_used} "
          f"dropped={_dropped[:6]}{'...' if len(_dropped) > 6 else ''} "
          f"dropped_tokens={_dtok}")
    if (_used > _budget or 0 not in _kept or not _dropped
            or _kept != _ranked[:len(_kept)]):
        print("[dry-run][ERR] token 预算装填自检失败（应为分数序前缀、最高分在块内、"
              "总 token ≤ 预算、dropped 非空）")
        return 2
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

    # Delta 写入策略自检（纯正则/纯标准库）：合成上下文含重复与覆写事实，
    # 三档写入条数必须互相区分，且 filtered 只留最终值 —— 与单测同一套断言逻辑。
    try:
        _demo = ("Session 1\n2021-03-05\nAlice lives in Paris.\nAlice lives in Paris.\n"
                 "In 2021 Alice moved to Lyon.\nAlice is a doctor.\n"
                 "Alice is vegetarian.\nShe worked there for years.\n"
                 "Session 2\n2022-01-09\nAlice lives in London.")
        _demo_all = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", _demo) if s.strip()]
        for _st in DELTA_WRITE_STRATEGIES:
            _t, _s = plan_delta_writes(_demo, strategy=_st,
                                       questions=["Where does Alice live?"],
                                       all_chunks=_demo_all, topk=2)
            print(f"[dry-run] delta strategy={_st:<12} write={_s['write_chunks']} "
                  f"skipped={_s['skipped_chunks']} reasons={_s['skip_reasons']} "
                  f"texts={_t if _st != 'all' else '<all chunks>'}")
        _tf, _sf = plan_delta_writes(_demo, strategy="filtered")
        if not (len(_tf) < len(_demo_all) and any("London" in t for t in _tf)
                and not any("Paris" in t for t in _tf)):
            print("[dry-run][ERR] delta 过滤自检失败：filtered 未减少写入或未保留最终值")
            return 2
        print("[dry-run] delta 过滤自检 ok（filtered 条数 < all 且只保留最终值）")
    except Exception as e:
        print(f"[dry-run][warn] delta 写入策略自检失败: {e}")

    # 诊断字段自检：假池 + 1 条命中
    if ar_diag is not None:
        class _DryPool:
            texts = ["Alice moved to Paris in 1997."]
            metas = [{"start": 0}]
        d = ar_diag.record(pool=_DryPool(), hits=[(0, 1.0)], used_ids=[0],
                           answer="Paris", query="Where did Alice move?", flat=True)
        print(f"[dry-run] ar_diag.record ok; answer_in_retrieved="
              f"{d['answer_in_retrieved']} "
              f"answer_recall_at_k={d.get('answer_recall_at_k')} "
              f"evidence_recall_at_k={d.get('evidence_recall_at_k')}")
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
    if READOUT_MODE == "latent":
        # 潜层读出复用 v9 的寻址层（非重叠单元 / IDF-MaxSim / 时间约束 / EventQA 窗口）。
        if MODE != "v9":
            raise RuntimeError(
                f"[ERR] AR_READOUT_MODE=latent 需要 AR_MODE=v9（潜层读出复用 v9 的寻址层），"
                f"当前 AR_MODE={MODE}")
        if LATENT_SLOTS_MAX <= 0:
            raise SystemExit(
                f"[ERR] 潜层预算算不出可读槽位数：LATENT_READ_BUDGET={LATENT_READ_BUDGET} / "
                f"LATENT_M={LATENT_M}（两者都必须 > 0）")
        if not LATENT_CKPT:
            raise SystemExit(
                "[ERR] AR_READOUT_MODE=latent 需要 LATENT_CKPT（train_latent_memory.py 的产物）；"
                "未训练的压缩器/桥接器只是随机投影，跑出来的分数不能当方法效果。")
    init_manifest()
    init_templates()
    claim_gpu()     # 先占显存，再慢慢加载权重
    init_model()
    # AR 读出默认只读；写入只在 prepare_delta_memory 内显式开关。
    set_delta_mem_write_enabled(model, False)
    latent_reader = None
    if READOUT_MODE == "latent":
        # checkpoint 缺失/结构不符/hidden_size 不匹配 -> 直接报错（load_latent_reader 内）。
        latent_reader = latent_import().load_latent_reader(LATENT_CKPT, model, tok, device=DEV)
        print(f"[latent] {latent_import().describe()}", flush=True)
        print(f"[latent] reader={latent_reader} "
              f"ckpt_sha256_16={(latent_reader.fingerprint or {}).get('sha256_16')}", flush=True)
    if MODE == "v9" and V9_L18:
        # 第 18 层稠密通道：建池时按单元文本前向取隐状态，查询期同层 mean-pool。
        # strict=True：provider 失败/形状不符必须报错，禁止静默退化成纯词面。
        v9_import().set_layer18_provider(_layer18_provider, strict=True)
        print(f"[v9] layer18 dense channel enabled (strict): "
              f"{v9_slotpool.layer18_spec()}", flush=True)
    rows_all = pq.read_table(PQ).to_pylist()
    rows = [x for x in rows_all if (x.get("metadata") or {}).get("source") == SRC]
    if not rows:
        raise SystemExit(f"[ERR] parquet 内没有 source={SRC} 的行")
    if LIMIT_ROWS > 0:
        rows = rows[:LIMIT_ROWS]
    print(f"[cfg] SRC={SRC} MODE={MODE} CHUNK={CHUNK} TOP_K={TOP_K} RERANK_M={RERANK_M} "
          f"EQA_WIN={EQA_WIN} CTX_MAX_TOK={CTX_MAX_TOK} "
          f"DELTA_WRITE={int(DELTA_WRITE)} DELTA_STRATEGY={DELTA_WRITE_STRATEGY} "
          f"DELTA_TOPK={DELTA_WRITE_TOPK} READOUT={READOUT_MODE} "
          f"DELTA_RESTORE={int(DELTA_RESTORE)} strict_retrieval="
          f"{int(strict_retrieval_enabled())} rows={len(rows)}", flush=True)
    print(f"[manifest] {(run_manifest.describe(MANIFEST) if run_manifest else 'unavailable')} "
          f"OUT={OUT}", flush=True)
    if READOUT_MODE == "latent":
        print(f"[latent] budget={LATENT_READ_BUDGET} soft tokens, m={LATENT_M}, "
              f"K<={LATENT_SLOTS_MAX} slots, layer={LATENT_LAYER}, "
              f"max_unit_tokens={LATENT_MAX_UNIT_TOKENS}, ckpt={LATENT_CKPT}, "
              f"qtxt_no_gold={int(QTXT_NO_GOLD)}", flush=True)

    done = set()
    prev_sf = []
    if RESUME and os.path.exists(OUT):
        if run_manifest is not None and os.environ.get("AR_RESUME_MANIFEST", "1") == "1":
            ok_mf, why = run_manifest.check_resume(OUT, MANIFEST)
            if not ok_mf:
                raise SystemExit(
                    f"[resume][ERR] 拒绝续跑：{why}\n"
                    f"  已有文件：{OUT}\n"
                    f"  处理：删除该文件 / 指定新的 AR_OUT / "
                    f"AR_RESUME_MANIFEST=0（不建议，会把不同配置混在一个文件里）")
            print(f"[resume][manifest] {why}", flush=True)
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
    if run_manifest is not None:
        run_manifest.write(OUT, MANIFEST)      # 先落 manifest，崩溃后也能校验

    results = []
    t_start = time.time()
    latent_bank = None
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
        # 查询前完成 delta 写入并拿快照；先于建池，保证 L18 单元/query 编码看到同一记忆状态。
        # questions 只给 topk_refresh 做写入前的 Top-K 槽位粗排（其余策略忽略）。
        snap = prepare_delta_memory(ctx, questions=qs)
        if MODE == "v9":
            epi, st, n_tok = build_pool_for_context_v9(ctx)
            pool = epi
            print(f"[ctx {ci+1}/{len(rows)}] tokens={n_tok} episodic={len(epi)} "
                  f"state={len(st)} stats={epi.stats()} ({time.time()-t_start:.0f}s)", flush=True)
            if READOUT_MODE == "latent":
                # 编码每个槽位的**原始内容**一次（读出只用载荷）：编码期模型冻结、Delta 写入关闭。
                latent_bank, lb = latent_import().build_latent_bank(
                    epi=epi, state_pool=st, encoder=latent_reader.encoder,
                    max_slots=LATENT_BANK_MAX)
                # 挂钩子（encode_existing=False：刚建的载荷已在库里，不重复编码）。
                # 之后任何 _store/_replace/_drop 都会同步潜载荷，不必改检索算法。
                epi.attach_latent_bank(latent_bank, "epi", latent_reader.encoder,
                                       encode_existing=False)
                st.attach_latent_bank(latent_bank, "state", latent_reader.encoder,
                                      encode_existing=False)
                _per = {k: v.get("n_slots") for k, v in (lb.get("per_pool") or {}).items()}
                print(f"[latent] bank slots={len(latent_bank)} "
                      f"soft_vectors={lb['n_soft_tokens']} tokens_in={lb['n_tokens_in']} "
                      f"ratio={lb.get('compression_ratio')} per_pool={_per} "
                      f"skipped={lb['n_skipped']} ({time.time()-t_start:.0f}s)", flush=True)
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
            # 查询模板：latent 模式强制不用金标（{label} 填 gold 等于把答案塞进问题）
            qtxt, qtxt_label = build_qtxt(q, a)

            latent_slot_ids = []
            latent_z = None
            gen_meta = None
            try:
                if READOUT_MODE == "latent":
                    # 潜层读出：只做寻址 + 取载荷，**不构造任何明文证据行**
                    latent_slot_ids, dbg = select_latent_slots(epi, st, q, meta, qi)
                    block = ""
                    diag_ctx = {"pool": epi, "hits": dbg.get("hits") or [],
                                "used_ids": [], "used_texts": [],
                                "state_pool": st, "used_state_ids": [],
                                "pool_kind": "epi"}
                elif MODE == "v9":
                    block, dbg, diag_ctx = build_block_v9(epi, st, q, meta, qi)
                elif MODE == "v8":
                    block, dbg, diag_ctx = build_block_v8(epi, st, q, meta, qi)
                else:
                    block, dbg, diag_ctx = build_block(pool, q, meta, qi)
            except Exception as e:                      # 单题失败不中断整轮
                if strict_retrieval_enabled():
                    # 稠密/潜层通道开启时不允许静默吞掉检索异常：空证据块会让指标
                    # 骤降，看起来像"方法变差"，实际是设备/形状 bug。
                    raise RuntimeError(
                        f"retrieve fail q={qi} (AR_STRICT_RETRIEVAL={STRICT_RETRIEVAL}): {e}"
                    ) from e
                print(f"[warn] retrieve fail q={qi}: {e}", flush=True)
                block, dbg = "", {"mode": "error", "err": str(e)}
                _p = epi if MODE in ("v8", "v9") else pool
                diag_ctx = {"pool": _p, "hits": [], "used_ids": [], "used_texts": [],
                            "state_pool": st if MODE in ("v8", "v9") else None,
                            "used_state_ids": []}

            # 逐题恢复写后快照：前序题目的读出不会污染后续题目（查询期禁止 reset）。
            if DELTA_RESTORE:
                delta_restore(snap)

            if READOUT_MODE == "latent":
                if latent_bank is None:
                    raise RuntimeError("latent_bank 未构建（AR_MODE 必须是 v9 且建池成功）")
                latent_z = (latent_bank.gather(latent_slot_ids, device=DEV)
                            if latent_slot_ids else None)
                # 签名只接受 (model, tokenizer, bridge, payloads, question, system_prompt)：
                # 没有任何参数能把明文证据块塞回生成端。
                pred, gen_meta = latent_import().generate_latent_answer(
                    model=model, tokenizer=tok, bridge=latent_reader.bridge,
                    payloads=latent_z, question=qtxt, system_prompt=sys_msg,
                    max_new_tokens=MAX_NEW_TOKENS)
                in_len, out_len = int(gen_meta["n_in"]), int(gen_meta["n_out"])
                trim = None
            else:
                # OOM 重试要按证据优先级裁剪：把逐行槽位 ID / 优先级 / 行文本交给
                # generate_readout，重试明细回填到 trim["retries"]
                # （禁止"保留最后 N token"的静默截断）。
                trim = None
                if dbg.get("line_ids") and diag_ctx.get("used_texts"):
                    trim = {"lines": list(diag_ctx["used_texts"]),
                            "prio": list(dbg.get("line_prio") or []),
                            "ids": list(dbg.get("line_ids") or [])}
                pred, in_len, out_len = generate_readout(block, qtxt, trim=trim)

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
                # block_chunks = 真正拼进 prompt 的行数（预算模式下与 readout 的 Top-K 不同）
                "retrieval": dbg,
                "block_chunks": len(diag_ctx.get("used_texts") or dbg.get("idxs") or []),
                # 证据预算账本（t6）：预算值、实际装入 token、被裁掉的槽位/token，
                # 以及 OOM 重试的 keep_ids/dropped_ids —— 全部落盘，不静默丢证据。
                "ctx_budget": dbg.get("ctx_budget"),
                "ctx_tokens": dbg.get("ctx_tokens"),
                "dropped_ids": dbg.get("dropped_ids", []),
                "dropped_tokens": dbg.get("dropped_tokens", 0),
                "gen_retries": (trim or {}).get("retries", []),
                "delta": {"write": bool(DELTA_WRITE), "restore": bool(DELTA_RESTORE),
                          "strategy": DELTA_WRITE_STRATEGY,
                          "snapshot_layers": len(snap),
                          # 写入量/跳过量/耗时：证明"Delta 有没有真正参与"而不是只看开关
                          "write_chunks": DELTA_STATS.get("write_chunks", 0),
                          "skipped_chunks": DELTA_STATS.get("skipped_chunks", 0),
                          "candidate_units": DELTA_STATS.get("candidate_units", 0),
                          "skip_reasons": DELTA_STATS.get("skip_reasons", {}),
                          "write_seconds": DELTA_STATS.get("write_seconds", 0.0)},
                "readout": (latent_import().latent_readout_record(
                    latent_reader, latent_slot_ids,
                    n_vectors=int(latent_z.shape[0]) if latent_z is not None else 0,
                    extra={"budget": LATENT_READ_BUDGET,
                           "budget_used": dbg.get("budget_used"),
                           "tokens_per_slot": LATENT_TOKENS_PER_SLOT,
                           "max_slots": LATENT_SLOTS_MAX,
                           "decode_mode": (gen_meta or {}).get("decode_mode"),
                           "n_sys_tokens": (gen_meta or {}).get("n_sys_tokens"),
                           "insert_at": (gen_meta or {}).get("insert_at"),
                           "has_payload": (gen_meta or {}).get("has_payload")})
                    if READOUT_MODE == "latent" else
                    {"mode": READOUT_MODE, "latent_slots": False,
                     "note": ("text_embeds 只是 EMB[ids] 的明文 token 嵌入，"
                              "不是槽位潜向量读出" if READOUT_MODE == "text_embeds"
                              else "prompt 模式：明文证据行直接拼进 prompt")}),
                "query_label": qtxt_label,
                "run_id": RUN_ID,
                "window": ({"back": EQA_BACK, "ahead": EQA_AHEAD}
                           if FAMILY == "eventqa" else None),
                **{k: v for k, v in m.items()}, **extra,
            }
            # 逐题诊断（answer_recall@K / answer_in_retrieved / answer_in_context）：
            # hits 必须是分数降序；used_ids/used_texts 是实际拼进 prompt 的槽位。
            if READOUT_MODE == "latent":
                # 潜层读出的账本：注入了哪些槽位、预算用掉多少、裁掉了哪些。
                # 与 ctx_budget/ctx_tokens 是**两套口径**（软 token vs 明文 token），不可混用。
                rec["latent"] = {
                    "slot_ids": list(latent_slot_ids),
                    "n_slots": len(latent_slot_ids),
                    "n_vectors": int(latent_z.shape[0]) if latent_z is not None else 0,
                    "budget": LATENT_READ_BUDGET,
                    "budget_used": dbg.get("budget_used"),
                    "tokens_per_slot": LATENT_TOKENS_PER_SLOT,
                    "dropped_epi_ids": dbg.get("dropped_epi_ids", []),
                    "dropped_epi_total": dbg.get("dropped_epi_total", 0),
                    "dropped_state_ids": dbg.get("dropped_state_ids", []),
                    "anchor": dbg.get("anchor"),
                    "anchor_forced": dbg.get("anchor_forced"),
                    "checkpoint": LATENT_CKPT,
                    "decode_mode": (gen_meta or {}).get("decode_mode"),
                }
            if ar_diag is not None:
                try:
                    rec.update(ar_diag.record(answer=a, query=q, flat=True, **diag_ctx))
                except Exception as e:                  # 诊断失败不影响主指标
                    rec["diag_status"] = "diag_error"
                    rec["diag_error"] = str(e)
            # EventQA 检索方向诊断：锚点单元本身是否含答案 vs 窗口是否覆盖答案。
            # 分开记才能区分"锚点定位失败"与"锚点之后覆盖不足"。
            if FAMILY == "eventqa" and ar_diag is not None and MODE == "v9":
                try:
                    ai = dbg.get("anchor")
                    if ai is not None and 0 <= int(ai) < len(epi.texts):
                        hit, _mt, _ix = ar_diag.answer_in_texts(a, [epi.texts[int(ai)]])
                        rec["anchor_answer_hit"] = hit
                    rec["anchor_window_hit"] = rec.get("answer_in_context")
                except Exception:
                    pass
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
