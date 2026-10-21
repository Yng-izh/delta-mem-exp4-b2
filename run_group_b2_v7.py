import sys, os, re, json, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/share/delta-Mem")
sys.path.insert(0, "/share/MemoryAgentBench")
import torch
import torch.nn.functional as F

# transformers / deltamem / MemoryAgentBench / pyarrow 只在真实跑分时需要。
# 放进 try：--dry-run 与 CPU 逻辑单测不得因缺依赖而失败（与 run_ar_v7.py 同款）。
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
    reset_delta_mem_states = set_delta_mem_write_enabled = None
    get_template = chunk_text_into_sentences = post_process = None
    pq = None
    _STACK_ERR = _e

# 可选：Delta 在线状态快照/恢复 API（老版本 deltamem 可能没有）。有则用官方实现，
# 没有则退回手工 clone/restore（与 run_ar_v7.delta_snapshot/delta_restore 同一语义）。
try:
    from deltamem.core.delta import (get_delta_mem_online_state,
                                     load_delta_mem_online_state,
                                     iter_delta_mem_modules)
except Exception:
    get_delta_mem_online_state = load_delta_mem_online_state = None
    iter_delta_mem_modules = None

import run_manifest  # 同目录：配置指纹 / 结果文件隔离 / 续跑校验

# =====================================================================
# B-v3：clean 重编码读出（fact-text slot pool）
#  存：全局编号事实提取（<10 条回退句切分）；delta 写入照常按 chunk 流式 forward；
#      每条事实对纯事实文本 tok(fact, add_special_tokens=False) 单独 forward 取 key
#      签名（主谓部分 = 切掉句尾 KEY_TAIL_CUT 个词的 embedding 均值）。
#  覆写：仅池内潜向量比较 + 文本替换；去中心化仅当 len(pool)>10，否则原始均值+L2。
#  读出：不拼接任何旧 chunk 的 KV（规避跨 chunk RoPE 错位）；把存活槽位事实
#      拼成干净 context 块，走 System+RAG 模板单次前向 generate。
# =====================================================================

CHUNK_TOKENS   = 512
THETA          = float(os.environ.get("B2_THETA", "0.72"))
POOL_MAX       = int(os.environ.get("B2_POOL_MAX", "5000"))
KEY_TAIL_CUT   = 2          # 主谓 key：切掉句尾最后 2 个词
WHITEN_MIN     = 10         # 去中心化仅当池内 key 数 > 10
MAX_NEW_TOKENS = 24

# CR 强化开关：v7=基线（默认，行为与历史结果完全一致）；v9=CR 强化路径
CR_MODE   = os.environ.get("CR_MODE", "v7").strip().lower()
CF_PROMPT = os.environ.get("V9_CR_CF_PROMPT", "1") == "1"   # 反事实强约束 Prompt（仅 v9）
if CR_MODE not in ("v7", "v9"):
    raise SystemExit(f"CR_MODE must be v7 or v9, got {CR_MODE!r}")

# Delta State 接入模式（**替代旧的 B2_KEEP_DELTA**：那个开关只进配置指纹，
# 从不控制查询前的 reset_delta_mem_states，所以 Delta State 永远不参与 CR 查询）：
#   off     —— 查询前清零（历史行为，逐字节不变）；
#   context —— 写入后快照、每题恢复（复用 AR 的 delta_snapshot/delta_restore 语义）；
#   refresh —— 在 context 之上，每题用本题 Top-K 检索结果重写这些槽位文本，
#              与 AR 侧 DELTA_WRITE_STRATEGY=topk_refresh 同口径（只刷新 Top-K）。
CR_DELTA_MODE = os.environ.get("CR_DELTA_MODE", "off").strip().lower() or "off"
CR_DELTA_MODES = ("off", "context", "refresh")
if CR_DELTA_MODE not in CR_DELTA_MODES:
    raise SystemExit(f"CR_DELTA_MODE must be one of {CR_DELTA_MODES}, got {CR_DELTA_MODE!r}")
CR_DELTA_REFRESH_TOPK = int(os.environ.get("CR_DELTA_REFRESH_TOPK", "6"))

# CR 读出方式（P1）：
#   prompt  默认：多跳寻址 -> 渲染明文证据块 -> 拼进 prompt（历史行为，逐字节不变）
#   latent  潜层读出：多跳**寻址** -> 取槽位载荷 Z -> LatentBridge 变 soft tokens，
#           明文事实行一行都不进 prompt。需要 CR_MODE=v9（寻址层在图检索上）与训练好的
#           LATENT_CKPT；预算按**软 token 数**（LATENT_READ_BUDGET）算，且按整条路径保留
#           —— 半条多跳路径没有意义（第二跳常排在后面）。
CR_READOUT_MODE = os.environ.get("CR_READOUT_MODE", "prompt").strip().lower() or "prompt"
if CR_READOUT_MODE not in ("prompt", "latent"):
    raise SystemExit(f"CR_READOUT_MODE must be prompt or latent, got {CR_READOUT_MODE!r}")
if CR_READOUT_MODE == "latent" and CR_MODE != "v9":
    raise SystemExit("CR_READOUT_MODE=latent 需要 CR_MODE=v9（潜层读出复用图检索寻址层）")
LATENT_CKPT            = os.environ.get("LATENT_CKPT", "").strip()
LATENT_M               = int(os.environ.get("LATENT_M", "4"))
LATENT_LAYER           = int(os.environ.get("LATENT_LAYER", "18"))
LATENT_INNER_DIM       = int(os.environ.get("LATENT_INNER_DIM", "512"))
LATENT_MAX_UNIT_TOKENS = int(os.environ.get("LATENT_MAX_UNIT_TOKENS", "64"))
LATENT_READ_BUDGET     = int(os.environ.get("LATENT_READ_BUDGET", "32"))
LATENT_BANK_MAX        = int(os.environ.get("LATENT_BANK_MAX", "0"))
latent_memory = None


def latent_import():
    """惰性导入 latent_memory（只有 CR_READOUT_MODE=latent 才需要 torch 侧神经网络）。"""
    global latent_memory
    if latent_memory is None:
        import importlib
        latent_memory = importlib.import_module("latent_memory")
    return latent_memory

# --dry-run：不加载模型、不读 parquet，只打印解析后的配置 + 指纹 + 纯逻辑自检
DRY_RUN = "--dry-run" in sys.argv[1:]
_ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
SRC = _ARGS[0] if _ARGS else "FactConsolidation"
GPU = _ARGS[1] if len(_ARGS) > 1 else "0"

# v9_cr 提供 CR 常量、Delta 模式纯函数（delta_plan/refresh_slots）与证据日志解析
# （block_fact_lines）；v7 模式也要用它解析证据行，故无条件导入（无模型副作用）。
import v9_cr        # noqa: E402
import v9_slotpool  # noqa: E402  _cr_params() 要读 v9 常量（tok/EMB 稍后注入）
import ar_diag      # noqa: E402  逐题检索证据日志（与 AR 同等级字段）

BASE = "/share/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
ADAPTER = "/share/hf_home/hub/models--declare-lab--delta-mem_qwen3_4b-instruct/snapshots/6a1a990e770974811b8fb27dab73430807f76cb5"
PQ = "/share/hf_home/hub/datasets--ai-hyz--MemoryAgentBench/snapshots/7ea066982b140a19337e17e60d45d4076e042faf/data/Conflict_Resolution-00000-of-00001.parquet"
# 输出路径由 init_run() 依据配置指纹决定：不同 CR 消融（beam/ego_k/rel_theta/窗口/
# path_view/cf_prompt）不再写进同一个文件；续跑前校验 manifest。
OUT = None
RUN_ID = ""


def _cr_params():
    """所有影响结果的 CR 配置（改这里必须同步 v9_cr / v9_slotpool 常量）。"""
    p = {"mode": CR_MODE, "theta": THETA, "pool_max": POOL_MAX,
         # 三档 Delta 模式必须进指纹：off/context/refresh 是互相对照的实验组，
         # 共用结果文件会让 B2_RESUME 把另一组的题目当成"已完成"直接跳过。
         "delta_mode": CR_DELTA_MODE, "delta_refresh_topk": CR_DELTA_REFRESH_TOPK,
         "key_tail_cut": KEY_TAIL_CUT,
         "chunk_tokens": CHUNK_TOKENS, "max_new_tokens": MAX_NEW_TOKENS,
         "whiten_min": WHITEN_MIN}
    if CR_MODE == "v9":
        p["v9_cr"] = {
            "beam": v9_cr.CR_BEAM, "ego_k": v9_cr.CR_EGO_K,
            "subj_gate": v9_cr.CR_SUBJ_GATE, "bridge": v9_cr.CR_BRIDGE,
            "hops": v9_cr.CR_HOPS, "w_subj": v9_cr.CR_W_SUBJ,
            "w_bridge": v9_cr.CR_W_BRIDGE, "w_pred": v9_cr.CR_W_PRED,
            "rel_theta": v9_slotpool.CR_REL_THETA,
            "state_theta": v9_slotpool.STATE_THETA,
            "pool_max": v9_cr.CR_POOL_MAX, "cf_prompt": v9_cr.CR_CF_PROMPT,
            "path_view": v9_cr.CR_PATH_VIEW, "paths": v9_cr.CR_PATHS,
            "hop1_m": v9_cr.CR_HOP1_M,
        }
    if CR_READOUT_MODE != "prompt":
        # 只有非 prompt 读出才加这一节：prompt 模式的 config_id 必须与历史结果逐字节一致，
        # 否则全部已完成的 CR 结果文件会失效、续跑被拒。
        try:
            fp = run_manifest.file_fingerprint(LATENT_CKPT, label="checkpoint")
        except Exception:
            fp = {"checkpoint": LATENT_CKPT, "exists": False}
        p["readout"] = {
            "mode": CR_READOUT_MODE, "checkpoint": fp,
            "n_slots": LATENT_M, "layer": LATENT_LAYER, "inner_dim": LATENT_INNER_DIM,
            "max_unit_tokens": LATENT_MAX_UNIT_TOKENS,
            "read_budget": LATENT_READ_BUDGET, "tokens_per_slot": LATENT_M,
            "bank_max": LATENT_BANK_MAX,
            "compression_ratio_max": (round(LATENT_M / LATENT_MAX_UNIT_TOKENS, 6)
                                      if LATENT_MAX_UNIT_TOKENS > 0 else None),
        }
    return p


def init_run():
    """建 manifest、确定输出文件、校验续跑；返回已完成题目集合。"""
    global OUT, RUN_ID
    m = run_manifest.build(kind="cr", src=SRC, params=_cr_params())
    RUN_ID = m["config_id"]
    prefix = "v9groupB2" if CR_MODE == "v9" else "v7groupB2"
    OUT = os.environ.get("B2_OUT") or run_manifest.default_out(
        "/share/exp4group/results", prefix, SRC, RUN_ID)
    print(f"[manifest] {run_manifest.describe(m)} OUT={OUT}", flush=True)
    done = set()
    if os.path.exists(OUT) and os.environ.get("B2_RESUME", "1") == "1":
        ok_mf, why = run_manifest.check_resume(OUT, m)
        if not ok_mf:
            raise SystemExit(f"[resume][ERR] 拒绝续跑：{why}\n  文件：{OUT}")
        with open(OUT) as f:
            for line in f:
                try:
                    done.add(int(json.loads(line)["question_idx"]))
                except Exception:
                    pass
        print(f"[resume] 已完成 {len(done)} 题（{why}）", flush=True)
    elif os.path.exists(OUT):
        os.remove(OUT)
    run_manifest.write(OUT, m)          # 先落 manifest，崩溃后也能校验
    return done

if not DRY_RUN and _STACK_ERR is not None:
    raise SystemExit(f"[fatal] 缺少运行依赖（--dry-run 之外必须装齐）：{_STACK_ERR}")

# ---------------- --dry-run：不加载模型 / 不读 parquet / 不占 GPU ----------------
if DRY_RUN:
    print(f"[dry-run] CR_MODE={CR_MODE} CR_DELTA_MODE={CR_DELTA_MODE} "
          f"CF_PROMPT={CF_PROMPT} delta_refresh_topk={CR_DELTA_REFRESH_TOPK} "
          f"CR_READOUT_MODE={CR_READOUT_MODE} SRC={SRC}",
          flush=True)
    _m = run_manifest.build(kind="cr", src=SRC, params=_cr_params())
    _prefix = "v9groupB2" if CR_MODE == "v9" else "v7groupB2"
    print(f"[dry-run] manifest {run_manifest.describe(_m)}", flush=True)
    print(f"[dry-run] OUT would be "
          f"{run_manifest.default_out('/share/exp4group/results', _prefix, SRC, _m['config_id'])}",
          flush=True)
    # 指纹隔离自检：三档 Delta 模式必须是三个不同的 config_id（否则共用结果文件）
    for _mode in CR_DELTA_MODES:
        _p = dict(_cr_params()); _p["delta_mode"] = _mode
        print(f"[dry-run] config_id(delta_mode={_mode:8s}) = "
              f"{run_manifest.config_id(_p, kind='cr', src=SRC)}", flush=True)
    # 纯逻辑自检：Delta 动作计划 + Top-K 刷新槽位选择（不碰模型）
    print(f"[dry-run] delta_plan({CR_DELTA_MODE}) = {v9_cr.delta_plan(CR_DELTA_MODE)}",
          flush=True)
    _hits = [(3, 0.91), (1, 0.88), (3, 0.90), (7, 0.70)]
    print(f"[dry-run] refresh_slots(hits, top_k={CR_DELTA_REFRESH_TOPK}) = "
          f"{v9_cr.refresh_slots(_hits, CR_DELTA_REFRESH_TOPK)}", flush=True)
    if CR_READOUT_MODE == "latent":
        # 潜层读出必须显式自检：缺 checkpoint / 预算算不出槽位 / 路径被截断，
        # 这三类错误在真实跑分里都会伪装成"方法变差"。
        if not LATENT_CKPT:
            print("[dry-run][ERR] CR_READOUT_MODE=latent 需要 LATENT_CKPT"
                  "（未训练的压缩器/桥接器只是随机投影）", flush=True)
            sys.exit(2)
        try:
            lm = latent_import()
        except Exception as e:                     # 没装 torch 也能看配置，但不能假装跑得通
            print(f"[dry-run][warn] latent_memory 不可导入（{e}），跳过载荷自检", flush=True)
            sys.exit(0)
        print(f"[dry-run] latent: {lm.describe()}", flush=True)
        print(f"[dry-run] latent checkpoint: "
              f"{run_manifest.file_fingerprint(LATENT_CKPT, label='checkpoint')}",
              flush=True)
        print(f"[dry-run] latent budget: {LATENT_READ_BUDGET} soft tokens / m={LATENT_M} "
              f"-> K<={LATENT_READ_BUDGET // max(LATENT_M, 1)} slots", flush=True)
        # 路径预算：整条保留（不 selected[:K] 截断，否则可能只留第一跳）
        _kept, _info = lm.select_latent_cr_paths(
            [0, 1, 2, 3],
            [{"path_len": 2, "facts": [0, 2]}, {"path_len": 2, "facts": [1, 3]}],
            LATENT_READ_BUDGET, LATENT_M)
        print(f"[dry-run] latent path budget: kept={_kept} "
              f"paths_kept={_info['n_paths_kept']} "
              f"skipped={len(_info['n_paths_skipped'])} overflow={_info['overflow']}",
              flush=True)
        # 指纹隔离：latent 与 prompt 必须是不同 config_id（否则两组结果互相覆盖）
        print(f"[dry-run] config_id(readout={CR_READOUT_MODE}) = "
              f"{run_manifest.config_id(_cr_params(), kind='cr', src=SRC)}", flush=True)
    print("[dry-run] OK（未加载模型、未占 GPU）", flush=True)
    sys.exit(0)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
        device_map=f"cuda:{GPU}", attn_implementation="sdpa")
cfg = HFDeltaMemConfig.from_pretrained(ADAPTER)
attach_delta_mem(model, cfg)
load_delta_mem_adapter(model, input_dir=ADAPTER)
model.eval()
NL = model.config.num_hidden_layers
NW = model.config.num_key_value_heads or model.config.num_attention_heads
HD = model.config.head_dim
EMB = model.get_input_embeddings().weight.detach()
DEV = f"cuda:{GPU}"

rows = pq.read_table(PQ).to_pylist()
row = [x for x in rows if x["metadata"]["source"] == SRC][0]
ctx, qs, ans = row["context"], row["questions"], row["answers"]

sys_msg = get_template(SRC, "system", "Long_context_agent_deltamem")
mem_tpl = get_template(SRC, "memorize", "Long_context_agent_deltamem")
qry_tpl = get_template(SRC, "query", "rag_agent")
def fmt(tpl, **kw):
    try: return tpl.format(time_stamp="", **kw)
    except KeyError: return tpl.format(**kw)

# ---------- (1) 全局事实提取 ----------
NUM_FACT_RE = re.compile(r"(?:^|\s)(\d+)\.\s+([^\n]+?)(?=(?:\s+\d+\.\s+)|\n|$)")
def extract_all_facts(text):
    """全局编号事实提取；<10 条时回退按句切分。"""
    facts = [(int(m.group(1)), m.group(2).strip())
             for m in NUM_FACT_RE.finditer(text) if m.group(2).strip()]
    if len(facts) < 10:
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        facts = [(i + 1, s) for i, s in enumerate(parts)]
    return facts

print("=== ctx[:500] ===", flush=True)
print(ctx[:500], flush=True)

VERB_EDGES = ("affiliated with", "associated with", "created in", "created by",
              "famous for", "educated at", "author of", "language of",
              "citizenship of", "head of", "founded", "written in",
              "plays", "played", "play", "is", "was", "are", "were")
VERB_PAT = re.compile(r"\b(" + "|".join(re.escape(v) for v in VERB_EDGES) + r")\b")
def strip_object(fact):
    """按系动词/谓语边界（词边界匹配）精准剥宾语；兜底切尾 3 词。"""
    m = VERB_PAT.search(fact.lower())
    if m and m.start() > 0:
        return fact[:m.start()].strip()
    toks = fact.split()
    if len(toks) <= 3:
        return fact
    return " ".join(toks[:-3])

import string as _string
def group_id(fact):
    """主谓组标识：strip_object 后归一化（小写、去标点、空白合一）。"""
    t = strip_object(fact).lower()
    t = t.translate(str.maketrans("", "", _string.punctuation))
    return " ".join(t.split())

STOPWORDS = set("the a an of in on at to for with by from is was are were "
                "be been what which who where when current name does did".split())
def weighted_key_embed(text):
    """STOPWORDS 过滤 + 非句首大写词 2.5 倍权重的加权 embedding。"""
    words = text.split()
    kept, weights = [], []
    for i, w in enumerate(words):
        if w.lower().strip(",.!?;:'\"()") in STOPWORDS:
            continue
        cap = w[:1].isupper() and i > 0
        kept.append(w)
        weights.append(2.5 if cap else 1.0)
    if not kept:
        kept, weights = words, [1.0] * len(words)
    # token 与词数不一致（子词切分），用 offset mapping 把词权重摊到 token 上
    enc = tok(" ".join(kept), return_tensors="pt", truncation=True, max_length=64,
              return_offsets_mapping=True)
    ids = enc.input_ids[0].to(EMB.device)
    spans = enc.offset_mapping[0].tolist()
    # 重建每个 kept 词在拼接文本中的字符起点
    starts, pos = [], 0
    for wd in kept:
        starts.append(pos)
        pos += len(wd) + 1
    tok_w = []
    for (a, b) in spans:
        if a == b:                      # special/empty
            tok_w.append(0.0); continue
        wi = 0
        for i, st in enumerate(starts):
            if a >= st:
                wi = i
            else:
                break
        tok_w.append(weights[wi])
    w = torch.tensor(tok_w, dtype=torch.float32, device=EMB.device)
    if w.sum() <= 0:
        w = torch.ones_like(w)
    w = w / w.sum()
    return F.normalize((EMB[ids].float() * w.unsqueeze(1)).sum(dim=0), dim=0)

def embed_norm(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(EMB.device)
    return F.normalize(EMB[ids[0]].mean(dim=0), dim=0)

# ---------- (2) 纯事实文本单独 forward 取 KV ----------
def fact_kv(fact_text):
    ids = tok(fact_text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    kv = []
    for layer in range(NL):
        kl = past.layers[layer]
        kv.append((kl.keys.clone(), kl.values.clone()))
    return tuple(kv)

# ---------- (5) key 池：去中心化仅当 len(pool)>10 ----------
class SlotPool:
    def __init__(self):
        self.keys = []          # raw embedding mean (未去中心化)
        self.texts = []         # 存活槽位存事实文本
        self.serial = []
        self.t0 = []
        self.clock = 0
        self.n_overwrites = 0
        self.n_evicts = 0
        self.n_group_overwrites = 0
        self.groups = {}        # group_id -> slot index (最新 serial 槽位)

    def __len__(self):
        return len(self.keys)

    def _drop_slot(self, idx):
        gid = getattr(self, '_slot_groups', None)
        if gid is not None:
            gid.pop(idx, None)

    def _sims(self, key_raw):
        """返回与池内各槽位的余弦相似度。"""
        if not self.keys:
            return None
        M = torch.stack(self.keys).float()
        k = key_raw.float()
        if len(self.keys) > WHITEN_MIN:      # 去中心化
            mu = M.mean(dim=0, keepdim=True)
            M = M - mu
            k = k - mu.squeeze(0)
        M = F.normalize(M, dim=1)
        k = F.normalize(k, dim=0)
        return M @ k

    def add(self, key_raw, text, serial, gid=None):
        if not hasattr(self, '_slot_groups'):
            self._slot_groups = [None] * len(self.keys)
        # ③ 主谓组去重：同组新事实无条件替换组内旧槽位，只留最新 serial
        if gid is not None and gid in self.groups:
            old_i = self.groups[gid]
            if self.serial[old_i] < serial:
                self.keys[old_i] = key_raw
                self.texts[old_i] = text
                self.serial[old_i] = serial
                self.t0[old_i] = self.clock; self.clock += 1
                self.n_group_overwrites += 1
                return "group_overwrite"
            return "stale_skip"
        self.keys.append(key_raw); self.texts.append(text)
        self.serial.append(serial); self.t0.append(self.clock); self.clock += 1
        self._slot_groups.append(gid)
        if gid is not None:
            self.groups[gid] = len(self.keys) - 1
        while len(self.keys) > POOL_MAX:
            self._evict_oldest()
        return "new"

    def _evict_oldest(self):
        idx = min(range(len(self.t0)), key=lambda i: self.t0[i])
        gid = self._slot_groups[idx]
        if gid is not None and self.groups.get(gid) == idx:
            self.groups.pop(gid, None)
        for lst in (self.keys, self.texts, self.serial, self.t0, self._slot_groups):
            lst.pop(idx)
        # groups 指向的索引整体左移
        for g, i in list(self.groups.items()):
            if i > idx:
                self.groups[g] = i - 1
        self.n_evicts += 1

    def try_overwrite(self, key_raw, text, serial):
        sims = self._sims(key_raw)
        if sims is None:
            return False
        top = int(sims.argmax())
        if float(sims[top]) > THETA:
            gid = self._slot_groups[top]
            self.texts[top] = text          # 文本替换
            self.keys[top] = key_raw
            self.serial[top] = serial
            self.t0[top] = self.clock; self.clock += 1
            self.n_overwrites += 1
            return True
        return False

    def context_block(self):
        """存活槽位事实拼成干净 context 块。"""
        order = sorted(range(len(self.serial)), key=lambda i: self.serial[i])
        return "\n".join(f"{self.serial[i]}. {self.texts[i]}" for i in order)

    def retrieve_context_block(self, query_text, top_k=15):
        """池≤top_k 全取；否则按问题加权 key 检索 Top-K，按 serial 排序拼块。"""
        if len(self.serial) <= top_k:
            return self.context_block()
        # 问题不 strip_object：strip_object 会把 "What is ..." 截成 "What"，破坏问题 key
        qk = weighted_key_embed(query_text)
        sims = self._sims(qk)
        if sims is None:
            return self.context_block()
        sims_np = sims.tolist()
        # 相似度按 serial 降序打破平局（新鲜度偏置）
        ranked = sorted(range(len(sims_np)), key=lambda i: (-float(sims_np[i]), -self.serial[i]))
        top = ranked[:top_k]
        order = sorted(top, key=lambda i: self.serial[i])
        return "\n".join(f"{self.serial[i]}. {self.texts[i]}" for i in order)

def _v7_diag_ctx(pool, block):
    """v7 池的逐题证据：只记录"实际拼进 prompt 的事实行"，不伪造检索排名。

    v7 SlotPool.retrieve_multihop_context 不返回分数序候选（没有 hits），
    所以 retrieved_* 留空、diag_status 标成 hits_not_exposed —— 宁可说"没暴露"，
    也不能把 used 顺序当成检索排名。
    """
    lines = v9_cr.block_fact_lines(pool, block)
    return {"pool": pool, "hits": [], "used_ids": [i for i, _ in lines],
            "used_texts": [t for _, t in lines], "pool_kind": "cr"}


# ---------------- Delta 在线状态：快照 / 恢复 / Top-K 刷新 ----------------
def delta_snapshot(model):
    """把当前 delta 在线状态 clone 到 CPU（每题恢复的基线）。语义同 run_ar_v7。"""
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
        p0 = next(m.parameters())
        m.delta_state = t.to(device=p0.device, dtype=p0.dtype)


def refresh_delta_texts(texts):
    """用本题 Top-K 证据文本重写 Delta（CR_DELTA_MODE=refresh）。返回写入条数。

    与 AR 侧 DELTA_WRITE_STRATEGY=topk_refresh 同口径：只刷新 Top-K 槽位，
    不重写整篇 context。写入前必须已经 delta_restore 到写后基线，否则会叠加上
    前一题的残留状态。
    """
    texts = [t for t in (texts or []) if str(t).strip()]
    if not texts:
        return 0
    set_delta_mem_write_enabled(model, True)
    try:
        with torch.no_grad():
            for t in texts:
                txt = tok.apply_chat_template(
                    [{"role": "system", "content": sys_msg},
                     {"role": "user", "content": fmt(mem_tpl, context=str(t))}],
                    add_generation_prompt=True, tokenize=False)
                ids = tok(txt, return_tensors="pt").input_ids.to(DEV)
                model(input_ids=ids, use_cache=True)
    finally:
        set_delta_mem_write_enabled(model, False)
    return len(texts)


DELTA_PLAN = (v9_cr.delta_plan(CR_DELTA_MODE) if CR_MODE == "v9" else
              {"mode": CR_DELTA_MODE, "reset_before_query": CR_DELTA_MODE == "off",
               "restore_snapshot": CR_DELTA_MODE != "off",
               "refresh_topk": CR_DELTA_REFRESH_TOPK if CR_DELTA_MODE == "refresh" else 0})
DELTA_SNAP = None

reset_delta_mem_states(model)      # 写前清零：两种模式都要（清掉上一 context 残差）
if CR_MODE == "v9":
    import v9_slotpool  # noqa: E402
    import v9_cr        # noqa: E402
    v9_slotpool.tok = tok
    v9_slotpool.EMB = EMB
    v9_cr.set_encoder(tok, EMB)
    pool = v9_cr.CRStatePool()
else:
    import v7_slotpool  # noqa: E402  (needs re/torch/F/tok/EMB in scope)
    v7_slotpool.tok = tok
    v7_slotpool.EMB = EMB
    from v7_slotpool import *  # noqa: E402,F403  覆盖本地 v4 SlotPool
    pool = SlotPool()
DELTA_PLAN["n_pool"] = len(pool)          # 诊断用：池规模写进计划
print(f"CR_MODE={CR_MODE} pool={type(pool).__name__} delta_mode={CR_DELTA_MODE} "
      f"pool_n={len(pool)}", flush=True)

all_facts = extract_all_facts(ctx)
print(f"extracted facts: {len(all_facts)} (numbered+fallback)", flush=True)
for s, f in all_facts[:5]:
    print(f"  fact[{s}] {f[:80]}", flush=True)

print(f"streaming... THETA={THETA} POOL={POOL_MAX}", flush=True)
with torch.no_grad():
    # (4) delta 写入仍按 chunk 流式 forward（与官方口径一致），但不从 chunk KV 截句子
    for ci, ch in enumerate(chunks := chunk_text_into_sentences(ctx, chunk_size=CHUNK_TOKENS)):
        txt = tok.apply_chat_template(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": fmt(mem_tpl, context=ch)}],
            add_generation_prompt=True, tokenize=False)
        ids = tok(txt, return_tensors="pt").input_ids.to(DEV)
        model(input_ids=ids, use_cache=True)
        if (ci + 1) % 5 == 0 or ci == len(chunks) - 1:
            print(f"chunk {ci+1}/{len(chunks)} pool={len(pool)} "
                  f"ow={pool.n_overwrites} ev={pool.n_evicts}", flush=True)
    # 事实槽位：v5 三元组解耦潜键 upsert（SUBJ+REL 双重命中原位覆写，同步出键 k_obj）
    for serial, fact in all_facts:
        pool.upsert(fact, serial)
set_delta_mem_write_enabled(model, False)
print(f"write done. pool_final={len(pool)} overwrites={pool.n_overwrites}", flush=True)

# 潜层载荷（CR_READOUT_MODE=latent）：每个**存活**事实槽位压成 Z 一次，之后读出只用载荷。
# 为什么在写入完成之后才编码：CR 的双阈值覆写发生在写入期，被覆写的旧值已经不存在于槽位里，
# 只需要为最终文本建载荷（等价于 Hard Overwrite：Alice→Microsoft 被 Alice→Raytheon 覆盖后，
# 槽里只剩 Raytheon，载荷自然也只有 Raytheon）。
# 挂钩子（encode_existing=False：载荷刚建好，不重复编码）保证后续 upsert/_evict 也同步。
latent_reader = None
latent_bank = None
if CR_READOUT_MODE == "latent":
    if not LATENT_CKPT:
        raise SystemExit("[ERR] CR_READOUT_MODE=latent 需要 LATENT_CKPT"
                         "（train_latent_memory.py 的产物）")
    latent_reader = latent_import().load_latent_reader(LATENT_CKPT, model, tok, device=DEV)
    latent_bank, lb = latent_import().build_latent_bank(
        state_pool=pool, encoder=latent_reader.encoder, namespaces={"state": "cr"},
        max_slots=LATENT_BANK_MAX)
    pool.attach_latent_bank(latent_bank, "cr", latent_reader.encoder,
                            encode_existing=False)
    print(f"[latent] {latent_import().describe()}", flush=True)
    print(f"[latent] reader={latent_reader} bank_slots={len(latent_bank)} "
          f"soft_vectors={lb['n_soft_tokens']} tokens_in={lb['n_tokens_in']} "
          f"ratio={lb.get('compression_ratio')} skipped={lb['n_skipped']} "
          f"budget={LATENT_READ_BUDGET}/m={LATENT_M}->K<={LATENT_READ_BUDGET // LATENT_M}",
          flush=True)

# ---------- (4) 读出：clean context 重编码，单次前向 generate ----------
# Delta 状态：context/refresh 取"写后快照"供每题恢复；off 则查询前清零（历史行为）。
if DELTA_PLAN["restore_snapshot"]:
    DELTA_SNAP = delta_snapshot(model)
    print(f"[delta] mode={CR_DELTA_MODE} snapshot_layers={len(DELTA_SNAP)} "
          f"refresh_topk={DELTA_PLAN['refresh_topk']}", flush=True)
else:
    reset_delta_mem_states(model)
    print(f"[delta] mode={CR_DELTA_MODE} reset_before_query=1（Delta 不参与查询）", flush=True)
done_q = init_run()
results = []
with torch.no_grad():
    for qi, (q, a) in enumerate(zip(qs, ans)):
        if qi in done_q:
            continue
        try: qtxt = qry_tpl.format(question=q)
        except KeyError: qtxt = qry_tpl.format(query=q)
        # 逐题恢复"写后快照"：前序题目的读出不得污染后续题目（查询期不 reset）
        if DELTA_PLAN["restore_snapshot"]:
            delta_restore(DELTA_SNAP)
        latent_slot_ids, latent_z, gen_meta, keep_info = [], None, None, None
        if CR_READOUT_MODE == "latent":
            # 只做寻址：拿到槽位 ID 与**完整路径**，不渲染任何明文证据块。
            ids, paths, cdbg = pool.retrieve_multihop_ids(q)
            latent_slot_ids, keep_info = latent_import().select_latent_cr_paths(
                ids, paths, LATENT_READ_BUDGET, LATENT_M)
            latent_z = (latent_bank.gather(latent_slot_ids, device=DEV)
                        if latent_slot_ids else None)
            block = ""
            # 诊断口径：retrieved 仍按 hop1 分数降序记录（检索侧命中率可单独报告），
            # 但 used_texts/used_ids 保持空 —— 没有任何明文进 prompt，
            # 因此 ar_diag 的 answer_in_context 会是 None 而不是被误记成 False。
            diag = {"pool": pool,
                    "hits": [(int(i), float(s)) for s, i in (cdbg.get("hop1") or [])],
                    "used_ids": [], "used_texts": [],
                    "used_state_ids": [], "state_pool": None}
        else:
            block = pool.retrieve_multihop_context(q)
            # 逐题检索证据（与 AR 同等级字段）：检索给了什么 / 实际拼进 prompt 什么
            diag = (pool.diag_ctx(block) if CR_MODE == "v9"
                    else _v7_diag_ctx(pool, block))
        # refresh：用本题 Top-K 检索结果重写这些槽位（AR 的 topk_refresh 同口径）
        ref_slots, n_refresh = [], 0
        if DELTA_PLAN["refresh_topk"] and diag.get("hits"):
            ref_slots = v9_cr.refresh_slots(diag["hits"], DELTA_PLAN["refresh_topk"])
            n_refresh = refresh_delta_texts([pool.texts[i] for i in ref_slots
                                             if 0 <= i < len(pool.texts)])
        if CR_READOUT_MODE == "latent":
            # 生成端只接收 system(正常 token) + soft tokens + 问题：
            # generate_latent_answer 的签名里没有任何参数能塞回明文证据块。
            pred, gen_meta = latent_import().generate_latent_answer(
                model=model, tokenizer=tok, bridge=latent_reader.bridge,
                payloads=latent_z, question=qtxt, system_prompt=sys_msg,
                max_new_tokens=MAX_NEW_TOKENS)
            in_len, out_len = int(gen_meta["n_in"]), int(gen_meta["n_out"])
        else:
            if CR_MODE == "v9" and CF_PROMPT:
                user_msg = v9_cr.build_cr_prompt(qtxt, block)   # 反事实强约束 Prompt
            else:
                user_msg = f"{block}\n\n{qtxt}" if block else qtxt
            text = tok.apply_chat_template(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": user_msg}],
                add_generation_prompt=True, tokenize=False)
            ids = tok(text, return_tensors="pt").input_ids.to(DEV)
            out = model.generate(input_ids=ids, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, use_cache=True,
                                 pad_token_id=tok.eos_token_id or 0)
            pred = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            in_len, out_len = int(ids.shape[1]), int(out.shape[1] - ids.shape[1])
        pred_norm = pred.strip()
        pred_norm = re.sub(r"^(?:answer|答案)\s*[:：]\s*", "", pred_norm, flags=re.IGNORECASE).strip().lower()  # 大小写归一，避免 "Rugby" vs "rugby" 白丢分
        ans_norm = [x.strip().lower() if isinstance(x, str) else [g.strip().lower() for g in x] if isinstance(x, list) else x for x in [a]][0]
        m, _ = post_process(
            {"output": pred_norm, "input_len": in_len,
             "output_len": out_len,
             "memory_construction_time": 0, "query_time_len": 0},
            ans_norm, {"sub_dataset": SRC, "debug": False})
        rec = {"question_idx": qi, "query": q, "answer": a, "output": pred,
               "run_id": RUN_ID, **m}
        if CR_READOUT_MODE == "latent":
            # 潜层读出的账本：注入了哪些槽位 / 预算用掉多少 / 哪些路径因预算被整条跳过。
            # 与 CR 的明文证据行是两套口径（软 token vs 明文 token），不可混用。
            rec["readout"] = latent_import().latent_readout_record(
                latent_reader, latent_slot_ids,
                n_vectors=int(latent_z.shape[0]) if latent_z is not None else 0,
                extra={"budget": LATENT_READ_BUDGET,
                       "budget_used": (keep_info or {}).get("budget_used"),
                       "tokens_per_slot": LATENT_M,
                       "n_paths_kept": (keep_info or {}).get("n_paths_kept"),
                       "n_paths_skipped": len((keep_info or {}).get("n_paths_skipped") or []),
                       "overflow": (keep_info or {}).get("overflow"),
                       "decode_mode": (gen_meta or {}).get("decode_mode")})
            rec["latent"] = {"slot_ids": list(latent_slot_ids),
                             "n_slots": len(latent_slot_ids),
                             "n_vectors": int(latent_z.shape[0]) if latent_z is not None else 0,
                             "budget": LATENT_READ_BUDGET,
                             "budget_used": (keep_info or {}).get("budget_used"),
                             "kept_paths": (keep_info or {}).get("kept_paths"),
                             "skipped_paths": (keep_info or {}).get("n_paths_skipped"),
                             "overflow": (keep_info or {}).get("overflow"),
                             "checkpoint": LATENT_CKPT,
                             "decode_mode": (gen_meta or {}).get("decode_mode")}
        if CR_MODE == "v9":
            rec["cr_dbg"] = pool.last_dbg
        # Delta 有没有真正参与：模式 / 快照层数 / 本题刷新了哪些槽位（不是只看开关）
        rec["delta"] = {"mode": DELTA_PLAN["mode"],
                        "restore_snapshot": bool(DELTA_PLAN["restore_snapshot"]),
                        "snapshot_layers": len(DELTA_SNAP or {}),
                        "refresh_topk": DELTA_PLAN["refresh_topk"],
                        "refresh_slots": ref_slots, "n_refresh": n_refresh}
        # 逐题检索证据日志（retrieved_slot_ids/scores/texts + used_slot_ids/texts +
        # 跨池统一 ID 'cr:<idx>'）：与 AR 的 ar_diag 同等级，供 diag-attribution 归因。
        rec.update(ar_diag.record(answer=a, query=q, flat=True, **diag))
        if CR_MODE != "v9" and not diag.get("hits"):
            # v7 池不暴露分数序候选：显式标注"没暴露"，而不是让它看起来像"没命中"
            rec["diag_status"] = "hits_not_exposed"
        results.append(rec)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (qi + 1) % 20 == 0:
            sf = sum(r["substring_exact_match"] for r in results) / len(results)
            print(f"{qi+1}/{len(qs)} running SF={sf:.3f}", flush=True)
# 汇总一律以文件为准（续跑时 results 只含本轮新题，不能拿它算总分）
all_recs = []
with open(OUT) as f:
    for line in f:
        if line.strip():
            try:
                all_recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
sf = sum(r.get("substring_exact_match", 0) for r in all_recs) / max(len(all_recs), 1)
print(f"FINAL groupB2 {SRC} [{CR_MODE}] run={RUN_ID}: SF={sf:.4f} n={len(all_recs)} "
      f"pool={len(pool)} overwrites={pool.n_overwrites} readout={CR_READOUT_MODE}", flush=True)

# CR 答案命中率（与 AR 的 answer_recall_at_k 同口径，分母只算可计算的记录）：
#   hit@K  = gold 出现在 Top-K 检索槽位文本里的比例（检索侧）；
#   hit_ctx= gold 出现在实际拼块文本里的比例（读出侧）。
# 分母为 0（v7 未暴露 hits / 全是 rubric 题）时打印 n=0 而不是 0.0000，避免误读。
_app = [r for r in all_recs if r.get("answer_in_retrieved") is not None]
_ctx = [r for r in all_recs if r.get("answer_in_context") is not None]


def _rate(rs, key):
    return (sum(1 for r in rs if r.get(key)) / len(rs)) if rs else None


_hit_k, _hit_ctx = _rate(_app, "answer_in_retrieved"), _rate(_ctx, "answer_in_context")
_fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
print(f"FINAL-CR {SRC} [{CR_MODE}] delta={CR_DELTA_MODE} run={RUN_ID}: "
      f"answer_hit@K={_fmt(_hit_k)} (n={len(_app)}) "
      f"answer_hit_ctx={_fmt(_hit_ctx)} (n={len(_ctx)}) "
      f"n_attr_rejected={getattr(pool, 'n_attr_rejected', 'n/a')}", flush=True)
