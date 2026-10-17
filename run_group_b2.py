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
KEEP_DELTA     = os.environ.get("B2_KEEP_DELTA", "1") == "1"
KEY_TAIL_CUT   = 2          # 主谓 key：切掉句尾最后 2 个词
WHITEN_MIN     = 10         # 去中心化仅当池内 key 数 > 10
MAX_NEW_TOKENS = 24

SRC = sys.argv[1]; GPU = sys.argv[2]
BASE = "/share/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
ADAPTER = "/share/hf_home/hub/models--declare-lab--delta-mem_qwen3_4b-instruct/snapshots/6a1a990e770974811b8fb27dab73430807f76cb5"
PQ = "/share/hf_home/hub/datasets--ai-hyz--MemoryAgentBench/snapshots/7ea066982b140a19337e17e60d45d4076e042faf/data/Conflict_Resolution-00000-of-00001.parquet"
OUT = f"/share/exp4group/results/groupB2v5_{SRC}.jsonl"

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

reset_delta_mem_states(model)
import v5_slotpool  # noqa: E402  (needs re/torch/F/tok/EMB in scope)
v5_slotpool.tok = tok
v5_slotpool.EMB = EMB
from v5_slotpool import *  # noqa: E402,F403  覆盖本地 v4 SlotPool
pool = SlotPool()

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

# ---------- (4) 读出：clean context 重编码，单次前向 generate ----------
reset_delta_mem_states(model)
results = []
with torch.no_grad():
    for qi, (q, a) in enumerate(zip(qs, ans)):
        try: qtxt = qry_tpl.format(question=q)
        except KeyError: qtxt = qry_tpl.format(query=q)
        block = pool.retrieve_multihop_context(q)
        text = tok.apply_chat_template(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": f"{block}\n\n{qtxt}" if block else qtxt}],
            add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt").input_ids.to(DEV)
        out = model.generate(input_ids=ids, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, use_cache=True,
                             pad_token_id=tok.eos_token_id or 0)
        pred = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        pred_norm = pred.strip().lower()  # 大小写归一，避免 "Rugby" vs "rugby" 白丢分
        ans_norm = [x.strip().lower() if isinstance(x, str) else [g.strip().lower() for g in x] if isinstance(x, list) else x for x in [a]][0]
        m, _ = post_process(
            {"output": pred_norm, "input_len": ids.shape[1],
             "output_len": out.shape[1] - ids.shape[1],
             "memory_construction_time": 0, "query_time_len": 0},
            ans_norm, {"sub_dataset": SRC, "debug": False})
        rec = {"question_idx": qi, "query": q, "answer": a, "output": pred, **m}
        results.append(rec)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (qi + 1) % 20 == 0:
            sf = sum(r["substring_exact_match"] for r in results) / len(results)
            print(f"{qi+1}/{len(qs)} running SF={sf:.3f}", flush=True)
sf = sum(r["substring_exact_match"] for r in results) / len(results)
print(f"FINAL groupB2 {SRC}: SF={sf:.4f} n={len(results)} "
      f"pool={len(pool)} overwrites={pool.n_overwrites}", flush=True)
