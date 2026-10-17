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
# B-v2：流式句子级 KV 槽位原位覆写（KV Slot Pool in-place overwrite）
#  存：逐 512-token chunk 流式前向，按句切事实（"N. " 行），
#      每条事实句子在 forward 后从各层 past_key_values 里截出该句 token
#      区间的 KV 片段存入槽位；Key 签名 = 该句主谓部分 token embedding 均值，
#      去中心化（减池内均值）后 L2 归一。
#  覆写：新事实 key 与池内旧 key 余弦 > THETA → 旧槽位 KV 原位替换为新句 KV。
#  池满：FIFO 淘汰最旧且从未被命中过的槽位。
#  读出：把存活槽位的 KV 拼成 past_key_values，问题单独前向做 attention；
#      delta state 同时保留（问题前向时关写，delta 态照常读出辅助）。
# =====================================================================

CHUNK_TOKENS   = 512
THETA          = float(os.environ.get("B2_THETA", "0.70"))
POOL_MAX       = int(os.environ.get("B2_POOL_MAX", "512"))
KEEP_DELTA     = os.environ.get("B2_KEEP_DELTA", "1") == "1"
KEY_TAIL_CUT   = 5          # 主谓 key：去掉句尾 5 个 token（宾语/新值）
SIM_PRUNE_THR  = 0.75       # 诊断 2 直方图双峰分界

SRC = sys.argv[1]; GPU = sys.argv[2]
BASE = "/share/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/" + os.listdir("/share/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots")[0]
ADAPTER = "/share/hf_home/hub/models--declare-lab--delta-mem_qwen3_4b-instruct/snapshots/6a1a990e770974811b8fb27dab73430807f76cb5"
PQ = "/share/hf_home/hub/datasets--ai-hyz--MemoryAgentBench/snapshots/7ea066982b140a19337e17e60d45d4076e042faf/data/Conflict_Resolution-00000-of-00001.parquet"
OUT = f"/share/exp4group/results/groupB2_{SRC}.jsonl"

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

rows = pq.read_table(PQ).to_pylist()
row = [x for x in rows if x["metadata"]["source"] == SRC][0]
ctx, qs, ans = row["context"], row["questions"], row["answers"]

sys_msg = get_template(SRC, "system", "Long_context_agent_deltamem")
mem_tpl = get_template(SRC, "memorize", "Long_context_agent_deltamem")
qry_tpl = get_template(SRC, "query", "rag_agent")
def fmt(tpl, **kw):
    try: return tpl.format(time_stamp="", **kw)
    except KeyError: return tpl.format(**kw)

# ---------- 句子切分：与 diag2 一致，按 "N. " 行 ----------
def split_facts(text):
    out = []
    for line in text.split("\n"):
        m = re.match(r"\s*(\d+)\.\s*(.*)", line)
        if m:
            out.append((int(m.group(1)), m.group(2).strip()))
    return out

def subj_pred_key(fact):
    toks = fact.split()
    if len(toks) <= KEY_TAIL_CUT + 2:
        return fact
    return " ".join(toks[:-KEY_TAIL_CUT])

def embed_norm(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(EMB.device)
    return F.normalize(EMB[ids[0]].mean(dim=0), dim=0)

class SlotPool:
    def __init__(self):
        self.keys = []          # list[Tensor dim=hidden] 归一化后
        self.kv  = []           # list[tuple(K,V)] K/V: [layers][1,NKV,T,HD]
        self.serial = []        # fact serial
        self.t0 = []            # 全局时间戳，用于 FIFO
        self.clock = 0
        self.n_overwrites = 0
        self.n_evicts = 0

    def _whiten(self):
        if not self.keys:
            return []
        M = torch.stack(self.keys).float()
        mu = M.mean(dim=0, keepdim=True)
        return [F.normalize(v, dim=0) for v in (M - mu)]

    def add(self, key_raw, kv, serial):
        self.keys.append(key_raw); self.kv.append(kv)
        self.serial.append(serial); self.t0.append(self.clock); self.clock += 1
        while len(self.keys) > POOL_MAX:
            self._evict_oldest()

    def _evict_oldest(self):
        idx = min(range(len(self.t0)), key=lambda i: self.t0[i])
        for lst in (self.keys, self.kv, self.serial, self.t0):
            lst.pop(idx)
        self.n_evicts += 1

    def try_overwrite(self, key_raw, kv_new, serial_new):
        """返回 True 表示命中并已原位替换。"""
        if not self.keys:
            return False
        W = torch.stack(self.keys).float()
        Wn = [F.normalize(v, dim=0) for v in (W - W.mean(dim=0, keepdim=True))]
        kn = F.normalize(key_raw - torch.stack(self.keys).float().mean(dim=0), dim=0)
        sims = torch.stack([torch.dot(kn, w) for w in Wn])
        top = int(sims.argmax())
        if float(sims[top]) > THETA:
            self.kv[top] = kv_new          # 原位替换 KV
            self.keys[top] = key_raw
            self.serial[top] = serial_new
            self.t0[top] = self.clock; self.clock += 1
            self.n_overwrites += 1
            return True
        return False

    def build_past_key_values(self):
        """把所有槽位 KV 沿 seq 拼接为 DynamicCache 兼容结构。"""
        if not self.kv:
            return None
        Ks, Vs = [], []
        for layer in range(NL):
            k_cat = torch.cat([kv[layer][0] for kv in self.kv], dim=2)
            v_cat = torch.cat([kv[layer][1] for kv in self.kv], dim=2)
            Ks.append(k_cat); Vs.append(v_cat)
        # transformers>=4.36 使用 tuple of tuple 也可被 generate 接受
        from transformers import DynamicCache
        cache = DynamicCache()
        for l in range(NL):
            cache.update(Ks[l], Vs[l], l)
        return cache

# ---------- 从一次 forward 的 KV cache 中截取某句 token 区间 ----------
def slice_kv(past, t0, t1):
    out = []
    for layer in range(NL):
        kl = past.layers[layer]; k, v = kl.keys, kl.values   # transformers 5.x DynamicLayer
        out.append((k[:, :, t0:t1, :].clone(), v[:, :, t0:t1, :].clone()))
    return tuple(out)

# ---------- 定位句子在 chunk token 序列中的 span ----------
def locate_spans(chunk_text, facts, ids_len):
    """用字符比例近似映射 token 位置（事实行有序且等密度，足够用于 KV 截取）。"""
    total_chars = len(chunk_text)
    spans = []
    for serial, fact in facts:
        pos = chunk_text.find(fact[:40])
        if pos < 0:
            pos = 0
        t0 = min(max(ids_len - 2, 0), int(ids_len * pos / max(total_chars, 1)))
        t1 = min(ids_len, t0 + max(8, int(ids_len * (len(fact) + 2) / max(total_chars, 1))))
        if t1 <= t0:
            t0 = max(0, ids_len - 24); t1 = ids_len
        spans.append((t0, t1))
    return spans

reset_delta_mem_states(model)
pool = SlotPool()
seen_serials = {}

print(f"streaming... THETA={THETA} POOL={POOL_MAX}", flush=True)
with torch.no_grad():
    for ci, ch in enumerate(chunks := chunk_text_into_sentences(ctx, chunk_size=CHUNK_TOKENS)):
        facts = split_facts(ch)
        txt = tok.apply_chat_template(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": fmt(mem_tpl, context=ch)}],
            add_generation_prompt=True, tokenize=False)
        ids = tok(txt, return_tensors="pt").input_ids.to(f"cuda:{GPU}")
        past = None
        if KEEP_DELTA:
            out = model(input_ids=ids, use_cache=True, past_key_values=past)
            past = out.past_key_values
        else:
            # delta 写入也要走 forward；use_cache=True 拿 KV
            out = model(input_ids=ids, use_cache=True)
            past = out.past_key_values
        # 注意：past 里包含 system+模板 token，句子 span 从尾部往前映射
        ctx_len = ids.shape[1]
        spans = locate_spans(ch, facts, ctx_len)
        for (serial, fact), (t0, t1) in zip(facts, spans):
            if t1 - t0 < 4:
                continue
            key_raw = embed_norm(subj_pred_key(fact))
            kv_new = slice_kv(past, t0, t1)
            hit = pool.try_overwrite(key_raw, kv_new, serial)
            if not hit:
                pool.add(key_raw, kv_new, serial)
        if (ci + 1) % 5 == 0 or ci == len(chunks) - 1:
            print(f"chunk {ci+1}/{len(chunks)} pool={len(pool.keys)} "
                  f"ow={pool.n_overwrites} ev={pool.n_evicts}", flush=True)
set_delta_mem_write_enabled(model, False)
print(f"write done. pool_final={len(pool.keys)} overwrites={pool.n_overwrites}", flush=True)

# ---------- 读出 ----------
results = []
with torch.no_grad():
    for qi, (q, a) in enumerate(zip(qs, ans)):
        try: qtxt = qry_tpl.format(question=q)
        except KeyError: qtxt = qry_tpl.format(query=q)
        text = tok.apply_chat_template(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": qtxt}],
            add_generation_prompt=True, tokenize=False)
        qids = tok(text, return_tensors="pt").input_ids.to(f"cuda:{GPU}")
        pkv = pool.build_past_key_values()
        # generate 与 delta 层在 cache 长度 != ids 长度时不兼容（scan 空列表崩溃），
        # 改为手写 greedy decode：显式传 cache_position 从池子长度起算。
        SL = pkv.get_seq_length() if pkv is not None else 0
        cur = qids
        gen_ids = []
        for step in range(10):
            cp = torch.arange(SL, SL + cur.shape[1], device=cur.device)
            logits = model(input_ids=cur, past_key_values=pkv,
                           cache_position=cp, use_cache=True).logits[:, -1, :]
            nxt = int(logits.argmax(dim=-1))
            gen_ids.append(nxt)
            if nxt == (tok.eos_token_id or -1):
                break
            cur = torch.tensor([[nxt]], device=cur.device)
            SL += 1
        pred = tok.decode(gen_ids, skip_special_tokens=True)
        m, _ = post_process(
            {"output": pred, "input_len": qids.shape[1],
             "output_len": len(gen_ids),
             "memory_construction_time": 0, "query_time_len": 0},
            a, {"sub_dataset": SRC, "debug": False})
        rec = {"question_idx": qi, "query": q, "answer": a, "output": pred, **m}
        results.append(rec)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (qi + 1) % 20 == 0:
            sf = sum(r["substring_exact_match"] for r in results) / len(results)
            print(f"{qi+1}/{len(qs)} running SF={sf:.3f}", flush=True)
sf = sum(r["substring_exact_match"] for r in results) / len(results)
print(f"FINAL groupB2 {SRC}: SF={sf:.4f} n={len(results)} "
      f"pool={len(pool.keys)} overwrites={pool.n_overwrites}", flush=True)
