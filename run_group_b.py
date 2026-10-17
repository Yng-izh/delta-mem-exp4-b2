import sys, os, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/share/delta-Mem")
sys.path.insert(0, "/share/MemoryAgentBench")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltamem.core import HFDeltaMemConfig, attach_delta_mem
from deltamem.core.delta_impl import DeltaMemAttention, load_delta_mem_adapter
from deltamem.core.delta import reset_delta_mem_states, set_delta_mem_write_enabled
from deltamem.eval.official_memory_agent_bench_templates import get_template
from utils.eval_other_utils import chunk_text_into_sentences, post_process

# ============ B 组：启发式原位覆写 ============
# 机制：写入第 i 个 chunk 前，用冻结 embedding（embed_tokens 均值）与之前所有 chunk
# 算余弦相似度；若 max_j<i cos(i,j) > THRESH，则该 chunk 写入时放大 beta
# （erase_seq=write_seq=beta↑，keep_seq=1-beta↓）→ 新事实更强地覆写 delta 态中旧值。

THRESH = float(os.environ.get("B_SIM_THRESH", "0.80"))
BETA_GAIN = float(os.environ.get("B_BETA_GAIN", "2.0"))

SRC = sys.argv[1]
GPU = sys.argv[2]
BASE = "/share/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/" + os.listdir("/share/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots")[0]
ADAPTER = "/share/hf_home/hub/models--declare-lab--delta-mem_qwen3_4b-instruct/snapshots/6a1a990e770974811b8fb27dab73430807f76cb5"
PQ = "/share/hf_home/hub/datasets--ai-hyz--MemoryAgentBench/snapshots/7ea066982b140a19337e17e60d45d4076e042faf/data/Conflict_Resolution-00000-of-00001.parquet"
OUT = f"/share/exp4group/results/groupB_{SRC}.jsonl"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
        device_map=f"cuda:{GPU}", attn_implementation="sdpa")
cfg = HFDeltaMemConfig.from_pretrained(ADAPTER)
attach_delta_mem(model, cfg)
load_delta_mem_adapter(model, input_dir=ADAPTER)
model.eval()

# ---- monkeypatch：per-forward beta 增益 ----
_orig_proj = DeltaMemAttention._memory_sequence_projections

def _patched_proj(self, hidden_states):
    q, k, v, beta, lam = _orig_proj(self, hidden_states)
    gain = getattr(self, "_overwrite_gain", None)
    if gain is not None and gain != 1.0:
        beta = (beta * gain).clamp(max=1.0)
        if self.couple_lambda:
            lam = 1.0 - beta
    return q, k, v, beta, lam

DeltaMemAttention._memory_sequence_projections = _patched_proj

def set_gain(model, gain):
    for m in model.modules():
        if isinstance(m, DeltaMemAttention):
            m._overwrite_gain = gain

import pyarrow.parquet as pq
rows = pq.read_table(PQ).to_pylist()
by_src = {x["metadata"]["source"]: x for x in rows}
row = by_src[SRC]
ctx, qs, ans = row["context"], row["questions"], row["answers"]

sys_msg = get_template(SRC, "system", "Long_context_agent_deltamem")
mem_tpl = get_template(SRC, "memorize", "Long_context_agent_deltamem")
qry_tpl = get_template(SRC, "query", "rag_agent")

def fmt(tpl, **kw):
    try:
        return tpl.format(time_stamp="", **kw)
    except KeyError:
        return tpl.format(**kw)

chunks = chunk_text_into_sentences(ctx, chunk_size=4096)
print(f"chunks: {len(chunks)} THRESH={THRESH} GAIN={BETA_GAIN}", flush=True)

# 冻结 embedding：embed_tokens 均值
emb = model.get_input_embeddings().weight.detach()

def chunk_emb(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(emb.device)
    e = emb[ids[0]].mean(dim=0)
    return F.normalize(e, dim=0)

chunk_vecs = [chunk_emb(c) for c in chunks]

# 相似度矩阵 → 每 chunk 覆写增益
gains = []
for i in range(len(chunks)):
    if i == 0:
        gains.append(1.0)
        continue
    sims = F.cosine_similarity(chunk_vecs[i].unsqueeze(0), torch.stack(chunk_vecs[:i])).max().item()
    gains.append(BETA_GAIN if sims > THRESH else 1.0)
n_ovw = sum(1 for g in gains if g != 1.0)
print(f"overwrite chunks: {n_ovw}/{len(chunks)}", flush=True)

reset_delta_mem_states(model)
set_delta_mem_write_enabled(model, True)
with torch.no_grad():
    for i, ch in enumerate(chunks):
        set_gain(model, gains[i])
        txt = tok.apply_chat_template(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": fmt(mem_tpl, context=ch)}],
            add_generation_prompt=True, tokenize=False)
        ids = tok(txt, return_tensors="pt").input_ids.to(f"cuda:{GPU}")
        _ = model(input_ids=ids, use_cache=False)
        if (i + 1) % 5 == 0 or i == len(chunks) - 1:
            print(f"chunk {i+1}/{len(chunks)} written (gain={gains[i]:.1f})", flush=True)
set_gain(model, 1.0)
set_delta_mem_write_enabled(model, False)
print("write phase done", flush=True)

results = []
with torch.no_grad():
    for qi, (q, a) in enumerate(zip(qs, ans)):
        try:
            qtxt = qry_tpl.format(question=q)
        except KeyError:
            qtxt = qry_tpl.format(query=q)
        text = tok.apply_chat_template(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": qtxt}],
            add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt").input_ids.to(f"cuda:{GPU}")
        out = model.generate(input_ids=ids, max_new_tokens=10, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        pred = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        m, _ = post_process(
            {"output": pred, "input_len": ids.shape[1],
             "output_len": int(out.shape[1] - ids.shape[1]),
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
print(f"FINAL groupB {SRC}: SF={sf:.4f} n={len(results)}", flush=True)
