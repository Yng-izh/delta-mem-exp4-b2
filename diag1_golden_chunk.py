import sys, os, json, re
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/share/delta-Mem")
sys.path.insert(0, "/share/MemoryAgentBench")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltamem.core import HFDeltaMemConfig, attach_delta_mem
from deltamem.core.delta_impl import load_delta_mem_adapter
from deltamem.core.delta import reset_delta_mem_states, set_delta_mem_write_enabled
from deltamem.eval.official_memory_agent_bench_templates import get_template
from utils.eval_other_utils import chunk_text_into_sentences, post_process
import pyarrow.parquet as pq

SRC = sys.argv[1]; GPU = sys.argv[2]
BASE = "/share/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/" + os.listdir("/share/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots")[0]
ADAPTER = "/share/hf_home/hub/models--declare-lab--delta-mem_qwen3_4b-instruct/snapshots/6a1a990e770974811b8fb27dab73430807f76cb5"
PQ = "/share/hf_home/hub/datasets--ai-hyz--MemoryAgentBench/snapshots/7ea066982b140a19337e17e60d45d4076e042faf/data/Conflict_Resolution-00000-of-00001.parquet"
OUT = "/share/exp4group/results/diag1_golden_chunk.jsonl"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
        device_map=f"cuda:{GPU}", attn_implementation="sdpa")
cfg = HFDeltaMemConfig.from_pretrained(ADAPTER)
attach_delta_mem(model, cfg)
load_delta_mem_adapter(model, input_dir=ADAPTER)
model.eval()

rows = pq.read_table(PQ).to_pylist()
row = [x for x in rows if x["metadata"]["source"] == SRC][0]
ctx, qs, ans = row["context"], row["questions"], row["answers"]

sys_msg = get_template(SRC, "system", "Long_context_agent_deltamem")
mem_tpl = get_template(SRC, "memorize", "Long_context_agent_deltamem")
qry_tpl = get_template(SRC, "query", "rag_agent")
def fmt(tpl, **kw):
    try: return tpl.format(time_stamp="", **kw)
    except KeyError: return tpl.format(**kw)

# 每 chunk 的原文文本（用于定位 golden chunk）
chunks = chunk_text_into_sentences(ctx, chunk_size=4096)
print(f"chunks: {len(chunks)}", flush=True)

def gold_chunk(q, a):
    """定位包含正确答案的 chunk；找不到则返回含 question 关键词最多的 chunk。"""
    for ai in (a if isinstance(a, list) else [a]):
        for ci, ch in enumerate(chunks):
            if str(ai).strip() in ch:
                return ci
    # fallback: question 里的专有名词匹配
    qwords = [w for w in re.findall(r"[A-Z][a-zA-Z]+", q)]
    best, bestc = 0, 0
    for ci, ch in enumerate(chunks):
        s = sum(ch.count(w) for w in qwords)
        if s > best: best, bestc = s, ci
    return bestc

results = []
with torch.no_grad():
    for qi, (q, a) in enumerate(zip(qs, ans)):
        ci = gold_chunk(q, a)
        reset_delta_mem_states(model)
        set_delta_mem_write_enabled(model, True)
        txt = tok.apply_chat_template(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": fmt(mem_tpl, context=chunks[ci])}],
            add_generation_prompt=True, tokenize=False)
        ids = tok(txt, return_tensors="pt").input_ids.to(f"cuda:{GPU}")
        _ = model(input_ids=ids, use_cache=False)
        set_delta_mem_write_enabled(model, False)
        try: qtxt = qry_tpl.format(question=q)
        except KeyError: qtxt = qry_tpl.format(query=q)
        text = tok.apply_chat_template(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": qtxt}],
            add_generation_prompt=True, tokenize=False)
        qids = tok(text, return_tensors="pt").input_ids.to(f"cuda:{GPU}")
        out = model.generate(input_ids=qids, max_new_tokens=10, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        pred = tok.decode(out[0][qids.shape[1]:], skip_special_tokens=True)
        m, _ = post_process(
            {"output": pred, "input_len": qids.shape[1],
             "output_len": int(out.shape[1]-qids.shape[1]),
             "memory_construction_time": 0, "query_time_len": 0},
            a, {"sub_dataset": SRC, "debug": False})
        rec = {"question_idx": qi, "chunk": ci, "query": q, "answer": a,
               "output": pred, **m}
        results.append(rec)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (qi+1) % 25 == 0:
            sf = sum(r["substring_exact_match"] for r in results)/len(results)
            print(f"{qi+1}/{len(qs)} running SF={sf:.3f}", flush=True)
sf = sum(r["substring_exact_match"] for r in results)/len(results)
print(f"FINAL diag1 {SRC} golden-chunk-only: SF={sf:.4f} n={len(results)}", flush=True)
