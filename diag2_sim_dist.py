import sys, os, re, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/share/MemoryAgentBench")
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from utils.eval_other_utils import chunk_text_into_sentences
import pyarrow.parquet as pq

SRC = sys.argv[1]
GPU = sys.argv[2]
BASE = "/share/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/" + os.listdir("/share/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots")[0]
PQ = "/share/hf_home/hub/datasets--ai-hyz--MemoryAgentBench/snapshots/7ea066982b140a19337e17e60d45d4076e042faf/data/Conflict_Resolution-00000-of-00001.parquet"

tok = AutoTokenizer.from_pretrained(BASE)
emb = tok.get_input_embeddings() if hasattr(tok, "get_input_embeddings") else None

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
        device_map=f"cuda:{GPU}", attn_implementation="sdpa")
W = model.get_input_embeddings().weight.detach()

rows = pq.read_table(PQ).to_pylist()
row = [x for x in rows if x["metadata"]["source"] == SRC][0]
ctx, qs, ans = row["context"], row["questions"], row["answers"]

# 1) 单句切分：按 "N. " 序号行拆分事实句子
def split_facts(text):
    facts = []
    for line in text.split("\n"):
        m = re.match(r"\s*(\d+)\.\s*(.*)", line)
        if m:
            serial, fact = int(m.group(1)), m.group(2).strip()
            facts.append((serial, fact))
    return facts

facts = split_facts(ctx)
print(f"total facts: {len(facts)}", flush=True)

# 2) Key 提取：去序列号 + 去句尾 3-5 个"宾语/新值" token，只取主语+谓语
def subj_pred_key(fact, n_tail=5):
    toks = fact.split()
    if len(toks) <= n_tail + 2:
        return fact
    return " ".join(toks[:-n_tail])

def embed(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(W.device)
    return F.normalize(W[ids[0]].mean(dim=0), dim=0)

serials = [s for s, _ in facts]
keys_raw = [embed(f) for _, f in facts]
keys_subj = [embed(subj_pred_key(f)) for _, f in facts]

def whiten(vecs):
    M = torch.stack(vecs).float()
    mu = M.mean(dim=0, keepdim=True)
    M = M - mu
    return [F.normalize(v, dim=0) for v in M]

keys_raw_w = whiten(keys_raw)
keys_subj_w = whiten(keys_subj)

def sim_dist(vecs, name):
    M = torch.stack(vecs).float()
    S = (M @ M.T).float()
    off = S[~torch.eye(len(vecs), dtype=torch.bool, device=S.device)]
    print(f"[{name}] pairwise cos: mean={off.mean():.3f} p50={off.median():.3f} "
          f"p90={off.quantile(0.9):.3f} p99={off.quantile(0.99):.3f} max={off.max():.3f}", flush=True)
    return S

S_raw = sim_dist(keys_raw, "raw-fact (with object)")
S_subj = sim_dist(keys_subj, "subj-pred (no object)")
S_raww = sim_dist(keys_raw_w, "raw whitened")
S_subjw = sim_dist(keys_subj, "subj-pred (no whitening)")  # placeholder, replaced below
S_subjw = sim_dist(keys_subj_w, "subj-pred whitened")

# 3) 用 gold 答案定位冲突组：对每题，找 gold 答案出现的句子，
#    再看同 subject 句子（去除宾语后 key 相似度最高）的相似度是否落在高区间
gold_in_high = 0; gold_total = 0
conflict_sims = []; random_sims = []
for q, a in zip(qs, ans):
    answers = a if isinstance(a, list) else [a]
    gold_sers = []
    for ai in answers:
        for (s, f) in facts:
            if str(ai).strip() in f:
                gold_sers.append(s)
    if not gold_sers:
        continue
    gold_total += 1
    for s in gold_sers:
        i = serials.index(s)
        ki = keys_subj_w[i]
        M = torch.stack(keys_subj_w).float(); sims = (M @ ki)/(1e-6 + M.norm(dim=1))
        sims[i] = -1
        top = sims.argsort(descending=True)[:5]
        conflict_sims += [float(v) for v in sims[top].flatten().tolist()]
        # 随机 100 对做对照
        rand_idx = torch.randint(0, len(facts), (100,))
        random_sims += [float(sims[j]) for j in rand_idx.tolist() if int(j) != i]

print(f"\nconflict-candidate sims (top5 per gold fact): "
      f"mean={sum(conflict_sims)/len(conflict_sims):.3f} "
      f"p50={sorted(conflict_sims)[len(conflict_sims)//2]:.3f} "
      f"p90={sorted(conflict_sims)[int(len(conflict_sims)*0.9)]:.3f}", flush=True)
print(f"random sims: mean={sum(random_sims)/len(random_sims):.3f}", flush=True)

# 分桶
buckets = {}
for s_ in conflict_sims:
    b = round(s_, 1)
    buckets[b] = buckets.get(b, 0) + 1
print("conflict-sim histogram (1-decimal):", dict(sorted(buckets.items())), flush=True)

# 保存冲突对 top 样例供人工检查
json.dump({"conflict_sims_p50": sorted(conflict_sims)[len(conflict_sims)//2],
           "random_sims_mean": sum(random_sims)/len(random_sims)},
          open("/share/exp4group/results/diag2_sim_summary.json", "w"))
print("DONE diag2", flush=True)
