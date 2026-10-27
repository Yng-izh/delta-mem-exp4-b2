# 注意：本脚本硬编码了服务器路径（/share/exp4group、/share/hf_home），需在 v100-2 上运行。
"""第二次干净重跑：修两个脚本级错误。

错误1（prompt）：此前手写 SYS，训练默认是 "You are a helpful assistant..."。已改为从源码提取。
错误2（delta 污染）：诊断脚本从未关闭 Delta 写入，而 LatentEncoder/训练循环里的
  delta_write_disabled 的 finally 把写入**永久置 False**（见 latent_memory.py:174）。
  => 训练进程内所有 forward 都不写 delta_state（干净）；诊断脚本写入默认开启，
     每条样本都往 delta_state 累积，于是"先跑的指标正常、后跑的全 0"。
  本脚本在加载后一次性 set_delta_mem_write_enabled(False)，与训练进程行为对齐。
错误3（oracle 解码）：generate(inputs_embeds=) 返回的序列不含输入前缀，
  原先用 out[0, e.shape[1]:] 切片得到空串 -> oracle 假 0。改为 decode(out[0])。
"""
import sys, re, torch, random
sys.path.insert(0, "/share/exp4group")
import latent_memory as LM, train_latent_memory as T
import deltamem_compat as DMC
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltamem.core.delta_impl import (set_delta_mem_write_enabled,
                                     reset_delta_mem_states)

_src = open("/share/exp4group/train_latent_memory.py").read()
SYS = re.search(r'"--system-prompt",\s*\n\s*default="([^"]+)"', _src).group(1)
print(f"[SYS] {SYS!r}", flush=True)

BASE = "/share/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
ADP = "/share/hf_home/hub/models--declare-lab--delta-mem_qwen3_4b-instruct/snapshots/6a1a990e770974811b8fb27dab73430807f76cb5"

def norm(t):
    t = str(t).strip().lower()
    for ch in ".,!?;:\"'()[]{}":
        t = t.replace(ch, " ")
    return " ".join(t.split())

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0",
                                             attn_implementation="sdpa")
cfg, attach, load_adp = DMC.import_delta_mem()
attach(model, cfg.from_pretrained(ADP)); load_adp(model, ADP); model.eval()
for p in model.parameters(): p.requires_grad_(False)
set_delta_mem_write_enabled(model, False)   # 与训练进程一致：写入永久关闭
print("[delta] write_enabled=False（对齐训练进程）", flush=True)
hidden = LM._model_hidden_size(model)

def load_reader(path):
    blob = LM._safe_torch_load(path); c = blob["config"]
    comp = LM.LatentCompressor(hidden, n_slots=c["n_slots"], inner_dim=c["inner_dim"],
                               num_heads=c["num_heads"]).to("cuda:0")
    bridge = LM.LatentBridge(hidden, bottleneck=c["bottleneck"],
                             target_norm=float(c.get("target_norm", 1.0))).to("cuda:0")
    comp.load_state_dict(blob["compressor"]); bridge.load_state_dict(blob["bridge"])
    comp.eval(); bridge.eval()
    enc = LM.LatentEncoder(model, tok, comp, layer=c["layer"],
                           max_unit_tokens=c["max_unit_tokens"])
    return comp, bridge, enc, c

train = T.load_synth_jsonl("/share/exp4group/synth/latent_exp2_train.jsonl")
val = T.load_synth_jsonl("/share/exp4group/synth/latent_exp2_val.jsonl")
min16 = T.load_synth_jsonl("/share/exp4group/synth/latent_min16.jsonl")
rng = random.Random(0)
min16_4slot = []
for i, it in enumerate(min16):
    others = [x["memory"][0] for j, x in enumerate(min16) if j != i]
    min16_4slot.append({"units": [it["memory"][0]] + rng.sample(others, 3),
                        "question": it["question"], "answer": it["answer"],
                        "evidence": it.get("evidence") or [it["memory"][0]]})

def reorder(it, mode):
    units = list(it.get("units") or [it.get("memory") or ""])
    tgt = it["evidence"][0] if isinstance(it.get("evidence"), list) and it["evidence"] else units[0]
    if mode == "full":
        return units
    if mode == "target_only":
        return [tgt]
    if mode == "target_end":
        return [u for u in units if u != tgt] + [tgt]
    raise ValueError(mode)

def run(ckpt, tag, mode, items, split, limit):
    comp, bridge, enc, c = load_reader(ckpt)
    n = strict = ent = 0; groups = {}
    for it in items[:limit]:
        reset_delta_mem_states(model)
        z, _ = T.encode_item(enc, {"units": reorder(it, mode), "question": it["question"]}, tag="diag")
        pred, _ = LM.generate_latent_answer(model=model, tokenizer=tok, bridge=bridge,
                                            payloads=z, question=it["question"],
                                            system_prompt=SYS, max_new_tokens=16)
        g = it["answer"]
        if isinstance(g, list): g = " ".join(str(x) for x in g)
        s = norm(pred) == norm(g); e = norm(g) in norm(pred)
        strict += s; ent += e; n += 1
        groups.setdefault(norm(it["question"]), []).append(s)
    pair = sum(1 for v in groups.values() if all(v))
    print(f"[{tag:6s}] {split:5s} {mode:12s} strict={strict:3d}/{n:3d} ({strict/n:5.1%}) "
          f"ent={ent:3d}/{n:3d} pair={pair}/{len(groups)}", flush=True)
    del comp, bridge, enc; torch.cuda.empty_cache()

C4 = "/share/exp4group/ckpt/latent_exp2_v2.pt"
C1 = "/share/exp4group/ckpt/latent_min16_v2.pt"

# 先做 sanity：1fact 在 min16 上必须回到 15/16
run(C1, "1fact", "full", min16, "min16", 16)
print("=== A. 4fact ckpt ===", flush=True)
for mode in ("full", "target_only", "target_end"):
    run(C4, "4fact", mode, train, "train", 200)
    run(C4, "4fact", mode, val, "val", 64)
print("=== C. 1fact ckpt 在同域 4 槽拼装上 ===", flush=True)
for mode in ("full", "target_only", "target_end"):
    run(C1, "1fact", mode, min16_4slot, "4slot", 16)
print("=== D. 上界对照（val 前 32 条）===", flush=True)
pt = 0
for it in val[:32]:
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": "Memory: " + " ".join(it["units"]) + "\nQuestion: " + it["question"]}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
    pred = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    pt += int(norm(pred) == norm(it["answer"]))
print(f"[plaintext] val前32 strict={pt}/32", flush=True)
orc = 0
el = model.get_input_embeddings()
for it in val[:32]:
    ids = tok(" ".join(it["units"]), return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
    with torch.no_grad():
        soft = el(ids)
    full, n_sys = LM.build_chat_parts(tok, SYS, it["question"])
    ids_full = tok(full, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
    ns = int(tok(full[:n_sys], return_tensors="pt", add_special_tokens=False).input_ids.shape[1])
    e = torch.cat([el(ids_full[:, :ns]), soft, el(ids_full[:, ns:])], dim=1)
    with torch.no_grad():
        out = model.generate(inputs_embeds=e,
                             attention_mask=torch.ones(e.shape[:2], dtype=torch.long, device="cuda:0"),
                             max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
    pred = tok.decode(out[0], skip_special_tokens=True)   # inputs_embeds: 输出不含前缀
    orc += int(norm(pred) == norm(it["answer"]))
print(f"[oracle]    val前32 strict={orc}/32", flush=True)
print("[done]", flush=True)
