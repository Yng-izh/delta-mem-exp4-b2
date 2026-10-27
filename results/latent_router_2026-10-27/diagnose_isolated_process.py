# 注意：本脚本硬编码了服务器路径（/share/exp4group、/share/hf_home），需在 v100-2 上运行。
"""独立进程单跑 C 段：1fact ckpt 在 min16 / min16-4slot 上的三模式。
目的：主表里 C 段全 0 而 B 段 15/16，二者输入字面相同（target_only == 原样 1 条），
疑似连续评估的状态污染。独立进程 + 每条 reset 复核。
"""
import sys, re, torch, random
sys.path.insert(0, "/share/exp4group")
import latent_memory as LM, train_latent_memory as T
import deltamem_compat as DMC
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltamem.core.delta_impl import set_delta_mem_write_enabled, reset_delta_mem_states
_src = open("/share/exp4group/train_latent_memory.py").read()
SYS = re.search(r'"--system-prompt",\s*\n\s*default="([^"]+)"', _src).group(1)
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
set_delta_mem_write_enabled(model, False)
hidden = LM._model_hidden_size(model)
blob = LM._safe_torch_load("/share/exp4group/ckpt/latent_min16_v2.pt"); c = blob["config"]
comp = LM.LatentCompressor(hidden, n_slots=c["n_slots"], inner_dim=c["inner_dim"],
                           num_heads=c["num_heads"]).to("cuda:0")
bridge = LM.LatentBridge(hidden, bottleneck=c["bottleneck"],
                         target_norm=float(c.get("target_norm", 1.0))).to("cuda:0")
comp.load_state_dict(blob["compressor"]); bridge.load_state_dict(blob["bridge"])
comp.eval(); bridge.eval()
enc = LM.LatentEncoder(model, tok, comp, layer=c["layer"], max_unit_tokens=c["max_unit_tokens"])
min16 = T.load_synth_jsonl("/share/exp4group/synth/latent_min16.jsonl")
rng = random.Random(0)
four = []
for i, it in enumerate(min16):
    # load_synth_jsonl 的 memory 是字符串，事实列表在 units 里
    others = [x["units"][0] for j, x in enumerate(min16) if j != i]
    four.append({"units": [it["units"][0]] + rng.sample(others, 3),
                 "question": it["question"], "answer": it["answer"]})
def go(mode, items, tag):
    n = ok = 0
    errs = []
    for it in items:
        u = [str(x) for x in (it.get("units") or [it.get("memory") or ""])]
        tgt = u[0]
        if mode == "full": uu = u
        elif mode == "target_only": uu = [tgt]
        elif mode == "target_end": uu = u[1:] + [tgt]
        reset_delta_mem_states(model)
        z, _ = T.encode_item(enc, {"units": uu, "question": it["question"]}, tag="c")
        pred, _ = LM.generate_latent_answer(model=model, tokenizer=tok, bridge=bridge,
                                            payloads=z, question=it["question"],
                                            system_prompt=SYS, max_new_tokens=16)
        g = it["answer"]; g = " ".join(g) if isinstance(g, list) else g
        if norm(pred) == norm(g): ok += 1
        elif len(errs) < 3: errs.append((uu[:1], g, pred[:60]))
        n += 1
    print(f"[1fact-mode] {tag:11s} strict={ok}/{n} ({ok/n:5.1%})", flush=True)
    for e in errs: print("    miss:", e, flush=True)
go("target_only", min16, "min16-only")      # == 原样单条，应 15/16
for m in ("target_only", "full", "target_end"):
    go(m, four, "4slot-" + m)
print("[done]", flush=True)
