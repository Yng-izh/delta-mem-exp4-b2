# 注意：本脚本硬编码了服务器路径（/share/exp4group、/share/hf_home），需在 v100-2 上运行。
"""Router 改动验证：推理路径 / checkpoint 往返 / 老 ckpt 兼容。"""
import sys, torch
sys.path.insert(0, "/share/exp4group")
import latent_memory as LM, train_latent_memory as T
import deltamem_compat as DMC
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltamem.core.delta_impl import set_delta_mem_write_enabled

SYS = "You are a helpful assistant. Answer with the shortest exact value."
BASE = "/share/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
ADP = "/share/hf_home/hub/models--declare-lab--delta-mem_qwen3_4b-instruct/snapshots/6a1a990e770974811b8fb27dab73430807f76cb5"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0",
                                             attn_implementation="sdpa")
cfg, attach, load_adp = DMC.import_delta_mem()
attach(model, cfg.from_pretrained(ADP)); load_adp(model, ADP); model.eval()
for p in model.parameters(): p.requires_grad_(False)
set_delta_mem_write_enabled(model, False)
hidden = LM._model_hidden_size(model)

comp = LM.LatentCompressor(hidden, n_slots=4, inner_dim=512, num_heads=8).to("cuda:0").eval()
bridge = LM.LatentBridge(hidden, bottleneck=512, target_norm=1.0873).to("cuda:0").eval()
router = LM.LatentRouter(hidden, n_slots=4, d_key=256).to("cuda:0").eval()
enc = LM.LatentEncoder(model, tok, comp, layer=18, max_unit_tokens=64)

val = T.load_synth_jsonl("/share/exp4group/synth/latent_exp2_val.jsonl")
it = val[0]
payloads, z, info = T.encode_item_slots(enc, it, tag="v")
print(f"[encode] slots={len(payloads)} z={tuple(z.shape)} units={len(it['units'])}", flush=True)
slot_z = LM.stack_payloads(payloads)
print(f"[stack] slot_z={tuple(slot_z.shape)}", flush=True)

qf = LM.extract_query_feature(model, tok, it["question"], SYS, layer=18)
print(f"[qfeat] {tuple(qf.shape)}", flush=True)
z_read, logits, ki, ko = router(qf, slot_z.unsqueeze(0), hard=False)
print(f"[router soft] z_read={tuple(z_read.shape)} logits={tuple(logits.shape)} "
      f"k_in={tuple(ki.shape)} k_out={tuple(ko.shape)}", flush=True)
z_hard, logits_h, _, _ = router(qf, slot_z.unsqueeze(0), hard=True)
print(f"[router hard] z_read={tuple(z_hard.shape)} picked={int(logits_h[0].argmax())}",
      flush=True)
chain = router.chain(qf, slot_z.unsqueeze(0), hops=2)
print(f"[router chain hops=2] picked={chain}", flush=True)

pred1, m1 = LM.generate_latent_answer(model=model, tokenizer=tok, bridge=bridge,
                                      payloads=z, question=it["question"],
                                      system_prompt=SYS, max_new_tokens=8,
                                      router=router, layer=18)
print(f"[gen router hops=1] routed={m1['routed']} n_soft={m1['n_soft']} "
      f"slots_in={m1['n_slots_in']} pred={pred1[:40]!r}", flush=True)
pred2, m2 = LM.generate_latent_answer(model=model, tokenizer=tok, bridge=bridge,
                                      payloads=z, question=it["question"],
                                      system_prompt=SYS, max_new_tokens=8,
                                      router=router, layer=18, hops=2)
print(f"[gen router hops=2] routed={m2['routed']} n_soft={m2['n_soft']} "
      f"pred={pred2[:40]!r}", flush=True)
pred3, m3 = LM.generate_latent_answer(model=model, tokenizer=tok, bridge=bridge,
                                      payloads=z, question=it["question"],
                                      system_prompt=SYS, max_new_tokens=8)
print(f"[gen no-router 回归] routed={m3['routed']} n_soft={m3['n_soft']} "
      f"(应为 16) pred={pred3[:40]!r}", flush=True)

# checkpoint 往返：has_router=True
LM.save_latent_checkpoint("/tmp/ck_router.pt", comp, bridge, {"layer": 18}, {}, router=router)
rd = LM.load_latent_reader("/tmp/ck_router.pt", model, tok)
print(f"[ckpt router] router_restored={rd.router is not None} "
      f"d_key={getattr(rd.router, 'd_key', None)}", flush=True)
# 老 ckpt 兼容（无 router）
LM.save_latent_checkpoint("/tmp/ck_norouter.pt", comp, bridge, {"layer": 18}, {})
rd2 = LM.load_latent_reader("/tmp/ck_norouter.pt", model, tok)
print(f"[ckpt legacy] router_restored={rd2.router is not None} (应为 False)", flush=True)
print("[done]", flush=True)
