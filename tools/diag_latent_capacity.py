#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/diag_latent_capacity.py — 潜层读出的**瓶颈定位**工具（真机 Qwen3-4B + Delta-Mem）

背景
----
`train_latent_memory.py --stage a` 在 15,000 条合成语料上跑完 1875 步（loss_qa 12.6→3.87），
但验证 exact-match = **0/32**。这个脚本用来把"到底哪一环坏了"钉死，而不是靠猜。

四个子命令，对应四条互斥的判据（全部在同一批验证样本上跑）：

    text    明文记忆进 prompt            → 数据/模型/任务是否成立
    oracle  跳过压缩器，把记忆原文的 token embedding 直接当软 token 注入
                                         → 软 token 的注入通路（位置/mask/拼接）是否成立
    scale   软 token 的范数 vs 真实 embedding 范数
            以及"把软 token 置零/换随机，loss 是否变化" → 模型是否根本没在看软 token
    probe   z 里还有多少答案信息（线性探针 + 平凡基线）→ 压缩器是否把信息丢了

实测结论（2026-10-21/22，v100-2，见 results/latent_stage_a_2026-10-21/）
------------------------------------------------------------------
    text    6/6      数据/模型/任务成立
    oracle  4/4      注入通路成立
    scale   软 token 范数 96~159，真实 embedding 1.05（差两个量级）→ 已修
    probe   压缩器重建记忆原文的 token-acc 22.8%，而"只吐前 10 高频词"的平凡基线 19.8%
            → 压缩器几乎没编码进任何内容；这就是 0/32 的原因

用法
----
    export PYTHONPATH=/share/exp4group/shim19:/share/exp4group:/share/delta-Mem
    python tools/diag_latent_capacity.py --sub text   --ckpt ckpt/latent_a.pt
    python tools/diag_latent_capacity.py --sub oracle
    python tools/diag_latent_capacity.py --sub scale  --ckpt ckpt/overfit200_recon.pt
    python tools/diag_latent_capacity.py --sub probe  --ckpt ckpt/overfit200_recon.pt

注意
----
- `--sub text/oracle` 不需要 checkpoint（只验证通路本身）。
- 所有子命令都需要真模型；不要在 --smoke-model 下用（随机小模型没有意义）。
- 探针必须给出**平凡基线**（最高频 token / 随机输入）才可解读：200 个样本配 4 万维输入，
  线性层无论如何都能硬背下来，没有基线的话任何数字都不能当信息量证据。
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import latent_memory as LM                      # noqa: E402
import train_latent_memory as T                 # noqa: E402

BASE = os.environ.get(
    "LATENT_BASE_MODEL",
    "/share/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/"
    "snapshots/cdbee75f17c01a7cc42f958dc650907174af0554")
ADAPTER = os.environ.get(
    "LATENT_ADAPTER",
    "/share/hf_home/hub/models--declare-lab--delta-mem_qwen3_4b-instruct/"
    "snapshots/6a1a990e770974811b8fb27dab73430807f76cb5")
SYS_PROMPT = "You are a memory agent. Answer the question using only the remembered facts."
RECON_Q = "Repeat the memory verbatim."


# --------------------------------------------------------------------- 模型
def load_model(device="cuda:0", with_adapter=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map=device)
    if with_adapter:
        import deltamem_compat as DMC
        _cfg, attach, load_adp = DMC.import_delta_mem()
        attach(model, _cfg.from_pretrained(ADAPTER))
        load_adp(model, ADAPTER)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def load_reader(ckpt, model, device="cuda:0"):
    blob = LM._safe_torch_load(ckpt)
    c = blob["config"]
    comp = LM.LatentCompressor(model.config.hidden_size, n_slots=c["n_slots"],
                               inner_dim=c["inner_dim"],
                               num_heads=c["num_heads"]).to(device)
    bridge = LM.LatentBridge(model.config.hidden_size, bottleneck=c["bottleneck"],
                             target_norm=float(c.get("target_norm", LM.LATENT_TARGET_NORM))).to(device)
    comp.load_state_dict(blob["compressor"])
    bridge.load_state_dict(blob["bridge"])
    comp.eval()
    bridge.eval()
    return blob, comp, bridge, c


def make_encoder(model, tok, comp, c):
    return LM.LatentEncoder(model, tok, comp, layer=c["layer"],
                            max_unit_tokens=c["max_unit_tokens"])


def _chat_embeds(model, tok, sysp, question, soft):
    """system + soft + question 的 inputs_embeds（与 generate_latent_answer 同构）。"""
    dev = LM._model_device(model)
    emb = model.get_input_embeddings()
    full, n_sys_chars = LM.build_chat_parts(tok, sysp, question)
    ids_full = tok(full, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    n_sys = 0
    if n_sys_chars:
        n_sys = int(tok(full[:n_sys_chars], return_tensors="pt",
                        add_special_tokens=False).input_ids.shape[1])
    parts = [emb(ids_full[:, :n_sys])]
    if soft is not None:
        parts.append(soft.reshape(1, -1, soft.shape[-1]).to(emb.weight.dtype))
    parts.append(emb(ids_full[:, n_sys:]))
    return torch.cat(parts, dim=1)


def _gen(model, tok, embeds, max_new_tokens=16):
    with torch.no_grad():
        out = model.generate(inputs_embeds=embeds,
                             attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long,
                                                       device=embeds.device),
                             max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None)
    return tok.decode(out[0], skip_special_tokens=True).strip()


# ------------------------------------------------------------------ 子命令
def sub_text(args, items):
    """明文记忆进 prompt：基准线。答不对就说明问题不在潜层读出。"""
    tok, model = load_model()
    hit = 0
    for it in items:
        msgs = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": "Remembered facts:\n" + it["memory"]
                 + "\n\nQuestion: " + it["question"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        with torch.no_grad():
            out = model.generate(input_ids=ids.to(model.device),
                                 attention_mask=torch.ones_like(ids, device=model.device),
                                 max_new_tokens=24, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
        pred = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        ok = str(it["answer"]).strip().lower() in pred.lower()
        hit += ok
        print(f"\nQ: {it['question']}\nA: {it['answer']}\n明文: {pred!r} {'✓' if ok else '✗'}")
    print(f"\n[text] {hit}/{len(items)}")


def sub_oracle(args, items):
    """把记忆原文的 token embedding 直接当软 token：验证注入通路。"""
    tok, model = load_model()
    hit = 0
    for it in items:
        mem_ids = tok(it["memory"], return_tensors="pt",
                      add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            soft = model.get_input_embeddings()(mem_ids)
        pred = _gen(model, tok, _chat_embeds(model, tok, SYS_PROMPT, it["question"], soft), 24)
        ok = str(it["answer"]).strip().lower() in pred.lower()
        hit += ok
        print(f"\nQ: {it['question']}\nA: {it['answer']}\n记忆 token={int(mem_ids.shape[1])} "
              f"oracle: {pred!r} {'✓' if ok else '✗'}")
    print(f"\n[oracle] {hit}/{len(items)}")


def sub_scale(args, items):
    """软 token 范数 vs 真实 embedding；以及置零/换随机后 loss 是否变化。"""
    tok, model = load_model()
    blob, comp, bridge, c = load_reader(args.ckpt, model)
    enc = make_encoder(model, tok, comp, c)
    print(f"[ckpt] {args.ckpt} m={c['n_slots']} target_norm={c.get('target_norm')}")
    acc = collections.defaultdict(list)
    for it in items:
        z, _ = T.encode_item(enc, it, tag="diag")
        with torch.no_grad():
            soft = bridge(z)
            real = model.get_input_embeddings()(
                tok(it["memory"], return_tensors="pt",
                    add_special_tokens=False).input_ids.to(model.device))
        acc["norm_real"].append(float(real.norm(dim=-1).mean()))
        acc["norm_soft"].append(float(soft.norm(dim=-1).mean()))
        for tag, s in (("loss_trained", soft), ("loss_zeros", torch.zeros_like(soft)),
                       ("loss_random", torch.randn_like(soft) * soft.std())):
            emb, lab, n = T.build_student_inputs(model, tok, s, SYS_PROMPT,
                                                 it["question"], it["answer"])
            with torch.no_grad():
                lg = model(inputs_embeds=emb,
                           attention_mask=torch.ones(emb.shape[:2], dtype=torch.long,
                                                     device=emb.device),
                           use_cache=False).logits
                acc[tag].append(float(torch.nn.functional.cross_entropy(
                    lg[0, -n - 1:-1], lab[0, -n:])))
    for k in ("norm_real", "norm_soft", "loss_trained", "loss_zeros", "loss_random"):
        v = acc[k]
        print(f"  {k:14s} {sum(v)/len(v):9.4f}")


def sub_probe(args, items):
    """z 里还有多少答案信息 + 重建保真度 + 平凡基线（三者缺一不可解读）。"""
    tok, model = load_model()
    blob, comp, bridge, c = load_reader(args.ckpt, model)
    enc = make_encoder(model, tok, comp, c)

    # ① 重建保真度：从软 token 重建记忆原文的 token 级准确率
    hit = tot = 0
    for it in items[:args.recon_n]:
        z, _ = T.encode_item(enc, it, tag="diag")
        with torch.no_grad():
            soft = bridge(z)
            emb, lab, n = T.build_student_inputs(model, tok, soft, SYS_PROMPT,
                                                 RECON_Q, it["memory"])
            with LM.delta_write_disabled(model):
                lg = model(inputs_embeds=emb,
                           attention_mask=torch.ones(emb.shape[:2], dtype=torch.long,
                                                     device=emb.device),
                           use_cache=False).logits
        # causal LM：logits[t] 预测 t+1，故第 i 个答案 token 的预测取它前一位的 logits
        hit += int((lg[0, -n - 1:-1].argmax(-1) == lab[0, -n:]).sum().item())
        tot += int(n)
    recon = hit / tot if tot else 0.0

    # ② 平凡基线：永远只吐最高频的 K 个 token 能拿多少
    cnt = collections.Counter()
    for it in items:
        cnt.update(tok(str(it["memory"]), add_special_tokens=False)["input_ids"])
    total = sum(cnt.values())
    top1 = cnt.most_common(1)[0][1] / total
    top10 = sum(v for _, v in cnt.most_common(10)) / total

    # ③ 线性探针：z -> 答案首 token（只在候选集合上分类，避免 4 万×15 万的巨型矩阵）
    Zs, Ys = [], []
    for it in items:
        z, _ = T.encode_item(enc, it, tag="diag")
        Zs.append(z.reshape(-1).float())
        Ys.append(tok(str(it["answer"]), add_special_tokens=False)["input_ids"][0])
    cands = sorted(set(Ys))
    idx = {t: i for i, t in enumerate(cands)}
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    Z = torch.stack(Zs).to(dev)
    Y = torch.tensor([idx[y] for y in Ys], device=dev)

    def _fit(Zin, tag, steps=1500, lr=5e-3):
        torch.manual_seed(0)
        W = torch.zeros(Zin.shape[1], len(cands), device=dev, requires_grad=True)
        opt = torch.optim.Adam([W], lr=lr)
        for _ in range(steps):
            loss = torch.nn.functional.cross_entropy(Zin @ W, Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        a = (Zin @ W).argmax(-1).eq(Y).float().mean().item()
        print(f"  probe[{tag}] acc {a:.1%}")
        return a

    probe = _fit(Z, "z")
    _fit(torch.randn_like(Z), "随机输入(对照)")

    print(f"\n[probe] 重建 token-acc {recon:.1%} | 平凡基线 top1 {top1:.1%} / top10 {top10:.1%}"
          f" | z 探针 {probe:.1%}（随机基线 {1/len(cands):.2%}）")
    print("  解读：重建准确率 ≈ 平凡基线 → 压缩器几乎没编码进内容（这正是 0/32 的原因）")


SUBS = {"text": sub_text, "oracle": sub_oracle, "scale": sub_scale, "probe": sub_probe}


def main():
    ap = argparse.ArgumentParser(description="潜层读出瓶颈定位")
    ap.add_argument("--sub", required=True, choices=sorted(SUBS))
    ap.add_argument("--ckpt", default="", help="scale/probe 需要：压缩器+桥接器 checkpoint")
    ap.add_argument("--data", default="synth/val_2k.jsonl")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--recon-n", type=int, default=16, help="probe 子命令里重建评估的样本数")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.sub in ("scale", "probe") and not args.ckpt:
        raise SystemExit(f"--sub {args.sub} 需要 --ckpt")
    if not os.path.exists(args.data):
        raise SystemExit(f"找不到数据：{args.data}（先跑 tools/gen_synth_qa.py）")
    items = T.load_synth_jsonl(args.data)[: max(args.limit, args.recon_n)]
    SUBS[args.sub](args, items)


if __name__ == "__main__":
    main()
