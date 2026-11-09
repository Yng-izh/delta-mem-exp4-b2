#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/diag_kv_by_type.py — 压缩误差落在哪些 token / 哪些层段上（实验 F 诊断）。

问题：阶段一 2 KiB/token 拿到 28.0% EM，但**日期/数字已追平原生、实体题只有一半**。
那么"实体的 KV 是不是被压没了"这件事必须直接量，而不是从 EM 反推。

做法：对每条样本取原生 U，过一个 codec 得到 Û，逐 token 算相对误差，再按 token 类型
（数字 / 实体 / 关系词 / 功能词）与层段组（seg0/1/2 × K/V）分层汇总。

判读：
  - 某类 token 的**误差占比 / token 占比** 明显 >1 -> 它承受了不成比例的损失
  - 实体类的这个倍数若显著高于 1，说明压缩确实在吞实体，方向应该转向"实体位保真"
  - 按层段看：若错误集中在某一段，说明该段的预算给少了（预算平均摊的代价）

用法：
    python tools/diag_kv_by_type.py --ckpt $B/results/exp_f_fadkv/fadkv_stage1.pt \
        --data $B/synth/hotpot_val500.jsonl --limit 60
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import torch

import latent_memory as LM
import train_latent_memory as T
import midlayer_kv as MK
import fadkv_codec as FC
import eval_midlayer_kv as ED

# 事实关键位用的停用词（与大写判断配合，避免把句首虚词算成实体）
_STOP = frozenset(
    "the a an it he she they we you i his her their its this that these those there "
    "however after before during in on at as by for from with while when where which "
    "who whom whose what how also both each other some many most more than then thus".split())
_REL = frozenset(
    "founded established born died married divorced directed produced written located "
    "served held became released published created member part appointed elected joined "
    "led owned designed built named known worked studied graduated awarded won lost "
    "replaced succeeded preceded attended represented signed defeated beat".split())


def token_types(tokenizer: Any, ids: Sequence[int]) -> List[str]:
    """逐个 token 判类型：digit / entity / relation / func。"""
    out: List[str] = []
    for i in ids:
        raw = tokenizer.convert_ids_to_tokens(int(i)) or ""
        word_start = raw[:1] in ("Ġ", "▁", " ")
        txt = raw.replace("Ġ", " ").replace("▁", " ").strip()
        low = txt.lower().strip(".,;:!?\"'()[]{}").strip()
        if any(ch.isdigit() for ch in txt):
            out.append("digit")
        elif word_start and txt[:1].isupper() and low not in _STOP:
            out.append("entity")
        elif low in _REL:
            out.append("relation")
        else:
            out.append("func")
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="压缩误差按 token 类型/层段分层")
    ap.add_argument("--ckpt", required=True, help="阶段一 codec（fadkv_stage1.pt）")
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--layers", default="6-35")
    ap.add_argument("--encoding", default="full_prefix", choices=MK.MID_KV_ENCODINGS)
    ap.add_argument("--base", default=T.BASE)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    ap.add_argument("--system-prompt",
                    default="You are a helpful assistant. Answer with the shortest exact value.")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    layers = list(range(int(args.layers.split("-")[0]), int(args.layers.split("-")[1]) + 1))
    dev = torch.device(f"cuda:{args.gpu}")
    T._setup_tf32(True)
    ns = argparse.Namespace(smoke_model=False, layer=0, base=args.base, dtype=args.dtype,
                            gpu=args.gpu, no_adapter=True, adapter="",
                            grad_checkpointing=False)
    model, tokenizer, dev, _hidden = T._load_backbone(ns, dev)
    T.freeze_model(model)
    codec, _cfg = FC.load_codec(args.ckpt, device=dev)
    codec.eval()
    print(f"[codec] {codec.describe()}", flush=True)
    groups = codec.groups
    data = T.load_synth_jsonl(args.data, limit=int(args.limit or 0), seed=int(args.seed))

    types = ["digit", "entity", "relation", "func"]
    tok_cnt = {t: 0 for t in types}
    tok_sse = {t: 0.0 for t in types}
    tok_rel = {t: [] for t in types}
    grp_sse = {n: 0.0 for n in groups}
    grp_den = {n: 0.0 for n in groups}
    n_tok = 0
    n_item = 0

    with torch.no_grad():
        for item in data:
            ev = ED.evidence_text(item)
            if not ev.strip():
                continue
            try:
                split = MK.split_prompt_tokens(tokenizer, args.system_prompt, ev,
                                               item["question"])
            except MK.SplitError:
                continue
            U, _layout = MK.capture_uv(model, tokenizer, split, args.encoding, layers)
            Uhat = codec.roundtrip(U, torch.bfloat16)   # 按真实存储精度（bf16）算
            Uf, Uf_hat = U.float(), Uhat.float()
            per_tok = (Uf_hat - Uf).pow(2).sum(dim=-1)
            tok_norm2 = Uf.pow(2).sum(dim=-1).clamp_min(1e-6)
            tt = token_types(tokenizer, split["ids_E"])
            if len(tt) != int(U.shape[0]):
                tt = tt[: int(U.shape[0])] + ["func"] * max(0, int(U.shape[0]) - len(tt))
            for j, tp in enumerate(tt):
                tok_cnt[tp] += 1
                tok_sse[tp] += float(per_tok[j])
                tok_rel[tp].append(float((per_tok[j] / tok_norm2[j]).sqrt()))
            for name, g in groups.items():
                cols = g["cols"]
                grp_sse[name] += float(((Uf_hat[:, cols] - Uf[:, cols]) ** 2).sum())
                grp_den[name] += float((Uf[:, cols] ** 2).sum())
            n_tok += int(U.shape[0])
            n_item += 1

    total_sse = sum(tok_sse.values())
    print(f"\n[样本] items={n_item} tokens={n_tok}\n")
    print("| token 类型 | token 数 | 占比 | 平均相对误差 | 误差占比 | 放大倍数 |")
    print("|---|---|---|---|---|---|")
    for t in types:
        share = tok_cnt[t] / max(1, n_tok)
        err_share = tok_sse[t] / max(total_sse, 1e-12)
        mean_rel = (sum(tok_rel[t]) / max(1, len(tok_rel[t])))
        amp = (err_share / share) if share > 0 else float("nan")
        print(f"| {t} | {tok_cnt[t]} | {100 * share:.1f}% | {100 * mean_rel:.2f}% | "
              f"{100 * err_share:.1f}% | {amp:.2f}× |")
    print("\n| 层段组 | 潜码维 | 相对误差 | 误差占比 |")
    print("|---|---|---|---|")
    tot_g = sum(grp_sse.values())
    for name in groups:
        rel = (grp_sse[name] / max(grp_den[name], 1e-12)) ** 0.5
        print(f"| {name} | {codec.latent_dims[name]} | {100 * rel:.2f}% | "
              f"{100 * grp_sse[name] / max(tot_g, 1e-12):.1f}% |")
    ent_share = tok_cnt["entity"] / max(1, n_tok)
    ent_err = tok_sse["entity"] / max(total_sse, 1e-12)
    print(f"\n[判据] 实体 token 占 {100 * ent_share:.1f}%、承担 {100 * ent_err:.1f}% 的误差，"
          f"放大 {ent_err / max(ent_share, 1e-9):.2f}×；"
          f"全局 KV 相对误差 {100 * (total_sse / max(sum(grp_den.values()), 1e-12)) ** 0.5:.2f}%",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
