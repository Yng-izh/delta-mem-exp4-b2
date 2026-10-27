#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/mem_probe.py — 显存到底被谁吃掉（分段计时/计显存）。

现象：4 步 × 40 条 HotpotQA（每条 34 个记忆单元）在 32GB V100 上 OOM，
PyTorch 已分配 27.7 GiB。这个脚本按阶段打印 `max_memory_allocated` 的**增量**，
定位到"加载模型 / 编码单元 / 教师前向 / 学生前向"里的哪一步。

用法：python tools/mem_probe.py --gpu 2 --n-items 3
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import torch

import latent_memory as LM
import train_latent_memory as T


def peak_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 2 ** 20


def mark(tag, t0, last):
    p = peak_mb()
    print(f"[mem] {tag:<28} peak={p:8.0f} MiB  (+{p - last:7.0f})  "
          f"dt={time.time() - t0:6.1f}s", flush=True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--data", default="synth/hotpot_train.jsonl")
    ap.add_argument("--n-items", type=int, default=3)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--max-unit-tokens", type=int, default=64)
    ap.add_argument("--max-slots", type=int, default=8)
    ap.add_argument("--base", default=T.BASE)
    ap.add_argument("--adapter", default=T.ADAPTER)
    ap.add_argument("--no-adapter", action="store_true")
    args = ap.parse_args()
    args.smoke_model = False
    args.device = ""
    args.gpu = str(args.gpu)

    t0 = time.time()
    model, tok, dev, hidden = T._load_backbone(args, dev=torch.device(f"cuda:{args.gpu}"))
    last = mark("加载主干", t0, 0.0)
    T.freeze_model(model)
    last = mark("freeze", time.time(), last)

    comp = LM.LatentCompressor(hidden, n_slots=args.m, inner_dim=512, num_heads=8).to(dev)
    bridge = LM.LatentBridge(hidden, bottleneck=512, target_norm=1.05).to(dev)
    router = LM.LatentRouter(hidden, n_slots=args.m, d_key=256).to(dev)
    enc = LM.LatentEncoder(model, tok, comp, layer=args.layer,
                           max_unit_tokens=args.max_unit_tokens, device=dev)
    last = mark("构建 comp/bridge/router", time.time(), last)

    data = T.load_synth_jsonl(args.data, limit=args.n_items)
    print(f"[data] {len(data)} 条，单元数={[len(T.item_units(it)) for it in data]}", flush=True)
    for k, item in enumerate(data):
        t0 = time.time()
        reset_ok = T.reset_delta_state(model)
        slots, _cat, info = T.encode_item_slots(enc, item, tag="probe")
        slots = LM.stack_payloads(list(slots)).unsqueeze(0)
        last = mark(f"item{k} 编码 {slots.shape[0]}x{slots.shape[1]} 槽 "
                    f"(reset={reset_ok})", t0, last)
        t0 = time.time()
        pred, meta = LM.generate_latent_answer(
            model=model, tokenizer=tok, bridge=bridge, payloads=slots,
            question=item["question"], system_prompt="You are a helpful assistant.",
            max_new_tokens=8, router=router, layer=args.layer,
            hops=max(1, int(item.get("hops", 1) or 1)))
        last = mark(f"item{k} 读出 n_soft={meta.get('n_soft')}", t0, last)
        print(f"   pred={pred[:30]!r}", flush=True)
        torch.cuda.reset_peak_memory_stats()
    print(f"[final] 当前分配={torch.cuda.memory_allocated()/2**20:.0f} MiB "
          f"保留={torch.cuda.memory_reserved()/2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
