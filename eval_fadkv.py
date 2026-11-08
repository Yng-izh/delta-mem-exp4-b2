#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_fadkv.py — FAD-KV 主表：统一 full_prefix 协议下比四档。

| 方法 | 存储 KiB/token | 序列保留率 | 说明 |
|---|---|---|---|
| `native` | 120 | 100% | 原生 L6~35 KV（上限） |
| `pca1024` | 2.0 | 100% | 实验 E 的全局线性基（旧基线） |
| `fadkv1` | 2.0 | 100% | 阶段一：分层 K/V 解耦 + 蒸馏 |
| `fadkv2` | ~1.03 | ~47.5% | 阶段二：锚点 + 池化 |

协议：`full_prefix` 编码、中间插入、同一套贪心解码、`max_new_tokens` 一致；EM 用归一化
整串相等，F1 用 HotpotQA 官方口径（`eval_latent_table.f1_score`）。主判据是配对 McNemar
（对 `native`），存储/序列只是记账。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import torch

import latent_memory as LM
import train_latent_memory as T
import midlayer_kv as MK
import kv_lowrank as LR
import fadkv_codec as FC
import eval_latent_table as EV
import eval_midlayer_kv as ED
import train_fadkv as TR


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="FAD-KV 主表")
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--layers", default="6-35")
    ap.add_argument("--encoding", default="full_prefix", choices=MK.MID_KV_ENCODINGS)
    ap.add_argument("--global-basis", default="", help="实验 E 的全局 LowRankKV（pca1024 臂）")
    ap.add_argument("--global-dim", type=int, default=1024)
    ap.add_argument("--stage1", default="", help="阶段一 codec ckpt")
    ap.add_argument("--stage2", default="", help="阶段二 codec ckpt")
    ap.add_argument("--wrong", action="store_true", help="给 fadkv1 加 wrong 反事实")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--system-prompt",
                    default="You are a helpful assistant. Answer with the shortest exact value.")
    ap.add_argument("--base", default=T.BASE)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--show-items", type=int, default=2)
    ap.add_argument("--save-json", default="")
    ap.add_argument("--dump-jsonl", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    layers = TR.parse_layers(args.layers)
    arms = ["native"] + (["pca1024"] if args.global_basis else []) \
        + (["fadkv1"] if args.stage1 else []) + (["fadkv2"] if args.stage2 else [])
    if args.wrong and "fadkv1" in arms:
        arms.append("fadkv1_wrong")
    if args.dry_run:
        print(f"[dry] layers={layers[0]}..{layers[-1]} arms={arms}")
        return 0
    dev = torch.device(f"cuda:{args.gpu}")
    T._setup_tf32(not args.no_tf32)
    ns = argparse.Namespace(smoke_model=False, layer=0, base=args.base, dtype=args.dtype,
                            gpu=args.gpu, no_adapter=True, adapter="",
                            grad_checkpointing=False)
    model, tokenizer, dev, hidden = T._load_backbone(ns, dev)
    T.freeze_model(model)
    layout = MK.uv_layout(model, layers)

    global_basis = None
    if args.global_basis:
        global_basis = LR.LowRankKV.load(args.global_basis, device=dev, dim=int(args.global_dim))
    codec1 = FC.load_codec(args.stage1, device=dev)[0] if args.stage1 else None
    codec2 = FC.load_codec(args.stage2, device=dev)[0] if args.stage2 else None
    for c in (codec1, codec2):
        if c is not None:
            c.eval()
    storage: Dict[str, Dict[str, Any]] = {"native": {
        "kib_per_token": round(layout["dim"] * 2 / 1024, 2), "seq_keep": 1.0}}
    if global_basis is not None:
        storage["pca1024"] = {"kib_per_token": round(global_basis.dim * 2 / 1024, 2),
                              "seq_keep": 1.0}
    if codec1 is not None:
        d = codec1.describe()
        storage["fadkv1"] = {"kib_per_token": round(d["bytes_per_token"] / 1024, 2),
                             "seq_keep": 1.0}
    if codec2 is not None:
        d = codec2.storage()
        storage["fadkv2"] = {"kib_per_token": d["kib_per_token"],
                             "seq_keep": d["seq_keep"]}
    print(f"[cfg] uv_dim={layout['dim']} arms={arms} storage={storage}", flush=True)

    data = T.load_synth_jsonl(args.data, limit=int(args.limit or 0), seed=int(args.seed))
    if not data:
        raise SystemExit(f"{args.data} 里没有样本")
    donor = ED.pick_donors(data, tokenizer, args) if "fadkv1_wrong" in arms else None

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, item in enumerate(data):
        ev = ED.evidence_text(item)
        gold = item["answer"]
        if isinstance(gold, (list, tuple)):
            gold = " ".join(str(g) for g in gold)
        try:
            split = MK.split_prompt_tokens(tokenizer, args.system_prompt, ev, item["question"])
        except MK.SplitError as e:
            print(f"[skip] idx={i} 切分失败：{e}", flush=True)
            continue
        row: Dict[str, Any] = {"idx": i, "gold_ans": str(gold), "question": item["question"],
                               "ans_type": EV.answer_type(gold), "n_e": int(split["ne"])}
        for arm in arms:
            mem = build_mem(arm, model, tokenizer, split, layers, args,
                            global_basis, codec1, codec2, data, donor, i)
            pred, meta = MK.midlayer_kv_generate(model, tokenizer, mem, split,
                                                 max_new_tokens=int(args.max_new_tokens))
            row[arm] = str(pred)
            row[f"{arm}__ok"] = bool(T._norm_ans(pred) == T._norm_ans(str(gold)))
            row[f"{arm}__f1"] = float(EV.f1_score(pred, str(gold)))
            row[f"{arm}__nmem"] = int(meta.get("n_mem", 0))
        rows.append(row)
        if args.show_items and i < int(args.show_items):
            print(f"  [item {i}] gold={gold!r} ne={row['n_e']}", flush=True)
            for arm in arms:
                mark = "✓" if row[f"{arm}__ok"] else "✗"
                print(f"      {arm:14s} {mark} {row[arm]!r}", flush=True)
        if args.progress_every and (i + 1) % int(args.progress_every) == 0:
            el = time.time() - t0
            part = " ".join(f"{a}={ED.arm_acc(rows, a):.3f}" for a in arms)
            print(f"[progress] {i + 1}/{len(data)} {part} elapsed={el:.0f}s "
                  f"eta={el / (i + 1) * (len(data) - i - 1):.0f}s", flush=True)

    summary = summarize(args, rows, arms, storage)
    print_table(summary, arms)
    if args.dump_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_jsonl)) or ".", exist_ok=True)
        with open(args.dump_jsonl, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[out] 逐题明细 -> {args.dump_jsonl}", flush=True)
    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "config": vars(args), "rows": rows}, f,
                      ensure_ascii=False, indent=1)
        print(f"[out] 汇总 -> {args.save_json}", flush=True)
    return 0


def build_mem(arm, model, tokenizer, split, layers, args, global_basis, codec1, codec2,
              data, donor, i) -> MK.MemoryKV:
    if arm == "native":
        return MK.build_evidence_kv(model, tokenizer, split, args.encoding,
                                    pos_start=int(split["ns"]), n_tokens=int(split["ne"]),
                                    layers=layers)
    if arm == "pca1024":
        U, _ = MK.capture_uv(model, tokenizer, split, args.encoding, layers)
        return MK.memory_kv_from_uv(model, global_basis.decode(global_basis.encode(U)),
                                    int(split["ns"]), layers)
    if arm in ("fadkv1", "fadkv2"):
        codec = codec1 if arm == "fadkv1" else codec2
        mem, _ = TR.encode_to_memory(model, tokenizer, codec, split, layers,
                                    1 if arm == "fadkv1" else 2, args.encoding)
        return mem
    if arm == "fadkv1_wrong":
        dsplit = MK.split_prompt_tokens(tokenizer, args.system_prompt,
                                        ED.evidence_text(data[donor[i]]),
                                        data[donor[i]]["question"])
        U, _ = MK.capture_uv(model, tokenizer, dsplit, args.encoding, layers,
                             n_tokens=int(split["ne"]), pad_repeat=True)
        return MK.memory_kv_from_uv(model, codec1.decode(codec1.encode(U)),
                                    int(split["ns"]), layers)
    raise ValueError(f"未知臂 {arm}")


def summarize(args, rows, arms, storage) -> Dict[str, Any]:
    nn = max(1, len(rows))
    em = {a: ED.arm_acc(rows, a) for a in arms}
    f1 = {a: sum(float(r[f"{a}__f1"]) for r in rows) / nn for a in arms}
    s: Dict[str, Any] = {"n": len(rows), "em": em, "f1": f1, "storage": storage,
                         "data": str(args.data), "encoding": args.encoding,
                         "layers": f"{TR.parse_layers(args.layers)[0]}-"
                                   f"{TR.parse_layers(args.layers)[-1]}"}
    s["paired"] = [ED.paired([bool(r[f"{a}__ok"]) for r in rows],
                             [bool(r["native__ok"]) for r in rows], a, "native")
                   for a in arms if a != "native"]
    by: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        t = str(r.get("ans_type") or "other")
        d = by.setdefault(t, {"n": 0, **{a: 0 for a in arms}})
        d["n"] += 1
        for a in arms:
            d[a] += int(bool(r[f"{a}__ok"]))
    for t, d in by.items():
        for a in arms:
            d[a] = d[a] / max(1, d["n"])
    s["by_ans_type"] = by
    s["verdict"] = verdict(s, arms)
    return s


def verdict(s: Dict[str, Any], arms: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"native": s["em"].get("native")}
    nat = s["em"].get("native") or 0.0
    for a in arms:
        if a == "native" or a.endswith("_wrong"):
            continue
        keep = s["em"][a] / nat if nat else 0.0
        out[a] = (f"EM={s['em'][a]:.3f} F1={s['f1'][a]:.3f} "
                  f"保留原生 {100 * keep:.1f}%")
    s1 = s["em"].get("fadkv1")
    if s1 is not None:
        out["stage1_hit40"] = bool(s1 >= 0.40)
        out["text"] = (f"阶段一 {s1:.3f}（目标 ≥0.40，{'达到' if s1 >= 0.40 else '未达到'}）")
    return out


def print_table(s: Dict[str, Any], arms: List[str]) -> None:
    nat = s["em"].get("native") or 0.0
    print("")
    print("| 方法 | 存储 KiB/Token | 序列保留率 | EM% | F1% | 相对原生 |")
    print("|---|---|---|---|---|---|")
    for a in arms:
        st = s["storage"].get(a, {})
        rel = "—" if (a == "native" or not nat) else f"{100 * s['em'][a] / nat:.0f}%"
        kib = st.get("kib_per_token", "—")
        keep = st.get("seq_keep")
        keep_s = "—" if keep is None else f"{100 * float(keep):.1f}%"
        print(f"| {a} | {kib} | {keep_s} | "
              f"{100 * s['em'][a]:.1f} | {100 * s['f1'][a]:.1f} | {rel} |")
    print("")
    print("| 配对（vs native） | A% | B% | 差 | 只有A对 | 只有B对 | p |")
    print("|---|---|---|---|---|---|---|")
    for p in s["paired"]:
        print(f"| {p['a']} | {100 * p['acc_a']:.1f} | {100 * p['acc_b']:.1f} "
              f"| {100 * p['delta']:+.1f} | {p['only_a']} | {p['only_b']} | {p['p']:.4f} |")
    if s.get("by_ans_type"):
        specs = [a for a in arms]
        print("")
        print("| 题型 | n | " + " | ".join(specs) + " |")
        print("|---" * (2 + len(specs)) + "|")
        for t in sorted(s["by_ans_type"]):
            d = s["by_ans_type"][t]
            print(f"| {t} | {d['n']} | " + " | ".join(f"{100 * d[a]:.1f}"
                                                      for a in specs) + " |")
    print("")
    for k in sorted(s["verdict"]):
        if k == "native":
            continue
        print(f"[判据] {k}: {s['verdict'][k]}", flush=True)
    print(json.dumps({k: round(v, 4) for k, v in s["em"].items()}, ensure_ascii=False),
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
