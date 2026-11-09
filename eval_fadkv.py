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
    ap.add_argument("--data", default="", help="eval 需要；fit-global 不需要")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--layers", default="6-35")
    ap.add_argument("--encoding", default="full_prefix", choices=MK.MID_KV_ENCODINGS)
    ap.add_argument("--stage", choices=("eval", "fit-global"), default="eval",
                    help="fit-global：在**同一个 full_prefix 协议**下重新拟合全局 PCA 基线")
    ap.add_argument("--fit-data", default="")
    ap.add_argument("--fit-limit", type=int, default=0)
    ap.add_argument("--fit-tokens", type=int, default=20000)
    ap.add_argument("--fit-basis", default="", help="fit-global 的输出路径")
    ap.add_argument("--global-basis", default="",
                    help="全局 LowRankKV（pca1024 臂）；必须与本表同前缀拟合，否则不可比")
    ap.add_argument("--global-dim", type=int, default=1024)
    ap.add_argument("--stage1", default="", help="阶段一 codec ckpt")
    ap.add_argument("--stage2", default="", help="阶段二 codec ckpt")
    ap.add_argument("--wrong", action="store_true", help="给 fadkv1 加 wrong 反事实")
    ap.add_argument("--latent-dtype", default="fp16", choices=("bf16", "fp16", "fp32"),
                    help="fadkv 臂默认的潜码精度")
    ap.add_argument("--latent-arms", default="",
                    help="逗号分隔的精度列表（如 bf16,fp16,fp32）：同一权重下逐精度出臂，"
                         "用来把'量化损失'从'训练配置差异'里拆出来")
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


def latent_dtype_of(args, name=None):
    return TR.latent_dtype_of(args, name)


def stage_fit_global(args, model, tokenizer, dev, layout, layers) -> int:
    """在训练集上用**同一个 full_prefix 协议**拟合全局 PCA 基线（pca1024 臂）。

    必须重拟合：实验 E 的基底是用 anchor 协议拟合的，而本表全部用 full_prefix。
    不重拟合，41.0% 对 23.5% 的提升里就混着"前缀分布不同"这一项。
    """
    if not (args.fit_data and args.fit_basis):
        raise SystemExit("fit-global 需要 --fit-data 与 --fit-basis")
    data = T.load_synth_jsonl(args.fit_data, limit=int(args.fit_limit or 0),
                              seed=int(args.seed))
    if not data:
        raise SystemExit(f"{args.fit_data} 里没有样本")

    def batches():
        used = 0
        for it in data:
            ev = ED.evidence_text(it)
            if not ev.strip():
                continue
            try:
                split = MK.split_prompt_tokens(tokenizer, args.system_prompt, ev,
                                               it["question"])
            except MK.SplitError:
                continue
            U, _ = MK.capture_uv(model, tokenizer, split, args.encoding, layers)
            yield U
            used += int(U.shape[0])
            if int(args.fit_tokens) > 0 and used >= int(args.fit_tokens):
                return

    lr = LR.fit_lowrank(batches, layout, dim=int(args.global_dim), oversample=64,
                        device=dev, seed=int(args.seed),
                        log=lambda m: print(m, flush=True))
    lr.save(args.fit_basis)
    print(f"[fit-global] {args.encoding} 协议，{lr.describe()} -> {args.fit_basis}",
          flush=True)
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if not hasattr(args, "fit_tokens"):
        args.fit_tokens = 20000
    layers = TR.parse_layers(args.layers)
    latent_arms = [x.strip() for x in str(args.latent_arms).split(",") if x.strip()]
    arms = ["native"] + (["pca1024"] if args.global_basis else [])
    if args.stage1:
        arms += [f"fadkv1_{d}" for d in latent_arms] if latent_arms else ["fadkv1"]
    if args.stage2:
        arms += ["fadkv2"]
    if args.wrong and any(a == "fadkv1" or a.startswith("fadkv1_") for a in arms):
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
    if args.stage == "fit-global":
        return stage_fit_global(args, model, tokenizer, dev, layout, layers)
    if not args.data:
        raise SystemExit("eval 需要 --data")

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
        dims = d["latent_dim"]
        for a in [x for x in arms if x == "fadkv1" or x.startswith("fadkv1_")]:
            dt = a.split("_", 1)[1] if "_" in a else str(args.latent_dtype)
            nbytes = 4 if dt == "fp32" else 2
            storage[a] = {"kib_per_token": round(dims * nbytes / 1024, 2),
                          "seq_keep": 1.0, "latent_dtype": dt}
    if codec2 is not None:
        d = codec2.storage()
        storage["fadkv2"] = {"kib_per_token": d["kib_per_token"],
                             "kib_per_token_latent_only": d["kib_per_token_latent_only"],
                             "seq_keep": d["seq_keep"],
                             "note": "含每个槽 4 字节位置元数据"}
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


def MK_FAGS(tokenizer, split):
    """锚点选择用的内容词标记（数字/实体/关系词）。"""
    return FC.content_flags(tokenizer, split["ids_E"])


def build_mem(arm, model, tokenizer, split, layers, args, global_basis, codec1, codec2,
              data, donor, i) -> MK.MemoryKV:
    if arm == "native":
        return MK.build_evidence_kv(model, tokenizer, split, args.encoding,
                                    pos_start=int(split["ns"]), n_tokens=int(split["ne"]),
                                    layers=layers)
    if arm == "pca1024":
        U, _ = MK.capture_uv(model, tokenizer, split, args.encoding, layers)
        z = global_basis.encode(U)
        if latent_dtype_of(args) is not None:
            z = z.to(latent_dtype_of(args)).float()
        return MK.memory_kv_from_uv(model, global_basis.decode(z),
                                    int(split["ns"]), layers)
    # 只有后缀是**已知精度**时才算精度臂：否则 `fadkv1_wrong` 会被解析成 dtype='wrong'
    if arm == "fadkv1" or (arm.startswith("fadkv1_")
                           and arm.split("_", 1)[1] in TR.LATENT_DTYPES):
        dt = latent_dtype_of(args, arm.split("_", 1)[1] if "_" in arm else None)
        mem, _ = TR.encode_to_memory(
            model, tokenizer, codec1, split, layers, 1, args.encoding,
            MK_FAGS(tokenizer, split), dt)
        return mem
    if arm == "fadkv2":
        mem, _ = TR.encode_to_memory(
            model, tokenizer, codec2, split, layers, 2, args.encoding,
            MK_FAGS(tokenizer, split), latent_dtype_of(args))
        return mem
    if arm == "fadkv1_wrong":
        dsplit = MK.split_prompt_tokens(tokenizer, args.system_prompt,
                                        ED.evidence_text(data[donor[i]]),
                                        data[donor[i]]["question"])
        U, _ = MK.capture_uv(model, tokenizer, dsplit, args.encoding, layers,
                             n_tokens=int(split["ne"]), pad_repeat=True)
        z = codec1.encode(U, latent_dtype_of(args))
        return MK.memory_kv_from_uv(model, codec1.decode(z), int(split["ns"]), layers)
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
