#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_kv_lowrank.py — 实验 E：跨层 KV 的可压缩性验证（零训练，纯线性压缩）。

两步：先 `--stage fit` 在训练集上拟合一套共享线性基（`kv_lowrank.fit_lowrank`），
再 `--stage eval` 在固定的 200 道验证题上比未压缩与各压缩维度的 QA。

臂
--
| 臂 | 记忆 | 作用 |
|---|---|---|
| `text_full` | 明文证据进 system | 上限参照（与实验 D 同一套解码） |
| `anchor6` | 锚点编码的**未压缩**逐层 KV（L6~35） | 实验 E 的基线（补 D 缺的 `anchor:L6`） |
| `anchor6_wrong` | 同上，换成别的题的证据 | 反事实 |
| `native_rt` | 同一份 KV 走"提取→还原"恒等路径 | **正确性门槛**：必须与 `anchor6` 一致 |
| `pca{k}` / `pca{k}_wrong` | 编码到 k 维再解码 | 可压缩性 |

门槛顺序不能反：`native_rt` 与 `anchor6` 不一致 -> 先修解码器，**不能**把掉点归因给 PCA。

记账两个指标：KV 重建相对误差与最终 EM。判据是 QA，不是重建误差。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import torch

import latent_memory as LM
import train_latent_memory as T
import midlayer_kv as MK
import kv_lowrank as LR
import eval_latent_table as EV
import eval_midlayer_kv as ED            # 复用配对检验/证据文本/供体选择的口径


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="实验 E：跨层 KV 低秩压缩")
    ap.add_argument("--stage", choices=("fit", "eval"), default="eval")
    ap.add_argument("--data", default="", help="eval 用的验证集")
    ap.add_argument("--fit-data", default="", help="fit 用的训练集")
    ap.add_argument("--basis", default="", help="基底 ckpt 路径（fit 写、eval 读）")
    ap.add_argument("--dim", type=int, default=2048, help="fit：拟合到多少维")
    ap.add_argument("--dims", default="512,1024,2048", help="eval：比哪几档")
    ap.add_argument("--layers", default="6-35", help="跨层 K/V 取哪几层，如 6-35")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--fit-limit", type=int, default=0, help="fit 最多用多少条样本")
    ap.add_argument("--fit-tokens", type=int, default=30000, help="fit 最多采多少 token 位置")
    ap.add_argument("--oversample", type=int, default=64)
    ap.add_argument("--encoding", default="anchor", choices=MK.MID_KV_ENCODINGS)
    ap.add_argument("--max-mem-tokens", type=int, default=MK.MID_KV_MAX_MEM_TOKENS)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--system-prompt",
                    default="You are a helpful assistant. Answer with the shortest exact value.")
    ap.add_argument("--base", default=T.BASE)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--show-items", type=int, default=0)
    ap.add_argument("--save-json", default="")
    ap.add_argument("--dump-jsonl", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="只解析参数与层区间，不加载模型")
    return ap.parse_args(argv)


def parse_layers(spec: str) -> List[int]:
    s = str(spec).strip()
    if "-" in s:
        lo, hi = s.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def parse_dims(spec: str) -> List[int]:
    return [int(x) for x in str(spec).split(",") if x.strip()]


# ---------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    layers = parse_layers(args.layers)
    if args.dry_run:
        print(f"[dry] layers={layers[0]}..{layers[-1]} ({len(layers)} 层) "
              f"dims={parse_dims(args.dims)} stage={args.stage}")
        return 0
    if not args.basis:
        raise SystemExit("必须给 --basis（fit 写入 / eval 读取）")
    dev = torch.device(f"cuda:{args.gpu}")
    ns = argparse.Namespace(smoke_model=False, layer=0, base=args.base, dtype=args.dtype,
                            gpu=args.gpu, no_adapter=True, adapter="",
                            grad_checkpointing=False)
    T._setup_tf32(not args.no_tf32)
    model, tokenizer, dev, hidden = T._load_backbone(ns, dev)
    T.freeze_model(model)
    n_layers = len(MK.decoder_layers(model))
    if max(layers) >= n_layers:
        raise SystemExit(f"--layers 上界 {max(layers)} 超出主干层数 {n_layers}")
    layout = MK.uv_layout(model, layers)
    print(f"[model] hidden={hidden} layers={n_layers} uv_dim={layout['dim']} "
          f"layers={layers[0]}..{layers[-1]}", flush=True)
    if args.stage == "fit":
        return stage_fit(args, model, tokenizer, dev, layout, layers)
    return stage_eval(args, model, tokenizer, dev, layout, layers)


# ---------------------------------------------------------------------
def _uv_batches(args, model, tokenizer, layers, data, token_budget: int):
    """返回一个闭包：每次调用都重新扫一遍数据、产出 U 批次（供三遍流式拟合共用）。"""
    def gen():
        used = 0
        for item in data:
            ev = ED.evidence_text(item)
            if not ev.strip():
                continue
            try:
                split = MK.split_prompt_tokens(tokenizer, args.system_prompt, ev,
                                               item["question"])
            except MK.SplitError:
                continue
            U, _ = MK.capture_uv(model, tokenizer, split, args.encoding, layers)
            yield U
            used += int(U.shape[0])
            if token_budget > 0 and used >= token_budget:
                return
    return gen


def stage_fit(args, model, tokenizer, dev, layout, layers) -> int:
    if not args.fit_data:
        raise SystemExit("fit 需要 --fit-data")
    data = T.load_synth_jsonl(args.fit_data, limit=int(args.fit_limit or 0),
                              seed=int(args.seed))
    if not data:
        raise SystemExit(f"{args.fit_data} 里没有样本")
    print(f"[fit] 训练样本 {len(data)} 条，编码协议={args.encoding}，"
          f"token 预算={args.fit_tokens}", flush=True)
    batches = _uv_batches(args, model, tokenizer, layers, data, int(args.fit_tokens))
    lr = LR.fit_lowrank(batches, layout, dim=int(args.dim),
                        oversample=int(args.oversample), device=dev, seed=int(args.seed),
                        log=lambda m: print(m, flush=True))
    lr.save(args.basis)
    print(f"[fit] 基底 -> {args.basis}  {lr.describe()}", flush=True)
    return 0


# ---------------------------------------------------------------------
def stage_eval(args, model, tokenizer, dev, layout, layers) -> int:
    if not args.data:
        raise SystemExit("eval 需要 --data")
    dims = parse_dims(args.dims)
    basis = LR.LowRankKV.load(args.basis, device=dev)
    for k in dims:
        if k > basis.dim:
            raise SystemExit(f"请求 {k} 维，但基底只拟合到 {basis.dim} 维")
    fits = {k: LR.LowRankKV.load(args.basis, device=dev, dim=k) for k in dims}
    data = T.load_synth_jsonl(args.data, limit=int(args.limit or 0), seed=int(args.seed))
    if not data:
        raise SystemExit(f"{args.data} 里没有样本")
    arms = (["text_full", "anchor6", "anchor6_wrong", "native_rt"]
            + [f"pca{k}" for k in dims] + [f"pca{k}_wrong" for k in dims])
    donor_idx = ED.pick_donors(data, tokenizer, args)
    print(f"[eval] n={len(data)} 臂={arms} 基底={basis.describe()}", flush=True)

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, item in enumerate(data):
        ev = ED.evidence_text(item)
        gold = item["answer"]
        if isinstance(gold, (list, tuple)):
            gold = " ".join(str(g) for g in gold)
        try:
            split = MK.split_prompt_tokens(tokenizer, args.system_prompt, ev,
                                           item["question"])
            dsplit = MK.split_prompt_tokens(tokenizer, args.system_prompt,
                                            ED.evidence_text(data[donor_idx[i]]),
                                            data[donor_idx[i]]["question"])
        except MK.SplitError as e:
            print(f"[skip] idx={i} 切分失败：{e}", flush=True)
            continue
        ne, nsi = int(split["ne"]), int(split["ns"])
        U, _ = MK.capture_uv(model, tokenizer, split, args.encoding, layers,
                             n_tokens=ne)
        Uw, _ = MK.capture_uv(model, tokenizer, dsplit, args.encoding, layers,
                              n_tokens=ne, pad_repeat=True)
        row: Dict[str, Any] = {"idx": i, "gold_ans": str(gold), "question": item["question"],
                               "ans_type": EV.answer_type(gold), "donor": donor_idx[i],
                               "n_e": ne, "n_s": nsi, "rel_err": {}}
        # ---- 参照与未压缩基线 ----
        pred, _ = MK.text_generate(model, tokenizer, split["full_text"],
                                   max_new_tokens=int(args.max_new_tokens))
        _record(row, "text_full", pred, gold)
        mem_native = MK.build_evidence_kv(model, tokenizer, split, args.encoding,
                                         pos_start=nsi, n_tokens=ne, layers=layers)
        mem_wrong = MK.build_evidence_kv(model, tokenizer, dsplit, args.encoding,
                                         pos_start=nsi, n_tokens=ne, layers=layers,
                                         pad_repeat=True)
        pred, _ = MK.midlayer_kv_generate(model, tokenizer, mem_native, split,
                                          max_new_tokens=int(args.max_new_tokens))
        _record(row, "anchor6", pred, gold)
        pred, _ = MK.midlayer_kv_generate(model, tokenizer, mem_wrong, split,
                                          max_new_tokens=int(args.max_new_tokens))
        _record(row, "anchor6_wrong", pred, gold)
        # ---- 恒等还原路径（正确性门槛）----
        mem_rt = MK.memory_kv_from_uv(model, U, nsi, layers)
        if i == 0:
            k0 = min(layers)
            dmax = float((mem_rt.k(k0).float() - mem_native.k(k0).float()).abs().max())
            print(f"[identity] native_rt vs anchor6 第 {k0} 层 K 的 max|Δ| = {dmax:.6f}"
                  f"（bf16 下应接近 0）", flush=True)
        pred, _ = MK.midlayer_kv_generate(model, tokenizer, mem_rt, split,
                                          max_new_tokens=int(args.max_new_tokens))
        _record(row, "native_rt", pred, gold)
        # ---- 各压缩档 ----
        for k, lr in fits.items():
            row["rel_err"][f"pca{k}"] = lr.rel_error(U)
            pred, _ = _gen_pca(model, tokenizer, lr, U, split, nsi, layers, args)
            _record(row, f"pca{k}", pred, gold)
            row["rel_err"][f"pca{k}_wrong"] = lr.rel_error(Uw)
            pred, _ = _gen_pca(model, tokenizer, lr, Uw, split, nsi, layers, args)
            _record(row, f"pca{k}_wrong", pred, gold)
        rows.append(row)
        if args.show_items and i < int(args.show_items):
            print(f"  [item {i}] gold={gold!r} ne={ne}", flush=True)
            for a in arms:
                mark = "✓" if row[f"{a}__ok"] else "✗"
                print(f"      {a:16s} {mark} {row[a]!r}", flush=True)
        if args.progress_every and (i + 1) % int(args.progress_every) == 0:
            el = time.time() - t0
            part = " ".join(f"{a}={ED.arm_acc(rows, a):.3f}" for a in arms)
            print(f"[progress] {i + 1}/{len(data)} {part} elapsed={el:.0f}s "
                  f"eta={el / (i + 1) * (len(data) - i - 1):.0f}s", flush=True)

    summary = summarize(args, rows, arms, dims, basis)
    print_table(summary, arms, dims)
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


def _gen_pca(model, tokenizer, lr, U, split, nsi, layers, args):
    Uh = lr.decode(lr.encode(U))
    mem = MK.memory_kv_from_uv(model, Uh, nsi, layers)
    return MK.midlayer_kv_generate(model, tokenizer, mem, split,
                                   max_new_tokens=int(args.max_new_tokens))


def _record(row: Dict[str, Any], arm: str, pred: str, gold: Any) -> None:
    row[arm] = str(pred)
    row[f"{arm}__ok"] = bool(T._norm_ans(pred) == T._norm_ans(str(gold)))


# ---------------------------------------------------------------------
def summarize(args, rows, arms, dims, basis: LR.LowRankKV) -> Dict[str, Any]:
    nn = max(1, len(rows))
    acc = {a: sum(int(r[f"{a}__ok"]) for r in rows) / nn for a in arms}
    text = acc.get("text_full")
    s: Dict[str, Any] = {"n": len(rows), "acc": acc, "dims": dims,
                         "native_dim": basis.native_dim,
                         "basis": basis.describe(), "data": str(args.data),
                         "layers": f"{basis.layout['layers'][0]}-{basis.layout['layers'][-1]}",
                         "encoding": args.encoding}
    s["paired"] = [ED.paired([bool(r[f"{a}__ok"]) for r in rows],
                             [bool(r[f"anchor6__ok"]) for r in rows], a, "anchor6")
                   for a in arms if a not in ("text_full", "anchor6", "anchor6_wrong")]
    if text:
        s["relative_to_text"] = {k: v / text for k, v in acc.items()}
        base = acc.get("anchor6")
        if base:
            s["retention_vs_uncompressed"] = {k: v / base for k, v in acc.items()}
    s["storage"] = {}
    for k in dims:
        s["storage"][f"pca{k}"] = {
            "bytes_per_token": int(k * 2),
            "native_bytes_per_token": int(basis.native_dim * 2),
            "ratio": round(basis.native_dim / max(1, k), 1)}
    dm = {}
    for k in dims:
        dm[f"pca{k}"] = {"correct": acc[f"pca{k}"], "wrong": acc[f"pca{k}_wrong"],
                         "delta": acc[f"pca{k}"] - acc[f"pca{k}_wrong"]}
    dm["anchor6"] = {"correct": acc["anchor6"], "wrong": acc["anchor6_wrong"],
                     "delta": acc["anchor6"] - acc["anchor6_wrong"]}
    s["delta_memory"] = dm
    errs = [r.get("rel_err", {}) for r in rows]
    s["rel_err_mean"] = {f"pca{k}": (sum(e.get(f"pca{k}", 0.0) for e in errs) / nn)
                         for k in dims}
    by_type: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        t = str(r.get("ans_type") or "other")
        d = by_type.setdefault(t, {"n": 0, **{a: 0 for a in arms}})
        d["n"] += 1
        for a in arms:
            d[a] += int(bool(r[f"{a}__ok"]))
    for t, d in by_type.items():
        for a in arms:
            d[a] = d[a] / max(1, d["n"])
    s["by_ans_type"] = by_type
    s["verdict"] = verdict(s, dims)
    return s


def verdict(s: Dict[str, Any], dims: List[int]) -> Dict[str, Any]:
    """按"未压缩基线保留率"给决策提示（阈值只做决策，不代替配对检验）。"""
    out: Dict[str, Any] = {}
    base = s["acc"].get("anchor6")
    out["baseline_anchor6"] = base
    if base is None or base <= 0:
        out["text"] = "未压缩基线为 0，实验无意义"
        return out
    best, bestk = -1.0, None
    for k in dims:
        v = s["acc"].get(f"pca{k}", 0.0)
        dm = s["delta_memory"].get(f"pca{k}", {}).get("delta", 0.0)
        tag = f"pca{k}: EM={v:.3f} 保留 {100 * v / base:.1f}% Δ_memory={100 * dm:+.1f}"
        out[f"pca{k}"] = tag
        if v > best:
            best, bestk = v, k
    if bestk is None:
        return out
    keep = best / base
    if keep >= 0.85:
        out["code"] = "compressible"
        out["text"] = (f"{bestk} 维保留未压缩基线的 {100 * keep:.1f}%（EM {best:.3f}）"
                       f"-> 跨层 KV 的低秩假设成立，可进入学习式压缩")
    elif keep >= 0.60:
        out["code"] = "marginal"
        out["text"] = (f"{bestk} 维只保留 {100 * keep:.1f}% -> 部分可压，需看精确实体/多跳"
                       f"是否集中受损（见分层表）")
    else:
        out["code"] = "not_compressible"
        out["text"] = (f"{bestk} 维只保留 {100 * keep:.1f}% -> 简单的全层共享低秩不成立，"
                       f"需先调整跨层分组或表示方式（例如分层/分 K-V 组各一套基）")
    return out


def print_table(s: Dict[str, Any], arms: List[str], dims: List[int]) -> None:
    ent = s.get("by_ans_type", {}).get("entity", {})
    print("")
    print("| 方法 | 每 token 存储 | EM% | Entity EM% | Wrong EM% | Correct−Wrong "
          "| 相对明文 | 保留未压缩 | KV 相对误差 |")
    print("|---|---|---|---|---|---|---|---|---|")
    norm_w = s["native_dim"] * 2
    for a in arms:
        if a.endswith("_wrong"):
            continue
        wrong = s["acc"].get(f"{a}_wrong")
        dm = s["delta_memory"].get(a, {}).get("delta")
        rel = s.get("relative_to_text", {}).get(a)
        keep = s.get("retention_vs_uncompressed", {}).get(a)
        if a.startswith("pca"):
            k = int(a[3:])
            store = f"{k * 2 // 1024} KiB（{s['native_dim'] // k}×）"
            err = f"{100 * s['rel_err_mean'].get(a, 0.0):.2f}%"
        elif a in ("anchor6", "native_rt", "anchor6_wrong"):
            store = f"{norm_w // 1024} KiB（1×）"
            err = "0（恒等）"
        else:
            store, err = "原始 token", "—"
        print(f"| {a} | {store} | {100 * s['acc'][a]:.1f} | "
              f"{100 * ent.get(a, 0.0):.1f} | "
              f"{'—' if wrong is None else f'{100 * wrong:.1f}'} | "
              f"{'—' if dm is None else f'{100 * dm:+.1f}'} | "
              f"{'—' if rel is None else f'{100 * rel:.0f}%'} | "
              f"{'—' if keep is None else f'{100 * keep:.0f}%'} | {err} |")
    print("")
    print("| 配对比较（vs 未压缩 anchor6） | A% | B% | 差 | 只有A对 | 只有B对 | p |")
    print("|---|---|---|---|---|---|---|")
    for p in s["paired"]:
        print(f"| {p['a']} | {100 * p['acc_a']:.1f} | {100 * p['acc_b']:.1f} "
              f"| {100 * p['delta']:+.1f} | {p['only_a']} | {p['only_b']} | {p['p']:.4f} |")
    if s.get("by_ans_type"):
        specs = [a for a in arms if a != "anchor6_wrong"]
        print("")
        print("| 题型 | n | " + " | ".join(specs) + " |")
        print("|---" * (2 + len(specs)) + "|")
        for t in sorted(s["by_ans_type"]):
            d = s["by_ans_type"][t]
            print(f"| {t} | {d['n']} | " + " | ".join(f"{100 * d[a]:.1f}"
                                                      for a in specs) + " |")
    print("")
    v = s["verdict"]
    for k in sorted(v):
        if k == "text":
            continue
        print(f"[判据] {k}: {v[k]}", flush=True)
    print(f"[判据] {v.get('code', '?')}: {v.get('text', '')}", flush=True)
    print(json.dumps({k: round(x, 4) for k, x in s["acc"].items()}, ensure_ascii=False),
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
