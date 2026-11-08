#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_midlayer_kv.py — 实验 D：恢复真实 KV 的明文读取能力（零训练）。

三个阶段，前一个不通过就不进下一个：

| 阶段 | 对比 | 回答什么 | 通过标准 |
|---|---|---|---|
| **D0-A** | `text_full` vs `prefix_kv` | 因果前缀拆成两次前向是否等价（原生 KV cache） | 首步 logits 与生成序列基本重合 |
| **D0-B** | `text_full` vs `kv:full_prefix:0` | 自定义注入（E 的 KV 插在 S 与 T 之间）能否复现明文 | 同上 |
| **D1** | `kv:{full_prefix,bare,anchor}:0` | 证据脱离原始上下文后还能不能被读 | 哪一个编码协议保得住 |
| **D2** | `kv:<enc>:{0,6,12,18}` | 冻结模型从哪一层起必须接触证据 | 最晚的有效注入起点 |

臂的写法：`none` / `text_full` / `prefix_kv` / `kv:<encoding>:<start>[:wrong]`。
`wrong` = 换成**别的题目**的证据（长度取最接近的），用来算
`Δ_memory = EM_correct − EM_wrong` —— 只涨 EM 而 Correct≈Wrong 不算读到事实。

三条统一口径（不这样做，四组数字之间不可比）：
1. 所有臂共用**同一个贪心解码函数**、同一个停止条件、同一个 `max_new_tokens`；
   明文基线必须**重新测量**，历史 52.5% 只作参照。
2. 证据只**编码一次**：同一题的 0/6/12/18 四档共用同一份捕获，`MemoryKV.subset()` 切片。
3. 证据段位置在在线序列里**留空档**（T 的 position_ids 从 `ns+ne` 起），不做压缩。

用法
----
    python eval_midlayer_kv.py --data $B/synth/hotpot_val500.jsonl --limit 15 \
        --mode equivalence --base ... --gpu 0
    python eval_midlayer_kv.py --data ... --limit 200 --mode main --save-json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import torch

import latent_memory as LM
import train_latent_memory as T
import train_reader as TR
import midlayer_kv as MK
import eval_latent_table as EV

SPECIAL_ARMS = ("none", "text_full", "prefix_kv")
DEFAULT_ARMS = ("none,text_full,prefix_kv,"
                "kv:full_prefix:0,kv:full_prefix:0:wrong,"
                "kv:bare:0,kv:bare:0:wrong,"
                "kv:anchor:0,kv:anchor:0:wrong,"
                "kv:full_prefix:6,kv:full_prefix:6:wrong,"
                "kv:full_prefix:12,kv:full_prefix:12:wrong,"
                "kv:full_prefix:18,kv:full_prefix:18:wrong")
EQUIV_ARMS = ("text_full,prefix_kv,kv:full_prefix:0,kv:full_prefix:0:wrong")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="实验 D：真实 KV 的明文读取能力恢复")
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--mode", choices=("equivalence", "main"), default="main",
                    help="equivalence=D0 数值/生成一致性检查；main=跑完整对照表")
    ap.add_argument("--arms", default="",
                    help=f"逗号分隔；空则 equivalence 用 {EQUIV_ARMS}，main 用默认 14 臂")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--max-mem-tokens", type=int, default=MK.MID_KV_MAX_MEM_TOKENS)
    ap.add_argument("--system-prompt",
                    default="You are a helpful assistant. Answer with the shortest exact value.")
    ap.add_argument("--base", default=T.BASE)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--partial-every", type=int, default=0)
    ap.add_argument("--save-json", default="")
    ap.add_argument("--dump-jsonl", default="")
    ap.add_argument("--show-items", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="只检查 S/E/T 切分是否严格可验证，不加载模型")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------
# 臂解析
# ---------------------------------------------------------------------
def parse_arm(spec: str) -> Dict[str, Any]:
    s = str(spec).strip()
    if s in SPECIAL_ARMS:
        return {"spec": s, "kind": s}
    parts = s.split(":")
    if len(parts) not in (3, 4) or parts[0] != "kv":
        raise SystemExit(f"臂写法非法：{spec!r}（应为 kv:<encoding>:<start>[:wrong]）")
    enc, start = parts[1], int(parts[2])
    wrong = len(parts) == 4 and parts[3] == "wrong"
    if len(parts) == 4 and parts[3] != "wrong":
        raise SystemExit(f"臂的后缀只能是 wrong：{spec!r}")
    if enc not in MK.MID_KV_ENCODINGS:
        raise SystemExit(f"未知编码协议 {enc!r}，可选 {MK.MID_KV_ENCODINGS}")
    return {"spec": s, "kind": "kv", "encoding": enc, "start": start, "wrong": wrong}


def arm_label(a: Dict[str, Any]) -> str:
    return a["spec"]


def correct_arm_of(a: Dict[str, Any]) -> Optional[str]:
    """`kv:*:*:wrong` -> 对应的 correct 臂名（用于算 Δ_memory）。"""
    if a["kind"] == "kv" and a["wrong"]:
        return f"kv:{a['encoding']}:{a['start']}"
    return None


# ---------------------------------------------------------------------
# 配对统计
# ---------------------------------------------------------------------
def mcnemar(b: int, c: int) -> float:
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    return float(min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)))


def paired(a: List[bool], b: List[bool], la: str, lb: str) -> Dict[str, Any]:
    if len(a) != len(b):
        raise ValueError(f"配对要求同批题：{la}={len(a)} {lb}={len(b)}")
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    return {"a": la, "b": lb, "acc_a": sum(a) / max(1, len(a)),
            "acc_b": sum(b) / max(1, len(b)),
            "delta": (sum(a) - sum(b)) / max(1, len(a)),
            "only_a": only_a, "only_b": only_b, "p": mcnemar(only_a, only_b)}


# ---------------------------------------------------------------------
# 一题的全部材料：切分 + 每份证据只编码一次
# ---------------------------------------------------------------------
class ItemCtx:
    def __init__(self, split: Dict[str, Any], kv: Dict[str, "MK.MemoryKV"]):
        self.split = split
        self.kv = kv

    def mem(self, encoding: str, start: int, wrong: bool) -> "MK.MemoryKV":
        base = self.kv["wrong" if wrong else "own"][encoding]
        return base if int(start) == base.start_layer else base.subset(int(start))


def build_item_ctx(model, tokenizer, item, donor, args, encodings: List[str],
                   layers_all: List[int]) -> ItemCtx:
    """一条样本：切 S/E/T，把每种编码协议的证据 KV 各捕获**一次**（全层）。"""
    split = MK.split_prompt_tokens(tokenizer, args.system_prompt,
                                   evidence_text(item), item["question"])
    rope = MK.rope_module(model)
    kv_own: Dict[str, MK.MemoryKV] = {}
    for enc in encodings:
        kv_own[enc] = MK.build_evidence_kv(
            model, tokenizer, split, enc, pos_start=split["ns"],
            n_tokens=int(split["ne"]), layers=layers_all, rope=rope)
    dsplit = MK.split_prompt_tokens(tokenizer, args.system_prompt,
                                    evidence_text(donor), donor["question"])
    kv_wrong: Dict[str, MK.MemoryKV] = {}
    for enc in encodings:
        # 错记忆必须占满**本题** E 的位置区间，位置也按本题算；只用别人的隐藏状态
        kv_wrong[enc] = MK.build_evidence_kv(
            model, tokenizer, dsplit, enc, pos_start=split["ns"],
            n_tokens=int(split["ne"]), layers=layers_all, rope=rope, pad_repeat=True)
    return ItemCtx(split, {"own": kv_own, "wrong": kv_wrong})


def evidence_text(item) -> str:
    idx = TR.gold_units(item)
    units = T.item_units(item)
    return "\n".join("- " + str(units[int(i)]) for i in idx)


# ---------------------------------------------------------------------
# 单题读取
# ---------------------------------------------------------------------
def read_arm(model, tokenizer, a: Dict[str, Any], ctx: ItemCtx, question: str, args,
             return_logits: bool = False):
    if a["kind"] == "none":
        # 不给记忆：prompt 里连证据段都没有，只有 system + 问题
        text = LM.build_chat_parts(tokenizer, args.system_prompt, question)[0]
        pred, meta = MK.text_generate(model, tokenizer, text,
                                      max_new_tokens=int(args.max_new_tokens))
        meta["arm_kind"] = "none"
        return (pred, meta, None) if return_logits else (pred, meta)
    if a["kind"] == "text_full":
        return MK.text_generate(model, tokenizer, ctx.split["full_text"],
                                max_new_tokens=int(args.max_new_tokens),
                                return_logits=return_logits)
    if a["kind"] == "prefix_kv":
        return MK.prefix_kv_generate(model, tokenizer, ctx.split,
                                     max_new_tokens=int(args.max_new_tokens),
                                     return_logits=return_logits)
    mem = ctx.mem(a["encoding"], a["start"], a["wrong"])
    return MK.midlayer_kv_generate(model, tokenizer, mem, ctx.split,
                                   max_new_tokens=int(args.max_new_tokens),
                                   return_logits=return_logits)


def arm_acc(rows: List[Dict[str, Any]], spec: str) -> float:
    return sum(int(r[f"{spec}__ok"]) for r in rows) / max(1, len(rows))


# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    specs = str(args.arms or (EQUIV_ARMS if args.mode == "equivalence" else DEFAULT_ARMS))
    arms = [parse_arm(s) for s in specs.split(",") if s.strip()]
    if not any(a["kind"] == "text_full" for a in arms):
        raise SystemExit("必须包含 text_full：明文基线要在统一解码协议下重新测量")
    encodings = sorted({a["encoding"] for a in arms if a["kind"] == "kv"})

    data = T.load_synth_jsonl(args.data, limit=int(args.limit or 0), seed=int(args.seed))
    if not data:
        raise SystemExit(f"{args.data} 里没有样本")

    if args.dry_run:
        return dry_run(args, data, arms, encodings)

    ns = argparse.Namespace(smoke_model=False, layer=0, base=args.base, dtype=args.dtype,
                            gpu=args.gpu, no_adapter=True, adapter="",
                            grad_checkpointing=False)
    dev = torch.device(f"cuda:{args.gpu}")
    T._setup_tf32(not args.no_tf32)
    model, tokenizer, dev, hidden = T._load_backbone(ns, dev)
    T.freeze_model(model)
    n_layers = len(MK.decoder_layers(model))
    layers_all = list(range(n_layers))
    print(f"[model] hidden={hidden} layers={n_layers} "
          f"mode={args.mode} arms={[a['spec'] for a in arms]}", flush=True)
    print(MK.describe(), flush=True)

    donor_idx = pick_donors(data, tokenizer, args)
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        try:
            ctx = build_item_ctx(model, tokenizer, item, data[donor_idx[i]], args,
                                 encodings, layers_all)
        except MK.SplitError as e:
            print(f"[skip] idx={i} S/E/T 无法严格切分：{e}", flush=True)
            continue
        gold = item["answer"]
        if isinstance(gold, (list, tuple)):
            gold = " ".join(str(g) for g in gold)
        row: Dict[str, Any] = {"idx": i, "gold_ans": str(gold),
                               "question": item["question"],
                               "ans_type": EV.answer_type(gold),
                               "split_method": ctx.split["split_method"],
                               "n_s": ctx.split["ns"], "n_e": ctx.split["ne"],
                               "n_t": ctx.split["nt"], "donor": int(donor_idx[i]),
                               "first_logits": {}}
        for a in arms:
            pred, meta, fl = read_arm(model, tokenizer, a, ctx, item["question"], args,
                                      return_logits=True)
            row[a["spec"]] = str(pred)
            row[f"{a['spec']}__ok"] = bool(T._norm_ans(pred) == T._norm_ans(str(gold)))
            if fl is not None:
                row["first_logits"][a["spec"]] = fl.detach().float().cpu()
        rows.append(row)
        if args.show_items and i < int(args.show_items):
            print(f"  [item {i}] gold={gold!r} split={ctx.split['split_method']} "
                  f"ns={ctx.split['ns']} ne={ctx.split['ne']} nt={ctx.split['nt']}", flush=True)
            for a in arms:
                spec = a["spec"]
                mark = "✓" if row[f"{spec}__ok"] else "✗"
                print(f"      {spec:26s} {mark} {row[spec]!r}", flush=True)
        if args.progress_every and (i + 1) % int(args.progress_every) == 0:
            el = time.time() - t0
            part = " ".join(f"{a['spec']}={arm_acc(rows, a['spec']):.3f}" for a in arms)
            print(f"[progress] {i + 1}/{len(data)} {part} "
                  f"elapsed={el:.0f}s eta={el / (i + 1) * (len(data) - i - 1):.0f}s", flush=True)
        if args.partial_every and (i + 1) % int(args.partial_every) == 0 and (i + 1) < len(data):
            _dump(args.save_json, args, rows, arms, partial=True)

    if args.mode == "equivalence":
        report_equivalence(rows, arms)
    summary = summarize(args, rows, arms)
    print_table(summary, arms)
    if args.dump_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_jsonl)) or ".", exist_ok=True)
        with open(args.dump_jsonl, "w", encoding="utf-8") as f:
            for r in rows:
                rr = {k: v for k, v in r.items() if k != "first_logits"}
                f.write(json.dumps(rr, ensure_ascii=False) + "\n")
        print(f"[out] 逐题明细 -> {args.dump_jsonl}", flush=True)
    _dump(args.save_json, args, rows, arms, partial=False)
    return 0


def dry_run(args, data, arms, encodings) -> int:
    """不加载模型也要验证的事：S/E/T 切分严格可验证、证据段非空。"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    n_ok = n_bad = 0
    methods: Dict[str, int] = {}
    for i, item in enumerate(data):
        ev = evidence_text(item)
        if not ev.strip():
            print(f"[dry] idx={i} 无金标证据，跳过", flush=True)
            continue
        try:
            sp = MK.split_prompt_tokens(tokenizer, args.system_prompt, ev, item["question"])
        except MK.SplitError as e:
            n_bad += 1
            print(f"[dry] idx={i} 切分失败：{e}", flush=True)
            continue
        n_ok += 1
        methods[sp["split_method"]] = methods.get(sp["split_method"], 0) + 1
        if i < 3:
            print(f"[dry] idx={i} method={sp['split_method']} "
                  f"ns={sp['ns']} ne={sp['ne']} nt={sp['nt']} "
                  f"len_full={len(sp['ids_full'])} 一致性={sp['ids_S'] + sp['ids_E'] + sp['ids_T'] == sp['ids_full']}",
                  flush=True)
    print(f"[dry] 可切分 {n_ok} / 失败 {n_bad}  切法分布={methods} "
          f"arms={[a['spec'] for a in arms]} encodings={encodings}", flush=True)
    return 0 if n_bad == 0 else 1


def pick_donors(data, tokenizer, args) -> List[int]:
    """wrong 臂的供体：**证据 token 长度最接近**的另一条样本（长度差本身不该成为变量）。"""
    lens = []
    for item in data:
        try:
            lens.append(len(MK.split_prompt_tokens(tokenizer, args.system_prompt,
                                                   evidence_text(item),
                                                   item["question"])["ids_E"]))
        except MK.SplitError:
            lens.append(0)
    out = []
    n = len(data)
    for i in range(n):
        best, bestd = (i + 1) % n, None
        for j in range(n):
            if j == i:
                continue
            d = abs(lens[j] - lens[i])
            if bestd is None or d < bestd:
                best, bestd = j, d
                if d == 0:
                    break
        out.append(int(best))
    return out


def summarize(args, rows: List[Dict[str, Any]], arms: List[Dict[str, Any]]) -> Dict[str, Any]:
    nn = max(1, len(rows))
    acc = {a["spec"]: sum(int(r[f"{a['spec']}__ok"]) for r in rows) / nn for a in arms}
    s: Dict[str, Any] = {"n": len(rows), "mode": args.mode, "acc": acc,
                         "data": str(args.data), "arms": [a["spec"] for a in arms],
                         "skip_split": None}
    text = acc.get("text_full")
    # 主对照：每个 kv 臂 vs 明文；每个 correct 臂 vs 自己的 wrong
    pairs: List[Tuple[str, str]] = []
    for a in arms:
        if a["spec"] in SPECIAL_ARMS or text is None:
            continue
        pairs.append((a["spec"], "text_full"))
        c = correct_arm_of(a)
        if c and c in acc:
            pairs.append((a["spec"], c))
    s["paired"] = [paired([bool(r[f"{x}__ok"]) for r in rows],
                          [bool(r[f"{y}__ok"]) for r in rows], x, y)
                   for x, y in pairs]
    if text is not None:
        s["relative_to_text"] = {k: (v / text if text > 0 else None) for k, v in acc.items()}
    # Δ_memory：correct − wrong（只涨 EM 而 Correct≈Wrong 不算读到事实）
    dm: Dict[str, Any] = {}
    for a in arms:
        c = correct_arm_of(a)
        if c and c in acc:
            dm[c] = {"correct": acc[c], "wrong": acc[a["spec"]],
                     "delta": acc[c] - acc[a["spec"]]}
    s["delta_memory"] = dm
    by_type: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        t = str(r.get("ans_type") or "other")
        d = by_type.setdefault(t, {"n": 0, **{a["spec"]: 0 for a in arms}})
        d["n"] += 1
        for a in arms:
            d[a["spec"]] += int(bool(r[f"{a['spec']}__ok"]))
    for t, d in by_type.items():
        for a in arms:
            d[a["spec"]] = d[a["spec"]] / max(1, d["n"])
    s["by_ans_type"] = by_type
    s["split_methods"] = _count(rows, "split_method")
    s["verdict"] = verdict(s)
    return s


def _count(rows, key) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        k = str(r.get(key))
        out[k] = out.get(k, 0) + 1
    return out


def verdict(s: Dict[str, Any]) -> Dict[str, Any]:
    acc = s["acc"]
    text = acc.get("text_full")
    if text is None:
        return {"code": "no_text", "text": "缺 text_full，无法判读"}
    out: Dict[str, Any] = {"code": "ok", "text": ""}
    eq = acc.get("kv:full_prefix:0")
    if eq is not None:
        gap = text - eq
        if gap > 0.05:
            out["d0"] = (f"D0 未通过：full_prefix@L0 {eq:.3f} 仍比明文低 {gap * 100:.1f} 点，"
                         f"先修等价性，别解读 D1/D2")
        else:
            out["d0"] = f"D0 通过：full_prefix@L0 {eq:.3f} ≈ 明文 {text:.3f}（差 {gap * 100:+.1f} 点）"
    best = None
    for k, v in acc.items():
        if k.startswith("kv:") and not k.endswith(":wrong"):
            if best is None or v > acc[best]:
                best = k
    if best:
        out["best_kv"] = best
        out["retention"] = (acc[best] / text) if text else None
        out["text"] = (f"最佳 KV 臂 {best} = {acc[best]:.3f}，保留明文 "
                       f"{100 * (acc[best] / text if text else 0):.1f}%（门槛 45% "
                       f"{'达到' if acc[best] >= 0.45 else '未达到'}）")
    return out


def print_table(s: Dict[str, Any], arms: List[Dict[str, Any]]) -> None:
    print("")
    print(f"| 实验模式 | EM% | Entity EM% | Wrong EM% | Correct−Wrong | 相对明文 |")
    print("|---|---|---|---|---|---|")
    ent = s.get("by_ans_type", {}).get("entity", {})
    for a in arms:
        k = a["spec"]
        c = correct_arm_of(a)
        if a.get("wrong"):
            continue
        wrong = s["acc"].get(f"{k}:wrong")
        d = s.get("delta_memory", {}).get(k, {}).get("delta")
        rel = s.get("relative_to_text", {}).get(k)
        wrong_s = "—" if wrong is None else f"{100 * wrong:.1f}"
        delta_s = "—" if d is None else f"{100 * d:+.1f}"
        rel_s = "—" if rel is None else f"{100 * rel:.0f}%"
        print(f"| {k} | {100 * s['acc'][k]:.1f} | {(100 * ent.get(k, 0)):.1f} | "
              f"{wrong_s} | {delta_s} | {rel_s} |")
    print("")
    print("| 配对比较 | A% | B% | 差 | 只有A对 | 只有B对 | p |")
    print("|---|---|---|---|---|---|---|")
    for p in s["paired"]:
        print(f"| {p['a']} vs {p['b']} | {100 * p['acc_a']:.1f} | {100 * p['acc_b']:.1f} "
              f"| {100 * p['delta']:+.1f} | {p['only_a']} | {p['only_b']} | {p['p']:.4f} |")
    if s.get("by_ans_type"):
        keys = sorted(s["by_ans_type"])
        specs = [a["spec"] for a in arms if not a.get("wrong")]
        print("")
        print("| 题型 | n | " + " | ".join(specs) + " |")
        print("|---" * (2 + len(specs)) + "|")
        for t in keys:
            d = s["by_ans_type"][t]
            print(f"| {t} | {d['n']} | " + " | ".join(f"{100 * d[k]:.1f}" for k in specs) + " |")
    print("")
    print(f"[切分口径] {s.get('split_methods')}", flush=True)
    for k in ("d0", "text"):
        if s["verdict"].get(k):
            print(f"[判据:{k}] {s['verdict'][k]}", flush=True)
    print(json.dumps({k: round(v, 4) for k, v in s["acc"].items()}, ensure_ascii=False),
          flush=True)


def report_equivalence(rows: List[Dict[str, Any]], arms: List[Dict[str, Any]]) -> None:
    """D0：首步 logits / Top-1 / 生成序列逐题比对（对 `text_full`）。"""
    ref = "text_full"
    if not rows or ref not in rows[0]["first_logits"]:
        print("[eq] 缺 text_full 的首步 logits，无法比对", flush=True)
        return
    print("")
    print("| 臂 | 首步 logits max|Δ| | Top-1 一致 | 生成串一致 | 均分（EM） |")
    print("|---|---|---|---|---|")
    for a in arms:
        spec = a["spec"]
        if spec == ref or spec not in rows[0]["first_logits"]:
            continue
        dmax, top1, same, hits = 0.0, 0, 0, 0
        for r in rows:
            lr = r["first_logits"].get(ref)
            la = r["first_logits"].get(spec)
            if lr is None or la is None:
                continue
            dmax = max(dmax, float((lr - la).abs().max()))
            top1 += int(int(lr.argmax()) == int(la.argmax()))
            same += int(str(r[ref]) == str(r[spec]))
            hits += int(bool(r[f"{spec}__ok"]))
        n = max(1, len(rows))
        print(f"| {spec} | {dmax:.3f} | {top1}/{len(rows)} | {same}/{len(rows)} | "
              f"{hits / n:.3f} |")
    print("", flush=True)
    ok = True
    for a in arms:
        spec = a["spec"]
        if spec == ref or a.get("wrong") or spec not in rows[0]["first_logits"]:
            continue
        bad_top1 = sum(int(int(r["first_logits"][ref].argmax())
                           != int(r["first_logits"][spec].argmax())) for r in rows)
        if bad_top1 > max(1, int(0.1 * len(rows))):
            ok = False
            print(f"[eq] {spec} 首步 Top-1 有 {bad_top1}/{len(rows)} 不一致 -> 未通过", flush=True)
    print("[eq] " + ("EQ_PASS 等价性通过" if ok else "EQ_FAIL 等价性未通过"), flush=True)


def _dump(path: str, args, rows, arms, partial: bool) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    blob = {"summary": summarize(args, rows, arms), "config": vars(args),
            "rows": [{k: v for k, v in r.items() if k != "first_logits"} for r in rows],
            "partial": bool(partial), "n_done": len(rows)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=1)
    print(f"[out] {'中间' if partial else '完整'}结果 -> {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
