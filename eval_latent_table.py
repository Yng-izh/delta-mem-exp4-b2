#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_latent_table.py — 四种方法在同一批数据、同一协议下的统一评测（结果表的一行）。

为什么要一个新脚本，而不是继续往训练脚本里塞评估
------------------------------------------------
结果表要求"同一冻结主模型、同一记忆输入、同一评测协议、同一 prompt 模板"。
把四组方法放进**同一个循环、同一次 `build_chat_parts`、同一次 generate**，差异只剩
"记忆以什么形式进入"：

| method | 记忆怎么进 prompt | 注入的 token 数 |
|---|---|---|
| `none`   | 不给任何记忆（下界：纯模型参数知识） | 0 |
| `text`   | system 段里写明文事实（BM25 取 top-k，k=0 表示全给） | 明文 token 数 |
| `oracle` | system 段里只放 evidence 原文（性能上界，不是可部署方法） | 明文 token 数 |
| `bridge` | 全部槽位的载荷过 Bridge → 软 token | K × m |
| `router` | Router 先选 1 个槽 → Bridge → 软 token | m |
| `fst`    | 两级寻址选 (事实, 版本) → Bridge → 软 token | m |

明文基线与潜层基线**走同一个 `_emit_latent` 代码路径**（明文只是 `soft=None`），
所以"潜层掉点"不可能用"两条路模板不一样"解释掉。

指标只保留论文要的五项：答案准确率、更新准确率、历史访问准确率、实际注入 token 数、查询耗时。
（后两项是潜层记忆的**唯一卖点**：不占上下文长度，但要多付一次编码前向。）

用法
----
    # 明文基线（上界）
    python eval_latent_table.py --method oracle --data synth/hotpot_dev.jsonl --limit 200
    # 明文 RAG
    python eval_latent_table.py --method text --data synth/hotpot_dev.jsonl --text-topk 5
    # 潜层：直接拼接 / 路由 / FST
    python eval_latent_table.py --method bridge --ckpt ckpt/a.pt --data ...
    python eval_latent_table.py --method router --ckpt ckpt/b.pt --data ...
    python eval_latent_table.py --method fst    --ckpt ckpt/fst.pt --data synth/fst_update_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import torch

import latent_memory as LM
import train_latent_memory as T

METHODS = ("none", "text", "oracle", "bridge", "router", "fst")


# =====================================================================
# 明文基线：BM25 检索（纯 numpy，不引外部依赖）
# =====================================================================
_WORD = re.compile(r"[a-z0-9]+")


def _toks(s):
    return _WORD.findall(str(s).lower())


def bm25_topk(units, query, k, k1=1.5, b=0.75):
    """返回按 BM25 分数降序的前 k 个单元（k<=0 表示全部，保持原顺序）。"""
    if k is None or k <= 0 or k >= len(units):
        return list(range(len(units)))
    q = _toks(query)
    docs = [_toks(u) for u in units]
    if not q or not docs:
        return list(range(min(k, len(units))))
    df = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    n = len(docs)
    avg = sum(len(d) for d in docs) / max(1, n)
    scores = []
    for d in docs:
        tf = {}
        for w in d:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df.get(w, 0) + 0.5) / (df.get(w, 0) + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / max(1e-6, avg)))
        scores.append((s, -len(d)))
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return order[:k]


def gold_slots(item, units):
    """黄金槽位（支持事实所在单元）。没有 evidence 就返回空集 —— 不猜。"""
    ev = {T._norm_ans(e) for e in (item.get("evidence") or [])}
    return {i for i, u in enumerate(units) if T._norm_ans(u) in ev}


def memory_block(units, idx):
    return "Memory:\n" + "\n".join(f"- {units[i]}" for i in idx) + "\n"


# =====================================================================
# 每条样本的四种读法
# =====================================================================
def n_tokens_of(tokenizer, text, dev):
    return int(LM._tokenize_no_special(tokenizer, text, dev).shape[1])


def read_one(method, item, ctx):
    """返回 (pred, info)，info 里有注入 token 数与各段耗时（毫秒）。"""
    model, tok, dev = ctx["model"], ctx["tokenizer"], ctx["dev"]
    T.reset_delta_state(model)      # 样本间隔离（在线状态是跨 forward 累积的）
    units = T.item_units(item)
    question = item["question"]
    sysp = ctx["system_prompt"]
    t0 = time.time()

    if method == "none":
        # 无记忆对照。**必须有**：HotpotQA 的答案大量存在于 Qwen 的参数记忆里，
        # 不给记忆也可能答对；没有这一行就无法判断"记忆有没有被真正读到"。
        pred, meta = LM._emit_latent(model, tok, None, question, sysp,
                                     max_new_tokens=ctx["max_new_tokens"],
                                     pad_token_id=ctx["pad_token_id"])
        info = {"mem_tokens": 0, "prompt_tokens": n_tokens_of(tok, question, dev),
                "pre_ms": 0.0, "total_ms": (time.time() - t0) * 1e3,
                "n_read_units": 0, "decode_mode": meta.get("decode_mode")}
        return pred, info

    if method in ("text", "oracle"):
        if method == "oracle":
            idx = [i for i, u in enumerate(units)
                   if T._norm_ans(u) in {T._norm_ans(e) for e in item.get("evidence") or []}]
            if not idx:                       # oracle 找不到目标事实 -> 退化为全给并记账
                idx = list(range(len(units)))
        else:
            idx = bm25_topk(units, question, ctx["text_topk"])
        sys_full = sysp + "\n\n" + memory_block(units, idx)
        t_pre = time.time()
        pred, meta = LM._emit_latent(model, tok, None, question, sys_full,
                                     max_new_tokens=ctx["max_new_tokens"],
                                     pad_token_id=ctx["pad_token_id"])
        t_gen = time.time()
        info = {"mem_tokens": n_tokens_of(tok, memory_block(units, idx), dev),
                "prompt_tokens": n_tokens_of(tok, sys_full, dev)
                + n_tokens_of(tok, question, dev),
                "pre_ms": (t_gen - t_pre) * 1e3, "total_ms": (time.time() - t0) * 1e3,
                "n_read_units": len(idx), "decode_mode": meta.get("decode_mode"),
                # 与潜层方法的"读对槽位"可直接比较的可部署指标：取回的 top-k 里
                # 有没有黄金事实。不设这一列的话，"文本 RAG 掉点"到底是因为
                # **没检索到**还是**读到了但答错**无法区分。
                "read_ok": bool(set(idx) & gold_slots(item, units))}
        return pred, info

    # ---- 潜层：先编码（所有潜层方法都要付这笔前向） ----
    if method == "fst":
        fi = T.build_fst_bank(ctx["encoder"], item, ctx["hist_max"], tag="eval",
                              grad=False)
        bank = fi["bank"]
        t_enc = time.time()
        pred, meta = LM.generate_latent_answer_fst(
            model=model, tokenizer=tok, bridge=ctx["bridge"], bank=bank,
            question=question, system_prompt=sysp,
            fst_router=ctx["fst_router"], max_new_tokens=ctx["max_new_tokens"],
            pad_token_id=ctx["pad_token_id"], layer=ctx["layer"], device=dev,
            query_time=item.get("as_of"), feat_cache=ctx["cache"])
        t_gen = time.time()
        info = {"mem_tokens": int(meta["n_soft"]), "prompt_tokens": int(meta["n_soft"]),
                "pre_ms": (t_enc - t0) * 1e3, "total_ms": (time.time() - t0) * 1e3,
                "n_read_units": 1,
                # 更新/历史口径的判定用得到（答案对 + 读对版本才算"更新准确率"）
                "group_ok": bool(meta["group_keys"][int(meta["group_picked"])] == fi["tgt_key"]),
                "ver_ok": bool(int(meta["ver_picked"]) == int(fi["tgt_ver"]) and fi["viable"]),
                "probe": item.get("probe") or "",
                "decode_mode": meta.get("decode_mode"),
                "read_ok": bool(meta["group_keys"][int(meta["group_picked"])] == fi["tgt_key"])}
        return pred, info

    slot_z, _cat, _ei = T.encode_item_slots(ctx["encoder"], item, tag="eval")
    slot_z = LM.stack_payloads(list(slot_z)).unsqueeze(0)     # [1,K,m,d]
    t_enc = time.time()
    router = ctx["router"]
    pred, meta = LM.generate_latent_answer(
        model=model, tokenizer=tok, bridge=ctx["bridge"], payloads=slot_z,
        question=question, system_prompt=sysp, max_new_tokens=ctx["max_new_tokens"],
        pad_token_id=ctx["pad_token_id"], router=router, layer=ctx["layer"],
        hops=max(1, int(item.get("hops", 1) or 1)))
    t_gen = time.time()
    _gs = gold_slots(item, units)
    _picked = [int(x) for x in (meta.get("routed") or [])]
    info = {"mem_tokens": int(meta.get("n_soft", 0)), "prompt_tokens": int(meta.get("n_soft", 0)),
            "pre_ms": (t_enc - t0) * 1e3, "total_ms": (time.time() - t0) * 1e3,
            "n_read_units": max(1, len(_picked)),
            "decode_mode": meta.get("decode_mode"),
            "read_ok": bool(set(_picked) & _gs) if _gs else None,
            "picked": _picked, "gold": sorted(_gs)}
    return pred, info


# =====================================================================
# 汇总
# =====================================================================
def summarize(rows, label):
    n = len(rows)
    if not n:
        return {"label": label, "n": 0}
    hit = sum(int(r["ok"]) for r in rows)
    groups = {}
    for r in rows:
        groups.setdefault(T._norm_ans(r["question"]), []).append(bool(r["ok"]))
    real = {k: v for k, v in groups.items() if len(v) >= 2 and len(set(v)) >= 2}
    pair_ok = sum(1 for v in real.values() if all(v))
    lat = [r["total_ms"] for r in rows]
    lat.sort()
    out = {
        "label": label, "n": n,
        "acc": hit / n,
        "pair_acc": (pair_ok / len(real)) if real else None,
        "pair_n": len(real),
        "mem_tokens": sum(r["mem_tokens"] for r in rows) / n,
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows) / n,
        "ms_mean": sum(lat) / n,
        "ms_p50": lat[n // 2],
        "pre_ms_mean": sum(r["pre_ms"] for r in rows) / n,
    }
    reads = [bool(r["read_ok"]) for r in rows if r.get("read_ok") is not None]
    out["read_acc"] = (sum(reads) / len(reads)) if reads else None
    # FST 专属：更新准确率（答案对且读对版本）、历史访问准确率（只看 history 探针）
    if any("probe" in r for r in rows):
        upd = [r for r in rows if r.get("probe") in ("current", "history") or "probe" in r]
        if upd:
            out["update_acc"] = sum(int(r["ok"] and r.get("ver_ok")) for r in upd) / len(upd)
        hist = [r for r in rows if r.get("probe") == "history"]
        if hist:
            out["hist_acc"] = sum(int(r["ok"]) for r in hist) / len(hist)
    return out


def fmt_line(s):
    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}"
    return (f"| {s['label']} | {s['n']} | {pct(s['acc'])} | {pct(s.get('read_acc'))}"
            f" | {pct(s.get('pair_acc'))}"
            f" | {pct(s.get('update_acc'))} | {pct(s.get('hist_acc'))}"
            f" | {s['mem_tokens']:.1f} | {s['prompt_tokens']:.1f}"
            f" | {s['ms_mean']:.0f} |")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="", help="潜层方法必需（bridge/router/fst）")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default=T.BASE)
    ap.add_argument("--adapter", default=T.ADAPTER)
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--device", default="")
    ap.add_argument("--smoke-model", action="store_true")
    ap.add_argument("--layer", type=int, default=LM.LATENT_LAYER)
    ap.add_argument("--max-unit-tokens", type=int, default=LM.LATENT_MAX_UNIT_TOKENS)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--text-topk", type=int, default=5, help="text 方法的检索条数（0=全给）")
    ap.add_argument("--hist-max", type=int, default=LM.LATENT_HIST_MAX)
    ap.add_argument("--system-prompt",
                    default="You are a helpful assistant. Answer with the shortest exact value.")
    ap.add_argument("--label", default="")
    ap.add_argument("--save-json", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    dev = torch.device(args.device if args.device else
                       ("cpu" if args.smoke_model else "cuda:0"))
    args.gpu = str(args.gpu)
    model, tok, dev, _hidden = T._load_backbone(args, dev)
    T.freeze_model(model)
    pad_id = getattr(tok, "eos_token_id", None) or getattr(tok, "pad_token_id", None) or 0

    ctx = {"model": model, "tokenizer": tok, "dev": dev,
           "system_prompt": args.system_prompt, "max_new_tokens": args.max_new_tokens,
           "pad_token_id": pad_id, "layer": args.layer,
           "text_topk": args.text_topk, "cache": {}}

    if args.method in ("bridge", "router", "fst"):
        if not args.ckpt:
            raise SystemExit(f"--method {args.method} 必须给 --ckpt（随机权重没有任何意义）")
        reader = LM.load_latent_reader(args.ckpt, model, tok, device=dev)
        ctx.update({"encoder": reader.encoder, "bridge": reader.bridge,
                    "router": reader.router, "fst_router": reader.fst_router,
                    "hist_max": args.hist_max})
        if args.method in ("router", "fst"):
            got = reader.router if args.method == "router" else reader.fst_router
            if got is None:
                raise SystemExit(f"checkpoint {args.ckpt} 里没有 "
                                 f"{'router' if args.method == 'router' else 'fst_router'} 权重")
        if args.method == "fst" and reader.router is not None:
            # FST 数据集上不要拿单级 router 混进来
            pass

    data = T.load_synth_jsonl(args.data, limit=args.limit, seed=args.seed)
    if not data:
        raise SystemExit(f"{args.data} 里没有可评测样本")
    label = args.label or f"{args.method}:{os.path.basename(args.data)}"

    rows = []
    for i, item in enumerate(data):
        pred, info = read_one(args.method, item, ctx)
        gold = item["answer"]
        if isinstance(gold, (list, tuple)):
            gold = " ".join(str(g) for g in gold)
        ok = T._norm_ans(pred) == T._norm_ans(gold)
        rows.append(dict(info, question=item["question"], gold=gold,
                         pred=str(pred), ok=bool(ok), idx=i))
        if args.verbose:
            print(f"  [{label}] {'✓' if ok else '✗'} q={item['question'][:50]!r} "
                  f"gold={gold!r} pred={str(pred)[:40]!r}", flush=True)

    s = summarize(rows, label)
    print("| 方法 | n | acc% | 读对槽位% | pair% | update% | hist% | 记忆token | prompt总token | 查询ms |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print(fmt_line(s), flush=True)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in s.items()}, ensure_ascii=False), flush=True)
    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump({"summary": s, "config": vars(args), "rows": rows}, f,
                      ensure_ascii=False, indent=1)
        print(f"[out] -> {args.save_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
