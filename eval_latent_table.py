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

# ---------------------------------------------------------------------
# 指标与字段口径
# ---------------------------------------------------------------------
_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_ARTICLES = {"a", "an", "the"}
_HOP_TYPES = ("bridge", "comparison")


def squad_norm(s):
    """SQuAD 式规范化：小写、去标点、去冠词、压空白。"""
    s = str(s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def em_score(pred, gold):
    """HotpotQA 官方主指标 Answer EM（规范化后精确匹配）。

    与 acc 的区别必须两边都报：acc 是"包含匹配"（_norm_ans 子串口径），
    22.5% 的样本答案串在干扰单元里也出现过，只要模型复读上下文就能拿到 acc，
    但 EM 拿不到 —— 只用 acc 会系统性高估"真的答对了"。
    """
    return float(squad_norm(pred) == squad_norm(gold))


def f1_score(pred, gold):
    p, g = squad_norm(pred).split(), squad_norm(gold).split()
    if not p or not g:
        return float(p == g)
    cnt = {}
    for w in p:
        cnt[w] = cnt.get(w, 0) + 1
    same = 0
    for w in g:
        if cnt.get(w, 0) > 0:
            cnt[w] -= 1
            same += 1
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def item_hops(item):
    """与 train.item_hops 同一套规则：item["hops"] → meta["type"] → 1。"""
    v = item.get("hops")
    try:
        if v:
            return max(1, int(v))
    except (TypeError, ValueError):
        pass
    t = str((item.get("meta") or {}).get("type") or "").strip().lower()
    return 2 if t in _HOP_TYPES else 1


def n_ev_of(item):
    """金标证据句数：读出预算的口径锚点（>2 句时固定 2 跳物理上不可能全覆盖）。"""
    return len(item.get("evidence") or [])


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
        _gs2 = gold_slots(item, units)
        info = {"mem_tokens": n_tokens_of(tok, memory_block(units, idx), dev),
                "prompt_tokens": n_tokens_of(tok, sys_full, dev)
                + n_tokens_of(tok, question, dev),
                "pre_ms": (t_gen - t_pre) * 1e3, "total_ms": (time.time() - t0) * 1e3,
                "n_read_units": len(idx), "decode_mode": meta.get("decode_mode"),
                # 与潜层方法的"读对槽位"可直接比较的可部署指标：取回的 top-k 里
                # 有没有黄金事实。不设这一列的话，"文本 RAG 掉点"到底是因为
                # **没检索到**还是**读到了但答错**无法区分。
                # 多跳口径统一：整条证据链都被取回才算读对（与潜层列同一条规则）。
                "read_ok": ((bool(_gs2 <= set(idx)) if len(_gs2) > 1
                             else bool(set(idx) & _gs2)) if _gs2 else None),
                # 完整证据召回：整条证据链都被取回才算。必须与潜层同一条规则落盘，
                # 否则 summarize 的 fa（read_all is not None）在这一支永远为空，
                # 于是 text/oracle 的 full_recall 恒为 null —— 表现为"这列没算"。
                "read_all": bool(_gs2 <= set(idx)) if _gs2 else None,
                "picked": list(idx), "gold": sorted(_gs2)}
        return pred, info

    # ---- 潜层：先编码（所有潜层方法都要付这笔前向） ----
    if method == "fst":
        fi = T.build_fst_bank(ctx["encoder"], item, ctx["hist_max"], tag="eval",
                              grad=False, cache=ctx.get("unit_cache"))
        bank = fi["bank"]
        t_enc = time.time()
        pred, meta = LM.generate_latent_answer_fst(
            model=model, tokenizer=tok, bridge=ctx["bridge"], bank=bank,
            question=question, system_prompt=sysp,
            fst_router=ctx["fst_router"], max_new_tokens=ctx["max_new_tokens"],
            pad_token_id=ctx["pad_token_id"], layer=ctx["layer"], device=dev,
            query_time=item.get("as_of"), feat_cache=ctx["cache"],
            hops=max(1, int(ctx.get("read_groups") or 1)),
            ver_policy=str(ctx.get("ver_policy") or "current"))
        t_gen = time.time()
        _groups = [int(x) for x in (meta.get("groups_picked") or [meta["group_picked"]])]
        _g_ok = bool(meta["group_keys"][int(meta["group_picked"])] == fi["tgt_key"])
        info = {"mem_tokens": int(meta["n_soft"]), "prompt_tokens": int(meta["n_soft"]),
                "pre_ms": (t_enc - t0) * 1e3, "total_ms": (time.time() - t0) * 1e3,
                "n_read_units": len(_groups),
                # 更新/历史口径的判定用得到（答案对 + 读对版本才算"更新准确率"）
                "group_ok": _g_ok,
                "ver_ok": bool(int(meta["ver_picked"]) == int(fi["tgt_ver"]) and fi["viable"]),
                "probe": item.get("probe") or "",
                "decode_mode": meta.get("decode_mode"),
                "read_ok": _g_ok}
        return pred, info

    slot_z, _cat, _ei = T.encode_item_slots(ctx["encoder"], item, tag="eval",
                                            cache=ctx.get("unit_cache"))
    # 键-载荷解耦：寻址键用**单元级池化键**，读出载荷用**块级保真**版本。
    # 旧代码在评测时用 stack_payloads 的块轴均值同时当键和载荷，把
    # --max-unit-tokens 32 切出来的多块长句内容直接抹平（训练侧是多块 cat）。
    slot_reads = LM.pack_slot_reads(list(slot_z))
    slot_z = LM.pool_slot_keys(list(slot_z)).unsqueeze(0)     # [1,K,m,d]
    t_enc = time.time()
    router = ctx["router"]
    pred, meta = LM.generate_latent_answer(
        model=model, tokenizer=tok, bridge=ctx["bridge"], payloads=slot_z,
        question=question, system_prompt=sysp, max_new_tokens=ctx["max_new_tokens"],
        pad_token_id=ctx["pad_token_id"], router=router, layer=ctx["layer"],
        hops=item_hops(item), reads=slot_reads)
    t_gen = time.time()
    _gs = gold_slots(item, units)
    _picked = [int(x) for x in (meta.get("routed") or [])]
    needs_full = len(_gs) > 1          # 多跳：证据链是**一组**槽位，不是随便一条
    info = {"mem_tokens": int(meta.get("n_soft", 0)), "prompt_tokens": int(meta.get("n_soft", 0)),
            "pre_ms": (t_enc - t0) * 1e3, "total_ms": (time.time() - t0) * 1e3,
            "n_read_units": max(1, len(_picked)),
            "decode_mode": meta.get("decode_mode"),
            # 多跳样本：只有**整条证据链**都被读到才算"读对"。
            # 旧口径"选中任意一条黄金事实就算对"会把"读了半条链"算成成功，
            # 于是 read_acc 虚高、与 acc 的落差被误读成"读出端有问题"。
            "read_ok": (bool(_gs <= set(_picked)) if needs_full
                        else (bool(set(_picked) & _gs) if _gs else None)),
            "read_all": bool(_gs <= set(_picked)) if _gs else None,
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
    # ---- HotpotQA 官方口径：EM / F1（acc 是包含匹配，必须与它们同时报） ----
    out["em"] = sum(float(r.get("em") or 0.0) for r in rows) / n
    out["f1"] = sum(float(r.get("f1") or 0.0) for r in rows) / n
    # ---- 完整证据召回：两种口径都落盘 ----
    # 主表看 full_recall_leq2：读出预算固定 2 跳时，金标证据 >2 句的样本
    # 物理上不可能全覆盖；混在一起算会把"口径缺陷"读成"机制缺陷"。
    fa = [r for r in rows if r.get("read_all") is not None]
    out["full_recall"] = (sum(int(r["read_all"]) for r in fa) / len(fa)) if fa else None
    # n_leq2 是**数据集属性**（金标证据句数 ≤2 的样本数），不能从 fa 里数：
    # none 这种"不注入记忆"的方法没有 read_all，从 fa 数会得到 0，
    # 让"本集有多少样本在 2 跳预算内可全覆盖"这个分母凭空消失。
    out["n_leq2"] = sum(1 for r in rows if 0 < int(r.get("n_ev") or 0) <= 2)
    sub_ = [r for r in fa if int(r.get("n_ev") or 0) <= 2]
    out["full_recall_leq2"] = (sum(int(r["read_all"]) for r in sub_) / len(sub_)) if sub_ else None
    # ---- 两跳子集单报（本轮核心变量：混在总数里看不出多跳有没有生效） ----
    h2 = [r for r in rows if int(r.get("hops") or 1) > 1]
    if h2:
        out["n_h2"] = len(h2)
        out["acc_h2"] = sum(int(r["ok"]) for r in h2) / len(h2)
        h2r = [r for r in h2 if r.get("read_ok") is not None]
        if h2r:
            out["read_acc_h2"] = sum(int(r["read_ok"]) for r in h2r) / len(h2r)
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
    return (f"| {s['label']} | {s['n']} | {pct(s['acc'])} | {pct(s.get('em'))}"
            f" | {pct(s.get('f1'))} | {pct(s.get('read_acc'))}"
            f" | {pct(s.get('full_recall_leq2'))}"
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
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"),
                    help="主干精度（必须与训练一致，auto 会在 A100 上选 bf16）")
    ap.add_argument("--no-tf32", action="store_true", help="关闭 TF32")
    ap.add_argument("--smoke-model", action="store_true")
    ap.add_argument("--layer", type=int, default=LM.LATENT_LAYER)
    ap.add_argument("--max-unit-tokens", type=int, default=LM.LATENT_MAX_UNIT_TOKENS)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--text-topk", type=int, default=5, help="text 方法的检索条数（0=全给）")
    ap.add_argument("--hist-max", type=int, default=LM.LATENT_HIST_MAX)
    ap.add_argument("--ver-policy", default="current",
                    choices=("current", "time", "learned"),
                    help="FST 版本级读出规则（必须与训练时一致，否则是两套方法）")
    ap.add_argument("--read-groups", type=int, default=1,
                    help="FST 一次读出几个事实键（多跳/多事实题 >1）")
    ap.add_argument("--train-max-slots", type=int, default=0,
                    help="训练时用的 --max-slots，仅用于 train/eval 槽数一致性告警")
    ap.add_argument("--system-prompt",
                    default="You are a helpful assistant. Answer with the shortest exact value.")
    ap.add_argument("--label", default="")
    ap.add_argument("--save-json", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    dev = torch.device(args.device if args.device else
                       ("cpu" if args.smoke_model else "cuda:0"))
    args.gpu = str(args.gpu)
    T._setup_tf32(not getattr(args, "no_tf32", False))
    T.gpu_report(args)
    model, tok, dev, _hidden = T._load_backbone(args, dev)
    T.freeze_model(model)
    pad_id = getattr(tok, "eos_token_id", None) or getattr(tok, "pad_token_id", None) or 0

    ctx = {"model": model, "tokenizer": tok, "dev": dev,
           "system_prompt": args.system_prompt, "max_new_tokens": args.max_new_tokens,
           "pad_token_id": pad_id, "layer": args.layer,
           "text_topk": args.text_topk, "cache": {},
           # 文本 -> 载荷缓存：同一段上下文被多问共享时不重复编码（FST 每题 11s 的主因）
           "unit_cache": {}, "read_groups": args.read_groups,
           "ver_policy": args.ver_policy}

    if args.method in ("bridge", "router", "fst"):
        if not args.ckpt:
            raise SystemExit(f"--method {args.method} 必须给 --ckpt（随机权重没有任何意义）")
        reader = LM.load_latent_reader(args.ckpt, model, tok, device=dev)
        # **bridge 不许带 router**：旧代码把 ckpt 里的 router 无条件塞进 ctx，
        # 于是 --method bridge 实际跑的是"有路由的单槽读出"，两行结果同源，
        # "Bridge vs Router" 这一列就不成立了（这是结果表里最贵的一处静默错误）。
        use_router = reader.router if args.method == "router" else None
        ctx.update({"encoder": reader.encoder, "bridge": reader.bridge,
                    "router": use_router, "fst_router": reader.fst_router,
                    "hist_max": args.hist_max})
        if args.method in ("router", "fst"):
            got = reader.router if args.method == "router" else reader.fst_router
            if got is None:
                raise SystemExit(f"checkpoint {args.ckpt} 里没有 "
                                 f"{'router' if args.method == 'router' else 'fst_router'} 权重")

    data = T.load_synth_jsonl(args.data, limit=args.limit, seed=args.seed)
    if not data:
        raise SystemExit(f"{args.data} 里没有可评测样本")
    label = args.label or f"{args.method}:{os.path.basename(args.data)}"

    _ks = sorted(len(it.get("units") or []) for it in data)
    _n_gt2 = sum(1 for it in data if n_ev_of(it) > 2)
    print(f"[slots] 评测集 units 中位={_ks[len(_ks) // 2]} max={_ks[-1]}；"
          f"训练侧 --max-slots={args.train_max_slots or '不设上限'}；"
          f"金标证据句数>2 的样本 {_n_gt2}/{len(data)}"
          f"（{100.0 * _n_gt2 / max(1, len(data)):.1f}%，固定 2 跳读出的覆盖上限）",
          flush=True)
    if args.train_max_slots and _ks[-1] > args.train_max_slots:
        print(f"[warn] 训练槽数上限 {args.train_max_slots} < 评测集最大槽数 {_ks[-1]}："
              f"router 训练时从未见过这么多槽，读对槽位会被 chance 稀释；"
              f"训练侧建议 --max-slots 0", flush=True)

    rows = []
    for i, item in enumerate(data):
        pred, info = read_one(args.method, item, ctx)
        gold = item["answer"]
        if isinstance(gold, (list, tuple)):
            gold = " ".join(str(g) for g in gold)
        ok = T._norm_ans(pred) == T._norm_ans(gold)
        rows.append(dict(info, question=item["question"], gold=gold,
                         pred=str(pred), ok=bool(ok), idx=i,
                         em=em_score(pred, gold), f1=f1_score(pred, gold),
                         n_ev=n_ev_of(item), hops=item_hops(item)))
        if args.verbose:
            print(f"  [{label}] {'✓' if ok else '✗'} q={item['question'][:50]!r} "
                  f"gold={gold!r} pred={str(pred)[:40]!r}", flush=True)

    s = summarize(rows, label)
    print("| 方法 | n | acc% | EM% | F1% | 读对槽位% | 完整证据%(≤2句) | pair% "
          "| update% | hist% | 记忆token | prompt总token | 查询ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
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
