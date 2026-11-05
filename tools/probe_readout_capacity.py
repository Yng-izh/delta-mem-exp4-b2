#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/probe_readout_capacity.py — 实验 A：读出接口的**能力上限**（不训练）。

要回答的问题
------------
"把一段事实写成潜层记忆之后，冻结模型还能不能把它读回来？" 这句话里有两个可能
的瓶颈，过去一直混在一起：

  1. **接口/容量**：冻结 Qwen 只肯从"几十个向量"里读事实，这本身就有上限；
  2. **压缩损失**：我们的 Compressor 在这几十个向量里塞进去的信息比上限更少。

区分办法只有一个：**同样多的 token、同样的 prompt 骨架，换成原生词向量**。
  - 原生 BPE 在同等预算下也只有这么多 -> 上限就是接口，再优化编码器收益有限；
  - 原生 BPE 明显更高 -> 压缩确实丢了东西，才值得继续研究编码保真。

为什么不能直接用现成的 `eval_latent_table.py --method oracle_embed*`
----------------------------------------------------------------
它是"按块数公式估算预算"的独立进程，和潜层那一行**不在同一道题上对齐**：
  - 潜层每题的注入量随金标槽数变化，公式估算只能对齐到平均值；
  - `oracle_embed` 只注入裸词向量，**没有围栏**；潜层那行有 `\\n[j] ` 围栏。
于是两行数字之间的差里混着"预算不同"和"prompt 骨架不同"两种成分。
`KNOWN_ISSUES.md` 第 10 条记的就是这件事。

本工具的做法（逐题、逐槽位严格对齐）
--------------------------------
对每道题，先跑一次潜层读出拿到它的**实际**注入结构（`seg_ids` 给出每个槽位几个
内容向量、围栏几个 token），再按这个结构拼原生 BPE：

    latent      : fence_0 + [m 个软向量] + fence_1 + [m 个软向量] + ...
    raw_*       : fence_0 + [c_0 个真实词向量] + fence_1 + [c_1 个真实词向量] + ...

`c_j` 就是潜层第 j 槽的内容向量数，**逐槽相等**，围栏用同一个字符串、同一个位置。
匹配成功时两条序列的 token 数完全相同（代码里有硬断言，不一致直接报错，不静默
降级）。差别只剩"那些位置上的向量是什么"—— 这正是要测的东西。

六条读法
--------
| arm | 记忆怎么进 prompt | 预算 |
|---|---|---|
| `none`        | 不给记忆（下界：纯参数知识） | 0 |
| `text`        | 金标证据原文写进 system（参照上界） | 原文 token |
| `latent`      | 当前方法：`z_lex + α·Bridge(z_sem)` | 每槽 m |
| `raw_prefix`  | 金标证据原文的**前 c_j 个**真实词向量（保留原序） | 与 latent 逐槽相等 |
| `raw_salient` | 同等预算下按**显著性**挑词，再按原序拼回 | 与 latent 逐槽相等 |
| `raw_full`    | 整条证据的真实词向量全给（不截断） | 不压缩的上限 |

`raw_prefix` 是"傻截断"，天然会把句尾的核心新值切掉（英语事实句的重心常在尾部），
所以判断容量时**主看 `raw_salient`**；`raw_prefix` 留着是为了量化"截断位置"本身
值多少分，`raw_full` 用来回答"token 给够就够了吗"。

判定与分层
----------
主指标 EM / F1（HotpotQA 官方口径），并**按题型分层**：
  - `hops`：1 跳 / 2 跳（`item_hops`，与训练同一套规则）
  - `ans_type`：yes-no / date / number / entity（按金标答案表面形式分，启发式）
配对检验复用 `tools/cmp_readout.py` 的 McNemar 精确二项（同一批题逐题配对）。

用法
----
    python tools/probe_readout_capacity.py --ckpt ckpt/x.pt \
        --data synth/hotpot_val500.jsonl --limit 200 --save-json out.json
    # 只跑两条
    python tools/probe_readout_capacity.py --ckpt ... --arms latent,raw_salient
"""
from __future__ import annotations

import argparse
import json
import os

import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import torch

import latent_memory as LM
import train_latent_memory as T
import eval_latent_table as EV
import cmp_readout as CMP

ARMS = ("none", "text", "latent", "raw_prefix", "raw_salient", "raw_full")
RAW_ARMS = ("raw_prefix", "raw_salient")     # 与 latent **逐槽等预算**
NEED_CKPT = ("latent", "raw_prefix", "raw_salient", "raw_full")

# 题型分类**只允许有一个实现**：分层标准一旦在评测脚本和探针里各写一份，两边报的
# "日期题 EM" 就会在某个边界（`May 2019`、`3rd`）上悄悄分到不同桶，表格之间不可比。
answer_type = EV.answer_type


def _fence_text(j: int, style: str, isolate: bool) -> str:
    """第 j 槽的围栏文本 —— **必须与 `compose_routed_soft` 逐字相同**。

    两条读法的 prompt 骨架如果连围栏都不一样，"原生 vs 潜层"的差就分不清是载荷
    不同还是模板不同。这里刻意照抄那段分支逻辑（而不是抽象成公共函数），因为围栏
    样式是评测协议的一部分，改动应当在两处同时显式发生，而不是被一个默认参数带走。
    """
    if style == "verbose":
        return (f"\n[Fact {j + 1}]: " if not isolate
                else f"\n-- Fact {j + 1} (memory slot, isolated) --\n")
    return f"\n[{j + 1}] "


def per_slot_counts(segs: Sequence[int], n_slots: int) -> List[int]:
    """从 `seg_ids` 数出每个槽位各占几个 token（seg==j+1 属于第 j 槽，0 是围栏）。"""
    out = [0] * int(n_slots)
    for v in segs:
        v = int(v)
        if 1 <= v <= n_slots:
            out[v - 1] += 1
    return out


def _slot_ids(tok, item, i: int, dev) -> torch.Tensor:
    """金标槽 i 的原文 token id（`unit_encode_text`，压缩器实际看到的那段文本）。"""
    return LM._tokenize_no_special(tok, T.unit_encode_text(item, i), dev)[0]


def pick_ids(tok, ids: torch.Tensor, budget: int, arm: str, question: str) -> torch.Tensor:
    """在 budget 个 token 的预算内挑词，**保持原文顺序**。

    raw_prefix  : 句首截断（旧行为；会把句尾的核心新值切掉）
    raw_salient : 显著性挑词（大写/数字/长词/非停用词/问题里没出现过），再按原序排回
    raw_full    : 不挑，全给
    预算 >= 可用 token 数时一律全给（不能凭空造词）。
    """
    n = int(ids.numel())
    if arm == "raw_full" or n <= 0:
        return ids
    if int(budget) >= n:
        return ids
    if arm == "raw_prefix":
        return ids[:int(budget)]
    return EV._salient_pick(tok, ids, int(budget), question)


def fence_runs(segs: Sequence[int]) -> List[int]:
    """数出每段围栏（段号 0 的连续段）占几个 token。用来核对骨架是否一致。"""
    runs: List[int] = []
    cur = 0
    for v in segs:
        if int(v) == 0:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def build_raw_soft(model, tok, item, idx: Sequence[int], counts: Sequence[int],
                   dev, arm: str, style: str, isolate: bool,
                   dump: bool = False
                   ) -> Tuple[Optional[torch.Tensor], List[int], Dict[str, Any]]:
    """按 latent 的槽位结构拼一段**原生词向量**序列，返回 (soft[1,T,d], segs, info)。

    info 记每槽的围栏 token 数、原文可用 token 数、实际用掉几个：
      - 围栏数与 latent 不一致 => 骨架漂了，调用方必须报错；
      - 内容比预算少 => 原文本身不够长（合法），但要记账，否则"原生基线吃亏"
        会被读成"原生不如潜层"。
    dump=True 时额外记下**实际挑中的词**（`picks`）：2 题的 sanity 只能看这个，
    否则总分是 0 还是 1 根本看不出"挑词逻辑对不对"。
    """
    emb = model.get_input_embeddings()
    dt = next(model.parameters()).dtype
    pieces: List[torch.Tensor] = []
    segs: List[int] = []
    fences: List[int] = []
    used: List[int] = []
    avail: List[int] = []
    picks: List[List[str]] = []
    for j, (i, c) in enumerate(zip(idx, counts)):
        fence = emb(LM._tokenize_no_special(tok, _fence_text(j, style, isolate), dev))
        pieces.append(fence.to(dtype=dt))
        segs.extend([0] * int(fence.shape[1]))
        fences.append(int(fence.shape[1]))
        ids = _slot_ids(tok, item, int(i), dev)
        avail.append(int(ids.numel()))
        keep = pick_ids(tok, ids, int(c), arm, str(item.get("question") or ""))
        used.append(int(keep.numel()))
        if dump:
            picks.append([str(tok.decode([int(t)], skip_special_tokens=True))
                          for t in keep.tolist()])
        if int(keep.numel()) <= 0:                 # 空槽：宁可给 0 个也不塞占位符
            continue
        e = emb(keep.view(1, -1)).to(dtype=dt)
        pieces.append(e)
        segs.extend([j + 1] * int(e.shape[1]))
    info = {"fences": fences, "used": used, "avail": avail,
            "counts": [int(c) for c in counts],
            "short": [int(u) < int(c) for u, c in zip(used, counts)]}
    if dump:
        info["picks"] = picks
    if not pieces:
        return None, [], info
    return torch.cat(pieces, dim=1), segs, info


# ---------------------------------------------------------------------
# 单题：一次算出所有读法
# ---------------------------------------------------------------------
def read_item(item, ctx) -> Dict[str, Any]:
    model, tok, dev = ctx["model"], ctx["tokenizer"], ctx["dev"]
    units = T.item_units(item)
    question = item["question"]
    sysp = ctx["system_prompt"]
    gold = EV.gold_slots(item, units)
    res: Dict[str, Dict[str, Any]] = {}

    def _run(arm: str, soft, segs, inject: bool = True):
        T.reset_delta_state(model)
        t0 = time.time()
        if inject:
            pred, meta = LM._emit_latent(
                model, tok, soft, question, sysp,
                max_new_tokens=ctx["max_new_tokens"],
                pad_token_id=ctx["pad_token_id"],
                seg_ids=segs, isolate=ctx["isolate"],
                inject_mode=ctx["inject_mode"], fence_style=ctx["fence_style"],
                extra_meta={"has_payload": soft is not None, "probe_arm": arm})
        else:
            pred, meta = LM._emit_latent(
                model, tok, None, question, sysp,
                max_new_tokens=ctx["max_new_tokens"],
                pad_token_id=ctx["pad_token_id"],
                extra_meta={"has_payload": False, "probe_arm": arm})
        dt = (time.time() - t0) * 1e3
        n_soft = int(meta.get("n_soft") or 0)
        res[arm] = {"pred": str(pred), "n_soft": n_soft,
                    "n_content": meta.get("n_soft_content"),
                    "n_fence": meta.get("n_fence"),
                    "n_in": int(meta.get("n_in") or 0),
                    "decode_mode": meta.get("decode_mode"),
                    "slot_isolation": meta.get("slot_isolation"),
                    "ms": dt}
        return res[arm]

    # ---- 无记忆下界 ----
    if "none" in ctx["arms"]:
        _run("none", None, None, inject=False)

    # ---- 明文参照（system 段放金标原文） ----
    if "text" in ctx["arms"]:
        idx_t = [i for i, u in enumerate(units)
                 if T._norm_ans(u) in {T._norm_ans(e) for e in item.get("evidence") or []}]
        if not idx_t:
            idx_t = list(range(len(units)))
        sys_full = sysp + "\n\n" + EV.memory_block(units, idx_t)
        T.reset_delta_state(model)
        t0 = time.time()
        pred, meta = LM._emit_latent(model, tok, None, question, sys_full,
                                     max_new_tokens=ctx["max_new_tokens"],
                                     pad_token_id=ctx["pad_token_id"],
                                     extra_meta={"has_payload": False, "probe_arm": "text"})
        n_mem = int(LM._tokenize_no_special(tok, EV.memory_block(units, idx_t), dev).shape[1])
        res["text"] = {"pred": str(pred), "n_soft": n_mem, "n_content": n_mem,
                       "n_fence": 0, "n_in": int(meta.get("n_in") or 0),
                       "decode_mode": meta.get("decode_mode"), "slot_isolation": "off",
                       "ms": (time.time() - t0) * 1e3}

    if not (ctx["arms"] & set(NEED_CKPT)):
        return {"res": res, "gold": sorted(gold), "budget": {}}

    # ---- 潜层读出（所有 raw_* 的预算基准都来自这里）----
    if not gold:
        # 没有金标证据就没有"预算"这回事：raw_* 与 latent 全部记为空，不拿 slot 0 冒充。
        for a in ctx["arms"] & set(NEED_CKPT):
            res[a] = {"pred": "", "n_soft": 0, "n_content": 0, "n_fence": 0, "n_in": 0,
                      "ms": 0.0, "oracle_no_evidence": True}
        return {"res": res, "gold": [], "budget": {"no_evidence": True}}

    slot_z, _cat, _ei, _kf, lex_payloads = T.encode_item_slots_full(
        ctx["encoder"], item, tag="eval", cache=ctx.get("unit_cache"))
    slot_reads = LM.pack_slot_reads(list(slot_z))
    idx = [i for i in sorted(gold) if i < len(slot_reads)]
    if not idx:
        for a in ctx["arms"] & set(NEED_CKPT):
            res[a] = {"pred": "", "n_soft": 0, "n_content": 0, "n_fence": 0, "n_in": 0,
                      "ms": 0.0, "oracle_no_evidence": True}
        return {"res": res, "gold": sorted(gold), "budget": {"no_evidence": True}}

    z = [slot_reads[i] for i in idx]
    lx = ([lex_payloads[i] for i in idx if i < len(lex_payloads)]
          if lex_payloads is not None else None)
    soft_lat, segs_lat = LM.compose_routed_soft(
        model, tok, ctx["bridge"], z, dev=dev, lex_list=lx,
        isolate=ctx["isolate"], return_segs=True, fence_style=ctx["fence_style"])
    counts = per_slot_counts(segs_lat, len(idx))
    if "latent" in ctx["arms"]:
        _run("latent", soft_lat, segs_lat)
    budget = {"idx": [int(i) for i in idx],
              "counts": [int(c) for c in counts],
              "n_content": int(sum(counts)),
              "n_soft": int(soft_lat.shape[1]),
              # 这条金标证据原文有多长（不压缩要吃多少 token）—— 压缩率的唯一诚实口径
              "n_raw_avail": int(sum(int(_slot_ids(tok, item, int(i), dev).numel())
                                     for i in idx))}

    # ---- 原生 BPE 对照 ----
    lat_fences = fence_runs(segs_lat)
    for arm in RAW_ARMS:
        if arm not in ctx["arms"]:
            continue
        soft_r, segs_r, info_r = build_raw_soft(model, tok, item, idx, counts, dev, arm,
                                               ctx["style"], ctx["isolate"],
                                               dump=bool(ctx.get("dump_picks")))
        if soft_r is None:
            res[arm] = {"pred": "", "n_soft": 0, "n_content": 0, "n_fence": 0,
                        "n_in": 0, "ms": 0.0}
            continue
        # **骨架核对**：围栏的 token 数必须与 compose_routed_soft 插入的逐个相等。
        # 不一致说明围栏文本或槽位顺序已经漂了，这时两条读法的差里混进了"模板不同"，
        # 宁可直接崩掉也不许出一行看着正常的数。
        if info_r["fences"] != lat_fences[:len(info_r["fences"])]:
            raise RuntimeError(
                f"围栏骨架没对齐（{arm}）：raw={info_r['fences']} vs "
                f"latent={lat_fences}。`_fence_text` 必须与 `compose_routed_soft` 逐字相同")
        # 内容 token 只允许**少于**预算（原文本身不够长），不允许多于预算。
        if int(soft_r.shape[1]) > int(soft_lat.shape[1]):
            raise RuntimeError(
                f"原生基线超出预算（{arm}）：raw={int(soft_r.shape[1])} > "
                f"latent={int(soft_lat.shape[1])}（counts={counts} used={info_r['used']}）")
        if any(info_r["short"]):
            budget.setdefault("short_slots", 0)
            budget["short_slots"] = int(budget["short_slots"]) + sum(
                1 for x in info_r["short"] if x)
        _run(arm, soft_r, segs_r)
        res[arm]["short"] = bool(any(info_r["short"]))
        if "picks" in info_r:
            res[arm]["picks"] = info_r["picks"]
    if "raw_full" in ctx["arms"]:
        soft_f, segs_f, _info_f = build_raw_soft(model, tok, item, idx, [0] * len(idx), dev,
                                                "raw_full", ctx["style"], ctx["isolate"])
        if soft_f is not None:
            _run("raw_full", soft_f, segs_f)
    return {"res": res, "gold": sorted(gold), "budget": budget}


# ---------------------------------------------------------------------
# 汇总：分层 + 配对
# ---------------------------------------------------------------------
def _agg(rows: List[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    vals = [r["res"][arm] for r in rows if arm in r["res"]]
    n = len(vals)
    if not n:
        return {"n": 0}
    out = {"n": n,
           "acc": sum(1 for v in vals if v["ok"]) / n,
           "em": sum(v["em"] for v in vals) / n,
           "f1": sum(v["f1"] for v in vals) / n,
           "n_content": sum(float(v.get("n_content") or 0) for v in vals) / n,
           "n_soft": sum(float(v.get("n_soft") or 0) for v in vals) / n,
           "n_in": sum(float(v.get("n_in") or 0) for v in vals) / n,
           "ms": sum(float(v.get("ms") or 0) for v in vals) / n}
    return out


def stratify(rows: List[Dict[str, Any]], arm: str, key: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if arm not in r["res"]:
            continue
        groups.setdefault(str(r.get(key)), []).append(r["res"][arm])
    return {k: {"n": len(v),
                "em": sum(x["em"] for x in v) / len(v),
                "f1": sum(x["f1"] for x in v) / len(v)}
            for k, v in sorted(groups.items())}


def paired(rows: List[Dict[str, Any]], a: str, b: str) -> Dict[str, Any]:
    """a vs b 的逐题配对（b 相对 a）。只看"一边对一边错"的题。"""
    oa = [bool(r["res"][a]["ok"]) for r in rows if a in r["res"] and b in r["res"]]
    ob = [bool(r["res"][b]["ok"]) for r in rows if a in r["res"] and b in r["res"]]
    only_a = sum(1 for x, y in zip(oa, ob) if x and not y)
    only_b = sum(1 for x, y in zip(oa, ob) if y and not x)
    return {"n": len(oa), "only_a": only_a, "only_b": only_b,
            "p": CMP.mcnemar(only_a, only_b)}


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}"


def print_items(rows: List[Dict[str, Any]], arms: Sequence[str], n: int) -> None:
    """逐样本明细：预算、每臂的注入构成、预测、原生基线挑中的词。

    小样本 sanity 只能看这个 —— 2 道题的总分是 0 还是 1 完全说明不了管线对不对，
    但"每槽 4 个内容向量 / 围栏 1 个 token / 挑中的词里有 1968"这些是能核对的。
    """
    for r in rows[:int(n)]:
        b = r.get("budget") or {}
        print(f"\n--- [{r['idx']}] {r['question']}")
        print(f"    金标答案={r['gold_ans']!r}  题型={r['ans_type']}  跳数={r['hops']}  "
              f"证据句数={r['n_ev']}")
        if b.get("no_evidence"):
            print("    [无金标证据] 所有读法记为空")
            continue
        print(f"    金标槽位={b.get('idx')}  每槽内容向量={b.get('counts')}  "
              f"合计内容={b.get('n_content')}  注入总token={b.get('n_soft')}  "
              f"原文可用={b.get('n_raw_avail')}")
        for a in arms:
            v = r["res"].get(a)
            if not v:
                continue
            print(f"    {a:<12} {'✓' if v.get('ok') else '✗'}  "
                  f"内容={v.get('n_content')} 围栏={v.get('n_fence')} "
                  f"prompt={v.get('n_in')} {v.get('ms', 0):.0f}ms  "
                  f"pred={str(v.get('pred'))[:50]!r}")
            if v.get("picks"):
                for j, words in enumerate(v["picks"]):
                    print(f"       槽{j + 1}: {words}")


def print_report(rows: List[Dict[str, Any]], arms: Sequence[str]) -> Dict[str, Any]:
    summ = {a: _agg(rows, a) for a in arms if any(a in r["res"] for r in rows)}
    print("\n| 读法 | n | EM% | F1% | 内容token | 注入总token | prompt | ms |")
    print("|---|---|---|---|---|---|---|---|")
    for a, s in summ.items():
        print(f"| `{a}` | {s['n']} | {_pct(s['em'])} | {_pct(s['f1'])} "
              f"| {s['n_content']:.1f} | {s['n_soft']:.1f} | {s['n_in']:.1f} | {s['ms']:.0f} |")

    # 预算口径核对：等预算的两条读法内容 token 必须相等
    for a in RAW_ARMS:
        if a in summ and "latent" in summ:
            d = summ[a]["n_content"] - summ["latent"]["n_content"]
            print(f"[budget] {a} 内容 token {summ[a]['n_content']:.2f} "
                  f"vs latent {summ['latent']['n_content']:.2f}（差 {d:+.2f}）")
    _short = sum(1 for r in rows
                 if any((r["res"].get(a) or {}).get("short") for a in RAW_ARMS))
    if _short:
        print(f"[budget][warn] {_short}/{len(rows)} 题有槽位原文短于预算"
              f"（原生基线因此少拿 token，差异偏保守）")
    avail = [r["budget"].get("n_raw_avail") for r in rows if r.get("budget")]
    avail = [int(x) for x in avail if x]
    if avail:
        print(f"[budget] 金标证据原文平均 {sum(avail) / len(avail):.1f} token"
              f"（不压缩要吃这么多）")

    strat = {}
    for key, name in (("hops", "跳数"), ("ans_type", "题型")):
        print(f"\n### 按{name}分层的 EM%")
        keys = sorted({str(r.get(key)) for r in rows})
        print("| 读法 | " + " | ".join(keys) + " |")
        print("|---" * (len(keys) + 1) + "|")
        for a in summ:
            st = stratify(rows, a, key)
            strat.setdefault(a, {})[key] = st
            cells = []
            for k in keys:
                g = st.get(k)
                cells.append("n/a" if not g else f"{_pct(g['em'])} (n={g['n']})")
            print(f"| `{a}` | " + " | ".join(cells) + " |")

    pair = {}
    if "latent" in summ:
        print("\n### 与 latent 的逐题配对（McNemar 精确二项）")
        for a in summ:
            if a == "latent":
                continue
            p = paired(rows, "latent", a)
            pair[f"latent_vs_{a}"] = p
            print(f"- `{a}`：只 latent 对 {p['only_a']}，只 {a} 对 {p['only_b']}，"
                  f"p = {p['p']:.4f}")
        if "raw_salient" in summ:
            p = paired(rows, "raw_salient", "latent")
            pair["raw_salient_vs_latent"] = p
            print(f"- 反向（raw_salient 为基线）：只 raw_salient 对 {p['only_a']}，"
                  f"只 latent 对 {p['only_b']}，p = {p['p']:.4f}")
    return {"summary": summ, "stratified": strat, "paired": pair}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="实验 A：读出接口能力上限（原生 BPE vs 潜层）")
    ap.add_argument("--ckpt", default="", help="潜层读法必需")
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default=",".join(ARMS),
                    help=f"逗号分隔，可选 {','.join(ARMS)}")
    ap.add_argument("--base", default=T.BASE)
    ap.add_argument("--adapter", default=T.ADAPTER)
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--device", default="")
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--smoke-model", action="store_true")
    ap.add_argument("--layer", type=int, default=LM.LATENT_LAYER)
    ap.add_argument("--max-unit-tokens", type=int, default=LM.LATENT_MAX_UNIT_TOKENS)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--inject-mode", choices=("", "user", "system", "between"), default="")
    ap.add_argument("--fence-style", choices=("", "bracket", "verbose"), default="")
    ap.add_argument("--slot-isolation", type=int, default=LM.LATENT_SLOT_ISOLATION)
    ap.add_argument("--system-prompt",
                    default="You are a helpful assistant. Answer with the shortest exact value.")
    ap.add_argument("--show-items", type=int, default=0,
                    help="打印前 N 题的逐样本明细（预算/注入构成/预测/挑中的词）")
    ap.add_argument("--dump-picks", action="store_true",
                    help="逐样本记录原生基线**实际挑中的词**。小样本 sanity 必开："
                         "只看总分看不出挑词逻辑对不对")
    ap.add_argument("--save-json", default="")
    ap.add_argument("--partial-every", type=int, default=25)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    arms = tuple(a.strip() for a in str(args.arms).split(",") if a.strip())
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"未知读法 {bad}；可选 {ARMS}")
    if (set(arms) & set(NEED_CKPT)) and not args.ckpt:
        raise SystemExit("latent/raw_* 需要 --ckpt（随机权重没有意义）")

    dev = torch.device(args.device if args.device else
                       ("cpu" if args.smoke_model else "cuda:0"))
    T._setup_tf32(not getattr(args, "no_tf32", False))
    T.gpu_report(args)
    model, tok, dev, _hidden = T._load_backbone(args, dev)
    T.freeze_model(model)
    pad_id = getattr(tok, "eos_token_id", None) or getattr(tok, "pad_token_id", None) or 0

    ctx: Dict[str, Any] = {"model": model, "tokenizer": tok, "dev": dev,
                           "system_prompt": args.system_prompt,
                           "max_new_tokens": args.max_new_tokens,
                           "pad_token_id": pad_id, "arms": set(arms),
                           "unit_cache": {},
                           "dump_picks": bool(args.dump_picks),
                           "isolate": bool(int(args.slot_isolation))}

    if args.ckpt:
        reader = LM.load_latent_reader(args.ckpt, model, tok, device=dev)
        # 预算口径以 ckpt 为准（块数由 max_unit_tokens 决定，m 由 n_slots 决定）：
        # 用 CLI 默认值会让"每槽几个向量"与训练时不同，等预算对照就不成立。
        _mut = int(reader.config.get("max_unit_tokens", args.max_unit_tokens))
        if int(args.max_unit_tokens) != _mut:
            print(f"[warn] --max-unit-tokens={args.max_unit_tokens} 与 ckpt 的 {_mut} "
                  f"不一致，已按 ckpt 运行", flush=True)
        args.max_unit_tokens = _mut
        args.layer = int(reader.config.get("layer", args.layer))
        ctx.update({"encoder": reader.encoder, "bridge": reader.bridge})
        _inj = str(reader.config.get("inject_mode", "") or "")
        _fen = str(reader.config.get("fence_style", "") or "")
        if not _inj:
            _inj = "between"
            print("[inject][warn] ckpt 未记录 inject_mode：按旧行为 between 评测", flush=True)
        if not _fen:
            _fen = "verbose"
            print("[inject][warn] ckpt 未记录 fence_style：按旧行为 verbose 评测", flush=True)
        ctx["inject_mode"] = args.inject_mode or _inj
        ctx["fence_style"] = args.fence_style or _fen
    else:
        ctx["inject_mode"] = args.inject_mode or LM.LATENT_INJECT_MODE
        ctx["fence_style"] = args.fence_style or LM.LATENT_FENCE_STYLE
    _, ctx["style"] = LM._check_inject_cfg(ctx["inject_mode"], ctx["fence_style"])
    print(f"[inject] inject_mode={ctx['inject_mode']} fence_style={ctx['fence_style']} "
          f"(style={ctx['style']}) isolate={ctx['isolate']}", flush=True)
    print(f"[arms] {','.join(arms)}", flush=True)

    data = T.load_synth_jsonl(args.data, limit=args.limit, seed=args.seed)
    if not data:
        raise SystemExit(f"{args.data} 里没有可评测样本")

    rows: List[Dict[str, Any]] = []
    t_run = time.time()
    for i, item in enumerate(data):
        got = read_item(item, ctx)
        gold = item["answer"]
        if isinstance(gold, (list, tuple)):
            gold = " ".join(str(g) for g in gold)
        for arm, v in got["res"].items():
            v["ok"] = bool(T._norm_ans(v["pred"]) == T._norm_ans(gold))
            v["em"] = EV.em_score(v["pred"], gold)
            v["f1"] = EV.f1_score(v["pred"], gold)
        rows.append({"idx": i, "question": item["question"], "gold_ans": str(gold),
                     "hops": EV.item_hops(item), "ans_type": answer_type(gold),
                     "n_ev": EV.n_ev_of(item), "res": got["res"],
                     "gold_slots": got["gold"], "budget": got["budget"]})
        if args.verbose:
            for arm, v in got["res"].items():
                print(f"  [{i}] {arm:<12} {'✓' if v['ok'] else '✗'} "
                      f"pred={v['pred'][:40]!r}", flush=True)
        if args.progress_every > 0 and (i + 1) % args.progress_every == 0:
            el = time.time() - t_run
            parts = [f"{a}={100 * _agg(rows, a)['em']:.1f}%" for a in arms
                     if any(a in r["res"] for r in rows)]
            print(f"[progress] {i + 1}/{len(data)} EM: {' '.join(parts)} "
                  f"elapsed={el:.0f}s eta={el / (i + 1) * (len(data) - i - 1):.0f}s",
                  flush=True)
        if (args.save_json and args.partial_every > 0
                and (i + 1) % args.partial_every == 0 and (i + 1) < len(data)):
            os.makedirs(os.path.dirname(os.path.abspath(args.save_json)) or ".",
                        exist_ok=True)
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump({"summary": {a: _agg(rows, a) for a in arms},
                           "config": vars(args), "rows": rows,
                           "partial": True, "n_done": len(rows)},
                          f, ensure_ascii=False, indent=1)

    if args.show_items > 0:
        print_items(rows, arms, args.show_items)
    rep = print_report(rows, arms)
    blob = {"config": vars(args), "n": len(rows), "rows": rows, **rep}
    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False, indent=1)
        print(f"\n[保存] {args.save_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
