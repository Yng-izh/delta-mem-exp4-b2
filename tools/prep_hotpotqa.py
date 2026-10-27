#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/prep_hotpotqa.py — HotpotQA（distractor 配置）-> 本项目训练 jsonl。

为什么换掉自造数据（这是本轮实验主线的一部分）
------------------------------------------------
`synth/latent_exp2_*` 与 `synth/fst_update_*` 一共只有 18 个答案值、6 个关系、
200/192 条样本，实体名是 `gen_names` 造的音节串。在这样的数据上"把准确率从
17% 提到 40%"无法区分是**机制有效**还是**模型背下了 18 个值的先验**。HotpotQA
提供真实维基段落、真实问题、真实答案与**支持事实标注**（supporting_facts），
后者正是 `target_slot_index` 需要的槽位监督。

数据布局（与推理期 bank 一致：一条事实句子 = 一个槽位）
------------------------------------------------------
    units      = 全部黄金段落的句子 + N 个干扰段落的句子（顺序可控）
    memory     = units 拼成一串（teacher 明文输入 / 事实重建任务用）
    evidence   = 支持事实那几句的**原文**（`target_slot_index` 按它精确定位槽位）
    question / answer
    meta       = {qid, type, level, n_gold_paras, n_distractor_paras, order,
                  leak（答案是否也出现在非黄金单元里）, n_units}

三个必须显式记录的点
--------------------
1. **`answer` 可能出现在干扰段落里**（真实维基文本难免）。这不是数据缺陷，是 HotpotQA
   的固有性质；但必须**计数并落盘**（`meta.leak`），否则无法区分"读对了槽位"和
   "在上下文里搜到了答案串"。合成数据上我们修的是"生成器自己造出的泄漏"，这里不能修，
   只能分层报告。
2. **顺序是实验变量**。`--order shuffled` 用同一个固定种子打乱 units，保证"同一组事实、
   不同顺序"是可配对的对照（抗干扰实验的 order 轴）。
3. **`hops`** 由 `type` 推出：`bridge` → 2，`comparison` → 2（HotpotQA 全是多跳）。
   当前 `generate_latent_answer` 的 `hops` 参数只控制路由链长，训练侧仍按单次寻址处理；
   写进 meta 是为了将来接线时能对得上，不是假装已经支持。

用法
----
    # 训练集（8 个干扰段落，句子顺序保持原文）
    python tools/prep_hotpotqa.py --split train --n-train 20000 \
        --n-distractors 8 --order natural --out synth/hotpot_train.jsonl

    # 干扰轴消融：同一批黄金事实，干扰段落数 0/2/4/8 × 顺序 natural/shuffled
    python tools/prep_hotpotqa.py --split validation --n-train 500 \
        --sweep-out synth/hotpot_sweep --levels 0,2,4,8 --orders natural,shuffled
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# =====================================================================
# 纯函数部分（可单测，不依赖 datasets/网络）
# =====================================================================
def _norm(x):
    return " ".join(str(x).lower().split())


def paragraphs(rec):
    """HF hotpot_qa 的 context 是 {"title": [...], "sentences": [[...], ...]}。

    也接受已展开成 [{"title":..,"sentences":[..]}, ...] 的形态（parquet 手工读取时常见）。
    """
    ctx = rec["context"]
    if isinstance(ctx, dict):
        titles = list(ctx["title"])
        sents = [list(s) for s in ctx["sentences"]]
        return [{"title": t, "sentences": s} for t, s in zip(titles, sents)]
    out = []
    for p in ctx:
        out.append({"title": p["title"], "sentences": list(p["sentences"])})
    return out


def support_pairs(rec):
    """supporting_facts -> [(title, sent_idx)]，去重并保序。"""
    sf = rec["supporting_facts"]
    if isinstance(sf, dict):
        pairs = list(zip(sf["title"], sf["sent_id"]))
    else:
        pairs = [(p["title"], p["sent_id"]) for p in sf]
    seen, out = set(), []
    for t, i in pairs:
        k = (str(t), int(i))
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def build_item(rec, n_distractors=8, order="natural", rng=None, qid=None):
    """一条 HotpotQA 记录 -> 训练 item。order ∈ {natural, shuffled}。"""
    if order not in ("natural", "shuffled"):
        raise ValueError(f"order 只能是 natural/shuffled：{order!r}")
    paras = paragraphs(rec)
    title_to_i = {}
    for i, p in enumerate(paras):
        title_to_i.setdefault(p["title"], i)
    pairs = support_pairs(rec)
    gold_titles = []
    for t, _i in pairs:
        if t not in title_to_i:
            raise ValueError(f"supporting_facts 的标题不在 context 里：{t!r}")
        if t not in gold_titles:
            gold_titles.append(t)
    gold_idx = {title_to_i[t] for t in gold_titles}

    # 黄金句子（按 supporting_facts 的 (title, sent_idx) 精确定位）
    ev_sents, ev_set = [], set()
    for t, si in pairs:
        p = paras[title_to_i[t]]
        if si >= len(p["sentences"]):
            raise ValueError(f"sent_id 越界：{t}#{si}")
        s = str(p["sentences"][si]).strip()
        if s and s not in ev_set:
            ev_set.add(s)
            ev_sents.append(s)
    if not ev_sents:
        raise ValueError("没有可用的支持事实句子")

    # 槽位：黄金段落全部句子 + N 个干扰段落全部句子
    units = []
    for i, p in enumerate(paras):
        if i in gold_idx:
            units.extend(str(s).strip() for s in p["sentences"] if str(s).strip())
    dis_titles = [p["title"] for i, p in enumerate(paras) if i not in gold_idx]
    n_dis = min(int(n_distractors), len(dis_titles))
    for t in dis_titles[:n_dis]:
        units.extend(str(s).strip() for s in paras[title_to_i[t]]["sentences"]
                     if str(s).strip())

    if order == "shuffled":
        if rng is None:
            raise ValueError("order=shuffled 时必须传 rng（要可复现）")
        rng.shuffle(units)

    # 槽位定位自检：evidence 必须能在 units 里逐条找到原文，否则 loss_addr 会标错槽
    u_norm = {_norm(u) for u in units}
    missing = [e for e in ev_sents if _norm(e) not in u_norm]
    if missing:
        raise ValueError(f"evidence 不在 units 里（槽位定位会失败）：{missing[:2]}")

    ans = rec["answer"]
    if isinstance(ans, (list, tuple)):
        ans = " ".join(str(x) for x in ans)
    a_norm = _norm(ans)
    # `leak` 只统计**干扰段落**里出现答案。不能把"黄金段落的其它句子"算进来：
    # HotpotQA 的答案本来就取自黄金段落原文，实测含黄金段落的 95%+ 样本里答案都能
    # 在同段别的句子里找到 —— 那是数据集性质，不是泄漏。把两种混成一个数会让人
    # 误判"数据有缺陷"，而真正的可比信号是"再加干扰段落，答案是否仍到处都在"。
    dis_units = [u for t in dis_titles[:n_dis]
                 for u in (str(x).strip() for x in paras[title_to_i[t]]["sentences"])
                 if u]
    leak = bool(a_norm) and any(a_norm in _norm(u) for u in dis_units)
    qtype = str(rec.get("type") or "")
    return {
        "kind": "hotpot", "hops": 2 if qtype in ("bridge", "comparison") else 1,
        "memory": " ".join(units), "units": list(units),
        "question": str(rec["question"]).strip(), "answer": str(ans).strip(),
        "evidence": ev_sents,
        "meta": {"qid": qid if qid is not None else rec.get("id", ""),
                 "type": qtype, "level": str(rec.get("level") or ""),
                 "n_gold_paras": len(gold_idx), "n_distractor_paras": n_dis,
                 "order": order, "n_units": len(units),
                 "n_gold_units": len(units) - len(dis_units), "leak": leak},
    }


# =====================================================================
# IO
# =====================================================================
def load_rows(which, from_parquet=None, hf_name="hotpot_qa", config="distractor"):
    """读原始记录。优先本地 parquet（离线），否则走 HF datasets。"""
    if from_parquet:
        import pyarrow.parquet as pq
        rows = []
        for p in from_parquet:
            rows.extend(pq.read_table(p).to_pylist())
        return rows
    from datasets import load_dataset
    ds = load_dataset(hf_name, config, split=which)
    return list(ds)


def write_jsonl(items, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def summarize(items, tag):
    if not items:
        print(f"[{tag}] 0 条", file=sys.stderr)
        return
    n = len(items)
    leak = sum(int(it["meta"]["leak"]) for it in items)
    units = [it["meta"]["n_units"] for it in items]
    types = {}
    for it in items:
        types[it["meta"]["type"]] = types.get(it["meta"]["type"], 0) + 1
    print(f"[{tag}] n={n} units/min/med/max={min(units)}/"
          f"{sorted(units)[n // 2]}/{max(units)} "
          f"leak={leak}({leak / n:.1%}) types={types}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    ap.add_argument("--n-train", type=int, default=0, help="取前 N 条（0=全部）")
    ap.add_argument("--n-distractors", type=int, default=8,
                    help="保留几个**干扰段落**（HotpotQA distractor 配置共 8 个）")
    ap.add_argument("--order", default="natural", choices=["natural", "shuffled"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "synth", "hotpot_train.jsonl"))
    ap.add_argument("--sweep-out", default="",
                    help="给了就产出干扰轴消融集（每个 levels×orders 一个 jsonl）")
    ap.add_argument("--levels", default="0,2,4,8")
    ap.add_argument("--orders", default="natural,shuffled")
    ap.add_argument("--from-parquet", nargs="*", default=None,
                    help="离线：直接读 parquet/arrow 文件，不走 datasets")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_rows(args.split, args.from_parquet)
    if args.n_train:
        rows = rows[:args.n_train]

    levels = [int(x) for x in str(args.levels).split(",") if str(x).strip()]
    orders = [x.strip() for x in str(args.orders).split(",") if x.strip()]

    def convert_one(r, i, lv, od):
        try:
            return build_item(r, lv, od, random.Random(args.seed * 100003 + i),
                              qid=r.get("id", i))
        except ValueError as e:
            # 真实数据有 0.02% 的 supporting_facts 越界/标题缺失。**跳过并计数**，
            # 不静默 clamp —— clamp 会把 ev 指到别的句子上，寻址标签直接错。
            bad.append(str(e))
            return None

    bad = []
    if not args.sweep_out:
        items = [x for x in (convert_one(r, i, args.n_distractors, args.order)
                             for i, r in enumerate(rows)) if x]
        if bad:
            print(f"[warn] 跳过 {len(bad)}/{len(rows)} 条坏样本（{len(bad)/len(rows):.3%}）"
                  f"首例：{bad[0]}", file=sys.stderr)
            if len(bad) > 0.01 * len(rows):
                raise SystemExit(f"坏样本比例 {len(bad)/len(rows):.2%} 超过 1%，先查数据")
        write_jsonl(items, args.out)
        print(f"[out] {len(items)} 条 -> {args.out}")
        return 0

    # 干扰轴：同一批记录 × (levels × orders)，**同一条问题在所有配置里都出现**
    os.makedirs(args.sweep_out, exist_ok=True)
    for lv in levels:
        if lv >= len(paragraphs(rows[0])) - 1 and lv > 8:
            raise SystemExit(f"levels 过大：{lv}")
        for od in orders:
            items = [x for x in (convert_one(r, i, lv, od) for i, r in enumerate(rows)) if x]
            path = os.path.join(args.sweep_out, f"hotpot_{args.split}_dis{lv}_{od}.jsonl")
            summarize(items, f"dis{lv}/{od}")
            write_jsonl(items, path)
            print(f"[out] -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
