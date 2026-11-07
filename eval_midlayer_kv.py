#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_midlayer_kv.py — 实验 C（零训练）：中间层原生 KV 读出能不能读回事实。

这是方案里的**架构决策关口**，训练一步都不做，只比较六条路径：

| arm | 记忆怎么进主干 | 回答什么问题 |
|---|---|---|
| `none`            | 不给记忆 | 冻结模型参数知识的下界 |
| `oracle_text`     | 明文证据写进 system，走全部层 | 任务可解性上界（与潜层无关） |
| `upper_kv`        | 证据在**第 18~35 层**的真实 K/V | 只在高层注入记忆够不够答题 |
| `upper_kv_wrong`  | 同上，但换成**别的题**的证据 | `upper_kv` 的主反事实 |
| `h18_kv`          | 同一份 H¹⁸ 过各层自己的 `W_k/W_v` | 单层状态天然有多少可读性 |
| `h18_kv_wrong`    | 同上，换成别的题的证据 | `h18_kv` 的主反事实 |

`upper_kv` 与 `h18_kv` 在**第 18 层完全相同**（都用 H¹⁸），从第 19 层起分叉。
所以"能否用单层替代逐层"这件事，只由这两行之间的差回答，不含别的变量。

为什么反事实必须是 wrong 而不是 shuffle
--------------------------------------
打乱记忆 token 顺序后，第 18 层隐藏状态里的"这是哪条事实"未必消失（相邻语义还在），
所以 `full ≈ shuffle` 不能证明模型没用记忆。**问题对、记忆错**（`*_wrong`）才是硬证据：
`upper_kv ≫ upper_kv_wrong` 才说明模型真的在读这段记忆，而不是被"多了一段上下文"带好。

判据（本实验结束后据此决定是否投入实验二）
------------------------------------------
  - `upper_kv` 不高于 `none`            -> 高层注入这条接口本身不成立，
                                           训 Adapter 也救不回来，要换注入位置/保存形式；
  - `upper_kv` 高，`h18_kv` ≈ `upper_kv` -> **最理想**：单层状态可读，
                                           值得训残差 Adapter 并研究 `N -> M` 压缩；
  - `upper_kv` 高，`h18_kv` ≈ `none`     -> 接口有效但单层不够，
                                           先做 Adapter（有逐层 teacher 监督）或改存多层。

运行口径与前面各轮一致：同一批题、同一个 prompt 骨架、同一套答案归一化；
`none` / `h18_kv` / `upper_kv` 三条走**同一个生成函数**（`LMKV.midlayer_kv_generate`），
所以 prompt 与解码逐位相同，唯一差别就是 K/V 里有没有记忆 —— 这是本实验唯一可信的前提。

用法
----
    python eval_midlayer_kv.py --data $B/synth/hotpot_val500.jsonl --limit 20 --dry-run
    python eval_midlayer_kv.py --data ... --limit 200 --save-json out.json \
        --base $B/models/Qwen3-4B-Instruct-2507 --gpu 1
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
import train_reader as TR          # 复用 gold_units：金标证据的匹配口径只能有一份
import midlayer_kv as MK
import eval_latent_table as EV     # 复用 answer_type：题型分层口径只能有一份

ARMS = ("none", "oracle_text", "upper_kv", "upper_kv_wrong", "h18_kv", "h18_kv_wrong")
# 配对比较：前四对是各自的主反事实，后两对是架构判据（单层 vs 逐层、高层 vs 无记忆）
PAIRS = (("upper_kv", "upper_kv_wrong"), ("h18_kv", "h18_kv_wrong"),
         ("upper_kv", "none"), ("h18_kv", "none"),
         ("upper_kv", "h18_kv"), ("oracle_text", "upper_kv"))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="实验 C：中间层原生 KV 读出的零训练对照")
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--arms", default=",".join(ARMS),
                    help=f"逗号分隔，可选 {ARMS}")
    ap.add_argument("--start-layer", type=int, default=MK.MID_KV_START_LAYER,
                    help="记忆从第几层开始注入（H¹⁸ 指 hidden_states[18]）")
    ap.add_argument("--max-mem-tokens", type=int, default=MK.MID_KV_MAX_MEM_TOKENS)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--system-prompt",
                    default="You are a helpful assistant. Answer with the shortest exact value.")
    ap.add_argument("--base", default=T.BASE)
    ap.add_argument("--adapter", default=T.ADAPTER)
    ap.add_argument("--with-adapter", action="store_true",
                    help="挂 delta-mem adapter（默认不挂：本实验的对照对象是**冻结的纯 Qwen**）")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--partial-every", type=int, default=0,
                    help="每 N 题落一次中间结果（长跑必须能保住已跑完的题）")
    ap.add_argument("--save-json", default="")
    ap.add_argument("--dump-jsonl", default="", help="逐题明细（含每臂预测）")
    ap.add_argument("--show-items", type=int, default=0,
                    help="打印前 N 题的逐臂预测（小样本 sanity）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="只检查数据与金标证据，不加载模型")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------
# 配对检验（与 tools/cmp_readout.py 同一口径：McNemar 精确二项）
# ---------------------------------------------------------------------
def mcnemar(b: int, c: int) -> float:
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    return float(min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)))


def paired(a: List[bool], b: List[bool], la: str, lb: str) -> Dict[str, Any]:
    if len(a) != len(b):
        raise ValueError(f"配对比较要求同批题：{la}={len(a)} {lb}={len(b)}")
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    return {"a": la, "b": lb, "acc_a": sum(a) / max(1, len(a)),
            "acc_b": sum(b) / max(1, len(b)),
            "delta": (sum(a) - sum(b)) / max(1, len(a)),
            "only_a": only_a, "only_b": only_b, "p": mcnemar(only_a, only_b)}


# ---------------------------------------------------------------------
# 记忆构建
# ---------------------------------------------------------------------
def evidence_text(item, idx: List[int]) -> str:
    """金标证据单元的**原文**拼成一段（与 `oracle_text` 臂给的完全是同一批句子）。

    不加 `unit_encode_text` 的 `[anchor]` 前缀：那个前缀是给压缩器看"这句话说的是谁"的，
    而本实验的记忆要能与明文 oracle 逐句对齐，多一个原文里没有的标记就不对齐了。
    """
    units = T.item_units(item)
    return "\n".join(str(units[int(i)]) for i in idx)


class ItemMemory:
    """一条样本的两份记忆：自己的（correct）与下一题的（wrong）。

    每份都同时带 `upper`（逐层真实状态）与 `h18`（单层复用）两种来源，且**都从同一份
    捕获里投影出来** —— 这是"第 18 层那一份 K/V 逐位相同"这个前提的实现保证。
    """

    def __init__(self, own: Dict[str, MK.MemoryKV], wrong: Dict[str, MK.MemoryKV]):
        self.own = own
        self.wrong = wrong


def build_item_memory(model, tokenizer, own_text: str, wrong_text: str,
                      args) -> ItemMemory:
    cap_own = MK.capture_evidence_hidden(model, tokenizer, own_text,
                                         start_layer=int(args.start_layer),
                                         max_tokens=int(args.max_mem_tokens), upper=True)
    cap_wrong = MK.capture_evidence_hidden(model, tokenizer, wrong_text,
                                           start_layer=int(args.start_layer),
                                           max_tokens=int(args.max_mem_tokens), upper=True)
    # 两条 source 从**同一份捕获**里取，才能保证第 18 层那一份 K/V 逐位相同
    own_upper = MK.build_memory_kv(model, cap_own, "oracle_upper")
    own_h18 = MK.build_memory_kv(model, cap_own, "shared_h18")
    wrong_upper = MK.build_memory_kv(model, cap_wrong, "oracle_upper")
    wrong_h18 = MK.build_memory_kv(model, cap_wrong, "shared_h18")
    return ItemMemory(own={"upper": own_upper, "h18": own_h18},
                      wrong={"upper": wrong_upper, "h18": wrong_h18})


def pick_memory(mem: ItemMemory, arm: str) -> Optional[MK.MemoryKV]:
    if arm == "none":
        return None
    base, sep, _tail = arm.partition("_wrong")     # 注意：看的是分隔符，不是尾串
    src = "upper" if base == "upper_kv" else "h18"
    return (mem.wrong if sep else mem.own)[src]


# ---------------------------------------------------------------------
# 单题读取
# ---------------------------------------------------------------------
def read_arm(model, tokenizer, arm: str, item, mem: ItemMemory, args) -> Tuple[str, Dict]:
    if arm == "oracle_text":
        idx = TR.gold_units(item)
        units = T.item_units(item)
        block = ("Memory:\n" + "\n".join("- " + str(units[j]) for j in idx) + "\n"
                 if idx else "")
        pred, meta = LM._emit_latent(model, tokenizer, None, item["question"],
                                     args.system_prompt + "\n\n" + block,
                                     max_new_tokens=int(args.max_new_tokens))
        meta = dict(meta)
        meta["n_mem"] = len(idx)
        return str(pred), meta
    m = pick_memory(mem, arm)
    pred, meta = MK.midlayer_kv_generate(
        model, tokenizer, m, item["question"], args.system_prompt,
        max_new_tokens=int(args.max_new_tokens))
    return str(pred), meta


# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    arms = [a.strip() for a in str(args.arms).split(",") if a.strip()]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"未知 arm {bad}，可选 {ARMS}")
    if "oracle_text" not in arms and "none" not in arms:
        raise SystemExit("至少要保留一个参照臂（none 或 oracle_text），否则无法判断读没读到")

    data = T.load_synth_jsonl(args.data, limit=int(args.limit or 0), seed=int(args.seed))
    if not data:
        raise SystemExit(f"{args.data} 里没有样本")
    golds = [TR.gold_units(it) for it in data]
    n_no_ev = sum(1 for g in golds if not g)
    print(f"[data] {args.data} n={len(data)} 无金标证据={n_no_ev} arms={arms}", flush=True)
    if n_no_ev:
        print("[warn] 无金标证据的样本其记忆为空，等价于 none 臂；保留但要在结果里注明",
              flush=True)
    if args.dry_run:
        for i in range(min(3, len(data))):
            print(f"  [dry] idx={i} gold={data[i]['answer']!r} "
                  f"units={len(golds[i])} text={evidence_text(data[i], golds[i])[:90]!r}",
                  flush=True)
        print("[dry-run] OK：数据与金标证据可用，未加载模型", flush=True)
        return 0

    ns = argparse.Namespace(
        smoke_model=False, layer=int(args.start_layer), base=args.base,
        dtype=args.dtype, gpu=args.gpu, no_adapter=not args.with_adapter,
        adapter=args.adapter, grad_checkpointing=False)
    dev = torch.device(f"cuda:{args.gpu}")
    T._setup_tf32(not args.no_tf32)
    model, tokenizer, dev, hidden = T._load_backbone(ns, dev)
    T.freeze_model(model)
    print(f"[model] hidden={hidden} layers={len(MK.decoder_layers(model))} "
          f"start_layer={args.start_layer} "
          f"adapter={'on' if args.with_adapter else 'off（纯冻结 Qwen）'}", flush=True)
    print(MK.describe(), flush=True)

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    n = len(data)
    for i, item in enumerate(data):
        idx = golds[i]
        own_text = evidence_text(item, idx)
        wrong_text = evidence_text(data[(i + 1) % n], golds[(i + 1) % n])
        gold = item["answer"]
        if isinstance(gold, (list, tuple)):
            gold = " ".join(str(g) for g in gold)
        if not own_text.strip():
            # 没有证据的题建不了记忆：仍跑 none/oracle_text，其余臂按 none 处理并记标记
            mem = None
        else:
            mem = build_item_memory(model, tokenizer, own_text, wrong_text or own_text, args)
        row: Dict[str, Any] = {"idx": i, "gold_ans": str(gold), "question": item["question"],
                              "n_units": len(idx), "ans_type": EV.answer_type(gold)}
        for arm in arms:
            if mem is None and arm not in ("none", "oracle_text"):
                pred, meta = "(no-evidence)", {"n_mem": 0}
            else:
                pred, meta = read_arm(model, tokenizer, arm, item, mem, args)
            row[arm] = pred
            row[f"{arm}__ok"] = bool(T._norm_ans(pred) == T._norm_ans(str(gold)))
            if arm not in ("none", "oracle_text"):
                row[f"{arm}__n_mem"] = int(meta.get("n_mem") or 0)
        rows.append(row)
        if args.show_items and i < int(args.show_items):
            print(f"  [item {i}] gold={gold!r} q={item['question'][:60]!r}", flush=True)
            for arm in arms:
                print(f"      {arm:16s} {'✓' if row[f'{arm}__ok'] else '✗'} "
                      f"{row[arm]!r}", flush=True)
        if args.progress_every and (i + 1) % int(args.progress_every) == 0:
            el = time.time() - t0
            part = " ".join(f"{a}={sum(int(r[f'{a}__ok']) for r in rows) / len(rows):.3f}"
                            for a in arms)
            print(f"[progress] {i + 1}/{n} {part} elapsed={el:.0f}s "
                  f"eta={el / (i + 1) * (n - i - 1):.0f}s", flush=True)
        if args.partial_every and (i + 1) % int(args.partial_every) == 0 and (i + 1) < n:
            _dump_records(args.save_json, args, rows, partial=True)

    summary = summarize(args, rows, arms)
    print_table(summary, arms)
    if args.dump_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_jsonl)) or ".", exist_ok=True)
        with open(args.dump_jsonl, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[out] 逐题明细 -> {args.dump_jsonl}", flush=True)
    _dump_records(args.save_json, args, rows, partial=False)
    return 0


def summarize(args, rows: List[Dict[str, Any]], arms: List[str]) -> Dict[str, Any]:
    nn = max(1, len(rows))
    acc = {a: sum(int(r[f"{a}__ok"]) for r in rows) / nn for a in arms}
    s: Dict[str, Any] = {"n": nn, "acc": acc, "start_layer": int(args.start_layer),
                         "data": str(args.data), "arms": list(arms)}
    s["paired"] = [paired([bool(r[f"{a}__ok"]) for r in rows],
                          [bool(r[f"{b}__ok"]) for r in rows], a, b)
                   for a, b in PAIRS if a in arms and b in arms]
    # 题型分层：日期/数字/实体正是前面几轮掉得最狠的三类，必须分开看
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
    s["verdict"] = verdict(s, arms)
    return s


def verdict(s: Dict[str, Any], arms: List[str]) -> Dict[str, Any]:
    """把方案里的三条判据落成一句话（只是决策辅助，不是统计检验）。"""
    acc = s["acc"]
    need = [a for a in ("none", "upper_kv", "h18_kv") if a in acc]
    if len(need) < 3:
        return {"code": "incomplete", "text": "缺 none/upper_kv/h18_kv 之一，无法判断"}
    up, h18, base = acc["upper_kv"], acc["h18_kv"], acc["none"]
    up_gain, h18_gain = up - base, h18 - base
    # 按 (a, b) 精确取"主反事实"的 p 值：同一个 a 会出现在多对里（upper_kv vs wrong /
    # upper_kv vs none），只按 a 建字典会取到最后一对，判据就会拿错 p。
    pmap = {(p["a"], p["b"]): p for p in s["paired"]}
    up_p = pmap.get(("upper_kv", "upper_kv_wrong"), {}).get("p", 1.0)
    h18_p = pmap.get(("h18_kv", "h18_kv_wrong"), {}).get("p", 1.0)
    if up_gain < 0.05:
        code = "upper_kv_dead"
        text = ("高层注入本身没读出事实（upper_kv 相对 none 不足 +5 点）"
                "-> 训 Adapter 救不回来，要换注入位置或保存形式")
    elif h18_gain >= 0.05 and h18_p < 0.05:
        code = "single_layer_readable"
        text = (f"单层状态可读（h18_kv {h18:.3f} 明显高于 none {base:.3f}，"
                f"且高于 wrong，p={h18_p:.4f}）-> 可以开实验二：训残差 Adapter + 研究 N->M 压缩")
    else:
        code = "single_layer_insufficient"
        text = (f"接口有效但单层不够（upper_kv {up:.3f} 高于 none，"
                f"h18_kv {h18:.3f} 没跟上）-> 先做 Adapter（有逐层 teacher 监督）或改存多层")
    return {"code": code, "text": text, "upper_kv_minus_none": up_gain,
            "h18_kv_minus_none": h18_gain, "upper_kv_p": up_p, "h18_kv_p": h18_p}


def print_table(s: Dict[str, Any], arms: List[str]) -> None:
    print("")
    print(f"| 臂 | EM% (n={s['n']}) |")
    print("|---|---|")
    for a in arms:
        print(f"| {a} | {100 * s['acc'][a]:.1f} |")
    print("")
    print("| 配对比较 | A% | B% | 差 | 只有A对 | 只有B对 | McNemar p |")
    print("|---|---|---|---|---|---|---|")
    for p in s["paired"]:
        print(f"| {p['a']} vs {p['b']} | {100 * p['acc_a']:.1f} | {100 * p['acc_b']:.1f} "
              f"| {100 * p['delta']:+.1f} | {p['only_a']} | {p['only_b']} | {p['p']:.4f} |")
    if s.get("by_ans_type"):
        keys = sorted(s["by_ans_type"])
        print("")
        print("| 题型 | n | " + " | ".join(arms) + " |")
        print("|---" * (2 + len(arms)) + "|")
        for t in keys:
            d = s["by_ans_type"][t]
            print(f"| {t} | {d['n']} | " +
                  " | ".join(f"{100 * d[a]:.1f}" for a in arms) + " |")
    print("")
    print(f"[判据] {s['verdict']['code']}: {s['verdict']['text']}", flush=True)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in s["acc"].items()}, ensure_ascii=False), flush=True)


def _dump_records(path: str, args, rows: List[Dict[str, Any]], partial: bool) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    arms = [a for a in ARMS if any(a in r for r in rows[:1])]
    blob = {"summary": summarize(args, rows, arms), "config": vars(args),
            "rows": rows, "partial": bool(partial), "n_done": len(rows)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=1)
    print(f"[out] {'中间' if partial else '完整'}结果 -> {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
