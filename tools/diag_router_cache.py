#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/diag_router_cache.py — P0-1：Router 的缓存一致性（cache=None vs cache={}）。

为什么必须单独验这一条
----------------------
2026-11-04 定位到的缺陷：`encode_item_slots_full()` 的缓存分支返回 `kfeat=None`。
训练/阶段一（cache=None）能用逐 token 冻结特征，端到端评测（cache={}）却退化成
"潜载荷键"，MaxSim 与新信息抑制静默失效 —— 同一份 router 权重，阶段一完整召回
52%，端到端只剩 6%。修完之后**必须**用同一批题逐题对拍，而不是只看总分涨了：
总分受采样与解码影响，逐题 picks 才是同一个函数的两个输入是否等价。

本脚本对每个样本跑两条路径，逐项比较：
    payload（语义载荷）/ lex（直通载荷）/ kfeat（逐槽 token 特征）
    / picks（路由选中的槽）/ read_all（完整证据是否被选中）

判定：
    kfeat 最大绝对差 <= --tol（bf16 下 batch 内 padding 会带来 1e-3 级差异）
    picks 一致率 == 100%（特征等价则路由必须给出同一组槽，否则还有别的漂移）
不满足时**非零退出**，不做"差一点也算过"。

用法（在仓库根目录）：
    python tools/diag_router_cache.py --ckpt $B/ckpt/hotpot_router_10k_v5.pt \
        --data $B/synth/hotpot_val500.jsonl --limit 200 \
        --base $B/models/Qwen3-4B-Instruct-2507 \
        --adapter $B/models/delta-mem_qwen3_4b_instruct \
        --save-json $B/results/diag_router_cache.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import torch

import latent_memory as LM
import train_latent_memory as T


def _maxdiff(a, b):
    """两个张量/列表的最大绝对差；形状不同直接返回 inf（不是静默跳过）。"""
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    if torch.is_tensor(a) and torch.is_tensor(b):
        if tuple(a.shape) != tuple(b.shape):
            return float("inf")
        return float((a.float() - b.float()).abs().max())
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max([_maxdiff(x, y) for x, y in zip(a, b)] + [0.0])
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return float("inf")
        return max([_maxdiff(a[k], b[k]) for k in a] + [0.0])
    return 0.0 if a == b else float("inf")


def _picks(reader, payloads, kfeat, lex, qf, topk, hops):
    slot_z = LM.pool_slot_keys(list(payloads)).unsqueeze(0)
    reads = LM.pack_slot_reads(list(payloads))
    out = reader.router.forward_hops(
        qf, slot_z, n_hops=max(1, int(hops)), hard=True, reads=reads,
        kfeat=kfeat, q_tok=qf.get("h0_tok"), topk_per_hop=int(topk),
        lex_reads=lex)
    return [int(x) for x in out["picks"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description="P0-1 Router 缓存一致性对拍")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--base", default=T.BASE)
    ap.add_argument("--adapter", default=T.ADAPTER)
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--topk-per-hop", type=int, default=4)
    ap.add_argument("--tol", type=float, default=5e-2,
                    help="kfeat 允许的最大绝对差（bf16 分批 padding 的正常抖动）")
    ap.add_argument("--save-json", default="")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    _argv = ["--gpu", str(a.gpu), "--base", a.base, "--adapter", a.adapter]
    if a.no_adapter:
        _argv.append("--no-adapter")
    args = T.parse_args(_argv)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    T._setup_tf32(True)
    T.gpu_report(args)
    model, tok, dev, hidden = T._load_backbone(args, dev)
    T.freeze_model(model)
    reader = LM.load_latent_reader(a.ckpt, model, tok, device=dev)
    if reader.router is None:
        raise SystemExit(f"{a.ckpt} 里没有 router 权重，无法对拍路由")
    enc = reader.encoder
    print(f"[ckpt] {a.ckpt} m={reader.config.get('n_slots')} "
          f"layer={reader.config.get('layer')} "
          f"max_unit_tokens={reader.config.get('max_unit_tokens')} "
          f"router_kind={reader.config.get('router_kind')}", flush=True)

    data = T.load_synth_jsonl(a.data, limit=a.limit, seed=0)
    if not data:
        raise SystemExit(f"{a.data} 里没有样本")

    rows = []
    for i, item in enumerate(data):
        T.reset_delta_state(model)
        units = T.item_units(item)
        qf = LM.extract_query_feature_ex(model, tok, item["question"],
                                         args.system_prompt, layer=enc.layer)
        # 路径 A：不传 cache（训练 / 阶段一的路径）
        pa, _za, _ia, kfa, lexa = T.encode_item_slots_full(
            enc, item, tag="diag", cache=None)
        # 路径 B：传空 cache（评测脚本的路径）
        pb, _zb, _ib, kfb, lexb = T.encode_item_slots_full(
            enc, item, tag="diag", cache={})
        hops = T.item_hops(item)
        picks_a = _picks(reader, pa, kfa, lexa, qf, a.topk_per_hop, hops)
        picks_b = _picks(reader, pb, kfb, lexb, qf, a.topk_per_hop, hops)
        gold = set(T.evidence_slot_indices(units, item.get("evidence")))
        row = {
            "idx": i,
            "payload_diff": _maxdiff(pa, pb),
            "lex_diff": _maxdiff(lexa, lexb),
            "kfeat_diff": _maxdiff(kfa, kfb),
            "kfeat_a_none": kfa is None,
            "kfeat_b_none": kfb is None,
            "picks_a": picks_a, "picks_b": picks_b,
            "picks_same": picks_a == picks_b,
            "read_all_a": (bool(gold <= set(picks_a)) if gold else None),
            "read_all_b": (bool(gold <= set(picks_b)) if gold else None),
            "n_gold": len(gold),
        }
        rows.append(row)
        if a.verbose:
            print(f"  [{i}] payload={row['payload_diff']:.3e} "
                  f"kfeat={row['kfeat_diff']:.3e} picks {picks_a} vs {picks_b}"
                  f" {'✓' if row['picks_same'] else '✗'}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"[progress] {i + 1}/{len(data)}", flush=True)

    n = len(rows)
    kf_diffs = [r["kfeat_diff"] for r in rows]
    same = sum(int(r["picks_same"]) for r in rows)
    ga = [r for r in rows if r["read_all_a"] is not None]
    rec_a = (sum(int(r["read_all_a"]) for r in ga) / len(ga)) if ga else None
    rec_b = (sum(int(r["read_all_b"]) for r in ga) / len(ga)) if ga else None
    kfeat_none = sum(int(r["kfeat_b_none"]) for r in rows)
    summary = {
        "n": n,
        "kfeat_maxdiff": max(kf_diffs) if kf_diffs else None,
        "kfeat_none_in_cache_path": kfeat_none,
        "payload_maxdiff": max(r["payload_diff"] for r in rows),
        "picks_same_rate": same / n,
        "full_recall_cache_none": rec_a,
        "full_recall_cache_empty": rec_b,
        "tol": a.tol,
        "ckpt": a.ckpt, "data": a.data,
        "topk_per_hop": int(a.topk_per_hop),
    }
    print("=" * 70)
    print(f"n={n}  kfeat_maxdiff={summary['kfeat_maxdiff']:.3e}  "
          f"payload_maxdiff={summary['payload_maxdiff']:.3e}")
    print(f"picks 一致率 = {100.0 * summary['picks_same_rate']:.1f}%")
    print(f"完整证据召回：cache=None {rec_a}  vs  cache={{}} {rec_b}")
    if kfeat_none:
        print(f"[FAIL] 缓存路径仍有 {kfeat_none}/{n} 条返回 kfeat=None（缺陷未修干净）")
    if summary["kfeat_maxdiff"] is not None and summary["kfeat_maxdiff"] > a.tol:
        print(f"[FAIL] kfeat 最大差 {summary['kfeat_maxdiff']:.3e} > tol={a.tol}")
    if same != n:
        print(f"[FAIL] picks 不一致 {n - same}/{n} 条：特征等价则路由必须给出同一组槽")
    ok = (kfeat_none == 0 and same == n
          and (summary["kfeat_maxdiff"] or 0.0) <= a.tol)
    print("[PASS] 两条路径等价" if ok else "[FAIL] 两条路径不等价（见上面的具体项）")
    if a.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(a.save_json)) or ".", exist_ok=True)
        with open(a.save_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
        print(f"[out] -> {a.save_json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
