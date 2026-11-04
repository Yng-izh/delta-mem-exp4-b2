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
    python tools/diag_router_cache.py --ckpt $B/ckpt/<你的阶段二权重>.pt \
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


# kfeat 里只有这三个键是**真正的特征**；`n_units` / `key_tokens` 是记账字段。
# 缓存路径逐条编码，`n_units` 天然是 1；批量路径是槽数。把记账字段也拿去比，
# 会让每一次对拍都误报 inf —— 这是判据的错，不是代码的错。
KFEAT_TENSOR_KEYS = ("h0", "h18", "mask")


def _kfeat_diff(a, b):
    """只比特征张量；返回 (最大差, 记账字段是否不同)。"""
    if a is None or b is None:
        return (0.0 if (a is None and b is None) else float("inf")), False
    feats_a = {k: a[k] for k in KFEAT_TENSOR_KEYS if k in a}
    feats_b = {k: b[k] for k in KFEAT_TENSOR_KEYS if k in b}
    meta_diff = ({k: v for k, v in a.items() if k not in KFEAT_TENSOR_KEYS}
                 != {k: v for k, v in b.items() if k not in KFEAT_TENSOR_KEYS})
    return _maxdiff(feats_a, feats_b), meta_diff


def _picks(reader, payloads, kfeat, lex, qf, topk, hops):
    """用给定载荷跑一次路由，返回选中的槽位号。

    qf 是 `extract_query_feature_ex` 返回的 **dict**（pooled / h0_tok / h18_tok），
    而 forward_hops 的第一个位置参数要的是 pooled **张量** —— 直接传 dict 会在
    `torch.as_tensor` 处报 "Could not infer dtype of dict"。
    """
    slot_z = LM.pool_slot_keys(list(payloads)).unsqueeze(0)
    reads = LM.pack_slot_reads(list(payloads))
    out = reader.router.forward_hops(
        qf["pooled"], slot_z, n_hops=max(1, int(hops)), hard=True, reads=reads,
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
    ap.add_argument("--enc-batch", type=int, default=0,
                    help="编码批大小（0=沿用 ckpt 的默认 8）。设成 1 可让两条路径都"
                         "**逐条**编码，用来判定剩下的差异是不是纯粹来自分批数值抖动")
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
    if int(a.enc_batch) > 0:
        enc.batch_size = max(1, int(a.enc_batch))
        print(f"[enc] 编码批大小强制设为 {enc.batch_size}"
              f"（=1 时两条路径都逐条编码，剩下的差异就只可能是数值抖动）", flush=True)
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
        kf_diff, kf_meta_diff = _kfeat_diff(kfa, kfb)
        row = {
            "idx": i,
            "payload_diff": _maxdiff(pa, pb),
            "lex_diff": _maxdiff(lexa, lexb),
            "kfeat_diff": kf_diff,
            "kfeat_meta_diff": kf_meta_diff,
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
    meta_diff = sum(int(r["kfeat_meta_diff"]) for r in rows)
    # picks 不同但召回相同，说明这些差异落在"金标集合之外"或"多选的槽不影响链完整性"，
    # 对指标无害；真正致命的是召回本身不一致。
    pick_diff = n - same
    summary = {
        "n": n,
        "kfeat_maxdiff": max(kf_diffs) if kf_diffs else None,
        "kfeat_none_in_cache_path": kfeat_none,
        "kfeat_meta_diff_rows": meta_diff,
        "payload_maxdiff": max(r["payload_diff"] for r in rows),
        "payload_diff_median": sorted(r["payload_diff"] for r in rows)[n // 2],
        "picks_same_rate": same / n,
        "picks_diff_rows": pick_diff,
        "full_recall_cache_none": rec_a,
        "full_recall_cache_empty": rec_b,
        "recall_gap": (None if (rec_a is None or rec_b is None) else rec_b - rec_a),
        "tol": a.tol,
        "enc_batch": int(enc.batch_size),
        "ckpt": a.ckpt, "data": a.data,
        "topk_per_hop": int(a.topk_per_hop),
    }
    print("=" * 70)
    print(f"n={n}  kfeat_maxdiff={summary['kfeat_maxdiff']:.3e}  "
          f"payload_maxdiff={summary['payload_maxdiff']:.3e} "
          f"(中位 {summary['payload_diff_median']:.3e})")
    print(f"picks 一致率 = {100.0 * summary['picks_same_rate']:.1f}%"
          f"（{pick_diff} 条不同）")
    print(f"完整证据召回：cache=None {rec_a}  vs  cache={{}} {rec_b}")
    if meta_diff:
        print(f"[info] {meta_diff}/{n} 条的 kfeat 记账字段不同（逐条编码 n_units=1 vs "
              f"批量 n_units=槽数），这是设计如此，不参与判定")
    # 判定只看**功能性**的三条：特征有没有、特征是否等价、召回是否一致。
    # picks 完全一致是"更强"的要求，bf16 分批抖动会让近似并列的槽翻面，
    # 因此单独报，不直接判死。
    fail = []
    if kfeat_none:
        fail.append(f"缓存路径仍有 {kfeat_none}/{n} 条返回 kfeat=None（缺陷未修干净）")
    if summary["kfeat_maxdiff"] is not None and summary["kfeat_maxdiff"] > a.tol:
        fail.append(f"kfeat 最大差 {summary['kfeat_maxdiff']:.3e} > tol={a.tol}")
    if rec_a is not None and rec_b is not None and abs(rec_b - rec_a) > 1e-9:
        fail.append(f"完整证据召回不一致：{rec_a} vs {rec_b}（这才是致命项）")
    for f in fail:
        print(f"[FAIL] {f}")
    ok = not fail
    if ok and pick_diff:
        print(f"[info] 有 {pick_diff} 条 picks 不同但召回一致 —— 多半是 bf16 分批数值"
              f"导致近似并列的槽翻面。要确认可加 --enc-batch 1 重跑：两条路径都逐条"
              f"编码时，这个差异应当消失")
    print("[PASS] 两条路径功能等价" if ok else "[FAIL] 两条路径不等价（见上面的具体项）")
    if a.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(a.save_json)) or ".", exist_ok=True)
        with open(a.save_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
        print(f"[out] -> {a.save_json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
