#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/diag_symbol_fidelity.py — P1-3：符号保真检查（Oracle 答错的题，压缩器到底锚在哪些词上）。

它回答的是一个**机制问题**，不是分数问题：
    答案实体/数字有没有进入潜层？如果进了，是不是被 Bridge 改坏了？

做法（逐题）：
    1. 只取"金标证据句"这个单元（单槽，避免多槽拼接的干扰）；
    2. 调 `LatentCompressor.forward_split`，读 `info["lex_pick"]`（每行选中的 token 列号）
       并用 `encode_latent_units` 返回的 `chunk_ids` 精确还原成**词**；
    3. 检查答案串的 token 是否出现在选中的词里（"实体有没有进潜层"）；
    4. 过 Bridge，读 `bridge.last_stats`（α / 残差比 / 输出范数 / 末层权重范数）；
    5. 走 oracle_latent 自由生成，记录预测，并标出对错。

输出两层：
    逐题明细（--verbose 或 --limit 较小的时候直接打）；
    汇总：实体命中率、答案 token 命中率、答错题的共同特征（α 偏高？残差比偏大？）。

用法：
    python tools/diag_symbol_fidelity.py --ckpt $B/ckpt/hotpot_router_10k_v5.pt \
        --data $B/synth/hotpot_val500.jsonl --limit 50 \
        --base $B/models/Qwen3-4B-Instruct-2507 \
        --adapter $B/models/delta-mem_qwen3_4b_instruct \
        --save-json $B/results/diag_symbol_fidelity.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import torch

import latent_memory as LM
import train_latent_memory as T

_WORD_RE = re.compile(r"\w+")


def _tokens(s):
    return _WORD_RE.findall(T._norm_ans(s))


def _picked_words(tok, chunk_ids, picks):
    """把 lex_pick 的列号还原成词（列号是**块内**下标，与 chunk_ids 一一对应）。"""
    out = []
    for p in picks:
        p = int(p)
        if 0 <= p < len(chunk_ids):
            out.append(tok.decode([int(chunk_ids[p])]).strip())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="P1-3 符号保真检查")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--limit", type=int, default=50,
                    help="最多检查多少条**答错**的题（先跑生成筛出错的）")
    ap.add_argument("--scan", type=int, default=200,
                    help="最多扫多少条样本以凑够 --limit 条答错题")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--base", default=T.BASE)
    ap.add_argument("--adapter", default=T.ADAPTER)
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--save-json", default="")
    ap.add_argument("--verbose", action="store_true", default=True)
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
    comp, bridge, enc = reader.compressor, reader.bridge, reader.encoder
    bridge.collect_stats = True
    pad = getattr(tok, "eos_token_id", None) or getattr(tok, "pad_token_id", None) or 0
    sysp = args.system_prompt
    mut = int(reader.config.get("max_unit_tokens", enc.max_unit_tokens))
    print(f"[ckpt] α={float(bridge.alpha()):.4f} scale={float(bridge.scale()):.3f} "
          f"|Wlast|={float(bridge.net[-1].weight.norm()):.3f} "
          f"lex_skip={bool(bridge.lex_skip)} m={reader.config.get('n_slots')} "
          f"max_unit_tokens={mut}", flush=True)

    data = T.load_synth_jsonl(a.data, limit=a.scan, seed=0)
    rows = []
    n_scanned = 0
    n_skipped_multi = 0
    for i, item in enumerate(data):
        if len(rows) >= a.limit:
            break
        n_scanned += 1
        T.reset_delta_state(model)
        units = T.item_units(item)
        gold_idx = T.evidence_slot_indices(units, item.get("evidence"))
        if not gold_idx:
            continue
        gi = int(gold_idx[0])
        text = T.unit_encode_text(item, gi)
        # 单块单元才做逐词对齐（多块时列号是块内下标，跨块映射要额外记账）
        feats = enc.features([text], parent_ids=["diag"])
        if int(feats["n_chunks_total"]) != 1:
            n_skipped_multi += 1
            continue
        cids = feats["chunk_ids"][0]
        with torch.no_grad():
            z, z_lex, cinfo = comp.forward_split(feats["h0"], feats["h18"],
                                                 feats["mask"])
        picks = (cinfo.get("lex_pick") or [[]])[0]
        words = _picked_words(tok, cids, picks)

        # oracle_latent 自由生成（与评测同一套构造）
        with torch.no_grad():
            payloads, _z, _info, _kf, lex_all = T.encode_item_slots_full(
                enc, item, tag="diag", cache=None)
            idx = [j for j in gold_idx if j < len(payloads)]
            reads = LM.pack_slot_reads(list(payloads))
            soft = LM.compose_routed_soft(
                model, tok, bridge, [reads[j] for j in idx], dev=dev,
                lex_list=([lex_all[j] for j in idx] if lex_all is not None else None),
                isolate=False)
            pred, _meta = LM._emit_latent(
                model, tok, soft, item["question"], sysp,
                max_new_tokens=int(a.max_new_tokens), pad_token_id=pad)
        gold = item["answer"]
        if isinstance(gold, (list, tuple)):
            gold = " ".join(str(x) for x in gold)
        ok = T._norm_ans(pred) == T._norm_ans(gold)
        if ok:
            continue                                  # 只看答错的题
        g_toks = set(_tokens(gold))
        w_toks = set(_tokens(" ".join(words)))
        # 答案 token 在"被选中的词"里的命中率：0 说明答案实体根本没进潜层
        hit_rate = (len(g_toks & w_toks) / len(g_toks)) if g_toks else None
        st = dict(bridge.last_stats or {})
        row = {
            "idx": i, "question": item["question"], "gold_ans": str(gold),
            "pred": str(pred), "gold_unit": str(units[gi]),
            "picked_words": words, "picked_ids": [int(p) for p in picks],
            "ans_token_hit_rate": hit_rate,
            "ans_in_unit": bool(T._norm_ans(gold) and
                                T._norm_ans(gold) in T._norm_ans(units[gi])),
            "bridge": st,
        }
        rows.append(row)
        if a.verbose:
            print(f"\n[{len(rows)}/{a.limit}] ✗ q={item['question'][:70]!r}")
            print(f"  gold={str(gold)!r}  pred={str(pred)[:60]!r}")
            print(f"  unit={str(units[gi])[:110]!r}")
            print(f"  压缩器选中词={words}  答案 token 命中率="
                  f"{'n/a' if hit_rate is None else f'{hit_rate:.2f}'}")
            print(f"  α={st.get('alpha', 0):.3f} res/lex={st.get('res_ratio', 0):.3f} "
                  f"|E|={st.get('e_norm', 0):.2f} |lex|={st.get('lex_norm', 0):.2f} "
                  f"|Wlast|={st.get('w_last', 0):.2f}", flush=True)

    n = len(rows)
    hits = [r["ans_token_hit_rate"] for r in rows if r["ans_token_hit_rate"] is not None]
    alphas = [r["bridge"].get("alpha", 0.0) for r in rows]
    ratios = [r["bridge"].get("res_ratio", 0.0) for r in rows]
    summary = {
        "n_wrong": n, "n_scanned": n_scanned, "n_skipped_multi_chunk": n_skipped_multi,
        "ans_token_hit_rate_mean": (sum(hits) / len(hits)) if hits else None,
        "ans_token_hit_rate_zero": sum(1 for h in hits if h == 0.0),
        "ans_in_unit_rate": (sum(int(r["ans_in_unit"]) for r in rows) / n) if n else None,
        "alpha_mean": (sum(alphas) / n) if n else None,
        "res_ratio_mean": (sum(ratios) / n) if n else None,
        "rows": rows,
    }
    print("\n" + "=" * 70)
    print(f"答错题 {n} 条（扫了 {n_scanned} 条，跳过 {n_skipped_multi} 条多块单元）")
    print(f"答案 token 在被选中的词里命中率均值 = "
          f"{'n/a' if summary['ans_token_hit_rate_mean'] is None else round(summary['ans_token_hit_rate_mean'], 3)}"
          f"（其中 {summary['ans_token_hit_rate_zero']} 条完全没命中）")
    print(f"α 均值 = {summary['alpha_mean']}  res/lex 均值 = {summary['res_ratio_mean']}")
    print("读法：命中率≈0 且 α 正常 -> 答案实体没进潜层（查 Compressor 选词/容量）；")
    print("      命中率高但答错   -> 实体进了但被读出/生成改坏（查 Bridge 残差与注入）")
    if a.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(a.save_json)) or ".", exist_ok=True)
        with open(a.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f"[out] -> {a.save_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
