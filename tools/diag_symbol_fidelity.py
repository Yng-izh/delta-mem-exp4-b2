#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/diag_symbol_fidelity.py — P1-3：符号保真检查（压缩器到底把哪些"词"压进去了）。

它回答三个具体的机制问题，不是分数问题：
    1. 答案实体/数字有没有进入潜层？
    2. 压缩器选中的是**整词**还是**子词碎片**？（τ_lex=0.05 的 argmax 在 token 级，
       一个词常被切成 2~4 个 token，可能只抓到一片）
    3. 如果改成"选中一个 token 就把它所属整词一起聚合"，命中率能提高多少？
       —— 这一项是给"整词选择"手术做定量预估的，不改代码也能先看到上限。

口径（比上一版严格，上一版有两个会让结论失真的地方）：
    - **看全部金标槽**，不是只看第一个：两跳题的答案常常在第二条证据句里，
      只看第 0 条会把"答案本来就不在这句里"算成"压缩器没抓到"。
    - **支持多块单元**：单元长于 max_unit_tokens 会被切块，`lex_pick` 的列号是
      **块内**下标；必须按块还原成词，不能整条跳过（上一版因此跳过了 151/193 条）。
    - **只在"确实含答案"的槽上统计命中率**：答案不在那句里时，命中率没有意义。

用法：
    python tools/diag_symbol_fidelity.py --ckpt $B/ckpt/hotpot_router_10k_v5.pt \
        --data $B/synth/hotpot_val500.jsonl --limit 50 --scan 200 \
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


def _is_word_start(tok, tid):
    """该 token 是否是一个新词的开头。

    字节级 BPE（Qwen 用这个）用 "Ġ" 标记词前空格，SentencePiece 用 "▁"；
    标点/符号单独成词（`-`、`(` 之类）。用"前缀标记或首字符非字母数字"来判定，
    不依赖具体分词器实现。
    """
    try:
        t = tok.convert_ids_to_tokens(int(tid))
    except Exception:
        return True
    if not isinstance(t, str) or not t:
        return True
    return t.startswith(("Ġ", "▁", " ")) or not t[0].isalnum()


def _word_span(tok, ids, col):
    """col 所在**整词**的 token 下标区间 [a, b)。"""
    n = len(ids)
    a = max(0, min(int(col), n - 1))
    while a > 0 and not _is_word_start(tok, ids[a]):
        a -= 1
    b = max(0, min(int(col), n - 1)) + 1
    while b < n and not _is_word_start(tok, ids[b]):
        b += 1
    return a, b


def _decode(tok, ids):
    try:
        return tok.decode([int(x) for x in ids]).strip()
    except Exception:
        return ""


def _hit(pieces, answer):
    """选中的词是否覆盖答案：整词相等、或作为子词前缀（`Mer` ⊂ `Mercedes`）。"""
    ans_toks = set(_tokens(answer))
    for p in pieces:
        pn = T._norm_ans(p)
        if not pn:
            continue
        if pn in ans_toks:
            return True
        if len(pn) >= 2 and any(t.startswith(pn) for t in ans_toks):
            return True
    return False


def _slot_picks(tok, feats, cinfo):
    """把 `lex_pick`（逐块列号）还原成 (子词列表, 整词列表)。

    feats 来自 `enc.features([text])`，`chunk_ids[k]` 是第 k 块的 token id；
    `lex_pick[k]` 是第 k 块里 m 个被选中 token 的**块内列号**。
    """
    picks = cinfo.get("lex_pick") or []
    cids_all = feats.get("chunk_ids") or []
    pieces, whole = [], []
    for row, cols in enumerate(picks):
        if row >= len(cids_all):
            continue
        cids = cids_all[row]
        for col in (cols or []):
            col = int(col)
            if not (0 <= col < len(cids)):
                continue
            pieces.append(_decode(tok, [cids[col]]))
            a, b = _word_span(tok, cids, col)
            whole.append(_decode(tok, cids[a:b]))
    # 去重但保序
    return list(dict.fromkeys(pieces)), list(dict.fromkeys(whole))


def main(argv=None):
    ap = argparse.ArgumentParser(description="P1-3 符号保真检查（全金标槽 / 多块 / 整词对照）")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--limit", type=int, default=50,
                    help="最多分析多少条**答错**的题（先跑生成筛出错的）")
    ap.add_argument("--scan", type=int, default=200,
                    help="最多扫多少条样本以凑够 --limit 条答错题")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--base", default=T.BASE)
    ap.add_argument("--adapter", default=T.ADAPTER)
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--show-ok", action="store_true",
                    help="答对的题也打印（默认只看答错的）")
    ap.add_argument("--save-json", default="")
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
    print(f"[ckpt] α={float(bridge.alpha()):.4f} scale={float(bridge.scale()):.3f} "
          f"|Wlast|={float(bridge.net[-1].weight.norm()):.3f} "
          f"lex_skip={bool(bridge.lex_skip)} m={reader.config.get('n_slots')} "
          f"max_unit_tokens={enc.max_unit_tokens}", flush=True)

    data = T.load_synth_jsonl(a.data, limit=a.scan, seed=0)
    rows = []
    slot_stats = []          # 每个"含答案的金标槽"一行
    n_scanned = 0
    for i, item in enumerate(data):
        if len(rows) >= a.limit:
            break
        n_scanned += 1
        T.reset_delta_state(model)
        units = T.item_units(item)
        gold_idx = [j for j in T.evidence_slot_indices(units, item.get("evidence"))]
        if not gold_idx:
            continue
        with torch.no_grad():
            payloads, _z, _info, _kf, lex_all = T.encode_item_slots_full(
                enc, item, tag="diag", cache=None)
        idx = [j for j in gold_idx if j < len(payloads)]
        if not idx:
            continue
        reads = LM.pack_slot_reads(list(payloads))
        # 生成：**用全部金标槽**（上一版只用第 0 条，等于人为制造失败）
        with torch.no_grad():
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
        if ok and not a.show_ok:
            continue
        st = dict(bridge.last_stats or {})
        # 逐金标槽：把该槽的 lex_pick 还原成词（多块也要覆盖）
        per_slot = []
        for j in idx:
            text = T.unit_encode_text(item, j)
            feats = enc.features([text], parent_ids=["diag"])
            with torch.no_grad():
                _z2, _zl2, cinfo = comp.forward_split(feats["h0"], feats["h18"],
                                                      feats["mask"])
            pieces, whole = _slot_picks(tok, feats, cinfo)
            ans_in_slot = bool(T._norm_ans(gold)) and (
                T._norm_ans(gold) in T._norm_ans(units[j]))
            rec = {
                "slot": int(j), "n_chunks": int(feats["n_chunks_total"]),
                "n_tokens": int(sum(feats["n_tokens"])),
                "ans_in_slot": ans_in_slot,
                "pieces": pieces, "whole_words": whole,
                "piece_hit": _hit(pieces, gold),
                "whole_hit": _hit(whole, gold),
                "unit": str(units[j])[:200],
            }
            per_slot.append(rec)
            if ans_in_slot:                     # 只在"这句里确实有答案"时统计命中率
                slot_stats.append({"piece_hit": rec["piece_hit"],
                                   "whole_hit": rec["whole_hit"],
                                   "n_chunks": rec["n_chunks"]})
        row = {
            "idx": i, "question": item["question"], "gold_ans": str(gold),
            "pred": str(pred), "ok": bool(ok), "n_gold": len(idx),
            "slots": per_slot, "bridge": st,
        }
        rows.append(row)
        print(f"\n[{len(rows)}] {'✓' if ok else '✗'} q={item['question'][:70]!r}")
        print(f"  gold={str(gold)!r}  pred={str(pred)[:60]!r}  金标槽={idx}")
        for rec in per_slot:
            mark = ("含答案" if rec["ans_in_slot"] else "不含答案")
            print(f"  槽{rec['slot']}({rec['n_chunks']}块/{rec['n_tokens']}tok,{mark}): "
                  f"选中={rec['pieces']}")
            if rec["whole_words"] != rec["pieces"]:
                print(f"      整词={rec['whole_words']}  "
                      f"命中 碎片={rec['piece_hit']} 整词={rec['whole_hit']}")
        print(f"  α={st.get('alpha', 0):.3f} res/lex={st.get('res_ratio', 0):.3f} "
              f"|E|={st.get('e_norm', 0):.2f} |Wlast|={st.get('w_last', 0):.2f}",
              flush=True)

    n = len(rows)
    n_slots = len(slot_stats)
    piece_rate = (sum(int(s["piece_hit"]) for s in slot_stats) / n_slots) if n_slots else None
    whole_rate = (sum(int(s["whole_hit"]) for s in slot_stats) / n_slots) if n_slots else None
    alphas = [r["bridge"].get("alpha", 0.0) for r in rows]
    ratios = [r["bridge"].get("res_ratio", 0.0) for r in rows]
    multi = [s for s in slot_stats if int(s["n_chunks"]) > 1]
    summary = {
        "n_items": n, "n_scanned": n_scanned,
        "n_answer_slots": n_slots,
        "n_answer_slots_multi_chunk": len(multi),
        "piece_hit_rate": piece_rate,
        "whole_word_hit_rate": whole_rate,
        "whole_word_gain": (None if (piece_rate is None or whole_rate is None)
                            else whole_rate - piece_rate),
        "alpha_mean": (sum(alphas) / n) if n else None,
        "res_ratio_mean": (sum(ratios) / n) if n else None,
        "rows": rows,
    }
    print("\n" + "=" * 70)
    print(f"答错题 {n} 条（扫了 {n_scanned} 条）；其中**含答案**的金标槽 {n_slots} 个"
          f"（多块槽 {len(multi)} 个）")
    if piece_rate is not None:
        print(f"命中率（碎片口径）= {100 * piece_rate:.1f}%   "
              f"（整词口径）= {100 * whole_rate:.1f}%   "
              f"整词能多救 = {100 * (whole_rate - piece_rate):.1f} 个点")
    print(f"α 均值 = {summary['alpha_mean']}  res/lex 均值 = {summary['res_ratio_mean']}")
    print("读法：")
    print("  碎片命中率低 + 整词命中率高 -> 病在**选词粒度**（argmax 落在子词上），改整词即可")
    print("  两者都低                     -> 答案实体没进潜层（查容量/选词目标/压缩）")
    if a.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(a.save_json)) or ".", exist_ok=True)
        with open(a.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f"[out] -> {a.save_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
