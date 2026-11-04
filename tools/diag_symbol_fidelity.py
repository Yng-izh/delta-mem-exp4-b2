#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/diag_symbol_fidelity.py — 选词保真检查（词级 / 覆盖率 / 无效选择）。

回答四个具体问题：
    1. 答案实体有没有进入潜层？（命中率）
    2. **完整答案覆盖率**：答案的每一个内容词都被选中了吗？（只命中一个子词不算成功）
    3. **数字覆盖率**：`9,984` 这种数字是否整体保留（不是只剩 `9`）？
    4. **无效选择比例**：4 个槽里还有多少被标点、停用词占着？

口径（三处会直接让结论失真，都已按最严格方式处理）：
    - 看**全部金标槽**，不是只看第一个：两跳题的答案常在第二条证据句里。
    - **支持多块单元**：单元长于 max_unit_tokens 会被切块，选词结果按块还原。
    - **只在"确实含答案"的槽上统计覆盖率**：答案不在那句里时，覆盖率没有意义。
    - 选词粒度由压缩器自己报告（`lex_granularity`）：词级用 `lex_word_pick` +
      `chunk_word_ids` 还原整词；token 级（旧 ckpt / 未传词边界）自动降级并标注。

用法：
    python tools/diag_symbol_fidelity.py --ckpt $B/ckpt/xxx.pt \
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
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _tokens(s):
    return _WORD_RE.findall(T._norm_ans(s))


def _is_invalid_word(w):
    """标点 / 停用词 / 空串 = 无效选择（不携带事实信息）。"""
    t = T._norm_ans(w)
    if not t:
        return True
    if all(not ch.isalnum() for ch in t):
        return True
    return t in LM._LATENT_STOPWORDS


def _answer_content_words(answer):
    """答案里的**内容词**（按空白切，去掉标点词与停用词）。

    注意按**空白**切而不是按 `\w+`：`9,984` 必须作为一个整体参与比对，
    否则会被拆成 `9` / `984`，"数字完整保留"这个口径就自相矛盾了。
    比较时两侧都用 `T._norm_ans` 归一（大小写/标点/空白一致）。
    """
    return [w for w in str(answer or "").split()
            if w and not _is_invalid_word(w)]


def _numbers(text):
    return _NUM_RE.findall(str(text or ""))


def _hit(words, answer):
    """答案的任一内容词被选中即算命中（宽松口径，用于与历史数字对照）。

    允许"选中词是答案词的前缀"（子词/截断），但要求长度 >= 3，避免 `a`/`in`
    这种碎片把命中率虚抬上去。
    """
    need = _answer_content_words(answer)
    if not need:
        return None
    have = [T._norm_ans(w) for w in words]
    for n in need:
        nn = T._norm_ans(n)
        if any(h == nn or (len(h) >= 3 and (h in nn or nn in h)) for h in have):
            return True
    return False


def _full_answer_covered(words, answer):
    """**完整覆盖**：答案的每一个内容词都必须被选中（只命中一个子词不算）。

    两侧都归一化后再比（大小写/标点/空白口径一致），且要求**整词相等**：
    子串匹配会让 "只抓到 `Mer`" 被判成覆盖了 `Mercedes`，那正是本指标要排除的情况。
    """
    need = _answer_content_words(answer)
    if not need:
        return None
    have = {T._norm_ans(w) for w in words}
    return all(T._norm_ans(n) in have for n in need)


def _numbers_covered(words, answer):
    """**数字整体保留**：`9,984` 必须作为整体出现，而不是只剩 `9`。"""
    nums = _numbers(answer)
    if not nums:
        return None
    have = " ".join(T._norm_ans(w) for w in words)
    return all(T._norm_ans(n) in have for n in nums)


def _slot_words(tok, feats, cinfo):
    """把选词结果还原成**整词**列表。

    `cinfo["lex_word_pick"]` 是逐块选中的**词编号**；用 `feats["chunk_word_ids"]`
    把该词的全部子词拼回整词。选词已经是词级，这里不再有 token 级降级路径
    （`forward_split` 缺词边界时直接报错，不会产出 token 列号）。
    """
    wids_all = feats.get("chunk_word_ids") or []
    cids_all = feats.get("chunk_ids") or []
    word_pick = cinfo.get("lex_word_pick")
    if word_pick is None:
        raise RuntimeError("压缩器没有返回 lex_word_pick：ckpt 不是词级选词版本，"
                           "或 forward_split 没有收到 word_ids")
    words = []
    for k, cols in enumerate(word_pick):
        if k >= len(wids_all) or k >= len(cids_all):
            continue
        wmap, cids = wids_all[k], cids_all[k]
        for wid in (cols or []):
            toks = [cids[i] for i, w in enumerate(wmap) if int(w) == int(wid)]
            if toks:
                words.append(_decode(tok, toks))
    return [w for w in words if w]


def _decode(tok, ids):
    try:
        return tok.decode([int(x) for x in ids]).strip()
    except Exception:
        return ""


def main(argv=None):
    ap = argparse.ArgumentParser(description="选词保真检查（词级 / 覆盖率 / 无效选择）")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--limit", type=int, default=50,
                    help="分析多少条样本。默认取**固定的前 N 条**（含答对与答错），"
                         "这样改代码前后可以用同一批样本对比")
    ap.add_argument("--scan", type=int, default=200,
                    help="仅在 --only-wrong 时生效：最多扫多少条以凑够 --limit 条答错题")
    ap.add_argument("--only-wrong", action="store_true",
                    help="只分析答错的题（旧行为）。注意它会让样本集随模型变化，"
                         "不能用于改代码前后的对比")
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
    # 注入位置/围栏样式以 ckpt 为准（旧 ckpt 未记录 -> 按旧口径 between+verbose）
    _ck_inj = str(reader.config.get("inject_mode", "") or "between")
    _ck_fen = str(reader.config.get("fence_style", "") or "verbose")
    print(f"[ckpt] α={float(bridge.alpha()):.4f} scale={float(bridge.scale()):.3f} "
          f"|Wlast|={float(bridge.net[-1].weight.norm()):.3f} "
          f"lex_skip={bool(bridge.lex_skip)} m={reader.config.get('n_slots')} "
          f"max_unit_tokens={enc.max_unit_tokens} "
          f"inject={_ck_inj}/{_ck_fen}", flush=True)

    data = T.load_synth_jsonl(a.data, limit=a.scan, seed=0)
    rows = []
    slot_stats = []          # 每个"含答案的金标槽"一行
    pick_stats = []          # 每个被选中的词一条（用于无效选择比例）
    n_scanned = 0
    for i, item in enumerate(data):
        if len(rows) >= a.limit:
            break
        n_scanned += 1
        T.reset_delta_state(model)
        units = T.item_units(item)
        gold_idx = list(T.evidence_slot_indices(units, item.get("evidence")))
        if not gold_idx:
            continue
        with torch.no_grad():
            payloads, _z, _info, _kf, lex_all = T.encode_item_slots_full(
                enc, item, tag="diag", cache=None)
        idx = [j for j in gold_idx if j < len(payloads)]
        if not idx:
            continue
        reads = LM.pack_slot_reads(list(payloads))
        with torch.no_grad():
            soft = LM.compose_routed_soft(
                model, tok, bridge, [reads[j] for j in idx], dev=dev,
                lex_list=([lex_all[j] for j in idx] if lex_all is not None else None),
                isolate=True, fence_style=_ck_fen)
            pred, _meta = LM._emit_latent(
                model, tok, soft, item["question"], sysp,
                max_new_tokens=int(a.max_new_tokens), pad_token_id=pad,
                inject_mode=_ck_inj, fence_style=_ck_fen)
        gold = item["answer"]
        if isinstance(gold, (list, tuple)):
            gold = " ".join(str(x) for x in gold)
        ok = T._norm_ans(pred) == T._norm_ans(gold)
        # 默认**固定取前 N 条**（答对答错都要），否则改代码前后比较的不是同一批样本。
        if a.only_wrong and ok:
            continue
        if (not a.only_wrong) and ok and not a.show_ok:
            pass                     # 默认统计全部，但仍只在 --show-ok 时打印明细
        st = dict(bridge.last_stats or {})
        per_slot = []
        for j in idx:
            text = T.unit_encode_text(item, j)
            feats = enc.features([text], parent_ids=["diag"])
            with torch.no_grad():
                _z2, _zl2, cinfo = comp.forward_split(
                    feats["h0"], feats["h18"], feats["mask"],
                    **_word_kwargs(feats))
            words = _slot_words(tok, feats, cinfo)
            ans_in_slot = bool(T._norm_ans(gold)) and (
                T._norm_ans(gold) in T._norm_ans(units[j]))
            n_invalid = sum(int(_is_invalid_word(w)) for w in words)
            rec = {
                "slot": int(j), "n_chunks": int(feats["n_chunks_total"]),
                "n_tokens": int(sum(feats["n_tokens"])),
                "ans_in_slot": ans_in_slot,
                "words": words, "n_picks": len(words), "n_invalid": n_invalid,
                "hit": _hit(words, gold),
                "full_answer": _full_answer_covered(words, gold),
                "numbers": _numbers_covered(words, gold),
                "unit": str(units[j])[:200],
            }
            per_slot.append(rec)
            for w in words:
                pick_stats.append(int(_is_invalid_word(w)))
            if ans_in_slot:
                slot_stats.append({"hit": rec["hit"],
                                   "full_answer": rec["full_answer"],
                                   "numbers": rec["numbers"],
                                   "n_chunks": rec["n_chunks"],
                                   "n_picks": rec["n_picks"],
                                   "n_invalid": rec["n_invalid"]})
        row = {
            "idx": i, "question": item["question"], "gold_ans": str(gold),
            "pred": str(pred), "ok": bool(ok), "n_gold": len(idx),
            "slots": per_slot, "bridge": st,
        }
        rows.append(row)
        if not a.show_ok and ok:
            continue                     # 统计包含答对题，但明细只在 --show-ok 时打印
        print(f"\n[{len(rows)}] {'✓' if ok else '✗'} q={item['question'][:70]!r}")
        print(f"  gold={str(gold)!r}  pred={str(pred)[:60]!r}  金标槽={idx}")
        for rec in per_slot:
            mark = "含答案" if rec["ans_in_slot"] else "不含答案"
            print(f"  槽{rec['slot']}({rec['n_chunks']}块/{rec['n_tokens']}tok,{mark},"
                  f"): 选中={rec['words']}")
            if rec["ans_in_slot"]:
                print(f"      命中={rec['hit']} 完整答案={rec['full_answer']} "
                      f"数字={rec['numbers']} 无效选择={rec['n_invalid']}/"
                      f"{rec['n_picks']}")
        print(f"  α={st.get('alpha', 0):.3f} res/lex={st.get('res_ratio', 0):.3f} "
              f"|E|={st.get('e_norm', 0):.2f} |Wlast|={st.get('w_last', 0):.2f}",
              flush=True)

    n = len(rows)
    n_slots = len(slot_stats)

    def _rate(key):
        vals = [s[key] for s in slot_stats if s.get(key) is not None]
        return (sum(int(v) for v in vals) / len(vals)) if vals else None

    invalid_rate = (sum(pick_stats) / len(pick_stats)) if pick_stats else None
    summary = {
        "n_items": n, "n_scanned": n_scanned, "n_answer_slots": n_slots,
        "n_ok": sum(int(r["ok"]) for r in rows),
        "n_wrong": sum(int(not r["ok"]) for r in rows),
        "sample_mode": ("only_wrong" if a.only_wrong else "first_n"),
        "hit_rate": _rate("hit"),
        "full_answer_rate": _rate("full_answer"),
        "number_rate": _rate("numbers"),
        "invalid_pick_rate": invalid_rate,
        "n_picks_total": len(pick_stats),
        "alpha_mean": (sum(r["bridge"].get("alpha", 0.0) for r in rows) / n) if n else None,
        "res_ratio_mean": (sum(r["bridge"].get("res_ratio", 0.0) for r in rows) / n)
        if n else None,
        "rows": rows,
    }
    print("\n" + "=" * 70)
    print(f"样本 {n} 条（扫了 {n_scanned} 条；答对 {summary['n_ok']} / "
          f"答错 {summary['n_wrong']}；取样={summary['sample_mode']}）；"
          f"含答案的金标槽 {n_slots} 个")
    if n_slots:
        _nr = summary["number_rate"]
        _nr_s = "n/a" if _nr is None else f"{100 * _nr:.1f}%"
        print(f"命中率（任一答案词）= {100 * summary['hit_rate']:.1f}%   "
              f"**完整答案覆盖** = {100 * summary['full_answer_rate']:.1f}%   "
              f"**数字完整保留** = {_nr_s}")
    print(f"**无效选择比例**（标点/停用词占被选词的比例）= "
          f"{'n/a' if invalid_rate is None else f'{100 * invalid_rate:.1f}%'}"
          f"（共 {len(pick_stats)} 个被选词）")
    print("读法：")
    print("  完整答案覆盖率低 -> 选中的词没有覆盖答案的全部内容词（只抓到一半）")
    print("  数字覆盖率低     -> 数字被拆开或没被选中（`9,984` 只剩 `9`）")
    print("  无效选择比例高   -> 4 个槽被标点/停用词占着，白占预算")
    if a.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(a.save_json)) or ".", exist_ok=True)
        with open(a.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f"[out] -> {a.save_json}")
    return 0


def _word_kwargs(feats):
    """把 feats 里的词边界信息整理成 forward_split 的关键字参数。"""
    wids = feats.get("chunk_word_ids") or []
    if not wids:
        return {}
    wid, pen = LM._pad_word_tensors(wids, feats.get("chunk_word_penalty") or [])
    return {"word_ids": wid, "word_penalty": pen}


if __name__ == "__main__":
    raise SystemExit(main())
