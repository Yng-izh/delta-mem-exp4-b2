#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ar_diag.py — AR 逐题诊断基建（Recall@K / answer_in_retrieved）

纯标准库实现，**不 import torch / transformers / deltamem**，因此：
  - 可被 `run_ar_v7.py` 在模块顶层直接 import（不触发模型加载、不占显存）；
  - 可在本机（无 /share 路径、无 GPU）跑 `--selftest` 与离线后处理。

=====================================================================
一、为什么需要它
=====================================================================
现有 AR 结果 JSONL 只记 `retrieval.idxs`（槽位下标），无法回答两个关键问题：
  1. **检索侧**：gold answer 到底有没有出现在 Top-K 检索结果里？（Recall@K）
  2. **读出侧**：检索命中但模型仍答错，是"没检索到"还是"喂进去了但没读出来"？
     （answer_in_retrieved vs answer_in_context）
两者分开统计，才能把 AR 的损失定位到检索器或 Reader。

=====================================================================
二、用法 A：运行时逐题记录（runner 调用，推荐）
=====================================================================
在 `run_ar_v7.py` 的每题循环里，检索完成、拼块完成后加两行：

    import ar_diag                                    # 顶层 import 即可

    hits = pool.retrieve(q_core, top_k=TOP_K)         # [(idx, score)] 分数降序
    used = sorted([i for i, _ in hits], key=...)      # 实际喂给 LLM 的槽位（原文序）
    rec.update(ar_diag.record(
        pool=pool, hits=hits, used_ids=used,
        answer=a, query=q, flat=True,
    ))

v9 池（`v9_slotpool.py`）已提供现成入口：
    hits = pool.retrieve(q, top_k=TOP_K)              # 分数序，返回 [(int, float)]
    used = pool.retrieve_ordered(q, top_k=TOP_K)      # 分数选 Top-K 后再按原文 start 排序
    used = pool.expand_backward(anchor, back=..., ahead=...)   # EventQA 后向非对称扩展
三者输出都可直接喂给 record()（hits 传分数序结果，used_ids 传最终拼块序）。

写入字段（flat=True 时平铺在 rec 上）：
    retrieved_slot_ids   list[int]    Top-K 槽位下标，**按检索分数降序**
    retrieved_scores     list[float]  与 ids 对齐的分数
    retrieved_texts      list[str]    与 ids 对齐的槽位原文
    retrieved_starts     list[int]    与 ids 对齐的原文句序位置（有 metas 时）
    used_slot_ids        list[int]    实际拼进 prompt 的槽位（原文序 / 扩展后）
    answer_in_retrieved  bool|None    gold 是否出现在 Top-K 任一槽位文本中
    answer_in_context    bool|None    gold 是否出现在实际拼块文本中
    answer_hit_slot      int|None     首个命中槽位（分数序）
    answer_matches       list[str]    命中的 gold 变体
    recall_at_k          dict[str,int] {"1":0/1, "3":..., "full":...}（仅 applicable 时有值）
    diag_status          str          ok | no_hits | no_texts | rubric_not_applicable

兼容 v7 `ChunkPool`（只有 `.texts`）与 v8/v9 池（`.texts` + `.metas`）。
`hits` 也接受 [idx, ...]（无分数）或 [(idx, score), ...]。

=====================================================================
三、用法 B：离线后处理（对已跑完的 JSONL 补诊断）
=====================================================================
    # 汇总 Recall@K（只读，不改原文件）
    python ar_diag.py --input results/ar_v9_longmemeval_s*.jsonl

    # 补写诊断字段到新文件（原文件不动）
    python ar_diag.py --input results/ar_v9_eventqa_65536.jsonl \
                      --enrich --enrich-out results/ar_v9_eventqa_65536.diag.jsonl

    # 分题型 / 分切片汇总，落盘 JSON
    python ar_diag.py --input a.jsonl b.jsonl --summary-out results/diag_summary.json

    # 纯 CPU 自测（无模型、无网络、无 GPU）
    python ar_diag.py --selftest

离线模式要求记录里已有 `retrieved_texts`（即 runner 已按用法 A 记录）。
只有 `retrieval.idxs` 的旧记录会被计为 `diag_status=legacy_idxs_only` 并跳过，
不会静默算成 0 —— 分母始终是"可计算"的记录数。

=====================================================================
四、指标口径（写死在这里，避免各脚本口径漂移）
=====================================================================
- **answer_in_retrieved**：Top-K（分数序）任一槽位文本包含 gold 即 1。
  gold 取 `answers` 列表里所有字符串变体；文本与 gold 都做 normalize
  （小写、去标点、压空白）后做子串匹配。
- **answer_in_context**：实际拼进 prompt 的槽位文本里是否包含 gold。
  与 answer_in_retrieved 的差集即"扩展/重排带来的新增命中"。
- **Recall@K**：K ∈ {1,3,5,10,12,20} ∩ [1, len(retrieved)]，外加 "full"。
  AR 每题只有 1 个 gold，故 Recall@K 是 0/1 命中率，不做多 gold 归一。
- **不适用（None）**：gold 是长 rubric（single-session-preference 类）时
  子串匹配无意义，标记 `rubric_not_applicable` 并从分母中剔除。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "normalize_text",
    "flatten_answers",
    "answer_in_texts",
    "core_question",
    "record",
    "compute_recall_at_k",
    "summarize",
    "load_jsonl",
    "enrich_record",
    "RECALL_KS",
]

# ----------------------------- 口径常量 -----------------------------
RECALL_KS: Tuple[int, ...] = (1, 3, 5, 10, 12, 20)
# gold 超过这个长度、或含换行 -> 视为 rubric（偏好类题），子串匹配不适用
RUBRIC_MAX_CHARS = 200
# 记录原文时的单槽位截断（0 = 不截断；仅在 --enrich 需要压缩体积时使用）
DEFAULT_MAX_TEXT_CHARS = 0

CORE_Q_RE = re.compile(r"^\s*[:：]\s*(.+)$", re.S)
_CORE_MARKERS = ("Now Answer the Question", "Question")

# 标点归一策略（与 gold 匹配的鲁棒性直接相关）：
#   - 撇号（' ’ `）直接删除："don't" -> "dont"，与 "dont" 同形；
#   - 连字符/破折号（- – —）折叠为空格："Fairly-OddParents" -> "fairly oddparents"，
#     这样 gold "The Fairly OddParents" 也能命中连字符写法（避免真命中被漏计）；
#   - 其余非 [字母/数字/空白] 折叠为空格。
_APOS_RE = re.compile(r"['`\u2018\u2019\u00b4]", re.UNICODE)
_DASH_RE = re.compile(r"[-\u2010-\u2015\u2212]", re.UNICODE)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)


# ----------------------------- 文本归一 -----------------------------
def normalize_text(s: Any) -> str:
    """小写 + NFKC + 撇号删除 + 连字符折叠为空格 + 其余标点折叠为空格。

    数字/单位原样保留（不做四舍五入、不做单位换算），保证 "18 days" 与
    "18 day" 不同形——宁可漏判也不放宽到误判。
    """
    if s is None:
        return ""
    t = unicodedata.normalize("NFKC", str(s)).lower()
    t = _APOS_RE.sub("", t)
    t = _DASH_RE.sub(" ", t)
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def flatten_answers(answer: Any) -> List[str]:
    """answers 字段可能是 str / list[str] / list[list[str]]，统一展平为非空字符串。"""
    out: List[str] = []

    def _walk(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, (list, tuple, set)):
            for y in x:
                _walk(y)
            return
        s = str(x).strip()
        if s:
            out.append(s)

    _walk(answer)
    # 去重保序
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def is_rubric_answer(answer: Any) -> bool:
    """偏好类题的 answer 是一段 rubric（长文本/多行），子串匹配无意义。"""
    golds = flatten_answers(answer)
    if not golds:
        return True
    if any("\n" in g for g in golds):
        return True
    return all(len(g) > RUBRIC_MAX_CHARS for g in golds)


def _text_hits_gold(text: str, golds_norm: Sequence[str]) -> List[str]:
    nt = normalize_text(text)
    if not nt:
        return []
    return [g for g in golds_norm if g and g in nt]


def answer_in_texts(answer: Any, texts: Sequence[str]) -> Tuple[Optional[bool], List[str], Optional[int]]:
    """gold 是否出现在 texts 任一文本中。

    返回 (hit, matched_gold_variants, first_hit_index)：
      - hit=None  -> 不适用（rubric gold 或 texts 为空）
      - hit=False -> 可计算但未命中（matched=[]，idx=None）
    """
    if is_rubric_answer(answer):
        return None, [], None
    if not texts:
        return None, [], None
    golds_norm = [normalize_text(g) for g in flatten_answers(answer)]
    golds_norm = [g for g in golds_norm if g]
    if not golds_norm:
        return None, [], None
    matched: List[str] = []
    first: Optional[int] = None
    for i, t in enumerate(texts):
        hit = _text_hits_gold(t, golds_norm)
        if hit:
            if first is None:
                first = i
            for g in hit:
                if g not in matched:
                    matched.append(g)
    return bool(matched), matched, first


def compute_recall_at_k(hit_flags: Sequence[int], ks: Sequence[int] = RECALL_KS) -> Dict[str, int]:
    """hit_flags[i] 表示分数序第 i+1 个槽位是否命中 gold（0/1）。

    返回 {"1":0/1, "3":..., "full":...}；K > len(hit_flags) 时用 full 代替，
    避免把"检索池本来就只有 5 条"误算成 Recall@20=0。
    """
    n = len(hit_flags)
    out: Dict[str, int] = {}
    for k in ks:
        kk = min(k, n) if n else 0
        out[str(k)] = int(any(hit_flags[:kk])) if kk > 0 else 0
    out["full"] = int(any(hit_flags)) if n else 0
    return out


def core_question(q: str) -> str:
    """剥掉 runner 查询模板外壳，取真正的问句（供 judge / 诊断复用）。

    模板可能同时出现 "Question: ..." 与 "Now Answer the Question: ..."，
    因此按 marker 从右往左找（优先 "Now Answer the Question"），取最后一个。
    """
    if not isinstance(q, str):
        return str(q or "")
    s = q
    for marker in _CORE_MARKERS:
        i = s.rfind(marker)
        if i >= 0:
            m = CORE_Q_RE.match(s[i + len(marker):])
            if m:
                return m.group(1).strip()
    return s.strip()


# ----------------------------- 记录侧 -----------------------------
def _pool_text(pool: Any, idx: int) -> str:
    texts = getattr(pool, "texts", None) or []
    if 0 <= idx < len(texts):
        return str(texts[idx])
    return ""


def _pool_start(pool: Any, idx: int) -> Optional[int]:
    metas = getattr(pool, "metas", None) or []
    if 0 <= idx < len(metas) and isinstance(metas[idx], dict):
        st = metas[idx].get("start")
        return int(st) if isinstance(st, (int, float)) else None
    return None


def _norm_hits(hits: Any) -> List[Tuple[int, Optional[float]]]:
    """兼容多种 hits 形态，避免"检索其实成功但诊断记成 no_hits"的静默失真：

        [(idx, score), ...] / [idx, ...] / [{"idx":..,"score":..}, ...]
        torch.Tensor [K,2] / [K]（逐行 iterable 也可）
    """
    out: List[Tuple[int, Optional[float]]] = []
    for h in hits or []:
        if isinstance(h, (list, tuple)) and h:
            idx = int(h[0])
            score = float(h[1]) if len(h) > 1 and h[1] is not None else None
            out.append((idx, score))
        elif isinstance(h, dict):
            idx = h.get("idx", h.get("index", h.get("slot_id")))
            if idx is None:
                continue
            sc = h.get("score")
            out.append((int(idx), float(sc) if sc is not None else None))
        else:
            # tensor / numpy 行 / 0-dim 标量
            try:
                out.append((int(h), None))
                continue
            except (TypeError, ValueError):
                pass
            try:
                seq = list(h)
            except TypeError:
                continue
            if not seq:
                continue
            idx = int(seq[0])
            score = float(seq[1]) if len(seq) > 1 and seq[1] is not None else None
            out.append((idx, score))
    return out


def record(
    pool: Any = None,
    hits: Any = None,
    used_ids: Optional[Sequence[int]] = None,
    answer: Any = None,
    query: str = "",
    texts: Optional[Sequence[str]] = None,
    used_texts: Optional[Sequence[str]] = None,
    ks: Sequence[int] = RECALL_KS,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    flat: bool = False,
) -> Dict[str, Any]:
    """构造逐题诊断字段。

    参数
    ----
    pool        : 检索池对象（需有 .texts；v8/v9 池额外有 .metas 提供 start）
    hits        : retrieve() 的返回值，**必须是分数降序**（Recall@K 依赖此顺序）
    used_ids    : 实际拼进 prompt 的槽位下标（原文序/邻域扩展后）
    answer      : gold（str / list[str] / list[list[str]]）
    query       : 原始 query（仅落盘备查）
    texts       : 直接给槽位原文时可不传 pool（离线场景）
    used_texts  : 直接给拼块原文时可不传 pool/used_ids
    flat        : True -> 平铺字段；False -> {"diag": {...}}（便于嵌进 retrieval 子字典）
    """
    pairs = _norm_hits(hits)

    if texts is None:
        texts = [_pool_text(pool, i) for i, _ in pairs] if pool is not None else []
    else:
        texts = [str(t) for t in texts]

    if used_texts is None and pool is not None and used_ids is not None:
        used_texts = [_pool_text(pool, int(i)) for i in used_ids]
    used_texts = [str(t) for t in (used_texts or [])]

    def _clip(s: str) -> str:
        if max_text_chars and len(s) > max_text_chars:
            return s[:max_text_chars]
        return s

    ids = [i for i, _ in pairs]
    scores = [s for _, s in pairs]
    starts = [_pool_start(pool, i) if pool is not None else None for i in ids]

    if not pairs:
        status = "no_hits"
    elif not any(texts):
        status = "no_texts"
    elif is_rubric_answer(answer):
        status = "rubric_not_applicable"
    else:
        status = "ok"

    # 逐槽位命中标记（分数序），用于 Recall@K
    hit_flags: List[int] = [0] * len(pairs)
    hit_slot: Optional[int] = None
    matched: List[str] = []
    in_retrieved: Optional[bool] = None
    in_context: Optional[bool] = None

    if status == "ok":
        golds_norm = [g for g in (normalize_text(x) for x in flatten_answers(answer)) if g]
        for i, t in enumerate(texts):
            if _text_hits_gold(t, golds_norm):
                hit_flags[i] = 1
                if hit_slot is None:
                    hit_slot = i
        in_retrieved = bool(any(hit_flags))
        in_context, ctx_matched, _ = answer_in_texts(answer, used_texts)
        matched = ctx_matched if ctx_matched else []
        for i, t in enumerate(texts):
            for g in _text_hits_gold(t, golds_norm):
                if g not in matched:
                    matched.append(g)

    recall_at = compute_recall_at_k(hit_flags, ks) if status == "ok" else {}

    diag: Dict[str, Any] = {
        "query": query,
        "retrieved_slot_ids": ids,
        "retrieved_scores": scores,
        "retrieved_texts": [_clip(t) for t in texts],
        "retrieved_starts": starts,
        "used_slot_ids": [int(i) for i in (used_ids or [])],
        "answer_in_retrieved": in_retrieved,
        "answer_in_context": in_context,
        "answer_hit_slot": hit_slot,
        "answer_matches": matched,
        "recall_at_k": recall_at,
        "diag_status": status,
    }
    if flat:
        diag.pop("query", None)   # query 已由 runner 的 rec["query"] 记录，不重复
        return diag
    return {"diag": diag}


# ----------------------------- 汇总侧 -----------------------------
def _get_diag(rec: Dict[str, Any]) -> Dict[str, Any]:
    """取诊断块：优先 rec["diag"]，其次平铺字段，最后兼容旧 retrieval 子字典。"""
    d = rec.get("diag")
    if isinstance(d, dict) and d:
        return d
    if "answer_in_retrieved" in rec or "retrieved_texts" in rec:
        return {k: rec.get(k) for k in (
            "retrieved_slot_ids", "retrieved_scores", "retrieved_texts",
            "retrieved_starts", "used_slot_ids", "answer_in_retrieved",
            "answer_in_context", "answer_hit_slot", "answer_matches",
            "recall_at_k", "diag_status")}
    r = rec.get("retrieval")
    if isinstance(r, dict) and (r.get("idxs") is not None):
        return {"retrieved_slot_ids": r.get("idxs"), "diag_status": "legacy_idxs_only"}
    return {"diag_status": "legacy_idxs_only"}


def _status_of(d: Dict[str, Any]) -> str:
    st = d.get("diag_status")
    if st:
        return str(st)
    if d.get("retrieved_texts"):
        return "ok"
    if d.get("retrieved_slot_ids"):
        return "no_texts"
    return "legacy_idxs_only"


def _rate(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


def _bucket() -> Dict[str, Any]:
    return {"n": 0, "n_applicable": 0, "retrieved_hit": 0, "context_hit": 0,
            "retrieved_hit_context_miss": 0, "context_hit_retrieved_miss": 0,
            "recall_at_k": {}}


def _accumulate(b: Dict[str, Any], d: Dict[str, Any], ks: Sequence[int]) -> None:
    b["n"] += 1
    if _status_of(d) != "ok":
        return
    b["n_applicable"] += 1
    air = d.get("answer_in_retrieved")
    aic = d.get("answer_in_context")
    if air:
        b["retrieved_hit"] += 1
    if aic:
        b["context_hit"] += 1
    if air and aic is False:
        b["retrieved_hit_context_miss"] += 1
    if aic and air is False:
        b["context_hit_retrieved_miss"] += 1
    rk = d.get("recall_at_k") or {}
    for k in list(ks) + ["full"]:
        key = str(k)
        if key in rk:
            slot = b["recall_at_k"].setdefault(key, {"n": 0, "hit": 0})
            slot["n"] += 1
            slot["hit"] += int(bool(rk[key]))


def _finalize(b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(b)
    out["answer_in_retrieved_rate"] = _rate(b["retrieved_hit"], b["n_applicable"])
    out["answer_in_context_rate"] = _rate(b["context_hit"], b["n_applicable"])
    out["recall_at_k"] = {
        k: {"n": v["n"], "hit": v["hit"], "rate": _rate(v["hit"], v["n"])}
        for k, v in sorted(b["recall_at_k"].items(),
                           key=lambda kv: (kv[0] == "full",
                                           int(kv[0]) if kv[0] != "full" else 0))
    }
    return out


def summarize(records: Iterable[Dict[str, Any]],
              ks: Sequence[int] = RECALL_KS,
              group_by: Sequence[str] = ("question_type", "source")) -> Dict[str, Any]:
    """汇总 Recall@K / answer_in_retrieved，支持按 question_type / source 分组。"""
    overall = _bucket()
    groups: Dict[str, Dict[str, Dict[str, Any]]] = {g: {} for g in group_by}
    group_seen: Dict[str, bool] = {g: False for g in group_by}
    status_counts: Dict[str, int] = {}
    n = 0

    for rec in records:
        n += 1
        d = _get_diag(rec)
        st = _status_of(d)
        status_counts[st] = status_counts.get(st, 0) + 1
        _accumulate(overall, d, ks)
        for g in group_by:
            key = rec.get(g)
            if key in (None, ""):
                key = "unknown"
            else:
                group_seen[g] = True
            _accumulate(groups[g].setdefault(str(key), _bucket()), d, ks)

    return {
        "n_records": n,
        "n_applicable": overall["n_applicable"],
        "diag_status_counts": dict(sorted(status_counts.items())),
        "overall": _finalize(overall),
        # 该字段在记录里完全缺失时（如 AR 结果没有 source），整组丢弃，
        # 避免输出一个只有 "unknown" 的无信息分组。
        "by": {g: {k: _finalize(v) for k, v in sorted(m.items())}
               for g, m in groups.items() if group_seen[g]},
        "recall_ks": list(ks),
    }


# ----------------------------- 离线 I/O -----------------------------
def load_jsonl(paths: Sequence[str]) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"[ERR] 输入文件不存在: {p}")
        with open(p, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[warn] {p}:{ln} 不是合法 JSON，已跳过（{e}）", file=sys.stderr)
    return recs


def enrich_record(rec: Dict[str, Any], ks: Sequence[int] = RECALL_KS,
                  max_text_chars: int = DEFAULT_MAX_TEXT_CHARS) -> Dict[str, Any]:
    """对单条记录重算/补写诊断字段（输入需已有 retrieved_texts）。

    已有 `diag` 且状态为 ok 时只重算 recall_at_k（口径可能变了），
    不覆盖 runner 记录的原值，避免"后处理把运行时事实改写掉"。
    """
    d = dict(_get_diag(rec))
    if _status_of(d) != "ok":
        return rec
    out = dict(rec)
    # 防御：记录里 gold 是 rubric（偏好类）却没有 diag_status 时，
    # 不能按子串匹配算命中，否则会把"本来就不适用"的题算成检索漏召。
    if is_rubric_answer(rec.get("answer")):
        d.update({"diag_status": "rubric_not_applicable", "answer_in_retrieved": None,
                  "answer_in_context": None, "answer_hit_slot": None,
                  "answer_matches": [], "recall_at_k": {}})
        out["diag"] = d
        for k in ("retrieved_slot_ids", "retrieved_scores", "retrieved_texts",
                  "retrieved_starts", "used_slot_ids", "answer_in_retrieved",
                  "answer_in_context", "answer_hit_slot", "answer_matches",
                  "recall_at_k", "diag_status"):
            out.pop(k, None)
        return out
    texts = d.get("retrieved_texts") or []
    # 离线无法还原 used 文本；若 runner 没记 used_texts，answer_in_context 保持原值
    golds_norm = [g for g in (normalize_text(x) for x in flatten_answers(rec.get("answer"))) if g]
    if golds_norm and texts:
        flags = [1 if _text_hits_gold(t, golds_norm) else 0 for t in texts]
        if "answer_in_retrieved" not in d or d.get("answer_in_retrieved") is None:
            d["answer_in_retrieved"] = bool(any(flags))
        d["recall_at_k"] = compute_recall_at_k(flags, ks)
        if d.get("answer_hit_slot") is None and any(flags):
            d["answer_hit_slot"] = flags.index(1)
        # 截断只影响落盘体积，命中判定始终基于未截断原文；
        # 代价是再次 enrich 截断后的记录时，若 gold 恰在截断点之后会漏判。
        if max_text_chars:
            d["retrieved_texts"] = [t[:max_text_chars] if len(t) > max_text_chars else t
                                    for t in texts]
    out["diag"] = d
    # 平铺字段与 diag 可能冲突，统一收进 diag，避免同一事实两处不一致
    for k in ("retrieved_slot_ids", "retrieved_scores", "retrieved_texts",
              "retrieved_starts", "used_slot_ids", "answer_in_retrieved",
              "answer_in_context", "answer_hit_slot", "answer_matches",
              "recall_at_k", "diag_status"):
        out.pop(k, None)
    return out


# ----------------------------- 打印 -----------------------------
def _fmt_rate(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def print_summary(summary: Dict[str, Any], title: str = "") -> None:
    if title:
        print(f"== {title} ==")
    print(f"records={summary['n_records']} applicable={summary['n_applicable']} "
          f"status={summary['diag_status_counts']}")
    o = summary["overall"]
    print(f"answer_in_retrieved={_fmt_rate(o['answer_in_retrieved_rate'])}  "
          f"answer_in_context={_fmt_rate(o['answer_in_context_rate'])}  "
          f"retrieved_hit_ctx_miss={o['retrieved_hit_context_miss']}  "
          f"ctx_hit_retrieved_miss={o['context_hit_retrieved_miss']}")
    rk = o["recall_at_k"]
    if rk:
        print("Recall@K: " + "  ".join(
            f"@{k}={_fmt_rate(v['rate'])}(n={v['n']})" for k, v in rk.items()))
    for g, buckets in summary["by"].items():
        if not buckets:
            continue
        print(f"[by {g}]")
        for key, b in buckets.items():
            rk_s = "  ".join(f"@{k}={_fmt_rate(v['rate'])}" for k, v in b["recall_at_k"].items()
                             if k != "full")
            print(f"  {key:<26} n={b['n_applicable']:<4} "
                  f"air={_fmt_rate(b['answer_in_retrieved_rate'])}  "
                  f"aic={_fmt_rate(b['answer_in_context_rate'])}  {rk_s}")


# ----------------------------- selftest -----------------------------
class _FakePool:
    def __init__(self, texts: List[str], starts: Optional[List[int]] = None):
        self.texts = texts
        self.metas = [{"start": s} for s in (starts or list(range(len(texts))))]


def _selftest() -> int:
    """纯 CPU 自测：口径数学 + 兼容性 + 汇总 + 离线补写。不联网、不加载模型。"""
    checks = []

    def ok(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"  {'ok  ' if cond else 'FAIL'} - {name}")

    # 1. normalize / 展平
    ok("normalize 去标点压空白",
       normalize_text("  The  Fairly-OddParents,  Inc. ") == "the fairly oddparents inc")
    ok("normalize 连字符与空格同形",
       normalize_text("Fairly-OddParents") == normalize_text("Fairly OddParents"))
    ok("normalize 撇号删除",
       normalize_text("don't") == "dont")
    ok("normalize 数字原样", normalize_text("18 days") == "18 days")
    ok("flatten nested", flatten_answers([["a"], ["b", ["c"]]]) == ["a", "b", "c"])
    ok("core_question 剥模板",
       core_question("Question: Who wrote X?\nNow Answer the Question: Who wrote X?")
       == "Who wrote X?")
    ok("core_question 单层模板", core_question("Question: Where is Y?") == "Where is Y?")
    ok("core_question 无模板原样", core_question("Who is Z?") == "Who is Z?")
    ok("rubric 判定（长文本）", is_rubric_answer("x" * (RUBRIC_MAX_CHARS + 1)))
    ok("rubric 判定（短答案）", not is_rubric_answer(["India"]))

    # 2. record：分数序 Recall@K + 命中槽位
    pool = _FakePool([
        "Unrelated sentence about weather.",
        "Nobuhiro Watsuki is famous for The Fairly OddParents.",
        "Another unrelated sentence.",
    ], starts=[0, 1, 2])
    hits = [(1, 0.9), (2, 0.5), (0, 0.4)]
    d = record(pool=pool, hits=hits, used_ids=[0, 1, 2], answer=["The Fairly OddParents"])
    dg = d["diag"]
    ok("retrieved ids 保分数序", dg["retrieved_slot_ids"] == [1, 2, 0])
    ok("scores 对齐", dg["retrieved_scores"] == [0.9, 0.5, 0.4])
    ok("starts 对齐", dg["retrieved_starts"] == [1, 2, 0])
    ok("answer_in_retrieved True", dg["answer_in_retrieved"] is True)
    ok("answer_in_context True", dg["answer_in_context"] is True)
    ok("hit_slot = 0（分数序首位）", dg["answer_hit_slot"] == 0)
    ok("Recall@1 = 1", dg["recall_at_k"]["1"] == 1)
    ok("Recall@3 = 1", dg["recall_at_k"]["3"] == 1)
    ok("status ok", dg["diag_status"] == "ok")

    # 3. 命中在 K 之外：Recall@1=0 / Recall@5=1
    hits2 = [(0, 0.9), (2, 0.8), (1, 0.1)]
    d2 = record(pool=pool, hits=hits2, answer=["The Fairly OddParents"])
    ok("Recall@1=0（gold 排第 3）", d2["diag"]["recall_at_k"]["1"] == 0)
    ok("Recall@3=1", d2["diag"]["recall_at_k"]["3"] == 1)
    ok("Recall@20 用 full 兜底", d2["diag"]["recall_at_k"]["20"] == 1)

    # 4. 未命中 + 不适用
    d3 = record(pool=pool, hits=hits, answer=["India"])
    ok("未命中 answer_in_retrieved False", d3["diag"]["answer_in_retrieved"] is False)
    ok("未命中 recall 全 0", all(v == 0 for v in d3["diag"]["recall_at_k"].values()))
    d4 = record(pool=pool, hits=hits, answer=["x" * (RUBRIC_MAX_CHARS + 1)])
    ok("rubric -> None + status", d4["diag"]["answer_in_retrieved"] is None
       and d4["diag"]["diag_status"] == "rubric_not_applicable")
    d5 = record(pool=pool, hits=[], answer=["India"])
    ok("no_hits status", d5["diag"]["diag_status"] == "no_hits")

    # 5. 兼容 hits 形态
    ok("hits 纯 idx", record(pool=pool, hits=[1, 0], answer=["India"])["diag"]["retrieved_slot_ids"] == [1, 0])
    ok("hits dict", record(pool=pool, hits=[{"idx": 2, "score": 0.3}],
                           answer=["India"])["diag"]["retrieved_scores"] == [0.3])

    class _Tensorish:
        """模拟 torch.Tensor 行：非 list/tuple，但可 int()/可迭代。"""
        def __init__(self, vals):
            self.vals = vals

        def __iter__(self):
            return iter(self.vals)

        def __len__(self):
            return len(self.vals)

        def __int__(self):
            if len(self.vals) != 1:
                raise TypeError("multi-element")
            return int(self.vals[0])

    d6 = record(pool=pool, hits=[_Tensorish([1, 0.9]), _Tensorish([0])],
                answer=["India"])["diag"]
    ok("hits tensor 行（2 列）", d6["retrieved_slot_ids"] == [1, 0]
       and d6["retrieved_scores"] == [0.9, None])
    ok("hits 标量 tensor", record(pool=pool, hits=[_Tensorish([2])],
                                  answer=["India"])["diag"]["retrieved_slot_ids"] == [2])

    # 6. flat 模式字段名
    flat = record(pool=pool, hits=hits, answer=["India"], flat=True)
    ok("flat 有 retrieved_slot_ids", "retrieved_slot_ids" in flat and "diag" not in flat)
    ok("flat 不含 query", "query" not in flat)

    # 7. 汇总：分组 + 分母只算 applicable
    recs = [
        {"question_type": "single-session-user", "source": "longmemeval_s*",
         "diag": record(pool=pool, hits=hits, answer=["The Fairly OddParents"])["diag"]},
        {"question_type": "single-session-user", "source": "longmemeval_s*",
         "diag": record(pool=pool, hits=hits2, answer=["The Fairly OddParents"])["diag"]},
        {"question_type": "single-session-preference", "source": "longmemeval_s*",
         "diag": record(pool=pool, hits=hits, answer=["y" * (RUBRIC_MAX_CHARS + 1)])["diag"]},
        {"question_type": "temporal-reasoning", "source": "longmemeval_s*",
         "retrieval": {"idxs": [0, 1]}},   # 旧记录：只有 idxs
    ]
    s = summarize(recs)
    ok("汇总 n_records=4", s["n_records"] == 4)
    ok("汇总 n_applicable=2（rubric+legacy 剔除）", s["n_applicable"] == 2)
    ok("汇总 Recall@1 = 0.5", abs(s["overall"]["recall_at_k"]["1"]["rate"] - 0.5) < 1e-9)
    ok("汇总 Recall@3 = 1.0", abs(s["overall"]["recall_at_k"]["3"]["rate"] - 1.0) < 1e-9)
    ok("分组 single-session-user 存在",
       "single-session-user" in s["by"]["question_type"])
    ok("legacy 记录计数",
       s["diag_status_counts"].get("legacy_idxs_only") == 1
       and s["diag_status_counts"].get("rubric_not_applicable") == 1)
    # 字段整体缺失时整组丢弃（AR 记录没有 source，不该出现只有 unknown 的分组）
    s_nosrc = summarize([{"question_type": "x",
                          "diag": record(pool=pool, hits=hits, answer=["India"])["diag"]}])
    ok("缺失 source 时丢弃该分组", "source" not in s_nosrc["by"])
    ok("缺失分组不影响 overall", s_nosrc["overall"]["n_applicable"] == 1)

    # 8. 离线补写：不改原记录、重算 recall
    #    注意：retrieved_texts 必须与 retrieved_slot_ids 同序（分数序），
    #    这是 record() 的契约；下面按 ids=[1,2,0] 重排文本构造真实形态。
    order = [1, 2, 0]
    legacy_flat = {"answer": ["The Fairly OddParents"],
                   "retrieved_texts": [pool.texts[i] for i in order],
                   "retrieved_slot_ids": order,
                   "diag_status": "ok"}
    e = enrich_record(legacy_flat)
    ok("enrich 收进 diag", "diag" in e and "retrieved_texts" not in e)
    ok("enrich 重算 recall@1=1", e["diag"]["recall_at_k"]["1"] == 1)
    ok("enrich 不改原对象", "diag" not in legacy_flat)

    failed = [n for n, c in checks if not c]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("ALL AR_DIAG SELFTESTS PASSED")
    return 0


# ----------------------------- CLI -----------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AR 逐题诊断：Recall@K / answer_in_retrieved")
    ap.add_argument("--input", nargs="*", default=[], help="AR 结果 JSONL（可多个）")
    ap.add_argument("--enrich", action="store_true", help="补写诊断字段到 --enrich-out")
    ap.add_argument("--enrich-out", default="", help="补写输出路径（默认 <input>.diag.jsonl）")
    ap.add_argument("--summary-out", default="", help="汇总 JSON 落盘路径")
    ap.add_argument("--ks", default=",".join(str(k) for k in RECALL_KS),
                    help="Recall@K 的 K 列表，逗号分隔（默认 1,3,5,10,12,20）")
    ap.add_argument("--max-text-chars", type=int, default=DEFAULT_MAX_TEXT_CHARS,
                    help="记录原文单槽位截断（0=不截断）")
    ap.add_argument("--quiet", action="store_true", help="只写文件不打印")
    ap.add_argument("--selftest", action="store_true", help="纯 CPU 自测后退出")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if not args.input:
        ap.error("需要 --input（或 --selftest）")

    try:
        ks = tuple(int(x) for x in str(args.ks).split(",") if str(x).strip())
    except ValueError:
        raise SystemExit(f"[ERR] --ks 非法: {args.ks}")

    recs = load_jsonl(args.input)
    summary = summarize(recs, ks=ks)
    if not args.quiet:
        print_summary(summary, title=", ".join(os.path.basename(p) for p in args.input))

    if args.summary_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_out)), exist_ok=True)
        with open(args.summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        if not args.quiet:
            print(f"[out] summary -> {args.summary_out}")

    if args.enrich:
        if len(args.input) != 1:
            raise SystemExit("[ERR] --enrich 一次只接受一个 --input")
        src = args.input[0]
        out = args.enrich_out or (os.path.splitext(src)[0] + ".diag.jsonl")
        n_ok = n_skip = 0
        with open(out, "w", encoding="utf-8") as f:
            for rec in recs:
                new = enrich_record(rec, ks=ks, max_text_chars=args.max_text_chars)
                if _status_of(_get_diag(new)) == "ok":
                    n_ok += 1
                else:
                    n_skip += 1
                f.write(json.dumps(new, ensure_ascii=False) + "\n")
        if not args.quiet:
            print(f"[out] enrich -> {out} (ok={n_ok} skipped={n_skip})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
