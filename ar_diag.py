#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ar_diag.py — AR 逐题诊断基建（答案命中代理 recall / answer_in_retrieved）

纯标准库实现，**不 import torch / transformers / deltamem**，因此：
  - 可被 `run_ar_v7.py` 在模块顶层直接 import（不触发模型加载、不占显存）；
  - 可在本机（无 /share 路径、无 GPU）跑 `--selftest` 与离线后处理。

=====================================================================
一、为什么需要它
=====================================================================
现有 AR 结果 JSONL 只记 `retrieval.idxs`（槽位下标），无法回答两个关键问题：
  1. **检索侧**：gold answer 到底有没有出现在 Top-K 检索结果里？
     （answer_recall_at_k：答案命中代理，**不是**证据级召回）
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
    used_slot_ids        list[int]    实际拼进 prompt 的**情景池**槽位（原文序 / 扩展后）
    used_texts           list[str]    实际拼进 prompt 的**逐行文本**（情景 + 状态，与
                                      used_slot_uids 一一对应）；与 retrieved_texts 分开存，
                                      二者不是同一个东西：前者是"喂进去了什么"，
                                      后者是"检索器给出了什么"
    retrieved_slot_uids  list[str]    跨池统一 ID（'epi:3' / 'state:2' / 'cr:0'），
                                      与 retrieved_slot_ids 同序，带池前缀
    used_slot_uids       list[str|None] 跨池统一 ID，与 used_texts 逐行对齐；
                                      无法反查到池内下标时为 None（绝不伪造）
    used_alignment       str           ok | state_ids_recovered | texts_without_ids |
                                      ids_truncated_to_texts（used_slot_uids 与 used_texts
                                      的对齐质量，见 _align_used）
    answer_in_retrieved  bool|None    gold 是否出现在 Top-K 任一槽位文本中
    answer_in_context    bool|None    gold 是否出现在实际拼块文本中
    answer_hit_slot      int|None     首个命中槽位（分数序）
    answer_matches       list[str]    命中的 gold 变体
    answer_recall_at_k  dict[str,int] {"1":0/1, "3":..., "full":...}（仅 applicable 时有值）
                                      答案命中代理：gold 答案字符串是否出现在 Top-K 槽位文本里
    evidence_recall_at_k dict|None    证据级召回：Top-K 槽位 id 是否命中 gold 证据 id
                                      **当前管线没有 gold 证据 span，默认 None（不伪造数值）**
    recall_at_k          dict[str,int] 已废弃：等价于 answer_recall_at_k，仅为旧 JSONL 兼容保留
    diag_status          str          ok | no_hits | no_texts | rubric_not_applicable

兼容 v7 `ChunkPool`（只有 `.texts`）与 v8/v9 池（`.texts` + `.metas`）。
`hits` 也接受 [idx, ...]（无分数）或 [(idx, score), ...]。

跨池统一槽位 ID（为什么需要）：
    情景池（EpisodicIndex）与状态池（StatePool / CRStatePool）各有自己的 0 基下标，
    同一个整数 `3` 在两张表里指向完全不同的文本，跨池证据无法对齐。故定义带池前缀的
    ID：`epi:<idx>` / `state:<idx>` / `cr:<idx>`（`slot_uid()` / `parse_slot_uid()`）。
    record() 里 `used_slot_ids` / `retrieved_slot_ids` 保持**旧语义（纯 int）不变**，
    新信息一律写进新字段 `used_slot_uids` / `retrieved_slot_uids`（只增不改，旧 JSONL
    消费方不受影响）。`pool_kind` 可显式指定，缺省按池类名推断（CR* -> cr，
    *State* -> state，其余 -> epi）。

=====================================================================
三、用法 B：离线后处理（对已跑完的 JSONL 补诊断）
=====================================================================
    # 汇总答案命中代理 recall（answer_recall_at_k；旧名 recall_at_k 仍可读）
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
- **answer_recall_at_k**（答案命中代理 recall）：K ∈ {1,3,5,10,12,20} ∩ [1, len(retrieved)]，
  外加 "full"。AR 每题只有 1 个 gold，故它是 0/1 命中率，不做多 gold 归一。
  它只回答「gold 答案字符串是否出现在 Top-K 槽位文本里」，**不等于证据级召回**：
  答案原文可能出现在非证据槽位，证据槽位也可能不含答案原文。
  旧名 `recall_at_k` 是它的废弃别名（同值），仅为兼容旧 JSONL 保留。
- **evidence_recall_at_k**（证据级召回）：只看分数序 Top-K 的槽位 id 是否命中
  gold 证据槽位 id（`evidence_ids`）。这需要逐题标注 gold 证据 span，
  **当前管线没有该标注**，故 `record()` 默认输出 None，绝不伪造数值；
  只有调用方显式传入 `evidence_ids` 时才有值，汇总只在有样本时输出。
- **不适用（None）**：gold 是长 rubric（single-session-preference 类）时
  子串匹配无意义，标记 `rubric_not_applicable` 并从分母中剔除。

=====================================================================
五、联合归因：Retrieval / Reader 四格表（summarize 的 `joint` 字段）
=====================================================================
只有命中率不够：必须把「检索没捞到」和「捞到了但没读出来」分开，否则无法判断
该改检索器还是改 Reader。故按 (answer_recall_at_k[k]=1 与否) × (SF 对/错) 分四格：

                  SF=1（读出对）      SF=0（读出错）
    recall=1      A 读到且答对          B 读到但答错
    recall=0      C 没检索到但答对      D 检索失败且答错

输出量（都在 `joint` 里，含 n 与 excluded 计数）：
    cells                      A/B/C/D 的样本数与占比
    reader_acc_given_recall1   A/(A+B)   ← Reader Acc | Recall=1（条件准确率）
    read_but_wrong_rate        B/n       ← 读到但答错率
    retrieval_failure_rate     (C+D)/n   ← 检索失败率
    retrieval_failure_and_wrong_rate  D/n
    sf_acc / answer_recall_rate      四格边际（SF 准确率 / 答案命中率）
    error_attribution         答错的样本里检索侧（D）与读出侧（B）各占多少
    evidence_based / evidence_note   是否证据级口径 + 必读告警

口径约束（硬性）：
  - 检索侧判据默认 answer_recall@1（`joint_k` 可换 K 或 "full"）；SF 默认取记录上的
    `substring_exact_match`（`sf_field` 可换）。SF 接受 bool / 0-1 / 0-100 记法。
  - **证据级召回为 None 时绝不冒充**：`evidence_recall_at_k` 全为 None 时
    `evidence_based=False` 且 `evidence_note` 明写"无金标证据位置，联合表仅基于
    答案命中代理"；只有真有 gold 证据标注时才额外输出 `evidence_cells`
    （证据级四格，与答案代理四格分开统计，不混用）。
  - 只有 `diag_status=ok` 的记录进四格；缺 recall / 缺 SF 的记录进 `excluded`
    计数而不是被当成"检索失败"，legacy（只有 retrieval.idxs）记录完全不进表。
  - 默认按 question_type 与 context_id 分组（`joint_by`，各组带 n）；context 组
    可能极多，超过 `joint_max_groups`（默认 200）时按 n 降序截断并在
    `joint_truncated` 里显式说明，不静默丢组。
  - `tools/summarize_sweep.py` 复用同一个 `joint_table()`，保证两个脚本口径一致。
"""

from __future__ import annotations

import argparse
import io
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
    "POOL_KINDS",
    "slot_uid",
    "parse_slot_uid",
    "joint_table",
    "JOINT_SF_FIELD",
    "JOINT_EVIDENCE_NOTE",
]

# ----------------------------- 口径常量 -----------------------------
RECALL_KS: Tuple[int, ...] = (1, 3, 5, 10, 12, 20)
# gold 超过这个长度、或含换行 -> 视为 rubric（偏好类题），子串匹配不适用
RUBRIC_MAX_CHARS = 200
# 记录原文时的单槽位截断（0 = 不截断；仅在 --enrich 需要压缩体积时使用）
DEFAULT_MAX_TEXT_CHARS = 0

# ---- 联合归因（Retrieval / Reader 四格表）口径 ----
# SF（读出是否正确）的字段名：runner 用官方 post_process 的 substring_exact_match（bool 或 0/1）
JOINT_SF_FIELD = "substring_exact_match"
# 四格表的检索侧判据默认用 answer_recall@1（Top-1 是否命中答案串）
JOINT_RECALL_K = 1
# 默认按这些维度给出分组联合表
JOINT_GROUP_BY: Tuple[str, ...] = ("question_type", "context_id")
# context_id 分组可能极多（LongMemEval 每个 context 一组），默认只保留 n 最大的前 200 组
JOINT_MAX_GROUPS = 200
# evidence_recall_at_k 为 None 时必须在输出里显式标注，避免把答案命中代理当成证据级召回
JOINT_EVIDENCE_NOTE = (
    "无金标证据位置：evidence_recall_at_k 全为 None（当前管线没有 gold 证据 span），"
    "联合表仅基于答案命中代理（answer_recall_at_k），不得当作证据级召回使用"
)
JOINT_EVIDENCE_NOTE_OK = (
    "evidence_recall_at_k 有值：evidence_cells 用证据级召回算，answer_cells 用答案命中代理算，"
    "两者不可混用"
)

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


# ------------------------- 跨池统一槽位 ID -------------------------
# 情景池 / 状态池 / CR 池各有自己的 0 基下标，整数 3 在三张表里指向不同文本，
# 跨池证据无法对齐。统一 ID = 池前缀 + ':' + 池内下标，如 'epi:3' / 'state:0'。
POOL_KINDS: Tuple[str, ...] = ("epi", "state", "cr")


def _norm_pool_kind(kind: Any) -> str:
    k = str(kind or "epi").strip().lower()
    if k not in POOL_KINDS:
        raise ValueError(f"未知池前缀 {kind!r}；只支持 {POOL_KINDS}")
    return k


def slot_uid(kind: Any, idx: Any) -> str:
    """构造统一槽位 ID：slot_uid('epi', 3) -> 'epi:3'。"""
    return f"{_norm_pool_kind(kind)}:{int(idx)}"


def parse_slot_uid(uid: Any, default_kind: Any = "epi") -> Tuple[str, int]:
    """解析统一槽位 ID，返回 (kind, idx)。

    兼容形态：'epi:3' / 'state:0' / 'cr:12'（带前缀）、3 / '3' / numpy.int64（纯下标，
    按 default_kind 归属）、{"uid": "epi:3"} / {"idx": 3} / {"slot_id": 3}（dict）。
    """
    if isinstance(uid, dict):
        for key in ("uid", "slot_uid", "slot_id", "idx", "index"):
            if key in uid:
                uid = uid[key]
                break
        else:
            raise ValueError(f"无法从 dict 解析槽位 ID：{uid!r}")
    s = str(uid).strip()
    if ":" in s:
        k, _, rest = s.partition(":")
        return _norm_pool_kind(k), int(rest)
    return _norm_pool_kind(default_kind), int(s)


def _infer_pool_kind(pool: Any) -> str:
    """按池类名推断池前缀：CRStatePool -> cr，StatePool -> state，其余 -> epi。"""
    name = type(pool).__name__ if pool is not None else ""
    up = name.upper()
    if up.startswith("CR") or "CRSTATE" in up:
        return "cr"
    if "STATE" in up:
        return "state"
    return "epi"


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


def _align_used(uids: List[Optional[str]], texts: List[str],
                state_pool: Any = None, pool: Any = None,
                pool_kind: str = "epi") -> Tuple[List[Optional[str]], str]:
    """让 used_slot_uids 与 used_texts 逐行对齐，返回 (uids, alignment_status)。

    拼块顺序是「情景行在前、状态行在后」（build_block_v8 / build_block_v9 的固定顺序），
    所以文本比 ID 多时，多出来的尾部行按文本反查池下标：先主池（epi/cr），再 StatePool。
    反查不到就补 None —— **绝不伪造下标**，只把对齐质量写进 used_alignment。
    """
    if len(texts) == len(uids):
        return uids, "ok"
    if len(texts) < len(uids):
        return uids[:len(texts)], "ids_truncated_to_texts"
    out: List[Optional[str]] = list(uids)

    def _idx_map(p: Any) -> Dict[str, int]:
        m: Dict[str, int] = {}
        for i, t in enumerate(getattr(p, "texts", None) or []):
            m.setdefault(str(t), i)
        return m

    main_map = _idx_map(pool) if pool is not None else {}
    state_map = _idx_map(state_pool) if state_pool is not None else {}
    recovered = unresolved = 0
    for t in texts[len(uids):]:
        s = str(t)
        if s in main_map:
            out.append(slot_uid(pool_kind, main_map[s]))
            recovered += 1
        elif s in state_map:
            out.append(slot_uid("state", state_map[s]))
            recovered += 1
        else:
            out.append(None)
            unresolved += 1
    if unresolved:
        return out, "texts_without_ids"
    return out, ("state_ids_recovered" if recovered else "ok")


def record(
    pool: Any = None,
    hits: Any = None,
    used_ids: Optional[Sequence[Any]] = None,
    answer: Any = None,
    query: str = "",
    texts: Optional[Sequence[str]] = None,
    used_texts: Optional[Sequence[str]] = None,
    ks: Sequence[int] = RECALL_KS,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    flat: bool = False,
    evidence_ids: Optional[Sequence[int]] = None,
    used_state_ids: Optional[Sequence[Any]] = None,
    state_pool: Any = None,
    pool_kind: Optional[str] = None,
    block: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构造逐题诊断字段。

    参数
    ----
    pool        : 检索池对象（需有 .texts；v8/v9 池额外有 .metas 提供 start）
    hits        : retrieve() 的返回值，**必须是分数降序**（Recall@K 依赖此顺序）
    used_ids    : 实际拼进 prompt 的槽位下标（原文序/邻域扩展后）；可传纯 int、
                  带前缀的 'epi:3' / 'state:2'，或 {"idx": 3} 形态
    answer      : gold（str / list[str] / list[list[str]]）
    query       : 原始 query（仅落盘备查）
    texts       : 直接给槽位原文时可不传 pool（离线场景）
    used_texts  : 直接给拼块原文时可不传 pool/used_ids；**会作为独立字段落盘**
    flat        : True -> 平铺字段；False -> {"diag": {...}}（便于嵌进 retrieval 子字典）
    evidence_ids: gold 证据槽位 id 列表（可选）。给定时按分数序 Top-K 的槽位 id
                  是否命中这些 id 计算 evidence_recall_at_k；未给定时该字段为 None。
    used_state_ids: 状态池侧的槽位下标（可选）。runner 拼块是「情景行 + 状态行」，
                  used_ids 只含情景下标，状态行通过它补成 'state:<idx>'。
    state_pool  : 状态池对象（用于把状态行反查成统一 ID；缺省则尾部行记 None）
    pool_kind   : 主检索池的池前缀 'epi'|'state'|'cr'；缺省按池类名推断
    block       : 直接给 runner 的拼块 dict（pool/hits/used_ids/used_texts/
                  used_state_ids/state_pool/pool_kind），显式参数优先。
                  run_ar_v7 用 `record(**diag_ctx, ...)` 调用，新增键自动生效。

    口径说明
    --------
    answer_recall_at_k 是**答案命中代理**：只判断 gold 答案字符串是否出现在 Top-K
    槽位文本里，不等于证据级召回。evidence_recall_at_k 才是证据级召回，但它需要
    每题标注 gold 证据 span——**当前管线没有该标注，因此默认必须是 None，
    这里绝不伪造证据级召回数值**。

    兼容性
    ------
    `used_slot_ids` / `retrieved_slot_ids` 保持旧语义（纯 int、主池下标）不变；
    跨池统一 ID 一律写进新字段 `used_slot_uids` / `retrieved_slot_uids`（只增不改）。
    """
    if block:
        if pool is None:
            pool = block.get("pool")
        if hits is None:
            hits = block.get("hits")
        if used_ids is None:
            used_ids = block.get("used_ids")
        if used_texts is None:
            used_texts = block.get("used_texts")
        if used_state_ids is None:
            used_state_ids = block.get("used_state_ids")
        if state_pool is None:
            state_pool = block.get("state_pool")
        if pool_kind is None:
            pool_kind = block.get("pool_kind")

    pool_kind = _norm_pool_kind(pool_kind) if pool_kind else _infer_pool_kind(pool)
    pairs = _norm_hits(hits)

    if texts is None:
        texts = [_pool_text(pool, i) for i, _ in pairs] if pool is not None else []
    else:
        texts = [str(t) for t in texts]

    # ---- 统一槽位 ID：used 侧（主池 + 状态池）----
    used_pairs: List[Tuple[str, int]] = [parse_slot_uid(u, pool_kind) for u in (used_ids or [])]
    used_pairs += [parse_slot_uid(u, "state") for u in (used_state_ids or [])]
    used_uids: List[Optional[str]] = [slot_uid(k, i) for k, i in used_pairs]
    # 旧字段：只保留主池下标（与历史行为完全一致）
    used_ids_legacy = [int(i) for k, i in used_pairs if k == pool_kind]

    # used 侧文本：优先用调用方给的拼块原文（那是真正进 prompt 的行），
    # 缺的尾部行按 ID 从池里补；补不出来就保持缺省，交给 _align_used 显式标记。
    pools = {pool_kind: pool, "state": state_pool}
    resolved = []
    for k, i in used_pairs:
        p = pools.get(k)
        resolved.append(_pool_text(p, i) if p is not None else "")
    # 一条都解析不出来 -> 保持"无 used 文本"，answer_in_context 仍是 None（未知），
    # 不能退化成 False（会把"无法判断"伪造成"没命中"）。
    if not any(resolved):
        resolved = []
    if used_texts is None:
        used_texts = resolved
    else:
        used_texts = [str(t) for t in used_texts]
        if len(used_texts) < len(resolved):
            for t in resolved[len(used_texts):]:
                if not t:
                    break
                used_texts.append(t)
    used_texts = [str(t) for t in (used_texts or [])]
    # 文本多于 ID（runner 的状态行没有下标）-> 用 StatePool 反查补齐，不伪造
    used_uids, used_alignment = _align_used(used_uids, used_texts, state_pool,
                                           pool=pool, pool_kind=pool_kind)

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

    # 答案命中代理 recall：gold 答案字符串是否出现在 Top-K 槽位文本里
    answer_recall_at = compute_recall_at_k(hit_flags, ks) if status == "ok" else {}
    # 证据级召回：分数序 Top-K 的槽位 id 是否命中 gold 证据 id。
    # 当前管线没有 gold 证据 span -> evidence_ids 默认 None -> 字段保持 None，绝不伪造数值。
    evidence_recall_at: Optional[Dict[str, int]] = None
    if evidence_ids is not None and pairs:
        gold_evidence = {int(i) for i in evidence_ids}
        ev_flags = [1 if i in gold_evidence else 0 for i in ids]
        evidence_recall_at = compute_recall_at_k(ev_flags, ks)

    diag: Dict[str, Any] = {
        "query": query,
        "retrieved_slot_ids": ids,
        "retrieved_scores": scores,
        "retrieved_texts": [_clip(t) for t in texts],
        "retrieved_starts": starts,
        "retrieved_slot_uids": [slot_uid(pool_kind, i) for i in ids],
        # used_slot_ids 保持旧语义（主池纯下标）；used_texts / used_slot_uids 是新字段，
        # 二者逐行一一对应（状态行补 'state:<idx>'），与 retrieved_* 分开存。
        "used_slot_ids": used_ids_legacy,
        "used_texts": [_clip(t) for t in used_texts],
        "used_slot_uids": used_uids,
        "used_alignment": used_alignment,
        "answer_in_retrieved": in_retrieved,
        "answer_in_context": in_context,
        "answer_hit_slot": hit_slot,
        "answer_matches": matched,
        "answer_recall_at_k": answer_recall_at,
        "evidence_recall_at_k": evidence_recall_at,
        "recall_at_k": answer_recall_at,  # 已废弃：等价于 answer_recall_at_k，仅为旧 JSONL 兼容保留
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
            "retrieved_starts", "retrieved_slot_uids", "used_slot_ids",
            "used_texts", "used_slot_uids", "used_alignment",
            "answer_in_retrieved",
            "answer_in_context", "answer_hit_slot", "answer_matches",
            "answer_recall_at_k", "evidence_recall_at_k",
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
            "recall_at_k": {}, "evidence_recall_at_k": {},
            "joint": _joint_bucket()}


def _accum_recall(bucket: Dict[str, Dict[str, int]],
                  rk: Dict[str, Any], ks: Sequence[int]) -> None:
    """把一个 {"1":0/1,...,"full":...} 累加进 bucket（每个 K 独立分母）。"""
    for k in list(ks) + ["full"]:
        key = str(k)
        if key in rk:
            slot = bucket.setdefault(key, {"n": 0, "hit": 0})
            slot["n"] += 1
            slot["hit"] += int(bool(rk[key]))


def _accumulate(b: Dict[str, Any], d: Dict[str, Any], ks: Sequence[int],
                rec: Optional[Dict[str, Any]] = None,
                joint_k: Any = JOINT_RECALL_K,
                sf_field: str = JOINT_SF_FIELD) -> None:
    b["n"] += 1
    st = _status_of(d)
    # 联合归因四格（Retrieval × Reader）：先累加，非 applicable 的记录记进 excluded
    _accum_joint(b["joint"], d, rec, joint_k, sf_field, status=st)
    if st != "ok":
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
    # 优先新键 answer_recall_at_k；缺失时回退旧键 recall_at_k（旧记录）
    rk = _recall_dict(d)
    _accum_recall(b["recall_at_k"], rk, ks)
    # 证据级召回单独累加：只有真正算过的记录（非 None）才进分母
    erk = d.get("evidence_recall_at_k")
    if erk:
        _accum_recall(b["evidence_recall_at_k"], erk, ks)


def _recall_sort_key(kv: Tuple[str, Any]) -> Tuple[int, int]:
    return (kv[0] == "full", int(kv[0]) if kv[0] != "full" else 0)


def _fmt_recall(bucket: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    return {k: {"n": v["n"], "hit": v["hit"], "rate": _rate(v["hit"], v["n"])}
            for k, v in sorted(bucket.items(), key=_recall_sort_key)}


def _recall_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """取 recall@K 字典：优先新键 answer_recall_at_k，缺失时回退废弃别名 recall_at_k。"""
    rk = d.get("answer_recall_at_k")
    if rk is None:
        rk = d.get("recall_at_k") or {}
    return rk or {}


def _as_flag(v: Any) -> Optional[bool]:
    """把记录里的 0/1、bool、'true'/'false'、0-100 分制指标统一成 bool（未知 -> None）。"""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v) > 0
    s = str(v).strip().lower()
    if s in ("", "none", "null", "nan"):
        return None
    if s in ("true", "yes", "y", "t"):
        return True
    if s in ("false", "no", "n", "f"):
        return False
    try:
        return float(s) > 0
    except ValueError:
        return None


def _joint_bucket() -> Dict[str, int]:
    """四格计数器：r1/r0 = 检索命中与否，sf1/sf0 = 读出对/错。"""
    return {"n": 0, "r1_sf1": 0, "r1_sf0": 0, "r0_sf1": 0, "r0_sf0": 0,
            "no_recall": 0, "no_sf": 0, "not_applicable": 0,
            "ev_n": 0, "ev_r1_sf1": 0, "ev_r1_sf0": 0, "ev_r0_sf1": 0, "ev_r0_sf0": 0}


def _accum_joint(j: Dict[str, int], d: Dict[str, Any], rec: Optional[Dict[str, Any]],
                 recall_k: Any, sf_field: str, status: str = "ok") -> None:
    """累加一条记录到四格表。

    - 检索侧：answer_recall_at_k[recall_k]（默认 @1）是否命中；
    - 读出侧：记录上的 sf_field（默认 substring_exact_match）是否为真。
    证据级召回（evidence_recall_at_k）有值时另算一组 ev_* 格，**不混进答案代理格**。
    """
    if status != "ok":
        # rubric / no_hits / legacy 记录不属于四格的任何一格，单独计数说明
        j["not_applicable"] += 1
        return
    rk = _recall_dict(d)
    rv = rk.get(str(recall_k))
    sfv = _as_flag((rec or {}).get(sf_field))
    ev = d.get("evidence_recall_at_k") or {}
    evv = ev.get(str(recall_k)) if isinstance(ev, dict) else None
    if rv is None and evv is None:
        j["no_recall"] += 1
        return
    if sfv is None:
        j["no_sf"] += 1
        return
    suffix = "_sf1" if sfv else "_sf0"
    if rv is not None:
        j["n"] += 1
        j["r1" + suffix if int(rv) else "r0" + suffix] += 1
    if evv is not None:
        j["ev_n"] += 1
        j["ev_r1" + suffix if int(evv) else "ev_r0" + suffix] += 1


def _cells(a: int, b: int, c: int, dd: int) -> Dict[str, Any]:
    n = a + b + c + dd
    return {
        "recall1_sf1": {"n": a, "share": _rate(a, n)},
        "recall1_sf0": {"n": b, "share": _rate(b, n)},
        "recall0_sf1": {"n": c, "share": _rate(c, n)},
        "recall0_sf0": {"n": dd, "share": _rate(dd, n)},
    }


def _finalize_joint(j: Dict[str, int], recall_k: Any = JOINT_RECALL_K,
                    sf_field: str = JOINT_SF_FIELD) -> Dict[str, Any]:
    """把四格计数器算成可读表：条件准确率 + 检索失败率 + 读到但答错率。"""
    a, b = j["r1_sf1"], j["r1_sf0"]
    c, dd = j["r0_sf1"], j["r0_sf0"]
    n = a + b + c + dd
    ev_a, ev_b = j["ev_r1_sf1"], j["ev_r1_sf0"]
    ev_c, ev_dd = j["ev_r0_sf1"], j["ev_r0_sf0"]
    ev_n = ev_a + ev_b + ev_c + ev_dd
    out: Dict[str, Any] = {
        "recall_k": str(recall_k),
        "sf_field": sf_field,
        "n": n,
        "excluded": {"no_recall": j["no_recall"], "no_sf": j["no_sf"],
                     "not_applicable": j["not_applicable"]},
        # 四格：行 = 检索是否命中（answer_recall_at_k=1/0），列 = 读出对错（SF）
        "cells": _cells(a, b, c, dd),
        # Reader Acc | Recall=1：检索命中时读出答对的条件准确率
        "reader_acc_given_recall1": _rate(a, a + b),
        # 读到但答错：检索命中却答错（占全部四格样本的比例）
        "read_but_wrong_rate": _rate(b, n),
        # 检索失败率：检索未命中（不论读出对错）
        "retrieval_failure_rate": _rate(c + dd, n),
        # 检索失败且答错（"既没检索到也没答对"）
        "retrieval_failure_and_wrong_rate": _rate(dd, n),
        # 参考量：SF 准确率与答案命中率（四格边际）
        "sf_acc": _rate(a + c, n),
        "answer_recall_rate": _rate(a + b, n),
        # 错误归因：答错的样本里，检索侧 vs 读出侧各占多少
        "error_attribution": {
            "n_errors": b + dd,
            "retrieval_side_share": _rate(dd, b + dd),
            "reader_side_share": _rate(b, b + dd),
        },
        "evidence_based": bool(ev_n),
        "evidence_note": JOINT_EVIDENCE_NOTE_OK if ev_n else JOINT_EVIDENCE_NOTE,
    }
    if ev_n:
        # 只有真有 gold 证据标注时才输出证据级四格；否则整块不出现，绝不拿代理冒充
        out["evidence_cells"] = _cells(ev_a, ev_b, ev_c, ev_dd)
        out["evidence_n"] = ev_n
        out["evidence_reader_acc_given_recall1"] = _rate(ev_a, ev_a + ev_b)
    return out


def _finalize(b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(b)
    out["answer_in_retrieved_rate"] = _rate(b["retrieved_hit"], b["n_applicable"])
    out["answer_in_context_rate"] = _rate(b["context_hit"], b["n_applicable"])
    # 规范键：答案命中代理 recall（旧别名 recall_at_k 不再出现在汇总输出里）
    out.pop("recall_at_k", None)
    out["answer_recall_at_k"] = _fmt_recall(b["recall_at_k"])
    # 证据级召回只在有样本时输出（当前管线没有 gold 证据 span -> 通常为空）
    ev = b.get("evidence_recall_at_k") or {}
    if ev:
        out["evidence_recall_at_k"] = _fmt_recall(ev)
    else:
        out.pop("evidence_recall_at_k", None)
    if "joint" in b:
        out["joint"] = _finalize_joint(b["joint"])
    return out


def joint_table(records: Iterable[Dict[str, Any]], recall_k: Any = JOINT_RECALL_K,
                sf_field: str = JOINT_SF_FIELD) -> Dict[str, Any]:
    """只算四格联合表（不跑其它汇总），供 tools/summarize_sweep.py 等复用同一口径。"""
    j = _joint_bucket()
    for rec in records:
        d = _get_diag(rec)
        if _status_of(d) != "ok":
            continue
        _accum_joint(j, d, rec, recall_k, sf_field)
    return _finalize_joint(j, recall_k, sf_field)


def summarize(records: Iterable[Dict[str, Any]],
              ks: Sequence[int] = RECALL_KS,
              group_by: Sequence[str] = ("question_type", "source"),
              joint_k: Any = JOINT_RECALL_K,
              sf_field: str = JOINT_SF_FIELD,
              joint_group_by: Sequence[str] = JOINT_GROUP_BY,
              joint_max_groups: int = JOINT_MAX_GROUPS) -> Dict[str, Any]:
    """汇总 answer_recall_at_k / evidence_recall_at_k / answer_in_retrieved / 四格联合归因。

    联合归因（joint）：按 (answer_recall_at_k[k]=1 与否) × (SF 对/错) 分四格，
    输出 Reader Acc | Recall=1（条件准确率）、检索失败率、读到但答错率，
    并分别按 question_type / context_id 分组给出样本量。
    """
    recs = list(records)
    overall = _bucket()
    groups: Dict[str, Dict[str, Dict[str, Any]]] = {g: {} for g in group_by}
    group_seen: Dict[str, bool] = {g: False for g in group_by}
    joint_groups: Dict[str, Dict[str, Dict[str, int]]] = {g: {} for g in joint_group_by}
    joint_seen: Dict[str, bool] = {g: False for g in joint_group_by}
    status_counts: Dict[str, int] = {}
    n = 0

    for rec in recs:
        n += 1
        d = _get_diag(rec)
        st = _status_of(d)
        status_counts[st] = status_counts.get(st, 0) + 1
        _accumulate(overall, d, ks, rec=rec, joint_k=joint_k, sf_field=sf_field)
        for g in group_by:
            key = rec.get(g)
            if key in (None, ""):
                key = "unknown"
            else:
                group_seen[g] = True
            _accumulate(groups[g].setdefault(str(key), _bucket()), d, ks,
                        rec=rec, joint_k=joint_k, sf_field=sf_field)
        for g in joint_group_by:
            key = rec.get(g)
            if key in (None, ""):
                key = "unknown"
            else:
                joint_seen[g] = True
            # 与 overall 同口径：只有 applicable（diag_status=ok）的记录才进四格表，
            # 其余记进 excluded（否则 legacy/rubric 会把分母污染成"检索失败"）。
            _accum_joint(joint_groups[g].setdefault(str(key), _joint_bucket()),
                         d, rec, joint_k, sf_field, status=st)

    joint_by: Dict[str, Dict[str, Any]] = {}
    joint_trunc: Dict[str, Dict[str, int]] = {}
    for g, m in joint_groups.items():
        if not joint_seen[g]:
            continue
        tables = {k: _finalize_joint(v, joint_k, sf_field) for k, v in m.items()}
        # 组太多时只保留样本量最大的前 N 组（并在 joint_truncated 里说明，不静默丢）
        ordered = sorted(tables.items(), key=lambda kv: (-kv[1]["n"], kv[0]))
        if joint_max_groups and len(ordered) > joint_max_groups:
            joint_trunc[g] = {"kept": int(joint_max_groups), "total": len(ordered)}
            ordered = ordered[:joint_max_groups]
        joint_by[g] = dict(ordered)

    overall_out = _finalize(overall)
    return {
        "n_records": n,
        "n_applicable": overall["n_applicable"],
        "diag_status_counts": dict(sorted(status_counts.items())),
        "overall": overall_out,
        # 该字段在记录里完全缺失时（如 AR 结果没有 source），整组丢弃，
        # 避免输出一个只有 "unknown" 的无信息分组。
        "by": {g: {k: _finalize(v) for k, v in sorted(m.items())}
               for g, m in groups.items() if group_seen[g]},
        "joint_by": joint_by,
        **({"joint_truncated": joint_trunc} if joint_trunc else {}),
        "joint_config": {
            "recall_k": str(joint_k),
            "sf_field": sf_field,
            "group_by": list(joint_group_by),
            "evidence_based": bool(overall_out["joint"]["evidence_based"]),
            "note": overall_out["joint"]["evidence_note"],
        },
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

    已有 `diag` 且状态为 ok 时只重算 answer_recall_at_k（口径可能变了），
    不覆盖 runner 记录的原值，避免"后处理把运行时事实改写掉"。
    重算时同时写规范键 `answer_recall_at_k` 与废弃别名 `recall_at_k`（同值）。
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
                  "answer_matches": [],
                  "answer_recall_at_k": {}, "recall_at_k": {}})
        out["diag"] = d
        for k in ("retrieved_slot_ids", "retrieved_scores", "retrieved_texts",
                  "retrieved_starts", "retrieved_slot_uids", "used_slot_ids",
                  "used_texts", "used_slot_uids", "used_alignment",
                  "answer_in_retrieved",
                  "answer_in_context", "answer_hit_slot", "answer_matches",
                  "answer_recall_at_k", "evidence_recall_at_k",
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
        d["answer_recall_at_k"] = compute_recall_at_k(flags, ks)
        d["recall_at_k"] = d["answer_recall_at_k"]  # 已废弃：等价于 answer_recall_at_k，仅为旧 JSONL 兼容保留
        if d.get("answer_hit_slot") is None and any(flags):
            d["answer_hit_slot"] = flags.index(1)
        # 截断只影响落盘体积，命中判定始终基于未截断原文；
        # 代价是再次 enrich 截断后的记录时，若 gold 恰在截断点之后会漏判。
        if max_text_chars:
            d["retrieved_texts"] = [t[:max_text_chars] if len(t) > max_text_chars else t
                                    for t in texts]
    # 记录里已有 used_texts（runner 落了真正拼进 prompt 的行）而 answer_in_context 缺失时，
    # 用它补算——**只在缺失时补**，绝不覆盖 runner 记录的运行时事实。
    used_texts = d.get("used_texts") or []
    if used_texts and d.get("answer_in_context") is None and not is_rubric_answer(rec.get("answer")):
        d["answer_in_context"], _m, _i = answer_in_texts(rec.get("answer"), used_texts)
    out["diag"] = d
    # 平铺字段与 diag 可能冲突，统一收进 diag，避免同一事实两处不一致
    for k in ("retrieved_slot_ids", "retrieved_scores", "retrieved_texts",
              "retrieved_starts", "retrieved_slot_uids", "used_slot_ids",
              "used_texts", "used_slot_uids", "used_alignment",
              "answer_in_retrieved",
              "answer_in_context", "answer_hit_slot", "answer_matches",
              "answer_recall_at_k", "evidence_recall_at_k",
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
    rk = o["answer_recall_at_k"]
    if rk:
        print("answer_recall@K（答案命中代理，非证据级召回）: " + "  ".join(
            f"@{k}={_fmt_rate(v['rate'])}(n={v['n']})" for k, v in rk.items()))
    ev = o.get("evidence_recall_at_k") or {}
    if ev:
        print("evidence_recall@K（证据级召回）: " + "  ".join(
            f"@{k}={_fmt_rate(v['rate'])}(n={v['n']})" for k, v in ev.items()))
    _print_joint(o.get("joint"), indent="")
    for g, buckets in summary["by"].items():
        if not buckets:
            continue
        print(f"[by {g}]")
        for key, b in buckets.items():
            rk_s = "  ".join(f"@{k}={_fmt_rate(v['rate'])}" for k, v in b["answer_recall_at_k"].items()
                             if k != "full")
            print(f"  {key:<26} n={b['n_applicable']:<4} "
                  f"air={_fmt_rate(b['answer_in_retrieved_rate'])}  "
                  f"aic={_fmt_rate(b['answer_in_context_rate'])}  {rk_s}")
    for g, tables in (summary.get("joint_by") or {}).items():
        print(f"[joint by {g}]")
        for key, jt in tables.items():
            print(f"  {key:<26} {_fmt_joint_line(jt)}")
    trunc = summary.get("joint_truncated") or {}
    for g, info in trunc.items():
        print(f"[joint by {g}] 仅显示样本量最大的 {info['kept']}/{info['total']} 组")


def _fmt_joint_line(j: Dict[str, Any]) -> str:
    """一行四格表：n / 四格计数 / Reader Acc|Recall=1 / 检索失败率 / 读到但答错率。"""
    c = j["cells"]
    return (f"n={j['n']:<4} "
            f"r1/sf1={c['recall1_sf1']['n']:<3} r1/sf0={c['recall1_sf0']['n']:<3} "
            f"r0/sf1={c['recall0_sf1']['n']:<3} r0/sf0={c['recall0_sf0']['n']:<3} "
            f"RAcc|R1={_fmt_rate(j['reader_acc_given_recall1'])} "
            f"RFail={_fmt_rate(j['retrieval_failure_rate'])} "
            f"RBW={_fmt_rate(j['read_but_wrong_rate'])}")


def _print_joint(j: Optional[Dict[str, Any]], indent: str = "") -> None:
    if not j:
        return
    print(f"{indent}联合归因（retrieval@K={j['recall_k']} × {j['sf_field']}）: {_fmt_joint_line(j)}")
    if j["n"] == 0:
        print(f"{indent}  （无同时具备 recall 与 SF 的样本；excluded={j['excluded']}）")
    print(f"{indent}  [注] {j['evidence_note']}")


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
    ok("answer_recall@1 = 1", dg["answer_recall_at_k"]["1"] == 1)
    ok("answer_recall@3 = 1", dg["answer_recall_at_k"]["3"] == 1)
    ok("answer_recall_at_k 等于废弃别名 recall_at_k",
       dg["answer_recall_at_k"] == dg["recall_at_k"])
    ok("evidence_recall_at_k 默认 None（当前管线无 gold 证据 span）",
       dg["evidence_recall_at_k"] is None)
    ok("status ok", dg["diag_status"] == "ok")

    # 2b. 证据级召回：显式传 evidence_ids 时才计算（基于分数序槽位 id）
    d_ev = record(pool=pool, hits=hits, answer=["The Fairly OddParents"],
                  evidence_ids=[1])["diag"]        # 分数序首位命中证据 -> @1=1
    ok("evidence@1 = 1（证据 id 排首位）", d_ev["evidence_recall_at_k"]["1"] == 1)
    ok("evidence full = 1", d_ev["evidence_recall_at_k"]["full"] == 1)
    d_ev2 = record(pool=pool, hits=hits, answer=["The Fairly OddParents"],
                   evidence_ids=[2])["diag"]       # 证据在第 2 位 -> @1=0 / @3=1
    ok("evidence@1 = 0（证据不在首位）", d_ev2["evidence_recall_at_k"]["1"] == 0)
    ok("evidence@3 = 1", d_ev2["evidence_recall_at_k"]["3"] == 1)
    ok("evidence 不影响 answer_recall",
       d_ev2["answer_recall_at_k"] == d_ev2["recall_at_k"])
    ok("evidence 未命中 -> 全 0 而非 None",
       all(v == 0 for v in record(pool=pool, hits=hits, answer=["India"],
                                  evidence_ids=[99])["diag"]["evidence_recall_at_k"].values()))
    ok("无检索结果时 evidence 仍为 None",
       record(pool=pool, hits=[], answer=["India"],
              evidence_ids=[1])["diag"]["evidence_recall_at_k"] is None)

    # 3. 命中在 K 之外：answer_recall@1=0 / @3=1
    hits2 = [(0, 0.9), (2, 0.8), (1, 0.1)]
    d2 = record(pool=pool, hits=hits2, answer=["The Fairly OddParents"])
    ok("answer_recall@1=0（gold 排第 3）", d2["diag"]["answer_recall_at_k"]["1"] == 0)
    ok("answer_recall@3=1", d2["diag"]["answer_recall_at_k"]["3"] == 1)
    ok("answer_recall@20 用 full 兜底", d2["diag"]["answer_recall_at_k"]["20"] == 1)

    # 4. 未命中 + 不适用
    d3 = record(pool=pool, hits=hits, answer=["India"])
    ok("未命中 answer_in_retrieved False", d3["diag"]["answer_in_retrieved"] is False)
    ok("未命中 answer_recall 全 0",
       all(v == 0 for v in d3["diag"]["answer_recall_at_k"].values()))
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
    ok("flat 含新字段 used_texts/used_slot_uids",
       "used_texts" in flat and "used_slot_uids" in flat and "used_alignment" in flat)

    # 6b. 跨池统一槽位 ID（epi:/state:/cr:）+ used_texts 独立落盘
    ok("slot_uid 构造", slot_uid("epi", 3) == "epi:3"
       and slot_uid("state", 0) == "state:0" and slot_uid("cr", 12) == "cr:12")
    ok("parse_slot_uid 带前缀", parse_slot_uid("state:2") == ("state", 2))
    ok("parse_slot_uid 纯下标用默认池",
       parse_slot_uid(4) == ("epi", 4) and parse_slot_uid("4", "state") == ("state", 4))
    ok("parse_slot_uid dict", parse_slot_uid({"idx": 5}) == ("epi", 5)
       and parse_slot_uid({"uid": "cr:1"}) == ("cr", 1))
    try:
        slot_uid("nope", 0)
        ok("未知池前缀报错", False)
    except ValueError:
        ok("未知池前缀报错", True)

    st_pool = _FakePool(["Alice prefers tea.", "Alice lives in Kyoto."], starts=[10, 11])
    # 模拟 build_block_v9 的拼块 dict：情景行在前、状态行在后（状态行没有下标）
    blk = {"pool": pool, "hits": hits, "used_ids": [0, 1], "state_pool": st_pool,
           "used_texts": [pool.texts[0], pool.texts[1], st_pool.texts[0]]}
    db = record(block=blk, answer=["The Fairly OddParents"], flat=True)
    ok("block dict -> used_texts 落盘", db["used_texts"] == blk["used_texts"])
    ok("used_texts 与 retrieved_texts 分开存",
       db["used_texts"] != db["retrieved_texts"] and len(db["retrieved_texts"]) == 3)
    ok("旧字段 used_slot_ids 语义不变（只含主池下标）",
       db["used_slot_ids"] == [0, 1] and all(isinstance(i, int) for i in db["retrieved_slot_ids"]))
    ok("used_slot_uids 与 used_texts 长度一致",
       len(db["used_slot_uids"]) == len(db["used_texts"]))
    ok("used_slot_uids 带池前缀且逐行对齐",
       db["used_slot_uids"] == ["epi:0", "epi:1", "state:0"])
    ok("状态行文本反查成功 -> alignment 标记",
       db["used_alignment"] == "state_ids_recovered")
    ok("retrieved_slot_uids 带 epi 前缀且与 ids 同序",
       db["retrieved_slot_uids"] == ["epi:1", "epi:2", "epi:0"])
    ok("answer_in_context 用 used_texts 计算（含状态行）", db["answer_in_context"] is True)
    ok("含状态行时 used_texts = 情景行 + 状态行",
       len(db["used_texts"]) == len(db["used_slot_ids"]) + 1)

    # 纯情景拼块（无状态行）：used_texts 与旧字段 used_slot_ids 严格等长
    db0 = record(block={"pool": pool, "hits": hits, "used_ids": [2, 0, 1]},
                 answer=["The Fairly OddParents"], flat=True)
    ok("纯情景拼块 used_texts 与 used_slot_ids 等长",
       len(db0["used_texts"]) == len(db0["used_slot_ids"]) == 3)
    ok("纯情景拼块 used_texts 按 used_slot_ids 取原文",
       db0["used_texts"] == [pool.texts[2], pool.texts[0], pool.texts[1]]
       and db0["used_slot_uids"] == ["epi:2", "epi:0", "epi:1"]
       and db0["used_alignment"] == "ok")

    # 显式给状态池下标：不需要文本反查，顺序仍是「情景行 + 状态行」
    db2 = record(pool=pool, hits=hits, used_ids=[0, 1], used_state_ids=[1],
                 state_pool=st_pool, answer=["India"], flat=True)
    ok("显式 used_state_ids -> state:<idx>",
       db2["used_slot_uids"] == ["epi:0", "epi:1", "state:1"])
    ok("显式 state ids 时文本按池解析",
       db2["used_texts"] == [pool.texts[0], pool.texts[1], st_pool.texts[1]])
    ok("显式 state ids 对齐 ok", db2["used_alignment"] == "ok")

    # used_ids 直接给带前缀字符串（跨池混合）
    db3 = record(pool=pool, hits=hits, used_ids=["epi:2", "state:1"], state_pool=st_pool,
                 answer=["India"], flat=True)
    ok("used_ids 接受带前缀字符串",
       db3["used_slot_uids"] == ["epi:2", "state:1"] and db3["used_slot_ids"] == [2])
    ok("带前缀 used_ids 时文本按池解析",
       db3["used_texts"] == [pool.texts[2], st_pool.texts[1]])

    # CR 池前缀：按类名推断 + 显式 pool_kind
    class _CRStatePool(_FakePool):
        """模拟 v9_cr.CRStatePool（类名含 CR -> 推断 'cr'）。"""

    cr_pool = _CRStatePool(["Alice: tea.", "Alice: Kyoto."], starts=[0, 1])
    db4 = record(pool=cr_pool, hits=[(1, 0.9), (0, 0.5)], used_ids=[0, 1],
                 answer=["India"], flat=True)
    ok("CR 池自动推断前缀 cr",
       db4["retrieved_slot_uids"] == ["cr:1", "cr:0"]
       and db4["used_slot_uids"] == ["cr:0", "cr:1"])
    ok("pool_kind 可显式覆盖",
       record(pool=pool, hits=hits, used_ids=[0], pool_kind="cr",
              answer=["India"], flat=True)["used_slot_uids"] == ["cr:0"])

    # 只有 used_texts（离线场景）：按文本反查统一 ID
    db5 = record(pool=pool, used_texts=[pool.texts[2], st_pool.texts[1]],
                 state_pool=st_pool, answer=["India"], flat=True)
    ok("只有 used_texts 时按文本反查统一 ID",
       db5["used_slot_uids"] == ["epi:2", "state:1"])

    # 文本少于 ID 但池里有原文：按 ID 补齐，仍保持一一对应
    db6 = record(pool=pool, hits=hits, used_ids=[0, 1, 2], used_texts=[pool.texts[0]],
                 answer=["India"], flat=True)
    ok("文本少于 ID 时按池补齐文本",
       db6["used_texts"] == [pool.texts[0], pool.texts[1], pool.texts[2]]
       and db6["used_slot_uids"] == ["epi:0", "epi:1", "epi:2"]
       and db6["used_alignment"] == "ok")
    # 池也没有（无 pool 的离线场景）：截断并显式标记，不静默错位
    db6b = record(used_ids=[0, 1, 2], used_texts=["只有一行"], answer=["India"], flat=True)
    ok("无法补齐时截断并标记",
       db6b["used_slot_uids"] == ["epi:0"]
       and db6b["used_alignment"] == "ids_truncated_to_texts")

    # 反查不到：补 None，绝不伪造下标
    db7 = record(pool=pool, hits=hits, used_ids=[0],
                 used_texts=["完全不在池里的行", "另一行也查不到"],
                 answer=["India"], flat=True)
    ok("无法反查时补 None 不伪造",
       db7["used_slot_uids"] == ["epi:0", None]
       and db7["used_alignment"] == "texts_without_ids")
    ok("新字段可 JSON 序列化",
       json.dumps(db, ensure_ascii=False) == json.dumps(db, ensure_ascii=False)
       and '"used_slot_uids"' in json.dumps(db, ensure_ascii=False))

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
    ok("汇总 Recall@1 = 0.5",
       abs(s["overall"]["answer_recall_at_k"]["1"]["rate"] - 0.5) < 1e-9)
    ok("汇总 Recall@3 = 1.0",
       abs(s["overall"]["answer_recall_at_k"]["3"]["rate"] - 1.0) < 1e-9)
    ok("汇总有 answer_recall_at_k、无废弃别名 recall_at_k",
       "answer_recall_at_k" in s["overall"] and "recall_at_k" not in s["overall"])
    ok("无证据样本时汇总不输出 evidence_recall_at_k",
       "evidence_recall_at_k" not in s["overall"])
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

    # 7b. 汇总：有证据级样本时才输出 evidence_recall_at_k（分母只算有值的记录）
    s_ev = summarize([
        {"question_type": "x", "diag": record(pool=pool, hits=hits,
                                              answer=["The Fairly OddParents"],
                                              evidence_ids=[1])["diag"]},
        {"question_type": "x", "diag": record(pool=pool, hits=hits,
                                              answer=["India"])["diag"]},
    ])
    ok("有证据样本时汇总输出 evidence_recall_at_k",
       abs(s_ev["overall"]["evidence_recall_at_k"]["1"]["rate"] - 1.0) < 1e-9
       and s_ev["overall"]["evidence_recall_at_k"]["1"]["n"] == 1)
    # 7c. 旧记录只有废弃别名 recall_at_k 时仍能汇总（兼容旧 JSONL）
    s_old = summarize([{"diag": {"diag_status": "ok", "answer_in_retrieved": True,
                                 "answer_in_context": True,
                                 "recall_at_k": {"1": 1, "3": 1, "full": 1}}}])
    ok("旧记录只有 recall_at_k 也能汇总",
       s_old["overall"]["answer_recall_at_k"]["1"]["rate"] == 1.0)

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
    ok("enrich 重算 recall@1=1", e["diag"]["answer_recall_at_k"]["1"] == 1)
    ok("enrich 同时写废弃别名",
       e["diag"]["recall_at_k"] == e["diag"]["answer_recall_at_k"])
    ok("enrich 不改原对象", "diag" not in legacy_flat)

    # 8b. 旧记录（无 used_texts）enrich 后不凭空造 used 字段；
    #     有 used_texts 且缺 answer_in_context 时用它补算（不覆盖已有值）
    e2 = enrich_record({"answer": ["India"],
                        "retrieved_texts": [pool.texts[i] for i in order],
                        "retrieved_slot_ids": order, "diag_status": "ok"})
    ok("无 used_texts 时 enrich 不伪造 used 字段（保持空，不造数据）",
       not e2["diag"].get("used_texts") and not e2["diag"].get("used_slot_uids"))
    e3 = enrich_record({"answer": ["The Fairly OddParents"],
                        "retrieved_texts": [pool.texts[0]],
                        "retrieved_slot_ids": [0], "used_texts": [pool.texts[1]],
                        "used_slot_uids": ["epi:1"], "diag_status": "ok"})
    ok("有 used_texts 时 enrich 补算 answer_in_context",
       e3["diag"]["answer_in_context"] is True
       and e3["diag"]["used_texts"] == [pool.texts[1]]
       and e3["diag"]["used_slot_uids"] == ["epi:1"])
    e4 = enrich_record({"answer": ["India"],
                        "retrieved_texts": [pool.texts[0]], "retrieved_slot_ids": [0],
                        "used_texts": [pool.texts[1]], "answer_in_context": False,
                        "diag_status": "ok"})
    ok("enrich 不覆盖 runner 记录的 answer_in_context",
       e4["diag"]["answer_in_context"] is False)

    # 9. 联合归因四格表（Retrieval × Reader）——数字全部手算核对
    def _jrec(recall1, sf, qtype="t1", ctx=0, ev=None, status="ok"):
        """造一条带 diag 的记录：recall1=answer_recall@1，sf=substring_exact_match。"""
        rec = {"question_type": qtype, "context_id": ctx, "substring_exact_match": sf,
               "diag": {"diag_status": status,
                        "answer_in_retrieved": bool(recall1),
                        "answer_in_context": bool(recall1),
                        "answer_recall_at_k": {"1": int(recall1), "3": 1, "full": 1},
                        "recall_at_k": {"1": int(recall1), "3": 1, "full": 1},
                        "evidence_recall_at_k": ev}}
        if sf is None:
            rec.pop("substring_exact_match")
        return rec

    # 12 条，四格分布（手算）：r1/sf1=4  r1/sf0=2  r0/sf1=3  r0/sf0=3
    jrecs = [
        _jrec(1, True, "t1", 0), _jrec(1, True, "t1", 0), _jrec(1, True, "t1", 0),
        _jrec(1, False, "t1", 0),                       # r1/sf0
        _jrec(0, True, "t1", 1), _jrec(0, True, "t1", 1),
        _jrec(1, 0, "t1", 1),                           # int 0 -> False -> r1/sf0
        _jrec(0, False, "t2", 1),                       # r0/sf0
        _jrec(0, 0, "t2", 2),                           # int 0 -> r0/sf0
        _jrec(1, True, "t2", 2),                        # r1/sf1
        _jrec(0, True, "t2", 2),                        # r0/sf1
        _jrec(0, False, "t2", 2),                       # r0/sf0
    ]
    sj = summarize(jrecs)
    j = sj["overall"]["joint"]
    ok("joint n=12", j["n"] == 12)
    ok("joint 四格计数与手算一致（4/2/3/3）",
       j["cells"]["recall1_sf1"]["n"] == 4 and j["cells"]["recall1_sf0"]["n"] == 2
       and j["cells"]["recall0_sf1"]["n"] == 3 and j["cells"]["recall0_sf0"]["n"] == 3)
    ok("joint Reader Acc | Recall=1 = 4/6", abs(j["reader_acc_given_recall1"] - 4 / 6) < 1e-9)
    ok("joint 读到但答错率 = 2/12", abs(j["read_but_wrong_rate"] - 2 / 12) < 1e-9)
    ok("joint 检索失败率 = 6/12", abs(j["retrieval_failure_rate"] - 0.5) < 1e-9)
    ok("joint 检索失败且答错率 = 3/12",
       abs(j["retrieval_failure_and_wrong_rate"] - 3 / 12) < 1e-9)
    ok("joint 边际：sf_acc=7/12、answer_recall=6/12",
       abs(j["sf_acc"] - 7 / 12) < 1e-9 and abs(j["answer_recall_rate"] - 0.5) < 1e-9)
    ok("joint 错误归因：5 个错误里检索侧 3、读出侧 2",
       j["error_attribution"]["n_errors"] == 5
       and abs(j["error_attribution"]["retrieval_side_share"] - 0.6) < 1e-9
       and abs(j["error_attribution"]["reader_side_share"] - 0.4) < 1e-9)
    ok("joint 四格 share 之和为 1（n>0）",
       abs(sum(c["share"] for c in j["cells"].values()) - 1.0) < 1e-9)
    ok("bool 与 int(0/1) 两种 SF 写法都识别（r1/sf0=2 含 int 0）",
       j["cells"]["recall1_sf0"]["n"] == 2)
    ok("joint 标注 recall_k / sf_field",
       j["recall_k"] == "1" and j["sf_field"] == "substring_exact_match")

    # 分组：question_type / context_id 的样本量与四格
    jq = sj["joint_by"]["question_type"]
    ok("joint_by question_type 有 t1/t2 两组", set(jq) == {"t1", "t2"})
    ok("joint_by t1：n=7、r1/sf1=3、r1/sf0=2、r0/sf1=2、r0/sf0=0",
       jq["t1"]["n"] == 7 and jq["t1"]["cells"]["recall1_sf1"]["n"] == 3
       and jq["t1"]["cells"]["recall1_sf0"]["n"] == 2
       and jq["t1"]["cells"]["recall0_sf1"]["n"] == 2
       and jq["t1"]["cells"]["recall0_sf0"]["n"] == 0)
    ok("joint_by t1 Reader Acc|R1 = 3/5", abs(jq["t1"]["reader_acc_given_recall1"] - 0.6) < 1e-9)
    ok("joint_by t2：n=5、r1/sf1=1、r0/sf0=3、Reader Acc|R1 = 1.0",
       jq["t2"]["n"] == 5 and jq["t2"]["cells"]["recall1_sf1"]["n"] == 1
       and jq["t2"]["cells"]["recall0_sf0"]["n"] == 3
       and abs(jq["t2"]["reader_acc_given_recall1"] - 1.0) < 1e-9)
    jc = sj["joint_by"]["context_id"]
    ok("joint_by context_id 三组各 n=4",
       set(jc) == {"0", "1", "2"} and all(v["n"] == 4 for v in jc.values()))
    ok("joint_by context_id=1：r0/sf1=2、r1/sf0=1、r0/sf0=1",
       jc["1"]["cells"]["recall0_sf1"]["n"] == 2
       and jc["1"]["cells"]["recall1_sf0"]["n"] == 1
       and jc["1"]["cells"]["recall0_sf0"]["n"] == 1)
    ok("joint 分组样本量之和 = 总 n",
       sum(v["n"] for v in jq.values()) == 12 == sum(v["n"] for v in jc.values()))

    # 换 K：recall@3 全命中 -> Reader Acc 退化为 sf_acc
    sj3 = summarize(jrecs, joint_k=3)
    ok("joint_k=3 时 r1 组全命中（r0 全 0）",
       sj3["overall"]["joint"]["cells"]["recall0_sf1"]["n"] == 0
       and sj3["overall"]["joint"]["cells"]["recall0_sf0"]["n"] == 0)
    ok("joint_k=3 的 Reader Acc|R1 = sf_acc = 7/12",
       abs(sj3["overall"]["joint"]["reader_acc_given_recall1"] - 7 / 12) < 1e-9)

    # 证据级召回：默认 None -> 明确标注；有值时才出 evidence_cells，且不混进答案代理格
    ok("evidence_recall_at_k 全 None 时 evidence_based=False",
       j["evidence_based"] is False and "evidence_cells" not in j)
    ok("无证据标注时 note 明说只基于答案命中代理",
       "无金标证据位置" in j["evidence_note"] and "答案命中代理" in j["evidence_note"])
    ok("joint_config 暴露口径与告警",
       sj["joint_config"]["evidence_based"] is False
       and "无金标证据位置" in sj["joint_config"]["note"]
       and sj["joint_config"]["recall_k"] == "1")
    ev_one = {"1": 1, "3": 1, "full": 1}
    sj_ev = summarize(jrecs[:3] + [_jrec(1, False, "t1", 0, ev=ev_one)]
                      + [_jrec(0, False, "t1", 0, ev={"1": 0, "3": 0, "full": 0})])
    jev = sj_ev["overall"]["joint"]
    ok("有证据标注时 evidence_based=True 且出 evidence_cells",
       jev["evidence_based"] is True and "evidence_cells" in jev
       and jev["evidence_n"] == 2)
    ok("证据四格与答案代理四格分开统计（不混用）",
       jev["evidence_cells"]["recall1_sf0"]["n"] == 1
       and jev["evidence_cells"]["recall0_sf0"]["n"] == 1
       and jev["cells"]["recall1_sf0"]["n"] == 1
       and jev["cells"]["recall0_sf0"]["n"] == 1
       and jev["evidence_note"] == JOINT_EVIDENCE_NOTE_OK)

    # 缺 SF / 缺 recall / legacy 记录：进 excluded 或完全不进表，不污染分母
    sj_ex = summarize(jrecs + [
        {"question_type": "t1", "context_id": 0, "diag": {
            "diag_status": "ok", "answer_recall_at_k": {"1": 1}}},          # 无 SF
        {"question_type": "t1", "context_id": 0, "substring_exact_match": True,
         "diag": {"diag_status": "ok"}},                                   # 无 recall
        {"question_type": "t1", "context_id": 0, "substring_exact_match": True,
         "retrieval": {"idxs": [0, 1]}},                                   # legacy
    ])
    jex = sj_ex["overall"]["joint"]
    ok("缺字段的记录只进 excluded，不进四格",
       jex["n"] == 12 and jex["excluded"]["no_sf"] == 1
       and jex["excluded"]["no_recall"] == 1)
    ok("legacy/rubric 记录记进 excluded.not_applicable",
       jex["excluded"]["not_applicable"] == 1)
    ok("legacy 记录不进分组四格",
       sum(v["n"] for v in sj_ex["joint_by"]["question_type"].values()) == 12)

    # 分组截断：joint_max_groups 生效且显式记录，不静默丢组
    sj_tr = summarize(jrecs, joint_max_groups=2)
    ok("joint_max_groups 截断并显式标记",
       len(sj_tr["joint_by"]["context_id"]) == 2
       and sj_tr["joint_truncated"]["context_id"] == {"kept": 2, "total": 3})

    # joint_table() 独立入口与 summarize 同口径
    ok("joint_table() 与 summarize().overall.joint 一致",
       joint_table(jrecs) == sj["overall"]["joint"])

    # 打印：四格表与告警必须出现（回归：print_summary 不能因缺字段崩）
    buf = io.StringIO()
    _stdout = sys.stdout
    sys.stdout = buf
    try:
        print_summary(sj, title="joint selftest")
    finally:
        sys.stdout = _stdout
    out_s = buf.getvalue()
    ok("print_summary 输出四格行",
       "RAcc|R1=0.6667" in out_s and "RFail=0.5000" in out_s and "RBW=0.1667" in out_s)
    ok("print_summary 输出证据告警与分组表",
       "无金标证据位置" in out_s and "[joint by context_id]" in out_s)

    failed = [n for n, c in checks if not c]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("ALL AR_DIAG SELFTESTS PASSED")
    return 0


# ----------------------------- CLI -----------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="AR 逐题诊断：answer_recall_at_k（答案命中代理）/ "
                    "evidence_recall_at_k（证据级召回）/ answer_in_retrieved")
    ap.add_argument("--input", nargs="*", default=[], help="AR 结果 JSONL（可多个）")
    ap.add_argument("--enrich", action="store_true", help="补写诊断字段到 --enrich-out")
    ap.add_argument("--enrich-out", default="", help="补写输出路径（默认 <input>.diag.jsonl）")
    ap.add_argument("--summary-out", default="", help="汇总 JSON 落盘路径")
    ap.add_argument("--ks", default=",".join(str(k) for k in RECALL_KS),
                    help="answer_recall_at_k 的 K 列表，逗号分隔（默认 1,3,5,10,12,20）")
    ap.add_argument("--max-text-chars", type=int, default=DEFAULT_MAX_TEXT_CHARS,
                    help="记录原文单槽位截断（0=不截断）")
    ap.add_argument("--joint-k", default=str(JOINT_RECALL_K),
                    help="四格联合表的检索侧 K（默认 1；也可给 full）")
    ap.add_argument("--joint-sf-field", default=JOINT_SF_FIELD,
                    help=f"四格联合表的读出侧字段名（默认 {JOINT_SF_FIELD}）")
    ap.add_argument("--joint-groups", default=",".join(JOINT_GROUP_BY),
                    help="联合表分组维度，逗号分隔（默认 question_type,context_id）")
    ap.add_argument("--joint-max-groups", type=int, default=JOINT_MAX_GROUPS,
                    help="每个分组维度最多输出多少组（按样本量降序，默认 200；0=不限）")
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

    joint_k: Any = str(args.joint_k).strip()
    joint_k = int(joint_k) if joint_k.isdigit() else joint_k
    joint_groups = tuple(g.strip() for g in str(args.joint_groups).split(",") if g.strip())

    recs = load_jsonl(args.input)
    summary = summarize(recs, ks=ks, joint_k=joint_k, sf_field=args.joint_sf_field,
                        joint_group_by=joint_groups,
                        joint_max_groups=args.joint_max_groups)
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
