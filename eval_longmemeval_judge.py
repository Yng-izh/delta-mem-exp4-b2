#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_longmemeval_judge.py — LongMemEval 官方 LLM-as-Judge 后处理（只写脚本，不跑）

用途：把 AR 结果 JSONL（`run_ar_v7.py` 产出）里的 (question, gold, prediction)
按 **LongMemEval 官方 `evaluate_qa.py` 的题型分支 prompt** 交给 judge 模型判定
yes/no，输出逐题判定与分题型 accuracy。用于替代当前只报 substring_exact_match
（SF）的口径——SF 会低估"语义等价但字面不同"的回答。

=====================================================================
零、硬约束
=====================================================================
- 本脚本**不加载任何本地模型、不占用 GPU、不做检索**；judge 走 HTTP API。
- `--dry-run` / `--selftest` **完全不联网**，只渲染 prompt 与做纯函数单测。
- 默认 `--dry-run` 不生效：真正判题必须显式给 `--input` 且不给 `--dry-run`。
  在拿到 API key 之前，所有验证都用 `--dry-run` / `--selftest`。

=====================================================================
一、官方口径与 prompt 来源（重要，别把它当成"我随手写的 prompt"）
=====================================================================
官方仓库：`/share/latent2/LongMemEval`，判题函数为
`src/evaluation/evaluate_qa.py::get_anscheck_prompt(task, question, answer, response, abstention)`，
按题型分 6 支 + abstention 1 支：
    single-session-user / single-session-assistant / multi-session  -> 通用支
    temporal-reasoning                                             -> 额外容忍天数 off-by-one
    knowledge-update                                               -> 容忍"旧值+新值"，只要新值正确
    single-session-preference                                      -> answer 是 rubric，看是否满足偏好
    abstention（question_id 以 `_abs` 结尾）                        -> 只看是否识别出"不可回答"

本文件内置一份**依据上述题型分支重写的镜像模板**（BUILTIN_PROMPTS）。
**注意**：本机没有官方仓库（`/share` 未挂载），内置模板的逐字一致性未在本机校验；
需要与官方逐字对齐时，用 `--prompt-source official --official-src <repo>` 直接
import 官方 `get_anscheck_prompt`（脚本会自动尝试多个模块路径）。
`--selftest` 会打印内置模板的 md5，便于与官方文件 diff 后核对。

=====================================================================
二、输入
=====================================================================
1) AR 结果 JSONL（必需）：每条至少含
     query（模板包装后的问句，脚本会剥壳）/ answer（gold，list 或 str）/ output（模型回答）
     question_type（可选，用于分题型汇总；缺失时记为 unknown）
     context_id / question_idx（可选，用于对齐与断点续跑）
2) 官方 cleaned 数据集 JSON（可选，强烈建议给）：
     `/share/paperwork/longmemeval/longmemeval_s_cleaned.json`
   用于补 question_id、rubric（preference 题的 answer 字段）、以及识别 abstention。
   脚本按归一化后的 question 文本做 join；join 不上的记录按"无官方元数据"处理并计数。

=====================================================================
三、运行（在有 API key 的机器上；本机只跑 --dry-run / --selftest）
=====================================================================
    export LME_JUDGE_API_KEY=sk-...            # 或 OPENAI_API_KEY
    export LME_JUDGE_BASE_URL=https://api.openai.com/v1
    export LME_JUDGE_MODEL=gpt-4o-2024-08-06

    # 1) 先 dry-run：只渲染 prompt，检查条数/题型分布/首条 prompt
    python eval_longmemeval_judge.py --input results/ar_v9_longmemeval_s_star.jsonl \
        --dataset /share/paperwork/longmemeval/longmemeval_s_cleaned.json \
        --dry-run --prompts-out /tmp/lme_prompts.jsonl

    # 2) 真跑（带缓存，可断点续跑）
    python eval_longmemeval_judge.py --input results/ar_v9_longmemeval_s_star.jsonl \
        --dataset /share/paperwork/longmemeval/longmemeval_s_cleaned.json \
        --out results/ar_v9_lme_judged.jsonl \
        --summary-out results/ar_v9_lme_judge_summary.json \
        --cache results/ar_v9_lme_judge_cache.jsonl --workers 8

    # 3) 与官方逐字对齐（若官方仓库可用）
    python eval_longmemeval_judge.py --input ... --prompt-source official \
        --official-src /share/latent2/LongMemEval --dry-run

输出：
    --out          逐题判定 JSONL（context_id/question_idx/question_type/verdict/raw/...）
    --summary-out  accuracy 汇总 JSON（overall + by question_type + abstention 分列）
    --cache        判题缓存（key=sha1(model+prompt)），重跑不重复付费

=====================================================================
四、判定口径
=====================================================================
- `verdict`：1=正确 / 0=错误 / None=空响应（官方实现是 `'yes' in response.lower()`：
  含 yes 即 1，否则 0；本脚本默认逐字复刻该口径，保证与官方数字可直接对照）。
  同时记录 `verdict_strict`（只看首个 token，其余 None），便于审计宽松口径。
- `accuracy = n_correct / n_total`，空响应与调用失败计入错误但单独计数
  （`parse_fail` / `error`），避免静默把 API 抖动算成模型错。
- 分题型 accuracy 与官方表一致；abstention 题单列，不混进主 accuracy 分母。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "BUILTIN_PROMPTS",
    "QUESTION_TYPES",
    "build_prompt",
    "parse_verdict",
    "load_cache",
    "summarize_judged",
    "main",
]

# ----------------------------- 官方题型 -----------------------------
QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)

# ----------------------------- 官方 prompt 镜像 -----------------------------
# 说明：以下模板按官方 evaluate_qa.py 的题型分支重写，键名与官方 task 名一致。
# 逐字一致性未在本机校验（本机无官方仓库）；需要严格一致请用 --prompt-source official。
_GENERIC = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no.\n\n"
    "Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. "
    "In addition, do not penalize off-by-one errors for the number of days. "
    "If the question asks for the number of days/weeks/months, etc., and the model makes "
    "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's "
    "response is still correct.\n\n"
    "Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response contains some previous information along with an updated answer, the "
    "response should be considered as correct as long as the updated answer is the required "
    "answer.\n\n"
    "Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a response "
    "from a model. Please answer yes if the response satisfies the desired response. "
    "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
    "The response is correct as long as it recalls and utilizes the user's personal "
    "information correctly.\n\n"
    "Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_ABSTENTION = (
    "I will give you an abstention question and a response from a model. "
    "Please answer yes if the response contains an answer that shows the model is aware "
    "that the question is unanswerable. Otherwise, answer no.\n\n"
    "Question: {question}\n\nModel Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

BUILTIN_PROMPTS: Dict[str, str] = {
    "single-session-user": _GENERIC,
    "single-session-assistant": _GENERIC,
    "multi-session": _GENERIC,
    "temporal-reasoning": _TEMPORAL,
    "knowledge-update": _KNOWLEDGE_UPDATE,
    "single-session-preference": _PREFERENCE,
    "__abstention__": _ABSTENTION,
    "__unknown__": _GENERIC,
}

# ----------------------------- 模板剥壳（与 ar_diag.core_question 同源） -----------
_CORE_Q_RE = re.compile(r"^\s*[:：]\s*(.+)$", re.S)
_CORE_MARKERS = ("Now Answer the Question", "Question")


def core_question(q: Any) -> str:
    """剥掉 runner 的查询模板外壳，取真正的问句。

    模板可能同时出现 "Question: ..." 与 "Now Answer the Question: ..."，
    因此按 marker 从右往左找（优先 "Now Answer the Question"），取最后一个。
    与 `ar_diag.core_question` 保持同一实现；此处复制一份是为了让本脚本
    在只拷贝单文件时也能独立运行（两处若改动需同步）。
    """
    s = q if isinstance(q, str) else str(q or "")
    for marker in _CORE_MARKERS:
        i = s.rfind(marker)
        if i >= 0:
            m = _CORE_Q_RE.match(s[i + len(marker):])
            if m:
                return m.group(1).strip()
    return s.strip()


def normalize_question(s: Any) -> str:
    """join 用归一：小写 + 压空白 + 去首尾标点。"""
    t = re.sub(r"\s+", " ", str(s or "")).strip().lower()
    return t.strip(" ?.!\"'")


# ----------------------------- prompt 构造 -----------------------------
def _builtin_prompt(qtype: str, question: str, answer: str, response: str,
                    abstention: bool = False) -> str:
    if abstention:
        tpl = BUILTIN_PROMPTS["__abstention__"]
    else:
        tpl = BUILTIN_PROMPTS.get(qtype or "__unknown__", BUILTIN_PROMPTS["__unknown__"])
    return tpl.format(question=question, answer=answer, response=response)


def load_official_prompt_fn(src_dir: str):
    """尝试从官方仓库 import `get_anscheck_prompt`；失败返回 (None, 原因)。"""
    if not src_dir:
        return None, "未提供 --official-src"
    if not os.path.isdir(src_dir):
        return None, f"目录不存在: {src_dir}"
    sys.path.insert(0, src_dir)
    try:
        import importlib
        for mod in ("evaluation.evaluate_qa", "src.evaluation.evaluate_qa",
                    "evaluate_qa"):
            try:
                m = importlib.import_module(mod)
            except Exception:
                continue
            fn = getattr(m, "get_anscheck_prompt", None)
            if callable(fn):
                return fn, f"imported {mod}.get_anscheck_prompt"
        return None, "官方仓库内未找到 get_anscheck_prompt"
    finally:
        try:
            sys.path.remove(src_dir)
        except ValueError:
            pass


def build_prompt(question: str, answer: str, response: str, qtype: str = "",
                 abstention: bool = False, official_fn=None) -> str:
    """构造 judge prompt。official_fn 非空时优先调用官方实现。"""
    if official_fn is not None:
        try:
            return str(official_fn(
                task=(qtype or "single-session-user"),
                question=question, answer=answer,
                response=response, abstention=bool(abstention)))
        except TypeError:
            # 官方某些版本签名是位置参数
            return str(official_fn(qtype or "single-session-user", question,
                                   answer, response, bool(abstention)))
    return _builtin_prompt(qtype, question, answer, response, abstention=abstention)


# ----------------------------- 判定解析 -----------------------------
def parse_verdict(raw: Any, strict: bool = False) -> Optional[int]:
    """把 judge 输出解析成 1/0/None。

    strict=False（默认）：**逐字复刻官方** `'yes' in response.lower()` 语义——
      含 "yes" 判 1，否则判 0；只有空响应才返回 None（调用失败/空回复）。
      这样本脚本的 accuracy 与官方数字可直接对照。
    strict=True：只看首个 token（yes/no），其余一律 None，用于审计"官方宽松口径
      是否把噪声回答误判为正确"。
    """
    t = str(raw or "").strip().lower()
    if not t:
        return None
    if strict:
        first = re.split(r"[\s.,!;:，。]", t)[0]
        if first == "yes":
            return 1
        if first == "no":
            return 0
        return None
    return 1 if "yes" in t else 0


# ----------------------------- HTTP judge -----------------------------
class JudgeError(RuntimeError):
    pass


def _http_post_json(url: str, payload: Dict[str, Any], api_key: str,
                    timeout: float = 120.0) -> Dict[str, Any]:
    """标准库 POST JSON（不依赖 openai SDK）。"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def call_judge(prompt: str, model: str, base_url: str, api_key: str,
               retries: int = 3, timeout: float = 120.0) -> str:
    """调用 OpenAI 兼容 /chat/completions。temperature=0、max_tokens=10（同官方）。"""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 10,
    }
    last = None
    for attempt in range(max(1, retries)):
        try:
            r = _http_post_json(url, payload, api_key, timeout=timeout)
            return str(r["choices"][0]["message"]["content"])
        except Exception as e:      # 网络/限流/解析都重试
            last = e
            if attempt + 1 < retries:
                time.sleep((2 ** attempt) + random.random())
    raise JudgeError(f"judge 调用失败（{retries} 次）: {last}")


# ----------------------------- 数据装载 -----------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] {path}:{ln} 非法 JSON，跳过（{e}）", file=sys.stderr)
    return recs


def load_dataset(path: str) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """官方 cleaned 数据集 -> (归一化 question -> item, 说明)。支持 .json / .jsonl。"""
    if not path:
        return {}, "未提供 --dataset"
    if not os.path.exists(path):
        return {}, f"数据集不存在: {path}"
    if path.endswith(".jsonl"):
        items = load_jsonl(path)
    else:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        items = obj if isinstance(obj, list) else list(obj.values())
    idx: Dict[str, Dict[str, Any]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        q = it.get("question")
        if q is None:
            continue
        idx[normalize_question(q)] = it
    return idx, f"装载 {len(idx)} 条官方题目"


def gold_text(rec: Dict[str, Any], ds_item: Optional[Dict[str, Any]] = None) -> str:
    """gold 文本：优先官方数据集（含 preference 的 rubric），否则用记录里的 answer。"""
    if ds_item:
        for k in ("answer", "rubric", "correct_answer"):
            v = ds_item.get(k)
            if v not in (None, "", []):
                return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    a = rec.get("answer")
    if isinstance(a, (list, tuple)):
        return "; ".join(str(x) for x in a)
    return str(a if a is not None else "")


def is_abstention(rec: Dict[str, Any], ds_item: Optional[Dict[str, Any]] = None) -> bool:
    if isinstance(rec.get("abstention"), bool):
        return rec["abstention"]
    if ds_item:
        qid = str(ds_item.get("question_id") or "")
        if qid.endswith("_abs"):
            return True
        if ds_item.get("is_abstention") is True:
            return True
    qid = str(rec.get("question_id") or "")
    return qid.endswith("_abs")


# ----------------------------- 缓存 -----------------------------
def cache_key(model: str, prompt: str) -> str:
    h = hashlib.sha1()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


def load_cache(path: str) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path or not os.path.exists(path):
        return cache
    for rec in load_jsonl(path):
        k = rec.get("key")
        if k:
            cache[k] = rec
    return cache


# ----------------------------- 汇总 -----------------------------
def _acc(n_correct: int, n_total: int) -> Optional[float]:
    return (n_correct / n_total) if n_total else None


def summarize_judged(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """按题型汇总 accuracy；abstention 单列，不进主分母。"""
    def bucket() -> Dict[str, int]:
        return {"n": 0, "correct": 0, "parse_fail": 0, "error": 0}

    overall = bucket()
    by_type: Dict[str, Dict[str, int]] = {}
    abst = bucket()

    for r in rows:
        b = abst if r.get("abstention") else overall
        if not r.get("abstention"):
            by_type.setdefault(str(r.get("question_type") or "unknown"), bucket())
            bt = by_type[str(r.get("question_type") or "unknown")]
        else:
            bt = None
        for tgt in ([b] if bt is None else [b, bt]):
            tgt["n"] += 1
            if r.get("error"):
                tgt["error"] += 1
            if r.get("verdict") is None:
                tgt["parse_fail"] += 1
            elif r.get("verdict") == 1:
                tgt["correct"] += 1

    def finalize(b: Dict[str, int]) -> Dict[str, Any]:
        return {
            "n": b["n"], "correct": b["correct"],
            "accuracy": _acc(b["correct"], b["n"]),
            "parse_fail": b["parse_fail"], "error": b["error"],
        }

    return {
        "n_total": len(rows),
        "overall": finalize(overall),
        "abstention": finalize(abst),
        "by_question_type": {k: finalize(v) for k, v in sorted(by_type.items())},
        "model": rows[0].get("model") if rows else None,
        "prompt_source": rows[0].get("prompt_source") if rows else None,
    }


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def print_summary(s: Dict[str, Any]) -> None:
    o = s["overall"]
    print(f"judged={s['n_total']}  accuracy={_fmt(o['accuracy'])}  "
          f"(correct={o['correct']} parse_fail={o['parse_fail']} error={o['error']})")
    if s["abstention"]["n"]:
        a = s["abstention"]
        print(f"[abstention] n={a['n']} accuracy={_fmt(a['accuracy'])} "
              f"parse_fail={a['parse_fail']} error={a['error']}")
    for t, v in s["by_question_type"].items():
        print(f"  {t:<26} n={v['n']:<4} accuracy={_fmt(v['accuracy'])} "
              f"parse_fail={v['parse_fail']} error={v['error']}")


# ----------------------------- 主流程 -----------------------------
def build_rows(records: Sequence[Dict[str, Any]],
               ds_index: Optional[Dict[str, Dict[str, Any]]] = None,
               official_fn=None,
               prompt_source: str = "builtin") -> List[Dict[str, Any]]:
    """把 AR 记录转换成待判行（含 prompt），不发起任何请求。"""
    rows: List[Dict[str, Any]] = []
    n_join = 0
    n_gold_missing = 0
    for rec in records:
        q_full = rec.get("query", "")
        q = core_question(q_full)
        ds_item = None
        if ds_index:
            ds_item = ds_index.get(normalize_question(q))
            if ds_item is not None:
                n_join += 1
        qtype = str(rec.get("question_type") or (ds_item or {}).get("question_type") or "unknown")
        gold = gold_text(rec, ds_item)
        if not str(gold).strip() and not is_abstention(rec, ds_item):
            n_gold_missing += 1
        pred = str(rec.get("output", ""))
        abst = is_abstention(rec, ds_item)
        prompt = build_prompt(q, gold, pred, qtype=qtype, abstention=abst,
                              official_fn=official_fn)
        rows.append({
            "context_id": rec.get("context_id"),
            "question_idx": rec.get("question_idx"),
            "question_id": rec.get("question_id") or (ds_item or {}).get("question_id"),
            "question_type": qtype,
            "abstention": abst,
            "question": q,
            "gold": gold,
            "prediction": pred,
            "prompt": prompt,
            "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
            "prompt_source": prompt_source,
        })
    if ds_index:
        print(f"[join] 官方题目命中 {n_join}/{len(records)}", file=sys.stderr)
    if n_gold_missing:
        # gold 缺失时 judge 只能判错，必须显式告警，否则 accuracy 会被静默拉低
        print(f"[warn] {n_gold_missing} 条记录缺 gold（且非 abstention），"
              f"这些题会被判错；建议补 --dataset", file=sys.stderr)
    return rows


def run_judge(rows: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    """并发判题（带缓存）。rows 需已含 prompt。"""
    cache = load_cache(args.cache) if args.cache and not args.no_resume else {}
    model, base_url, api_key = args.model, args.base_url, args.api_key
    out: List[Dict[str, Any]] = []
    lock = threading.Lock()
    cache_f = None
    if args.cache:
        os.makedirs(os.path.dirname(os.path.abspath(args.cache)) or ".", exist_ok=True)
        cache_f = open(args.cache, "a", encoding="utf-8")

    def judge_one(row: Dict[str, Any]) -> Dict[str, Any]:
        key = cache_key(model, row["prompt"])
        hit = cache.get(key)
        # 缓存里只有"成功判定"才复用；上次是 error 的条目必须重试，
        # 否则一次 API 抖动会被永久固化进结果。
        if hit is not None and not hit.get("error"):
            raw = hit.get("raw", "")
            err = None
        else:
            raw, err = "", None
            try:
                raw = call_judge(row["prompt"], model, base_url, api_key,
                                 retries=args.retries, timeout=args.timeout)
            except JudgeError as e:
                err = str(e)
            if cache_f is not None:
                with lock:
                    cache_f.write(json.dumps({
                        "key": key, "model": model, "raw": raw, "error": err,
                        "ts": time.time()}, ensure_ascii=False) + "\n")
                    cache_f.flush()
        return {
            "context_id": row["context_id"], "question_idx": row["question_idx"],
            "question_id": row["question_id"], "question_type": row["question_type"],
            "abstention": row["abstention"], "question": row["question"],
            "gold": row["gold"], "prediction": row["prediction"],
            "raw": raw, "error": err,
            "verdict": parse_verdict(raw) if not err else None,
            "verdict_strict": parse_verdict(raw, strict=True) if not err else None,
            "prompt_sha1": row["prompt_sha1"], "model": model,
            "prompt_source": row["prompt_source"],
        }

    try:
        if args.workers <= 1:
            for r in rows:
                out.append(judge_one(r))
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(judge_one, r): r for r in rows}
                for fut in as_completed(futs):
                    out.append(fut.result())
    finally:
        if cache_f is not None:
            cache_f.close()

    out.sort(key=lambda r: (str(r.get("context_id")), str(r.get("question_idx"))))
    return out


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# ----------------------------- selftest -----------------------------
def _selftest() -> int:
    """纯 CPU 自测：模板渲染 + 判定解析 + dry-run（禁网）+ 缓存 + 汇总。

    这里 monkeypatch 掉 `_http_post_json`，任何真实网络调用都会直接失败，
    从而在结构上保证 --selftest 不联网。
    """
    checks: List[Tuple[str, bool]] = []

    def ok(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"  {'ok  ' if cond else 'FAIL'} - {name}")

    # 0. 禁网哨兵
    orig_post = globals()["_http_post_json"]

    def _boom(*a, **k):
        raise AssertionError("selftest 中发生了真实 HTTP 调用")

    globals()["_http_post_json"] = _boom
    try:
        # 1. 模板渲染：6 题型 + abstention 都能出 prompt，且含关键约束
        for qt in QUESTION_TYPES:
            p = build_prompt("Who?", "Alice", "Bob", qtype=qt)
            ok(f"prompt[{qt}] 含问题/答案/回答", "Who?" in p and "Alice" in p and "Bob" in p)
            ok(f"prompt[{qt}] 含 yes/no 约束", "Answer yes or no only" in p)
        ok("temporal 含 off-by-one 容忍", "off-by-one" in build_prompt("q", "a", "r", qtype="temporal-reasoning"))
        ok("knowledge-update 含旧值容忍",
           "previous information" in build_prompt("q", "a", "r", qtype="knowledge-update"))
        ok("preference 用 Rubric 字段", "Rubric:" in build_prompt("q", "rubric-text", "r",
                                                                qtype="single-session-preference"))
        pa = build_prompt("q", "", "r", qtype="single-session-user", abstention=True)
        ok("abstention 模板不含 Correct Answer", "Correct Answer" not in pa
           and "unanswerable" in pa)
        ok("未知题型回退通用模板", "Correct Answer" in build_prompt("q", "a", "r", qtype="???"))
        # 模板无残留占位符
        ok("无未替换占位符", all("{" not in build_prompt("q", "a", "r", qtype=qt)
                                and "}" not in build_prompt("q", "a", "r", qtype=qt)
                                for qt in QUESTION_TYPES))

        # 2. 判定解析
        ok("yes -> 1", parse_verdict("yes") == 1)
        ok("Yes. -> 1", parse_verdict("Yes.") == 1)
        ok("no -> 0", parse_verdict("no") == 0)
        ok("No, because ... -> 0", parse_verdict("No, the answer differs.") == 0)
        ok("空 -> None（调用失败/空回复）", parse_verdict("") is None)
        ok("maybe -> 0（官方 yes 口径）", parse_verdict("maybe") == 0)
        ok("strict: maybe -> None", parse_verdict("maybe", strict=True) is None)
        ok("噪声 yes -> 1（官方宽松口径）", parse_verdict("No ... yes") == 1)
        ok("strict 口径不看噪声", parse_verdict("No ... yes", strict=True) == 0)
        ok("strict: Yes -> 1", parse_verdict("Yes, correct", strict=True) == 1)

        # 3. 记录 -> 行（含剥壳 + 官方数据集 join + abstention）
        recs = [
            {"context_id": 0, "question_idx": 0, "question_type": "single-session-user",
             "query": "Question: Who wrote X?\nNow Answer the Question: Who wrote X?",
             "answer": ["Alice"], "output": "Alice"},
            {"context_id": 0, "question_idx": 1, "question_type": "single-session-preference",
             "query": "Question: Plan my trip", "answer": ["likes window seats"],
             "output": "ok"},
            {"context_id": 0, "question_idx": 2, "query": "Question: Unanswerable?",
             "answer": ["unanswerable"], "output": "no idea", "abstention": True},
        ]
        ds = {normalize_question("Who wrote X?"): {"question_id": "q1",
                                                   "question_type": "single-session-user",
                                                   "answer": "Alice"}}
        rows = build_rows(recs, ds_index=ds)
        ok("剥壳后问题正确", rows[0]["question"] == "Who wrote X?")
        ok("join 命中官方 gold", rows[0]["gold"] == "Alice")
        ok("preference gold 用记录 answer", rows[1]["gold"] == "likes window seats")
        ok("abstention 识别", rows[2]["abstention"] is True)
        ok("abstention 用 abstention 模板", "Correct Answer" not in rows[2]["prompt"])

        # 4. dry-run：写 prompt 文件、零网络
        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "in.jsonl")
            with open(inp, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            prompts_out = os.path.join(td, "prompts.jsonl")
            rc = main(["--input", inp, "--dry-run", "--prompts-out", prompts_out])
            ok("dry-run 退出码 0", rc == 0)
            ok("dry-run 写出 prompt 文件", os.path.exists(prompts_out))
            n = len(load_jsonl(prompts_out)) if os.path.exists(prompts_out) else 0
            ok("dry-run prompt 条数=3", n == 3)

            # 5. 缓存：伪造一条缓存记录，judge 应命中缓存且不联网
            cache_path = os.path.join(td, "cache.jsonl")
            rows2 = build_rows(recs, ds_index=ds)
            with open(cache_path, "w", encoding="utf-8") as f:
                for r in rows2:
                    f.write(json.dumps({"key": cache_key("m", r["prompt"]),
                                        "model": "m", "raw": "yes", "error": None},
                                       ensure_ascii=False) + "\n")
            args = argparse.Namespace(model="m", base_url="http://invalid", api_key="k",
                                      workers=2, retries=1, timeout=1,
                                      cache=cache_path, no_resume=False)
            judged = run_judge(rows2, args)
            ok("缓存命中 3 条", len(judged) == 3)
            ok("缓存命中不联网且判对", all(r["verdict"] == 1 and not r["error"] for r in judged))

            # 5b. 缓存里是 error 的条目必须重试（而不是把抖动永久固化）
            cache_err = os.path.join(td, "cache_err.jsonl")
            with open(cache_err, "w", encoding="utf-8") as f:
                for r in rows2:
                    f.write(json.dumps({"key": cache_key("m", r["prompt"]),
                                        "model": "m", "raw": "", "error": "boom"},
                                       ensure_ascii=False) + "\n")
            args_err = argparse.Namespace(model="m", base_url="http://invalid", api_key="k",
                                          workers=1, retries=1, timeout=1,
                                          cache=cache_err, no_resume=False)
            judged_err = run_judge(rows2[:1], args_err)
            ok("error 缓存不复用（触发重试）",
               bool(judged_err[0]["error"]) and judged_err[0]["verdict"] is None)

            # 6. 汇总
            s = summarize_judged(judged)
            ok("汇总 n_total=3", s["n_total"] == 3)
            ok("abstention 单列 n=1", s["abstention"]["n"] == 1)
            ok("主 accuracy=1.0", abs((s["overall"]["accuracy"] or 0) - 1.0) < 1e-9)
            ok("by_question_type 含 single-session-user",
               "single-session-user" in s["by_question_type"])
            ok("abstention 不进主分母", s["overall"]["n"] == 2)
    finally:
        globals()["_http_post_json"] = orig_post

    # 7. 内置模板指纹（供与官方文件 diff 核对）
    for k, v in BUILTIN_PROMPTS.items():
        print(f"  tmpl[{k}] md5={hashlib.md5(v.encode()).hexdigest()} len={len(v)}")

    # 8. official 加载失败要优雅降级
    fn, why = load_official_prompt_fn("/nonexistent/LongMemEval")
    ok("official 缺失时返回 None", fn is None and "不存在" in why)

    failed = [n for n, c in checks if not c]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("ALL LONGMEMEVAL JUDGE SELFTESTS PASSED (no network, no model)")
    return 0


# ----------------------------- CLI -----------------------------
def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="LongMemEval 官方 LLM-as-Judge 后处理（不加载本地模型/不占 GPU）")
    ap.add_argument("--input", default="", help="AR 结果 JSONL")
    ap.add_argument("--dataset", default="", help="官方 cleaned 数据集（.json/.jsonl，可选）")
    ap.add_argument("--out", default="", help="逐题判定输出 JSONL")
    ap.add_argument("--summary-out", default="", help="accuracy 汇总 JSON")
    ap.add_argument("--prompts-out", default="", help="dry-run 时 prompt 落盘路径")
    ap.add_argument("--cache", default="", help="判题缓存 JSONL（断点续跑，省钱）")
    ap.add_argument("--no-resume", action="store_true", help="忽略已有缓存")
    ap.add_argument("--model", default=os.environ.get("LME_JUDGE_MODEL", "gpt-4o-2024-08-06"))
    ap.add_argument("--base-url", default=os.environ.get("LME_JUDGE_BASE_URL",
                                                        "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("LME_JUDGE_API_KEY",
                                                       os.environ.get("OPENAI_API_KEY", "")))
    ap.add_argument("--workers", type=int, default=8, help="并发判题线程数")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=0, help="只判前 N 题（0=全部）")
    ap.add_argument("--dry-run", action="store_true", help="只渲染 prompt，不联网")
    ap.add_argument("--prompt-source", choices=("builtin", "official"), default="builtin",
                    help="prompt 来源：内置镜像 / 官方仓库函数")
    ap.add_argument("--official-src", default=os.environ.get("LONGMEMEVAL_SRC", ""),
                    help="官方仓库根目录（--prompt-source official 时必需）")
    ap.add_argument("--selftest", action="store_true", help="纯 CPU 自测后退出（不联网）")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.input:
        raise SystemExit("需要 --input（或 --selftest）")
    if not os.path.exists(args.input):
        raise SystemExit(f"[ERR] 输入不存在: {args.input}")

    official_fn = None
    if args.prompt_source == "official":
        official_fn, why = load_official_prompt_fn(args.official_src)
        if official_fn is None:
            raise SystemExit(f"[ERR] --prompt-source official 不可用：{why}\n"
                             f"      改用 --prompt-source builtin，或修正 --official-src")
        print(f"[prompt] {why}", file=sys.stderr)

    records = load_jsonl(args.input)
    ds_index, ds_note = load_dataset(args.dataset)
    if args.dataset:
        print(f"[dataset] {ds_note}", file=sys.stderr)
    if args.limit > 0:
        records = records[:args.limit]

    rows = build_rows(records, ds_index=ds_index or None,
                      official_fn=official_fn, prompt_source=args.prompt_source)
    if not rows:
        raise SystemExit("[ERR] 没有可判题记录（input 为空或字段缺失）")

    # dry-run：只落盘 prompt，绝不联网
    if args.dry_run:
        out = args.prompts_out or (os.path.splitext(args.input)[0] + ".prompts.jsonl")
        n = _write_jsonl(out, rows)
        print(f"[dry-run] {n} 条 prompt -> {out}（未发起任何网络请求）")
        print(f"[dry-run] 首条 prompt 前 400 字：\n{rows[0]['prompt'][:400]}")
        types: Dict[str, int] = {}
        for r in rows:
            types[r["question_type"]] = types.get(r["question_type"], 0) + 1
        print("[dry-run] 题型分布: " + "  ".join(f"{k}={v}" for k, v in sorted(types.items())))
        return 0

    if not args.api_key:
        raise SystemExit("[ERR] 缺少 judge API key：设 LME_JUDGE_API_KEY 或 --api-key"
                         "（仅想检查 prompt 请加 --dry-run）")
    if not args.out:
        raise SystemExit("[ERR] 真判题需要 --out 指定逐题判定输出")

    t0 = time.time()
    judged = run_judge(rows, args)
    n = _write_jsonl(args.out, judged)
    summary = summarize_judged(judged)
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    summary["input"] = args.input
    summary["dataset"] = args.dataset or None
    print(f"[out] {n} 条判定 -> {args.out}")
    print_summary(summary)
    if args.summary_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_out)) or ".", exist_ok=True)
        with open(args.summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[out] summary -> {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
