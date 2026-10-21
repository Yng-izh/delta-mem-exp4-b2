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
- 两种 judge 后端，由 `--judge-backend` 选择，**prompt / verdict 语义 / 输出字段 / 缓存格式
  完全相同**（差异只在"怎么拿到 raw 文本"）：
    http  （默认）OpenAI 兼容 /chat/completions，不加载本地模型、不占 GPU；
    local 用 transformers 加载本地 Qwen 权重（`--judge-model`），需要 torch + GPU。
- `--dry-run` / `--selftest` **完全不联网、不加载任何权重**，只渲染 prompt 与做纯函数单测；
  `--selftest` 用注入的假 generate 验证 local 全链路，`_TRANSFORMERS_IMPORTS` 计数器
  会断言"全程没碰 transformers"。
- **local 后端绝不静默回退 http**：缺 torch/transformers、无可用 GPU、权重路径不存在、
  缺 `--judge-model`，一律抛 `LocalBackendError` → 非零退出码 + 可执行的修复提示
  （无 GPU 时必须显式 `--judge-device cpu` 才肯跑 CPU，避免误以为在 GPU 上跑）。
- 默认 `--dry-run` 不生效：真正判题必须显式给 `--input` 且不给 `--dry-run`。
  在拿到 API key（或 GPU 权重）之前，所有验证都用 `--dry-run` / `--selftest`。

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

=====================================================================
五、两种 judge 后端（http | local）与**等价性验证方法**
=====================================================================
后端选择：
    --judge-backend http   默认。--model/--base-url/--api-key（或 LME_JUDGE_* 环境变量）。
    --judge-backend local  本地权重。--judge-model 给**目录**或 **HF 模型名**
                           （如 Qwen/Qwen2.5-7B-Instruct；本地目录请写绝对路径或 ./ 开头）。

local 后端的显存/速度保护（写死，不靠调用方记得加）：
    max_new_tokens=10（与 http 的 max_tokens=10 对齐，判题只需 yes/no）
    do_sample=False   （贪心解码，与 http 的 temperature=0 语义对齐，结果可复现）
    --judge-batch-size=8  按批 generate，显存不够就调小；单批 OOM 只污染这一批（记 error）
    --judge-device auto|cuda|cpu|mps    auto=有 CUDA 用 CUDA、有 MPS 用 MPS，否则**报错**
    --judge-dtype  auto|float16|bfloat16|float32   auto=GPU float16 / CPU float32
    instruct 权重会自动套 chat template（与 http 的单条 user message 等价）。
    --workers 只对 http 后端有效（local 走单进程批量生成，靠 --judge-batch-size 吃满显存）。

**等价性验证方法**（唯一可信做法：跑同一批题，比逐题判定，而不是凭感觉）：

    # 1) 同一批题分别用两个后端判，落到两份逐题判定 JSONL
    python eval_longmemeval_judge.py --input results/ar_v9_lme.jsonl \
        --judge-backend local --judge-model /weights/Qwen2.5-7B-Instruct \
        --out results/lme_judged_local.jsonl --summary-out results/lme_sum_local.json
    python eval_longmemeval_judge.py --input results/ar_v9_lme.jsonl \
        --judge-backend http --model gpt-4o-2024-08-06 \
        --out results/lme_judged_http.jsonl --summary-out results/lme_sum_http.json

    # 2) 逐题比对（离线、零网络）：prompt 是否逐字一致 + verdict 一致率
    python eval_longmemeval_judge.py --compare-backends \
        results/lme_judged_local.jsonl results/lme_judged_http.jsonl \
        --compare-out results/lme_backend_equiv.json
    # 等价 -> 退出码 0 且打印 "等价性判定: PASS"；不等价 -> 退出码 1 + 差异清单

判定等价的三条硬标准（`compare_backends()` 逐条检查）：
    1) prompt 逐字一致（prompt_sha1 全等）——否则两边问的不是同一个问题，比 verdict 没意义；
    2) verdict 一致率（默认口径 'yes' in text）——不同后端允许有抖动，但必须在报告里可见；
    3) verdict_strict 一致率——严格口径审计，用来暴露"宽松口径掩盖的抖动"。
    任何一条不满足都判不等价，`diffs` 里给出题号/题型/两边 raw，供人工复核是抖动还是口径差异。
    注意：local 与 http 的 model 名不同 ⇒ 缓存 key 不同，两后端缓存天然隔离，不会互相污染。
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
    "LOCAL_MAX_NEW_TOKENS",
    "LOCAL_DO_SAMPLE",
    "LOCAL_BATCH_SIZE",
    "LocalBackendError",
    "build_prompt",
    "parse_verdict",
    "load_cache",
    "summarize_judged",
    "resolve_local_device",
    "load_local_judge",
    "local_generate_batch",
    "run_judge",
    "run_judge_http",
    "run_judge_local",
    "compare_backends",
    "judge_backend",
    "judge_model_name",
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


# ----------------------------- 本地 Judge 后端 -----------------------------
# 目的：不依赖任何 HTTP API，直接加载本地 Qwen 权重判题。硬约束：
#   1) 与 http 后端共用**同一套** prompt 模板（build_prompt）与**同一套** verdict 语义
#      （parse_verdict：'yes' in text），因此两后端的判定可直接对照；
#   2) 显存/速度保护：max_new_tokens=10、do_sample=False（贪心，确定性）、按 batch 生成；
#   3) **绝不静默回退到 http**：设备/依赖/权重任一不满足就抛 LocalBackendError，
#      由 main 转成非零退出码 + 可执行的修复提示。
LOCAL_MAX_NEW_TOKENS = 10          # 与 http 的 max_tokens=10 对齐
LOCAL_DO_SAMPLE = False            # 贪心解码：与 http 的 temperature=0 语义对齐
LOCAL_BATCH_SIZE = 8
LOCAL_DEVICES = ("auto", "cuda", "cuda:0", "cuda:1", "cpu", "mps")
LOCAL_DTYPES = ("auto", "float16", "bfloat16", "float32")


class LocalBackendError(RuntimeError):
    """本地 Judge 后端不可用（缺依赖/缺 GPU/权重路径错）。绝不被降级成 http。"""


def _cuda_available(torch_mod=None) -> bool:
    """可被测试替换的探针；不 import torch 时返回 False。"""
    if torch_mod is None:
        try:
            import torch as torch_mod
        except Exception:
            return False
    try:
        return bool(torch_mod.cuda.is_available())
    except Exception:
        return False


def _mps_available(torch_mod=None) -> bool:
    if torch_mod is None:
        try:
            import torch as torch_mod
        except Exception:
            return False
    try:
        return bool(getattr(torch_mod.backends, "mps", None)
                    and torch_mod.backends.mps.is_available())
    except Exception:
        return False


def resolve_local_device(spec: str = "auto", cuda_available: Optional[bool] = None,
                         mps_available: Optional[bool] = None) -> str:
    """决定本地权重跑在哪块设备上；不可用时抛**明确**错误，不猜也不回退。

    默认 auto 要求有 GPU（CUDA，退而求其次 Apple MPS）；纯 CPU 必须显式
    `--judge-device cpu`，避免有人在无 GPU 机器上误跑出"看起来正常"的慢结果。
    """
    spec = str(spec or "auto").strip().lower()
    if cuda_available is None:
        cuda_available = _cuda_available()
    if mps_available is None:
        mps_available = _mps_available()
    if spec in ("auto", ""):
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        raise LocalBackendError(
            "本地 Judge 后端未找到可用 GPU（CUDA 不可用，MPS 也不可用）。\n"
            "     这不是可以静默忽略的问题，脚本不会回退到 http 后端。可选：\n"
            "       a) 在带 GPU 的机器上跑（推荐：Qwen judge 在 CPU 上慢到不可用）；\n"
            "       b) 显式加 --judge-device cpu 强制 CPU（会很慢，仅用于少量冒烟）；\n"
            "       c) 改用 --judge-backend http（OpenAI 兼容接口）。")
    if spec.startswith("cuda"):
        if not cuda_available:
            raise LocalBackendError(
                f"--judge-device {spec} 要求 CUDA，但 torch.cuda.is_available()=False。\n"
                "     请换到带 GPU 的机器、改用 --judge-device cpu（慢），"
                "或改用 --judge-backend http。")
        return spec
    if spec == "mps":
        if not mps_available:
            raise LocalBackendError(
                "--judge-device mps 要求 Apple Silicon + torch.backends.mps 可用，当前不可用。\n"
                "     请改用 --judge-device cpu（慢）或 --judge-backend http。")
        return spec
    if spec == "cpu":
        return "cpu"
    raise LocalBackendError(f"未知 --judge-device={spec!r}；可选 {LOCAL_DEVICES}")


def _resolve_dtype(torch_mod, dtype: str, device: str):
    """dtype 策略：auto -> GPU 上 float16（省显存）、CPU 上 float32（CPU 半精度常崩）。"""
    dtype = str(dtype or "auto").strip().lower()
    if dtype == "auto":
        return torch_mod.float16 if device.startswith("cuda") else torch_mod.float32
    table = {"float16": torch_mod.float16, "bfloat16": torch_mod.bfloat16,
             "float32": torch_mod.float32}
    if dtype not in table:
        raise LocalBackendError(f"未知 --judge-dtype={dtype!r}；可选 {LOCAL_DTYPES}")
    return table[dtype]


_TRANSFORMERS_IMPORTS = 0          # selftest 用它断言"零权重依赖"


def _torch_module():
    """轻量探测 torch（不 import transformers）；不可用时返回 None。"""
    try:
        import torch
        return torch
    except Exception:
        return None


def _require_transformers():
    """惰性 import torch + transformers；缺失时给可执行的修复提示（不回退 http）。"""
    global _TRANSFORMERS_IMPORTS
    _TRANSFORMERS_IMPORTS += 1
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise LocalBackendError(
            "本地 Judge 后端需要 torch + transformers，当前导入失败："
            f"{type(e).__name__}: {e}\n"
            "     请 pip install 'transformers>=4.40' torch（Qwen 需要 trust_remote_code 兼容版本），"
            "或改用 --judge-backend http。脚本不会静默回退。")
    return torch, AutoModelForCausalLM, AutoTokenizer


def load_local_judge(model_path: str, device: str = "auto", dtype: str = "auto",
                     max_new_tokens: int = LOCAL_MAX_NEW_TOKENS,
                     batch_size: int = LOCAL_BATCH_SIZE):
    """加载本地权重，返回 `generate(batch_prompts) -> List[str]`（纯 CPU 上不可测，故隔离）。"""
    model_path = str(model_path or "").strip()
    if not model_path:
        raise LocalBackendError(
            "本地 Judge 后端必须给 --judge-model（本地权重目录或 HF 模型名，如 "
            "Qwen/Qwen2.5-7B-Instruct）。脚本不会回退到 http。")
    # 区分"本地路径"与"HF 模型名"（Qwen/Qwen2.5-7B-Instruct 里也有 '/'，不能一律当路径）：
    #   存在的路径 -> 本地权重；绝对路径 / ./ ../ ~ 开头 -> 必须存在，否则报错；
    #   其余（org/name 形状）-> 交给 transformers 当 HF 模型名解析，并明确提示。
    expanded = os.path.expanduser(model_path)
    path_hint = (os.path.isabs(expanded) or model_path.startswith((".", "~"))
                 or os.sep in model_path and not re.match(r"^[\w.\-]+/[\w.\-]+$", model_path))
    if os.path.exists(expanded):
        model_path = expanded
    elif path_hint:
        raise LocalBackendError(
            f"--judge-model 指向的本地路径不存在: {expanded}\n"
            "     若是 HF 模型名请写成 org/name（如 Qwen/Qwen2.5-7B-Instruct）；"
            "若是本地目录请给绝对路径或 ./ 开头。")
    else:
        print(f"[local] --judge-model={model_path} 按 HF 模型名解析"
              f"（本地目录请写成绝对路径或 ./ 开头）", file=sys.stderr)

    # 检查顺序刻意如此：先确认有 torch（否则谈不上设备），再判设备（无 GPU 是最常见的
    # 失败点，报错必须最直白），最后才 import transformers / 读权重。
    torch_mod = _torch_module()
    if torch_mod is None:
        raise LocalBackendError(
            "本地 Judge 后端需要 torch，当前 import torch 失败。\n"
            "     请 pip install torch（按 CUDA 版本装对应 wheel），"
            "或改用 --judge-backend http。脚本不会静默回退。")
    dev = resolve_local_device(device, _cuda_available(torch_mod), _mps_available(torch_mod))
    torch, AutoModelForCausalLM, AutoTokenizer = _require_transformers()
    torch_dtype = _resolve_dtype(torch, dtype, dev)

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
    model.to(dev).eval()

    def generate(batch_prompts: Sequence[str]) -> List[str]:
        # Qwen 等 instruct 权重必须套 chat template，否则判题质量会明显下降；
        # 没有 template 的权重退化为裸 prompt（与 http 后端的单条 user message 等价）。
        texts = []
        for p in batch_prompts:
            if getattr(tok, "chat_template", None):
                texts.append(tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True))
            else:
                texts.append(p)
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True).to(dev)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max(1, int(max_new_tokens)),
                do_sample=LOCAL_DO_SAMPLE,          # 贪心：与 http temperature=0 对齐
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        return [tok.decode(t, skip_special_tokens=True).strip() for t in new_tokens]

    generate.model_path = model_path          # 供审计/日志
    generate.device = dev
    generate.batch_size = int(batch_size)
    return generate


def _chunks(seq: Sequence[Any], size: int) -> List[List[Any]]:
    size = max(1, int(size))
    return [list(seq[i:i + size]) for i in range(0, len(seq), size)]


def local_generate_batch(generate, prompts: Sequence[str],
                         batch_size: int = LOCAL_BATCH_SIZE) -> List[str]:
    """按 batch 逐块生成；块内长度不齐时由 tokenizer padding 兜住。"""
    prompts = list(prompts)
    raws: List[str] = []
    for chunk in _chunks(prompts, batch_size):
        got = list(generate(chunk))
        if len(got) != len(chunk):
            raise LocalBackendError(
                f"本地 generate 返回 {len(got)} 条，期望 {len(chunk)} 条（batch 对齐错误）")
        raws.extend(str(x) for x in got)
    return raws


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


def judge_backend(args) -> str:
    """归一化 --judge-backend；非法值直接报错（不猜、不回退）。"""
    b = str(getattr(args, "judge_backend", "http") or "http").strip().lower()
    if b not in ("http", "local"):
        raise LocalBackendError(f"未知 --judge-backend={b!r}；可选 http | local")
    return b


def judge_model_name(args) -> str:
    """写进缓存 key 与结果行 model 字段的名字。

    local 后端用权重路径（或 HF 模型名），因此**与 http 的 model 名不同**，
    两后端的缓存天然隔离，不会互相污染判定。
    """
    if judge_backend(args) == "local":
        return str(getattr(args, "judge_model", "") or "")
    return str(getattr(args, "model", "") or "")


def _result_row(row: Dict[str, Any], raw: str, err: Optional[str], model: str,
                backend: str) -> Dict[str, Any]:
    """把 (raw, err) 变成一条逐题判定记录；两后端共用同一 verdict 语义。"""
    return {
        "context_id": row["context_id"], "question_idx": row["question_idx"],
        "question_id": row["question_id"], "question_type": row["question_type"],
        "abstention": row["abstention"], "question": row["question"],
        "gold": row["gold"], "prediction": row["prediction"],
        "raw": raw, "error": err,
        "verdict": parse_verdict(raw) if not err else None,
        "verdict_strict": parse_verdict(raw, strict=True) if not err else None,
        "prompt_sha1": row["prompt_sha1"], "model": model, "judge_backend": backend,
        "prompt_source": row["prompt_source"],
    }


def run_judge(rows: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    """判题入口：按 --judge-backend 分派到 http（并发）或 local（批量生成）。

    两个后端共用缓存格式、prompt、verdict 语义与输出字段；local 失败时抛
    LocalBackendError，**不会**降级成 http。
    """
    if judge_backend(args) == "local":
        return run_judge_local(rows, args)
    return run_judge_http(rows, args)


def run_judge_http(rows: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    """并发判题（带缓存）。rows 需已含 prompt。"""
    cache = load_cache(args.cache) if args.cache and not args.no_resume else {}
    model, base_url, api_key = judge_model_name(args), args.base_url, args.api_key
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
        return _result_row(row, raw, err, model, "http")

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


def run_judge_local(rows: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    """本地权重批量判题（带同一套缓存）。

    与 http 后端的差异只有"怎么拿到 raw 文本"：这里一次性加载权重、按 batch
    贪心生成；prompt / verdict / 输出字段 / 缓存格式完全相同，因此两后端结果可直接比对。
    加载失败（缺依赖、无 GPU、权重路径错）抛 LocalBackendError 直接上抛 —— 不回退 http。
    """
    model = judge_model_name(args)
    if not model:
        raise LocalBackendError(
            "本地 Judge 后端必须给 --judge-model（本地权重目录或 HF 模型名）。"
            "脚本不会回退到 http 后端。")
    cache = load_cache(args.cache) if args.cache and not args.no_resume else {}
    batch_size = max(1, int(getattr(args, "judge_batch_size", LOCAL_BATCH_SIZE)))
    max_new_tokens = int(getattr(args, "judge_max_new_tokens", LOCAL_MAX_NEW_TOKENS))
    device = str(getattr(args, "judge_device", "auto") or "auto")
    dtype = str(getattr(args, "judge_dtype", "auto") or "auto")

    out: List[Dict[str, Any]] = []
    pending: List[Tuple[Dict[str, Any], str]] = []
    for row in rows:
        key = cache_key(model, row["prompt"])
        hit = cache.get(key)
        # 同 http：只有成功判定才复用；error 条目必须重试
        if hit is not None and not hit.get("error"):
            out.append(_result_row(row, hit.get("raw", ""), None, model, "local"))
        else:
            pending.append((row, key))

    if not pending:
        out.sort(key=lambda r: (str(r.get("context_id")), str(r.get("question_idx"))))
        return out

    # 设备/依赖检查在加载前做，报错信息与"要不要回退"的判断都发生在这里
    generate = load_local_judge(model, device=device, dtype=dtype,
                               max_new_tokens=max_new_tokens, batch_size=batch_size)
    print(f"[local] judge 权重={getattr(generate, 'model_path', model)} "
          f"device={getattr(generate, 'device', device)} batch={batch_size} "
          f"max_new_tokens={max_new_tokens} do_sample={LOCAL_DO_SAMPLE} "
          f"待判={len(pending)}（缓存命中 {len(out)}）", file=sys.stderr)

    cache_f = None
    if args.cache:
        os.makedirs(os.path.dirname(os.path.abspath(args.cache)) or ".", exist_ok=True)
        cache_f = open(args.cache, "a", encoding="utf-8")
    try:
        for chunk in _chunks(pending, batch_size):
            try:
                raws = local_generate_batch(generate, [r["prompt"] for r, _ in chunk],
                                            batch_size=len(chunk))
                errs: List[Optional[str]] = [None] * len(chunk)
            except Exception as e:                       # 单批 OOM/生成异常 -> 只污染这一批
                raws = [""] * len(chunk)
                errs = [f"{type(e).__name__}: {e}"] * len(chunk)
            for (row, key), raw, err in zip(chunk, raws, errs):
                out.append(_result_row(row, raw, err, model, "local"))
                if cache_f is not None and not err:
                    cache_f.write(json.dumps({
                        "key": key, "model": model, "raw": raw, "error": err,
                        "judge_backend": "local", "ts": time.time()},
                        ensure_ascii=False) + "\n")
                    cache_f.flush()
    finally:
        if cache_f is not None:
            cache_f.close()

    out.sort(key=lambda r: (str(r.get("context_id")), str(r.get("question_idx"))))
    return out


def _row_identity(row: Dict[str, Any]) -> Tuple[str, str]:
    """跨后端对齐用的题目身份：优先 question_id，退化到 (context_id, question_idx)。"""
    qid = row.get("question_id")
    if qid not in (None, ""):
        return ("qid", str(qid))
    return ("idx", f"{row.get('context_id')}::{row.get('question_idx')}")


def compare_backends(rows_a: Sequence[Dict[str, Any]],
                     rows_b: Sequence[Dict[str, Any]],
                     label_a: str = "a", label_b: str = "b") -> Dict[str, Any]:
    """两个后端的逐题判定一致性报告（等价性验证的**唯一**可信方法）。

    判据（任一不满足即视为不等价）：
      1) prompt 完全一致（prompt_sha1 相同）—— 保证两边问的是同一个问题；
      2) verdict 一致率（默认口径 'yes' in text）；
      3) verdict_strict 一致率（严格口径审计，暴露"宽松口径掩盖的抖动"）。
    输出同时给出不一致样本清单，便于人工复核是 judge 抖动还是口径差异。
    """
    ia = {_row_identity(r): r for r in rows_a}
    ib = {_row_identity(r): r for r in rows_b}
    common = sorted(set(ia) & set(ib))
    n = len(common)
    same_prompt = sum(1 for k in common
                      if ia[k].get("prompt_sha1") == ib[k].get("prompt_sha1"))
    both_ok = [k for k in common
               if ia[k].get("verdict") is not None and ib[k].get("verdict") is not None]
    agree = sum(1 for k in both_ok if ia[k]["verdict"] == ib[k]["verdict"])
    strict_ok = [k for k in common
                 if ia[k].get("verdict_strict") is not None
                 and ib[k].get("verdict_strict") is not None]
    strict_agree = sum(1 for k in strict_ok
                       if ia[k]["verdict_strict"] == ib[k]["verdict_strict"])
    diffs = []
    for k in common:
        va, vb = ia[k].get("verdict"), ib[k].get("verdict")
        if va != vb or ia[k].get("prompt_sha1") != ib[k].get("prompt_sha1"):
            diffs.append({
                "id": k[1], "question_type": ia[k].get("question_type"),
                "verdict": {label_a: va, label_b: vb},
                "raw": {label_a: ia[k].get("raw"), label_b: ib[k].get("raw")},
                "prompt_sha1_same": ia[k].get("prompt_sha1") == ib[k].get("prompt_sha1"),
                "question": ia[k].get("question"),
            })
    return {
        "n_a": len(rows_a), "n_b": len(rows_b), "n_common": n,
        "n_only_a": len(set(ia) - set(ib)), "n_only_b": len(set(ib) - set(ia)),
        "prompt_identical": same_prompt,
        "prompt_identical_rate": (same_prompt / n) if n else None,
        "n_verdict_comparable": len(both_ok),
        "n_verdict_agree": agree,
        "verdict_agreement": (agree / len(both_ok)) if both_ok else None,
        "n_strict_comparable": len(strict_ok),
        "n_strict_agree": strict_agree,
        "strict_agreement": (strict_agree / len(strict_ok)) if strict_ok else None,
        "equivalent": bool(n and same_prompt == n and both_ok and agree == len(both_ok)),
        "n_diff": len(diffs),
        "diffs": diffs[:20],
        "note": ("等价性判据：prompt_sha1 全等 + verdict 全一致；"
                 "任一不满足都算不等价（diffs 里是待人工复核的样本）"),
    }


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

    def _rejects(argv: Sequence[str]) -> bool:
        """argparse 应当拒绝该参数组合（choices 校验）。"""
        try:
            _parse_args(list(argv))
        except SystemExit:
            return True
        return False

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

    # 9. local 后端（零权重、零网络：设备/依赖/参数校验 + 注入假权重跑批）
    ok("默认 judge 后端是 http", _parse_args([]).judge_backend == "http")
    ok("--judge-backend local 可解析",
       _parse_args(["--judge-backend", "local", "--input", "x"]).judge_backend == "local")
    ok("--judge-device 非法值被 argparse 拒绝",
       _rejects(["--judge-device", "tpu", "--input", "x"]))

    # 9a 设备解析：无 GPU 必须**明确报错**，不许静默跑 CPU 也不许回退 http
    err_auto = None
    try:
        resolve_local_device("auto", cuda_available=False, mps_available=False)
    except LocalBackendError as e:
        err_auto = str(e)
    ok("无 GPU 时 auto 抛 LocalBackendError", err_auto is not None and "GPU" in err_auto)
    ok("报错同时给出 --judge-device cpu 与 --judge-backend http 两条出路",
       "--judge-device cpu" in (err_auto or "") and "--judge-backend http" in (err_auto or ""))
    ok("显式 cpu 才允许 CPU",
       resolve_local_device("cpu", cuda_available=False, mps_available=False) == "cpu")
    ok("有 CUDA 时 auto -> cuda",
       resolve_local_device("auto", cuda_available=True, mps_available=False) == "cuda")
    err_cuda = None
    try:
        resolve_local_device("cuda", cuda_available=False, mps_available=False)
    except LocalBackendError as e:
        err_cuda = str(e)
    ok("显式 cuda 但不可用时报错", err_cuda is not None and "CUDA" in err_cuda)
    err_dev = None
    try:
        resolve_local_device("tpu", cuda_available=True)
    except LocalBackendError as e:
        err_dev = str(e)
    ok("未知设备名报错", err_dev is not None and "未知" in err_dev)

    # 9b 权重/依赖：先做便宜的路径校验，缺依赖时给可执行提示（都不 import transformers）
    err_nomodel = None
    try:
        load_local_judge("")
    except LocalBackendError as e:
        err_nomodel = str(e)
    ok("缺 --judge-model 报错并说明不回退 http",
       err_nomodel is not None and "--judge-model" in err_nomodel and "回退" in err_nomodel)
    err_nopath = None
    try:
        load_local_judge("/nonexistent/qwen-judge")
    except LocalBackendError as e:
        err_nopath = str(e)
    ok("本地权重路径不存在时报错", err_nopath is not None and "不存在" in err_nopath)

    # 9c 批量生成：按 batch 切块、保序、条数不符要报错
    calls: List[int] = []

    def fake_gen(batch):
        calls.append(len(batch))
        return ["yes" if i % 2 == 0 else "no" for i in range(len(batch))]

    raws = local_generate_batch(fake_gen, ["p1", "p2", "p3", "p4", "p5"], batch_size=2)
    ok("batch 切块为 [2,2,1]", calls == [2, 2, 1])
    ok("batch 生成保序", raws == ["yes", "no", "yes", "no", "yes"])
    err_align = None
    try:
        local_generate_batch(lambda b: ["yes"], ["p1", "p2"], batch_size=2)
    except LocalBackendError as e:
        err_align = str(e)
    ok("generate 返回条数与 batch 不符时报错", err_align is not None and "对齐" in err_align)

    # 9d run_judge_local：注入假权重，验证 prompt/verdict/字段与 http 完全一致
    saved_loader = globals()["load_local_judge"]
    seen: Dict[str, Any] = {}
    gen_prompts: List[str] = []

    def fake_loader(model_path, device="auto", dtype="auto",
                    max_new_tokens=LOCAL_MAX_NEW_TOKENS, batch_size=LOCAL_BATCH_SIZE):
        seen.update({"model": model_path, "device": device, "dtype": dtype,
                     "max_new_tokens": max_new_tokens, "batch_size": batch_size})
        seen["loads"] = seen.get("loads", 0) + 1

        def g(batch):
            gen_prompts.extend(batch)
            return ["yes"] * len(batch)
        return g

    def mk_local_args(**kw):
        base = dict(model="gpt-4o-2024-08-06", base_url="", api_key="", workers=1,
                    retries=1, timeout=1, cache="", no_resume=True,
                    judge_backend="local", judge_model="Qwen/Qwen2.5-7B-Instruct",
                    judge_device="auto", judge_dtype="auto", judge_batch_size=2,
                    judge_max_new_tokens=LOCAL_MAX_NEW_TOKENS)
        base.update(kw)
        return argparse.Namespace(**base)

    globals()["load_local_judge"] = fake_loader
    try:
        local_rows = run_judge(rows2, mk_local_args())
        ok("local 后端产出全部题目判定", len(local_rows) == 3)
        ok("local 与 http 用同一套 prompt（prompt_sha1 全等）",
           [r["prompt_sha1"] for r in local_rows] == [r["prompt_sha1"] for r in rows2])
        ok("local verdict 与 http 同语义（'yes' in text -> 1）",
           all(r["verdict"] == 1 and r["verdict_strict"] == 1 for r in local_rows))
        ok("local 记录 judge_backend=local 与权重名",
           all(r["judge_backend"] == "local" and r["model"] == "Qwen/Qwen2.5-7B-Instruct"
               for r in local_rows))
        ok("显存保护参数透传：max_new_tokens=10、batch=2、do_sample=False",
           seen.get("max_new_tokens") == 10 and seen.get("batch_size") == 2
           and LOCAL_DO_SAMPLE is False)
        ok("权重只加载一次（3 题 1 次 load、2 次 generate）",
           seen.get("loads") == 1 and len(gen_prompts) == 3)

        # 9e 缓存隔离：http 后端的缓存条目不会被 local 复用（model 名不同 -> key 不同）
        with tempfile.TemporaryDirectory() as td:
            mix = os.path.join(td, "cache_mix.jsonl")
            with open(mix, "w", encoding="utf-8") as f:
                for r in rows2:
                    f.write(json.dumps({"key": cache_key("gpt-4o-2024-08-06", r["prompt"]),
                                        "model": "gpt-4o-2024-08-06", "raw": "yes",
                                        "error": None}, ensure_ascii=False) + "\n")
            gen_prompts.clear()
            run_judge(rows2, mk_local_args(cache=mix, no_resume=False))
            ok("http 缓存不被 local 复用（仍重新生成）", len(gen_prompts) == 3)
            # local 自己写进缓存后，第二次应全命中
            gen_prompts.clear()
            run_judge(rows2, mk_local_args(cache=mix, no_resume=False))
            ok("local 自己写的缓存第二次全命中", gen_prompts == [])

        # 9f 后端不可用时抛错，绝不回退 http
        orig_post2 = globals()["_http_post_json"]
        globals()["_http_post_json"] = _boom
        try:
            def dead_loader(*a, **k):
                raise LocalBackendError("模拟：本机无可用 GPU")
            globals()["load_local_judge"] = dead_loader
            err_fallback = None
            try:
                run_judge(rows2, mk_local_args())
            except LocalBackendError as e:
                err_fallback = str(e)
            ok("local 不可用 -> 抛 LocalBackendError（未回退 http）",
               err_fallback is not None and "无可用 GPU" in err_fallback)
        finally:
            globals()["_http_post_json"] = orig_post2

        # 9g 等价性比较工具：同源 -> PASS；一处翻转 -> FAIL 且列出差异
        rep_same = compare_backends(judged, judged, "local", "http")
        ok("同源判定比较 -> 等价", rep_same["equivalent"] is True
           and rep_same["verdict_agreement"] == 1.0 and rep_same["n_diff"] == 0)
        flipped = [dict(r) for r in judged]
        flipped[0]["verdict"] = 0
        rep_diff = compare_backends(judged, flipped, "local", "http")
        ok("一处 verdict 不同 -> 不等价且列出 1 条差异",
           rep_diff["equivalent"] is False and rep_diff["n_diff"] == 1)
        ok("差异条目带两边 raw/verdict 与题型",
           rep_diff["diffs"][0]["question_type"] and "verdict" in rep_diff["diffs"][0]
           and "raw" in rep_diff["diffs"][0])
        sha_flip = [dict(r) for r in judged]
        sha_flip[0]["prompt_sha1"] = "deadbeef"
        rep_sha = compare_backends(judged, sha_flip, "local", "http")
        ok("prompt 不一致也算不等价（prompt_identical 暴露口径漂移）",
           rep_sha["equivalent"] is False
           and rep_sha["prompt_identical"] == len(judged) - 1)
    finally:
        globals()["load_local_judge"] = saved_loader

    # 9h local dry-run：零权重、零网络、退出码 0；缺 --judge-model 直接失败
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "in_local.jsonl")
        with open(inp, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        pout = os.path.join(td, "prompts_local.jsonl")
        # 同样上禁网哨兵：local dry-run 也必须零网络
        orig_post3 = globals()["_http_post_json"]
        globals()["_http_post_json"] = _boom
        try:
            rc_local = main(["--input", inp, "--dry-run", "--judge-backend", "local",
                             "--judge-model", "Qwen/Qwen2.5-7B-Instruct",
                             "--prompts-out", pout])
        finally:
            globals()["_http_post_json"] = orig_post3
        ok("local dry-run 退出码 0", rc_local == 0)
        ok("local dry-run 写出 prompt 文件", os.path.exists(pout))
        ok("local dry-run 未加载任何权重/依赖", _TRANSFORMERS_IMPORTS == 0)
        msg_missing = ""
        try:
            main(["--input", inp, "--dry-run", "--judge-backend", "local"])
        except SystemExit as e:
            msg_missing = str(e)
        ok("local 缺 --judge-model 直接失败并写明不回退 http",
           "judge-model" in msg_missing and "回退" in msg_missing)

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
    ap.add_argument("--judge-backend", choices=("http", "local"), default="http",
                    help="judge 后端：http=OpenAI 兼容接口（默认）/ local=本地权重")
    ap.add_argument("--judge-model", default=os.environ.get("LME_JUDGE_LOCAL_MODEL", ""),
                    help="local 后端的权重目录或 HF 模型名（如 Qwen/Qwen2.5-7B-Instruct）；"
                         "http 后端忽略此项")
    ap.add_argument("--judge-device", default=os.environ.get("LME_JUDGE_DEVICE", "auto"),
                    choices=LOCAL_DEVICES,
                    help="local 后端设备；默认 auto=有 GPU 才跑，纯 CPU 必须显式 cpu")
    ap.add_argument("--judge-dtype", default="auto", choices=LOCAL_DTYPES,
                    help="local 后端权重精度；auto=GPU 用 float16、CPU 用 float32")
    ap.add_argument("--judge-batch-size", type=int, default=LOCAL_BATCH_SIZE,
                    help="local 后端每次 generate 的 batch 大小（显存不够就调小）")
    ap.add_argument("--judge-max-new-tokens", type=int, default=LOCAL_MAX_NEW_TOKENS,
                    help=f"local 后端生成长度（默认 {LOCAL_MAX_NEW_TOKENS}，与 http max_tokens 对齐）")
    ap.add_argument("--compare-backends", nargs=2, default=None,
                    metavar=("A_JSONL", "B_JSONL"),
                    help="不判题：比较两份逐题判定 JSONL 的一致性（后端等价性验证）")
    ap.add_argument("--compare-out", default="", help="把一致性报告写成 JSON")
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


def run_compare(args) -> int:
    """`--compare-backends A B`：两个后端逐题判定的一致性报告（不判题、不联网）。"""
    pa, pb = args.compare_backends
    for path in (pa, pb):
        if not os.path.exists(path):
            raise SystemExit(f"[ERR] 待比较文件不存在: {path}")
    ra, rb = load_jsonl(pa), load_jsonl(pb)
    if not ra or not rb:
        raise SystemExit(f"[ERR] 待比较文件为空: {pa if not ra else pb}")
    rep = compare_backends(ra, rb,
                           label_a=f"{(ra[0].get('judge_backend') or 'a')}",
                           label_b=f"{(rb[0].get('judge_backend') or 'b')}")
    rep["file_a"], rep["file_b"] = pa, pb
    rep["model_a"] = ra[0].get("model")
    rep["model_b"] = rb[0].get("model")
    print(f"== 后端等价性比较 ==")
    print(f"  A: {pa}  backend={ra[0].get('judge_backend')} model={rep['model_a']}")
    print(f"  B: {pb}  backend={rb[0].get('judge_backend')} model={rep['model_b']}")
    print(f"  题目数 A={rep['n_a']} B={rep['n_b']} 共同={rep['n_common']} "
          f"仅A={rep['n_only_a']} 仅B={rep['n_only_b']}")
    print(f"  prompt 逐字一致: {rep['prompt_identical']}/{rep['n_common']} "
          f"({_fmt(rep['prompt_identical_rate'])})")
    print(f"  verdict 一致: {rep['n_verdict_agree']}/{rep['n_verdict_comparable']} "
          f"({_fmt(rep['verdict_agreement'])})")
    print(f"  verdict_strict 一致: {rep['n_strict_agree']}/{rep['n_strict_comparable']} "
          f"({_fmt(rep['strict_agreement'])})")
    print(f"  等价性判定: {'PASS' if rep['equivalent'] else 'FAIL'}  "
          f"不一致 {rep['n_diff']} 条")
    for d in rep["diffs"]:
        print(f"    - {d['id']} [{d['question_type']}] verdict={d['verdict']} "
              f"prompt_sha1_same={d['prompt_sha1_same']}")
    print(f"  {rep['note']}")
    if args.compare_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.compare_out)) or ".", exist_ok=True)
        with open(args.compare_out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"[out] 比较报告 -> {args.compare_out}")
    return 0 if rep["equivalent"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.compare_backends:
        return run_compare(args)
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

    # local 后端参数校验：在 dry-run / 真判题之前就失败，且**绝不**回退到 http
    try:
        backend = judge_backend(args)
    except LocalBackendError as e:
        raise SystemExit(f"[ERR] {e}")
    if backend == "local" and not str(args.judge_model).strip():
        raise SystemExit(
            "[ERR] --judge-backend local 需要 --judge-model（本地权重目录或 HF 模型名，"
            "如 Qwen/Qwen2.5-7B-Instruct）。\n"
            "      脚本不会静默回退到 http 后端；若想用接口判题请显式 --judge-backend http")
    if backend == "local" and args.dry_run:
        # dry-run 只验证 prompt，不加载权重：设备检查留给真判题，但先把结论说出来
        try:
            resolve_local_device(args.judge_device)
        except LocalBackendError as e:
            print(f"[warn] --dry-run 不加载权重、不做设备硬校验；真判题时这会失败：{e}",
                  file=sys.stderr)

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

    if backend == "http" and not args.api_key:
        raise SystemExit("[ERR] 缺少 judge API key：设 LME_JUDGE_API_KEY 或 --api-key"
                         "（仅想检查 prompt 请加 --dry-run）")
    if not args.out:
        raise SystemExit("[ERR] 真判题需要 --out 指定逐题判定输出")

    t0 = time.time()
    try:
        judged = run_judge(rows, args)
    except LocalBackendError as e:
        # 本地后端不可用是硬错误：给出修复路径并退出非零，绝不回退 http
        raise SystemExit(f"[ERR] 本地 Judge 后端不可用：\n      {e}")
    n = _write_jsonl(args.out, judged)
    summary = summarize_judged(judged)
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    summary["input"] = args.input
    summary["dataset"] = args.dataset or None
    summary["judge_backend"] = backend
    summary["judge_model"] = judge_model_name(args)
    if backend == "local":
        summary["judge_device"] = args.judge_device
        summary["judge_batch_size"] = args.judge_batch_size
        summary["judge_max_new_tokens"] = args.judge_max_new_tokens
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
