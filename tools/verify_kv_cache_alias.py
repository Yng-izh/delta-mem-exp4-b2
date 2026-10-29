#!/usr/bin/env python
"""验证 Delta-Mem 包装器的 KV Cache 参数名兼容补丁（deltamem_compat.patch_kv_cache_alias）。

两项检查（用户指定）：
  1. 带 adapter、`input_ids + use_cache=True` 生成是否正常；
  2. 整段 prompt 前向后 `outputs.past_key_values.get_seq_length()` 是否等于输入 token 数。

对照组：`DELTA_NO_KV_PATCH=1` 关掉补丁重跑同一份权重，用于确认差异确实来自补丁。

用法：
  PYTHONPATH=<B>/nodns:<R>:<B>/delta-Mem python tools/verify_kv_cache_alias.py \
      --base <B>/models/Qwen3-4B-Instruct-2507 \
      --adapter <B>/models/delta-mem_qwen3_4b_instruct --gpu 0
  （探针只需一张卡；`--no-adapter` 可做纯主干对照）
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import train_latent_memory as T  # noqa: E402

QUESTION = ("Which film has the director who is also the screenwriter of "
            "The Third Part of the Night?")


def build_prompt(tok):
    msgs = [{"role": "user", "content": QUESTION +
             "\nAnswer the question with a short phrase."}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return "Question: " + QUESTION + "\nAnswer:"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--smoke-model", action="store_true")
    a = ap.parse_args(argv)
    a.grad_checkpointing = False

    import deltamem_compat as DMC
    patched = DMC.patch_kv_cache_alias()
    print(f"[patch] 本次是否新打补丁={patched}；"
          f"DELTA_NO_KV_PATCH={os.environ.get('DELTA_NO_KV_PATCH', '')!r}", flush=True)

    model, tok, dev, _hidden = T._load_backbone(a, torch.device(f"cuda:{a.gpu}"))
    model.eval()

    prompt = build_prompt(tok)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    n_tok = int(enc["input_ids"].shape[1])
    print(f"[prompt] {n_tok} tokens", flush=True)

    # ---- 检查 2：整段 prompt 的缓存长度 ----
    with torch.no_grad():
        out = model(**enc, use_cache=True, return_dict=True)
    pkv = out.past_key_values
    if pkv is None:
        seq_len = -1
    elif hasattr(pkv, "get_seq_length"):
        seq_len = int(pkv.get_seq_length())
    else:
        try:
            seq_len = int(pkv[0][0].shape[-2])
        except Exception:
            seq_len = -2
    print(f"[check2] past_key_values seq_len={seq_len} vs 输入 {n_tok} "
          f"→ {'OK' if seq_len == n_tok else 'FAIL'}", flush=True)

    # ---- 检查 1：带缓存生成 ----
    with torch.no_grad():
        gen = model.generate(**enc, use_cache=True, do_sample=False,
                             max_new_tokens=a.max_new_tokens,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    text = tok.decode(gen[0, n_tok:], skip_special_tokens=True)
    print(f"[check1] gen(use_cache=True) = {text!r}", flush=True)

    # 参考：一次性前向给 logits 的 argmax，用于区分"缓存坏了"还是"模型本身不会"
    with torch.no_grad():
        nxt = out.logits[0, -1].argmax().item()
    print(f"[ref] 单次前向 argmax 首 token = {tok.decode([nxt])!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
