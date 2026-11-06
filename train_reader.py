#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_reader.py — 实验 A / B：训练「完整隐藏状态记忆 + 轻量 Query-conditioned Reader」。

两个阶段共用一个脚本，只换 `--mem`：

    实验 A（验证读取接口）：--mem full
        记忆 = 金标证据的**逐 token 隐藏状态**（不压缩，通常 100~200 个位置），
        只训 Reader。必须证明 full 明显高于 none 与 shuffle。

    实验 B（验证潜层压缩）：--mem slots8 / slots4 / slots2
        记忆 = 可训练压缩器输出的 K 个潜槽，Reader 结构**完全不变**（--init 从 A 的
        Reader 初始化，可选 --freeze-reader 只训压缩器）。判据是"同读取预算下超过
        Query-Agnostic 原生 Token 基线，同时保持关系绑定准确率"。

训练目标：`L = L_QA + λ · L_fact_recon`

只靠 QA 损失是不够的 —— 答案通常只有一两个 token，要它去学"把 2560 维的第 18 层
隐藏状态翻译成输入词向量空间"这种大跨度映射，信号既稀又高方差（实测 750 步、
`full` 一直贴着 `none` 的 17.5%）。所以加一条**密集的事实重建损失**：

  - 让模型用记忆复述证据原文，但**只在事实关键位算损失**（数字 / 实体 / 关系词），
    见 `reader_memory.key_span_flags`。整句重建会把梯度花在可预测的功能词上，
    Reader 学到的与"记住事实"无关 —— 旧实验里"重建 tokF1 涨了但 QA 不涨"可能就是
    这个原因。
  - 前缀随机破坏（`--recon-corrupt`）：Teacher Forcing 下模型能靠已给的金标前缀
    自回归猜出下一个词，不破坏前缀的话重建会退化成语言模型练习。

评测固定跑五条（缺一不可）：
    none      不给记忆                    —— 冻结模型的参数知识下界
    shuffle   给记忆但打乱 token 顺序      —— 顺序/绑定是否被用到
    wrong     给**别的样本**的记忆          —— **主反事实**：问题对、记忆错
    full      给完整记忆                  —— 这条读取通路的上限
    oracle_text 明文原文进 system          —— 参照上界（与 Reader 无关）

为什么主反事实是 `wrong` 而不是 `shuffle`：第 18 层隐藏状态本身编码了上下文，
打乱 token 顺序之后"这是哪条事实"的信息未必消失。所以 `full ≈ shuffle` 不能证明
模型没在用记忆；只有 `full ≫ wrong` 才是"确实在读这段记忆"的硬证据。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import torch

import latent_memory as LM
import train_latent_memory as T
import reader_memory as RM
import eval_latent_table as EV


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="实验 A/B：完整记忆 + Query-conditioned Reader")
    ap.add_argument("--mem", choices=RM.READER_MEM_MODES, default="full",
                    help="记忆来源：full=完整隐藏状态（实验 A）；slotsK=压缩成 K 个潜槽（实验 B）")
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-data", default="")
    ap.add_argument("--n-train", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=120)
    ap.add_argument("--layer", type=int, default=RM.READER_LAYER)
    ap.add_argument("--bottleneck", type=int, default=RM.READER_BOTTLENECK)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--max-mem-tokens", type=int, default=RM.READER_MEM_MAX)
    ap.add_argument("--inner-dim", type=int, default=LM.LATENT_INNER_DIM,
                    help="实验 B 的压缩器内部宽度")
    ap.add_argument("--heads", type=int, default=LM.LATENT_N_HEADS)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-limit", type=int, default=40)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--fact-recon-weight", type=float, default=0.5,
                    help="事实重建损失的权重（0=关闭，退回纯 QA 基线 A）")
    ap.add_argument("--recon-corrupt", type=float, default=0.5,
                    help="重建时随机破坏输入侧前缀的概率（防止靠已知前缀自回归猜答案）")
    ap.add_argument("--recon-max-tokens", type=int, default=96,
                    help="重建目标最多取多少 token（省显存，不影响方法定义）")
    ap.add_argument("--shuffle-train", choices=("none", "tokens"), default="none",
                    help="训练时是否打乱记忆。默认不打乱；把它当正则/鲁棒性对照时再开")
    ap.add_argument("--init", default="", help="从已有 reader（+压缩器）ckpt 初始化")
    ap.add_argument("--freeze-reader", action="store_true",
                    help="实验 B 第二步：固定 Reader，只训压缩器")
    ap.add_argument("--freeze-compressor", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=T.BASE)
    ap.add_argument("--adapter", default=T.ADAPTER)
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--device", default="")
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--smoke-model", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inject-mode", choices=("user", "system"), default="user")
    ap.add_argument("--fence-style", choices=("bracket",), default="bracket")
    ap.add_argument("--system-prompt",
                    default="You are a helpful assistant. Answer with the shortest exact value.")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------
# 记忆与查询特征
# ---------------------------------------------------------------------
def gold_units(item) -> List[int]:
    """金标证据单元下标（没有 evidence 就返回空 —— 不猜）。"""
    units = T.item_units(item)
    ev = {T._norm_ans(e) for e in (item.get("evidence") or [])}
    if not ev:
        return []
    return [i for i, u in enumerate(units) if T._norm_ans(u) in ev]


def build_memory(args, model, tokenizer, compressor, item) -> Dict[str, Any]:
    idx = gold_units(item)
    if not idx:
        return {"h": torch.zeros((1, 0, int(model.config.hidden_size))),
                "mask": torch.zeros((1, 0), dtype=torch.long), "n": 0, "texts": []}
    if args.mem == "full":
        return RM.build_full_memory(model, tokenizer, item, idx, layer=int(args.layer),
                                    max_tokens=int(args.max_mem_tokens))
    return RM.build_slot_memory(model, tokenizer, compressor, item, idx,
                                max_tokens=int(args.max_mem_tokens),
                                layer=int(args.layer))


def query_vec(model, tokenizer, item, args) -> torch.Tensor:
    """问题特征：取第 layer 层的 mean-pool（与记忆同一层）。

    `extract_query_feature_ex` 返回 cat([h0.mean, hL.mean])，后半段就是记忆那一层，
    取它而不是再训一个 query encoder —— Reader 已经够轻，再叠一层就说不清是谁的功劳。
    """
    qf = LM.extract_query_feature_ex(model, tokenizer, item["question"],
                                     args.system_prompt, layer=int(args.layer))
    pooled = qf["pooled"]
    d = int(pooled.shape[-1]) // 2
    return pooled[:, d:]


# ---------------------------------------------------------------------
# 事实重建损失（密集监督）
# ---------------------------------------------------------------------
def fact_recon_loss(args, model, tokenizer, reader, mem, q, rng) -> torch.Tensor:
    """用记忆复述证据原文，**只在事实关键位上算损失**。

    返回标量张量；没有可监督位置时返回 0（不抛错 —— 有些样本全是功能词，属正常）。

    三个防作弊设计，缺一个这条损失就会退化成语言模型练习：
      1. 目标文本**不在输入里**（输入只有 soft tokens + 一句固定指令）；
      2. 只监督关键位（数字/实体/关系词），功能词不给分；
      3. 以 `--recon-corrupt` 的概率把输入侧的答案前缀换成随机词表 token，
         否则 Teacher Forcing 下模型能靠已知前缀自回归猜出下一个词。
    """
    texts = list(mem.get("texts") or [])
    if not texts:
        return torch.zeros((), device=q.device)
    recon = " ".join(str(t) for t in texts)
    ids = tokenizer(recon, add_special_tokens=False)["input_ids"][0].tolist()
    if int(args.recon_max_tokens) > 0:
        ids = ids[: int(args.recon_max_tokens)]
    if not ids:
        return torch.zeros((), device=q.device)
    token_texts = [tokenizer.decode([int(i)]) for i in ids]
    flags = RM.key_span_flags(token_texts)
    if not any(flags):
        return torch.zeros((), device=q.device)
    recon_trunc = tokenizer.decode([int(i) for i in ids], skip_special_tokens=True)

    soft = reader(mem["h"], mem["mask"], q)
    embeds, labels, n_ans = T.build_student_inputs(
        model, tokenizer, soft, args.system_prompt,
        "Repeat the memory verbatim.", recon_trunc, inject_mode=args.inject_mode)
    # labels 覆盖"答案 token + EOS"；把非关键位抹成 -100
    lab = labels.clone()
    if n_ans > 0:
        f = torch.zeros((n_ans,), dtype=torch.bool, device=lab.device)
        k = min(int(len(flags)), int(n_ans))
        f[:k] = torch.tensor(flags[:k], dtype=torch.bool, device=lab.device)
        region = lab[0, lab.shape[1] - n_ans:]
        region[~f] = -100
    if not bool((lab != -100).any()):
        return torch.zeros((), device=q.device)
    if float(args.recon_corrupt) > 0.0 and n_ans > 1 and \
            float(torch.rand((), generator=rng)) < float(args.recon_corrupt):
        first = int(embeds.shape[1]) - int(n_ans)
        n_rep = int(n_ans) - 1
        vocab = int(getattr(model.config, "vocab_size", 0) or 0)
        if vocab > 0 and n_rep > 0:
            rand_ids = torch.randint(0, vocab, (1, n_rep), generator=rng).to(embeds.device)
            repl = model.get_input_embeddings()(rand_ids).to(embeds.dtype)
            embeds = embeds.clone()
            embeds[:, first:first + n_rep] = repl
    with LM.delta_write_disabled(model):
        logits = model(inputs_embeds=embeds,
                       attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long,
                                                 device=embeds.device),
                       use_cache=False).logits
    return T.qa_loss(logits, lab)


# ---------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------
def train(args) -> int:
    dev = torch.device(args.device if args.device else
                       ("cpu" if args.smoke_model else "cuda:0"))
    T._setup_tf32(not getattr(args, "no_tf32", False))
    model, tokenizer, dev, hidden = T._load_backbone(args, dev)
    T.freeze_model(model)
    torch.manual_seed(int(args.seed))

    with torch.no_grad():
        target_norm = float(model.get_input_embeddings().weight.float()
                            .norm(dim=-1).mean())
    reader = RM.QueryCondReader(hidden, bottleneck=int(args.bottleneck),
                                target_norm=target_norm, tau=float(args.tau)).to(dev)
    reader._mem_mode = str(args.mem)
    compressor = None
    if args.mem != "full":
        k = int(str(args.mem).replace("slots", ""))
        compressor = LM.LatentCompressor(hidden, n_slots=k, inner_dim=int(args.inner_dim),
                                         num_heads=int(args.heads), lex_skip=False).to(dev)
        compressor.max_unit_tokens = int(getattr(args, "max_unit_tokens",
                                                 LM.LATENT_MAX_UNIT_TOKENS))
    if args.init:
        r2, c2, cfg2 = RM.load_reader_checkpoint(args.init, model=model, device=dev)
        if r2 is not None:
            # 结构必须一致：实验 B 的第一步就是"沿用 A 的 Reader"，形状不符要立刻报
            if (int(r2.hidden_size), int(r2.bottleneck)) != (hidden, int(args.bottleneck)):
                raise SystemExit(f"[init] Reader 结构不一致：ckpt={r2.hidden_size}/"
                                 f"{r2.bottleneck}，当前={hidden}/{args.bottleneck}")
            reader.load_state_dict(r2.state_dict(), strict=True)
            print(f"[init] Reader 从 {args.init} 载入", flush=True)
        if c2 is not None and compressor is not None:
            compressor.load_state_dict(c2.state_dict(), strict=True)
            print(f"[init] 压缩器从 {args.init} 载入", flush=True)
        elif c2 is not None and compressor is None:
            print("[init][warn] ckpt 里有压缩器但 --mem full，忽略", flush=True)

    params: List[torch.nn.Parameter] = []
    if not args.freeze_reader:
        params += list(reader.parameters())
    if compressor is not None and not args.freeze_compressor:
        params += list(compressor.parameters())
    if not params:
        raise SystemExit("[错误] 没有任何可训练参数（reader 与 compressor 都被冻结）")
    opt = torch.optim.AdamW(params, lr=float(args.lr), betas=(0.9, 0.95))
    n_param = sum(p.numel() for p in params)
    print(f"[params] 可训练 {n_param/1e6:.3f}M（mem={args.mem} "
          f"reader_frozen={args.freeze_reader} comp_frozen={args.freeze_compressor}）",
          flush=True)

    data = T.load_synth_jsonl(args.data, limit=int(args.n_train or 0), seed=int(args.seed))
    if not data:
        raise SystemExit(f"{args.data} 里没有样本")
    val = []
    if args.val_data:
        val = T.load_synth_jsonl(args.val_data, limit=int(args.n_val or 0),
                                 seed=int(args.seed))
    print(f"[data] train={len(data)} val={len(val)} 记忆模式={args.mem}", flush=True)

    rng = torch.Generator().manual_seed(int(args.seed))
    step = 0
    micro = 0
    t0 = time.time()
    seen = 0
    while step < int(args.steps):
        for item in data:
            mem = build_memory(args, model, tokenizer, compressor, item)
            if int(mem["n"]) <= 0:
                continue      # 没有金标证据：这条样本对"读记忆"没有任何信息，跳过
            if args.shuffle_train != "none":
                mem = RM.shuffle_memory(mem, args.shuffle_train, rng=rng)
            q = query_vec(model, tokenizer, item, args)
            soft = reader(mem["h"], mem["mask"], q)
            embeds, labels, n_ans = T.build_student_inputs(
                model, tokenizer, soft, args.system_prompt, item["question"],
                item["answer"], inject_mode=args.inject_mode)
            with LM.delta_write_disabled(model):
                logits = model(inputs_embeds=embeds,
                               attention_mask=torch.ones(embeds.shape[:2],
                                                         dtype=torch.long, device=embeds.device),
                               use_cache=False).logits
            loss_qa = T.qa_loss(logits, labels)
            loss_rec = torch.zeros((), device=loss_qa.device)
            if float(args.fact_recon_weight) > 0.0:
                loss_rec = fact_recon_loss(args, model, tokenizer, reader, mem, q, rng)
            loss = (loss_qa + float(args.fact_recon_weight) * loss_rec) / max(1, int(args.accum))
            loss.backward()
            micro += 1
            seen += 1
            if micro % int(args.accum) == 0:
                torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % int(args.log_every) == 0:
                    el = time.time() - t0
                    print(f"[step {step}] loss_qa={float(loss_qa):.4f} "
                          f"loss_recon={float(loss_rec):.4f} "
                          f"mem_tokens={int(mem['n'])} elapsed={el:.0f}s "
                          f"({el / max(1, step):.2f}s/step)", flush=True)
                if int(args.eval_every) > 0 and step % int(args.eval_every) == 0:
                    run_eval(args, model, tokenizer, reader, compressor, val, step, dev)
                if step >= int(args.steps):
                    break
        if step >= int(args.steps):
            break

    if args.dry_run:
        print(f"[dry-run] OK：跑了 {step} 步，未写 checkpoint", flush=True)
        return 0
    RM.save_reader_checkpoint(
        args.out, reader, compressor,
        config=RM.reader_config(reader, compressor,
                                {"mem": str(args.mem), "layer": int(args.layer),
                                 "max_mem_tokens": int(args.max_mem_tokens),
                                 "shuffle_train": str(args.shuffle_train),
                                 "inject_mode": str(args.inject_mode),
                                 "steps": int(args.steps), "lr": float(args.lr)}),
        stats={"steps": step, "seen": seen})
    print(f"[save] {args.out}", flush=True)
    if val:
        run_eval(args, model, tokenizer, reader, compressor, val, step, dev)
    return 0


# ---------------------------------------------------------------------
# 评测：none / shuffle / full / oracle_text
# ---------------------------------------------------------------------
@torch.no_grad()
def run_eval(args, model, tokenizer, reader, compressor, val, step, dev) -> Dict[str, Any]:
    if not val:
        return {}
    reader.eval()
    if compressor is not None:
        compressor.eval()
    lim = int(args.eval_limit or len(val))
    items = list(val[:lim])
    rows: List[Dict[str, Any]] = []
    rng = torch.Generator().manual_seed(int(args.seed) + 7)
    # 先把所有记忆建好：wrong-memory 对照要拿**别的样本**的记忆，必须预先把整批建出来
    mems = [build_memory(args, model, tokenizer, compressor, it) for it in items]
    n_i = len(items)
    for i, item in enumerate(items):
        mem = mems[i]
        wrong = mems[(i + 1) % n_i] if n_i > 1 else mems[i]
        q = query_vec(model, tokenizer, item, args)
        out: Dict[str, Any] = {"idx": i, "gold": str(item["answer"])}
        for mode in ("none", "shuffle", "wrong", "full"):
            if mode == "none":
                m = None
            elif mode == "shuffle":
                m = RM.shuffle_memory(mem, "tokens", rng=rng)
            elif mode == "wrong":
                # 主反事实：**问题对、记忆错**。若 full 与它没有明显差距，
                # 说明模型根本没在用这段记忆的内容。
                m = wrong
            else:
                m = mem
            pred, _meta = RM.reader_generate(
                model, tokenizer, reader, m, item["question"], args.system_prompt,
                q_vec=q, max_new_tokens=16, inject_mode=args.inject_mode,
                fence_style=args.fence_style)
            out[mode] = str(pred)
        # 明文参照：把金标原文写进 system（与 Reader 无关）
        idx = gold_units(item)
        units = T.item_units(item)
        block = "Memory:\n" + "\n".join("- " + units[j] for j in idx) + "\n" if idx else ""
        pred_t, _m = LM._emit_latent(model, tokenizer, None, item["question"],
                                     args.system_prompt + "\n\n" + block,
                                     max_new_tokens=16)
        out["oracle_text"] = str(pred_t)
        rows.append(out)
    summary: Dict[str, Any] = {"step": int(step), "n": len(rows)}
    for mode in ("none", "shuffle", "wrong", "full", "oracle_text"):
        hit = sum(1 for r in rows if T._norm_ans(r[mode]) == T._norm_ans(r["gold"]))
        summary[mode] = hit / len(rows)
    # **关键量**：full 与 wrong 的差（问题对、记忆错）。这是"确实在读这段记忆"的硬证据；
    # full-shuffle 只说明"顺序/绑定有没有被用到"，打乱后上下文信息未必消失，所以只作参考。
    summary["full_minus_shuffle"] = summary["full"] - summary["shuffle"]
    summary["full_minus_wrong"] = summary["full"] - summary["wrong"]
    summary["full_minus_none"] = summary["full"] - summary["none"]
    print(f"[eval] step={step} n={len(rows)} "
          f"none={summary['none']:.3f} shuffle={summary['shuffle']:.3f} "
          f"wrong={summary['wrong']:.3f} full={summary['full']:.3f} "
          f"oracle_text={summary['oracle_text']:.3f} "
          f"| **full-wrong={summary['full_minus_wrong']:+.3f}** "
          f"full-shuffle={summary['full_minus_shuffle']:+.3f} "
          f"full-none={summary['full_minus_none']:+.3f}", flush=True)
    reader.train()
    if compressor is not None:
        compressor.train()
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    print(f"[cfg] mem={args.mem} layer={args.layer} bottleneck={args.bottleneck} "
          f"lr={args.lr} steps={args.steps} accum={args.accum} "
          f"inject={args.inject_mode} shuffle_train={args.shuffle_train}", flush=True)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
