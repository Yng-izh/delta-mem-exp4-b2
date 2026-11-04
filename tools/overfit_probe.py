#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/overfit_probe.py — P1-4：64 题小样本过拟合 + 三个目标函数的对照。

要回答的两个问题
----------------
1. **能不能拟合**：固定 64 道训练题，只训 Compressor + Bridge（Router 冻结），
   在同一批题上测 oracle_latent 自由生成 EM。
     连训练集都拟合不上 -> 优先查梯度/接口/读出通路（而不是泛化、容量）；
     能拟合却不泛化     -> 才是信息瓶颈或数据问题。
2. **重建损失有没有伤害主任务**：同一份初始化、同一批题、同样步数，只换目标：
     qa_kl      ：QA + KL，不加重建（干净对照）
     recon_old  ：QA + KL + 重建，重建走 bridge(z_sem)（旧写法，与 QA 目标不一致）
     recon_qa   ：QA + KL + 重建，重建走 bridge(z_sem, z_lex)（与 QA 同一条路径）
   三条曲线在同一步数上的差值，就是"重建目标是否在拖主任务"的直接证据。

为什么必须固定 64 题、同一初始化、同步数
    任何一项不同，"哪条曲线更好"都可能只是数据/步数差异。脚本里三个 arm 共用
    同一份 items 列表、同一随机种子、同一评估函数与同一 ckpt 初始化。

用法（本脚本**不会**自动跑，只在你明确要求时执行）：
    python tools/overfit_probe.py --ckpt $B/ckpt/hotpot_router_10k_v5.pt \
        --data $B/synth/hotpot_train_10k.jsonl --n 64 --steps 200 --accum 4 \
        --arms qa_kl,recon_old,recon_qa \
        --base $B/models/Qwen3-4B-Instruct-2507 \
        --adapter $B/models/delta-mem_qwen3_4b_instruct \
        --save-json $B/results/overfit_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import torch

import latent_memory as LM
import train_latent_memory as T

ARMS = ("qa_kl", "recon_old", "recon_qa")


def _gold_read(comp, bridge, enc, model, tok, item, sysp, dev, pad, max_new=16,
               fence_style=None, inject_mode=None):
    """oracle_latent 自由生成：金标槽 -> compose -> 注入生成。返回 (pred, gold)。

    训练与评测**必须用同一套** isolate / fence_style / inject_mode：
    SDPA 下 isolate=False 与 isolate=True 的掩码与围栏分布完全不同，
    一边开一边关，测出来的 EM 就不是同一个方法。
    """
    units = T.item_units(item)
    gold_idx = T.evidence_slot_indices(units, item.get("evidence"))
    payloads, _z, _info, _kf, lex_all = T.encode_item_slots_full(
        enc, item, tag="probe-eval", cache=None)
    idx = [i for i in gold_idx if i < len(payloads)] or [0]
    reads = LM.pack_slot_reads(list(payloads))
    soft = LM.compose_routed_soft(
        model, tok, bridge, [reads[i] for i in idx], dev=dev,
        lex_list=([lex_all[i] for i in idx] if lex_all is not None else None),
        isolate=True, fence_style=fence_style)
    pred, _meta = LM._emit_latent(model, tok, soft, item["question"], sysp,
                                  max_new_tokens=int(max_new), pad_token_id=pad,
                                  inject_mode=inject_mode, fence_style=fence_style)
    gold = item["answer"]
    if isinstance(gold, (list, tuple)):
        gold = " ".join(str(x) for x in gold)
    return str(pred), str(gold)


def eval_oracle(comp, bridge, enc, model, tok, items, sysp, dev, pad, max_new=16,
                fence_style=None, inject_mode=None):
    """同一批题上的 oracle_latent EM（自由生成，与评测同一条构造）。"""
    hit = 0
    for item in items:
        T.reset_delta_state(model)
        pred, gold = _gold_read(comp, bridge, enc, model, tok, item, sysp, dev,
                                pad, max_new=max_new, fence_style=fence_style,
                                inject_mode=inject_mode)
        hit += int(T._norm_ans(pred) == T._norm_ans(gold))
    return hit / max(1, len(items))


def train_arm(arm, reader, model, tok, items, args, dev, pad):
    """在一个 arm 上训练 --steps 步，返回 (曲线, 最终 EM)。"""
    comp, bridge = reader.compressor, reader.bridge
    enc = reader.encoder
    for p in list(comp.parameters()) + list(bridge.parameters()):
        p.requires_grad_(True)
    comp.train()
    bridge.train()
    opt = torch.optim.AdamW(list(comp.parameters()) + list(bridge.parameters()),
                            lr=float(args.lr))
    rng = random.Random(int(args.seed) + 7)
    sysp = args.system_prompt
    curve = []
    micro = 0
    t0 = time.time()
    for step in range(int(args.steps)):
        item = items[step % len(items)]
        T.reset_delta_state(model)
        payloads, _z, _info, _kf, lex_all = T.encode_item_slots_full(
            enc, item, tag="probe", grad=True)
        units = T.item_units(item)
        gold_idx = T.evidence_slot_indices(units, item.get("evidence"))
        idx = [i for i in gold_idx if i < len(payloads)] or [0]
        reads = LM.pack_slot_reads(list(payloads))
        soft = LM.compose_routed_soft(
            model, tok, bridge, [reads[i] for i in idx], dev=dev,
            lex_list=([lex_all[i] for i in idx] if lex_all is not None else None),
            # 与正式 v5 读出对齐：训练侧也开 isolate（旧写法两边都是 False，
            # 测出来的曲线不能代表正式读出）
            isolate=True, fence_style=str(getattr(args, "fence_style", "") or None))
        embeds, labels, n_ans = T.build_student_inputs(
            model, tok, soft, sysp, item["question"], item["answer"])
        with LM.delta_write_disabled(model):
            s_logits = model(inputs_embeds=embeds,
                             attention_mask=torch.ones(embeds.shape[:2],
                                                       dtype=torch.long, device=dev),
                             use_cache=False).logits
        loss_qa = T.qa_loss(s_logits, labels)
        # KL：teacher 只给金标证据（与 student 应读到的一致，见 --teacher-scope）
        t_ev, _no_ev = T.teacher_evidence_text(item, units, "gold")
        t_embeds, t_n = T.build_teacher_inputs(
            model, tok, t_ev, sysp, item["question"], item["answer"])
        with LM.delta_write_disabled(model), torch.no_grad():
            t_logits = model(inputs_embeds=t_embeds,
                             attention_mask=torch.ones(t_embeds.shape[:2],
                                                       dtype=torch.long, device=dev),
                             use_cache=False).logits
        loss_kl = T.kl_loss(s_logits, t_logits, [min(n_ans, t_n)], tau=float(args.kl_tau))
        loss_r = None
        if arm != "qa_kl":
            loss_r = T.recon_loss_units(
                model, tok, bridge, payloads, units, rng, 2, sysp, dev,
                gold_idx=gold_idx, ent_weight=float(args.ent_weight),
                stop_weight=float(args.stop_weight), corrupt_p=float(args.recon_corrupt),
                lex_payloads=lex_all, use_lex=(arm == "recon_qa"))
        loss = T.assemble_loss(loss_qa, None, loss_kl, loss_r,
                               addr_weight=0.0, kl_weight=float(args.kl_weight),
                               recon_weight=(0.0 if arm == "qa_kl"
                                             else float(args.recon_weight)))
        (loss / max(1, int(args.accum))).backward()
        micro += 1
        if micro % max(1, int(args.accum)) == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in list(comp.parameters()) + list(bridge.parameters())
                 if p.requires_grad], float(args.grad_clip))
            opt.step()
            opt.zero_grad(set_to_none=True)
            if args.eval_every > 0 and (step + 1) % int(args.eval_every) == 0:
                em = eval_oracle(comp, bridge, enc, model, tok, items, sysp, dev, pad,
                                 fence_style=str(getattr(args, "fence_style", "") or None),
                                 inject_mode=str(getattr(args, "inject_mode", "") or None))
                curve.append({"step": step + 1, "em": em,
                              "loss_qa": float(loss_qa.detach()),
                              "loss_kl": float(loss_kl.detach()) if loss_kl is not None else None,
                              "loss_r": float(loss_r.detach()) if loss_r is not None else None})
                print(f"[{arm}] step={step + 1} trainEM={em:.4f} "
                      f"qa={float(loss_qa):.4f} "
                      f"kl={'n/a' if loss_kl is None else round(float(loss_kl), 4)} "
                      f"rec={'n/a' if loss_r is None else round(float(loss_r), 4)} "
                      f"α={float(bridge.alpha()):.3f} ({time.time() - t0:.0f}s)",
                      flush=True)
    em_final = eval_oracle(comp, bridge, enc, model, tok, items, sysp, dev, pad,
                           fence_style=str(getattr(args, "fence_style", "") or None),
                           inject_mode=str(getattr(args, "inject_mode", "") or None))
    curve.append({"step": int(args.steps), "em": em_final, "final": True})
    print(f"[{arm}] 训练集 oracle_latent EM = {em_final:.4f} "
          f"（{len(items)} 题，{args.steps} 步）", flush=True)
    return curve, em_final


def main(argv=None):
    ap = argparse.ArgumentParser(description="P1-4 64 题过拟合 + 目标函数对照")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=64, help="固定训练/评估用的题数")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--kl-tau", type=float, default=1.0)
    ap.add_argument("--recon-weight", type=float, default=0.5)
    ap.add_argument("--recon-corrupt", type=float, default=0.5)
    ap.add_argument("--ent-weight", type=float, default=5.0)
    ap.add_argument("--stop-weight", type=float, default=0.2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="qa_kl,recon_old,recon_qa")
    ap.add_argument("--fence-style", choices=("", "bracket", "verbose"), default="",
                    help="围栏样式（留空=用 latent_memory 的默认；训练与评测共用）")
    ap.add_argument("--inject-mode", choices=("", "user", "system", "between"),
                    default="", help="软 token 注入位置（训练与评测共用）")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--system-prompt",
                    default="You are a helpful assistant. Answer with the shortest exact value.",
                    help="必须与训练/评测用同一个 system prompt，否则读出的分布不一致")
    ap.add_argument("--base", default=T.BASE)
    ap.add_argument("--adapter", default=T.ADAPTER)
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--save-json", default="")
    a = ap.parse_args(argv)

    arms = [x.strip() for x in str(a.arms).split(",") if x.strip()]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        raise SystemExit(f"未知 arm: {bad}；可选 {ARMS}")

    _argv = ["--gpu", str(a.gpu), "--base", a.base, "--adapter", a.adapter]
    if a.no_adapter:
        _argv.append("--no-adapter")
    args = T.parse_args(_argv)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    T._setup_tf32(True)
    T.gpu_report(args)
    model, tok, dev, hidden = T._load_backbone(args, dev)
    T.freeze_model(model)
    pad = getattr(tok, "eos_token_id", None) or getattr(tok, "pad_token_id", None) or 0
    sysp = args.system_prompt

    data = T.load_synth_jsonl(a.data, limit=0, seed=0)
    items = [it for it in data if T.evidence_slot_indices(
        T.item_units(it), it.get("evidence"))][:int(a.n)]
    if len(items) < int(a.n):
        print(f"[warn] 只凑到 {len(items)} 条带证据的样本（要求 {a.n}）", flush=True)
    if not items:
        raise SystemExit(f"{a.data} 里没有带 evidence 的样本")

    results = {}
    for arm in arms:
        # 每个 arm 都从**同一份 ckpt** 重新加载，保证初始化逐位相同
        reader = LM.load_latent_reader(a.ckpt, model, tok, device=dev)
        print(f"\n===== arm={arm} =====", flush=True)
        # **Step 0 必须在同一批 64 题上测**：否则"64 题比验证集简单"这个偏差
        # 会被算进"训练带来的提升"里，三条曲线也就不可比。
        em0 = eval_oracle(reader.compressor, reader.bridge, reader.encoder,
                          model, tok, items, sysp, dev, pad,
                          fence_style=str(a.fence_style or None),
                          inject_mode=str(a.inject_mode or None))
        print(f"[{arm}] step=0 训练集 EM = {em0:.4f}（同一批 {len(items)} 题，"
              f"未训练）", flush=True)
        curve, em = train_arm(arm, reader, model, tok, items, a, dev, pad)
        curve.insert(0, {"step": 0, "em": em0, "untrained": True})
        results[arm] = {"curve": curve, "train_em": em, "train_em_step0": em0,
                        "gain": em - em0,
                        "alpha": float(reader.bridge.alpha()),
                        "n_items": len(items)}

    print("\n" + "=" * 70)
    print("| arm | step0 EM | 训练后 EM | 增益 | 说明 |")
    print("|---|---|---|---|---|")
    _note = {"qa_kl": "不加重建（干净对照）",
             "recon_old": "重建走 bridge(z_sem)（与 QA 目标不一致）",
             "recon_qa": "重建走 bridge(z_sem, z_lex)（与 QA 同路径）"}
    for arm in arms:
        r = results[arm]
        print(f"| {arm} | {r['train_em_step0']:.4f} | {r['train_em']:.4f} "
              f"| {r['gain']:+.4f} | {_note[arm]} |")
    if a.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(a.save_json)) or ".", exist_ok=True)
        with open(a.save_json, "w", encoding="utf-8") as f:
            json.dump({"ckpt": a.ckpt, "data": a.data, "n": len(items),
                       "steps": int(a.steps), "results": results}, f,
                      ensure_ascii=False, indent=1)
        print(f"[out] -> {a.save_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
