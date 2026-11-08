#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_fadkv.py — FAD-KV 的功能蒸馏训练（冻结 Qwen，只训 codec）。

损失：`L = L_QA + λ_kv·L_kv + λ_kl·L_KL + λ_attn·L_attn`

| 项 | 监督从哪来 | 为什么需要 |
|---|---|---|
| `L_kv` | 原生跨层 KV（逐组归一化 MSE） | 最密的信号：每个 token、每个组都有目标 |
| `L_QA` | 答案 token 的交叉熵（用压缩后 KV 前向） | 直接对准最终指标 |
| `L_KL` | 教师（**原生 KV**）在同一位置上答案分布 | 只有 CE 时答案对了但分布可能整体偏，KL 把整条曲线拉近 |
| `L_attn` | 教师/学生在被注入层上的注意力输出 MSE | KV 空间误差与"读出来是否一样"不是一回事；这一项直接对读出端 |

训练与评测必须走同一条前缀协议（默认 `full_prefix`）：评测侧实验 D 已量出
`full_prefix@L6` = 45.5 高于 `anchor@L6` = 40.5，本轮目标就是逼近前者，所以训练也用它。

生成侧不参与训练；训练用 teacher forcing 在答案 token 上算损失，位置与中间插入一致
（`T` 段的 position_ids 保留 `E` 占掉的空档）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import torch
import torch.nn.functional as F

import latent_memory as LM
import train_latent_memory as T
import midlayer_kv as MK
import kv_lowrank as LR
import fadkv_codec as FC
import eval_midlayer_kv as ED


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="FAD-KV 功能蒸馏训练")
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-data", default="")
    ap.add_argument("--n-train", type=int, default=0)
    ap.add_argument("--layers", default="6-35")
    ap.add_argument("--segments", type=int, default=3, help="层段数（阶段一/二都用）")
    ap.add_argument("--kv-split", default="",
                    help="K/V 预算比，如 0.3,0.7（空=按组均分）。实测 V 的重建误差是 K 的 1.6 倍")
    ap.add_argument("--latent-dim", type=int, default=1024,
                    help="阶段一总潜码（bf16 下 2 KiB/token）")
    ap.add_argument("--anchor-ratio", type=float, default=0.3)
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--anchor-dim", type=int, default=1536)
    ap.add_argument("--bg-dim", type=int, default=384)
    ap.add_argument("--encoding", default="full_prefix", choices=MK.MID_KV_ENCODINGS)
    ap.add_argument("--init", default="",
                    help="阶段一：分组 PCA ckpt（GroupedLowRankKV）；阶段二：阶段一 codec ckpt")
    ap.add_argument("--fit-tokens", type=int, default=20000,
                    help="没给 --init 时，现场拟合分组 PCA 用多少 token")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--losses", default="kv,qa,kl",
                    help="逗号分隔：kv,qa,kl,attn（attn 要额外捕获两层注意力，最贵）")
    ap.add_argument("--lambda-kv", type=float, default=1.0)
    ap.add_argument("--lambda-kl", type=float, default=0.5)
    ap.add_argument("--lambda-attn", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-limit", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=T.BASE)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--system-prompt",
                    default="You are a helpful assistant. Answer with the shortest exact value.")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def parse_kind_weights(spec: str) -> Optional[Dict[str, float]]:
    s = str(spec or "").strip()
    if not s:
        return None
    parts = [x for x in s.split(",") if x.strip()]
    if len(parts) != 2:
        raise SystemExit(f"--kv-split 要写成 'w_k,w_v'，得到 {spec!r}")
    return {"k": float(parts[0]), "v": float(parts[1])}


def parse_layers(spec: str) -> List[int]:
    s = str(spec).strip()
    if "-" in s:
        lo, hi = s.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------
# 前向：把答案 token 接在 T 后面，位置保留 E 的空档
# ---------------------------------------------------------------------
@contextmanager
def capture_attn(model, layers: Sequence[int]):
    """捕获被注入层注意力模块的输出（`out[0]`），用于 L_attn。"""
    store: Dict[int, torch.Tensor] = {}
    handles = []
    dec = MK.decoder_layers(model)

    def mk(l):
        def hk(mod, args, kw, out):
            store[int(l)] = (out[0] if isinstance(out, tuple) else out).detach()
        return hk

    for l in layers:
        handles.append(MK.attn_of(dec[int(l)]).register_forward_hook(mk(l), with_kwargs=True))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def answer_logits(model, split: Dict[str, Any], mem: MK.MemoryKV,
                  ans_ids: Sequence[int], capture_layers: Optional[Sequence[int]] = None
                  ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """返回 [n_ans, V] 的答案位置 logits（与中间插入、位置空档一致）。"""
    dev = LM._model_device(model)
    ns, ne, nt = int(split["ns"]), int(split["ne"]), int(split["nt"])
    ans = [int(i) for i in ans_ids]
    n_ans = len(ans)
    if n_ans <= 0:
        raise ValueError("答案 token 为空")
    ids = list(split["ids_S"]) + list(split["ids_T"]) + ans
    pos = torch.cat([torch.arange(0, ns, device=dev),
                     torch.arange(ns + ne, ns + ne + nt, device=dev),
                     torch.arange(ns + ne + nt, ns + ne + nt + n_ans, device=dev)])
    emb = model.get_input_embeddings()(torch.tensor([ids], dtype=torch.long, device=dev))
    cap: Dict[int, torch.Tensor] = {}
    with MK.inject_memory_kv(model, mem), LM.delta_write_disabled(model):
        if capture_layers:
            with capture_attn(model, capture_layers) as store:
                out = model(inputs_embeds=emb, use_cache=False, position_ids=pos.unsqueeze(0))
                cap = dict(store)
        else:
            out = model(inputs_embeds=emb, use_cache=False, position_ids=pos.unsqueeze(0))
    logits = out.logits[0]
    rows = logits[int(logits.shape[0]) - n_ans - 1: int(logits.shape[0]) - 1]
    if int(rows.shape[0]) != n_ans:
        raise RuntimeError(f"答案段对齐失败：{int(rows.shape[0])} != {n_ans}")
    return rows, cap


# ---------------------------------------------------------------------
# codec 编解码 -> MemoryKV
# ---------------------------------------------------------------------
def encode_to_memory(model, tokenizer, codec, split: Dict[str, Any], layers: List[int],
                     stage: int, encoding: str, token_flags: Optional[Sequence[bool]] = None
                     ) -> Tuple[MK.MemoryKV, Dict[str, Any]]:
    """证据 -> 原生 U -> 潜码 -> 还原 -> MemoryKV（阶段二还会缩短序列长度）。"""
    U, layout = MK.capture_uv(model, tokenizer, split, encoding, layers)
    ns = int(split["ns"])
    info: Dict[str, Any] = {"n_native": int(U.shape[0])}
    if int(stage) == 2:
        packed = codec.compress(U, token_flags)
        U_hat, pos = codec.decompress(packed)
        info.update({"n_anchor": int(packed["n_anchor"]), "n_slot": int(packed["n_slot"]),
                     "n_mem": int(U_hat.shape[0])})
        mem = MK.memory_kv_from_uv(model, U_hat.to(U.dtype),
                                   insert_at=ns + 0, layers=layers,
                                   positions=[ns + int(p) for p in pos])
    else:
        U_hat = codec.decode(codec.encode(U))
        info["n_mem"] = int(U_hat.shape[0])
        mem = MK.memory_kv_from_uv(model, U_hat.to(U.dtype), insert_at=ns, layers=layers)
    info["rel_err"] = float((U_hat.float() - U.float()).norm()
                            / U.float().norm().clamp_min(1e-6))
    return mem, info


def kv_penalty(codec, model, tokenizer, split, layers, stage, encoding) -> torch.Tensor:
    """L_kv：在原生 U 空间上逐组归一化 MSE（阶段二按 锚点/池化背景 分别算）。"""
    U, _ = MK.capture_uv(model, tokenizer, split, encoding, layers)
    if int(stage) == 1:
        return codec.kv_loss(U)
    packed = codec.compress(U)
    total = U.new_zeros(())
    n_terms = 0
    if int(packed["n_anchor"]) > 0:
        idx = torch.as_tensor(packed["idx_anchor"], device=U.device)
        total = total + codec.anchor_codec.kv_loss(U[idx])
        n_terms += 1
    if int(packed["n_slot"]) > 0:
        U_b = torch.stack([U[torch.as_tensor(c, device=U.device)].float().mean(dim=0)
                           for c in packed["chunks_bg"]]).to(U.dtype)
        total = total + codec.bg_codec.kv_loss(U_b)
        n_terms += 1
    return total / max(1, n_terms)


# ---------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    layers = parse_layers(args.layers)
    want = [x.strip() for x in str(args.losses).split(",") if x.strip()]
    bad = [x for x in want if x not in ("kv", "qa", "kl", "attn")]
    if bad:
        raise SystemExit(f"--losses 里有未知项 {bad}")
    dev = torch.device(f"cuda:{args.gpu}")
    T._setup_tf32(not args.no_tf32)
    ns = argparse.Namespace(smoke_model=False, layer=0, base=args.base, dtype=args.dtype,
                            gpu=args.gpu, no_adapter=True, adapter="",
                            grad_checkpointing=False)
    model, tokenizer, dev, hidden = T._load_backbone(ns, dev)
    T.freeze_model(model)
    layout = MK.uv_layout(model, layers)
    print(f"[model] hidden={hidden} uv_dim={layout['dim']} layers={layers[0]}..{layers[-1]} "
          f"stage={args.stage} losses={want} encoding={args.encoding}", flush=True)

    data = T.load_synth_jsonl(args.data, limit=int(args.n_train or 0), seed=int(args.seed))
    if not data:
        raise SystemExit(f"{args.data} 里没有样本")
    print(f"[data] train={len(data)}", flush=True)

    codec = build_codec(args, model, tokenizer, dev, layout, layers, data)
    n_param = sum(p.numel() for p in codec.parameters())
    print(f"[codec] {codec.describe()} 可训练 {n_param / 1e6:.2f}M", flush=True)
    if args.dry_run:
        print("[dry-run] codec 与数据可用，未训练", flush=True)
        return 0

    params = [p for p in codec.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(args.lr), betas=(0.9, 0.95))
    val = T.load_synth_jsonl(args.val_data, limit=int(args.eval_limit or 0), seed=0) \
        if args.val_data else []
    codec.train()
    step = micro = 0
    t0 = time.time()
    while step < int(args.steps):
        for item in data:
            losses = train_step(args, model, tokenizer, codec, item, layers, want, dev)
            loss = losses["total"] / max(1, int(args.accum))
            loss.backward()
            micro += 1
            if micro % int(args.accum) == 0:
                torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % int(args.log_every) == 0:
                    el = time.time() - t0
                    msg = " ".join(f"{k}={v:.4f}" for k, v in losses.items())
                    print(f"[step {step}] {msg} elapsed={el:.0f}s "
                          f"({el / max(1, step):.2f}s/step)", flush=True)
                if args.eval_every and step % int(args.eval_every) == 0 and val:
                    run_eval(args, model, tokenizer, codec, val, layers, step, dev)
                if step >= int(args.steps):
                    break
        if step >= int(args.steps):
            break

    FC.save_codec(args.out, codec,
                  config={"stage": int(args.stage), "layers": layers,
                          "segments": int(args.segments), "latent_dim": int(args.latent_dim),
                          "anchor_ratio": float(args.anchor_ratio), "pool": int(args.pool),
                          "encoding": args.encoding, "steps": int(args.steps),
                          "losses": want, "lr": float(args.lr)},
                  stats={"steps": step, "params": int(n_param)})
    print(f"[save] {args.out}", flush=True)
    return 0


def build_codec(args, model, tokenizer, dev, layout, layers, data):
    if int(args.stage) == 1:
        if args.init:
            gr = LR.GroupedLowRankKV.load(args.init, device=dev)
            print(f"[init] 分组 PCA <- {args.init} {gr.describe()}", flush=True)
            codec = FC.GroupedLinearCodec.from_grouped(gr)
        else:
            print(f"[init] 现场拟合分组 PCA（≤{args.fit_tokens} token）", flush=True)
            gr = LR.fit_grouped(_sub_batches(args, model, tokenizer, layers, data,
                                            int(args.fit_tokens)),
                                layout, latent_dim=int(args.latent_dim),
                                n_segments=int(args.segments), device=dev,
                                seed=int(args.seed),
                                kind_weights=parse_kind_weights(args.kv_split),
                                log=lambda m: print(m, flush=True))
            codec = FC.GroupedLinearCodec.from_grouped(gr)
        return codec.to(dev)

    # 阶段二：两套 codec，各自 PCA 初始化；锚点/背景的划分与训练时一致
    if not args.init:
        raise SystemExit("阶段二需要 --init 指向阶段一 codec（或先跑阶段一）")
    stage1, _cfg = FC.load_codec(args.init, device=dev)
    ratio, pool = float(args.anchor_ratio), int(args.pool)

    def anchor_batches():
        for U in _raw_uv(args, model, tokenizer, layers, data, 0):
            idx = torch.as_tensor(FC.select_anchors(U, ratio), device=U.device)
            yield U[idx]

    def bg_batches():
        for U in _raw_uv(args, model, tokenizer, layers, data, 0):
            idx = set(FC.select_anchors(U, ratio))
            rest = [i for i in range(int(U.shape[0])) if i not in idx]
            chunks = FC.pool_groups(rest, pool)
            if not chunks:
                continue
            yield torch.stack([U[torch.as_tensor(c, device=U.device)].float().mean(dim=0)
                               for c in chunks]).to(U.dtype)

    gr_a = LR.fit_grouped(anchor_batches, layout, latent_dim=int(args.anchor_dim),
                          n_segments=int(args.segments), device=dev, seed=int(args.seed),
                          log=lambda m: print("[anchor]" + m, flush=True))
    gr_b = LR.fit_grouped(bg_batches, layout, latent_dim=int(args.bg_dim),
                          n_segments=int(args.segments), device=dev,
                          seed=int(args.seed) + 1,
                          log=lambda m: print("[bg]" + m, flush=True))
    codec = FC.DualGranularityCodec(FC.GroupedLinearCodec.from_grouped(gr_a),
                                    FC.GroupedLinearCodec.from_grouped(gr_b),
                                    anchor_ratio=ratio, pool=pool)
    return codec.to(dev)


def _raw_uv(args, model, tokenizer, layers, data, budget):
    used = 0
    for item in data:
        ev = ED.evidence_text(item)
        if not ev.strip():
            continue
        try:
            split = MK.split_prompt_tokens(tokenizer, args.system_prompt, ev, item["question"])
        except MK.SplitError:
            continue
        U, _ = MK.capture_uv(model, tokenizer, split, args.encoding, layers)
        yield U
        used += int(U.shape[0])
        if budget > 0 and used >= budget:
            return


def _sub_batches(args, model, tokenizer, layers, data, budget):
    def gen():
        yield from _raw_uv(args, model, tokenizer, layers, data, budget)
    return gen


# ---------------------------------------------------------------------
def train_step(args, model, tokenizer, codec, item, layers, want, dev) -> Dict[str, float]:
    ev = ED.evidence_text(item)
    if not ev.strip():
        return {"total": torch.zeros((), device=dev)}
    split = MK.split_prompt_tokens(tokenizer, args.system_prompt, ev, item["question"])
    ans_ids = tokenizer(str(item["answer"]), return_tensors="pt",
                        add_special_tokens=False)["input_ids"][0].tolist()
    out: Dict[str, Any] = {"total": torch.zeros((), device=dev)}
    # 学生：压缩后的 KV
    mem_s, info = encode_to_memory(model, tokenizer, codec, split, layers,
                                   int(args.stage), args.encoding)
    cap_layers = layers if "attn" in want else None
    logit_s, cap_s = answer_logits(model, split, mem_s, ans_ids, cap_layers)
    y = torch.tensor(ans_ids, dtype=torch.long, device=dev)
    total = out["total"]
    if "qa" in want:
        l_qa = F.cross_entropy(logit_s, y)
        out["qa"] = float(l_qa)
        total = total + l_qa
    if "kl" in want or "attn" in want:
        mem_t = MK.build_evidence_kv(model, tokenizer, split, args.encoding,
                                     pos_start=int(split["ns"]), n_tokens=int(split["ne"]),
                                     layers=layers)
        with torch.no_grad():
            logit_t, cap_t = answer_logits(model, split, mem_t, ans_ids, cap_layers)
        if "kl" in want:
            l_kl = F.kl_div(F.log_softmax(logit_s, dim=-1),
                            F.softmax(logit_t.float(), dim=-1),
                            reduction="batchmean") * (1.0 ** 2)
            out["kl"] = float(l_kl)
            total = total + float(args.lambda_kl) * l_kl
        if "attn" in want and cap_s and cap_t:
            terms = []
            for l in cap_s:
                if l not in cap_t:
                    continue
                a, b = cap_s[l].float(), cap_t[l].float()
                terms.append(((a - b) ** 2).mean() / b.var().clamp_min(1e-6))
            if terms:
                l_at = torch.stack(terms).mean()
                out["attn"] = float(l_at)
                total = total + float(args.lambda_attn) * l_at
    if "kv" in want:
        l_kv = kv_penalty(codec, model, tokenizer, split, layers, int(args.stage),
                          args.encoding)
        out["kv"] = float(l_kv)
        total = total + float(args.lambda_kv) * l_kv
    out["total"] = total
    out["n_mem"] = float(info.get("n_mem", 0))
    return out


@torch.no_grad()
def run_eval(args, model, tokenizer, codec, val, layers, step, dev) -> None:
    codec.eval()
    hits = 0
    for item in list(val)[: int(args.eval_limit or len(val))]:
        ev = ED.evidence_text(item)
        if not ev.strip():
            continue
        try:
            split = MK.split_prompt_tokens(tokenizer, args.system_prompt, ev, item["question"])
        except MK.SplitError:
            continue
        mem, _ = encode_to_memory(model, tokenizer, codec, split, layers,
                                  int(args.stage), args.encoding)
        pred, _ = MK.midlayer_kv_generate(model, tokenizer, mem, split,
                                          max_new_tokens=int(args.max_new_tokens))
        hits += int(T._norm_ans(pred) == T._norm_ans(str(item["answer"])))
    n = len(val)
    print(f"[eval] step={step} n={n} EM={hits / max(1, n):.3f}", flush=True)
    codec.train()


if __name__ == "__main__":
    raise SystemExit(main())
