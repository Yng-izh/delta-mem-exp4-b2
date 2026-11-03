#!/usr/bin/env python3
"""阶段一：**只训 Router**（不跑生成、不跑重建、不算 KL）。

为什么单独拎出来
----------------
`HybridLatentRouter` 的输入特征 kfeat（冻结主干的 h0/h18）完全静态，不随
Compressor/Bridge 更新而改变。把它绑在"生成 + 重建"的主循环里训练，等于用十几个
小时去等一个本该几十分钟收敛的东西；更糟的是 router 还没收敛时采出的难负例会把
Bridge 一起带偏（v4 就是这样把读出从 17.2% 带到 10.0% 的）。

为什么必须批量编码
------------------
本脚本唯一昂贵的一步是"过冻结 4B 取 h0/h18"。逐条编码时 3 万次前向要 4 小时，
其中绝大部分时间花在 kernel 启动和同步上（单条才 ~700 token）。把 16 条样本的
单元拼成一次前向（`encode_full` 本来就按 parent_id 分组），吞吐能到原来的 6~8 倍。

用法：
    python tools/train_router_stage1.py --base ... --adapter ... \
        --data $B/synth/hotpot_train_10k.jsonl --val-data $B/synth/hotpot_val500.jsonl \
        --epochs 3 --out $B/ckpt/router_stage1.pt
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import latent_memory as LM          # noqa: E402
import train_latent_memory as T     # noqa: E402


def encode_batch(encoder, items, tag="fact"):
    """一次前向编码**一批**样本的全部单元，返回逐样本的 (payloads, kfeat, lex)。

    关键点：parent_ids 用 f"{tag}:{bi}:{k}" 把"第几条样本 / 第几个单元"编进去，
    encode_full 会按 parent 分组并保持首次出现顺序，于是返回的 payloads/kfeat
    与传入顺序一一对应，可以直接按下标切片。
    """
    texts, pids, bounds = [], [], []
    for bi, item in enumerate(items):
        units = T.item_units(item)
        bounds.append(len(units))
        for k in range(len(units)):
            texts.append(T.unit_encode_text(item, k))
            pids.append(f"{tag}:{bi}:{k}")
    if not texts:
        return [{"payloads": [], "kfeat": None, "lex": None} for _ in items]
    payloads, lex, kfeat, _info = encoder.encode_full(texts, parent_ids=pids, grad=False)
    out, off = [], 0
    for n in bounds:
        kf = None
        if kfeat is not None:
            kf = {kk: (vv[off:off + n] if torch.is_tensor(vv) else vv)
                  for kk, vv in kfeat.items()}
        out.append({"payloads": list(payloads[off:off + n]),
                    "kfeat": kf,
                    "lex": (list(lex[off:off + n]) if lex is not None else None)})
        off += n
    return out


def iter_batches(seq, bs):
    for i in range(0, len(seq), bs):
        yield seq[i:i + bs]


def build_router(args, hidden, dev):
    cls = LM.HybridLatentRouter if args.router_kind == "hybrid" else LM.LatentRouter
    kw = {"gamma": float(args.addr_gamma)} if cls is LM.HybridLatentRouter else {}
    return cls(hidden, n_slots=args.m, d_key=args.d_key, **kw).to(dev)


def recall_eval(router, encoder, items, args, dev, limit=0, bs=16):
    """不生成答案，只算寻址指标：首跳命中 / 完整证据召回（整条链都进 picks）。"""
    router.eval()
    use = items[:limit] if limit else items
    n = hit1 = full = 0
    t0 = time.time()
    for batch in iter_batches(use, bs):
        enc = encode_batch(encoder, batch, tag="val")
        for item, e in zip(batch, enc):
            units = T.item_units(item)
            gold = T.evidence_chain_indices(units, item.get("evidence"))
            if not gold or not e["payloads"]:
                continue
            slot_z = LM.pool_slot_keys(list(e["payloads"])).unsqueeze(0)
            reads = LM.pack_slot_reads(list(e["payloads"]))
            qf_ex = T.query_feature_ex_cached(encoder.model, encoder.tokenizer,
                                              item["question"], args.system_prompt,
                                              args.layer, dev)
            with torch.no_grad():
                out = router.forward_hops(qf_ex["pooled"], slot_z, n_hops=2, hard=True,
                                          reads=reads, kfeat=e["kfeat"],
                                          q_tok=qf_ex.get("h0_tok"),
                                          topk_per_hop=int(args.topk_per_hop))
            picks = set(int(p) for p in (out.get("picks") or []))
            n += 1
            hit1 += int(bool(picks & set(int(g) for g in gold)))
            full += int(set(int(g) for g in gold) <= picks)
    router.train()
    dt = time.time() - t0
    if not n:
        return {"n": 0, "hit1": None, "full": None, "sec": dt}
    return {"n": n, "hit1": hit1 / n, "full": full / n, "sec": dt}


def save(router, args, hidden, step, path):
    torch.save({"router": router.state_dict(),
                "config": {"router_kind": args.router_kind, "n_slots": args.m,
                           "d_key": args.d_key, "hidden_size": hidden,
                           "addr_gamma": args.addr_gamma},
                "step": step}, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-data", default="")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--val-limit", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--enc-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--inner-dim", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--bottleneck", type=int, default=512)
    ap.add_argument("--max-unit-tokens", type=int, default=32)
    ap.add_argument("--max-slots", type=int, default=0)
    ap.add_argument("--d-key", type=int, default=LM.LATENT_D_KEY)
    ap.add_argument("--router-kind", default="hybrid", choices=("latent", "hybrid"))
    ap.add_argument("--addr-gamma", type=float, default=LM.LATENT_ADDR_GAMMA)
    ap.add_argument("--topk-per-hop", type=int, default=2)
    ap.add_argument("--system-prompt",
                    default="You are a helpful assistant. Answer with the shortest exact value.")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke-model", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--no-adapter", action="store_true")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.smoke_model = bool(args.smoke_model)
    model, tok, dev, hidden = T._load_backbone(args, dev)
    comp = LM.LatentCompressor(hidden, n_slots=args.m, inner_dim=args.inner_dim,
                               num_heads=args.heads, lex_skip=False).to(dev).eval()
    comp.requires_grad_(False)
    encoder = LM.LatentEncoder(model, tok, comp, layer=args.layer,
                               max_unit_tokens=args.max_unit_tokens, device=dev)
    router = build_router(args, hidden, dev)
    opt = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=0.0)
    items = T.load_synth_jsonl(args.data, args.n_train, seed=args.seed)
    val = T.load_synth_jsonl(args.val_data, 0, seed=args.seed) if args.val_data else []
    print(f"[stage1] train={len(items)} val={len(val)} m={args.m} d_key={args.d_key} "
          f"kind={args.router_kind} lr={args.lr} accum={args.accum} "
          f"enc_batch={args.enc_batch}", flush=True)

    rng = random.Random(args.seed)
    step = 0
    t_start = time.time()
    # 问题特征缓存：q_feat/q_tok 只取决于问题文本，跨 epoch 完全不变。
    # 实测逐条编码时瓶颈根本不在 encode，而在**每条样本一次的问题前向**
    # （1.83 it/s 里约 87% 是它）；缓存后第 2、3 轮几乎只剩 encode 的开销。
    qf_cache: dict = {}
    for ep in range(args.epochs):
        for _start in range(0, len(items), args.enc_batch):
            batch = items[_start:_start + args.enc_batch]
            enc = encode_batch(encoder, batch, tag="fact")
            for _j, (item, e) in enumerate(zip(batch, enc)):
                units = T.item_units(item)
                chain = T.evidence_chain_indices(units, item.get("evidence"))
                if not chain or not e["payloads"]:
                    continue
                tgt = T.target_slot_index(units, item["answer"],
                                          evidence=item.get("evidence"))
                slot_z, _reads, tgt_slot, keep = T.sample_slot_subset(
                    e["payloads"], (chain[0] if chain else tgt), rng, args.max_slots or None,
                    must_keep=T.evidence_slot_indices(units, item.get("evidence")))
                keep_pos = {int(o): j for j, o in enumerate(keep)}
                chain_new = [keep_pos[i] for i in chain if i in keep_pos]
                n_hops = 2 if (T.item_hops(item) > 1 and len(chain_new) >= 2) else 1
                kfeat = e["kfeat"]
                if kfeat is not None:
                    kfeat = {k: (v[keep] if torch.is_tensor(v) else v)
                             for k, v in kfeat.items()}
                _qid = _start + _j
                qf_ex = qf_cache.get(_qid)
                if qf_ex is None:
                    qf_ex = T.query_feature_ex_cached(model, tok, item["question"],
                                                      args.system_prompt, args.layer, dev)
                    qf_cache[_qid] = qf_ex
                loss, _i2 = router.addr_loss_set(
                    qf_ex["pooled"], slot_z,
                    (chain_new[:2] if n_hops > 1 else [int(tgt_slot)]),
                    n_hops=n_hops, kfeat=kfeat, q_tok=qf_ex.get("h0_tok"))
                if loss is None:
                    continue
                (loss / args.accum).backward()
                step += 1
                if step % args.accum == 0:
                    torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                if step % 100 == 0:
                    el = max(time.time() - t_start, 1e-6)
                    print(f"[step {step}] loss_addr={float(loss.detach()):.4f} ep={ep} "
                          f"{step / el:.2f} it/s", flush=True)
                if val and args.eval_every > 0 and step % args.eval_every == 0:
                    r = recall_eval(router, encoder, val, args, dev,
                                    limit=args.val_limit, bs=args.enc_batch)
                    print(f"[eval step={step}] hit1={r['hit1']} full={r['full']} "
                          f"n={r['n']} ({r['sec']:.0f}s)", flush=True)
                    save(router, args, hidden, step, args.out)
    save(router, args, hidden, step, args.out)
    if val:
        r = recall_eval(router, encoder, val, args, dev, limit=args.val_limit,
                        bs=args.enc_batch)
        print(f"[final] hit1={r['hit1']} full={r['full']} n={r['n']} "
              f"({r['sec']:.0f}s)", flush=True)
    print(f"[stage1] done step={step} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
