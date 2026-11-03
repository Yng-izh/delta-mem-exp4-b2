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
    """寻址诊断（不生成答案）。三个指标必须分开报，否则无法定位问题在哪一跳：

      hop1      : **第 1 跳** top-b 是否命中任一金标（旧口径把两跳混在一起算，93% 是假的）
      hop2_given: 第 1 跳命中时，整条证据链是否被补全（衡量第二跳的补全能力）
      full      : 全部金标是否都进了 picks（受读出容量限制：picks 上限 = 跳数 × b）

    并按"金标证据数 = 2 / >= 3"分组：>=3 条的题需要更多 picks，混在一起会把
    "读出容量不足"误判成"第二跳选错"。
    """
    router.eval()
    use = items[:limit] if limit else items
    st = {"n": 0, "hop1": 0, "hop2_given": 0, "hop2_n": 0, "full": 0,
          "g2": {"n": 0, "hop1": 0, "full": 0}, "g3": {"n": 0, "hop1": 0, "full": 0}}
    t0 = time.time()
    for batch in iter_batches(use, bs):
        enc = encode_batch(encoder, batch, tag="val")
        for item, e in zip(batch, enc):
            units = T.item_units(item)
            gold = set(int(g) for g in T.evidence_chain_indices(units, item.get("evidence")))
            if not gold or not e["payloads"]:
                continue
            slot_z = LM.pool_slot_keys(list(e["payloads"])).unsqueeze(0)
            reads = LM.pack_slot_reads(list(e["payloads"]))
            qf_ex = T.query_feature_ex_cached(encoder.model, encoder.tokenizer,
                                              item["question"], args.system_prompt,
                                              args.layer, dev)
            with torch.no_grad():
                o1 = router.forward_hops(qf_ex["pooled"], slot_z, n_hops=1, hard=True,
                                         reads=reads, kfeat=e["kfeat"],
                                         q_tok=qf_ex.get("h0_tok"),
                                         topk_per_hop=int(args.topk_per_hop))
                of = router.forward_hops(qf_ex["pooled"], slot_z, n_hops=2, hard=True,
                                         reads=reads, kfeat=e["kfeat"],
                                         q_tok=qf_ex.get("h0_tok"),
                                         topk_per_hop=int(args.topk_per_hop))
            p1 = set(int(p) for p in (o1.get("picks") or []))
            pf = set(int(p) for p in (of.get("picks") or []))
            hit1 = bool(p1 & gold)
            full = gold <= pf
            st["n"] += 1
            st["hop1"] += int(hit1)
            st["full"] += int(full)
            if hit1:
                st["hop2_n"] += 1
                st["hop2_given"] += int(full)
            g = st["g2"] if len(gold) == 2 else st["g3"]
            g["n"] += 1
            g["hop1"] += int(hit1)
            g["full"] += int(full)
    router.train()
    dt = time.time() - t0
    n = max(1, st["n"])
    out = {"n": st["n"], "sec": dt,
           "hop1": st["hop1"] / n,
           "full": st["full"] / n,
           "hop2_given": (st["hop2_given"] / st["hop2_n"]) if st["hop2_n"] else None}
    for key in ("g2", "g3"):
        d = st[key]
        out[key] = {"n": d["n"],
                    "hop1": (d["hop1"] / d["n"]) if d["n"] else None,
                    "full": (d["full"] / d["n"]) if d["n"] else None}
    return out


def save(router, opt, args, hidden, step, epoch, epoch_pos, rng, path):
    """保存**可续训**的完整状态。

    只存 router 权重是不够的：AdamW 的一阶/二阶动量、全局步数、epoch 内位置、
    以及 python/torch 两套 RNG 状态缺一不可，否则"接着训"等于重新开始一条不同的轨迹。
    写入走 tmp + os.replace：训练中途被杀也不会留下半截 ckpt。
    """
    blob = {
        "format": 2,
        "router": router.state_dict(),
        "optimizer": (opt.state_dict() if opt is not None else None),
        "step": int(step),
        "epoch": int(epoch),
        "epoch_pos": int(epoch_pos),
        "rng": {"torch": torch.get_rng_state(),
                "py": (rng.getstate() if rng is not None else None)},
        "config": {"router_kind": args.router_kind, "n_slots": args.m,
                   "d_key": args.d_key, "hidden_size": hidden,
                   "addr_gamma": args.addr_gamma, "lr": args.lr,
                   "accum": args.accum, "enc_batch": args.enc_batch,
                   "topk_per_hop": args.topk_per_hop, "seed": args.seed},
    }
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)
    return path


def load_state(path, router, opt, rng):
    """从 ckpt 恢复 router/优化器/步数/位置/RNG。返回 (step, epoch, epoch_pos)。"""
    blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict) or "router" not in blob:
        raise ValueError(f"{path} 不是本脚本产出的 ckpt（缺 router 字段）")
    router.load_state_dict(blob["router"])
    if opt is not None and blob.get("optimizer") is not None:
        opt.load_state_dict(blob["optimizer"])
    st = blob.get("rng") or {}
    if st.get("torch") is not None:
        torch.set_rng_state(st["torch"])
    if rng is not None and st.get("py") is not None:
        rng.setstate(st["py"])
    return int(blob.get("step", 0)), int(blob.get("epoch", 0)), int(blob.get("epoch_pos", 0))


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
    ap.add_argument("--save-every", type=int, default=500,
                    help="每多少步存一次可续训 ckpt（0=只在评测点/结束时存）")
    ap.add_argument("--resume", default="",
                    help="从 ckpt 接着训（恢复权重/优化器动量/步数/epoch 位置/RNG）")
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
    step, start_ep, start_pos = 0, 0, 0
    if str(args.resume or ""):
        step, start_ep, start_pos = load_state(args.resume, router, opt, rng)
        print(f"[resume] {args.resume} -> step={step} epoch={start_ep} "
              f"epoch_pos={start_pos}", flush=True)
    t_start = time.time()
    # 问题特征缓存：q_feat/q_tok 只取决于问题文本，跨 epoch 完全不变。
    # 实测逐条编码时瓶颈根本不在 encode，而在**每条样本一次的问题前向**
    # （1.83 it/s 里约 87% 是它）；缓存后第 2、3 轮几乎只剩 encode 的开销。
    qf_cache: dict = {}
    for ep in range(start_ep, args.epochs):
        for _start in range(0, len(items), args.enc_batch):
            batch = items[_start:_start + args.enc_batch]
            if ep == start_ep and (_start + len(batch)) <= start_pos:
                continue                     # 整批已完成：连编码都不做
            enc = encode_batch(encoder, batch, tag="fact")
            for _j, (item, e) in enumerate(zip(batch, enc)):
                if ep == start_ep and (_start + _j) < start_pos:
                    continue
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
                # 监督目标必须覆盖**整条证据链**，上限是读出容量（跳数 × 每跳 top-b）。
                # 旧写法只取前 2 条，而评测要求整条链都进 picks —— 161/500 条
                # （金标 >=3 句）的题从训练目标上就没被要求过，指标自然上不去。
                _cap = max(1, int(n_hops) * int(args.topk_per_hop))
                kfeat = e["kfeat"]
                if kfeat is not None:
                    # 关键性能点：encode_full 把特征留在 CPU（省显存），而
                    # addr_loss_set 对**每个金标分支**都会调一次 scores()，每次都要把
                    # [K,cap,d]（约 7.5 MB）从 CPU 搬到 GPU。4 个金标 -> 12 个分支 ->
                    # 每条样本近 90 MB 的 H2D 拷贝，这才是 1.9 it/s 的真正瓶颈。
                    # 这里只搬一次，后面全部复用。
                    kfeat = {k: (v[keep].to(dev, non_blocking=True)
                                 if torch.is_tensor(v) else v)
                             for k, v in kfeat.items()}
                _qid = _start + _j
                qf_ex = qf_cache.get(_qid)
                if qf_ex is None:
                    qf_ex = T.query_feature_ex_cached(model, tok, item["question"],
                                                      args.system_prompt, args.layer, dev)
                    qf_cache[_qid] = qf_ex
                loss, _i2 = router.addr_loss_set(
                    qf_ex["pooled"], slot_z,
                    (chain_new[:_cap] if n_hops > 1 else [int(tgt_slot)]),
                    n_hops=n_hops, kfeat=kfeat, q_tok=qf_ex.get("h0_tok"))
                if loss is None:
                    continue
                (loss / args.accum).backward()
                step += 1
                if step % args.accum == 0:
                    torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                if args.save_every > 0 and step % args.save_every == 0:
                    save(router, opt, args, hidden, step, ep, _start + _j, rng, args.out)
                    print(f"[save] step={step} -> {args.out}", flush=True)
                if step % 100 == 0:
                    el = max(time.time() - t_start, 1e-6)
                    print(f"[step {step}] loss_addr={float(loss.detach()):.4f} ep={ep} "
                          f"{step / el:.2f} it/s", flush=True)
                if val and args.eval_every > 0 and step % args.eval_every == 0:
                    r = recall_eval(router, encoder, val, args, dev,
                                    limit=args.val_limit, bs=args.enc_batch)
                    print(f"[eval step={step}] hop1={r['hop1']:.3f} "
                          f"hop2|hop1={r['hop2_given']} full={r['full']:.3f} "
                          f"n={r['n']} | ev=2: n={r['g2']['n']} hop1={r['g2']['hop1']} "
                          f"full={r['g2']['full']} | ev>=3: n={r['g3']['n']} "
                          f"hop1={r['g3']['hop1']} full={r['g3']['full']} "
                          f"({r['sec']:.0f}s)", flush=True)
                    save(router, opt, args, hidden, step, ep, _start + _j, rng, args.out)
    save(router, opt, args, hidden, step, max(0, args.epochs - 1), len(items), rng, args.out)
    if val:
        r = recall_eval(router, encoder, val, args, dev, limit=args.val_limit,
                        bs=args.enc_batch)
        print(f"[final] hop1={r['hop1']:.3f} hop2|hop1={r['hop2_given']} "
              f"full={r['full']:.3f} n={r['n']} | ev=2: n={r['g2']['n']} "
              f"hop1={r['g2']['hop1']} full={r['g2']['full']} | ev>=3: n={r['g3']['n']} "
              f"hop1={r['g3']['hop1']} full={r['g3']['full']}", flush=True)
    if torch.cuda.is_available():
        # 显存峰值必须打出来：调大 enc_batch 之前得知道余量有多少，而不是靠猜
        print(f"[mem] peak_alloc={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
              f"peak_reserved={torch.cuda.max_memory_reserved() / 2**30:.2f} GiB "
              f"enc_batch={args.enc_batch}", flush=True)
    print(f"[stage1] done step={step} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
