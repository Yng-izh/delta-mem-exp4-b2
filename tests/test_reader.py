#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tests/test_reader.py — 实验 A/B（完整记忆 + Query-conditioned Reader）的回归测试（全 CPU）。

每条都对应一个"改错了不会报错、只会让实验结论静默失真"的地方：

  R. Reader：输出形状必须与记忆长度一致（不压缩）；范数必须按**整向量 L2**对齐；
     必须真的被问题条件化（不同问题 -> 不同输出）；零初始化时输出方向等于原记忆。
  S. 打乱记忆：必须是**排列**（内容不变、顺序变），N<=1 时无操作。
  M. 记忆构建：完整记忆的长度等于 token 数（不压缩），槽记忆等于 K×块数。
  C. checkpoint：Reader（+压缩器）必须能存能读，结构参数全部落盘。

用法：python tests/test_reader.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import torch

import latent_memory as LM
import train_latent_memory as T
import reader_memory as RM

N_OK = 0
FAILS = []


def ok(name, cond, extra=""):
    global N_OK
    if cond:
        N_OK += 1
    else:
        FAILS.append(name)
        print(f"[FAIL] {name} {extra}")


def _raises(fn):
    try:
        fn()
    except KeyboardInterrupt:
        raise
    except BaseException:
        return True
    return False


# =====================================================================
# R. Reader
# =====================================================================
def test_reader_basic():
    torch.manual_seed(0)
    d, b, tn, N = 32, 8, 1.05, 7
    rd = RM.QueryCondReader(d, bottleneck=b, target_norm=tn)
    h = torch.randn(1, N, d)
    q = torch.randn(1, d)
    out = rd(h, None, q)
    ok("R1 输出形状与记忆长度一致（不压缩、不改变位置数）",
       tuple(out.shape) == (1, N, d), extra=str(tuple(out.shape)))
    ok("R2 每个向量的**整向量 L2 范数**对齐到 target_norm",
       torch.allclose(out.norm(dim=-1), torch.full((1, N), tn), atol=1e-4),
       extra=str([round(float(v), 4) for v in out.detach().norm(dim=-1)[0][:3]]))
    ok("R3 范数不是 RMSNorm×常数的量级（≈target·√d）",
       float(out.detach().norm(dim=-1).max()) < tn * 3.0,
       extra=str(float(out.detach().norm(dim=-1).max())))
    # 去掉了残差：输出是 MLP 直接投影出来的，**不再是恒等映射**。
    # 这一条是 2026-11-06 的修正：恒等（=原样隐藏状态）实测只有 10%，等于无记忆。
    cos = torch.nn.functional.cosine_similarity(out[0], h[0], dim=-1)
    ok("R4 输出不再是原记忆的恒等映射（残差已去掉）",
       float(cos.max()) < 0.99, extra=f"max_cos={float(cos.max()):.3f}")
    # 输入先过 LayerNorm：把记忆整体乘 100，输出应当几乎不变
    with torch.no_grad():
        o1 = rd(h, None, q)
        o2 = rd(h * 100.0, None, q)
    ok("R4b 输入尺度被 LayerNorm 吸收（乘 100 后输出基本不变）",
       float((o1 - o2).abs().max()) < 1e-3,
       extra=f"max_diff={float((o1 - o2).abs().max()):.2e}")


def test_reader_query_conditioned():
    torch.manual_seed(0)
    d, b, N = 32, 8, 6
    rd = RM.QueryCondReader(d, bottleneck=b, target_norm=1.05)
    # 打破零初始化，让问题条件化那条路有输出
    with torch.no_grad():
        rd.mlp2.weight.normal_(0, 0.05)
        rd.mlp2.bias.zero_()
    h = torch.randn(1, N, d)
    q1 = torch.randn(1, d)
    q2 = torch.randn(1, d)
    o1 = rd(h, None, q1)
    o2 = rd(h, None, q2)
    ok("R5 不同问题 -> 不同输出（Reader 真的在看问题）",
       float((o1 - o2).abs().max()) > 1e-4,
       extra=f"max_diff={float((o1 - o2).abs().max()):.2e}")
    o0 = rd(h, None, None)
    ok("R6 q_vec=None 时退化成不看问题的 Reader（消融路径可用）",
       tuple(o0.shape) == (1, N, d))
    # 同一问题、同一记忆 -> 逐位确定
    ok("R7 同一输入逐位确定（没有隐藏的随机性）",
       torch.allclose(rd(h, None, q1), o1, atol=0.0))


def test_reader_mask():
    torch.manual_seed(0)
    d, N = 16, 5
    rd = RM.QueryCondReader(d, bottleneck=4, target_norm=1.0)
    h = torch.randn(1, N, d)
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    out = rd(h, mask, torch.randn(1, d))
    ok("R8 padding 位置输出全零（不会把空位当成内容注入）",
       float(out[0, 3:].abs().sum()) == 0.0,
       extra=str(float(out[0, 3:].abs().sum())))


# =====================================================================
# S. 打乱记忆（实验 A 的关键对照）
# =====================================================================
def test_shuffle():
    torch.manual_seed(0)
    N, d = 9, 8
    mem = {"h": torch.randn(1, N, d), "mask": torch.ones(1, N, dtype=torch.long), "n": N}
    out = RM.shuffle_memory(mem, "tokens", rng=torch.Generator().manual_seed(1))
    ok("S1 打乱后长度不变", tuple(out["h"].shape) == (1, N, d))
    # 是排列：排序后的元素集合完全相同
    a = mem["h"][0].sum(dim=-1).sort().values
    b = out["h"][0].sum(dim=-1).sort().values
    ok("S2 打乱是**排列**（内容一个不多一个不少，只是顺序变了）",
       torch.allclose(a, b, atol=1e-5), extra=f"max={float((a - b).abs().max()):.2e}")
    ok("S3 打乱确实改变了顺序", not torch.allclose(mem["h"], out["h"]))
    ok("S4 同一 seed 结果可复现",
       torch.allclose(out["h"], RM.shuffle_memory(
           mem, "tokens", rng=torch.Generator().manual_seed(1))["h"]))
    one = {"h": torch.randn(1, 1, d), "mask": torch.ones(1, 1, dtype=torch.long), "n": 1}
    ok("S5 只有 1 个 token 时打乱是无操作",
       torch.allclose(RM.shuffle_memory(one, "tokens")["h"], one["h"]))
    ok("S6 未知的打乱方式直接报错（不能静默当成不打乱）",
       _raises(lambda: RM.shuffle_memory(mem, "banana")))


# =====================================================================
# K. 事实关键位（Masked Evidence Reconstruction 的监督掩码）
# =====================================================================
def test_key_spans():
    flags = RM.key_span_flags(["The", " city", " was", " founded", " in", " 1998",
                               " by", " Alice", ".", " He", " served", " as"])
    ok("K1 关系词被标为关键位（founded / served）",
       flags[3] and flags[10], extra=str(flags))
    ok("K2 数字被标为关键位（1998）", flags[5], extra=str(flags))
    ok("K3 实体（大写、非句首虚词）被标为关键位（Alice）", flags[7], extra=str(flags))
    ok("K4 句首虚词不算关键位（The / He 也大写但不是实体）",
       not flags[0] and not flags[9], extra=str(flags))
    ok("K5 功能词与标点不算关键位",
       not flags[1] and not flags[2] and not flags[4] and not flags[6] and not flags[8])
    ok("K6 空 token 不算关键位", RM.key_span_flags([""]) == [False])


# =====================================================================
# M. 记忆构建
# =====================================================================
def test_memory_build():
    model = T.SmokeModel(vocab_size=256, hidden_size=32, n_layers=3, n_heads=2)
    tok = T.StubTok(vocab_size=256)
    item = {"units": ["alpha beta gamma", "delta epsilon zeta eta"],
            "evidence": ["alpha beta gamma", "delta epsilon zeta eta"],
            "question": "which?", "answer": "gamma"}
    mem = RM.build_full_memory(model, tok, item, [0, 1], layer=1, max_tokens=100)
    n_tok = sum(len(tok._ids(T.unit_encode_text(item, i))) for i in (0, 1))
    ok("M1 完整记忆的 token 数 == 原文 token 数（**不压缩**）",
       int(mem["n"]) == int(n_tok), extra=f"{mem['n']} vs {n_tok}")
    ok("M2 完整记忆的隐藏维 == 模型 hidden",
       int(mem["h"].shape[-1]) == 32, extra=str(tuple(mem["h"].shape)))
    capped = RM.build_full_memory(model, tok, item, [0, 1], layer=1, max_tokens=3)
    ok("M3 max_tokens 生效（防显存爆，不改变方法定义）", int(capped["n"]) == 3)
    empty = RM.build_full_memory(model, tok, item, [], layer=1)
    ok("M4 没有金标证据时返回空记忆而不是编一段", int(empty["n"]) == 0)

    comp = LM.LatentCompressor(32, n_slots=4, inner_dim=16, num_heads=2, lex_skip=False)
    sm = RM.build_slot_memory(model, tok, comp, item, [0, 1], max_tokens=100, layer=1)
    ok("M5 槽记忆的长度 == K × 块数（K=4 的整数倍）",
       int(sm["n"]) > 0 and int(sm["n"]) % 4 == 0, extra=str(sm["n"]))
    # 实验 B 要训压缩器：build_slot_memory **不能**带 no_grad，否则梯度被掐断
    comp.zero_grad(set_to_none=True)
    sm2 = RM.build_slot_memory(model, tok, comp, item, [0, 1], max_tokens=100, layer=1)
    sm2["h"].sum().backward()
    ok("M6 槽记忆对压缩器保留梯度（不能包 @torch.no_grad，否则实验 B 训不动）",
       comp.queries.grad is not None and float(comp.queries.grad.abs().sum()) > 0.0,
       extra="None" if comp.queries.grad is None else
       f"{float(comp.queries.grad.abs().sum()):.2e}")


# =====================================================================
# C. checkpoint
# =====================================================================
def test_ckpt_roundtrip():
    import tempfile
    torch.manual_seed(0)
    rd = RM.QueryCondReader(32, bottleneck=8, target_norm=1.05)
    comp = LM.LatentCompressor(32, n_slots=4, inner_dim=16, num_heads=2, lex_skip=False)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "r.pt")
        RM.save_reader_checkpoint(path, rd, comp,
                                  config=RM.reader_config(rd, comp, {"mem": "slots4"}))
        r2, c2, cfg = RM.load_reader_checkpoint(path, device="cpu")
        ok("C1 Reader 权重逐位一致",
           torch.equal(r2.mlp1.weight, rd.mlp1.weight))
        ok("C2 压缩器权重逐位一致",
           torch.equal(c2.queries, comp.queries))
        ok("C3 压缩器的构造参数全部落盘（否则 strict load 会形状不符）",
           int(cfg["compressor_inner_dim"]) == 16
           and int(cfg["compressor_num_heads"]) == 2
           and int(cfg["compressor_n_slots"]) == 4, extra=str(sorted(cfg))[:80])
        ok("C4 mem 口径随 ckpt 还原", str(cfg.get("mem")) == "slots4")


def main():
    for fn in (test_reader_basic, test_reader_query_conditioned, test_reader_mask,
               test_shuffle, test_key_spans, test_memory_build, test_ckpt_roundtrip):
        print(f"--- {fn.__name__}")
        fn()
    print("=" * 70)
    if FAILS:
        print(f"FAILED {len(FAILS)} / {N_OK + len(FAILS)}：{FAILS}")
        return 1
    print(f"ALL PASS ({N_OK} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
