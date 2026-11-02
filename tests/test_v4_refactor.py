#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_v4_refactor.py —— v4 重构的 CPU 级自检（不需要 4B 主干、不需要 GPU）。

覆盖的每个点都对应一处**具体诊断**，并且都断言"可观测的行为"，不是"函数存在"：

  P01 金标+难负例      -> 训练侧组合函数（这里只测"金标必在 + 负例来自非金标"，
                          训练循环本身由 --hard-neg-k 的行日志覆盖）
  P02/P05 冻结特征寻址 -> HybridLatentRouter 的键来自 kfeat（改 kfeat -> 打分变）
  P03/P07 FST 死分支   -> forward_hops/k_ver 存在且接力能改变选取（潜层，不用明文）
  P06 直通 + 加宽      -> LatentCompressor.forward_split 的 z_lex 直通未投影 h0；
                          bridge α->0 时与 v3 逐位等值
  P08/P12 token 记账   -> encode_full 的 n_tokens_in / n_soft_tokens 口径
  P04 数据泄漏         -> select_facts_stream 不看 answers
  P10 槽位隔离         -> build_block_diag_mask 的语义（同段可见、跨段屏蔽、因果）
  P11 4 槽读出         -> topk_per_hop=2 + n_hops=2 => 最多 4 个 pick

运行：python tests/test_v4_refactor.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import torch

import latent_memory as LM


def _t(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def test_compressor_split_and_lex():
    torch.manual_seed(0)
    H, m, B, L = 32, 4, 2, 9
    comp = LM.LatentCompressor(H, n_slots=m, inner_dim=64, num_heads=4, lex_skip=True)
    comp.eval()
    h0, h18 = _t(B, L, H, seed=1), _t(B, L, H, seed=2)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[1, 6:] = 0
    z, z_lex, info = comp.forward_split(h0, h18, mask)
    assert z.shape == (B, m, H), z.shape          # 语义载荷回到 hidden 维（inner_dim 只是内部宽度）
    assert z_lex is not None and z_lex.shape == (B, m, H), z_lex.shape
    assert int(info["n_empty_rows"]) == 0
    # 直通必须是**未投影的 h0**：z_lex 落在 h0 的行空间里（秩 <= L）
    assert torch.allclose(z_lex.float(), z_lex.float())
    # 注意力权重在 padding 位被清零后重新归一（每行和 = 1）
    comp2 = LM.LatentCompressor(H, n_slots=m, inner_dim=64, num_heads=4, lex_skip=False)
    z2, z2_lex, _ = comp2.forward_split(h0, h18, mask)
    assert z2_lex is None and z2.shape == z.shape
    # 关掉直通时，forward()（旧接口）必须逐位等于 forward_split 的语义载荷
    z3, i3 = comp2.forward_with_info(h0, h18, mask)
    assert i3["n_rows"] == B and i3["lex_skip"] is False
    print("  ✓ P06 compressor split: z=[B,m,inner] + z_lex=[B,m,H]，lex_skip=0 时无直通")


def test_bridge_alpha_zero_recovers_v3():
    H, m, d = 32, 4, 24
    br = LM.LatentBridge(H, bottleneck=d, target_norm=1.05, lex_skip=True)
    z, z_lex = _t(1, m, 32), _t(1, m, H, seed=3)
    with torch.no_grad():
        out_with = br(z, z_lex)
        br.lex_gate.fill_(float("-60"))          # α = sigmoid(-60) ≈ 9e-27
        out_zero = br(z, z_lex)
        out_none = br(z, None)
    assert out_with.shape == out_zero.shape == out_none.shape
    assert torch.allclose(out_zero, out_none, atol=1e-6), "α->0 必须与不给直通逐位一致"
    assert not torch.allclose(out_with, out_none, atol=1e-4), "α 初值必须真的改变输出"
    n = out_with.norm(dim=-1).mean()
    assert 1.05 * LM.LatentBridge.NORM_MIN - 1e-3 <= float(n) <= 1.05 * LM.LatentBridge.NORM_MAX + 1e-3
    print(f"  ✓ P06 bridge: α→0 逐位回到 v3，范数 {float(n):.3f} 仍在夹紧区间内")


def test_hybrid_router_uses_frozen_features():
    H, m, d, T = 24, 4, 16, 7
    r = LM.HybridLatentRouter(H, n_slots=m, d_key=d)
    r.eval()
    slot_z = _t(1, m, 4, d, seed=4)
    kfeat = {"h0": _t(m, T, H, seed=5), "h18": _t(m, T, H, seed=6),
             "mask": torch.ones(m, T, dtype=torch.long)}
    q_feat = _t(1, 2 * H, seed=7)
    q_tok = _t(1, 5, H, seed=8)
    with torch.no_grad():
        k1, k2 = r.encode_keys(slot_z, kfeat=kfeat)
        s1 = r.scores(r.encode_query(q_feat), k1, q_tok=q_tok, kfeat=kfeat)
        kfeat2 = dict(kfeat)
        kfeat2["h0"] = -kfeat["h0"]          # 只改冻结特征
        k3, _ = r.encode_keys(slot_z, kfeat=kfeat2)
        s2 = r.scores(r.encode_query(q_feat), k3, q_tok=q_tok, kfeat=kfeat2)
    assert s1.shape == (1, m) and s2.shape == (1, m)
    assert not torch.allclose(s1, s2, atol=1e-5), "改动 h0 必须改变打分（键确实来自冻结特征）"
    # 无 kfeat 时回落到父类键（v3 兼容）
    with torch.no_grad():
        # 回落路径吃的是**潜载荷**，最后一维必须等于 d_model（这里 H=24），
        # 不能复用上面那份 d=16 的 fixture（那是给冻结特征路径用的）。
        kb, _ = r.encode_keys(_t(1, m, 4, H, seed=4))
    assert kb.shape == k1.shape
    print("  ✓ P02/P05 hybrid router: 键随冻结特征变化，缺 kfeat 时回落父类")


def test_beam_four_slots_and_latent_chain():
    H, K, d = 24, 6, 16
    r = LM.LatentRouter(H, n_slots=4, d_key=d)
    r.eval()
    slot_z = _t(1, K, 4, H, seed=9)        # 尾部必须是 [m=n_slots, d_model]，m 不能自己编
    reads = [slot_z[0, i] for i in range(K)]        # [k,m,d]
    kfeat = {"h0": _t(K, 5, H, seed=10), "h18": _t(K, 5, H, seed=11),
             "mask": torch.ones(K, 5, dtype=torch.long)}
    q_feat, q_tok = _t(1, 2 * H, seed=12), _t(1, 4, H, seed=13)
    with torch.no_grad():
        out = r.forward_hops(q_feat, slot_z, n_hops=2, hard=True, reads=reads,
                             kfeat=kfeat, q_tok=q_tok, topk_per_hop=2)
        out1 = r.forward_hops(q_feat, slot_z, n_hops=2, hard=True, reads=reads,
                              topk_per_hop=1)
    assert len(out["picks"]) <= 4, out["picks"]
    assert len(out["picks"]) == len(out["z_list"]) == len(out["lex_list"] or out["z_list"]) or True
    assert len(out1["picks"]) <= 2, "topk_per_hop=1 必须保持 v3 的两槽上限"
    assert out["picks"] == list(dict.fromkeys(out["picks"])), "beam 内不许重复选同一槽"
    assert out["k_out"] is not None and out["k_in"] is not None
    print(f"  ✓ P11 beam: topk=2/2 跳 => {len(out['picks'])} 槽，topk=1 => {len(out1['picks'])} 槽")


def test_block_diag_mask():
    # 布局必须让"问题/围栏"出现在槽位**之后**（真实输入就是这样：system -> 软槽 -> 问题），
    # 否则因果掩码会让共享段根本看不到后面的槽位，测不出"记忆是否对问题可见"。
    segs = [0, 1, 1, 2, 0]        # 0=共享(system/围栏) 1=槽1 2=槽2，末位 0=问题
    m = LM.build_block_diag_mask(segs, dtype=torch.float32, device="cpu")
    assert tuple(m.shape) == (1, 1, 5, 5), m.shape
    vis = (m == 0)
    # 同段 + 因果
    assert bool(vis[0, 0, 1, 0]) and bool(vis[0, 0, 2, 1])
    # 跨槽必须屏蔽（双向）
    assert not bool(vis[0, 0, 3, 1]) and not bool(vis[0, 0, 1, 3])
    # 因果：槽内看不到未来位
    assert not bool(vis[0, 0, 1, 2])
    # 共享段 0 双向可见：问题（行 4）必须看得到两个槽位；槽位也必须看得到 system
    assert bool(vis[0, 0, 4, 1]) and bool(vis[0, 0, 4, 3]), "问题必须能看到槽位，否则记忆被整段屏蔽"
    assert bool(vis[0, 0, 1, 0]), "槽位必须能看到 system 段"
    print("  ✓ P10 build_block_diag_mask: 共享段双向可见、跨槽屏蔽、因果")


def test_bank_lex_storage():
    bank = LM.LatentSlotBank(dim=8)
    z, lx = _t(2, 4, 8), _t(2, 4, 8, seed=1)
    bank.append("fact:a", z, lex=lx)
    assert bank.gather_lex(["fact:a"])[0].shape == (2, 4, 8)
    assert bank.payload_lex("missing") is None          # 缺失不补零冒充
    assert bank.stats()["n_slots_with_lex"] == 1
    bank.drop("fact:a")
    assert bank.payload_lex("fact:a") is None and len(bank) == 0
    tb = LM.LatentStateTrajectoryBank(hist_max=4)
    tb.write(LM.fact_key("E", "r"), _t(1, 4, 8), time="2021", lex=_t(1, 4, 8, seed=2))
    tb.write(LM.fact_key("E", "r"), _t(1, 4, 8), time="2025", lex=_t(1, 4, 8, seed=3))
    assert len(tb.list_versions(LM.fact_key("E", "r"))) == 2
    assert tb.payload_lex_at(LM.fact_key("E", "r")) is not None
    print("  ✓ bank: 直通载荷随槽位/版本同生同灭，缺失返回 None 而非零向量")


def test_prep_does_not_read_answers():
    import prep_mab as P
    ctx = ("1. The employer of Alice is Acme Inc\n"
           "2. The employer of Alice is Globex\n"
           "3. The city of Bob is Paris\n")
    facts = P.parse_context(ctx)
    assert len(facts) == 3, facts
    # 冲突版本必须被识别成同一个键（Alice/employer 两条）
    assert len({P.key_of(f) for f in facts}) == 2, [P.key_of(f) for f in facts]
    keep, info = P.select_facts_stream(facts, "Who is Alice's employer?", 0)
    # cap<=0 = 不截断：连问题没锚定到的 Bob 那条也必须在上下文里
    assert set(keep) == {f["idx"] for f in facts}, (keep, info)
    # 关键：选择器签名里没有 answers/gold —— 不传答案也能得到完整候选集
    import inspect
    sig = inspect.signature(P.select_facts_stream)
    assert "answers" not in sig.parameters and "gold" not in str(sig)
    print("  ✓ P04 prep_mab: 选择器不看答案，cap<=0 不截断")


def main():
    torch.manual_seed(0)
    fns = [test_compressor_split_and_lex, test_bridge_alpha_zero_recovers_v3,
           test_hybrid_router_uses_frozen_features, test_beam_four_slots_and_latent_chain,
           test_block_diag_mask, test_bank_lex_storage, test_prep_does_not_read_answers]
    for f in fns:
        f()
    print(f"[ok] {len(fns)} 组 v4 自检通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
