#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tests/test_readout_experiments.py — 读出结构三轮实验（E1~E4）的回归测试（全 CPU）。

每条都对应一个"改错了不会报错、只会让实验结论静默失真"的地方：

  L. E1 独立词形头：零初始化必须让**选择行为逐位等于旧实现**（否则新实验组与已有
     两组不可比）；L_cov 的梯度必须只进词形头 —— 语义注意力（queries/attn/
     input_proj/output_proj）一根都不能拿。这条是 E1 的全部意义所在。
  B. E2 原生 BPE：载荷行必须是**完整词的连续子词向量**，不许出现"半截词"或均值；
     整词放不下就跳过（有效词变少是可接受的），不能截断。
  S. E3 分开注入：语义 token 的范数必须按**整向量 L2**对齐到 target_norm，
     不是 RMSNorm 之后再乘常数（那会把范数推到 ≈target·√d）；词形行必须原样复制。
  H. E4 隐状态蒸馏：位置必须是"答案首 token 的前一位"；余弦只看方向不看范数；
     权重为 0 时完全不进 loss。

用法：python tests/test_readout_experiments.py
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

N_OK = 0
FAILS = []


def ok(name, cond, extra=""):
    global N_OK
    if cond:
        N_OK += 1
    else:
        FAILS.append(name)
        print(f"[FAIL] {name} {extra}")
    return bool(cond)


def _comp(seed=0, lex_head=0, token_mode="mean", m=4, d=16, n_words=None):
    torch.manual_seed(seed)
    return LM.LatentCompressor(d, n_slots=m, inner_dim=d, num_heads=2, lex_skip=True,
                               lex_head=lex_head, lex_token_mode=token_mode)


def _inputs(B=1, L=8, d=16, wid=None):
    torch.manual_seed(1)
    h0 = torch.randn(B, L, d)
    h18 = torch.randn(B, L, d)
    mask = torch.ones(B, L, dtype=torch.long)
    wid = torch.arange(L).view(1, L).repeat(B, 1) if wid is None else wid
    W = int(wid.max().item()) + 1
    pen = torch.ones(B, W)
    wgt = torch.zeros(B, W)
    return h0, h18, mask, wid, pen, wgt


# =====================================================================
# L. E1：独立词形选择头
# =====================================================================
def test_lex_head_zero_init():
    """零初始化 => delta≡0 => 选词与共享注意力的旧实现逐位相同。"""
    torch.manual_seed(0)
    base = _comp(seed=0, lex_head=0)
    head = _comp(seed=0, lex_head=1)
    ok("L1 lex_head=1 时注册了词形头", head.lex_scorer is not None)
    ok("L2 lex_head=0 时不注册词形头", base.lex_scorer is None)

    # 共享参数必须逐位相同（词形头在所有既有模块之后创建，不扰动 RNG 顺序）
    same = all(torch.equal(a, b) for a, b in zip(
        base.attn.parameters(), head.attn.parameters()))
    ok("L3 打开词形头不改变共享注意力权重（同一 seed 下逐位相同）", same)
    ok("L4 queries 也逐位相同", torch.equal(base.queries, head.queries))

    h0, h18, mask, wid, pen, wgt = _inputs()
    _z0, _zl0, i0 = base.forward_split(h0, h18, mask, word_ids=wid, word_penalty=pen)
    _z1, _zl1, i1 = head.forward_split(h0, h18, mask, word_ids=wid, word_penalty=pen)
    ok("L5 第 0 步选词与旧实现逐位相同",
       i0["lex_word_pick"] == i1["lex_word_pick"],
       extra=f"{i0['lex_word_pick']} vs {i1['lex_word_pick']}")
    ok("L6 第 0 步词形载荷也逐位相同（delta 严格为 0）",
       torch.equal(_zl0, _zl1))
    ok("L7 info 里标明用的是独立选择头", i1.get("lex_head") is True
       and i0.get("lex_head") is False)


def test_lex_head_gradient_isolation():
    """只反传 L_cov：词形头必须有梯度，语义注意力/投影必须**一个都没有**。"""
    torch.manual_seed(0)
    comp = _comp(seed=0, lex_head=1)
    h0, h18, mask, wid, pen, _wgt = _inputs(B=1, L=8, d=16)
    wgt = torch.tensor([[3.0, 2.0, 2.0, 0.5, 0.5, 0.0, 0.0, 0.0]])
    _z, _zl, info = comp.forward_split(h0, h18, mask, word_ids=wid, word_penalty=pen,
                                       word_weight=wgt, want_cov=True)
    cl = info.get("cov_loss")
    ok("L8 词形头模式下 L_cov 仍然建了计算图",
       cl is not None and bool(cl.requires_grad))
    if cl is None:
        return
    comp.zero_grad(set_to_none=True)
    cl.backward()

    def _gsum(p):
        return 0.0 if p is None or p.grad is None else float(p.grad.abs().sum())

    # 末层是零初始化，所以第 1 步只有**末层**能拿到梯度（d(loss)/d(W_hidden) ∝ W_last = 0）。
    # 这是零初始化的固有性质，不是死路：末层一动，隐层下一步就有梯度（L9a 验证）。
    ok("L9 梯度回到词形头末层（lex_scorer.net[-1]）",
       _gsum(comp.lex_scorer.net[-1].weight) > 0.0,
       extra=str(_gsum(comp.lex_scorer.net[-1].weight)))
    ok("L9a 第 1 步隐层梯度为 0（零初始化的固有性质，符合预期）",
       _gsum(comp.lex_scorer.net[0].weight) == 0.0)
    # 走一步之后隐层必须能拿到梯度（否则词形头永远只能学一个线性残差）
    if _gsum(comp.lex_scorer.net[-1].weight) > 0.0:
        with torch.no_grad():
            comp.lex_scorer.net[-1].weight -= 1e-3 * comp.lex_scorer.net[-1].weight.grad
            comp.lex_scorer.net[-1].bias -= 1e-3 * comp.lex_scorer.net[-1].bias.grad
        comp.zero_grad(set_to_none=True)
        _z9, _zl9, i9 = comp.forward_split(
            h0, h18, mask, word_ids=wid, word_penalty=pen,
            word_weight=wgt, want_cov=True)
        if i9.get("cov_loss") is not None:
            i9["cov_loss"].backward()
        ok("L9b 末层动过之后隐层也能拿到梯度（不是死锁在纯线性）",
           _gsum(comp.lex_scorer.net[0].weight) > 0.0,
           extra=str(_gsum(comp.lex_scorer.net[0].weight)))
    ok("L10 语义 queries 拿不到 L_cov 的梯度", _gsum(comp.queries) == 0.0,
       extra=str(_gsum(comp.queries)))
    ok("L11 语义注意力（attn）拿不到 L_cov 的梯度",
       _gsum(comp.attn.in_proj_weight) == 0.0, extra=str(_gsum(comp.attn.in_proj_weight)))
    ok("L12 input_proj 拿不到 L_cov 的梯度",
       _gsum(comp.input_proj.weight) == 0.0)
    ok("L13 output_proj 拿不到 L_cov 的梯度",
       _gsum(comp.output_proj.weight) == 0.0)

    # 对照：旧行为（共享注意力）下**确实**会更新语义注意力 —— 否则 L10~L13 是空测
    torch.manual_seed(0)
    comp2 = _comp(seed=0, lex_head=0)
    _z2, _zl2, i2 = comp2.forward_split(h0, h18, mask, word_ids=wid, word_penalty=pen,
                                        word_weight=wgt, want_cov=True)
    i2["cov_loss"].backward()
    ok("L14 对照：共享注意力模式下 queries 确实被 L_cov 更新（说明 L10 不是空测）",
       comp2.queries.grad is not None and float(comp2.queries.grad.abs().sum()) > 0.0)

    # 关掉 want_cov 时两条路径都不能产生 cov_loss
    _z3, _zl3, i3 = comp.forward_split(h0, h18, mask, word_ids=wid, word_penalty=pen,
                                       word_weight=wgt, want_cov=False)
    ok("L15 want_cov=False 时 info 里没有 cov_loss", i3.get("cov_loss") is None)


# =====================================================================
# B. E2：原生 BPE 直通
# =====================================================================
def _marker_h0(B, L, d):
    """每个 token 一个可辨识的向量：第 i 个 token = [i+1]*d，方便核对"原样复制"。"""
    h0 = torch.zeros(B, L, d)
    for i in range(L):
        h0[:, i, :] = float(i + 1)
    return h0


def test_bpe_payload():
    """bpe 模式：载荷行是整词子词向量的**原样复制**，整词放不下就跳过。"""
    torch.manual_seed(0)
    comp = _comp(seed=0, token_mode="bpe", m=4, d=16)
    B, L, d = 1, 10, 16
    # 词 0=2 token, 词 1=3 token, 词 2=1 token, 词 3=4 token（刚好铺满 m）
    wid = torch.tensor([[0, 0, 1, 1, 1, 2, 3, 3, 3, 3]])
    pen = torch.ones(B, 4)
    h0 = _marker_h0(B, L, d)
    h18 = torch.zeros(B, L, d)
    mask = torch.ones(B, L, dtype=torch.long)
    _z, zl, info = comp.forward_split(h0, h18, mask, word_ids=wid, word_penalty=pen)

    ok("B1 bpe 载荷形状仍是 [B,m,d]（下游不用改）",
       tuple(zl.shape) == (B, 4, d), extra=str(tuple(zl.shape)))
    ok("B2 info 记录口径为 bpe", info.get("lex_token_mode") == "bpe")

    # 每一行要么是零（空槽），要么正好等于某个 h0 token 向量 —— 不允许出现均值
    rows = zl[0]
    h0_rows = {round(float(v[0]), 6) for v in h0[0]}
    for r in rows:
        v = round(float(r[0]), 6)
        ok(f"B3 载荷行 {v} 是原样复制的 token（不是均值）",
           (abs(v) < 1e-9) or (v in h0_rows), extra=f"v={v} 候选={sorted(h0_rows)}")

    packed = info.get("lex_words_packed")
    ok("B4 info 记录了真正铺进去的词编号", isinstance(packed, list) and len(packed) == B,
       extra=str(type(packed)))
    if packed:
        used_rows = info["lex_rows_used"][0]
        ok("B5 用掉的 token 行数 <= m", used_rows <= 4, extra=str(used_rows))
        # 每个词完整放进去了：行数 = 该词 token 数之和
        counts = {0: 2, 1: 3, 2: 1, 3: 4}
        ok("B6 放进去的行数正好等于这些词的 token 总数",
           used_rows == sum(counts[w] for w in packed[0]),
           extra=f"rows={used_rows} words={packed[0]}")

    # 空位必须留零（不能把别的词塞进来冒充）
    n_used = int(info["lex_rows_used"][0])
    ok("B7 没用到的行是全零（空槽语义保留）",
       bool((rows[n_used:].abs().sum() == 0)) if n_used < 4 else True)


def test_bpe_no_truncation():
    """超过 m 个 token 的词必须整词跳过，绝不截断（否则"精确恢复"这个指标就假了）。"""
    torch.manual_seed(0)
    comp = _comp(seed=0, token_mode="bpe", m=2, d=16)
    B, L, d = 1, 6, 16
    # 词 0 = 1 token，词 1 = 4 token（> m=2，永远放不下），词 2 = 1 token
    wid = torch.tensor([[0, 1, 1, 1, 1, 2]])
    pen = torch.ones(B, 3)
    h0 = _marker_h0(B, L, d)
    h18 = torch.zeros(B, L, d)
    mask = torch.ones(B, L, dtype=torch.long)
    _z, zl, info = comp.forward_split(h0, h18, mask, word_ids=wid, word_penalty=pen)
    packed = info.get("lex_words_packed") or [[]]
    ok("B8 超长词被整词跳过（没有出现在 packed 里）", 1 not in packed[0],
       extra=str(packed))
    ok("B9 跳过长词后仍然可以放下别的整词", len(packed[0]) >= 1, extra=str(packed))
    ok("B10 每行都不含半截词：行数 = 已放词的 token 数之和",
       int(info["lex_rows_used"][0]) == sum({0: 1, 2: 1}[w] for w in packed[0]),
       extra=str(info["lex_rows_used"]))


def test_lex_mean_unchanged():
    """mean 模式必须是"该词全部子词取均值"，与 bpe 模式口径不同。"""
    torch.manual_seed(0)
    comp = _comp(seed=0, token_mode="mean", m=4, d=16)
    B, L, d = 1, 4, 16
    wid = torch.tensor([[0, 0, 1, 1]])
    pen = torch.ones(B, 2)
    h0 = _marker_h0(B, L, d)
    h18 = torch.zeros(B, L, d)
    mask = torch.ones(B, L, dtype=torch.long)
    _z, zl, info = comp.forward_split(h0, h18, mask, word_ids=wid, word_penalty=pen)
    ok("B11 mean 模式 info 口径正确", info.get("lex_token_mode") == "mean")
    ok("B12 mean 模式不记录 packed（那是 bpe 专属）",
       info.get("lex_words_packed") is None)
    # 词 0 的两个 token 是 1.0 和 2.0，均值 1.5；它一定被选（两词里必选其一进前两槽）
    vals = [round(float(r[0]), 4) for r in zl[0][:2]]
    ok("B13 mean 模式出现均值 1.5 或 3.5（词内平均，不是原样复制）",
       1.5 in vals or 3.5 in vals, extra=str(vals))


# =====================================================================
# S. 混合表示（实验 2）：n_sym 个符号位 + n_rel 个关系位，总位置数不变
# =====================================================================
def test_bridge_hybrid():
    torch.manual_seed(0)
    d, m, tn = 32, 4, 1.05
    br = LM.LatentBridge(d, bottleneck=8, target_norm=tn, lex_skip=True,
                         compose="split", n_sym=2, n_rel=2)
    B = 2
    z = torch.randn(B, m, d) * 0.4
    lex = torch.randn(B, 2, d)
    lex = lex / lex.norm(dim=-1, keepdim=True) * tn
    out = br.forward_split(z, lex)
    ok("S1 输出形状 = [B, n_sym+n_rel, d]（总位置数与残差模式相同）",
       tuple(out.shape) == (B, 4, d), extra=str(tuple(out.shape)))
    ok("S2 符号位**原样复制**词形向量（不做任何混合）",
       torch.allclose(out[:, :2, :], lex, atol=1e-6),
       extra=str(float((out[:, :2, :] - lex).abs().max())))
    sem = out[:, 2:, :]
    ok("S3 关系位的**整向量 L2 范数**对齐到 target_norm",
       torch.allclose(sem.norm(dim=-1), torch.full((B, 2), tn), atol=1e-4),
       extra=str([round(float(v), 4) for v in sem.detach().norm(dim=-1).flatten()[:4]]))
    ok("S4 关系范数不是 RMSNorm×常数的量级（≈target·√d）",
       float(sem.detach().norm(dim=-1).max()) < tn * 3.0)
    # 纯符号臂：n_rel=0
    br_sym = LM.LatentBridge(d, bottleneck=8, target_norm=tn, lex_skip=True,
                             compose="split", n_sym=4, n_rel=0)
    lex4 = torch.randn(B, 4, d)
    lex4 = lex4 / lex4.norm(dim=-1, keepdim=True) * tn
    out_sym = br_sym.forward_split(z, lex4)
    ok("S5 n_rel=0 时输出就是 4 个符号位（纯符号臂）",
       tuple(out_sym.shape) == (B, 4, d)
       and torch.allclose(out_sym, lex4, atol=1e-6), extra=str(tuple(out_sym.shape)))
    # 无信息占位：关系位与输入无关、且每块每位置都相同
    br_blank = LM.LatentBridge(d, bottleneck=8, target_norm=tn, lex_skip=True,
                               compose="split", n_sym=2, n_rel=2, rel_mode="blank")
    o1 = br_blank.forward_split(z, lex)
    o2 = br_blank.forward_split(torch.randn(B, m, d) * 3.0, lex)
    ok("S6 占位模式的关系位与输入内容无关（换一份语义载荷结果不变）",
       torch.allclose(o1[:, 2:, :], o2[:, 2:, :], atol=1e-6))
    ok("S7 占位模式的关系位在块/位置之间完全相同，但范数仍对齐 target_norm",
       torch.allclose(o1[:, 2:, :], o1[:, 2:3, :].expand_as(o1[:, 2:, :]), atol=1e-6)
       and abs(float(o1[0, 2].norm()) - tn) < 1e-3,
       extra=str(round(float(o1[0, 2].norm()), 4)))
    # 结构校验
    ok("S8 n_rel 超过语义槽数时直接报错（不能静默少给位置）",
       _raises(lambda: br.forward_split(z, lex, n_sym=2, n_rel=9)))
    ok("S9 n_sym>0 但没给词形载荷时直接报错（不能静默变成纯语义）",
       _raises(lambda: br.forward_split(z, None)))
    ok("S10 forward() 在 split 模式下自动分派（漏一处就会静默退回残差）",
       tuple(br(z, lex).shape) == (B, 4, d))
    torch.manual_seed(0)
    br_res = LM.LatentBridge(d, bottleneck=8, target_norm=tn, lex_skip=True,
                             compose="residual")
    ok("S11 残差模式仍然零初始化末层（第 0 步 E=z_lex 这个不变量不能破）",
       float(br_res.net[-1].weight.abs().sum()) == 0.0)
    torch.manual_seed(0)
    br_sp = LM.LatentBridge(d, bottleneck=8, target_norm=tn, lex_skip=True,
                            compose="split", n_sym=2, n_rel=2)
    ok("S12 split 模式不零初始化（零输出等于往序列里塞全零向量）",
       float(br_sp.net[-1].weight.abs().sum()) > 0.0)


def test_compose_hybrid_tokens():
    """compose 在 split 模式下每块的**位置数必须仍等于 m**（同预算的硬约束）。"""
    model = T.SmokeModel(vocab_size=256, hidden_size=32, n_layers=2, n_heads=2)
    tok = T.StubTok(vocab_size=256)
    z = [torch.randn(1, 4, 32)]
    lex = [torch.randn(1, 2, 32)]
    br_res = LM.LatentBridge(32, bottleneck=8, target_norm=1.05, lex_skip=True,
                             compose="residual")
    soft_res, _ = LM.compose_routed_soft(model, tok, br_res, z, lex_list=[torch.randn(1, 4, 32)],
                                         isolate=True, return_segs=True, fence_style="bracket")
    br_sp = LM.LatentBridge(32, bottleneck=8, target_norm=1.05, lex_skip=True,
                            compose="split", n_sym=2, n_rel=2)
    soft_sp, segs_sp = LM.compose_routed_soft(model, tok, br_sp, z, lex_list=lex,
                                              isolate=True, return_segs=True,
                                              fence_style="bracket")
    ok("S13 split(2+2) 与 residual(4) 的注入 token 总数相同",
       int(soft_sp.shape[1]) == int(soft_res.shape[1]),
       extra=f"split={int(soft_sp.shape[1])} residual={int(soft_res.shape[1])}")
    ok("S14 每槽内容 token 数仍是 4（2 符号 + 2 关系）",
       sum(1 for v in segs_sp if int(v) > 0) == 4,
       extra=str([int(v) for v in segs_sp]))
    ok("S15 纯语义臂（n_sym=0）不要求词形载荷",
       int(LM.compose_routed_soft(
           model, tok,
           LM.LatentBridge(32, bottleneck=8, target_norm=1.05, lex_skip=True,
                           compose="split", n_sym=0, n_rel=4),
           z, lex_list=None, fence_style="bracket").shape[1]) > 0)


# =====================================================================
# H. E4：隐状态蒸馏
# =====================================================================
def test_hid_distill():
    labels = torch.tensor([[-100, -100, 7, 8, -100]])
    ok("H1 位置 = 答案首 token 的前一位", T._hid_answer_pos(labels) == [1],
       extra=str(T._hid_answer_pos(labels)))
    ok("H2 没有答案时返回 -1（调用方跳过）",
       T._hid_answer_pos(torch.full((1, 4), -100)) == [-1])
    ok("H3 答案是第一个 token 时返回 -1（没有前一位可用）",
       T._hid_answer_pos(torch.tensor([[5, -100]])) == [-1])

    torch.manual_seed(0)
    v = torch.randn(1, 4, 8)
    s_hid = v.clone()
    t_hid = v.clone()
    l1 = torch.tensor([[-100, 6, 7, -100]])
    l2 = torch.tensor([[-100, 6, 7, -100]])
    d0 = T.hid_distill_loss(s_hid, l1, t_hid, l2)
    ok("H4 完全相同的隐状态 -> 损失 0", d0 is not None and abs(float(d0)) < 1e-6,
       extra=str(None if d0 is None else float(d0)))

    t_hid2 = -v.clone()           # 反向 => cos=-1 => loss=2
    d1 = T.hid_distill_loss(s_hid, l1, t_hid2, l2)
    ok("H5 反向向量 -> 损失接近 2", d1 is not None and abs(float(d1) - 2.0) < 1e-4,
       extra=str(None if d1 is None else float(d1)))

    # 只看方向不看范数：把 teacher 整体放大 100 倍，损失不变
    d2 = T.hid_distill_loss(s_hid, l1, t_hid * 100.0, l2)
    ok("H6 余弦口径：teacher 缩放不影响损失",
       d2 is not None and abs(float(d2)) < 1e-6, extra=str(None if d2 is None else float(d2)))

    # 层选择
    ok("H7 --hid-layer last 取最后一层",
       T._hid_pick_layer([1, 2, 3], "last") == 3)
    ok("H8 非法层号直接报错（不静默取最后一层）",
       _raises(lambda: T._hid_pick_layer([1, 2, 3], "9")))
    ok("H9 last+18 明确拒绝（避免两处同时改）",
       _raises(lambda: T._hid_pick_layer([1, 2, 3], "last+18")))

    # assemble_loss：权重为 0 时这一项完全不进 loss
    base = T.assemble_loss(torch.tensor(1.0), None, None, None, 1.0, 1.0, 0.5)
    with_hid = T.assemble_loss(torch.tensor(1.0), None, None, None, 1.0, 1.0, 0.5,
                               loss_hid=torch.tensor(10.0), hid_weight=0.5)
    off = T.assemble_loss(torch.tensor(1.0), None, None, None, 1.0, 1.0, 0.5,
                          loss_hid=torch.tensor(10.0), hid_weight=0.0)
    ok("H10 开了权重才加（1 + 0.5×10 = 6）", abs(float(with_hid) - 6.0) < 1e-5,
       extra=str(float(with_hid)))
    ok("H11 权重 0 时完全不进 loss", abs(float(off) - float(base)) < 1e-9)
    # recon_stage 会重置 loss，hid 项仍要生效（与 L_cov 同一条规则）
    rs = T.assemble_loss(torch.tensor(1.0), None, None, torch.tensor(2.0), 1.0, 1.0, 0.5,
                         recon_stage=True, loss_hid=torch.tensor(10.0), hid_weight=0.5)
    ok("H12 recon_stage 下 hid 项仍然生效（不被阶段切换吃掉）",
       abs(float(rs) - 7.0) < 1e-5, extra=str(float(rs)))


def test_ckpt_roundtrip():
    """E1/E2/E3 的结构开关必须能存进 ckpt 再原样还原（strict load 不容错）。"""
    import tempfile
    model = T.SmokeModel(vocab_size=256, hidden_size=32, n_layers=2, n_heads=2)
    tok = T.StubTok(vocab_size=256)
    torch.manual_seed(0)
    comp = LM.LatentCompressor(32, n_slots=4, inner_dim=16, num_heads=2, lex_skip=True,
                              lex_head=True, lex_token_mode="bpe", bpe_backfill=1,
                              n_sym=2)
    bridge = LM.LatentBridge(32, bottleneck=8, target_norm=1.05, lex_skip=True,
                            compose="split", n_sym=2, n_rel=2)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "rt.pt")
        LM.save_latent_checkpoint(path, comp, bridge, {"n_slots": 4, "inner_dim": 16,
                                                       "bottleneck": 8})
        rd = LM.load_latent_reader(path, model, tok, device="cpu")
        ok("C1 lex_scorer 被还原", getattr(rd.compressor, "lex_scorer", None) is not None)
        ok("C2 词形载荷口径被还原", rd.compressor.lex_token_mode == "bpe")
        ok("C3 bpe_backfill 被还原", int(rd.compressor.bpe_backfill) == 1)
        ok("C4 compose 模式被还原", str(rd.bridge.compose) == "split",
           extra=str(rd.bridge.compose))
        ok("C5 n_sym / n_rel 被还原（压缩器侧也要还原，否则符号位数量会变）",
           int(rd.bridge.n_sym) == 2 and int(rd.bridge.n_rel) == 2
           and int(rd.compressor.n_sym) == 2,
           extra=f"{rd.bridge.n_sym}+{rd.bridge.n_rel} comp={rd.compressor.n_sym}")
        # 权重逐位一致（strict=True 已经保证键集合；这里再核一次数值）
        ok("C6 词形头权重逐位一致",
           torch.equal(rd.compressor.lex_scorer.net[-1].weight,
                       comp.lex_scorer.net[-1].weight))
        # 只有词形头这一份参数是新增的：参数量必须变大，且前进/反向都能跑
        rd.compressor.train()
        h0, h18, mask, wid, pen, _wgt0 = _inputs(B=1, L=8, d=32)
        # 监督权重必须非零，否则 L_cov 没有目标、返回 None（那是对的，不是 bug）
        wgt = torch.tensor([[3.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0]])
        _z, _zl, info = rd.compressor.forward_split(
            h0, h18, mask, word_ids=wid, word_penalty=pen, word_weight=wgt, want_cov=True)
        # n_sym=2：词形载荷只保留 2 行（其余位置留给关系槽），这就是 split 的定义
        ok("C7 还原后的压缩器前向正常（词形载荷 = n_sym 行）",
           tuple(_zl.shape) == (1, 2, 32), extra=str(tuple(_zl.shape)))
        ok("C8 还原后的路径仍能算 L_cov",
           info.get("cov_loss") is not None)
        ok("C9 口径标记随 info 出来（诊断/日志要靠它归因）",
           info.get("lex_token_mode") == "bpe" and info.get("lex_head") is True)


def _raises(fn):
    """期望抛异常。**必须连 SystemExit 一起抓**：训练脚本的守卫用的是
    `raise SystemExit(...)`（命令行工具的正常退出方式），它继承 BaseException 而不是
    Exception —— 只抓 Exception 会让测试进程被守卫直接终止，看起来像"测试崩了"。
    """
    try:
        fn()
    except KeyboardInterrupt:                                 # pragma: no cover
        raise
    except BaseException:
        return True
    return False


# =====================================================================
# PA. 实验 A：原生 BPE 与潜层的**逐槽等预算**对照
# =====================================================================
def _probe():
    """懒加载探针模块：它 import 评测脚本，放在函数里可以让其它测试不受牵连。"""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import probe_readout_capacity as PRC
    return PRC


def test_budget_matching():
    """等预算的实现细节：数得对、挑得对、围栏与 compose 逐字一致。"""
    PRC = _probe()
    ok("PA1 per_slot_counts 只数段号>0 的 token",
       PRC.per_slot_counts([0, 1, 1, 1, 0, 2, 2], 2) == [3, 2],
       extra=str(PRC.per_slot_counts([0, 1, 1, 1, 0, 2, 2], 2)))
    ids = torch.tensor([10, 11, 12, 13, 14, 15, 16])
    tok = T.StubTok(vocab_size=256)
    pre = PRC.pick_ids(tok, ids, 3, "raw_prefix")
    ok("PA2 raw_prefix 是原序前缀（不是乱序挑）",
       pre.tolist() == [10, 11, 12], extra=str(pre.tolist()))
    sal = PRC.pick_ids(tok, ids, 3, "raw_salient")
    ok("PA3 raw_salient 结果按原文顺序单调递增（拼回原序，不打乱句子）",
       sal.tolist() == sorted(sal.tolist()) and len(set(sal.tolist())) == len(sal),
       extra=str(sal.tolist()))
    ok("PA4 预算 >= 可用 token 时全给（不能凭空造词）",
       PRC.pick_ids(tok, ids, 99, "raw_prefix").tolist() == ids.tolist())
    ok("PA5 raw_full 不受预算限制",
       PRC.pick_ids(tok, ids, 2, "raw_full").tolist() == ids.tolist())


    # 围栏文本必须与 compose_routed_soft 逐字一致：这是"骨架相同"的唯一保证。
    model = T.SmokeModel(vocab_size=256, hidden_size=32, n_layers=2, n_heads=2)
    bridge = LM.LatentBridge(32, bottleneck=8, target_norm=1.05, lex_skip=False)
    z = [torch.randn(1, 4, 32), torch.randn(1, 4, 32)]
    for style, iso in (("bracket", False), ("bracket", True), ("verbose", False)):
        soft, segs = LM.compose_routed_soft(model, tok, bridge, z, lex_list=None,
                                            isolate=iso, return_segs=True,
                                            fence_style=style)
        runs, cur = [], 0
        for v in segs:                        # 数出每段围栏实际占几个 token
            if int(v) == 0:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        want = [int(LM._tokenize_no_special(
            tok, PRC._fence_text(j, style, iso), torch.device("cpu")).shape[1])
            for j in range(len(z))]
        ok(f"PA6 围栏 token 数与 compose 实际插入的一致（{style}/isolate={iso}）",
           runs[:len(want)] == want, extra=f"实际={runs[:len(want)]} 期望={want}")


def test_word_units():
    """整词切分与去重：这是 `raw_word` 相对 token 级 `raw_salient` 的全部差别。"""
    PRC = _probe()

    class _Tok:
        """极简 BPE 替身：词首带空格，词内子词不带。"""
        VOCAB = {0: "[", 1: " Mari", 2: "etta", 3: " Air", 4: " Force",
                 5: " Station", 6: " is", 7: " in", 8: " Georgia",
                 9: " Robert", 10: " Salisbury", 11: ",", 12: " 1968"}

        def decode(self, ids, skip_special_tokens=True):
            return "".join(self.VOCAB[int(i)] for i in ids)

    tk = _Tok()
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    spans = PRC._word_spans(tk, ids)
    words = [tk.decode(ids[a:b].tolist()).strip() for a, b in spans]
    ok("PA13 整词切分：词首空格是边界（Marietta 是一个词，Air/Force 是两个）",
       words[:5] == ["[", "Marietta", "Air", "Force", "Station"], extra=str(words))
    keep, picked = PRC.pick_word_ids(tk, ids, 6)
    ok("PA14 整词挑选不产生半截词（每个被选词都能在原句里找到完整词面）",
       all(w in "".join(words) for w in picked), extra=str(picked))
    def _is_subseq(sub, full):
        it = iter(full.tolist())
        return all(any(x == y for y in it) for x in sub.tolist())

    ok("PA15 整词挑选按原文顺序拼回（结果是原序列的子序列，不是重排）",
       _is_subseq(keep, ids), extra=str(keep.tolist()))
    # 同一词面重复出现时只选一次（token 级没有这个保证）
    ids2 = torch.cat([ids, torch.tensor([9, 10])])          # Robert Salisbury 再来一遍
    _k2, picked2 = PRC.pick_word_ids(tk, ids2, 12)
    ok("PA16 同一词面只选一次（token 级会重复挑 Robert / Salisbury）",
       len(picked2) == len(set(picked2)), extra=str(picked2))


def test_answer_type():
    import eval_latent_table as EVM
    cases = {"1968": "date", "May 2019": "date", "3rd": "date", "yes": "yesno",
             "1,234": "number", "Virginia Commonwealth University": "entity",
             "": "empty"}
    got = {k: EVM.answer_type(k) for k in cases}
    ok("PA7 题型分类（日期/数字/实体/是非）逐例正确", got == cases, extra=str(got))


def test_raw_skeleton_parity():
    """同一道题、同一批槽位：原生 BPE 与潜层的注入骨架必须**逐 token 对齐**。

    这条是实验 A 的全部立论基础。骨架一旦漂了（围栏少一个 token、槽位顺序变了），
    两条读法的差就不再是"载荷不同"，而会把"模板不同"算成"压缩损失"。
    """
    PRC = _probe()
    model = T.SmokeModel(vocab_size=512, hidden_size=32, n_layers=2, n_heads=2)
    tok = T.StubTok(vocab_size=512)
    bridge = LM.LatentBridge(32, bottleneck=8, target_norm=1.05, lex_skip=False)
    z = [torch.randn(1, 4, 32), torch.randn(1, 4, 32)]
    soft_lat, segs_lat = LM.compose_routed_soft(model, tok, bridge, z, lex_list=None,
                                                isolate=False, return_segs=True,
                                                fence_style="bracket")
    counts = PRC.per_slot_counts(segs_lat, len(z))
    ok("PA8 latent 每槽 4 个内容向量（m=4）", counts == [4, 4], extra=str(counts))
    item = {"units": ["alpha beta gamma delta epsilon zeta",
                      "one two three four five six"],
            "evidence": ["alpha beta gamma delta epsilon zeta",
                         "one two three four five six"],
            "question": "which one?"}
    for arm in ("raw_prefix", "raw_salient"):
        soft_r, segs_r, info_r = PRC.build_raw_soft(
            model, tok, item, [0, 1], counts, torch.device("cpu"), arm, "bracket", False)
        ok(f"PA9a {arm} 围栏数与 latent 逐个相等（骨架没漂）",
           info_r["fences"] == PRC.fence_runs(segs_lat), extra=str(info_r["fences"]))
        ok(f"PA9 {arm} 的总 token 数与 latent 完全相同",
           int(soft_r.shape[1]) == int(soft_lat.shape[1]),
           extra=f"raw={int(soft_r.shape[1])} latent={int(soft_lat.shape[1])}")
        ok(f"PA10 {arm} 的段号序列与 latent 完全相同（围栏位置也没漂）",
           [int(v) for v in segs_r] == [int(v) for v in segs_lat],
           extra=f"raw={segs_r} latent={[int(v) for v in segs_lat]}")
        # 内容位置上必须是**真实词表向量**：这是"原生 BPE 直通"的定义，
        # 一旦变成软向量（或均值），这条基线就不是它声称的那条基线了。
        _emb_w = model.get_input_embeddings().weight.detach()
        _content = soft_r[0, 1:5].detach()
        _dmin = (_content[:, None, :] - _emb_w[None, :, :]).abs().sum(-1).min(dim=1).values
        ok(f"PA11 {arm} 每个内容位置都是词表里真实存在的 embedding 行",
           float(_dmin.max()) < 1e-6, extra=f"最大距离={float(_dmin.max()):.2e}")
    soft_f, segs_f, _info_f = PRC.build_raw_soft(
        model, tok, item, [0, 1], [0, 0], torch.device("cpu"), "raw_full", "bracket", False)
    ok("PA12 raw_full 不受预算限制（token 数多于 latent）",
       int(soft_f.shape[1]) > int(soft_lat.shape[1]),
       extra=f"full={int(soft_f.shape[1])} latent={int(soft_lat.shape[1])}")


# =====================================================================
# PB. 实验 B：纯语义读出（--no-lex-skip）的口径与守卫
# =====================================================================
def test_pure_sem_guards():
    """冲突开关必须**直接报错**：静默不生效会让一轮一小时的训练白跑。"""
    def _args(*extra):
        return T.parse_args(["--data", "x.jsonl"] + list(extra))

    ok("PB1 --no-lex-skip + --lex-head 1 直接报错（否则独立词形头根本不会被创建）",
       _raises(lambda: T.validate_lex_config(_args("--no-lex-skip", "--lex-head", "1"))))
    ok("PB2 --no-lex-skip + --cov-weight>0 直接报错（L_cov 目标恒为空）",
       _raises(lambda: T.validate_lex_config(
           _args("--no-lex-skip", "--cov-weight", "0.1"))))
    ok("PB3 --no-lex-skip + --split-inject 1 直接报错（没有词形可分开注入）",
       _raises(lambda: T.validate_lex_config(
           _args("--no-lex-skip", "--split-inject", "1"))))
    try:
        T.validate_lex_config(_args("--no-lex-skip", "--cov-weight", "0",
                                    "--lex-head", "0", "--recon-lex", "0"))
        ok("PB4 纯语义的合法组合不报错", True)
    except Exception as e:                                    # pragma: no cover
        ok("PB4 纯语义的合法组合不报错", False, extra=f"{type(e).__name__}: {e}")
    try:
        T.validate_lex_config(_args())                        # 默认 lex_skip=True
        ok("PB5 默认配置（有词形直通）不受影响", True)
    except Exception as e:                                    # pragma: no cover
        ok("PB5 默认配置（有词形直通）不受影响", False, extra=f"{type(e).__name__}: {e}")


def test_pure_sem_readout_identity():
    """实验 B 的核心性质：训练读出与 oracle_sem 评测**同一个函数同一种输入**。

    `lex_skip=False` 时 `alpha()` 恒 0，且 forward 的残差分支要求 `self.lex_skip`，
    所以就算有人把词形载荷传进来也会被忽略 —— 这条不变量破了，"训练/评测同构"
    这个立论就没了。
    """
    torch.manual_seed(0)
    d, m, tn = 32, 4, 1.05
    br = LM.LatentBridge(d, bottleneck=8, target_norm=tn, lex_skip=False)
    ok("PB6 纯语义 Bridge 的 α 恒为 0", float(br.alpha()) == 0.0,
       extra=str(float(br.alpha())))
    z = torch.randn(2, m, d)
    out = br(z)
    ok("PB7 纯语义输出仍是**满范数**（不是被 α 缩小的残差）",
       torch.allclose(out.norm(dim=-1), torch.full((2, m), tn), atol=1e-4),
       extra=str([round(float(v), 4) for v in out.detach().norm(dim=-1).flatten()[:4]]))
    lex = torch.randn(2, m, d)
    lex = lex / lex.norm(dim=-1, keepdim=True) * tn
    ok("PB8 传进来的词形载荷被忽略（没有直通分支就不可能有残差混合）",
       torch.allclose(br(z, lex), out, atol=1e-6))
    model = T.SmokeModel(vocab_size=256, hidden_size=32, n_layers=2, n_heads=2)
    tok = T.StubTok(vocab_size=256)
    s_none, seg_none = LM.compose_routed_soft(model, tok, br, [z[:1]], lex_list=None,
                                              return_segs=True, fence_style="bracket")
    s_lex, seg_lex = LM.compose_routed_soft(model, tok, br, [z[:1]], lex_list=[lex[:1]],
                                            return_segs=True, fence_style="bracket")
    ok("PB9 compose 在纯语义权重下给不给 lex_list 结果逐位相同",
       torch.allclose(s_none, s_lex, atol=1e-6)
       and [int(v) for v in seg_none] == [int(v) for v in seg_lex])


def test_split_budget():
    """实验 2 的「同预算」是硬约束：n_sym + n_rel 必须正好等于 m。"""
    def _a(*extra):
        return T.parse_args(["--data", "x.jsonl"] + list(extra))

    ok("S16 split 2+2 解析成 (2,2)",
       T.resolve_split_budget(_a("--compose", "split", "--n-sym", "2", "--n-rel", "2"))
       == ("split", 2, 2))
    ok("S17 split 4+0（纯符号臂）解析成 (4,0)",
       T.resolve_split_budget(_a("--compose", "split", "--n-sym", "4", "--n-rel", "0"))
       == ("split", 4, 0))
    ok("S18 只给 n-sym 时 n-rel 自动补到 m",
       T.resolve_split_budget(_a("--compose", "split", "--n-sym", "1"))
       == ("split", 1, 3))
    ok("S19 位置总数不等于 m 直接报错（否则同预算不成立）",
       _raises(lambda: T.resolve_split_budget(
           _a("--compose", "split", "--n-sym", "2", "--n-rel", "1"))))
    ok("S20 split 且没给任何位置数直接报错",
       _raises(lambda: T.resolve_split_budget(_a("--compose", "split"))))
    ok("S21 split + n_sym>0 + --no-lex-skip 直接报错（符号槽没有东西可放）",
       _raises(lambda: T.resolve_split_budget(
           _a("--compose", "split", "--n-sym", "2", "--no-lex-skip"))))
    ok("S22 residual 模式忽略 n-sym/n-rel，位置数恒为 m",
       T.resolve_split_budget(_a("--compose", "residual", "--n-sym", "2"))
       == ("residual", 4, 0))


def test_eval_stratify():
    """评测侧的分层统计：总 EM 会把"日期题全灭 + 实体题变好"平均成"没变化"。"""
    import eval_latent_table as EVM
    rows = [
        {"ok": True, "em": 1.0, "f1": 1.0, "ans_type": "date", "hops": 2,
         "question": "q1", "mem_tokens": 8, "mem_plain_tokens": 0, "prompt_tokens": 40,
         "total_ms": 1.0, "pre_ms": 0.0, "read_ok": True},
        {"ok": False, "em": 0.0, "f1": 0.5, "ans_type": "entity", "hops": 1,
         "question": "q2", "mem_tokens": 8, "mem_plain_tokens": 0, "prompt_tokens": 40,
         "total_ms": 1.0, "pre_ms": 0.0, "read_ok": True},
    ]
    s = EVM.summarize(rows, "t")
    ok("PB10 summarize 产出按题型分层（date 1/1、entity 0/1）",
       s["by_ans_type"]["date"]["em"] == 1.0
       and s["by_ans_type"]["entity"]["em"] == 0.0,
       extra=str({k: round(v["em"], 3) for k, v in s["by_ans_type"].items()}))
    ok("PB11 summarize 产出按跳数分层", set(s["by_hops"]) == {"1", "2"},
       extra=str(list(s["by_hops"])))
    ok("PB12 题型分类只有一个实现（探针与评测脚本共用同一个函数）",
       _probe().answer_type is EVM.answer_type)


def main():
    for fn in (test_lex_head_zero_init, test_lex_head_gradient_isolation,
               test_bpe_payload, test_bpe_no_truncation, test_lex_mean_unchanged,
               test_bridge_hybrid, test_compose_hybrid_tokens, test_hid_distill,
               test_ckpt_roundtrip, test_budget_matching, test_word_units,
               test_answer_type,
               test_raw_skeleton_parity, test_pure_sem_guards,
               test_pure_sem_readout_identity, test_split_budget,
               test_eval_stratify):
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
