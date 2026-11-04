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
# S. E3：词形与语义分开注入
# =====================================================================
def test_bridge_split():
    torch.manual_seed(0)
    d, m, tn = 32, 4, 1.05
    br = LM.LatentBridge(d, bottleneck=8, target_norm=tn, lex_skip=True,
                         split_inject=True, split_n_lex=3)
    B = 2
    z = torch.randn(B, m, d) * 0.4
    lex = torch.randn(B, m, d)
    lex = lex / lex.norm(dim=-1, keepdim=True) * tn
    lex[1, 2] = 0.0                              # 造一个空槽
    out = br.forward_separate(z, lex)
    ok("S1 输出形状 = [B, n_lex+1, d]", tuple(out.shape) == (B, 4, d),
       extra=str(tuple(out.shape)))
    sem = out[:, 3, :]
    ok("S2 语义行的**整向量 L2 范数**对齐到 target_norm",
       torch.allclose(sem.norm(dim=-1), torch.full((B,), tn), atol=1e-4),
       extra=str([round(float(v), 4) for v in sem.detach().norm(dim=-1)]))
    ok("S3 语义范数不是 RMSNorm×常数的量级（≈target·√d）",
       float(sem.detach().norm(dim=-1).max()) < tn * 3.0,
       extra=f"max={float(sem.detach().norm(dim=-1).max()):.3f} 上界={tn * 3.0:.3f}")
    ok("S4 词形行原样复制（不做任何残差混合）",
       torch.allclose(out[0, :3, :], lex[0, :3, :]),
       extra=str(float((out[0, :3, :] - lex[0, :3, :]).abs().max())))
    ok("S5 空槽的词形行用语义向量兜底（不是全零向量）",
       torch.allclose(out[1, 2, :], sem[1], atol=1e-6))
    # split 模式下 Bridge 末层**不能**零初始化（否则语义 token 恒为零向量）
    ok("S6 split 模式不零初始化末层",
       float(br.net[-1].weight.abs().sum()) > 0.0)
    torch.manual_seed(0)
    br2 = LM.LatentBridge(d, bottleneck=8, target_norm=tn, lex_skip=True,
                          split_inject=False)
    ok("S7 残差模式仍然零初始化末层（第 0 步 E=z_lex 这个不变量不能破）",
       float(br2.net[-1].weight.abs().sum()) == 0.0)


def test_compose_split_tokens():
    """compose 在 split 模式下每槽的 content token 数 = k+1（k=词形行数）。"""
    model = T.SmokeModel(vocab_size=256, hidden_size=32, n_layers=2, n_heads=2)
    tok = T.StubTok(vocab_size=256)
    bridge = LM.LatentBridge(32, bottleneck=8, target_norm=1.05, lex_skip=True,
                             split_inject=True, split_n_lex=3)
    z = [torch.randn(1, 4, 32)]
    lex = [torch.randn(1, 4, 32)]
    soft_split, segs_split = LM.compose_routed_soft(
        model, tok, bridge, z, lex_list=lex, isolate=True, return_segs=True,
        fence_style="bracket")
    bridge.split_inject = False
    soft_res, _segs_res = LM.compose_routed_soft(
        model, tok, bridge, z, lex_list=lex, isolate=True, return_segs=True,
        fence_style="bracket")
    n_split = int(soft_split.shape[1])
    n_res = int(soft_res.shape[1])
    ok("S9 两种模式的 token 总数相同（4 content/槽，预算没变）",
       n_split == n_res, extra=f"split={n_split} residual={n_res}")
    ok("S10 split 模式下 z_lex 为空时退回残差路径（oracle_sem 不能被改口径）",
       int(LM.compose_routed_soft(model, tok, bridge, z, lex_list=None,
                                  fence_style="bracket").shape[1]) > 0)


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
                              lex_head=True, lex_token_mode="bpe", bpe_backfill=1)
    bridge = LM.LatentBridge(32, bottleneck=8, target_norm=1.05, lex_skip=True,
                            split_inject=True, split_n_lex=3)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "rt.pt")
        LM.save_latent_checkpoint(path, comp, bridge, {"n_slots": 4, "inner_dim": 16,
                                                       "bottleneck": 8})
        rd = LM.load_latent_reader(path, model, tok, device="cpu")
        ok("C1 lex_scorer 被还原", getattr(rd.compressor, "lex_scorer", None) is not None)
        ok("C2 词形载荷口径被还原", rd.compressor.lex_token_mode == "bpe")
        ok("C3 bpe_backfill 被还原", int(rd.compressor.bpe_backfill) == 1)
        ok("C4 split_inject 被还原", bool(rd.bridge.split_inject) is True)
        ok("C5 split_n_lex 被还原", int(rd.bridge.n_lex) == 3)
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
        ok("C7 还原后的压缩器前向正常（bpe 载荷形状不变）",
           tuple(_zl.shape) == (1, 4, 32), extra=str(tuple(_zl.shape)))
        ok("C8 还原后的路径仍能算 L_cov",
           info.get("cov_loss") is not None)
        ok("C9 口径标记随 info 出来（诊断/日志要靠它归因）",
           info.get("lex_token_mode") == "bpe" and info.get("lex_head") is True)


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def main():
    for fn in (test_lex_head_zero_init, test_lex_head_gradient_isolation,
               test_bpe_payload, test_bpe_no_truncation, test_lex_mean_unchanged,
               test_bridge_split, test_compose_split_tokens, test_hid_distill,
               test_ckpt_roundtrip):
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
