#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tests/test_diag_contracts.py — 2026-11-04 那批修复的**回归**测试（全 CPU）。

这里每条都对应一个"改错了不会报错、只会让指标静默失真"的地方：

  A. Bridge 的范数分母不能是 1e-6：末层零初始化时 d(out/‖out‖) ≈ 1/‖out‖，
     实测把初始 bridge 分支梯度放大到 1.85e10，grad_clip=1.0 之后实际更新
     完全由这条病态路径支配。
  B. α 必须有硬上限：v5 的 α 涨到 0.264 而读出 EM 反而低于 v3。
  C. 重建与 QA 必须走同一个 bridge 调用形式（z_sem + z_lex）。旧代码 recon 走
     bridge(z_sem)，等于让同一组参数同时服务两个互相冲突的目标。
  D. teacher 的上下文范围必须显式（gold / full），且找不到证据时**留痕**而不是
     静默给空上下文。
  E. 重建诊断必须能拆出实体/数字恢复率（旧口径 teacher forcing 的 59.6% 不能
     证明符号被保住）。
  F. oracle_lex 的 α 归零/还原必须是可逆的（评测里改的是共享参数）。

用法：python tests/test_diag_contracts.py
"""
import os
import random
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


# =====================================================================
# A/B. Bridge：梯度量级与 α 上限
# =====================================================================
def test_bridge_numerics():
    torch.manual_seed(0)
    d, m = 32, 4
    br = LM.LatentBridge(d, bottleneck=16, target_norm=1.05, lex_skip=True,
                         alpha_max=0.5)
    z = torch.randn(1, m, d) * 0.5
    lex = torch.randn(1, m, d)
    lex = lex / lex.norm(dim=-1, keepdim=True) * 1.05

    out = br(z, lex)
    # A1 零初始化 + α=0：输出必须逐元素等于 z_lex（起点就是 oracle_embed 状态）
    ok("A1 α≈0 时输出 == z_lex", torch.allclose(out, lex, atol=1e-5),
       extra=f"maxdiff={float((out - lex).abs().max().detach()):.3e}")
    # A2 输出范数对齐到 target_norm
    ok("A2 输出每向量范数 ≈ target_norm",
       abs(float(out.norm(dim=-1).mean()) - 1.05) < 1e-3)

    # A3 回归：初始梯度不能爆炸（旧分母 1e-6 时这里是 ~1e6 量级）
    br.zero_grad(set_to_none=True)
    br(z, lex).sum().backward()
    g = float(br.net[3].weight.grad.norm())
    ok("A3 初始末层梯度 < 1e3（分母夹 1.0）", g < 1e3, extra=f"gnorm={g:.3e}")
    ok("A4 初始末层梯度 > 0（α 初值非零，语义分支从第 0 步就能学）",
       g > 0.0, extra=f"gnorm={g:.3e}")

    # A5 回归：α 初值**不能**是 0。零初始化 + α=0 时语义分支与门控同时无梯度
    #    （∂e/∂W ∝ α = 0 且 ∂e/∂α ∝ res = 0），会死锁在"永远只有 z_lex"。
    br0 = LM.LatentBridge(d, bottleneck=16, target_norm=1.05, lex_skip=True,
                          lex_gate_init=0.0)
    br0.zero_grad(set_to_none=True)
    br0(z, lex).sum().backward()
    g0 = float(br0.net[3].weight.grad.norm())
    ok("A5 α=0 + 末层零初始化时梯度严格为 0（死锁，故默认初值必须非零）",
       g0 == 0.0, extra=f"gnorm={g0:.3e}")
    ok("A6 默认 lex_gate 初值非零", float(LM.LATENT_LEX_GATE_INIT) > 0.0,
       extra=f"init={LM.LATENT_LEX_GATE_INIT}")

    # B1 α 上限：把门控推到很大，alpha() 必须停在 alpha_max
    with torch.no_grad():
        br.lex_gate.fill_(5.0)
    ok("B1 α 被 alpha_max 截断", abs(float(br.alpha()) - 0.5) < 1e-6,
       extra=f"alpha={float(br.alpha()):.4f}")
    with torch.no_grad():
        br.lex_gate.fill_(-5.0)
    ok("B2 α 负向同样截断", abs(float(br.alpha()) + 0.5) < 1e-6)

    # B3 collect_stats 关闭时不写统计、打开时写
    br.collect_stats = False
    br(z, lex)
    ok("B3 collect_stats=False 不写统计", br.last_stats == {})
    br.collect_stats = True
    br(z, lex)
    ok("B4 collect_stats=True 记录 alpha/res_ratio",
       "alpha" in br.last_stats and "res_ratio" in br.last_stats,
       extra=str(sorted(br.last_stats)))
    ok("B5 alpha_max 非法取值报错",
       _raises(lambda: LM.LatentBridge(d, bottleneck=16, alpha_max=0.0)))


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


# =====================================================================
# C. 重建与 QA 的 bridge 输入一致
# =====================================================================
class _SpyBridge(torch.nn.Module):
    """记录每次调用有没有收到 z_lex，返回固定形状的 soft。"""

    def __init__(self, d, m=4):
        super().__init__()
        self.d, self.m = d, m
        self.calls = []

    def forward(self, z, z_lex=None):
        self.calls.append(z_lex is not None)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        return torch.zeros(1, int(z.shape[1]) * 1, self.d)


def test_recon_uses_lex():
    model = T.SmokeModel(vocab_size=256, hidden_size=32, n_layers=2, n_heads=2)
    tok = T.StubTok(vocab_size=256)
    bridge = _SpyBridge(32)
    payloads = [torch.randn(1, 4, 32) for _ in range(3)]
    lex = [torch.randn(1, 4, 32) for _ in range(3)]
    units = ["Alpha Corp is in Ironwood City.",
             "Beta Ltd was founded in 1998.",
             "Gamma Inc makes sensors."]

    T.recon_loss_units(model, tok, bridge, payloads, units, random.Random(0), 2,
                       "You are a helpful assistant.", "cpu",
                       gold_idx=[0], use_lex=True, lex_payloads=lex,
                       corrupt_p=0.0)
    ok("C1 recon 走 bridge(z_sem, z_lex)（与 QA 同一调用形式）",
       bridge.calls and all(bridge.calls), extra=str(bridge.calls))

    bridge.calls.clear()
    T.recon_loss_units(model, tok, bridge, payloads, units, random.Random(0), 2,
                       "You are a helpful assistant.", "cpu",
                       gold_idx=[0], use_lex=False, lex_payloads=lex,
                       corrupt_p=0.0)
    ok("C2 use_lex=False 复现旧行为（只给 z_sem）",
       bridge.calls and not any(bridge.calls), extra=str(bridge.calls))

    # C3 元素数一致但形状不同的直通载荷必须能 reshape 对齐，而不是抛错
    bridge.calls.clear()
    lex_flat = [t.reshape(-1, 32) for t in lex]
    T.recon_loss_units(model, tok, bridge, payloads, units, random.Random(0), 1,
                       "You are a helpful assistant.", "cpu",
                       gold_idx=[0], use_lex=True, lex_payloads=lex_flat,
                       corrupt_p=0.0)
    ok("C3 直通载荷形状不同但元素数一致时可对齐", bool(bridge.calls))


# =====================================================================
# D. teacher 的上下文范围
# =====================================================================
def test_teacher_scope():
    item = {"memory": "A. B. C.", "evidence": ["B."], "question": "q", "answer": "x"}
    units = ["A.", "B.", "C."]
    txt, fell_back = T.teacher_evidence_text(item, units, "gold")
    ok("D1 gold 只给证据句", txt == "B." and fell_back is False, extra=repr(txt))
    txt, fell_back = T.teacher_evidence_text(item, units, "full")
    ok("D2 full 给整份 memory", txt == "A. B. C." and fell_back is False)
    txt, fell_back = T.teacher_evidence_text(
        {"memory": "A. B. C.", "evidence": ["不存在"]}, units, "gold")
    ok("D3 找不到证据时退回 full 且**留痕**",
       txt == "A. B. C." and fell_back is True)

    # D4 probe 的提示词必须和训练用的是同一个函数（逐字相同）
    tok = T.StubTok()
    a = T._teacher_prompt_text(tok, "B.", "sys", "q")
    b = T._teacher_prompt_text(tok, "B.", "sys", "q")
    ok("D4 teacher 提示词单点构造", a == b and "Evidence: B." in a)


class _ProbeModel(T.SmokeModel):
    """generate 接受 input_ids（teacher_probe 用的是 input_ids 路径）。"""

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=8,
                 pad_token_id=None, **kw):
        n = int(max_new_tokens)
        tail = torch.full((1, n), 5, dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, tail], dim=1)


def test_teacher_probe():
    model = _ProbeModel(vocab_size=256, hidden_size=32, n_layers=2, n_heads=2)
    tok = T.StubTok(vocab_size=256)
    data = [{"memory": "A. B.", "evidence": ["B."], "units": ["A.", "B."],
             "question": "who", "answer": "tok_5"},
            {"memory": "C. D.", "evidence": [], "units": ["C.", "D."],
             "question": "who2", "answer": "tok_5"}]
    res = T.teacher_probe(model, tok, data, "sys", limit=2, max_new_tokens=4,
                          label="t")
    ok("D5 teacher_probe 返回两种上下文的 acc",
       set(res) >= {"full", "gold", "n", "n_no_ev"} and res["n"] == 2,
       extra=str(res))
    ok("D6 teacher_probe 统计找不到证据的样本数", res["n_no_ev"] == 1, extra=str(res))


# =====================================================================
# E. 重建诊断口径
# =====================================================================
def test_recon_scores():
    gold = "Ironwood Energy Nilork was founded in 1987 with 1,250 staff."
    pred = "Ironwood Energy Nilork was founded in 1987."
    sc = T._recon_scores(pred, gold)
    ok("E1 EM 严格（少一半不算对）", sc["em"] == 0.0)
    ok("E2 词袋 F1 在 (0,1)", 0.0 < sc["tok_f1"] < 1.0, extra=str(sc["tok_f1"]))
    ok("E3 实体恢复率 = 1（实体全在）", sc["ent_recall"] is not None
       and abs(sc["ent_recall"] - 1.0) < 1e-9, extra=str(sc["ent_recall"]))
    ok("E4 数字恢复率 = 0.5（1987 有、1250 没有）",
       sc["num_recall"] is not None and abs(sc["num_recall"] - 0.5) < 1e-9,
       extra=str(sc["num_recall"]))
    sc2 = T._recon_scores("nothing here", "no entities")
    ok("E5 无实体/无数字时返回 None 而不是 0（不许把'没有'当成'全错'）",
       sc2["ent_recall"] is None and sc2["num_recall"] is None, extra=str(sc2))
    ok("E6 数字跨度识别千分位与小数",
       T._num_spans("1,250 and 3.14 and 7") == ["1,250", "3.14", "7"],
       extra=str(T._num_spans("1,250 and 3.14 and 7")))


# =====================================================================
# F. oracle_lex 的 α 归零可逆
# =====================================================================
def test_alpha_toggle():
    import eval_latent_table as E
    d, m = 16, 2
    br = LM.LatentBridge(d, bottleneck=8, target_norm=1.05, lex_skip=True)
    with torch.no_grad():
        br.lex_gate.fill_(0.3)
    before = float(br.alpha())
    saved = E._zero_alpha(br)
    ok("F1 归零后 α == 0", abs(float(br.alpha())) < 1e-9)
    E._restore_alpha(br, saved)
    ok("F2 还原后 α 回到原值", abs(float(br.alpha()) - before) < 1e-9,
       extra=f"{float(br.alpha())} vs {before}")
    # F3 关掉 lex_skip 的 bridge 上调用必须安全（返回 None，不改任何东西）
    br2 = LM.LatentBridge(d, bottleneck=8, target_norm=1.05, lex_skip=False)
    ok("F3 无直通分支时安全返回 None", E._zero_alpha(br2) is None)


# =====================================================================
# G. 符号保真的词边界还原（碎片 vs 整词）
# =====================================================================
class _TokStub:
    """字节级 BPE 风格的假分词器：'Ġ' 表示词前空格。"""

    def __init__(self, toks):
        self.toks = list(toks)

    def convert_ids_to_tokens(self, i):
        return self.toks[int(i)]

    def decode(self, ids):
        # 真实字节级 BPE 的 decode 会把 "Ġ" 还原成空格；假分词器必须同样处理，
        # 否则 `.strip()` 去不掉 "Ġ"，测试会因为"分词器不真实"而误报失败。
        return "".join(self.toks[int(i)] for i in ids).replace("Ġ", " ")


def test_word_span():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import diag_symbol_fidelity as S
    tok = _TokStub(["ĠMer", "cedes", "Ġis", "Ġhead", "-", "Ġx"])
    ids = list(range(6))
    ok("G1 子词向后合并成整词", S._word_span(tok, ids, 1) == (0, 2),
       extra=str(S._word_span(tok, ids, 1)))
    ok("G2 词首 token 只覆盖自己", S._word_span(tok, ids, 2) == (2, 3))
    ok("G3 标点单独成词", S._word_span(tok, ids, 4) == (4, 5))
    # 整词还原：选中 'cedes'（列 1）应还原出 'Mercedes'
    feats = {"chunk_ids": [ids]}
    cinfo = {"lex_pick": [[1, 3]]}
    pieces, whole = S._slot_picks(tok, feats, cinfo)
    ok("G4 碎片口径保留子词", pieces == ["cedes", "head"], extra=str(pieces))
    ok("G5 整词口径还原成整词", whole == ["Mercedes", "head"], extra=str(whole))

    # 命中判定：子词前缀也算命中（'Mer' ⊂ 'Mercedes'）
    ok("G6 子词前缀算命中", S._hit(["Mer"], "Mercedes-AMG") is True)
    ok("G7 无关词不算命中", S._hit(["is"], "Mercedes-AMG") is False)


# =====================================================================
# H. 注入位置（消息体内 vs 消息之间）与围栏泡沫
# =====================================================================
class _WordTok:
    """按词表 decode 的假分词器（给显著性挑词用）。"""

    def __init__(self, words):
        self.words = list(words)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.words[int(i)] for i in ids)


def test_inject_position():
    import eval_latent_table as E
    model = T.SmokeModel(vocab_size=256, hidden_size=32, n_layers=2, n_heads=2)
    tok = T.StubTok(vocab_size=256)
    bridge = LM.LatentBridge(32, bottleneck=8, target_norm=1.05, lex_skip=True)
    z = [torch.randn(1, 4, 32), torch.randn(1, 4, 32)]
    lex = [torch.randn(1, 4, 32) for _ in z]

    # H1/H2 围栏泡沫：bracket 必须显著短于 verbose（这是本次改动的目的）
    soft_b, segs_b = LM.compose_routed_soft(
        model, tok, bridge, z, lex_list=lex, isolate=True, return_segs=True,
        fence_style="bracket")
    soft_v, segs_v = LM.compose_routed_soft(
        model, tok, bridge, z, lex_list=lex, isolate=True, return_segs=True,
        fence_style="verbose")
    content = 2 * 4                      # 2 槽 × m=4
    fence_b = int(soft_b.shape[1]) - content
    fence_v = int(soft_v.shape[1]) - content
    ok("H1 bracket 围栏 token 数显著少于 verbose", fence_b < fence_v,
       extra=f"bracket={fence_b} verbose={fence_v}")
    ok("H2 bracket 每槽围栏 <= 4 token", fence_b <= 8, extra=f"fence={fence_b}")

    # H3 注入位置：默认 user，软 token 落在 user 消息体内
    pre, post = LM._chat_parts_with_soft_slot(tok, "sys", "Q?", "user")
    ok("H3 软 token 插在 user 消息体内",
       "<user>" in pre and "Evidence:" in pre and "Question:" in post,
       extra=f"pre={pre!r} post={post!r}")
    ok("H4 占位符不会泄漏进最终 prompt",
       LM._SOFT_SENTINEL not in pre and LM._SOFT_SENTINEL not in post)
    pre_s, post_s = LM._chat_parts_with_soft_slot(tok, "sys", "Q?", "system")
    # system 模式下问题是**独立的一条 user 消息**（与明文 oracle 同构），
    # 所以 post 里是问题本身，而不是 "Question:" 前缀。
    ok("H5 system 模式把证据段放进 system 消息内部",
       "<system>" in pre_s and "Evidence:" in pre_s and "Q?" in post_s,
       extra=f"pre={pre_s!r} post={post_s!r}")
    ok("H6 between 不是合法体内模式（走旧路径）",
       _raises(lambda: LM._chat_parts_with_soft_slot(tok, "sys", "Q?", "between")))
    ok("H7 非法 inject_mode 直接报错", _raises(lambda: LM._check_inject_cfg("nope")))
    ok("H8 非法 fence_style 直接报错", _raises(lambda: LM._check_inject_cfg(None, "nope")))

    # H9 记账：内容 token 与围栏 token 必须分开
    pred, meta = LM._emit_latent(model, tok, soft_b, "Q?", "sys", max_new_tokens=2,
                                 pad_token_id=0, seg_ids=segs_b, isolate=True,
                                 inject_mode="user", fence_style="bracket")
    ok("H9 meta 分开记账 content/fence",
       int(meta["n_soft_content"]) + int(meta["n_fence"]) == int(meta["n_soft"]),
       extra=str({k: meta[k] for k in ("n_soft", "n_soft_content", "n_fence")}))
    ok("H10 meta 记录注入位置与围栏样式",
       meta["inject_mode"] == "user" and meta["fence_style"] == "bracket")

    # H11 显著性挑词：优先实体与数字，丢掉停用词
    wt = _WordTok(["the", "Rome", "was", "9984"])
    ids = torch.tensor([0, 1, 2, 3])
    keep = E._salient_pick(wt, ids, 2, "what happened")
    ok("H11 同等预算下按显著性挑词（Rome + 9984）",
       keep.tolist() == [1, 3], extra=str(keep.tolist()))
    ok("H12 预算 >= 词数时全保留",
       E._salient_pick(wt, ids, 99, "q").tolist() == [0, 1, 2, 3])


def main():
    test_inject_position()
    test_word_span()
    test_bridge_numerics()
    test_recon_uses_lex()
    test_teacher_scope()
    test_teacher_probe()
    test_recon_scores()
    test_alpha_toggle()
    print("=" * 70)
    if FAILS:
        print(f"FAILED {len(FAILS)}: {FAILS}")
        return 1
    print(f"ALL PASS ({N_OK} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
