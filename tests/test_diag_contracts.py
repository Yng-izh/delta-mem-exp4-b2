#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tests/test_diag_contracts.py — 2026-11-04 那批修复的**回归**测试（全 CPU）。

这里每条都对应一个"改错了不会报错、只会让指标静默失真"的地方：

  A. Bridge 的范数分母不能是 1e-6：末层零初始化时 d(out/‖out‖) ≈ 1/‖out‖，
     实测把初始 bridge 分支梯度放大到 1.85e10，grad_clip=1.0 之后实际更新
     完全由这条病态路径支配。
  B. α 必须有硬上限：旧权重的 α 涨到 0.264 而读出 EM 反而低于 v3。
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
# G. 选词结果的还原（词级 / token 级降级）
# =====================================================================
class _TokStub:
    """字节级 BPE 风格的假分词器：'Ġ' 表示词前空格。"""

    def __init__(self, toks):
        self.toks = list(toks)

    def convert_ids_to_tokens(self, i):
        return self.toks[int(i)]

    def decode(self, ids):
        # 真实字节级 BPE 的 decode 会把 "Ġ" 还原成空格；假分词器必须同样处理。
        return "".join(self.toks[int(i)] for i in ids).replace("Ġ", " ")


def test_slot_words():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import diag_symbol_fidelity as S
    tok = _TokStub(["ĠMer", "cedes", "Ġis", "Ġhead", "-", "Ġx"])
    ids = list(range(6))
    # 词结构：[Mer cedes] [is] [head] [-] [x]
    feats = {"chunk_ids": [ids], "chunk_word_ids": [[0, 0, 1, 2, 3, 4]]}
    words = S._slot_words(tok, feats, {"lex_word_pick": [[0, 2]]})
    ok("G1 词级还原出整词（不是子词）",
       words == ["Mercedes", "head"], extra=str(words))
    ok("G1b 没有 lex_word_pick 时直接报错（不再静默降级）",
       _raises(lambda: S._slot_words(tok, feats, {})))

    ok("G3 命中判定用内容词（子词拼回整词后命中）",
       S._hit(["Mercedes"], "Mercedes-AMG") is True)
    ok("G4 停用词不算命中", S._hit(["is"], "Mercedes-AMG") is False)


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


# =====================================================================
# I. 词边界分组 / 词级选词 / 覆盖率口径
# =====================================================================
def test_word_grouping():
    # 用户给的例子：Tosca was filmed in Rome.
    texts = ["T", "os", "ca", " was", " filmed", " in", " Rome", "."]
    # 真实偏移**包含词前空格**：" was" 的 span 是 (5,10)，起点正好等于 "ca" 的终点。
    offs = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 17), (17, 20), (20, 25), (25, 26)]
    wid = LM._word_ids_from_offsets(offs, texts)
    ok("I1 偏移分组：T+os+ca 属于同一个词", wid[:3] == [0, 0, 0], extra=str(wid))
    ok("I2 每个词一个编号，标点单独成词",
       wid == [0, 0, 0, 1, 2, 3, 4, 5], extra=str(wid))
    # 回退路径（无 offsets）对同一句话给出同样的分组
    wid2 = LM._word_ids_from_texts(texts)
    ok("I3 前缀回退路径与偏移路径一致", wid2 == wid, extra=str(wid2))
    # 特殊 token / padding 记为 -1
    wid3 = LM._word_ids_from_offsets([(0, 0), (0, 1)], ["<s>", "Hi"])
    ok("I4 特殊 token 记 -1", wid3 == [-1, 0], extra=str(wid3))

    # 词级降权：标点 0.1 / 停用词 0.35 / 内容词 1.0
    wt = _WordTok(["Tosca", "was", "in", "Rome", "."])
    pen = LM._word_penalty_for(wt, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    ok("I5 纯标点降权为 0（不可选）", pen[4] == 0.0, extra=str(pen))
    ok("I6 停用词降权 0.35", abs(pen[1] - 0.35) < 1e-9 and abs(pen[2] - 0.35) < 1e-9,
       extra=str(pen))
    ok("I7 内容词不降权", pen[0] == 1.0 and pen[3] == 1.0, extra=str(pen))


def test_word_level_selection():
    torch.manual_seed(0)
    comp = LM.LatentCompressor(16, n_slots=4, inner_dim=16, num_heads=2, lex_skip=True)
    B, L, d = 2, 8, 16
    h0 = torch.randn(B, L, d)
    h18 = torch.randn(B, L, d)
    mask = torch.ones(B, L, dtype=torch.long)
    wid = torch.tensor([[0, 0, 1, 2, 3, 4, 5, 6]] * B)   # 第 0 个词有 2 个 token
    pen = torch.ones(B, 7)
    pen[:, 0] = 0.1                                      # 把第 0 个词降权
    z, zlex, info = comp.forward_split(h0, h18, mask, word_ids=wid, word_penalty=pen)
    ok("I8 词级选择仍是 m=4 个向量", tuple(zlex.shape) == (B, 4, d),
       extra=str(tuple(zlex.shape)))
    ok("I9 info 记录词数与选中的词", info["n_words"] == 7
       and len(info["lex_word_pick"][0]) == 4,
       extra=f"{info.get('n_words')}/{info.get('lex_word_pick')}")
    ok("I10 每次选的是**不同的词**（不重复占槽）",
       all(len(set(row)) == len(row) for row in info["lex_word_pick"]),
       extra=str(info["lex_word_pick"]))
    ok("I11 被降权的词不会优先被选", all(row[0] != 0 for row in info["lex_word_pick"]),
       extra=str(info["lex_word_pick"]))
    # 不传词边界 -> 直接报错（不再静默退回 token 级）
    ok("I13 缺 word_ids 时直接报错（不允许静默降级）",
       _raises(lambda: comp.forward_split(h0, h18, mask)))


def test_coverage_metrics():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import diag_symbol_fidelity as S
    ok("I14 标点/停用词算无效选择",
       S._is_invalid_word(".") and S._is_invalid_word("the")
       and not S._is_invalid_word("Rome"))
    words = ["Rome", "9,984"]
    ok("I15 完整答案覆盖：答案两个词都在 -> True",
       S._full_answer_covered(words, "Rome 9,984") is True)
    ok("I16 只覆盖一半 -> False（只命中一个子词不算成功）",
       S._full_answer_covered(["Rome"], "Rome 9,984") is False)
    ok("I17 数字整体保留才算覆盖",
       S._numbers_covered(["9,984"], "population was 9,984") is True)
    ok("I18 数字被拆开 -> False",
       S._numbers_covered(["9"], "population was 9,984") is False)
    ok("I19 无数字时返回 None（不是 False）",
       S._numbers_covered(["Rome"], "in Rome") is None)


def test_numeric_and_stopwords():
    # 数字链：9 , 984 / 3 . 14 必须合成一个词
    texts = ["9", ",", "984"]
    offs = [(0, 1), (1, 2), (2, 5)]
    ok("I20 千分位数字合成一个词",
       LM._word_ids_from_offsets(offs, texts) == [0, 0, 0],
       extra=str(LM._word_ids_from_offsets(offs, texts)))
    texts2 = ["3", ".", "14"]
    offs2 = [(0, 1), (1, 2), (2, 4)]
    ok("I21 小数合成一个词",
       LM._word_ids_from_offsets(offs2, texts2) == [0, 0, 0])
    # 句末句点仍要单独成词
    texts3 = ["Rome", "."]
    offs3 = [(0, 4), (4, 5)]
    ok("I22 句末句点不并入前一个词",
       LM._word_ids_from_offsets(offs3, texts3) == [0, 1])
    # 句末逗号不能被当成千分位（真实数据里 "1868, was" 曾被粘成 "1868,"）
    ok("I24 句末逗号与数字分开",
       LM._word_ids_from_offsets([(0, 4), (4, 5), (5, 9)], ["1868", ",", " was"])
       == [0, 1, 2],
       extra=str(LM._word_ids_from_offsets([(0, 4), (4, 5), (5, 9)],
                                           ["1868", ",", " was"])))
    # 空格/未知字符 token（解码为空但占字符宽）必须断开前后两个词
    ok("I25 空 token 断开前后词",
       LM._word_ids_from_offsets([(0, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)],
                                 ["before", "", "1", "8", "6", "5"])
       == [0, -1, 1, 1, 1, 1],
       extra=str(LM._word_ids_from_offsets(
           [(0, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)],
           ["before", "", "1", "8", "6", "5"])))
    ok("I26 破折号后的数字不与破折号同词",
       LM._word_ids_from_offsets([(0, 4), (4, 5), (5, 6), (6, 8)],
                                 ["1830", "\u2013", "", "22"]) == [0, 1, -1, 2],
       extra=str(LM._word_ids_from_offsets([(0, 4), (4, 5), (5, 6), (6, 8)],
                                           ["1830", "\u2013", "", "22"])))
    # 否定/时间/比较词不能被当停用词降权
    bad = [w for w in ("not", "no", "before", "after", "than")
           if w in LM._LATENT_STOPWORDS]
    ok("I23 not/no/before/after/than 不在停用词表", not bad, extra=str(bad))


def test_per_slot_ranking():
    """每个槽用自己的注意力偏好选词，而不是四槽共用一份全局排名。

    构造：槽 0 偏好词 0（0.5）胜过词 1（0.4）；槽 1 偏好词 1（0.9）远胜词 0（0.1）。
    逐槽选 -> [0, 1, ...]；旧写法 `mass.max(dim=1)` 合并后词 1（0.9）会排第一，
    于是槽 0 也被迫先选词 1 -> [1, 0, ...]。两者结果不同，因此这条能钉住语义。
    """
    comp = LM.LatentCompressor(16, n_slots=4, inner_dim=16, num_heads=2, lex_skip=True)
    B, L, d = 1, 6, 16
    h0 = torch.randn(B, L, d)
    mask = torch.ones(B, L, dtype=torch.long)
    wid = torch.tensor([[0, 0, 1, 1, 2, 3]])
    w = torch.zeros(B, 4, L)
    w[0, 0] = torch.tensor([0.5, 0.5, 0.4, 0.4, 0.0, 0.0])
    w[0, 1] = torch.tensor([0.1, 0.1, 0.9, 0.9, 0.0, 0.0])
    _z, pick, n_words, _cin = comp._lex_by_word(h0, w, mask, wid, None)
    ok("I24 每槽按自己的偏好选词（不是全局 Top-k）",
       pick[0].tolist()[:2] == [0, 1] and n_words == 4,
       extra=str(pick[0].tolist()))


def test_student_inputs_inject_mode():
    """训练前向必须和评测走同一个注入位置。"""
    model = T.SmokeModel(vocab_size=256, hidden_size=32, n_layers=2, n_heads=2)
    tok = T.StubTok(vocab_size=256)
    soft = torch.zeros(1, 3, 32)
    e_u, l_u, n_u = T.build_student_inputs(
        model, tok, soft, "sys", "Q?", "A", inject_mode="user")
    e_b, l_b, n_b = T.build_student_inputs(
        model, tok, soft, "sys", "Q?", "A", inject_mode="between")
    ok("I25 user 与 between 两种布局长度不同（确实换了位置）",
       e_u.shape[1] != e_b.shape[1], extra=f"{e_u.shape} vs {e_b.shape}")
    ok("I26 两种布局的答案标签一致（只换注入位置，不换监督）",
       n_u == n_b and torch.equal(l_u[0, -n_u:], l_b[0, -n_b:]))


def test_train_eval_layout_identity():
    """训练前向的 prompt 必须与评测侧**逐位相同**（不只是"调了同一个函数"）。

    做法：给 `_emit_latent` 打一个捕获钩子，拿到它真正喂给模型的 inputs_embeds，
    再和 `build_student_inputs` 造出来的前缀逐元素比较。三种注入位置都要对。
    """
    class _Cap(T.SmokeModel):
        def generate(self, inputs_embeds=None, attention_mask=None, max_new_tokens=8,
                     pad_token_id=None, **kw):
            self.cap = inputs_embeds.detach().clone()
            return super().generate(inputs_embeds=inputs_embeds,
                                    attention_mask=attention_mask,
                                    max_new_tokens=max_new_tokens,
                                    pad_token_id=pad_token_id)

    for mode in ("user", "system", "between"):
        model = _Cap(vocab_size=256, hidden_size=32, n_layers=2, n_heads=2)
        tok = T.StubTok(vocab_size=256)
        soft = torch.randn(1, 4, 32)
        LM._emit_latent(model, tok, soft, "Q?", "sys", max_new_tokens=1,
                        pad_token_id=0, inject_mode=mode)
        cap = model.cap[0]
        emb, _lab, _n = T.build_student_inputs(
            model, tok, soft, "sys", "Q?", "ANS", inject_mode=mode)
        ok(f"I27 训练/评测 prompt 逐位相同（{mode}）",
           torch.equal(emb[0, :cap.shape[0]], cap),
           extra=f"n_in={cap.shape[0]} n_train={emb.shape[1]}")


def test_punct_hard_mask():
    """纯标点必须**不可选**，而不是"分数低但仍可能被选"。"""
    texts = ["[Smyrna", "]", "is", "a", "city", "northwest", "of", "Atlanta", "."]
    tok = _WordTok(texts)
    ids = list(range(len(texts)))
    wids = list(range(len(texts)))
    pen = LM._word_penalty_for(tok, ids, wids)
    grp = LM._word_groups(texts)
    ok("J1 纯标点降权为 0", pen[1] == 0.0 and pen[8] == 0.0, extra=str(pen))
    ok("J2 带标点的实词仍可选（[Smyrna / Atlanta）",
       pen[0] == 1.0 and pen[7] == 1.0, extra=str(pen))
    ok("J3 连字符姓名不被当标点", LM._word_penalty_for(
        _WordTok(["Robert", "Gascoyne-Cecil"]), [0, 1], [0, 1])[1] == 1.0)

    comp = LM.LatentCompressor(16, n_slots=4, inner_dim=16, num_heads=2, lex_skip=True)
    L, d = len(texts), 16
    h0 = torch.randn(1, L, d)
    mask = torch.ones(1, L, dtype=torch.long)
    w = torch.zeros(1, 4, L)
    # 把注意力**全压在标点**上：如果没有硬屏蔽，标点必然被选中
    w[0, :, 1] = 10.0
    w[0, :, 8] = 10.0
    w[0, :, 0] = 1.0
    w[0, :, 4] = 0.9
    _z, pick, _n, _cin = comp._lex_by_word(
        h0, w, mask, torch.tensor([wids]), torch.tensor([pen]), torch.tensor([grp]))
    picked = [texts[i] for i in pick[0].tolist() if i >= 0]
    ok("J4 标点即使注意力最高也不会被选中",
       not any(t in ("]", ".") for t in picked), extra=str(picked))

    # 去重：标题里的 [Smyrna 与正文 Smyrna 只占一个槽
    texts2 = ["[Smyrna", "]", "Smyrna", "city"]
    pen2 = LM._word_penalty_for(tok, [0, 1, 2, 3], [0, 1, 2, 3])
    grp2 = LM._word_groups(texts2)
    ok("J5 标题与正文的同名词分到同一组", grp2[0] == grp2[2] != -1, extra=str(grp2))
    w2 = torch.zeros(1, 4, len(texts2))
    w2[0, :, 0] = 5.0
    w2[0, :, 2] = 4.0
    w2[0, :, 3] = 1.0
    _z2, pick2, _n2, _c2 = comp._lex_by_word(
        h0[:, :4], w2, mask[:, :4], torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([pen2]), torch.tensor([grp2]))
    got2 = [i for i in pick2[0].tolist() if i >= 0]
    ok("J6 同名词只占一个槽（不重复占坑）",
       sum(1 for i in got2 if i in (0, 2)) == 1, extra=str(pick2[0].tolist()))

    # 有效词不足 4 个：空位记 -1，不拿无效词凑数
    texts3 = ["city", "."]
    pen3 = LM._word_penalty_for(tok, [0, 1], [0, 1])
    grp3 = LM._word_groups(texts3)
    w3 = torch.zeros(1, 4, 2)
    w3[0, :, 0] = 1.0
    w3[0, :, 1] = 9.0
    _z3, pick3, _n3, _c3 = comp._lex_by_word(
        h0[:, :2], w3, mask[:, :2], torch.tensor([[0, 1]]),
        torch.tensor([pen3]), torch.tensor([grp3]))
    ok("J7 有效词不足时空位记 -1（不拿标点凑数）",
       pick3[0].tolist() == [0, -1, -1, -1], extra=str(pick3[0].tolist()))

    # 数字必须作为完整候选
    texts4 = ["population", "was", "9,984"]
    pen4 = LM._word_penalty_for(tok, [0, 1, 2], [0, 1, 2])
    grp4 = LM._word_groups(texts4)
    w4 = torch.zeros(1, 4, 3)
    w4[0, :, 2] = 1.0
    _z4, pick4, _n4, _c4 = comp._lex_by_word(
        h0[:, :3], w4, mask[:, :3], torch.tensor([[0, 1, 2]]),
        torch.tensor([pen4]), torch.tensor([grp4]))
    ok("J8 9,984 作为完整数字候选被选中", pick4[0, 0].item() == 2,
       extra=str(pick4[0].tolist()))


# =====================================================================
# K. L_cov：重要事实词覆盖监督
# =====================================================================
def test_cov_weights():
    """词重要性标签只能由**记忆原文**生成（Query-Agnostic），且四类词分开。"""
    texts = ["Smyrna", "is", "a", "city", "founded", "in", "1830", ".", "9,984"]
    tok = _WordTok(texts)
    ids = list(range(len(texts)))
    wgt = LM._word_weight_for(tok, ids, ids)
    ok("K1 数字/年份权重 3.0", wgt[6] == 3.0 and wgt[8] == 3.0, extra=str(wgt))
    ok("K2 专有名词权重 2.0", wgt[0] == 2.0, extra=str(wgt))
    ok("K3 普通内容词权重 0.5", wgt[3] == 0.5 and wgt[4] == 0.5, extra=str(wgt))
    ok("K4 停用词/纯标点权重 0（不进监督目标）",
       wgt[1] == 0.0 and wgt[2] == 0.0 and wgt[5] == 0.0 and wgt[7] == 0.0,
       extra=str(wgt))
    # 同一条规则必须对"干扰事实"生效：换个实体名、换个数字，权重结构不变
    t2 = ["Atlanta", "is", "a", "city", "with", "22", "districts"]
    w2 = LM._word_weight_for(_WordTok(t2), list(range(len(t2))), list(range(len(t2))))
    ok("K5 干扰事实走同一套规则（无金标泄漏）",
       w2[0] == 2.0 and w2[5] == 3.0 and w2[1] == 0.0, extra=str(w2))

    # 序数 / 年代：真实数据里 95 次全部掉成"内容词 0.5"，是最该被存下来的时间信息
    t3 = ["7th", "1980s", "22nd", "3rd", "1950s", "R509", "F1", "analysis"]
    w3 = LM._word_weight_for(_WordTok(t3), list(range(len(t3))), list(range(len(t3))))
    ok("K5a 序数与年代算数字（3.0）",
       w3[:5] == [3.0] * 5, extra=str(w3))
    ok("K5b 字母打头的编号仍是实体（R509/F1），不被误判成数字",
       w3[5] == 2.0 and w3[6] == 2.0, extra=str(w3))
    ok("K5c 后缀削除不能误伤普通词（analysis 是内容词）",
       w3[7] == 0.5, extra=str(w3))
    ok("K5d 后缀只在数字后面才削（Analysis/Status 不是数字）",
       not LM._is_numeric_word("Analysis") and not LM._is_numeric_word("Status")
       and LM._is_numeric_word("1980s") and LM._is_numeric_word("7th"))


def test_cov_loss_formula():
    """L_cov 的公式语义：完美覆盖 -> 0；全压在同一个词上 -> 大惩罚。"""
    eps = 1e-6
    W, m = 4, 4
    # 4 个槽各自 one-hot 命中一个词 -> 全覆盖
    mass = torch.zeros(1, m, W)
    for r in range(m):
        mass[0, r, r] = 1.0
    elig = torch.ones(1, W, dtype=torch.bool)
    wgt = torch.tensor([[3.0, 2.0, 2.0, 0.5]])
    loss, st = LM.word_coverage_loss(mass, elig, wgt)
    ok("K6 完美覆盖时 L_cov≈0", loss is not None and float(loss) < 1e-5,
       extra=str(None if loss is None else float(loss)))
    ok("K7 cov_pcov 在完美覆盖时≈1", abs(st["cov_pcov"] - 1.0) < 1e-5, extra=str(st))

    # 4 个槽全部压在词 0 上：其余 3 个目标覆盖概率为 0 -> 惩罚接近 -log(eps)
    mass2 = torch.zeros(1, m, W)
    mass2[0, :, 0] = 1.0
    loss2, st2 = LM.word_coverage_loss(mass2, elig, wgt)
    ok("K8 全压在同一个词上时 L_cov 很大", float(loss2) > 1.0, extra=str(float(loss2)))
    ok("K9 未被覆盖的词把 cov_pcov 拉低", st2["cov_pcov"] < 0.6, extra=str(st2))

    # 均匀摊开（每个槽对每个词 0.25）：不是"覆盖"，pcov 只有 1-0.75^4
    mass3 = torch.full((1, m, W), 1.0 / W)
    loss3, st3 = LM.word_coverage_loss(mass3, elig, wgt)
    ok("K10 均匀摊开不等于覆盖（pcov≈0.68 而不是 1）",
       0.5 < st3["cov_pcov"] < 0.8, extra=str(st3["cov_pcov"]))
    ok("K11 均匀摊开的损失明显大于完美覆盖",
       float(loss3) > float(loss) + 0.1, extra=f"{float(loss3)} vs {float(loss)}")

    # 全零权重（全是停用词/标点）-> 没有监督目标
    loss4, _st4 = LM.word_coverage_loss(mass, elig, torch.zeros(1, W))
    ok("K12 没有重要词时返回 None（不产生虚假监督）", loss4 is None)


def test_cov_capacity_and_dedup():
    """4 个槽装不下所有重要词时必须**筛目标**，同表面键必须去重。"""
    W, m = 10, 4
    mass = torch.zeros(1, m, W)
    mass[0, :, 0] = 1.0
    elig = torch.ones(1, W, dtype=torch.bool)
    wgt = torch.tensor([[3.0] * 10])                 # 10 个数字，槽只有 4 个
    _loss, st = LM.word_coverage_loss(mass, elig, wgt)
    ok("K13 重要词超过容量时只监督 m 个（不强迫全覆盖）",
       abs(st["n_cov_tgt"] - 4.0) < 1e-6, extra=str(st))
    _loss2, st2 = LM.word_coverage_loss(mass, elig, wgt, max_targets=2)
    ok("K14 max_targets 可显式调小", abs(st2["n_cov_tgt"] - 2.0) < 1e-6, extra=str(st2))

    # 同表面键：`[Smyrna` 与 `Smyrna` 分到同一组，只能算**一个**监督目标。
    # 构造：词 0 独立（槽 0 命中），词 1 与词 2 同组（槽 1 只命中词 1）。
    # 不做去重时 3 个词都是目标，词 2 的覆盖概率是 0 -> 损失被拉到 -log(eps)；
    # 去重后目标只剩 {0,1}，损失 0。两个数差一个数量级，能钉住"到底去没去重"。
    W3, m3 = 3, 2
    mass3 = torch.zeros(1, m3, W3)
    mass3[0, 0, 0] = 1.0
    mass3[0, 1, 1] = 1.0
    elig3 = torch.ones(1, W3, dtype=torch.bool)
    wgt3 = torch.tensor([[2.0, 2.0, 2.0]])
    grp3 = torch.tensor([[-1, 0, 0]])
    l_no, st_no = LM.word_coverage_loss(mass3, elig3, wgt3, max_targets=3)
    l_dd, st_dd = LM.word_coverage_loss(mass3, elig3, wgt3, word_group=grp3,
                                        max_targets=3)
    ok("K15 同组同名词只占一个监督目标（3 -> 2）",
       abs(st_no["n_cov_tgt"] - 3.0) < 1e-6 and abs(st_dd["n_cov_tgt"] - 2.0) < 1e-6,
       extra=f"no_dedup={st_no['n_cov_tgt']} dedup={st_dd['n_cov_tgt']}")
    ok("K16 去重后不再有结构性无法覆盖的目标（损失显著下降）",
       float(l_dd) < 1e-5 and float(l_no) > 1.0,
       extra=f"dedup={float(l_dd):.4f} no_dedup={float(l_no):.4f}")


def test_cov_gradient_path():
    """L_cov 必须**保留计算图**并把梯度送回选择器（argmax 路径给不了梯度）。"""
    torch.manual_seed(0)
    comp = LM.LatentCompressor(16, n_slots=4, inner_dim=16, num_heads=2, lex_skip=True)
    B, L, d = 1, 8, 16
    h0 = torch.randn(B, L, d)
    h18 = torch.randn(B, L, d)
    mask = torch.ones(B, L, dtype=torch.long)
    wid = torch.arange(L).view(1, L)
    pen = torch.ones(B, L)
    wgt = torch.tensor([[3.0, 2.0, 2.0, 0.5, 0.5, 0.0, 0.0, 0.0]])
    _z, _zlex, info = comp.forward_split(
        h0, h18, mask, word_ids=wid, word_penalty=pen, word_weight=wgt,
        want_cov=True)
    cl = info.get("cov_loss")
    ok("K17 want_cov=True 时 info 里带 cov_loss", cl is not None, extra=str(info.keys()))
    ok("K18 cov_loss 是**可微**张量（保留了计算图）",
       cl is not None and cl.requires_grad)
    if cl is not None:
        comp.zero_grad(set_to_none=True)
        cl.backward()
        qg = comp.queries.grad
        ok("K19 梯度回到 queries（选择器的注意力参数）",
           qg is not None and float(qg.abs().sum()) > 0.0,
           extra=str(None if qg is None else float(qg.abs().sum())))
        ok("K20 梯度也回到 input_proj",
           comp.input_proj.weight.grad is not None
           and float(comp.input_proj.weight.grad.abs().sum()) > 0.0)
    # want_cov=False 时 info 里必须是 None（推理路径不付这份开销）
    _z2, _zl2, i2 = comp.forward_split(h0, h18, mask, word_ids=wid, word_penalty=pen,
                                       word_weight=wgt)
    ok("K21 want_cov=False 时 cov_loss 为 None", i2.get("cov_loss") is None)


def test_cov_plumbing():
    """--pack 路径与 assemble_loss 都不能漏掉 L_cov。"""
    qa = torch.tensor(2.0)
    cov = torch.tensor(4.0)
    base = T.assemble_loss(qa, None, None, None, 1.0, 1.0, 0.0)
    with_cov = T.assemble_loss(qa, None, None, None, 1.0, 1.0, 0.0,
                               loss_cov=cov, cov_weight=0.1)
    ok("K22 assemble_loss 加上 cov_weight*L_cov",
       abs(float(with_cov) - (float(base) + 0.4)) < 1e-6,
       extra=f"{float(with_cov)} vs {float(base)}")
    off = T.assemble_loss(qa, None, None, None, 1.0, 1.0, 0.0,
                          loss_cov=cov, cov_weight=0.0)
    ok("K23 cov_weight=0 时与旧行为逐位一致", float(off) == float(base))
    # recon_stage（纯重建阶段）也必须保留 cov：它监督的是压缩器存了什么
    rs = T.assemble_loss(qa, None, None, torch.tensor(1.0), 1.0, 1.0, 0.5,
                         recon_stage=True, loss_cov=cov, cov_weight=0.1)
    ok("K24 recon_stage 下 L_cov 仍然生效",
       abs(float(rs) - (1.0 + 0.4)) < 1e-6, extra=str(float(rs)))

    # _pad_word_tensors 必须把词重要性一起填出来，缺位填 0（不监督）
    wid, pen, grp, wgt = LM._pad_word_tensors(
        [[0, 1], [0, 1, 2]], [[1.0, 1.0], [1.0, 0.0, 1.0]],
        [[0, 1], [0, 1, 2]], [[3.0, 2.0], [0.5, 0.0, 2.0]])
    ok("K25 _pad_word_tensors 返回 4 个张量且形状对齐",
       tuple(wgt.shape) == (2, 3) and float(wgt[0, 2]) == 0.0,
       extra=str(wgt.tolist()))
    ok("K26 词重要性逐块填入正确位置",
       float(wgt[1, 0]) == 0.5 and float(wgt[1, 2]) == 2.0, extra=str(wgt.tolist()))


def test_resume_state():
    """断点续训必须把**两条独立随机流**也存下来，否则续训跑的是另一条轨迹。"""
    rng_slots = random.Random(11)
    rng_recon = random.Random(22)
    for _ in range(3):
        rng_slots.random()
        rng_recon.random()
    st = T._train_state(None, 17, 2, 41,
                        extra_rng={"rng_slots": rng_slots.getstate(),
                                   "rng_recon": rng_recon.getstate()})
    ok("K27 train_state 记录 step/epoch/pos", (st["step"], st["epoch"], st["pos"])
       == (17, 2, 41), extra=f"{st['step']}/{st['epoch']}/{st['pos']}")
    ok("K28 train_state 带两条独立随机流", set((st.get("rng_extra") or {}).keys())
       == {"rng_slots", "rng_recon"}, extra=str(list(st.keys())))
    # 恢复后必须与"不中断"的抽样序列逐位一致
    want = [rng_slots.random() for _ in range(4)]
    r2 = random.Random(999)
    r2.setstate(st["rng_extra"]["rng_slots"])
    ok("K29 恢复 rng_slots 后续抽样与不中断那次完全相同",
       [r2.random() for _ in range(4)] == want)
    ok("K30 不传 extra_rng 时兼容旧签名（不写 rng_extra 键）",
       "rng_extra" not in T._train_state(None, 0, 0, -1))


def main():
    test_resume_state()
    test_cov_plumbing()
    test_cov_gradient_path()
    test_cov_capacity_and_dedup()
    test_cov_loss_formula()
    test_cov_weights()
    test_punct_hard_mask()
    test_train_eval_layout_identity()
    test_numeric_and_stopwords()
    test_per_slot_ranking()
    test_student_inputs_inject_mode()
    test_word_grouping()
    test_word_level_selection()
    test_coverage_metrics()
    test_inject_position()
    test_slot_words()
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
