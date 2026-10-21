#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tests/test_latent_readout.py — 潜层读出的**机制性**验证（CPU、无模型、无 GPU）

这里验证的不是"分数变好了"（那需要真实训练 + 真实跑分），而是方案一那几条
**一旦做错就会静默失效**的机制：

  A. 读出接口不含明文：generate_latent_answer 的签名里没有任何能塞回证据块的参数；
     实际送进 tokenizer 的文本里也没有证据原文（不是"我没传"，而是"不可能传"）。
  B. 软 token 的位置 / 掩码 / 内容：插在 system 段之后，mask 全 1，
     soft 段逐元素等于 bridge(Z)，长度 = n_sys + K*m + n_query。
  C. 梯度真的到得了 Compressor / Bridge：冻结的是权重，不是梯度路径；
     存储路径默认 detach，训练路径必须 detach=False。
  D. 长单元**再分块而不截断**：父槽位 ID 一致、span 覆盖全部 token、
     sum(n_tokens) 等于原 token 数（截断会静默丢答案）。
  E. 载荷库语义：重复 ID / 缺失 ID 报错、版本号 +1、淘汰后 shift_index 重编号、
     epi 与 state 命名空间不串味。
  F. CR 路径预算：整条路径保留，装不下的整条跳过，绝不出现半条路径。
  G. 配置指纹：不同 m / 不同 checkpoint 必须不同 config_id（否则消融互相覆盖）。
  H. 数值安全：全 padding 输入不得产生 NaN。
  I. checkpoint 往返：save -> load 一致；缺失文件必须报错（禁止随机初始化顶上）。

用法：python tests/test_latent_readout.py
"""
import inspect
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import torch
import torch.nn as nn

import latent_memory as LM

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
# 桩件：字符/词级 tokenizer + 极小模型（只提供 latent_memory 需要的接口）
# =====================================================================
class StubTok:
    """词级 tokenizer。记录所有被送进来的文本，用于"没有明文进读出通路"的断言。"""

    def __init__(self, vocab_size=512):
        self.vocab_size = int(vocab_size)
        self.eos_token_id = 1
        self.pad_token_id = 0
        self.texts = []
        self._w2id = {}
        self._id2w = {}
        self._next = 8

    def _wid(self, w):
        if w not in self._w2id:
            self._w2id[w] = self._next % (self.vocab_size - 2) + 2
            self._id2w[self._w2id[w]] = w
            self._next += 1
        return self._w2id[w]

    def __call__(self, texts, return_tensors=None, add_special_tokens=True,
                 padding=False, truncation=False, max_length=None):
        if isinstance(texts, str):
            texts = [texts]
        self.texts.extend([str(t) for t in texts])
        seqs = [[self._wid(w) for w in str(t).split()] or [2] for t in texts]
        if truncation and max_length:
            seqs = [s[:int(max_length)] for s in seqs]
        L = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), L), dtype=torch.long)
        am = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            am[i, :len(s)] = 1
        return {"input_ids": ids, "attention_mask": am}

    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=False):
        out = ""
        for m in msgs:
            out += f"<|{m['role']}|> {m['content']} </|{m['role']}|> "
        if add_generation_prompt:
            out += "<|assistant|> "
        return out

    def decode(self, ids, skip_special_tokens=True):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return " ".join(self._id2w.get(int(i), f"<{int(i)}>") for i in ids)


class _Cfg:
    hidden_size = 32
    vocab_size = 512
    num_hidden_layers = 3
    num_key_value_heads = 4
    num_attention_heads = 4


class _Out:
    def __init__(self, logits, hidden_states=None):
        self.logits = logits
        if hidden_states is not None:
            self.hidden_states = hidden_states


class StubModel(nn.Module):
    """return_mode='new' 只返回新 token；'full' 返回 prompt+新 token（两种真实实现都有）。"""

    def __init__(self, return_mode="new", hidden=32):
        super().__init__()
        self.config = _Cfg()
        self.config.hidden_size = int(hidden)
        self.emb = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.head = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.return_mode = return_mode
        self.last_embeds = None
        self.last_mask = None
        self.forward_calls = 0

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                output_hidden_states=False, use_cache=False, **kw):
        self.forward_calls += 1
        x = inputs_embeds if inputs_embeds is not None else self.emb(input_ids)
        # 跨位置混合：纯点式 head 会让"前面的软 token 影响答案位置"的梯度**恒为 0**，
        # C 组（梯度是否到得了 Compressor/Bridge）就变成恒假式、测不出真问题。
        # 这里加一个跨位置均值项，模拟真实 attention 的信息混合（不改变形状与 mask 语义）。
        x = x + 0.1 * x.mean(dim=1, keepdim=True)
        hs = [x] * (self.config.num_hidden_layers + 1) if output_hidden_states else None
        return _Out(self.head(x), hs)

    def generate(self, inputs_embeds=None, attention_mask=None, max_new_tokens=8, **kw):
        self.last_embeds = inputs_embeds
        self.last_mask = attention_mask
        new = torch.full((1, int(max_new_tokens)), 7, dtype=torch.long)
        if self.return_mode == "full":
            pre = torch.full((1, inputs_embeds.shape[1]), 5, dtype=torch.long)
            return torch.cat([pre, new], dim=1)
        return new


def make_modules(hidden=32, m=4, inner=64, heads=4, bottleneck=16):
    comp = LM.LatentCompressor(hidden, n_slots=m, inner_dim=inner, num_heads=heads)
    bridge = LM.LatentBridge(hidden, bottleneck=bottleneck)
    comp.eval()
    bridge.eval()
    return comp, bridge


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


# =====================================================================
# A. 读出接口不含明文
# =====================================================================
def test_no_plaintext_interface():
    params = set(inspect.signature(LM.generate_latent_answer).parameters)
    ok("A1 签名参数集合", params == {"model", "tokenizer", "bridge", "payloads",
                                     "question", "system_prompt", "max_new_tokens",
                                     "pad_token_id"}, extra=str(sorted(params)))
    for bad in ("block", "evidence", "evidence_text", "memory", "context", "facts",
                "passages", "documents"):
        ok(f"A2 签名不含 {bad}", bad not in params)

    tok, model = StubTok(), freeze(StubModel())
    comp, bridge = make_modules(model.config.hidden_size)
    enc = LM.LatentEncoder(model, tok, comp, layer=2, max_unit_tokens=8)
    EVID = "ZQXKVV"
    payloads, _info = enc.encode([f"Alice works at {EVID}."], parent_ids=["cr:0"])
    n_before = len(tok.texts)
    pred, meta = LM.generate_latent_answer(
        model=model, tokenizer=tok, bridge=bridge, payloads=payloads[0],
        question="Where does Alice work?", system_prompt="Be brief.",
        max_new_tokens=4)
    seen = tok.texts[n_before:]
    ok("A3 读出期送进 tokenizer 的文本不含证据原文",
       all(EVID not in t for t in seen), extra=str(seen))
    ok("A4 读出期只喂了 system+question（不重新编码证据）", len(seen) <= 3, extra=str(len(seen)))
    ok("A5 有载荷时 n_soft = K*m", meta["n_soft"] == 1 * comp.n_slots, extra=str(meta["n_soft"]))
    ok("A6 has_payload", meta["has_payload"] is True)
    ok("A7 空载荷也能跑（退化为纯问题）",
       LM.generate_latent_answer(model=model, tokenizer=tok, bridge=bridge,
                                 payloads=None, question="hi",
                                 system_prompt="s", max_new_tokens=2)[1]["has_payload"] is False)


# =====================================================================
# B. 软 token 的位置 / 掩码 / 内容
# =====================================================================
def test_soft_token_placement():
    tok, model = StubTok(), freeze(StubModel())
    comp, bridge = make_modules(model.config.hidden_size)
    z = torch.randn(3, comp.n_slots, model.config.hidden_size)
    full, n_sys_chars = LM.build_chat_parts(tok, "Be brief.", "Where does Alice work?")
    ids_full = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"]
    pred, meta = LM.generate_latent_answer(
        model=model, tokenizer=tok, bridge=bridge, payloads=z,
        question="Where does Alice work?", system_prompt="Be brief.", max_new_tokens=2)
    emb = model.last_embeds
    n_sys, n_soft = meta["n_sys_tokens"], meta["n_soft"]
    n_in = meta["n_in"]
    n_q = int(ids_full.shape[1]) - n_sys
    ok("B1 输入长度 = n_sys + K*m + n_query",
       n_in == n_sys + 3 * comp.n_slots + n_q,
       extra=f"{n_in} vs {n_sys}+{3*comp.n_slots}+{n_q}")
    ok("B2 n_soft = K*m", n_soft == 3 * comp.n_slots)
    ok("B3 attention_mask 全 1（软 token 全部参与注意）",
       model.last_mask is not None and int(model.last_mask.sum()) == n_in
       and model.last_mask.shape[1] == n_in)
    with torch.no_grad():
        soft_expected = bridge(z)
    if soft_expected.dim() == 3:
        soft_expected = soft_expected.reshape(1, -1, soft_expected.shape[-1])
    got = emb[:, n_sys:n_sys + n_soft]
    ok("B4 soft 段逐元素等于 bridge(Z)",
       got.shape == soft_expected.shape
       and torch.allclose(got, soft_expected.to(got.dtype), atol=1e-5),
       extra=f"{tuple(got.shape)} vs {tuple(soft_expected.shape)}")
    # system 段必须是正常 token 的嵌入（不是软 token）
    with torch.no_grad():
        sys_expected = model.get_input_embeddings()(ids_full[:, :n_sys])
    ok("B5 前 n_sys 个位置仍是 system 的正常 token 嵌入",
       n_sys == sys_expected.shape[1]
       and torch.allclose(emb[:, :n_sys], sys_expected, atol=1e-6))
    ok("B6 insert_at == n_sys_tokens", meta["insert_at"] == n_sys)
    # 无 system 时软 token 在最前
    _p2, m2 = LM.generate_latent_answer(model=model, tokenizer=tok, bridge=bridge,
                                        payloads=z, question="q", system_prompt="",
                                        max_new_tokens=2)
    ok("B7 空 system 时 n_sys=0 且软 token 落在最前", m2["n_sys_tokens"] == 0
       and m2["n_soft"] == 3 * comp.n_slots)


# =====================================================================
# C. 梯度路径
# =====================================================================
def test_gradients():
    import train_latent_memory as T

    tok, model = StubTok(), freeze(StubModel())
    comp, bridge = make_modules(model.config.hidden_size)
    enc = LM.LatentEncoder(model, tok, comp, layer=2, max_unit_tokens=16)

    payloads, _ = enc.encode(["Alice works at Raytheon."], parent_ids=["fact:0"], grad=True)
    z = payloads[0]
    ok("C1 训练路径 z 带梯度", z.requires_grad)
    payloads_s, _ = enc.encode(["x y z"], parent_ids=["f:9"], grad=False)
    ok("C2 grad=False 的存储路径输出不带梯度（特征在 no_grad 下取得）",
       not payloads_s[0].requires_grad)

    # 训练必须过 bridge：推理期 generate_latent_answer 内部就是 soft = bridge(payloads)。
    # 曾经训练时直接把 z 当 soft token，bridge 不在计算图里 —— 训完是随机投影。
    soft = bridge(z)
    embeds, labels, n_ans = T.build_student_inputs(
        model, tok, soft, "Be brief.", "Where does Alice work?", "Raytheon")
    ok("C3 labels 只标注答案 token",
       int((labels != -100).sum()) == n_ans and n_ans > 0)
    logits = model(inputs_embeds=embeds,
                   attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long),
                   use_cache=False).logits
    loss = T.qa_loss(logits, labels)
    loss.backward()
    # 注意断言的是**梯度非零**，不是 is not None：全零梯度同样说明这条路没通
    ok("C4 compressor.queries 拿到非零梯度", comp.queries.grad is not None
       and float(comp.queries.grad.abs().sum()) > 0)
    ok("C5 compressor.input_proj 拿到非零梯度", comp.input_proj.weight.grad is not None
       and float(comp.input_proj.weight.grad.abs().sum()) > 0)
    ok("C6 bridge 末层 Linear 拿到非零梯度", bridge.net[3].weight.grad is not None
       and float(bridge.net[3].weight.grad.abs().sum()) > 0)
    ok("C7 冻结的模型权重没有梯度",
       all(p.grad is None for p in model.parameters()))

    # 存储路径 detach：默认 detach=True，训练路径必须显式 False
    bank = LM.LatentSlotBank(n_slots=comp.n_slots, dim=model.config.hidden_size,
                             dtype=torch.float32)
    payloads2, _ = enc.encode(["Bob works at Contoso."], parent_ids=["fact:1"], grad=True)
    bank.append("fact:1", payloads2[0])
    ok("C8 默认存储 detach（不把训练图带进库）",
       bank.payload("fact:1").requires_grad is False)
    payloads3, _ = enc.encode(["Cara works at Initech."], parent_ids=["fact:2"], grad=True)
    bank.append("fact:2", payloads3[0], detach=False)
    ok("C9 detach=False 保留梯度（训练路径）",
       bank.payload("fact:2").requires_grad is True)
    ok("C10 存储后 dtype 归一（bf16/fp32 由库决定）",
       bank.payload("fact:2").dtype == torch.float32)

    # 回归守卫：把 z 直接当 soft token（绕过 bridge）时，bridge 必须拿不到梯度。
    # 这条正是上面那个 bug 的判据 —— 一旦有人改回直连，这里会立刻失败。
    comp.zero_grad(set_to_none=True)
    bridge.zero_grad(set_to_none=True)
    payloads4, _ = enc.encode(["Dan works at Umbra."], parent_ids=["fact:3"], grad=True)
    emb4, lab4, _ = T.build_student_inputs(
        model, tok, payloads4[0], "Be brief.", "Where does Dan work?", "Umbra")
    logits4 = model(inputs_embeds=emb4,
                    attention_mask=torch.ones(emb4.shape[:2], dtype=torch.long),
                    use_cache=False).logits
    T.qa_loss(logits4, lab4).backward()
    ok("C11 绕过 bridge 时 bridge 拿不到梯度（训练必须过 bridge）",
       all(p.grad is None for p in bridge.parameters()))


# =====================================================================
# D. 长单元再分块而不截断
# =====================================================================
def test_chunking():
    tok, model = StubTok(), freeze(StubModel())
    comp, _bridge = make_modules(model.config.hidden_size)
    enc = LM.LatentEncoder(model, tok, comp, layer=2, max_unit_tokens=4)
    text = "one two three four five six seven eight nine ten"
    n_words = len(text.split())
    feats = enc.features([text], parent_ids=["epi:0"])
    payloads, info = enc.encode([text], parent_ids=["epi:0"])
    z = payloads[0]
    ok("D1 长单元被分成多块", z.shape[0] > 1, extra=str(tuple(z.shape)))
    ok("D2 每块仍是 m 个向量", z.shape[1] == comp.n_slots)
    ok("D3 分块宽度不超过 max_unit_tokens",
       feats["max_unit_tokens"] == 4 and info["n_chunks"] == 3, extra=str(info["n_chunks"]))
    ok("D4 没有截断：总 token 数等于原文 token 数",
       int(info["n_tokens_in"]) == n_words, extra=f"{info['n_tokens_in']} vs {n_words}")
    ok("D5 span 覆盖全部 token 且连续",
       feats["span"][0][0] == 0 and feats["span"][-1][1] == n_words
       and all(feats["span"][i][1] == feats["span"][i + 1][0]
               for i in range(len(feats["span"]) - 1)), extra=str(feats["span"]))
    ok("D6 所有块记在同一个父槽位 ID 下", set(feats["parents"]) == {"epi:0"})
    bank = LM.LatentSlotBank(n_slots=comp.n_slots, dim=model.config.hidden_size,
                             dtype=torch.float32)
    bank.append("epi:0", z)
    ok("D7 gather 把 k 块摊平成 [Σk, m, d]",
       tuple(bank.gather(["epi:0"]).shape) == (z.shape[0], comp.n_slots,
                                               model.config.hidden_size))
    # 短单元只有 1 块
    _p, info2 = enc.encode(["short one"], parent_ids=["epi:1"])
    ok("D8 短单元只有 1 块", info2["n_chunks"] == 1)


# =====================================================================
# E. 载荷库语义
# =====================================================================
def test_bank_semantics():
    d, m = 8, 4
    bank = LM.LatentSlotBank(n_slots=m, dim=d, dtype=torch.float32)
    z = torch.ones(1, m, d)
    bank.append("epi:0", z)
    raised = False
    try:
        bank.append("epi:0", z)
    except ValueError:
        raised = True
    ok("E1 重复 append 报错（不静默覆盖）", raised)
    raised = False
    try:
        bank.overwrite("epi:9", z)
    except KeyError:
        raised = True
    ok("E2 overwrite 缺失 ID 报错", raised)
    raised = False
    try:
        bank.gather(["epi:9"])
    except KeyError:
        raised = True
    ok("E3 gather 缺失 ID 报错（不静默补零）", raised)
    v0 = bank.versions["epi:0"]
    bank.overwrite("epi:0", torch.zeros(1, m, d))
    ok("E4 覆写后版本号 +1", bank.versions["epi:0"] == v0 + 1)
    ok("E5 覆写后载荷内容变了", float(bank.payload("epi:0").abs().sum()) == 0.0)
    # 命名空间：epi:0 与 state:0 互不影响
    bank.append("state:0", torch.full((1, m, d), 2.0))
    g = bank.gather(["epi:0", "state:0"])
    ok("E6 命名空间不串味", float(g[0].abs().sum()) == 0.0
       and float(g[1].abs().sum()) == float(2.0 * m * d))
    ok("E7 slot_ids 按命名空间+下标排序",
       bank.slot_ids() == ["epi:0", "state:0"], extra=str(bank.slot_ids()))
    # 淘汰重编号：删 state:0 后 state:1 -> state:0
    bank.append("state:1", torch.full((1, m, d), 3.0))
    bank.drop("state:0")
    moved = bank.shift_index("state", 0)
    ok("E8 淘汰后 shift_index 重编号", moved == 1
       and "state:0" in bank and "state:1" not in bank, extra=str(bank.slot_ids()))
    ok("E9 重编号后载荷跟着移动（不指向别的事实）",
       float(bank.payload("state:0").abs().sum()) == float(3.0 * m * d))
    empty = LM.LatentSlotBank(n_slots=m, dim=d, dtype=torch.float32)
    ok("E10 空库 gather 返回 (0,m,d)",
       tuple(empty.gather([]).shape) == (0, m, d))


# =====================================================================
# F. CR 路径预算
# =====================================================================
def test_cr_path_budget():
    paths = [{"path_len": 2, "facts": [0, 5]}, {"path_len": 3, "facts": [1, 6, 7]}]
    kept, info = LM.select_latent_cr_paths([0, 1, 5, 6, 7], paths, 32, 4)
    ok("F1 预算充足时两条路径都保留（含第二跳）",
       set(kept) == {0, 5, 1, 6, 7}, extra=str(kept))
    ok("F2 n_paths_kept=2", info["n_paths_kept"] == 2)
    kept, info = LM.select_latent_cr_paths([0, 1, 5, 6, 7], paths, 8, 4)  # K=2
    ok("F3 预算不足时保留整条 2 跳路径", set(kept) == {0, 5}, extra=str(kept))
    ok("F4 装不下的 3 跳路径整条跳过", len(info["n_paths_skipped"]) == 1)
    ok("F5 没有半条路径（第二跳 6/7 不得被补位塞回）",
       not ({6, 7} & set(kept)), extra=str(kept))
    ok("F6 被跳过的路径记了 facts",
       set(info["n_paths_skipped"][0]["facts"]) == {1, 6, 7})
    kept, info = LM.select_latent_cr_paths([0, 5], [{"path_len": 2, "facts": [0, 5]}], 4, 4)
    ok("F7 一条都装不下时保留首条并标 overflow",
       set(kept) == {0, 5} and info["overflow"] is True)
    kept, info = LM.select_latent_cr_paths([0, 5], paths, 0, 4)
    ok("F8 预算为 0 时 disabled 且返回空", kept == [] and info.get("disabled") is True)
    kept, info = LM.select_latent_cr_paths([0, 1, 2], [], 12, 4)
    ok("F9 无路径时按槽位顺序补满预算", kept == [0, 1, 2])


# =====================================================================
# G. 配置指纹隔离
# =====================================================================
def test_manifest_fingerprint():
    import run_manifest as RM

    base = {"mode": "v9", "top_k": 12,
            "latent": {"n_slots": 4, "layer": 18, "read_budget": 32,
                       "checkpoint": {"checkpoint": "/x/a.pt", "exists": True,
                                      "sha256_16": "aaaa"}}}
    cid_a = RM.config_id(base, kind="ar", src="eventqa_65536")
    same = json.loads(json.dumps(base))
    ok("G1 同配置 config_id 稳定", RM.config_id(same, kind="ar", src="eventqa_65536") == cid_a)
    b = json.loads(json.dumps(base))
    b["latent"]["n_slots"] = 8
    ok("G2 不同 m -> 不同 config_id",
       RM.config_id(b, kind="ar", src="eventqa_65536") != cid_a)
    c = json.loads(json.dumps(base))
    c["latent"]["checkpoint"]["sha256_16"] = "bbbb"
    ok("G3 换了 checkpoint -> 不同 config_id",
       RM.config_id(c, kind="ar", src="eventqa_65536") != cid_a)
    # 文件指纹
    fp_missing = RM.file_fingerprint("/nonexistent/latent.pt", label="checkpoint")
    ok("G4 缺失 checkpoint 的指纹 exists=False", fp_missing["exists"] is False)
    with tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False) as f:
        f.write(b"latent-ckpt-bytes")
        tmp = f.name
    try:
        fp = RM.file_fingerprint(tmp, label="checkpoint")
        ok("G5 指纹含字节数与 16 位 sha256",
           fp["exists"] is True and fp["bytes"] == len(b"latent-ckpt-bytes")
           and len(fp["sha256_16"]) == 16, extra=str(fp))
        ok("G6 latent_memory 与 run_manifest 指纹口径一致",
           LM.checkpoint_fingerprint(tmp)["sha256_16"] == fp["sha256_16"])
    finally:
        os.remove(tmp)


# =====================================================================
# H. 数值安全（全 padding 行）
# =====================================================================
def test_empty_rows():
    comp, _bridge = make_modules(32)
    h0 = torch.randn(2, 5, 32)
    h18 = torch.randn(2, 5, 32)
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.long)
    z, info = comp.forward_with_info(h0, h18, mask)
    ok("H1 全 padding 行不产生 NaN", bool(torch.isfinite(z).all()))
    ok("H2 全 padding 行输出被置零", float(z[1].abs().sum()) == 0.0)
    ok("H3 记了 n_empty_rows", info.get("n_empty_rows") == 1, extra=str(info.get("n_empty_rows")))
    ok("H4 正常行非零", float(z[0].abs().sum()) > 0)


# =====================================================================
# I. checkpoint 往返
# =====================================================================
def test_checkpoint_roundtrip():
    tok, model = StubTok(), freeze(StubModel())
    comp, bridge = make_modules(model.config.hidden_size)
    cfg = {"schema": LM.LATENT_SCHEMA, "n_slots": comp.n_slots,
           "layer": 2, "inner_dim": comp.inner_dim, "num_heads": comp.num_heads,
           "bottleneck": bridge.bottleneck, "max_unit_tokens": 16,
           "read_budget": 32, "dtype": "float32"}
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "latent.pt")
        LM.save_latent_checkpoint(path, comp, bridge, cfg, {"note": "unit-test"})
        ok("I1 checkpoint 落盘", os.path.isfile(path))
        reader = LM.load_latent_reader(path, model, tok)
        ok("I2 读回 n_slots 一致", reader.n_slots() == comp.n_slots)
        ok("I3 读回 layer 一致", reader.config.get("layer") == 2)
        ok("I4 读回权重一致",
           torch.allclose(reader.compressor.queries.detach(), comp.queries.detach()))
        ok("I5 读出模块被冻结（推理期只读）",
           all(not p.requires_grad for p in reader.compressor.parameters())
           and all(not p.requires_grad for p in reader.bridge.parameters()))
        ok("I6 指纹带 sha256_16", len(reader.fingerprint.get("sha256_16", "")) == 16)
        raised = False
        try:
            LM.load_latent_reader("", model, tok)
        except FileNotFoundError:
            raised = True
        ok("I7 未给 checkpoint 必须报错（禁止随机初始化顶上）", raised)
        raised = False
        try:
            LM.load_latent_reader(os.path.join(td, "nope.pt"), model, tok)
        except FileNotFoundError:
            raised = True
        ok("I8 checkpoint 不存在必须报错", raised)
        # hidden_size 不一致必须报错：拿 32 维训练出来的载荷去喂 48 维模型只会静默错位
        raised = False
        try:
            LM.load_latent_reader(path, StubModel(hidden=48), tok)
        except ValueError:
            raised = True
        ok("I9 hidden_size 不一致必须报错", raised)


# =====================================================================
# J. 预算算术
# =====================================================================
def test_budget_math():
    ok("J1 32 软 token / m=4 -> 8 槽位", LM.latent_budget_slots(32, 4) == 8)
    ok("J2 预算为 0 -> 0 槽位", LM.latent_budget_slots(0, 4) == 0)
    ok("J3 预算小于一个槽位时仍保留 1 个（宁可轻微超预算也不返回空读出）",
       LM.latent_budget_slots(3, 4) == 1)
    ok("J4 m<=0 不崩", LM.latent_budget_slots(32, 0) == 0)
    cfg = LM.latent_config()
    ok("J5 latent_config 含关键字段",
       {"n_slots", "layer", "inner_dim", "max_unit_tokens", "read_budget", "dtype"} <= set(cfg),
       extra=str(sorted(cfg)))


def main():
    print("=" * 70)
    print("tests/test_latent_readout.py — 潜层读出机制验证（CPU / 无模型）")
    print("=" * 70)
    for fn in (test_no_plaintext_interface, test_soft_token_placement, test_gradients,
               test_chunking, test_bank_semantics, test_cr_path_budget,
               test_manifest_fingerprint, test_empty_rows, test_checkpoint_roundtrip,
               test_budget_math):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n" + "=" * 70)
    if FAILS:
        print(f"FAILED {len(FAILS)}/{N_OK + len(FAILS)}: {FAILS}")
        return 1
    print(f"ALL PASS ({N_OK} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
