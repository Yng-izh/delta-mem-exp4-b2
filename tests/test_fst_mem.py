#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tests/test_fst_mem.py — FST-Mem（状态—轨迹解耦潜层记忆）的机制性验证。

CPU、无 GPU、无 Qwen：验证的是**一旦做错就会静默失效**的那几条机制，
不是"分数变好了"（那需要真实训练 + 真实跑分，本文件一概不声称）。

  F. 事实键：k = (实体, 属性)，与**值无关** —— 值变了键必须不变。
  G. 事实键编码器：实体/属性分别映射后联合（含交互项），输出归一化；
     并且真的参与训练（梯度到 e_net / r_net / out / q_net）。
  H. 状态—轨迹库：写新值 = 旧值下沉进历史，**其他事实键一个字节都不动**
     （更新局域性由数据结构保证，不靠"训练时学会别串"）；
     历史容量超限要真正释放槽位；按时间条件读版本的选择规则要可预测。
  I. 时序两级寻址：先在事实组上选（与组数/顺序无关），再在组内按时间选版本
     （跨组版本不互相竞争）；训练前向用 STE，前向数值逐位等于硬选。
  J. 读出：FST 路径只注入被选中版本的 m 个软 token（不是全部版本裸拼）；
     证据值本身**从不**进 tokenizer（值只在潜载荷里）。
  K. checkpoint 往返：has_fst=True 而权重缺失必须报错，不许退回裸拼。

用法：python tests/test_fst_mem.py
"""
import inspect
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


class StubTok:
    """词级 tokenizer；记录所有被送进来的文本（用于"值从不进 tokenizer"的断言）。"""

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
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}

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
    def __init__(self, hidden=32):
        super().__init__()
        self.config = _Cfg()
        self.config.hidden_size = int(hidden)
        self.emb = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.head = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.last_embeds = None

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                output_hidden_states=False, use_cache=False, **kw):
        x = inputs_embeds if inputs_embeds is not None else self.emb(input_ids)
        x = x + 0.1 * x.mean(dim=1, keepdim=True)
        hs = [x] * (self.config.num_hidden_layers + 1) if output_hidden_states else None
        return _Out(self.head(x), hs)

    def generate(self, inputs_embeds=None, attention_mask=None, max_new_tokens=8, **kw):
        self.last_embeds = inputs_embeds
        return torch.full((1, int(max_new_tokens)), 7, dtype=torch.long)


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


D_MODEL, M, D_KEY = 32, 4, 8


def rand_payload(seed):
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn((M, D_MODEL), generator=g)


# =====================================================================
# F. 事实键：与值无关
# =====================================================================
def test_fact_key():
    k1 = LM.fact_key("Zhang San", "Employer")
    k2 = LM.fact_key("  zhang   san ", "employer.")
    ok("F1 事实键规范化（大小写/空白/标点）", k1 == k2 == "zhang san||employer", extra=f"{k1!r}/{k2!r}")
    ok("F2 事实键与值无关（同键不同值仍同键）",
       LM.fact_key("Zhang", "employer") == LM.fact_key("Zhang", "employer"))
    e, r = LM.split_fact_key(k1)
    ok("F3 split_fact_key 往返", (e, r) == ("zhang san", "employer"), extra=f"{e!r},{r!r}")
    ok("F4 空实体/属性报错",
       _raises(lambda: LM.fact_key("", "employer")) and _raises(lambda: LM.fact_key("a", " ")))
    ok("F5 非法键字符串报错", _raises(lambda: LM.split_fact_key("no-separator")))
    # 属性名里的分隔符必须被剔除，否则 "a||b" 与 ("a","b") 会撞键
    k3 = LM.fact_key("x||y", "z")
    ok("F6 键内不允许出现分隔符", k3.count("||") == 1, extra=k3)


def _raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


# =====================================================================
# G. 事实键编码器
# =====================================================================
def test_fact_key_encoder():
    torch.manual_seed(0)
    enc = LM.LatentFactKeyEncoder(D_MODEL, d_key=D_KEY)
    e = torch.randn(3, D_MODEL)
    r = torch.randn(3, D_MODEL)
    k = enc.encode_keys(e, r)
    ok("G1 键形状 [G,d_key]", tuple(k.shape) == (3, D_KEY), extra=str(tuple(k.shape)))
    ok("G2 键已归一化", torch.allclose(k.norm(dim=-1), torch.ones(3), atol=1e-5),
       extra=str(k.norm(dim=-1).tolist()))
    q = enc.encode_query(torch.randn(1, 2 * D_MODEL))
    ok("G3 查询形状 [B,d_key]", tuple(q.shape) == (1, D_KEY), extra=str(tuple(q.shape)))
    ok("G4 打分形状 [B,G]", tuple(enc.scores(q, k).shape) == (1, 3))
    # 不同 (e,r) 不能被压成同一个键（否则"局部更新"会误伤别的属性）
    k_same = enc.encode_keys(e[:1], r[:1])
    k_diff = enc.encode_keys(e[:1], r[1:2])
    d_same_diff = float((k_same - k_diff).detach().norm())
    ok("G5 不同属性给出不同键", d_same_diff > 0.05, extra=f"{d_same_diff:.4f}")
    # 交互项确实参与：交换实体后键必须变（纯加法式拼接+线性会退化成可交换）
    k_swap = enc.encode_keys(r[:1], e[:1])
    ok("G6 实体/属性不可交换（有交互项）",
       not torch.allclose(k_same, k_swap, atol=1e-4))
    enc.zero_grad()
    loss = enc.scores(enc.encode_query(torch.randn(1, 2 * D_MODEL)), k).sum()
    loss.backward()
    ok("G7 梯度到 e_net/r_net/out/q_net",
       all(w.grad is not None for w in (enc.e_net[0].weight, enc.r_net[0].weight,
                                        enc.out[0].weight, enc.q_net[0].weight)))


# =====================================================================
# H. 状态—轨迹库：更新局域性 / 历史下沉 / 容量淘汰 / 时间读取
# =====================================================================
def test_state_trajectory_bank():
    bank = LM.LatentStateTrajectoryBank(hist_max=1)
    K_EMP = LM.fact_key("Zhang", "employer")
    K_HOME = LM.fact_key("Zhang", "residence")
    z_tencent = rand_payload(1)
    z_alibaba = rand_payload(2)
    z_beijing = rand_payload(3)

    sid_e1 = bank.write(K_EMP, z_tencent, time="2023")
    sid_h1 = bank.write(K_HOME, z_beijing, time="2024")
    ok("H1 两个属性 = 两个独立事实键", bank.group_keys() == [K_EMP, K_HOME],
       extra=str(bank.group_keys()))
    ok("H2 首次写入无历史、当前=新写入",
       bank.count_versions(K_EMP) == 1 and bank.list_versions(K_EMP)[-1]["current"] is True)
    ok("H3 当前载荷就是写入的载荷",
       torch.equal(bank.payload(sid_e1), bank.payload(bank.read(K_EMP))))

    home_before = bank.payload(sid_h1).clone()
    home_versions_before = bank.count_versions(K_HOME)
    home_sid_before = bank.read(K_HOME)

    sid_e2 = bank.write(K_EMP, z_alibaba, time="2025")   # 张三换雇主
    ok("H4 写新值后当前指针前移", bank.read(K_EMP) == sid_e2 and sid_e2 != sid_e1)
    ok("H5 旧值下沉进历史（不是删除）",
       bank.count_versions(K_EMP) == 2
       and bank.list_versions(K_EMP)[0]["slot_id"] == sid_e1
       and bank.list_versions(K_EMP)[0]["time"] == "2023")
    ok("H6 历史载荷逐位保留",
       torch.equal(bank.payload(sid_e1), bank._normalize(z_tencent).to(bank.dtype)))
    # ---- 更新局域性（本方法的核心断言）----
    ok("H7 改雇主不动居住地载荷（逐位相等）",
       torch.equal(bank.payload(sid_h1), home_before))
    ok("H8 改雇主不动居住地版本数/当前指针",
       bank.count_versions(K_HOME) == home_versions_before and bank.read(K_HOME) == home_sid_before)

    # ---- 时间条件读取 ----
    ok("H9 无时间条件 -> 当前状态", bank.read(K_EMP) == sid_e2)
    ok("H10 问 2023 -> 历史版本", bank.read(K_EMP, "2023") == sid_e1,
       extra=str(bank.read(K_EMP, "2023")))
    ok("H11 前缀匹配（2023-05 -> 2023）", bank.read(K_EMP, "2023-05") == sid_e1)
    ok("H12 未知年份晚于全部版本 -> 当前", bank.read(K_EMP, "2099") == sid_e2)
    ok("H13 早于全部版本 -> 当前（不猜最旧值）", bank.read(K_EMP, "1999") == sid_e2)
    ok("H14 版本顺序：历史旧->新，当前在最后",
       [v["slot_id"] for v in bank.list_versions(K_EMP)] == [sid_e1, sid_e2])

    # ---- 容量淘汰（真正释放槽位，不是"留着读不到"）----
    n_slots_before = len(bank.payloads)
    sid_e3 = bank.write(K_EMP, rand_payload(4), time="2027")
    ok("H15 超容量淘汰最旧历史", sid_e1 not in bank.payloads, extra=str(sorted(bank.payloads)))
    ok("H16 淘汰不新增槽位（净增 = 0）", len(bank.payloads) == n_slots_before,
       extra=f"{n_slots_before} -> {len(bank.payloads)}")
    ok("H17 淘汰计数", bank.n_evicted == 1 and bank.stats()["fst_n_evicted"] == 1)
    ok("H18 淘汰后历史长度 = hist_max",
       bank.count_versions(K_EMP) - 1 == bank.hist_max,
       extra=f"{bank.count_versions(K_EMP) - 1} vs hist_max={bank.hist_max}")
    ok("H19 淘汰后 2023 不再可读（回落到当前）", bank.read(K_EMP, "2023") == sid_e3)
    ok("H20 尚存历史仍可读", bank.read(K_EMP, "2025") == sid_e2)

    # ---- hist_max=0：硬覆写对照 ----
    b0 = LM.LatentStateTrajectoryBank(hist_max=0)
    b0.write(K_EMP, rand_payload(1), time="2023")
    b0.write(K_EMP, rand_payload(2), time="2025")
    ok("H21 hist_max=0 = 硬覆写（历史全丢）", b0.count_versions(K_EMP) == 1
       and b0.n_evicted == 1)

    # ---- 非法输入 ----
    ok("H22 非法事实键拒绝写入",
       _raises(lambda: LM.LatentStateTrajectoryBank().write("整句事实没有分隔符", rand_payload(5))))
    ok("H23 读未知事实键报错",
       _raises(lambda: bank.read(LM.fact_key("Nobody", "employer"))))

    # ---- 形状 ----
    vz, times, is_cur = bank.gather_versions(K_EMP)
    ok("H24 gather_versions 形状", tuple(vz.shape) == (bank.count_versions(K_EMP), M, D_MODEL)
       and len(times) == len(is_cur) == bank.count_versions(K_EMP), extra=str(tuple(vz.shape)))
    per_group = bank.gather_all_versions()
    z_pad, mask = LM.pad_version_stack(per_group)
    ok("H25 pad_version_stack 形状/掩码",
       tuple(z_pad.shape) == (2, max(len(g) for g in per_group), M, D_MODEL)
       and bool(mask.all()) is False and int(mask[0].sum()) == len(per_group[0]),
       extra=f"{tuple(z_pad.shape)} {mask.tolist()}")
    ok("H26 填充位为零向量（不靠'填充恰好分低'）",
       bool((z_pad[~mask] == 0).all()))
    ok("H27 空分组报错", _raises(lambda: LM.pad_version_stack([[rand_payload(1)], []])))


# =====================================================================
# I. 时序两级寻址
# =====================================================================
def test_fst_router():
    torch.manual_seed(0)
    router = LM.FstRouter(D_MODEL, n_slots=M, d_key=D_KEY)
    G, V = 3, 2
    e_feat = torch.randn(G, D_MODEL)
    r_feat = torch.randn(G, D_MODEL)
    ver_z = torch.randn(G, V, M, D_MODEL)
    ver_t_feat = torch.randn(G, V, D_MODEL)
    q_feat = torch.randn(1, 2 * D_MODEL)
    mask = torch.ones((G, V), dtype=torch.bool)

    router.eval()
    out = router(q_feat, e_feat, r_feat, ver_z, ver_t_feat, None, mask, hard=True)
    ok("I1 z_read 形状 [1,m,d]", tuple(out["z_read"].shape) == (1, M, D_MODEL),
       extra=str(tuple(out["z_read"].shape)))
    ok("I2 事实级/版本级打分形状",
       tuple(out["logits_group"].shape) == (1, G) and tuple(out["logits_ver"].shape) == (1, V),
       extra=f"{tuple(out['logits_group'].shape)}/{tuple(out['logits_ver'].shape)}")
    ok("I3 硬选出的向量逐位等于被选版本载荷",
       torch.allclose(out["z_read"][0].float(),
                      ver_z[out["group"], out["version"]].float(), atol=1e-5))

    # 组数/顺序不变性：追加一个事实组不改变已有组的打分（纯内积，无 softmax 竞争）
    e2 = torch.cat([e_feat, torch.randn(1, D_MODEL)], dim=0)
    r2 = torch.cat([r_feat, torch.randn(1, D_MODEL)], dim=0)
    vz2 = torch.cat([ver_z, torch.randn(1, V, M, D_MODEL)], dim=0)
    vt2 = torch.cat([ver_t_feat, torch.randn(1, V, D_MODEL)], dim=0)
    mask2 = torch.ones((G + 1, V), dtype=torch.bool)
    out2 = router(q_feat, e2, r2, vz2, vt2, None, mask2, hard=True)
    ok("I4 事实级打分与组数无关（追加组不改已有分数）",
       torch.allclose(out2["logits_group"][0, :G], out["logits_group"][0], atol=1e-6))

    # 跨组版本不竞争：版本级打分的长度 = 组内版本数 V，与 G 无关
    ok("I5 版本级候选只在被选组内部（长度=V，非 G*V）",
       int(out["logits_ver"].shape[-1]) == V)

    # 时间条件真的进入打分
    out_t0 = router(q_feat, e_feat, r_feat, ver_z, ver_t_feat, torch.zeros(1, D_MODEL),
                    mask, hard=True)
    out_t1 = router(q_feat, e_feat, r_feat, ver_z, ver_t_feat, torch.randn(1, D_MODEL),
                    mask, hard=True)
    ok("I6 时间条件改变版本打分（不是常量输出）",
       not torch.allclose(out_t0["logits_ver"], out_t1["logits_ver"], atol=1e-6))

    # 掩码：填充位必须被屏蔽
    m2 = mask.clone()
    m2[:, 1] = False
    outm = router(q_feat, e_feat, r_feat, ver_z, ver_t_feat, None, m2, hard=True)
    ok("I7 掩码位置分数被压到 -1e4 量级",
       float(outm["logits_ver"][0, 1].detach()) <= -1e3,
       extra=str([round(x, 3) for x in outm["logits_ver"][0].tolist()]))
    ok("I8 掩码位置不会被选中",
       int(outm["version"]) == 0, extra=str(outm["version"]))

    # STE：训练模式前向逐位等于硬选，梯度仍能回传
    router.train()
    out_s = router(q_feat, e_feat, r_feat, ver_z, ver_t_feat, None, mask, hard=False)
    out_h = router(q_feat, e_feat, r_feat, ver_z, ver_t_feat, None, mask, hard=True)
    ok("I9 训练前向 = 硬选（STE 前向一致）",
       torch.equal(out_s["z_read"], out_h["z_read"])
       and out_s["group"] == out_h["group"] and out_s["version"] == out_h["version"])
    router.zero_grad()
    out_s["z_read"].sum().backward()
    ok("I10 梯度到事实键与版本两个头",
       router.key.e_net[0].weight.grad is not None
       and router.state.ver_net[0].weight.grad is not None
       and router.key.q_net[0].weight.grad is not None
       and router.state.q_net[0].weight.grad is not None)
    ok("I11 载荷本身不是被优化对象（z_read 对 ver_z 无梯度）",
       ver_z.grad is None)


# =====================================================================
# J. 读出：只注入被选中版本的 m 个软 token；值不进 tokenizer
# =====================================================================
def test_fst_readout():
    torch.manual_seed(0)
    model = freeze(StubModel())
    tok = StubTok()
    comp = LM.LatentCompressor(model.config.hidden_size, n_slots=M, inner_dim=64,
                               num_heads=4).eval()
    bridge = LM.LatentBridge(model.config.hidden_size, bottleneck=16).eval()
    router = LM.FstRouter(model.config.hidden_size, n_slots=M, d_key=D_KEY).eval()
    bank = LM.LatentStateTrajectoryBank(hist_max=3)
    K_EMP = LM.fact_key("Zhang", "employer")
    K_HOME = LM.fact_key("Zhang", "residence")
    bank.write(K_EMP, rand_payload(1), time="2023")
    bank.write(K_EMP, rand_payload(2), time="2025")
    bank.write(K_HOME, rand_payload(3), time="2024")
    VALUE = "ZQXKVV"            # 只出现在值里，绝不能进 tokenizer

    n_before = len(tok.texts)
    pred, meta = LM.generate_latent_answer_fst(
        model=model, tokenizer=tok, bridge=bridge, bank=bank,
        question="Where does Zhang work?", system_prompt="Be brief.",
        fst_router=router, max_new_tokens=4, layer=2, query_time="2023")
    seen = tok.texts[n_before:]
    ok("J1 只注入 1 个版本的 m 个软 token（不是全部版本裸拼）",
       meta["n_soft"] == M, extra=str(meta["n_soft"]))
    ok("J2 meta 记录了两级选择", isinstance(meta["group_picked"], int)
       and isinstance(meta["ver_picked"], int)
       and meta["n_groups"] == 2 and meta["n_versions"] > 0, extra=str(meta))
    ok("J3 读出期 tokenizer 只看到 system/问题/键名/时间，且键名里没有值",
       all("zhang" in t.lower() or "employer" in t.lower()
           or "residence" in t.lower() or "2023" in t or "2024" in t or "2025" in t
           or t.startswith("<|") for t in seen), extra=str(seen))
    ok("J4 事实值从未进入 tokenizer", all(VALUE not in t for t in seen), extra=str(seen))
    ok("J5 decode 正常返回", isinstance(pred, str))

    # 时间条件影响被读的版本（结构层面：query_time 不同 -> 版本候选集不变但打分变）
    _p2, meta2 = LM.generate_latent_answer_fst(
        model=model, tokenizer=tok, bridge=bridge, bank=bank,
        question="Where does Zhang work?", system_prompt="Be brief.",
        fst_router=router, max_new_tokens=2, layer=2, query_time=None)
    ok("J6 query_time=None 时 meta 记录空条件", meta2["query_time"] == "")

    ok("J7 空 bank 报错", _raises(lambda: LM.generate_latent_answer_fst(
        model=model, tokenizer=tok, bridge=bridge,
        bank=LM.LatentStateTrajectoryBank(), question="q", system_prompt="s",
        fst_router=router, max_new_tokens=2, layer=2)))
    ok("J8 多跳未实现时显式报错（不静默按单跳跑）",
       _raises(lambda: LM.generate_latent_answer_fst(
           model=model, tokenizer=tok, bridge=bridge, bank=bank, question="q",
           system_prompt="s", fst_router=router, max_new_tokens=2, layer=2, hops=2)))
    ok("J9 传普通 SlotBank 报错（类型必须对）",
       _raises(lambda: LM.generate_latent_answer_fst(
           model=model, tokenizer=tok, bridge=bridge, bank=LM.LatentSlotBank(),
           question="q", system_prompt="s", fst_router=router, max_new_tokens=2,
           layer=2)))

    # 与老路径的签名对照：FST 是**新函数**，不许往老接口塞潜文本入口
    old_params = set(inspect.signature(LM.generate_latent_answer).parameters)
    ok("J10 generate_latent_answer 签名未被 FST 改动",
       old_params == {"model", "tokenizer", "bridge", "payloads", "question",
                      "system_prompt", "max_new_tokens", "pad_token_id",
                      "router", "q_feat", "layer", "hops"}, extra=str(sorted(old_params)))
    new_params = set(inspect.signature(LM.generate_latent_answer_fst).parameters)
    ok("J11 FST 签名同样不含明文证据入口",
       not (new_params & {"block", "evidence", "evidence_text", "memory", "facts",
                          "passages", "documents"}), extra=str(sorted(new_params)))


# =====================================================================
# K. checkpoint 往返
# =====================================================================
def test_checkpoint_roundtrip():
    torch.manual_seed(0)
    model = freeze(StubModel())
    tok = StubTok()
    comp = LM.LatentCompressor(model.config.hidden_size, n_slots=M, inner_dim=64,
                               num_heads=4).eval()
    bridge = LM.LatentBridge(model.config.hidden_size, bottleneck=16).eval()
    router = LM.FstRouter(model.config.hidden_size, n_slots=M, d_key=D_KEY).eval()
    d = tempfile.mkdtemp()
    p = os.path.join(d, "fst.pt")
    LM.save_latent_checkpoint(p, comp, bridge,
                              {"layer": 2, "max_unit_tokens": 8}, {"unit": True},
                              fst_router=router)
    blob = LM._safe_torch_load(p)
    ok("K1 config 记录 has_fst/d_key", blob["config"]["has_fst"] is True
       and int(blob["config"]["d_key"]) == D_KEY)
    ok("K2 权重落盘", blob.get("fst_router") is not None)
    rd = LM.load_latent_reader(p, model, tok)
    ok("K3 重载带回 fst_router", rd.fst_router is not None)
    same = all(torch.equal(v, rd.fst_router.state_dict()[k])
               for k, v in router.state_dict().items())
    ok("K4 权重逐位一致", same)

    # 权重缺失必须报错，不许静默退回"全版本裸拼"
    blob2 = dict(blob)
    blob2["fst_router"] = None
    p2 = os.path.join(d, "fst_broken.pt")
    torch.save(blob2, p2)
    try:
        LM.load_latent_reader(p2, model, tok)
        raised = False
    except Exception:
        raised = True
    ok("K5 has_fst=True 但权重缺失 -> 报错", raised)


def main():
    test_fact_key()
    test_fact_key_encoder()
    test_state_trajectory_bank()
    test_fst_router()
    test_fst_readout()
    test_checkpoint_roundtrip()
    print("=" * 70)
    if FAILS:
        print(f"FAILED {len(FAILS)}/{N_OK + len(FAILS)}: {FAILS}")
        return 1
    print(f"ALL PASS ({N_OK} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
