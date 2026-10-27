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
    # F2 原先是 `fact_key("Zhang","employer") == fact_key("Zhang","employer")`：
    # 同一函数同一参数自比，恒真，等于没测。真正的性质是"值变了键不变"，
    # 只能通过**写入两条不同值**来验：同键两写 -> 仍然只有一组、两个版本。
    _b = LM.LatentStateTrajectoryBank(hist_max=2)
    _k = LM.fact_key("Zhang", "employer")
    _b.write(_k, rand_payload(1), time="2023")
    _b.write(_k, rand_payload(2), time="2025")
    ok("F2 事实键与值无关（同键写两个不同值仍是一组两个版本）",
       _b.group_keys() == [_k] and _b.count_versions(_k) == 2,
       extra=f"{_b.group_keys()} {_b.count_versions(_k)}")
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
    # I11 原先是 `ver_z.grad is None`，而这里的 ver_z 来自 torch.randn（非叶子，
    # requires_grad=False），grad 必然为 None —— 恒真，且恰好掩盖了真正的毛病：
    # 未 detach 的软期望会把梯度按 wg*wv 铺到**全部 G×V 条**载荷上（压缩器被迫同时
    # 满足所有事实）。现在改成"叶子 + 检查梯度分布"：载荷只能经由硬选那一路拿梯度。
    # 注意：载荷的梯度有**两个合法来源** —— (a) 硬选那一路（z_hard 直接取选中版本），
    # (b) 版本打分那一路（TemporalStateSelector 的输入本身就是载荷）。所以"非选中版本
    # 梯度为 0"是错的断言。要验的是第三条路（软期望混合）必须被 detach：
    #   STE 前向后的载荷梯度 必须与 hard=True 前向的载荷梯度**逐位相同**。
    def _grad_with(hard_mode):
        leaf = ver_z.clone().requires_grad_(True)
        router.zero_grad()
        o = router(q_feat, e_feat, r_feat, leaf, ver_t_feat, None, mask, hard=hard_mode)
        o["z_read"].sum().backward()
        return leaf.grad.clone(), int(o["group"]), int(o["version"])

    # 载荷还有**第三个**合法梯度来源：版本打分函数的输入本身就是载荷
    # （TemporalStateSelector 吃 z.flatten(-2)），所以必须先把这一路切断才能单独
    # 检查"软期望混合"有没有漏梯度——否则测的是三条路的和，什么都测不出来。
    _orig_ver = router.state.encode_versions
    _fixed_ver = _orig_ver(ver_z, ver_t_feat).detach()
    router.state.encode_versions = lambda z, t: _fixed_ver
    g_ste, gs, vs = _grad_with(False)
    g_hard, gh, vh = _grad_with(True)
    router.state.encode_versions = _orig_ver
    ok("I11 STE 不给载荷额外梯度（与 hard=True 的载荷梯度逐位相同）",
       g_ste is not None and g_hard is not None and (gs, vs) == (gh, vh)
       and torch.allclose(g_ste, g_hard, atol=1e-6),
       extra=f"max|Δ|={float((g_ste - g_hard).abs().max()):.4g}")


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
    # 值只可能存在于载荷/事实原文里。原来的 J4 断言 "VALUE 不在 tokenizer 文本中"，
    # 但全文没有任何一处把 VALUE 送进管线 —— 恒真。现在改成**白名单集合断言**：
    # 读出期 tokenizer 只能看到 {system, 问题, 实体名, 属性名, 版本时间, 查询时间}，
    # 任何"把事实原文拼进去"的实现都会让集合多出元素而被抓住。
    VALUE = "ZQXKVV"
    fact_text = f"Zhang works at {VALUE}."

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
    joined = " ".join(seen)
    # 实质检查：任何事实原文被拼进 prompt 都会让 "works at"/"lives in"/"residence" 这类
    # 事实句成分出现在文本里（问题文本是 "Where does Zhang work?"，不含这些搭配）。
    # 同时反向确认问题与 system **确实**进过 tokenizer，否则这条断言是空转的。
    ok("J3 没有事实原文进入 tokenizer（且问题/system 确实进过，不是空转）",
       ("works at" not in joined) and ("lives in" not in joined)
       and ("Where does Zhang work?" in joined) and ("Be brief." in joined),
       extra=str(seen[:6]))
    ok("J4 事实原文与事实值从未进入 tokenizer",
       all(VALUE not in t for t in seen) and fact_text not in joined, extra=str(seen[:6]))
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


# =====================================================================
# L. 组序/版本序/掩码的**内容级**对齐（错了不报错，只会"读到另一条属性"）
# =====================================================================
def test_alignment():
    """上一轮把 gather_all_versions 的返回顺序反转，67 项断言**全过**，而读出
    实际读到了另一条属性的载荷、meta 却照样报对的键 —— 这类静默错位必须用
    "载荷归属"级别的断言才能抓住，光测形状没用。
    """
    bank = LM.LatentStateTrajectoryBank(hist_max=3)
    KA, KB = LM.fact_key("A", "x"), LM.fact_key("B", "y")
    bank.write(KA, rand_payload(11), time="2020")
    bank.write(KA, rand_payload(12), time="2021")
    bank.write(KB, rand_payload(13), time="2022")
    keys = bank.group_keys()
    per = bank.gather_all_versions(keys)
    ok("L1 组数与版本数", len(keys) == 2 and len(per) == 2
       and len(per[0]) == 2 and len(per[1]) == 1, extra=f"{keys} {[len(x) for x in per]}")
    aligned = all(torch.equal(per[i][j], bank.payload(v["slot_id"]))
                  for i, k in enumerate(keys)
                  for j, v in enumerate(bank.list_versions(k)))
    ok("L2 gather_all_versions 与 list_versions/payload 逐条对齐（组序错位必被抓）",
       aligned)
    ok("L3 两组载荷本来就不同（否则 L2 抓不到交换）",
       not torch.equal(per[0][0], per[1][0]))
    # 内容级：第 0 组的版本时间顺序必须是 2020 -> 2021（旧->新），且当前=NULL
    ok("L4 组内版本顺序为旧->新且当前在最后",
       [v["time"] for v in bank.list_versions(KA)] == ["2020", "2021"]
       and bank.list_versions(KA)[-1]["current"] is True)

    # ---- 组标记载荷：z_read 必须来自被选中的那一组 ----
    torch.manual_seed(1)
    router = LM.FstRouter(D_MODEL, n_slots=M, d_key=D_KEY).eval()
    G, V = 2, 2
    vz = torch.zeros(G, V, M, D_MODEL)
    vz[0, 0] = 1.0
    vz[0, 1] = 3.0            # 组 0 的"填充位"：若被误选，z_read 会变成 3.0
    vz[1] = 7.0
    zt = torch.zeros(G, V, D_MODEL)
    # 组 0 只有 1 个版本（第 2 个是填充位），组 1 有 2 个
    mask = torch.tensor([[True, False], [True, True]])
    o_g, o_v = router.key.scores, router.state.scores
    try:
        router.key.scores = lambda q, k: torch.tensor([[-9.0, 9.0]])
        router.state.scores = lambda q, k: torch.full((1, int(k.shape[0])), 0.5)
        out = router(torch.randn(1, 2 * D_MODEL), torch.randn(G, D_MODEL),
                     torch.randn(G, D_MODEL), vz, zt, None, mask, hard=True)
        ok("L5 z_read 来自被选中的那一组（组序错位会被抓住）",
           int(out["group"]) == 1
           and torch.allclose(out["z_read"], torch.full((1, M, D_MODEL), 7.0)),
           extra=f"group={out['group']} mean={float(out['z_read'].mean()):.3f}")
        # 掩码端到端：组 0 只有 1 个版本，第 2 个候选是填充位；即使打分偏爱它也不能被选中
        # （填充位若被选中，z_read 会是 3.0 而不是 1.0，所以这条断言有分辨力）
        router.key.scores = lambda q, k: torch.tensor([[9.0, -9.0]])
        router.state.scores = lambda q, k: torch.tensor([[0.0, 99.0, -99.0, -99.0]])
        out2 = router(torch.randn(1, 2 * D_MODEL), torch.randn(G, D_MODEL),
                      torch.randn(G, D_MODEL), vz, zt, None, mask, hard=True)
        ok("L6 掩码位（填充版本）即使分高也不会被选中",
           int(out2["group"]) == 0 and int(out2["version"]) == 0
           and torch.allclose(out2["z_read"], torch.full((1, M, D_MODEL), 1.0)),
           extra=f"g={out2['group']} v={out2['version']}")
    finally:
        router.key.scores, router.state.scores = o_g, o_v


# =====================================================================
# M. 数据 -> 库 -> 张量：目标定位与"训练/读出同源"（真实 jsonl 驱动）
# =====================================================================
def test_data_pipeline():
    import train_latent_memory as T

    model = freeze(StubModel())
    tok = StubTok()
    comp = LM.LatentCompressor(model.config.hidden_size, n_slots=M, inner_dim=64,
                              num_heads=4).eval()
    bridge = LM.LatentBridge(model.config.hidden_size, bottleneck=16).eval()
    router = LM.FstRouter(model.config.hidden_size, n_slots=M, d_key=D_KEY).eval()
    encoder = LM.LatentEncoder(model, tok, comp, layer=2, max_unit_tokens=16)

    path = os.path.join(ROOT, "synth", "fst_update_val.jsonl")
    if not os.path.exists(path):
        ok("M0 需要 synth/fst_update_val.jsonl（先生成数据）", False, extra=path)
        return
    rows = T.load_synth_jsonl(path, limit=8, seed=0)
    hist = [r for r in rows if r.get("probe") == "history"][:1]
    cur = [r for r in rows if r.get("probe") == "current"][:1]
    ok("M1 数据里同时有 current 与 history 探针", bool(hist) and bool(cur))

    fi = T.build_fst_bank(encoder, cur[0], 4, tag="val", grad=False)
    item = cur[0]
    ok("M2 目标组指向该行的 fact_key",
       fi["keys"][int(fi["tgt_group"])] == fi["tgt_key"] == LM.fact_key(
           item["entity"], item["relation"]), extra=str(fi["tgt_key"]))
    vers = fi["bank"].list_versions(fi["tgt_key"])
    ok("M3 tgt_ver 的时间 == answer_fact_time（当前版本）",
       fi["viable"] and vers[int(fi["tgt_ver"])]["time"] == LM._norm_time(
           item["answer_fact_time"]), extra=f"{vers} / {item['answer_fact_time']}")
    ok("M4 tgt_ver 与 bank.read() 的选择一致（并列/兜底规则同源）",
       fi["bank"].read(fi["tgt_key"], item["answer_fact_time"])
       == vers[int(fi["tgt_ver"])]["slot_id"])
    # 大小写：键是规范化过的（小写），readout 若从键反推就会拿到不同文本
    k0 = fi["keys"][0]
    ok("M5 库里存的是**原始大小写**实体名（与从键反推的规范化文本不同）",
       fi["bank"].group_text(k0)[0] == item["entity"]
       and fi["bank"].group_text(k0)[0] != LM.split_fact_key(k0)[0],
       extra=f"{fi['bank'].group_text(k0)} vs {LM.split_fact_key(k0)}")

    # 训练侧与读出侧必须把**同一份字符串**喂进编码器
    # 两侧各用**独立**缓存：否则读出侧会命中训练侧留下的缓存，根本不 tokenize，
    # 断言就退化成"缓存命中"（我第一版就这么写错了，M6 直接假失败）。
    tok.texts = []
    T.fst_build_inputs(fi, model, tok, 2, torch.device("cpu"), {})
    train_txt = set(tok.texts)
    tok.texts = []
    LM.generate_latent_answer_fst(model=model, tokenizer=tok, bridge=bridge,
                                  bank=fi["bank"], question=item["question"],
                                  system_prompt="Be brief.", fst_router=router,
                                  max_new_tokens=2, layer=2,
                                  query_time=item.get("as_of"), feat_cache={})
    read_txt = set(tok.texts)
    want = {item["entity"], item["relation"]}
    ok("M6 训练侧与读出侧用同一份实体/属性文本（原始大小写）",
       want <= train_txt and want <= read_txt,
       extra=f"train={sorted(train_txt)[:6]} read={sorted(read_txt)[:6]}")
    # 反向确认缓存键口径一致：共享缓存时读出侧不应再 tokenize 实体/属性
    shared = {}
    T.fst_build_inputs(fi, model, tok, 2, torch.device("cpu"), shared)
    tok.texts = []
    LM.generate_latent_answer_fst(model=model, tokenizer=tok, bridge=bridge,
                                  bank=fi["bank"], question=item["question"],
                                  system_prompt="Be brief.", fst_router=router,
                                  max_new_tokens=2, layer=2,
                                  query_time=item.get("as_of"), feat_cache=shared)
    ok("M6b 共享缓存时读出侧不再重算实体/属性特征（缓存键口径一致）",
       not (want & set(tok.texts)), extra=str(sorted(set(tok.texts))[:6]))

    if hist:
        fh = T.build_fst_bank(encoder, hist[0], 4, tag="val", grad=False)
        vh = fh["bank"].list_versions(fh["tgt_key"])
        ok("M7 历史题指向旧版本（不是当前）",
           fh["viable"] and vh[int(fh["tgt_ver"])]["time"]
           == LM._norm_time(hist[0]["answer_fact_time"])
           and int(fh["tgt_ver"]) < len(vh) - 1, extra=str(vh))
        f0 = T.build_fst_bank(encoder, hist[0], 0, tag="val", grad=False)
        ok("M8 hist_max=0 时历史题的目标版本被真正淘汰（evicted=True, viable=False）",
           bool(f0["evicted"]) and not f0["viable"], extra=str(f0["tgt_ver"]))
        ok("M9 hist_max=0 时不误判为'标签缺失'",
           not f0["label_missing"] and not f0["label_mismatch"])


# =====================================================================
# N. 指标口径（纯函数 fst_scores）+ 并列时间戳的监督定位
# =====================================================================
def test_metrics_and_targets():
    import train_latent_memory as T

    def rec(**kw):
        base = {"okq": False, "g_ok": True, "ver_picked": 0, "tgt_ver": 0,
                "viable": True, "tgt_ver_n": 2, "tgt_group": 0, "evicted": False,
                "label_missing": False, "probe": "current", "question": "q?", "gold": "a",
                "n_soft": 4}
        base.update(kw)
        return base

    # N1 版本命中必须绑组别：组选错但版本下标碰巧相同，不得算命中
    bad = [rec(g_ok=False, ver_picked=0, tgt_ver=0)] * 4
    m = T.fst_scores(bad)
    ok("N1 组选错时版本级一律不计命中", m["ver_acc"] == 0.0 and m["ver_multi_acc"] == 0.0,
       extra=str({k: m[k] for k in ("ver_acc", "ver_multi_acc")}))
    # N2 单版本目标组不计入版本级分母（否则恒真样本虚抬）
    mix = [rec(tgt_ver_n=1, ver_picked=0, tgt_ver=0) for _ in range(6)] + \
          [rec(tgt_ver_n=2, ver_picked=1, tgt_ver=0)]
    m2 = T.fst_scores(mix)
    ok("N2 单版本样本不进版本级分母", m2["ver_multi_n"] == 1 and m2["ver_multi_acc"] == 0.0,
       extra=str((m2["ver_multi_n"], m2["ver_multi_acc"])))
    # N3 单成员配对不算通过；gold 相同的配对也不算反事实配对
    m3 = T.fst_scores([rec(question="q1", gold="a", okq=True),
                       rec(question="q2", gold="a", okq=True),
                       rec(question="q2", gold="a", okq=True),
                       rec(question="q3", gold="a", okq=True),
                       rec(question="q3", gold="b", okq=False)])
    ok("N3 单成员/同值配对不计入 pair_acc", m3["pair_n"] == 1 and m3["pair_acc"] == 0.0
       and m3["pair_dropped_n"] == 2,
       extra=str({k: m3[k] for k in ("pair_n", "pair_acc", "pair_dropped_n")}))
    # N4 常数基线：永远选最新版
    m4 = T.fst_scores([rec(tgt_ver_n=2, tgt_ver=1, ver_picked=1)] * 3)
    ok("N4 最新版常数基线", m4["ver_newest_acc"] == 1.0 and m4["ver_multi_acc"] == 1.0)

    # N5 并列时间戳：监督必须取**最新**那条，且与 bank.read() 的选择一致
    model = freeze(StubModel())
    tok = StubTok()
    comp = LM.LatentCompressor(model.config.hidden_size, n_slots=M, inner_dim=64,
                              num_heads=4).eval()
    enc = LM.LatentEncoder(model, tok, comp, layer=2, max_unit_tokens=16)
    item = {"units": ["A works at X.", "A works at Y."], "hops": 1,
            "unit_meta": [{"entity": "A", "relation": "works at", "time": "2025",
                           "fact": "A works at X.", "distractor": False},
                          {"entity": "A", "relation": "works at", "time": "2025",
                           "fact": "A works at Y.", "distractor": False}],
            "question": "Where does A work?", "answer": "Y",
            "evidence": ["A works at Y."], "entity": "A", "relation": "works at",
            "fact_key": "a||works at", "as_of": "", "answer_fact_time": "2025",
            "probe": "current", "timeline": []}
    fi = T.build_fst_bank(enc, item, 4, tag="val", grad=False)
    vers = fi["bank"].list_versions(fi["tgt_key"])
    ok("N5 并列时间戳时监督取最新版本，且与 bank.read() 一致",
       len(vers) == 2 and int(fi["tgt_ver"]) == 1
       and fi["bank"].read(fi["tgt_key"], "2025") == vers[1]["slot_id"],
       extra=f"tgt_ver={fi['tgt_ver']} read={fi['bank'].read(fi['tgt_key'], '2025')}")


def main():
    test_fact_key()
    test_fact_key_encoder()
    test_state_trajectory_bank()
    test_fst_router()
    test_fst_readout()
    test_checkpoint_roundtrip()
    test_alignment()
    test_data_pipeline()
    test_metrics_and_targets()
    print("=" * 70)
    if FAILS:
        print(f"FAILED {len(FAILS)}/{N_OK + len(FAILS)}: {FAILS}")
        return 1
    print(f"ALL PASS ({N_OK} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
