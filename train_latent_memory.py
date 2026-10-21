#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_latent_memory.py — 冻结 Qwen，训练轻量模块（LatentCompressor + LatentBridge）

这是方案一最关键、也最可能失败的部分：**没有训练的压缩器与投影器，即使张量维度完全正确，
也没有理由认为冻结 Qwen 能理解这些连续向量。** 本脚本就是让"读出通路真正学会恢复事实"的
训练过程。

训练范围（与设计一致）
    Qwen3-4B 主体          冻结
    Delta-Mem Adapter      冻结，且关闭在线写入（delta_write_disabled）
    LatentCompressor       训练
    LatentBridge           训练
    原有检索器             不训练（也不在本脚本内）
    潜层门控更新网络       第一阶段不训练、不接入（先做 Hard Overwrite，见 v9_cr 的说明）

两阶段
    阶段 A：单事实读出训练。模型只接收压缩后的 Z 和问题，**不接收 memory 明文**。
           损失 = 只在答案 token 上计算的标准 Causal LM 交叉熵。
           可选附加"事实重建"任务（--recon-weight>0）：只给 Z，让模型重建原事实，
           用来检查专有名词的符号保真能力（不参与主指标）。
    阶段 B：Teacher–Student 自蒸馏。Teacher 用明文证据，Student 用潜载荷 + 同一问题；
           在**相同目标答案位置**上对齐输出分布：L = L_QA + λ·L_KL。
           建议先纯 QA 跑通（--kl-weight 0），再加 KL —— 否则无法判断蒸馏到底有没有用。

精度约定
    Qwen 主体 bf16 冻结；Compressor / Bridge 保持 **fp32**（LATENT_DTYPE 只决定**载荷存储**
    精度，不决定训练精度）。理由：h0/h18 会先被 cast 到参数 dtype 再进投影层，
    小模块用 fp32 训练数值最稳；用 bf16 训练这两个小模块容易在 LayerNorm/softmax 处失稳。

必须守住的实现细节
    1. Student 前向**不能**放进 torch.no_grad()：冻结的是权重，不是梯度路径；
       一旦整段 no_grad，Bridge 与 Compressor 永远学不到东西（loss 会正常下降的假象也不会有，
       只会一直不动）。
    2. 冻结用 requires_grad_(False)，不用 no_grad：前者让 autograd 跳过权重更新但仍能
       把梯度回传到输入侧的 soft tokens。
    3. 编码特征（h0/h18）本身在 no_grad 下取得：模型冻结，特征就是常量。
    4. Teacher 前向必须 no_grad（它只提供目标分布）。

⚠️ "单卡数小时收敛"这类估计不能当结论：冻结模型的反向传播仍有显著显存开销，
   训练时长与显存需要实际 profiling（--profile 会打印参数量/显存峰值）。

用法
    # 只做自检（CPU、小随机模型、不加载 Qwen）：验证数据/梯度/损失通路
    # --dry-run = 跑通一次前向+反向自检后退出，**不写 checkpoint**；
    # 配合 --smoke-model 可在纯 CPU 上完成，不需要 Qwen 权重。
    python train_latent_memory.py --dry-run --stage a --smoke-model

    # 阶段 A（需要 GPU + Qwen 权重）
    LATENT_M=4 LATENT_LAYER=18 LATENT_INNER_DIM=512 LATENT_LR=1e-4 \
      python train_latent_memory.py --stage a --out /share/exp4group/ckpt/latent_a.pt

    # 阶段 B（在阶段 A 的产物上继续，先纯 QA 再开 KL）
    python train_latent_memory.py --stage b --init /share/exp4group/ckpt/latent_a.pt \
      --kl-weight 0.5 --out /share/exp4group/ckpt/latent_b.pt
"""
import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

import latent_memory as LM

BASE = ("/share/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/"
        "cdbee75f17c01a7cc42f958dc650907174af0554")
ADAPTER = ("/share/hf_home/hub/models--declare-lab--delta-mem_qwen3_4b-instruct/snapshots/"
           "6a1a990e770974811b8fb27dab73430807f76cb5")


# =====================================================================
# 1) 合成训练数据：实体 / 数字 / 日期 / 反事实关系
# =====================================================================
_FIRST = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Heidi",
          "Ivan", "Judy", "Karl", "Linda", "Mona", "Nate", "Olga", "Pete",
          "Quinn", "Rita", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier"]
_LAST = ["Smith", "Jones", "Brown", "Clark", "Davis", "Evans", "Foster", "Green"]
_CITY = ["Kyoto", "Lisbon", "Quito", "Riga", "Osaka", "Porto", "Tbilisi", "Utrecht",
         "Valencia", "Wroclaw", "Zagreb", "Bergen"]
_ORG = ["Northwind", "Contoso", "Fabrikam", "Initech", "Umbrella", "Vandelay",
        "Soylent", "Globex", "Hooli", "Massive Dynamic"]
_COLOR = ["teal", "crimson", "amber", "indigo", "ochre", "violet"]
_ANIMAL = ["otter", "lynx", "heron", "ibex", "marmot", "puffin"]

# (属性键, 值池, 事实模板, 问题模板)
_ATTRS = [
    ("favorite_city", _CITY, "{E}'s favorite city is {V}.", "What is {E}'s favorite city?"),
    ("employer", _ORG, "{E} works at {V}.", "Where does {E} work?"),
    ("favorite_color", _COLOR, "{E}'s favorite color is {V}.", "What is {E}'s favorite color?"),
    ("pet", _ANIMAL, "{E} keeps a pet {V}.", "What pet does {E} keep?"),
]
_NUM_ATTRS = [
    ("shares", "{E} owns {V} shares of {O}.", "How many shares of {O} does {E} own?"),
]
_DATE_ATTRS = [
    ("birth_year", "{E} was born in {V}.", "When was {E} born?"),
    ("move_year", "{E} moved to {C} in {V}.", "When did {E} move to {C}?"),
]


def _name(rng):
    return f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"


def make_dataset(n, rng, entities, value_shift=0, counterfactual=False):
    """生成 n 条 (memory, question, answer)。

    - entities：本 split 允许使用的实体（train/val 用**不相交**的实体集合）；
    - value_shift：值池偏移（val 用训练集没出现过的具体值，防止 Reader 靠背答案得分）；
    - counterfactual：同一实体同一属性生成两条互相矛盾的记忆（只应记住后一条）。
      这种样本只用于阶段 B 的"反事实对照"，answer 取后一条。
    """
    out = []
    for _ in range(n):
        e = rng.choice(entities)
        kind = rng.random()
        if kind < 0.15:
            _key, tpl, q = rng.choice(_NUM_ATTRS)
            o = rng.choice(_ORG)
            v = str(rng.randint(10, 9999) + value_shift)
            mem = tpl.format(E=e, V=v, O=o)
            question = q.format(E=e, O=o)
            ans = v
            old = str(int(v) + rng.randint(1, 50))
            old_mem = tpl.format(E=e, V=old, O=o)
        elif kind < 0.30:
            _key, tpl, q = rng.choice(_DATE_ATTRS)
            c = rng.choice(_CITY)
            y = rng.randint(1930, 2020) + (value_shift % 7)
            mem = tpl.format(E=e, V=y, C=c)
            question = q.format(E=e, C=c)
            ans = str(y)
            old = str(int(y) - rng.randint(1, 12))
            old_mem = tpl.format(E=e, V=old, C=c)
        else:
            _key, pool, tpl, q = rng.choice(_ATTRS)
            v = pool[(rng.randrange(len(pool)) + value_shift) % len(pool)]
            mem = tpl.format(E=e, V=v)
            question = q.format(E=e)
            ans = v
            cand = [x for x in pool if x != v]
            old = rng.choice(cand) if cand else v
            old_mem = tpl.format(E=e, V=old)
        if counterfactual and old != ans:
            # 同一主语同一属性的旧值 + 新值：只有新值有效（与 CR 的覆写语义一致）
            out.append({"memory": f"{old_mem} {mem}", "question": question,
                        "answer": ans, "cf": True})
            continue
        out.append({"memory": mem, "question": question, "answer": ans, "cf": False})
    return out


def split_entities(seed=0):
    """train / val 使用**不相交**的实体集合（验证集不能是训练实体的同义重复）。"""
    rng = random.Random(seed)
    names = sorted({_name(rng) for _ in range(200)})
    rng.shuffle(names)
    half = len(names) // 2
    return names[:half], names[half:]


# =====================================================================
# 2) 序列构造：soft tokens + 答案 token 的标签
# =====================================================================
def build_student_inputs(model, tokenizer, soft, system_prompt, question, answer):
    """student：system(正常 token) + soft tokens + user/assistant 骨架 + answer token。

    返回 (embeds [1,L,d], labels [1,L], n_ans)。labels 只有答案 token 非 -100。
    """
    dev = LM._model_device(model)
    emb_layer = model.get_input_embeddings()
    full, n_sys_chars = LM.build_chat_parts(tokenizer, system_prompt, question)
    ids_full = tokenizer(full, return_tensors="pt",
                         add_special_tokens=False)["input_ids"].to(dev)
    n_sys = 0
    if n_sys_chars:
        ids_pre = tokenizer(full[:n_sys_chars], return_tensors="pt",
                            add_special_tokens=False)["input_ids"].to(dev)
        n_sys = int(ids_pre.shape[1])
        if ids_full[0, :n_sys].tolist() != ids_pre[0].tolist():
            raise RuntimeError("system 段分词边界不一致，无法确定 soft token 插入点")
    ans_ids = tokenizer(str(answer), return_tensors="pt",
                        add_special_tokens=False)["input_ids"].to(dev)
    pre = emb_layer(ids_full[:, :n_sys])
    post = emb_layer(ids_full[:, n_sys:])
    ans_emb = emb_layer(ans_ids)
    parts = [pre]
    if soft is not None and soft.numel():
        s = soft
        if s.dim() == 2:
            s = s.unsqueeze(0)
        s = s.reshape(1, -1, s.shape[-1]).to(dtype=pre.dtype, device=dev)
        parts.append(s)
    parts += [post, ans_emb]
    embeds = torch.cat(parts, dim=1)
    n_ans = int(ans_ids.shape[1])
    labels = torch.full((1, embeds.shape[1]), -100, dtype=torch.long, device=dev)
    labels[0, -n_ans:] = ans_ids[0]
    return embeds, labels, n_ans


def build_teacher_inputs(model, tokenizer, evidence, system_prompt, question, answer):
    """teacher：明文证据进 user 消息（只用于提供目标分布，全程 no_grad）。"""
    dev = LM._model_device(model)
    emb_layer = model.get_input_embeddings()
    user = f"Evidence: {evidence}\n\nQuestion: {question}" if evidence else question
    msgs = []
    if (system_prompt or "").strip():
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user})
    full = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(full, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
    ans_ids = tokenizer(str(answer), return_tensors="pt",
                        add_special_tokens=False)["input_ids"].to(dev)
    embeds = torch.cat([emb_layer(ids), emb_layer(ans_ids)], dim=1)
    n_ans = int(ans_ids.shape[1])
    return embeds, n_ans


def answer_logits(logits, n_ans):
    """取"预测答案 token"的那几个位置：位置 L-n_ans-1 .. L-2。

    与 build_*_inputs 的拼接顺序绑定（答案在序列末尾）。用 -1 的相对下标而不是绝对
    位置，是因为 teacher / student 的 prompt 长度不同，只能在**答案 token 序号**上对齐。
    """
    L = int(logits.shape[1])
    if n_ans <= 0 or L < n_ans + 1:
        raise ValueError(f"序列长度 {L} 不足以容纳 {n_ans} 个答案 token")
    return logits[0, L - n_ans - 1:L - 1]


# =====================================================================
# 3) 冻结 / 小模型自检
# =====================================================================
def freeze_model(model):
    """冻结全部权重，但保留梯度路径（不做 no_grad）。返回可训练参数数（应为 0）。"""
    n = 0
    for p in model.parameters():
        p.requires_grad_(False)
        n += p.numel()
    model.eval()
    return n


class SmokeModel(nn.Module):
    """--smoke-model：极小随机模型，用来在 CPU 上验证数据/梯度/损失通路（不是 Qwen）。

    行为上尽量贴近真实调用：有 get_input_embeddings()、config.hidden_size、
    接受 inputs_embeds + attention_mask 并返回带 .logits 的对象。
    """

    class _Cfg:
        hidden_size = 64
        num_hidden_layers = 2
        vocab_size = 512

    class _Out:
        def __init__(self, logits):
            self.logits = logits
            self.hidden_states = None

    def __init__(self, vocab_size=512, hidden_size=64, n_layers=2, n_heads=4):
        super().__init__()
        self.config = SmokeModel._Cfg()
        self.config.vocab_size = vocab_size
        self.config.hidden_size = hidden_size
        self.config.num_hidden_layers = n_layers
        self.emb = nn.Embedding(vocab_size, hidden_size)
        layer = nn.TransformerEncoderLayer(hidden_size, n_heads, 4 * hidden_size,
                                           batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                output_hidden_states=False, use_cache=False, **kw):
        x = inputs_embeds if inputs_embeds is not None else self.emb(x)
        if attention_mask is not None and attention_mask.shape[1] == x.shape[1]:
            pad = (1.0 - attention_mask.to(x.dtype)).unsqueeze(-1)
            x = x * (1.0 - pad)
        h = self.body(x)
        return SmokeModel._Out(self.head(h))


class StubTok:
    """--smoke-model 配套的字符级 tokenizer（只求能跑通，不代表 Qwen 分词）。"""

    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size
        self.eos_token_id = 1
        self.pad_token_id = 0

    def _ids(self, text):
        return [(sum(ord(c) for c in w) % (self.vocab_size - 8)) + 4
                for w in (str(text).split() or ["<empty>"])]

    def __call__(self, texts, return_tensors=None, add_special_tokens=True,
                 padding=False, truncation=False, max_length=None):
        if isinstance(texts, str):
            texts = [texts]
        seqs = [self._ids(t) for t in texts]
        if truncation and max_length:
            seqs = [s[:max_length] for s in seqs]
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
            out += f"<{m['role']}> {m['content']} </{m['role']}> "
        if add_generation_prompt:
            out += "<assistant> "
        return out

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


# =====================================================================
# 4) 训练
# =====================================================================
def qa_loss(logits, labels):
    """标准 Causal LM 交叉熵，只对答案 token（labels != -100）计算。"""
    shift_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].reshape(-1)
    sel = shift_labels != -100
    if not bool(sel.any()):
        raise ValueError("本 batch 没有可训练的答案 token（labels 全是 -100）")
    return F.cross_entropy(shift_logits[sel], shift_labels[sel])


def kl_loss(student_logits, teacher_logits, n_ans, tau=1.0):
    """答案位置上的分布对齐（逐行按答案 token 序号对齐，不按绝对位置）。"""
    losses = []
    for b, n in enumerate(n_ans):
        if n <= 0:
            continue
        s = answer_logits(student_logits[b:b + 1], n)
        t = answer_logits(teacher_logits[b:b + 1], n)
        losses.append(F.kl_div(F.log_softmax(s / tau, dim=-1),
                               F.softmax(t / tau, dim=-1),
                               reduction="batchmean") * (tau * tau))
    if not losses:
        return None
    return torch.stack(losses).mean()


def evaluate(reader_like, model, tokenizer, data, system_prompt, limit=32,
             layer=LM.LATENT_LAYER, max_unit_tokens=LM.LATENT_MAX_UNIT_TOKENS):
    """用潜层读出（generate_latent_answer）在验证集上算 exact-match，仅供训练期观察。"""
    if not hasattr(model, "generate"):
        return None, 0
    comp, bridge = reader_like
    comp.eval()
    bridge.eval()
    enc = LM.LatentEncoder(model, tokenizer, comp, layer=layer,
                           max_unit_tokens=max_unit_tokens)
    hit = 0
    n = 0
    for item in data[:limit]:
        payloads, _info = enc.encode([item["memory"]], parent_ids=["val:0"])
        z = payloads[0]
        pred, _meta = LM.generate_latent_answer(
            model=model, tokenizer=tokenizer, bridge=bridge, payloads=z,
            question=item["question"], system_prompt=system_prompt, max_new_tokens=16)
        n += 1
        if str(item["answer"]).strip().lower() in str(pred).strip().lower():
            hit += 1
    return (hit / n if n else None), n


def train(args):
    t0 = time.time()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = torch.device(args.device if args.device else
                       ("cpu" if args.smoke_model else
                        ("cuda" if torch.cuda.is_available() else "cpu")))

    # ---- 模型 ----
    if args.smoke_model:
        model = SmokeModel().to(dev)
        tokenizer = StubTok()
        hidden = model.config.hidden_size
        print(f"[smoke] 使用极小随机模型（hidden={hidden}, vocab={model.config.vocab_size}）："
              f"只验证通路，不代表 Qwen 行为", flush=True)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base)
        model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, device_map=f"cuda:{args.gpu}",
            attn_implementation="sdpa")
        if not args.no_adapter:
            from deltamem.core import HFDeltaMemConfig, attach_delta_mem
            from deltamem.core.delta_impl import load_delta_mem_adapter
            cfg = HFDeltaMemConfig.from_pretrained(args.adapter)
            attach_delta_mem(model, cfg)
            load_delta_mem_adapter(model, input_dir=args.adapter)
        dev = torch.device(f"cuda:{args.gpu}")
        hidden = LM._model_hidden_size(model)
    n_frozen = freeze_model(model)
    print(f"[freeze] 冻结参数 {n_frozen/1e6:.1f}M（Delta 写入同时关闭）", flush=True)

    # ---- 可训练模块 ----
    comp = LM.LatentCompressor(hidden, n_slots=args.m, inner_dim=args.inner_dim,
                               num_heads=args.heads).to(dev)
    bridge = LM.LatentBridge(hidden, bottleneck=args.bottleneck).to(dev)
    if args.init:
        blob = LM._safe_torch_load(args.init)
        comp.load_state_dict(blob["compressor"])
        bridge.load_state_dict(blob["bridge"])
        print(f"[init] 从 {args.init} 初始化（阶段 B 常规做法）", flush=True)
    n_train = sum(p.numel() for p in comp.parameters()) + \
              sum(p.numel() for p in bridge.parameters())
    print(f"[params] 可训练 {n_train/1e6:.3f}M "
          f"(compressor={sum(p.numel() for p in comp.parameters())/1e6:.3f}M, "
          f"bridge={sum(p.numel() for p in bridge.parameters())/1e6:.3f}M)", flush=True)

    # ---- 数据 ----
    ent_train, ent_val = split_entities(args.seed)
    train_data = make_dataset(args.n_train, random.Random(args.seed + 1), ent_train,
                              value_shift=0, counterfactual=args.counterfactual)
    val_data = make_dataset(args.n_val, random.Random(args.seed + 2), ent_val,
                            value_shift=3, counterfactual=False)
    print(f"[data] train={len(train_data)} val={len(val_data)} "
          f"entities(train/val)={len(ent_train)}/{len(ent_val)}（不相交）", flush=True)
    print(f"[data] 例：memory={train_data[0]['memory']!r} "
          f"q={train_data[0]['question']!r} a={train_data[0]['answer']!r}", flush=True)

    if args.dry_run:
        print("[dry-run] 数据/模型/参数已就绪，开始一次前向+反向自检（不写 checkpoint）",
              flush=True)

    opt = torch.optim.AdamW([p for p in list(comp.parameters()) + list(bridge.parameters())
                             if p.requires_grad], lr=args.lr)
    encoder = LM.LatentEncoder(model, tokenizer, comp, layer=args.layer,
                               max_unit_tokens=args.max_unit_tokens, device=dev)
    system_prompt = args.system_prompt
    max_steps = args.max_steps if args.max_steps > 0 else max(1, len(train_data) // args.accum)
    comp.train()
    bridge.train()
    step = 0
    micro = 0
    acc_items = []
    grad_seen = None        # 必须在 opt.zero_grad() 之前取样，否则永远是 None -> 误报失败
    t_start = time.time()
    for epoch in range(args.epochs):
        order = list(range(len(train_data)))
        random.Random(args.seed + 100 + epoch).shuffle(order)
        for i in order:
            item = train_data[i]
            # ① 压缩（grad=True：保留到压缩器输出的梯度路径）
            payloads, enc_info = encoder.encode([item["memory"]], parent_ids=["fact:0"],
                                                 grad=True)
            z = payloads[0]
            # ② student 前向：**不加 no_grad**（冻结的是权重，不是梯度）
            embeds, labels, n_ans = build_student_inputs(
                model, tokenizer, z, system_prompt, item["question"], item["answer"])
            with LM.delta_write_disabled(model):
                s_logits = model(inputs_embeds=embeds,
                                 attention_mask=torch.ones(embeds.shape[:2],
                                                           dtype=torch.long, device=dev),
                                 use_cache=False).logits
            loss_qa = qa_loss(s_logits, labels)
            loss = loss_qa
            loss_kl = None
            if args.stage == "b" and args.kl_weight > 0:
                # ③ teacher：明文证据，只提供目标分布
                t_embeds, t_n = build_teacher_inputs(
                    model, tokenizer, item["memory"], system_prompt,
                    item["question"], item["answer"])
                with torch.no_grad():
                    t_logits = model(inputs_embeds=t_embeds,
                                     attention_mask=torch.ones(t_embeds.shape[:2],
                                                               dtype=torch.long, device=dev),
                                     use_cache=False).logits
                loss_kl = kl_loss(s_logits, t_logits, [min(n_ans, t_n)], tau=args.kl_tau)
                if loss_kl is not None:
                    loss = loss_qa + args.kl_weight * loss_kl
            if args.recon_weight > 0:
                # 事实重建辅助任务：只给 Z，重建原事实（检查专有名词符号保真）
                r_embeds, r_labels, _ = build_student_inputs(
                    model, tokenizer, z, system_prompt,
                    "Repeat the memory verbatim.", item["memory"])
                with LM.delta_write_disabled(model):
                    r_logits = model(inputs_embeds=r_embeds,
                                     attention_mask=torch.ones(r_embeds.shape[:2],
                                                               dtype=torch.long, device=dev),
                                     use_cache=False).logits
                loss = loss + args.recon_weight * qa_loss(r_logits, r_labels)
            (loss / args.accum).backward()
            acc_items.append((float(loss_qa), None if loss_kl is None else float(loss_kl)))
            micro += 1
            if micro % args.accum == 0:
                if grad_seen is None:
                    # 第一个梯度累积边界处取样：此时 .grad 已填充、尚未 step/zero_grad
                    grad_seen = {
                        "compressor_queries": comp.queries.grad is not None,
                        "compressor_input_proj": comp.input_proj.weight.grad is not None,
                        "bridge_last_linear": bridge.net[3].weight.grad is not None,
                        "model_has_no_grad": all(p.grad is None for p in model.parameters()),
                    }
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(comp.parameters()) + list(bridge.parameters()), args.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                qa_avg = sum(x[0] for x in acc_items) / len(acc_items)
                kls = [x[1] for x in acc_items if x[1] is not None]
                kl_avg = sum(kls) / len(kls) if kls else None
                print(f"[step {step}] loss_qa={qa_avg:.4f} "
                      f"loss_kl={'n/a' if kl_avg is None else f'{kl_avg:.4f}'} "
                      f"soft_tokens={enc_info['n_soft_tokens']} "
                      f"compress={enc_info['compression_ratio']} "
                      f"({time.time()-t_start:.0f}s)", flush=True)
                acc_items = []
                if args.max_steps > 0 and step >= max_steps:
                    break
        if args.max_steps > 0 and step >= max_steps:
            break

    # ---- 梯度自检：Compressor / Bridge 必须真的拿到梯度，Qwen 必须没有 ----
    if grad_seen is None:
        raise RuntimeError("训练没有完成任何一个优化步：检查 --accum/--max-steps/--epochs")
    grad_ok = dict(grad_seen)
    print(f"[grad-check] {json.dumps(grad_ok, ensure_ascii=False)}", flush=True)
    if not all(v for k, v in grad_ok.items() if k != "model_has_no_grad"):
        raise RuntimeError(
            "梯度没有回传到 Compressor/Bridge —— 检查是否误把 student 前向放进了 "
            "torch.no_grad()，或 z 被提前 detach。")
    if not grad_ok["model_has_no_grad"]:
        print("[grad-check][warn] 冻结权重拿到了梯度：请确认 requires_grad_(False) 生效",
              flush=True)

    em, n_eval = evaluate((comp, bridge), model, tokenizer, val_data, system_prompt,
                          limit=args.eval_limit, layer=args.layer,
                          max_unit_tokens=args.max_unit_tokens)
    print(f"[eval] val exact-match={em if em is None else round(em, 4)} (n={n_eval}) "
          f"—— 训练期观察值，不是论文指标", flush=True)

    if args.dry_run:
        print(f"[dry-run] OK：通路验证完成（{time.time()-t0:.1f}s），未写 checkpoint",
              flush=True)
        return 0

    cfg = {"schema": LM.LATENT_SCHEMA, "n_slots": args.m, "layer": args.layer,
           "inner_dim": args.inner_dim, "num_heads": args.heads,
           "bottleneck": args.bottleneck, "max_unit_tokens": args.max_unit_tokens,
           "read_budget": LM.LATENT_READ_BUDGET, "dtype": LM.LATENT_DTYPE,
           "stage": args.stage, "kl_weight": args.kl_weight, "kl_tau": args.kl_tau,
           "recon_weight": args.recon_weight, "lr": args.lr,
           "steps": step, "epochs": args.epochs, "seed": args.seed,
           "smoke_model": bool(args.smoke_model)}
    meta = {"trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git": _git_commit(), "n_train": len(train_data), "n_val": len(val_data),
            "val_exact_match": em, "n_trainable_params": n_train,
            "grad_check": grad_ok, "elapsed_sec": round(time.time() - t0, 1)}
    out = LM.save_latent_checkpoint(args.out, comp, bridge, cfg, meta)
    print(f"[save] {out} config={json.dumps(cfg, ensure_ascii=False)}", flush=True)
    return 0


def _git_commit():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))
                                       ).decode().strip()
    except Exception:
        return None


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="训练潜层记忆压缩器 + 桥接器（冻结 Qwen）")
    ap.add_argument("--stage", choices=["a", "b"], default="a")
    ap.add_argument("--out", default="/share/exp4group/ckpt/latent.pt")
    ap.add_argument("--init", default="", help="阶段 B 从阶段 A 的 checkpoint 初始化")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--no-adapter", action="store_true", help="不挂 Delta-Mem adapter（纯 Qwen）")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--device", default="", help="显式设备（smoke 用 cpu）")
    ap.add_argument("--smoke-model", action="store_true", help="极小随机模型，CPU 自检通路")
    ap.add_argument("--dry-run", action="store_true", help="跑通自检，不写 checkpoint")
    # 结构 / 训练超参（与设计表一致）
    ap.add_argument("--m", type=int, default=LM.LATENT_M)
    ap.add_argument("--layer", type=int, default=LM.LATENT_LAYER)
    ap.add_argument("--inner-dim", type=int, default=LM.LATENT_INNER_DIM)
    ap.add_argument("--heads", type=int, default=LM.LATENT_N_HEADS)
    ap.add_argument("--bottleneck", type=int, default=LM.LATENT_BOTTLENECK)
    ap.add_argument("--max-unit-tokens", type=int, default=LM.LATENT_MAX_UNIT_TOKENS)
    ap.add_argument("--lr", type=float, default=float(os.environ.get("LATENT_LR", "1e-4")))
    ap.add_argument("--accum", type=int,
                    default=int(os.environ.get("LATENT_GRAD_ACCUM", "16")))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--kl-weight", type=float, default=0.0)
    ap.add_argument("--kl-tau", type=float, default=1.0)
    ap.add_argument("--recon-weight", type=float, default=0.0)
    ap.add_argument("--counterfactual", action="store_true",
                    help="训练集加入『同主语同属性冲突，只记新值』样本")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--eval-limit", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--system-prompt",
                    default="You are a helpful assistant. Answer with the shortest exact value.")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"[cfg] stage={args.stage} m={args.m} layer={args.layer} "
          f"inner={args.inner_dim} heads={args.heads} bottleneck={args.bottleneck} "
          f"max_unit_tokens={args.max_unit_tokens} lr={args.lr} accum={args.accum} "
          f"kl={args.kl_weight} recon={args.recon_weight} "
          f"smoke={args.smoke_model} dry_run={args.dry_run}", flush=True)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
