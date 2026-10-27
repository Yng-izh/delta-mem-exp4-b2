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
import json
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
# 1.5) 外部语料：tools/gen_synth_qa.py 产出的 jsonl
# =====================================================================
def load_synth_jsonl(path, limit=0, seed=0):
    """读 `tools/gen_synth_qa.py` 产出的 jsonl，转成训练用的 item。

    每条记录：memory(事实列表) / question / answer / evidence / meta.cluster。
    转换后的 item 同时保留两种粒度：
      - `units`：**每条事实一个记忆单元**（一个事实一个槽位，与推理期 bank 的布局一致）；
      - `memory`：事实拼接成的单串（teacher 明文输入、事实重建任务用）。
    没有 `units` 的 item（内部 make_dataset）走单单元路径，行为不变。

    为什么按事实切槽而不是整段压成一个槽：推理期每个槽位就是一条事实，
    整段压缩会让"检索到某一条事实"在训练里根本不存在，训练/推理布局不一致。
    """
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            mem = rec.get("memory") or rec.get("evidence") or []
            if isinstance(mem, str):
                mem = [mem]
            mem = [str(m).strip() for m in mem if str(m).strip()]
            if not mem or rec.get("question") in (None, "") or rec.get("answer") in (None, ""):
                continue
            meta = rec.get("meta") or {}
            ev = rec.get("evidence") or []
            if isinstance(ev, str):
                ev = [ev]
            items.append({
                "memory": " ".join(mem),
                "units": mem,
                "question": str(rec["question"]).strip(),
                "answer": str(rec["answer"]).strip(),
                # evidence = 目标事实（唯一能唯一定位"哪个槽该被选中"的字段）。
                # 丢掉它就只能靠答案串匹配，而 22.5% 的样本里答案值在干扰事实里
                # 也出现过，实测会把 13.5% 的寻址标签标错 —— router 直接被教歪。
                "evidence": [str(e).strip() for e in ev if str(e).strip()],
                "cf": bool(rec.get("cf", False)),
                "hops": int(rec.get("hops") or 1),
                "cluster": meta.get("cluster", (meta.get("clusters") or [None])[0]),
            })
    if limit and len(items) > limit:
        rng = random.Random(seed)
        idx = list(range(len(items)))
        rng.shuffle(idx)
        keep = sorted(idx[:limit])
        items = [items[i] for i in keep]
    return items


def check_split_leakage(train_items, val_items):
    """训练/验证切分必须干净，否则验证指标没有意义。返回问题列表（空=通过）。"""
    probs = []
    tc = {it["cluster"] for it in train_items if it.get("cluster") is not None}
    vc = {it["cluster"] for it in val_items if it.get("cluster") is not None}
    if tc and vc:
        inter = tc & vc
        if inter:
            probs.append(f"实体簇重叠 {len(inter)} 个（如 {sorted(inter)[:3]}）")
    same_q = ({it["question"].lower() for it in train_items} &
              {it["question"].lower() for it in val_items})
    if same_q:
        probs.append(f"问句完全相同 {len(same_q)} 条（如 {sorted(same_q)[:2]}）")
    return probs


def item_units(item):
    """取一条样本的记忆单元：外部数据按事实切槽，内部数据整段一槽。"""
    units = item.get("units")
    if units:
        return [str(u) for u in units]
    return [str(item.get("memory") or "")]


def encode_item(encoder, item, tag="fact", grad=False):
    """编码一条样本的全部记忆单元，拼成一段 soft token 序列。

    多单元时沿**块轴**拼接：每个 payload 是 [k, m, d]，cat 后仍是 [Σk, m, d]，
    正好等价于"把多条事实的软 token 接在同一个位置上"。
    """
    units = item_units(item)
    payloads, info = encoder.encode(units, parent_ids=[f"{tag}:{k}" for k in range(len(units))],
                                    grad=grad)
    if not payloads:
        raise RuntimeError("encode 未返回任何 payload")
    z = torch.cat(payloads, dim=0) if len(payloads) > 1 else payloads[0]
    return z, info


# =====================================================================
# 2) 序列构造：soft tokens + 答案 token 的标签
# =====================================================================
def encode_item_slots(encoder, item, tag="fact", grad=False):
    """编码一条样本，返回 (per-slot payload 列表, 拼接后的 z, info)。

    为什么要 per-slot：LatentRouter 要在**槽位集合**上打分，需要每个槽独立成形
    [m, d]；encode_item 只给拼好的 [Σk, m, d]，把槽边界丢了，无法路由。
    """
    units = item_units(item)
    payloads, info = encoder.encode(units,
                                    parent_ids=[f"{tag}:{k}" for k in range(len(units))],
                                    grad=grad)
    if not payloads:
        raise RuntimeError("encode 未返回任何 payload")
    z = torch.cat(payloads, dim=0) if len(payloads) > 1 else payloads[0]
    return payloads, z, info


_QF_CACHE: Dict[Any, torch.Tensor] = {}


def query_feature_cached(model, tokenizer, question, system_prompt, layer, device):
    """问题特征缓存：同一问题在训练里会被反复采样，重复跑 L0/L18 前向纯属浪费。

    缓存键含 system_prompt 与 layer —— 换了提示词或取层，特征必须重算（否则会拿
    另一套配置的查询向量去和当前键打分，路由悄悄错位而指标看不出来）。
    """
    key = (str(question), str(system_prompt), int(layer), str(device))
    hit = _QF_CACHE.get(key)
    if hit is None:
        hit = LM.extract_query_feature(model, tokenizer, question, system_prompt,
                                       layer=layer, device=device)
        _QF_CACHE[key] = hit.detach()
    return _QF_CACHE[key]


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
    # 答案末尾必须监督 EOS：否则模型从没被教过"答完就停"，推理时贪心解码会继续编
    # （实测失败模式：Velcor->Velcora、Kestmor->Kestmorville、Elian->Elian Cantonendon...，
    # 实体都对但拖尾巴，严格匹配全错）。重建任务同样受益（教它在记忆末尾停）。
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        eos_t = torch.tensor([[eos_id]], device=dev)
        ans_ids = torch.cat([ans_ids, eos_t], dim=1)
        ans_emb = torch.cat([ans_emb, emb_layer(eos_t)], dim=1)
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
    # teacher 必须和 student 一样监督 EOS：student 的答案段是 [answer..., EOS]，
    # teacher 少了 EOS 会让两边答案长度差 1，kl_loss 只能按 min(n) 截，最后一个
    # 位置的分布（也就是"什么时候停"）永远蒸不到 —— 表现为蒸馏了但生成还在拖尾巴。
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        eos_t = torch.tensor([[eos_id]], device=dev)
        ans_ids = torch.cat([ans_ids, eos_t], dim=1)
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
        def __init__(self, logits, hidden_states=None):
            self.logits = logits
            self.hidden_states = hidden_states

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
        # 原来写成 self.emb(x)：input_ids 路径必然 UnboundLocalError。
        # 这条路径只在 --smoke-model 自检里走，之前从未执行过，所以没暴露。
        x = inputs_embeds if inputs_embeds is not None else self.emb(input_ids)
        if attention_mask is not None and attention_mask.shape[1] == x.shape[1]:
            pad = (1.0 - attention_mask.to(x.dtype)).unsqueeze(-1)
            x = x * (1.0 - pad)
        h = self.body(x)
        # 潜层压缩要读 token 级隐藏状态：hs[0]=embedding，hs[-1]=末层。
        # 之前恒为 None，--smoke-model 一进编码就报 "没有返回 hidden_states"。
        hs = (x, h) if output_hidden_states else None
        return SmokeModel._Out(self.head(h), hs)


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


def _norm_ans(t: str) -> str:
    """答案规范化：小写、去标点、压空白。配对判定和 exact-match 都用同一个口径。"""
    t = str(t).strip().lower()
    for ch in ".,!?;:\"'()[]{}":
        t = t.replace(ch, " ")
    return " ".join(t.split())


def target_slot_index(units, answer, fallback=0, evidence=None):
    """目标槽位下标：优先按 evidence（目标事实原文）精确匹配，退化才用答案串。

    为什么不能只用答案串：干扰事实常常**恰好包含同一个值**。实测
    latent_exp2_train 里 45/200（22.5%）的样本答案值在 >1 条事实中出现，
    只用答案串匹配会把 27/200（13.5%）的寻址标签标到错误的槽位上 ——
    loss_addr 于是把 router 往错的方向训，且训练损失照样下降，看不出来。

    为什么也不用"第 0 条就是目标"：数据生成会 shuffle，位置本身是随机的。
    """
    if isinstance(answer, (list, tuple)):
        answer = " ".join(str(x) for x in answer)
    ev = evidence or []
    if isinstance(ev, str):
        ev = [ev]
    ev_n = {_norm_ans(e) for e in ev if str(e).strip()}
    if ev_n:
        for i, u in enumerate(units):
            if _norm_ans(u) in ev_n:
                return i
    a = _norm_ans(answer)
    if a:
        for i, u in enumerate(units):
            if a in _norm_ans(u):
                return i
    return int(fallback)


def sample_slot_subset(payloads, target_idx, rng, max_slots=None, keep_all=False):
    """随机取 K_cur ∈ [1, K] 个槽位（**必含目标槽**）、随机顺序，返回 (slot_z, new_tgt)。

    这是 Phase 2 的核心课程：模型必须学会"无论给了几个槽、目标排第几，都按问题
    选到同一个槽"。固定 4 槽训练只会学到"第 0 位是答案"这个位置先验（实测：
    训练集 39.5%，把目标挪到最后掉到 21.5%，只给 1 槽直接 0%）。
    """
    K = len(payloads)
    if K <= 0:
        raise ValueError("sample_slot_subset 收到空 payload 列表")
    tgt = int(target_idx) if 0 <= int(target_idx) < K else 0
    if keep_all:
        # 多跳样本：链上的每一跳都必须留着，随机丢槽会把推理链剪断，
        # 模型只会学到"单槽里正好有答案"。顺序仍然打乱，位置无关性照样训练。
        keep = list(range(K))
        rng.shuffle(keep)
        return (LM.stack_payloads([payloads[i] for i in keep]), keep.index(tgt), keep)
    hi = K if max_slots is None else max(1, min(K, int(max_slots)))
    k_cur = rng.randint(1, hi)
    others = [i for i in range(K) if i != tgt]
    rng.shuffle(others)
    keep = [tgt] + others[:max(0, k_cur - 1)]
    rng.shuffle(keep)
    new_tgt = keep.index(tgt)
    slot_z = LM.stack_payloads([payloads[i] for i in keep])       # [K_cur, m, d]
    return slot_z, new_tgt, keep


def evaluate(reader_like, model, tokenizer, data, system_prompt, limit=32,
             layer=LM.LATENT_LAYER, max_unit_tokens=LM.LATENT_MAX_UNIT_TOKENS,
             verbose=False, label="eval", bridge_only=False):
    """潜层读出生成评估：**规范化后的完整答案匹配**（不再是子串匹配）。

    为什么改成完整匹配：子串匹配会把 "Ironwood Energy Nilork, a" 判成对
    （正确答案 "Ironwood Energy Nilork" 是它的子串），系统性高估。
    规范化 = 小写 + 去标点 + 压空白，且要求整段生成与答案一致。

    反事实配对：按规范化问题分组（同问题不同记忆值），一组内**全部**答对才算
    配对正确 —— 这是"模型确实在读记忆"的判据，纯背 Question→Answer 永远做不到。
    """
    if not hasattr(model, "generate"):
        return None, 0
    comp, bridge = reader_like[0], reader_like[1]
    router = reader_like[2] if len(reader_like) > 2 else None
    comp.eval()
    bridge.eval()
    if router is not None:
        router.eval()
    enc = None if bridge_only else LM.LatentEncoder(
        model, tokenizer, comp, layer=layer, max_unit_tokens=max_unit_tokens)
    hit = 0
    n = 0
    groups = {}                     # 规范化问题 -> [该组各条是否全对]
    for item in data[:limit]:
        if bridge_only:
            z, _info = encode_item_raw_embed(model, tokenizer, item)
        else:
            # 必须传 **per-slot 列表**：encode_item 返回的是 cat 后的 [Σk,m,d]，
            # 槽位一旦被切成多块（unit 超 max_unit_tokens），Σk>槽数，generate 会把
            # "块"当成"槽"去路由，选中半条事实。列表形式才能把边界带过去。
            z, _zcat, _info = encode_item_slots(enc, item, tag="val")
        pred, _meta = LM.generate_latent_answer(
            model=model, tokenizer=tokenizer, bridge=bridge, payloads=z,
            question=item["question"], system_prompt=system_prompt, max_new_tokens=16,
            router=router, layer=layer,
            # 多跳样本必须走链式选槽；否则只读第一跳，两跳题永远不可能对
            hops=max(1, int(item.get("hops", 1) or 1)))
        n += 1
        gold = item["answer"]
        if isinstance(gold, list):
            gold = " ".join(str(g) for g in gold)
        ok = _norm_ans(pred) == _norm_ans(gold)
        hit += ok
        key = _norm_ans(item["question"])
        groups.setdefault(key, []).append(ok)
        if verbose:
            print(f"  [{label}] {'✓' if ok else '✗'} q={item['question']!r} "
                  f"gold={gold!r} pred={str(pred)[:60]!r}", flush=True)
    # 反事实配对：一组（同问题）内全部正确才算
    n_pairs = len(groups)
    n_pairs_ok = sum(1 for v in groups.values() if all(v))
    if n_pairs:
        print(f"[{label}] pair-acc={n_pairs_ok}/{n_pairs} "
              f"（同问题不同记忆值：整组全对才算，纯背 Q→A 做不到）", flush=True)
    return (hit / n if n else None), n


def _grad_norm(params) -> float:
    """参数组的全局梯度范数（L2）。None 梯度跳过；全 None 返回 0.0。"""
    total = 0.0
    seen = False
    for p in params:
        if p.grad is not None:
            seen = True
            total += float(p.grad.norm().item()) ** 2
    return (total ** 0.5) if seen else 0.0


def encode_item_raw_embed(model, tokenizer, item):
    """--bridge-only 消融：跳过压缩器，记忆原文 token embedding 直接当 z（压缩率 1:1）。

    用途：区分"bridge 学不会映射"还是"compressor 丢了信息"。z 形状 [1, T, d]，
    token 数与原文一致（不是等压缩率对照，是通路消融）。
    """
    dev = LM._model_device(model)
    text = " ".join(item_units(item))
    ids = tokenizer(text, return_tensors="pt",
                    add_special_tokens=False)["input_ids"].to(dev)
    z = model.get_input_embeddings()(ids)          # 冻结权重，无梯度流入
    return z, {"n_soft_tokens": int(z.shape[1]), "compression_ratio": 1.0}


def evaluate_recon(reader_like, model, tokenizer, data, system_prompt, limit=16,
                   layer=LM.LATENT_LAYER, max_unit_tokens=LM.LATENT_MAX_UNIT_TOKENS):
    """从软 token 重建记忆原文的 **token 级准确率**（衡量压缩器保不保真）。

    这是 recon 辅助任务真正要回答的问题：z 里还剩多少原文的符号信息。
    exact-match 太严（一条 65 token 的句子差一个 token 就全错），所以报 token 级。
    位置对齐：causal LM 的 logits[t] 预测 token t+1，因此第 i 个答案 token 的预测
    取自它**前一个**位置的 logits（写错会整体错一位，看起来像"永远差一点"）。
    """
    comp, bridge = reader_like[0], reader_like[1]
    router = reader_like[2] if len(reader_like) > 2 else None
    comp.eval()
    bridge.eval()
    if router is not None:
        router.eval()
    enc = LM.LatentEncoder(model, tokenizer, comp, layer=layer,
                           max_unit_tokens=max_unit_tokens)
    hit = tot = 0
    for item in data[:limit]:
        z, _zcat, _info = encode_item_slots(enc, item, tag="recon")
        with torch.no_grad():
            if router is not None:
                # recon 必须和训练时走同一条读出路径：训练里 recon_stage 用的是
                # 路由后的 soft（m 个 token），这里若用裸 bridge(z)（K*m 个 token），
                # 评估口径与训练不一致，recon 指标不能反映压缩器实际保真度。
                slot_z = LM.stack_payloads(list(z)).unsqueeze(0)
                qf = LM.extract_query_feature(model, tokenizer, item["question"],
                                              system_prompt, layer=layer)
                z_sel, _lg, _ki, _ko = router(qf, slot_z, hard=True)
                soft = bridge(z_sel)
            else:
                soft = bridge(torch.cat(list(z), dim=0) if len(z) > 1 else z[0])
            embeds, labels, n_ans = build_student_inputs(
                model, tokenizer, soft, system_prompt,
                "Repeat the memory verbatim.", " ".join(item_units(item)))
            with LM.delta_write_disabled(model):
                logits = model(inputs_embeds=embeds,
                               attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long),
                               use_cache=False).logits
        pred = logits[0, -n_ans - 1:-1].argmax(-1)
        gold = labels[0, -n_ans:]
        hit += int((pred == gold).sum().item())
        tot += int(n_ans)
    return (hit / tot if tot else None), tot


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
        if args.layer >= model.config.num_hidden_layers:
            # smoke 模型只有 2 层（hs=(embedding, 末层)），--layer 18 会直接越界。
            # 这里显式降级并说明，不做"重复末层"之类的假包装。
            print(f"[smoke] --layer {args.layer} 超出 smoke 模型层数，"
                  f"降到 {model.config.num_hidden_layers - 1}（只影响自检）", flush=True)
            args.layer = model.config.num_hidden_layers - 1
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base)
        model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, device_map=f"cuda:{args.gpu}",
            attn_implementation="sdpa")
        if not args.no_adapter:
            # deltamem.core 顶层硬依赖 transformers.models.smollm3（4.51 无此子包），
            # 这里经 deltamem_compat 注入最小占位后再 import，详见该模块的说明。
            import deltamem_compat as DMC
            HFDeltaMemConfig, attach_delta_mem, load_delta_mem_adapter = \
                DMC.import_delta_mem()
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
    # 软 token 的范数必须对齐真实 token embedding 的尺度，否则会变成 attention sink
    # 把真实 token 淹没（实测未约束时软 token 范数 96~159，而真实 embedding 约 1.05，
    # 结果：loss 降到 2.8 但同一批样本 200 条只答对 1~3 条，加宽 m 完全无效）。
    with torch.no_grad():
        _emb_w = model.get_input_embeddings().weight
        target_norm = float(_emb_w.float().norm(dim=-1).mean())
    bridge = LM.LatentBridge(hidden, bottleneck=args.bottleneck,
                             target_norm=target_norm).to(dev)
    print(f"[bridge] 软 token 目标范数 = 真实 embedding 每向量范数 {target_norm:.4f}"
          f"（可学习，夹在 [{LM.LatentBridge.NORM_MIN}, {LM.LatentBridge.NORM_MAX}] 倍）", flush=True)
    router = None
    if args.router:
        router = LM.LatentRouter(hidden, n_slots=args.m, d_key=args.d_key).to(dev)
        print(f"[router] LatentRouter 启用：d_key={args.d_key} τ0="
              f"{float(router.log_temp.exp()):.4f} addr_weight={args.addr_weight}"
              f"{'（Compressor 主干冻结）' if args.freeze_compressor else ''}", flush=True)
    if args.init:
        blob = LM._safe_torch_load(args.init)
        comp.load_state_dict(blob["compressor"])
        bridge.load_state_dict(blob["bridge"])
        if router is not None and blob.get("router"):
            router.load_state_dict(blob["router"])
            print(f"[init] Router 也一并从 {args.init} 载入", flush=True)
        print(f"[init] 从 {args.init} 初始化（阶段 B 常规做法）", flush=True)
    if args.freeze_compressor:
        if router is None:
            raise SystemExit("--freeze-compressor 只在 --router 下有意义（Phase 2）")
        # Phase 2 只放开寻址头：压缩器主干冻住，保证值编码不被寻址梯度改写。
        for prm in comp.parameters():
            prm.requires_grad_(False)
        comp.eval()
        print("[freeze] Compressor 主干已冻结（只训 Router + Bridge）", flush=True)
    n_train = sum(p.numel() for p in comp.parameters() if p.requires_grad) + \
              sum(p.numel() for p in bridge.parameters()) + \
              (sum(p.numel() for p in router.parameters()) if router is not None else 0)
    print(f"[params] 可训练 {n_train/1e6:.3f}M "
          f"(compressor={sum(p.numel() for p in comp.parameters() if p.requires_grad)/1e6:.3f}M, "
          f"bridge={sum(p.numel() for p in bridge.parameters() if p.requires_grad)/1e6:.3f}M"
          + (f", router={sum(p.numel() for p in router.parameters())/1e6:.3f}M"
             if router is not None else "") + ")", flush=True)

    # ---- 数据 ----
    if args.data:
        # 外部语料（tools/gen_synth_qa.py 产出）：与最终评测集无关的独立事实文本
        n_train = args.n_train or 0
        train_data = load_synth_jsonl(args.data, limit=n_train, seed=args.seed)
        if not train_data:
            raise SystemExit(f"--data 读不到任何样本：{args.data}")
        if args.val_data:
            val_data = load_synth_jsonl(args.val_data, limit=(args.n_val or 0),
                                        seed=args.seed + 1)
        else:
            # 没有验证文件时只做通路检查，**不能**当验证指标（同实体同分布）
            val_data = train_data[:max(1, min(len(train_data), args.n_val or 32))]
            print("[data][warn] 未给 --val-data，用训练集前若干条做通路检查，"
                  "该数字不是验证指标", flush=True)
        for prob in check_split_leakage(train_data, val_data):
            print(f"[data][warn] 切分不干净：{prob}", flush=True)
        hops = Counter(it["hops"] for it in train_data)
        n_units = sum(len(item_units(it)) for it in train_data) / max(1, len(train_data))
        print(f"[data] {args.data}: train={len(train_data)} val={len(val_data)} "
              f"hops={dict(sorted(hops.items()))} 平均记忆单元={n_units:.1f}", flush=True)
        print(f"[data] 例：units={item_units(train_data[0])[:3]} "
              f"q={train_data[0]['question']!r} a={train_data[0]['answer']!r}", flush=True)
    else:
        ent_train, ent_val = split_entities(args.seed)
        n_train = args.n_train or 2000
        n_val = args.n_val or 100
        train_data = make_dataset(n_train, random.Random(args.seed + 1), ent_train,
                                  value_shift=0, counterfactual=args.counterfactual)
        val_data = make_dataset(n_val, random.Random(args.seed + 2), ent_val,
                                value_shift=3, counterfactual=False)
        print(f"[data] train={len(train_data)} val={len(val_data)} "
              f"entities(train/val)={len(ent_train)}/{len(ent_val)}（不相交）", flush=True)
        print(f"[data] 例：memory={train_data[0]['memory']!r} "
              f"q={train_data[0]['question']!r} a={train_data[0]['answer']!r}", flush=True)

    if args.dry_run:
        print("[dry-run] 数据/模型/参数已就绪，开始一次前向+反向自检（不写 checkpoint）",
              flush=True)

    opt_params = list(comp.parameters()) + list(bridge.parameters()) + \
        (list(router.parameters()) if router is not None else [])
    opt = torch.optim.AdamW([p for p in opt_params if p.requires_grad], lr=args.lr)
    encoder = LM.LatentEncoder(model, tokenizer, comp, layer=args.layer,
                               max_unit_tokens=args.max_unit_tokens, device=dev)
    system_prompt = args.system_prompt
    reader_like = (comp, bridge) if router is None else (comp, bridge, router)
    rng_slots = random.Random(args.seed + 777)      # 变槽数采样的独立随机流
    max_steps = args.max_steps if args.max_steps > 0 else max(1, len(train_data) // args.accum)
    if args.freeze_compressor:
        comp.eval()          # 冻结主干：训练循环里的 comp.train() 会覆盖，这里兜住
    else:
        comp.train()
    bridge.train()
    if router is not None:
        router.train()
    def _cfg(steps_done):
        return {"schema": LM.LATENT_SCHEMA, "n_slots": args.m, "layer": args.layer,
                "inner_dim": args.inner_dim, "num_heads": args.heads,
                "bottleneck": args.bottleneck, "max_unit_tokens": args.max_unit_tokens,
                "target_norm": round(float(target_norm), 6),
                "read_budget": LM.LATENT_READ_BUDGET, "dtype": LM.LATENT_DTYPE,
                "stage": args.stage, "kl_weight": args.kl_weight, "kl_tau": args.kl_tau,
                "recon_weight": args.recon_weight, "lr": args.lr,
                "steps": steps_done, "epochs": args.epochs, "seed": args.seed,
                "smoke_model": bool(args.smoke_model),
                "router": bool(args.router), "d_key": int(args.d_key),
                "addr_weight": float(args.addr_weight)}

    step = 0
    micro = 0
    acc_items = []
    grad_seen = None        # 必须在 opt.zero_grad() 之前取样，否则永远是 None -> 误报失败
    t_start = time.time()
    if args.eval_every > 0 and not args.dry_run:
        # step=0 基线：随机初始化的压缩器/桥接器在这批样本上的表现（对照起点）
        comp.eval()
        bridge.eval()
        if router is not None:
            router.eval()
        em0, n0 = evaluate(reader_like, model, tokenizer, val_data, system_prompt,
                           limit=args.eval_limit, layer=args.layer,
                           max_unit_tokens=args.max_unit_tokens, verbose=True,
                           label="eval@0", bridge_only=args.bridge_only)
        print(f"[eval] step=0 full-match={em0 if em0 is None else round(em0, 4)} (n={n0})",
              flush=True)
        if not args.freeze_compressor:
            comp.train()
        bridge.train()
        if router is not None:
            router.train()
    for epoch in range(args.epochs):
        order = list(range(len(train_data)))
        random.Random(args.seed + 100 + epoch).shuffle(order)
        for i in order:
            item = train_data[i]
            # ① 压缩（grad=True：保留到压缩器输出的梯度路径）
            #    一条样本有几条事实就编几个槽位（与推理期 bank 布局一致）
            #    --bridge-only：跳过压缩器，原文 token embedding 直接进 bridge（消融用）
            payloads = None
            if args.bridge_only:
                z, enc_info = encode_item_raw_embed(model, tokenizer, item)
            elif router is not None:
                # 路由需要每个槽独立成形 [m,d]，所以保留 per-slot payload。
                # Phase 2 冻结压缩器时用 grad=False：反正要 detach，建图纯浪费显存。
                payloads, z, enc_info = encode_item_slots(
                    encoder, item, tag="fact", grad=not args.freeze_compressor)
            else:
                z, enc_info = encode_item(encoder, item, tag="fact", grad=True)
            # ② Z -> soft tokens（**必须过 bridge**）
            #    推理期 generate_latent_answer 内部就是 soft = bridge(payloads)；
            #    训练时若直接把 z 当 soft token 喂进去，bridge 不在计算图里、
            #    梯度检查会报 bridge_last_linear=false，且推理期用的是随机投影。
            r_logits = None
            tgt_slot = 0
            if router is not None and payloads is not None:
                # 变槽数 + 随机顺序：逼模型学"按问题选槽"，而不是"第 0 位是答案"
                tgt_slot = target_slot_index(item_units(item), item["answer"],
                                             evidence=item.get("evidence"))
                slot_z, tgt_slot, _keep = sample_slot_subset(
                    payloads, tgt_slot, rng_slots, args.max_slots or None,
                    keep_all=(int(item.get("hops", 1) or 1) > 1))
                if args.freeze_compressor:
                    slot_z = slot_z.detach()
                q_feat = query_feature_cached(model, tokenizer, item["question"],
                                              system_prompt, args.layer, dev)
                z_read, r_logits, _ki, _ko = router(q_feat, slot_z, hard=False)
                soft = bridge(z_read)
            else:
                soft = bridge(z)
            # ③ student 前向：**不加 no_grad**（冻结的是权重，不是梯度）
            embeds, labels, n_ans = build_student_inputs(
                model, tokenizer, soft, system_prompt, item["question"], item["answer"])
            with LM.delta_write_disabled(model):
                s_logits = model(inputs_embeds=embeds,
                                 attention_mask=torch.ones(embeds.shape[:2],
                                                           dtype=torch.long, device=dev),
                                 use_cache=False).logits
            loss_qa = qa_loss(s_logits, labels)
            loss = loss_qa
            loss_addr = None
            if r_logits is not None:
                # 显式寻址监督：让"选哪个槽"有直接梯度。只靠 QA 隐式回传时，
                # 4 槽里 3 个干扰槽会把信号摊薄到学不出来（实测 target_only 直接 0%）。
                tgt_t = torch.tensor([int(tgt_slot)], device=r_logits.device)
                loss_addr = F.cross_entropy(r_logits, tgt_t)
                loss = loss + args.addr_weight * loss_addr
            loss_kl = None
            if args.stage == "b" and args.kl_weight > 0:
                # ④ teacher：明文证据，只提供目标分布
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
            recon_stage = args.recon_only_steps > 0 and step < args.recon_only_steps
            if args.recon_weight > 0 or recon_stage:
                # ⑤ 事实重建辅助任务：只给 soft tokens，重建原事实（检查符号保真）
                #    memory 字段是列表，必须先 join 成句 —— str(list) 会得到
                #    "['Taron works at Velcor.']" 这种 Python repr（实测踩过：
                #    recon token-acc 被这个 bug 压低了 ~39%）。
                r_embeds, r_labels, _ = build_student_inputs(
                    model, tokenizer, soft, system_prompt,
                    "Repeat the memory verbatim.", " ".join(item_units(item)))
                with LM.delta_write_disabled(model):
                    r_logits = model(inputs_embeds=r_embeds,
                                     attention_mask=torch.ones(r_embeds.shape[:2],
                                                               dtype=torch.long, device=dev),
                                     use_cache=False).logits
                loss_r = qa_loss(r_logits, r_labels)
                # 两阶段（--recon-only-steps N）：前 N 步只训重建 —— 先逼压缩器把内容
                # 保住，再引入 QA 学读出。两个目标在同一组软 token 上会争预算，同时开
                # 会震荡（实测 200/400/600 步准确率 0.0625->0.125->0.0625 非单调）。
                if recon_stage:
                    loss = loss_r
                    if loss_addr is not None:
                        loss = loss + args.addr_weight * loss_addr
                else:
                    loss = loss + args.recon_weight * loss_r
            (loss / args.accum).backward()
            acc_items.append((float(loss_qa.detach()),
                               None if loss_kl is None else float(loss_kl.detach()),
                               None if loss_addr is None else float(loss_addr.detach())))
            micro += 1
            if micro % args.accum == 0:
                # 梯度范数（step 前）：.grad is not null 只说明"有梯度"，不说明"梯度可用"；
                # 范数才知道是不是在有效学习（太小=没信号，爆炸=不稳）
                gnorm_comp = 0.0 if (args.bridge_only or args.freeze_compressor) \
                    else _grad_norm(comp.parameters())
                gnorm_bridge = _grad_norm(bridge.parameters())
                gnorm_router = (_grad_norm(router.parameters())
                                if router is not None else 0.0)
                if grad_seen is None:
                    # 第一个梯度累积边界处取样：此时 .grad 已填充、尚未 step/zero_grad
                    grad_seen = {"model_has_no_grad":
                                     all(p.grad is None for p in model.parameters())}
                    if not args.bridge_only:
                        grad_seen["compressor_queries"] = comp.queries.grad is not None
                        grad_seen["compressor_input_proj"] = \
                            comp.input_proj.weight.grad is not None
                    grad_seen["bridge_last_linear"] = bridge.net[3].weight.grad is not None
                if args.save_every > 0 and step % args.save_every == 0 and not args.dry_run:
                    # 长跑（数小时）必须周期落盘：只在最后保存的话，中途崩掉全丢。
                    # 直接覆盖同一个文件，不做 step 编号副本，避免 ckpt 目录膨胀。
                    LM.save_latent_checkpoint(
                        args.out, comp, bridge, _cfg(step),
                        {"trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                         "git": _git_commit(), "partial": True, "step": step,
                         "n_train": len(train_data)}, router=router)
                    print(f"[ckpt] step {step} 已保存 -> {args.out}", flush=True)
                if args.grad_clip > 0:
                    clip_params = list(comp.parameters()) + list(bridge.parameters()) + \
                        (list(router.parameters()) if router is not None else [])
                    torch.nn.utils.clip_grad_norm_(
                        [x for x in clip_params if x.requires_grad], args.grad_clip)
                opt.step()
                if args.eval_every > 0 and step % args.eval_every == 0:
                    # 训练中周期评估（用户要求的 0/200/400/800 节点；step=0 在循环前跑）
                    if router is not None:
                        router.eval()
                    em_i, n_i = evaluate(
                        reader_like, model, tokenizer, val_data, system_prompt,
                        limit=args.eval_limit, layer=args.layer,
                        max_unit_tokens=args.max_unit_tokens, verbose=True,
                        label=f"eval@{step}", bridge_only=args.bridge_only)
                    print(f"[eval] step={step} full-match={em_i if em_i is None else round(em_i, 4)}"
                          f" (n={n_i}) gnorm_comp={gnorm_comp:.3e}"
                          f" gnorm_bridge={gnorm_bridge:.3e}"
                          f" gnorm_router={gnorm_router:.3e}", flush=True)
                    if not args.freeze_compressor:
                        comp.train()
                    bridge.train()
                    if router is not None:
                        router.train()
                opt.zero_grad(set_to_none=True)
                step += 1
                qa_avg = sum(x[0] for x in acc_items) / len(acc_items)
                kls = [x[1] for x in acc_items if x[1] is not None]
                kl_avg = sum(kls) / len(kls) if kls else None
                addrs = [x[2] for x in acc_items if x[2] is not None]
                addr_avg = sum(addrs) / len(addrs) if addrs else None
                print(f"[step {step}] loss_qa={qa_avg:.4f} "
                      f"loss_kl={'n/a' if kl_avg is None else f'{kl_avg:.4f}'} "
                      f"loss_addr={'n/a' if addr_avg is None else f'{addr_avg:.4f}'} "
                      f"soft_tokens={int(soft.reshape(1, -1, soft.shape[-1]).shape[1])} "
                      f"compress={enc_info['compression_ratio']} "
                      f"gnorm_comp={gnorm_comp:.3e} gnorm_bridge={gnorm_bridge:.3e} "
                      f"{'gnorm_router=%.3e ' % gnorm_router if router is not None else ''}"
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

    em, n_eval = evaluate(reader_like, model, tokenizer, val_data, system_prompt,
                          limit=args.eval_limit, layer=args.layer,
                          max_unit_tokens=args.max_unit_tokens, verbose=True,
                          label="eval-final", bridge_only=args.bridge_only)
    print(f"[eval] val full-match={em if em is None else round(em, 4)} (n={n_eval}) "
          f"—— 规范化完整答案匹配（不再是子串匹配）", flush=True)

    # 存盘->重载->评：排除"训练进程内状态"对最终评估的影响（悬案：进程内 0.9375 vs
    # 重载 0，曾发生在 min16 v2 / exp2 v2 上）。若两者不一致，铁证状态污染。
    if not args.dry_run and not args.bridge_only:
        import tempfile, os as _os
        _snap = _os.path.join(tempfile.gettempdir(), "reload_check.pt")
        LM.save_latent_checkpoint(_snap, comp, bridge, _cfg(args.max_steps),
                                  {"partial": False}, router=router)
        _blob = LM._safe_torch_load(_snap)
        _comp2 = LM.LatentCompressor(LM._model_hidden_size(model),
                                     n_slots=_blob["config"]["n_slots"],
                                     inner_dim=_blob["config"]["inner_dim"],
                                     num_heads=_blob["config"]["num_heads"]).to(dev)
        _bridge2 = LM.LatentBridge(LM._model_hidden_size(model),
                                   bottleneck=_blob["config"]["bottleneck"],
                                   target_norm=float(_blob["config"]["target_norm"])).to(dev)
        _comp2.load_state_dict(_blob["compressor"]); _bridge2.load_state_dict(_blob["bridge"])
        _comp2.eval(); _bridge2.eval()
        _router2 = None
        if _blob.get("router"):
            _router2 = LM.LatentRouter(LM._model_hidden_size(model),
                                       n_slots=_blob["config"]["n_slots"],
                                       d_key=int(_blob["config"].get("d_key",
                                                                    LM.LATENT_D_KEY))).to(dev)
            _router2.load_state_dict(_blob["router"])
            _router2.eval()
        _reload_reader = (_comp2, _bridge2) if _router2 is None \
            else (_comp2, _bridge2, _router2)
        em_rl, n_rl = evaluate(_reload_reader, model, tokenizer, val_data,
                               system_prompt, limit=args.eval_limit, layer=args.layer,
                               max_unit_tokens=args.max_unit_tokens, verbose=False,
                               label="eval-reload")
        print(f"[eval-reload] full-match={em_rl if em_rl is None else round(em_rl, 4)} "
              f"(n={n_rl}) —— 存盘重载后的对照值", flush=True)
        # 钉死 delta 状态依赖：清空 delta_state 后再评一次。若从高分掉到 0，
        # 说明训出的 comp/bridge 只在"带着训练累积 delta 状态的模型"上工作 ——
        # 潜 token 本身没有携带答案信息（悬案 v2 0.9375 vs 重载 0 的根因）。
        from deltamem.core.delta_impl import reset_delta_mem_states as _rds
        _rds(model)
        em_rs, n_rs = evaluate(_reload_reader, model, tokenizer, val_data,
                               system_prompt, limit=args.eval_limit, layer=args.layer,
                               max_unit_tokens=args.max_unit_tokens, verbose=False,
                               label="eval-after-reset")
        print(f"[eval-after-reset] full-match={em_rs if em_rs is None else round(em_rs, 4)} "
              f"(n={n_rs}) —— 清空 delta 状态后的对照值", flush=True)
        if em is not None and em_rl is not None and abs(em - em_rl) > 0.2:
            print(f"[WARNING] 进程内评估({em:.3f}) 与 重载评估({em_rl:.3f}) 差异巨大！"
                  f"训练进程内存在影响生成的状态（delta_state / dropout / 缓存）。", flush=True)

    if args.dry_run:
        print(f"[dry-run] OK：通路验证完成（{time.time()-t0:.1f}s），未写 checkpoint",
              flush=True)
        return 0

    cfg = _cfg(step)
    meta = {"trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git": _git_commit(), "n_train": len(train_data), "n_val": len(val_data),
            "val_exact_match": em, "n_trainable_params": n_train,
            "grad_check": grad_ok, "elapsed_sec": round(time.time() - t0, 1)}
    if args.recon_weight > 0:
        rec_acc, rec_tok = evaluate_recon(
            reader_like, model, tokenizer, val_data, system_prompt,
            limit=min(16, len(val_data)), layer=args.layer,
            max_unit_tokens=args.max_unit_tokens)
        print(f"[eval] recon token-acc={rec_acc:.3f}（n_tok={rec_tok}）—— "
              f"z 里还留了多少原文符号信息", flush=True)
        meta["recon_token_acc"] = rec_acc

    out = LM.save_latent_checkpoint(args.out, comp, bridge, cfg, meta, router=router)
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
    ap.add_argument("--recon-only-steps", type=int, default=0,
                    help="前 N 个优化步只训重建（QA 关闭），之后切 QA+recon。"
                         "用途：先保真再学读出，避免两个目标同时争软 token 预算")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="每 N 个优化步做一次生成评估（0=只在结束后评）。"
                         "实验 1 用 200：节点 0/200/400/800")
    ap.add_argument("--bridge-only", action="store_true",
                    help="条件消融 A：跳过压缩器，记忆原文 token embedding 直接进 bridge"
                         "（压缩率 1:1，保留全部 token 信息；区分 bridge vs compressor 瓶颈）")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=500,
                    help="每 N 个优化步落盘一次（0=只在结束时保存）")
    ap.add_argument("--kl-weight", type=float, default=0.0)
    ap.add_argument("--kl-tau", type=float, default=1.0)
    ap.add_argument("--recon-weight", type=float, default=0.0)
    ap.add_argument("--counterfactual", action="store_true",
                    help="训练集加入『同主语同属性冲突，只记新值』样本")
    ap.add_argument("--data", default="",
                    help="外部训练语料 jsonl（tools/gen_synth_qa.py 产出）；"
                         "给了就用它，否则用内部模板 make_dataset")
    ap.add_argument("--val-data", default="", help="外部验证语料 jsonl（与 --data 实体簇不重叠）")
    ap.add_argument("--n-train", type=int, default=0,
                    help="训练条数；0=内部默认 2000 / --data 时全量")
    ap.add_argument("--n-val", type=int, default=0,
                    help="验证条数；0=内部默认 100 / --data 时全量")
    ap.add_argument("--eval-limit", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--router", action="store_true",
                    help="启用 LatentRouter 集合内容寻址（多槽读出必须开，否则位置敏感）")
    ap.add_argument("--d-key", type=int, default=LM.LATENT_D_KEY,
                    help="Router 的查询/键维度")
    ap.add_argument("--addr-weight", type=float, default=1.0,
                    help="寻址交叉熵损失权重（router 的显式监督）")
    ap.add_argument("--freeze-compressor", action="store_true",
                    help="冻结 Compressor 主干（Phase 2：只训 router 的键/查询头）")
    ap.add_argument("--max-slots", type=int, default=0,
                    help="变槽数采样的上限（0=用数据里的全部槽位）")
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
