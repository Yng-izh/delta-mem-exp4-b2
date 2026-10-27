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
            # `units` 为**显式槽位列表**时以它为准：真实数据（HotpotQA / LME）一条样本有
            # 几十句，`memory` 往往已经是拼好的整串；若还按"memory 是一条事实"去切槽，
            # 整段会被压成 1 个槽位 —— 训练里"从 40 个槽里选 1 个"当场退化成"压缩整段"，
            # 而这正是真实数据要练的能力。合成数据的 memory 本来就是事实列表，未受影响。
            units = rec.get("units")
            if isinstance(units, list) and units:
                mem = [str(u).strip() for u in units if str(u).strip()]
            else:
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
                # ---- FST-Mem（阶段 B）需要的结构化字段 ----
                # unit_meta 与 units **逐条对齐**：每条记忆单元属于哪个 (实体,属性)、
                # 写入时间、是不是干扰项。没有它就只能靠正则解析句子猜三元组，
                # 而"猜错一个属性"在本方法里等价于"更新打到了别的事实键"——静默致命。
                "unit_meta": rec.get("unit_meta") or [],
                "kind": str(rec.get("kind") or ""),
                "entity": rec.get("entity"),
                "relation": rec.get("relation"),
                "fact_key": rec.get("fact_key"),
                "as_of": rec.get("as_of"),
                "answer_fact_time": rec.get("answer_fact_time"),
                "probe": str(rec.get("probe") or ""),
                "timeline": rec.get("timeline") or [],
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

    def generate(self, inputs_embeds=None, attention_mask=None, max_new_tokens=8,
                 pad_token_id=None, **kw):
        """最小贪心生成：只为 --smoke-model 跑通 generate 通路（不代表任何真实行为）。

        没有这个方法时 --smoke-model 一旦走到评估就 AttributeError
        （evaluate 里靠 hasattr 兜住、直接放弃评估，于是"通路验证"根本没验证生成）。
        返回形态与"只返回新 token"的真实实现一致。
        """
        if inputs_embeds is None:
            raise ValueError("SmokeModel.generate 需要 inputs_embeds")
        x = inputs_embeds
        ids = []
        for _ in range(int(max_new_tokens)):
            h = self.body(x)
            nxt = int(self.head(h[:, -1:]).argmax(-1).item())
            ids.append(nxt)
            e = self.emb(torch.tensor([[nxt]], dtype=torch.long, device=x.device))
            x = torch.cat([x, e.to(dtype=x.dtype)], dim=1)
        return torch.tensor([ids], dtype=torch.long, device=x.device)

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
def assemble_loss(loss_qa, loss_addr, loss_kl, loss_r, addr_weight, kl_weight,
                  recon_weight, recon_stage=False):
    """把四项损失组合起来（**纯函数**，可直接单测）。

    唯一允许的行为：寻址/KL/重建都**累加**到 QA 损失上。
    唯一的例外是 `recon_stage`（两阶段训练的前 N 步）：那时切换到"纯重建 + 寻址"，
    故意丢掉 QA —— 因为两个目标在同一组软 token 上争预算。

    这里曾有一个静默 bug：KL 用 `loss = loss_qa + kl_weight*loss_kl` 直接赋值，
    把已经加进去的 `addr_weight*loss_addr` **整条覆盖掉**，于是"开了 KL 之后寻址
    再没被监督过"；而日志里 loss_addr 照常打印，看起来一切正常。
    """
    loss = loss_qa
    if loss_addr is not None:
        loss = loss + addr_weight * loss_addr
    if loss_kl is not None:
        loss = loss + kl_weight * loss_kl
    if recon_stage:
        if loss_r is None:
            raise ValueError("recon_stage=True 但 loss_r is None")
        # 纯重建阶段：只保留重建 + 寻址
        loss = loss_r
        if loss_addr is not None:
            loss = loss + addr_weight * loss_addr
    elif loss_r is not None:
        loss = loss + recon_weight * loss_r
    return loss


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
        reset_delta_state(model)     # 样本间隔离：上一题的在线状态不许泄进下一题
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


def _peak_mb():
    """本进程 CUDA 峰值显存（MiB）。打在每步日志里，任何"显存慢慢涨"都能一眼看出。"""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 2 ** 20


def reset_delta_state(model):
    """清空 Delta-Mem 在线状态（**每条样本前必须调用**）。

    为什么必须显式清：在线状态是挂在模块上的、跨 forward 累积的量。训练里我们
    关了写入（`delta_write_disabled`），但状态里仍是**上一条样本**留下的内容；
    只要有一条路径漏关写入，状态就会越积越大，最终不是报错而是
    "显存慢慢涨到 OOM"或"后面的样本莫名受前面影响"，两者都很难定位。

    这也是评估纪律的一部分：同一 context 内必须前后一致，不同 context 之间必须隔离。
    """
    if model is None:
        return False
    try:
        from deltamem.core.delta import reset_delta_mem_states as _r
    except Exception:
        return False
    try:
        _r(model)
        return True
    except Exception:
        return False


def _load_backbone(args, dev=None):
    """加载冻结主干（Qwen 或 --smoke-model 的极小随机模型）。返回 (model, tok, dev, hidden)。

    抽成函数是为了让常规训练（train）与 FST-Mem 训练（train_fst）共享**同一份**
    加载逻辑：两份拷贝迟早漂移，而"其中一条路忘了挂 adapter / 忘了关 Delta 写入"
    只会表现为"某组实验莫名其妙差几个点"，极难定位。
    """
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
        return model, tokenizer, dev, hidden
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
    return model, tokenizer, dev, LM._model_hidden_size(model)


def train(args):
    t0 = time.time()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = torch.device(args.device if args.device else
                       ("cpu" if args.smoke_model else
                        ("cuda" if torch.cuda.is_available() else "cpu")))

    # ---- 模型 ----
    model, tokenizer, dev, hidden = _load_backbone(args, dev)
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
    rng_recon = random.Random(args.seed + 31337)    # recon 取单元用的独立随机流
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
            # 每条样本前清 Delta 在线状态：状态是跨 forward 累积的模块属性，
            # 不清就会"显存慢慢涨"/"后面的样本受前面影响"（都不报错）。
            reset_delta_state(model)
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
            loss_addr = None
            if r_logits is not None:
                # 显式寻址监督：让"选哪个槽"有直接梯度。只靠 QA 隐式回传时，
                # 4 槽里 3 个干扰槽会把信号摊薄到学不出来（实测 target_only 直接 0%）。
                tgt_t = torch.tensor([int(tgt_slot)], device=r_logits.device)
                loss_addr = F.cross_entropy(r_logits, tgt_t)
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
            recon_stage = args.recon_only_steps > 0 and step < args.recon_only_steps
            if args.recon_weight > 0 or recon_stage:
                # ⑤ 事实重建辅助任务：只给 soft tokens，重建原事实（检查符号保真）
                #    memory 字段是列表，必须先 join 成句 —— str(list) 会得到
                #    "['Taron works at Velcor.']" 这种 Python repr（实测踩过：
                #    recon token-acc 被这个 bug 压低了 ~39%）。
                # recon 只重建**少量**随机单元：整段记忆（HotpotQA 一条 ~35 句、
                # 上千 token）做一次带梯度的 LM 前向+反传要 ~11GB，v100-32GB 上跑到
                # 第 3 步就 OOM（实测 21.8GB→30GB）。取固定条数既把显存钉死，
                # 又保持"检查压缩器符号保真"这个用途不变。
                _ru = list(item_units(item))
                if args.recon_units > 0 and len(_ru) > args.recon_units:
                    _ru = [x[1] for x in
                           sorted((rng_recon.random(), u) for u in _ru)[:args.recon_units]]
                r_embeds, r_labels, _ = build_student_inputs(
                    model, tokenizer, soft, system_prompt,
                    "Repeat the memory verbatim.", " ".join(_ru))
                with LM.delta_write_disabled(model):
                    r_logits = model(inputs_embeds=r_embeds,
                                     attention_mask=torch.ones(r_embeds.shape[:2],
                                                               dtype=torch.long, device=dev),
                                     use_cache=False).logits
                loss_r = qa_loss(r_logits, r_labels)
            # 四项损失的组合规则集中在 assemble_loss（旧代码在 KL 分支里重新赋值，
            # 把寻址项整条覆盖掉了；两阶段的重建切换规则也一并挪进来便于单测）。
            loss = assemble_loss(loss_qa, loss_addr, loss_kl,
                                 None if not (args.recon_weight > 0 or recon_stage) else loss_r,
                                 addr_weight=args.addr_weight, kl_weight=args.kl_weight,
                                 recon_weight=args.recon_weight, recon_stage=recon_stage)
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
                      f"peakmem={_peak_mb():.0f}MiB "
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


# =====================================================================
# 5) FST-Mem（阶段 B）：状态—轨迹解耦潜层记忆的训练与评估
# =====================================================================
# 这一节**不改动**上面的 train()/evaluate()：它们是阶段 A（单级 Router）的既有基线，
# 任何改动都会让历史实验不可复现。FST 走独立的入口（--fst），共享的是底层模块
# （Compressor / Bridge / 编码器 / 事实键与两级 Router），不是训练循环。
#
# 三个机制与代码的对应关系（论文里要逐条对得上）：
#   机制一 实体—属性级稳定寻址  -> LM.LatentFactKeyEncoder（键与值无关）
#   机制二 状态—轨迹解耦槽位      -> LM.LatentStateTrajectoryBank（cur + 有限历史）
#   机制三 时序条件两级读出      -> LM.FstRouter（先选键，再选版本）
def _fst_feat(model, tokenizer, text, layer, dev, cache):
    """短文本特征缓存：实体名/属性名/年份在数据里高度重复，重复前向纯属浪费。

    缓存键必须含 layer 与 device（口径同 query_feature_cached）：只按文本命中，
    换个层号/设备就会拿到另一套特征去打分，而指标完全看不出问题。
    """
    k = ("txt", str(text), int(layer), str(dev))
    if k in cache:
        return cache[k]
    v = LM.extract_text_feature(model, tokenizer, text, layer=layer, device=dev)
    cache[k] = v.detach()
    return cache[k]


def fst_units_and_meta(item):
    """校验 unit_meta 与 units 逐条对齐。不齐直接报错——猜就是错的开始。"""
    units = item_units(item)
    ums = list(item.get("unit_meta") or [])
    if not ums:
        raise ValueError(f"样本缺 unit_meta（FST 需要每条记忆的 实体/属性/时间）："
                         f"q={item.get('question')!r}")
    if len(ums) != len(units):
        raise ValueError(f"unit_meta({len(ums)}) 与 units({len(units)}) 条数不一致："
                         f"q={item.get('question')!r}")
    for i, um in enumerate(ums):
        if not um.get("entity") or not um.get("relation"):
            raise ValueError(f"unit_meta[{i}] 缺 entity/relation：{um!r}")
        if _norm_ans(um.get("fact")) != _norm_ans(units[i]):
            raise ValueError(f"unit_meta[{i}].fact 与 units[{i}] 不是同一条事实："
                             f"{um.get('fact')!r} vs {units[i]!r}")
    return units, ums


def build_fst_bank(encoder, item, hist_max, tag="fst", grad=False,
                   shuffle_order=False, rng=None):
    """把一条样本的时间线按要求顺序写进 LatentStateTrajectoryBank，并定位监督目标。

    顺序（order）在这里是**显式可选**的：默认按数据的写入顺序（= 真实的时间线），
    shuffle_order=True 时随机排列写入顺序。事实键与时间戳都写在数据里，因此
    两种顺序写出来的库在语义上必须等价 —— 这正是"组合稳定性"要测的东西。

    返回 dict，其中 tgt_group/tgt_ver 是被监督的"正确事实键 + 正确版本"。
    viable=False 表示目标版本已被容量淘汰（hist_max 太小），此时该样本**没法学**，
    训练要跳过、评估要算错（这正是"硬覆写"基线该拿 0 分的原因）。
    """
    units, ums = fst_units_and_meta(item)
    payloads, _z, info = encode_item_slots(encoder, item, tag=tag, grad=grad)
    if len(payloads) != len(units):
        raise RuntimeError(f"编码返回 {len(payloads)} 个槽位，但样本有 {len(units)} 条记忆")
    order = list(range(len(units)))
    if shuffle_order:
        (rng or random).shuffle(order)
    # fp32 存储：训练路径要把梯度带回去，bf16 往返会引入无意义的舍入；
    # 载荷本身来自 bf16 压缩器，转 fp32 是无损的。
    bank = LM.LatentStateTrajectoryBank(hist_max=hist_max, dtype=torch.float32)
    for i in order:
        um = ums[i]
        k = LM.fact_key(um["entity"], um["relation"])
        # entity/relation 的**原始文本**随键一起存进库：读出侧从这里取，
        # 不能让它自己从（规范化过的）键反推，否则训练/读出两条路的键特征不同源。
        bank.write(k, payloads[i], time=um.get("time", ""), detach=not grad,
                   entity=um["entity"], relation=um["relation"])
    keys = bank.group_keys()
    ent = {k: bank.group_text(k)[0] for k in keys}
    rel = {k: bank.group_text(k)[1] for k in keys}
    tgt_key = item.get("fact_key") or LM.fact_key(item.get("entity"), item.get("relation"))
    tgt_key = LM.fact_key(*LM.split_fact_key(tgt_key))
    out = {"bank": bank, "keys": keys, "tgt_key": tgt_key, "viable": False,
           "tgt_group": -1, "tgt_ver": -1, "d_units": len(units), "info": info,
           "n_groups": len(keys)}
    if tgt_key not in keys:
        raise RuntimeError(f"目标事实键不在库里：{tgt_key!r}（键={keys}）")
    out["tgt_group"] = keys.index(tgt_key)
    want_t = LM._norm_time(item.get("answer_fact_time")
                           if item.get("answer_fact_time") is not None
                           else item.get("as_of"))
    vers = bank.list_versions(tgt_key)
    # 并列时间戳时取**最后**（= 更新）的那个匹配，与 bank.read() 的 `>=` 并列规则一致；
    # 若这里 break 在第一个匹配（最旧），监督就会指向 read() 永远不会选的那条槽位。
    for vi, v in enumerate(vers):
        if LM._time_match_score(want_t, v["time"]) == 2:
            out["tgt_ver"] = vi
            out["viable"] = True
    # 目标组/版本的张量：只把**这个键**的版本送去版本级打分
    out["tgt_ver_n"] = len(vers)          # 目标组里还剩几个版本（=1 表示没得选）
    # 三种"定位不到版本"的情形必须分开，否则训练/评估会把不同原因混成一个数：
    #   evicted       : 该键写过的版本多于现存版本 -> 目标版本真被容量淘汰（样本不可学）
    #   label_missing : 数据里没有答案时间戳 -> 与淘汰无关，仍然可训 QA/组级
    #   label_mismatch: 有时间戳但与任何版本都不精确相等 -> 数据/格式问题，必须显式报
    out["evicted"] = bool(int(bank.groups[tgt_key]["n_write"]) > len(vers))
    out["label_missing"] = bool(not want_t)
    out["label_mismatch"] = bool(not out["viable"] and not out["evicted"]
                                 and not out["label_missing"])
    out.update(fst_group_tensors(bank, keys, tgt_group=out["tgt_group"],
                                 hist_max=hist_max))
    out["ent"], out["rel"] = ent, rel
    return out


def fst_group_tensors(bank, keys, tgt_group, hist_max):
    """整库版本张量 + 目标组内部的版本张量（两级打分各自需要什么就给什么）。"""
    per_group = bank.gather_all_versions(keys)
    ver_z, ver_mask = LM.pad_version_stack(per_group, dtype=torch.float32, device="cpu")
    tgt_payloads = per_group[int(tgt_group)]
    tgt_z, tgt_mask = LM.pad_version_stack([tgt_payloads], dtype=torch.float32,
                                           device="cpu")
    times = [[v["time"] for v in bank.list_versions(k)] for k in keys]
    return {"ver_z": ver_z, "ver_mask": ver_mask,
            "tgt_ver_z": tgt_z, "tgt_ver_mask": tgt_mask, "times": times}


def fst_build_inputs(fst_item, model, tokenizer, layer, dev, cache):
    """库 + 时间特征 -> 两级 Router 的完整输入张量。"""
    keys = fst_item["keys"]
    ent, rel = fst_item["ent"], fst_item["rel"]
    e_feat = torch.cat([_fst_feat(model, tokenizer, ent[k], layer, dev, cache)
                        for k in keys], dim=0)
    r_feat = torch.cat([_fst_feat(model, tokenizer, rel[k], layer, dev, cache)
                        for k in keys], dim=0)
    ver_z = fst_item["ver_z"].to(dev)
    ver_mask = fst_item["ver_mask"].to(dev)
    d = int(ver_z.shape[-1])
    ver_t_feat = torch.zeros((len(keys), int(ver_z.shape[1]), d),
                             dtype=ver_z.dtype, device=dev)
    for gi, ts in enumerate(fst_item["times"]):
        for vi, t in enumerate(ts):
            if _norm_key_time(t):
                ver_t_feat[gi, vi] = _fst_feat(model, tokenizer, t, layer, dev,
                                               cache).to(device=dev,
                                                         dtype=ver_t_feat.dtype)[0]
    return e_feat, r_feat, ver_z, ver_mask, ver_t_feat


def _norm_key_time(t):
    return LM._norm_time(t)


def fst_query_time_feat(item, model, tokenizer, layer, dev, cache):
    """问题里的时间条件 -> 特征；没有条件返回 None（Router 内部按零向量处理）。"""
    t = LM._norm_time(item.get("as_of"))
    if not t:
        return None
    return _fst_feat(model, tokenizer, t, layer, dev, cache)


def fst_eval_subset(data, limit):
    """按"整组问题"截断，而不是按行截断。

    反事实配对的两条样本共享同一句问题文本。若按行截断到 limit，配对经常被切在
    中间：一半样本凑不成对，pair_acc 就退化成 acc（甚至因为单成员对 `all([x])`
    恒真而虚高）。这里以"问题"为不可分割单位取前若干条，保证配对完整。
    """
    if limit is None or limit <= 0 or limit >= len(data):
        return list(data)
    groups, order = {}, []
    for r in data:
        k = _norm_ans(r.get("question"))
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(r)
    out = []
    for k in order:
        out.extend(groups[k])
        if len(out) >= limit:
            break
    return out



def fst_scores(recs):
    """把逐条记录折算成 FST 指标（**纯函数**，单测直接喂造假记录）。

    抽出来是为了能被测：指标口径本身写错了（版本命中不绑组别、单版本样本计入
    版本准确率、单成员配对记作通过）不会让任何"跑得通"的测试失败，只会让论文
    里的数字虚高。见 tools/audit_mutations.py 的 ver_metric_unbound 变异。

    每条 record 需要的键：
        okq, g_ok, ver_picked, tgt_ver, viable, tgt_ver_n, tgt_group,
        evicted, label_missing, probe, question, gold, n_soft
    """
    n = len(recs)
    hit = sum(int(r["okq"]) for r in recs)
    grp_ok = sum(int(r["g_ok"]) for r in recs)
    grp0_ok = sum(int(int(r["tgt_group"]) == 0) for r in recs)
    # 版本级命中必须**同时**选对组：ver_picked 是"被选中组内部"的下标，
    # 组选错时拿它跟目标组的下标比较没有意义（还会因为下标碰巧相等而误记命中）。
    v_ok = [int(bool(r["g_ok"]) and int(r["viable"])
                and int(r["tgt_ver"]) >= 0
                and int(r["ver_picked"]) == int(r["tgt_ver"])) for r in recs]
    # 只有"目标组有多个可选项"时版本级指标才有信息量：单版本组选谁都是对（恒真）。
    multi = [i for i, r in enumerate(recs)
             if int(r["tgt_ver_n"]) > 1 and bool(r["viable"])]
    probes, pairs = {}, {}
    for r in recs:
        d = probes.setdefault(r["probe"], {"n": 0, "hit": 0})
        d["n"] += 1
        d["hit"] += int(r["okq"])
        # 配对必须带 gold：同问题两个变体 gold 相同的（如 unrelated 探针）不构成
        # 反事实配对，不能算进 pair_acc；单成员对也不能算（all([x]) 恒真）。
        pairs.setdefault(_norm_ans(r["question"]), []).append(
            (bool(r["okq"]), _norm_ans(r["gold"])))
    real_pairs = {k: v for k, v in pairs.items()
                  if len(v) >= 2 and len({g for _ok, g in v}) >= 2}
    n_pairs_ok = sum(1 for v in real_pairs.values() if all(ok for ok, _g in v))
    return {
        "n": n,
        "acc": (hit / n if n else None),
        "group_acc": (grp_ok / n if n else None),
        "group_const_acc": (grp0_ok / n if n else None),
        "ver_acc": (sum(v_ok) / n if n else None),
        "ver_multi_acc": (sum(v_ok[i] for i in multi) / len(multi) if multi else None),
        "ver_multi_n": len(multi),
        # 常数基线：永远选"最新版"（= 当前状态）。不比它高 = 时间条件没起作用。
        "ver_newest_acc": (sum(int(int(recs[i]["tgt_ver"])
                                  == int(recs[i]["tgt_ver_n"]) - 1)
                               for i in multi) / len(multi) if multi else None),
        "evicted_n": sum(int(r["evicted"]) for r in recs),
        "label_missing_n": sum(int(r["label_missing"] and not r["evicted"]) for r in recs),
        "pair_acc": (n_pairs_ok / len(real_pairs) if real_pairs else None),
        "pair_n": len(real_pairs),
        "pair_dropped_n": len(pairs) - len(real_pairs),
        "probes": {k: {"n": v["n"], "acc": (v["hit"] / v["n"] if v["n"] else None),
                       "hit": v["hit"]} for k, v in sorted(probes.items())},
        "n_soft": (int(sum(int(r["n_soft"]) for r in recs) / n) if n else 0),
    }


def evaluate_fst(fst_router, comp, bridge, encoder, model, tokenizer, data,
                 system_prompt, layer, hist_max, limit=32, cache=None,
                 verbose=False, label="fst-eval", shuffle_order=False, seed=0,
                 ignore_query_time=False):
    """FST 的评估：严格答案匹配 + **两级寻址准确率** + 探针分解 + 反事实配对 + 顺序稳定性。

    只报答案准确率是不够的：答案对了可能是"没选对版本但那条刚好也能答对"
    （2021 与 2025 的雇主都可能出现在干扰事实里）。所以这里必须同时报
    group_acc（选对属性了吗）与 ver_multi_acc（选对时间版本了吗），
    并且都带**常数基线**，否则"随机权重下 0.25"会被误读成"有点效果"。
    """
    comp.eval()
    bridge.eval()
    fst_router.eval()
    cache = {} if cache is None else cache
    dev = torch.device(LM._model_device(model))
    rng = random.Random(seed + 991)
    recs = []
    order_same = order_n = 0
    fails = []
    for item in fst_eval_subset(data, limit):
        reset_delta_state(model)
        try:
            fi = build_fst_bank(encoder, item, hist_max, tag="val", grad=False,
                                shuffle_order=shuffle_order, rng=rng)
        except Exception as e:                       # 数据问题要显式报，不许跳过不回话
            raise RuntimeError(f"[{label}] 构造 FST 库失败：q={item.get('question')!r}: {e}")
        # 张量由 generate_latent_answer_fst 内部从**真实 bank** 构造：
        # 评估必须走产品路径，不能在这里另建一套张量（两套迟早不一致）。
        pred, meta = LM.generate_latent_answer_fst(
            model=model, tokenizer=tokenizer, bridge=bridge, bank=fi["bank"],
            question=item["question"], system_prompt=system_prompt,
            fst_router=fst_router, max_new_tokens=16, layer=layer,
            query_time=(None if ignore_query_time else item.get("as_of")),
            device=dev, feat_cache=cache)
        gold = item["answer"]
        if isinstance(gold, (list, tuple)):
            gold = " ".join(str(g) for g in gold)
        okq = _norm_ans(pred) == _norm_ans(gold)
        probe = item.get("probe") or ("history" if LM._norm_time(item.get("as_of"))
                                      else "current")
        recs.append({
            "okq": bool(okq),
            "g_ok": bool(meta["group_keys"][int(meta["group_picked"])] == fi["tgt_key"]),
            "ver_picked": int(meta["ver_picked"]),
            "tgt_ver": int(fi["tgt_ver"]),
            "viable": bool(fi["viable"]),
            "tgt_ver_n": int(fi["tgt_ver_n"]),
            "tgt_group": int(fi["tgt_group"]),
            "evicted": bool(fi["evicted"]),
            "label_missing": bool(fi["label_missing"]),
            "probe": probe,
            "question": item["question"],
            "gold": gold,
            "n_soft": int(meta["n_soft"]),
        })
        if shuffle_order:
            # 顺序稳定性：同一库用另一种写入顺序，答案必须一致
            fi2 = build_fst_bank(encoder, item, hist_max, tag="val", grad=False,
                                 shuffle_order=False)
            pred2, _m2 = LM.generate_latent_answer_fst(
                model=model, tokenizer=tokenizer, bridge=bridge, bank=fi2["bank"],
                question=item["question"], system_prompt=system_prompt,
                fst_router=fst_router, max_new_tokens=16, layer=layer,
                query_time=(None if ignore_query_time else item.get("as_of")),
                device=dev, feat_cache=cache)
            order_n += 1
            order_same += int(_norm_ans(pred2) == _norm_ans(pred))
        if verbose and not okq and len(fails) < 8:
            fails.append((item["question"], gold, str(pred)[:40],
                          meta["group_keys"][meta["group_picked"]],
                          meta["ver_time"], item.get("as_of")))
    res = fst_scores(recs)
    res["order_stable"] = (order_same / order_n if order_n else None)
    if verbose:
        for row in fails:
            print(f"  [{label}] ✗ q={row[0]!r} gold={row[1]!r} pred={row[2]!r} "
                  f"pick={row[3]!r} ver={row[4]!r} as_of={row[5]!r}", flush=True)
    return res


def train_fst(args):
    """阶段 B：在"同实体多属性 + 多次覆写 + 时间条件"的数据上训练 FST-Mem。"""
    t0 = time.time()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = torch.device(args.device if args.device else
                       ("cpu" if args.smoke_model else
                        ("cuda" if torch.cuda.is_available() else "cpu")))
    if not args.data:
        raise SystemExit("--fst 需要 --data（tools/gen_fst_update.py 的产出）")
    model, tokenizer, dev, hidden = _load_backbone(args, dev)
    n_frozen = freeze_model(model)
    print(f"[freeze] 冻结参数 {n_frozen/1e6:.1f}M（Delta 写入同时关闭）", flush=True)

    comp = LM.LatentCompressor(hidden, n_slots=args.m, inner_dim=args.inner_dim,
                               num_heads=args.heads).to(dev)
    with torch.no_grad():
        _emb_w = model.get_input_embeddings().weight
        target_norm = float(_emb_w.float().norm(dim=-1).mean())
    bridge = LM.LatentBridge(hidden, bottleneck=args.bottleneck,
                             target_norm=target_norm).to(dev)
    fst_router = LM.FstRouter(hidden, n_slots=args.m, d_key=args.d_key).to(dev)
    if args.init:
        blob = LM._safe_torch_load(args.init)
        comp.load_state_dict(blob["compressor"])
        bridge.load_state_dict(blob["bridge"])
        if blob.get("fst_router"):
            fst_router.load_state_dict(blob["fst_router"])
            print(f"[init] FST Router 也一并从 {args.init} 载入", flush=True)
        print(f"[init] 从 {args.init} 初始化", flush=True)
    if args.freeze_compressor:
        for prm in comp.parameters():
            prm.requires_grad_(False)
        comp.eval()
        print("[freeze] Compressor 主干已冻结（只训 FST Router + Bridge）", flush=True)
    n_train = (sum(p.numel() for p in comp.parameters() if p.requires_grad)
               + sum(p.numel() for p in bridge.parameters())
               + sum(p.numel() for p in fst_router.parameters()))
    print(f"[params] 可训练 {n_train/1e6:.3f}M "
          f"(compressor={sum(p.numel() for p in comp.parameters() if p.requires_grad)/1e6:.3f}M,"
          f" bridge={sum(p.numel() for p in bridge.parameters())/1e6:.3f}M,"
          f" fst_router={sum(p.numel() for p in fst_router.parameters())/1e6:.3f}M) "
          f"hist_max={args.hist_max}", flush=True)

    train_data = load_synth_jsonl(args.data, limit=args.n_train, seed=args.seed)
    if not train_data:
        raise SystemExit(f"--data 读不到任何样本：{args.data}")
    if args.val_data:
        val_data = load_synth_jsonl(args.val_data, limit=args.n_val, seed=args.seed + 1)
    else:
        val_data = train_data[:max(1, min(len(train_data), args.n_val or 8))]
        print("[data][warn] 未给 --val-data，用训练集前若干条做通路检查，"
              "该数字不是验证指标", flush=True)
    for prob in check_split_leakage(train_data, val_data):
        print(f"[data][warn] 切分不干净：{prob}", flush=True)
    probes = Counter(it.get("probe") or "?" for it in train_data)
    print(f"[data] {args.data}: train={len(train_data)} val={len(val_data)} "
          f"探针={dict(sorted(probes.items()))} 平均记忆单元="
          f"{sum(len(item_units(it)) for it in train_data)/max(1, len(train_data)):.1f}",
          flush=True)
    print(f"[data] 例：units={item_units(train_data[0])} "
          f"q={train_data[0]['question']!r} a={train_data[0]['answer']!r} "
          f"as_of={train_data[0].get('as_of')!r}", flush=True)

    encoder = LM.LatentEncoder(model, tokenizer, comp, layer=args.layer,
                               max_unit_tokens=args.max_unit_tokens, device=dev)
    system_prompt = args.system_prompt
    opt_params = list(comp.parameters()) + list(bridge.parameters()) + \
        list(fst_router.parameters())
    opt = torch.optim.AdamW([p for p in opt_params if p.requires_grad], lr=args.lr)
    cache = {}
    max_steps = args.max_steps if args.max_steps > 0 else max(1, len(train_data) // args.accum)
    if args.freeze_compressor:
        comp.eval()
    else:
        comp.train()
    bridge.train()
    fst_router.train()

    def _cfg(steps_done):
        return {"schema": LM.LATENT_SCHEMA, "n_slots": args.m, "layer": args.layer,
                "inner_dim": args.inner_dim, "num_heads": args.heads,
                "bottleneck": args.bottleneck, "max_unit_tokens": args.max_unit_tokens,
                "target_norm": round(float(target_norm), 6),
                "read_budget": LM.LATENT_READ_BUDGET, "dtype": LM.LATENT_DTYPE,
                "stage": "fst", "hist_max": int(args.hist_max),
                "addr_weight": float(args.addr_weight), "ver_weight": float(args.ver_weight),
                "lr": args.lr, "steps": steps_done, "epochs": args.epochs,
                "seed": args.seed, "smoke_model": bool(args.smoke_model),
                "router": False, "fst": True, "d_key": int(args.d_key)}

    step = micro = 0
    acc = []
    grad_seen = None
    skipped_evicted = 0
    skipped_ver = 0
    t_start = time.time()
    for epoch in range(args.epochs):
        order = list(range(len(train_data)))
        random.Random(args.seed + 100 + epoch).shuffle(order)
        for i in order:
            item = train_data[i]
            reset_delta_state(model)
            fi = build_fst_bank(encoder, item, args.hist_max, tag="fact",
                                grad=not args.freeze_compressor)
            if fi["evicted"]:
                # 目标版本已被容量淘汰：这条样本在当前 hist_max 下**无解**。
                # 拿它算 CE 等于把 Router 往"选一个已经被删掉的版本"上训。
                skipped_evicted += 1
                continue
            if fi["label_mismatch"]:
                raise RuntimeError(
                    f"答案时间戳与任何版本都不相等（数据/格式问题，不许静默当 0）："
                    f"q={item['question']!r} answer_fact_time="
                    f"{item.get('answer_fact_time')!r} 版本时间="
                    f"{[v['time'] for v in fi['bank'].list_versions(fi['tgt_key'])]}")
            # 没有时间戳 -> 只有版本级监督无法给出，QA 与组级监督照常训练
            skipped_ver += int(not fi["viable"])
            e_feat, r_feat, ver_z, ver_mask, ver_t_feat = fst_build_inputs(
                fi, model, tokenizer, args.layer, dev, cache)
            tq = (None if args.ignore_query_time else
                  fst_query_time_feat(item, model, tokenizer, args.layer, dev, cache))
            q_feat = query_feature_cached(model, tokenizer, item["question"],
                                          system_prompt, args.layer, dev)
            out = fst_router(q_feat, e_feat, r_feat, ver_z, ver_t_feat, tq_feat=tq,
                             ver_mask=ver_mask, hard=False)
            soft = bridge(out["z_read"])
            embeds, labels, n_ans = build_student_inputs(
                model, tokenizer, soft, system_prompt, item["question"], item["answer"])
            with LM.delta_write_disabled(model):
                s_logits = model(inputs_embeds=embeds,
                                 attention_mask=torch.ones(embeds.shape[:2],
                                                           dtype=torch.long, device=dev),
                                 use_cache=False).logits
            loss_qa = qa_loss(s_logits, labels)
            t_g = torch.tensor([int(fi["tgt_group"])], device=out["logits_group"].device)
            # 版本监督打在**目标组**那一行：用 logits_ver（被选中组）会在选错组时
            # 监督到另一组的版本布局（甚至填充位），loss 瞬间到 1e4 量级。
            loss_grp = F.cross_entropy(out["logits_group"], t_g)
            if fi["viable"]:
                t_v = torch.tensor([int(fi["tgt_ver"])],
                                   device=out["logits_ver"].device)
                loss_ver = F.cross_entropy(
                    out["logits_ver_all"][int(fi["tgt_group"]):int(fi["tgt_group"]) + 1],
                    t_v)
            else:
                # 没有可定位的目标版本：不能凭空造监督，也不能因此丢掉整条样本
                loss_ver = out["logits_ver_all"].sum() * 0.0
            loss = loss_qa + args.addr_weight * loss_grp + args.ver_weight * loss_ver
            (loss / args.accum).backward()
            acc.append((float(loss_qa.detach()), float(loss_grp.detach()),
                        float(loss_ver.detach())))
            micro += 1
            if micro % args.accum:
                continue
            if grad_seen is None:
                grad_seen = {
                    "bridge_last_linear": bridge.net[3].weight.grad is not None,
                    "fst_key_e_net": fst_router.key.e_net[0].weight.grad is not None,
                    "fst_key_output": fst_router.key.out[0].weight.grad is not None,
                    "fst_state_ver_net": fst_router.state.ver_net[0].weight.grad is not None,
                    "model_has_no_grad": all(p.grad is None for p in model.parameters()),
                }
                if not args.freeze_compressor:
                    grad_seen["compressor_queries"] = comp.queries.grad is not None
            g_comp = 0.0 if args.freeze_compressor else _grad_norm(comp.parameters())
            g_bridge = _grad_norm(bridge.parameters())
            g_fst = _grad_norm(fst_router.parameters())
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in opt_params if p.requires_grad], args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            qa_avg = sum(x[0] for x in acc) / len(acc)
            grp_avg = sum(x[1] for x in acc) / len(acc)
            ver_avg = sum(x[2] for x in acc) / len(acc)
            print(f"[step {step}] loss_qa={qa_avg:.4f} loss_grp={grp_avg:.4f} "
                  f"loss_ver={ver_avg:.4f} "
                  f"soft_tokens={int(soft.reshape(1, -1, soft.shape[-1]).shape[1])} "
                  f"compress={fi['info']['compression_ratio']} "
                  f"gnorm_comp={g_comp:.3e} gnorm_bridge={g_bridge:.3e} "
                  f"gnorm_fst={g_fst:.3e} skip_evicted={skipped_evicted} "
                  f"skip_ver={skipped_ver} "
                  f"({time.time()-t_start:.0f}s)", flush=True)
            acc = []
            if args.eval_every > 0 and step % args.eval_every == 0 and not args.dry_run:
                res_ev = evaluate_fst(fst_router, comp, bridge, encoder, model,
                                      tokenizer, val_data, system_prompt, args.layer,
                                      args.hist_max, limit=args.eval_limit,
                                      cache=cache, verbose=False,
                                      label=f"fst@{step}",
                                      ignore_query_time=args.ignore_query_time)
                print(f"[eval] step={step} {_fmt_fst(res_ev)}", flush=True)
                if not args.freeze_compressor:
                    comp.train()
                bridge.train()
                fst_router.train()
            if args.save_every > 0 and step % args.save_every == 0 and not args.dry_run:
                LM.save_latent_checkpoint(
                    args.out, comp, bridge, _cfg(step),
                    {"trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "git": _git_commit(), "partial": True, "step": step},
                    fst_router=fst_router)
                print(f"[ckpt] step {step} 已保存 -> {args.out}", flush=True)
            if args.max_steps > 0 and step >= max_steps:
                break
        if args.max_steps > 0 and step >= max_steps:
            break

    if grad_seen is None:
        raise RuntimeError("训练没有完成任何一个优化步：检查 --accum/--max-steps/--epochs "
                           "以及 skipped_evicted（hist_max 太小会让所有样本都不可学）")
    print(f"[grad-check] {json.dumps(grad_seen, ensure_ascii=False)}", flush=True)
    if not all(v for k, v in grad_seen.items() if k != "model_has_no_grad"):
        raise RuntimeError("梯度没有回传到 Bridge/FST Router 的某个头 —— "
                           "检查是否误把 student 前向放进 no_grad() 或提前 detach")
    if not grad_seen["model_has_no_grad"]:
        print("[grad-check][warn] 冻结权重拿到了梯度：确认 requires_grad_(False) 生效",
              flush=True)

    res_val = evaluate_fst(fst_router, comp, bridge, encoder, model, tokenizer, val_data,
                           system_prompt, args.layer, args.hist_max,
                           limit=args.eval_limit, cache=cache, verbose=True,
                           label="fst-final", ignore_query_time=args.ignore_query_time)
    print(f"[eval] val {_fmt_fst(res_val)}", flush=True)
    res_train = evaluate_fst(fst_router, comp, bridge, encoder, model, tokenizer,
                             train_data, system_prompt, args.layer, args.hist_max,
                             limit=min(args.eval_limit, len(train_data)), cache=cache,
                             verbose=False, label="fst-train",
                             ignore_query_time=args.ignore_query_time)
    print(f"[eval] train {_fmt_fst(res_train)}", flush=True)
    # 顺序稳定性：换一个写入顺序，答案必须一样（组合稳定性的直接检验）
    res_ord = evaluate_fst(fst_router, comp, bridge, encoder, model, tokenizer, val_data,
                           system_prompt, args.layer, args.hist_max,
                           limit=args.eval_limit, cache=cache, verbose=False,
                           label="fst-order", shuffle_order=True, seed=args.seed,
                           ignore_query_time=args.ignore_query_time)
    print(f"[eval] val(写入顺序打乱) {_fmt_fst(res_ord)}", flush=True)

    if args.dry_run:
        print(f"[dry-run] OK：FST 通路验证完成（{time.time()-t0:.1f}s），未写 checkpoint",
              flush=True)
        return 0

    cfg = _cfg(step)
    meta = {"trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git": _git_commit(), "n_train": len(train_data), "n_val": len(val_data),
            "fst": True, "hist_max": int(args.hist_max),
            "val": res_val, "train": res_train, "val_order_shuffled": res_ord,
            "grad_check": grad_seen, "skipped_evicted": int(skipped_evicted),
            "elapsed_sec": round(time.time() - t0, 1)}
    out = LM.save_latent_checkpoint(args.out, comp, bridge, cfg, meta,
                                    fst_router=fst_router)
    print(f"[save] {out} config={json.dumps(cfg, ensure_ascii=False)}", flush=True)
    # 存盘->重载->评：FST 多了一个 Router 要恢复，这一步顺便验证 ckpt 完整性
    rd = LM.load_latent_reader(out, model, tokenizer, device=dev)
    if rd.fst_router is None:
        raise RuntimeError("重载后 fst_router 为空：checkpoint 不完整")
    res_rl = evaluate_fst(rd.fst_router, rd.compressor, rd.bridge, rd.encoder, model,
                          tokenizer, val_data, system_prompt, args.layer, args.hist_max,
                          limit=args.eval_limit, cache=cache, verbose=False,
                          label="fst-reload", ignore_query_time=args.ignore_query_time)
    print(f"[eval-reload] {_fmt_fst(res_rl)}", flush=True)
    if res_val["acc"] is not None and res_rl["acc"] is not None \
            and abs(res_val["acc"] - res_rl["acc"]) > 0.2:
        print(f"[WARNING] 进程内({res_val['acc']:.3f}) 与 重载({res_rl['acc']:.3f}) "
              f"差异巨大：训练进程内存在影响生成的状态", flush=True)
    return 0


def _f3(v):
    """None-safe 的 3 位小数（空集时指标是 None，不是 0）。"""
    return "n/a" if v is None else f"{v:.3f}"


def _fmt_fst(res):
    """把 FST 指标压成一行（成绩单口径固定，便于多次实验横向对比）。"""
    if not res:
        return "n/a"
    pr = " ".join(f"{k}={_f3(v['acc'])}({v['hit']}/{v['n']})"
                  for k, v in res["probes"].items())
    return (f"acc={_f3(res['acc'])} group={_f3(res['group_acc'])}"
            f"(const={_f3(res.get('group_const_acc'))}) "
            f"ver={_f3(res['ver_acc'])} "
            f"ver_multi={_f3(res['ver_multi_acc'])}({res['ver_multi_n']},"
            f"newest={_f3(res.get('ver_newest_acc'))}) pair={_f3(res['pair_acc'])}"
            f"({res['pair_n']}) {pr} n_soft={res['n_soft']}"
            f" ev={res.get('evicted_n')} nover={res.get('label_missing_n')}"
            + (f" order_stable={_f3(res['order_stable'])}"
               if res.get("order_stable") is not None else ""))


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
    ap.add_argument("--recon-units", type=int, default=2,
                    help="recon 每次只重建这么多条记忆单元（0=整段；整段在长记忆上会 OOM）")
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
    ap.add_argument("--fst", action="store_true",
                    help="FST-Mem（阶段 B）：事实键 + 状态/轨迹解耦 + 时序两级读出")
    ap.add_argument("--hist-max", type=int, default=LM.LATENT_HIST_MAX,
                    help="每个事实键保留的历史版本数；0 = 硬覆写对照（历史全丢）")
    ap.add_argument("--ignore-query-time", action="store_true",
                    help="消融：丢掉查询里的时间条件（版本级只看问题+载荷，不看年份）")
    ap.add_argument("--ver-weight", type=float, default=1.0,
                    help="版本级（时间）寻址的交叉熵权重")
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
    if args.fst:
        print(f"[cfg] fst=True hist_max={args.hist_max} "
              f"addr_weight={args.addr_weight} ver_weight={args.ver_weight}", flush=True)
        return train_fst(args)
    if args.hist_max != LM.LATENT_HIST_MAX and not args.fst:
        print("[cfg][warn] --hist-max 只在 --fst 下生效，当前被忽略", flush=True)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
