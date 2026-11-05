#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_latent_table.py — 四种方法在同一批数据、同一协议下的统一评测（结果表的一行）。

为什么要一个新脚本，而不是继续往训练脚本里塞评估
------------------------------------------------
结果表要求"同一冻结主模型、同一记忆输入、同一评测协议、同一 prompt 模板"。
把四组方法放进**同一个循环、同一次 `build_chat_parts`、同一次 generate**，差异只剩
"记忆以什么形式进入"：

| method | 记忆怎么进 prompt | 注入的 token 数 |
|---|---|---|
| `none`   | 不给任何记忆（下界：纯模型参数知识） | 0 |
| `text`   | system 段里写明文事实（BM25 取 top-k，k=0 表示全给） | 明文 token 数 |
| `oracle` | system 段里只放 evidence 原文（性能上界，不是可部署方法） | 明文 token 数 |
| `bridge` | 全部槽位的载荷过 Bridge → 软 token | K × m |
| `router` | Router 先选 1 个槽 → Bridge → 软 token | m |
| `fst`    | 两级寻址选 (事实, 版本) → Bridge → 软 token | m |
| `oracle_latent` | 金标槽 → Bridge(z_sem, z_lex)（残差锚定，当前方法） | K_gold × m |
| `oracle_lex`    | 金标槽 → **只用 z_lex**（α=0，纯 h0 词形直通） | K_gold × m |
| `oracle_sem`    | 金标槽 → **只用 Bridge(z_sem)**（不给 z_lex） | K_gold × m |
| `oracle_embed`  | 金标证据原文的真实 token embedding（不压缩，**句首截断**到预算） | 与 oracle_latent 对齐 |
| `oracle_embed_full`    | 同上但**不截断**（整条证据全给）；`--embed-inject {user,between}` 选位置 | 不压缩的上限 |
| `oracle_embed_salient` | 同等预算下按显著性（大写/数字/长词/非停用词）挑词 | 与 oracle_latent 对齐 |

软 token 的**注入位置**由 `--inject-mode` 控制（默认 `user`：user 消息体内的 `Evidence:` 段；
`between` 是旧行为，插在消息之间的角色真空区）。槽位围栏由 `--fence-style` 控制
（默认 `bracket`：每槽 `\n[j] `，2~3 token；`verbose` 是旧长围栏）。
ckpt 里记录了两者，**留空即以 ckpt 为准**（旧 ckpt 自动按 between+verbose 评测）。

明文基线与潜层基线**走同一个 `_emit_latent` 代码路径**（明文只是 `soft=None`），
所以"潜层掉点"不可能用"两条路模板不一样"解释掉。

指标只保留论文要的五项：答案准确率、更新准确率、历史访问准确率、实际注入 token 数、查询耗时。
（后两项是潜层记忆的**唯一卖点**：不占上下文长度，但要多付一次编码前向。）

用法
----
    # 明文基线（上界）
    python eval_latent_table.py --method oracle --data synth/hotpot_dev.jsonl --limit 200
    # 明文 RAG
    python eval_latent_table.py --method text --data synth/hotpot_dev.jsonl --text-topk 5
    # 潜层：直接拼接 / 路由 / FST
    python eval_latent_table.py --method bridge --ckpt ckpt/a.pt --data ...
    python eval_latent_table.py --method router --ckpt ckpt/b.pt --data ...
    python eval_latent_table.py --method fst    --ckpt ckpt/fst.pt --data synth/fst_update_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import torch

import latent_memory as LM
import train_latent_memory as T

METHODS = ("none", "text", "oracle", "oracle_latent", "oracle_embed",
           "oracle_embed_full", "oracle_embed_salient",
           "oracle_lex", "oracle_sem", "bridge", "router", "fst")

# 无压缩原生词嵌入的三条对照（同一批金标槽，只换"取哪些 token"）：
#   prefix  旧行为：从句首截 budget 个 token —— 英语事实句的核心新值常在句尾，
#           这个截法天然把答案切掉，是 19.0% 这个数字被低估的主因。
#   full    完全不截断：整条金标证据的原生词嵌入全给，量化"不压缩的上限"。
#   salient 同等预算下按显著性挑：大写开头 / 含数字 / 长词 / 非停用词 / 问题里没出现
#           过的词优先，再按原序拼回。用来回答"18 个 token 到底能装多少信息"。
EMBED_ARMS = {"oracle_embed": "prefix",
              "oracle_embed_full": "full",
              "oracle_embed_salient": "salient"}


def _salient_pick(tok, ids, budget, question):
    """在 budget 个 token 的预算内按**显著性**挑词，而不是从句首硬截。

    评分是代理指标（离线环境拿不到语料 IDF）：
        大写开头 +3（实体） / 含数字 +3（年份、数量） / 长度>=6 +1（实词）
        停用词或空 -2 / 问题里没出现过 +1（新信息）
    选完按**原序**排回去，保持句子的可读顺序（乱序拼接会额外引入噪声）。
    """
    n = int(ids.numel())
    if n <= 0:
        return ids
    budget = max(1, min(int(budget), n))
    q_toks = set(T._norm_ans(question).split())
    scored = []
    for pos in range(n):
        t = tok.decode([int(ids[pos])]).strip()
        low = t.lower()
        sc = 0.0
        if t[:1].isupper():
            sc += 3.0
        if any(ch.isdigit() for ch in t):
            sc += 3.0
        if len(low) >= 6:
            sc += 1.0
        if not low or low in T._STOPWORDS:
            sc -= 2.0
        if low and low not in q_toks:
            sc += 1.0
        scored.append((sc, pos))
    scored.sort(key=lambda x: (-x[0], x[1]))
    keep = sorted(p for _sc, p in scored[:budget])
    return ids[torch.tensor(keep, dtype=torch.long)]


def embed_arm_ids(item, ctx, arm):
    """按 arm 取"金标证据的原生 token id"。返回 (ids 或 None, 预算, 槽数)。

    arm:
        prefix  —— 每槽从**句首**截 budget 个（旧行为）
        full    —— 每槽全部 token，不截断
        salient —— 每槽在同等 budget 内按显著性挑
    预算口径与 `oracle_latent` 对齐：每槽 ceil(unit_tokens/max_unit_tokens) 个块 × m 个
    向量，且 token 数按 `unit_encode_text`（"[标题] 原文"，压缩器实际看到的文本）计算。
    """
    tok = ctx["tokenizer"]
    units = T.item_units(item)
    idxe = [i for i in sorted(gold_slots(item, units)) if 0 <= i < len(units)]
    if not idxe:
        return None, 0, 0
    mut = max(1, int(ctx.get("max_unit_tokens") or 32))
    m = max(1, int(ctx.get("m_slots") or 4))
    fixed = int(ctx.get("embed_budget") or 0)
    if arm == "full":
        budgets = [None] * len(idxe)
    elif fixed > 0:
        share = max(1, fixed // max(1, len(idxe)))
        budgets = [share] * len(idxe)
    else:
        budgets = []
        for i in idxe:
            _ids = LM._tokenize_no_special(tok, T.unit_encode_text(item, i), ctx["dev"])
            n = max(1, int(_ids.shape[1]))
            budgets.append(max(1, (n + mut - 1) // mut) * m)
    chunks = []
    for i, b in zip(idxe, budgets):
        ids = LM._tokenize_no_special(tok, T.unit_encode_text(item, i), ctx["dev"])[0]
        if b is not None:
            ids = (_salient_pick(tok, ids, b, item["question"])
                   if arm == "salient" else ids[:int(b)])
        if int(ids.numel()) > 0:
            chunks.append(ids)
    if not chunks:
        return None, 0, len(idxe)
    return torch.cat(chunks, dim=0), int(sum(len(c) for c in chunks)), len(idxe)

# ---------------------------------------------------------------------
# 指标与字段口径
# ---------------------------------------------------------------------
_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_ARTICLES = {"a", "an", "the"}
_HOP_TYPES = ("bridge", "comparison")


def squad_norm(s):
    """SQuAD 式规范化：小写、去标点、去冠词、压空白。"""
    s = str(s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def em_score(pred, gold):
    """HotpotQA 官方主指标 Answer EM（规范化后精确匹配）。

    与 acc 的区别必须两边都报：acc 是"包含匹配"（_norm_ans 子串口径），
    22.5% 的样本答案串在干扰单元里也出现过，只要模型复读上下文就能拿到 acc，
    但 EM 拿不到 —— 只用 acc 会系统性高估"真的答对了"。
    """
    return float(squad_norm(pred) == squad_norm(gold))


def f1_score(pred, gold):
    p, g = squad_norm(pred).split(), squad_norm(gold).split()
    if not p or not g:
        return float(p == g)
    cnt = {}
    for w in p:
        cnt[w] = cnt.get(w, 0) + 1
    same = 0
    for w in g:
        if cnt.get(w, 0) > 0:
            cnt[w] -= 1
            same += 1
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


# 答案表面形式 -> 题型。HotpotQA 官方 type 只有 bridge/comparison 两类，但"答案长
# 什么样"决定了读出端要从记忆里捞什么（一个日期？一个实体名？），这是不同难度的
# 任务，混在总 EM 里会互相抵消 —— 例如"日期题全灭、实体题变好"会显示成没变化。
_MONTHS = {m.lower() for m in (
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec")}
_ORD = re.compile(r"^\d+(st|nd|rd|th)$", re.I)


def answer_type(ans):
    """金标答案 -> 题型（启发式，只用于分层统计，不参与判定）。"""
    a = str(ans or "").strip()
    low = a.lower()
    if low in ("yes", "no"):
        return "yesno"
    if not a:
        return "empty"
    if re.fullmatch(r"\d{4}", a) or _ORD.match(a):
        return "date"
    if any(t in _MONTHS for t in low.replace(",", " ").split()):
        return "date"
    if re.fullmatch(r"[\d\s.,%-]+", a):
        return "number"
    return "entity"


def item_hops(item):
    """与 train.item_hops 同一套规则：item["hops"] → meta["type"] → 1。"""
    v = item.get("hops")
    try:
        if v:
            return max(1, int(v))
    except (TypeError, ValueError):
        pass
    t = str((item.get("meta") or {}).get("type") or "").strip().lower()
    return 2 if t in _HOP_TYPES else 1


def n_ev_of(item):
    """金标证据句数：读出预算的口径锚点（>2 句时固定 2 跳物理上不可能全覆盖）。"""
    return len(item.get("evidence") or [])


# =====================================================================
# 明文基线：BM25 检索（纯 numpy，不引外部依赖）
# =====================================================================
_WORD = re.compile(r"[a-z0-9]+")


def _toks(s):
    return _WORD.findall(str(s).lower())


def bm25_topk(units, query, k, k1=1.5, b=0.75):
    """返回按 BM25 分数降序的前 k 个单元（k<=0 表示全部，保持原顺序）。"""
    if k is None or k <= 0 or k >= len(units):
        return list(range(len(units)))
    q = _toks(query)
    docs = [_toks(u) for u in units]
    if not q or not docs:
        return list(range(min(k, len(units))))
    df = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    n = len(docs)
    avg = sum(len(d) for d in docs) / max(1, n)
    scores = []
    for d in docs:
        tf = {}
        for w in d:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df.get(w, 0) + 0.5) / (df.get(w, 0) + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / max(1e-6, avg)))
        scores.append((s, -len(d)))
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return order[:k]


def gold_slots(item, units):
    """黄金槽位（支持事实所在单元）。没有 evidence 就返回空集 —— 不猜。"""
    ev = {T._norm_ans(e) for e in (item.get("evidence") or [])}
    return {i for i, u in enumerate(units) if T._norm_ans(u) in ev}


def memory_block(units, idx):
    return "Memory:\n" + "\n".join(f"- {units[i]}" for i in idx) + "\n"


# =====================================================================
# 每条样本的四种读法
# =====================================================================
def n_tokens_of(tokenizer, text, dev):
    return int(LM._tokenize_no_special(tokenizer, text, dev).shape[1])


def _zero_alpha(bridge):
    """把 bridge 的残差门控临时压到 0（返回原值，供 _restore_alpha 还原）。

    用途：oracle_lex 要测"**只有** z_lex 的读出"。直接传 lex_list 仍会叠加
    α·Bridge(z_sem) 残差，那就不是纯词形分支了。改参数而不是改代码路径，
    保证除 α 之外与 oracle_latent 逐位相同。
    """
    if not isinstance(bridge, LM.LatentBridge) or not getattr(bridge, "lex_skip", False):
        return None
    saved = bridge.lex_gate.detach().clone()
    with torch.no_grad():
        bridge.lex_gate.zero_()
    return saved


def _restore_alpha(bridge, saved):
    if saved is None:
        return
    with torch.no_grad():
        bridge.lex_gate.copy_(saved)


def read_one(method, item, ctx):
    """返回 (pred, info)，info 里有注入 token 数与各段耗时（毫秒）。"""
    model, tok, dev = ctx["model"], ctx["tokenizer"], ctx["dev"]
    T.reset_delta_state(model)      # 样本间隔离（在线状态是跨 forward 累积的）
    units = T.item_units(item)
    question = item["question"]
    sysp = ctx["system_prompt"]
    t0 = time.time()

    if method == "none":
        # 无记忆对照。**必须有**：HotpotQA 的答案大量存在于 Qwen 的参数记忆里，
        # 不给记忆也可能答对；没有这一行就无法判断"记忆有没有被真正读到"。
        pred, meta = LM._emit_latent(model, tok, None, question, sysp,
                                     max_new_tokens=ctx["max_new_tokens"],
                                     pad_token_id=ctx["pad_token_id"])
        info = {"mem_tokens": 0, "mem_plain_tokens": 0,
                "prompt_tokens": n_tokens_of(tok, question, dev),
                "pre_ms": 0.0, "enc_ms": 0.0, "gen_ms": (time.time() - t0) * 1e3,
                "total_ms": (time.time() - t0) * 1e3,
                "n_read_units": 0, "decode_mode": meta.get("decode_mode")}
        return pred, info

    if method in ("text", "oracle"):
        if method == "oracle":
            idx = [i for i, u in enumerate(units)
                   if T._norm_ans(u) in {T._norm_ans(e) for e in item.get("evidence") or []}]
            if not idx:                       # oracle 找不到目标事实 -> 退化为全给并记账
                idx = list(range(len(units)))
        else:
            idx = bm25_topk(units, question, ctx["text_topk"])
        sys_full = sysp + "\n\n" + memory_block(units, idx)
        t_pre = time.time()
        pred, meta = LM._emit_latent(model, tok, None, question, sys_full,
                                     max_new_tokens=ctx["max_new_tokens"],
                                     pad_token_id=ctx["pad_token_id"])
        t_gen = time.time()
        _gs2 = gold_slots(item, units)
        _n_mem = n_tokens_of(tok, memory_block(units, idx), dev)
        info = {"mem_tokens": _n_mem, "mem_plain_tokens": _n_mem,
                "prompt_tokens": n_tokens_of(tok, sys_full, dev)
                + n_tokens_of(tok, question, dev),
                "pre_ms": 0.0, "enc_ms": 0.0, "gen_ms": (t_gen - t_pre) * 1e3,
                "total_ms": (time.time() - t0) * 1e3,
                "n_read_units": len(idx), "decode_mode": meta.get("decode_mode"),
                # 与潜层方法的"读对槽位"可直接比较的可部署指标：取回的 top-k 里
                # 有没有黄金事实。不设这一列的话，"文本 RAG 掉点"到底是因为
                # **没检索到**还是**读到了但答错**无法区分。
                # 多跳口径统一：整条证据链都被取回才算读对（与潜层列同一条规则）。
                "read_ok": ((bool(_gs2 <= set(idx)) if len(_gs2) > 1
                             else bool(set(idx) & _gs2)) if _gs2 else None),
                # 完整证据召回：整条证据链都被取回才算。必须与潜层同一条规则落盘，
                # 否则 summarize 的 fa（read_all is not None）在这一支永远为空，
                # 于是 text/oracle 的 full_recall 恒为 null —— 表现为"这列没算"。
                "read_all": bool(_gs2 <= set(idx)) if _gs2 else None,
                "picked": list(idx), "gold": sorted(_gs2)}
        return pred, info

    if method in EMBED_ARMS:
        # ---- 读出机制归因：**不压缩**，直接注入金标证据的原生 token embedding ----
        # 三条 arm 只差"取哪些 token"，注入位置由 --embed-inject 控制：
        #   prefix  句首截 budget 个（旧行为，会把句尾的核心新值切掉）
        #   full    完全不截断（量化"不压缩的上限"）
        #   salient 同等 budget 下按显著性挑（大写/数字/长词/非停用词/问题里没出现过）
        # 位置对照（in-message vs between-message）用 --embed-inject 切换：
        # 同一个 arm 跑两遍，差值就是"注入位置不当"造成的损失。
        arm = EMBED_ARMS[method]
        _ids, _budget, _nslots = embed_arm_ids(item, ctx, arm)
        if _ids is None:
            return "", {"oracle_no_evidence": True, "mem_tokens": 0,
                        "mem_plain_tokens": 0, "prompt_tokens": 0, "pre_ms": 0.0,
                        "enc_ms": 0.0, "gen_ms": 0.0,
                        "total_ms": (time.time() - t0) * 1e3, "n_read_units": 0,
                        "read_ok": None, "read_all": None, "picked": [], "gold": []}
        emb = model.get_input_embeddings()(_ids.view(1, -1))
        _inj = str(ctx.get("embed_inject") or "user")
        pred, meta = LM._emit_latent(
            model, tok, emb, question, sysp,
            max_new_tokens=ctx["max_new_tokens"], pad_token_id=ctx["pad_token_id"],
            inject_mode=_inj,
            extra_meta={"has_payload": True, "oracle_embed": True, "embed_arm": arm,
                        "embed_inject": _inj, "embed_budget": int(_budget)})
        info = {"mem_tokens": int(meta.get("n_soft", 0)), "mem_plain_tokens": 0,
                "prompt_tokens": int(meta.get("n_in", meta.get("n_soft", 0))),
                "pre_ms": 0.0, "enc_ms": 0.0,
                "gen_ms": (time.time() - t0) * 1e3,
                "total_ms": (time.time() - t0) * 1e3,
                "n_read_units": int(_nslots),
                "read_ok": True, "read_all": True,
                "picked": sorted(gold_slots(item, units)),
                "gold": sorted(gold_slots(item, units)),
                "embed_arm": arm, "embed_inject": _inj,
                "embed_budget": int(_budget),
                # 注入 token 的构成必须分开记账，否则"给得多反而更差"会被误读成
                # "压缩器不行"，而实际上是围栏泡沫占了大多数位置。
                "n_soft_content": meta.get("n_soft_content"),
                "n_fence": meta.get("n_fence"),
                "inject_mode": meta.get("inject_mode")}
        return pred, info

    # ---- 潜层：先编码（所有潜层方法都要付这笔前向） ----
    if method == "fst":
        fi = T.build_fst_bank(ctx["encoder"], item, ctx["hist_max"], tag="eval",
                              grad=False, cache=ctx.get("unit_cache"))
        bank = fi["bank"]
        t_enc = time.time()
        pred, meta = LM.generate_latent_answer_fst(
            model=model, tokenizer=tok, bridge=ctx["bridge"], bank=bank,
            question=question, system_prompt=sysp,
            fst_router=ctx["fst_router"], max_new_tokens=ctx["max_new_tokens"],
            pad_token_id=ctx["pad_token_id"], layer=ctx["layer"], device=dev,
            query_time=item.get("as_of"), feat_cache=ctx["cache"],
            hops=max(1, int(ctx.get("read_groups") or 1)),
            ver_policy=str(ctx.get("ver_policy") or "hybrid"))
        t_gen = time.time()
        _groups = [int(x) for x in (meta.get("groups_picked") or [meta["group_picked"]])]
        _g_ok = bool(meta["group_keys"][int(meta["group_picked"])] == fi["tgt_key"])
        # 统一 token 口径：mem_tokens = 注入的**软** token 数；prompt_tokens =
        # 这次前向实际吃进去的全部上下文（软 token + system + question），
        # 取 meta["n_in"]。旧代码把 prompt_tokens 也写成 n_soft，
        # 于是"潜层只占 m 个 token"与"整个 prompt 有多长"两件事被混成一个数，
        # 表里的 token 列无法与明文基线对照（明文那列是真·上下文长度）。
        info = {"mem_tokens": int(meta["n_soft"]), "mem_plain_tokens": 0,
                "prompt_tokens": int(meta.get("n_in", meta["n_soft"])),
                "pre_ms": (t_enc - t0) * 1e3,
                "enc_ms": (t_enc - t0) * 1e3, "gen_ms": (t_gen - t_enc) * 1e3,
                "total_ms": (time.time() - t0) * 1e3,
                "n_read_units": len(_groups),
                # 更新/历史口径的判定用得到（答案对 + 读对版本才算"更新准确率"）
                "group_ok": _g_ok,
                "ver_ok": bool(int(meta["ver_picked"]) == int(fi["tgt_ver"]) and fi["viable"]),
                "probe": item.get("probe") or "",
                "decode_mode": meta.get("decode_mode"),
                "read_ok": _g_ok}
        return pred, info

    slot_z, _cat, _ei, kfeat, lex_payloads = T.encode_item_slots_full(
        ctx["encoder"], item, tag="eval", cache=ctx.get("unit_cache"))
    # 键-载荷解耦：寻址键用**单元级池化键**，读出载荷用**块级保真**版本。
    # 旧代码在评测时用 stack_payloads 的块轴均值同时当键和载荷，把
    # --max-unit-tokens 32 切出来的多块长句内容直接抹平（训练侧是多块 cat）。
    slot_reads = LM.pack_slot_reads(list(slot_z))
    slot_z = LM.pool_slot_keys(list(slot_z)).unsqueeze(0)     # [1,K,m,d]
    t_enc = time.time()
    router = ctx["router"]
    qf_ex = LM.extract_query_feature_ex(model, tok, question, sysp, layer=ctx["layer"])
    _gs = gold_slots(item, units)
    if method == "oracle_latent":
        # ---- 诊断模式：完美寻址下的**读出上界** ----
        # 直接把金标槽送进 bridge（不做任何寻址），其余（bridge、软 token 注入、
        # 解码）与 router 方法逐位相同。它回答的是唯一一个问题：
        # "如果寻址全对，这套潜层读出能到多少？"
        # oracle_latent ≈ oracle 但 router 差 => 瓶颈全在寻址（P01/P02 的方向）；
        # oracle_latent 也低          => bridge/压缩本身丢了信息（P06 的方向）。
        # 没有 evidence 就没有"黄金槽位"这回事：不拿 slot 0 冒充（那会把一根
        # 无关事实当成上界，凭空抬高或压低 oracle_latent），显式记账为无上界样本。
        _idx = [i for i in sorted(_gs) if i < len(slot_reads)]
        if not _idx:
            info_no_oracle = {"oracle_no_evidence": True, "mem_tokens": 0,
                              "mem_plain_tokens": 0, "prompt_tokens": 0,
                              "pre_ms": 0.0, "enc_ms": (t_enc - t0) * 1e3,
                              "gen_ms": 0.0, "total_ms": (time.time() - t0) * 1e3,
                              "n_read_units": 0, "read_ok": None, "read_all": None,
                              "picked": [], "gold": []}
            return "", info_no_oracle
        _z = [slot_reads[i] for i in _idx]
        _lx = ([lex_payloads[i] for i in _idx if i < len(lex_payloads)]
               if lex_payloads is not None else None)
        # 隔离参数必须**一路传到 _emit_latent**：旧代码只把 isolate 传给
        # compose_routed_soft（拿到的是加倍的围栏），却没把 seg_ids 交给注入端，
        # 于是 oracle_latent 在 eager 后端拿不到块对角掩码，而 router 方法拿得到 ——
        # 两个"只差寻址"的方法在隔离行为上并不同构，oracle 上界就被系统性压低。
        _iso = bool(ctx.get("slot_isolation"))
        _soft_seg = LM.compose_routed_soft(model, tok, ctx["bridge"], _z, dev=dev,
                                          lex_list=_lx, isolate=_iso,
                                          return_segs=True,
                                          fence_style=ctx.get("fence_style"))
        soft, _seg_ids = _soft_seg
        pred, meta = LM._emit_latent(
            model, tok, soft, question, sysp,
            max_new_tokens=ctx["max_new_tokens"], pad_token_id=ctx["pad_token_id"],
            seg_ids=_seg_ids, isolate=_iso,
            inject_mode=ctx.get("inject_mode"),
            fence_style=ctx.get("fence_style"),
            extra_meta={"has_payload": True, "routed": _idx,
                        "hops": item_hops(item), "oracle_addressing": True})
        meta = dict(meta)
        meta["routed"] = _idx
    elif method in ("oracle_lex", "oracle_sem"):
        # ---- 读出分解（2026-11-04）：把"当前 oracle_latent"拆成两条腿 ----
        # oracle_lex：只用 z_lex（h0 符号直通，α 强制 0，不叠加任何语义残差）
        # oracle_sem：只用 Bridge(z_sem) 的单位化输出（完全不给 z_lex）
        # 两条腿分别对应"词形保真"与"语义映射"两个假设：
        #   oracle_lex 高、oracle_sem 低  -> 潜层能读出全靠原生词向量，语义分支在拖后腿
        #   oracle_sem 高、oracle_lex 低  -> 压缩语义确实编码了答案，词形不是必需
        #   两者都低                     -> 信息在 Compressor 里就没了（查容量/选词）
        _idx = [i for i in sorted(_gs) if i < len(slot_reads)]
        if not _idx:
            return "", {"oracle_no_evidence": True, "mem_tokens": 0,
                        "mem_plain_tokens": 0, "prompt_tokens": 0,
                        "pre_ms": 0.0, "enc_ms": (t_enc - t0) * 1e3,
                        "gen_ms": 0.0, "total_ms": (time.time() - t0) * 1e3,
                        "n_read_units": 0, "read_ok": None, "read_all": None,
                        "picked": [], "gold": []}
        _z = [slot_reads[i] for i in _idx]
        _lx = ([lex_payloads[i] for i in _idx if i < len(lex_payloads)]
               if lex_payloads is not None else None)
        _iso = bool(ctx.get("slot_isolation"))
        if method == "oracle_lex":
            if _lx is None:
                raise SystemExit("oracle_lex 需要 ckpt 打开 lex_skip（否则没有直通载荷）")
            _saved = _zero_alpha(ctx["bridge"])
            try:
                soft, _seg_ids = LM.compose_routed_soft(
                    model, tok, ctx["bridge"], _z, dev=dev, lex_list=_lx,
                    isolate=_iso, return_segs=True,
                    fence_style=ctx.get("fence_style"))
            finally:
                _restore_alpha(ctx["bridge"], _saved)
        else:
            soft, _seg_ids = LM.compose_routed_soft(
                model, tok, ctx["bridge"], _z, dev=dev, lex_list=None,
                isolate=_iso, return_segs=True,
                fence_style=ctx.get("fence_style"))
        pred, meta = LM._emit_latent(
            model, tok, soft, question, sysp,
            max_new_tokens=ctx["max_new_tokens"], pad_token_id=ctx["pad_token_id"],
            seg_ids=_seg_ids, isolate=_iso,
            inject_mode=ctx.get("inject_mode"),
            fence_style=ctx.get("fence_style"),
            extra_meta={"has_payload": True, "routed": _idx,
                        "hops": item_hops(item), "oracle_addressing": True,
                        "oracle_branch": method})
        meta = dict(meta)
        meta["routed"] = _idx
    else:
        pred, meta = LM.generate_latent_answer(
            model=model, tokenizer=tok, bridge=ctx["bridge"], payloads=slot_z,
            question=question, system_prompt=sysp, q_feat=qf_ex["pooled"],
            max_new_tokens=ctx["max_new_tokens"],
            pad_token_id=ctx["pad_token_id"], router=router, layer=ctx["layer"],
            hops=item_hops(item), reads=slot_reads, kfeat=kfeat,
            q_tok=qf_ex.get("h0_tok"), lex_reads=lex_payloads,
            topk_per_hop=int(ctx.get("topk_per_hop") or 1),
            isolate=int(ctx.get("slot_isolation") or 0),
            inject_mode=ctx.get("inject_mode"),
            fence_style=ctx.get("fence_style"))
        # q_feat 已由 extract_query_feature_ex 算好并显式传入，避免 generate 内重跑前向
    t_gen = time.time()
    _gs = gold_slots(item, units)
    _picked = [int(x) for x in (meta.get("routed") or [])]
    needs_full = len(_gs) > 1          # 多跳：证据链是**一组**槽位，不是随便一条
    info = {"mem_tokens": int(meta.get("n_soft", 0)), "mem_plain_tokens": 0,
            "prompt_tokens": int(meta.get("n_in", meta.get("n_soft", 0))),
            "pre_ms": (t_enc - t0) * 1e3,
            "enc_ms": (t_enc - t0) * 1e3, "gen_ms": (time.time() - t_enc) * 1e3,
            "total_ms": (time.time() - t0) * 1e3,
            "n_read_units": max(1, len(_picked)),
            "decode_mode": meta.get("decode_mode"),
            # 注入 token 的构成：内容向量 vs 围栏泡沫。这是本轮改动的核心观测量，
            # 不分开记账就分不清"给得多"和"给了有用的东西"。
            "n_soft_content": meta.get("n_soft_content"),
            "n_fence": meta.get("n_fence"),
            "inject_mode": meta.get("inject_mode"),
            "fence_style": meta.get("fence_style"),
            # 隔离到底有没有生效必须逐样本落盘：SDPA 下块对角会被静默丢弃，
            # 只报"开了 isolate=1"而不报实际生效模式，等于把降级藏起来。
            "slot_isolation": meta.get("slot_isolation"),
            # 多跳样本：只有**整条证据链**都被读到才算"读对"。
            # 旧口径"选中任意一条黄金事实就算对"会把"读了半条链"算成成功，
            # 于是 read_acc 虚高、与 acc 的落差被误读成"读出端有问题"。
            "read_ok": (bool(_gs <= set(_picked)) if needs_full
                        else (bool(set(_picked) & _gs) if _gs else None)),
            "read_all": bool(_gs <= set(_picked)) if _gs else None,
            "picked": _picked, "gold": sorted(_gs)}
    return pred, info


# =====================================================================
# 汇总
# =====================================================================
def summarize(rows, label):
    n = len(rows)
    if not n:
        return {"label": label, "n": 0}
    hit = sum(int(r["ok"]) for r in rows)
    groups = {}
    for r in rows:
        groups.setdefault(T._norm_ans(r["question"]), []).append(bool(r["ok"]))
    real = {k: v for k, v in groups.items() if len(v) >= 2 and len(set(v)) >= 2}
    pair_ok = sum(1 for v in real.values() if all(v))
    lat = [r["total_ms"] for r in rows]
    lat.sort()
    out = {
        "label": label, "n": n,
        "acc": hit / n,
        "pair_acc": (pair_ok / len(real)) if real else None,
        "pair_n": len(real),
        "mem_tokens": sum(r["mem_tokens"] for r in rows) / n,
        "mem_plain_tokens": sum(int(r.get("mem_plain_tokens") or 0) for r in rows) / n,
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows) / n,
        # 编码前向 vs 解码前向分开报：潜层的"多付一次编码"全部落在 enc_ms 上，
        # 合成一个 total_ms 就没法回答"400ms 预算里到底哪一段超了"。
        "enc_ms_mean": sum(float(r.get("enc_ms") or 0.0) for r in rows) / n,
        "gen_ms_mean": sum(float(r.get("gen_ms") or 0.0) for r in rows) / n,
        "ms_mean": sum(lat) / n,
        "ms_p50": lat[n // 2],
        "pre_ms_mean": sum(r["pre_ms"] for r in rows) / n,
    }
    reads = [bool(r["read_ok"]) for r in rows if r.get("read_ok") is not None]
    out["read_acc"] = (sum(reads) / len(reads)) if reads else None
    # ---- HotpotQA 官方口径：EM / F1（acc 是包含匹配，必须与它们同时报） ----
    out["em"] = sum(float(r.get("em") or 0.0) for r in rows) / n
    out["f1"] = sum(float(r.get("f1") or 0.0) for r in rows) / n
    # ---- 分层 EM / F1 ----
    # 总 EM 会把"日期题全灭、实体题变好"平均成"没变化"。分层之后才能看出
    # 潜层记忆到底丢了哪一类事实（符号精确值 vs 实体名）。
    for key, name in (("ans_type", "by_ans_type"), ("hops", "by_hops")):
        groups = {}
        for r in rows:
            groups.setdefault(str(r.get(key)), []).append(r)
        out[name] = {k: {"n": len(v),
                         "em": sum(float(x.get("em") or 0.0) for x in v) / len(v),
                         "f1": sum(float(x.get("f1") or 0.0) for x in v) / len(v)}
                     for k, v in sorted(groups.items())}
    # ---- 完整证据召回：两种口径都落盘 ----
    # 主表看 full_recall_leq2：读出预算固定 2 跳时，金标证据 >2 句的样本
    # 物理上不可能全覆盖；混在一起算会把"口径缺陷"读成"机制缺陷"。
    fa = [r for r in rows if r.get("read_all") is not None]
    out["full_recall"] = (sum(int(r["read_all"]) for r in fa) / len(fa)) if fa else None
    # n_leq2 是**数据集属性**（金标证据句数 ≤2 的样本数），不能从 fa 里数：
    # none 这种"不注入记忆"的方法没有 read_all，从 fa 数会得到 0，
    # 让"本集有多少样本在 2 跳预算内可全覆盖"这个分母凭空消失。
    out["n_leq2"] = sum(1 for r in rows if 0 < int(r.get("n_ev") or 0) <= 2)
    sub_ = [r for r in fa if int(r.get("n_ev") or 0) <= 2]
    out["full_recall_leq2"] = (sum(int(r["read_all"]) for r in sub_) / len(sub_)) if sub_ else None
    # ---- 两跳子集单报（本轮核心变量：混在总数里看不出多跳有没有生效） ----
    h2 = [r for r in rows if int(r.get("hops") or 1) > 1]
    if h2:
        out["n_h2"] = len(h2)
        out["acc_h2"] = sum(int(r["ok"]) for r in h2) / len(h2)
        h2r = [r for r in h2 if r.get("read_ok") is not None]
        if h2r:
            out["read_acc_h2"] = sum(int(r["read_ok"]) for r in h2r) / len(h2r)
    # FST 专属：更新准确率（答案对且读对版本）、历史访问准确率（只看 history 探针）
    if any("probe" in r for r in rows):
        upd = [r for r in rows if r.get("probe") in ("current", "history") or "probe" in r]
        if upd:
            out["update_acc"] = sum(int(r["ok"] and r.get("ver_ok")) for r in upd) / len(upd)
        hist = [r for r in rows if r.get("probe") == "history"]
        if hist:
            out["hist_acc"] = sum(int(r["ok"]) for r in hist) / len(hist)
    return out


def fmt_line(s):
    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}"
    return (f"| {s['label']} | {s['n']} | {pct(s['acc'])} | {pct(s.get('em'))}"
            f" | {pct(s.get('f1'))} | {pct(s.get('read_acc'))}"
            f" | {pct(s.get('full_recall_leq2'))}"
            f" | {pct(s.get('pair_acc'))}"
            f" | {pct(s.get('update_acc'))} | {pct(s.get('hist_acc'))}"
            f" | {s['mem_tokens']:.1f} | {s['prompt_tokens']:.1f}"
            f" | {s.get('enc_ms_mean', 0.0):.0f} | {s.get('gen_ms_mean', 0.0):.0f}"
            f" | {s['ms_mean']:.0f} |")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", "--mode", dest="method", required=True, choices=METHODS,
                    help="评测方法；--mode 是同一参数（与方案文档里的 `--mode oracle_latent` 对齐）")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="", help="潜层方法必需（bridge/router/fst）")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default=T.BASE)
    ap.add_argument("--adapter", default=T.ADAPTER)
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--device", default="")
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"),
                    help="主干精度（必须与训练一致，auto 会在 A100 上选 bf16）")
    ap.add_argument("--no-tf32", action="store_true", help="关闭 TF32")
    ap.add_argument("--smoke-model", action="store_true")
    ap.add_argument("--layer", type=int, default=LM.LATENT_LAYER)
    ap.add_argument("--max-unit-tokens", type=int, default=LM.LATENT_MAX_UNIT_TOKENS)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--embed-budget", type=int, default=0,
                    help="oracle_embed 每样本注入的真实 token 数（0=按潜层块数自动算）；"
                         "与 oracle_latent 的 mem_tokens 对齐后才能比较")
    ap.add_argument("--text-topk", type=int, default=5, help="text 方法的检索条数（0=全给）")
    ap.add_argument("--hist-max", type=int, default=LM.LATENT_HIST_MAX)
    ap.add_argument("--ver-policy", default="hybrid",
                    choices=("current", "time", "learned", "hybrid"),
                    help="FST 版本级读出规则（必须与训练时一致，否则是两套方法）")
    ap.add_argument("--read-groups", type=int, default=1,
                    help="FST 一次读出几个事实键（多跳/多事实题 >1）")
    # 与训练保持一致：2 跳 × 4 = 8 个读出名额（金标 >=3 条的题在 4 名额下召回到顶）
    ap.add_argument("--topk-per-hop", type=int, default=4,
                    help="路由方法每跳读出几个槽（1=v3 上限 2 槽；2=两跳共 4 槽）")
    ap.add_argument("--inject-mode", choices=("", "user", "system", "between"),
                    default="",
                    help="软 token 插在哪。留空=按 ckpt 里训练时的配置（推荐，训练/评测同源）；"
                         "between 是旧行为（消息之间的角色真空区），只用于 A/B 对照")
    ap.add_argument("--fence-style", choices=("", "bracket", "verbose"), default="",
                    help="槽位围栏样式。留空=按 ckpt 配置；verbose 是旧长围栏")
    ap.add_argument("--embed-inject", choices=("user", "system", "between"),
                    default="user",
                    help="oracle_embed* 三条基线把原生词嵌入放在哪：user=消息体内（默认）、"
                         "between=消息之间（旧软 token 插入点）。同一 arm 跑两遍即可量化"
                         "\"位置不当\"造成的损失")
    # ---- E2/E3：读出结构的评测期开关（默认跟随 ckpt，显式给了才覆盖）----
    ap.add_argument("--lex-token-mode", choices=("ckpt",) + tuple(LM._LEX_TOKEN_MODES),
                    default="ckpt",
                    help="E2：词形载荷口径。mean=词内子词均值（ckpt 训练时的默认）；"
                         "bpe=按原文顺序复制完整 BPE 向量（整词放不下就跳过，不截断）。"
                         "这是**零训练**对照：同一份权重换一种词形编码")
    ap.add_argument("--bpe-backfill", type=int, default=-1,
                    help="bpe 模式下是否用后续候选词补空位（-1=跟随 ckpt，默认不补）")
    ap.add_argument("--split-inject", type=int, default=-1, choices=(-1, 0, 1),
                    help="E3：1=词形与语义分开注入（[lex...]+[sem]）；0=残差叠加；"
                         "-1=跟随 ckpt。用残差模式训出来的权重直接开 1 只是个 sanity "
                         "check（Bridge 末层是零初始化，方向会退回池化后的语义向量），"
                         "真正的 E3 结论必须用独立训过的权重")
    ap.add_argument("--split-n-lex", type=int, default=-1,
                    help="E3：每块留给词形的 token 数（-1=跟随 ckpt）")
    ap.add_argument("--slot-isolation", type=int, default=LM.LATENT_SLOT_ISOLATION,
                    help="1=尝试槽位块对角隔离（仅 eager 后端生效，否则降级并记账）")
    ap.add_argument("--train-max-slots", type=int, default=0,
                    help="训练时用的 --max-slots，仅用于 train/eval 槽数一致性告警")
    ap.add_argument("--system-prompt",
                    default="You are a helpful assistant. Answer with the shortest exact value.")
    ap.add_argument("--label", default="")
    ap.add_argument("--save-json", default="")
    ap.add_argument("--progress-every", type=int, default=50,
                    help="每 N 题打印一次进度（acc / 已用时 / ETA）；0=关闭")
    ap.add_argument("--partial-every", type=int, default=100,
                    help="每 N 题把**当前已完成**的逐样本结果落一次盘（长跑中途崩掉不丢）")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--stratify", action="store_true",
                    help="按题型（yesno/date/number/entity）与跳数分层打印 EM/F1。"
                         "总 EM 会把\"日期题全灭 + 实体题变好\"平均成\"没变化\"")
    args = ap.parse_args(argv)

    dev = torch.device(args.device if args.device else
                       ("cpu" if args.smoke_model else "cuda:0"))
    args.gpu = str(args.gpu)
    T._setup_tf32(not getattr(args, "no_tf32", False))
    T.gpu_report(args)
    model, tok, dev, _hidden = T._load_backbone(args, dev)
    T.freeze_model(model)
    pad_id = getattr(tok, "eos_token_id", None) or getattr(tok, "pad_token_id", None) or 0

    ctx = {"model": model, "tokenizer": tok, "dev": dev,
           "system_prompt": args.system_prompt, "max_new_tokens": args.max_new_tokens,
           "pad_token_id": pad_id, "layer": args.layer,
           "text_topk": args.text_topk, "cache": {},
           # 文本 -> 载荷缓存：同一段上下文被多问共享时不重复编码（FST 每题 11s 的主因）
           "unit_cache": {}, "read_groups": args.read_groups,
           "ver_policy": args.ver_policy,
           "topk_per_hop": int(args.topk_per_hop),
           "slot_isolation": int(args.slot_isolation),
           "inject_mode": (args.inject_mode or LM.LATENT_INJECT_MODE),
           "fence_style": (args.fence_style or LM.LATENT_FENCE_STYLE),
           "embed_inject": str(args.embed_inject),
           # oracle_embed 用：每个金标槽给 m 个软 token，预算与 oracle_latent 对齐
           "m_slots": int(getattr(args, "m", 4) or 4),
           "max_unit_tokens": int(args.max_unit_tokens),
           "embed_budget": int(args.embed_budget)}

    if args.method in ("bridge", "router", "fst", "oracle_latent",
                       "oracle_lex", "oracle_sem"):
        if not args.ckpt:
            raise SystemExit(f"--method {args.method} 必须给 --ckpt（随机权重没有任何意义）")
        reader = LM.load_latent_reader(args.ckpt, model, tok, device=dev)
        # **ckpt 的配置优先于命令行默认值**：oracle_embed 的 token 预算按
        # "每个金标槽 ceil(unit_tokens/max_unit_tokens) 个块 × m" 算，若这里用 CLI
        # 默认 64 而 ckpt 训练时是 32，两个方法的预算就不是同一个数，对比不成立。
        # 命令行显式给了不同值时要**报警并采用 ckpt 的值**（不是静默采用命令行的）。
        _cfg_mut = int(reader.config.get("max_unit_tokens", args.max_unit_tokens))
        if int(args.max_unit_tokens) != _cfg_mut:
            print(f"[warn] --max-unit-tokens={args.max_unit_tokens} 与 ckpt 训练时的 "
                  f"{_cfg_mut} 不一致：预算对比会失真，已按 ckpt 的值运行", flush=True)
        args.max_unit_tokens = _cfg_mut
        _cfg_m = int(reader.config.get("n_slots", 4))
        ctx["max_unit_tokens"] = _cfg_mut
        ctx["m_slots"] = _cfg_m
        if int(args.layer) != int(reader.config.get("layer", args.layer)):
            print(f"[warn] --layer={args.layer} 与 ckpt 的 {reader.config.get('layer')} "
                  f"不一致，已按 ckpt 的值运行", flush=True)
        args.layer = int(reader.config.get("layer", args.layer))
        ctx["layer"] = args.layer
        # **bridge 不许带 router**：旧代码把 ckpt 里的 router 无条件塞进 ctx，
        # 于是 --method bridge 实际跑的是"有路由的单槽读出"，两行结果同源，
        # "Bridge vs Router" 这一列就不成立了（这是结果表里最贵的一处静默错误）。
        use_router = reader.router if args.method == "router" else None
        # oracle_* 刻意**不带 router**：它们要测的是"给定正确槽位时的读出上界"，
        # 一旦让 router 参与，就不再是无寻址误差的量。
        ctx.update({"encoder": reader.encoder, "bridge": reader.bridge,
                    "router": use_router, "fst_router": reader.fst_router,
                    "hist_max": args.hist_max})
        # 注入位置/围栏样式**以 ckpt 为准**：训练时插在消息体内、评测却插在消息之间，
        # 两行数字就不是同一个方法。命令行显式给了才覆盖（A/B 对照用）。
        _ck_inj = str(reader.config.get("inject_mode", "") or "")
        _ck_fen = str(reader.config.get("fence_style", "") or "")
        # **旧 ckpt 必须按旧口径评测**：2026-11-04 之前的权重是在"消息之间 + 长围栏"下训出来的，
        # 直接按新默认（消息体内 + 极简围栏）评测就是训练/推理不一致，数字没有意义。
        if not _ck_inj:
            _ck_inj = "between"
            print("[inject][warn] ckpt 未记录 inject_mode（2026-11-04 之前）：按旧行为 "
                  "between（消息之间）评测。要评估新位置必须用新训的 ckpt", flush=True)
        if not _ck_fen:
            _ck_fen = "verbose"
            print("[inject][warn] ckpt 未记录 fence_style（2026-11-04 之前）：按旧行为 "
                  "verbose（长围栏）评测", flush=True)
        if not args.inject_mode:
            ctx["inject_mode"] = _ck_inj
        if not args.fence_style:
            ctx["fence_style"] = _ck_fen
        print(f"[inject] inject_mode={ctx['inject_mode']} "
              f"fence_style={ctx['fence_style']} "
              f"（ckpt: {_ck_inj or '未记录'} / {_ck_fen or '未记录'}）", flush=True)
        # oracle_lex 必须有直通载荷；oracle_sem 不需要 —— 在 `--no-lex-skip` 训出来的
        # 权重上（纯语义读出，Bridge 只吃 z_sem，α 恒 0），oracle_sem 与 oracle_latent
        # **逐位同构**，此时它是这组权重的**主读法**而不是消融。
        if args.method == "oracle_lex" and not getattr(reader.bridge, "lex_skip", False):
            raise SystemExit("--method oracle_lex 需要 ckpt 打开 lex_skip")
        if args.method == "oracle_sem" and not getattr(reader.bridge, "lex_skip", False):
            print("[oracle_sem] ckpt 未开 lex_skip（纯语义读出权重）：oracle_sem 与 "
                  "oracle_latent 逐位同构，读数即训练时的读出方式", flush=True)
        # E2/E3：把命令行覆盖应用到已加载的 reader 上（只影响"载荷怎么拼"，
        # 不影响权重）。默认值 ckpt / -1 表示完全沿用 ckpt 里记录的口径。
        _ck_tok = str(reader.config.get("lex_token_mode", "mean"))
        if str(args.lex_token_mode) != "ckpt":
            reader.compressor.lex_token_mode = str(args.lex_token_mode)
            print(f"[lex] 词形载荷口径 {_ck_tok} -> {args.lex_token_mode}", flush=True)
        else:
            print(f"[lex] 词形载荷口径 = ckpt 记录的 {_ck_tok}", flush=True)
        if int(args.bpe_backfill) >= 0:
            reader.compressor.bpe_backfill = int(args.bpe_backfill)
        _ck_split = bool(reader.config.get("split_inject", False))
        if int(args.split_inject) in (0, 1):
            reader.bridge.split_inject = bool(int(args.split_inject))
            print(f"[split] 分注入 {_ck_split} -> {bool(int(args.split_inject))}", flush=True)
        else:
            print(f"[split] 分注入 = ckpt 记录的 {_ck_split}", flush=True)
        if int(args.split_n_lex) > 0:
            reader.bridge.n_lex = int(args.split_n_lex)
        print(f"[lex] lex_head={'on' if getattr(reader.compressor, 'lex_scorer', None) is not None else 'off'}"
              f" split_n_lex={int(getattr(reader.bridge, 'n_lex', 0))}", flush=True)
        if args.method in ("router", "fst"):
            got = reader.router if args.method == "router" else reader.fst_router
            if got is None:
                raise SystemExit(f"checkpoint {args.ckpt} 里没有 "
                                 f"{'router' if args.method == 'router' else 'fst_router'} 权重")

    if args.method in EMBED_ARMS:
        # oracle_embed 不加载 bridge/encoder，但**预算必须和 oracle_latent 对齐**：
        # 它按"每个金标槽 ceil(unit_tokens / max_unit_tokens) 块 × m 个向量"算注入
        # token 数。CLI 默认 64 而旧权重训练时是 32，直接跑会让两边的预算差一倍，
        # "软 token 注入机制 vs 压缩损失"的归因就不成立了。给了 --ckpt 就按 ckpt 的配置。
        if args.ckpt:
            _blob = LM._safe_torch_load(args.ckpt)
            _c = dict((_blob or {}).get("config") or {})
            if _c:
                _mut = int(_c.get("max_unit_tokens", ctx["max_unit_tokens"]))
                _ms = int(_c.get("n_slots", ctx["m_slots"]))
                print(f"[budget] oracle_embed 按 ckpt 配置对齐："
                      f"max_unit_tokens={_mut}（CLI {ctx['max_unit_tokens']}）、"
                      f"m={_ms}（CLI {ctx['m_slots']}）", flush=True)
                ctx["max_unit_tokens"] = _mut
                ctx["m_slots"] = _ms
        else:
            print(f"[budget][warn] {args.method} 未给 --ckpt：按 CLI max_unit_tokens="
                  f"{ctx['max_unit_tokens']} 算预算，与训练配置不一致时该行不可比",
                  flush=True)

    data = T.load_synth_jsonl(args.data, limit=args.limit, seed=args.seed)
    if not data:
        raise SystemExit(f"{args.data} 里没有可评测样本")
    label = args.label or f"{args.method}:{os.path.basename(args.data)}"

    _ks = sorted(len(it.get("units") or []) for it in data)
    _n_gt2 = sum(1 for it in data if n_ev_of(it) > 2)
    print(f"[slots] 评测集 units 中位={_ks[len(_ks) // 2]} max={_ks[-1]}；"
          f"训练侧 --max-slots={args.train_max_slots or '不设上限'}；"
          f"金标证据句数>2 的样本 {_n_gt2}/{len(data)}"
          f"（{100.0 * _n_gt2 / max(1, len(data)):.1f}%，固定 2 跳读出的覆盖上限）",
          flush=True)
    if args.train_max_slots and _ks[-1] > args.train_max_slots:
        print(f"[warn] 训练槽数上限 {args.train_max_slots} < 评测集最大槽数 {_ks[-1]}："
              f"router 训练时从未见过这么多槽，读对槽位会被 chance 稀释；"
              f"训练侧建议 --max-slots 0", flush=True)

    rows = []
    _t_run = time.time()
    for i, item in enumerate(data):
        pred, info = read_one(args.method, item, ctx)
        gold = item["answer"]
        if isinstance(gold, (list, tuple)):
            gold = " ".join(str(g) for g in gold)
        ok = T._norm_ans(pred) == T._norm_ans(gold)
        # 逐样本记录里 gold 是**答案文本**，而 info["gold"] 是金标槽位下标；
        # 两者同名会让审计时把"槽位号"当答案读（反之亦然），因此答案改名为 gold_ans。
        rows.append(dict(info, question=item["question"], gold_ans=gold,
                         pred=str(pred), ok=bool(ok), idx=i,
                         em=em_score(pred, gold), f1=f1_score(pred, gold),
                         n_ev=n_ev_of(item), hops=item_hops(item),
                         ans_type=answer_type(gold)))
        if args.verbose:
            print(f"  [{label}] {'✓' if ok else '✗'} q={item['question'][:50]!r} "
                  f"gold_ans={gold!r} pred={str(pred)[:40]!r}", flush=True)
        # 长跑（500 题 ≈ 25 分钟）必须能看出"跑到哪、现在多少分"，
        # 而且中途崩掉不能把已跑完的几百题丢掉 —— 这两件事之前都缺。
        if args.progress_every > 0 and (i + 1) % args.progress_every == 0:
            _el = time.time() - _t_run
            _acc = sum(int(r["ok"]) for r in rows) / len(rows)
            _eta = _el / (i + 1) * (len(data) - i - 1)
            print(f"[progress] {i + 1}/{len(data)} acc={_acc:.4f} "
                  f"elapsed={_el:.0f}s eta={_eta:.0f}s", flush=True)
        if (args.save_json and args.partial_every > 0
                and (i + 1) % args.partial_every == 0 and (i + 1) < len(data)):
            _s_part = summarize(rows, label)
            _dump = {"summary": _s_part, "config": vars(args), "rows": rows,
                     "partial": True, "n_done": len(rows)}
            os.makedirs(os.path.dirname(os.path.abspath(args.save_json)) or ".",
                        exist_ok=True)
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(_dump, f, ensure_ascii=False, indent=1)

    s = summarize(rows, label)
    print("| 方法 | n | acc% | EM% | F1% | 读对槽位% | 完整证据%(≤2句) | pair% "
          "| update% | hist% | 记忆token | prompt总token | 查询ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    print(fmt_line(s), flush=True)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in s.items()}, ensure_ascii=False), flush=True)
    if args.stratify:
        for name, title in (("by_ans_type", "题型"), ("by_hops", "跳数")):
            st = s.get(name) or {}
            print(f"\n[分层] 按{title}：")
            print("| 分组 | n | EM% | F1% |")
            print("|---|---|---|---|")
            for k, v in st.items():
                print(f"| {k} | {v['n']} | {100 * v['em']:.1f} | {100 * v['f1']:.1f} |")
        print("", flush=True)
    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump({"summary": s, "config": vars(args), "rows": rows}, f,
                      ensure_ascii=False, indent=1)
        print(f"[out] -> {args.save_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
