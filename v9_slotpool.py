# v9_slotpool.py — v9 非重叠索引 + 分数优先检索 + EventQA 后向非对称扩展
#                 + 时间戳参与打分/时序重排 + 第 18 层隐状态混合稠密打分
#
# 相对 v8（v8_slotpool.py）的改动，对应 v9 重构的 5 个检索问题：
#   (1) 非重叠索引 extract_units：句级非重叠单元（stride == block_sents，无覆盖重叠）。
#       v8 用 3 句滑窗 stride=1，同一句最多出现在 3 个单元里 -> 池膨胀 3 倍、
#       Top-K 的"K 个单元"实际只覆盖 K/3 个不同句、位置排序有歧义。
#       v9 每个句子恰好属于一个单元，Recall@K 与位置排序都良定义（block_sents=1 即单句索引）。
#   (2) 先按分数选 Top-K、再按原文位置排序：retrieve() 只按分数返回；
#       retrieve_ordered()/readout() 对【已选中的 K 个】按 start 排序，
#       绝不做"先把全池按位置排序再截断"（那会把高分尾部单元截掉）。
#   (3) EventQA 后向 12-16 句非对称扩展：expand_backward(anchor, back=16, ahead=4)
#       —— 以"已发生事件"锚点为准，向前多取、向后少取（非对称），默认 16 > 4。
#       v7/v8 只向后（anchor..anchor+EQA_WIN）取 4 块，锚点之前的证据全部拿不到。
#   (4) 时间戳参与检索打分与时序重排：
#       - 打分：单元编码用 index_text = "In <ts>: <text>"（ts 可为显式或继承），
#         query 里的日期 token 因此能与单元时间戳逐 token MaxSim 命中；
#         texts 仍保存原文，拼给 LLM 的上下文不含人工前缀。
#       - 重排：temporal_rerank 按 |year_unit - year_query| 的指数衰减给分加成
#         （V9_TEMPORAL_TAU / V9_TEMPORAL_BONUS），无年份单元不加分也不罚分。
#   (5) 第 18 层隐状态混合稠密打分接口：
#       - 调用方注入 set_layer18_provider(fn)：fn(texts, kind) -> [N, d] | None，
#         kind ∈ {"units", "query"}；fn 内部用模型 hidden_states[V9_L18_LAYER] 的
#         mean-pool（层号默认 18）。
#       - 或直接 attach_layer18(hidden) + retrieve(..., q_hidden=...)。
#       - hybrid = (1-alpha) * IDF-MaxSim + alpha * ((cos+1)/2)，alpha=V9_L18_ALPHA。
#       - provider 缺失/抛错/形状不符 -> 自动退化为纯词面 IDF-MaxSim（CPU 单测路径）。
#
# 注入约定（同 v6/v7/v8）：v9_slotpool.tok = tok ; v9_slotpool.EMB = EMB。
# CR（FactConsolidation）主路径仍走 v7_slotpool.py；本文件 StatePool 只作为 AR 的
# 可选状态补充与 CR v9 扩展点（_should_merge 可覆写为主语+关系双阈值）。
#
# AR 接入示例（run_ar_v7.py, AR_MODE=v9）：
#   epi, st = v9.build_pools_for_context(ctx)
#   idxs, dbg = epi.readout(q_core, top_k=TOP_K, mode="topk")
#   idxs, dbg = epi.readout(q_core, mode="eventqa", anchor_events=evs,
#                           back=EQA_BACK, ahead=EQA_AHEAD)   # anchor_events: last/last2
#   block = "\n".join(epi.texts[i] for i in idxs)      # texts 是原文，无时间戳前缀
#
# CPU 可测：注入 stub tok + 小 EMB（+ 可选 stub layer18 provider）即可跑全部逻辑单测。

import math
import os
import re
import string
from collections import Counter

import torch
import torch.nn.functional as F

# ----------------------------- 环境参数 -----------------------------
BLOCK_SENTS    = int(os.environ.get("V9_BLOCK_SENTS", "1"))       # 非重叠单元句数（1=单句索引）
UNIT_MAX_WORDS = int(os.environ.get("V9_UNIT_MAX_WORDS", "512"))  # 单元词级硬截断
ENCODE_BS      = int(os.environ.get("V9_ENCODE_BS", "1024"))      # 词级编码分批
BATCH          = int(os.environ.get("V9_BATCH", "64"))            # 全池 MaxSim 分批
IDF_UNSEEN     = float(os.environ.get("V9_IDF_UNSEEN", "0"))      # 0 -> 用池内最大 IDF
TOKV_BF16      = os.environ.get("V9_TOKV_BF16", "1") == "1"       # 逐 token 向量 bf16 存储（省内存）
EQA_BACK       = int(os.environ.get("V9_EQA_BACK", "16"))         # EventQA 锚点前取几句
EQA_AHEAD      = int(os.environ.get("V9_EQA_AHEAD", "4"))         # EventQA 锚点后取几句（非对称）
TEMPORAL_TAU   = float(os.environ.get("V9_TEMPORAL_TAU", "2.0"))  # 时序加成衰减（年）
TEMPORAL_BONUS = float(os.environ.get("V9_TEMPORAL_BONUS", "0.15"))  # 时序加成上限（相对）
L18_ALPHA      = float(os.environ.get("V9_L18_ALPHA", "0.3"))     # 稠密通道权重
L18_LAYER      = int(os.environ.get("V9_L18_LAYER", "18"))        # 调用方应取的隐状态层号
STATE_THETA    = float(os.environ.get("V9_STATE_THETA", "0.92"))  # 状态池语义合并阈值

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "is", "was", "are", "were", "be", "been", "what", "which", "who", "where",
    "when", "that", "this", "does", "did", "has", "have", "had", "and", "or",
    "it", "its", "his", "her", "their", "he", "she", "they", "as", "but",
}

# 事件线索：过去时事件动词
EVENT_HINT_RE = re.compile(
    r"\b(?:was|were|had|did|done|became|become|moved|founded|created|born|died|killed|"
    r"wrote|directed|released|won|lost|opened|closed|joined|left|started|"
    r"ended|married|graduated|retired|elected|appointed|discovered|invaded|"
    r"attended|worked|played|traveled|visited|returned|occurred|happened|crossed|"
    r"swam|climbed|escaped|arrived|departed|sent|gave|took|made|built|sold|bought)\b",
    flags=re.IGNORECASE,
)

# 时间标记：March 3, 2001 / 12 January 2005 / 1997 / the 1990s / 2001-03-03
TIMESTAMP_RE = re.compile(
    r"(?:\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b"
    r"|\b(?:19|20)\d{2}\b"
    r"|\bthe\s+\d{4}s\b"
    r"|\b\d{4}-\d{2}-\d{2}\b)",
    flags=re.IGNORECASE,
)

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
WORD_RE = re.compile(r"[A-Za-z0-9_\u00C0-\u024F'-]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    """简单句切分（可被调用方覆盖：v9_slotpool.split_sentences = fn）。"""
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_year(ts):
    """时间戳 -> 年份 int（'the 1990s' -> 1990；'12 January 2005' -> 2005）；无则 None。"""
    if not ts:
        return None
    m = YEAR_RE.search(str(ts))
    return int(m.group(0)) if m else None


def ts_prefix(ts):
    """时间戳前缀（只用于打分文本，不进入给 LLM 的原文）。"""
    return f"In {ts}:" if ts else ""


def build_index_text(ts, text):
    """打分文本 = 时间戳前缀 + 原文；让日期 token 参与逐 token MaxSim。"""
    pre = ts_prefix(ts)
    return f"{pre} {text}".strip() if pre else text


# ----------------------------- 非重叠单元抽取 -----------------------------
def extract_units(context, block_sents=None):
    """把上下文切成【非重叠】检索单元。

    block_sents=1 -> 单句索引；block_sents>1 -> 连续 b 句成一块（stride == b，无重叠）。
    每个句子恰好属于一个单元（不丢句、不重复），因此 Top-K / 位置排序 / 邻域扩展良定义。

    返回 list[dict]：
      text       原文（拼上下文用，不含人工前缀）
      index_text 打分文本（"In <ts>: <text>"，ts 可为显式或继承）
      start/end  句索引区间（闭区间）
      ts         该单元的时间戳（单元内首个显式标记，否则继承起点之前最近一次）
      ts_explicit / ts_inherited
      year       ts 的年份 int（时序重排用；无则 None）
      is_event   单元内是否含事件动词
    """
    sents = split_sentences(context)
    n = len(sents)
    b = max(1, int(BLOCK_SENTS if block_sents is None else block_sents))

    ts_at = [None] * n          # 每句显式时间戳
    resolved_ts = [None] * n    # 逐句"最近一次显式时间戳"（含自身），只读
    is_event = [False] * n
    last_ts = None
    for i, s in enumerate(sents):
        m = TIMESTAMP_RE.search(s)
        if m:
            last_ts = m.group(0)
            ts_at[i] = last_ts
        resolved_ts[i] = last_ts
        is_event[i] = bool(EVENT_HINT_RE.search(s))

    units = []
    for u0 in range(0, n, b):                 # 非重叠：起点每次前进 b
        u1 = min(u0 + b - 1, n - 1)
        explicit = next((ts_at[j] for j in range(u0, u1 + 1) if ts_at[j]), None)
        prev_ts = resolved_ts[u0 - 1] if u0 > 0 else None
        ts = explicit if explicit is not None else prev_ts
        text = " ".join(sents[u0:u1 + 1])
        units.append({
            "text": text,
            "index_text": build_index_text(ts, text),
            "start": u0,
            "end": u1,
            "ts": ts,
            "ts_explicit": explicit is not None,
            "ts_inherited": explicit is None and ts is not None,
            "year": extract_year(ts),
            "is_event": any(is_event[j] for j in range(u0, u1 + 1)),
        })
    return units


# ----------------------------- 编码工具 -----------------------------
tok = None    # 调用方注入 tokenizer（CPU 测试可注入 stub）
EMB = None    # 调用方注入嵌入矩阵 [V, d]（同 device）


def _norm_word(w):
    return w.lower().translate(str.maketrans("", "", string.punctuation)).strip("'")


def _content_words(text):
    ws = [_norm_word(w) for w in WORD_RE.findall(text)]
    ws = [w for w in ws if w]
    kept = [w for w in ws if w not in STOPWORDS] or ws or [text.strip().lower()]
    return kept


def encode_words(words):
    """批量编码词列表 -> ([Σlw, d] 归一化逐 token fp32 向量, counts)。分批编码控制显存峰值。"""
    dim = EMB.shape[1] if EMB is not None else 1
    if not words:
        return torch.zeros((0, dim), dtype=torch.float32), []
    dev = EMB.device if EMB is not None else torch.device("cpu")
    out, counts = [], []
    for s in range(0, len(words), ENCODE_BS):
        chunk = list(words[s:s + ENCODE_BS])
        enc = tok(chunk, add_special_tokens=False, padding=True,
                  truncation=True, max_length=16, return_tensors="pt")
        ids = (enc["input_ids"] if isinstance(enc, dict) else enc.input_ids).to(dev)
        am = (enc["attention_mask"] if isinstance(enc, dict) else enc.attention_mask).to(dev)
        vecs = F.normalize(EMB[ids].float(), dim=-1)     # [W, Lw, d]
        for i in range(len(chunk)):
            L = max(int(am[i].sum()), 0)
            if L == 0:
                out.append(torch.zeros((0, dim), dtype=torch.float32))
                counts.append(0)
                continue
            out.append(vecs[i, :L].cpu())
            counts.append(L)
        del enc, ids, am, vecs
    flat = torch.cat([o for o in out], dim=0) if out else torch.zeros((0, dim), dtype=torch.float32)
    return flat, counts


# ----------------------------- 第 18 层隐状态接口 -----------------------------
L18_PROVIDER = None   # fn(texts: list[str], kind: "units"|"query") -> [N, d] | None


def set_layer18_provider(fn):
    """注入第 18 层隐状态提供者；fn=None 关闭稠密通道。

    fn 契约：
      - fn(texts, kind) 返回 [len(texts), d] 张量（CPU/GPU 均可，内部转 CPU fp32 并 L2 归一化）；
      - kind == "units" 用于建索引（一次批量），"query" 用于查询（一次一条）；
      - 返回 None / 抛异常 / 形状不符 -> 静默退化为纯词面打分（保证 CPU 单测与无模型路径可用）；
      - 单元与 query 必须来自同一层（V9_L18_LAYER，默认 18）、同一 pool 方式（mean-pool）。
    """
    global L18_PROVIDER
    L18_PROVIDER = fn


def mean_pool_hidden(hidden, attention_mask=None):
    """把某一层 hidden_states 做 mask mean-pool -> [N, d]。

    hidden: [N, L, d]（批量）或 [L, d]（单条）；
    attention_mask: [N, L]（可选，0 位置不计入均值）。
    典型用法（run_ar_v7.py）：
        hs = model(..., output_hidden_states=True).hidden_states[V9_L18_LAYER]
        unit_vecs = v9.mean_pool_hidden(hs, attn_mask)     # [N, d]
        v9.set_layer18_provider(lambda texts, kind: ...)   # 或 epi.attach_layer18(unit_vecs)
    """
    h = torch.as_tensor(hidden, dtype=torch.float32)
    if h.dim() == 2:
        return h.mean(dim=0, keepdim=True)
    if h.dim() != 3:
        raise ValueError(f"hidden 需要 [N,L,d] 或 [L,d]，实际 {tuple(h.shape)}")
    if attention_mask is None:
        return h.mean(dim=1)
    m = torch.as_tensor(attention_mask, dtype=torch.float32)
    if m.dim() == 1:
        m = m.unsqueeze(0)
    m = m.unsqueeze(-1)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)


def layer18_spec():
    """日志用：当前稠密通道配置。"""
    return {"layer": L18_LAYER, "alpha": L18_ALPHA, "enabled": L18_PROVIDER is not None}


def _call_l18(texts, kind):
    if L18_PROVIDER is None or not texts:
        return None
    try:
        h = L18_PROVIDER(list(texts), kind)
    except Exception:
        return None
    if h is None:
        return None
    h = torch.as_tensor(h, dtype=torch.float32)
    if h.dim() == 1:
        h = h.unsqueeze(0)
    if h.dim() != 2 or h.shape[0] != len(texts):
        return None
    return F.normalize(h, dim=-1)


# ----------------------------- 打分底座 -----------------------------
class _ScoreStore:
    """单元存储 + 三路打分（词面 IDF-MaxSim / 第 18 层稠密 / 时序加成）。"""

    def __init__(self):
        self.texts = []         # 原文（输出用）
        self.index_texts = []   # 打分文本（时间戳前缀 + 原文）
        self.metas = []         # 元数据（start/end/ts/year/...）
        self.tokv = []          # list[cpu bf16/fp32 [L,d]] 逐 token 归一化向量
        self.words = []         # list[list[str]] 与 token 对齐的词（IDF 用）
        self.mean = []          # list[cpu fp32 [d]]
        self.years = []         # list[float]（nan = 无时间戳）
        self.word_df = Counter()
        self.n_dup = 0
        self.l18 = None         # [N, d_l18] cpu fp32 归一化（稠密通道）
        self._idf = None
        self._mean_mat = None

    def __len__(self):
        return len(self.texts)

    # ---- 写入 ----
    def _store(self, texts, index_texts, metas):
        """编码并追加单元。词级截断先于 span 记账（v8 Bug2 修复后的顺序）。"""
        keep = [(t, it, m) for t, it, m in zip(texts, index_texts, metas)
                if t and t.strip()]
        if not keep:
            return
        texts = [k[0] for k in keep]
        index_texts = [k[1] for k in keep]
        metas = [k[2] for k in keep]

        n_before = len(self)
        all_words, spans = [], []
        for it in index_texts:
            ws = _content_words(it)[:UNIT_MAX_WORDS]
            spans.append((len(all_words), len(all_words) + len(ws)))
            all_words.extend(ws)
        vecs_flat, counts = encode_words(all_words)
        off = 0
        for k, (text, it, meta) in enumerate(zip(texts, index_texts, metas)):
            s, e = spans[k]
            w_list = all_words[s:e]
            c_counts = counts[s:e]
            L = sum(c_counts)
            if L <= 0:
                # 零 token 兜底：vecs_flat 无对应行 -> 不推进 off
                tokv = torch.zeros((1, vecs_flat.shape[1]), dtype=torch.float32)
                w_list = [w_list[0] if w_list else "<empty>"]
            else:
                tokv = vecs_flat[off:off + L]
                off += L
            if TOKV_BF16 and tokv.dtype != torch.bfloat16:
                tokv = tokv.to(torch.bfloat16)
            self.texts.append(text)
            self.index_texts.append(it)
            self.metas.append(meta)
            self.tokv.append(tokv)
            self.words.append(w_list)
            self.mean.append(F.normalize(tokv.float().mean(dim=0), dim=0))
            year = meta.get("year")
            if year is None:
                year = extract_year(meta.get("ts"))
            self.years.append(float(year) if year is not None else float("nan"))
            for w in set(w_list):
                self.word_df[w] += 1
        self._idf = None
        self._mean_mat = None
        # 稠密通道：provider 可用时按本批文本编码并追加；形状/同步不符则跳过（不破坏词面路径）
        if L18_PROVIDER is not None:
            h = _call_l18(index_texts, "units")
            if h is not None and h.shape[0] == len(index_texts):
                if self.l18 is None and n_before == 0:
                    self.l18 = h
                elif self.l18 is not None and self.l18.shape[0] == n_before:
                    self.l18 = torch.cat([self.l18, h], dim=0)

    def attach_layer18(self, hidden):
        """直接注入单元侧第 18 层隐状态 [N, d]（N 必须等于当前单元数）。"""
        h = torch.as_tensor(hidden, dtype=torch.float32)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if h.dim() != 2 or h.shape[0] != len(self):
            raise ValueError(f"layer18 shape {tuple(h.shape)} != ({len(self)}, d)")
        self.l18 = F.normalize(h, dim=-1)

    # ---- 词面打分 ----
    def idf_table(self):
        if self._idf is None:
            n = max(len(self), 1)
            self._idf = {w: math.log(1.0 + n / (1.0 + df)) for w, df in self.word_df.items()}
        return self._idf

    def _query_repr(self, query):
        """query -> (qe [Lq,d] dev fp32, q_idf [Lq] dev)。"""
        dev = EMB.device if EMB is not None else torch.device("cpu")
        ws = _content_words(query)[:64]
        qv, counts = encode_words(ws)
        if qv.shape[0] == 0:                       # 空 query 兜底（避免 mean 出 nan）
            qv = torch.zeros((1, EMB.shape[1] if EMB is not None else 1), dtype=torch.float32)
            counts = [1]
            ws = ["<q>"]
        idf_t = self.idf_table()
        max_idf = max(idf_t.values()) if idf_t else 1.0
        w_list = []
        for w, c in zip(ws, counts):
            w_list.extend([w] * c)
        if len(w_list) != qv.shape[0]:             # 截断保护
            w_list = (w_list + [ws[0]] * qv.shape[0])[:qv.shape[0]]
        qw = torch.tensor([idf_t.get(w, max_idf if IDF_UNSEEN == 0 else IDF_UNSEEN)
                           for w in w_list], dtype=torch.float32, device=dev)
        qe = F.normalize(qv.to(dev), dim=-1)
        return qe, qw

    def lexical_scores(self, query):
        """全池 IDF-MaxSim：score(u) = Σ_t idf(q_t)·max_j cos(q_t, u_j) / Σ_t idf(q_t)。"""
        dev = EMB.device if EMB is not None else torch.device("cpu")
        n = len(self)
        qe, qw = self._query_repr(query)
        denom = qw.sum().clamp(min=1e-6)
        scores = torch.zeros(n, dtype=torch.float32, device=dev)
        for s in range(0, n, BATCH):
            e = min(s + BATCH, n)
            Ls = [self.tokv[i].shape[0] for i in range(s, e)]
            Lm = max(Ls)
            buf = torch.zeros((e - s, Lm, qe.shape[1]), dtype=torch.float32, device=dev)
            for k, i in enumerate(range(s, e)):
                buf[k, :Ls[k]] = self.tokv[i].to(device=dev, dtype=torch.float32)
            sim = torch.einsum("qd,bld->bql", qe, buf)      # [B, Lq, Lm]
            mx = sim.max(dim=2).values                      # [B, Lq]
            scores[s:e] = (mx * qw).sum(dim=1) / denom
        return scores

    # ---- 稠密打分（第 18 层） ----
    def dense_scores(self, query=None, q_hidden=None):
        """单元-query 的第 18 层隐状态余弦；未接稠密通道时返回 None。"""
        if self.l18 is None or self.l18.shape[0] != len(self):
            return None
        if q_hidden is None:
            if query is None:
                return None
            h = _call_l18([query], "query")
            if h is None:
                return None
            q_hidden = h[0]
        qh = torch.as_tensor(q_hidden, dtype=torch.float32)
        if qh.dim() > 1:
            qh = qh.mean(dim=0)
        qh = F.normalize(qh, dim=0)
        return (self.l18 @ qh).clamp(-1.0, 1.0)

    def hybrid_scores(self, query, q_hidden=None, alpha=None):
        """混合打分：(1-alpha)·词面 + alpha·((稠密cos+1)/2)；无稠密通道 -> 纯词面。"""
        s = self.lexical_scores(query)
        d = self.dense_scores(query, q_hidden)
        if d is None:
            return s
        a = L18_ALPHA if alpha is None else float(alpha)
        a = min(max(a, 0.0), 1.0)
        return (1.0 - a) * s + a * ((d + 1.0) * 0.5)

    # ---- 时序重排 ----
    def temporal_rerank(self, scores, query_ts=None, query_year=None):
        """按 |year_u - year_q| 指数衰减给分加成；无年份单元不加分也不罚分。"""
        yq = query_year if query_year is not None else extract_year(query_ts)
        if yq is None or not self.years:
            return scores
        y = torch.tensor(self.years, dtype=torch.float32, device=scores.device)
        known = ~torch.isnan(y)
        aff = torch.zeros_like(y)
        if known.any():
            aff[known] = torch.exp(-(y[known] - float(yq)).abs() / max(TEMPORAL_TAU, 1e-6))
        return scores * (1.0 + TEMPORAL_BONUS * aff)

    def score(self, query, query_ts=None, q_hidden=None, alpha=None, temporal=True):
        """最终打分张量 [N]（混合 + 可选时序加成）。"""
        s = self.hybrid_scores(query, q_hidden=q_hidden, alpha=alpha)
        if temporal:
            yq = extract_year(query_ts) if query_ts is not None else extract_year(query)
            if yq is not None:
                s = self.temporal_rerank(s, query_year=yq)
        return s

    # ---- 检索 ----
    def retrieve(self, query, top_k=12, query_ts=None, q_hidden=None,
                 alpha=None, temporal=True):
        """【先按分数选 Top-K】。返回 [(idx, score)] 分数降序，无位置排序。"""
        if len(self) == 0:
            return []
        scores = self.score(query, query_ts=query_ts, q_hidden=q_hidden,
                            alpha=alpha, temporal=temporal)
        k = min(int(top_k), len(self))
        vals, idxs = scores.topk(k)
        return [(int(i), float(v)) for i, v in zip(idxs.tolist(), vals.tolist())]

    def retrieve_ordered(self, query, top_k=12, **kw):
        """先按分数选 Top-K，再把【已选中的 K 个】按原文位置排序（不是全池排序后截断）。"""
        hits = self.retrieve(query, top_k=top_k, **kw)
        return sorted((i for i, _ in hits), key=lambda i: self.metas[i].get("start", i))

    def best_unit_for(self, text, query_ts=None, **kw):
        """时序锚定：定位某段文本（已发生事件）最匹配的单元。返回 (idx, score)。"""
        hits = self.retrieve(text, top_k=1, query_ts=query_ts, **kw)
        return hits[0] if hits else (-1, 0.0)

    def mean_mat(self):
        if self._mean_mat is None and self.mean:
            self._mean_mat = torch.stack(self.mean)
        return self._mean_mat

    def stats(self):
        L = [t.shape[0] for t in self.tokv]
        return {"n": len(self),
                "n_ts": sum(1 for y in self.years if not math.isnan(y)),
                "mean_tokens": round(sum(L) / len(L), 2) if L else 0.0,
                "has_l18": self.l18 is not None,
                "n_dup": self.n_dup}


# ----------------------------- 情景索引 -----------------------------
class EpisodicIndex(_ScoreStore):
    """非重叠情景索引：单元按原文位置一一对应，检索分数优先、扩展非对称后向。"""

    def add_units(self, units):
        """units: list[dict]（extract_units 的产物）。按位置全量保留，不做文本去重
        （去重会让同一句只留一个位置，破坏位置排序与邻域扩展）。"""
        if not units:
            return
        texts = [u["text"] for u in units]
        index_texts = [u.get("index_text") or build_index_text(u.get("ts"), u["text"])
                       for u in units]
        self._store(texts, index_texts, list(units))

    def expand_backward(self, anchor_idx, back=None, ahead=None):
        """EventQA 后向非对称扩展：以 anchor 为界，向前 back 个、向后 ahead 个单元。

        默认 back=V9_EQA_BACK=16 > ahead=V9_EQA_AHEAD=4（非对称）。
        非重叠索引下单元下标即原文顺序，返回按位置升序的连续下标。
        """
        if len(self) == 0:
            return []
        b = EQA_BACK if back is None else max(0, int(back))
        a = EQA_AHEAD if ahead is None else max(0, int(ahead))
        anchor_idx = int(anchor_idx)
        if anchor_idx < 0:
            return []
        lo = max(0, anchor_idx - b)
        hi = min(len(self) - 1, anchor_idx + a)
        return list(range(lo, hi + 1))

    def anchor_from_events(self, events, cov_threshold=0.5, query_ts=None, **kw):
        """EventQA 时序锚定：取最后一个已发生事件；覆盖率 < cov_threshold 时
        拼接最后两个事件重锚（与 v7/v8 的 anchor_used=last/last2 口径一致）。

        返回 (idx, cov, used)，used ∈ {"last", "last2", "none"}。
        """
        events = [e for e in (events or []) if e]
        if not events or len(self) == 0:
            return -1, 0.0, "none"
        idx, cov = self.best_unit_for(events[-1], query_ts=query_ts, **kw)
        used = "last"
        if cov < cov_threshold and len(events) > 1:
            idx2, cov2 = self.best_unit_for(events[-2] + " " + events[-1],
                                            query_ts=query_ts, **kw)
            if cov2 > cov:
                idx, cov, used = idx2, cov2, "last2"
        return idx, cov, used

    def readout(self, query, top_k=12, mode="topk", anchor_idx=None, anchor_text=None,
                anchor_events=None,
                query_ts=None, q_hidden=None, alpha=None, temporal=True,
                back=None, ahead=None):
        """读出：返回 (idxs, dbg)，idxs 一律按原文位置升序。

        mode="topk"    : 先按分数选 Top-K，再把选中的 K 个按原文位置排序；
        mode="eventqa" : 定位锚点（anchor_idx 优先；否则 anchor_events 走
                         anchor_from_events 的 last/last2 逻辑；否则 anchor_text/query
                         的 best_unit_for），再做后向非对称扩展（back/ahead）。
        """
        if len(self) == 0:
            return [], {"mode": "empty", "n_units": 0}
        yq = extract_year(query_ts) if query_ts is not None else extract_year(query)
        if mode == "eventqa":
            cov = 0.0
            used = "given" if anchor_idx is not None else "none"
            if anchor_idx is None and anchor_events:
                anchor_idx, cov, used = self.anchor_from_events(
                    anchor_events, query_ts=query_ts, q_hidden=q_hidden,
                    alpha=alpha, temporal=temporal)
            if anchor_idx is None:
                anchor_idx, cov = self.best_unit_for(anchor_text or query,
                                                     query_ts=query_ts,
                                                     q_hidden=q_hidden,
                                                     alpha=alpha, temporal=temporal)
                used = "text" if anchor_text else "query"
            if int(anchor_idx) < 0:
                idxs = self.retrieve_ordered(query, top_k=top_k, query_ts=query_ts,
                                             q_hidden=q_hidden, alpha=alpha,
                                             temporal=temporal)
                return idxs, {"mode": "eventqa_fallback_topk", "idxs": idxs}
            b = EQA_BACK if back is None else int(back)
            a = EQA_AHEAD if ahead is None else int(ahead)
            idxs = self.expand_backward(anchor_idx, back=b, ahead=a)
            return idxs, {"mode": "eventqa_backward", "anchor": int(anchor_idx),
                          "anchor_cov": round(float(cov), 4), "anchor_used": used,
                          "back": b, "ahead": a, "idxs": idxs, "query_year": yq}
        hits = self.retrieve(query, top_k=top_k, query_ts=query_ts, q_hidden=q_hidden,
                             alpha=alpha, temporal=temporal)
        idxs = sorted((i for i, _ in hits), key=lambda i: self.metas[i].get("start", i))
        return idxs, {"mode": "topk_score_first",
                      "hits": [[i, round(s, 4)] for i, s in hits],
                      "idxs": idxs, "query_year": yq}


# ----------------------------- CR 双阈值（v9 CR 用） -----------------------------
# 谓词短语：长谓词优先，避免 "is" 抢先匹配 "is located in"。
CR_REL_THETA = float(os.environ.get("V9_CR_REL_THETA", "0.80"))
CR_VP_RE = re.compile(
    r"\b(?:is associated with|is affiliated with|is located in the continent of|"
    r"is located in the country of|was educated at|is educated at|"
    r"was created in|is created in|was created by|is created by|"
    r"was founded by|is founded by|is located in|was located in|"
    r"is famous for|was famous for|was written in|is written in|"
    r"was written by|is written by|was directed by|is directed by|"
    r"was born in|was born on|is married to|was married to|"
    r"is a citizen of|was a citizen of|held the position of|holds the position of|"
    r"is the capital of|is the language of|worked at|works at|worked for|works for|"
    r"plays|played|founded|wrote|is|was|are|were)\b",
    flags=re.IGNORECASE,
)


def _cr_parse_fact(text):
    """轻量 (subject, relation) 切分，仅供 CR 双阈值使用。

    不做 v7 的 WHERE/PREFIX 全解析：同一模板的冲突事实切分一致即可
    （主语用于精确匹配，谓词短语用于余弦比较）。v9_cr 的图检索用 v7 全解析。
    """
    f = (text or "").strip().rstrip(".")
    m = CR_VP_RE.search(f)
    if not m:
        toks = f.split()
        return (" ".join(toks[:-2]) if len(toks) > 3 else f), (toks[-1] if toks else "")
    return f[:m.start()].strip(), m.group(0).strip()


def _cr_norm(s):
    s = (s or "").lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(w for w in s.split() if w not in STOPWORDS) or s.strip()


def _cr_key_vec(text):
    """谓词短语 -> 归一化均值向量；未注入 tok/EMB 时返回 None。"""
    if tok is None or EMB is None:
        return None
    ws = _content_words(text)
    if not ws:
        return None
    v, _c = encode_words(ws)
    if v.shape[0] == 0:
        return None
    return F.normalize(v.float().mean(dim=0), dim=0)


# ----------------------------- 状态池（可选补充 / CR 扩展点） -----------------------------
class StatePool(_ScoreStore):
    """稳定事实单元池：近重复语义合并（v9 CR：主语+关系双阈值；否则 mean-cos）。

    CR v9 扩展点已落到 `_should_merge`：主语归一化字符串精确匹配，且关系谓词
    向量余弦 > V9_CR_REL_THETA（默认 0.80）才覆写；v8 的单阈值 mean-cos(0.92)
    对"同主语+同谓词、不同宾语"的冲突事实不够敏感。AR 状态补充路径仍可用。
    """

    def __init__(self, theta=None):
        super().__init__()
        self.theta = STATE_THETA if theta is None else float(theta)
        self.cr_subj = []      # 主语归一化字符串（与 mean 槽一一对应）
        self.cr_relk = []      # 关系谓词均值向量 [d]

    def _store(self, texts, index_texts, metas):
        keep = [(t, it, m) for t, it, m in zip(texts, index_texts, metas)
                if t and t.strip()]
        super()._store(texts, index_texts, metas)
        for t, _it, _m in keep:
            subj, rel = _cr_parse_fact(t)
            self.cr_subj.append(_cr_norm(subj))
            self.cr_relk.append(_cr_key_vec(rel))

    def _should_merge(self, text, meta, qm, base):
        """双阈值：主语精确匹配 + 关系谓词余弦 > V9_CR_REL_THETA（默认 0.80）。

        以 add_batch 的 argmax 槽（实际覆写目标）为准：只有该槽与候选主语完全
        相同、且谓词余弦过阈值时才合并；谓词键缺失（pending / 未注入 tok+EMB）
        或主语为空时退回 mean-cos 单阈值。
        """
        sims = base @ qm
        j = int(sims.argmax())
        if j >= len(self.cr_subj):                 # pending 单元，无已存储键
            return float(sims[j]) > self.theta
        subj, rel = _cr_parse_fact(text)
        s_norm = _cr_norm(subj)
        if not s_norm:
            return float(sims[j]) > self.theta
        if s_norm != self.cr_subj[j]:
            return False
        k_r = _cr_key_vec(rel)
        if k_r is None or self.cr_relk[j] is None:
            return float(sims[j]) > self.theta
        return float(self.cr_relk[j] @ k_r) > CR_REL_THETA

    def _drop(self, j):
        """回收第 j 个槽（df/tokv/words/mean/years 同步），供覆写前使用。"""
        if j < len(self.cr_subj):
            self.cr_subj.pop(j)
            self.cr_relk.pop(j)
        for w in set(self.words[j]):
            self.word_df[w] -= 1
            if self.word_df[w] <= 0:
                self.word_df.pop(w, None)
        for lst in (self.texts, self.index_texts, self.metas, self.tokv,
                    self.words, self.mean, self.years):
            lst.pop(j)
        if self.l18 is not None and self.l18.shape[0] == len(self):
            self.l18 = (torch.cat([self.l18[:j], self.l18[j + 1:]], dim=0)
                        if self.l18.shape[0] > 1 else None)
        self._idf = None
        self._mean_mat = None

    def _mean_only(self, index_text):
        dev = EMB.device if EMB is not None else torch.device("cpu")
        ws = _content_words(index_text)[:UNIT_MAX_WORDS]
        qv, _c = encode_words(ws)
        if qv.shape[0] == 0:
            return torch.zeros((EMB.shape[1] if EMB is not None else 1,),
                               dtype=torch.float32, device=dev)
        qe = F.normalize(qv.to(dev), dim=-1)
        return F.normalize(qe.mean(dim=0), dim=0)

    def add_batch(self, units):
        """units: list[(text, meta)]。近重复（默认 mean cos > theta）覆写旧槽。"""
        pending = []
        for text, meta in units:
            if not text or not text.strip():
                continue
            it = build_index_text(meta.get("ts"), text)
            qm = self._mean_only(it).cpu()
            base = None
            if self.mean:
                base = torch.stack([m.cpu() for m in self.mean])
            if pending:
                pm = torch.stack([p[2] for p in pending])
                base = pm if base is None else torch.cat([base, pm])
            merged = False
            if base is not None and self._should_merge(text, meta, qm, base):
                sims = base @ qm
                j = int(sims.argmax())
                if j < len(self.mean):
                    self._drop(j)
                    pending.append((text, meta, qm))
                else:
                    pending[j - len(self.mean)] = (text, meta, qm)
                self.n_dup += 1
                merged = True
            if not merged:
                pending.append((text, meta, qm))
        if not pending:
            return
        texts = [p[0] for p in pending]
        metas = [p[1] for p in pending]
        index_texts = [build_index_text(m.get("ts"), t) for t, m in zip(texts, metas)]
        self._store(texts, index_texts, metas)


# ----------------------------- 组装入口 -----------------------------
def build_index(ctx, block_sents=None):
    """只建情景索引（v9 AR 主路径）。"""
    epi = EpisodicIndex()
    epi.add_units(extract_units(ctx, block_sents=block_sents))
    return epi


def build_pools_for_context(ctx, block_sents=None):
    """建 (情景索引, 状态池)。状态池只收无事件动词、无显式时间戳的稳定陈述句，
    供 AR 读出可选补充；CR 路径不依赖本函数。

    注意：v9 情景索引是非重叠全覆盖（每句恰好一个单元），因此状态单元与情景单元
    文本重叠。AR v9 主路径只用 epi；若拼入状态池补充，必须按文本去重
    （v8 的 build_block_v8 已有 seen 去重，可直接复用）。
    """
    units = extract_units(ctx, block_sents=block_sents)
    epi = EpisodicIndex()
    epi.add_units(units)
    st = StatePool()
    st.add_batch([(u["text"], u) for u in units
                  if (not u["is_event"]) and (not u["ts_explicit"])])
    return epi, st
