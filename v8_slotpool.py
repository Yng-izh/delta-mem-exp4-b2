# v8_slotpool.py — v8 双模态潜槽位记忆 (DM-ELSM: Dual-Modal Episodic Latent Slot Memory)
# 相对 v7 AR 管线（run_ar_v7.py ChunkPool）的改动：
#   (1) 情景/状态自适应分流 extract_adaptive_units：
#       - 情景微块（episodic）：3 句滑窗（stride=1），每个窗口是独立检索单元；
#       - 状态单元（state）：无时间标记、无过去时事件动词的稳定事实句，聚合去重；
#       - 时间戳继承：显式时间标记（In 1997 / On March 3, 2001 / in the 1990s）
#         写入 cur_ts，后续无标记窗口继承最近一次时间戳。
#   (2) EpisodicFastPool：全池无截断 IDF-MaxSim 检索——废除 v7 的 n>3500 Top-2500 截断，
#       对全池每个块分批（batch）做 ColBERT MaxSim，query 逐 token 用池内 IDF 加权；
#       情景池永不覆写、永不淘汰（AR 无"新值覆盖旧值"语义）。
#   (3) 邻域扩展：检索命中情景微块后，按句序扩展相邻窗口（时间连续性）。
#   (4) StatePool：近重复语义合并（mean cos > V8_STATE_THETA 时新文本覆写旧槽），
#       检索同用 IDF-MaxSim。
# 依赖调用方注入（同 v6/v7 约定）：v8_slotpool.tok = tok ; v8_slotpool.EMB = EMB。
# CPU 可测：注入 stub tok + 小 EMB 即可跑全部逻辑单测。
#
# v8.1（确定性 Bug 修复，接口不变）：
#   Bug1 cur_ts 第二轮污染 —— 逐句预计算 resolved_ts[i]，窗口/状态两轮只读；
#        窗口继承"起点之前最近一次"显式时间戳，不再继承全篇末尾时间戳。
#   Bug2 512 词 span 偏移 —— 先按 UNIT_MAX_WORDS 截断再记 span；零 token 兜底不再推进 off。
#   Bug3 ev 时间触发 —— ev = 事件动词 or 显式时间标记（与 README/设计一致）；
#        state 只收无时间标记、无事件动词的稳定陈述句。

import os
import re
import math
import string
from collections import Counter

import torch
import torch.nn.functional as F

# ----------------------------- 环境参数 -----------------------------
WIN        = int(os.environ.get("V8_WIN", "3"))           # 情景微块滑窗句数
STATE_THETA = float(os.environ.get("V8_STATE_THETA", "0.92"))  # 状态池语义合并阈值
BATCH      = int(os.environ.get("V8_BATCH", "64"))        # 全池 MaxSim 分批大小
MAX_TOK    = int(os.environ.get("V8_MAX_TOK", "512"))     # 单元 token 硬截断
IDF_UNSEEN = float(os.environ.get("V8_IDF_UNSEEN", "0"))  # 0 -> 用池内最大 IDF

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "is", "was", "are", "were", "be", "been", "what", "which", "who", "where",
    "when", "that", "this", "does", "did", "has", "have", "had", "and", "or",
    "it", "its", "his", "her", "their", "he", "she", "they", "as", "but",
}

# 情景（事件）线索：过去时事件动词 / 时间标记
EVENT_HINT_RE = re.compile(
    r"\b(?:was|were|had|did|done|became|become|moved|founded|created|born|died|killed|"
    r"wrote|wrote|directed|released|won|lost|opened|closed|joined|left|started|"
    r"ended|married|graduated|retired|elected|appointed|discovered|invaded|"
    r"attended|worked|played|traveled|visited|returned|occurred|happened|crossed|"
    r"swam|climbed|escaped|arrived|departed|sent|gave|took|made|built|sold|bought)\b",
    flags=re.IGNORECASE,
)

# 时间标记：March 3, 2001 / 1997 / the 1990s / 12 January 2005 / 2001-03-03
TIMESTAMP_RE = re.compile(
    r"(?:\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b"
    r"|\b(?:19|20)\d{2}\b"
    r"|\bthe\s+\d{4}s\b"
    r"|\b\d{4}-\d{2}-\d{2}\b)",
    flags=re.IGNORECASE,
)

WORD_RE = re.compile(r"[A-Za-z0-9_\u00C0-\u024F'-]+")

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    """简单句切分（可被调用方用 chunk_text_into_sentences 覆盖：v8_slotpool.split_sentences = fn）。"""
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ----------------------------- 分流：extract_adaptive_units -----------------------------
def extract_adaptive_units(context, win=WIN):
    """情景/状态自适应分流。
    返回 (episodic_units, state_units)：
      episodic unit: {"text", "sents", "start", "end", "ts", "ts_inherited"}
      state unit:    {"text", "start", "end", "ts", "ts_inherited"}
    规则：
      - 3 句滑窗（stride=1）为情景微块；窗口任一句含【显式时间标记】或【事件动词】→ episodic；
      - 不落入任何情景窗口的稳定陈述句（无显式时间标记、无事件动词）→ state；
      - 时间戳继承：窗口内无显式标记时，继承窗口起点之前最近的一次显式时间戳
        （不是全篇末尾时间戳）。
    v8.1 修复三个确定性 Bug：
      Bug1 cur_ts 第二轮污染：旧版首轮扫描结束后 cur_ts 停留在全篇最后一个时间戳，
           第二轮窗口/状态判定直接复用它（且状态轮还继续改写它），导致【位于任何时间标记之前】
           的单元被写入【文档末尾】的时间戳。现改为首轮一次性预计算逐句 resolved_ts[i]
           （= 该位置之前最近显式时间戳），两轮只读、不再跨轮改写。
      Bug3 ev 时间触发：旧版 ev 只看事件动词，纯时间句（含 "In 2001" 但无过去时动词）落进
           state，与"时间标记或事件动词 → episodic"的设计矛盾。现 ev 同时接受显式时间标记。
    """
    sents = split_sentences(context)
    n = len(sents)
    ts_at = [None] * n          # 每句自己的显式时间戳（仅显式，用于触发与 ts_inherited）
    resolved_ts = [None] * n    # 逐句"最近一次显式时间戳"（含自身），继承用，只读
    is_event = [False] * n      # 每句是否事件句（仅事件动词）
    last_ts = None
    for i, s in enumerate(sents):
        m = TIMESTAMP_RE.search(s)
        if m:
            last_ts = m.group(0)
            ts_at[i] = last_ts
        resolved_ts[i] = last_ts
        is_event[i] = bool(EVENT_HINT_RE.search(s))

    episodic, state = [], []
    state_covered = [False] * n
    for i in range(n):
        w = sents[i:i + win]
        if not w:
            break
        end = min(i + win - 1, n - 1)
        explicit = next((ts_at[j] for j in range(i, end + 1) if ts_at[j]), None)
        # 继承源：窗口起点之前最近一次显式时间戳（i==0 时无前文可继承）
        prev_ts = resolved_ts[i - 1] if i > 0 else None
        inherited = explicit if explicit is not None else prev_ts
        # Bug3：显式时间标记同样触发 episodic（纯时间句不再落入 state）
        ev = any(is_event[j] or ts_at[j] is not None for j in range(i, end + 1))
        if ev:
            episodic.append({
                "text": " ".join(w), "sents": list(w), "start": i, "end": end,
                "ts": inherited, "ts_inherited": explicit is None and inherited is not None,
            })
            for j in range(i, end + 1):
                state_covered[j] = True

    # 状态单元：未被情景窗口覆盖的稳定陈述句（无显式时间标记、无事件动词）
    for i, s in enumerate(sents):
        if state_covered[i] or is_event[i] or ts_at[i] is not None:
            continue
        # Bug1：只读 resolved_ts[i]（该句之前最近的时间戳），不再复用/改写跨轮 cur_ts
        state.append({"text": s, "start": i, "end": i, "ts": resolved_ts[i],
                      "ts_inherited": resolved_ts[i] is not None})
    return episodic, state


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


ENCODE_BS = int(os.environ.get("V8_ENCODE_BS", "1024"))   # 词级编码分批大小（防长上下文 OOM）
UNIT_MAX_WORDS = int(os.environ.get("V8_UNIT_MAX_WORDS", "512"))  # 单元词级硬截断（建池与 IDF 共用）


def encode_words(words):
    """批量编码词列表 -> [Σlw, d] 归一化逐 token 向量（每个词 >=1 个 token）。
    分批编码（每批 ENCODE_BS 个词），峰值显存与上下文长度解耦。
    返回 (vecs_cpu_fp32, counts list[int])。"""
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
            L = int(am[i].sum())
            L = max(L, 1)
            out.append(vecs[i, :L].cpu())
            counts.append(L)
        del enc, ids, am, vecs
    return torch.cat(out, dim=0), counts


class _UnitStore:
    """情景/状态池公共底座：追加式单元存储 + 全池 IDF-MaxSim 检索（无截断）。"""

    def __init__(self):
        self.texts = []        # 单元文本
        self.metas = []        # 附加元数据（ts/start/end 等）
        self.tokv = []         # list[cpu fp32 [L,d]] 逐 token 归一化向量
        self.words = []        # list[list[str]] 与 token 对齐的词（IDF 用）
        self.mean = []         # list[tensor [d]]（mean 余弦粗排可选用）
        self.word_df = Counter()   # word -> 出现过的单元数（文档频率）
        self.n_dup = 0
        self._idf = None       # 缓存 {word: idf}
        self._mean_mat = None

    def __len__(self):
        return len(self.texts)

    def add_batch(self, units):
        """units: list[(text, meta)]。文本级精确去重。"""
        uniq = []
        seen_local = set()
        for text, meta in units:
            if not text or not text.strip():
                continue
            k = " ".join(text.split()).lower()
            if k in seen_local:
                self.n_dup += 1
                continue
            seen_local.add(k)
            uniq.append((text, meta, k))
        if not uniq:
            return
        # 展平全部词一次编码
        # Bug2 512 词 span 偏移：旧版 spans 用未截断的 len(ws) 记账，却只 extend(ws[:512])，
        # 长单元（>512 词）的 span 会越过自身边界、吃进后续单元的词，并让 off 偏移累积，
        # 导致后续单元的 tokv/words/df 全部错位。现改为【先截断再记账】。
        all_words, spans = [], []
        for text, _, _ in uniq:
            ws = _content_words(text)[:UNIT_MAX_WORDS]
            spans.append((len(all_words), len(all_words) + len(ws)))
            all_words.extend(ws)
        vecs_flat, counts = encode_words(all_words)       # [Σtokens, d]
        off = 0
        for wi, (text, meta, k) in enumerate(uniq):
            s, e = spans[wi]
            w_list = all_words[s:e]
            c_counts = counts[s:e]
            L = sum(c_counts)
            if L <= 0:
                # 兜底：零 token 单元用零向量占位。vecs_flat 没有对应行，
                # 因此【不推进 off】——推进会让后续所有单元整体偏移。
                tokv = torch.zeros((1, vecs_flat.shape[1]), dtype=torch.float32)
                w_list, c_counts = [w_list[0] if w_list else "<empty>"], [1]
            else:
                tokv = vecs_flat[off:off + L]
                off += L
            self.texts.append(text)
            self.metas.append(meta)
            self.tokv.append(tokv)
            self.words.append(w_list)
            m = F.normalize(tokv.mean(dim=0), dim=0)
            self.mean.append(m)
            for w in set(w_list):
                self.word_df[w] += 1
        self._idf = None
        self._mean_mat = None

    # ---- IDF ----
    def idf_table(self):
        if self._idf is None:
            n = max(len(self), 1)
            self._idf = {w: math.log(1.0 + n / (1.0 + df)) for w, df in self.word_df.items()}
        return self._idf

    def _query_repr(self, query):
        """query -> (qe [Lq,d] dev fp32, q_idf [Lq] dev, qm [d] dev)。"""
        dev = EMB.device
        ws = _content_words(query)[:64]
        qv, counts = encode_words(ws)                     # cpu [Lq,d]
        idf_t = self.idf_table()
        max_idf = max(idf_t.values()) if idf_t else 1.0
        w_list = []
        for w, c in zip(ws, counts):
            w_list.extend([w] * c)
        if len(w_list) != qv.shape[0]:                    # 截断保护
            w_list = (w_list + [ws[0] if ws else "<q>"] * qv.shape[0])[:qv.shape[0]]
        qw = torch.tensor([idf_t.get(w, max_idf if IDF_UNSEEN == 0 else IDF_UNSEEN)
                           for w in w_list], dtype=torch.float32, device=dev)
        qe = F.normalize(qv.to(dev), dim=-1)
        qm = F.normalize(qe.mean(dim=0), dim=0)
        return qe, qw, qm

    def idf_maxsim_scores(self, query):
        """全池无截断 IDF-MaxSim：对每个单元分批计算
            score(u) = Σ_t idf(q_t) * max_j cos(q_t, u_j) / Σ_t idf(q_t)
        返回 [N] 分数张量（dev fp32）。"""
        dev = EMB.device if EMB is not None else torch.device("cpu")
        n = len(self)
        qe, qw, _ = self._query_repr(query)
        denom = qw.sum().clamp(min=1e-6)
        scores = torch.zeros(n, dtype=torch.float32, device=dev)
        for s in range(0, n, BATCH):
            e = min(s + BATCH, n)
            Ls = [self.tokv[i].shape[0] for i in range(s, e)]
            Lm = max(Ls)
            buf = torch.zeros((e - s, Lm, qe.shape[1]), dtype=torch.float32, device=dev)
            for k, i in enumerate(range(s, e)):
                buf[k, :Ls[k]] = self.tokv[i].to(dev)
            # [B, Lq, Lm] -> 每 chunk 每 query token 的 max
            sim = torch.einsum("qd,bld->bql", qe, buf)               # [B, Lq, Lm]
            mx = sim.max(dim=2).values                               # [B, Lq]
            scores[s:e] = (mx * qw).sum(dim=1) / denom
        return scores

    def retrieve(self, query, top_k=12, pool_mask=None):
        """全池 IDF-MaxSim Top-K（返回 [(idx, score)] 分数降序，无任何 n 截断）。"""
        if len(self) == 0:
            return []
        scores = self.idf_maxsim_scores(query)
        if pool_mask is not None:
            scores = scores.clone()
            scores[~pool_mask] = -1e9
        k = min(top_k, len(self))
        vals, idxs = scores.topk(k)
        return [(int(i), float(v)) for i, v in zip(idxs.tolist(), vals.tolist())]

    def mean_mat(self):
        if self._mean_mat is None and self.mean:
            self._mean_mat = torch.stack(self.mean).to(
                EMB.device if EMB is not None else torch.device("cpu"))
        return self._mean_mat


class EpisodicFastPool(_UnitStore):
    """情景池：3 句滑窗微块全量追加（永不覆写/淘汰），全池 IDF-MaxSim 检索。

    与 v7 检索的关键差异：v7 在 n>3500 时先取 mean-cos Top-2500 再精算 MaxSim
    （截断）；本池对全部 n 个微块分批完成 MaxSim，无 Top-M 截断。"""

    def neighbor_expand(self, hits, extra=1):
        """邻域扩展：命中窗口按句序向前后各扩展 extra 个相邻微块（去重保序）。"""
        if not hits:
            return []
        order = {}
        for idx, _ in hits:
            order.setdefault(idx, True)
        picked = []
        seen = set()
        metas = self.metas
        # 按窗口 start 句序展开
        seq = sorted(order.keys(), key=lambda i: metas[i].get("start", i))
        for idx in seq:
            st = metas[idx].get("start", idx)
            en = metas[idx].get("end", idx)
            start_map = {}
            for j, m in enumerate(metas):
                start_map.setdefault(m.get("start", j), j)
            for d in range(1, extra + 1):
                for key in (st - d, en + d):
                    j = start_map.get(key)
                    if j is not None and j not in seen:
                        seen.add(j)
                        picked.append(j)
        out = list(dict.fromkeys(seq + picked))
        return out

    def best_unit_for(self, text):
        """时序锚定：定位某段文本（已发生事件）最匹配的微块。返回 (idx, score)。"""
        hits = self.retrieve(text, top_k=1)
        return hits[0] if hits else (-1, 0.0)


class StatePool(_UnitStore):
    """状态池：稳定事实单元；近重复语义合并（mean cos > STATE_THETA 覆写旧槽）。"""

    def add_batch(self, units):
        """近重复语义合并：与【已有池 + 本批 pending】比较 mean cos，
        > STATE_THETA 时新文本覆写旧槽（保位置），否则追加。"""
        pending = []
        for text, meta in units:
            v, _, qm = self._mean_only(text)
            dev = qm.device
            base = None
            if self.mean:
                base = torch.stack([m.to(dev) for m in self.mean])
            if pending:
                pm = torch.stack([p[2].to(dev) for p in pending])
                base = pm if base is None else torch.cat([base, pm])
            merged = False
            if base is not None:
                sims = base @ qm
                if float(sims.max()) > STATE_THETA:
                    j = int(sims.argmax())
                    if j < len(self.mean):
                        # 覆写已有槽：先移除旧条目（tokv/words/df 同步回收），
                        # 再把新文本作为 pending 追加，由 super().add_batch 重编码。
                        old_words = self.words[j]
                        for w in set(old_words):
                            self.word_df[w] -= 1
                            if self.word_df[w] <= 0:
                                self.word_df.pop(w, None)
                        for lst in (self.texts, self.metas, self.tokv, self.words, self.mean):
                            lst.pop(j)
                        # 释放精确去重键，允许覆写文本重新入库
                        if hasattr(self, "_seen"):
                            self._seen.discard(" ".join(text.split()).lower())
                        self._mean_mat = None
                        self._idf = None
                        pending.append((text, meta, qm.cpu()))
                    else:
                        pending[j - len(self.mean)] = (text, pending[j - len(self.mean)][1], qm.cpu())
                    self.n_dup += 1
                    merged = True
            if not merged:
                pending.append((text, meta, qm.cpu()))
        super().add_batch([(t, m) for t, m, _ in pending])

    def _mean_only(self, text):
        dev = EMB.device if EMB is not None else torch.device("cpu")
        ws = _content_words(text)[:UNIT_MAX_WORDS]
        qv, _c = encode_words(ws)
        qe = F.normalize(qv.to(dev), dim=-1)
        qm = F.normalize(qe.mean(dim=0), dim=0)
        return qe, None, qm


# ----------------------------- 组装入口 -----------------------------
def build_pools_for_context(ctx):
    """把上下文分流为情景/状态两池。返回 (epi_pool, state_pool)。"""
    epi_units, state_units = extract_adaptive_units(ctx)
    epi = EpisodicFastPool()
    epi.add_batch([(u["text"], u) for u in epi_units])
    st = StatePool()
    st.add_batch([(u["text"], u) for u in state_units])
    return epi, st
