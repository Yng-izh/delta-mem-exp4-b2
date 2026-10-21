# v9_cr.py — CR（FactConsolidation）v9 强化：双阈值覆写 + Beam=6 ego-graph + 双跳链联合打分 + 反事实 Prompt
#
# 相对 v7（v7_slotpool.SlotPool，CR 基线）的改动：
#   (1) 覆写：沿用 v9_slotpool.StatePool 的【主语精确匹配 + 关系谓词余弦 >
#       V9_CR_REL_THETA(0.80)】双阈值，替换 v8 的 mean-cos(0.92) 单阈值；
#       本文件只补图检索所需的 (subj, rel, obj) 逐 token 键。
#       v9 修复：目标槽由 StatePool._merge_target 按【同主语槽集合】定位（不再看全池
#       argmax），且覆写走 `_replace` **原位更新**（槽 id 稳定，不 drop+append）。
#   (2) Beam=6 ego-graph 第 1 跳：Hop1 取 6 个主语门控种子后，立刻并入每个种子的
#       ego 邻域（同主语的其他属性 + 出边实体），不再只留 2 个孤立种子。
#       v9 修复：Hop1 先用主语均值向量做一次矩阵初筛（V9_CR_HOP1_M）再逐 token MaxSim，
#       避免对全池逐槽 Python 循环（262K/约一万槽时查询延迟不可接受）。
#   (3) 双跳链谓词联合打分 Score_chain：评分主体是**2 事实 2 关系**结构
#       (fact1, fact2)——fact1: subj1 --rel1--> mid，fact2: mid --rel2--> answer，
#       答案就是 obj(fact2)（例：Alice --attended--> UnivB --located in--> CityC）。
#       硬约束：obj(fact1) 与 subj(fact2) 归一化后必须相等，不相等直接不构成链
#       （chain_score 返回 None）。分数 =
#       W_SUBJ·subj_cov(fact1) + W_BRIDGE·bridge(fact1,fact2)
#       + W_PRED·两段关系分别与 query 两段关系的匹配（query 只有一段关系时两段都比它）。
#       3 事实链是**独立分支** path_len=3（chain_score3）：单独打分、单独渲染，
#       不与 2 跳混在一起排序。v9 早期版本把 3 事实链当"双跳"打分（seed→mid→target），
#       第三跳会污染前两跳评分、真正的 2 跳路径可能进不了路径展示（v9 修复）。
#   (4) 反事实强约束 Prompt：编号越大越新；冲突时只认最大编号；作答前先做
#       "若候选答案为真则会与更晚事实矛盾" 的自检；禁止输出被覆盖值。
#   (5) 路径式读出（V9_CR_PATH_VIEW=1，默认开）：不再把"种子 + 中间节点 + 目标"
#       混成一个大集合平铺，而是选少量完整路径按 seed → intermediate → answer 展示；
#       每条路径**只渲染真正参与评分的事实与边**，事实行数 == 边数（2 跳 2 行 2 边、
#       3 跳 3 行 3 边）。未进入路径的已选事实放 OTHER SELECTED FACTS。
#       V9_CR_PATH_VIEW=0 退回按 serial 升序的平铺块（与早期 v9 行为一致）。
#
# 注入约定（同 v7/v8）：v9_cr.set_encoder(tok, EMB)（内部同步 v9_slotpool.tok/EMB）。
# CPU 可测：注入 stub tokenizer + 小嵌入矩阵即可跑全部逻辑（不加载模型/不占 GPU）。
#
# 接入（run_group_b2_v7.py，CR_MODE=v9）：
#   import v9_cr; v9_cr.set_encoder(tok, EMB)
#   pool = v9_cr.CRStatePool()
#   for serial, fact in enumerate(facts, 1): pool.upsert(fact, serial)
#   block = pool.retrieve_multihop_context(question)
#   prompt = v9_cr.build_cr_prompt(question, block)

import os
import re

import torch
import torch.nn.functional as F

import v9_slotpool as v9
from v7_slotpool import parse_fact_triple, VP_RE

# ----------------------------- 环境参数 -----------------------------
CR_BEAM       = int(os.environ.get("V9_CR_BEAM", "6"))            # Hop1 种子数（v7=2）
CR_EGO_K      = int(os.environ.get("V9_CR_EGO_K", "12"))          # ego 邻域最多并入几条
CR_SUBJ_GATE  = float(os.environ.get("V9_CR_SUBJ_GATE", "0.62"))  # 主语门控（同 v7）
CR_BRIDGE     = float(os.environ.get("V9_CR_BRIDGE", "0.65"))     # 桥接阈值（同 v7）
CR_HOPS       = int(os.environ.get("V9_CR_HOPS", "4"))            # 最多跳数（同 v7）
CR_W_SUBJ     = float(os.environ.get("V9_CR_W_SUBJ", "0.40"))     # 链分：目标主语覆盖
CR_W_BRIDGE   = float(os.environ.get("V9_CR_W_BRIDGE", "0.30"))   # 链分：两跳桥接
CR_W_PRED     = float(os.environ.get("V9_CR_W_PRED", "0.30"))     # 链分：两跳谓词
CR_POOL_MAX   = int(os.environ.get("V9_CR_POOL_MAX", "0"))        # 0=弹性无损（同 v7 默认）
CR_CF_PROMPT  = os.environ.get("V9_CR_CF_PROMPT", "1") == "1"     # 反事实 Prompt 开关
CR_HOP1_M     = int(os.environ.get("V9_CR_HOP1_M", "96"))         # Hop1 矩阵初筛候选数
CR_PATH_VIEW  = os.environ.get("V9_CR_PATH_VIEW", "1") == "1"     # 路径式读出（否则平铺）
CR_PATHS      = int(os.environ.get("V9_CR_PATHS", "3"))           # 最多展示几条完整路径

# 反事实强约束 Prompt（CR 的"新值覆盖旧值"语义写进指令，而不是只靠事实顺序）
CR_PROMPT_HEADER = (
    "You are given numbered facts extracted from a long document. "
    "Larger numbers appear LATER in the document, so they are NEWER.\n"
    "STRICT RULES:\n"
    "1. If two facts conflict (same subject and relation, different value), only the fact "
    "with the LARGEST number is current. Every earlier value is OUTDATED and must be ignored.\n"
    "2. Counterfactual self-check before answering: if your candidate answer were true, it "
    "would contradict a later-numbered fact. When that happens, discard the candidate and "
    "use the later fact instead.\n"
    "3. Never output a superseded value, even if it appears in more facts or looks familiar.\n"
    "4. Answer with the shortest exact entity/value only. No explanation.\n"
    "5. If the facts do not contain the answer, answer exactly: unknown."
)


def build_cr_prompt(question, block):
    """反事实强约束 Prompt：FACTS 块 + 冲突/覆盖规则 + 问题。"""
    facts = block.strip() if block and block.strip() else "(no facts retrieved)"
    return f"{CR_PROMPT_HEADER}\n\nFACTS:\n{facts}\n\nQUESTION: {question}"


def set_encoder(tokenizer, emb):
    """注入 tokenizer / 嵌入矩阵（同时同步 v9_slotpool 的模块级注入点）。"""
    v9.tok = tokenizer
    v9.EMB = emb


def _enc(text):
    """文本 -> (逐 token 向量 [L,d] fp32, 均值向量 [d], 词数)。"""
    ws = v9._content_words(text or "")
    if not ws:
        ws = [(text or "").strip().lower() or "<empty>"]
    ws = ws[:v9.UNIT_MAX_WORDS]
    v, _counts = v9.encode_words(ws)
    if v.shape[0] == 0:
        dim = v9.EMB.shape[1] if v9.EMB is not None else 1
        v = torch.zeros((1, dim), dtype=torch.float32)
    v = v.float()
    return v, F.normalize(v.mean(dim=0), dim=0), len(ws)


def _maxsim(A, B):
    if A is None or B is None or A.shape[0] == 0 or B.shape[0] == 0:
        return 0.0
    return float((A @ B.T).max(dim=1).values.mean())


def _bridge(cand_E_subj, prev_E_obj):
    """双向桥接覆盖率（同 v7）：0.65*正向 + 0.35*反向，打通跨粒度实体。"""
    if cand_E_subj is None or prev_E_obj is None:
        return 0.0
    if cand_E_subj.shape[0] == 0 or prev_E_obj.shape[0] == 0:
        return 0.0
    sim = cand_E_subj @ prev_E_obj.T
    return 0.65 * float(sim.max(dim=1).values.mean()) + \
           0.35 * float(sim.max(dim=0).values.mean())


class CRStatePool(v9.StatePool):
    """CR v9 槽位池：双阈值覆写（继承 StatePool）+ 三元组图检索。

    与 v7 的差异：Hop1 Beam=6 并做 ego-graph 扩展；Hop2+ 用 Score_chain 联合打分；
    输出按 serial（事实原文序号）升序，保证 Prompt 的"编号越大越新"成立。
    """

    def __init__(self, theta=None, beam=None, ego_k=None, subj_gate=None,
                 bridge_th=None, hops=None, pool_max=None):
        super().__init__(theta=theta)
        self.beam = CR_BEAM if beam is None else int(beam)
        self.ego_k = CR_EGO_K if ego_k is None else int(ego_k)
        self.subj_gate = CR_SUBJ_GATE if subj_gate is None else float(subj_gate)
        self.bridge_th = CR_BRIDGE if bridge_th is None else float(bridge_th)
        self.hops = CR_HOPS if hops is None else int(hops)
        self.pool_max = CR_POOL_MAX if pool_max is None else int(pool_max)
        self.w_subj = CR_W_SUBJ
        self.w_bridge = CR_W_BRIDGE
        self.w_pred = CR_W_PRED

        # 图检索键（与 mean 槽一一对应；_store/_drop 同步）
        self.cr_E_subj = []
        self.cr_E_rel = []
        self.cr_E_obj = []
        self.cr_k_subj = []
        self.cr_k_obj = []
        self.cr_subj_norm = []
        self.cr_obj_norm = []
        self.cr_n_subj = []
        self._subj_mat_cache = None

        self.n_overwrites = 0
        self.n_evicts = 0
        self.last_dbg = {}
        self._rendered_paths = []        # 上一次 _render_paths 真正渲染的路径（dbg 用）

    # ---------------- 写入 ----------------
    def _store(self, texts, index_texts, metas):
        keep = [(t, it, m) for t, it, m in zip(texts, index_texts, metas)
                if t and t.strip()]
        super()._store(texts, index_texts, metas)       # StatePool：双阈值键 + 基础槽
        for t, _it, _m in keep:
            subj, rel, obj = parse_fact_triple(t)
            E_s, k_s, nw_s = _enc(subj)
            E_r, k_r, _ = _enc(rel)
            E_o, k_o, _ = _enc(obj)
            self.cr_E_subj.append(E_s)
            self.cr_E_rel.append(E_r)
            self.cr_E_obj.append(E_o)
            self.cr_k_subj.append(k_s)
            self.cr_k_obj.append(k_o)
            self.cr_subj_norm.append(v9._cr_norm(subj))
            self.cr_obj_norm.append(v9._cr_norm(obj))
            self.cr_n_subj.append(nw_s)
        self._subj_mat_cache = None

    def _drop(self, j):
        if j < len(self.cr_E_subj):
            for lst in (self.cr_E_subj, self.cr_E_rel, self.cr_E_obj,
                        self.cr_k_subj, self.cr_k_obj, self.cr_subj_norm,
                        self.cr_obj_norm, self.cr_n_subj):
                lst.pop(j)
        self._subj_mat_cache = None
        super()._drop(j)

    def _replace(self, j, text, meta):
        """原位覆写（槽 id 稳定）：同步 CR 图检索键，保证图键与槽下标仍一一对应。"""
        super()._replace(j, text, meta)
        subj, rel, obj = parse_fact_triple(text)
        E_s, k_s, nw_s = _enc(subj)
        E_r, _k_r, _ = _enc(rel)
        E_o, k_o, _ = _enc(obj)
        self.cr_E_subj[j] = E_s
        self.cr_E_rel[j] = E_r
        self.cr_E_obj[j] = E_o
        self.cr_k_subj[j] = k_s
        self.cr_k_obj[j] = k_o
        self.cr_subj_norm[j] = v9._cr_norm(subj)
        self.cr_obj_norm[j] = v9._cr_norm(obj)
        self.cr_n_subj[j] = nw_s
        self._subj_mat_cache = None

    def upsert(self, fact, serial):
        """写入一条事实；同主语+同谓词（双阈值）命中则原位覆写语义（旧槽被替换）。

        返回 True 表示发生了覆写（冲突解决），False 表示新增独立状态。
        """
        fact = (fact or "").strip()
        if not fact:
            return False
        n_dup0 = self.n_dup
        self.add_batch([(fact, {"serial": int(serial),
                                "year": v9.extract_year(fact)})])
        merged = self.n_dup > n_dup0
        if merged:
            self.n_overwrites += 1
        if self.pool_max and len(self) > self.pool_max:
            self._evict()
        return merged

    def _evict(self):
        """受限容量（V9_CR_POOL_MAX>0）时优先淘汰孤立叶节点；默认弹性容量不触发。"""
        if not len(self):
            return
        all_obj = set(self.cr_obj_norm)
        active_subj = set(self.cr_subj_norm)
        def deg(i):
            return (1 if self.cr_subj_norm[i] in all_obj else 0) + \
                   (1 if self.cr_obj_norm[i] in active_subj else 0)
        best = min(range(len(self)),
                   key=lambda i: (deg(i), self.metas[i].get("serial", i)))
        self._drop(best)
        self.n_evicts += 1

    # ---------------- 打分 ----------------
    def _subj_mean_mat(self):
        if self._subj_mat_cache is None or self._subj_mat_cache.shape[0] != len(self):
            self._subj_mat_cache = (torch.stack(self.cr_k_subj)
                                    if self.cr_k_subj else torch.zeros((0, 1)))
        return self._subj_mat_cache

    def _buckets(self):
        b = {}
        for i, s in enumerate(self.cr_subj_norm):
            b.setdefault(s, []).append(i)
        return b

    def hop1_candidates(self, E_q):
        """主语均值向量矩阵初筛：返回候选槽下标（至多 V9_CR_HOP1_M 个）。

        O(N·d) 的一次矩阵乘；逐 token MaxSim 只在这批候选上做。
        n <= M 时直接全量，保证小池行为与旧版完全一致。
        """
        n = len(self)
        if n == 0:
            return []
        m = min(max(int(CR_HOP1_M), self.beam * 4, 16), n)
        if m >= n:
            return list(range(n))
        q_k = F.normalize(E_q.float().mean(dim=0), dim=0)
        cos = self._subj_mean_mat() @ q_k
        idx = torch.topk(cos, m).indices.tolist()
        return sorted(int(i) for i in idx)

    def hop1_scores(self, E_q):
        """Hop1 主语门控 + 特异性长度奖励 + 谓词覆盖（同 v7 口径）。

        性能：先做主语均值向量矩阵初筛（hop1_candidates），再只对候选做逐 token MaxSim。
        v9 早期版本对全池逐槽 Python 循环 MaxSim，262K（约一万槽）时查询延迟不可接受。
        保底：初筛后若无候选过门控，退回全池软打分（不因初筛丢召回）。
        """
        n = len(self)
        if n == 0:
            return []
        out = []
        for i in self.hop1_candidates(E_q):
            cov = _maxsim(self.cr_E_subj[i], E_q)
            if cov < self.subj_gate:
                continue
            spec = 1.0 + 0.025 * min(self.cr_n_subj[i], 5)
            out.append((cov * spec + 0.45 * _maxsim(self.cr_E_rel[i], E_q), i))
        if not out:                                   # 门控全空：退回软打分
            out = [(_maxsim(self.cr_E_subj[i], E_q) +
                    0.35 * _maxsim(self.cr_E_rel[i], E_q), i) for i in range(n)]
        out.sort(key=lambda x: (-x[0], x[1]))
        return out

    # ---- 链的硬约束 / 关系匹配 ----
    def linked(self, f1, f2):
        """链的硬约束：fact1 的宾语与 fact2 的主语归一化后必须相等。

        这是"链"的定义本身（同一实体把两条事实接起来），不相等就根本不是链，
        不能用语义相似度糊过去（v9 早期版本用 bridge 软门控，导致伪链入榜）。
        """
        return self.cr_obj_norm[f1] == self.cr_subj_norm[f2]

    @staticmethod
    def _as_qrels(E_qrels):
        """兼容单条 query 关系（tensor）与多条（list）。"""
        if E_qrels is None:
            return []
        if torch.is_tensor(E_qrels):
            return [E_qrels]
        return list(E_qrels)

    def _edge_rel_scores(self, facts, E_qrels):
        """每条边（= 每条事实的关系）与 query 对应关系的覆盖分。

        第 i 条边对 query 的第 i 段关系；query 关系少于边数时，多出的边沿用最后一段
        （2 跳 query 只有一段关系时两段都比它，与旧口径一致）。
        """
        qs = self._as_qrels(E_qrels)
        if not qs:
            return [0.0] * len(facts)
        return [_maxsim(self.cr_E_rel[f], qs[min(i, len(qs) - 1)])
                for i, f in enumerate(facts)]

    def _rel_pair(self, f1, f2, E_qrels):
        """两段链关系分别与 query 两段关系匹配（单段 query 时两段都比它）。"""
        return 0.5 * sum(self._edge_rel_scores([f1, f2], E_qrels))

    def chain_score(self, f1, f2, E_q, E_qrels):
        """双跳链联合打分 Score_chain：(fact1, fact2) 二元结构。

        fact1: subj1 --rel1--> mid，fact2: mid --rel2--> answer。
        硬约束 obj(fact1) == subj(fact2)；不满足或 f1 == f2 时返回 None（不构成链）。
        链分三项：起点主语覆盖、两跳桥接（硬约束成立即 1.0）、两段关系与 query 匹配。
        """
        if f1 == f2 or not self.linked(f1, f2):
            return None
        subj_cov = _maxsim(self.cr_E_subj[f1], E_q)
        bridge = 1.0                       # 硬约束成立，桥接必然闭合
        pred = self._rel_pair(f1, f2, E_qrels)
        return self.w_subj * subj_cov + self.w_bridge * bridge + self.w_pred * pred

    def chain_score3(self, f1, f2, f3, E_q, E_qrels):
        """3 事实链（path_len=3）独立打分：不与 2 跳分支混排。

        硬约束逐边成立：obj(f1) == subj(f2) 且 obj(f2) == subj(f3)。
        多出来的第三跳只影响本分支分数，绝不会回流污染 2 跳分支的分数。
        """
        if len({f1, f2, f3}) != 3:
            return None
        if not (self.linked(f1, f2) and self.linked(f2, f3)):
            return None
        subj_cov = _maxsim(self.cr_E_subj[f1], E_q)
        pred = sum(self._edge_rel_scores([f1, f2, f3], E_qrels)) / 3.0
        return self.w_subj * subj_cov + self.w_bridge * 1.0 + self.w_pred * pred

    def single_hop_score(self, f1, f2, E_qrels):
        """单事实扩展分（path_len=1 兜底分支，不参与 2/3 跳排序）。"""
        b1 = 1.0 if self.linked(f1, f2) else \
            _bridge(self.cr_E_subj[f2], self.cr_E_obj[f1])
        return 0.6 * b1 + 0.4 * self._edge_rel_scores([f2], E_qrels)[0]

    def _bridge_candidates(self, node, buckets):
        """node 的出边候选：宾语精确命中的主语桶 + 主语向量余弦初筛。"""
        n = len(self)
        if n == 0:
            return []
        exact = list(buckets.get(self.cr_obj_norm[node], []))
        cos = self._subj_mean_mat() @ self.cr_k_obj[node]
        sem = torch.where(cos > self.bridge_th)[0].tolist()
        if not sem:
            sem = cos.argsort(descending=True)[:12].tolist()
        out = []
        for j in dict.fromkeys(exact + sem):
            if j == node:
                continue
            b = 1.0 if j in exact else _bridge(self.cr_E_subj[j], self.cr_E_obj[node])
            if b >= self.bridge_th:
                out.append((b, j))
        out.sort(key=lambda x: (-x[0], x[1]))
        return out

    def _ego_candidates(self, seeds, E_q, E_qrel, buckets):
        """Beam=6 种子的 ego 邻域：同主语的其他属性 + 出边实体（1 跳）。"""
        seed_set = set(seeds)
        cand = {}
        for s in seeds:
            neigh = list(buckets.get(self.cr_subj_norm[s], [])) + \
                    list(buckets.get(self.cr_obj_norm[s], []))
            for j in dict.fromkeys(neigh):
                if j in seed_set or j in cand:
                    continue
                bridge = 1.0 if self.cr_subj_norm[j] == self.cr_obj_norm[s] else \
                    _bridge(self.cr_E_subj[j], self.cr_E_obj[s])
                cand[j] = 0.4 * bridge + \
                    0.4 * _maxsim(self.cr_E_rel[j], E_qrel) + \
                    0.2 * _maxsim(self.cr_E_subj[j], E_q)
        return sorted(((v, k) for k, v in cand.items()), key=lambda kv: (-kv[0], kv[1]))

    # ---------------- 检索 ----------------
    def _query_rels(self, query_text):
        """query 的两段关系（长谓词优先，保持出现顺序）；不足两段时只有一段。"""
        rels = []
        for m in VP_RE.finditer(query_text):
            r = m.group(0).strip()
            if r and r not in rels:
                rels.append(r)
            if len(rels) == 2:
                break
        return rels

    def _path_entry(self, facts, score, E_qrels):
        """路径记录：facts 与 edges 严格一一对应（事实行数 == 边数），供渲染与诊断。"""
        triples = [parse_fact_triple(self.texts[i]) for i in facts]
        return {
            "score": round(float(score), 4),
            "path_len": len(facts),
            "facts": [int(i) for i in facts],
            "labels": [triples[0][0]] + [obj for _s, _r, obj in triples],
            "edges": [{"from": subj, "rel": rel, "to": obj}
                      for subj, rel, obj in triples],
            "edge_scores": [round(x, 4) for x in self._edge_rel_scores(facts, E_qrels)],
        }

    def retrieve_multihop_context(self, query_text, hops=None, beam=None):
        """多跳读出，返回可直接拼进 prompt 的证据块；dbg 写入 self.last_dbg。

        链检索分三条**互不混排**的分支：
          - path_len=2（主分支）：(fact1, fact2) 二元双跳链，chain_score 打分；
          - path_len=3（独立分支）：(fact1, fact2, fact3) 三事实链，chain_score3 打分，
            单独排序、单独渲染，不与 2 跳混在一起排序；
          - path_len=1（兜底）：无链可扩展时用单事实扩展分补位。
        两种渲染（V9_CR_PATH_VIEW）：
          - 1（默认）：先给至多 CR_PATHS 条完整路径（2 跳在前、3 跳在后），
                       再列其余已选事实（OTHER SELECTED FACTS）；
          - 0：平铺，按 serial 升序。
        两种渲染的事实行都保留**全局 serial 编号**，反事实 Prompt 的"编号越大越新"仍成立。
        """
        n = len(self)
        self.last_dbg = {"n_pool": n}
        if n == 0:
            return ""
        E_q, _kq, _ = _enc(query_text)
        q_rels = self._query_rels(query_text)
        E_qrels = [_enc(r)[0] for r in q_rels] or [_enc(query_text)[0]]
        hops = self.hops if hops is None else int(hops)
        beam = self.beam if beam is None else int(beam)
        buckets = self._buckets()

        h1 = self.hop1_scores(E_q)
        seeds = [i for _s, i in h1[:beam]]
        selected, seen = [], set()
        for i in seeds:
            if i not in seen:
                selected.append(i)
                seen.add(i)
        ego = self._ego_candidates(seeds, E_q, E_qrels[0], buckets)[:self.ego_k]
        for _s, j in ego:
            if j not in seen:
                selected.append(j)
                seen.add(j)

        chains2, chains3, paths2, paths3 = [], [], [], []
        frontier = list(seeds)
        for _h in range(2, hops + 1):
            ent1, ent2, ent3 = [], [], []
            for s in frontier:
                for _b1, mid in self._bridge_candidates(s, buckets):
                    ent1.append((self.single_hop_score(s, mid, E_qrels), s, mid))
                    sc2 = self.chain_score(s, mid, E_q, E_qrels)
                    if sc2 is not None:
                        ent2.append((sc2, s, mid))
                    for _b2, tgt in self._bridge_candidates(mid, buckets):
                        if tgt in (s, mid):
                            continue
                        sc3 = self.chain_score3(s, mid, tgt, E_q, E_qrels)
                        if sc3 is not None:
                            ent3.append((sc3, s, mid, tgt))
            ent1.sort(key=lambda x: (-x[0], x[1], x[2]))
            ent2.sort(key=lambda x: (-x[0], x[1], x[2]))
            ent3.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))

            added = []

            def _take(node):
                if node in seen:
                    return False
                selected.append(node)
                seen.add(node)
                added.append(node)
                return True

            # 分支 A：2 跳（各分支独立取槽，分支之间不按分数混排）
            for sc, s, mid in ent2[:beam]:
                _take(mid)
                chains2.append((round(sc, 4), int(s), int(mid)))
                paths2.append(self._path_entry((s, mid), sc, E_qrels))
            # 分支 B：3 跳（独立打分 / 独立渲染，第三跳不回流污染 2 跳分支）
            for sc, s, mid, tgt in ent3[:beam]:
                _take(mid)
                _take(tgt)
                chains3.append((round(sc, 4), int(s), int(mid), int(tgt)))
                paths3.append(self._path_entry((s, mid, tgt), sc, E_qrels))
            if not added:                       # 兜底：无链可扩，退回单事实扩展
                for _sc, s, mid in ent1[:beam]:
                    _take(mid)
            if not added:
                break
            frontier = added[:beam]

        cap = max(int(beam), 4)
        paths = paths2[:cap] + paths3[:cap]
        order = sorted(selected, key=lambda i: self.metas[i].get("serial", i))
        rendered, self._rendered_paths = [], []
        if CR_PATH_VIEW and paths:
            block = self._render_paths(paths, order)
            rendered = self._rendered_paths
        else:
            block = "\n".join(f"{self.metas[i].get('serial', i + 1)}. {self.texts[i]}"
                              for i in order)
        self.last_dbg.update({
            "hop1": [[round(s, 4), i] for s, i in h1[:beam]],
            "seeds": seeds,
            "ego": [[round(s, 4), i] for s, i in ego],
            "query_rel": q_rels[0] if q_rels else query_text,
            "query_rels": q_rels,
            "chains2": chains2,             # [(score, fact1, fact2)]        path_len=2
            "chains3": chains3,             # [(score, fact1, fact2, fact3)] path_len=3
            "chains": [{"path_len": 2, "score": sc, "facts": [a, b]}
                       for sc, a, b in chains2] +
                      [{"path_len": 3, "score": sc, "facts": [a, b, c]}
                       for sc, a, b, c in chains3],
            "paths": paths,                 # 候选路径（2 跳在前、3 跳在后），带 path_len
            "paths_rendered": rendered,
            "n_paths_by_len": {2: len(paths2), 3: len(paths3)},
            "path_view": bool(CR_PATH_VIEW and paths),
            "selected": order,
            "n_selected": len(order),
            "beam": beam,
            "n_overwrites": self.n_overwrites,
        })
        return block

    def diag_ctx(self, block=None):
        """逐题检索证据（可直接 `ar_diag.record(block=...)` 展开）。

        retrieved_*：检索器给出的候选，**严格分数降序**（ar_diag 的 Recall@K 依赖
        这个顺序）。CR 的排序单位是链而不是单槽，因此先把链展开成槽位序列：同一
        槽位只在首次出现处记一次，分数 = 该链分；未进任何链但被选中的槽位用 hop1
        分补上。展开后统一按分数降序排序（同分保持首次出现顺序），**不按分支分组**
        ——2 跳/3 跳分支分组只用于路径渲染，不能拿来当检索排名。
        used_*：真正拼进 prompt 的**事实行**（block 里 '<serial>. <原文>' 的行，
        PATH 头等结构行不是证据），与槽位一一对应（used_alignment 应为 ok）。
        同一槽位可能重复出现（多条路径共享实体），这是 Prompt 的真实形态，不合并。
        """
        dbg = self.last_dbg or {}
        lines = block_fact_lines(self, block)
        used_ids = [i for i, _ in lines]
        used_texts = [t for _, t in lines]
        score_of, order = {}, []
        for ch in dbg.get("chains") or []:
            try:
                sc = float(ch.get("score") or 0.0)
            except (TypeError, ValueError):
                sc = 0.0
            for i in ch.get("facts") or []:
                if i is None:
                    continue
                i = int(i)
                if i in score_of:
                    continue
                score_of[i] = sc
                order.append(i)
        hop1 = {}
        for pair in dbg.get("hop1") or []:
            try:
                hop1.setdefault(int(pair[1]), float(pair[0]))
            except (TypeError, ValueError, IndexError):
                continue
        for i in dbg.get("selected") or []:
            i = int(i)
            if i in score_of:
                continue
            score_of[i] = hop1.get(i, 0.0)
            order.append(i)
        ranked = sorted(range(len(order)), key=lambda k: (-score_of[order[k]], k))
        order = [order[k] for k in ranked]
        return {"pool": self, "hits": [(i, score_of[i]) for i in order],
                "used_ids": used_ids, "used_texts": used_texts, "pool_kind": "cr"}

    def _render_paths(self, paths, selected):
        """路径式读出：每条路径只渲染**真正参与评分**的事实与边（事实行数 == 边数）。

        2 跳路径 = 2 行事实 + 2 条边，3 跳路径 = 3 行事实 + 3 条边；不再出现
        "渲染 3 行事实却只有 2 条边"（v9 早期把 tgt 当路径组成部分的错位）。
        路径顺序由调用方给定（2 跳分支在前、3 跳分支在后，分支内按链分降序），
        因此两类路径不会被混在一起排序。事实行仍带全局 serial 编号，
        反事实 Prompt 的"编号越大越新"规则不受影响。

        名额分配：CR_PATHS 是总名额；3 跳分支存在时**至少保留 1 个名额**给它
        （2 跳最多占 CR_PATHS-1 个），保证 3 跳路径不会被 2 跳分支整块吞掉。
        去重只跳过事实完全相同的路径（同一路径不重复渲染），允许不同路径共享实体
        ——共享实体正是多跳链的正常形态。
        """
        budget = max(1, CR_PATHS)
        two = [p for p in paths if p["path_len"] == 2]
        three = [p for p in paths if p["path_len"] == 3]
        reserve3 = 1 if (three and two and budget >= 2) else 0
        lines, used, k, rendered = [], set(), 0, []

        def _emit(group, limit):
            nonlocal k
            for p in group:
                if limit is not None and k >= limit:
                    return
                if k >= budget:
                    return
                facts = [i for i in p["facts"] if i is not None]
                if tuple(facts) in used:        # 同一条路径不重复渲染
                    continue
                k += 1
                used.add(tuple(facts))
                chain = p["labels"][0]
                for e in p["edges"]:
                    chain += f" --[{e['rel']}]--> {e['to']}"
                rel_s = "/".join(f"{x:.2f}" for x in p["edge_scores"])
                lines.append(f"PATH {k} (len={p['path_len']}, score={p['score']:.3f}, "
                             f"rel={rel_s}): {chain}")
                for i in facts:                 # 事实行数 == 边数
                    lines.append(f"{self.metas[i].get('serial', i + 1)}. {self.texts[i]}")
                rendered.append(p)

        _emit(two, budget - reserve3)           # 2 跳分支：不越界到 3 跳的名额
        _emit(three, None)                      # 3 跳分支：独立渲染，用剩余名额
        covered = {i for p in rendered for i in p["facts"]}
        rest = [i for i in selected if i not in covered]
        if rest:
            lines.append("OTHER SELECTED FACTS:")
            for i in rest:
                lines.append(f"{self.metas[i].get('serial', i + 1)}. {self.texts[i]}")
        self._rendered_paths = rendered
        return "\n".join(lines)


# ----------------------------- Delta 快照接入（CR_DELTA_MODE） -----------------------------
# 旧 runner 在查询前无条件 reset_delta_mem_states(model)：写进 Delta 的记忆在读出时
# 已经清零，B2_KEEP_DELTA 只进配置指纹、从不控制这次 reset，所以 CR 里 Delta State
# 永远不参与查询（用户审计已实测）。三档模式把"Delta 参不参与查询"变成显式消融项：
#   off     —— 查询前清零（历史行为，逐字节不变）；
#   context —— 写入后快照、每题恢复（复用 AR 的 delta_snapshot/delta_restore 语义），
#              所有题目看到同一份"写后"记忆状态；
#   refresh —— 在 context 之上，每题用本题 Top-K 检索结果重写这些槽位文本，
#              与 AR 侧 DELTA_WRITE_STRATEGY=topk_refresh 同口径（只刷新 Top-K 槽位）。
CR_DELTA_MODES = ("off", "context", "refresh")
CR_DELTA_REFRESH_TOPK = int(os.environ.get("V9_CR_DELTA_REFRESH_TOPK", "6"))


def parse_delta_mode(value):
    """CR_DELTA_MODE -> 规范值；非法值直接报错（不静默退回 off）。"""
    v = str("off" if value is None else value).strip().lower() or "off"
    if v not in CR_DELTA_MODES:
        raise ValueError(f"CR_DELTA_MODE must be one of {CR_DELTA_MODES}, got {value!r}")
    return v


def delta_plan(mode, n_pool=0):
    """每题读出前的 Delta 动作计划（纯函数，runner 照着执行）。

    n_pool 只把池规模记进计划（诊断用），不影响动作本身。
    """
    m = parse_delta_mode(mode)
    if m == "off":
        return {"mode": m, "reset_before_query": True, "restore_snapshot": False,
                "refresh_topk": 0, "n_pool": int(n_pool)}
    return {"mode": m, "reset_before_query": False, "restore_snapshot": True,
            "refresh_topk": (CR_DELTA_REFRESH_TOPK if m == "refresh" else 0),
            "n_pool": int(n_pool)}


def refresh_slots(hits, top_k):
    """检索结果 -> 需要刷新的槽位（分数序去重后取前 top_k）。纯函数。

    hits 形态与 pool.retrieve / ar_diag 一致：[(idx, score), ...]（分数降序）。
    top_k<=0 表示不刷新，返回空列表（off/context 不应走到刷新逻辑）。
    """
    if not top_k or int(top_k) <= 0:
        return []
    out, seen = [], set()
    for h in hits or []:
        if isinstance(h, (list, tuple)) and h:
            i = h[0]
        elif isinstance(h, dict):
            i = h.get("idx", h.get("index", h.get("slot_id")))
        else:
            i = h
        try:
            i = int(i)
        except (TypeError, ValueError):
            continue
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
        if len(out) >= int(top_k):
            break
    return out


CR_BLOCK_LINE_RE = re.compile(r"^(\d+)\.\s+(.*)$")


def block_fact_lines(pool, block):
    """证据块 -> [(槽位下标, 事实行原文)]；跳过 'PATH ...' / 'OTHER SELECTED FACTS:'。

    只认 '<serial>. <原文>' 事实行（全局 serial -> 池内下标）。反查不到下标的行直接
    丢弃：宁可在日志里少一行，也不编造槽位 ID。
    """
    texts = list(getattr(pool, "texts", None) or [])
    metas = list(getattr(pool, "metas", None) or [])
    by_serial = {}
    for i, m in enumerate(metas):
        if isinstance(m, dict) and m.get("serial") is not None:
            try:
                by_serial.setdefault(int(m["serial"]), i)
            except (TypeError, ValueError):
                continue
    out = []
    for line in str(block or "").split("\n"):
        m = CR_BLOCK_LINE_RE.match(line)
        if not m:
            continue
        body = m.group(2)
        idx = by_serial.get(int(m.group(1)))
        if idx is None:                      # 没有 serial 元数据 -> 按原文反查
            idx = next((i for i, t in enumerate(texts) if str(t) == body), None)
        if idx is None:
            continue
        out.append((int(idx), line))
    return out
