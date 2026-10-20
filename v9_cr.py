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
#   (3) 双跳链谓词联合打分 Score_chain：seed→mid→target 链的分数 =
#       W_SUBJ·subj_cov(target) + W_BRIDGE·(bridge(seed,mid)+bridge(mid,target))/2
#       + W_PRED·(rel_cov(mid)+rel_cov(target))/2，两条边谓词都与 query 谓词比较。
#   (4) 反事实强约束 Prompt：编号越大越新；冲突时只认最大编号；作答前先做
#       "若候选答案为真则会与更晚事实矛盾" 的自检；禁止输出被覆盖值。
#   (5) 路径式读出（V9_CR_PATH_VIEW=1，默认开）：不再把"种子 + 中间节点 + 目标"
#       混成一个大集合平铺，而是选少量完整路径按 seed → intermediate → target 展示，
#       每条路径保留实体连接与关系分数；未进入路径的已选事实放 OTHER SELECTED FACTS。
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

    def chain_score(self, seed, mid, tgt, E_q, E_qrel):
        """双跳链联合打分 Score_chain。

        tgt is None 时退化为单跳扩展分（0.6*bridge + 0.4*rel_cov(mid)）。
        链分三项：目标主语覆盖、两跳桥接、两条边谓词与 query 谓词的平均覆盖。
        """
        b1 = 1.0 if self.cr_subj_norm[mid] == self.cr_obj_norm[seed] else \
            _bridge(self.cr_E_subj[mid], self.cr_E_obj[seed])
        if tgt is None:
            return 0.6 * b1 + 0.4 * _maxsim(self.cr_E_rel[mid], E_qrel)
        b2 = 1.0 if self.cr_subj_norm[tgt] == self.cr_obj_norm[mid] else \
            _bridge(self.cr_E_subj[tgt], self.cr_E_obj[mid])
        subj_cov = _maxsim(self.cr_E_subj[tgt], E_q)
        pred = 0.5 * (_maxsim(self.cr_E_rel[mid], E_qrel) +
                      _maxsim(self.cr_E_rel[tgt], E_qrel))
        return self.w_subj * subj_cov + self.w_bridge * 0.5 * (b1 + b2) + \
            self.w_pred * pred

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
    def retrieve_multihop_context(self, query_text, hops=None, beam=None):
        """多跳读出，返回可直接拼进 prompt 的证据块；dbg 写入 self.last_dbg。

        两种渲染（V9_CR_PATH_VIEW）：
          - 1（默认）：先给至多 CR_PATHS 条完整路径 `种子 --[关系]--> 中间 --[关系]--> 目标`，
                       再列其余已选事实（OTHER SELECTED FACTS）；
          - 0：平铺，按 serial 升序。
        两种渲染的事实行都保留**全局 serial 编号**，反事实 Prompt 的"编号越大越新"仍成立。
        """
        n = len(self)
        self.last_dbg = {"n_pool": n}
        if n == 0:
            return ""
        E_q, _kq, _ = _enc(query_text)
        q_rel_txt = query_text
        qm = list(VP_RE.finditer(query_text))
        if qm:
            q_rel_txt = qm[-1].group(0)
        E_qrel, _kr, _ = _enc(q_rel_txt)
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
        ego = self._ego_candidates(seeds, E_q, E_qrel, buckets)[:self.ego_k]
        for _s, j in ego:
            if j not in seen:
                selected.append(j)
                seen.add(j)

        chains_dbg, path_entries, frontier = [], [], list(seeds)
        for h in range(2, hops + 1):
            entries = []
            for s in frontier:
                for _b1, mid in self._bridge_candidates(s, buckets):
                    entries.append((0.6 * _b1 + 0.4 * _maxsim(self.cr_E_rel[mid], E_qrel),
                                    s, mid, None))
                    for _b2, tgt in self._bridge_candidates(mid, buckets):
                        if tgt == s:
                            continue
                        entries.append((self.chain_score(s, mid, tgt, E_q, E_qrel),
                                        s, mid, tgt))
            entries.sort(key=lambda x: (-x[0], x[1], x[2],
                                        -1 if x[3] is None else x[3]))
            added = []
            for sc, s, mid, tgt in entries:
                if len(added) >= beam:
                    break
                if mid not in seen:
                    selected.append(mid)
                    seen.add(mid)
                    added.append(mid)
                if tgt is not None and tgt not in seen:
                    selected.append(tgt)
                    seen.add(tgt)
                    added.append(tgt)
                chains_dbg.append((round(sc, 4), int(mid),
                                   None if tgt is None else int(tgt)))
                if tgt is not None:
                    path_entries.append((float(sc), int(s), int(mid), int(tgt)))
            if not added:
                break
            frontier = added[:beam]

        # 路径候选：按链分降序，记录实体连接与每条边的谓词覆盖分（渲染/诊断用）
        paths = []
        for sc, s, mid, tgt in sorted(path_entries, key=lambda x: (-x[0], x[1], x[2], x[3])):
            subj_s, rel_s, obj_s = parse_fact_triple(self.texts[s])
            subj_m, rel_m, obj_m = parse_fact_triple(self.texts[mid])
            paths.append({
                "score": round(sc, 4),
                "seed": s, "mid": mid, "target": tgt,
                "facts": [s, mid, tgt],
                "labels": [subj_s, obj_s or subj_m, obj_m],
                "edges": [{"from": subj_s, "rel": rel_s, "to": obj_s},
                          {"from": subj_m, "rel": rel_m, "to": obj_m}],
                "edge_scores": [round(_maxsim(self.cr_E_rel[s], E_qrel), 4),
                                round(_maxsim(self.cr_E_rel[mid], E_qrel), 4)],
            })

        order = sorted(selected, key=lambda i: self.metas[i].get("serial", i))
        if CR_PATH_VIEW and paths:
            block = self._render_paths(paths, order)
        else:
            block = "\n".join(f"{self.metas[i].get('serial', i + 1)}. {self.texts[i]}"
                              for i in order)
        self.last_dbg.update({
            "hop1": [[round(s, 4), i] for s, i in h1[:beam]],
            "seeds": seeds,
            "ego": [[round(s, 4), i] for s, i in ego],
            "chains": chains_dbg,
            "paths": paths[:max(1, CR_PATHS)],
            "path_view": bool(CR_PATH_VIEW and paths),
            "selected": order,
            "n_selected": len(order),
            "beam": beam,
            "query_rel": q_rel_txt,
            "n_overwrites": self.n_overwrites,
        })
        return block

    def _render_paths(self, paths, selected):
        """路径式读出：先给少量完整路径（seed → intermediate → target），再列其余事实。

        每条路径保留：整体链分、每条边的谓词覆盖分、实体连接；
        事实行仍带全局 serial 编号，因此反事实 Prompt 的"编号越大越新"规则不受影响。
        """
        lines, used, k = [], set(), 0
        for p in paths:
            if k >= max(1, CR_PATHS):
                break
            ids = [i for i in p["facts"] if i is not None]
            if any(i in used for i in ids):
                continue
            k += 1
            used.update(ids)
            e = p["edges"]
            chain = (f"{p['labels'][0]} --[{e[0]['rel']}]--> {p['labels'][1]} "
                     f"--[{e[1]['rel']}]--> {p['labels'][2]}")
            lines.append(f"PATH {k} (score={p['score']:.3f}, "
                         f"rel={p['edge_scores'][0]:.2f}/{p['edge_scores'][1]:.2f}): {chain}")
            for i in ids:
                lines.append(f"{self.metas[i].get('serial', i + 1)}. {self.texts[i]}")
        rest = [i for i in selected if i not in used]
        if rest:
            lines.append("OTHER SELECTED FACTS:")
            for i in rest:
                lines.append(f"{self.metas[i].get('serial', i + 1)}. {self.texts[i]}")
        return "\n".join(lines)
