# v7_slotpool.py — v6 升级版：自适应弹性槽位池 (Adaptive Elastic SlotPool)
# 相对 v6 的改动：
#   (1) 容量自适应：B2_POOL_MAX=0（默认）时无损弹性倍增缓存（2048->4096->8192->...），
#       262k 全量 ~11.3k 独立槽位零淘汰；B2_POOL_MAX>0 时启用拓扑桥接度保护淘汰
#       （保多跳桥接节点，淘汰孤立叶节点），替代 v6 的纯 FIFO。
#   (2) 预分配连续张量缓存 _k_subj_buf/_k_rel_buf，消除每步 torch.stack 的 O(N^2) 开销；
#       gid 精确哈希 + 主语倒排桶二级路由，覆写 O(1)。
#   (3) 检索自适应：主语特异性长度加权（防短实体抢 Beam）+ 双向桥接覆盖率
#       （打通 "Zimbabwe" <-> "Zimbabwe government" 跨粒度搭桥）+ 主语尾部
#       " government" 剥离进 rel。
#   (4) 保留 v6 全部解析能力：VP_RE 长谓词优先、PREFIX_REL_RE、WHERE_CLAUSE_RE。
# 依赖调用方注入：v7_slotpool.tok = tok ; v7_slotpool.EMB = EMB（同 v6 约定）。

import os
import re
import string
import torch
import torch.nn.functional as F

# B2_POOL_MAX=0 表示【自适应无损弹性容量】（按流内独立 (subj,rel) 状态数倍增扩容）
# B2_POOL_MAX>0（如 5000）则启用【拓扑桥接度保护的自适应淘汰】
POOL_MAX = int(os.environ.get("B2_POOL_MAX", "0"))
INIT_CAP = int(os.environ.get("B2_INIT_CAP", "2048"))
THETA_SUBJ = float(os.environ.get("B2_THETA_SUBJ", "0.82"))
THETA_REL = float(os.environ.get("B2_THETA_REL", "0.70"))
HOP_MAX = int(os.environ.get("B2_HOPS", "4"))
BEAM_WIDTH = int(os.environ.get("B2_BEAM", "2"))
BRIDGE_TH = float(os.environ.get("B2_BRIDGE", "0.65"))
SUBJ_GATE = float(os.environ.get("B2_SUBJGATE", "0.62"))

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "is", "was", "are", "were", "be", "been", "what", "which", "who", "where",
    "when", "that", "this", "does", "did", "has", "have", "had",
}

VP_RE = re.compile(
    r"\b(?:"
    r"is associated with the sport of|is affiliated with the religion of|"
    r"is located in the continent of|is located in the country of|"
    r"was educated at|is educated at|was created in|is created in|"
    r"was created by|is created by|was founded by|is founded by|"
    r"is located in|was located in|is famous for|was famous for|"
    r"is associated with|is affiliated with|was written in|"
    r"was born in|was born on|is born in|"
    r"was died in|died in|died on|was killed in|"
    r"worked at|works at|worked for|works for|worked in|works in|"
    r"is a citizen of|was a citizen of|citizen of|"
    r"was married to|is married to|married to|"
    r"was buried in|is buried in|buried in|"
    r"held the position of|holds the position of|position of|"
    r"citizenship is|nationality is|"
    r"was developed by|is developed by|developed by|"
    r"was directed by|is directed by|directed by|"
    r"was written by|is written by|written by|"
    r"speaks|spoke|is the capital of|is the language of|"
    r"plays|played|founded|wrote|is|was|are|were"
    r")\b",
    flags=re.IGNORECASE,
)

# 裸系动词：不允许作为谓词拆分点（杜绝客体污染）
BARE_COPULA = {"is", "was", "are", "were"}

PREFIX_REL_RE = re.compile(
    r"^The\s+([a-z\s]+?\s+of(?: the)?)\s+(.+?)\s+"
    r"(?:was written in is|is located in|was|is|are|were)\s+(.+)$",
    flags=re.IGNORECASE,
)

WHERE_CLAUSE_RE = re.compile(
    r"^The\s+(.+?)\s+where\s+(.+?)\s+(was|is|are|were)\s+(.+?)\s+(?:is|was|are|were)\s+(.+)$",
    flags=re.IGNORECASE,
)

WHERE_NOUN_TO_REL = {
    "university": "educated at", "univeristy": "educated at",
    "universities": "educated at", "school": "educated at",
    "institution": "educated at", "place": "located in", "city": "located in",
    "town": "located in", "country": "located in", "state": "located in",
    "province": "located in", "region": "located in", "location": "located in",
    "continent": "located in", "building": "located in", "structure": "located in",
    "company": "founded by", "organization": "founded by",
    "organisation": "founded by", "year": "created in", "date": "created in",
    "person": "created by", "film": "directed by", "movie": "directed by",
    "book": "written by", "language": "spoken in", "birth": "born in",
    "death": "died in", "grave": "buried in", "cemetery": "buried in",
}

# 主语/宾语尾部机构性后缀 -> 归一化谓词后缀（打通 国家 <-> 国家政府 首脑 等桥接）
SUBJ_TAIL_TO_REL = {" government": " government"}

# PREFIX_REL_RE 也会把 "The current head of the government of Zimbabwe" 的主语误判为
# "government of Zimbabwe"（rel 位置词）；对其 obj 中出现的 "<X> government" 做同款剥离
# 由 _clean_subj_tail 统一处理即可，因为该句 v7 解析为 subj="government of Zimbabwe"，
# 尾部不是 " government"，需要专门的剥离：主语以 "government of" 结尾时把主语降为实体。


def _clean_subj_tail(subj, rel):
    """两种规范化：
    (1) subj 尾部 ' government' 移入 rel：'Zimbabwe government' -> subj='Zimbabwe', rel+=' government'
    (2) subj 以 'government of' 开头（PREFIX_REL 误判）-> subj='X', rel='government of <rel>'"""
    low = subj.lower()
    if low.endswith(" government") and len(subj.split()) >= 2:
        return subj[: -len(" government")].strip(), f"{rel} government".strip()
    if low.startswith("government of ") and len(subj.split()) >= 3:
        return subj[len("government of "):].strip(), f"government of {rel}".strip()
    return subj, rel


def _parse_where_clause(f):
    m = WHERE_CLAUSE_RE.match(f)
    if not m:
        return None
    noun, subj, obj = m.group(1).strip(), m.group(2).strip(), m.group(5).strip()
    words = [w for w in noun.lower().split() if w not in STOPWORDS] or noun.lower().split()
    rel = None
    for w in words:
        if w in WHERE_NOUN_TO_REL:
            rel = WHERE_NOUN_TO_REL[w]
            break
    if rel is None:
        rel = noun
    return subj, rel, obj


def parse_fact_triple(fact):
    """v6 同款：VP_RE 长谓词优先 + PREFIX_REL + WHERE 从句；额外做主语尾部机构后缀剥离。"""
    f = fact.strip().rstrip(".")
    matches = list(VP_RE.finditer(f))
    if matches:
        specific = [m for m in matches if m.group(0).lower() not in BARE_COPULA]
        last_m = specific[-1] if specific else matches[-1]
        lhs = f[: last_m.end()].strip()
        rhs = f[last_m.end():].strip()
    else:
        toks = f.split()
        lhs = " ".join(toks[:-2]) if len(toks) > 3 else f
        rhs = " ".join(toks[-2:]) if len(toks) > 3 else f

    m_pre = PREFIX_REL_RE.match(f)
    if m_pre:
        subj, rel = m_pre.group(2).strip(), m_pre.group(1).strip()
        subj, rel = _clean_subj_tail(subj, rel)
        return subj, rel, m_pre.group(3).strip()

    m_where = _parse_where_clause(f)
    if m_where:
        subj, rel, obj = m_where
        subj, rel = _clean_subj_tail(subj, rel)
        return subj, rel, obj

    m_vp = list(VP_RE.finditer(lhs))
    if m_vp:
        spec_vp = [m for m in m_vp if m.group(0).lower() not in BARE_COPULA]
        last_vp = spec_vp[-1] if spec_vp else m_vp[-1]
        subj = lhs[: last_vp.start()].strip()
        rel = lhs[last_vp.start():].strip()
        if subj:
            subj, rel = _clean_subj_tail(subj, rel)
            return subj, rel, rhs
    return lhs, lhs, rhs


def _norm_str(s):
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(w for w in s.split() if w not in STOPWORDS) or s.strip()


def get_token_vecs(text):
    """去停用词逐词归一化向量 [L, D]、均值向量 [D]、词数（特异性加权用）"""
    words = [
        w for w in re.findall(r"[A-Za-z0-9_\u00C0-\u024F'-]+", text)
        if w.lower() not in STOPWORDS
    ]
    if not words:
        words = re.findall(r"[A-Za-z0-9_\u00C0-\u024F'-]+", text) or [text]
    clean_txt = " ".join(words)
    ids = tok(clean_txt, add_special_tokens=False, return_tensors="pt",
              truncation=True).input_ids.to(EMB.device)
    if ids.numel() == 0:
        v = F.normalize(EMB[0:1].float(), dim=-1)
        return v, v.squeeze(0), 1
    vecs = F.normalize(EMB[ids[0]].float(), dim=-1)
    mean_v = F.normalize(vecs.mean(dim=0), dim=0)
    return vecs, mean_v, len(words)


def maxsim_coverage(sub_vecs, target_vecs):
    sim_mat = sub_vecs @ target_vecs.T
    return float(sim_mat.max(dim=1).values.mean())


def bridge_coverage(cand_subj_vecs, prev_obj_vecs):
    """双向桥接覆盖率：0.65*正向(cand子集在obj中的覆盖) + 0.35*反向(obj在cand中的覆盖)，
    打通 'Zimbabwe'(1词) <-> 'Zimbabwe government'(2词) 的跨粒度搭桥。"""
    sim_mat = cand_subj_vecs @ prev_obj_vecs.T
    cov_f = float(sim_mat.max(dim=1).values.mean())
    cov_b = float(sim_mat.max(dim=0).values.mean())
    return 0.65 * cov_f + 0.35 * cov_b


class SlotPool:
    """v7 自适应弹性槽位池：
    1. 预分配倍增连续张量缓存（无 torch.stack O(N^2)）；
    2. gid 精确哈希 + 主语倒排桶二级路由 O(1) 覆写；
    3. 默认无损弹性容量；受限 POOL_MAX>0 时拓扑桥接度保护淘汰。"""

    def __init__(self):
        self.cap = INIT_CAP
        self.dim = None
        self._k_subj_buf = None  # [cap, D] float32 GPU
        self._k_rel_buf = None

        self.E_subj = []         # [L, D] 逐 token 主语向量
        self.n_subj_words = []   # 主语词数（特异性加分）
        self.E_rel = []
        self.k_obj = []
        self.E_obj = []
        self.texts = []
        self.serial = []
        self.t0 = []
        self.subj_norm = []      # 归一化主语字符串
        self.obj_norm = []
        self.gid_list = []

        self.gid_to_idx = {}     # 精确 (subj_norm||rel_norm) -> idx
        self.subj_buckets = {}   # subj_norm -> [idx, ...] 倒排桶

        self.clock = 0
        self.n_overwrites = 0
        self.n_evicts = 0

    def __len__(self):
        return len(self.texts)

    # ---------- 容量自适应倍增缓存 ----------
    def _ensure_buf(self, d, device):
        if self._k_subj_buf is None:
            self.dim = d
            self._k_subj_buf = torch.zeros((self.cap, d), dtype=torch.float32, device=device)
            self._k_rel_buf = torch.zeros((self.cap, d), dtype=torch.float32, device=device)
        elif len(self.texts) >= self.cap:
            new_cap = self.cap * 2
            new_s = torch.zeros((new_cap, self.dim), dtype=torch.float32, device=device)
            new_r = torch.zeros((new_cap, self.dim), dtype=torch.float32, device=device)
            n = len(self.texts)
            new_s[:n] = self._k_subj_buf[:n]
            new_r[:n] = self._k_rel_buf[:n]
            self._k_subj_buf, self._k_rel_buf = new_s, new_r
            self.cap = new_cap

    def _extract_all_keys(self, fact):
        subj, rel, obj = parse_fact_triple(fact)
        E_s, k_s, nw_s = get_token_vecs(subj)
        E_r, k_r, _ = get_token_vecs(rel)
        E_o, k_o, _ = get_token_vecs(obj)
        s_norm = _norm_str(subj)
        r_norm = _norm_str(rel)
        o_norm = _norm_str(obj)
        gid = f"{s_norm}||{r_norm}"
        return (E_s, k_s, nw_s, E_r, k_r, E_o, k_o, s_norm, o_norm, gid)

    def _overwrite_at(self, idx, fact, serial, E_s, k_s, nw_s, E_r, k_r,
                      E_o, k_o, s_norm, o_norm, gid):
        old_gid = self.gid_list[idx]
        old_s = self.subj_norm[idx]
        if old_gid != gid and self.gid_to_idx.get(old_gid) == idx:
            self.gid_to_idx.pop(old_gid, None)
        if old_s != s_norm and old_s in self.subj_buckets and idx in self.subj_buckets[old_s]:
            self.subj_buckets[old_s].remove(idx)

        self._k_subj_buf[idx] = k_s
        self._k_rel_buf[idx] = k_r
        self.E_subj[idx] = E_s
        self.n_subj_words[idx] = nw_s
        self.E_rel[idx] = E_r
        self.k_obj[idx] = k_o
        self.E_obj[idx] = E_o
        self.texts[idx] = fact
        self.serial[idx] = serial
        self.t0[idx] = self.clock
        self.subj_norm[idx] = s_norm
        self.obj_norm[idx] = o_norm
        self.gid_list[idx] = gid
        self.gid_to_idx[gid] = idx
        self.subj_buckets.setdefault(s_norm, [])
        if idx not in self.subj_buckets[s_norm]:
            self.subj_buckets[s_norm].append(idx)
        self.clock += 1
        self.n_overwrites += 1

    def upsert(self, fact, serial):
        E_s, k_s, nw_s, E_r, k_r, E_o, k_o, s_norm, o_norm, gid = self._extract_all_keys(fact)
        self._ensure_buf(k_s.shape[0], k_s.device)
        n = len(self.texts)

        # 1. O(1) 精确 (subj,rel) gid 命中 -> 原位覆写
        if gid in self.gid_to_idx:
            self._overwrite_at(self.gid_to_idx[gid], fact, serial,
                               E_s, k_s, nw_s, E_r, k_r, E_o, k_o, s_norm, o_norm, gid)
            return True

        # 2. 二级主语桶内语义覆写（同主语桶通常 1~4 槽，替代全池扫描）
        bucket = self.subj_buckets.get(s_norm)
        if bucket:
            b_t = torch.tensor(bucket, device=k_r.device, dtype=torch.long)
            s_rel = self._k_rel_buf[b_t] @ k_r
            best_local = int(s_rel.argmax())
            if float(s_rel[best_local]) > THETA_REL:
                self._overwrite_at(bucket[best_local], fact, serial,
                                   E_s, k_s, nw_s, E_r, k_r, E_o, k_o, s_norm, o_norm, gid)
                return True

        # 3. 全局连续缓存语义兜底（无 torch.stack）
        if n > 0:
            S_subj = self._k_subj_buf[:n] @ k_s
            mask_s = S_subj > THETA_SUBJ
            if mask_s.any():
                cand = torch.where(mask_s)[0]
                S_rel = self._k_rel_buf[cand] @ k_r
                comb = S_subj[cand] + S_rel
                comb[S_rel <= THETA_REL] = -1e9
                if comb.max() > -1e8:
                    top = int(cand[int(comb.argmax())])
                    self._overwrite_at(top, fact, serial,
                                       E_s, k_s, nw_s, E_r, k_r, E_o, k_o, s_norm, o_norm, gid)
                    return True

        # 4. 新颖状态：弹性追加
        idx = n
        self._k_subj_buf[idx] = k_s
        self._k_rel_buf[idx] = k_r
        self.E_subj.append(E_s)
        self.n_subj_words.append(nw_s)
        self.E_rel.append(E_r)
        self.k_obj.append(k_o)
        self.E_obj.append(E_o)
        self.texts.append(fact)
        self.serial.append(serial)
        self.t0.append(self.clock)
        self.subj_norm.append(s_norm)
        self.obj_norm.append(o_norm)
        self.gid_list.append(gid)
        self.gid_to_idx[gid] = idx
        self.subj_buckets.setdefault(s_norm, []).append(idx)
        self.clock += 1

        if POOL_MAX > 0 and len(self.texts) > POOL_MAX:
            self._evict_bridge_aware()
        return False

    # ---------- 受限容量时的拓扑桥接保护淘汰 ----------
    def _evict_bridge_aware(self):
        """优先淘汰 bridge_deg==0 的最旧孤立叶节点；O(1) 交换删除。"""
        all_objs = set(self.obj_norm)
        active_subjs = set(self.subj_norm)
        best_idx, best_key = 0, (999, float("inf"))
        for i in range(len(self.texts)):
            deg = (1 if self.subj_norm[i] in all_objs else 0) + \
                  (1 if self.obj_norm[i] in active_subjs else 0)
            key = (deg, self.t0[i])
            if key < best_key:
                best_key, best_idx = key, i

        last = len(self.texts) - 1
        ev_gid, ev_s = self.gid_list[best_idx], self.subj_norm[best_idx]
        if self.gid_to_idx.get(ev_gid) == best_idx:
            self.gid_to_idx.pop(ev_gid, None)
        if ev_s in self.subj_buckets and best_idx in self.subj_buckets[ev_s]:
            self.subj_buckets[ev_s].remove(best_idx)

        if best_idx != last:
            self._k_subj_buf[best_idx] = self._k_subj_buf[last]
            self._k_rel_buf[best_idx] = self._k_rel_buf[last]
            for lst in (self.E_subj, self.n_subj_words, self.E_rel, self.k_obj,
                        self.E_obj, self.texts, self.serial, self.t0,
                        self.subj_norm, self.obj_norm, self.gid_list):
                lst[best_idx] = lst[last]
            moved_gid, moved_s = self.gid_list[best_idx], self.subj_norm[best_idx]
            self.gid_to_idx[moved_gid] = best_idx
            if moved_s in self.subj_buckets:
                self.subj_buckets[moved_s] = [
                    best_idx if x == last else x for x in self.subj_buckets[moved_s]
                ]

        for lst in (self.E_subj, self.n_subj_words, self.E_rel, self.k_obj,
                    self.E_obj, self.texts, self.serial, self.t0,
                    self.subj_norm, self.obj_norm, self.gid_list):
            lst.pop()
        self.n_evicts += 1

    # ---------- 多跳检索 ----------
    def retrieve_multihop_context(self, query_text, hops=HOP_MAX, beam=BEAM_WIDTH):
        n = len(self.texts)
        if n == 0:
            return ""
        E_q, k_q, _ = get_token_vecs(query_text)
        selected, selected_set = [], set()

        # Hop 1: 大池先用矩阵内积初筛（<=3500 全扫，否则 Top-2500），再精算特异性 MaxSim
        if n <= 3500:
            scan_range = range(n)
        else:
            scan_range = (self._k_subj_buf[:n] @ k_q).argsort(descending=True)[:2500].tolist()

        hop1_scores = []
        for i in scan_range:
            subj_cov = maxsim_coverage(self.E_subj[i], E_q)
            if subj_cov < SUBJ_GATE:
                continue
            # 特异性长度奖励：多词完整主语优先于单字子集实体（防 'Tunisia' 抢 Beam）
            spec = 1.0 + 0.025 * min(self.n_subj_words[i], 5)
            rel_cov = maxsim_coverage(self.E_rel[i], E_q)
            hop1_scores.append((subj_cov * spec + 0.45 * rel_cov, i))

        if not hop1_scores:
            for i in range(n):
                subj_cov = maxsim_coverage(self.E_subj[i], E_q)
                rel_cov = maxsim_coverage(self.E_rel[i], E_q)
                hop1_scores.append((subj_cov + 0.35 * rel_cov, i))

        hop1_scores.sort(reverse=True)
        frontier = [idx for _, idx in hop1_scores[:beam]]
        selected.extend(frontier)
        selected_set.update(frontier)

        # 问题谓词（分支展开排序用，保留 v6 谓词选择性机制）
        q_m = list(VP_RE.finditer(query_text))
        q_rel_txt = q_m[-1].group(0) if q_m else query_text
        _, k_qrel, _ = get_token_vecs(q_rel_txt)

        for h in range(2, hops + 1):
            candidates = []
            for p_idx in frontier:
                p_k_obj, p_E_obj = self.k_obj[p_idx], self.E_obj[p_idx]
                p_o_norm = self.obj_norm[p_idx]
                # 路径 A: O(1) 主语桶精确命中（与池规模无关！）
                exact = list(self.subj_buckets.get(p_o_norm, []))
                # 路径 B: 连续矩阵余弦初筛
                cos_all = self._k_subj_buf[:n] @ p_k_obj
                if (cos_all > 0.62).any():
                    sem = torch.where(cos_all > 0.62)[0].tolist()
                else:
                    sem = cos_all.argsort(descending=True)[:12].tolist()
                for j in dict.fromkeys(exact + sem):
                    if j in selected_set:
                        continue
                    if j in exact:
                        bridge_sim = 1.0
                    else:
                        b_cov = bridge_coverage(self.E_subj[j], p_E_obj)
                        bridge_sim = 0.5 * (float(cos_all[j]) + b_cov)
                    if bridge_sim < BRIDGE_TH:
                        continue
                    rel_cov = maxsim_coverage(self.E_rel[j], E_q)
                    candidates.append((bridge_sim + 0.45 * rel_cov, j))

            candidates.sort(reverse=True)
            new_frontier = []
            for _, j in candidates:
                if j not in selected_set and len(new_frontier) < beam:
                    new_frontier.append(j)
                    selected.append(j)
                    selected_set.add(j)
            if not new_frontier:
                break
            frontier = new_frontier

            # 谓词选择性分支展开（v6 机制，改用主语倒排桶替代全池余弦扫描）
            for j in list(new_frontier):
                j_s = self.subj_norm[j]
                branches = []
                for i in self.subj_buckets.get(j_s, []):
                    if i in selected_set:
                        continue
                    if float(self._k_subj_buf[i] @ self._k_subj_buf[j]) > 0.82:
                        branches.append((float(self._k_rel_buf[i] @ k_qrel), i))
                branches.sort(reverse=True)
                for _, i in branches[:2]:
                    selected.append(i)
                    selected_set.add(i)

        return "\n".join(f"{self.serial[i]}. {self.texts[i]}" for i in selected)
