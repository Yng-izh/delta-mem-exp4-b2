# v5_slotpool.py — 三元组解耦潜键 (k_subj, k_rel, k_obj) + 主语门控多跳接力（Beam Chaining）
# 由用户提供方案；DeepSeek 仅转写执行。依赖调用方作用域：re, torch, F, tok, EMB。
# 注：def _ extract_all_keys 中间多余空格已修正为 _extract_all_keys。

import os
import re
import torch
import torch.nn.functional as F
# 依赖调用方注入：run_group_b2.py 在 pool = SlotPool() 前设置
#   v5_slotpool.tok = tok ; v5_slotpool.EMB = EMB
# get_token_vecs 通过模块级 globals 查找 tok/EMB（def 在本模块，tok/EMB 为模块属性注入）
POOL_MAX = int(os.environ.get("B2_POOL_MAX", "5000"))
THETA_SUBJ = float(os.environ.get("B2_THETA_SUBJ", "0.82"))
THETA_REL = float(os.environ.get("B2_THETA_REL", "0.70"))
HOP_MAX = int(os.environ.get("B2_HOPS", "4"))    # 支持 1~4 跳自动接力
BEAM_WIDTH = int(os.environ.get("B2_BEAM", "2")) # 每跳最多留 2 条，杜绝干扰项
BRIDGE_TH = float(os.environ.get("B2_BRIDGE", "0.65"))   # 搭桥硬门控阈值（原 0.72）
SUBJ_GATE = float(os.environ.get("B2_SUBJGATE", "0.62")) # Hop1 主语硬门控阈值（原 0.68）

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
    r"plays|played|founded|wrote|is|was|are|were"
    r")\b",
    flags=re.IGNORECASE,
)

PREFIX_REL_RE = re.compile(
    r"^The\s+("
    r"name of the current head of(?: the)?|country of citizenship of|"
    r"official language of|original language of|language|"
    r"headquarters of|capital of|author of|partner of|spouse of|"
    r"founder of|director of|creator of|developer of|performer of|"
    r"manufacturer of|parent organization of|employer of|mother of|father of|"
    r"child of|member of|genre of|continent of|country of|sport of|religion of|"
    r"[a-z\s]+?\s+of"
    r")\s+(.+?)\s+(?:was written in is|is located in|was|is|are|were)$",
    flags=re.IGNORECASE,
)


def parse_fact_triple(fact):
    """将单条事实精准拆解为 (subj 主语实体, rel 关系谓词, obj 句尾客体)"""
    f = fact.strip().rstrip(".")
    matches = list(VP_RE.finditer(f))
    if matches:
        last_m = matches[-1]
        lhs = f[: last_m.end()].strip()
        rhs = f[last_m.end():].strip()
    else:
        toks = f.split()
        lhs = " ".join(toks[:-2]) if len(toks) > 3 else f
        rhs = " ".join(toks[-2:]) if len(toks) > 3 else f

    # 情况 A: "The partner of Nicolae Ceaușescu is" -> rel="partner of", subj="Nicolae Ceaușescu"
    m_pre = PREFIX_REL_RE.match(lhs)
    if m_pre:
        return m_pre.group(2).strip(), m_pre.group(1).strip(), rhs

    # 情况 B: "Elena Ceaușescu was educated at" -> subj="Elena Ceaușescu", rel="was educated at"
    m_vp = list(VP_RE.finditer(lhs))
    if m_vp:
        last_vp = m_vp[-1]
        subj = lhs[: last_vp.start()].strip()
        rel = lhs[last_vp.start():].strip()
        if subj:
            return subj, rel, rhs
    return lhs, lhs, rhs


def get_token_vecs(text):
    """提取去停用词后的逐词归一化向量矩阵 [L, D] 与均值向量 [D]"""
    words = [
        w
        for w in re.findall(r"[A-Za-z0-9_\u00C0-\u024F'-]+", text)
        if w.lower() not in STOPWORDS
    ]
    if not words:
        words = re.findall(r"[A-Za-z0-9_\u00C0-\u024F'-]+", text) or [text]
    clean_txt = " ".join(words)
    ids = tok(
        clean_txt, add_special_tokens=False, return_tensors="pt", truncation=True
    ).input_ids.to(EMB.device)
    if ids.numel() == 0:
        v = F.normalize(EMB[0:1].float(), dim=-1)
        return v, v.squeeze(0)
    vecs = F.normalize(EMB[ids[0]].float(), dim=-1)  # [L, D]
    mean_v = F.normalize(vecs.mean(dim=0), dim=0)    # [D]
    return vecs, mean_v


def maxsim_coverage(sub_vecs, target_vecs):
    """计算 sub_vecs 中的每个 token 在 target_vecs 中的最大余弦相似度均值 (ColBERT MaxSim)"""
    sim_mat = sub_vecs @ target_vecs.T  # [L_sub, L_tgt]
    return float(sim_mat.max(dim=1).values.mean())


class SlotPool:

    def __init__(self):
        self.k_subj = []  # [D] 主语均值向量
        self.E_subj = []  # [L, D] 主语逐 token 向量（用于精确匹配）
        self.k_rel = []   # [D] 关系均值向量
        self.E_rel = []   # [L, D] 关系逐 token 向量
        self.k_obj = []   # [D] 宾语（出键）均值向量 —— 多跳搭桥核心！
        self.E_obj = []   # [L, D] 宾语逐 token 向量
        self.texts = []
        self.serial = []
        self.t0 = []
        self.clock = 0
        self.n_overwrites = 0
        self.n_evicts = 0

    def __len__(self):
        return len(self.texts)

    def _extract_all_keys(self, fact):
        subj, rel, obj = parse_fact_triple(fact)
        E_s, k_s = get_token_vecs(subj)
        E_r, k_r = get_token_vecs(rel)
        E_o, k_o = get_token_vecs(obj)
        return (E_s, k_s, E_r, k_r, E_o, k_o)

    def upsert(self, fact, serial):
        """原位覆写或新增槽位：仅当 SUBJ 和 REL 同时命中时原位覆写，并同步更新出键 k_obj"""
        E_s, k_s, E_r, k_r, E_o, k_o = self._extract_all_keys(fact)
        if self.k_subj:
            S_subj = torch.stack(self.k_subj) @ k_s
            S_rel = torch.stack(self.k_rel) @ k_r
            # 双重条件：主语相同 且 关系相同
            mask = (S_subj > THETA_SUBJ) & (S_rel > THETA_REL)
            if mask.any():
                comb = S_subj + S_rel
                comb[~mask] = -1e9
                top = int(comb.argmax())
                # 原位覆写：更新文本与多跳出键 k_obj / E_obj！
                self.texts[top] = fact
                self.k_subj[top], self.E_subj[top] = k_s, E_s
                self.k_rel[top], self.E_rel[top] = k_r, E_r
                self.k_obj[top], self.E_obj[top] = k_o, E_o
                self.serial[top] = serial
                self.t0[top] = self.clock
                self.clock += 1
                self.n_overwrites += 1
                return True

        # 未命中则新增槽位
        self.k_subj.append(k_s)
        self.E_subj.append(E_s)
        self.k_rel.append(k_r)
        self.E_rel.append(E_r)
        self.k_obj.append(k_o)
        self.E_obj.append(E_o)
        self.texts.append(fact)
        self.serial.append(serial)
        self.t0.append(self.clock)
        self.clock += 1
        while len(self.texts) > POOL_MAX:
            idx = min(range(len(self.t0)), key=lambda i: self.t0[i])
            for lst in (
                self.k_subj,
                self.E_subj,
                self.k_rel,
                self.E_rel,
                self.k_obj,
                self.E_obj,
                self.texts,
                self.serial,
                self.t0,
            ):
                lst.pop(idx)
            self.n_evicts += 1
        return False

    def retrieve_multihop_context(self, query_text, hops=HOP_MAX, beam=BEAM_WIDTH):
        """主语门控多跳链式检索：Hop 1 锚定问题实体，Hop 2..H 用上一跳 k_obj 接力匹配下一跳 k_subj"""
        if not self.texts:
            return ""
        E_q, k_q = get_token_vecs(query_text)
        selected = []
        selected_set = set()

        # --- Hop 1: 寻找主语出现在问题中的起始事实 ---
        hop1_scores = []
        for i in range(len(self.texts)):
            subj_cov = maxsim_coverage(self.E_subj[i], E_q)
            if subj_cov < SUBJ_GATE:  # 主语硬门控：主语不在问题里的（如其他国家的大洲事实）直接淘汰！
                continue
            rel_cov = maxsim_coverage(self.E_rel[i], E_q)
            hop1_scores.append((subj_cov + 0.45 * rel_cov, i))

        # Fallback: 若硬门控未命中，退化为按 subj_cov + 0.35 * rel_cov 取前 beam 个
        if not hop1_scores:
            for i in range(len(self.texts)):
                subj_cov = maxsim_coverage(self.E_subj[i], E_q)
                rel_cov = maxsim_coverage(self.E_rel[i], E_q)
                hop1_scores.append((subj_cov + 0.35 * rel_cov, i))

        hop1_scores.sort(reverse=True)
        frontier = [idx for _, idx in hop1_scores[:beam]]
        for idx in frontier:
            selected.append(idx)
            selected_set.add(idx)

        # --- Hop 2 .. H: 拿上一跳的宾语出键 (k_obj / E_obj) 去匹配下一跳的主语入键 (k_subj / E_subj) ---
        for h in range(2, hops + 1):
            candidates = []
            for p_idx in frontier:
                p_k_obj, p_E_obj = self.k_obj[p_idx], self.E_obj[p_idx]
                # 先用矩阵乘法快速初筛 k_subj 与上一跳 k_obj 余弦较高的候选
                cos_all = torch.stack(self.k_subj) @ p_k_obj
                top_cand_idxs = (
                    torch.where(cos_all > 0.65)[0].tolist()
                    if (cos_all > 0.65).any()
                    else cos_all.argsort(descending=True)[:10].tolist()
                )
                for j in top_cand_idxs:
                    if j in selected_set:
                        continue
                    bridge_cov = maxsim_coverage(self.E_subj[j], p_E_obj)
                    bridge_sim = 0.5 * (float(cos_all[j]) + bridge_cov)
                    if (
                        bridge_sim < BRIDGE_TH
                    ):  # 搭桥硬门控：主语不是上一跳宾语的干扰项全部拦截！
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

        # 关键细节：按【Hop 1 -> Hop 2 -> Hop 3 -> Hop 4】的推理链顺序排列，方便模型顺藤摸瓜！
        return "\n".join(f"{self.serial[i]}. {self.texts[i]}" for i in selected)
