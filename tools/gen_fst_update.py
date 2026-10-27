#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/gen_fst_update.py — FST-Mem 阶段 B 数据：同一实体多属性、多次覆写、带时间条件。

与 latent_exp2 的差别（这是新方法要解决的新问题）：

    exp2    : 4 条**互不相关**的静态事实，问其中一条 —— 只有"选哪条"的问题。
    本数据  : 同一实体的多个属性各自被**多次覆写**，问"现在"或"某一年" ——
              同时有"选哪个属性"和"选哪个时间版本"两个竞争，外加"改一个属性
              不能影响另一个属性"的局域性要求。

一条样本（row）回答**一个**问题，但携带完整时间线，因此同一个时间线会被展开成
三种问题（雇主-当前 / 雇主-历史 / 居住地-当前），其中"居住地-当前"就是
**无关属性保持率**的探针：它的答案与雇主怎么变毫无关系。

反事实配对的口径与 exp2 一致：同一实体、同一问题文本，两个变体的雇主值互换
（A→B 与 B→A）。只有真的读了**当前版本**才能两组都答对；背 Question→Answer 必错。
**注意配对只对 current / history 两个探针成立**：unrelated 探针问的是居住地，
居住地在两个变体里是同一个值，两个变体的 gold 相同 —— 那种"对"不含任何反事实信息，
评估侧必须把它排除在 pair_acc 之外（`evaluate_fst` 按"同问题且 gold 不同"筛）。

刻意不做的事：
  - 不 shuffle memory 顺序。时间线顺序本身是"追加式记忆"最强的形态，故意留给
    "全部历史追加"基线；FST 不依赖顺序（版本由 time 字段索引），这样对比才公平。
  - 不制造"两条事实同时间戳"的样本：那种样本的正确读法本身有歧义，
    混进来只会让指标无法归因。
  - **干扰事实的值不与本行任何答案值重合**。旧版从同一个 VALUES 池随机取，
    结果 21.9% 的训练行里答案字符串直接出现在干扰事实中 —— 那等于"值在场即可答"，
    不需要正确读目标槽位/版本，准确率被系统性虚高（与 exp2 的 22.5% 缺陷同一类）。
  - 干扰事实的实体**只从本侧实体池取**，否则"train/val 实体不重叠"只对目标键成立，
    干扰文本里仍会出现对侧实体名，"未见实体泛化"这个说法就站不住。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from gen_latent_exp2 import SYL1, SYL2, VALUES, SENT, Q, gen_names  # noqa: E402

# 两个属性：雇主 + 居住地。刻意选一对"语义上互不相关"的属性，
# 这样"改雇主后居住地是否还读得对"才是一个干净的局域性探针。
REL_EMP = "works at"
REL_HOME = "lives in"

Q_PAST = {
    REL_EMP: "In {t}, where did {s} work?",
    REL_HOME: "In {t}, where did {s} live?",
}


def build_timeline(rng, entity, emp_values, home_value, years):
    """按年份交替写入：雇主 2 次覆写 + 居住地 1 次。返回 [(time, rel, value, fact)]。"""
    assert len(emp_values) == 2, emp_values
    tl = [
        (years[0], REL_EMP, emp_values[0]),
        (years[1], REL_HOME, home_value),
        (years[2], REL_EMP, emp_values[1]),
    ]
    return [(t, rel, v, SENT[rel].format(s=entity, v=v)) for t, rel, v in tl]


def make_rows(rng, entity, other_entities, years, n_distractors=2):
    """一个实体 -> 2 个反事实变体 × 3 个问题 = 6 行。

    other_entities 必须是**同侧**实体池（train 行的干扰实体不能来自 val，反之亦然）。
    """
    emp_a, emp_b = rng.sample(VALUES[REL_EMP], 2)
    home = rng.choice(VALUES[REL_HOME])
    rows = []
    for emp_values in ([emp_a, emp_b], [emp_b, emp_a]):
        tl = build_timeline(rng, entity, emp_values, home, years)
        cur_emp = tl[2]
        hist_emp = tl[0]
        cur_home = tl[1]
        # 干扰事实：**同关系不同实体**（逼模型做实体绑定）+ 其他随机事实
        dis, dis_meta, used_keys = [], [], {f"{entity.lower()}||{REL_EMP}",
                                           f"{entity.lower()}||{REL_HOME}"}
        # 三个答案值都不能出现在干扰事实里（两个变体的当前/历史雇主 + 居住地）
        forbid = {emp_a, emp_b, home}
        tries = 0
        while len(dis) < n_distractors:
            tries += 1
            if tries > 200:
                raise SystemExit(f"干扰事实生成失败（值池太小？）：entity={entity} "
                                 f"forbid={sorted(forbid)}")
            e2 = rng.choice(other_entities)
            r2 = rng.choice([REL_EMP, REL_HOME])
            k2 = f"{e2.lower()}||{r2}"
            pool = [v for v in VALUES[r2] if v not in forbid]
            if not pool:
                raise SystemExit(f"VALUES[{r2}] 去掉答案值后为空：{sorted(VALUES[r2])}")
            f = SENT[r2].format(s=e2, v=rng.choice(pool))
            # 干扰事实的 (实体,属性) 键必须与已有键互不冲突：
            # 同一个键被写成"两条同时刻版本"会让"该读哪个版本"本身有歧义，
            # 这种样本混进来会让指标无法归因（分不清是机制错还是数据歧义）。
            if f in dis or e2 == entity or k2 in used_keys:
                continue
            used_keys.add(k2)
            dis.append(f)
            dis_meta.append({"entity": e2, "relation": r2, "time": "",
                             "fact": f, "distractor": True})
        memory = [f for _t, _r, _v, f in tl] + dis
        unit_meta = [{"entity": entity, "relation": r, "time": t, "fact": f,
                      "distractor": False} for t, r, v, f in tl] + dis_meta
        base = {"kind": "update", "hops": 1, "entity": entity, "unit_meta": unit_meta,
                "timeline": [{"time": t, "entity": entity, "relation": r, "value": v,
                              "fact": f} for t, r, v, f in tl],
                "memory": memory, "units": list(memory),
                "distractors": dis, "hist_years": list(years)}
        # ① 雇主-当前（as_of 为空）：必须选"当前版本"，且不受居住地/干扰影响
        rows.append(dict(base, **{
            "relation": REL_EMP, "fact_key": f"{entity.lower()}||{REL_EMP}",
            "question": Q[REL_EMP].format(s=entity), "answer": cur_emp[2],
            "as_of": "", "evidence": [cur_emp[3]],
            "answer_fact_time": cur_emp[0], "probe": "current"}))
        # ② 雇主-历史（问最早那一年）：旧值必须还在，且按时间选中
        rows.append(dict(base, **{
            "relation": REL_EMP, "fact_key": f"{entity.lower()}||{REL_EMP}",
            "question": Q_PAST[REL_EMP].format(s=entity, t=hist_emp[0]),
            "answer": hist_emp[2], "as_of": hist_emp[0],
            "evidence": [hist_emp[3]],
            "answer_fact_time": hist_emp[0], "probe": "history"}))
        # ③ 居住地-当前（当局局性探针：雇主怎么改都不该影响它）
        rows.append(dict(base, **{
            "relation": REL_HOME, "fact_key": f"{entity.lower()}||{REL_HOME}",
            "question": Q[REL_HOME].format(s=entity), "answer": cur_home[2],
            "as_of": "", "evidence": [cur_home[3]],
            "answer_fact_time": cur_home[0], "probe": "unrelated"}))
    return rows


def validate(rows, label=""):
    """生成后立刻自检。任何一条不成立就 SystemExit —— 数据缺陷必须在生成时炸，
    而不是等到训练完发现准确率虚高再回头怀疑方法。"""
    bad = []

    def norm(x):
        return " ".join(str(x).lower().split())

    for i, r in enumerate(rows):
        ums = r.get("unit_meta") or []
        units = r.get("units") or []
        if len(ums) != len(units):
            bad.append((i, f"unit_meta {len(ums)} != units {len(units)}"))
            continue
        for um, u in zip(ums, units):
            if norm(um.get("fact")) != norm(u):
                bad.append((i, f"unit_meta.fact != units: {um.get('fact')!r} vs {u!r}"))
        # ① 答案值不得出现在**任何非目标单元**里（否则"值在场就能答对"）
        ans = norm(r.get("answer"))
        ev = {norm(e) for e in (r.get("evidence") or [])}
        for um, u in zip(ums, units):
            if norm(u) in ev:
                continue
            if ans and ans in norm(u):
                bad.append((i, f"答案值出现在非目标单元：answer={r.get('answer')!r} "
                               f"unit={u!r}"))
        # ② 干扰项的事实键不得与目标键冲突，也不得互相重复
        tgt_key = f"{str(r.get('entity')).lower()}||{r.get('relation')}"
        seen = set()
        for um in ums:
            if not um.get("distractor"):
                continue
            k = f"{str(um.get('entity')).lower()}||{um.get('relation')}"
            if k == tgt_key:
                bad.append((i, f"干扰项键与目标键相同：{k}"))
            if k in seen:
                bad.append((i, f"干扰项键重复：{k}"))
            seen.add(k)
    # ③ 反事实配对：current/history 探针必须同问题两行且 gold 不同
    groups = {}
    for r in rows:
        groups.setdefault((norm(r.get("question")), r.get("probe")), []).append(
            norm(r.get("answer")))
    for (q, probe), golds in groups.items():
        if probe in ("current", "history"):
            if len(golds) != 2:
                bad.append((-1, f"配对行数不是 2：probe={probe} q={q!r} n={len(golds)}"))
            elif len(set(golds)) != 2:
                bad.append((-1, f"配对两行 gold 相同（不含反事实信息）：q={q!r}"))
        if probe == "unrelated" and len(set(golds)) != 1:
            bad.append((-1, f"unrelated 探针两变体 gold 应相同：q={q!r} {set(golds)}"))
    if bad:
        for b in bad[:10]:
            print(f"[validate][FAIL] {label} 行{b[0]}: {b[1]}", file=sys.stderr)
        raise SystemExit(f"[validate] {label} 共 {len(bad)} 处不成立，拒绝落盘")
    print(f"[validate] {label} OK：{len(rows)} 行，答案值无泄漏、干扰键无冲突、配对完整")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", default=os.path.join(ROOT, "synth", "fst_update_train.jsonl"))
    ap.add_argument("--val-out", default=os.path.join(ROOT, "synth", "fst_update_val.jsonl"))
    ap.add_argument("--n-train-entities", type=int, default=32)
    ap.add_argument("--n-val-entities", type=int, default=12)
    ap.add_argument("--n-distractors", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--years", default="2021,2023,2025",
                    help="三个写入年份（历史、无关属性、当前）")
    args = ap.parse_args()

    years = [y.strip() for y in str(args.years).split(",")]
    if len(years) != 3 or len(set(years)) != 3:
        raise SystemExit(f"--years 必须是 3 个不同年份：{args.years!r}")

    rng = random.Random(args.seed)
    all_names = gen_names(rng, args.n_train_entities + args.n_val_entities + 64)
    train_ent = all_names[:args.n_train_entities]
    val_ent = all_names[args.n_train_entities:args.n_train_entities + args.n_val_entities]
    extra = all_names[args.n_train_entities + args.n_val_entities:]

    def dump(entities, path):
        rows = []
        for e in entities:
            # 同侧实体池：干扰事实的实体也不跨侧（见文件头"刻意不做的事"）
            others = [x for x in list(entities) + list(extra) if x != e]
            rows.extend(make_rows(rng, e, others, years, args.n_distractors))
        validate(rows, label=os.path.basename(path))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return rows

    tr = dump(train_ent, args.train_out)
    va = dump(val_ent, args.val_out)
    # 实体不重叠是这套数据的核心性质：新实体上的泛化才算泛化（否则只是背名字）
    leak = set(train_ent) & set(val_ent)
    if leak:
        raise SystemExit(f"训练/验证实体重叠：{sorted(leak)[:5]}")
    # 目标键不重叠还不够：干扰文本里出现对侧实体名，"未见实体"就不是真的未见
    def texts(rows):
        return " ".join(" ".join(r.get("units") or []) for r in rows)
    tr_txt, va_txt = texts(tr), texts(va)
    cross = sorted({e for e in val_ent if e in tr_txt} |
                   {e for e in train_ent if e in va_txt})
    if cross:
        raise SystemExit(f"对侧实体名出现在记忆文本里（{len(cross)} 个）：{cross[:5]}")
    probes = {}
    for r in tr:
        probes[r["probe"]] = probes.get(r["probe"], 0) + 1
    print(f"[out] train={len(tr)} -> {args.train_out}")
    print(f"[out] val={len(va)} -> {args.val_out}")
    print(f"[out] 实体 train/val={len(train_ent)}/{len(val_ent)}（不相交）"
          f"+{len(extra)} 个只做干扰")
    print(f"[out] 探针分布(train)={probes}")
    print(f"[out] 例：{json.dumps(tr[0], ensure_ascii=False)[:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
