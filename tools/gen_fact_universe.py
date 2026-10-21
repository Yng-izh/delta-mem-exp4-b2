#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/gen_fact_universe.py — 生成大规模**虚构**事实宇宙（训练语料源）

为什么不用现成数据集
--------------------
训练事实必须**独立于最终评测集**。服务器上唯一的本地数据集是
`Conflict_Resolution-*.parquet` —— 那正是评测集本身，用它训练等于背答案。
所以这里造一个虚构事实宇宙：实体名全部由音节表合成，与真实世界知识不冲突
（不会出现"模型凭先验就能答"的情况，强制它必须读记忆），也零泄漏。

每条事实仍渲染成**自然陈述句**（而不是直接给三元组），这样才能走
`parse_fact_triple` 的真实抽取路径 —— 训练与推理用同一套解析器，
解析口径不会错位。

一个"簇"（cluster）= 一个人 + 其雇主/出生地/母校/配偶/著作/职位……
簇内实体自然形成多跳链（person -> org -> country），簇间实体名不重复，
所以多跳链不会意外跨簇。

用法
----
    # 4000 个簇 -> 约 6 万条事实
    python tools/gen_fact_universe.py --clusters 4000 --out /tmp/facts.txt

    # 只要事实条数（便于配比）
    python tools/gen_fact_universe.py --clusters 1200 --out /tmp/facts.txt --stats
"""
import argparse
import json
import os
import random
import sys

# 音节表：名字由"辅音+元音"拼装，保证发音自然且组合空间足够大
ONSET = ["b", "d", "f", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v",
         "z", "br", "dr", "gr", "kr", "tr", "vr", "sh", "th", "st", "sk", "ph"]
NUCLEUS = ["a", "e", "i", "o", "u", "ai", "au", "ei", "ia", "io", "ou", "ea"]
CODA = ["", "", "", "n", "r", "l", "s", "m", "th", "nd", "rk", "st"]

ORG_HEAD = ["Astra", "Boreal", "Cinder", "Dovetail", "Ember", "Fathom", "Gale",
            "Harbor", "Ironwood", "Juniper", "Kestrel", "Lantern", "Meridian",
            "Northwind", "Opal", "Pinnacle", "Quarry", "Ridgeway", "Summit",
            "Tidewater", "Umber", "Vantage", "Willow", "Yarrow", "Zephyr"]
ORG_TAIL = ["Systems", "Labs", "Analytics", "Robotics", "Energy", "Dynamics",
            "Holdings", "Logistics", "Materials", "Networks", "Partners",
            "Foundry", "Mining", "Biotech", "Studios", "Shipping"]

ROLES = ["chief engineer", "head of research", "finance director",
         "operations manager", "lead architect", "chief counsel",
         "regional director", "head of logistics", "principal scientist",
         "compliance officer", "plant supervisor", "creative director"]

WORK_ADJ = ["Silent", "Crimson", "Northern", "Hollow", "Golden", "Distant",
            "Broken", "Hidden", "Restless", "Quiet", "Burning", "Frozen"]
WORK_NOUN = ["Fjord", "Lantern", "Harvest", "Compass", "Orchard", "Signal",
             "Horizon", "Passage", "Archive", "Meridian", "Citadel", "Voyage"]


def word(rng, syllables=2):
    out = []
    for _ in range(syllables):
        out.append(rng.choice(ONSET) + rng.choice(NUCLEUS) + rng.choice(CODA))
    w = "".join(out)
    return w[:1].upper() + w[1:]


class Names:
    """去重的名字池（保证簇内/簇间实体不重名，否则多跳链会被意外桥接）。"""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.used = set()

    def one(self, syllables=2):
        """单个唯一词（用于机构名的品牌后缀，避免出现三四个词的怪名字）。

        注意 syllables=1 时组合空间只有 26*12*12=3744，2000 簇需要 4000 个机构名
        会直接耗尽（踩过）；2 音节有 ~1400 万组合，够用。
        """
        for _ in range(500):
            w = word(self.rng, syllables)
            if w.lower() not in self.used:
                self.used.add(w.lower())
                return w
        raise RuntimeError("单词池耗尽")

    def get(self, syllables=2, space="x"):
        for _ in range(200):
            a, b = word(self.rng, syllables), word(self.rng, syllables)
            name = f"{a} {b}"
            key = name.lower()
            if key not in self.used:
                self.used.add(key)
                return name
        raise RuntimeError(f"名字池耗尽（space={space}），请增大音节组合或减少簇数")


def make_cluster(rng, names, idx):
    """一个簇：人 + 雇主 + 出生地 + 母校 + 配偶 + 著作 + 职位 ……（10~16 条事实）"""
    person = names.get(2)
    spouse = names.get(2)
    # 机构名/作品名也必须进唯一池：原来用 word() 直接拼，不注册唯一性，
    # 4000 个机构里会撞名 -> train/val 共享实体（实测交集 1 个）。
    org = f"{rng.choice(ORG_HEAD)} {rng.choice(ORG_TAIL)} {names.one()}"
    org2 = f"{rng.choice(ORG_HEAD)} {rng.choice(ORG_TAIL)} {names.one()}"
    city = names.get(2)
    city2 = names.get(2)
    country = names.get(1)
    country2 = names.get(1)
    uni = f"{names.get(2)} University"
    # 作品名同样要唯一：只有 12x12=144 种组合，跨簇必然重复
    work = f"The {rng.choice(WORK_ADJ)} {rng.choice(WORK_NOUN)} of {names.get(1)}"
    role = rng.choice(ROLES)
    year = rng.randint(1930, 2005)

    f = []
    # --- 单跳事实 ---
    f.append(rng.choice([f"{person} works at {org}.",
                         f"{person} works for {org}.",
                         f"{person} worked at {org}."]))
    f.append(rng.choice([f"{org} is located in the country of {country}.",
                         f"{org} is located in {country}."]))
    f.append(rng.choice([f"{person} was born in {city}.",
                         f"{person} was born in {year}."]))
    f.append(f"{city} is located in the country of {country2}.")
    f.append(f"{city} is the capital of {names.get(1)}.")
    f.append(f"{person} was educated at {uni}.")
    f.append(f"{uni} is located in the country of {names.get(1)}.")
    f.append(f"{person} is married to {spouse}.")
    f.append(f"{spouse} works at {org2}.")
    f.append(f"{org2} is located in the country of {names.get(1)}.")
    f.append(f"{person} founded {org2}." if rng.random() < 0.5
             else f"{org2} was founded by {person}.")
    f.append(f"{person} wrote {work}.")
    f.append(f"{person} holds the position of {role}.")
    f.append(f"{person} is a citizen of {country}.")
    f.append(f"{person} is famous for {work}.")
    # 跨簇不重复的额外事实：city2 的归属（给多跳链备用）
    f.append(f"{city2} is located in the country of {country2}.")
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成虚构事实宇宙（训练语料）")
    ap.add_argument("--clusters", type=int, default=1000)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--jsonl", action="store_true",
                    help="输出 {text, cluster} jsonl（便于按簇切 train/val，杜绝实体泄漏）")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    names = Names(args.seed)
    facts, clusters = [], []
    for i in range(args.clusters):
        batch = make_cluster(rng, names, i)
        facts.extend(batch)
        clusters.extend([i] * len(batch))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            for text, c in zip(facts, clusters):
                if args.jsonl:
                    fh.write(json.dumps({"text": text, "cluster": c}, ensure_ascii=False) + "\n")
                else:
                    fh.write(text + "\n")
    if args.stats or not args.out:
        print(json.dumps({"clusters": args.clusters, "facts": len(facts),
                          "unique_names": len(names.used),
                          "per_cluster": round(len(facts) / max(1, args.clusters), 2)},
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
