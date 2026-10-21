#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tests/test_synth_data.py — 外部语料接入训练脚本的单元测试

覆盖 `train_latent_memory.py` 里新增的 4 个接口：
    load_synth_jsonl / item_units / check_split_leakage / encode_item
以及 `--data / --val-data / --n-train / --n-val` 的 CLI 契约。

这些接口决定"训练分布是否与推理期一致"，错一个不会报错、只会让指标莫名其妙地低，
所以每条都写成**可失败的断言**，不写恒真式。

运行（需要 torch，故不在无 torch 的环境跑）：
    python tests/test_synth_data.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import train_latent_memory as T

_N = [0]


def ok(name, cond):
    _N[0] += 1
    if not cond:
        raise AssertionError(f"FAIL {name}")
    print(f"ok {_N[0]:3d} {name}")


def _write(rows):
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    fh.close()
    return fh.name


def _rec(q="Q?", a="A.", mem=("f1.", "f2."), hops=1, cluster=0, **kw):
    rec = {"question": q, "answer": a, "memory": list(mem), "evidence": list(mem),
           "hops": hops, "meta": {"cluster": cluster}}
    rec.update(kw)
    return rec


# ---------------------------------------------------------------------
# A. load_synth_jsonl
# ---------------------------------------------------------------------
def test_load():
    p = _write([
        _rec("Q1?", "A1.", ("s1.", "s2.", "s3."), hops=2, cluster=7),
        _rec("Q2?", "A2.", ("s4.",), hops=1, cluster=8),
        {"question": "no answer", "memory": ["x."]},            # 缺 answer -> 丢弃
        {"question": "no memory", "answer": "y.", "memory": []},  # 空记忆 -> 丢弃
        {"memory": ["z."], "answer": "w."},                     # 缺 question -> 丢弃
    ])
    items = T.load_synth_jsonl(p)
    os.unlink(p)
    ok("A1 丢弃缺字段记录后剩 2 条", len(items) == 2)
    ok("A2 memory 由 units 空格拼接", items[0]["memory"] == "s1. s2. s3.")
    ok("A3 units 保持逐条事实", items[0]["units"] == ["s1.", "s2.", "s3."])
    ok("A4 hops 透传", items[0]["hops"] == 2 and items[1]["hops"] == 1)
    ok("A5 cluster 从 meta 透传", items[0]["cluster"] == 7 and items[1]["cluster"] == 8)
    ok("A6 默认 cf=False", items[0]["cf"] is False)
    ok("A7 字符串 memory 也能读", T.load_synth_jsonl(_write([dict(
        _rec(), memory="single fact.")]))[0]["units"] == ["single fact."])

    # limit 生效且可复现
    p = _write([_rec(f"Q{i}?", f"A{i}.", (f"s{i}.",), cluster=i) for i in range(50)])
    a = T.load_synth_jsonl(p, limit=10, seed=3)
    b = T.load_synth_jsonl(p, limit=10, seed=3)
    c = T.load_synth_jsonl(p, limit=10, seed=4)
    os.unlink(p)
    ok("A8 limit 截断到指定条数", len(a) == 10)
    ok("A9 同 seed 抽样可复现", [x["question"] for x in a] == [x["question"] for x in b])
    ok("A10 不同 seed 抽样不同", [x["question"] for x in a] != [x["question"] for x in c])
    ok("A11 limit=0 表示全量", len(T.load_synth_jsonl(
        _write([_rec() for _ in range(5)]), limit=0)) == 5)


# ---------------------------------------------------------------------
# B. item_units
# ---------------------------------------------------------------------
def test_units():
    ok("B1 有 units 时按事实切槽",
       T.item_units({"units": ["a.", "b."], "memory": "a. b."}) == ["a.", "b."])
    ok("B2 无 units 时整段一槽（内部数据路径不变）",
       T.item_units({"memory": "a. b."}) == ["a. b."])
    ok("B3 units 为空列表时回退到 memory",
       T.item_units({"units": [], "memory": "m."}) == ["m."])
    ok("B4 全空时返回空串单槽（不抛异常）",
       T.item_units({}) == [""])


# ---------------------------------------------------------------------
# C. check_split_leakage
# ---------------------------------------------------------------------
def test_leakage():
    tr = [{"cluster": i, "question": f"q{i}"} for i in range(5)]
    ok("C1 簇不相交且问句不同 -> 无告警",
       T.check_split_leakage(tr, [{"cluster": i, "question": f"v{i}"}
                                 for i in range(5, 9)]) == [])
    probs = T.check_split_leakage(tr, [{"cluster": 3, "question": "v1"}])
    ok("C2 簇重叠被检出", any("簇重叠" in x for x in probs))
    probs = T.check_split_leakage(tr, [{"cluster": 99, "question": "q2"}])
    ok("C3 问句相同被检出", any("问句完全相同" in x for x in probs))
    ok("C4 cluster 全为 None 时不误报",
       T.check_split_leakage([{"cluster": None, "question": "a"}],
                             [{"cluster": None, "question": "b"}]) == [])


# ---------------------------------------------------------------------
# D. encode_item
# ---------------------------------------------------------------------
class _Enc:
    """假编码器：每个单元返回 [1,2,3] 的 payload，记录调用参数。"""

    def __init__(self):
        self.calls = []

    def encode(self, texts, parent_ids=None, grad=False):
        self.calls.append({"texts": list(texts),
                           "parents": list(parent_ids or []), "grad": grad})
        return ([torch.zeros(1, 2, 3) for _ in texts],
                {"n_units": len(texts)})


class _EmptyEnc:
    def encode(self, texts, parent_ids=None, grad=False):
        return [], {}


def test_encode_item():
    enc = _Enc()
    z, info = T.encode_item(enc, {"units": ["a.", "b.", "c."]}, tag="fact", grad=True)
    ok("D1 多单元沿块轴拼接 -> [Σk, m, d]", tuple(z.shape) == (3, 2, 3))
    ok("D2 parent_ids 按槽位编号", enc.calls[0]["parents"] == ["fact:0", "fact:1", "fact:2"])
    ok("D3 grad 标志透传（训练路径不能丢）", enc.calls[0]["grad"] is True)
    ok("D4 info 透传", info["n_units"] == 3)

    enc = _Enc()
    z, _ = T.encode_item(enc, {"units": ["only."]})
    ok("D5 单单元保持原形状（不额外 cat）", tuple(z.shape) == (1, 2, 3))
    ok("D6 默认 tag 前缀为 fact", enc.calls[0]["parents"] == ["fact:0"])

    enc = _Enc()
    T.encode_item(enc, {"memory": "m."}, tag="val")
    ok("D7 tag 可改（验证集用 val 前缀）", enc.calls[0]["parents"] == ["val:0"])

    try:
        T.encode_item(_EmptyEnc(), {"memory": "m."})
        raise AssertionError("FAIL D8 空 payload 应抛错")
    except RuntimeError:
        ok("D8 空 payload 抛 RuntimeError（不能静默产出空 soft tokens）", True)


# ---------------------------------------------------------------------
# E. CLI 契约
# ---------------------------------------------------------------------
def test_cli():
    a = T.parse_args([])
    ok("E1 --data 默认空（沿用内部模板）", a.data == "")
    ok("E2 --val-data 默认空", a.val_data == "")
    ok("E3 --n-train 默认 0 表示自动", a.n_train == 0 and a.n_val == 0)
    a = T.parse_args(["--data", "/tmp/x.jsonl", "--val-data", "/tmp/y.jsonl",
                      "--n-train", "100"])
    ok("E4 --data/--val-data/--n-train 可解析",
       a.data == "/tmp/x.jsonl" and a.val_data == "/tmp/y.jsonl" and a.n_train == 100)
    ok("E5 旧参数未受影响", T.parse_args([]).stage == "a" and T.parse_args([]).m == T.LM.LATENT_M)


def main():
    test_load()
    test_units()
    test_leakage()
    test_encode_item()
    test_cli()
    print(f"\n全部通过：{_N[0]} 条断言")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
