#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/audit_mutations.py — 用"故意改坏"来验证单测真的有约束力。

背景（这轮核查真实踩到的）：`tests/test_fst_mem.py` 里有三条断言恒真 ——
`fact_key(x,y) == fact_key(x,y)`、`ver_z.grad is None`（ver_z 本来就不需要梯度）、
"秘密值不在 tokenizer 文本里"（秘密值从来没进过任何文本）。它们**永远通过**，
于是 `gather_all_versions` 顺序反转、"读出漏传 ver_mask"、STE 不 detach 载荷
这三个真 bug 全都能在 67 项断言下安然无恙。

这个脚本对每个"应该被抓住的错误"做一次最小变异，然后跑测试文件，
断言**测试必须失败**。变异存活 = 测试没有约束力。

用法：
    python3 tools/audit_mutations.py            # 全部
    python3 tools/audit_mutations.py --only mask
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (名字, 文件, 原始代码片段, 变异后片段, 期望捕获方式说明)
MUTATIONS = [
    ("ste_no_detach", "latent_memory.py",
     '        z = z_all.to(device=wg.device, dtype=wg.dtype).flatten(-2)   # [G,V,m*d]\n'
     '        z = z.detach()\n',
     '        z = z_all.to(device=wg.device, dtype=wg.dtype).flatten(-2)   # [G,V,m*d]\n',
     "STE 软期望不再 detach 载荷 -> 载荷被铺上 G×V 混合梯度"),
    ("gather_reversed", "latent_memory.py",
     '''        return [[self.payload(v["slot_id"]) for v in self.list_versions(k)] for k in ks]''',
     '''        return [[self.payload(v["slot_id"]) for v in self.list_versions(k)] for k in ks][::-1]''',
     "组序反转 -> 读到另一条属性的载荷"),
    ("mask_ignored", "latent_memory.py",
     "            lg_ver_all = lg_ver_all.masked_fill(~mask, -1e4)",
     "            pass  # 突变：不掩码填充位",
     "填充版本不再掩码 -> 零向量可能进 Reader"),
    ("key_text_from_key", "latent_memory.py",
     "    ents_rels = [bank.group_text(k) for k in keys]",
     "    ents_rels = [split_fact_key(k) for k in keys]",
     "读出退回从(小写)键反推实体文本 -> 训练/读出键特征不同源"),
    ("ver_metric_unbound", "train_latent_memory.py",
     '''    v_ok = [int(bool(r["g_ok"]) and int(r["viable"])''',
     '''    v_ok = [int(int(r["viable"])''',
     "版本命中不绑组别 -> 组选错也可能记命中"),
    ("tgt_ver_first_match", "train_latent_memory.py",
     '''    for vi, v in enumerate(vers):
        if LM._time_match_score(want_t, v["time"]) == 2:
            out["tgt_ver"] = vi
            out["viable"] = True''',
     '''    for vi, v in enumerate(vers):
        if LM._time_match_score(want_t, v["time"]) == 2:
            out["tgt_ver"] = vi
            out["viable"] = True
            break''',
     "并列时间戳取最旧（与 bank.read 的并列规则不一致）"),
]


def run_one(name, fname, old, new, note):
    tmp = tempfile.mkdtemp(prefix="mut_")
    try:
        dst = os.path.join(tmp, "repo")
        shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(
            ".git", "__pycache__", "runs", "*.pt", "*.log"))
        path = os.path.join(dst, fname)
        src = open(path, encoding="utf-8").read()
        if src.count(old) != 1:
            return name, "SKIP", f"{fname} 里片段出现 {src.count(old)} 次（期望 1）：{old[:40]!r}"
        open(path, "w", encoding="utf-8").write(src.replace(old, new))
        env = dict(os.environ, DELTA_MEM_SCAN_IMPL="torch", PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, os.path.join(dst, "tests", "test_fst_mem.py")],
                           cwd=dst, env=env, capture_output=True, text=True)
        tail = (r.stdout or "").strip().splitlines()
        last = tail[-1] if tail else (r.stderr or "").strip().splitlines()[-1:]
        caught = r.returncode != 0
        return name, ("CAUGHT" if caught else "SURVIVED"), f"{last if isinstance(last, str) else last}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只跑名字里含该子串的变异")
    args = ap.parse_args()
    rows = []
    for name, fname, old, new, note in MUTATIONS:
        if args.only and args.only not in name:
            continue
        rows.append((name,) + run_one(name, fname, old, new, note)[1:] + (note,))
    print("=" * 78)
    bad = 0
    for name, status, detail, note in rows:
        mark = {"CAUGHT": "✓", "SURVIVED": "✗", "SKIP": "-"}[status]
        print(f"{mark} {name:22s} {status:9s} {note}")
        print(f"    {detail}")
        bad += int(status == "SURVIVED")
    print("=" * 78)
    if bad:
        print(f"FAILED：{bad} 个变异存活 —— 对应断言没有约束力，必须补测")
        return 1
    print(f"OK：{len(rows)} 个变异全部被单测抓住")
    return 0


if __name__ == "__main__":
    sys.exit(main())
