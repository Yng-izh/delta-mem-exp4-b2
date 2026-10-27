#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tests/test_train_objectives.py — 训练目标组合 + Delta 写入开关的**状态恢复**。

这两条都是"改错了不会报错、只会让结果静默失真"的地方，因此单独成文件：

  P. `assemble_loss`：KL / 寻址 / 重建只能**累加**到 QA 上。历史 bug 是 KL 分支用
     `loss = loss_qa + kl_weight*loss_kl` 直接赋值，把 `addr_weight*loss_addr`
     整条覆盖掉 —— 于是"开了 KL 之后寻址再没被监督过"，而日志里 loss_addr 照常打印。
  Q. `delta_write_disabled`：退出时必须**恢复调用前的状态**。历史 bug 是 finally 里
     一律 `fn(model, False)`，第一次进上下文之后写入被永久关闭：后续 chunk 写不进在线
     状态，跑分照样出数、只是全错，日志毫无痕迹。

用法：python tests/test_train_objectives.py
"""
import contextlib
import os
import random
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import torch

import latent_memory as LM
import train_latent_memory as T

N_OK = 0
FAILS = []


def ok(name, cond, extra=""):
    global N_OK
    if cond:
        N_OK += 1
    else:
        FAILS.append(name)
        print(f"[FAIL] {name} {extra}")
    return bool(cond)


def tens(x):
    return torch.tensor(float(x))


# =====================================================================
# P. 损失组合
# =====================================================================
def test_assemble_loss():
    qa, addr, kl, rec = tens(2.0), tens(1.0), tens(0.5), tens(4.0)
    aw, kw, rw = 1.0, 0.3, 0.1

    # P1 只有 QA
    ok("P1 只给 QA", float(T.assemble_loss(qa, None, None, None, aw, kw, rw)) == 2.0)

    # P2 QA + 寻址
    ok("P2 QA+寻址累加",
       abs(float(T.assemble_loss(qa, addr, None, None, aw, kw, rw)) - (2.0 + 1.0)) < 1e-6)

    # P3 **回归**：开 KL 后寻址项不能被覆盖
    v = float(T.assemble_loss(qa, addr, kl, None, aw, kw, rw))
    ok("P3 开 KL 时寻址项仍在（旧写法会覆盖）",
       abs(v - (2.0 + 1.0 + 0.3 * 0.5)) < 1e-6, extra=f"got {v}")

    # P4 三项 + 重建全累加
    v = float(T.assemble_loss(qa, addr, kl, rec, aw, kw, rw))
    want = 2.0 + 1.0 + 0.3 * 0.5 + 0.1 * 4.0
    ok("P4 四项全累加", abs(v - want) < 1e-6, extra=f"got {v} want {want}")

    # P5 权重为 0 时等于关掉该项
    v = float(T.assemble_loss(qa, addr, kl, rec, 0.0, 0.0, 0.0))
    ok("P5 权重 0 = 关闭", abs(v - 2.0) < 1e-6, extra=f"got {v}")

    # P6 两阶段：纯重建 + 寻址（故意丢 QA 与 KL）
    v = float(T.assemble_loss(qa, addr, kl, rec, aw, kw, rw, recon_stage=True))
    ok("P6 recon_stage 只保留重建+寻址", abs(v - (4.0 + 1.0)) < 1e-6, extra=f"got {v}")

    # P7 recon_stage 但没算重建 -> 报错（不静默退化）
    try:
        T.assemble_loss(qa, addr, kl, None, aw, kw, rw, recon_stage=True)
        raised = False
    except ValueError:
        raised = True
    ok("P7 recon_stage 缺 loss_r 报错", raised)

    # P8 梯度真的能同时流到寻址头与 QA（不是只"数值上加对了"）
    router_w = torch.zeros(2, requires_grad=True)
    addr_t = torch.nn.functional.cross_entropy(router_w.unsqueeze(0),
                                               torch.tensor([1]))
    total = T.assemble_loss(tens(1.0), addr_t, tens(0.2), None, 1.0, 0.5, 0.0)
    total.backward()
    ok("P8 反传后寻址参数有梯度", router_w.grad is not None
       and float(router_w.grad.abs().sum()) > 0)


# =====================================================================
# Q. Delta 写入开关的状态恢复
# =====================================================================
class _FakeMod:
    def __init__(self, enabled):
        self.write_enabled = bool(enabled)
        self.read_context_mask = "mask"
        self.write_message_ids = "mid"


def _install_fake_deltamem(mods):
    """把 deltamem.core.delta 塞进 sys.modules，返回 (restore_fn)。"""
    pkg = types.ModuleType("deltamem")
    core = types.ModuleType("deltamem.core")
    delta = types.ModuleType("deltamem.core.delta")

    def iter_delta_mem_modules(model):
        return [("layer%d" % i, m) for i, m in enumerate(getattr(model, "mods", []))]

    def set_delta_mem_write_enabled(model, enabled):
        for m in getattr(model, "mods", []):
            m.write_enabled = bool(enabled)

    delta.iter_delta_mem_modules = iter_delta_mem_modules
    delta.set_delta_mem_write_enabled = set_delta_mem_write_enabled
    core.delta = delta
    pkg.core = core
    saved = {k: sys.modules.get(k) for k in ("deltamem", "deltamem.core",
                                             "deltamem.core.delta")}
    sys.modules.update({"deltamem": pkg, "deltamem.core": core,
                        "deltamem.core.delta": delta})

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return restore


def test_delta_write_restore():
    m1, m2 = _FakeMod(True), _FakeMod(False)
    model = types.SimpleNamespace(mods=[m1, m2])
    restore = _install_fake_deltamem([m1, m2])
    try:
        with LM.delta_write_disabled(model) as active:
            ok("Q1 上下文内写入被关闭", active is True and m1.write_enabled is False
               and m2.write_enabled is False)
        # Q2 回归：退出后必须恢复原值（m1 原本 True、m2 原本 False）
        ok("Q2 退出后恢复**各自**原状态（不是一律置 False）",
           m1.write_enabled is True and m2.write_enabled is False,
           extra=f"m1={m1.write_enabled} m2={m2.write_enabled}")

        # Q3 旁路状态不许被清掉（set_write_enabled(True) 会清 read_context_mask）
        ok("Q3 旁路状态（read_context_mask/write_message_ids）未被清空",
           m1.read_context_mask == "mask" and m1.write_message_ids == "mid")

        # Q4 连续两次进入，第二次仍应能恢复
        with LM.delta_write_disabled(model):
            pass
        ok("Q4 可重复进入且每次都恢复", m1.write_enabled is True)

        # Q5 嵌套：内层退出后外层仍然处于"关闭"状态
        with LM.delta_write_disabled(model):
            with LM.delta_write_disabled(model):
                pass
            inner_after = m1.write_enabled
        ok("Q5 嵌套退出后恢复到外层状态", inner_after is False and m1.write_enabled is True,
           extra=f"inner_after={inner_after} final={m1.write_enabled}")

        # Q6 异常路径同样恢复
        try:
            with LM.delta_write_disabled(model):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        ok("Q6 异常退出后同样恢复", m1.write_enabled is True)
    finally:
        restore()


def test_slot_curriculum():
    """R1–R5：变槽数课程的三个不变量（写错不会报错，只会让 Router 停在 chance）。"""
    payloads = [torch.zeros(1, 4, 8) for _ in range(23)]
    for i, p in enumerate(payloads):
        p.fill_(float(i))                       # 用填充值当"身份"，便于核对保留下来的槽

    rng = random.Random(0)
    seen_sizes = set()
    for _ in range(200):
        slot_z, new_tgt, keep = T.sample_slot_subset(
            payloads, 7, rng, max_slots=8, must_keep=[7, 11])
        seen_sizes.add(len(keep))
        if not ({7, 11} <= set(keep)):
            break
        if float(slot_z[new_tgt][0, 0]) != 7.0:
            break
    ok("R1 must_keep 与目标槽始终保留、且 new_tgt 指向目标槽",
        {7, 11} <= set(keep) and float(slot_z[new_tgt][0, 0]) == 7.0)
    ok("R2 变槽数课程真正生效（样本内槽数 < 总槽数，且不止一种取值）",
       max(seen_sizes) <= 8 and len(seen_sizes) > 1, extra=f"sizes={sorted(seen_sizes)}")
    ok("R3 抽到的槽位按需截断到 max_slots（多一点：must_keep 优先）",
       max(seen_sizes) <= 8)

    # R4：没有 must_keep 时也必含目标
    rng2 = random.Random(1)
    for _ in range(50):
        slot_z, new_tgt, keep = T.sample_slot_subset(payloads, 3, rng2, max_slots=4)
        assert 3 in keep and float(slot_z[new_tgt][0, 0]) == 3.0

    # R5：keep_all 仍然全留（兼容旧行为）
    slot_z, new_tgt, keep = T.sample_slot_subset(payloads, 5, random.Random(2),
                                                 keep_all=True)
    ok("R5 keep_all=True 仍保留全部 23 槽", len(keep) == 23)

    # R6：证据下标解析（多跳链的每一跳）
    units = ["A one.", "B two.", "C three."]
    idx = T.evidence_slot_indices(units, ["C three.", "A one."])
    ok("R6 evidence_slot_indices 返回全部命中的跳", sorted(idx) == [0, 2], extra=str(idx))
    ok("R7 无 evidence 时返回空（不猜）", T.evidence_slot_indices(units, []) == [])


def main():
    test_assemble_loss()
    test_delta_write_restore()
    test_slot_curriculum()
    print("=" * 70)
    if FAILS:
        print(f"FAILED {len(FAILS)}/{N_OK + len(FAILS)}: {FAILS}")
        return 1
    print(f"ALL PASS ({N_OK} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
