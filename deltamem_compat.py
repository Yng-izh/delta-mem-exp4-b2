#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""deltamem_compat.py — 让 `deltamem.core` 能在 transformers 4.51 上被 import

问题
----
`deltamem/core/backbone_compat.py` 在**模块顶层**无条件执行：

    from transformers.models.smollm3.modeling_smollm3 import (
        SmolLM3Attention, apply_rotary_pos_emb, eager_attention_forward)

而服务器 env（`/share/conda/envs/latent2`，transformers 4.51.3）**没有** smollm3 子包
（smollm3 是 transformers 5.x 才有的）。于是 `import deltamem.core` 直接 ModuleNotFoundError，
连带 `HFDeltaMemConfig` / `attach_delta_mem` / `load_delta_mem_adapter` 全部拿不到 ——
即使本实验根本不用 SmolLM3，只是加载 Qwen3-4B。

为什么不在 env 里塞一份 5.x 的 transformers
-------------------------------------------
`/share/exp4group/pylib5` 里确实有一份 5.19.0，但它是**不完整拷贝**（`pylib5/numpy`
缺编译扩展 `_multiarray_umath`），一旦进 PYTHONPATH 就先炸 numpy。改共享 env 或全局
PYTHONPATH 影响面太大，所以这里做**最小注入**：缺模块时往 `sys.modules` 塞一个占位模块，
只提供 deltamem 顶层 import 需要的那 3 个名字。

安全性
------
- 本仓库只用 Qwen3；`deltamem` 内部对 `SmolLM3Attention` 只有 `isinstance` 判断，
  对 `apply_rotary_pos_emb` / `eager_attention_forward` 没有调用点（已 grep 全仓确认）。
- 占位类不实现 forward；万一将来有代码真的用到它，会**显式报错**而不是静默算错。
- 真实模块存在时（transformers ≥5）**完全不介入**。
"""
from __future__ import annotations

import os
import sys
import types

_INJECTED = False


def ensure_smollm3_stub(verbose: bool = True) -> bool:
    """缺 `transformers.models.smollm3` 时注入占位模块。返回是否注入。"""
    global _INJECTED
    if _INJECTED:
        return True
    try:
        import transformers.models.smollm3  # noqa: F401
        return False                     # 真的有，不动
    except Exception:
        pass

    import torch.nn as nn

    class SmolLM3Attention(nn.Module):
        """占位：本实验不加载 SmolLM3，任何真实使用都应立刻失败。"""

        def __init__(self, *a, **kw):
            super().__init__()
            raise NotImplementedError(
                "SmolLM3Attention 是 deltamem_compat 注入的占位实现："
                "当前 transformers 无 smollm3 子包，本实验只用 Qwen3。")

    def _unavailable(name):
        def _f(*a, **kw):
            raise NotImplementedError(
                f"{name} 是 deltamem_compat 注入的占位实现，当前 transformers 无 smollm3 子包")
        return _f

    mod = types.ModuleType("transformers.models.smollm3.modeling_smollm3")
    mod.SmolLM3Attention = SmolLM3Attention
    mod.apply_rotary_pos_emb = _unavailable("apply_rotary_pos_emb")
    mod.eager_attention_forward = _unavailable("eager_attention_forward")

    pkg = types.ModuleType("transformers.models.smollm3")
    pkg.modeling_smollm3 = mod

    sys.modules["transformers.models.smollm3"] = pkg
    sys.modules["transformers.models.smollm3.modeling_smollm3"] = mod

    # 让 `transformers.models.smollm3` 这种属性访问也成立
    try:
        import transformers.models as _tm
        setattr(_tm, "smollm3", pkg)
    except Exception:
        pass

    _INJECTED = True
    if verbose:
        print("[compat] transformers 无 smollm3 子包，已注入占位模块"
              "（deltamem.core 顶层 import 需要；本实验只用 Qwen3，不涉及 SmolLM3）",
              flush=True)
    return True


def patch_kv_cache_alias() -> bool:
    """修 `DeltaMemAttention.forward` 的 KV Cache 参数名不兼容（返回是否改动）。

    transformers 4.51.3 的 `Qwen3DecoderLayer.forward` 传的是**单数**
    `past_key_value=past_key_value`（`modeling_qwen3.py:293`），而
    `DeltaMemAttention.forward` 声明的是**复数** `past_key_values=None`，
    且带 `**kwargs`。于是真正的 Cache 对象被静默吞进 `kwargs`：包装器内部
    `if past_key_values is not None` 永远为假 → 缓存永不更新 → 解码阶段每个
    新 token 都只能看到自己，历史上下文全丢。这类错误不抛异常，只表现为
    "开头还算正常、随后开始重复数字"的退化生成。

    补丁只做参数名归一，不改注意力计算、不动 adapter 权重、不影响 checkpoint。
    新旧 transformers 都兼容（谁传哪个都行）。
    """
    if os.environ.get("DELTA_NO_KV_PATCH", "") not in ("", "0"):
        return False
    ensure_smollm3_stub(verbose=False)
    from deltamem.core.delta_impl import DeltaMemAttention

    if getattr(DeltaMemAttention, "_cache_alias_fixed", False):
        return False

    original_forward = DeltaMemAttention.forward

    def forward_compat(self, *args, past_key_value=None, past_key_values=None, **kwargs):
        if past_key_value is not None and past_key_values is not None:
            if past_key_value is not past_key_values:
                raise ValueError("Conflicting KV cache arguments")
        cache = past_key_values if past_key_values is not None else past_key_value
        return original_forward(self, *args, past_key_values=cache, **kwargs)

    forward_compat.__wrapped__ = original_forward
    DeltaMemAttention.forward = forward_compat
    DeltaMemAttention._cache_alias_fixed = True
    return True


def import_delta_mem():
    """import deltamem 的四个入口，必要时先注入 smollm3 占位模块。"""
    ensure_smollm3_stub()
    patch_kv_cache_alias()
    from deltamem.core import HFDeltaMemConfig, attach_delta_mem
    from deltamem.core.delta_impl import load_delta_mem_adapter
    return HFDeltaMemConfig, attach_delta_mem, load_delta_mem_adapter


if __name__ == "__main__":
    injected = ensure_smollm3_stub()
    HFDeltaMemConfig, attach_delta_mem, load_delta_mem_adapter = import_delta_mem()
    print(f"[ok] injected_stub={injected} HFDeltaMemConfig={HFDeltaMemConfig.__name__}")
