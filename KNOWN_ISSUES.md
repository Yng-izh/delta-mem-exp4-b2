# KNOWN_ISSUES.md — 已知问题登记

**最后更新**：2026-10-27（潜层 Router 重构轮）

只登记**确实存在、已确认、但尚未修复**的问题。每条都写明「怎么发现的 / 影响什么 / 怎么修」。
已修的问题记在 [`CHANGELOG.md`](CHANGELOG.md) 对应版本里。

---

## P0 — 会影响实验结论的正确性

### 1. `delta_write_disabled` 会永久关闭 Delta 写入

**位置**：`latent_memory.py`，`delta_write_disabled` contextmanager

```python
finally:
    fn(model, False)      # ← 本意是"恢复原值"，实际是"强制关闭"
```

**现象**：任何一次 `with delta_write_disabled(model)` 退出后，该**进程内**模型的 Delta 写入永久为关，
即使原来是开的。

**怎么发现的**：追查"训练 17%、重载 0%"时，发现训练进程因反复进入该 contextmanager 而被永久关闭写入
（所以训练期数字是干净的），诊断进程没有反复进入、于是逐步累积 `delta_state` 而崩溃。

**影响**：
- 目前潜层读出只走软 token，不依赖 Delta 写入，**暂时无害**；
- 但只要同一进程里既有"需要写"（AR 管线在线写入）又有"需要禁写"（潜层编码）的路径，
  后者的第一次调用就会悄悄废掉前者的全部能力，且**没有任何日志或异常**。

**修法**：进入时记录原值，`finally` 恢复原值。

```python
prev = fn(model, None) if fn supports else current_state
try:
    yield
finally:
    fn(model, prev)
```

需要先确认 Delta 实现是否有"读当前状态"的接口（`delta_impl.py` 里 `set_delta_mem_write_enabled` 只写不读）。

**为什么还没修**：一改会改变**所有既有实验**的行为基线，必须单独一轮回归，不能夹在重构里顺手改。

---

### 2. Router 的重构效果完全未验证

**现象**：`LatentRouter` 只做了机制验证（形状、梯度、ckpt 往返、单测 89 项全过），
**一步训练都没跑**。

**影响**：不能声称"重构解决了位置敏感"。4 槽 target_end 从 0/16 能不能救回来，现在**没有任何证据**。

**修法**：按 `results/latent_router_2026-10-27/README.md` 第六节的阻塞项，先补单槽数据 → Phase 1 → Phase 2。

---

### 3. 数据歧义：22.5% 的样本答案值在干扰事实里重复

**位置**：`tools/gen_latent_exp2.py`，`make_split()` 生成干扰事实时未排除与答案同值

| 数据集 | 歧义样本 | 占比 |
|---|---|---|
| `latent_exp2_train` | 45/200 | 22.5% |
| `latent_exp2_val` | 11/64 | 17.2% |

**影响**：模型读错槽也能恰好答对 → 准确率**被高估**；这类样本也**无法检验内容寻址是否 work**。

**修法**（二选一，待定）：
- 改生成器排除同值干扰，重新出数据（干净，但旧 ckpt 全部作废）
- 保留数据，评估时额外报一列「无歧义子集」准确率（不破坏已有结果）

---

## P1 — 会让某些路径静默失效

### 4. 多跳路径（`LatentRouter.chain`）从未在真实数据上跑过

**现象**：`chain()` 只有形状验证（`hops=2 → picked=[0,1]`）。
`generate_latent_answer(hops=2)` 的换行隔离也只在随机权重下验证过不崩。

**影响**：三套在用数据全是 `hops=1`。`train_15k` 里虽有 2998 条多跳，但**从未进过潜层管线**。
两跳题能不能做，完全未知。

**修法**：把 `train_15k` 的多跳子集接进 `load_synth_jsonl` + Phase 2，评估时按 `item["hops"]` 走链。
（`sample_slot_subset(keep_all=True)` 与 `evaluate(hops=...)` 都已就位，缺的是数据接入与验证。）

---

### 5. `stack_payloads` 对一个槽位的多块取均值

**位置**：`latent_memory.py`，`stack_payloads()`

**现象**：一个记忆单元被切成 `k` 块时，`stack_payloads` 对块轴取 `mean`，把 `k` 块压成一个槽位描述。

**影响**：当前数据事实 ≤9 token ≪ `max_unit_tokens=64`，永远 `k=1`，**不触发**。
但换成长事实后，路由的键是"多块均值"，与 Phase 1 逐块编码的语义有偏差。

**修法**：给 Router 增加块级注意力（对 `k` 轴做 attention pooling 而非 mean），或限制"一槽一块"并显式报错。

---

### 6. 评测管线 3 处已知修复未落地

| 位置 | 问题 |
|---|---|
| `v8_slotpool.py` | top-K 后按 `orig_idx` 重排的顺序 bug |
| `v8_slotpool.py` | `cur_ts` 单句时间戳冻结 |
| 软 token 预算统计 | 预算应按真实 `num_chunks * 4` 算，而非配置值 |

**影响**：不在潜层主链路上，但会影响 AR 管线的检索与预算口径。

---

## P2 — 环境与流程

### 7. 服务器 Python 环境被其他用户破坏

editable 安装的 `transformers` 指向不存在的 `/root/transformers-main`。所有运行必须带：

```bash
CUDA_VISIBLE_DEVICES=7 \
HF_HOME=/share/hf_home HF_HUB_OFFLINE=1 \
DELTA_MEM_SCAN_IMPL=torch \
PYTHONPATH=/share/exp4group/shim19:/share/exp4group:/share/delta-Mem \
/share/conda/envs/latent2/bin/python <script>
```

**影响**：不知道这一点会得到 `ModuleNotFoundError` 或 import 到错误版本，误判成代码 bug。

### 8. 机器共享，可能被清空

`/share/exp4group` 是普通 scp 同步目录，无版本控制、无备份。**checkpoint 不入库**（30 MB/个），
配置记在各日志 `[save]` 行。GPU 7 约 3.3 GB 基线占用。长跑每 200 步必须落盘。

### 9. SSH 到 v100-2 偶发超时

`ssh: connect to host 100.94.9.44 port 22: Operation timed out`（exit 255）。
用 `-o ConnectTimeout=25` 重试循环可恢复。

---

## 登记规范

新增一条前先确认：

1. **能复现**（给出具体命令或最小代码片段）
2. **说清影响**（哪些实验/结论会被它污染）
3. **给出修法**（哪怕只是方向）
4. **标注优先级**：`P0` 影响结论正确性 / `P1` 让某路径静默失效 / `P2` 环境与流程
