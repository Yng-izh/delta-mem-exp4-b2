# KNOWN_ISSUES.md — 已知问题登记

**最后更新**：2026-10-27（FST-Mem 阶段 B 轮）

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

### 3.5 FST-Mem 的效果完全未验证（与第 2 条并列，单独立项）

**位置**：`latent_memory.LatentFactKeyEncoder / LatentStateTrajectoryBank / FstRouter`、
`train_latent_memory.py --fst`

**现状**：三个机制**结构上都已做到**并有 88 项单测断言（键与值无关、改 A 不动 B 逐位相等、
两级竞争隔离、前向硬选 + STE、ckpt 严格往返、内容级对齐、指标口径、并列时间戳），
且 `tools/audit_mutations.py` 用 6 个变异确认这些断言**真有约束力**（核查轮之前有 3 条恒真，
让 4 个静默 bug 在 67 项断言下安然无恙，见 CHANGELOG v9.6.1）。但：
- **一步真实训练都没跑** → 没有任何 `acc / group_acc / ver_multi_acc` 的可用数字；
- `--fst --smoke-model` 的 smoke 指标是**随机权重下的噪声**，不能引用；
- 没有与"直接拼接潜槽位 / 阶段 A Router / 硬覆写"的对照跑分。

**影响**：任何"FST 解决了组合不稳定 / 更新局部性 / 时间一致性"的表述**目前都不成立**。
单测保证的是"实现符合设计"，不是"设计有效"。

**怎么修**：按 `ARCHITECTURE.md §6.5` 的消融矩阵在 v100-2 上依次跑
（`--router` → `--fst --hist-max 0` → `--fst --hist-max 2` → `--fst --ignore-query-time`），
每档至少 3 个 seed，报 `acc + group_acc + ver_multi_acc + 三探针 + pair_acc + order_stable`，
并且**必须同时报两个常数基线**（`group_const_acc` = 永远选第一组、`ver_newest_acc` = 永远选最新版），
否则"随机权重下 0.33"会被误读成有信号 —— 核查轮 smoke 里基线其实是 0.667。
另：评估集必须**按整组问题**截断（`fst_eval_subset`），按行截断会把反事实配对切两半，
`pair_n` 直接变 0。

### 3.6 FST 没有多跳、没有接进线上管线

**位置**：`generate_latent_answer_fst`、`run_ar_v7.py` / `run_group_b2_v7.py`

**现状**：
- `hops>1` **显式报错**（`ValueError`）。FST 的键是静态身份键，没有 `k_out` 那一路"客体值接力"，
  直接按单跳跑会得到"看起来正常的错答案"，所以选择报错而不是静默降级。
- AR/CR runner 仍走 `generate_latent_answer`（单级 Router），**没有任何代码路径会构造
  `LatentStateTrajectoryBank`**：需要从 `StatePool` / `EpisodicMemory` 提取
  `(实体, 属性, 时间)` 来建键与时间线，这是阶段 C 的工作。

**影响**：FST 目前只能通过 `train_latent_memory.py --fst` 训练与评估，**线上跑分数字无法产生**。
论文里"端到端对比"这一栏现在空白。

**怎么修**：见 `ARCHITECTURE.md §10` 阶段 C。

### 3.7 历史容量与保真度的权衡未量化

**位置**：`LATENT_HIST_MAX`（默认 4）

**现状**：超容量时 FIFO 淘汰最旧版本并**真正释放槽位**（不保留"读不到的僵尸槽"）。
但"保留几条历史够用""压缩后的历史丢了多少信息""历史变长后读出开销与定位精度如何退化"
**一个都没测**。

**影响**：`hist_max` 目前是拍的常数，不是实验结论。审稿人会问"历史很长怎么办"。

**已修的相关数据坑**：`hist_max=0` 时"目标版本被淘汰"与"数据缺时间戳"此前共用一个
`viable=False` 标志，导致无法区分"容量机制在起作用"和"数据有洞"。现已拆成
`evicted / label_missing / label_mismatch`，训练只跳过 `evicted`（日志 `skip_evicted`），
遇到 `label_mismatch`（答案不在库里）直接报错。

**怎么修**：造 `hist_max ∈ {0,1,2,4,8}` × 时间跨度 ∈ {3,6,12} 的网格，
报 `history` 探针准确率与 `n_versions` 的关系曲线。

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
