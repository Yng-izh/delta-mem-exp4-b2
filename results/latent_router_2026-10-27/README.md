# 潜层读出重构：17% 之谜定位 → LatentRouter 内容寻址（2026-10-27，v100-2 / 本地核查）

**结论先说**：

1. **"训练 17%、重载 0%" 不是产品 bug，是我自己诊断脚本的 4 个 bug**。用干净进程重测后，
   训练期与重载后的数字**完全一致**，checkpoint 存取没有问题。
2. 真正的失败原因被锁定：**读出完全由"槽位排在软 token 序列的第几位"决定，与内容无关**。
   同一事实单槽 15/16 → 4 槽 target_only 仍 15/16 → 4 槽 full 1/16 → 4 槽 target_end **0/16**。
3. 已按"键值解耦 + 集合内容寻址"重构（`LatentRouter`），但**尚未训练、不知道能不能救回来**。
   本轮只做到"代码改完 + 机制验证通过"，任何效果结论都还没有。
4. 重构后**两轮自查又发现 7 个会静默失效的问题**（1 个是我自己的设计错误），已全部修复并附实测证据。

---

## 一、17% 之谜：4 个自伤脚本 bug（无产品缺陷）

现象：`train_latent_memory.py` 训练结束时自报 17.2%，但用同一 checkpoint 重载测试得 0%。

| # | Bug | 后果 |
|---|---|---|
| 1 | 诊断脚本硬编码 system prompt `"You are a memory agent..."`，而训练默认是 `"You are a helpful assistant. Answer with the shortest exact value."` | 输入分布不同，指标不可比 |
| 2 | 诊断脚本未关闭 Delta 写入。训练进程因 `delta_write_disabled` 的 `finally: fn(model, False)` 被**永久关闭**写入（干净），诊断进程则逐条累积 `delta_state` | 前几次看着正常，之后整体崩到 0 |
| 3 | `generate(inputs_embeds=...)` 返回序列**不含输入前缀**，脚本却做 `out[0, e.shape[1]:]` 切片 | 解出空串 → 假 0/32 |
| 4 | `load_synth_jsonl` 把 `memory` 变成**拼接后的字符串**，脚本却用 `it["memory"][0]` 取"第一条事实" | 实际取到第一个**字符**，造出 `["T","M","S"]` 这种畸形槽位 |

已排除的假说（直接测量，全部无关）：delta_state 污染最终分数、`comp/bridge` 的 train-vs-eval 模式、
`torch_dtype=` vs `dtype=`、`attn_implementation="sdpa"` vs `"eager"`、adapter 路径默认值、
`load_delta_mem_adapter` 位置参数 vs 关键字、forward 是否改写权重（722 个参数张量哈希前后一致）、
`load_state_dict` 严格性（无缺失键）。

### 干净重测（同一进程内不累积状态：训练同款 prompt + Delta 写入关闭 + 每条 reset）

| ckpt | 条件 | 严格全匹配 | pair-acc |
|---|---|---|---|
| 1 事实 | min16 单槽 | **15/16 (93.8%)** | 7/8 |
| 1 事实 | 4 槽 target_only（内容完全相同，仅外壳是 4 槽） | **15/16 (93.8%)** | — |
| 1 事实 | 4 槽 full（+3 干扰，目标在首位） | **1/16 (6.2%)** | — |
| 1 事实 | 4 槽 target_end（目标在末位） | **0/16 (0%)** | — |
| 4 事实 | 训练集 full | 79/200 (39.5%) | 16/100 |
| 4 事实 | 验证集 full | **11/64 (17.2%)** | **0/32** |
| 4 事实 | 训练集 target_only | **0/200 (0%)** | 0/100 |
| 4 事实 | 验证集 target_only | **0/64 (0%)** | 0/32 |
| 4 事实 | 训练集 target_end | 43/200 (21.5%) | 1/100 |
| 4 事实 | 验证集 target_end | 14/64 (21.9%) | 0/32 |
| — | 明文 4 槽读出（32 条） | 24/32 (75%) | — |
| — | Oracle（原文 embedding）32 条 | 17/32 (53%) | — |

### 三条判据

1. **target_only 是最强的诊断**：给 4 槽模型只喂**正确那一条**，它反而 0/200 全错。
   如果它真在按内容寻址，这是最容易的情形。→ 它学的是"4 槽这个形状"，不是 query→slot。
2. **1 槽 ckpt 套上 4 槽外壳就崩**（15/16 → 1/16），内容一模一样 → 纯位置效应。
3. **17.2% 低于 31.2% 的 relation-prior 基线**（数据只有 6 种关系、18 个答案值，按关系猜值就能到 31.2%）
   → 17.2% 是噪声，不是能力。

瓶颈在 **Compressor + Bridge**，不在 Reader：明文 75%、Oracle 53% 都远高于 17.2%。

---

## 二、重构做了什么

| 组件 | 实现 |
|---|---|
| `latent_memory.LatentRouter` | `s_k = <q, k_in_k>/τ` 纯内积集合打分；**推理 argmax 只送 1 个槽**（m 个软 token）；`chain()` 用 `k_out` 接力做多跳 |
| 键值解耦 | `k_in`（主语+关系，寻址用）、`k_out`（客体值，多跳用），各自 MLP + L2 归一化 |
| `extract_query_feature` | 问题 → `[L0 mean ‖ L18 mean]`，词形与语境各一半 |
| `stack_payloads` | per-slot `[k,m,d]` → `[K,m,d]`，块轴取均值保证键稳定 |
| `generate_latent_answer` | 新增 `router/q_feat/layer/hops`；多跳槽位间插 Qwen 原生 `\n` embedding；meta 加 `routed/hops/n_slots_in` |
| save/load | `config.has_router` + `d_key`；加载时缺路由权重**直接报错**；老 ckpt 兼容 |
| 训练侧 | `--router / --d-key / --addr-weight / --freeze-compressor / --max-slots`；`sample_slot_subset` 变槽数+随机序；`loss_addr` 显式寻址 CE |

---

## 三、两轮自查发现的 7 个问题（都附着实测证据）

### 第一轮（提交 `60421d8` 之后自查）

| # | 问题 | 证据 | 严重度 |
|---|---|---|---|
| 1 | **软门控聚合做训练前向，与推理硬选不同分布**（我自己的设计错误） | 4 槽 τ=0.07：`‖z_soft‖=61.9` vs `‖z_hard‖=100.2`，`‖z_soft-z_hard‖` = 54.6%，`w=[0.18,0.05,0.23,0.54]` 远非 one-hot | 🔴 |
| 2 | `generate_latent_answer` 里**原有的** `torch.as_tensor(payloads)` 短路新加的 stack 分支 | 多块列表下 `ValueError: only one element tensors can be converted to scalars` | 🔴 |
| 3 | **槽边界丢失**：`encode_item` 返回 cat 后的 `[Σk,m,d]`，Σk>槽数时把"块"当"槽" | 4 槽其中 1 槽切 2 块 → `cat=(5,4,2560)` 被当成 5 个槽；`stack_payloads=(4,4,2560)` 才对 | 🔴 |
| 4 | **评测管线不传 router**：`run_ar_v7:2059` / `run_group_b2_v7:665` | Phase 2 的 ckpt 会退回 K*m 裸拼，位置敏感老问题原样复现且看不出原因 | 🔴 |
| 5 | `evaluate_recon` 不走路由 | 训练 recon 用路由 soft（4 token），评估用裸 `bridge(z)`（16 token） | 🟠 |
| 6 | 多跳样本被随机丢槽会剪断推理链 | 现有数据全 `hops=1`，此路径未被真实数据覆盖 | 🟠 |

### 第二轮（查数据集时发现）

| # | 问题 | 证据 | 严重度 |
|---|---|---|---|
| 7 | **寻址监督 13.5% 标签是错的**：`target_slot_index` 只用答案串匹配，而 22.5% 的样本答案值在干扰事实里也出现 | 见下 | 🔴 |

问题 7 实例（`latent_exp2_train`，答案 `Fenwick`）：

```
真值位置 2 错标成 0
units = ['Quinent founded Fenwick.',   ← 命中这条（错）
         'Veyost founded Ondrey.',
         'Quinent works at Fenwick.',  ← 真正的目标
         'Veyira founded Nivora.']
```

根因在数据侧：`load_synth_jsonl` 把 `evidence`（目标事实原文）丢了，唯一可靠的定位依据消失。
`loss_addr` 因此把 router 往错方向训，而**训练损失照样下降**，从曲线上完全看不出。

修法：保留 `evidence`；`target_slot_index` 优先用它精确匹配，退化才用答案串。
**标签错误 200→0、64→0。**

### 问题 1 的修法（直通估计）

```python
z_read = z_hard + (z_soft - z_soft.detach())
```

前向与推理**逐位相同**（已断言 `torch.equal(z_train, z_hard)` = True），梯度仍回传 logits。
`eval` 模式自动走纯硬选。

---

## 四、验证证据（本轮日志，均在 `results/` 目录内）

| 日志 | 验证内容 | 关键数字 |
|---|---|---|
| `verify_router.log` | 推理路径 + ckpt 往返 + 老 ckpt 兼容 | `slot_z=(4,4,2560)`、`logits=(1,4)`、`chain(hops=2)=[0,1]`、`n_soft=4`（hops=1）/`9`（hops=2，含换行）、无 router 回归 `n_soft=16`、`has_router=True→d_key=256`、老 ckpt `router=None` |
| `smoke_router.log` | min16 单槽 smoke | **暴露** `loss_addr=0.0000`、`gnorm_router=0`（K=1 时 CE 恒为 0，符合预期） |
| `smoke_router4.log` | 4 槽 router 训练 | `loss_addr` 0.68→1.33→0.67、`gnorm_router` 6.1→0.53（梯度确实存在） |
| `smoke_phase2.log` | Phase 2 冻结压缩器 | `gnorm_comp=0.000e+00`（冻结生效）；init ckpt 在 router 随机初始化下仍 3/4（未破坏已有通路） |
| `smoke_chunk.log` | `--max-unit-tokens 8` 强制多块 | **暴露问题 2/3**；修复后 `soft_tokens=4`（槽边界保住） |
| `smoke_final.log` | 端到端（多块+router+recon+eval） | `[params] 14.375M (compressor=4.994M, bridge=2.630M, router=6.752M)`、`soft_tokens=4`、eval-reload/after-reset 均走含 router 的重载 reader |
| `smoke_evidence.log` | evidence 修复后 25 步 | `loss_addr` 0.66~1.62 波动、`gnorm_router` 非零、`recon token-acc=0.241` |

单元测试：`tests/test_latent_readout.py` 由 86 项扩到 **89 项全过**
（A1 白名单纳入 `router/q_feat/layer/hops` 并保留明文参数禁令；
新增 A8 有 router 时 `n_soft=m`、A9 `routed` 下标、A10 `n_slots_in` 保持输入槽数）。

### 可复跑的脚本（一并入库）

| 脚本 | 作用 |
|---|---|
| `diagnose_clean.py` | **产出第一节干净诊断表的脚本**。内部修掉了 4 个自伤 bug：从源码提取 system prompt、加载后一次性 `set_delta_mem_write_enabled(False)`、每条 `reset_delta_mem_states`、`decode(out[0])` 不做前缀切片 |
| `diagnose_isolated_process.py` | 独立进程单跑 C 段，复核"连续评估导致的状态污染" |
| `audit_soft_hard.py` | 量化软门控聚合 vs 硬选的分布差异（产出 61.9 vs 100.2、54.6%、w 分布）+ 槽边界丢失的构造 |
| `audit_ste.py` | 验证直通估计：`torch.equal(z_train, z_hard)` 必须为 True |
| `verify_router.py` | 推理路径 / 多跳 chain / ckpt 往返 / 老 ckpt 兼容 |

> 这些脚本硬编码了服务器路径（`/share/exp4group`、`/share/hf_home`），需在 v100-2 上运行。
> 运行前必须带完整环境变量（见 [`KNOWN_ISSUES.md`](../../KNOWN_ISSUES.md) 第 7 条）。

---

## 五、明确没有做的事

- **没有训练**。Phase 1/Phase 2 一步都没跑，所以"路由能不能救回 4 槽"**目前没有任何证据**。
- **没有验证多跳**。`chain()` 只有形状/流程验证，没有任何真实多跳数据（三套数据全 `hops=1`）。
- **没有扩容 bottleneck**（512 未动）。单槽已 93.8%，瓶颈不是容量而是寻址；扩容会破坏 ckpt 兼容。
- **没有改评测管线那 3 处已知修复**（`v8_slotpool` 排序截断、`cur_ts` 时间戳、软 token 预算统计）。

## 六、下一步（阻塞项）

1. **缺单槽拆分数据** → Phase 1 无法按方案开跑（现只能拿 4 槽数据或 16 条 min16 凑）。需 `tools/explode_units.py`。
2. **缺外部 benchmark** → `MemoryAgentBench` / `LongMemEval` / `EventQA` 原始数据一套都没有，实验 2 无数据可跑。
3. **`train_15k` 的 2998 条多跳从未进过潜层管线**，多跳路径欠验证。

详见根目录 [`DATASETS.md`](../../DATASETS.md) 与 [`KNOWN_ISSUES.md`](../../KNOWN_ISSUES.md)。
