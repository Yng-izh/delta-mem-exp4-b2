# 潜层记忆实验：HotpotQA / MemoryAgentBench 上的四方法对照（Text-RAG / Bridge / Router+Bridge / FST-Mem）

基于 delta-mem（declare-lab/delta-mem_qwen3_4b-instruct）+ MemoryAgentBench（Conflict_Resolution 切片）的对照实验，验证"流式事实槽位原位覆写（B2）"能否提升 Conflict Resolution 的 SF（substring exact match）。

> **架构说明**：系统目标形态、设计不变量与各组件实现状态见 [ARCHITECTURE.md](ARCHITECTURE.md)。
> **实验数据**：各数据集规模、字段、缺口与评估纪律见 [DATASETS.md](DATASETS.md)。
> **已知问题**：未修缺陷登记（含 P0）见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)。
> **本轮实验记录**：FST-Mem（阶段 B）见 [results/fst_mem_2026-10-27/](results/fst_mem_2026-10-27/README.md)；
> 上一轮 Router 根因诊断见 [results/latent_router_2026-10-27/](results/latent_router_2026-10-27/README.md)。
> 本文按时间倒序记录版本进展与实测数字。

## 最新进展（架构修正轮，2026-10-28）——先修实现，再谈结论

代码评审发现上一轮结果表里的潜层两法**在实现层面站不住**（多跳第二跳无监督、recon 目标错位、
MAB 的 BM25 排序取反、FST 只读单槽、STE 未 detach、单元缺实体锚点、版本规则被重新学习、
编码不复用、评测桥/router 混用与多跳 read_acc 口径过松）。9 处已逐一修掉，
详见 `CHANGELOG.md` 的「架构修正轮」条目。

**这一轮没有跑任何训练/评测**：因此本文件与 `results/RESULTS.md` 里 2026-10-27 的数字
仍是**旧口径**（HotpotQA dis4 100q：none 12.0 / text-top4 36.0 / oracle 58.0 /
bridge 17.5 / router 17.5；MAB-CR cap32：none 4.0 / text-top4 17.0 / oracle 32.0 /
router 2.5 / fst 2.5）。它们只说明"旧实现跑出来的东西"，不能当作四种方法的性能对比；
下一轮重训后整体替换，汇报时必须同时给 `none` 底线与"读对槽位"列。

## 上一轮（v9.7-real-data 轮，2026-10-27）——玩具数据退场：接入 HotpotQA / MAB-CR，修掉两处会让结果作废的 bug

**一句话**：问题不再是"缺新探针"，而是"缺能支撑论文核心结论的实验"。本轮把实验主线收束成三个问题
（真实潜层读出 / 多事实抗干扰 / 动态更新与历史访问），正式数据只留 **HotpotQA（训练+读出）** 与
**MemoryAgentBench-CR / LongMemEval（冲突覆盖、知识更新）**，方法对比压缩成四组
（Text-RAG+StatePool / Compressor+Bridge / Compressor+Router+Bridge / FST-Mem）。

### 1. 两处污染性 bug（已修，有定点回归测试）

- `delta_write_disabled` 退出时恒 `fn(model, False)` → **进程内永久关闭 Delta 写入**；
  改为逐模块快照/恢复各自原本的 `write_enabled`。
- KL 分支 `loss = loss_qa + kl_weight * loss_kl` **覆盖**了 `addr_weight * loss_addr`
  → 一开 KL，寻址监督静默消失；抽出纯函数 `assemble_loss` 统一累加。
- 新增 `reset_delta_state(model)`：每条样本前清 Delta 在线状态（不清 = 显存慢涨 / 样本互相污染，都不报错）。

回归测试：`tests/test_train_objectives.py`（14 项，含"KL 打开后 addr 项仍在"和"逐模块各自恢复"）。

### 2. 公开数据接入

| 数据 | 规模 | 用途 | 转换器 |
|---|---|---|---|
| HotpotQA distractor（镜像下载到 `/share/exp4group/data/hotpot/`） | train 90447 / val 7405 | 训练 + 评测 | `tools/prep_hotpotqa.py` |
| MAB Conflict Resolution | 8 context × 100 问 | 冲突覆盖 | `tools/prep_mab.py` |
| LongMemEval cleaned（264MB / 500 问） | 6 类问题 | 更新与时间推理 | 待接入（容量限制见 KNOWN_ISSUES §6） |

转换器里三条"写错不报错、只是让实验失去意义"的坑已固定并有单测：
`units` 必须逐条保留（一条事实一个槽）、`evidence` 必须是支持事实原文、同键冲突版本必须整组进上下文。

### 3. 统一评测入口 `eval_latent_table.py` 与第一个真实数字

四组方法 + 两个参照走**同一条 generate 路径**，差别只是"记忆以什么形式进入"；
指标 = 答案准确率 / 读对槽位率 / 更新准确率 / 历史访问准确率 / 注入 token 数 / 查询耗时。

HotpotQA dev（dis4，100 题）首个基线：

| 方法 | n | acc% | 读对槽位% | 注入记忆 token |
|---|---|---|---|---|
| **none（不给记忆，纯参数知识）** | 100 | **12.0** | n/a | 0 |

这 12% 是**所有方法必须超过的地板**——HotpotQA 的答案大量存在于模型参数知识里，
不报这一行就无法判断"记忆是否真被读到"。

### 4. 连通性修复

`--recon-weight>0` 用整段记忆做带梯度的 LM 前向+反传，实测峰值 21.8GB→30GB，第 3 步 OOM；
改为 `--recon-units`（默认 2）后峰值钉回 9.19GB / 6 步稳定。每步日志新增 `peakmem=`，
用于区分"自己涨"和"被同卡其他用户的进程挤掉"。

## v9.6.1-fst-audit 核查轮（2026-10-27）——核查并修掉 6 个静默 bug + 3 条恒真断言 + 3 个数据缺陷

**结论先行**：用户要求"核查代码"。三路并行审查 + 逐条源码复核，确认 **6 个不会报错、只会让指标悄悄失效的 bug**（训练/读出键特征不同源、STE 未 detach 载荷、版本命中不绑组别、配对恒真、淘汰/缺标签混淆、并列时间戳监督取最旧）、**3 条恒真断言**、**3 个数据缺陷**（21.9% 训练行的答案字符串出现在干扰事实里）。全部已修，变异审计 **6/6 变异被单测抓住**（工具随废案移除）。详见 [CHANGELOG v9.6.1](CHANGELOG.md) 与 [本轮记录](results/fst_mem_2026-10-27/README.md)。

<details>
<summary>v9.6-fst-mem 轮的原始记录（已折叠）</summary>

## v9.6-fst-mem 轮（2026-10-27）——FST-Mem：把 Router 降级为基线，加**事实键 + 状态/轨迹解耦 + 时序两级读出**（代码与单测齐备，**未训练**）

**结论先行**：

1. **研究定位修正**：`Compressor + Router + Bridge + 软 token` 三件套都有成熟研究基础，
   只解决**静态多事实读取**，没用到项目已有的事实覆写与时间记忆能力 → 只能当**基线**。
   真正的研究问题是：**事实压成潜向量后，在不断新增、局部更新、历史回溯的过程中，
   如何保持实体—属性绑定、避免无关事实干扰、并按时间取回历史值。**
2. **机制一 · 事实键与值无关**：寻址键从"整条事实"改成 `(实体, 属性)`。整句事实的键会随**值**变化，
   于是"换雇主"表现为"一个键消失、另一个键出现"，系统没有"同一属性"的先验；
   改成 `(实体,属性)` 后值变了键不动，局部更新才有意义。
3. **机制二 · 状态—轨迹解耦**：每个键 = `当前状态 + 有限历史轨迹`。写新值 = 旧值下沉历史、
   新值上位、**其他键逐位不动**（单测用 `torch.equal` 断言，不是"训练时学会别串"）。
4. **机制三 · 时序两级读出**：先选 (实体,属性)，再在被选中组内部按时间条件选版本，
   Reader **只收到 1 个版本的 m 个软 token**。时间条件与属性选择被隔离开，
   不让"2023 年的居住地"和"2025 年的雇主"抢同一个 softmax。
5. **一步真实训练都没跑**。本轮只有机制保证（核查后 **88** 项单测）+ CPU 极小模型的全链路通路验证。

| 项 | 数字 |
|---|---|
| 新数据集 | `fst_update_train` **192** 条 / `fst_update_val` **72** 条（实体不相交；三探针各 64 条） |
| 新单测 | `tests/test_fst_mem.py` **88 项 ALL PASS**（核查轮新增 21 项：内容级对齐 / 指标口径 / 并列时间戳 / 真实数据端到端）|
| 变异审计 | 核查轮 6/6 变异被抓住（该工具随废案移除，防"恒真断言"的方法保留在测试里）|
| 旧单测 | `tests/test_latent_readout.py` **89 项 ALL PASS**（`_emit_latent` 重构后不回归）|
| 全链路 | 训练 6 步 → 评估 → 存盘 → 重载 → 评估；`grad-check` 五头全 true；`n_soft=4`（= m）|
| 消融开关 | `--hist-max 0`（硬覆写对照）· `--ignore-query-time`（去时序条件）均可跑 |
| 效果 | **无**（随机权重下的 smoke 数字不是结论）|

**没做的事**：`hops>1` 未实现（显式报错）；未接进 AR/CR 线上管线；历史容量的容量—保真权衡未量化；
块对角掩码/正交键**刻意未做**（两级读出已在输入层隔离，先证明必要性）。完整记录见
[`results/fst_mem_2026-10-27/`](results/fst_mem_2026-10-27/README.md)。

</details>

## 文件

| 路径 | 内容 |
|---|---|
| `latent_memory.py` | 核心库：槽位库、LatentRouter、FST 状态—轨迹库、软 token 注入、checkpoint 指纹 |
| `train_latent_memory.py` | 训练入口：`--router`（Compressor+Router+Bridge）/ `--fst`（FST-Mem）/ 纯 Bridge；`assemble_loss` 组合 QA/寻址/重建 |
| `eval_latent_table.py` | **统一四方法评测入口**：`--method none\|text\|oracle\|bridge\|router\|fst`，明文与潜层同一条生成路径 |
| `deltamem_compat.py` | deltamem.core 在 transformers 5.x 下的最小兼容注入 |
| `tools/prep_hotpotqa.py` | HotpotQA distractor → 句级槽位 jsonl（含 dis0~8 × natural/shuffled 扫描集） |
| `tools/prep_mab.py` | MemoryAgentBench Conflict Resolution → cap 截断 jsonl（unit_meta 带事实键/时间） |
| `tools/gen_synth_qa.py`、`tools/gen_fst_update.py` | 玩具 smoke 数据生成（仅 smoke 用） |
| `tools/mem_probe.py`、`tools/diag_latent_capacity.py` | 显存探针 / 潜层容量诊断 |
| `runs/run_baselines.sh`、`runs/run_mab_baselines.sh` | 两基准的明文/无记忆基线脚本 |
| `results/RESULTS.md` | **第一版完整实验结果表**（两基准 × 四方法 × 五指标） |
| `results/<日期>/README.md` + json | 各轮结果与逐题明细 |
| `logs/2026-10-27/` | 清洗后的训练/评测日志 + 复现命令（唯一保留的日志轮） |
| `tests/` | 6 个测试文件 263 项断言 |

## 已废除方案（代码已删，历史见 CHANGELOG 与 git log ≤ e290382）

v5–v9 的自造数据管线（EventQA 窗口、合成事实宇宙、AR/CR 线上 runner、slotpool v6/v7/v8/v9、
run_manifest、ar_diag 逐题诊断、LME 旧 judge、对应生成器与数据）已全部移除；
废案时期的跑分明细（results/groupA/groupB2/v6/v7groupB2_*.jsonl、archive、latent_min16_exp2）
与旧训练日志（logs/ 根下 b2_*/smoke_*）也一并清除。
废除原因与收束决策见 `ARCHITECTURE.md` §11 与 `CHANGELOG.md`。
引用 2026-10-27 之前的任何数字前，先确认它出自哪一版管线。

## 运行

```bash
# 测试（全部本地 CPU，无需 GPU）
for f in tests/test_*.py; do DELTA_MEM_SCAN_IMPL=torch python3 $f; done

# 数据转换（服务器，需数据集）
python tools/prep_hotpotqa.py --split validation --n-distractors 4 --out synth/hotpot_val.jsonl
python tools/prep_mab.py --cap 32 --out synth/mab_cr_cap32.jsonl

# 基线（无需 checkpoint）
bash runs/run_baselines.sh
bash runs/run_mab_baselines.sh

# 训练 + 潜层评测（命令与超参见 logs/2026-10-27/README.md）
```
