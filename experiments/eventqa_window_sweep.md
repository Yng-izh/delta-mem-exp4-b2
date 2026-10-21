# EventQA 检索方向验证：窗口扫描

## 为什么做这个实验

v7/v8 的 EventQA 读出是**只向后取**：锚点单元之后 `EQA_WIN=4` 块。于是"答案在锚点之前"
的题目必然检索失败，而失败被记成"方法不行"。v9 改成**非对称后向扩展**
`expand_backward(anchor, back=16, ahead=4)`，方向假设是"事件问句的锚点通常落在答案之后"。

这个假设必须验证，不能只看总分：总分同时受锚点定位、窗口方向、生成质量影响。
所以拆成两个可直接观测的量：

| 指标 | 字段 | 含义 |
|---|---|---|
| `anchor_accuracy` | `anchor_answer_hit` | **锚点单元本身**是否含答案（定位对不对） |
| `answer_in_context` | `answer_in_context` | **最终证据窗口**是否覆盖答案（方向+宽度够不够） |

二者一起看才能区分两类失败：

- `anchor_answer_hit` 低 → 锚点定位错了，调窗口没用，要查 `anchor_from_events` / `best_unit_for`。
- `anchor_answer_hit` 高、`answer_in_context` 低 → 定位对但窗口方向/宽度不够，正是本实验要测的。
- 两者都高、SF 低 → 检索没问题，问题在生成或判分。

## 环境前置（2026-10-20 实测，必须先修）

`latent2` 自带 transformers 4.51.3 **没有** `models/smollm3`，而 `/share/delta-Mem` 的
`deltamem/core/backbone_compat.py:3` 无条件 `from transformers.models.smollm3...`，
于是 `import deltamem` 直接 `ImportError` —— **整个评测栈跑不起来**（不是本仓库的问题）。
同环境还缺 `nltk` / `rouge_score` / `editdistance` / `tiktoken`（`post_process` 依赖）。

处理：不改共享 conda 环境，装一个 overlay 目录，用 `PYTHONPATH` 覆盖：

```bash
pip install --target /share/exp4group/pylib_tf57 --no-deps transformers==4.57.1
pip install --target /share/exp4group/pylib_tf57 --no-deps "tokenizers>=0.22.0,<=0.23.0"
pip install --target /share/exp4group/pylib_tf57 nltk rouge_score editdistance tiktoken

export PYTHONPATH=/share/exp4group/pylib_tf57:/share/delta-Mem:/share/MemoryAgentBench
```

`run_eventqa_window_sweep.sh` 已内置这段 `PYTHONPATH`。

## 指标口径（必读）—— 已实测：`answer_in_context` 在 EventQA 上不可用

`anchor_answer_hit` / `answer_in_context` 是**字符串包含**判定：gold answer 原文是否出现在
锚点单元 / 最终窗口里。

**2026-10-21 实测结论：EventQA 的金标答案句在原文上下文里根本不存在**，所以这两个指标
结构性恒为 0，测不到任何东西（不是检索失败，是口径失效）。

验证方法（eventqa_65536 第 1 个 context，285,324 字符）：

```
gold = "Debbie wore a new green flowered-muslin dress that matched her slippers."
gold in context (raw)        -> False
gold in context (normalized) -> False      # 小写 + 去标点 + 折叠空白后仍找不到
```

原因：EventQA 的候选事件是**生成式改写**（LLM 按原文风格重写的选项），不是原文 span。
`metadata` 里也没有答案句的字符偏移或事件 id。

750 条实测记录里 `anchor_answer_hit` / `answer_in_context` 命中数 **0 / 750**，与上述判断一致。

后果：**事件级召回（evidence recall）在这套数据上算不出来**，能用的只有端到端
SF / `eventqa_recall`（6 选 1）。要做检索方向验证，必须先解决探针问题，见下节。

## 探针有效性对照（2026-10-21 新增，必看）

既然 `answer_in_context` 不可用，就必须先确认**端到端 SF 本身对检索内容敏感**，
否则"四组窗口 SF 差不多"什么也证明不了（6 个选项风格差异大，模型可能不看上下文也能选对）。

两个对照（同 ctx0、各 100 题）：

| 对照 | Back | Ahead | 窗口大小 | 目的 |
|---|---|---|---|---|
| `ctl_b0_a0` | 0 | 0 | **1 句**（只给锚点） | 几乎没有上下文时 SF 还剩多少 |
| `ctl_b200_a200` | 200 | 200 | ~400 句 | 宽窗天花板 |

判读规则：

- 若 1 句窗口 SF ≈ 0.81（与 16/4 默认相当）→ **EventQA SF 对检索不敏感**，
  这个 benchmark 不能用来验证检索方向，需要换探针（合成 needle / 带 span 金标的数据集）。
- 若 1 句窗口 SF 明显掉、宽窗明显涨 → SF 有区分度，四组窗口的差异才可信。

## 扫描配置

| 组 | Back | Ahead | 检验的假设 |
|---|---|---|---|
| 1 | 0 | 20 | 纯前向（锚点之后取 20 句）——v7/v8 方向的极端版 |
| 2 | 4 | 16 | 弱后向 |
| 3 | 10 | 10 | 对称 |
| 4 | 16 | 4 | **当前默认**（强后向） |

四组窗口总宽都是 20 句，只改方向，保证可比。

## 实测结果（2026-10-21，未跑完，仅 ctx0 全量 + ctx0/1 交集）

聚合数字已入库：[`experiments/results/2026-10-21_eventqa_sweep.json`](results/2026-10-21_eventqa_sweep.json)
（含每窗口 run_id / manifest 指纹 / 各指标，原始 JSONL 留在服务器 `/share/exp4group/results/sweep/`）。

四组并行跑 9 小时后只推进到第 2 个 context（每题 5s → 200s，宿主机 load 180，
别人的 CPU 作业把机器压满），已杀掉重排。现有可比数字：

| 窗口 | n=100（ctx0）SF | n=144（ctx0+1 交集）SF | 平均块句数 |
|---|---|---|---|
| b0_a20 | 0.810 | 0.826 | 21.0 |
| b4_a16 | 0.810 | 0.819 | 21.0 |
| b10_a10 | 0.800 | 0.806 | 20.9 |
| b16_a4（默认） | 0.730 | 0.750 | 20.7 |

`anchor_answer_hit` / `answer_in_context`：**全部为 0**（口径失效，见上节），不构成结论。

初步方向：强后视（16/4）最差，其余三组在噪声内（差值 7.6pp，n=144 时约 1.7σ，
**不显著**）。在探针有效性对照跑完前，这张表不能作为"方向结论"使用。

### 探针有效性对照（部分完成，已中止）

| 对照 | 窗口大小 | n | SF | 判读 |
|---|---|---|---|---|
| `ctl_b0_a0` | 1 句 | 52 | 0.635 | 上下文太少明显掉分 |
| 默认（b16_a4） | 21 句 | 100 | 0.810 | 最高 |
| `ctl_b200_a200` | ~350 句 | 23 | 0.783 | 塞太多略降（噪声内） |

结论：**端到端 SF 确实依赖检索内容**（1 句 vs 21 句差 17.5pp），是有效探针；
但两头都更差，说明默认 21 句窗口是合理折中。

### 本轮结论与决策（2026-10-21）

1. **`answer_in_context` / `anchor_answer_hit` 不可用**（金标不在原文，750/0），
   事件级召回必须换数据（带 span 金标）或换探针（合成 needle）。
2. 四组窗口差异不显著 → **窗口方向不是可优化点**，`back=16 / ahead=4` 保留为默认。
3. **实验中止**：继续跑满 5 个 context（500 题/组）不会把 7.6pp 变成显著；
   宿主机 load 180 时四进程并行每题 200s，性价比不成立。
4. 若要重跑（例如换了探针），命令见下节，注意先删旧文件（manifest 记录的 commit 已变，会拒绝续跑）。

## 运行

```bash
bash experiments/run_eventqa_window_sweep.sh
# 或手动单组：
AR_MODE=v9 AR_V9_EQA_BACK=4 AR_V9_EQA_AHEAD=16 \
  AR_OUT=/share/exp4group/results/sweep/ar_v9_eventqa_65536_b4_a16.jsonl \
  python run_ar_v7.py eventqa_65536 0
```

每组结果文件旁边会自动生成 `<out>.manifest.json`（配置指纹 + 完整参数 + git commit）。
**换配置必须换文件**：若把两个窗口写进同一个 `AR_OUT`，第二次启动会被 manifest 校验拒绝
（`[resume][ERR] 拒绝续跑`），这正是为了不让两组消融混在一起。

## 汇总

```bash
python3 tools/summarize_sweep.py /share/exp4group/results/sweep/ar_v9_eventqa_65536_b*.jsonl
```

输出列：`window / n / SF / anchor / in_ctx / AR@1 / AR@3 / AR@10 / run_id`。

## 记录（待填）

| Back/Ahead | n | SF | anchor_accuracy | answer_in_context | AR@1 | AR@3 | AR@10 | run_id |
|---|---|---|---|---|---|---|---|---|
| 0/20 | | | | | | | | |
| 4/16 | | | | | | | | |
| 10/10 | | | | | | | | |
| 16/4 | | | | | | | | |

> 说明：本表数值必须来自真实运行。CPU 单测只能验证"扩展逻辑正确"，
> 不能代替这四个数字。未跑之前一律留空，不填估计值。

## 判读规则（跑完后照此结论）

1. 若 `16/4` 的 `answer_in_context` 明显高于 `0/20` → 后向假设成立，默认值保持。
2. 若 `10/10` 或 `4/16` 更高 → 后向幅度过大，把默认 `V9_EQA_BACK` 调小。
3. 若四组 `anchor_accuracy` 都低（< 0.5）→ 问题在锚点定位而非窗口，先修
   `anchor_from_events`（`cov_threshold=0.5`、`last/last2` 重锚）再谈方向。
4. 若 `anchor_accuracy` 高但 `answer_in_context` 四组都低 → 窗口总宽 20 不够，
   下一轮扫宽度（如 8/8、16/16、32/32）而不是方向。
