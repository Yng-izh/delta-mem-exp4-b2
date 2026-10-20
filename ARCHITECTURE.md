# 架构设计：DM-ELSM（双模态弹性潜槽位记忆）

> 本文描述**当前预期架构**，即代码正在收敛到的目标形态，而不是"已完成并验证"的清单。
> 每一项都标注实现状态：`已实现` / `已接线·未验证` / `规划中`。
> 指标与实验结论见 `README.md` 与 `CHANGELOG.md`；本文只谈结构与契约。

---

## 1. 一句话定位

一个**双模态**长上下文记忆系统：把上下文流式拆成"语义状态"与"情景事件"两条记忆流，
用可寻址的**槽位（slot）**组织，检索时全池无损打分、按分数选证据，
读出时让证据以**潜层**方式（delta 状态或 `inputs_embeds`）参与生成，而不是只拼成一段明文提示。

## 2. 设计不变量（invariants）

这四条是架构的"宪法"，任何实现都不得违反。它们正是 v7/v8 三个 Bug 的根因所在。

| # | 不变量 | 反例（已修复） |
|---|---|---|
| I1 | **检索不截断全池**：候选池多大都参与打分，禁止"池 > N 就截到前 M" | v7 曾有 `n > 3500` 截断 |
| I2 | **先按分数选证据，最后才按原文位置排序**：位置只用于拼装可读性，绝不参与淘汰 | v8 `sorted(...)[:40]` 把后文高分证据整段截掉 |
| I3 | **时间是数据不是装饰**：每个单元绑定"该位置之前最近一次显式时间戳"，不跨 Session 继承，不得用全篇末尾时间回填 | v8 `cur_ts` 第二轮污染 |
| I4 | **读出必须走潜层**：明文证据块只是给人看的视图；真正的记忆注入经由 delta 状态或 `inputs_embeds` | 早期 `AR_DELTA_WRITE` 写在回答之后、且读出前 `reset` |

## 3. 数据流

```
                   ┌──────────────── 写入阶段（每个 context 一次）────────────────┐
Context ──切句──►  extract_units ──┬─► EpisodicIndex   （情景流：只追加，去重仅精确重复）
                                   └─► StatePool       （状态流：主语+关系双阈值原位覆写）
                                   │
                                   └─► delta 流式写入（可选，写后 clone 快照）
                   └────────────────────────────────────────────────────────────┘

                   ┌──────────────── 读出阶段（每道题）───────────────────────────┐
Query ──► 检索打分（IDF-MaxSim ⊕ 第 18 层稠密 ⊕ 时序加成）
      ──► readout：Top-K 或 EventQA 锚点非对称扩展
      ──► 组装：先按分数定证据集，再按原文位置排序
      ──► 读出：① 明文块 + prompt   ② delta 状态恢复快照   ③ inputs_embeds
      ──► 判分：SF / eventqa_recall；诊断：Recall@K + answer_in_retrieved
                   └────────────────────────────────────────────────────────────┘
```

## 4. 组件契约

| 组件 | 文件 | 职责 | 状态 |
|---|---|---|---|
| 切分与分流 | `v9_slotpool.extract_units` | 单句非重叠索引；逐句固化 `resolved_ts[i]`；`ev` 判定（事件动词 ∪ 显式时间标记） | 已实现 |
| 情景记忆 | `v9_slotpool.EpisodicIndex` | 只追加、精确去重；`expand_backward(back=16, ahead=4)`；`anchor_from_events` | 已实现 |
| 状态记忆 | `v9_slotpool.StatePool` | 主语精确匹配 + 关系谓词余弦 > `V9_CR_REL_THETA` 才原位覆写 | 已实现 |
| 检索打分 | `_ScoreStore.lexical_scores / dense_scores / hybrid_scores / temporal_rerank` | `hybrid = (1−α)·IDF-MaxSim + α·((cos+1)/2)`；provider 缺失静默退化为纯词面 | 已实现（稠密通道未在真模型上验证） |
| 证据组装 | `EpisodicIndex.readout` + `run_ar_v7._cap_before_sort` | 先分数选 Top-K / 锚点扩展，再位置排序，最后按配额裁剪 | 已实现 |
| CR 强化 | `v9_cr.CRStatePool` | Beam=6 ego-graph 第 1 跳；双跳链谓词联合打分；反事实强约束 Prompt | 已实现·未跑分 |
| 诊断 | `ar_diag.py` | 逐题 `retrieved_slot_ids/scores/texts`、`answer_in_retrieved`、`recall_at_k` | 已实现 |
| 判题 | `eval_longmemeval_judge.py` | 官方 6 题型 LLM-as-Judge 后处理；`--dry-run` 零网络 | 已实现·未真实判题 |
| AR runner | `run_ar_v7.py` | `AR_MODE=plain/v7/v8/v9`；诊断字段；`--dry-run` | 已实现 |
| CR runner | `run_group_b2_v7.py` | `CR_MODE=v7/v9`，默认 v7 行为不变 | 已接线·未跑分 |
| Delta 接口 | `prepare_delta_memory` + 快照 API | 查询前写完 → 关写 → `clone` 快照 → 每题 `load` 恢复 | 已接线·未在真模型上验证 |

## 5. 两条合法读出路径

架构上只承认这两条路。二者都以"记忆进入潜层"为前提，区别在于记忆载体。

**路径 A：Dual-Path 协同读出（delta 状态）**
- 写入：`AR_DELTA_WRITE=1` 时在**建池/查询之前**完成 512-token 分块流式 forward，随后关闭写开关。
- 快照：每个 context 结束 `clone` 一份各层 `delta_state`；每题生成前 `load` 恢复，**不清零**（`AR_DELTA_RESTORE=1` 默认开）。
- 目的：避免第 q_i 题的解码污染第 q_{i+1} 题，同时保留上下文结束时的非空状态。
- 状态：已接线，未在真实模型上验证；快照 API 缺失时回退手工 `iter_delta_mem_modules` clone/restore。

**路径 B：纯潜槽位表征（`inputs_embeds`）**
- 读出：`AR_INPUTS_EMBEDS=1` 时用连续隐向量而非 `input_ids` 调 `generate`。
- 目标形态（规划中）：`SlotPool` 存的不是 `texts`，而是进入第 1 层 Attention 之前的隐向量序列（或池化 latent tokens），彻底摆脱"外部明文 RAG"质疑。
- 现状：`inputs_embeds` 读出通路已接线；**槽位表征替换尚未实现**，池里仍是明文 + 可选的第 18 层稠密向量。

## 6. 关键参数

| 类别 | 变量 | 默认 | 含义 |
|---|---|---|---|
| 模式 | `AR_MODE` | `v7` | `plain / v7 / v8 / v9` |
| 模式 | `CR_MODE` | `v7` | `v7 / v9` |
| 索引 | `V9_BLOCK_SENTS` | 1 | 单元句数；=1 即单句非重叠 |
| 索引 | `V9_UNIT_MAX_WORDS` | 512 | 单元截断上限（先截断再记 span） |
| 检索 | `AR_TOP_K` / `V9_TOPK_STATE` | 12 / 4 | 情景 Top-K / 状态补充数 |
| 检索 | `V9_L18_ALPHA` / `V9_L18_LAYER` / `AR_V9_L18` | 0.3 / 18 / 0 | 稠密通道权重、层号、开关 |
| 检索 | `V9_TEMPORAL_TAU` / `V9_TEMPORAL_BONUS` | 2.0 / 0.15 | 时序衰减与加成 |
| 扩展 | `V9_EQA_BACK` / `V9_EQA_AHEAD` | 16 / 4 | EventQA 锚点非对称扩展 |
| 组装 | `AR_CTX_MAX_TOK` | 0 | 0 = 全上下文不截断（对外披露项） |
| Delta | `AR_DELTA_WRITE` / `AR_DELTA_RESTORE` / `AR_INPUTS_EMBEDS` | 0 / 1 / 0 | 写开关 / 快照恢复 / 潜向量读出 |
| CR | `V9_CR_BEAM` / `V9_CR_EGO_K` / `V9_CR_REL_THETA` / `V9_CR_HOPS` | 6 / 12 / 0.80 / 2 | 束宽、ego 子图上限、关系阈值、跳数 |

## 7. 实现状态矩阵

| 能力 | 状态 | 证据 |
|---|---|---|
| v8 三个确定性 Bug 修复 | 已实现 | `test_v8_slotpool.py` 含反向回归（旧版必失败） |
| 非重叠索引 / 分数优先 / 非对称扩展 / 时序重排 | 已实现 | `test_v9_slotpool.py` 95 断言 |
| 第 18 层混合稠密打分 | 已实现·未验证 | 单测用 stub provider；真实前向未跑 |
| 逐题诊断字段 | 已实现 | `ar_diag --selftest` 43/43 |
| LongMemEval 官方 Judge | 已实现·未判题 | `--selftest` 45/45，`--dry-run` 零网络 |
| CR 双阈值 / Beam=6 / 双跳链 / 反事实 Prompt | 已实现·未跑分 | `test_v9_cr.py` 59 断言 |
| Delta 写入顺序 / 快照恢复 | 已接线·未验证 | `test_ar_delta_alignment.py` 27 断言 |
| `inputs_embeds` 读出 | 已接线·未验证 | dry-run 覆盖，真实 generate 未跑 |
| **槽位存潜向量（替换明文）** | **规划中** | 无实现 |
| 可学习门控（何时写/何时读） | 规划中 | 无实现 |

## 8. 已知风险（必须在论文中披露）

1. **稠密通道依赖冻结模型中间层**，第 18 层是否最优未做消融；`α=0.3` 是拍定的。
2. **Delta 状态与 rank=8 冻结适配器**是否对细粒度问答有正贡献，尚无证据；若为负，架构主张须退到路径 B。
3. **CR 与 AR 的池语义不同**（覆写 vs 追加），统一为"双模态"是设计意图，尚缺跨任务族的一致性验证。
4. **v9 没有任何指标**：全部结论来自 dry-run 与 CPU 单测；v7/v8 的历史数字不得当作 v9 效果。
5. `ruler_qa1_197K` / `ruler_qa2_421K` 需要 choice 专用指标，当前 runner 显式拒绝，覆盖不全。

## 9. 演进路线

| 阶段 | 目标 | 退出条件 |
|---|---|---|
| P0（已完成） | 修 Bug、补诊断、对齐评测口径 | dry-run + 单测全绿 |
| P1（待跑分） | v9 检索在 EventQA 全切片与 LongMemEval 上对照 v7 | 四切片 + Recall@K 归因表 |
| P2（待跑分） | CR `CR_MODE=v9` 对照 262K 多跳 | sh/mh 不掉点，mh_262k 提升 |
| P3 | 路径 A 消融：`AR_DELTA_WRITE=1` vs 0，验证潜层是否真参与 | 同切片 A/B |
| P4 | 路径 B：槽位改存潜向量 + `inputs_embeds` 读出 | 摆脱"明文 RAG"质疑 |
| P5 | 可学习写入/读出策略（替代手工阈值 0.80 / 0.92） | 与手工阈值对照 |

## 10. 与旧版本的关系

- **CR（FactConsolidation）默认路径仍是 v7**：`CR_MODE=v7` 时 `run_group_b2_v7.py` 行为与历史一致（sh/mh 0.94/0.67 基线不受影响）。
- **v8 保留**为可对照实现，三个 Bug 已在原地修复；其历史四切片数字（0.7100/0.5120/0.3820/0.3167）是修复前产出，未重跑。
- **v9 是新主线**，但不改变任何默认值：所有新能力都靠 env 开关开启，未开启时行为等于 v7。
