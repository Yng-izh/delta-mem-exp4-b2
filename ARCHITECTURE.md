# 架构设计：DM-ELSM（双模态弹性潜槽位记忆）

> 本文区分两件事：**已经写进代码并有单测的能力**，和**架构上承诺但尚未实现的目标**。
> 每个组件都带状态标记：`已实现` / `已接线·未验证` / `规划中`。
> 指标与实验结论见 `README.md`、`CHANGELOG.md`、`experiments/`；本文只谈结构与契约。
>
> **最重要的一句**：目前**没有任何"潜槽位读出"**。读出层只有"明文 prompt"和
> "明文 token 的嵌入矩阵（`EMB[ids]`）"两种模式，二者都基于外部明文证据。
> 真正的 latent readout 是路径 B，仍在规划中（§6）。

---

## 1. 一句话定位

一个**双模态**长上下文记忆系统：把上下文流式拆成"语义状态"与"情景事件"两条记忆流，
用可寻址的**槽位（slot）**组织，检索时全池无损打分、按分数选证据，
读出时把选中的证据送进生成模型。

"双模态"指**记忆的组织方式**（状态流 vs 事件流）双模态，**不是**指读出已经潜层化。
读出是否潜层化，见 §6 的三种模式。

## 2. 三层结构

```
① 输入事件流 (Input Event Stream)
   原始 context ──切句──► unit[i] = (text, span, session_id, event_time, ingestion_order)
   契约：非重叠、覆盖全文、时间是数据（I3）

        ├──────────────────────────┬──────────────────────────┐
        ▼                          ▼                          ▼
② State Memory              ③ Episodic Memory          ④ 写入侧 delta（可选）
   StatePool                   EpisodicIndex              分块流式 forward
   槽位 = 事实三元组           槽位 = 单个句子单元         写后 clone 快照
   主语+关系双阈值原位覆写      只追加、精确去重            逐题 load 恢复
   （有损、可更新、有"当前值"）  （无损、全量、可回溯）      （路径 A，未验证）

        └──────────────┬───────────┘
                       ▼
              ④ Evidence Router（证据路由）
                 全池打分：IDF-MaxSim ⊕ 第 18 层稠密 ⊕ 时序加成
                 选证据：Top-K（分数优先）或 EventQA 锚点非对称扩展
                 组装：先按分数定集合 → 再按原文位置排序 → 最后按配额裁剪

                       ▼
              ⑤ Latent Readout（潜层读出）
                 现状：明文块 + prompt  |  EMB[ids] 的 inputs_embeds
                 目标：槽位内潜向量直接进入第 1 层 Attention（未实现）
```

**层间边界**：②③ 是两套**独立**的槽位语义，共享同一套 unit 与同一套打分器；
④ 只决定"取哪些槽位"；⑤ 只决定"怎么把槽位送进模型"。当前 ⑤ 拿到的是**文本**。

## 3. 设计不变量（invariants）

架构的"宪法"，任何实现都不得违反。每一条都对应一个真实修过的 Bug。

| # | 不变量 | 反例（已修复） |
|---|---|---|
| I1 | **检索不截断全池**：池多大都参与打分，禁止"池 > N 就截到前 M" | v7 曾有 `n > 3500` 截断 |
| I2 | **先按分数选证据，最后才按原文位置排序**：位置只用于拼装可读性，绝不参与淘汰 | v8 `sorted(...)[:40]` 把后文高分证据整段截掉 |
| I3 | **时间是数据不是装饰**：每个单元绑定"该位置之前最近一次显式时间戳"，**不跨 Session 继承**，不得用全篇末尾时间回填 | v8 `cur_ts` 跨 Session 污染 |
| I4 | **读出必须走潜层**（目标态）：明文证据块只是给人看的视图 | 早期 `AR_DELTA_WRITE` 写在回答之后、读出前又 `reset` |
| I5 | **结果是配置的函数**：不同配置不得共用结果文件，续跑必须校验配置与代码版本 | 旧版固定 `ar_{MODE}_{SRC}.jsonl` + `AR_RESUME=1`，消融互相污染 |

I4 是**目标态**，当前代码不满足（§6）。I5 由 `run_manifest.py` 强制。

## 4. 数据流（含设备边界）

```
写入：context ─► extract_units ─┬─► EpisodicIndex（追加）
                               ├─► StatePool（双阈值原位覆写）
                               └─► delta 流式写（可选）→ snapshot

读出：query ─► 词面 IDF-MaxSim（token 张量所在设备，通常 GPU）
            ─► 第 18 层稠密（provider 一律返回 CPU fp32）
            ─► hybrid_scores：把稠密分搬到词面分同一设备后相加（唯一对齐点）
            ─► 选证据 → 位置排序 → 配额裁剪 → block
            ─► generate(input_ids) 或 generate(inputs_embeds=EMB[ids])
```

设备约定（v9 曾因此崩过）：`l18` 向量常驻 **CPU fp32**（262K 单元 × d 放显存会爆），
`hybrid_scores` 在相加前把两侧对齐到同一 device/dtype；**任何新打分项都必须显式对齐**。

## 5. 组件契约

| 组件 | 文件 | 职责 | 状态 |
|---|---|---|---|
| 切分与分流 | `v9_slotpool.extract_units` | 单句非重叠索引；`session_id` / `event_time` / `ingestion_order`；**按 Session 重置**时间戳 | 已实现 |
| 会话切分 | `v9_slotpool.split_sessions` | 识别 `Session N` / `## Session` / `=== Session` / `Date:` / `YYYY/MM/DD`；无标记则整篇一个 Session | 已实现 |
| 情景记忆 | `v9_slotpool.EpisodicIndex` | 只追加、精确去重；`expand_backward(back, ahead)`；`anchor_from_events` | 已实现 |
| 状态记忆 | `v9_slotpool.StatePool` | 主语索引 `_subj_index`（O(1) 查同主语）+ 关系谓词余弦 > `V9_CR_REL_THETA` → **原位 `_replace`**（槽 id 稳定） | 已实现 |
| 检索打分 | `_ScoreStore.*` | `hybrid = (1−α)·IDF-MaxSim + α·((cos+1)/2)`；provider 缺失时退化纯词面，**strict 模式下报错** | 已实现·稠密未真跑 |
| 证据组装 | `EpisodicIndex.readout` + `run_ar_v7._cap_before_sort` | 先分数选 Top-K / 锚点扩展，再位置排序，最后配额裁剪 | 已实现 |
| CR 强化 | `v9_cr.CRStatePool` | Beam=6 ego-graph 第 1 跳（矩阵初筛 + 逐 token MaxSim）；双跳链联合打分；**路径式读出**；反事实 Prompt | 已实现·未跑分 |
| 诊断 | `ar_diag.py` | `answer_recall_at_k`（答案串命中代理）、`answer_in_retrieved`、`answer_in_context`、`evidence_recall_at_k`（无金标证据时为 None） | 已实现 |
| 实验管理 | `run_manifest.py` | 配置指纹 → 输出文件隔离；sidecar manifest；续跑校验（配置/参数/commit） | 已实现 |
| AR runner | `run_ar_v7.py` | `AR_MODE=plain/v7/v8/v9`；manifest；strict retrieval；`--dry-run` | 已实现 |
| CR runner | `run_group_b2_v7.py` | `CR_MODE=v7/v9`；manifest；续跑跳过已完成题 | 已接线·未跑分 |
| Delta 接口 | `prepare_delta_memory` + 快照 API | 查询前写完 → 关写 → `clone` → 每题 `load`；**空快照直接报错** | 已接线·未真模型验证 |
| 判题 | `eval_longmemeval_judge.py` | 官方 6 题型 LLM-as-Judge；`--dry-run` 零网络 | 已实现·未真实判题 |

## 6. 读出层的三种模式（诚实边界）

| 模式 | 命令 | 实际发生什么 | 能否称为"潜槽位记忆" |
|---|---|---|---|
| `prompt` | `AR_READOUT_MODE=prompt`（默认） | 证据文本拼进 user prompt，`generate(input_ids=...)` | 否，明文 RAG |
| `text_embeds` | `AR_READOUT_MODE=text_embeds` | `generate(inputs_embeds=EMB[ids])`：把**同一段明文**换成查表嵌入 | **否**。等价于 prompt，只是绕过 embedding 层；无任何槽位内部状态 |
| `latent` | `AR_READOUT_MODE=latent` | **未实现**，runner 直接退出码 2 报错 | — |

因此：**"双模态潜槽位记忆"目前只有记忆组织层成立，读出层仍是明文。**
论文里若声称 latent memory，必须有 §7 路径 B 的实现与消融；否则只能主张
"结构化双流记忆 + 无损证据路由"。

**路径 A（delta 状态）**：`AR_DELTA_WRITE=1` 时建池前分块流式写，写后关写并 `clone` 快照，
每题生成前 `load` 恢复。这是**唯一**真正把上下文写进模型内部状态的通道，
但它对细粒度问答的贡献尚无证据（见 §9 风险 2）。

**路径 B（纯潜槽位表征，规划中）**：槽位不再存 `texts`，而存进入第 1 层 Attention 前的
隐向量序列（或池化 latent tokens），读出时直接拼接 `inputs_embeds`。当前**无实现**。

## 7. 关键参数

| 类别 | 变量 | 默认 | 含义 |
|---|---|---|---|
| 模式 | `AR_MODE` | `v7` | `plain / v7 / v8 / v9` |
| 模式 | `CR_MODE` | `v7` | `v7 / v9` |
| 索引 | `V9_BLOCK_SENTS` | 1 | 单元句数；=1 即单句非重叠 |
| 索引 | `V9_UNIT_MAX_WORDS` | 512 | 单元截断上限（先截断再记 span） |
| 索引 | `V9_SESSION_RESET` / `V9_SPLIT_ON_NEWLINE` | 1 / 1 | 按 Session 重置时间戳 / 按行切句 |
| 检索 | `AR_TOP_K` / `V9_TOPK_STATE` | 12 / 4 | 情景 Top-K / 状态补充数 |
| 检索 | `V9_L18_ALPHA` / `V9_L18_LAYER` / `AR_V9_L18` | 0.3 / 18 / 0 | 稠密通道权重、层号、开关 |
| 检索 | `V9_TEMPORAL_TAU` / `V9_TEMPORAL_BONUS` | 2.0 / 0.15 | 时序衰减与加成 |
| 扩展 | `V9_EQA_BACK` / `V9_EQA_AHEAD`（runner 用 `AR_V9_EQA_BACK/AHEAD`） | 16 / 4 | EventQA 锚点非对称扩展 |
| 组装 | `AR_CTX_MAX_TOK` | 0 | 0 = 全上下文不截断（对外披露项） |
| 读出 | `AR_READOUT_MODE` | `prompt` | `prompt / text_embeds / latent(未实现)` |
| Delta | `AR_DELTA_WRITE` / `AR_DELTA_RESTORE` / `AR_ALLOW_EMPTY_DELTA_SNAP` | 0 / 1 / 0 | 写开关 / 快照恢复 / 空快照放行 |
| 严格性 | `AR_STRICT_RETRIEVAL` | `auto` | auto：稠密/潜层/写入开启时检索异常直接抛 |
| 复现 | `AR_RESUME` / `AR_RESUME_MANIFEST` / `AR_OUT` | 1 / 1 / 空 | 续跑 / 校验 manifest / 显式输出路径 |
| CR | `V9_CR_BEAM` / `V9_CR_EGO_K` / `V9_CR_REL_THETA` / `V9_CR_HOPS` | 6 / 12 / 0.80 / 4 | 束宽、ego 上限、关系阈值、跳数 |
| CR | `V9_CR_HOP1_M` / `V9_CR_PATH_VIEW` / `V9_CR_PATHS` | 96 / 1 / 3 | Hop1 矩阵初筛候选数 / 路径式读出 / 最多路径数 |

## 8. 实现状态矩阵

| 能力 | 状态 | 证据 |
|---|---|---|
| v8 三个确定性 Bug 修复 | 已实现 | `test_v8_slotpool.py` 57 断言 |
| 非重叠索引 / 分数优先 / 非对称扩展 / 时序重排 | 已实现 | `test_v9_slotpool.py` 129 断言 |
| Session 边界不继承时间戳（I3） | 已实现 | `test_v9_slotpool.py::test_session_boundary_ts` |
| 第 18 层混合打分的设备对齐 | 已实现 | `test_hybrid_device_consistency`（cpu/cuda/mps 三选一） |
| 第 18 层稠密通道**效果** | 未验证 | 单测用 stub provider；真实前向未跑 |
| 状态池同主语原位覆写（槽 id 稳定） | 已实现 | `test_cr_overwrite_finds_same_subject_slot`（旧实现必失败） |
| 状态池 O(N) 写入（主语索引，非全池扫描） | 已实现 | `test_cr_pending_merge_and_stable_ids` |
| CR Hop1 矩阵初筛（P1 性能） | 已实现 | `test_hop1_matrix_prefilter`（120 槽 → 96 候选，gold 不丢） |
| CR 路径式读出（seed→mid→target + 其余事实） | 已实现·未跑分 | `test_path_view_and_flat_fallback` |
| 逐题诊断字段 | 已实现 | `ar_diag --selftest` 57 项 |
| 配置指纹 / 结果文件隔离 / 续跑校验 | 已实现 | `test_run_manifest.py` 27 断言 |
| Delta 空快照必须报错 | 已实现 | `test_run_manifest.py::test_empty_delta_snapshot_raises` |
| LongMemEval 官方 Judge | 已实现·未判题 | `--selftest` 45 项，`--dry-run` 零网络 |
| Delta 写入顺序 / 快照恢复 | 已接线·未验证 | `test_ar_delta_alignment.py` 27 断言 |
| `inputs_embeds` 读出通路 | 已接线·未验证 | 明文嵌入，**不是**潜槽位 |
| **槽位存潜向量（替换明文）** | **规划中** | 无实现 |
| 可学习门控（何时写/何时读） | 规划中 | 无实现 |

## 9. 已知风险（必须在论文中披露）

1. **稠密通道依赖冻结模型第 18 层**，层号与 `α=0.3` 都是拍定的，无消融。
2. **Delta 状态 + rank=8 冻结适配器**对细粒度问答是否有正贡献，**无证据**；
   若为负，架构主张须退回路径 B（而路径 B 未实现）。
3. **CR 与 AR 池语义不同**（覆写 vs 追加），统一为"双模态"是设计意图，
   缺跨任务族一致性验证。
4. **v9 没有任何指标**：全部结论来自 dry-run 与 CPU 单测；v7/v8 历史数字不得当作 v9 效果。
5. `ruler_qa1_197K` / `ruler_qa2_421K` 需要 choice 专用指标，runner 显式拒绝，覆盖不全。
6. `answer_recall_at_k` 是**答案字符串命中代理**（gold answer 是否出现在第 k 个槽位），
   **不是**证据级召回；本管线没有金标证据 span，`evidence_recall_at_k` 恒为 None。

## 10. 演进路线

| 阶段 | 目标 | 退出条件 |
|---|---|---|
| P0（已完成） | 修 Bug、补诊断、对齐评测口径、结果可复现 | 单测全绿 + dry-run |
| P1（待跑分） | v9 检索在 EventQA 全切片与 LongMemEval 上对照 v7 | 四切片 + 召回归因表 |
| P2（待跑分） | CR `CR_MODE=v9` 对照 262K 多跳 | sh/mh 不掉点，mh_262k 提升 |
| P3 | 路径 A 消融：`AR_DELTA_WRITE=1` vs 0 | 同切片 A/B |
| P4 | 路径 B：槽位改存潜向量 + `inputs_embeds` 读出 | 摆脱"明文 RAG"质疑 |
| P5 | 可学习写入/读出策略（替代手工阈值 0.80 / 0.92） | 与手工阈值对照 |

## 11. 与旧版本的关系

- **CR 默认路径仍是 v7**：`CR_MODE=v7` 时 `run_group_b2_v7.py` 行为与历史一致。
- **v8 保留**为可对照实现，三个 Bug 已原地修复；历史四切片数字是修复前产出，未重跑。
- **v9 是新主线**，不改变任何默认值：新能力全部靠 env 开关，未开启时行为等于 v7。
- **v9 CR 的路径式读出默认开启**（`V9_CR_PATH_VIEW=1`）：因为 v9 CR 从未跑过分，
  不存在与历史数字的可比性问题；若要对照旧版平铺输出，设 `V9_CR_PATH_VIEW=0`。
