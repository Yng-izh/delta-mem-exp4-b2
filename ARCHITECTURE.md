# 架构设计：DM-ELSM（双模态弹性潜槽位记忆）

> 本文区分两件事：**已经写进代码并有单测的能力**，和**架构上承诺但尚未实现的目标**。
> 每个组件都带状态标记：`已实现` / `已接线·未验证` / `规划中`。
> 指标与实验结论见 `README.md`、`CHANGELOG.md`、`experiments/`；本文只谈结构与契约。
>
> **最重要的一句**：潜层读出（路径 B）**代码已落地，多槽形态已被证伪，重构版未训练**。
> `AR_READOUT_MODE=latent` 现在会真的把槽位压成软 token 送进 `inputs_embeds`（明文进不去），
> 但压缩器/桥接器是随机初始化的 —— 没有 `LATENT_CKPT` 时 runner 直接报错退出。
> 因此**目前所有指标仍来自明文 prompt 模式**，不能声称任何 latent 效果（§6）。
>
> 2026-10-27 更新：已用干净诊断（训练同款 prompt + Delta 写入关闭 + 每条 reset）确认
> **多槽读出失败的原因是纯位置效应、零内容寻址**——同一事实单槽 15/16，套上 4 槽外壳即 1/16，
> 目标挪到末位 0/16。已引入 `LatentRouter`（集合内积寻址、推理只送 1 槽）作为结构性修法，
> **但一步训练都没跑**，能否救回无任何证据。详见
> `results/latent_router_2026-10-27/`、`DATASETS.md`、`KNOWN_ISSUES.md`。
>
> 2026-10-27 追加（FST-Mem，阶段 B）：在 `LatentRouter` 之上再加**事实键级寻址 +
> 状态/轨迹解耦 + 时序两级读出**（§6.5）。代码与机制单测齐备（`tests/test_fst_mem.py` 67 项），
> 训练入口 `--fst` 在 CPU 极小模型上跑通全链路（训练→评估→存盘→重载→评估）。
> **同样一步真实训练都没跑**，没有任何效果数字；`LatentRouter` 的定位从"候选方法"
> 下调为"静态多事实内容寻址基线"。详见 `results/fst_mem_2026-10-27/`。

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
                 明文模式（默认）：明文块 + prompt  |  EMB[ids] 的 inputs_embeds
                 潜层模式（latent，需训练）：槽位 [h0;h18] → Z∈R^{m×d} → 软 token → inputs_embeds
```

**层间边界**：②③ 是两套**独立**的槽位语义，共享同一套 unit 与同一套打分器；
④ 只决定"取哪些槽位"；⑤ 只决定"怎么把槽位送进模型"。⑤ 默认拿到的是**文本**；
`AR_READOUT_MODE=latent` 时拿到的是**槽位载荷 `Z`**（无明文，但需先训练）。

## 3. 设计不变量（invariants）

架构的"宪法"，任何实现都不得违反。每一条都对应一个真实修过的 Bug。

| # | 不变量 | 反例（已修复） |
|---|---|---|
| I1 | **检索不截断全池**：池多大都参与打分，禁止"池 > N 就截到前 M" | v7 曾有 `n > 3500` 截断 |
| I2 | **先按分数选证据，最后才按原文位置排序**：位置只用于拼装可读性，绝不参与淘汰 | v8 `sorted(...)[:40]` 把后文高分证据整段截掉 |
| I3 | **时间是数据不是装饰**：每个单元绑定"该位置之前最近一次显式时间戳"，**不跨 Session 继承**，不得用全篇末尾时间回填 | v8 `cur_ts` 跨 Session 污染 |
| I4 | **读出必须走潜层**（目标态）：明文证据块只是给人看的视图 | 早期 `AR_DELTA_WRITE` 写在回答之后、读出前又 `reset` |
| I5 | **结果是配置的函数**：不同配置不得共用结果文件，续跑必须校验配置与代码版本 | 旧版固定 `ar_{MODE}_{SRC}.jsonl` + `AR_RESUME=1`，消融互相污染 |
| I6 | **训练事实必须独立于评测集**：不得用 MemoryAgentBench / LongMemEval / EventQA 的任何切片训练；训练样本按事实切槽（一事实一槽位），与推理期 bank 布局一致 | 服务器上唯一的本地数据集就是 `Conflict_Resolution-*.parquet`（即评测集），用它训练等于背答案；训练时若整段压成一个槽，"检索到某一条事实"在训练里根本不存在 |
| I7 | **软 token 必须与真实 token embedding 同尺度**：`bridge` 输出要做范数对齐（实测未约束时范数 96–159，真实 embedding 1.05，差两个量级会让软 token 变成 attention sink 把真实 token 淹没） | 首轮真机训练 loss 12.6→3.87 但验证 0/32，200 条过拟合 1/200；加宽 m 4→16 无改善 |

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
| `latent` | `AR_READOUT_MODE=latent`（需 `AR_MODE=v9` + `LATENT_CKPT`） | 槽位先压成 `Z∈R^{m×d}`，经桥接器变成 k×m 个**软 token 嵌入**直接拼进 `inputs_embeds`；生成端拿不到任何明文 | **是**（通路成立）；但**未训练 = 无意义**，见下方"必须先训练" |

因此：**"双模态潜槽位记忆"的读出层现在有了实现，但仍没有指标。**
`latent` 通路的正确性有机制性单测兜底（明文进不去、软 token 位置/掩码/梯度、分块不截断、
载荷库语义、路径预算、checkpoint 严格加载），**效果一个数字都没有**。

**路径 A（delta 状态）**：`AR_DELTA_WRITE=1` 时建池前分块流式写，写后关写并 `clone` 快照，
每题生成前 `load` 恢复。这是**唯一**真正把上下文写进模型内部状态的通道，
但它对细粒度问答的贡献尚无证据（见 §9 风险 2）。

**路径 B（潜槽位读出，v9.3 已实现）**：

```
槽位原文 ──冻结 Qwen──> [h0; h18] token 级 [L,d]
                              │  LatentCompressor: m 个可学习 query 做 cross-attention
                              ▼
                        Z ∈ R^{m×d}（每槽 m 个软向量）──存入 LatentSlotBank（键=稳定槽位 ID）
                              │  读出时按检索结果取载荷 → LatentBridge
                              ▼
                    [k·m, d] 软 token 嵌入 ──拼进 inputs_embeds（system 之后、query 之前）
```

- **为什么单独写 `encode_latent_units()`**：`_layer18_provider()` 返回的是 mean-pool 后的
  `[N,d]`，token 轴已经没了，拿它做 token-level 压缩等于对着一句向量再注意一次。
- **分块而非截断**：单元长于 `LATENT_MAX_UNIT_TOKENS=64` 时切成不重叠块并记父槽位 ID；
  截断可能正好丢掉答案。
- **载荷以 ID 为键**：`_replace` 原位覆写（下标稳定、版本 +1），`_drop`/`_evict` 会移动下标，
  必须 `shift_index(namespace, drop_index)`，否则 `state:5` 指向另一条事实。
- **必须先训练**：`LATENT_CKPT` 缺失时 runner 直接报错退出，**不会**拿随机投影跑对照。
  训练见 `train_latent_memory.py`（冻结 Qwen + Adapter，只训 Compressor + Bridge）。
- **CR 侧同一通路**：`CR_READOUT_MODE=latent` 时 CR 生成从潜载荷构造，不再拼事实块；
  多跳路径整条保留/整条跳过（`select_latent_cr_paths`），不出现半条路径。

## 6.5 FST-Mem：状态—轨迹解耦的潜层记忆（阶段 B，已实现 · 未训练）

**定位**：`LatentRouter`（§6 路径 B 的寻址头）只解决**静态多事实**里的"选哪一条"。
论文主线要回答的是另一件事：事实被压缩成潜向量之后，**在不断新增、局部更新、历史回溯**的
过程中，如何保持正确的实体—属性绑定、避免无关事实干扰、并支持按时间访问历史状态。
`LatentRouter` 因此降级为**基线**，FST-Mem 是在它之上的结构性扩展。

### 三个机制与代码的对应

| 机制 | 要解决的问题 | 实现 | 关键契约 |
|---|---|---|---|
| 一 · 实体—属性级稳定寻址 | 整句事实的键会随**值**变化，于是"换雇主"表现为"一个键消失、另一个键出现"，系统没有"同一属性"的先验 | `LatentFactKeyEncoder`：`k_{e,r}=normalize(out([h_e,h_r,h_e⊙h_r]))`，`h_e=e_net(实体特征)`、`h_r=r_net(属性特征)` | 键**与值无关**（值变了键不变）；留交互项（`h_e⊙h_r`）区分"泛属性名 + 具体实体"的搭配 |
| 二 · 状态—轨迹解耦槽位 | 硬覆写丢历史；全部追加则新旧事实在同一读出里竞争，更新退化成"多了一条矛盾记忆" | `LatentStateTrajectoryBank`：每个事实键 = `{cur, cur_time, hist[(slot,time)]}`；`write()` 把旧 cur 下沉进 hist，超 `LATENT_HIST_MAX` 淘汰最旧并**真正释放槽位** | **更新局域性由数据结构保证**：写 `zhang||employer` 时 `zhang||residence` 的载荷逐位不变（单测 H7/H8 断言） |
| 三 · 时序条件两级读出 | 时间条件若混进同一个打分式，"2023 年的居住地"会与"2025 年的雇主"抢同一个 softmax | `FstRouter`：第一级在**事实键**上选（`<q,k_{e,r}>`，与组数/顺序无关），第二级在**被选中组内部**按 `<q_t,k_ver>` 选版本；Reader 只收到一个版本的 m 个软 token | 两级的竞争被隔离（`logits_ver` 长度 = 组内版本数，不是 G×V）；前向一律 argmax + STE |

### 刻意**未**采用的候选机制（需先证明必要性）

- **块对角注意力掩码 / 绑定子空间**：论文计划里列为"待验证候选"。当前两级读出已经在**输入层**
  把"只送一个版本"做到位（其余键与版本根本不进 `inputs_embeds`），在证明这一步不够之前
  改造冻结模型的注意力结构属于"加模块 ≠ 加贡献"。
- **正交键约束**：只要求"可靠区分 + 局部更新"（组级 CE = 批内 softmax 的对比约束），
  不要求数学正交——正交既不必要，也难以在冻结 Qwen 的语义空间里维持。

### 训练入口与评估口径

```
tools/gen_fst_update.py              # 同实体多属性 + 多次覆写 + 时间条件（192/72 条）
train_latent_memory.py --fst ...     # 独立入口，不改动阶段 A 的 train()/evaluate()
```

必报指标（缺一不可，只报 acc 会掩盖"选错版本但那条也能答对"）：

| 指标 | 含义 |
|---|---|
| `acc` | 规范化**完整**答案匹配（不是子串） |
| `group_acc` | 事实级寻址：选对 (实体,属性) 了吗 |
| `ver_multi_acc` | 版本级寻址，**只统计目标组有 >1 个版本的样本**（单版本样本"选对"恒真，计入会虚抬） |
| `probes{current,history,unrelated}` | 三探针分解；`unrelated` = 无关属性保持率（改雇主后居住地还答不答得对） |
| `pair_acc` | 反事实配对：同问题不同值，整组全对才算 |
| `order_stable` | 写入顺序打乱后答案一致率（组合稳定性） |
| `n_soft` | 实际注入的软 token 数（= m，与版本数无关） |

### 消融矩阵（当前可跑）

| 消融 | 启动方式 |
|---|---|
| 直接拼接潜槽位（原始多事实基线） | 不传 router |
| Compressor + Router + Bridge（普通内容寻址基线） | 阶段 A `--router` |
| + 属性级稳定键（无历史、无时间） | `--fst --hist-max 0`（硬覆写对照） |
| + 状态—历史分离 | `--fst --hist-max N`（N≥1） |
| 去掉时序条件寻址 | `--fst --ignore-query-time` |
| 跨槽隔离（块对角掩码等） | **未实现**（见上"刻意未采用"） |

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
| 检索 | `V9_L18_WHITEN` / `V9_L18_WHITEN_EPS` / `V9_L18_REUSE_EPI` | 1 / 1e-4 / 1 | 池内白化 / 白化正则（兼特征值下限） / 复用 epi 已算向量 |
| 扩展 | `V9_EQA_BACK` / `V9_EQA_AHEAD`（runner 用 `AR_V9_EQA_BACK/AHEAD`） | 16 / 4 | 窗口扩展；**方向未定**（配对检验 Holm p=0.2314、MDE=9.7pp） |
| 扩展 | `V9_LME_TURN_WINDOW` / `V9_MAX_PER_SESSION` / `V9_SESSION_QUOTA` | 1 / 2 / 1 | LME 同 Session turn 扩展 / 跨 Session 配额 / 配额开关（**runner 尚未接线**） |
| 组装 | `AR_V9_CTX_TOKEN_BUDGET` / `AR_V9_CTX_MAX_UNITS` | 2500 / 0 | 证据 token 预算 / 单元数上限（0=关闭，由预算决定） |
| 组装 | `AR_V9_PROMPT_DATE` / `AR_V9_PROMPT_SERIAL` | 1 / 跟随 DATE | 行前缀 `[date]` / 行前缀编号（消融用） |
| 组装 | `AR_V9_DROPPED_ID_CAP` | 200 | 预算/裁剪记账里 `dropped_ids` 的落盘上限 |
| 组装 | `AR_CTX_MAX_TOK` | 0 | 0 = 全上下文不截断（对外披露项） |
| 读出 | `AR_READOUT_MODE` | `prompt` | `prompt / text_embeds / latent`（latent 需 `AR_MODE=v9` + `LATENT_CKPT`） |
| Delta | `AR_DELTA_WRITE` / `AR_DELTA_RESTORE` / `AR_ALLOW_EMPTY_DELTA_SNAP` | 0 / 1 / 0 | 写开关 / 快照恢复 / 空快照放行 |
| Delta | `AR_DELTA_WRITE_STRATEGY` / `AR_DELTA_WRITE_TOPK` / `AR_DELTA_WRITE_CHUNK` | `all` / 12 / 512 | 写入策略（`topk_refresh` 为词面近似） / 刷新条数 / 分块大小 |
| 严格性 | `AR_STRICT_RETRIEVAL` | `auto` | auto：稠密/潜层/写入开启时检索异常直接抛 |
| 复现 | `AR_RESUME` / `AR_RESUME_MANIFEST` / `AR_OUT` | 1 / 1 / 空 | 续跑 / 校验 manifest / 显式输出路径 |
| v8 | `V8_SESSION_RESET` | 1 | Session 边界重置 `last_ts`、滑窗不跨 Session |
| CR | `V9_CR_BEAM` / `V9_CR_EGO_K` / `V9_CR_REL_THETA` / `V9_CR_HOPS` | 6 / 12 / 0.80 / 4 | 束宽、ego 上限、关系阈值、跳数 |
| CR | `V9_CR_HOP1_M` / `V9_CR_PATH_VIEW` / `V9_CR_PATHS` | 96 / 1 / 3 | Hop1 矩阵初筛候选数 / 路径式读出 / 最多路径数 |
| CR | `CR_DELTA_MODE` / `CR_DELTA_REFRESH_TOPK` | `off` / 6 | Delta 参与查询的方式（非法值 SystemExit） / 刷新条数 |
| 潜层 | `LATENT_CKPT` | 空 | 压缩器+桥接器权重路径；latent 模式下缺失直接报错退出 |
| 潜层 | `LATENT_M` / `LATENT_LAYER` / `LATENT_INNER_DIM` | 4 / 18 / 512 | 每槽软 token 数 / 取哪层 hidden_states / 注意力内宽 |
| 潜层 | `LATENT_MAX_UNIT_TOKENS` | 64 | 单元再分块宽度（**不截断**，超出切成多块并记父 ID） |
| 潜层 | `LATENT_READ_BUDGET` / `LATENT_TOKENS_PER_SLOT` | 32 / 4 | 软 token 预算 → K ≤ budget//m 个槽位（预算>0 时至少 1） |
| 潜层 | `LATENT_BANK_MAX` | 0 | >0 时只编码前 N 个槽位（smoke / 大池调试）；0=不限制 |
| 潜层 | `LATENT_DTYPE` | `bf16` | 载荷**存储**精度（不决定训练精度：Compressor/Bridge 恒 fp32） |
| 潜层 | `LATENT_D_KEY` | 256 | 寻址键宽度（`LatentRouter` 与 FST 事实键/版本键共用） |
| 潜层 | `LATENT_HIST_MAX` | 4 | **FST**：每个 (实体,属性) 键保留的历史版本数；0 = 硬覆写对照（历史全丢） |
| 潜层 | `CR_READOUT_MODE` | `prompt` | CR 生成从潜载荷构造（`latent`）还是拼事实块（`prompt`） |
| 潜层 | `AR_QTXT_NO_GOLD` | latent 下强制 1 | 查询模板不再注入金标（`{label}` 填空串，记 `query_label=empty(no_gold)`） |
| 训练 | `LATENT_TRAIN_BATCH` / `LATENT_GRAD_ACCUM` / `LATENT_LR` | 1 / 16 / 1e-4 | 训练超参（**未验证**） |

## 8. 实现状态矩阵

| 能力 | 状态 | 证据 |
|---|---|---|
| v8 三个确定性 Bug 修复 | 已实现 | `test_v8_slotpool.py` 86 断言 |
| v8 Session 重置 + 相对时间戳（yesterday / last week / 星期几） | 已实现 | `test_v8_slotpool.py`（`V8_SESSION_RESET=1`） |
| 非重叠索引 / 分数优先 / 非对称扩展 / 时序重排 | 已实现 | `test_v9_slotpool.py` 284 断言 |
| Session 边界不继承时间戳（I3） | 已实现 | `test_v9_slotpool.py::test_session_boundary_ts` |
| 第 18 层混合打分的设备对齐 | 已实现 | `test_hybrid_device_consistency`（cpu/cuda/mps 三选一） |
| 第 18 层池内 whitening + 复用 epi 向量（免二次前向） | 已实现·**效果未验证** | `test_v9_slotpool.py`（`V9_L18_WHITEN=0` 与旧口径逐位一致；按文本查表不串行号） |
| 第 18 层稠密通道**效果** | 未验证 | 单测用 stub provider；真实前向未跑 |
| 状态池同主语原位覆写（槽 id 稳定） | 已实现 | `test_cr_overwrite_finds_same_subject_slot`（旧实现必失败） |
| CR 属性类别门控（不同属性不得判为同一谓词） | 已实现·**双向风险未量化** | `test_v9_slotpool.py`（doctor/vegetarian、born in/born on） |
| 状态池 O(N) 写入（主语索引，非全池扫描） | 已实现 | `test_cr_pending_merge_and_stable_ids` |
| CR Hop1 矩阵初筛（P1 性能） | 已实现 | `test_hop1_matrix_prefilter`（120 槽 → 96 候选，gold 不丢） |
| CR 路径式读出（seed→mid→target + 其余事实） | 已实现·未跑分 | `test_path_view_and_flat_fallback` |
| CR Delta 参与查询（`CR_DELTA_MODE=off\|context\|refresh`） | 已实现·**效果未验证** | `test_cr_delta_log.py` 84 断言；三档指纹互异 |
| 潜层读出：软 token 注入通路（位置/mask/拼接/位置编码） | 已实现·**真机验证通过** | `tools/diag_latent_capacity.py --sub oracle` 4/4（记忆原文 embedding 直接当软 token） |
| 潜层读出：压缩器**能否编码内容** | **未达成**（真实负结果） | 重建 token-acc 22.8% vs 平凡基线 top10 19.8%；QA 0/32、200 条过拟合 1/200；加宽 m 无改善。见 `results/latent_stage_a_2026-10-21/` |
| 潜层读出：**单槽**通路（1 事实 → 1 槽） | **已达成** | 严格全匹配 15/16、反事实配对 7/8（`results/latent_router_2026-10-27/` 第一节） |
| 潜层读出：**多槽**读出（K 个槽裸拼） | **已证伪（位置效应，非内容寻址）** | 同一事实：单槽 15/16 → 4 槽 target_only 15/16 → 4 槽 full 1/16 → 4 槽 target_end **0/16**；4 槽 ckpt 在 target_only 上 **0/200**；验证 17.2% < relation-prior 基线 31.2% |
| 潜层读出：**集合内容寻址**（`LatentRouter`） | 已实现·**未训练** | 机制验证通过（形状/梯度/ckpt 往返/单测 89 项）；`s_k=<q,k_in_k>/τ`、推理 argmax 只送 1 槽、`chain()` 多跳。**效果未知** |
| FST · 事实键（实体—属性级稳定寻址） | 已实现·**未训练** | `latent_memory.LatentFactKeyEncoder`；`test_fst_mem.py` F/G 组（键与值无关、实体/属性不可交换、梯度到四头） |
| FST · 状态—轨迹解耦库 | 已实现·**未训练** | `LatentStateTrajectoryBank`；单测 H 组：旧值下沉、**改 A 不动 B（逐位相等）**、容量淘汰真释放、按时间读版本 |
| FST · 时序两级读出 | 已实现·**未训练** | `FstRouter`；单测 I 组：与组数/顺序无关、版本候选只在组内、掩码、STE 前向逐位等于硬选 |
| FST · 训练与评估入口 | 已实现·**仅 CPU 通路验证** | `--fst`（`train_fst`）；评估报 acc/group/ver_multi/三探针/pair/order_stable/n_soft；`results/fst_mem_2026-10-27/` 有 smoke 日志 |
| FST · 动态事实数据（同实体多属性 + 多次覆写 + 时间条件） | 已实现·**真实产出** | `tools/gen_fst_update.py`：192 训练 + 72 验证，实体不相交，三探针各 64 条 |
| FST · 接进 AR/CR 线上管线 | **未接线** | 需从 `StatePool`/`EpisodicMemory` 构造事实键与时间线，属阶段 C |
| 逐题诊断字段（used_texts / 跨池统一 ID `epi:/state:/cr:`） | 已实现 | `ar_diag --selftest` 121 项 |
| 四格联合归因（Reader Acc \| Recall=1 等） | 已实现·**仅合成验证** | `ar_diag --selftest` 121 项；仓库内无带 `diag` 字段的真实 jsonl |
| 配置指纹 / 结果文件隔离 / 续跑校验 | 已实现 | `test_run_manifest.py` 27 断言 |
| 训练语料生成（虚构事实宇宙 → 三元组 → 单跳/多跳，32 进程并发） | 已实现·**真实执行** | 15,000 训练 + 2,000 验证；唯一 (问,答) 15000/15000，train/val 实体交集 0 |
| 训练脚本消费外部语料（`--data/--val-data`，一事实一槽位） | 已实现·**仅 CPU 通路验证** | `tests/test_synth_data.py` 32 断言；`--smoke-model` 阶段 A/B 均 OK；**GPU 训练未跑** |
| Delta 空快照必须报错 | 已实现 | `test_run_manifest.py::test_empty_delta_snapshot_raises` |
| Delta 有效写入策略（`all` / `filtered` / `topk_refresh`） | 部分实现 | `test_ar_delta_write_strategy.py` 47 断言；`topk_refresh` 是词面 IDF 近似 |
| Delta 写入**效果** | 未验证 | 默认仍 `AR_DELTA_WRITE=0`，三档策略无对照 |
| 时间进生成 prompt（`<serial>. [<date>] <原文>`） | 已实现·收益未验证 | `test_ar_v9_wiring.py`（`AR_V9_PROMPT_DATE=0` 逐字节等于旧输出） |
| 2500 token 证据预算 + OOM 按证据优先级裁剪 | 已实现 | `test_ar_v9_wiring.py` 105 / `test_ar_delta_alignment.py` 37 断言 |
| LongMemEval Session/Turn 扩展 + 跨 Session Max-2 + first/last/before/after | 已实现·**待接线** | `test_v9_slotpool.py` LME 用例；runner 仍走 `mode="topk"` → 线上不生效 |
| LongMemEval 官方 Judge（http / local 双后端） | 已实现·未判题 | `--selftest` 76 项，`--dry-run` 零网络；`--compare-backends` 验等价性 |
| Delta 写入顺序 / 快照恢复 | 已接线·未验证 | `test_ar_delta_alignment.py` 37 断言 |
| `inputs_embeds` 读出通路 | 已接线·未验证 | 明文嵌入，**不是**潜槽位 |
| **槽位存潜向量（附加，不替换明文）** | 已实现·**未训练** | `latent_memory.LatentSlotBank`：以稳定槽位 ID 为键存 `[k,m,d]`；`test_latent_readout.py` 库语义用例 |
| **潜层读出（latent readout，路径 B）** | 已实现·**未训练 / 未跑分** | `AR_READOUT_MODE=latent`；明文进不去（签名 + tokenizer 文本双重断言）、软 token = `bridge(Z)`、梯度到 Compressor/Bridge |
| 长单元再分块（不截断） | 已实现 | `test_latent_readout.py`：`Σn_tokens` 守恒、span 连续、父 ID 一致 |
| CR 多跳路径预算（整条保留/整条跳过） | 已实现 | `test_latent_readout.py`：不出现半条路径、被跳过路径的 facts 不得补位 |
| checkpoint 进配置指纹 | 已实现 | `run_manifest.file_fingerprint()`；缺失/隐藏层不一致/未给路径三种情况均报错 |
| EventQA 窗口方向验证（back/ahead） | 不成立（指标失效） | 金标为改写句、不在原文中，见 `experiments/eventqa_window_sweep.md` |
| 证据级召回指标 | **缺失** | 无金标 span，`evidence_recall_at_k` 恒 None；EventQA 无法算 |
| 可学习门控（何时写/何时读） | 规划中 | 无实现 |

## 9. 已知风险（必须在论文中披露）

1. **稠密通道依赖冻结模型第 18 层**，层号与 `α=0.3` 都是拍定的，无消融。
2. **Delta 状态 + rank=8 冻结适配器**对细粒度问答是否有正贡献，**无证据**；
   若为负，架构主张须退回路径 B（路径 B 已实现，但**未训练未跑分**，见风险 11）。
3. **CR 与 AR 池语义不同**（覆写 vs 追加），统一为"双模态"是设计意图，
   缺跨任务族一致性验证。
4. **v9 没有任何指标**：全部结论来自 dry-run 与 CPU 单测；v7/v8 历史数字不得当作 v9 效果。
5. `ruler_qa1_197K` / `ruler_qa2_421K` 需要 choice 专用指标，runner 显式拒绝，覆盖不全。
6. `answer_recall_at_k` 是**答案字符串命中代理**（gold answer 是否出现在第 k 个槽位），
   **不是**证据级召回；本管线没有金标证据 span，`evidence_recall_at_k` 恒为 None。
7. **EventQA 的 `answer_in_context` / `anchor_answer_hit` 口径失效**（2026-10-21 实测）：
   金标答案句在 285K 字符原文中**不存在**（选项是生成式改写），750 条记录命中 0/750。
   任何基于"答案是否落在证据窗口"的结论都不可用；事件级召回必须换数据或换探针。
8. **端到端 SF 是 6 选 1 任务分**，混着生成能力与选项风格。原"SF 对检索内容敏感、是有效探针"的说法
   **已撤回**：那是**未配对**样本造成的假象。配对后同 52 题 1 句窗口 0.6346 vs 默认 0.6731（−3.85pp，p=0.625）、
   同 23 题 ~350 句宽窗 0.7826 vs 默认 0.6957（+8.70pp，p=0.500），**均不显著**；n=100 时 MDE=9.7pp。
   结论：**现有数据无法证明 SF 能区分检索好坏**，需换自带金标 span 的合成探针（`evidence_recall@K`）。
   另："默认 21 句 = 0.810"是笔误，默认 b16_a4 为 **0.730**（0.810 属 b0_a20 / b4_a16）。
9. **Delta 写入策略的 `topk_refresh` 名不副实**：AR 侧写入先于建池、拿不到真实检索结果，只能用问题词面
   IDF 近似；CR 侧 `CR_DELTA_MODE=refresh` 才是真实检索 Top-K 刷新。两者**不可横向比较**。
10. **块内 serial 可能回退**：拼块是"情景行在前、状态行在后"，而每行 serial 取绝对位置，故块尾状态行的
    serial 可能小于前面的情景行，与反事实 prompt 的"编号越大越新"约定冲突。有 `[date]` 前缀时影响小，
    无日期时会让旧状态看起来最新。修它需同时改 `ar_diag` 的对齐位置约定，**本轮未修**。
11. **潜层读出的软 token 通路无效果证据**（v9.3）：压缩器/桥接器是随机初始化的，
    没有 `LATENT_CKPT` 时 runner 直接报错退出；即便训练，也没有任何真实数据上的对照跑分。
    **2026-10-27 补充**：单槽通路已达成（15/16），但**多槽读出被证伪**——失败原因是纯位置效应
    （K 个槽的 K×m 个软 token 裸拼时，冻结 Qwen 只能按位置注意，"第几个槽"而非"内容"决定读出）。
    `LatentRouter` 是针对此的结构性修法，**尚未训练，不能声称有效**。
    另注：`delta_write_disabled` 退出时会**永久关闭** Delta 写入（见 `KNOWN_ISSUES.md` P0-1），
    任何同一进程内既要写又要禁写的路径都会被静默破坏。
    另外 `answer_in_context` 在 latent 模式下是 **None**（`used_texts` 保持空）而非 `False`，
    检索侧命中率与读出准确率必须分开报，不能拿明文匹配当"答案进了上下文"。
12. **CR 两处硬门控是保守选择**（t19 已量化，`experiments/results/cr_link_recall.json`）：
    - 属性类别门控 `_attr_compatible`：槽级真 stale **0/260 = 0.00%**（同主语同属性下池中没有并存多值的活槽），
      pair 级属性冲突率亦 0.00% → **未超阈，不加归一**。子串粗代理 61.7% 是噪声，不可引用。
    - 2 跳 `linked()` 要求 `obj(f1) == subj(f2)` 归一化精确相等：失败率 mh_6k 0.00%（判定样本 10 条）、
      sh_6k 8.33%（36 条）——但**覆盖率仅 6% / 17%**（172/207 条需链题目在解析层找不到桥接候选），
      样本不足 → **未定论**，不据此加别名归一。要定论需 sh/mh_262k 切片（本机未落盘）。

13. **FST-Mem 目前只有机制保证，没有效果证据。** 已实现的是"结构上做对了"：
    键与值无关、更新不动其他键（逐位断言）、两级竞争隔离、前向硬选 + STE。
    **未做的事**：真实训练一步没跑；`hops>1` 未实现（`generate_latent_answer_fst` 显式报错）；
    未接进 AR/CR 线上管线；历史容量 `LATENT_HIST_MAX` 的容量—保真权衡未量化；
    `stack_payloads` 对多块槽位取块均值仍是潜伏问题（事实 ≤9 token ≪ 64，实际 `k=1`）；
    阶段 A 的 22.5% 数据歧义缺陷在新数据集里已规避（干扰项键不与目标键冲突），
    但**旧数据集未重生成**。任何"FST 解决了组合不稳定/更新局部性"的说法在跑出对照前都不成立。

## 10. 演进路线

| 阶段 | 目标 | 退出条件 |
|---|---|---|
| P0（已完成） | 修 Bug、补诊断、对齐评测口径、结果可复现 | 单测全绿 + dry-run |
| P1（待跑分） | v9 检索在 EventQA 全切片与 LongMemEval 上对照 v7 | 四切片 + 召回归因表 |
| P2（待跑分） | CR `CR_MODE=v9` 对照 262K 多跳 | sh/mh 不掉点，mh_262k 提升 |
| P3 | 路径 A 消融：`AR_DELTA_WRITE=1` vs 0 | 同切片 A/B |
| P4 | **潜层读出**：槽位向量接读出头（A/B/C 三档） | 摆脱"明文 RAG"质疑 |
| P4-A | 线性 probe：top-k 槽向量池化 → 与候选答案 embedding 比相似度（零生成） | 回答"槽向量里有没有答案信息"；若 <0.5 则表征层需重做，B 不必做 |
| P4-B（**代码已实现，未训练未跑分**） | soft prefix：槽位 `[h0;h18]` → m 个可学习 query 交叉注意压成 `Z∈R^{m×d}` → 桥接器变软 token 拼进 `inputs_embeds` | 生成式潜层读出；退出条件未达成（需先训练 + 与 `prompt` / `text_embeds` 三臂对照） |
| P4-C | cross-attention：问题向量当 query attend 全部槽向量 | 按需寻址；需改 attention + KV cache，最后做 |
| P5 | 可学习写入/读出策略（替代手工阈值 0.80 / 0.92） | 与手工阈值对照 |

## 11. 与旧版本的关系

- **CR 默认路径仍是 v7**：`CR_MODE=v7` 时 `run_group_b2_v7.py` 行为与历史一致。
- **v8 保留**为可对照实现，三个 Bug 已原地修复；历史四切片数字是修复前产出，未重跑。
- **v9 是新主线**，绝大多数新能力靠 env 开关、默认关闭。**但收口轮有两处默认值变化**，
  对比旧 v9 跑分前必须显式回到旧口径：
  - `AR_V9_CTX_MAX_UNITS` 默认 **40 → 0**，拼块改由 `AR_V9_CTX_TOKEN_BUDGET=2500` 决定；
    复现旧拼块口径设 `AR_V9_CTX_TOKEN_BUDGET=0`。
  - `V9_L18_WHITEN` 默认 **1**（仅 `AR_V9_L18=1` 时生效）；复现旧 L18 口径设 `=0`（逐位一致，有断言）。
  - 两处都已进 `run_manifest` 配置指纹，不同口径不会写进同一个结果文件。
- **v9 CR 的路径式读出默认开启**（`V9_CR_PATH_VIEW=1`）：因为 v9 CR 从未跑过分，
  不存在与历史数字的可比性问题；若要对照旧版平铺输出，设 `V9_CR_PATH_VIEW=0`。
