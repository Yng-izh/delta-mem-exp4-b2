# 架构设计：DM-ELSM（双模态弹性潜槽位记忆）

> 本文区分两件事：**已经写进代码并有单测的能力**，和**架构上承诺但尚未实现的目标**。
> 每个组件都带状态标记：`已实现` / `已接线·未验证` / `规划中`。
> 指标与实验结论见 `README.md`、`CHANGELOG.md`、`results/RESULTS.md`；本文只谈结构与契约。
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
> 状态/轨迹解耦 + 时序两级读出**（§6.5）。代码与机制单测齐备（`tests/test_fst_mem.py` 88 项、`tests/test_train_objectives.py` 14 项覆盖损失装配与 Delta 写入状态恢复、`tests/test_prep_hotpotqa.py` 14 项 + `tests/test_prep_mab.py` 22 项覆盖公开数据转换，变异审计验证断言有约束力），
> 训练入口 `--fst` 在 CPU 极小模型上跑通全链路（训练→评估→存盘→重载→评估）。
> **同样一步真实训练都没跑**，没有任何效果数字；`LatentRouter` 的定位从"候选方法"
> 下调为"静态多事实内容寻址基线"。详见 `results/fst_mem_2026-10-27/`。

---

## 1. 一句话定位

把上下文压成**一事实一槽**的潜层记忆（每槽 `[k,m,d]` 载荷），按问题寻址后以
**4~9 个软 token** 注入冻结 LLM 生成 —— 与明文 Text-RAG / 无记忆地板在同一条
生成路径上对照。方法分四组：Text-RAG / Compressor+Bridge / Compressor+Router+Bridge /
FST-Mem（两级寻址 + 状态—历史解耦）。评测用公开基准（HotpotQA、MemoryAgentBench-CR），
结果表在 `results/RESULTS.md`。

## 2. 三层结构

```
① 记忆单元（一事实一槽）
   来源：tools/prep_hotpotqa.py（HotpotQA 句级）
        tools/prep_mab.py（MAB-CR 事实级，unit_meta 带事实键/时间）
        tools/gen_fst_update.py（FST smoke 数据，仅 smoke）
   契约：非重叠、一事实一槽、时间/编号是数据

        ▼
② 槽位编码（Compressor）
   每槽 [k, m, d] 载荷（k=寻址键，m×d=内容），bf16
   整段 recon 只取 --recon-units N 个随机单元（长记忆 OOM 修复）

        ▼
③ 寻址与读出（三档，按论文方法分组）
   a. Bridge：全部槽位 → 软 token 裸拼（Compressor+Bridge）
   b. Router：按问题选 1 槽 → bridge → 4 软 token（Compressor+Router+Bridge）
   c. FST-Mem：两级寻址（事实键 entity/relation → 组内版本）
      + 状态—轨迹解耦（新值覆盖旧值、历史下沉保留）

        ▼
④ 注入与生成
   soft token 经 bridge 范数对齐后拼进 inputs_embeds（system 段之后）
   明文基线（text/oracle/none）走同一条 _emit_latent 路径，soft=None
   —— 四组方法唯一的差异是"记忆怎么来"，生成与判定完全同协议
```

## 3. 设计不变量（invariants）

每条都对应一个真实修过的 Bug 或一次真实翻车。

| # | 不变量 | 反例（已修复/已证伪） |
|---|---|---|
| I1 | **Delta 写入开关必须逐模块快照/恢复** | 旧实现 finally 一律 `fn(model,False)`，第一次使用后写入被进程级永久关闭，跑分照出、全错无声 |
| I2 | **损失只能累加**：KL/寻址/重建不得互相覆盖 | 旧 KL 分支 `loss = loss_qa + kl*loss_kl` 把寻址损失整条覆盖，日志照样打印 loss_addr |
| I3 | **每条样本前必须 reset Delta 在线状态** | 不 reset：显存慢涨 + 样本互污，两者都无报错 |
| I4 | **训练事实必须独立于评测集** | 不得用 MAB/LME/HotpotQA 评测切片训练；HotpotQA train(90447)/validation(7405) 官方本就分离 |
| I5 | **多跳样本的变槽数课程必须保留证据链、丢其余** | 旧 keep_all 在 23 槽真实数据上等于关课，Router 停在 chance（dev 5% < 无记忆 12%） |
| I6 | **软 token 必须与真实 token embedding 同尺度** | bridge 未约束时范数 96–159 vs 真实 1.05，软 token 变 attention sink |
| I7 | **寻址标签优先 evidence 原文匹配，退化才用答案串** | 干扰事实常含同值（旧合成数据实测 22.5% 样本答案值多槽出现），纯答案串匹配 13.5% 标错槽 |
| I8 | **答案在上下文里是数据集性质不是缺陷**（HotpotQA 96.3%） | 必须 always 报 none 地板与读对槽位列，否则"蒙对"会冒充记忆读出 |

## 4. 数据流（含设备边界）

```
写入：units ─► encode_item_slots（逐槽 [k,m,d]）─► 槽位库 / FST 轨迹库
读出：question ─► query_feature(第18层) ─► Router/FstRouter 选槽
        ─► bridge(Z) 范数对齐 ─► soft tokens
        ─► _emit_latent(model, soft=...) ─► generate ─► _norm_ans 全匹配
明文：units/top-k 句子 ─► 同一 _emit_latent(soft=None) ─► 同一 generate
```

设备约定：压缩器/路由在 GPU bf16；槽位载荷进 FST 轨迹库时转 fp32（训练要回传梯度，
bf16 往返会引入无意义舍入）；Delta 在线状态每样本 reset（I3）。

## 5. 组件契约

| 组件 | 位置 | 职责 | 状态 |
|---|---|---|---|
| 槽位库 / 载荷栈 | `latent_memory.LatentSlotBank` / `stack_payloads` | 以稳定槽 ID 存 `[k,m,d]`；变槽数拼装 | 已实现 |
| Compressor+Bridge | `latent_memory` + `train_latent_memory.py` | 每事实 → m×d 潜码 → 软 token | 已实现·已跑分 |
| LatentRouter | `latent_memory.LatentRouter` | `s_k=<q,k>/τ` 内容寻址；推理 argmax 只送 1 槽 | 已实现·已训练（acc 停在 chance，瓶颈=寻址） |
| FST 状态—轨迹库 | `latent_memory.LatentStateTrajectoryBank` | 旧值下沉、改 A 不动 B（逐位相等）、容量淘汰、按时间读版本 | 已实现·已训练（group 0.650 / ver 0.525 / order_stable 1.0） |
| FstRouter | `latent_memory.FstRouter` | 两级：组 softmax + 组内版本 softmax，掩码 -1e4，STE | 已实现·已训练（版本级输给"恒取最新"0.885） |
| 统一评测 | `eval_latent_table.py` | 六方法同路径；acc/read_acc/pair/update/hist/token/ms 七列汇总 | 已实现·已跑分 |
| 数据转换 | `tools/prep_hotpotqa.py` / `prep_mab.py` | 公开数据 → 一事实一槽 jsonl + meta（leak/cap/gold） | 已实现·已跑分 |
| Delta 接口 | `deltamem_compat` + `reset_delta_state` | adapter 挂载、在线状态逐样本恢复 | 已实现 |
| checkpoint 指纹 | `latent_memory.checkpoint_fingerprint` | 路径+字节+sha256 前 16 位（进 manifest/日志） | 已实现 |

## 6. 读出层的三种档位（诚实边界）

| 档位 | 入口 | 实际发生什么 | 潜层？ |
|---|---|---|---|
| Bridge | `eval_latent_table.py --method bridge` | 全部槽位载荷 → bridge → K×m 软 token 裸拼 | 是，但多槽裸拼已被证伪（位置效应） |
| Router | `--method router` | 问题特征选 1 槽 → bridge → 4 软 token | 是；acc 当前停在 chance，瓶颈=寻址 |
| FST-Mem | `--method fst` | 两级寻址（事实键→版本）+ 状态—历史解耦，读 1 槽 | 是；group 超平凡基线、ver 未学到 recency 先验 |
| 明文对照 | `--method none/text/oracle` | 同一条 `_emit_latent` 路径，soft=None | 否（地板/对手/上界） |

判定与生成对四组方法完全一致（同冻结模型、同 system prompt、同 `_norm_ans` 全匹配）；
方法间唯一差异是"记忆怎么表示、怎么进上下文"。

## 6.4 路径 B 细节（潜层通路的机制约束）

```
槽位原文 ──冻结 LLM──> [h0; h18] token 级 [L,d]
             │  LatentCompressor：m 个可学习 query 做 cross-attention
             ▼
       Z ∈ R^{m×d}（每槽 m 个软向量）──存入槽位库（键=稳定槽 ID）
             │  读出：Router/FstRouter 选槽 → LatentBridge（范数对齐）
             ▼
     [k·m, d] 软 token 嵌入 ──拼进 inputs_embeds（system 之后、query 之前）
```

- **分块不截断**：单元长于 `LATENT_MAX_UNIT_TOKENS=64` 时切不重叠块并记父 ID。
- **明文进不去**：latent 读出签名 + tokenizer 文本双重断言。
- **严格加载**：`LATENT_CKPT` 缺失直接报错，不拿随机投影跑对照。
- **多槽裸拼已证伪**（2026-10-27）：失败原因是纯位置效应——冻结模型只按"第几个槽"注意，
  不按内容。Router/FST 是针对此的结构性修法。

## 6.5 FST-Mem：状态—轨迹解耦的潜层记忆（已实现 · 已在 MAB-CR 上训练）

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

### 6.5.1 静默失效清单（核查轮固定下来的不变量）

这些错**都不会报错**，只会让指标悄悄变差或虚高。每条都对应一个单测断言和一个
变异审计抽查（变异存活即判失败；工具随废案移除，方法保留在测试设计里）。

| 不变量 | 错了会怎样 | 断言 |
|---|---|---|
| **载荷在 STE 软期望里必须 detach** | 梯度按 `wg⊗wv` 铺到全部 G×V 条载荷，压缩器被迫同时满足所有事实；前向数值**完全正确**，只有梯度被污染 | I11：STE 载荷梯度必须与 `hard=True` 逐位相同（测前需切断"版本打分吃载荷"这一路） |
| **版本分布必须每组各算一次 softmax** | 把选中组的 `wv` 套到所有组上，软期望不再是任何一致的期望量 | I9 + STE 前向 `torch.equal` |
| **训练侧与读出侧的实体/属性文本必须同源** | 训练用原始大小写、读出从（小写）事实键反推 → 评估喂分布外输入 | M5/M6/M6b |
| **版本级命中必须绑组别** | `ver_picked` 是"被选中组内"的下标，与目标组下标比较会在组选错时误记命中 | N1 |
| **单版本目标组不进版本级分母** | 选谁都对，`ver_multi_acc` 被恒真样本虚抬 | N2 |
| **配对必须有 ≥2 成员且 gold 不同** | `all([])`/`all([x])` 恒真；`unrelated` 探针两变体 gold 相同也不含反事实信息 | N3 |
| **并列时间戳取最新** | `bank.read()` 取最新的，监督取最旧 → 监督指向永远不会被读的槽位 | N5 |
| **填充版本必须掩码（`-1e4`）** | 零向量载荷可能进 Reader | L6 |
| **`gather_all_versions` 组序必须与 `group_keys` 对齐** | 读到**另一条属性**的载荷，而 meta 照旧报对的键 | L2/L5 |
| **淘汰与缺标签必须分开** | `hist_max` 调小时分不清"容量机制"与"数据有洞" | M8/M9 |

## 7. 关键参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `--m` | 4 | 每槽软 token 数（潜层记忆的"读取带宽"） |
| `--layer` | 18 | 取哪一层隐状态做压缩/查询特征 |
| `--inner-dim` / `--heads` / `--bottleneck` | 512 / 8 / 512 | Compressor 结构 |
| `--max-unit-tokens` | 64 | 单元超长切块上限（不截断） |
| `--max-slots` | 8 | 变槽数课程上界（must_keep 之外随机丢） |
| `--recon-weight` / `--recon-units` | 0 / 2 | 整段重建损失权重与参与单元数（长记忆 OOM 修复） |
| `--addr-weight` / `--ver-weight` | 1.0 / 1.0 | 组级 / 版本级寻址监督权重 |
| `--hist-max` | 4 | FST 每键保留的版本数（历史容量） |
| `--d-key` | 256 | FST 事实键维度 |
| `--read-budget` | 32 | 读出 token 预算 |
| `--text-topk` | 4 | 明文 BM25 取回数（text 基线） |
| `--cap`（prep_mab） | 32 | MAB-CR 每题可见事实数（必须随表报告） |
## 8. 实现状态矩阵

| 能力 | 状态 | 证据 |
|---|---|---|
| Delta 写入状态恢复（逐模块快照/恢复 write_enabled） | 已实现·已修复 | `test_train_objectives.py` Q1–Q6（旧实现 finally 一律关闭写入，进程级静默失效） |
| 损失组合 `assemble_loss`（KL/寻址/重建只累加不覆盖） | 已实现·已修复 | `test_train_objectives.py` P1–P8（旧 KL 分支覆盖寻址损失） |
| 变槽数课程对多跳样本生效（must_keep 保证据链） | 已实现·已修复 | `test_train_objectives.py` R1–R7（旧 keep_all 在 23 槽真实数据上等于关课） |
| 潜层读出：软 token 注入通路 | 已实现·真机验证 | `tools/diag_latent_capacity.py --sub oracle` 4/4 |
| 潜层读出：单槽通路（1 事实 → 1 槽） | 已达成 | 严格全匹配 15/16、反事实配对 7/8（`results/latent_router_2026-10-27/`） |
| 潜层读出：多槽裸拼 | 已证伪（位置效应） | 同一事实：单槽 15/16 → 4 槽 full 1/16 → target_end 0/16；val 17.2% < relation-prior 31.2% |
| Compressor 内容编码（整段 recon） | 真实负结果 | token-acc 22.8% vs 平凡基线 19.8%；QA 0/32；加宽 m 无改善。见 `results/latent_stage_a_2026-10-21/` |
| 潜层四方法统一评测入口 | 已实现·已跑分 | `eval_latent_table.py`（none/text/oracle/bridge/router/fst 同一条生成路径）；`results/RESULTS.md` |
| HotpotQA 真实数据接入 | 已实现·已跑分 | `tools/prep_hotpotqa.py`（train 90447/val 7405，8 个干扰×顺序对照集）；`test_prep_hotpotqa.py` 14 项 |
| MAB-CR 真实数据接入 | 已实现·已跑分 | `tools/prep_mab.py`（800 题 cap=32，76.8% 含冲突版本）；`test_prep_mab.py` 24 项 |
| FST · 事实键 / 状态—轨迹库 / 两级读出 | 已实现·**已训练（MAB-CR 300 题）** | `test_fst_mem.py` 88 项；`results/mab_cr_2026-10-27/`：group 0.650>0.550、ver 0.525<恒取最新 0.885、order_stable 1.0、read_acc 65% vs Router 5% |
| FST · 两跳链式读出 | **未接入** | `generate_latent_answer_fst` 硬编码 hops=1；CR 题多为两跳 → 当前 CR 潜层列是不对等比较 |
| LongMemEval 接入 | **未接入（容量不匹配）** | 500 题 cleaned 已在服务器；会话级 ~115k token vs 64-token 槽位，需分块两级读出或写成局限 |
| 训练峰值显存与并发争用区分 | 已实现 | 每步 `peakmem=`；`tools/mem_probe.py` |
| 逐题诊断字段 / 四格归因 | 随旧 AR 管线废除 | 历史见 CHANGELOG v9.6 及更早 |

## 9. 已知风险（必须在论文中披露）

1. **潜层 acc 尚未超过明文基线**（17.5% vs 36.0%）：瓶颈定位在寻址（bridge 读对槽位 35%
   vs 明文检索命中 97%，oracle 58% 证明读出端无碍）。
2. **FST 版本级输给"恒取最新"**（0.525 vs 0.885）：300 训练题内没学到 recency 先验；
   组级（0.650>0.550）与顺序稳定（1.0）是结构上成立的证据。
3. **CR 潜层列是不对等比较**：CR 题多为两跳链，潜层每次只读 1 槽；oracle 给 1 条金标也只有 32%。
   FST 两跳链式读出未接入前，这一列只能与"读对槽位"一起解读。
4. **层号 18 与超参（m=4、α 等）是拍定的**，无消融。
5. **答案在上下文是数据集性质**（HotpotQA 96.3%）：只报 acc 会把"字符串检索"当记忆读出；
   必须 always 同时报 none 地板与读对槽位列。
6. **共享 GPU 争用**：每步 `peakmem=` 区分自身增长与他人进程，引用显存数字前先核对。
7. **LongMemEval 未接入**：115k token 会话 vs 64-token 槽位，容量不匹配；接入前不能声称覆盖 LME。
8. **潜层查询开销更高**（FST 11.0s vs text-top4 1.5s）：32 槽逐条编码线性增长，
   报"读对槽位"优势时必须同时报 ms。
## 10. 演进路线

| 阶段 | 目标 | 退出条件 |
|---|---|---|
| P0（已完成） | 公开数据接入 + 修复污染结果的实现问题 + 统一四方法评测 | 6 个测试文件 263 项全绿；HotpotQA/MAB-CR 基线表落地（`results/RESULTS.md`） |
| P1（进行中） | 真实数据训练：HotpotQA Router（课程修复后）/ MAB-CR Router 与 FST | 潜层行填入结果表；当前信号：acc 停在 chance 附近，瓶颈定位到寻址 |
| P2 | 抗干扰扫描（dis0/2/4/8 × natural/shuffled）在训练后的 ckpt 上重跑 | 顺序轴稳定（已验证 17.5=17.5）+ 干扰数量轴曲线 |
| P3 | FST 两跳链式读出（`generate_latent_answer_fst` 支持 hops=2） | MAB-CR 潜层列变为对等比较 |
| P4 | LongMemEval：分块两级读出或明确写成局限 | knowledge-update 78 题给出 update/hist 两列 |
| P5 | hist-max 扫描（1/2/4/8）：历史容量与当前值准确率权衡 | FST 相对 StatePool 的核心卖点量化 |

## 11. 已废除方案（代码已移除，历史见 CHANGELOG）

| 方案 | 废除原因 |
|---|---|
| v6/v7/v8/v9_slotpool、v9_cr、run_group_b / b2 / b2_v6 / b2_v7、run_ar_v7 | 自造数据管线（EventQA 窗口、262K 合成事实等）不支撑论文核心结论；主线收束为三个问题后由 `train_latent_memory.py` + `eval_latent_table.py` 取代。考古可查 git 历史（≤ e290382） |
| run_manifest / 配置指纹隔离 | 只服务上述 runner 的结果文件管理；指纹逻辑内联进 `latent_memory.checkpoint_fingerprint` |
| ar_diag / 逐题诊断字段、四格归因 | 服务已废 AR 管线 |
| eval_longmemeval_judge | LME 未接入；接入时按当前评测协议重写（旧 judge 是另一套口径） |
| diag1_golden_chunk / diag2_sim_dist / smoke_b2 | 一次性诊断脚本，结论已固化进文档 |
| tools/gen_fact_universe、gen_latent_exp2/min16、summarize_sweep、audit_mutations、demo_facts | 合成数据生成器/审计，玩具数据只留 smoke（fst_update + gen_synth_qa 仍在） |
| synth/latent_exp2_*、synth/latent_min16 | 对应废案的数据 |
| results/ 废案产物：groupA/groupB2/v6/v7groupB2_*.jsonl、summary_ar_v8.md、diag2_sim_summary.json、results/archive/、latent_min16_exp2_2026-10-24/ | 旧 B2/AR 管线的跑分明细，不支撑当前四方法结论；考古可查 git 历史 |
| logs/ 根下 b2_*/dryrun_*/smoke_* 日志与 .pid/.start 文件 | 旧 B2 管线的训练日志，仅 v6/v7 runner 使用 |
