# CHANGELOG

## [v9.2-l18] — 2026-10-21 — 第 18 层稠密通道：池内 whitening（默认开）+ 复用流式中间态

**一句话**：L18 稠密通道从"纯 L2 余弦"升级为"池内中心化 + 协方差白化"，并把调用方
已算好的向量做成可按文本复用的预计算矩阵 —— **默认行为变化**（见下），旧口径保留为消融对照。

### 默认行为变化（AR_V9_L18=1 时）

| 项 | 旧（v9.0/v9.1） | 新（v9.2，默认） |
|---|---|---|
| 稠密余弦口径 | 原始隐空间 L2 归一化后点积 | 池内中心化 + 白化后再点积（`V9_L18_WHITEN=1`） |
| 池内两两余弦 | 被"共同方向"抬高（合成数据实测均值 0.86、max 0.98） | 均值≈0、分布显著分散（排序才有区分度） |
| 状态池向量 | 对重叠文本再前向一次 | 复用 epi 池已算好的向量（`V9_L18_REUSE_EPI=1`） |

- `V9_L18_WHITEN=0` 恢复旧的纯 L2 行为（逐位一致，已加断言）；`V9_L18_WHITEN_EPS`
  （默认 1e-4）同时是协方差特征值下限，保证数值稳定。
- 数值稳定性：`d>=N` 或 `N<=1`（协方差秩不足，白化会放大噪声）自动退化为**只中心化**，
  并把实际走的模式写进 `layer18_spec()` / `l18_whiten_fit()` 的 `info.mode`。
- 复用接口：`set_layer18_provider(fn, precomputed=[N,d], precomputed_texts=[...])` /
  `set_layer18_precomputed(hidden, texts=...)`。查表按**文本**匹配，因此 epi 池与状态池
  共享同一份矩阵也不会串行号；没有预计算时退回按需前向并**告警一次**。
- 指纹：`l18_whiten / l18_whiten_eps / l18_reuse_epi` 已进 `run_manifest` 的 config_id，
  WHITEN=0/1 两次跑不会撞同一个 id（否则续跑会把两种口径混进一个文件）。

### 测试

`test_v9_slotpool.py` 249 → 284 条断言（新增 6 个用例 / 35 条断言）：whitening 前后余弦
分布变化、`d>N`/`d==N`/`N<=1` 退化路径、`WHITEN=0` 与旧口径逐位一致、预计算矩阵命中时
provider 零调用（计数器）、按文本查表不串行号、状态池复用 epi 向量（开/关对照）。

**未做的部分（如实披露）**：本轮无 GPU 机器，whitening 的**效果**只在合成/CPU 单测上验证，
EventQA/LongMemEval 上的真实指标对照尚未跑；`V9_L18_WHITEN` 默认改为 1 属于口径变更，
与旧跑分不可直接横比，跨口径比较请显式设 `V9_L18_WHITEN=0` 复现旧数字。

## [v9.2-audit] — 2026-10-21 — 20 项审计逐条核验表

逐条核验用户 2026-10-21 审计报告的 20 项要求（代码 / 测试 / 文档三方对齐），产出
`experiments/audit_2026-10-21.md`：20 行核验表（每行带 `文件:行` + 测试名）+ 状态统计
（已修 12 / 部分修 2 / 仅合成验证 1 / 待接线 1 / 已知风险 1 / 方向未定 1 / 未修 1 / 未验证 1）
+ 风险表（P0：Delta 效果未验证、证据级召回缺金标；P1：四格仅合成验证、LME 未接线、CR 双向风险；
P2：块内 serial 回退、v8/v9 时间语义双份实现）+ t6 预算口径 + 复现命令 + 文档遗留清单。

**这不是成绩单**：表里明确区分"已修 / 部分修 / 仅合成验证 / 待接线 / 未修 / 未验证"，
不得把 `topk_refresh`（词面 IDF 近似）写成"检索器驱动刷新"、不得把 Delta 写成"已验证有效"、
不得把窗口方向写成已定论、不得把合成探针结果表述成真实数据集的证据级召回。

## [v9.2-t19] — 2026-10-21 — CR 两处硬门控的反向风险量化（纯解析，无 GPU）

**一句话**：把"漏链 / 误拒覆写"从口头风险变成数字 —— 属性门控的代价是 **0**，
2 跳桥接的代价**测不出来**（判定样本太小），两者都**不触发**归一改动。

### 指标与结论（`experiments/results/cr_link_recall.json`）

| 项 | 口径 | 结果 | 结论 |
|---|---|---|---|
| B 覆写丢失 | **槽级**：同主语 + 同 loose 属性下池中并存多个不同值的活槽 | **0 / 260 = 0.00%** | 未超 5% 阈值 → 不动 `_cr_verb_core` |
| B 覆写丢失（pair 级） | 应覆写对里属性键不同的比例 | 0 / 47 = 0.00% | 同上 |
| B（粗代理，不采用） | old/new 两值都作为子串出现在池文本 | 29 / 47 = 61.7% | **噪声**，会被别的事实的同名字符串污染，不可引用 |
| A 2 跳漏链 | 需链题目里因归一化不等而无法成链的比例 | mh 0.00%（n=10）/ sh 8.33%（n=36） | **未定论**：覆盖率仅 6% / 17% |

### 为什么 A 记"未定论"而不是"未超阈"

172（mh）/ 207（sh）条"需链"里，解析层有 **162 / 171 条找不到桥接候选**，真正被判过的只有
10 / 36 条。8.33% 是"36 条里 3 条"，不足以支撑一次归一改动。脚本内置样本量门槛
（`MIN_JUDGED=30`、`MIN_COVERAGE=30%`），不达标时结论字段直接写 `A_underpowered: true`。
要提高覆盖率需换事实库更全的切片（`sh_262k` / `mh_262k` 本机未落盘，JSON 里标 `unavailable`，不编造）。

### 数据来源

`results/groupA_factconsolidation_mh_6k.jsonl` 的 fact pool（100 行共享）+ mh_6k 100 题 +
`results/v6groupB2_factconsolidation_sh_6k.jsonl` 的 sh_6k 100 题；真实 CR 跑分产物，非合成。

## [v9.2-closeout] — 2026-10-21 — 审计收口：Delta 写入策略 / 时间进 prompt / token 预算 / CR Delta / LME 读出 / 联合归因 / 本地 Judge

**一句话**：把审计报告里"机制缺失"的项全部补上并接线，同时把**默认行为的两处变化**和
**五项仍未验证的效果**写清楚 —— 本轮仍然**没有任何可用于论文的指标**。

### 新增能力

| 能力 | 开关（默认） | 状态 |
|---|---|---|
| Delta 有效写入策略 | `AR_DELTA_WRITE_STRATEGY=all\|filtered\|topk_refresh`（`all`） | `all`/`filtered` 已实现；`topk_refresh` 是**问题词面 IDF 近似**（写入先于建池，拿不到真实检索结果）→ 部分实现 |
| 时间进生成 prompt | `AR_V9_PROMPT_DATE=1` | 行格式 `<serial>. [<date>] <原文>`；日期继承 Session 头；解析不出则不伪造（只到年份写 `[1997]`）。`=0` 与旧输出**逐字节一致** |
| 2500 token 证据预算 | `AR_V9_CTX_TOKEN_BUDGET=2500`、`AR_V9_CTX_MAX_UNITS=0` | 按分数降序装填、超预算即停、装完按位置排序（I2 不变）；OOM 重试改按证据优先级整行丢，删除"保留最后 N token"的静默截尾 |
| CR Delta 参与查询 | `CR_DELTA_MODE=off\|context\|refresh`（`off`） | 删除死开关 `B2_KEEP_DELTA`（只进指纹、从不控制 reset）；非法值 `SystemExit` 不静默退回 |
| CR 逐题证据日志 | 默认开启 | `retrieved_*` / `used_*` / 统一 ID `cr:<idx>` / `rec["delta"]`；FINAL-CR 行给 `answer_hit@K` 与 `answer_hit_ctx` |
| LongMemEval 读出 | `readout(mode="lme")`，开关 `V9_LME_TURN_WINDOW=1` / `V9_MAX_PER_SESSION=2` / `V9_SESSION_QUOTA=1` | Session/Turn 内扩展、跨 Session Max-2 配额、first/last/before/after 时间约束；`before/after` 是硬过滤、`first/last` 是重排、解析失败即 none。**runner 尚未接线 → 线上不生效** |
| v8 时间语义 | `V8_SESSION_RESET=1` | Session 边界重置 `last_ts`；`TIMESTAMP_RE` 扩到 yesterday / last week / 星期几；滑窗不跨 Session |
| 四格联合归因 | `ar_diag --joint-k`（默认 1） | `Reader Acc \| Recall=1`、检索失败率、读到但答错率；`evidence_recall_at_k` 全 None 时标 `evidence_based=False` |
| 本地 Judge | `--judge-backend local` | 与 http 共用 `build_prompt` / `parse_verdict`；缺 GPU / 缺权重 / 缺 `--judge-model` 一律 exit≠0 且**不回退 http**；`--compare-backends` 做等价性验证 |

### 修掉的确定性 Bug

1. **CR 系表谓词解析把不同属性判成同一谓词**：`Alice is a doctor.` 与 `Alice is vegetarian.` 都解析成
   `('Alice','is')`，写入后 `size=1 / n_dup=1`，医生事实被素食事实吃掉。现表语冠词+头词并入谓词键
   （`is a doctor` / `is vegetarian`），并加属性类别门控 `_attr_compatible` + `n_attr_rejected` 诊断。
2. **CR 双跳评分与路径渲染错位**：`chain_score` 改为显式 `(fact1, fact2)` 二元结构，
   `linked()` 要求 `obj(f1) == subj(f2)` 归一化相等，不再用软门控糊过去。
3. **L18 预计算矩阵按 offset 盲切 → 静默取错向量**：`StatePool._replace` 传的是状态池自己的槽 id，
   多池共享同一矩阵时会拿到别人的向量且不报错。改为按**文本**查表（见 [v9.2-l18]）。
4. **`extract_units` 的 `ts` 只到年**（`2023/05/20` → `2023`），使 LME 的 `before/after` 在同一年内
   毫无判别力。新增 `anchor_ts`（Session 头日期，整日精度），`unit_date` 取最精确者。
5. **块行缺日期**：时间此前只进检索打分、不进模型输入；现每行带 `[date]` 前缀。

### 默认行为变化（对比旧跑分前必看）

- `AR_V9_CTX_MAX_UNITS` **40 → 0**：拼块改由 token 预算决定，复现旧口径设 `AR_V9_CTX_TOKEN_BUDGET=0`。
- `V9_L18_WHITEN` **默认 1**（仅 `AR_V9_L18=1` 时生效）：复现旧 L18 口径设 `=0`。
- 两处均已进 `run_manifest` 配置指纹，不同口径不会共用结果文件。
- 预算模式下 **Top-K 不再决定拼块集合**，`hits` / `recall@K` 仅用于诊断。

### 未验证（硬约束）

1. `AR_DELTA_WRITE=0/1` × 三种策略 —— **Delta 到底有没有用，无任何对照**。
2. `AR_V9_PROMPT_DATE=1/0` 配对 —— 日期进 prompt 的收益未测。
3. `AR_V9_L18=0/1`（含 whitening）—— 稠密通道收益未测（只有合成池几何证据）。
4. `CR_DELTA_MODE=off/context/refresh` —— CR 里 Delta 参不参与查询的差别未测。
5. 本地 Judge 真实跑分未做；`evidence_recall_at_k` 在真实数据上仍恒为 None
   （`results/*.jsonl` 18 文件 1800 行中带 `diag` 字段 0 条）。

### 文档修正

- 撤回"端到端 SF 对检索内容敏感、是有效探针"：配对后 1 句 vs 默认 −3.85pp（p=0.625）、
  宽窗 vs 默认 +8.70pp（p=0.500），**均不显著**；n=100 时 MDE=9.7pp。
- 修正数字张冠李戴："默认 21 句 = 0.810" 应为 **0.730**（0.810 属 b0_a20 / b4_a16）；
  b16_a4 的 all_rows 0.750 应为 **0.7483**；交集口径 n=144 → **145**。
- 窗口方向明确写"**未定**"：12 对比较 Holm 后全不显著（最小 Holm p=0.0764），默认 16/4 仅暂留。

## [v9-validate] — 2026-10-21 — 首次真实评测：窗口扫描中止 + 两个口径问题 + 潜层读出立项

**一句话**：10 项修复全部落地并双环境验证；首次跑真实评测（EventQA_65536）暴露出
"指标口径失效"与"服务器环境跑不起来"两个必须先解决的问题；窗口方向实验中止，潜层读出立项。

### 新增文档

| 文件 | 内容 |
|---|---|
| `experiments/eventqa_window_sweep.md` | 窗口扫描设计、指标口径失效的实测证据、四组 + 两个对照的真实数字、中止决策 |
| `experiments/run_eventqa_window_sweep.sh` | 四组顺序扫描脚本（内置 overlay `PYTHONPATH`，每组独立结果文件 + manifest） |
| `tools/summarize_sweep.py` | 扫描结果汇总（window/n/SF/anchor/in_ctx/AR@1/3/10/run_id） |
| `README.md` | 新增"v9 验证轮"条目：修复清单、首次评测数字、口径/环境问题、P4 计划 |
| `ARCHITECTURE.md` | 状态矩阵补 4 行（潜层读出未实现 / 窗口验证不成立 / 证据级召回缺失）、风险补 7–8 条、P4 拆成 A/B/C |
| `test_v9_cr.py` | 新增 `test_subject_index_survives_eviction`：淘汰/覆写后主语索引必须指向活槽 |

### 实测数字（EventQA_65536，单卡 V100，ctx0 各 100 题）

| 窗口 | n=100 SF | n=145（ctx0+ctx1 交集） |
|---|---|---|
| b0_a20 | 0.810 | 0.8276 |
| b4_a16 | 0.810 | 0.8207 |
| b10_a10 | 0.800 | 0.8069 |
| b16_a4（默认） | 0.730 | 0.7517 |

> **数字修正（2026-10-21 收口轮）**：本节原写"n=144（ctx0+1）"与 b16_a4 `0.750`，实为
> **n=145 交集**与 `0.7517`（文件级 all_rows 为 0.7483）；"默认 21 句 0.810"是笔误，默认是 **0.730**。
> 配对检验见 `experiments/results/paired_window_test.json`。

对照（配对口径）：同 52 题 1 句窗口 0.6346 vs 默认 0.6731；同 23 题 ~350 句宽窗 0.7826 vs 默认 0.6957。

### 关键发现（必须披露）

1. **EventQA 金标不在原文中**：`gold in context`（原文 / 归一化）均 False → 选项是生成式改写。
   `answer_in_context` / `anchor_answer_hit` 750 条命中 0/750，**口径失效**，不是检索失败。
2. ~~**端到端 SF 是有效探针但粗**：1 句窗口掉 17.5pp，说明依赖检索~~
   **→ 已撤回（2026-10-21 收口轮）**：17.5pp 来自**未配对**样本（1 句 0.635 vs 默认 0.810，而两者题号不同）。
   配对后同 52 题 1 句 vs 默认仅 **−3.85pp（p=0.625）**，同 23 题宽窗 vs 默认 **+8.70pp（p=0.500）**，
   均不显著。结论改为：**现有数据无法证明 SF 能区分检索好坏**，需换自带金标 span 的合成探针。
3. **窗口方向差异不显著**（配对 McNemar：b0_a20 / b4_a16 vs b16_a4 各 +8.00pp，raw p=0.0386，
   **Holm p=0.2314**；MDE@80% power = 9.7pp，观测 8.0pp < MDE）→ 方向**未定**，默认值暂留，实验中止。
4. **服务器环境原本跑不起来**：transformers 4.51.3 缺 `smollm3`（`import deltamem` 失败）+ 缺
   `nltk/rouge_score/editdistance/tiktoken`；用 overlay `/share/exp4group/pylib_tf57` 解决，跨版本数字不可比。
5. **宿主机负载**：四进程并行时 load 180，每题 5s → 200s，9 小时只推进到第 2 个 context。

### 未做 / 未验证

- 窗口扫描未跑满 5 个 context（中止，见上）。
- `evidence_recall_at_k` 仍无数据源（EventQA 无 span，LongMemEval 未接）。
- **潜层读出（P4-A/B/C）未开工**：`AR_READOUT_MODE=latent` 仍直接 exit 2，所有数字都是明文 prompt 读出。
- 第 18 层稠密通道的真实效果、Delta 写入的真实效果，仍无证据。

本仓库的实验记录按版本追加。指标一律注明"由哪一版代码产出"，避免把重构前数字误当重构后结果。

## [v9-fix] — 2026-10-20 — 设备不一致 / 覆写丢槽 / 结果文件复用（P0）+ 诊断诚实化 + CR 性能与路径读出

**性质：正确性与可复现性修复，不含任何新指标。** 未跑真实评测、未加载模型权重、未初始化 CUDA。
所有结论来自 CPU 单测与 `--dry-run`；EventQA 窗口扫描脚本已就绪但**未执行**（见
`experiments/eventqa_window_sweep.md`，表中数值留空）。

### Fixed — P0（会让结果错误却看起来正常）

- **第 18 层稠密通道设备不一致**（`v9_slotpool.py:545` `hybrid_scores`）：旧版把 `dense_scores` 的结果
  （provider 在 GPU 上算）与 `lexical_scores`（CPU/另一设备）直接相加，触发
  `RuntimeError: Expected all tensors to be on the same device`。该异常被 `run_ar_v7.py` 里
  宽 `except` 吞掉，于是**整题退化成空证据块**，指标下降被误读为"方法变差"。
  改为：`l18` 向量一律常驻 **CPU fp32**（262K 单元 × d 放显存会爆），`dense_scores` 返回 CPU 张量、
  `hybrid_scores` 在相加前把两侧对齐到 `s.device/s.dtype`，并加形状断言。
- **CR 覆写可能找不到正确的旧槽**（`v9_slotpool.py:843` `_merge_target`）：旧版在全池 `mean` 上取
  `argmax` 选候选，再按主语过滤——当别的槽主语向量更接近时，同主语的旧槽**根本没进候选**，
  于是"覆写"变成新增，池里留下过期事实。改为 `_subj_index` 主语倒排表（`_reindex_subject`），
  先 O(1) 取同主语槽，再在其中按关系谓词余弦 + `V9_CR_REL_THETA` 双阈值判定。
- **不同实验配置复用同一个结果文件**（`run_ar_v7.py:224` `init_manifest` + 新增 `run_manifest.py`）：
  旧版输出固定 `ar_{MODE}_{SRC}.jsonl` 且 `AR_RESUME=1` 默认开，`AR_DELTA_WRITE=0/1`、
  `AR_V9_L18=0/1`、不同 EventQA 窗口全部落到同一文件，第二次运行直接跳过已完成题并打印第一次的分数。
  改为：配置指纹（8 位）→ 文件名 `ar_{MODE}_{SRC}_{id}.jsonl`；同目录 `<out>.manifest.json` 记录
  完整参数 + git commit + 环境；续跑前校验，不一致**拒绝恢复**（`AR_RESUME_MANIFEST=0` 可关，不建议）。

### Fixed — P1

- **时间戳跨 Session 继承**（`v9_slotpool.py:135` `split_sessions` / `:182` `extract_units`）：
  违反不变量 I3。现在按 `Session N` / `## Session` / `=== Session` / `Date:` / `YYYY/MM/DD` 切 Session，
  每个 Session 重置 `last_ts`，单元新增 `session_id` / `event_time` / `ingestion_order` 字段。
  无显式标记的上下文仍视为单 Session（行为不变）。
- **CR 覆写是 drop+append**（`v9_slotpool.py:909` `_replace`）：旧版删除旧槽再追加新槽，槽 id 漂移、
  图键与槽下标错位、`cr_E_*` 索引失效。改为原位替换（texts/index_texts/metas/tokv/words/mean/years/df/
  CR 图键/l18 全部就地更新），槽 id 与池序稳定。`v9_cr.py:179` 覆写 `_replace` 同步 CR 图结构与
  主语/宾语键表，并清空 `_subj_mat_cache`。
- **StatePool 建池 O(N²)**：旧版每条事实都对全池 stack 一次再打分。改为主语倒排索引 +
  批量内 pending 冲突合并（`_merge_target` 返回 `("stored"|"pending", idx)`），单条事实只碰同主语槽。
- **第 18 层编码失败静默降级**（`v9_slotpool.py:305` `set_layer18_provider(fn, strict=True)`、
  `:358` `_call_l18`）：runner 注入 provider 时传 `strict=True`，失败/形状不符直接抛错，不再退化成纯词面。
  默认 `strict=None` 时仍按 `V9_L18_STRICT` 决定，保持单测与旧脚本可用。
- **检索异常被宽 except 吞掉**（`run_ar_v7.py:244` `strict_retrieval_enabled`）：`AR_STRICT_RETRIEVAL=auto`
  时，只要稠密通道 / delta 写入 / 潜层读出任一开启，检索异常直接 `raise`，不再产出空证据块。
- **`inputs_embeds` 被当成潜层读出**（`run_ar_v7.py:395` `generate_readout`）：实际是 `EMB[ids]`——
  **同一段明文 prompt** 的查表嵌入，没有注入任何槽位状态。新增 `AR_READOUT_MODE=prompt|text_embeds|latent`，
  `latent` **显式报错退出**（未实现）；每条记录写入 `readout.mode` 与 `readout.latent_slots=false`，
  避免论文里把明文嵌入说成 latent memory。

### Fixed — P2

- **诊断 `Recall@K` 名不副实**（`ar_diag.py`）：它是"gold answer 字符串是否出现在第 K 个槽位"的代理，
  不是证据级召回。规范键改为 `answer_recall_at_k`（旧键 `recall_at_k` 保留为别名以兼容历史 JSONL），
  新增 `evidence_recall_at_k`：没有金标证据 span 时恒为 `None`，**不伪造**。
- **Delta 空快照允许实验继续**（`run_ar_v7.py:357` `prepare_delta_memory`）：写入开启但写后快照为空时，
  逐题恢复退化为 no-op、题目互相污染。改为直接 `RuntimeError`；确需放行用 `AR_ALLOW_EMPTY_DELTA_SNAP=1`。

### Added

- **`run_manifest.py`**：配置指纹、原子写 sidecar、`check_resume`（config_id / 参数 / commit 三级校验）、
  `default_out`（文件名带指纹）、`describe`。纯标准库，不依赖 torch。
- **`run_group_b2_v7.py`** 接入 manifest：`CR_MODE`/`BEAM`/`EGO_K`/`REL_THETA`/`HOP1_M`/`PATH_VIEW`/
  `CF_PROMPT`/`THETA`/`POOL_MAX` 全部进指纹；续跑按 `question_idx` 跳过已完成题；汇总改为**读文件**
  而非只算本轮 `results`（续跑时旧算法会把总分算成本轮小样本）。
- **`v9_cr.hop1_candidates`**（`v9_cr.py:241`）：Hop1 先用主语均值矩阵 `@ q_k` 初筛至多
  `V9_CR_HOP1_M=96` 个候选（`n<=M` 时全量，等价旧行为），再逐 token MaxSim；1 万槽池的 Python
  循环从 1 万次降到 96 次。gold 槽不丢（`test_hop1_matrix_prefilter`）。
- **CR 路径式读出**（`v9_cr.py:337` / `:437` `_render_paths`）：每条完整路径渲染成
  `PATH i: 种子 --关系--> 中间实体 --关系--> 目标`，附实体链与关系分，再列其余已选事实；
  事实仍用**全局编号**，反事实 Prompt 的"编号越大越新"规则不变。`V9_CR_PATH_VIEW=1` 默认开
  （v9 CR 从未跑分，无历史可比性问题），`=0` 回到平铺块。
- **`experiments/eventqa_window_sweep.md` + `run_eventqa_window_sweep.sh`**：Back/Ahead = 0/20、4/16、
  10/10、16/4 四组，记录 `anchor_accuracy`（`anchor_answer_hit`）与 `answer_in_context`。
- **`tools/summarize_sweep.py`**：读若干 jsonl，输出 `window/n/SF/anchor/in_ctx/AR@1/3/10/run_id` 表。
- **`test_run_manifest.py`**（27 断言）：指纹随消融开关变化、文件隔离、续跑三级校验、
  `latent` 读出被拒、delta 空快照报错。

### Changed

- `run_ar_v7.py` 记录新增 `run_id` / `readout` / `window` / `anchor_answer_hit` / `anchor_window_hit`；
  `AR_INPUTS_EMBEDS=1` 仍可用，等价于 `AR_READOUT_MODE=text_embeds`。
- `ARCHITECTURE.md` 重写为**三层结构**（输入事件流 / State Memory / Episodic Memory / Evidence Router /
  Latent Readout），新增不变量 I5（结果是配置的函数），并明确"读出层仍是明文"这一事实边界。

### Tests

| 套件 | 断言 | 状态 |
|---|---|---|
| `test_v9_slotpool.py` | 129 | 全绿（含同主语覆写 P0 回归、设备对齐、Session 边界、strict 模式） |
| `test_v9_cr.py` | 80 | 全绿（含原位覆写图键一致、Hop1 初筛、路径读出/平铺回退） |
| `test_ar_v9_wiring.py` | 36 | 全绿 |
| `test_ar_delta_alignment.py` | 27 | 全绿 |
| `test_run_manifest.py` | 27 | 全绿 |
| `test_v8_slotpool.py` | 57 | 全绿 |
| `ar_diag.py --selftest` | 57 | 全绿 |

### 环境阻塞（2026-10-20 实测，非本仓库问题）

- `latent2` 的 transformers 4.51.3 **缺 `models/smollm3`**，而 `/share/delta-Mem/deltamem/core/
  backbone_compat.py:3` 无条件 import 它 → `import deltamem` 直接 `ImportError`，
  **任何评测都跑不起来**（`run_ar_v7.py` 会打印"缺少真实跑分依赖"）。
- 同环境还缺 `nltk` / `rouge_score` / `editdistance` / `tiktoken`（`post_process` 依赖）。
- 处理：不改共享 conda 环境，装 overlay 目录 `/share/exp4group/pylib_tf57`
  （transformers 4.57.1 + tokenizers 0.22.2 + 上述四个包），运行时
  `PYTHONPATH=/share/exp4group/pylib_tf57:/share/delta-Mem:/share/MemoryAgentBench`。
  已实测 `import deltamem` + 模型加载 + 3 题冒烟通过（SF=1.0）。
- 副作用：transformers 版本与历史 v7 结果不同，**跨版本数字不可直接比较**；
  窗口扫描四组同栈，内部可比。

### 未做 / 未验证（不得在论文中声称）

1. 真实 GPU 跑分：v9 AR、CR v9、EventQA 窗口扫描四组**均未执行**。
2. 第 18 层稠密通道的**效果**（只有设备与失败路径被验证）。
3. Delta 状态 + 快照恢复在真实模型上的行为（仅单测 stub）。
4. 槽位潜向量表征（路径 B）——**无实现**，`AR_READOUT_MODE=latent` 直接报错。

## [docs] — 2026-10-20 — 新增 `ARCHITECTURE.md`（预期架构）+ 环境路径订正

### Added

- **`ARCHITECTURE.md`**：描述当前**预期架构**（DM-ELSM 双模态弹性潜槽位记忆），而非完成清单。
  含 4 条设计不变量（全池不截断 / 先分数后位置 / 时间是数据 / 读出走潜层）、数据流图、组件契约表、
  两条合法潜层读出路径（A: delta 状态 + 快照恢复；B: `inputs_embeds` + 槽位潜向量）、参数表、
  实现状态矩阵（`已实现` / `已接线·未验证` / `规划中`）、5 项已知风险与 P0–P5 演进路线。
- README 顶部与"文件"章节加入指向该文档的入口。

### Changed

- README「运行」章节补充**环境订正**：`/share/conda/envs/latent2/bin/python`（Python 3.12.4 +
  torch 2.8.0+cu126 + transformers 4.51.3，2026-10-20 重建）；明确**不得再带 `t28` / `pylib5`
  叠加层**（新环境下会使 `import torch` 失败、dry-run 退化为 `cuda_initialized=n/a`）。
- 「v9 真实跑分」示例命令由 `/opt/conda/bin/python` 改为重建后的解释器路径。

## [v9] — 2026-10-20 — 确定性 Bug 修复 + 逐题诊断 + AR 检索重写 + CR 强化 + Delta 对齐

**性质：代码 / 可诊断性重构，不含任何新指标。** 本轮按硬约束只做 dry-run（`py_compile` / CPU 单测 / `--dry-run`），
未跑真实评测、未加载模型权重、未初始化 CUDA、未占用 GPU。

### Fixed — v8 三个确定性 Bug（`v8_slotpool.py`，接口零变更）

- **Bug1 `cur_ts` 第二轮污染**（`v8_slotpool.py:79-138`）：旧版首轮扫描结束后 `cur_ts` 停在全篇最后一个时间戳，
  第二轮窗口/状态判定直接复用它（状态轮还继续改写它），使位于任何时间标记之前的单元被写入"文档末尾"时间戳。
  改为首轮一次性预计算逐句 `resolved_ts[i]`（= 该位置之前最近一次显式时间戳），窗口与状态两轮只读；
  窗口无显式标记时继承 `resolved_ts[i-1]`。合成 150 句上下文：旧版 31 个情景单元被写入末尾时间戳，新版仅 2 个
  （窗口内确实含该时间戳句，属正常显式标记）。
- **Bug2 512 词 span 偏移**（`v8_slotpool.py:159,222-236,401`）：旧版 `spans` 用未截断的 `len(ws)` 记账、却只
  `extend(ws[:512])`，长单元 span 越过自身边界吃进后续单元的词；零 token 兜底分支还 `off += 1`，使后续单元
  `tokv` 整体错位。改为**先截断再记账**（`UNIT_MAX_WORDS=512`，env `V8_UNIT_MAX_WORDS`），零 token 分支不推进 `off`；
  `StatePool._mean_only` 复用同一常量。
- **Bug3 `ev` 时间触发**（`v8_slotpool.py:123,131-138`）：旧版 `ev` 只看事件动词，纯时间句（含 `In 2001` 但无过去时动词）
  落入 state，与"时间标记或事件动词 → episodic"的设计矛盾。改为 `ev` = 事件动词 **or** 窗口内显式时间标记；
  state 只收无显式时间标记且无事件动词的稳定陈述句。仅"继承到时间戳"、窗口内无显式标记且无事件动词的窗口仍归 state。
- 接口：episodic unit 字段不变；state unit 只新增附加字段 `ts_inherited`；`run_ar_v7.py` 无需改动。
- 回归：`test_v8_slotpool.py` 从 9 组扩到 16 组（+7 个 Bug 回归用例）。有效性已验证——把 git HEAD 的旧版
  `v8_slotpool.py` 放到临时目录跑新测试，bug1/bug2/bug3 用例均按预期失败，新版全绿。

### Added

- **`v9_slotpool.py`**（777 行）— v9 AR 检索：
  - 非重叠索引 `extract_units`（`:124`）：`stride == block_sents`，`V9_BLOCK_SENTS=1` 即单句索引，每句恰好属于一个单元；
    修掉 v8 3 句滑窗 stride=1 的 3 倍池膨胀、Top-K 只覆盖 K/3 句、位置排序有歧义三个问题。
  - 分数优先：`retrieve`（`:466`）只按分数降序返回；`retrieve_ordered`（`:477`）/`readout`（`:550`）只对**已选中的 K 个**
    按 `start` 排序，绝不做"全池按位置排序后截断"。
  - EventQA 后向非对称扩展 `expand_backward`（`:515`）：默认 `back=16`、`ahead=4`（`V9_EQA_BACK` / `V9_EQA_AHEAD`），
    含边界裁剪；`anchor_events` 走 last/last2 重锚。
  - 时间戳参与打分与重排：单元编码文本 `In <ts>: <原文>`（`build_index_text`，`:117`），日期 token 进入 IDF-MaxSim；
    `texts` 仍保存原文；`temporal_rerank`（`:444`）按 `|year_u − year_q|` 指数衰减加成
    （`V9_TEMPORAL_TAU=2.0` / `V9_TEMPORAL_BONUS=0.15`），无年份单元不加分不罚分。
  - 第 18 层混合稠密打分：`set_layer18_provider`（`:223`）/`mean_pool_hidden`/`attach_layer18`/`dense_scores`/
    `hybrid_scores`（`:433`），`hybrid=(1−α)·IDF-MaxSim + α·((cos+1)/2)`，`V9_L18_ALPHA=0.3`、`V9_L18_LAYER=18`；
    provider 缺失/抛错/形状不符一律静默退化为纯词面。
  - `StatePool`（`:647`）只作 AR 可选补充与 CR v9 扩展点，`_should_merge` 可覆写。
- **`v9_cr.py`**（359 行）— CR 强化：主语+关系双阈值覆写（`V9_CR_REL_THETA=0.80`）、Hop1 `Beam=6` + ego-graph
  （`V9_CR_EGO_K=12`，同主语属性 + 出边实体）、双跳链联合打分
  `0.40·subj_cov + 0.30·(bridge1+bridge2)/2 + 0.30·(rel_mid+rel_tgt)/2`、反事实强约束 Prompt（`build_cr_prompt`，`:61`）。
- **`ar_diag.py`**（823 行）— 逐题诊断：`record()`（`:306`）产出 `retrieved_slot_ids / retrieved_scores / retrieved_texts /
  retrieved_starts / used_slot_ids / answer_in_retrieved / answer_in_context / answer_hit_slot / answer_matches /
  recall_at_k / diag_status`；`hits` 必须分数降序（Recall@K 依赖）。口径：NFKC + 小写 + 撇号删除 + 连字符折叠后子串匹配；
  rubric（偏好类）gold 标 `rubric_not_applicable` 并剔除分母；只有 `idxs` 的 legacy 记录标 `legacy_idxs_only`。
  CLI：`--input / --summary-out / --enrich / --selftest`。
- **`eval_longmemeval_judge.py`**（836 行）— LongMemEval 官方 `evaluate_qa` 口径的 LLM-as-Judge 后处理：6 题型模板
  （通用 / temporal off-by-one / knowledge-update 旧值容忍 / preference rubric）+ abstention 模板；
  `--prompt-source official` 直接 import 官方 `get_anscheck_prompt`；OpenAI 兼容 `/chat/completions`
  （`temperature=0, max_tokens=10`）、线程池并发、重试退避、sha1 缓存断点续跑；verdict 复刻官方 `'yes' in text` 语义，
  另存 `verdict_strict` 审计；abstention 单列不进主分母；`--dry-run` 零网络、`--selftest` 用 monkeypatch 哨兵保证零网络。
- 新单测：`test_v9_slotpool.py`（95 断言）、`test_v9_cr.py`（56）、`test_ar_v9_wiring.py`（36）、
  `test_ar_delta_alignment.py`（27）；`test_v8_slotpool.py` 扩到 56 断言。

### Changed

- **`run_ar_v7.py`**（976 行）：
  - 新增 `AR_MODE=v9`（惰性 `v9_import`，`build_pool_for_context_v9` / `build_block_v9`，`:589` / `:620`）。
  - 逐题诊断字段：v7/v8/v9/plain 四种模式都写；诊断异常只写 `diag_status=diag_error`，不影响主指标。
  - 修正"先排序后截断"：新增 `_cap_before_sort()`（`:490`），先按分数优先裁到情景配额（EventQA 以 anchor 为中心裁窗）
    再按原文位置排序；旧写法会丢高分尾部单元（单测构造 11 个候选、旧实现丢 8 个）。
  - 新增 `--dry-run`（`dry_run()`，`:740`）：不加载权重、不读 parquet、不初始化 CUDA；打印配置与依赖可导入性，
    跑两项纯 CPU 自检（v9 `extract_units` 时间戳继承、`ar_diag.record` 假池命中），末行打印
    `cuda_initialized=False model_loaded=False`；非法 `AR_MODE` / source 返回退出码 2。
    `transformers/deltamem/pyarrow` 改为 try 导入，缺依赖时真实跑分显式报错，dry-run 不受影响。
  - Delta Memory 对齐：写入移到**查询前**（`prepare_delta_memory`，`:255`），每个 context 只 `reset_delta_mem_states` 一次，
    题目循环内不再 reset；每题生成前 `delta_restore(snap)`（`:239`，优先官方 `load_delta_mem_online_state`，
    缺 API 时退回 `iter_delta_mem_modules` 手工 clone/restore，两套都缺则 `[warn]` 退化为 no-op）；
    新增 `AR_INPUTS_EMBEDS=1` 的 `inputs_embeds` 读出路径（`generate_readout`，`:288`），OOM 渐进截短两条路径都生效；
    每题 JSONL 新增 `delta: {write, restore, inputs_embeds, snapshot_layers}`。
- **`run_group_b2_v7.py`**：新增 `CR_MODE=v7|v9`（默认 v7，v7 分支与历史完全一致）；v9 分支注入 `v9_cr`、
  可选 CF Prompt、每题追加 `cr_dbg`、OUT 改 `v9groupB2_*.jsonl`、FINAL 打印 `CR_MODE`；非法值直接退出。

### Docs

- `README.md`：新增 v9 章节（改动清单 + dry-run 结论 + 未验证项声明 + v9 env 表），v8 章节标注"指标为修复前数字"，
  补 `--dry-run` 与 v9 运行命令，更新文件清单。
- 本 `CHANGELOG.md`。

### Verification（dry-run，本机 Python 3.9.6 + torch 2.8.0 CPU，`device_count=0`）

| 检查 | 结果 |
|---|---|
| `python3 -m py_compile *.py` | 21/21 通过 |
| CPU 单测（5 文件） | 95 + 56 + 56 + 36 + 27 = 270 断言全部 PASS |
| `ar_diag.py --selftest` | 43/43 |
| `eval_longmemeval_judge.py --selftest` | 45/45（零网络） |

> **当前数字（v9 收口轮）**：`py_compile` **28/28**；CPU 单测 **9 个文件** —— `test_v9_slotpool` 284 /
> `test_v9_cr` 143 / `test_v8_slotpool` 86 / `test_ar_v9_wiring` 105 / `test_ar_delta_alignment` 37 /
> `test_run_manifest` 27 / `test_ar_delta_write_strategy` 47 / `test_cr_delta_log` 84 / `test_v6_parse`；
> `ar_diag --selftest` **121/121**；`eval_longmemeval_judge --selftest` **76/76**。上表是 v9 验证轮的历史快照。
| `run_ar_v7.py --dry-run`（v7/v8/v9/plain） | 四种均退出码 0，末行 `cuda_initialized=False model_loaded=False` |
| `eval_longmemeval_judge.py --dry-run` | 合成 3 题渲染 3 条 prompt，零网络请求 |
| 守卫断言 | 零网络、零 `transformers/pyarrow/deltamem` 导入、无文件写入、无残留进程 |

### Not verified（硬约束，非"已通过"）

1. **没有任何 v9 指标**：真实 tokenizer/模型下的 v9 建池与生成、五项检索改动对 SF / Recall 的影响均未测量。
2. **v8 历史指标（0.7100 / 0.5120 / 0.3820 / 0.3167）由修复前代码产出**，修复后未重跑。
3. `AR_V9_L18=1` 的 provider 前向未跑（仅静态审查，默认关闭）。
4. Delta Memory：`/share/delta-Mem` 实际版本是否含 `get/load_delta_mem_online_state` 未确认（已做 fallback）；
   真实 Qwen3 `generate(inputs_embeds=...)` 行为、写入/恢复开销与指标影响未测。
5. `run_group_b2_v7.py` 未运行（模块级 `from_pretrained` + `/share` 路径、无 `--dry-run`）；CR v9 只由单测覆盖。
6. `eval_longmemeval_judge.py` 未做真实判题（未配 API key、未联网）。
7. `ar_diag.py` 未跑真实 AR 结果，只在 legacy CR JSONL 上验证不崩（全部 `legacy_idxs_only`，applicable=0）。

## [v8] — 2026-10-19 — 双模态潜槽位记忆 DM-ELSM（AR）

- 新增 `v8_slotpool.py`：`extract_adaptive_units` 情景/状态自适应分流（3 句滑窗微块 + 时间戳继承）、
  EpisodicFastPool 全池无截断 IDF-MaxSim、邻域扩展；`run_ar_v7.py AR_MODE=v8`。
- AR 四切片：eventqa_65536 0.7100 / 131072 0.5120 / full 0.3820 / longmemeval_s* 0.3167
  （v7 基线 0.7240 / 0.5620 / 0.3500 / 0.2767）。**注：这三个 Bug 已于 v9 修复，上表未重跑。**
- 长上下文 OOM 修复：`encode_words` 分批编码 + 逐切片 `.cpu()`；generate 前 `empty_cache` + 渐进截短重试。

## [v7] — 2026-10-18 — 自适应弹性槽位池打通 6K→262K

- 新增 `v7_slotpool.py`：无损弹性容量（2048→…，`B2_POOL_MAX=0`）、二级路由 O(1) 覆写、主语特异性长度加权、
  双向桥接覆盖率、WHERE 倒装解析；CR 8 切片 SF：sh 0.94/0.91/0.89/0.85，mh 0.67/0.63/0.59/0.49。
- AR 四切片：eventqa_65536 0.7240 / 131072 0.5620 / full 0.3500 / longmemeval_s* 0.2767。

## [v6] — 2026-10-18 — 一套代码通吃 sh + mh

- `v6_slotpool.py`：VP_RE 谓词补全 + 长谓词优先、WHERE 倒装从句解析、(subj, rel) gid 精确覆写。
- CR：sh_6k 0.940 / sh_64k 0.890 / mh_6k 0.670 / mh_64k 0.590。

## [v5] — 2026-10-17 — 三元组解耦潜键 + Beam Chaining

- `v5_slotpool.py`：`parse_fact_triple` 三元组解耦、双重命中覆写、主语门控多跳检索（Beam=2）。
- CR mh：6k 0.510 / 64k 0.450。

## [v4] — 2026-10-17 — 大池 + 主谓组去重 + Top-15 检索读出

- 修 4 个瓶颈（POOL_MAX=512 盲淘、全量拼块稀释注意力、KEY_TAIL_CUT 漏覆写、读出前未 reset）。
- CR sh：6k 0.860 / 64k 0.870。

## [v3] 及更早

见 git 历史与 `versions/` 目录。初版（v1）因 5 个致命 Bug（事实切分漏召、模板 KV 污染、decode 步进、跨 chunk RoPE 拼接、
key 切分失效）被判 No-Go，v2 修复后 sh_6k 0.110→0.850。
