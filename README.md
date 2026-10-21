# 潜层记忆 B2 实验：流式事实槽位原位覆写 + 检索式读出

基于 delta-mem（declare-lab/delta-mem_qwen3_4b-instruct）+ MemoryAgentBench（Conflict_Resolution 切片）的对照实验，验证"流式事实槽位原位覆写（B2）"能否提升 Conflict Resolution 的 SF（substring exact match）。

> **架构说明**：系统目标形态、设计不变量与各组件实现状态见 [ARCHITECTURE.md](ARCHITECTURE.md)。
> 本文按时间倒序记录版本进展与实测数字。

## 最新进展（v9 收口轮，2026-10-21）——20 项审计落地 + 能力矩阵更新

> 逐条核验表（每项带 `文件:行` 与测试名）：[experiments/audit_2026-10-21.md](experiments/audit_2026-10-21.md)

**结论先行**：

1. 审计报告的 **20 项**现状：已修 12 / 部分修 2 / 仅合成验证 1 / 待接线 1 / 已知风险 1 / 方向未定 1 / 未修 1 / 未验证 1。
2. **确定性 Bug 已清完**：本轮修的是"结果不对 / 不可复现 / 静默降级"三类，**不是性能提升**。
3. **v9 仍然没有任何可用于论文的指标**：新增能力全部默认关闭或尚未接线，且没有任何一项在真实数据上跑过对照。
4. 一条硬事实：仓库内 `results/*.jsonl` 共 **18 文件 1800 行，带 `diag` 字段 0 条、带 `delta` 字段 0 条**。
   所有历史跑分都早于这些字段存在 —— 所以"四格联合归因"只能算合成验证、"Delta 效果"只能算未验证。

### 1. 本轮新增能力（全部靠 env 开关，括号内是默认值）

| 能力 | 开关（默认） | 状态 |
|---|---|---|
| Delta 有效写入策略 | `AR_DELTA_WRITE_STRATEGY=all\|filtered\|topk_refresh`（`all`） | `all` / `filtered` 已实现；`topk_refresh` 是**问题词面 IDF 近似**，不是检索器驱动刷新 → 部分实现 |
| 时间进生成 prompt | `AR_V9_PROMPT_DATE=1` | 已实现（行格式 `<serial>. [<date>] <原文>`）；**收益未验证** |
| 2500 token 证据预算 | `AR_V9_CTX_TOKEN_BUDGET=2500`、`AR_V9_CTX_MAX_UNITS=0` | 已实现；OOM 重试改按证据优先级整行丢，不再静默截尾 |
| LongMemEval Session/Turn 扩展 + 跨 Session Max-2 + 时间约束 | `V9_LME_TURN_WINDOW=1` / `V9_MAX_PER_SESSION=2` / `V9_SESSION_QUOTA=1` | 代码已实现（`readout(mode="lme")`），**runner 尚未接线 → 线上不生效** |
| L18 池内 whitening + 复用 epi 向量 | `V9_L18_WHITEN=1` / `V9_L18_REUSE_EPI=1` | 已实现；**效果未验证**（只有合成池几何证据） |
| v8 Session 重置 + 相对时间戳 | `V8_SESSION_RESET=1` | 已实现 |
| CR 中 Delta 参与查询 | `CR_DELTA_MODE=off\|context\|refresh`（`off`） | 已实现；**效果未验证**；死开关 `B2_KEEP_DELTA` 已删除 |
| CR 属性类别覆写门控 | 无开关（`_attr_compatible`） | 已实现；**双向风险已量化**（t19）：槽级真 stale 0/260 = 0.00% → 不加归一；2 跳漏链 mh 0.00%(n=10) / sh 8.33%(n=36)，覆盖率仅 6%/17% → **未定论** |
| 四格联合归因（Reader Acc \| Recall=1） | `ar_diag --joint-k` | 已实现；**仅合成验证** |
| 本地 Qwen Judge | `--judge-backend local` | 已实现；**未真实跑分** |

### 2. 未验证清单（引用任何数字前必读）

以下五项**开关已就位、效果未知**，都需要在有 GPU 的机器上跑对照：

1. `AR_DELTA_WRITE=0/1` × 三种写入策略 —— Delta State 到底有没有用
2. `AR_V9_PROMPT_DATE=1/0` 配对 —— 日期进 prompt 有没有用
3. `AR_V9_L18=0/1`（含 `V9_L18_WHITEN`）—— 第 18 层稠密通道有没有用
4. `CR_DELTA_MODE=off/context/refresh` —— CR 里 Delta 参不参与查询有没有差别
5. 本地 Qwen Judge 真实跑分 + `--compare-backends` 与 http 后端的等价性验证

### 3. 默认行为变化（对比旧跑分前必看）

- `AR_V9_CTX_MAX_UNITS` 默认 **40 → 0**，拼块改由 `AR_V9_CTX_TOKEN_BUDGET=2500` 决定。
  **复现旧 v9 拼块口径需显式设 `AR_V9_CTX_TOKEN_BUDGET=0`。**
- `V9_L18_WHITEN` 默认 **1**（仅 `AR_V9_L18=1` 时生效）：与旧 L18 跑分**不可横比**，复现旧口径设 `=0`。
- 预算模式下 **Top-K 不再决定拼块集合**，`hits` / `recall@K` 只用于诊断。

## 历史进展（v9 验证轮，2026-10-21）——正确性修复落地 + 首次真实评测 + 两个口径问题

**结论先行**：

1. 报告里的 10 项 P0/P1/P2 问题**全部修复并推送**，单测 413 项在本机（py3.9）与服务器（py3.12）双绿。
2. 本轮**首次跑真实评测**（EventQA_65536，单卡 V100），暴露出两个必须先解决的问题，见 §2 / §3。
3. **窗口方向（back/ahead）扫描不构成结论**：用来衡量的指标在 EventQA 上结构性失效，且四组差异不显著。
   该实验已中止，默认值 `back=16 / ahead=4` 保留，结论与验证方法见
   [experiments/eventqa_window_sweep.md](experiments/eventqa_window_sweep.md)。
4. **v9 仍然没有任何可用于论文的指标**：所有修复都是正确性 / 可复现性，不是性能提升。

### 1. 修复清单（10 项）

| 编号 | 问题 | 修复位置 | 证据 |
|---|---|---|---|
| P0 | 第 18 层混合检索 CPU/GPU 设备不一致 | `v9_slotpool.py:543/559`：dense 恒返回 CPU fp32，相加前对齐 device/dtype + 形状校验 | `test_hybrid_device_consistency` |
| P0 | 稠密通道失败静默退化为纯词面 | `set_layer18_provider(fn, strict=True)` + `_l18_fail` | `test_layer18_strict_mode` |
| P0 | CR 覆写可能找不到正确的旧槽 | `v9_slotpool.py:843`：主语倒排定位同主语槽，再按关系谓词双阈值判定 | `test_cr_overwrite_finds_same_subject_slot`（旧实现必失败） |
| P0 | 不同实验配置复用同一个结果文件 | `run_manifest.py`：配置指纹进文件名 + sidecar manifest + 续跑三级校验（配置/参数/commit） | `test_run_manifest.py` 27 断言 |
| P1 | `inputs_embeds` 读出被当成潜层读出 | `AR_READOUT_MODE=prompt\|text_embeds\|latent`，`latent` 显式报错退出；记录内写 `readout.mode` / `latent_slots=false` | `test_ar_delta_alignment.py` |
| P1 | 时间戳跨 Session 继承（违反 I3） | `split_sessions` + 每 Session 重置 `last_ts` | `test_session_boundary_ts` |
| P1 | CR 覆写是 drop+append（槽 id 漂移） | `_replace` 原位更新，CR 图键与主语索引同步 | `test_inplace_replace_keeps_graph_keys` |
| P1 | StatePool O(N²) 建池 | 主语倒排 + 批内 pending 合并，单条事实只碰同主语槽 | `test_cr_pending_merge_and_stable_ids`、`test_subject_index_survives_eviction` |
| P2 | `Recall@K` 名不副实 | 更名 `answer_recall_at_k`（旧键留别名）+ 新增 `evidence_recall_at_k`（无金标恒 None，不伪造） | `ar_diag --selftest` 121 项 |
| P2 | Delta 快照为空仍继续实验 | 直接 `RuntimeError`，`AR_ALLOW_EMPTY_DELTA_SNAP=1` 才放行 | `test_empty_delta_snapshot_raises` |

附带（用户额外要求）：CR Hop1 全池 Python MaxSim 循环 → 主语均值向量矩阵初筛（1 万槽 → 96 候选，`test_hop1_matrix_prefilter`）；
CR 读出改为完整路径 `种子 --[关系]--> 中间 --[关系]--> 目标` + 每条边关系分（`V9_CR_PATH_VIEW=1`，`=0` 回到平铺）。

### 2. 首次真实评测：EventQA 窗口扫描（已中止，不构成结论）

四组窗口总宽都是 20 句、只改方向，同 ctx 各 100 题：

| 窗口 | n=100（ctx0）SF | n=145（ctx0+ctx1 交集）SF | 平均块句数 |
|---|---|---|---|
| b0_a20（纯前视） | 0.810 | 0.8276 | 21.0 |
| b4_a16（弱后视） | 0.810 | 0.8207 | 21.0 |
| b10_a10（对称） | 0.800 | 0.8069 | 20.9 |
| b16_a4（当前默认） | 0.730 | 0.7517 | 20.7 |

- **配对检验**（同 `(context_id, question_idx)` 交集 + McNemar 精确 + Newcombe 配对 CI）：12 对比较经 Holm 校正后
  **全部不显著**（最小 Holm p=0.0764）。b0_a20 / b4_a16 vs b16_a4 各 **+8.00pp**（raw p=0.0386、**Holm p=0.2314**）；
  MDE@80% power = **9.7pp**，观测 8.0pp < MDE，即 power 不足 0.80。详见
  [experiments/results/paired_window_test.json](experiments/results/paired_window_test.json)。
- 因此**方向未定**：默认 `back=16 / ahead=4` 保留，但**不得**表述为"已验证更优"。
- **探针有效性未证实（原结论已撤回）**：同 52 题 1 句窗口 0.6346 vs 默认 0.6731（−3.85pp，p=0.625）；
  同 23 题 ~350 句宽窗 0.7826 vs 默认 0.6957（+8.70pp，p=0.500）。此前"端到端 SF 确实依赖检索内容（有效探针）"
  的说法来自**未配对**样本，配对后不成立，已撤回。
- 样本量规划（π_d=0.12）：10pp→n≈95、8pp→n≈148、5pp→n≈377。当前 ctx0 n=100 **只够检出 10pp 以上**。
- 注：聚合快照（n=144/144/315/147）与原始 JSONL（145/145/316/147）差 1 行，是快照早于末行写入所致；
  上表用配对交集口径，原始文件级 all_rows 为 0.8276 / 0.8207 / 0.7373 / 0.7483。

### 3. 两个口径 / 环境问题（必须先解决）

1. **EventQA 的金标答案句不在原文里**。实测：`gold in 285,324 字符上下文`（原文与归一化后）均为 False ——
   选项是 LLM 按原文风格**改写**的句子，`metadata` 也没有答案 span。因此 `answer_in_context` / `anchor_answer_hit`
   在 750 条记录里命中 **0/750**，是口径失效而非检索失败。**事件级召回在这套数据上算不出来。**
2. **服务器环境本身跑不起来**：`latent2` 的 transformers 4.51.3 缺 `models/smollm3`，而 `/share/delta-Mem`
   无条件 import 它 → `import deltamem` 直接失败；还缺 `nltk/rouge_score/editdistance/tiktoken`。
   用 overlay 目录 `/share/exp4group/pylib_tf57`（transformers 4.57.1）通过 `PYTHONPATH` 覆盖，不动共享 conda。
   副作用：**跨 transformers 版本的数字不可直接比较**。
3. **宿主机负载**：四进程并行时 load average 180（他人 CPU 作业），每题 5s → 200s，9 小时只推进到第 2 个 context。

### 4. 下一步：P4 潜层读出（让论文立起来的唯一一条路）

现状：所有跑出来的数字都是 `AR_READOUT_MODE=prompt`——检索明文拼进 prompt，**就是普通 RAG**。
`StatePool` 里每个槽的向量（`mean` / `tokv` / 可选 `l18`）只用于检索打分，**从未进入模型输入**。

三档实现（成本递增），设计细节见 [ARCHITECTURE.md](ARCHITECTURE.md) §10：

| 档 | 做法 | 目的 | 成本 |
|---|---|---|---|
| A | 线性 probe：top-k 槽向量池化后与候选答案 embedding 比相似度 | 最快回答"槽向量里有没有答案信息" | 几小时 |
| B | soft prefix：top-k 槽向量经 MLP 投影成 8 个软 token，前置到 `inputs_embeds` | 生成式潜层读出，输入 1500 → ~8 token | 1–2 天 |
| C | cross-attention：问题向量当 query attend 全部槽向量 | 按需寻址的记忆读取 | 需改 attention + KV cache |

对比口径固定为三臂：`prompt`（基线）/ `text_embeds`（明文嵌入对照）/ `latent`，同检索结果、同解码参数，
记录准确率 + 输入 token 数 + 延迟。

## 上一版进展（v9 重构轮，2026-10-20）——确定性 Bug 修复 + 逐题诊断 + AR 检索重写 + CR 强化 + Delta 对齐

**结论先行**：v9 是**代码 / 可诊断性重构**。本轮按硬约束只做 dry-run（`py_compile` / CPU 单测 / `--dry-run`），
**未跑任何真实评测、未加载模型权重、未初始化 CUDA、未占用 GPU**，因此**没有任何新指标数字**。
下文出现的 v7 / v8 指标全部是重构前代码的历史结果，不能当作 v9 效果。

### 1. v8 三个确定性 Bug 修复（`v8_slotpool.py`，接口零变更）

| Bug | 症状 | 修复 |
|---|---|---|
| Bug1 `cur_ts` 第二轮污染 | 首轮扫描结束后 `cur_ts` 停在**全篇最后一个**时间戳，第二轮窗口/状态判定直接复用它（状态轮还继续改写它），导致位于任何时间标记**之前**的单元被写入"文档末尾"时间戳 | 首轮一次性预计算逐句 `resolved_ts[i]`（= 该位置之前最近一次显式时间戳），两轮只读；窗口无显式标记时继承 `resolved_ts[i-1]` |
| Bug2 512 词 span 偏移 | `spans` 用未截断的 `len(ws)` 记账、却只 `extend(ws[:512])`，长单元 span 越过自身边界吃进后续单元的词；零 token 兜底分支还 `off += 1`，使后续单元 tokv 整体错位 | 先按 `UNIT_MAX_WORDS=512`（env `V8_UNIT_MAX_WORDS`）截断再记 span；零 token 分支**不**推进 `off` |
| Bug3 `ev` 时间触发 | `ev` 只看事件动词，纯时间句（含 `In 2001` 但无过去时动词）落入 state，与"时间标记或事件动词 → episodic"的设计矛盾 | `ev` = 事件动词 **or** 窗口内显式时间标记；state 只收无显式时间标记且无事件动词的稳定陈述句 |

- 合成 150 句上下文：旧版有 31 个情景单元被写入"文档末尾时间戳"，新版仅剩 2 个（窗口内确实含该时间戳句，属正常显式标记）。
- 接口零变更：episodic unit 字段不变；state unit 只新增附加字段 `ts_inherited`。`run_ar_v7.py` 只用 `start/end/texts/metas`，无需改动。
- **披露**：v8 四切片指标（0.7100 / 0.5120 / 0.3820 / 0.3167）由**修复前**代码产出，修复后**未重跑**。

### 2. AR 检索重构（`v9_slotpool.py`，`AR_MODE=v9`）

针对 v8 的五个检索问题逐条改：

1. **非重叠索引**：v8 用 3 句滑窗 stride=1，同一句最多出现在 3 个单元里（池膨胀 3 倍、Top-K 实际只覆盖 K/3 个不同句、位置排序有歧义）。v9 改为 `stride == block_sents`，`V9_BLOCK_SENTS=1` 即单句索引，每句恰好属于一个单元，Recall@K 与位置排序良定义。
2. **先按分数选 Top-K、再按原文位置排序**：`retrieve()` 只按分数降序返回；`retrieve_ordered()` / `readout()` 只对**已选中的 K 个**按 `start` 排序，绝不做"全池按位置排序后截断"。
3. **EventQA 后向非对称扩展**：`expand_backward(anchor, back=16, ahead=4)`——以"已发生事件"锚点为准，向前多取、向后少取（env `V9_EQA_BACK` / `V9_EQA_AHEAD`，含低/高边界裁剪）。v7 只向后取 4 块、v8 只在命中窗口两侧各扩 2 个，锚点之前的证据都拿不到。
4. **时间戳参与打分与时序重排**：单元编码文本 = `In <ts>: <原文>`（`ts` 可为显式或继承），日期 token 因此参与 IDF-MaxSim；`texts` 仍保存原文，**拼给 LLM 的上下文没有人工前缀**。`temporal_rerank` 按 `|year_unit − year_query|` 指数衰减加成（`V9_TEMPORAL_TAU=2.0` / `V9_TEMPORAL_BONUS=0.15`），无年份单元不加分也不罚分。
5. **第 18 层隐状态混合稠密打分**：`hybrid = (1−α)·IDF-MaxSim + α·((cos+1)/2)`，`α=V9_L18_ALPHA=0.3`、层号 `V9_L18_LAYER=18`；provider 注入（`fn(texts, kind) -> [N,d]`）或手工 `attach_layer18()` + `retrieve(q_hidden=...)`。provider 缺失 / 抛错 / 形状不符一律**静默退化为纯词面**，无模型路径不崩。

另修一处 v8 读出缺陷：新增 `_cap_before_sort()`，把"全候选按位置排序 → `lines[:CTX_SENTS]`"改为"先按分数优先裁到情景配额（EventQA 以 anchor 为中心裁窗）→ 再按原文位置排序"。旧写法会截掉高分尾部单元（单测构造 11 个候选、旧实现丢 8 个）。

### 3. 逐题诊断 JSONL + LongMemEval judge（两个新文件，仅脚本，未跑）

- **`ar_diag.py`**：`record()` 产出逐题字段 `retrieved_slot_ids / retrieved_scores / retrieved_texts / retrieved_starts / used_slot_ids / answer_in_retrieved / answer_in_context / answer_hit_slot / answer_matches / recall_at_k / diag_status`。`hits` 必须是**分数降序**（Recall@K 依赖此顺序），`used_ids` 是实际拼进 prompt 的槽位。口径：NFKC + 小写 + 撇号删除 + 连字符折叠后子串匹配；gold 为 rubric（偏好类）标 `rubric_not_applicable` 并**剔除分母**；只有 `idxs` 的 legacy 记录标 `legacy_idxs_only`，分母只算可计算记录。CLI 支持 `--input / --summary-out / --enrich / --selftest`。
- **`eval_longmemeval_judge.py`**：官方 `evaluate_qa` 口径的 LLM-as-Judge 后处理。6 题型模板（通用 / temporal off-by-one / knowledge-update 旧值容忍 / preference rubric）+ abstention 模板；`--prompt-source official` 可直接 import 官方 `get_anscheck_prompt`。OpenAI 兼容 `/chat/completions`（`temperature=0, max_tokens=10`）、线程池并发、重试退避、sha1 缓存断点续跑；verdict 复刻官方 `'yes' in text` 语义，另存 `verdict_strict` 审计。汇总含 overall / by_question_type，`parse_fail`/`error` 单列，abstention 单列不进主分母。`--dry-run` 只渲染 prompt 落盘（**零网络**），`--selftest` 用 monkeypatch 哨兵保证零网络。
- **接入**：`run_ar_v7.py` 的 v7 / v8 / v9 / plain 四种模式每题都写诊断字段；诊断异常只写 `diag_status=diag_error`，不影响主指标。

### 4. CR 强化（`v9_cr.py`，`CR_MODE=v9`；默认 `CR_MODE=v7`，历史行为不变）

1. **主语 + 关系双阈值覆写**：`StatePool._should_merge` 覆写为"主语归一化精确匹配 + 关系谓词余弦 > `V9_CR_REL_THETA=0.80`"，替换 v8 的 mean-cos(0.92) 单阈值；以 `add_batch` 的 argmax 槽（实际覆写目标）为准。
2. **Beam=6 ego-graph 第 1 跳**：Hop1 取 6 个主语门控种子后立刻并入每个种子的 ego 邻域（同主语其他属性 + 出边实体，`V9_CR_EGO_K=12`），不再只留 2 个孤立种子。
3. **双跳链谓词联合打分**：`Score_chain = 0.40·subj_cov(target) + 0.30·(bridge1+bridge2)/2 + 0.30·(rel_mid+rel_tgt)/2`，两条边的谓词都与 query 谓词比较；输出按 serial 升序（配合"编号越大越新"）。
4. **反事实强约束 Prompt**（`build_cr_prompt`）：OUTDATED 值禁止输出 / 冲突时只认最大编号 / 作答前做反事实自检 / 缺证据回 `unknown`。
5. **接线**：`run_group_b2_v7.py` 加 `CR_MODE=v7|v9`（默认 v7），v9 分支注入 `v9_cr`、可选 CF Prompt、每题追加 `cr_dbg`，OUT 改 `v9groupB2_*.jsonl`，FINAL 打印 `CR_MODE`；非法值直接退出。

### 5. Delta Memory 读出对齐（`run_ar_v7.py`）

1. **写入移到查询前**：每个 context 起点 `reset_delta_mem_states` **一次**；随后、建池之前 `prepare_delta_memory(ctx)`——开写 → 512-token chunk 流式 forward → `finally` 关写 → 快照 clone。原"所有题目问完后才写 delta"的块已删除。先写后建池，保证 `AR_V9_L18=1` 时单元编码与 query 编码看到同一记忆状态。
2. **查询期禁止 reset**：题目循环内不再出现 `reset_delta_mem_states`。
3. **逐题快照恢复**：每题生成前 `delta_restore(snap)`（官方 `load_delta_mem_online_state`；缺该 API 时退回 `iter_delta_mem_modules` 手工 clone/restore；两套都缺则 `[warn]` 退化为 no-op，不中断跑分）。`AR_DELTA_RESTORE=0` 可关。
4. **`inputs_embeds` 读出路径**：`AR_INPUTS_EMBEDS=1` 时 `EMB[ids]` → `model.generate(inputs_embeds=..., attention_mask=...)`，`output_len` 按"只返回新 token"计；OOM 渐进截短在两条路径下都生效。
5. **可观测性**：每题 JSONL 新增 `delta: {write, restore, inputs_embeds, snapshot_layers}`；`--dry-run` 打印 delta 开关与 `state_api=ok/fallback`。

### 6. dry-run 结论（本轮唯一的验证形式）

| 检查 | 结果（v9 收口轮结束时，2026-10-21） |
|---|---|
| `py_compile` | 仓库内 **30/30** 个 git 跟踪的 `.py` 全部通过 |
| CPU 单测 9 个文件 | `test_v9_slotpool` **284** / `test_v9_cr` 143 / `test_v8_slotpool` 86 / `test_ar_v9_wiring` 105 / `test_ar_delta_alignment` 37 / `test_run_manifest` 27 / `test_ar_delta_write_strategy` 47 / `test_cr_delta_log` 84 / `test_v6_parse` —— **全部 PASS** |
| `ar_diag.py --selftest` | **121/121** |
| `eval_longmemeval_judge.py --selftest` | **76/76**（零网络、零权重） |
| 消融环境重跑 | `V9_L18_WHITEN=0`、`V9_L18_REUSE_EPI=0` 下 `test_v9_slotpool` 同样全绿 |
| `run_ar_v7.py --dry-run` | `AR_MODE=v7 / v8 / v9 / plain` 四种均退出码 0，末行 `cuda_initialized=False model_loaded=False` |
| `eval_longmemeval_judge.py --dry-run` | 合成 3 题渲染 3 条 prompt，未发起任何网络请求 |
| 守卫断言 | 零网络、零 `transformers/pyarrow/deltamem` 导入、无文件写入、无残留进程 |

环境事实：本机 Python 3.9.6 + torch 2.8.0（CPU，`device_count=0`），`transformers` / `pyarrow` / `deltamem` **缺失**——真实跑分路径本机不可执行，dry-run 已证明不依赖它们。

### 7. 未验证项声明（必读）

以下**全部未验证**，引用本文档时须一并说明：

1. **没有任何 v9 指标**。真实 tokenizer / 模型下的 v9 建池与生成、以及五项检索改动对 SF / Recall 的影响，均未测量。
2. **v8 三个 Bug 修复对历史指标的影响未知**：0.7100 / 0.5120 / 0.3820 / 0.3167 是修复前数字，未重跑对照。
3. **`AR_V9_L18=1` 的 provider 前向未跑**（仅静态审查，默认关闭）；真实模型取 `hidden_states[18]` 的显存与耗时未知。
4. **Delta Memory**：`/share/delta-Mem` 实际版本是否含 `get/load_delta_mem_online_state` 未确认（已做 fallback）；真实 Qwen3 `generate(inputs_embeds=...)` 行为、写入/恢复开销与指标影响未测。
5. **`run_group_b2_v7.py` 未运行**：模块级 `from_pretrained` + `/share` 路径、无 `--dry-run`；CR v9 只由 `test_v9_cr.py` 覆盖池 / 双跳链 / Prompt / 调用序列。
6. **`eval_longmemeval_judge.py` 未做真实判题**（未配 API key、未联网），只验证了 prompt 渲染与自测。
7. **`ar_diag.py` 未跑真实 AR 结果**：只在 legacy CR JSONL 上验证不崩（全部标 `legacy_idxs_only`，applicable=0）。

### v9 env 参数表

| 变量 | 默认 | 含义 |
|---|---|---|
| `AR_MODE` | `v7` | `v7` / `v8` / `v9` / `plain` |
| `AR_V9_EQA_BACK` | 16 | EventQA 锚点**前**取几句 |
| `AR_V9_EQA_AHEAD` | 4 | EventQA 锚点**后**取几句（非对称） |
| `AR_V9_CTX_MAX_UNITS` | 40 | 拼给 LLM 的单元上限 |
| `AR_V9_TOPK_STATE` | 4 | 状态池附带单元数 |
| `AR_V9_L18` | 0 | 第 18 层隐状态混合稠密打分开关 |
| `AR_V9_L18_LAYER` | 18 | 取哪一层 hidden_states |
| `V9_BLOCK_SENTS` | 1 | 非重叠单元句数（1=单句索引） |
| `V9_UNIT_MAX_WORDS` | 512 | 单元词级硬截断 |
| `V9_TOKV_BF16` | 1 | 逐 token 向量 bf16 存储（省内存） |
| `V9_TEMPORAL_TAU` / `V9_TEMPORAL_BONUS` | 2.0 / 0.15 | 时序重排衰减尺度 / 加成上限 |
| `V9_L18_ALPHA` | 0.3 | 稠密通道权重 α |
| `V9_CR_REL_THETA` | 0.80 | CR 双阈值覆写的谓词余弦阈值 |
| `CR_MODE` | `v7` | `v7`（基线）/ `v9`（CR 强化） |
| `V9_CR_BEAM` / `V9_CR_EGO_K` | 6 / 12 | CR Hop1 种子数 / ego 邻域上限 |
| `V9_CR_CF_PROMPT` | 1 | 反事实强约束 Prompt 开关（仅 CR_MODE=v9） |
| `AR_DELTA_WRITE` / `AR_DELTA_RESTORE` | 0 / 1 | 查询前写 delta / 逐题恢复快照 |
| `AR_INPUTS_EMBEDS` | 0 | 读出走 `inputs_embeds` 路径 |

## 上一版进展（v8，2026-10-19）——双模态潜槽位记忆 DM-ELSM（AR 管线）

v8 针对 Accurate_Retrieval（AR）管线引入双模态潜槽位记忆（DM-ELSM: Dual-Modal Episodic Latent Slot Memory，
`v8_slotpool.py`，经 `run_ar_v7.py` 的 `AR_MODE=v8` 启用）。**CR（FactConsolidation）路径保持 v7 不动，向后兼容。**

> ⚠️ 本节四切片指标由 **v8.1 Bug 修复前**的代码产出（见顶部 v9 第 1 节三个确定性 Bug），修复后**未重跑**。

### v8 方法

1. **情景/状态自适应分流 `extract_adaptive_units`**：
   - 情景微块（episodic）：3 句滑窗（stride=1），任一句含时间标记或过去时事件动词的窗口成为独立检索单元；
   - 状态单元（state）：无时间标记、无事件动词的稳定事实句，单句成窗，StatePool 近重复语义合并（mean cos > `V8_STATE_THETA=0.92` 时新文本覆写旧槽）；
   - 时间戳继承：显式时间标记（In 1997 / On March 3, 2001 / in the 1990s）记为逐句 `resolved_ts`，后续无标记窗口继承"起点之前最近一次"时间戳（v8.1 修复，见顶部）。
2. **EpisodicFastPool：全池无截断 IDF-MaxSim 检索**：废除 v7 的 n>3500 时先取 mean-cos Top-2500 再精算 MaxSim 的截断；对全池每个微块分批（`V8_BATCH=64`）完成 ColBERT MaxSim，query 逐 token 用池内 IDF 加权。情景池只追加，永不覆写、永不淘汰（AR 无"新值覆盖旧值"语义）。
3. **邻域扩展**：检索命中情景微块后，按句序向前后各扩展 `AR_V8_NEIGHBOR=2` 个相邻窗口（时间连续性）。
4. 拼给 LLM 的单元上限 `AR_V8_CTX_MAX_UNITS=40`，状态池附带 `AR_V8_TOPK_STATE=4` 个状态单元。

### v8 vs v7 对照表（AR 四切片，v100-2 全量）

| 切片 | n | v7 基线 | v8 (DM-ELSM) | Δ |
|---|---|---|---|---|
| eventqa_65536 | 500 | 0.7240 | 0.7100 | −0.0140 |
| eventqa_131072 | 500 | 0.5620 | 0.5120 | −0.0500 |
| eventqa_full | 500 | 0.3500 | **0.3820** | **+0.0320** |
| longmemeval_s* | 300 | 0.2767 | **0.3167** | **+0.0400** |

详细分题型与运行备注见 `results/summary_ar_v8.md`。

### v8 结论与披露

1. **长上下文切片 v8 超基线**：eventqa_full（40.7万–74.6万 token）+0.0320、longmemeval_s*（36万–39万 token）+0.0400。全池无截断检索（废除 n>3500 截断）在超长上下文上收益最大。
2. **短上下文略低**：eventqa_65536 −0.0140、eventqa_131072 −0.0500。微块粒度（3 句滑窗）+ 40 单元拼块上限在 65k/131k 上弱于 v7 的 128-token 块检索。
3. **长上下文建池成本高**：eventqa_full 各 ctx 建池 627–8894s（episodic 1.2万–2.2万 / state 1.2k–4.9k 单元，dup=0）。
4. **长上下文 OOM 修复**：`v8_slotpool.encode_words` 分批编码（`V8_ENCODE_BS=1024`）+ 逐切片 `.cpu()` 回传；`run_ar_v7.py` generate 前 `empty_cache`，OOM 渐进截短重试（16k/8k/4k）。
5. **GPU 环境**：GPU1 全程被外部进程占用（28GB），四切片用 GPU0+GPU2 并行跑完；GPU 选择不影响指标（greedy decode）。
6. 上表的 v7 基线为 2026-10-18 同脚本 `AR_MODE=v7` 的全量结果（见下节），四切片均为 500/500（eventqa）与 300（longmemeval 5 ctx × 60 题）。

## 最终结论（v7，2026-10-18）——自适应弹性槽位池打通 6K→262K 全长度

Conflict_Resolution 全量 8 切片 × 100 题（共 800 题，无采样）：

| 切片 | v7 SF | v6 SF | v5/v4 SF | 池 final | overwrites | 淘汰 |
|---|---|---|---|---|---|---|
| sh_6k | 0.94 | 0.94 | 0.86（v4） | 292 | 163 | 0 |
| sh_32k | 0.91 | 0.91 | — | 1444 | 866 | 0 |
| sh_64k | 0.89 | 0.89 | 0.87（v4） | 2826 | 1754 | 0 |
| **sh_262k** | **0.85** | **0.45** | — | 10721 | 7612 | 0 |
| mh_6k | 0.67 | 0.67 | 0.51（v5）/0.08（v4） | 293 | 162 | 0 |
| mh_32k | 0.63 | 0.63 | — | 1444 | 866 | 0 |
| mh_64k | 0.59 | 0.59 | 0.45（v5） | 2826 | 1754 | 0 |
| **mh_262k** | **0.49** | **0.13** | — | 10721 | 7612 | 0 |

v6 在 262k 上触顶 `POOL_MAX=5000`（上下文约 18.3k 条事实、约 11.3k 个独立 (subj,rel) 状态），FIFO 提前逐出早期槽位，导致单跳只用 44% 的幸存率（实测 0.45 = 5000/11304），多跳按 $p^k$ 幂次崩塌（实测 0.13 ≈ 0.442^2.5~3）。v7 去掉固定上限后两条曲线基本拉平：6K→262K 单跳 0.94→0.85（−9pp），多跳 0.67→0.49（−18pp）。

### v7 改动（`v7_slotpool.py`，runner `run_group_b2_v7.py`）

1. **容量自适应（无损弹性倍增缓存）**：默认 `B2_POOL_MAX=0`，池按 2048→4096→8192→16384 自动翻倍，容量由流内独立 (subj,rel) 状态数决定，262k 停在 10721 槽、`evicts=0`。预分配连续张量 `_k_subj_buf/_k_rel_buf` 取代每步 `torch.stack(k_subj)`，消除 O(N²) 拷贝。
2. **二级路由 O(1) 覆写**：`gid_to_idx`（精确 `subj||rel` 哈希）+ `subj_buckets`（主语倒排桶）两级命中，覆写不再全池扫描；桶内仅 1~4 槽时语义覆写也只算桶内余弦。
3. **检索自适应（治大池噪声）**：
   - 主语特异性长度加权 `cov * (1 + 0.025 * min(L_subj, 5))`，防 "Tunisia" 抢 "Tunisia national football team" 的 Beam；
   - 双向桥接覆盖率 `0.65*cov(cand→obj) + 0.35*cov(obj→cand)`，打通 "Zimbabwe"(1 词) ↔ "Zimbabwe government"(2 词) 跨粒度搭桥；
   - 解析侧把主语尾部 ` government` / 前缀 `government of X` 归一到 (实体, 谓词)，补上"国家 → 国家政府首脑"这一跳。
4. **受限容量可选（消融用）**：`B2_POOL_MAX>0` 时启用拓扑桥接度保护淘汰——按 `bridge_deg`（该槽的 subj 是否为他人 obj、obj 是否为他人 subj）优先淘汰 `deg=0` 的最旧孤立叶节点，O(1) 交换删除，替代纯 FIFO。

### v7 实测备注

- 冒烟 mh_6k 复跑 0.67（pool 293/ow 162，与 v6 的 292/163 差 1 槽，来自桶路由），确认改动不伤短上下文。
- mh_262k 0.49 低于 p² 上界（0.85²≈0.72）：剩余缺口来自 10.7k 槽位下 Beam=2 的候选竞争与链尾实体歧义，属检索侧残余，非容量问题。按现数字定稿，后续如需再调 Beam/桥接阈值。
- 全部 8 切片 jsonl 各 100 条，SF 与 FINAL 日志一致。

## 上一版结论（v6，2026-10-18）——一套代码通吃 sh + mh

| 切片 | B2 v6 SF | B2 v5 | B2 v4 | A chunked 基线 | Δ vs v5 | Go/No-Go |
|---|---|---|---|---|---|---|
| factconsolidation_sh_6k | **0.940** | — | 0.860 | 0.12 | +8pp（vs v4） | **Go** |
| factconsolidation_sh_64k | **0.890** | — | 0.870 | 0.23 | +2pp（vs v4） | **Go** |
| factconsolidation_mh_6k | **0.670** | 0.510 | 0.080 | 0.02 | **+16pp** | **Go** |
| factconsolidation_mh_64k | **0.590** | 0.450 | — | — | **+14pp** | **Go** |

v6 = v5_slotpool.py 两处极小修补升级而来（`v6_slotpool.py`），同一套代码同时跑 sh 与 mh：

1. **VP_RE 谓词正则补全 + 长谓词优先**：`parse_fact_triple` 优先匹配非裸系动词的具体谓词短语（禁止退化为 is/was）；新增 WHERE 倒装从句解析（双系动词句型 `The X where SUBJ was VERB is OBJ`，30 词条名词→谓词映射），修复教育类冲突对的 subj 解析污染（"The univeristy where Elena Ceaușescu was educated" → subj=Elena Ceaușescu, rel=educated at）。
2. **(subj, rel) 精确组标识 gid 原位覆写**：`gid = f"{subj.lower()}||{rel.lower()}"` 命中即无条件原位覆写 texts/k_obj/E_obj（找回 v5 漏掉的覆写），双重命中余弦覆写退为二级兜底；每实体保留多条不同 rel 槽位，多跳检索命中跳板实体时展开其全部属性分支（Hop2+ 按问题谓词相关性 Top-2 进块）。

### v6 实测

- sh_6k：SF=0.94、pool=292、ow=163；sh_64k：SF=0.89、pool=2826、ow=1754
- mh_6k：SF=0.67、pool=292、ow=163；mh_64k：SF=0.59、pool=2826、ow=1754
- mh 残余失败主因：链尾候选语义竞争（Grosseto/Harrisville 类小城复用误答），检索侧已尽，需模型侧候选重排再推。
- 复算验证：四路 jsonl 各 100 条，SF 与 FINAL 日志一致（EM=SF，missing=0）。

## 历史结论（v5，2026-10-17）

| 切片 | B2 v5 SF | B2 v4 | v3 | A chunked 基线 | Δ vs A | Go/No-Go（≥20pp） |
|---|---|---|---|---|---|---|
| factconsolidation_sh_6k | —（未重跑） | **0.860** | 0.850 | 0.12 | **+74pp** | **Go**（v4） |
| factconsolidation_sh_64k | —（未重跑） | **0.870** | 0.300 | 0.23 | **+64pp** | **Go**（v4） |
| factconsolidation_mh_6k | **0.510**（51/100） | 0.080 | — | — | — | Go（vs v4 +43pp） |
| factconsolidation_mh_64k | **0.450**（45/100） | — | — | — | — | 待评估 |

- **单跳（sh）任务**：v4 已双双 Go（0.86/0.87），v5 未重跑 sh 切片（v5 主打多跳）。
- **多跳（mh）任务**：v4 仅 0.08（断链），v5 三元组解耦潜键 + 主语门控多跳接力提升至 0.51/0.45；池从 294→2847（×9.7）时 SF 仅降 6pp，规模适应性良好。

## v4 → v5：三元组解耦潜键 + 主语门控多跳接力（Beam Chaining）

v4 的失败模式（mh_6k SF=0.08）：多跳链中段事实检索不到（Top-15 无搭桥机制）、洲级/干扰事实混入块。v5（`v5_slotpool.py`，由主脚本 `from v5_slotpool import *` 覆盖内置 SlotPool）：

1. **三元组解耦潜键**：`parse_fact_triple`（VP_RE 谓词正则 + PREFIX_REL_RE 前缀关系）把事实拆为 (subj, rel, obj)，每槽存均值键 (k_subj, k_rel, k_obj) 与逐 token 向量矩阵 (E_subj, E_rel, E_obj)。
2. **双重命中覆写**：cos(k_subj)>0.82 且 cos(k_rel)>0.70 时原位覆写，并同步更新出键 k_obj/E_obj（多跳搭桥核心）。
3. **主语门控多跳检索** `retrieve_multihop_context`：Hop1 用 maxsim_coverage（ColBERT MaxSim）做主语硬门控（subj_cov≥0.62），Hop2-4 拿上一跳 k_obj/E_obj 搭桥（bridge_sim≥0.65），Beam=2，按跳序拼块。

### env 参数表

| 变量 | 默认 | 含义 |
|---|---|---|
| `B2_POOL_MAX` | 0（v7）/5000（v6） | v7：0=无损弹性容量，>0 触发拓扑桥接保护淘汰；v6：FIFO 上限 |
| `B2_INIT_CAP` | 2048 | v7 弹性缓存初始容量（之后自动翻倍） |
| `B2_THETA_SUBJ` | 0.82 | 覆写：主语键余弦阈值 |
| `B2_THETA_REL` | 0.70 | 覆写：关系键余弦阈值 |
| `B2_BRIDGE` | 0.65 | Hop2+ 搭桥硬门控阈值 |
| `B2_SUBJGATE` | 0.62 | Hop1 主语硬门控阈值 |
| `B2_HOPS` | 4 | 最大跳数 |
| `B2_BEAM` | 2 | 每跳保留槽数 |

### 运行参数实测

- mh_6k：pool=294、overwrites=161、约 26min（GPU2）
- mh_64k：pool=2847、overwrites=1733、约 25min（GPU9）

## v3 → v4 修复（救 sh_64k）

v3 的 4 个瓶颈（sh_64k SF=0.30）：①POOL_MAX=512 触顶 FIFO 盲淘（64k 约 2600+ 独立事实）；②context_block 全量拼 512 条（~7500 token）稀释注意力；③KEY_TAIL_CUT=2 漏覆写多词宾语 + 均值 embed 模板词撞车；④131 chunk 累积 delta 状态未在读出前 reset。

v4 改动：
1. **大池 + 主谓组去重**：POOL_MAX 默认 5000（env `B2_POOL_MAX` 可覆盖），THETA 默认 0.72；新增主谓组去重——`group_id = strip_object` 归一化文本，同组新事实**无条件替换**旧槽（`n_group_overwrites` 计数），余弦 θ 只负责非同组语义去重。保证"同主谓新值覆盖旧值"语义确定生效。
2. **Top-K 检索读出**：`SlotPool.retrieve_context_block(query_text, top_k=15)`——用问题加权 key 检索池内 Top-15 槽位按 serial 排序拼块（并列时 serial 降序偏置新鲜度）。每题输入仅 ~430 token，替代 v3 的 ~7500 token 全量拼接。
3. **weighted_key_embed**：STOPWORDS 过滤（the/of/is/... 等）+ 非句首大写专有名词 2.5× 权重（offset mapping 摊权重到子词），避免不同实体因共享句式模板被误覆写；事实 key = `weighted_key_embed(strip_object(fact))`。`strip_object` 用词边界谓词正则（修复了子串匹配把 'British' 截成 'Brit' 的 bug）。
4. **读出前 reset**：query 循环前调用 `reset_delta_mem_states(model)`，消除 131 chunk 累积的 delta 残差干扰。

v4 运行参数：sh_64k pool=2598/5000、overwrites=2598（组去重约 1982）、evicts=0、约 60min；sh_6k pool=275、overwrites=275、evicts=0、约 26min。

## 修复历史

### 第一轮（v2，commit `244e4d8` → `2b073fb`）：5 个致命 Bug

初版代码存在 5 个致命 Bug，导致误判 No-Go：

| 指标 | 修复前 | v2 修复后 |
|---|---|---|
| sh_6k SF | 0.110 | 0.850 |
| sh_64k SF | 0.010 | 0.300 |
| 事实提取（sh_6k） | 7 条（98% 漏召） | 455 条 |
| overwrites | 0 | 198 / 991 |
| 输出症状 | 复读 "I have learned the facts" / 半截乱码 | 正确实体回答 |

5 个 Bug 与修复：
1. **事实切分漏召**：punkt 加载失败回退切分把多行拼成一行，`re.match` 每行只留第 1 条 → 改全局 `re.findall` 编号提取（`(?:^|\s)(\d+)\.\s+...`），<10 条回退按句切分 `(?<=[.!?])\s+|\n+`。
2. **模板 KV 污染**：从含 sys_msg/mem_tpl 的 ids 按字符比例映射截 KV，把 Memorize 指令截进池子 → 改为对纯事实文本 `tok(fact, add_special_tokens=False)` 单独 forward 取 key。
3. **decode SL 步进错误**：手写 greedy decode `SL += 1` 应为 `SL += cur.shape[1]`，第 2 token 起 RoPE 位置全乱。
4. **跨 chunk KV 拼接违反 RoPE**：不同 chunk 的 Key 相位重叠/非单调，Query cache_position 可能小于记忆 Key 位置（Δpos 为负）→ 彻底放弃 KV 拼接读出。
5. **key 切分/whitening 失效**：`KEY_TAIL_CUT=5` 盲切掉主语、pool≤3 时去中心化归零 → `KEY_TAIL_CUT=2`；去中心化仅 len(pool)>10 启用，否则原始 embedding 均值 + L2。

v2 架构（clean 重编码读出）：
- 写入：512-token chunk 流式 forward（与官方 delta 写入口径一致），槽位池存事实文本 + 潜层 Key（纯事实文本 embedding 均值）；新事实 key 与池内 key 余弦 > θ=0.70 → 原位覆写；池满 FIFO。
- 读出：**clean 重编码**——把存活槽位事实拼成 System+RAG 模板干净 context 块，单次 `model.generate(do_sample=False)`。绕开 RoPE 拼接与 cache_position 全部坑。

## 文件

- `ARCHITECTURE.md` — **当前预期架构设计**（设计不变量、数据流、组件契约、两条潜层读出路径、参数表、实现状态矩阵、风险与演进路线）
- `v9_slotpool.py` — **v9 AR 检索**（非重叠单句/微块索引 + 分数优先 Top-K + EventQA 后向非对称扩展 + 时间戳参与打分/时序重排 + 第 18 层混合稠密打分；`AR_MODE=v9`）
- `v9_cr.py` — **v9 CR 强化**（主语+关系双阈值覆写 + Beam=6 ego-graph + 双跳链联合打分 + 反事实 Prompt；`CR_MODE=v9`）
- `ar_diag.py` — 逐题诊断（`retrieved_* / answer_in_retrieved / answer_in_context / recall_at_k`）+ Recall@K 汇总 CLI
- `eval_longmemeval_judge.py` — LongMemEval 官方口径 LLM-as-Judge 后处理（`--dry-run` / `--selftest` 零网络）
- `test_v9_slotpool.py` / `test_v9_cr.py` / `test_ar_v9_wiring.py` / `test_ar_delta_alignment.py` — v9 CPU 单测（stub tok + 小 EMB，无需模型/GPU）
- `v8_slotpool.py` — **v8 双模态潜槽位记忆 DM-ELSM**（v8.1 已修 3 个确定性 Bug；`extract_adaptive_units` 情景/状态分流 + EpisodicFastPool 全池无截断 IDF-MaxSim；经 `run_ar_v7.py AR_MODE=v8` 启用，AR 四切片结果见顶部，**为修复前数字**）
- `test_v8_slotpool.py` — v8 CPU 单测（stub tok + 小 EMB，含 3 个 Bug 的回归用例）
- `results/summary_ar_v8.md` — v8 vs v7 四切片对照表与 longmemeval 分题型
- `v7_slotpool.py` — **v7 自适应弹性槽位池**（CR 主力，覆盖内置 SlotPool；默认无损弹性容量，8 切片全量结果见上）
- `run_group_b2_v7.py` — v7 runner（`import v7_slotpool` 注入 tok/EMB 后 `from v7_slotpool import *`；`CR_MODE=v7|v9`，默认 v7）
- `v6_slotpool.py` / `run_group_b2_v6.py` — v6（gid 精确覆写 + WHERE 倒装解析，POOL_MAX=5000）
- `v5_slotpool.py` — v5（三元组解耦潜键 + Beam Chaining）
- `run_group_b2.py` — B2 主脚本 **v4**（大池 + 主谓组去重 + Top-15 检索读出 + 加权 key，SF=0.86/0.87）
- `run_ar_v7.py` — Accurate_Retrieval runner（`AR_MODE=v7|v8|v9|plain`，含 `--dry-run`；每题写诊断字段；v7 四切片结果见下节，v8 见顶部）
- `smoke_b2.py` — smoke 测试（1 chunk + 2 个已知答案问题，验证召回/覆写/输出内容）
- `run_group_b.py` — B1 chunk 级增益脚本（已放弃，留作参考）
- `diag1_golden_chunk.py` — 诊断1：golden chunk 单块读出（证明 delta state 无独立读出能力，SF=0.12）
- `diag2_sim_dist.py` — 诊断2：句子相似度分布（确定 θ=0.70）
- `results/` — 运行日志与 jsonl 结果（v4 最终 + 历史版本见 git 历史）

## 运行

> **环境（2026-10-20 重建）**：`/share/conda/envs/latent2/bin/python`
> = Python 3.12.4 + torch 2.8.0+cu126 + transformers 4.51.3。
> 重建脚本 `env_rebuild.sh`，日志 `logs/env_rebuild.log`。
> **不要再带 `t28` / `pylib5` 这两个 PYTHONPATH 叠加层**——它们是旧环境缺包时的补丁，
> 在新环境下会让 `import torch` 直接失败（`cuda_initialized` 退化为 `n/a`）。
> 正确前缀：`PYTHONPATH=/share/delta-Mem:/share/MemoryAgentBench`。
> 架构说明见 [ARCHITECTURE.md](ARCHITECTURE.md)。

```bash
# 全量（100 题）。v7 默认 B2_POOL_MAX=0（无损弹性容量）
DELTA_MEM_SCAN_IMPL=torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/opt/conda/bin/python run_group_b2_v7.py factconsolidation_sh_262k <GPU_ID>

# 受限容量消融（触发拓扑桥接保护淘汰，替代 FIFO）
B2_POOL_MAX=5000 /opt/conda/bin/python run_group_b2_v7.py factconsolidation_mh_262k <GPU_ID>

# smoke（1 chunk + 2 问）
python smoke_b2.py
```

### dry-run（零模型 / 零 GPU，本机可跑）

```bash
# 配置 + 依赖可导入性 + 两项纯 CPU 自检；四种 AR_MODE 都可
python3 run_ar_v7.py eventqa_65536 --dry-run
AR_MODE=v9 python3 run_ar_v7.py eventqa_65536 --dry-run

# CPU 单测（stub tok + 小 EMB）
for f in test_v9_slotpool.py test_v9_cr.py test_v8_slotpool.py \
         test_ar_v9_wiring.py test_ar_delta_alignment.py; do python3 $f; done

# 诊断 / 判题脚本自测（零网络）
python3 ar_diag.py --selftest
python3 eval_longmemeval_judge.py --selftest

# judge 只渲染 prompt，不联网
python3 eval_longmemeval_judge.py --input results/ar_v9_eventqa_65536.jsonl \
    --dry-run --prompts-out /tmp/lme_prompts.jsonl
```

### v9 真实跑分（未执行，需 GPU 机器）

```bash
# AR v9（非重叠索引 + 分数优先 + EventQA 非对称扩展）
AR_MODE=v9 DELTA_MEM_SCAN_IMPL=torch PYTHONPATH=/share/delta-Mem:/share/MemoryAgentBench \
/share/conda/envs/latent2/bin/python /share/exp4group/run_ar_v7.py eventqa_65536 6

# 叠第 18 层混合稠密打分（默认关闭）
AR_MODE=v9 AR_V9_L18=1 ... run_ar_v7.py eventqa_65536 6

# 查询前写 delta + 逐题恢复快照
AR_MODE=v9 AR_DELTA_WRITE=1 ... run_ar_v7.py eventqa_65536 6

# CR 强化
CR_MODE=v9 /share/conda/envs/latent2/bin/python run_group_b2_v7.py factconsolidation_sh_64k 6
```

## Accurate_Retrieval（v7 已跑完，2026-10-18；v8 见顶部"上一版进展"，v9 见顶部"最新进展"）

跨任务族验证：`run_ar_v7.py` 在 v100-2（V100 32GB）上完成全部 4 个可用切片。
数据从磁盘读（`/share/latent/MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet`），
按 `metadata.source` 过滤。与 CR 管线刻意不同的两点：池语义改为"保留"而非"覆写"（`ChunkPool`，
句子块 128 token + 文本去重 + 只追加，永不淘汰）；检索加时序锚定（EventQA 用
`metadata.previous_events` 最后一条事件定位块，mean 粗筛 → MaxSim 精排，向后取 `AR_EQA_WIN=4` 块；
LongMemEval_s* 用 MaxSim rerank 的 top-12 块）。`AR_MODE=plain` 关掉锚定与 rerank，作同脚本内基线。

### AR 最终结果

| 切片 | SF | eventqa_recall | n | 上下文 tokens | 结果文件 |
|---|---|---|---|---|---|
| eventqa_65536 | **0.7240** | 0.7240 | 500 | 65,755–66,426 | `results/ar_v7_eventqa_65536.jsonl` |
| eventqa_131072 | **0.5620** | 0.5620 | 500 | 131,512–133,024 | `results/ar_v7_eventqa_131072.jsonl` |
| eventqa_full | **0.3500** | 0.3500 | 500 | 407,175–745,586 | `results/ar_v7_eventqa_full.jsonl` |
| longmemeval_s* | **0.2767** | 0.0000（不适用） | 300 | 360,057–392,906 | `results/ar_v7_longmemeval_s.jsonl` |

### 四点披露（必读）

1. **`AR_CTX_MAX_TOK=0` 全上下文不截断**：证据池只做块检索，不受模型 262k 窗口约束
   （eventqa_full 上下文最高 745k tokens 也不截尾部）。这是本方法与官方 stuff-context 基线的
   结构性差异，报告时须写明。
2. **LongMemEval 超窗为已知优势**：360k+ 上下文超出模型 262k 窗口，stuff 基线无法完整处理，
   而本方法（块检索 + clean 重编码）不受影响，该切片分数需结合此结构性优势解读。
3. **ruler_qa1_197K / ruler_qa2_421K 未跑**：被 `run_ar_v7.py` 显式拒绝——这两个切片需要
   choice 专用指标（非 substring exact match），脚本未实现，未产出任何结果。
4. **GPU 环境备注**：eventqa_full 原计划 v100-2 GPU1，但 GPU1 被外部进程占满（30.7GB，
   宿主 pid 不可见），改用 GPU0（与 claim_gpu 叠加峰值 ~11.8GB，无 OOM）。GPU 选择不影响
   指标（greedy decode、do_sample=False）。

### AR 运行环境

- 机器：v100-2（V100 32GB），复用 `launch_ar2.sh` 环境
- transformers 5.19.0，BASE=Qwen3-4B-Instruct-2507，ADAPTER=declare-lab/delta-mem_qwen3_4b-instruct
- 全部 500/500（eventqa）与 300 行（longmemeval 5 ctx × 60 题）落盘，SF 与 FINAL 日志一致
- 指标走官方 `post_process`：SF（`substring_exact_match`）+ EventQA 的 `eventqa_recall`

```bash
# eventqa_65536 / longmemeval_s* 各一次；GPU6/7 目前空闲（各 3.3 GiB 占用）
DELTA_MEM_SCAN_IMPL=torch PYTHONPATH=/share/delta-Mem:/share/MemoryAgentBench \
/opt/conda/bin/python /share/exp4group/run_ar_v7.py eventqa_65536 6
/opt/conda/bin/python /share/exp4group/run_ar_v7.py 'longmemeval_s*' 7
# 同脚本内基线（纯均值 top-K，不 rerank/不锚定）
AR_MODE=plain /opt/conda/bin/python /share/exp4group/run_ar_v7.py eventqa_65536 6
# smoke（1 个上下文 × 3 题）
AR_LIMIT_ROWS=1 AR_LIMIT_Q=3 /opt/conda/bin/python /share/exp4group/run_ar_v7.py eventqa_65536 6
```

## 技术备注（transformers 5.19.0）

- forward 返回 `DynamicCache`：用 `past.layers[l].keys / .values` 取 KV；
- `build_past_key_values` 直接构造 `DynamicCache` 并 `update(K, V, layer_idx)`；
- generate 与 delta 层在 cache 长度 ≠ ids 长度时崩溃（`_memory_affine_scan_torch` 空列表）——v2 中仅在纯 chunk 写入时用 forward，读出走 clean 重编码 + `model.generate`，不触发该路径。
