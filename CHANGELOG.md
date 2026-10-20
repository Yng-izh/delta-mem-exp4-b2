# CHANGELOG

本仓库的实验记录按版本追加。指标一律注明"由哪一版代码产出"，避免把重构前数字误当重构后结果。

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
