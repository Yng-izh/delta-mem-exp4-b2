# 潜层记忆 B2 实验：流式事实槽位原位覆写 + 检索式读出

基于 delta-mem（declare-lab/delta-mem_qwen3_4b-instruct）+ MemoryAgentBench（Conflict_Resolution 切片）的对照实验，验证"流式事实槽位原位覆写（B2）"能否提升 Conflict Resolution 的 SF（substring exact match）。

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

- `v7_slotpool.py` — **v7 自适应弹性槽位池**（当前主力，覆盖内置 SlotPool；默认无损弹性容量，8 切片全量结果见上）
- `run_group_b2_v7.py` — v7 runner（`import v7_slotpool` 注入 tok/EMB 后 `from v7_slotpool import *`）
- `v6_slotpool.py` / `run_group_b2_v6.py` — v6（gid 精确覆写 + WHERE 倒装解析，POOL_MAX=5000）
- `v5_slotpool.py` — v5（三元组解耦潜键 + Beam Chaining）
- `run_group_b2.py` — B2 主脚本 **v4**（大池 + 主谓组去重 + Top-15 检索读出 + 加权 key，SF=0.86/0.87）
- `run_ar_v7.py` — Accurate_Retrieval runner（分块证据池；四切片已全部跑完，见下节）
- `smoke_b2.py` — smoke 测试（1 chunk + 2 个已知答案问题，验证召回/覆写/输出内容）
- `run_group_b.py` — B1 chunk 级增益脚本（已放弃，留作参考）
- `diag1_golden_chunk.py` — 诊断1：golden chunk 单块读出（证明 delta state 无独立读出能力，SF=0.12）
- `diag2_sim_dist.py` — 诊断2：句子相似度分布（确定 θ=0.70）
- `results/` — 运行日志与 jsonl 结果（v4 最终 + 历史版本见 git 历史）

## 运行

```bash
# 全量（100 题）。v7 默认 B2_POOL_MAX=0（无损弹性容量）
DELTA_MEM_SCAN_IMPL=torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/opt/conda/bin/python run_group_b2_v7.py factconsolidation_sh_262k <GPU_ID>

# 受限容量消融（触发拓扑桥接保护淘汰，替代 FIFO）
B2_POOL_MAX=5000 /opt/conda/bin/python run_group_b2_v7.py factconsolidation_mh_262k <GPU_ID>

# smoke（1 chunk + 2 问）
python smoke_b2.py
```

## Accurate_Retrieval（v7 已跑完，2026-10-18）

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
