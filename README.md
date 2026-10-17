# 潜层记忆 B2 实验：流式事实槽位原位覆写 + 检索式读出

基于 delta-mem（declare-lab/delta-mem_qwen3_4b-instruct）+ MemoryAgentBench（Conflict_Resolution 切片）的对照实验，验证"流式事实槽位原位覆写（B2）"能否提升 Conflict Resolution 的 SF（substring exact match）。

## 最终结论（v5，2026-10-17）

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
| `B2_POOL_MAX` | 5000 | 槽位池上限（FIFO 淘汰） |
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

- `run_group_b2.py` — B2 主脚本 **v4**（大池 + 主谓组去重 + Top-15 检索读出 + 加权 key，SF=0.86/0.87）
- `smoke_b2.py` — smoke 测试（1 chunk + 2 个已知答案问题，验证召回/覆写/输出内容）
- `run_group_b.py` — B1 chunk 级增益脚本（已放弃，留作参考）
- `diag1_golden_chunk.py` — 诊断1：golden chunk 单块读出（证明 delta state 无独立读出能力，SF=0.12）
- `diag2_sim_dist.py` — 诊断2：句子相似度分布（确定 θ=0.70）
- `results/` — 运行日志与 jsonl 结果（v4 最终 + 历史版本见 git 历史）

## 运行

```bash
# 全量（100 题），v4 默认 POOL_MAX=5000 / THETA=0.72，可用 B2_POOL_MAX / B2_THETA 覆盖
DELTA_MEM_SCAN_IMPL=torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python run_group_b2.py factconsolidation_sh_6k <GPU_ID>

# smoke（1 chunk + 2 问）
python smoke_b2.py
```

环境：v100-1（V100 32GB），transformers 5.19.0，BASE=Qwen3-4B-Instruct-2507，ADAPTER=declare-lab/delta-mem_qwen3_4b-instruct。SRC 参数须传完整名（如 `factconsolidation_sh_6k`）。

## 技术备注（transformers 5.19.0）

- forward 返回 `DynamicCache`：用 `past.layers[l].keys / .values` 取 KV；
- `build_past_key_values` 直接构造 `DynamicCache` 并 `update(K, V, layer_idx)`；
- generate 与 delta 层在 cache 长度 ≠ ids 长度时崩溃（`_memory_affine_scan_torch` 空列表）——v2 中仅在纯 chunk 写入时用 forward，读出走 clean 重编码 + `model.generate`，不触发该路径。
