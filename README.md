# 潜层记忆 B2 实验：流式事实槽位原位覆写 + Clean 重编码读出

基于 delta-mem（declare-lab/delta-mem_qwen3_4b-instruct）+ MemoryAgentBench（Conflict_Resolution 切片）的对照实验，验证"流式事实槽位原位覆写（B2）"能否提升 Conflict Resolution 的 SF（substring exact match）。

## 最终结论（修复后，2026-10-17）

| 切片 | B2 修复后 SF | A chunked 基线 | Δ | 官方整段口径 | Δ | Go/No-Go（≥20pp） |
|---|---|---|---|---|---|---|
| factconsolidation_sh_6k | **0.850**（85/100） | 0.12 | **+73pp** | 0.40 | +45pp | **Go** |
| factconsolidation_sh_64k | **0.300**（30/100） | 0.23 | +7pp | 0.23 | +7pp | No-Go |

- sh_6k 两种口径均远超门槛：**潜层记忆 + 原位覆写思路成立**。
- sh_64k 未达门槛，主瓶颈是**池容量**：pool_final=512 触顶 POOL_MAX，64k 事实集（数千条）远超容量，覆盖率不足（overwrites=991 正常工作，非覆写失效）。
- 救 sh_64k 的方向：提高 POOL_MAX（如 2048）或改 LFU/相关性淘汰后重跑。

## 修复前 vs 修复后

初版代码存在 5 个致命 Bug（详见 git 历史 `244e4d8`），导致误判 No-Go：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| sh_6k SF | 0.110 | **0.850** |
| sh_64k SF | 0.010 | **0.300** |
| 事实提取（sh_6k） | 7 条（98% 漏召） | 455 条 |
| overwrites | 0 | 198 / 991 |
| 输出症状 | 复读 "I have learned the facts" / 半截乱码 | 正确实体回答 |

5 个 Bug 与修复：
1. **事实切分漏召**：punkt 加载失败回退切分把多行拼成一行，`re.match` 每行只留第 1 条 → 改全局 `re.findall` 编号提取（`(?:^|\s)(\d+)\.\s+...`），<10 条回退按句切分 `(?<=[.!?])\s+|\n+`。
2. **模板 KV 污染**：从含 sys_msg/mem_tpl 的 ids 按字符比例映射截 KV，把 Memorize 指令截进池子 → 改为对纯事实文本 `tok(fact, add_special_tokens=False)` 单独 forward 取 key。
3. **decode SL 步进错误**：手写 greedy decode `SL += 1` 应为 `SL += cur.shape[1]`，第 2 token 起 RoPE 位置全乱。
4. **跨 chunk KV 拼接违反 RoPE**：不同 chunk 的 Key 相位重叠/非单调，Query cache_position 可能小于记忆 Key 位置（Δpos 为负）→ 彻底放弃 KV 拼接读出。
5. **key 切分/whitening 失效**：`KEY_TAIL_CUT=5` 盲切掉主语、pool≤3 时去中心化归零 → `KEY_TAIL_CUT=2`；去中心化仅 len(pool)>10 启用，否则原始 embedding 均值 + L2。

修复后架构（B2 v2）：
- 写入：512-token chunk 流式 forward（与官方 delta 写入口径一致），槽位池存事实文本 + 潜层 Key（纯事实文本 embedding 均值）；新事实 key 与池内 key 余弦 > θ=0.70 → 原位覆写；池满 FIFO。
- 读出：**clean 重编码**——把存活槽位事实拼成 System+RAG 模板干净 context 块，单次 `model.generate(do_sample=False)`。绕开 RoPE 拼接与 cache_position 全部坑。

## 文件

- `run_group_b2.py` — B2 主脚本 v2（修复后，SF=0.85/0.30）
- `smoke_b2.py` — smoke 测试（1 chunk + 2 个已知答案问题，验证召回/覆写/输出内容）
- `run_group_b.py` — B1 chunk 级增益脚本（已放弃，留作参考）
- `diag1_golden_chunk.py` — 诊断1：golden chunk 单块读出（证明 delta state 无独立读出能力，SF=0.12）
- `diag2_sim_dist.py` — 诊断2：句子相似度分布（确定 θ=0.70）
- `results/` — 运行日志与 jsonl 结果（修复后重跑）

## 运行

```bash
# 全量（100 题）
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
