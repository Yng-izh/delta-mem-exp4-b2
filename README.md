# 潜层记忆 B2 实验：流式句子级 KV 槽位原位覆写

基于 delta-mem（declare-lab/delta-mem_qwen3_4b-instruct）+ MemoryAgentBench（Conflict_Resolution 切片）的对照组实验，验证"流式句子级 KV 槽位原位覆写（B2）"能否提升 Conflict Resolution 的 SF（substring exact match）。

## 结论：No-Go

B2 相对 A 基线（chunked 口径）无提升，未达到 ≥20pp 的 Go/No-Go 门槛。

| 切片 | B2 SF | A chunked 基线 | 差值 |
|---|---|---|---|
| factconsolidation_sh_6k | **0.110** (n=100) | 0.12 | -1pp |
| factconsolidation_sh_64k | **0.010** (n=100) | 0.23 | **-22pp** |

关键证据：
- **overwrites=0**：两路全程 0 次原位覆写（θ=0.70 去中心化余弦阈值在实际 streaming 前向中从未触发；sh_6k pool=7，sh_64k pool=85，ev=0）。
- 读出质量差：仅靠存活槽位 KV 拼接读出，输出多为空壳续写，64k 上明显弱于 chunked delta state。
- 此前 B1（chunk 级增益覆写）也已验证无效：sh_64k B=0.220 vs A=0.230；sh_6k B=0.120 与 gain=1.0 对照相同。

## B2 方案

仍按 512-token chunk 流式前向，但维护 KV Slot Pool：
- 每条事实句子（`"N. "` 行切分）存 Key 签名 + 该句 token 区间的各层 KV 片段；
- 覆写：新事实 key 与池内 key 去中心化余弦 > θ=0.70 → 原位替换旧槽 KV；池满 FIFO 淘汰；
- 读出：存活槽位 KV 拼成 `past_key_values`（DynamicCache），问题单独前向做 attention。

Key 签名方案（diag2 验证）：`embed_tokens` 均值 + 主谓 key（去句尾 5 token）+ whitening。离线分布：无关句 cos mean=-0.002，冲突对 top-5 mean=0.561。

## 文件

- `run_group_b2.py` — B2 主脚本（KV Slot Pool + 原位覆写 + 手写 greedy decode）
- `run_group_b.py` — B1 chunk 级增益脚本（已放弃，留作参考）
- `diag1_golden_chunk.py` — 诊断1：golden chunk 单块读出（证明 delta state 无独立读出能力，SF=0.12）
- `diag2_sim_dist.py` — 诊断2：句子相似度分布（确定 θ=0.70）
- `results/` — 运行日志与 jsonl 结果

## 运行

```bash
DELTA_MEM_SCAN_IMPL=torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python run_group_b2.py factconsolidation_sh_6k <GPU_ID>
```

## 技术备注（transformers 5.19.0）

- forward 返回 `DynamicCache`：用 `past.layers[l].keys / .values` 取 KV；
- `build_past_key_values` 直接构造 `DynamicCache` 并 `update(K, V, layer_idx)`；
- generate 与 delta 层在 cache 长度 ≠ ids 长度时崩溃（`_memory_affine_scan_torch` 空列表），改为手写 greedy decode 并显式传 `cache_position`（从池子长度起算）。
