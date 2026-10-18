# AR 四切片 v8 (DM-ELSM) vs v7 对照表

- 环境：v100-2（`/share/exp4group`），`AR_MODE=v8`，`run_ar_v7.py` + `v8_slotpool.py`（extract_adaptive_units + EpisodicFastPool 全池 IDF-MaxSim 检索，废除 n>3500 截断）。
- 指标：eventqa 切片为 substring_exact_match（=eventqa_recall）；longmemeval_s* 为 substring_exact_match（SF）。n = 5 ctx × 每题数（eventqa 100，longmemeval 60）。
- 结果文件：`/share/exp4group/results/ar_v8_<src>.jsonl`。

## 总表

| 切片 | n | v7 基线 | v8 (DM-ELSM) | Δ |
|---|---|---|---|---|
| eventqa_65536 | 500 | 0.7240 | 0.7100 | −0.0140 |
| eventqa_131072 | 500 | 0.5620 | 0.5120 | −0.0500 |
| eventqa_full | 500 | 0.3500 | 0.3820 | +0.0320 |
| longmemeval_s* | 300 | 0.2767 | 0.3167 | +0.0400 |

小结：上下文越长（eventqa_full ~40-75 万 token）v8 优势越明显（+0.03~+0.04）；短上下文（65k/131k）v8 低于 v7 基线（−0.01~−0.05）。

## longmemeval_s* 分题型（v8 SF）

| 题型 | n | v8 SF |
|---|---|---|
| multi-session | 75 | 0.1333 |
| knowledge-update | 45 | 0.5333 |
| temporal-reasoning | 75 | 0.2000 |
| single-session-assistant | 30 | 0.4333 |
| single-session-preference | 30 | 0.0000 |
| single-session-user | 45 | 0.7333 |

## 运行备注

- GPU1 全程被外部进程占用（28GB），四切片实际用 GPU0+GPU2 并行跑完。
- 长上下文建池耗时：eventqa_full 各 ctx 627–8894s（40.7 万–74.6 万 token，episodic 1.2 万–2.2 万 / state 1.2k–4.9k，dup=0）。
- 跑长切片时修复 2 处 OOM（已同步本地 `delta-mem-exp4/`）：
  1. `v8_slotpool.encode_words`：逐切片 `.cpu()` 回传，降 GPU 峰值；
  2. `run_ar_v7.py`：`model.generate` 前 `torch.cuda.empty_cache()`，OOM 时渐进截短重试（保尾部 16k/8k/4k token）。
- CR（FactConsolidation）路径未改动，向后兼容。
