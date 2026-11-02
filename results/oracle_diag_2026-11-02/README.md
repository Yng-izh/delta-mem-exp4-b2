# 2026-11-02 潜层读出诊断运行记录

**本文件只记录运行条件、命令与原始数字，不含结论或解释。** 数字直接由各 `*.json` 的
`summary` 字段生成，未手工改写。

## 1. 运行环境

| 项 | 值 |
|---|---|
| 机器 | `zhangjiayang@100.118.6.87 -p 50022`，根目录 `/data/zhangjiayang/yyz3` |
| GPU | NVIDIA RTX 6000 Ada Generation ×2；本轮 `CUDA_VISIBLE_DEVICES=1`（GPU 0 被其他任务占用 37951/40960 MiB） |
| Python | 3.11.15（`miniconda3/envs/flux/bin/python`） |
| torch | 2.11.0+cu130 |
| transformers | 4.51.3 |
| 主干 | `models/Qwen3-4B-Instruct-2507` |
| adapter | `models/delta-mem_qwen3_4b_instruct`（加载校验 324/324 张量对应） |
| 数据 | `synth/hotpot_val500.jsonl`，`--limit 500` |
| 潜层 ckpt | `ckpt/hotpot_router_10k_v3.pt` |
| ckpt config | `n_slots=4, inner_dim=512, max_unit_tokens=32, layer=18` |
| 代码版本 | git `b56873e` |

通用环境变量：`DELTA_MEM_SCAN_IMPL=torch`、`PYTHONPATH=$R:$B/delta-Mem`、
`HF_HOME=$B/hf_home`、`HF_HUB_OFFLINE=1`、`TOKENIZERS_PARALLELISM=false`、`CUDA_VISIBLE_DEVICES=1`。

## 2. 命令（逐条原样）

```bash
# 1) 金标槽位 -> bridge（读出上界）
python eval_latent_table.py --mode oracle_latent \
  --ckpt $B/ckpt/hotpot_router_10k_v3.pt --data $B/synth/hotpot_val500.jsonl \
  --limit 500 --max-new-tokens 16 --gpu 0 \
  --base $B/models/Qwen3-4B-Instruct-2507 --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --label v3-oracle_latent --save-json $B/results/v3_oracle_latent_500.json

# 2) 金标证据原文的真实 token embedding，预算按潜层块数自动算
python eval_latent_table.py --mode oracle_embed \
  --data $B/synth/hotpot_val500.jsonl --limit 500 --max-new-tokens 16 --gpu 0 \
  --base $B/models/Qwen3-4B-Instruct-2507 --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --label v3-oracle_embed --save-json $B/results/v3_oracle_embed_500.json

# 3) 同上，预算按块数对齐后重跑
python eval_latent_table.py --mode oracle_embed \
  --data $B/synth/hotpot_val500.jsonl --limit 500 --max-new-tokens 16 --gpu 0 \
  --base $B/models/Qwen3-4B-Instruct-2507 --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --label v3-oracle_embed_aligned --save-json $B/results/v3_oracle_embed_aligned_500.json

# 4) 同上，显式 --embed-budget 34（与 oracle_latent 的 mem_tokens 对齐）
python eval_latent_table.py --mode oracle_embed --embed-budget 34 \
  --data $B/synth/hotpot_val500.jsonl --limit 500 --max-new-tokens 16 --gpu 0 \
  --base $B/models/Qwen3-4B-Instruct-2507 --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --label v3-oracle_embed_b34 --save-json $B/results/v3_oracle_embed_b34_500.json
```

## 3. 指标（各 json 的 `summary` 字段原样）

| label | n | acc | EM | F1 | read_acc | full_recall | mem_tokens | mem_plain_tokens | prompt_tokens | enc_ms_mean | gen_ms_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `none` | 500 | 0.1120 | 0.1120 | 0.1948 | — | — | 0.00 | — | 21.57 | — | — |
| `text` | 500 | 0.3120 | 0.3200 | 0.4592 | 0.3600 | 0.3600 | 151.20 | — | 185.77 | — | — |
| `text-top4` | 500 | 0.3120 | 0.3200 | 0.4592 | 0.3600 | — | 151.20 | — | 185.77 | — | — |
| `oracle` | 500 | 0.5460 | 0.5540 | 0.7242 | 1.0000 | 1.0000 | 91.30 | — | 125.87 | — | — |
| `bridge-v3` | 500 | 0.1600 | 0.1660 | 0.2489 | 0.0000 | 0.0000 | 95.54 | — | 95.54 | — | — |
| `router-v3` | 500 | 0.1260 | 0.1280 | 0.2197 | 0.0560 | 0.0560 | 22.00 | — | 22.00 | — | — |
| `v3-oracle_latent` | 500 | 0.1700 | 0.1720 | 0.2659 | 1.0000 | 1.0000 | 34.52 | 0.00 | 82.10 | 2060.4 | 474.1 |
| `v3-oracle_embed` | 500 | 0.1600 | 0.1620 | 0.2645 | 1.0000 | 1.0000 | 9.72 | 0.00 | 57.29 | 0.0 | 758.6 |
| `v3-oracle_embed_aligned` | 500 | 0.1560 | 0.1620 | 0.2872 | 1.0000 | 1.0000 | 10.50 | 0.00 | 58.08 | 0.0 | 763.7 |
| `v3-oracle_embed_b34` | 500 | 0.3180 | 0.3280 | 0.4695 | 1.0000 | 1.0000 | 33.09 | 0.00 | 80.66 | 0.0 | 636.8 |

`read_acc` / `full_recall` 在本表单位与 json 一致（0~1 比例）。`—` 表示该 json 无此字段。

## 4. 文件清单

| 文件 | 内容 |
|---|---|
| `v3_oracle_latent_500.json` | 500 条逐样本记录 + summary |
| `v3_oracle_embed_500.json` | 同上（自动预算） |
| `v3_oracle_embed_aligned_500.json` | 同上（块数对齐预算） |
| `v3_oracle_embed_b34_500.json` | 同上（`--embed-budget 34`） |
| `../../logs/2026-11-02/oracle_latent_v3.log` | 第 1 条命令的完整 stdout/stderr |
| `../../logs/2026-11-02/oracle_embed_v3.log` | 第 2 条 |
| `../../logs/2026-11-02/oracle_embed_aligned_v3.log` | 第 3 条 |
| `../../logs/2026-11-02/oracle_embed_b34_v3.log` | 第 4 条 |

对照基线 `exp1_*.json` 未复制到本目录，直接引用仓库根 `results/` 下的同名文件
（服务器上的副本与本地副本 md5 一致）。
