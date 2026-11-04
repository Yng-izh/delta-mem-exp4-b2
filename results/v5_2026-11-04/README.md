# 2026-11-04 v5 训练与评测记录

**只记录运行条件、命令与原始数字，不含结论。** 指标由各 `*.json` 的 `summary` 字段生成。

## 1. 阶段一：只训 Router（8 槽读出）

| 项 | 值 |
|---|---|
| 脚本 | `tools/train_router_stage1.py`（`--epochs 1 --accum 8 --enc-batch 16 --lr 2e-3 --topk-per-hop 4`） |
| 数据 | `synth/hotpot_train_10k.jsonl`（9997 条，全 2 跳）；评测 `synth/hotpot_val500.jsonl` |
| 耗时 | 2026-11-03 16:12 → 16:56，9997 步，约 2.1 条/秒 |
| 产物 | `ckpt/router_stage1_8slot.pt`（100,991,977 bytes） |

召回曲线（n=500，原样摘自上表日志）：

```
    [eval step=2000] hop1=0.934 hop2|hop1=0.5717344753747323 full=0.550 n=500 | ev=2: n=339 hop1=0.9203539823008849 full=0.6637168141592921 | ev>=3: n=161 hop1=0.9627329192546584 full=0.3105590062111801 (262s)
    [eval step=4000] hop1=0.878 hop2|hop1=0.5466970387243736 full=0.494 n=500 | ev=2: n=339 hop1=0.8613569321533924 full=0.6076696165191741 | ev>=3: n=161 hop1=0.9130434782608695 full=0.2546583850931677 (222s)
    [eval step=6000] hop1=0.946 hop2|hop1=0.5560253699788583 full=0.540 n=500 | ev=2: n=339 hop1=0.9321533923303835 full=0.6637168141592921 | ev>=3: n=161 hop1=0.9751552795031055 full=0.2795031055900621 (222s)
    [eval step=8000] hop1=0.946 hop2|hop1=0.5539112050739958 full=0.542 n=500 | ev=2: n=339 hop1=0.9351032448377581 full=0.6519174041297935 | ev>=3: n=161 hop1=0.968944099378882 full=0.3105590062111801 (220s)
    [final] hop1=0.928 hop2|hop1=0.5538793103448276 full=0.520 n=500 | ev=2: n=339 hop1=0.911504424778761 full=0.6312684365781711 | ev>=3: n=161 hop1=0.9627329192546584 full=0.2857142857142857
```

## 2. 阶段二：训 Compressor + Bridge

| 项 | 值 |
|---|---|
| 脚本 | `runs/run_stage2_bridge.sh`（`--init-router router_stage1_8slot.pt --freeze-router`） |
| 关键参数 | `topk_per_hop=4`、`kl_weight=1.0`、`recon_weight=0.5`、`pack=4`、`accum=4`、`gold_readout_ratio=0.5`、`hard_neg_k=2`、`epochs=2` |
| 起止 | 2026-11-03 17:35 → 22:44（约 5 小时 09 分），4990 步 |
| 产物 | `ckpt/hotpot_router_10k_v5.pt`（224,505,677 bytes，含优化器动量） |
| KL 跳过计数 | 0（蒸馏全程生效） |

## 3. 评测（val500，n=500）

命令：`runs/eval_v5.sh`（GPU 0，`--topk-per-hop 4`，00:30:44 → 01:18:26）

| label | n | acc | EM | F1 | read_acc | full_recall | mem_tokens | prompt_tokens | enc_ms | gen_ms | total_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `v5-router` | 500 | 0.1060 | 0.1060 | 0.1839 | 0.0600 | 0.0600 | 188.55 | 236.12 | 2053.7 | 804.8 | 2858.5 |
| `v5-oracle_latent` | 500 | 0.1180 | 0.1180 | 0.2195 | 1.0000 | 1.0000 | 63.68 | 111.26 | 2054.6 | 773.3 | 2827.9 |

## 4. 已知作废项

`v5-router` 这一次运行使用的代码在 `encode_item_slots_full()` 的缓存分支里固定返回
`kfeat=None`，导致评测期 `HybridLatentRouter` 退回潜载荷键（MaxSim 与新信息抑制未生效）。
该缺陷在 2026-11-04 修复（缓存分支同时缓存并返回逐槽特征），**本目录的 router 数字需重跑替换**。
`v5-oracle_latent` 不经过 Router，不受该缺陷影响。

## 5. 文件清单

| 文件 | 内容 |
|---|---|
| `v5_router_500.json` / `v5_oracle_latent_500.json` | 逐样本记录 + summary |
| `../../logs/2026-11-04/eval_v5*.log` | 评测日志 |
| `../../logs/2026-11-04/stage1_router_8slot.log` | 阶段一训练日志 |
| `../../logs/2026-11-04/train_stage2.log` | 阶段二训练日志 |
