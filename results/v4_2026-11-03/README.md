# 2026-11-03 v4 训练与评测记录

**只记录运行条件、命令与原始数字，不含结论。** 数字由各 `*.json` 的 `summary` 字段生成。

## 1. 训练

| 项 | 值 |
|---|---|
| 脚本 | `runs/run_train_v4.sh`（`GPU=1`） |
| 参数 | `stage=a m=4 layer=18 inner=1024 heads=8 bottleneck=512 max_unit_tokens=32 lr=2e-4 accum=4 pack=4 kl=1.0 recon=0.5 epochs=3` |
| 数据 | `synth/hotpot_train_10k.jsonl`（train 9997 / val 120，hops 全为 2） |
| 起止 | 2026-11-02 15:31 → 2026-11-03 06:10（约 14 小时 39 分），共 7500 步 |
| 收尾标记 | `ROUTER-10K-V4 DONE` |
| 产物 | `ckpt/hotpot_router_10k_v4.pt`（106828096 bytes，06:09） |
| KL | 全程 `loss_kl=n/a`（stage=a 下不生效，启动时打印警告） |
| 训练期评测 | n=40 子集，step 250→7250 共 29 次，除 step 0（0.05）外全部 0.075；收尾评测 0.100（4/40） |

## 2. 评测（val500，n=500）

命令由 `runs/watch_v4_eval.sh` 在训练结束后自动执行（GPU 1，06:10 起）：

```bash
python eval_latent_table.py --mode router        --ckpt $B/ckpt/hotpot_router_10k_v4.pt \
  --data $B/synth/hotpot_val500.jsonl --limit 500 --max-new-tokens 16 --gpu 0 \
  --base $B/models/Qwen3-4B-Instruct-2507 --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --label v4-router --save-json $B/results/v4_router_500.json
python eval_latent_table.py --mode oracle_latent --ckpt $B/ckpt/hotpot_router_10k_v4.pt \
  --data $B/synth/hotpot_val500.jsonl --limit 500 --max-new-tokens 16 --gpu 0 \
  --base $B/models/Qwen3-4B-Instruct-2507 --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --label v4-oracle_latent --save-json $B/results/v4_oracle_latent_500.json
```

## 3. 指标（`summary` 原样）

| label | n | acc | EM | F1 | read_acc | full_recall | mem_tokens | prompt_tokens | enc_ms | gen_ms | total_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `v3-router` | 500 | 0.1260 | 0.1280 | 0.2197 | 0.0560 | 0.0560 | 22.00 | 22.00 | — | — | 2716.3 |
| `v4-router` | 500 | 0.0980 | 0.0980 | 0.1794 | 0.1560 | 0.1560 | 58.17 | 105.74 | 2004.3 | 906.7 | 2911.0 |
| `v3-oracle_latent` | 500 | 0.1700 | 0.1720 | 0.2659 | 1.0000 | 1.0000 | 34.52 | 82.10 | 2060.4 | 474.1 | 2534.5 |
| `v4-oracle_latent` | 500 | 0.1000 | 0.1000 | 0.1783 | 1.0000 | 1.0000 | 34.52 | 82.10 | 2033.0 | 909.9 | 2942.9 |
| `none` | 500 | 0.1120 | 0.1120 | 0.1948 | — | — | 0.00 | 21.57 | — | — | 675.8 |
| `text` | 500 | 0.3120 | 0.3200 | 0.4592 | 0.3600 | 0.3600 | 151.20 | 185.77 | — | — | 681.1 |
| `oracle` | 500 | 0.5460 | 0.5540 | 0.7242 | 1.0000 | 1.0000 | 91.30 | 125.87 | — | — | 514.0 |

`read_acc` / `full_recall` 为 0~1 比例；`—` 表示该 json 无此字段。

## 4. 文件清单

| 文件 | 内容 |
|---|---|
| `v4_router_500.json` | v4 端到端 500 题逐样本记录 + summary |
| `v4_oracle_latent_500.json` | v4 金标槽位读出 500 题逐样本记录 + summary |
| `../../logs/2026-11-03/train_v4.log` | 训练完整日志 |
| `../../logs/2026-11-03/watch_v4_eval.status` | 看门狗时间戳流水 |
| `../../logs/2026-11-03/eval_router_v4.log` | router 模式评测日志 |
| `../../logs/2026-11-03/eval_oracle_latent_v4.log` | oracle_latent 模式评测日志 |

v3 对照数据在 `results/exp1_*.json` 与 `results/oracle_diag_2026-11-02/`。
