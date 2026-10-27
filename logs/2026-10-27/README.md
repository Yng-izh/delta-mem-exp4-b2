# logs/2026-10-27 — 第一轮公开基准实验日志

| 文件 | 内容 | 关键结论 |
|---|---|---|
| `baselines.log` | HotpotQA dev（dis4）none/oracle/text 基线 | none 12.0 / top4 36.0 / top16 47.0 / oracle 58.0 |
| `mab_baselines.log` | MAB-CR val 基线 | none 4.0 / top4 17.0 / top32 14.0 / oracle 32.0 |
| `hotpot_r1.log` | 第一次 Router 训练（课程被 keep_all 关掉的旧代码） | 200 步后 dev 5.0% < 无记忆 12% → 暴露课程失效，触发 must_keep 修复 |
| `hotpot_r2.log` | 第二次 Router 训练（课程修复后，仍在跑） | acc 仍停在 ~5%（chance 4.3%）；ckpt@300 用于 bridge/router 消融行 |
| `mab_cr_fst.log` | FST-Mem 在 MAB-CR 训练（150 步完成） | group 0.650>0.550，ver 0.525<恒取最新 0.885，order_stable 1.0 |
| `mab_cr_router.log` | 普通 Router 在 MAB-CR 训练（150 步完成） | FST 的对照组；eval 读对槽位 5% vs FST 65% |

## 对应结果 JSON（results/ 目录）

- HotpotQA：`hotpot_{none,oracle,text_top4,text_top16}.json`、`hp_bridge_dis{0,4}.json`、`hp_bridge_dis4_shuffled.json`、`hp_router_dis4.json`
- MAB-CR：`mab_cr_{none,oracle,text_top4,text_top32,router,fst}.json`

汇总表：[`results/RESULTS.md`](../results/RESULTS.md)

## 复现命令

```bash
# 基线（无需 checkpoint）
bash runs/run_baselines.sh        # HotpotQA dis4
bash runs/run_mab_baselines.sh     # MAB-CR val

# 训练（Hyperparam 与 [save] 行一致，GPU 为共享 V100-32G）
python train_latent_memory.py --router --data synth/hotpot_train.jsonl \
  --val-data synth/hotpot_sweep/hotpot_validation_dis4_natural.jsonl \
  --n-train 2500 --n-val 120 --epochs 1 --accum 4 --eval-every 100 --eval-limit 40 \
  --recon-weight 0.3 --recon-units 2 --addr-weight 1.0 --kl-weight 0 \
  --m 4 --layer 18 --inner-dim 512 --heads 8 --bottleneck 512 \
  --max-unit-tokens 64 --max-slots 8 --lr 2e-4 --out ckpt/hotpot_r2.pt --save-every 150

python train_latent_memory.py --fst --hist-max 4 \
  --data synth/mab_cr_train.jsonl --val-data synth/mab_cr_val.jsonl \
  --n-train 300 --n-val 60 --epochs 1 --accum 2 --eval-every 100 --eval-limit 40 \
  --m 4 --layer 18 --inner-dim 512 --heads 8 --bottleneck 512 --max-unit-tokens 64 \
  --d-key 256 --addr-weight 1.0 --ver-weight 1.0 --lr 1e-4 \
  --out ckpt/mab_cr_fst.pt --save-every 100

# 潜层评测（与明文同一条生成路径）
python eval_latent_table.py --method {bridge,router,fst} --ckpt <ckpt.pt> \
  --data <val.jsonl> --limit 40 --label <label> --save-json results/<out>.json
```

## 环境备注

- 服务器无外网，数据经 hf-mirror 下载；代码经 tar 管道同步（scp 在该机不稳定）
- GPU 为 10 卡共享，每步日志带 `peakmem=` 区分自身增长与他人争用
- 日志已清洗：删去 transformers 权重加载进度条，只留结构化行（`[step]/[eval]/[save]` 与表格）
