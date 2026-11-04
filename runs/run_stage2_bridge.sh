#!/bin/bash
# 阶段二：固定阶段一训好的 Router，只训 Compressor + Bridge。
#   前 50% 步数：金标读出（QA CE + KL 蒸馏 oracle 明文 + 重建）
#   后 50% 步数：按 --readout-mode 混入 router 自选分布（与推理同一分布）+ 抗噪
#
# 与已删除的旧权重（2026-11-04 之前那一版）的差别，逐条对应已定位的问题：
#   1. --readout-mode mixed --router-readout-p 0.5
#      旧训练全程只有"金标 + 2 个难负例"，从未见过推理期"router 自选 8 条、可能缺证据"
#      的输入分布；训练 loss 正常而生成 EM 不动，两个分布不是同一个任务。
#   2. --teacher-scope gold
#      旧 teacher 吃整份 memory（20+ 条事实含全部干扰），它自己也可能答错，
#      KL 于是在蒸馏一个噪声分布。开跑前的 --teacher-probe-n 会先量两种上下文的 acc。
#   3. --recon-lex 1
#      旧重建走 bridge(z_sem) 而 QA 走 bridge(z_sem, z_lex)：同一组参数被两个互相冲突的
#      目标拉扯，recon_weight=0.5 持续把语义分支往"能独立还原原文"的方向拽。
#   4. --alpha-max 0.5 --lex-gate-init 0.05
#      残差 α 在旧权重上涨到 0.264、bridge 末层权重 13.29、out_norm 1.05->1.36，读出反而退化。
#      α 现在有硬上限；初值必须非零（末层零初始化下 α=0 会让语义分支与门控同时失去梯度）。
#   5. --diag-every 50
#      每 50 步打印 α / 残差比 / 输出范数 / 末层权重范数，并记录裁剪前总梯度与 clip_coef；
#      旧权重的初始 bridge 梯度是 1.85e10，当时日志里看不出来。
#   6. --cov-weight（默认 0 = 关）
#      L_cov：让词级选择器学会优先保存实体和数字。开关只在 Compressor 真的在训练时
#      有意义（--freeze-compressor 下自动不生效），训练日志里会出现
#      loss_cov / cov_pcov / cov_tgt 三个数。跑完会打印 [cov-check]，若为 0 次直接报错。
#
# A/B 用法（同一份脚本、同一套超参，只差 L_cov）：
#   基线： COV=0.0 OUT=$B/ckpt/hotpot_router_10k_v6.pt      bash runs/run_stage2_bridge.sh
#   实验： COV=0.1 OUT=$B/ckpt/hotpot_router_10k_v6_cov.pt  bash runs/run_stage2_bridge.sh
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=/data/zhangjiayang/miniconda3/envs/flux/bin/python
OUT=${OUT:-$B/ckpt/hotpot_router_10k_v6.pt}
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd $R || exit 1

$PY train_latent_memory.py --router \
  --base $B/models/Qwen3-4B-Instruct-2507 \
  --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --data $B/synth/hotpot_train_10k.jsonl --val-data $B/synth/hotpot_val500.jsonl \
  --n-train 10000 --n-val 120 --epochs 2 --accum 4 --pack 4 \
  --eval-every 250 --eval-limit 40 --log-every 10 --enc-batch 24 \
  --recon-weight 0.5 --recon-units 2 --recon-lex 1 --addr-weight 1.0 --kl-weight 1.0 \
  --gold-readout-ratio 0.5 --readout-mode mixed --router-readout-p 0.5 --hard-neg-k 2 \
  --teacher-scope gold --teacher-probe-n 16 \
  --topk-per-hop 4 \
  --router-kind hybrid --addr-gamma 0.5 --addr-maxsim 1 \
  --init-router $B/ckpt/router_stage1_8slot.pt --freeze-router \
  --ent-weight 5.0 --stop-weight 0.2 --recon-corrupt 0.5 \
  --cov-weight ${COV:-0.0} --cov-targets ${COV_TARGETS:-0} \
  --alpha-max 0.5 --lex-gate-init 0.05 --diag-every 50 \
  --inject-mode user --fence-style bracket \
  --m 4 --layer 18 --inner-dim 1024 --heads 8 --bottleneck 512 \
  --max-unit-tokens 32 --max-slots 0 --lr 2e-4 \
  --out "$OUT" --save-every 250
# 断点续训：同一条命令加 --resume "$OUT"
# ckpt 里存了权重 + AdamW 动量 + step/epoch 位置 + RNG，接着训不会换轨迹
