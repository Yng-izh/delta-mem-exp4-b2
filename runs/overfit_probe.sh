#!/bin/bash
# P1-4：64 题小样本过拟合 + 三个目标函数对照（qa_kl / recon_old / recon_qa）
#   qa_kl      = QA + KL，不加重建（干净对照）
#   recon_old  = 重建走 bridge(z_sem)（旧写法，与 QA 目标不一致）
#   recon_qa   = 重建走 bridge(z_sem, z_lex)（与 QA 同一条路径）
# 三条曲线在同一步数上的差值 = "重建目标有没有在拖主任务"的直接证据。
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=/data/zhangjiayang/miniconda3/envs/flux/bin/python
CKPT=$B/ckpt/hotpot_router_10k_v5.pt
OUT=${OUT:-$B/results/overfit_probe.json}
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$R" || exit 1

echo "[diag $(date '+%F %T')] 过拟合探针开始"
"$PY" tools/overfit_probe.py \
  --ckpt "$CKPT" --data $B/synth/hotpot_train_10k.jsonl \
  --n 64 --steps 200 --accum 4 --lr 1e-4 --eval-every 25 \
  --kl-weight 1.0 --recon-weight 0.5 --recon-corrupt 0.5 \
  --arms qa_kl,recon_old,recon_qa \
  --base $B/models/Qwen3-4B-Instruct-2507 \
  --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --save-json "$OUT"
echo "[diag $(date '+%F %T')] 结束 rc=$? -> $OUT"
