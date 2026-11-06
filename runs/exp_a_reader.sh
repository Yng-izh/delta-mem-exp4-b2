#!/bin/bash
# 实验 A：验证读取接口 —— 不压缩，完整隐藏状态当记忆，只训轻量 Query-conditioned Reader。
#
# 判据（缺一不可）：
#   full 明显高于 none     -> Reader 确实把记忆送进去了
#   full 明显高于 shuffle  -> Reader 用的是**记忆内容**，不是"多一段有信息量的噪声"
#   full 接近 oracle_text  -> 这条读取通路接近明文上界，值得往下做压缩
#
# 三个对照都在训练脚本的评测里自动跑（none / shuffle / full / oracle_text）。
#
# ---- 用法 ----
#   bash runs/exp_a_reader.sh
#   STEPS=2000 bash runs/exp_a_reader.sh
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=${PY:-/data/zhangjiayang/miniconda3/envs/flux/bin/python}
STEPS=${STEPS:-1000}
LAYER=${LAYER:-18}
BOTTLENECK=${BOTTLENECK:-256}
LR=${LR:-2e-4}
EVAL_EVERY=${EVAL_EVERY:-250}
EVAL_LIMIT=${EVAL_LIMIT:-40}
FACT_RECON=${FACT_RECON:-0.5}     # 0 = 基线 A（只训 QA）；0.5 = 改进版 B
RECON_CORRUPT=${RECON_CORRUPT:-0.5}
OUT=${OUT:-$B/ckpt/reader_full_${STEPS}.pt}
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-1}
cd "$R" || exit 1

echo "[expA $(date '+%F %T')] 完整记忆 + Reader -> $OUT"
"$PY" train_reader.py \
  --mem full \
  --data $B/synth/hotpot_train_10k.jsonl --val-data $B/synth/hotpot_val500.jsonl \
  --n-train 10000 --n-val $EVAL_LIMIT \
  --layer $LAYER --bottleneck $BOTTLENECK --lr $LR \
  --steps $STEPS --accum 4 --grad-clip 1.0 \
  --eval-every $EVAL_EVERY --eval-limit $EVAL_LIMIT --log-every 10 \
  --base $B/models/Qwen3-4B-Instruct-2507 \
  --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --inject-mode user --fence-style bracket \
  --fact-recon-weight $FACT_RECON --recon-corrupt $RECON_CORRUPT \
  --out "$OUT" \
  > "$B/logs/$(date +%F)/expA_reader_${STEPS}_recon${FACT_RECON}.log" 2>&1
echo "[expA $(date '+%F %T')] rc=$? -> $OUT"
tail -20 "$B/logs/$(date +%F)/expA_reader_${STEPS}_recon${FACT_RECON}.log"
