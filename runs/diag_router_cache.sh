#!/bin/bash
# P0-1：Router 缓存一致性对拍（cache=None vs cache={}），200 题逐题比较
# 期望：kfeat 最大差 < tol、picks 一致率 100%、两条路径的完整证据召回相同。
# 不满足则非零退出 —— 不要"差一点也算过"。
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=/data/zhangjiayang/miniconda3/envs/flux/bin/python
CKPT=$B/ckpt/hotpot_router_10k_v5.pt
OUT=${OUT:-$B/results/diag_router_cache.json}
# ENC_BATCH=1 时两条路径都逐条编码：用来判定 picks 差异是否纯粹来自分批数值抖动
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$R" || exit 1

echo "[diag $(date '+%F %T')] router 缓存一致性开始（enc_batch=${ENC_BATCH:-0}）"
"$PY" tools/diag_router_cache.py \
  --ckpt "$CKPT" --data $B/synth/hotpot_val500.jsonl --limit 200 \
  --topk-per-hop 4 --tol 5e-2 --enc-batch "${ENC_BATCH:-0}" \
  --base $B/models/Qwen3-4B-Instruct-2507 \
  --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --save-json "$OUT"
rc=$?
echo "[diag $(date '+%F %T')] 结束 rc=$rc -> $OUT"
exit $rc
