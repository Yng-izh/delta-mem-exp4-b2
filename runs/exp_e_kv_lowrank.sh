#!/bin/bash
# 实验 E：跨层 KV 的可压缩性验证（零训练，纯线性压缩）
#
#   1) 先补实验 D 缺的 anchor:L6 及其 Wrong 对照（未压缩基线，必须先在 200 题上确认）
#   2) 在训练集上拟合一套共享线性基（最高 2048 维，512/1024 取前 k 列）
#   3) 在固定 200 题上比未压缩 / PCA-512/1024/2048 的 QA，各带 Wrong 反事实
#
# 门槛顺序（不能反）：
#   native_rt 与 anchor6 不一致 -> 先修解码器，不能把掉点归因给 PCA
#   最佳档保留未压缩基线 <60% -> 全层共享低秩不成立，先改跨层分组
#
# ---- 用法 ----
#   PY=<envs/knowlearn_data/bin/python> GPU=1 STAGE=all bash runs/exp_e_kv_lowrank.sh
#   STAGE=anchor bash runs/exp_e_kv_lowrank.sh          # 只补 anchor:L6（实验 D 的臂）
#   STAGE=fit bash runs/exp_e_kv_lowrank.sh             # 只拟合基底
#   STAGE=eval DIMS=1024,2048 bash runs/exp_e_kv_lowrank.sh
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=${PY:?必须指定 PY（如 \$B/../miniconda3/envs/knowlearn_data/bin/python）}
TAG=${TAG:-exp_e_lowrank}
STAGE=${STAGE:-all}
LIMIT=${LIMIT:-200}
GPU=${GPU:-1}
LAYERS=${LAYERS:-6-35}
DIM=${DIM:-2048}
DIMS=${DIMS:-512,1024,2048}
FIT_TOKENS=${FIT_TOKENS:-30000}
FIT_LIMIT=${FIT_LIMIT:-0}
OUT=${OUT:-$B/results/$TAG}
mkdir -p "$OUT"
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU
cd "$R" || exit 1
BASE=$B/models/Qwen3-4B-Instruct-2507
VAL=$B/synth/hotpot_val500.jsonl
TRAIN=$B/synth/hotpot_train_10k.jsonl
BASIS=${BASIS:-$OUT/kv_lowrank_${DIM}.pt}

run_anchor() {
  echo "[E-acchor $(date '+%F %T')] 补实验 D 的 anchor:L6（未压缩基线）"
  $PY eval_midlayer_kv.py --data "$VAL" --limit "$LIMIT" --mode main \
    --arms "text_full,kv:anchor:6,kv:anchor:6:wrong,kv:anchor:0" \
    --base "$BASE" --gpu 0 --max-new-tokens 16 --progress-every 10 \
    --dump-jsonl "$OUT/anchor_l6_rows.jsonl" \
    --save-json "$OUT/anchor_l6.json" > "$OUT/anchor_l6.log" 2>&1
  local rc=$?
  grep -E "^\| " "$OUT/anchor_l6.log" | head -12
  return $rc
}

run_fit() {
  echo "[E-fit $(date '+%F %T')] layers=$LAYERS dim=$DIM tokens<=$FIT_TOKENS -> $BASIS"
  $PY eval_kv_lowrank.py --stage fit --fit-data "$TRAIN" --basis "$BASIS" \
    --dim "$DIM" --layers "$LAYERS" --fit-tokens "$FIT_TOKENS" --fit-limit "$FIT_LIMIT" \
    --base "$BASE" --gpu 0 > "$OUT/fit.log" 2>&1
  local rc=$?
  tail -6 "$OUT/fit.log"
  return $rc
}

run_eval() {
  echo "[E-eval $(date '+%F %T')] layers=$LAYERS dims=$DIMS basis=$BASIS"
  $PY eval_kv_lowrank.py --stage eval --data "$VAL" --basis "$BASIS" \
    --dims "$DIMS" --layers "$LAYERS" --limit "$LIMIT" \
    --base "$BASE" --gpu 0 --progress-every 10 --show-items 2 \
    --dump-jsonl "$OUT/rows.jsonl" --save-json "$OUT/summary.json" \
    > "$OUT/eval.log" 2>&1
  local rc=$?
  grep -E "^\[identity\]|^\| |^\[判据" "$OUT/eval.log" | tail -40
  return $rc
}

case "$STAGE" in
  anchor) run_anchor ;;
  fit)    run_fit ;;
  eval)   run_eval ;;
  all)
    run_anchor || { echo "[E] anchor:L6 失败，停止"; exit 1; }
    run_fit    || { echo "[E] 拟合失败，停止"; exit 1; }
    run_eval ;;
  *) echo "未知 STAGE=$STAGE（anchor|fit|eval|all）"; exit 2 ;;
esac
