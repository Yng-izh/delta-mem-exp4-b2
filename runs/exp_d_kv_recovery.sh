#!/bin/bash
# 实验 D：恢复真实 KV 的明文读取能力（零训练，三阶段，前一个不过不进下一个）
#
#   D0-A  text_full  vs prefix_kv            因果前缀拆两次前向是否等价（原生 KV cache）
#   D0-B  text_full  vs kv:full_prefix:0      自定义注入（E 插在 S 与 T 之间）能否复现明文
#   D1    kv:{full_prefix,bare,anchor}:0      证据脱离原始上下文后还能不能被读
#   D2    kv:<enc>:{0,6,12,18}                冻结模型从哪一层起必须接触证据
#
# 判据（写在 EXPERIMENTS.md「实验 D」）：
#   D0 首步 Top-1 出现系统性分歧 -> 停，先修等价性，别解读 D1/D2
#   D1/D2 目标：最佳 KV 臂保留明文 >= 86%（EM >= 45%）；同时 Correct-Wrong 必须显著
#
# ---- 用法 ----
#   PY=<envs/knowlearn_data/bin/python> GPU=1 bash runs/exp_d_kv_recovery.sh
#   STAGE=dry bash runs/exp_d_kv_recovery.sh        # 只查 S/E/T 切分，不加载模型
#   STAGE=eq   LIMIT_EQ=15 bash runs/exp_d_kv_recovery.sh
#   STAGE=main LIMIT=200 SWEEP_ENC=full_prefix bash runs/exp_d_kv_recovery.sh
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=${PY:?必须指定 PY（如 \$B/../miniconda3/envs/knowlearn_data/bin/python）}
TAG=${TAG:-exp_d_kv}
STAGE=${STAGE:-all}
LIMIT=${LIMIT:-200}
LIMIT_EQ=${LIMIT_EQ:-15}
GPU=${GPU:-1}
SWEEP_ENC=${SWEEP_ENC:-full_prefix}
MAX_MEM=${MAX_MEM:-320}
OUT=${OUT:-$B/results/$TAG}
mkdir -p "$OUT"
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU
cd "$R" || exit 1
BASE=$B/models/Qwen3-4B-Instruct-2507
DATA=$B/synth/hotpot_val500.jsonl
COMMON="--base $BASE --gpu 0 --max-mem-tokens $MAX_MEM --max-new-tokens 16"

run_dry() {
  echo "[D-dry $(date '+%F %T')] 只验证 S/E/T 切分"
  $PY eval_midlayer_kv.py --data $DATA --limit "$LIMIT" --dry-run $COMMON \
    > "$OUT/split_check.log" 2>&1
  local rc=$?
  tail -6 "$OUT/split_check.log"
  echo "[D-dry] rc=$rc（0 = 全部样本都能在 token 边界上严格切开）"
  return $rc
}

run_eq() {
  echo "[D0-eq $(date '+%F %T')] 等价性检查 limit=$LIMIT_EQ"
  $PY eval_midlayer_kv.py --data $DATA --limit "$LIMIT_EQ" --mode equivalence \
    --show-items 2 $COMMON > "$OUT/d0_equiv.log" 2>&1
  local rc=$?
  grep -E "^\| (text_full|prefix_kv|kv:)|^\[eq\]" "$OUT/d0_equiv.log" | tail -12
  if grep -q "EQ_PASS" "$OUT/d0_equiv.log"; then
    echo "[D0-eq] 通过 -> 进入 D0-B/D1/D2 的 200 题主表"
    return 0
  fi
  echo "[D0-eq] 未通过，按实验设计停下（先修等价性，不解读后面的数字）"
  return 1
}

run_main() {
  local arms="none,text_full,prefix_kv,"
  arms+="kv:full_prefix:0,kv:full_prefix:0:wrong,"
  arms+="kv:bare:0,kv:bare:0:wrong,"
  arms+="kv:anchor:0,kv:anchor:0:wrong,"
  arms+="kv:$SWEEP_ENC:6,kv:$SWEEP_ENC:6:wrong,"
  arms+="kv:$SWEEP_ENC:12,kv:$SWEEP_ENC:12:wrong,"
  arms+="kv:$SWEEP_ENC:18,kv:$SWEEP_ENC:18:wrong"
  echo "[D-main $(date '+%F %T')] limit=$LIMIT sweep_enc=$SWEEP_ENC arms=$arms"
  $PY eval_midlayer_kv.py --data $DATA --limit "$LIMIT" --mode main --arms "$arms" \
    --show-items 2 --progress-every 10 --partial-every 25 \
    --dump-jsonl "$OUT/rows.jsonl" --save-json "$OUT/midlayer_kv.json" $COMMON \
    > "$OUT/midlayer_kv.log" 2>&1
  local rc=$?
  echo "[D-main] rc=$rc -> $OUT/midlayer_kv.json"
  tail -30 "$OUT/midlayer_kv.log"
  return $rc
}

case "$STAGE" in
  dry)  run_dry ;;
  eq)   run_eq ;;
  main) run_main ;;
  all)
    run_dry || { echo "[D] 切分不通过，停止"; exit 1; }
    run_eq  || { echo "[D] D0 不通过，停止"; exit 1; }
    run_main ;;
  *) echo "未知 STAGE=$STAGE（dry|eq|main|all）"; exit 2 ;;
esac
