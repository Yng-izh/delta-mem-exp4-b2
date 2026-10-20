#!/usr/bin/env bash
# run_eventqa_window_sweep.sh — EventQA 检索方向验证（Back/Ahead 四组，顺序跑）
#
# 每组写入独立文件 + 独立 manifest：不同窗口配置不可能混进同一个结果文件。
# 用法（服务器）：
#   bash experiments/run_eventqa_window_sweep.sh
#   SRC=eventqa_65536 GPU=0 OUTDIR=/share/exp4group/results/sweep bash ...
set -euo pipefail

# latent2 自带的 transformers 4.51.3 没有 models/smollm3，而 /share/delta-Mem 无条件
# import 它 -> 直接 ImportError。用本地 overlay（4.57.1）覆盖，不动共享环境。
OVERLAY="${OVERLAY:-/share/exp4group/pylib_tf57}"
export PYTHONPATH="$OVERLAY:/share/delta-Mem:/share/MemoryAgentBench${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-/share/conda/envs/latent2/bin/python}"
SRC="${SRC:-eventqa_65536}"
GPU="${GPU:-0}"
ROOT="${ROOT:-/share/exp4group}"
OUTDIR="${OUTDIR:-$ROOT/results/sweep}"
mkdir -p "$OUTDIR"

for win in "0 20" "4 16" "10 10" "16 4"; do
  set -- $win
  B="$1"; A="$2"
  tag="b${B}_a${A}"
  out="$OUTDIR/ar_v9_${SRC}_${tag}.jsonl"
  log="$OUTDIR/log_${tag}.txt"
  echo "[sweep] back=$B ahead=$A -> $out"
  AR_MODE=v9 \
  AR_V9_EQA_BACK="$B" AR_V9_EQA_AHEAD="$A" \
  AR_RESUME=1 AR_RESUME_MANIFEST=1 AR_OUT="$out" \
    "$PY" "$ROOT/run_ar_v7.py" "$SRC" "$GPU" 2>&1 | tee "$log"
done

echo "[sweep] done. 汇总："
"$PY" "$ROOT/tools/summarize_sweep.py" "$OUTDIR"/ar_v9_${SRC}_b*.jsonl
