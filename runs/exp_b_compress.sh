#!/bin/bash
# 实验 B：验证潜层压缩 —— 固定成功的 Reader，把完整记忆逐步压成 8 / 4 个潜槽。
#
# 与实验 A 的唯一差别是 `--mem slotsK`：记忆来源从"完整隐藏状态"换成"压缩器输出的
# K 个潜槽"，**Reader 结构完全不变**（从实验 A 的权重初始化）。
# 这样"压缩比"只改变记忆长度，读取通路保持不变 —— 这正是"固定 Reader"的含义。
#
# 判据：
#   同读取预算下超过 Query-Agnostic 原生 Token 基线（raw_word@K），且关系绑定准确率不降。
#   原生基线的数字由 `tools/probe_readout_capacity.py` 给（它强制不看问题）。
#
# ---- 用法 ----
#   bash runs/exp_b_compress.sh                 # 8 -> 4 两档都跑
#   KS="8" bash runs/exp_b_compress.sh          # 只跑一档
#   READER=$B/ckpt/reader_full_1000.pt bash runs/exp_b_compress.sh
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=${PY:-/data/zhangjiayang/miniconda3/envs/flux/bin/python}
KS=${KS:-"8 4"}
STEPS=${STEPS:-1000}
LAYER=${LAYER:-18}
BOTTLENECK=${BOTTLENECK:-256}
LR=${LR:-2e-4}
EVAL_EVERY=${EVAL_EVERY:-250}
EVAL_LIMIT=${EVAL_LIMIT:-40}
READER=${READER:-$B/ckpt/reader_full_1000.pt}
mkdir -p "$B/logs/$(date +%F)"
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-1}
cd "$R" || exit 1

if [ ! -f "$READER" ]; then
  echo "[expB] 找不到实验 A 的 Reader：$READER" >&2
  echo "[expB] 先跑 bash runs/exp_a_reader.sh，或者用 READER=... 指定" >&2
  exit 1
fi

for K in $KS; do
  OUT=${OUT_PREFIX:-$B/ckpt/reader_slots${K}_${STEPS}}.pt
  LOG=$B/logs/$(date +%F)/expB_slots${K}_${STEPS}.log
  echo "[expB $(date '+%F %T')] 压缩到 $K 个潜槽 -> $OUT"
  # 第一步：**固定 Reader**，只训压缩器（这是"验证压缩"最干净的做法：
  # 读取通路由实验 A 固定住，唯一变量就是"记忆被压成几个槽"）。
  "$PY" train_reader.py \
    --mem slots$K \
    --data $B/synth/hotpot_train_10k.jsonl --val-data $B/synth/hotpot_val500.jsonl \
    --n-train 10000 --n-val $EVAL_LIMIT \
    --layer $LAYER --bottleneck $BOTTLENECK --lr $LR \
    --steps $STEPS --accum 4 --grad-clip 1.0 \
    --eval-every $EVAL_EVERY --eval-limit $EVAL_LIMIT --log-every 10 \
    --init "$READER" --freeze-reader \
    --base $B/models/Qwen3-4B-Instruct-2507 \
    --adapter $B/models/delta-mem_qwen3_4b_instruct \
    --inject-mode user --fence-style bracket \
    --out "$OUT" \
    > "$LOG" 2>&1
  echo "[expB $(date '+%F %T')] slots$K rc=$?"
  grep -E "^\[eval\]" "$LOG" | tail -3
done
echo "[expB $(date '+%F %T')] 全部完成"
