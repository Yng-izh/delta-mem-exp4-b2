#!/bin/bash
# 实验 C（零训练）：中间层原生 KV 读出能不能读回事实 —— 架构决策关口。
#
# 只比较六条路径，不训练任何东西：
#   none              不给记忆（冻结模型参数知识下界）
#   oracle_text       明文证据进 system（任务可解性上界）
#   upper_kv          证据在 18~35 层的**真实** K/V      + upper_kv_wrong（换成别的题）
#   h18_kv            同一份 H18 过各层自己的 W_k/W_v      + h18_kv_wrong
#
# 判据（写在 EXPERIMENTS.md「实验 C」）：
#   upper_kv 不高于 none            -> 高层注入接口不成立，换位置/换保存形式
#   upper_kv 高 & h18_kv ≈ upper_kv -> 单层可读，开实验二（残差 Adapter + N->M 压缩）
#   upper_kv 高 & h18_kv ≈ none     -> 接口有效但单层不够，先训 Adapter 或改存多层
#
# ---- 用法 ----
#   PY=/data/.../envs/knowlearn_data/bin/python GPU=1 LIMIT=200 bash runs/exp_c_kv_readout.sh
#   LIMIT=5 SHOW=3 bash runs/exp_c_kv_readout.sh          # 先跑 5 题看预测长什么样
#   DRY=1 bash runs/exp_c_kv_readout.sh                    # 只检查数据，不加载模型
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
# flux 环境中途消失过，这里不猜路径：必须显式给一个能 import torch+transformers 的解释器
PY=${PY:?必须指定 PY（如 \$B/../miniconda3/envs/knowlearn_data/bin/python，需 torch 2.5 + transformers 5.x）}
TAG=${TAG:-exp_c_kv}
LIMIT=${LIMIT:-200}
GPU=${GPU:-1}
ARMS=${ARMS:-none,oracle_text,upper_kv,upper_kv_wrong,h18_kv,h18_kv_wrong}
START_LAYER=${START_LAYER:-18}
MAX_MEM=${MAX_MEM:-320}
OUT=${OUT:-$B/results/$TAG}
mkdir -p "$OUT"
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU
cd "$R" || exit 1

EXTRA=""
if [ "${SHOW:-0}" != "0" ]; then EXTRA="--show-items $SHOW"; fi
if [ "${DRY:-0}" = "1" ]; then EXTRA="$EXTRA --dry-run"; fi

echo "[exp_c $(date '+%F %T')] limit=$LIMIT start=$START_LAYER arms=$ARMS -> $OUT"
$PY eval_midlayer_kv.py \
  --data $B/synth/hotpot_val500.jsonl --limit "$LIMIT" --arms "$ARMS" \
  --start-layer "$START_LAYER" --max-mem-tokens "$MAX_MEM" --max-new-tokens 16 \
  --base $B/models/Qwen3-4B-Instruct-2507 \
  --gpu 0 --progress-every 10 --partial-every 25 $EXTRA \
  --dump-jsonl "$OUT/rows.jsonl" \
  --save-json "$OUT/midlayer_kv.json" \
  > "$OUT/midlayer_kv.log" 2>&1
rc=$?
echo "[exp_c $(date '+%F %T')] 结束 rc=$rc -> $OUT/midlayer_kv.json"
tail -45 "$OUT/midlayer_kv.log"
