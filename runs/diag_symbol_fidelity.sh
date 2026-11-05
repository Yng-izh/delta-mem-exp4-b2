#!/bin/bash
# P1-3：符号保真检查 —— 只看 oracle_latent 答错的题，打印压缩器选中的词、
# 金标句、答案实体是否命中、Bridge 的 α / 残差比 / 输出范数 / 末层权重范数。
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=${PY:-/data/zhangjiayang/miniconda3/envs/flux/bin/python}
CKPT=${CKPT:?必须显式指定 CKPT（没有默认权重了）}
OUT=${OUT:-$B/results/diag_symbol_fidelity.json}
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$R" || exit 1

echo "[diag $(date '+%F %T')] 符号保真开始"
"$PY" tools/diag_symbol_fidelity.py \
  --ckpt "$CKPT" --data $B/synth/hotpot_val500.jsonl \
  --limit 50 --scan 200 --max-new-tokens 16 \
  --base $B/models/Qwen3-4B-Instruct-2507 \
  --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --save-json "$OUT"
echo "[diag $(date '+%F %T')] 结束 rc=$? -> $OUT"
