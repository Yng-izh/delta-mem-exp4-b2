#!/usr/bin/env bash
# MAB-CR 的明文/无记忆基线（不需要 checkpoint）。与 HotpotQA 完全同一协议。
set -u
cd /share/exp4group
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
echo "gpu=$G"
export CUDA_VISIBLE_DEVICES=$G HF_HOME=/share/hf_home HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=/share/exp4group/shim19:/share/exp4group:/share/delta-Mem
PY=/share/conda/envs/latent2/bin/python
D=synth/mab_cr_val.jsonl
mkdir -p results
for M in none oracle; do
  $PY eval_latent_table.py --method "$M" --data "$D" --limit 100 \
      --label "cr-$M" --save-json "results/mab_cr_${M}.json"
done
for K in 4 32; do
  $PY eval_latent_table.py --method text --data "$D" --limit 100 --text-topk "$K" \
      --label "cr-text-top${K}" --save-json "results/mab_cr_text_top${K}.json"
done
echo "[mab-baselines] done"
