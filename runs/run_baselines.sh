#!/usr/bin/env bash
# 明文/无记忆基线的统一评测（不需要 checkpoint）。产出结果表的前三行。
# 顺序：none（纯参数知识）→ oracle（只给支持事实，上界）→ text-top4 / text-top16（真实 RAG）
set -u
cd /share/exp4group
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
echo "gpu=$G"
export CUDA_VISIBLE_DEVICES=$G HF_HOME=/share/hf_home HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=/share/exp4group/shim19:/share/exp4group:/share/delta-Mem
PY=/share/conda/envs/latent2/bin/python
D=synth/hotpot_sweep/hotpot_validation_dis4_natural.jsonl
mkdir -p results
for M in none oracle; do
  $PY eval_latent_table.py --method "$M" --data "$D" --limit 100 \
      --label "$M" --save-json "results/hotpot_${M}.json"
done
for K in 4 16; do
  $PY eval_latent_table.py --method text --data "$D" --limit 100 --text-topk "$K" \
      --label "text-top${K}" --save-json "results/hotpot_text_top${K}.json"
done
echo "[baselines] done"
