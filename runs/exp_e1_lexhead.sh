#!/bin/bash
# E1：解耦 Attention —— L_cov 只更新**独立词形选择头**，不碰语义注意力。
#
# 背景（results/2026-11-04_readout_probe/）：共享注意力下 `--cov-weight 0.1` 把
# oracle_sem 从 14.0% 打到 7.0%（配对 McNemar p=0.0013），oracle_lex 基本不动。
# E1 把选词分数拆成 `mass.detach() + lex_scorer(w.detach(), x.detach())`：
#   - lex_scorer 零初始化 => 第 0 步选择与旧实现**逐位相同**；
#   - L_cov 只能回到 lex_scorer => 语义分支不再被覆盖目标拉扯。
#
# 与已有两组的唯一差别是 `--lex-head 1`（其余超参逐字相同、同为 1000 步）：
#   已有：COV=0.0（probe_base_1000.pt）/ COV=0.1（probe_cov_1000.pt）
#   本组：COV=0.1 + LEX_HEAD=1
#
# 判据：同一批 200 题跑 oracle_sem / oracle_lex / oracle_latent（runs/exp_e2 一起做）。
#   成功 = oracle_sem 明显恢复（回到 ~14%）**且** 词形选择仍在改善（无效选择率降）。
#   若 oracle_sem 恢复但 oracle_lex 不涨 => 解耦解决了干扰，但覆盖监督本身没收益。
#
# ---- 用法 ----
#   首次：  bash runs/exp_e1_lexhead.sh
#   续训：  RESUME=1 bash runs/exp_e1_lexhead.sh
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=/data/zhangjiayang/miniconda3/envs/flux/bin/python
OUT=${OUT:-$B/ckpt/probe_e1_lexhead.pt}
COV=${COV:-0.1}
LEX_HEAD=${LEX_HEAD:-1}
LEX_HEAD_HIDDEN=${LEX_HEAD_HIDDEN:-0}
LEX_TOKEN_MODE=${LEX_TOKEN_MODE:-mean}
RESUME=${RESUME:-0}
STEPS=${STEPS:-1000}
SAVE_EVERY=${SAVE_EVERY:-100}
EVAL_EVERY=${EVAL_EVERY:-500}
TEACHER_PROBE=${TEACHER_PROBE:-16}
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd $R || exit 1

RESUME_ARG=""
if [ "$RESUME" = "1" ]; then
  if [ ! -f "$OUT" ]; then echo "[resume] 找不到 $OUT" >&2; exit 1; fi
  RESUME_ARG="--resume $OUT"
  echo "[resume] 从 $OUT 接着训"
fi

$PY train_latent_memory.py --router \
  --base $B/models/Qwen3-4B-Instruct-2507 \
  --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --data $B/synth/hotpot_train_10k.jsonl --val-data $B/synth/hotpot_val500.jsonl \
  --n-train 10000 --n-val 120 --epochs 2 --accum 4 --pack 4 \
  --max-steps $STEPS \
  --eval-every $EVAL_EVERY --eval-limit 40 --log-every 10 --enc-batch 24 \
  --recon-weight 0.5 --recon-units 2 --recon-lex 1 --addr-weight 1.0 --kl-weight 1.0 \
  --gold-readout-ratio 0.5 --readout-mode mixed --router-readout-p 0.5 --hard-neg-k 2 \
  --teacher-scope gold --teacher-probe-n $TEACHER_PROBE \
  --topk-per-hop 4 \
  --router-kind hybrid --addr-gamma 0.5 --addr-maxsim 1 \
  --init-router $B/ckpt/router_stage1_8slot.pt --freeze-router \
  --ent-weight 5.0 --stop-weight 0.2 --recon-corrupt 0.5 \
  --cov-weight $COV --cov-targets ${COV_TARGETS:-0} \
  --lex-head $LEX_HEAD --lex-head-hidden $LEX_HEAD_HIDDEN \
  --lex-token-mode $LEX_TOKEN_MODE \
  --alpha-max 0.5 --lex-gate-init 0.05 --diag-every 50 \
  --inject-mode user --fence-style bracket \
  --m 4 --layer 18 --inner-dim 1024 --heads 8 --bottleneck 512 \
  --max-unit-tokens 32 --max-slots 0 --lr 2e-4 \
  --out "$OUT" --save-every $SAVE_EVERY $RESUME_ARG
