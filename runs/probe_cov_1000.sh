#!/bin/bash
# L_cov 的 1000 步小规模探针（约 1 小时）。
#
# 目的不是出成绩，而是回答一个问题：**L_cov 到底有没有在动选择器**。
# 判据（跑完看这三样）：
#   1. 日志里 cov_pcov 是否在涨（0.42 起步，涨到 0.6+ 说明覆盖真的变好）；
#   2. `tools/diag_symbol_fidelity.py` 的无效选择比例 / 完整答案覆盖率相对
#      results/2026-11-04_v6_select/symbol_fidelity_after.json 有没有动；
#   3. 生成的答案有没有肉眼可见变好（eval 那一行的 full-match）。
#
# 与 runs/run_stage2_bridge.sh 的差别只有三处，其余超参逐字相同：
#   --max-steps 1000      只跑 1000 步（正式跑是 5000 步）
#   --save-every 100      每 100 步落盘，便于中断续训
#   --eval-every 500      少评两次，省时间
#
# 注意（有意为之）：正式配置的预热步数是 _total_opt(5000) * 0.5 = 2500，
# 所以这 1000 步**全程都在"金标读出"预热阶段**，不会切到 router 自选分布。
# 对本次探针这是好事：把"读出分布切换"这个变量固定住，只看 L_cov 对选词的影响。
#
# ---- 用法 ----
#   首次：  COV=0.1 bash runs/probe_cov_1000.sh
#   续训：  RESUME=1 COV=0.1 bash runs/probe_cov_1000.sh
#           （COV 必须与首次一致；不一致时脚本会用 ckpt 里的值并告警）
#   基线：  COV=0.0 OUT=.../probe_base_1000.pt bash runs/probe_cov_1000.sh
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=/data/zhangjiayang/miniconda3/envs/flux/bin/python
OUT=${OUT:-$B/ckpt/probe_cov_1000.pt}
COV=${COV:-0.1}
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
  if [ ! -f "$OUT" ]; then
    echo "[resume] 找不到 $OUT，无法续训" >&2
    exit 1
  fi
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
  --alpha-max 0.5 --lex-gate-init 0.05 --diag-every 50 \
  --inject-mode user --fence-style bracket \
  --m 4 --layer 18 --inner-dim 1024 --heads 8 --bottleneck 512 \
  --max-unit-tokens 32 --max-slots 0 --lr 2e-4 \
  --out "$OUT" --save-every $SAVE_EVERY $RESUME_ARG
