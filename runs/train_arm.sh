#!/bin/bash
# 训练**一个臂**：1000 步，超参与 `probe_base_1000` 逐字相同，只由环境变量换一个变量。
#
# 为什么只有一个脚本：每个实验都复制一份脚本，迟早会漂成"两份会漂移的实现"。
# 这里把所有"会变的"都提成环境变量，其余（数据、步数、lr、pack、accum、注入位置、
# 围栏、router）全部写死，保证任意两臂之间只差你显式改的那一个变量。
#
# ---- 变量 ----
#   OUT          产物路径（必填）
#   MAX_UNIT     每块多少 token 才压成 m 个向量（默认 32）
#   M            每块输出几个记忆位置（默认 4）
#                压缩比 = MAX_UNIT / M：32/4 = 8 倍；8/4 = 2 倍；16/8 = 2 倍。
#                16/8 是当前推荐档：压缩比仍是 2 倍，但 8 个位置放得下 6 token 的数字
#                （Qwen3 一个数字一个 token，51,271 = 6 个）
#   COV          L_cov 权重（默认 0；0.1 = 第十二轮探针那档）
#   LEX_HEAD     1 = 独立词形选择头（E1）
#   COMPOSE       residual（旧行为）/ split（符号位 + 关系位，实验 2 的核心）
#   N_SYM / N_REL split 下每块的符号位 / 关系位数，两者之和必须等于 m（默认 4）
#   REL_MODE      latent（连续关系向量）/ blank（无信息占位，实验 2 的对照臂）
#   ROUTER_CKPT   阶段一 Router 权重。**换 m 之后旧 Router 不能复用**（k_in_net 的
#                 输入维度是 m*d，形状对不上）。做实验 1/2（金标证据）时把它设成空串，
#                 并用 READOUT=gold HARDNEG=0 —— 读出只看金标槽，Router 不参与。
#                 实验 3 需要 m 对应的 Router，要另外跑一遍阶段一。
#   READOUT       gold（只走金标，实验 1/2 用）/ mixed / router
#   HARDNEG       难负例个数（无 Router 时可设 0）
#   STEPS        训练步数（默认 1000）
#   RESUME=1     从 OUT 续训
#
# ---- 用法 ----
#   基线（cov=0）：        OUT=$B/ckpt/probe_base_1000.pt bash runs/train_arm.sh
#   L_cov 探针：           COV=0.1 OUT=$B/ckpt/probe_cov_1000.pt bash runs/train_arm.sh
#   E1 独立词形头：        COV=0.1 LEX_HEAD=1 OUT=$B/ckpt/probe_e1_lexhead.pt bash runs/train_arm.sh
#   降压缩比（2 倍）：     MAX_UNIT=8 OUT=$B/ckpt/probe_mut8_1000.pt bash runs/train_arm.sh
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=${PY:-/data/zhangjiayang/miniconda3/envs/flux/bin/python}
OUT=${OUT:?必须显式指定 OUT}
MAX_UNIT=${MAX_UNIT:-32}
M=${M:-4}
COV=${COV:-0}
LEX_HEAD=${LEX_HEAD:-0}
LEX_HEAD_HIDDEN=${LEX_HEAD_HIDDEN:-0}
LEX_TOKEN_MODE=${LEX_TOKEN_MODE:-mean}
COMPOSE=${COMPOSE:-residual}
N_SYM=${N_SYM:--1}
N_REL=${N_REL:--1}
REL_MODE=${REL_MODE:-latent}
STEPS=${STEPS:-1000}
SAVE_EVERY=${SAVE_EVERY:-100}
EVAL_EVERY=${EVAL_EVERY:-500}
RESUME=${RESUME:-0}
ROUTER_CKPT=${ROUTER_CKPT-$B/ckpt/router_stage1_8slot.pt}
READOUT=${READOUT:-mixed}
HARDNEG=${HARDNEG:-2}
TEACHER_PROBE=${TEACHER_PROBE:-16}
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$R" || exit 1

RESUME_ARG=""
if [ "$RESUME" = "1" ]; then
  if [ ! -f "$OUT" ]; then echo "[resume] 找不到 $OUT" >&2; exit 1; fi
  RESUME_ARG="--resume $OUT"
  echo "[resume] 从 $OUT 接着训"
fi

ROUTER_ARGS=""
if [ -n "${ROUTER_CKPT}" ]; then ROUTER_ARGS="--init-router $ROUTER_CKPT --freeze-router"; fi
echo "[train_arm] OUT=$OUT M=$M MAX_UNIT=$MAX_UNIT COV=$COV LEX_HEAD=$LEX_HEAD" \
     "COMPOSE=$COMPOSE N_SYM=$N_SYM N_REL=$N_REL REL_MODE=$REL_MODE STEPS=$STEPS" \
     "READOUT=$READOUT HARDNEG=$HARDNEG ROUTER=${ROUTER_CKPT:-无（金标读出）}"
$PY train_latent_memory.py --router \
  --base $B/models/Qwen3-4B-Instruct-2507 \
  --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --data $B/synth/hotpot_train_10k.jsonl --val-data $B/synth/hotpot_val500.jsonl \
  --n-train 10000 --n-val 120 --epochs 2 --accum 4 --pack 4 \
  --max-steps $STEPS \
  --eval-every $EVAL_EVERY --eval-limit 40 --log-every 10 --enc-batch 24 \
  --recon-weight 0.5 --recon-units 2 --recon-lex 1 --addr-weight 1.0 --kl-weight 1.0 \
  --gold-readout-ratio 0.5 --readout-mode $READOUT --router-readout-p 0.5 \
  --hard-neg-k $HARDNEG \
  --teacher-scope gold --teacher-probe-n $TEACHER_PROBE \
  --topk-per-hop 4 \
  --router-kind hybrid --addr-gamma 0.5 --addr-maxsim 1 \
  $ROUTER_ARGS \
  --ent-weight 5.0 --stop-weight 0.2 --recon-corrupt 0.5 \
  --cov-weight $COV --cov-targets 0 \
  --lex-head $LEX_HEAD --lex-head-hidden $LEX_HEAD_HIDDEN \
  --lex-token-mode $LEX_TOKEN_MODE \
  --compose $COMPOSE --n-sym $N_SYM --n-rel $N_REL --rel-mode $REL_MODE \
  --alpha-max 0.5 --lex-gate-init 0.05 --diag-every 50 \
  --inject-mode user --fence-style bracket \
  --m $M --layer 18 --inner-dim 1024 --heads 8 --bottleneck 512 \
  --max-unit-tokens $MAX_UNIT --max-slots 0 --lr 2e-4 \
  --out "$OUT" --save-every $SAVE_EVERY $RESUME_ARG
