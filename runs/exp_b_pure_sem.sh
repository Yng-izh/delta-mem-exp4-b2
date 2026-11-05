#!/bin/bash
# 实验 B：真正训练一次**纯语义读出**（训练与评测同构）。
#
# 与过去所有 oracle_sem 数字的区别（这是这个实验存在的全部理由）：
#   过去：Bridge 训练时吃 `z_sem + z_lex`（残差锚定在词形上），评测时才单喂 z_sem ——
#         那是训练时**没有优化过**的输入，低分不能证明"z_sem 丢了事实"。
#   现在：`--no-lex-skip` 从第 0 步起就只喂 z_sem，α 恒 0，没有词形直通；
#         训练前向 / QA 损失 / 评测 oracle_sem 用的是同一个函数同一种输入。
#
# 关掉的：L_cov（依赖词级选词，纯语义下没有目标）、recon-lex（没有词形载荷）、
#         独立词形头（只在 lex_skip=True 时注册）。--no-lex-skip 与它们冲突时
#         训练脚本会**直接报错**，不会静默不生效（见 validate_lex_config）。
# 保留的：QA/KL 目标、重建目标（只用 z_sem）、金标读出、冻结 router。
#
# 判据（写在 EXPERIMENTS.md「实验 B」）：
#   显著高于 none 且随步数持续上升 -> 语义压缩器学得会，可以继续做事实绑定
#   始终贴着 none               -> 潜层接口/容量/训练方式需要重新考虑，E3/E4 暂停
# 参照线：base_1000 的 oracle_sem = 28/200，cov_1000 = 14/200，E1 = 12/200
#         （但注意：那三组都是"训练时吃两份输入"的权重，与本组不完全可比）
#
# ---- 用法 ----
#   首次：  bash runs/exp_b_pure_sem.sh
#   续训：  RESUME=1 bash runs/exp_b_pure_sem.sh
#   只训练：EVAL=0 bash runs/exp_b_pure_sem.sh
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=${PY:-/data/zhangjiayang/miniconda3/envs/flux/bin/python}
OUT=${OUT:-$B/ckpt/probe_b_puresem.pt}
RES=${RES:-$B/results/exp_b_puresem}
RESUME=${RESUME:-0}
STEPS=${STEPS:-1000}
SAVE_EVERY=${SAVE_EVERY:-100}
EVAL_EVERY=${EVAL_EVERY:-500}
EVAL=${EVAL:-1}
LIMIT=${LIMIT:-200}
mkdir -p "$RES"
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

echo "[exp_b $(date '+%F %T')] 纯语义读出训练 -> $OUT"
# 超参与 E1 / base / cov 三组**逐字相同**，唯一差别是 --no-lex-skip + cov=0 + recon-lex=0，
# 这样四条曲线（base / cov / e1 / b）之间只有读出结构这一个变量。
$PY train_latent_memory.py --router \
  --base $B/models/Qwen3-4B-Instruct-2507 \
  --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --data $B/synth/hotpot_train_10k.jsonl --val-data $B/synth/hotpot_val500.jsonl \
  --n-train 10000 --n-val 120 --epochs 2 --accum 4 --pack 4 \
  --max-steps $STEPS \
  --eval-every $EVAL_EVERY --eval-limit 40 --log-every 10 --enc-batch 24 \
  --recon-weight 0.5 --recon-units 2 --recon-lex 0 --addr-weight 1.0 --kl-weight 1.0 \
  --gold-readout-ratio 0.5 --readout-mode mixed --router-readout-p 0.5 --hard-neg-k 2 \
  --teacher-scope gold --teacher-probe-n 16 \
  --topk-per-hop 4 \
  --router-kind hybrid --addr-gamma 0.5 --addr-maxsim 1 \
  --init-router $B/ckpt/router_stage1_8slot.pt --freeze-router \
  --ent-weight 5.0 --stop-weight 0.2 --recon-corrupt 0.5 \
  --cov-weight 0 --lex-head 0 \
  --no-lex-skip \
  --alpha-max 0.5 --lex-gate-init 0.05 --diag-every 50 \
  --inject-mode user --fence-style bracket \
  --m 4 --layer 18 --inner-dim 1024 --heads 8 --bottleneck 512 \
  --max-unit-tokens 32 --max-slots 0 --lr 2e-4 \
  --out "$OUT" --save-every $SAVE_EVERY $RESUME_ARG \
  > "$RES/train.log" 2>&1
_rc=$?
echo "[exp_b $(date '+%F %T')] 训练结束 rc=$_rc"
if [ $_rc -ne 0 ] || [ "$EVAL" != "1" ]; then
  tail -30 "$RES/train.log"
  exit $_rc
fi

# ---- 评测：只跑纯语义这一条腿（oracle_sem 在这组权重上与 oracle_latent 逐位同构）----
for m in oracle_sem none oracle; do
  echo "[exp_b $(date '+%F %T')] eval $m"
  $PY eval_latent_table.py --mode "$m" --ckpt "$OUT" \
    --data $B/synth/hotpot_val500.jsonl --limit "$LIMIT" --max-new-tokens 16 \
    --topk-per-hop 4 --slot-isolation 1 --progress-every 25 --stratify \
    --base $B/models/Qwen3-4B-Instruct-2507 \
    --adapter $B/models/delta-mem_qwen3_4b_instruct \
    --label "exp_b-$m" --save-json "$RES/$m.json" \
    > "$RES/$m.log" 2>&1
  echo "[exp_b $(date '+%F %T')] eval $m rc=$?"
done
echo "[exp_b $(date '+%F %T')] 全部完成 -> $RES"
grep -h "| " "$RES"/*.log | tail -20
