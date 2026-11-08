#!/bin/bash
# 实验 F：FAD-KV 核心压缩实验（冻结 Qwen，只训 codec）
#
#   阶段一：L6~35 按"层段 × K/V"分 6 组，每组独立线性编解码，总潜码 1024 维 = 2 KiB/token
#           初始化 = 分组 PCA（第 0 步就等于 PCA），用 L_QA + L_KL + L_kv 功能蒸馏
#   阶段二：30% 事实锚点逐个保留（1536 维）+ 70% 背景 4:1 池化（384 维/槽）
#           平均 ≈1.03 KiB/token，序列保留 ≈47.5%
#
# 判据（写在 EXPERIMENTS.md「实验 F」）：
#   阶段一目标 EM >= 40%（Global PCA-1024 是 23.0%，原生是 45.5%）
#   阶段二目标在存储再减半后仍 ~40%；掉太多就只调锚点比例与池化率，不加新架构
#
# ---- 用法 ----
#   PY=<envs/knowlearn_data/bin/python> GPU=1 STAGE=all bash runs/exp_f_fadkv.sh
#   STAGE=train1 STEPS=200 bash runs/exp_f_fadkv.sh       # 先小步数看损失
#   STAGE=eval bash runs/exp_f_fadkv.sh                   # 只跑主表
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=${PY:?必须指定 PY（如 \$B/../miniconda3/envs/knowlearn_data/bin/python）}
TAG=${TAG:-exp_f_fadkv}
STAGE=${STAGE:-all}
LIMIT=${LIMIT:-200}
GPU=${GPU:-1}
LAYERS=${LAYERS:-6-35}
SEGMENTS=${SEGMENTS:-3}
LATENT=${LATENT:-1024}
STEPS1=${STEPS1:-1000}
STEPS2=${STEPS2:-600}
LOSSES=${LOSSES:-kv,qa,kl}
ANCHOR_RATIO=${ANCHOR_RATIO:-0.3}
POOL=${POOL:-4}
ANCHOR_DIM=${ANCHOR_DIM:-1536}
BG_DIM=${BG_DIM:-384}
FIT_TOKENS=${FIT_TOKENS:-20000}
# K/V 预算比（空=按组均分）。实测 V 的重建误差是 K 的 1.6 倍、吃掉 67% 总误差
KV_SPLIT=${KV_SPLIT:-}
OUT=${OUT:-$B/results/$TAG}
mkdir -p "$OUT"
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=$B/nodns:$R:$B/delta-Mem
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU
cd "$R" || exit 1
BASE=$B/models/Qwen3-4B-Instruct-2507
TRAIN=$B/synth/hotpot_train_10k.jsonl
VAL=$B/synth/hotpot_val500.jsonl
COMMON="--base $BASE --gpu 0 --layers $LAYERS --segments $SEGMENTS"
# eval 不接 --segments：层段数从 codec 的 meta 里读，传了会 argparse 直接报错
COMMON_EVAL="--base $BASE --gpu 0 --layers $LAYERS"

run_init() {
  echo "[F-init $(date '+%F %T')] 分组 PCA（${SEGMENTS}段 × K/V，总 $LATENT 维）"
  $PY - "$@" <<'PY' > "$OUT/init.log" 2>&1
import sys, argparse, torch
sys.path.insert(0,'.')
import train_latent_memory as T, midlayer_kv as MK, kv_lowrank as LR
import train_fadkv as TR
a=argparse.Namespace(base=sys.argv[1], gpu=0, dtype='auto', layers=sys.argv[2],
    segments=int(sys.argv[3]), latent=int(sys.argv[4]), tokens=int(sys.argv[5]),
    out=sys.argv[6], kv_split=(sys.argv[7] if len(sys.argv) > 7 else ''),
    encoding='full_prefix',
    system_prompt='You are a helpful assistant. Answer with the shortest exact value.')
ns=argparse.Namespace(smoke_model=False, layer=0, base=a.base, dtype=a.dtype, gpu=0,
    no_adapter=True, adapter='', grad_checkpointing=False)
dev=torch.device('cuda:0'); T._setup_tf32(True)
model,tok,dev,_=T._load_backbone(ns,dev); T.freeze_model(model)
layers=TR.parse_layers(a.layers); layout=MK.uv_layout(model,layers)
data=T.load_synth_jsonl('/data/zhangjiayang/yyz3/synth/hotpot_train_10k.jsonl',
                        limit=0, seed=0)
import train_fadkv as TR2
gr=LR.fit_grouped(TR._sub_batches(a,model,tok,layers,data,a.tokens), layout,
                  latent_dim=a.latent, n_segments=a.segments, device=dev,
                  kind_weights=TR2.parse_kind_weights(a.kv_split),
                  log=lambda m: print(m, flush=True))
gr.save(a.out); print('[init]', gr.describe())
PY
  local rc=$?
  tail -4 "$OUT/init.log"
  return $rc
}

BASIS=$OUT/fadkv_groups_${LATENT}.pt

run_train1() {
  echo "[F-train1 $(date '+%F %T')] steps=$STEPS1 losses=$LOSSES latent=$LATENT"
  $PY train_fadkv.py --stage 1 --data "$TRAIN" --val-data "$VAL" --eval-limit 30 \
    --latent-dim "$LATENT" --steps "$STEPS1" --losses "$LOSSES" \
    --init "$BASIS" --out "$OUT/fadkv_stage1.pt" $COMMON \
    > "$OUT/train1.log" 2>&1
  local rc=$?
  tail -8 "$OUT/train1.log"
  return $rc
}

run_train2() {
  echo "[F-train2 $(date '+%F %T')] steps=$STEPS2 anchor=$ANCHOR_RATIO pool=$POOL"
  $PY train_fadkv.py --stage 2 --data "$TRAIN" --val-data "$VAL" --eval-limit 30 \
    --anchor-ratio "$ANCHOR_RATIO" --pool "$POOL" \
    --anchor-dim "$ANCHOR_DIM" --bg-dim "$BG_DIM" \
    --steps "$STEPS2" --losses "$LOSSES" \
    --init "$OUT/fadkv_stage1.pt" --out "$OUT/fadkv_stage2.pt" $COMMON \
    > "$OUT/train2.log" 2>&1
  local rc=$?
  tail -8 "$OUT/train2.log"
  return $rc
}

run_eval() {
  echo "[F-eval $(date '+%F %T')] limit=$LIMIT"
  local extra=""
  [ -f "$OUT/fadkv_stage1.pt" ] && extra="$extra --stage1 $OUT/fadkv_stage1.pt"
  [ -f "$OUT/fadkv_stage2.pt" ] && extra="$extra --stage2 $OUT/fadkv_stage2.pt"
  [ -f "$GLOBAL" ] && extra="$extra --global-basis $GLOBAL --global-dim 1024"
  $PY eval_fadkv.py --data "$VAL" --limit "$LIMIT" --encoding full_prefix \
    --wrong --progress-every 10 --show-items 2 \
    --dump-jsonl "$OUT/rows.jsonl" --save-json "$OUT/summary.json" $extra $COMMON_EVAL \
    > "$OUT/eval.log" 2>&1
  local rc=$?
  grep -E "^\[cfg\]|^\| |^\[判据" "$OUT/eval.log" | tail -30
  return $rc
}

GLOBAL=${GLOBAL:-}
case "$STAGE" in
  init)   run_init "$BASE" "$LAYERS" "$SEGMENTS" "$LATENT" "$FIT_TOKENS" "$BASIS" "$KV_SPLIT" ;;
  train1) run_train1 ;;
  train2) run_train2 ;;
  eval)   run_eval ;;
  all)
    run_init "$BASE" "$LAYERS" "$SEGMENTS" "$LATENT" "$FIT_TOKENS" "$BASIS" "$KV_SPLIT" \
      || { echo "[F] PCA 初始化失败，停止"; exit 1; }
    run_train1 || { echo "[F] 阶段一训练失败，停止"; exit 1; }
    run_train2 || { echo "[F] 阶段二训练失败，停止"; exit 1; }
    run_eval ;;
  *) echo "未知 STAGE=$STAGE（init|train1|train2|eval|all）"; exit 2 ;;
esac
