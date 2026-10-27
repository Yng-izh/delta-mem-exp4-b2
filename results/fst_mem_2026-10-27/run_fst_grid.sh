#!/usr/bin/env bash
# FST-Mem 阶段 B 的对照网格（阶段 A 基线 / 硬覆写 / FST 完整 / 去时序条件）
#
# 用法：在 v100-2 上  bash results/fst_mem_2026-10-27/run_fst_grid.sh [seed...]
# 默认 3 个 seed。每档产出独立目录，含 config + 指标 json（由 save_latent_checkpoint 的 meta 落盘）。
#
# 注意：服务器 Python 环境被其他用户破坏（editable transformers 指向不存在的 /root/transformers-main），
# 所以这里把 5 个必需的环境变量写死；换机器时只需改 PY / PYTHONPATH。
set -euo pipefail

REPO=${REPO:-/share/exp4group}
PY=${PY:-/share/conda/envs/latent2/bin/python}
SEEDS=${@:-0 1 2}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export HF_HOME=${HF_HOME:-/share/hf_home}
export HF_HUB_OFFLINE=1
export DELTA_MEM_SCAN_IMPL=torch
export PYTHONPATH=/share/exp4group/shim19:/share/exp4group:/share/delta-Mem

cd "$REPO"
OUT=${OUT:-runs/fst_mem_2026-10-27}
mkdir -p "$OUT"

COMMON="--epochs 3 --accum 16 --eval-every 200 --eval-limit 72 --save-every 0 --seed"

for S in $SEEDS; do
  # ① 阶段 A：静态多事实内容寻址（基线）
  echo "=== [1/4] stage-A router  seed=$S"
  $PY train_latent_memory.py --router \
      --data synth/latent_exp2_train.jsonl --val-data synth/latent_exp2_val.jsonl \
      $COMMON "$S" --out "$OUT/a_router_s$S.pt"

  # ② 只有稳定事实键，历史全丢（硬覆写对照）
  echo "=== [2/4] FST hist_max=0（硬覆写）  seed=$S"
  $PY train_latent_memory.py --fst --hist-max 0 \
      --data synth/fst_update_train.jsonl --val-data synth/fst_update_val.jsonl \
      $COMMON "$S" --out "$OUT/b_hard_s$S.pt"

  # ③ FST 完整：事实键 + 状态/历史分离 + 时序两级读出
  echo "=== [3/4] FST hist_max=4（完整）  seed=$S"
  $PY train_latent_memory.py --fst --hist-max 4 \
      --data synth/fst_update_train.jsonl --val-data synth/fst_update_val.jsonl \
      $COMMON "$S" --out "$OUT/c_fst_s$S.pt"

  # ④ 去掉查询里的时间条件（时序寻址的消融）
  echo "=== [4/4] FST hist_max=4 --ignore-query-time  seed=$S"
  $PY train_latent_memory.py --fst --hist-max 4 --ignore-query-time \
      --data synth/fst_update_train.jsonl --val-data synth/fst_update_val.jsonl \
      $COMMON "$S" --out "$OUT/d_no_time_s$S.pt"
done

echo
echo "完成。每档必报：acc / group_acc / ver_multi_acc / 三探针 / pair_acc / order_stable / n_soft"
echo "对照判读：②③ 的差值才是'状态—轨迹解耦'的贡献；③④ 的差值才是'时序条件寻址'的贡献。"
echo "注意：pair_acc 会因为 --eval-limit 随机截断而拆散部分反事实对，报数时必须同时报 pair_n。"
