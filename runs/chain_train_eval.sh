#!/bin/bash
# 训练结束 -> 2 题 smoke -> 通过才跑正式评测；不通过就停下并写清原因。
#
# 为什么要在服务器上挂一条链，而不是本地定时提醒：
# 训练一跑就是 90 分钟，中间不需要人看着；但"训练结束"和"评测能不能跑"之间有一道
# 闸门 —— 权重可能根本没存、可能存了但结构不对、可能能加载但读出全是空。这三种
# 情况都不该继续烧 20 分钟去跑 200 题。所以闸门必须自动判、判不过就停。
#
# ---- 用法 ----
#   TRAIN_PID=<训练进程号> CKPT=$B/ckpt/xxx.pt bash runs/chain_train_eval.sh
#   可选：LIMIT（正式评测题数，默认 200）  SMOKE_N（smoke 题数，默认 2）
set -uo pipefail
B=/data/zhangjiayang/yyz3; R=$B/delta-mem-exp4
PY=${PY:-/data/zhangjiayang/miniconda3/envs/flux/bin/python}
TRAIN_PID=${TRAIN_PID:?必须给训练进程号}
CKPT=${CKPT:?必须给目标 ckpt 路径}
LIMIT=${LIMIT:-200}
SMOKE_N=${SMOKE_N:-2}
OUT=${OUT:-$B/results/chain_$(basename "$CKPT" .pt)}
mkdir -p "$OUT" "$B/logs/2026-11-05"
LOG=${LOG:-$B/logs/2026-11-05/chain_$(basename "$CKPT" .pt).log}
exec > >(tee -a "$LOG") 2>&1
say() { echo "[chain $(date '+%F %T')] $*"; }

say "等待训练进程 $TRAIN_PID 结束（ckpt=$CKPT）"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
say "训练进程已退出"

# ---- 闸门 1：权重存在且不像半截文件 ----
if [ ! -f "$CKPT" ]; then
  say "[FAIL] 训练没有产出 $CKPT —— 不跑评测。看训练日志：$B/logs/2026-11-05/"
  exit 2
fi
SZ=$(stat -c%s "$CKPT")
say "ckpt 大小 $SZ 字节"
if [ "$SZ" -lt 100000000 ]; then
  say "[FAIL] ckpt 只有 $SZ 字节（正常 ~220MB），疑似写了一半 —— 不跑评测"
  exit 2
fi

# ---- 闸门 2：2 题 smoke ----
say "smoke：$SMOKE_N 题，模式 oracle_lex（mean 编码）"
GPU=${GPU:-1} LIMIT=$SMOKE_N TAG=chain_smoke MODES="oracle_lex" \
  CKPT="$CKPT" OUT="$OUT/smoke" bash "$R/runs/eval_stage2.sh" > "$OUT/smoke.out" 2>&1
SMOKE_RC=$?
say "smoke rc=$SMOKE_RC"
SMOKE_JSON="$B/results/chain_smoke_oracle_lex_${SMOKE_N}.json"
[ -f "$SMOKE_JSON" ] || SMOKE_JSON=$(ls -t "$B"/results/chain_smoke_*_${SMOKE_N}.json 2>/dev/null | head -1)

VERDICT=$("$PY" - "$SMOKE_JSON" "$SMOKE_N" <<'PY'
import json, sys
path, want = sys.argv[1], int(sys.argv[2])
try:
    d = json.load(open(path))
except Exception as e:
    print(f"FAIL 读不出 smoke 产出 {path}: {e}")
    raise SystemExit(0)
rows = d.get("rows") or []
s = d.get("summary") or {}
problems = []
if len(rows) != want:
    problems.append(f"题数 {len(rows)} != {want}")
if not s.get("mem_tokens"):
    problems.append("mem_tokens 为 0（没有注入任何记忆）")
if not s.get("prompt_tokens") or float(s["prompt_tokens"]) < 80:
    problems.append(f"prompt_tokens={s.get('prompt_tokens')} 太小（注入段可能没进去）")
if not any((r.get("pred") or "").strip() for r in rows):
    problems.append("所有预测都是空串（生成没跑通）")
if not all((r.get("n_soft_content") or 0) > 0 for r in rows):
    problems.append("有样本的内容 token 数为 0")
if problems:
    print("FAIL " + "；".join(problems))
else:
    em = 100 * float(s.get("em") or 0.0)
    print(f"PASS n={len(rows)} EM={em:.0f}% 注入={s.get('mem_tokens'):.1f} "
          f"prompt={s.get('prompt_tokens'):.1f}")
PY
)
say "smoke 判定：$VERDICT"
case "$VERDICT" in
  PASS*) : ;;
  *) say "[FAIL] smoke 没通过 —— 不跑正式评测。看 $OUT/smoke.out"; exit 3 ;;
esac

# ---- 正式评测：实验 1A（mean vs bpe）+ 完整方法参照 ----
say "正式评测：$LIMIT 题，实验 1A（oracle_lex mean vs bpe）+ oracle_latent"
GPU=${GPU:-1} LIMIT="$LIMIT" STAGE=A OUT="$OUT/exp1" CKPT="$CKPT" \
  bash "$R/runs/exp1_bottleneck.sh" > "$OUT/exp1.out" 2>&1
say "实验 1A rc=$?"

say "补一条 oracle_latent（当前完整方法）作为参照"
GPU=${GPU:-1} LIMIT="$LIMIT" TAG=chain_latent MODES="oracle_latent" CKPT="$CKPT" \
  OUT="$OUT/latent" bash "$R/runs/eval_stage2.sh" > "$OUT/latent.out" 2>&1
say "oracle_latent rc=$?"

say "===== 汇总 ====="
"$PY" - "$OUT" "$LIMIT" "$B" <<'PY'
import json, os, sys
root, limit, base = sys.argv[1], sys.argv[2], sys.argv[3]
def p(x):
    return "n/a" if x is None else f"{100*x:.1f}"
for name, path in (("lex_mean", f"{root}/exp1/lex_mean.json"),
                   ("lex_bpe",  f"{root}/exp1/lex_bpe.json"),
                   ("latent",   f"{base}/results/chain_latent_oracle_latent_{limit}.json")):
    if not os.path.isfile(path):
        continue
    d = json.load(open(path)); s = d.get("summary") or {}
    by = s.get("by_ans_type") or {}
    num = by.get("number", {}).get("em"); dat = by.get("date", {}).get("em")
    print(f"{name:<10} EM={p(s.get('em'))}% F1={p(s.get('f1'))}% "
          f"date={p(dat)}% number={p(num)}% 注入={s.get('mem_tokens')}")
PY
say "全部完成 -> $OUT"
