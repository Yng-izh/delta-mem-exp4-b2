# SETUP.md — 在一台全新服务器上从零重建本实验

**一句话**：装依赖 → 拿权重 → 生数据 → 训练 → 评测，五步。所有路径用环境变量 `B` 统一，
本仓库假设 `B=<数据盘>/exp4`，脚本里出现 `$B` 就是这个。

> **必须知道的两件事**
> 1. **仓库里没有 `.pt` 权重**（`.gitignore` 排除了 `ckpt/` 与 `*.pt`）。想复现已有数字，
>    要么从旧机器拷 `ckpt/hotpot_router_10k_v3.pt`，要么按第 4 步自己重训。
> 2. **多卡显存要 ≥24 GB 空闲**。4B 主干 + adapter + 潜层读写，加载阶段 OOM 会直接退出；
>    跑之前先 `nvidia-smi` 看一眼有没有别的进程占着卡。

---

## 1. 依赖

```bash
conda create -y -n latent python=3.11
conda activate latent
pip install -r requirements.txt
# CUDA 版本按驱动选：cu121 / cu126 / cu130 都能跑，本仓库实测 cu126
```

**另有三个仓库外的依赖代码**（不是 pip 包）：

| 依赖 | 作用 | 获取 |
|---|---|---|
| `delta-Mem` | adapter 与 `deltamem.core` 实现 | `git clone https://github.com/declare-lab/delta-mem` |
| `deltamem_compat.py` | 让 `deltamem.core` 能在 transformers 4.51 上 import（本仓库已含） | 仓库内 |
| MemoryAgentBench parquet | MAB-CR 原始数据 | 见第 3 步 |

运行时统一：

```bash
export B=/data/<你>/exp4
export R=$B/delta-mem-exp4
export PYTHONPATH=$R:$B/delta-Mem
export DELTA_MEM_SCAN_IMPL=torch        # 不开会走 numpy 逐层实现，慢十倍
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$B/hf_home               # 权重缓存，避免重复下载
```

## 2. 权重（约 10 GB）

服务器通常连不上 `huggingface.co`，用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=0
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-4B-Instruct-2507", local_dir="/data/<你>/exp4/models/Qwen3-4B-Instruct-2507")
snapshot_download("declare-lab/delta-mem_qwen3_4b_instruct", local_dir="/data/<你>/exp4/models/delta-mem_qwen3_4b_instruct")
PY
```

**adapter 拉不下来（401 / Repository Not Found）时**：`declare-lab/delta-mem_qwen3_4b_instruct`
在镜像上是 gated 仓库，需要 `HF_TOKEN`（`HF_ENDPOINT=https://hf-mirror.com HF_TOKEN=hf_xxx python ...`）。
没有 token 就走拷贝：任何一台已经跑过实验的机器上都有 `models/delta-mem_qwen3_4b_instruct`
（adapter 只有几百 MB），直接 `rsync -av <老机器>:.../models/delta-mem_qwen3_4b_instruct/ models/`。
主干 4B 权重是公开的，不受影响。

之后所有训练/评测都加 `HF_HUB_OFFLINE=1`。**主干必须冻结**：本实验只训练 Compressor / Bridge / Router，
主干参数一个都不更新（`_load_backbone` 已 `requires_grad_(False)`）。

## 3. 数据

```bash
mkdir -p $B/synth $B/results $B/ckpt $B/logs

# HotpotQA distractor -> 训练/评测 jsonl（2 跳）
python tools/prep_hotpotqa.py --out-train $B/synth/hotpot_train_10k.jsonl \
       --out-val $B/synth/hotpot_val500.jsonl --n-train 10000 --n-val 500

# MAB Conflict Resolution -> 冲突覆盖集（**不看答案选候选**，v4 起 --cap 0 = 不截断）
python tools/prep_mab.py --src <Conflict_Resolution-*.parquet> --out $B/synth/mab_cr_cap32_v2.jsonl
```

数据字段、规模、缺口与"答案本来就在上下文里"这类数据集性质，见 [DATASETS.md](DATASETS.md)。
转换器有硬约束：`units` 一条事实一个槽、`evidence` 必须是支持事实原文、同键冲突版本必须整组进上下文。

## 4. 训练

```bash
# 阶段一：只训 Router（冻结特征上优化寻址，训完直接报召回率）
GPU=0 bash runs/run_stage1_router.sh

# 阶段二：固定阶段一 Router，只训 Compressor + Bridge（KL 蒸馏 + 重建 + 抗噪）
GPU=0 bash runs/run_stage2_bridge.sh
```

断点续训：两条命令加 `--resume <ckpt>`（权重 + 优化器动量 + step/epoch 位置 + RNG 全部恢复）。

关键参数含义（`train_latent_memory.py --help` 有全量）：

| 参数 | 含义 |
|---|---|
| `--m 4 --layer 18` | 每条事实 4 个潜槽，取第 18 层隐状态 |
| `--inner-dim 1024` | Compressor 内部宽度（v3 是 512，v4 加宽） |
| `--hard-neg-k 2` | 金标 + 难负例路径里掺入的干扰槽数 |
| `--readout-mode {gold,mixed,router}` | QA 读出的槽位分布。默认 `mixed`：预热后一半步数走 router 的真实 picks（与推理同分布）。`gold` 是旧行为（作对照） |
| `--teacher-scope {gold,full}` | KL teacher 看到的事实。默认 `gold`（与 student 应读到的一致）；`full` 是旧行为（整份 memory 含干扰） |
| `--teacher-probe-n 16` | 开跑前先量 teacher 在 memory/gold 两种上下文上的自由生成 acc |
| `--recon-lex {1,0}` | 重建是否与 QA 走同一条 `bridge(z_sem, z_lex)`。默认 1；0 复现"重建只训语义分支"的旧行为 |
| `--kl-weight 1.0 --recon-weight 0.5` | 蒸馏与重建权重 |
| `--alpha-max 0.5 --lex-gate-init 0.05` | 残差 α 的硬上限与初值。初值**不能为 0**（末层零初始化下 α=0 会让语义分支与门控同时失去梯度） |
| `--inject-mode {user,system,between}` | 软 token 插在哪。默认 `user`（user 消息体内的 Evidence: 段，与明文 oracle 同构）；`between` 是旧行为（消息之间的角色真空区，仅 A/B 对照） |
| `--fence-style {bracket,verbose}` | 槽位围栏。默认 `bracket`（每槽 `\n[j] `，2~3 token）；`verbose` 是旧长围栏（每题约 46.6 token） |
| `--diag-every 50` | 每 50 步打印 α / 残差比 / 输出范数 / 末层权重范数，并记录裁剪前总梯度与 clip_coef |
| `--topk-per-hop 4` | 2 跳 Top-4 beam → 最多 8 槽读出 |
| `--router-kind {latent,hybrid}` | hybrid = 冻结 h0/h18 注意力池化 + Token-MaxSim 兜底 |
| `--slot-isolation` | 槽位块对角掩码（**仅 eager 后端生效**，SDPA 会静默丢弃 4D 掩码） |

产物：`$B/ckpt/hotpot_router_10k_v6.pt`（v5 的产物是 `hotpot_router_10k_v5.pt`，那一轮早于
本轮的目标/读出对齐，见 `results/v5_2026-11-04/`）；`save_latent_checkpoint` 会写入
`lex_skip / router_kind / inner_dim` 等构建参数，加载时按 ckpt 重建结构，**不要手动对齐超参**。

## 5. 评测

```bash
python eval_latent_table.py --data $B/synth/hotpot_val500.jsonl --limit 500 \
  --ckpt $B/ckpt/hotpot_router_10k_v6.pt \
  --base $B/models/Qwen3-4B-Instruct-2507 \
  --adapter $B/models/delta-mem_qwen3_4b_instruct \
  --label v5-router --save-json $B/results/v5_router_500.json
```

`--mode`（别名 `--method`）七选一：

| mode | 记忆以什么形式进入 | 用途 |
|---|---|---|
| `none` | 什么都不给 | **必须报的底线** |
| `text` | 明文 top-k 段落 | Text-RAG 对照 |
| `oracle` | 明文金标证据 | 明文上界 |
| `oracle_latent` | **金标槽位直送 Bridge**（不经 Router） | 潜层读出上界：测"100% 选对证据"时能到多少 |
| `oracle_lex` | 金标槽位、**只用 z_lex**（α=0） | 读出分解：纯词形保真那条腿 |
| `oracle_sem` | 金标槽位、**只用 Bridge(z_sem)** | 读出分解：纯语义映射那条腿 |
| `oracle_embed` | 金标证据原文的真实 token embedding（不压缩，**句首截断**到预算） | 区分"软 token 注入机制"与"压缩损失"；预算按 ckpt 的 `max_unit_tokens/m` 对齐 |
| `oracle_embed_full` | 同上但**不截断**（整条证据的原生词嵌入全给） | 不压缩的上限；配 `--embed-inject {user,between}` 可量化"注入位置不当"的损失 |
| `oracle_embed_salient` | 同等预算下按**显著性**挑词（大写/数字/长词/非停用词/问题里没出现过） | 回答"18 个 token 到底能装多少信息"，排除句首截断的物理假象 |
| `bridge` | 压缩器潜向量，单槽 | 保真诊断 |
| `router` | Router 寻址 + Bridge | 主方法 |
| `fst` | 多跳潜层接力 + 版本策略 | 动态更新场景 |

报告口径：**`oracle − oracle_latent` = 压缩损失，`oracle_latent − router` = 寻址损失**。
日志与 json 里 `enc_ms / gen_ms / prompt_tokens / mem_tokens` 已分开记账，不要只报总耗时。

## 6. 诊断（不训练主模型，只回答一个机制问题）

```bash
GPU=0 bash runs/diag_router_cache.sh      # 缓存路径 vs 训练路径逐题对拍（P0）
GPU=0 bash runs/diag_readout.sh           # 五种读法分解：注入 / 词形 / 语义 / 压缩（P0）
GPU=0 bash runs/diag_symbol_fidelity.sh   # 压缩器选中的词 vs 答案实体（P1）
GPU=0 bash runs/overfit_probe.sh          # 64 题过拟合 + 三种目标函数对照（P1）
```

每个脚本的判定路径写在脚本头部注释与 `README.md` 的"诊断脚本"表里。
它们都**只读** ckpt，不写模型权重。

## 7. 自检

```bash
python -m pytest tests/ -q          # 含 v4 七项重构自检（CPU，秒级）
python -m py_compile latent_memory.py train_latent_memory.py eval_latent_table.py tools/prep_mab.py
```

## 8. 常见坑

| 现象 | 原因 / 处理 |
|---|---|
| `import deltamem` 报 `ImportError: smollm3` | transformers 版本与 `backbone_compat.py` 不匹配 → 用 `deltamem_compat.py`，或把 transformers 钉到 4.51.3 |
| 加载权重 OOM | 卡上还有别人的进程；4B 需要 ≥24 GB 空闲 |
| 槽位隔离"开了等于没开" | SDPA 只接受 2D/3D 掩码，4D 块对角被静默丢弃 → 设 `attn_implementation="eager"`；降级原因记在每条样本的 `meta["slot_isolation"]` |
| `loss_addr=inf` | 旧集合损失写法；v4 用 `log_softmax + logsumexp` 的集合边缘化，若复现旧代码需自查 |
| tokenizer 反序列化崩 | protobuf 7.x → 钉 `protobuf==4.25.9` |
| 数字对不上历史 | transformers/torch 版本不同；跨版本不可比，同栈内部才可比 |

更多未解风险见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)，决策记录见 [CHANGELOG.md](CHANGELOG.md)。
