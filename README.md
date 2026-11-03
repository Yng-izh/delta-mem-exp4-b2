# 潜层记忆实验：把事实压成软 token，交给冻结的 Qwen3-4B 读

在 **HotpotQA distractor（2 跳）** 上做对照实验：历史事实经 `LatentCompressor` 压成
少量潜向量，经 `LatentBridge` 变成软 token 注入冻结主干；`HybridLatentRouter` 负责
在候选槽位里选出该读哪几条。评测集 `hotpot_val500`（500 题），判定为归一化整串相等。

## 当前流程（三段，缺一不可）

```
① 阶段一：只训 Router        tools/train_router_stage1.py   （约 1.5 小时 / 1 epoch）
   冻结主干，在冻结特征 h0/h18 上优化寻址，训完立刻报 500 题召回率
        ↓  产出 ckpt/router_stage1.pt（召回 52.0%）
② 阶段二：训 Compressor+Bridge  train_latent_memory.py --init-router --freeze-router
   Router 冻结；KL 蒸馏（对齐明文 teacher）+ 事实重建 + 抗噪读出（金标恒在 + 难负例）
        ↓  产出 ckpt/hotpot_router_10k_v5.pt
③ 评测                        runs/eval_v5.sh
   router（端到端） + oracle_latent（金标槽位直送 bridge 的读出上界）
```

**为什么拆两段**：Router 的输入特征（冻结 h0/h18）与 Compressor 无关，是静态的。
把它绑在生成循环里训练，等于用十几小时等一个本该一小时收敛的东西，而且 Router 没收敛
时采出的难负例会把 Bridge 一起带偏（v4 就是这样把读出从 17.2% 带到 10.0% 的）。

## 当前数字（HotpotQA val500，n=500）

| 方法 | 读出名额 | acc% | EM% | 完整证据召回% |
|---|---|---|---|---|
| 无记忆（`none`） | — | 11.2 | 11.2 | — |
| 明文 BM25 top-4（`text`） | 151 token | 31.2 | 32.0 | 36.0 |
| 明文金标（`oracle`，上界） | 91 token | 54.6 | 55.4 | 100 |
| Router（v3 权重） | 4 槽 | 12.6 | 12.8 | 5.6 |
| Router（v4 权重，接线有 bug） | 4 槽 | 9.8 | 9.8 | 15.6 |
| **阶段一 Router** | **8 槽** | — | — | **52.0** |

阶段一（8 槽读出）把完整证据召回从 15.6% 拉到 **52.0%**（=2 条证据的题 63.1%，
≥3 条的题 28.6%）。端到端 EM 等阶段二评测完成后再补。

## 仓库结构

| 路径 | 作用 |
|---|---|
| `latent_memory.py` | 压缩器 / Bridge / Router / 槽位银行 / 软 token 注入 / 生成 |
| `train_latent_memory.py` | 阶段二训练 + 评测（`--fst` 分支为动态更新场景） |
| `eval_latent_table.py` | 统一评测入口（`none/text/oracle/oracle_latent/oracle_embed/bridge/router/fst`） |
| `tools/train_router_stage1.py` | 阶段一：只训 Router，训完直接报召回 |
| `tools/prep_hotpotqa.py` / `tools/prep_mab.py` | 数据转换（**不按答案挑候选**） |
| `runs/run_stage1_router.sh` / `runs/run_stage2_bridge.sh` / `runs/eval_v5.sh` | 三段流程的启动脚本 |
| `tests/test_v4_refactor.py` | 8 组 CPU 自检（压缩/直通/Bridge/Router/beam/掩码/银行/数据） |
| `SETUP.md` | 从零重建：依赖、权重、数据、训练、评测 |
| `DATASETS.md` | 数据集字段、规模与已知性质 |
| `KNOWN_ISSUES.md` | 当前仍未解决的风险 |

## 三条设计不变量（踩过坑，别改回去）

1. **软 token 的范数必须锚在真实 embedding 尺度（≈1.05）**：未约束时实测范数 96–159，
   软 token 变成 attention sink，200 条只答对 1–3 条。
2. **读出以原生词嵌入为主干**：同 33 token 预算下，直接注入金标 token 的真实 embedding
   有 32.8% EM，而平滑平均出来的潜向量只有 17.2%。因此 Bridge 是**残差**：
   `E = z_lex + tanh(α)·Bridge(z_sem)`，α 初值 0、最后一层零初始化。
3. **训练与推理必须是同一套算法**：v4 的教训是训练期 `kfeat/lex_payloads` 被清空、
   第二跳接力方向写反、KL 被 stage 门控静默跳过 —— 三处都让"训练在跑、指标不动"。

## 已知边界

- 读出名额 = 跳数 × 每跳 top-b。8 槽（2×4）是目前配置；≥3 条证据的题仍只有 28.6%。
- 槽位块对角隔离只在 `attn_implementation="eager"` 下生效，SDPA 会静默丢弃 4D 掩码，
  降级原因写在每条样本的 `meta["slot_isolation"]`。
- 详见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)。
