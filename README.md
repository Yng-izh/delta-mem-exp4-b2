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
| v5 Router（端到端，**作废**） | 8 槽 | 10.6 | 10.6 | 6.0 |
| v5 oracle_latent（金标槽直送） | 8 槽 | 11.8 | 11.8 | 100 |

阶段一（8 槽读出）把完整证据召回从 15.6% 拉到 **52.0%**（=2 条证据的题 63.1%，
≥3 条的题 28.6%）。

**v5 的两个数字说明什么**（详见 `results/v5_2026-11-04/`）：

- `v5 Router` 的 6.0% 与阶段一的 52.0% 矛盾，原因是 `encode_item_slots_full()` 的
  缓存分支丢 `kfeat`（已修），这一行**不能用来判断 Router 能力**，需重跑；
- `v5 oracle_latent` 11.8% 是**有效**的，且它完全绕过 Router —— 说明即使寻址全对，
  当前压缩+读出也只能到"无记忆 +0.6 个点"。因此下一步不是继续训 Router，
  而是先定位读出端（`runs/diag_readout.sh`）。

## 诊断脚本（每条只回答一个问题，不训练）

| 脚本 | 回答什么 | 判定 |
|---|---|---|
| `runs/diag_router_cache.sh` | 缓存路径与训练路径是否等价 | kfeat 最大差 < tol 且 picks 一致率 100%，否则非零退出 |
| `runs/diag_readout.sh` | 读出损失在哪一段：注入 / 词形 / 语义 / 压缩 | 明文高而 embed 低 → 查注入；embed 高而 lex 低 → 查选词；lex 高而 latent 低 → 查残差 |
| `runs/diag_symbol_fidelity.sh` | 答案实体有没有进潜层、进了有没有被改坏 | 命中率≈0 且 α 正常 → 压缩器选词/容量；命中率高但答错 → 读出端 |
| `oracle_embed_full` / `oracle_embed_salient`（同一脚本） | 不压缩的原生词嵌入上限；同预算下按显著性挑词 | 与 `oracle_embed`（句首截断）对比，量化"截断/位置"造成的损失 |
| `runs/overfit_probe.sh` | 64 题能不能拟合；重建目标是否在拖主任务 | 连训练集都拟合不上 → 先修梯度/接口；`qa_kl` vs `recon_old` vs `recon_qa` 三条曲线同框比较 |

## 定位实验结果（2026-11-04，四组诊断）

完整记录与口径见 [`results/2026-11-04_diag/README.md`](results/2026-11-04_diag/README.md)。

| 实验 | 关键数字 | 结论 |
|---|---|---|
| P0-1 Router 缓存一致性 | 完整证据召回 **52.5% vs 52.5%** | 缓存丢 `kfeat` 的缺陷确认修好，v5 的 `6.0%` 作废 |
| P0-2 Oracle 读出分解 | 明文 **54.0%** / 原始词向量 19.0%（18 token）/ 潜层 12.5%（65 token） | 不压缩只给 18 个 token 也只有 19%；潜层给 3.6 倍 token 反而更低 |
| P1-3 符号保真 | 碎片命中 **29.3%**，整词 **36.2%**（58 个含答案槽） | 整词能多救 6.9 点；但每块 32 token 只准带 4 个，随机基线已 17% |
| P1-4 小样本过拟合 | 64 题 200 步：31.25 / 31.25 / 32.81 | 能拟合（起点 12.5%）；三种目标无差别，但步数太少不足以下结论 |

**读出来的病**：读出端的限制是"每块 32 token 只带 4 个，且没有直接监督教它带哪个"，
不是单纯的压缩失真。下一步候选：每块带 8 个 + 整词聚合（便宜），或给选词加显式监督（治本）。

## 仓库结构

| 路径 | 作用 |
|---|---|
| `latent_memory.py` | 压缩器 / Bridge / Router / 槽位银行 / 软 token 注入 / 生成 |
| `train_latent_memory.py` | 阶段二训练 + 评测（`--fst` 分支为动态更新场景） |
| `eval_latent_table.py` | 统一评测入口（`none/text/oracle/oracle_latent/oracle_embed/bridge/router/fst`） |
| `tools/train_router_stage1.py` | 阶段一：只训 Router，训完直接报召回 |
| `tools/diag_router_cache.py` | P0：缓存路径与训练路径逐题对拍（kfeat / picks / 召回） |
| `tools/diag_symbol_fidelity.py` | P1：压缩器选中的词 vs 答案实体，附 Bridge 的 α / 残差比 |
| `tools/overfit_probe.py` | P1：64 题过拟合 + 三种目标（`qa_kl`/`recon_old`/`recon_qa`）对照 |
| `tools/prep_hotpotqa.py` / `tools/prep_mab.py` | 数据转换（**不按答案挑候选**） |
| `runs/run_stage1_router.sh` / `runs/run_stage2_bridge.sh` / `runs/eval_v5.sh` | 三段流程的启动脚本 |
| `runs/diag_*.sh` / `runs/overfit_probe.sh` | 上面四个诊断实验的启动脚本（不训练主模型） |
| `tests/test_v4_refactor.py` | 8 组 CPU 自检（压缩/直通/Bridge/Router/beam/掩码/银行/数据） |
| `tests/test_diag_contracts.py` | 29 项回归：Bridge 梯度量级、α 上限/死锁、重建与 QA 同路径、teacher scope、重建口径 |
| `SETUP.md` | 从零重建：依赖、权重、数据、训练、评测 |
| `DATASETS.md` | 数据集字段、规模与已知性质 |
| `KNOWN_ISSUES.md` | 当前仍未解决的风险 |

## 三条设计不变量（踩过坑，别改回去）

1. **软 token 的范数必须锚在真实 embedding 尺度（≈1.05）**：未约束时实测范数 96–159，
   软 token 变成 attention sink，200 条只答对 1–3 条。
2. **读出以原生词嵌入为主干**：同 33 token 预算下，直接注入金标 token 的真实 embedding
   有 32.8% EM，而平滑平均出来的潜向量只有 17.2%。因此 Bridge 是**残差**：
   `E = z_lex + tanh(α)·Bridge(z_sem)`，最后一层零初始化，α 初值 **0.05**、硬上限 **0.5**。
   α 不能取 0：末层零初始化下 α=0 会让语义分支与门控**同时**失去梯度（死锁在"永远只有 z_lex"）。
   α 也不能无限涨：v5 涨到 0.264 时读出 EM 反而低于 v3。
3. **软 token 必须插在合法消息体内部**（v6）：`<|im_end|>` 与 `<|im_start|>user` 之间是
   "角色真空区"，Qwen3 只把它当对话块之间的换行空白读，跨层注意力会被系统性衰减。
   现在插进 user 消息体内的 `Evidence:` 段（`<|im_start|>user\nEvidence:\n[1] <soft>...`），
   与明文 oracle 的证据段同构；槽位围栏也从 46.6 token 的长文本换成 `\n[j] ` 的极简分隔符。
   旧 ckpt 按旧口径（`between` + `verbose`）评测，不混用。
4. **训练与推理必须是同一套算法**：v4 的教训是训练期 `kfeat/lex_payloads` 被清空、
   第二跳接力方向写反、KL 被 stage 门控静默跳过 —— 三处都让"训练在跑、指标不动"。
   同一条规则现在也适用于**读出分布**（`--readout-mode`）、**teacher 上下文**
   （`--teacher-scope`）与**重建的 bridge 输入**（`--recon-lex`）：任何训练期与推理期
   不一致的地方，都必须显式成参数并写进 ckpt 的 `config`。

## 已知边界

- 读出名额 = 跳数 × 每跳 top-b。8 槽（2×4）是目前配置；≥3 条证据的题仍只有 28.6%。
- 槽位块对角隔离只在 `attn_implementation="eager"` 下生效，SDPA 会静默丢弃 4D 掩码，
  降级原因写在每条样本的 `meta["slot_isolation"]`。
- 详见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)。
