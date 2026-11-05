# 潜层记忆实验：把事实压成软 token，交给冻结的 Qwen3-4B 读

在 **HotpotQA distractor（2 跳）** 上做对照实验：历史事实经 `LatentCompressor` 压成
少量潜向量，经 `LatentBridge` 变成软 token 注入冻结主干；`HybridLatentRouter` 负责
在候选槽位里选出该读哪几条。评测集 `hotpot_val500`，判定为归一化整串相等（EM）。

- 数字与结论：**[RESULTS.md](RESULTS.md)**
- 实验协议与判据：**[EXPERIMENTS.md](EXPERIMENTS.md)**
- 当前仍未解决的风险：**[KNOWN_ISSUES.md](KNOWN_ISSUES.md)**
- 架构为什么长这样：**[CHANGELOG.md](CHANGELOG.md)**

## 当前流程（三段，缺一不可）

```
① 阶段一：只训 Router        tools/train_router_stage1.py   （约 1.5 小时 / 1 epoch）
   冻结主干，在冻结特征 h0/h18 上优化寻址，训完立刻报召回率
        ↓  产出 ckpt/router_stage1_8slot.pt（完整证据召回 52.0%）
② 阶段二：训 Compressor+Bridge  train_latent_memory.py --init-router --freeze-router
   Router 冻结；KL 蒸馏（对齐明文 teacher）+ 事实重建 + 抗噪读出（金标恒在 + 难负例）
        ↓  产出 ckpt/*.pt
③ 评测                        eval_latent_table.py / tools/probe_readout_capacity.py
   router（端到端） + oracle_latent（金标槽位直送 bridge 的读出上界）
```

**为什么拆两段**：Router 的输入特征（冻结 h0/h18）与 Compressor 无关，是静态的。
把它绑在生成循环里训练，等于用十几小时等一个本该一小时收敛的东西，而且 Router 没收敛
时采出的难负例会把 Bridge 一起带偏。

## 当前在研究什么

主线问题：**把一段事实写成潜层记忆之后，未来出现不同问题，模型还能否正确恢复事实关系。**

只做两个实验（协议见 `EXPERIMENTS.md`）：

| 实验 | 新训练量 | 一句话 |
|---|---|---|
| A 读出接口上限 | **0 步** | 同题同骨架、逐槽等预算，原生 BPE vs 潜层 |
| B 纯语义读出 | 1 × 1000 步 | `--no-lex-skip`，训练/评测同构 |

E2/E3/E4、`L_cov` 调参、独立词形头**全部暂停**（E1 已证伪"语义退化是 L_cov 梯度造成"）。

## 当前权重（只存在于服务器 `ckpt/`，不入库）

| 权重 | 用途 |
|---|---|
| `router_stage1_8slot.pt` | 阶段一 Router（冻结用），完整证据召回 52.0% |
| `probe_base_1000.pt` | 1000 步基线（cov=0），实验 A 的参照权重 |
| `probe_cov_1000.pt` | 1000 步 + `--cov-weight 0.1`（共享注意力） |
| `probe_e1_lexhead.pt` | 1000 步 + `--cov-weight 0.1 --lex-head 1`（E1，已证伪） |

## 脚本

| 脚本 | 回答什么 |
|---|---|
| `runs/exp_a_capacity.sh` | **实验 A**：接口容量 vs 压缩损失（不训练，约 20 分钟） |
| `runs/exp_b_pure_sem.sh` | **实验 B**：纯语义读出能不能学会（1000 步 + 自动评测） |
| `runs/train_arm.sh` | 训练单个臂（1000 步），用环境变量换一个变量：`MAX_UNIT`/`COV`/`LEX_HEAD`/`SPLIT_INJECT` |
| `runs/run_stage1_router.sh` | 阶段一 Router |
| `runs/eval_stage2.sh` | 通用评测入口（`CKPT` 必填） |
| `runs/diag_router_cache.sh` | 缓存路径与训练路径是否等价 |
| `runs/diag_symbol_fidelity.sh` | 答案实体有没有进潜层、进了有没有被改坏 |
| `runs/overfit_probe.sh` | 64 题能不能拟合；重建目标是否在拖主任务 |

## 仓库结构

| 路径 | 作用 |
|---|---|
| `latent_memory.py` | 压缩器 / Bridge / Router / 槽位银行 / 软 token 注入 / 生成 |
| `train_latent_memory.py` | 阶段二训练 + 评测（`--fst` 分支为动态更新场景，不在主线） |
| `eval_latent_table.py` | 统一评测入口（`none/text/oracle/oracle_latent/oracle_lex/oracle_sem/oracle_embed/bridge/router/fst`） |
| `tools/train_router_stage1.py` | 阶段一：只训 Router，训完直接报召回 |
| `tools/probe_readout_capacity.py` | 实验 A：逐槽等预算下原生 BPE vs 潜层 |
| `tools/cmp_readout.py` | 两份 eval 产出逐题配对比较（McNemar 精确二项） |
| `tools/diag_router_cache.py` | 缓存路径与训练路径逐题对拍（kfeat / picks / 召回） |
| `tools/diag_symbol_fidelity.py` | 压缩器选中的词 vs 答案实体，附 Bridge 的 α / 残差比 |
| `tools/overfit_probe.py` | 64 题过拟合 + 三种目标（`qa_kl`/`recon_old`/`recon_qa`）对照 |
| `tools/prep_hotpotqa.py` / `tools/prep_mab.py` | 数据转换（**不按答案挑候选**） |
| `results/<日期>_*/` | **只有最新一轮**的原始 json/log；历史轮次见 `RESULTS.md` |
| `logs/<日期>/` | 只有最新一轮的训练日志 |
| `tests/` | CPU 自检：`test_diag_contracts` 118 项、`test_readout_experiments` 94 项、`test_v4_refactor` 8 组、`test_train_objectives` 20 项、数据转换 38 项 |
| `SETUP.md` / `DATASETS.md` | 从零重建；数据字段与规模 |

## 四条设计不变量（踩过坑，别改回去）

1. **软 token 的范数必须锚在真实 embedding 尺度（≈1.087）**：未约束时实测范数 96–159，
   软 token 变成 attention sink，200 条只答对 1–3 条。对齐方式是 `unit(out)·scale`，
   **不是** RMSNorm×常数（那会把范数推到 ≈target·√d）。
2. **读出以原生词嵌入为主干**：同预算下直接注入金标 token 的真实 embedding 有 32.8% EM，
   而平滑平均出来的潜向量只有 17.2%。因此 Bridge 是**残差**：
   `E = z_lex + tanh(α)·Bridge(z_sem)`，最后一层零初始化，α 初值 **0.05**、硬上限 **0.5**。
   α 不能取 0（末层零初始化下 α=0 会让语义分支与门控同时失去梯度，死锁在"只有 z_lex"），
   也不能无限涨（旧权重到 0.264 时读出反而更低）。
   **零初始化的前提是"有锚"**：`--no-lex-skip` / `split_inject` 没有锚，不零初始化。
3. **软 token 必须插在合法消息体内部**：`<|im_end|>` 与 `<|im_start|>user` 之间是
   "角色真空区"，Qwen3 只当换行空白读，跨层注意力被系统性衰减。现在插进 user 消息体内的
   `Evidence:` 段，围栏用极简 `\n[j] `（旧长围栏每题吃掉 46.6 个注入 token 的 72.2%）。
   旧 ckpt 按旧口径（`between` + `verbose`）评测，不混用。
4. **训练与推理必须是同一套算法**：任何训练期与推理期不一致的地方，都必须显式成参数
   并写进 ckpt 的 `config`（`--readout-mode`、`--teacher-scope`、`--recon-lex`、
   `--inject-mode`、`--fence-style`）。实验 B 就是这条规则的直接产物。

## 已知边界

- 读出名额 = 跳数 × 每跳 top-b，8 槽（2×4）是当前配置；≥3 条证据的题仍只有 28.6%。
- 槽位块对角隔离只在 `attn_implementation="eager"` 下生效，SDPA 会静默丢弃 4D 掩码，
  降级原因写在每条样本的 `meta["slot_isolation"]`。
- 检索键（每条记忆 64 token 的 h0/h18）与词形载荷 `z_lex` 都**不在**"注入 token 数"里，
  谈压缩率时必须一起算（`KNOWN_ISSUES.md` 第 21 条）。
