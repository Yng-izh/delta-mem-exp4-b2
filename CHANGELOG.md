# 决策记录（当前架构为什么长这样）

只保留对**当前代码**有约束力的条目。历史实验的逐轮流水已随旧版本一起移除。

## 软 token 的注入位置与围栏结构（2026-11-04 第四轮，v6 手术一）

**问题**：软 token 插在 `<|im_end|>` 与 `<|im_start|>user` 之间的"角色真空区"，而
Qwen3 在指令微调里从没在消息边界外见过连续向量，自注意力只把这段当换行空白；
同时 `slot_isolation=1` 的围栏是 `-- Fact j (memory slot, isolated) --` 前后两段长文本，
实测每题 64.5 个注入 token 里有 46.6 个（72.2%）是重复英文围栏，真正的内容向量只有 17.9 个。

| 改动 | 做法 | 影响 |
|---|---|---|
| 注入位置 | `--inject-mode user`（默认）：`<\|im_start\|>user\nEvidence:\n[1] <soft>\nQuestion: q`；`system` 放进 system 消息内部（与明文 oracle 同构）；`between` 保留旧行为供 A/B | 软向量落在"被当内容读"的位置 |
| 围栏 | `--fence-style bracket`（默认）：每槽 `\n[j] `，2~3 token；`verbose` 保留旧长围栏 | 注入 token 里围栏占比从 72% 降到个位数 |
| 记账 | `meta["n_soft_content"]` / `meta["n_fence"]` 分开报 | "给得多"与"给了有用的东西"不再混为一谈 |
| 兼容 | ckpt 未记录这两个字段（v5 及更早）时**按旧口径**（between + verbose）评测并告警 | 旧结果仍可复现，不会静默换口径 |

**注意**：bracket 样式下 `isolate` 不再改变围栏文本，隔离只剩块对角掩码一条路；
SDPA 后端拿不到 4D 掩码时会降级为"单个分隔符"，比旧的双围栏更弱。实际模式仍写在
`meta["slot_isolation"]` 里。

同轮新增三条原生词嵌入对照（`eval_latent_table.py`）：`oracle_embed_full`（不截断，
可分别放消息体内/消息之间，量化"位置不当"的损失）、`oracle_embed_salient`（同等预算下
按大写/数字/长词/非停用词挑词，而不是从句首硬截）。`overfit_probe.py` 补上**同题 Step 0**
并把训练/评测统一为 `isolate=1`。

## 四组定位实验的结论（2026-11-04 第三轮）

四组实验全部执行完毕，完整记录见 [`results/2026-11-04_diag/README.md`](results/2026-11-04_diag/README.md)。

| 实验 | 结果 | 对当前代码的约束 |
|---|---|---|
| P0-1 缓存一致性 | 召回 52.5% vs 52.5%，kfeat 均非 None | `cache={}` 路径已与训练路径同构；v5 的 6.0% 作废 |
| P0-2 读出分解 | 明文 54.0 / 原始词向量 19.0（18 token）/ 潜层 12.5（65 token） | **"压缩损失"不能单独解释掉点**：token 预算与选词粒度同样是主因 |
| P1-3 符号保真 | 碎片命中 29.3%，整词 36.2% | 选词停在**子词**粒度；每块 32 token 只带 4 个 |
| P1-4 过拟合 | 64 题 200 步：31.25/31.25/32.81 | 能拟合；三目标无差别，但 50 次更新不足以判定 |

由此新增两条已知问题（`KNOWN_ISSUES.md` 第 4、5 条）：
读出端"每块只带 4 个 token + 选词无直接监督"，以及过拟合实验步数不足。

诊断脚本自身也修了三处会让结论失真的缺陷（见 `3e86323`）：符号保真只看第一个金标槽、
跳过全部多块单元；`oracle_embed` 预算按裸原文算块数（差 1/3，把 19.0% 算成 14.5%）。

## 读出端与训练目标的六处对齐（2026-11-04 第二轮）

v5 的 oracle_latent 只有 11.8%（无记忆 11.2%），说明**寻址全对也读不出来**。这一轮只动
"训练目标与读出路径"，不动架构：

| 问题 | 根因 | 修法 | 回归测试 |
|---|---|---|---|
| 重建与 QA 用不同的 bridge 输入 | recon 走 `bridge(z_sem)`，QA 走 `bridge(z_sem, z_lex)`：同一组参数被两个冲突目标拉扯 | `--recon-lex 1`（默认）：recon 也走 `compose_routed_soft`，输入逐位同构；`0` 复现旧行为 | C1–C3 |
| 初始 bridge 梯度 1.85e10 | 末层零初始化 + `unit = out / n.clamp_min(1e-6)`：雅可比 ≈1e6 | 分母夹在 **1.0**（‖out‖≥1 时逐位等价），梯度回到 O(1) | A3–A4 |
| α 无约束地涨到 0.264 | `tanh(lex_gate)` 无上限，残差把读出拖离词表流形 | `--alpha-max`（默认 0.5）+ 诊断打印 α / 残差比 / ‖W_last‖ | B1–B2 |
| α=0 时语义分支永远拿不到梯度 | 零初始化下 ∂e/∂W ∝ α = 0 且 ∂e/∂α ∝ res = 0，只能靠"另一条不走 α 的重建路径"启动 | α 初值 **0.05**（res 仍严格为 0，输出仍等于 z_lex，但梯度可流） | A5–A6 |
| 训练从未见过推理的读出分布 | 训练=金标+难负例，推理=router 自选 8 条（可能缺证据） | `--readout-mode gold/mixed/router`（默认 mixed，预热后 50% 走 `forward_hops` 的真实 picks），并累计 `readout_stats` 写进 ckpt | — |
| teacher 吃整份 memory | 20+ 条干扰下 teacher 自己也可能答错，KL 在蒸馏噪声 | `--teacher-scope gold`（默认）+ 开跑前 `--teacher-probe-n` 量两种上下文的 acc | D1–D6 |

另外三处口径修正：

- `evaluate_recon` 改为**单槽输入 / 单槽目标 / 自由生成**，并单独报实体与数字恢复率；
  旧口径（整份 memory 当目标 + teacher forcing）的 59.6% token-acc 不能当保真结论。
- `oracle_latent` 之前没把 `seg_ids`/`isolate` 传给 `_emit_latent`，eager 后端拿不到
  块对角掩码，与 `router` 不同构；现在两条路都传。
- `oracle_embed` 的预算按 ckpt 的 `max_unit_tokens`/`m` 对齐（旧默认 64 对 32 会让
  "软 token 注入 vs 压缩损失"的归因失真）。

## 三段流程（2026-11-04 定稿）

| 阶段 | 做什么 | 为什么 |
|---|---|---|
| 一：只训 Router | 冻结主干，在静态特征 h0/h18 上优化 `addr_loss_set` | Router 的输入与 Compressor 无关；绑在生成循环里训 14 小时，指标反被未收敛的难负例带偏 |
| 二：训 Compressor+Bridge | Router 冻结，KL 蒸馏 + 重建 + 抗噪读出 | 读出瓶颈已定位在压缩保真，不在寻址 |
| 三：评测 | `router` + `oracle_latent` 同 ckpt | 前者是端到端，后者是"100% 选对证据"的读出上界 |

## 已修的致命接线错误（每条都曾让实验白跑）

| 现象 | 根因 | 修法 |
|---|---|---|
| v4 端到端 9.8%，低于无记忆 11.2% | 训练期 `kfeat`/`lex_payloads` 被无条件清空，hybrid 键与直通分支**从未训练**，评测却全程启用 | 预初始化挪到编码分支之前 |
| `loss_kl` 恒为 `n/a` | ① KL 只在 `stage=="b"` 生效；② 打包路径要求 student/teacher 形状一致（144 vs 933 token，永不相等） | 解除 stage 门控；改为逐样本按答案位置对齐 |
| 完整证据召回平台在 24% | 第二跳接力 `sim.topk()` 取的是**最相似** token（方向反了）；训练用的第二跳不含接力；两条 beam 共用一份桥接词 | 新增 `relay_bridge_tokens`（升序取 + 排除 padding + 显著性代理）；训练与推理共用同一算法；分支状态扩成 `(qv, path, q_tok)` |
| ≥3 条证据的题召回到顶 7.5% | 4 个读出名额装不下 3–5 条必选证据 | `--topk-per-hop 2→4`（2 跳 = 8 槽），召回 52.0% |

## 结构决策（有实测支撑，别改回去）

1. **Bridge 用残差锚定**：`E = z_lex + tanh(α)·Bridge(z_sem)`，末层零初始化，
   α 初值 **0.05**（不能是 0：零初始化下 α=0 会让语义分支与门控同时失去梯度）、
   硬上限 **0.5**（v5 涨到 0.264 时读出退化）。
   依据：同 33 token 预算，原生 h0 embedding 32.8% EM，平滑平均的潜向量 17.2%。
2. **直通分支低温尖锐选通**：`τ_lex=0.05` + 逐行互斥，m 个槽分别锚到 m 个不同实词 token。
3. **寻址键来自冻结特征**：`h0/h18` 注意力池化 + 去均值 IDF-MaxSim；k_out 做新信息抑制，
   标题区（前 3 个有效 token）加成。
4. **软 token 范数锚定 ≈1.05**：未约束时范数 96–159，软 token 变 attention sink。
5. **训练 = 推理**：同一 `compose_routed_soft`、同一接力、同一 topk；任何"训练期关掉、
   推理期打开"的开关都必须写进日志计数。

## 数据与判定

- 训练：`synth/hotpot_train_10k.jsonl`（9997 条，全 2 跳）；评测：`synth/hotpot_val500.jsonl`。
- 候选选择**不看答案**（`select_facts_stream`）；不可解样本计数上报，不静默丢弃。
- 判定：归一化整串相等（EM），另报 acc / F1 / 读对槽位 / 完整证据召回。
