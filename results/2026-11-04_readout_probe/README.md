# L_cov 探针两臂的读出分解（2026-11-04，200 题，不重训）

目的：把 `oracle_lex` / `oracle_sem` 这两条"单腿读出"放到 **L_cov 探针的两个权重**上，
看 L_cov 到底改了哪一条腿。顺便补上 `oracle_latent`，把三条读法在同一批题上凑齐。

## 设置

| 项 | 值 |
|---|---|
| 权重 A（基线） | `ckpt/probe_base_1000.pt`，`cov_weight=0.0`，lex_gate=0.1392 |
| 权重 B（实验） | `ckpt/probe_cov_1000.pt`，`cov_weight=0.1`，lex_gate=0.1232 |
| 训练差异 | **只有 `--cov-weight`**，其余超参逐字相同，都 1000 步 |
| 训练阶段 | 1000 步全在"金标读出"预热内（正式预热是 5000 步的一半 = 2500） |
| 评测数据 | `synth/hotpot_val500.jsonl` 前 200 题，与历次读出分解同一批 |
| 注入口径 | 由 ckpt 自带的 `inject_mode=user` / `fence_style=bracket` 决定，不额外传 |
| 每臂注入量 | 30.17 token（两臂完全相同），prompt 83.45 token |
| 判定 | 归一化整串相等（EM） |

三条读法都是**金标槽位直送**（绕过 Router），唯一区别是喂 bridge 的哪一半：

- `oracle_lex`：只用 `z_lex`（原生词向量直通，α 强制 0）
- `oracle_sem`：只用 `Bridge(z_sem)`（完全不给词形）
- `oracle_latent`：两者相加，即当前正式形式 `E = z_lex + tanh(α)·Bridge(z_sem)`

## 结果

| 读法 | 基线（cov=0） | 实验（cov=0.1） | 变化 | 配对检验 |
|---|---|---|---|---|
| `oracle_lex`（词形腿） | 17.5% (35/200) | 16.0% (32/200) | −1.5 pt | base 独赢 12 / cov 独赢 9，**p=0.66 无差异** |
| `oracle_sem`（语义腿） | 14.0% (28/200) | **7.0%** (14/200) | **−7.0 pt** | base 独赢 16 / cov 独赢 2，**p=0.0013 显著变差** |
| `oracle_latent`（两者相加） | 18.0% (36/200) | 14.5% (29/200) | −3.5 pt | base 独赢 13 / cov 独赢 6，p=0.17 方向负但不显著 |

配对检验 = McNemar 精确二项（同一批 200 题逐题配对，不是两个独立比例相减）。

## 怎么读

**1. L_cov 把语义腿打掉了，词形腿没动。**

这正好对应 L_cov 的作用机制：它把 4 个槽的注意力质量重新分配给"实体/数字"这些
**表面重要词**。`z_lex` 是原生词向量的直通，词形本身没被改，所以不受影响（−1.5 pt，
在噪声内）；而 `z_sem` 是压缩后的语义瓶颈，质量一旦向实体词倾斜，它就越来越像
"关键词袋"，丢掉了回答问题需要的语境 —— 14.0% → 7.0%，16 比 2 的配对差，
在这批数据上不太可能是偶然。

**2. 相加的形式替语义腿挡掉了一半损失。**

`oracle_latent` 只掉 3.5 pt，比语义腿的 7 pt 少一半，因为残差形式
`E = z_lex + tanh(α)·Bridge(z_sem)` 里 α 只有 0.12，词形主干还能兜底。
换句话说：**L_cov 的伤害被残差结构吸收了一半，但没有被消除**。

**3. 两臂都明显高于已删除的旧权重那轮（10.5 / 10.5 / 12.5），但不能直接相减。**

这一轮的注入量是 30.17 token，旧那轮是 64.47 token —— **给得更少反而更高**。
差别来自训练管线（v6 的"注入进 user 消息体 + `\n[j] ` 围栏"）和训练步数，
不是同一组对照，只能作为方向参考，不能当成"提升幅度"。

## 结论

在 `--cov-weight 0.1` 这个档位上，**L_cov 与读出准确率是负相关的**：它确实让选词更
偏向实体和数字（见 `results/2026-11-04_probe_cov/`），代价是压缩语义分支明显变差。
所以调参方向不是加大权重，而是要么降权重，要么换监督口径（当前是 Query-Agnostic，
不知道问题问什么，只能盲选"看起来重要"的词 —— 而答案往往恰恰不是最显眼的那几个）。

## 复现

```bash
B=/data/zhangjiayang/yyz3
# 基线臂
MODES="oracle_lex oracle_sem oracle_latent" GPU=0 \
  CKPT=$B/ckpt/probe_base_1000.pt OUT=$B/results/readout_probe_base \
  bash runs/diag_readout.sh
# 实验臂
MODES="oracle_lex oracle_sem oracle_latent" GPU=1 \
  CKPT=$B/ckpt/probe_cov_1000.pt OUT=$B/results/readout_probe_cov \
  bash runs/diag_readout.sh
```

单臂约 33 分钟（每条读法约 11 分钟）。两臂可同时在 GPU 0 / 1 上跑。

文件：`{base,cov}_oracle_{lex,sem,latent}.json`（逐题记录，含 picked / gold / pred）
与同名 `.log`（进度与 summary 行）。
