# 潜层读出阶段 A 首次真机训练（2026-10-21/22，v100-2）

**结论先说：训练跑完了，但没学会 —— 验证 exact-match = 0/32。**
瓶颈定位到**压缩器学不会编码**（不是通路、不是数据、不是训练量）。下面的日志与数字都是实测，
不是估算；checkpoint 未入库（每个 30 MB），配置写在每条日志末尾的 `[save]` 行里。

## 1. 正式训练（`train_stage_a_full_1875steps.log`）

| 项 | 值 |
|---|---|
| 数据 | `synth/train_15k.jsonl` 15,000 条（12,002 单跳 + 2,998 多跳） |
| 超参 | stage a，m=4，layer=18，inner=512，heads=8，bottleneck=512，accum=8，lr=1e-4，1 epoch |
| 步数 / 耗时 | 1,875 步 / 13,454 s（3.7 h），卡 5 |
| 可训练参数 | 7.62 M（compressor 4.99 M + bridge 2.63 M），冻结 4,027.9 M |
| 显存 | 9.8 GB（V100-32G），利用率 87–97% |
| loss_qa | 12.59（1–10 步均值）→ 8.73（11–20）→ 7.73（21–30）→ **3.87（末步）** |
| **验证 exact-match** | **0.0（n=32）** |
| 梯度自检 | `{"compressor_queries": true, "compressor_input_proj": true, "bridge_last_linear": true, "model_has_no_grad": true}` |

生成结果不是乱码，是**格式对、名字错**：

```
Q: What did Stelsal Trurkdest found?
A: Ironwood Energy Nilork
生成: Kestrel Analytics Vrindkraulmiothd, a
```

## 2. 两次外部事故（都不是代码问题）

| 日志 | 现象 | 处理 |
|---|---|---|
| `train_stage_a_killed_at_step532.log` | 卡 1 上跑到 step 532（59 min）被**外部杀掉**：无 traceback，同刻全机 10 张卡任务全被清空（2–5 号卡各 31.7 GB 的任务也没了），load average 72→17，**机器没重启** | 从 step 400 的 checkpoint 续训，救回 45 min |
| `train_stage_a_oom_on_card1.log` | 换回卡 1 重试时 `CUDA error: out of memory`（加载模型阶段）：卡 1 已被另一个任务重新占满 31.7/32.7 GB | 换到空闲卡 5 |

> 这台机器随时可能被清。所以训练加了 `--save-every`（本次 100 步），最坏丢 10 分钟。

## 3. 瓶颈定位：四个过拟合 / 对照实验

同一批 200 条样本、8 epoch、lr 2e-4、accum 4（共 400 步）。**全部在训练集上评**，是最宽松的判据。

| 日志 | 条件 | QA exact-match | 说明 |
|---|---|---|---|
| `overfit200_m4.log` | m=4（16 软 token） | **1/200** | 连 200 条都记不住 |
| `overfit200_m16.log` | m=16（64 软 token，加宽 4 倍） | **3/200** | 加宽无效 → 不是容量宽度问题 |
| `overfit200_m4_norm.log` | m=4 + 软 token 范数对齐 | **0/200** | 尺度是真问题，但事后修不够 |
| `overfit200_m4_recon.log` | m=4 + `--recon-weight 1.0`（强制重建原文） | **0/200**，重建 token-acc **22.8%** | 逼不出来 |

真模型上的三条对照（见 `tools/diag_latent_capacity.py`）：

| 子命令 | 结果 | 结论 |
|---|---|---|
| `--sub text`（明文记忆） | **6/6** | 数据、模型、任务都成立 |
| `--sub oracle`（记忆原文 embedding 当软 token，跳过压缩器） | **4/4** | 注入通路（位置/mask/拼接）完全正常 |
| `--sub scale` | 软 token 范数 **96–159**，真实 embedding **1.05** | 差两个量级 → 软 token 成了 attention sink；已修（`LatentBridge` 范数对齐） |
| `--sub probe` | 重建 token-acc **22.8%**；平凡基线 top1 6.0% / **top10 19.8%** | 只比"无脑吐前 10 高频词"高 3 个点 → 压缩器几乎没编码进内容 |

## 4. 判据链（为什么可以排除其他解释）

1. 明文 6/6 → 排除数据、模型、任务、评测口径。
2. Oracle 4/4 → 排除软 token 注入通路（位置、mask、拼接、位置编码）。
3. 4 项 grad-check 全 true → 排除梯度断路（`bridge` 曾经真的不在计算图里，已修）。
4. 200 条过拟合 1/200 → 排除数据量、epoch 数、泛化问题。
5. m 4→16 无改善 → 排除"软 token 数量不够"。
6. recon 22.8% ≈ 平凡基线 19.8% → **锁定：压缩器没有把内容编码进 z**。

## 5. 尚未验证 / 未排除

- **没跑过更长训练**：只跑了 400 步（过拟合）和 1,875 步（全量）。soft-prompt 类方法通常要几千步、
  lr 大一个量级（这里 2e-4）。**"没训够"与"结构学不出"尚未区分**，这是下一步实验。
- **没跑过 stage B（KL 蒸馏）**：本目录所有数字都来自 stage a 的纯 QA 交叉熵。
- **没在评测集上跑过**：0/32 是训练期验证观察值，不是 MemoryAgentBench / LongMemEval 指标。
- **语料是虚构的**：27 种关系、用词分布窄于真实文本；换真实语料只需把 `--facts` 指向它。
- 探针的对照实验（随机输入 → 100%）说明：200 样本 × 4 万维输入，线性层怎么都能硬背，
  所以探针数字**只能当"有信号"的弱证据**，不能当信息量度量。真正的判据是重建准确率对平凡基线。

## 6. 复现

```bash
export HF_HOME=/share/hf_home HF_HUB_OFFLINE=1 DELTA_MEM_SCAN_IMPL=torch
# 注意：env 里的 transformers 在 2026-10-21 21:25 被别人改坏（editable 指向不存在的
# /root/transformers-main），所以必须用 vendored 副本绕开：
export PYTHONPATH=/share/exp4group/shim19:/share/exp4group:/share/delta-Mem
P=/share/conda/envs/latent2/bin/python

# 正式训练（本次日志的来源）
CUDA_VISIBLE_DEVICES=5 $P train_latent_memory.py --stage a \
  --data synth/train_15k.jsonl --val-data synth/val_2k.jsonl --gpu 0 \
  --accum 8 --epochs 1 --save-every 100 --out ckpt/latent_a.pt

# 瓶颈定位
$P tools/diag_latent_capacity.py --sub text
$P tools/diag_latent_capacity.py --sub oracle
$P tools/diag_latent_capacity.py --sub scale --ckpt ckpt/overfit200_recon.pt
$P tools/diag_latent_capacity.py --sub probe --ckpt ckpt/overfit200_recon.pt
```

`shim19` 只是一个只含 `transformers` 符号链接的目录：

```bash
mkdir -p /share/exp4group/shim19
ln -sfn /share/exp4group/pylib5/transformers /share/exp4group/shim19/transformers
```
