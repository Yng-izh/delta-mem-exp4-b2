# HotpotQA 四方法对照实验（2026-10-27 起）

本目录记录**第一组基于公开 benchmark 的结果**。旧的自造玩具数据只作为 smoke 测试保留，
不再作为科研结论来源。

## 1. 协议（四组方法必须完全一致）

| 项 | 设置 |
|---|---|
| 冻结主模型 | Qwen3-4B-Instruct-2507 + Delta-Mem adapter（`/share/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507`），全程冻结 |
| 数据 | HotpotQA distractor；dev 来自**官方 validation**（7405 条），训练来自官方 **train**（90447 条）；两侧不可能重叠 |
| dev 子集 | `synth/hotpot_sweep/hotpot_validation_dis4_natural.jsonl`（前 100 题，2 黄金段落 + 4 干扰段落，句级槽位） |
| 记忆单元 | 一条句子一个槽位（1 句 = 1 事实） |
| prompt 模板 | 明文与潜层走**同一条** `_emit_latent` 路径（`build_chat_parts` → system + [记忆] + question），明文只是 `soft=None`，潜层是软 token |
| 解码 | 贪心，`max_new_tokens=16` |
| 判定 | 归一化后**整串相等**（不做子串包含） |
| 命令 | `python eval_latent_table.py --method {none,text,oracle,bridge,router,fst} --data <jsonl> --limit 100 --label ... --save-json ...` |

## 2. 已测到的参照点（100 题，全部同一协议）

| 方法 | n | acc% | 读对槽位% | pair% | update% | hist% | 记忆 token | prompt 总 token | 查询 ms |
|---|---|---|---|---|---|---|---|---|---|
| `none`：不给任何记忆 | 100 | **12.0** | n/a | n/a | n/a | n/a | 0.0 | 22.3 | 940 |
| `oracle`：只给支持事实（上界） | 100 | **58.0** | 100.0 | n/a | n/a | n/a | 94.1 | 129.4 | 1469 |
| `text`：BM25 top-4 明文 RAG | 100 | **36.0** | 97.0 | n/a | n/a | n/a | 150.5 | 185.7 | 1916 |
| `text`：BM25 top-16 明文 RAG | 100 | **47.0** | 100.0 | n/a | n/a | n/a | 558.1 | 613.3 | 4928 |
| `bridge`（Compressor+Bridge） | — | 待训练 | — | — | — | — | 4·K | — | — |
| `router`（Compressor+Router+Bridge） | — | 待训练 | — | — | — | — | 4 | — | — |
| `fst`（FST-Mem） | — | 待训练 | — | — | — | — | 4 | — | — |

### 这三个数字各自的意思（引用时必须区分）

1. **12.0% = 纯参数知识地板。** HotpotQA 的很多答案本来就在 Qwen 的参数记忆里。
   任何"记忆方法"的准确率都要与 **12%** 比，不是与 0 比。少报这一行 = 把模型的先验当成方法的功劳。
2. **58.0% = 上界。** 只把 2 条支持事实（94 token）放进 system 段，其余全不给。
   说明瓶颈**不在读出端**：只要有正确的句级事实 + 明文，这台模型能答到 58%。
3. **36.0% / 47.0% = 可部署的明文基线。** 同一批题，BM25 取回 4 句（150 token）得 36.0%，
   取回 16 句（558 token）得 47.0%，检索命中率 97~100%。
   **这就是潜层记忆要打的那条线**：用 4 个软 token 去够 150~558 个明文 token 的效果。
   注意 token 数与准确率不是线性关系（150→558 token 只换来 +11 个点），
   说明"多给明文"很快就饱和，"能不能读到对的那句"更关键。

## 3. 必须随表一起报告的数据事实

- **答案在上下文里是常态**：含黄金段落 95.0%、加 8 段干扰 96.3%（`meta.leak`，只统计干扰段落）。
  所以 exact-match 单独看会把"字符串检索"当成"记忆读出"，这也是必须给 `none` 与"读对槽位率"的原因。
- **96% 是 2 跳题**（bridge/comparison 全为 multi-hop）：路由必须开 `hops=2` 链式选槽。
- **0.017% 的 supporting_facts 越界**（1/6000）：转换时跳过并计数，不 clamp。

## 4. 训练配置（`bridge` / `router` 行）

```
python train_latent_memory.py --router \
  --data synth/hotpot_train.jsonl --val-data synth/hotpot_sweep/hotpot_validation_dis4_natural.jsonl \
  --n-train 2500 --n-val 120 --epochs 1 --accum 4 --recon-weight 0.3 --recon-units 2 \
  --addr-weight 1.0 --kl-weight 0 --m 4 --layer 18 --inner-dim 512 --heads 8 --bottleneck 512 \
  --max-unit-tokens 64 --max-slots 8 --lr 1e-4 --out ckpt/hotpot_r1.pt --save-every 300
```

- 可训练参数 14.375M（compressor 4.994M + bridge 2.630M + router 6.752M），主干 4027.9M 冻结
- 峰值显存 9.2→11.0 GB（每步日志的 `peakmem`）；同卡还有其他用户的进程，OOM 需先看是不是自己的
- 每步 ~16-27s（4 条样本），2500 条 1 epoch ≈ 2~4 小时
- 日志：服务器 `/share/exp4group/runs/hotpot_r1.log`，checkpoint `ckpt/hotpot_r1.pt`

## 5. 本轮之后必须补的三件事

1. `bridge` / `router` 两行的实测（训练完直接用上面的 `eval_latent_table.py --method` 跑同一 dev）。
2. **抗干扰控制实验**：同一批题在 dis0 / dis2 / dis4 / dis8 与 natural / shuffled 下重跑，
   看准确率与"读对槽位率"是否稳定（`tools/prep_hotpotqa.py --sweep-out` 已生成 8 个对照集）。
3. MAB-CR（`tools/prep_mab.py`，cap=32）上的更新准确率与历史访问准确率，FST-Mem 与普通 Router 直接对比。
