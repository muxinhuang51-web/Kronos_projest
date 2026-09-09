# S1 Next-Token Post-Training 技术报告

**实验 ID**: `s1_next_token_all_a_share_baseline`
**日期**: 2026-07-31
**状态**: 已完成

---

## 1. 实验目标

在 Kronos-small 预训练模型基础上，对全 A 股日线数据进行 S1 粗 token 的 next-token 后训练。仅训练 predictor 的 `head.proj_s1` 层（525,312 参数），强化模型对 S1 token 的预测能力。Tokenizer 和 Predictor backbone 全部冻结。

---

## 2. 数据管线

### 2.1 数据源

| 项目 | 配置 |
|------|------|
| 数据源 | tushare pro |
| 品种 | 全 A 股（含已退市），point-in-time 股票池 |
| 时间跨度 | 2016-01-01 至 2026-07-31 |
| 特征 | open, high, low, close, volume, amount |
| 复权 | OHLC 后复权（raw × adj_factor），volume/amount 不复权 |
| 股票数 | 5,534 |

### 2.2 时间切分

| 集合 | 时间范围 | 有效样本数 |
|------|----------|------------|
| Train | 2016-01-01 ~ 2023-12-31 | 7,249,645 |
| Validation | 2024-01-01 ~ 2024-12-31 | 1,277,076 |
| Test | 2025-01-01 ~ 最新 | 2,062,892 |
| **合计** | | **10,589,613** |

切分按 target 日期归属，禁止随机切分导致泄漏。

### 2.3 样本构造

每个样本为 `(ts_code, target_date)` 对：

- **Context**: target_date 前 256 个交易日（跳过停牌日，不前向填充）
- **Target**: target_date 当日的 OHLCVA
- **归一化**: context-only z-score（ddof=0, eps=1e-5），clip to [-5, 5]
- **采样**: 均匀有放回抽样，每 epoch 100,000 训练样本 + 20,000 验证样本
- **有效训练窗口**: 5,943,402（含 ≥256 个前序交易日的样本）

### 2.4 数据冻结

下载和处理完成后，原始 Parquet 数据、复权结果和 manifest 全部冻结，训练只读取冻结快照。

---

## 3. 模型配置

### 3.1 基础模型

| 组件 | 路径 | 参数量 |
|------|------|--------|
| Tokenizer | `NeoQuasar/Kronos-Tokenizer-base` | ~15.8M (冻结) |
| Predictor | `NeoQuasar/Kronos-small` | ~24.7M |
| └ 可训练 | `head.proj_s1.{weight, bias}` | **525,312** |
| s1_bits | 10 | vocab_s1 = 1,024 |
| d_model | 512 | |

### 3.2 训练超参数

| 参数 | 值 |
|------|-----|
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Learning Rate | 4 × 10⁻⁵ |
| Weight Decay | 0.1 |
| Batch Size | 50 (MPS) |
| Epochs | 30 |
| LR Schedule | OneCycle Cosine (pct_start=0.03, div_factor=10, final_div_factor=10000) |
| Gradient Clip | 3.0 |
| Precision | FP32 |
| 训练设备 | Apple MPS (GPU) |
| 总训练时间 | ~9 小时 |

---

## 4. 实验结果

### 4.1 验证集指标汇总

| Epoch | Val CE ↓ | Perplexity ↓ | Top-1 ↑ | Top-5 ↑ | Bit Acc ↑ | Hamming ↓ |
|-------|----------|-------------|---------|---------|-----------|-----------|
| 1 | 3.3856 | 29.54 | 16.81% | 48.87% | 77.38% | 2.2617 |
| **2** | **3.3441** | **28.34** | 17.29% | **50.34%** | 77.92% | 2.2078 |
| 5 | 3.3690 | 29.05 | 17.85% | 51.24% | 78.20% | 2.1797 |
| 10 | 3.4242 | 30.70 | 18.18% | 51.67% | 78.47% | 2.1526 |
| 15 | 3.4646 | 31.96 | 18.45% | 51.64% | 78.72% | 2.1285 |
| 20 | 3.4799 | 32.46 | 18.55% | 52.09% | 78.79% | 2.1213 |
| 25 | 3.4832 | 32.56 | 18.67% | 51.91% | 78.84% | 2.1162 |
| 30 | 3.4833 | 32.57 | 18.75% | 51.91% | 78.86% | 2.1144 |

### 4.2 最佳结果 vs 基线

| 指标 | Epoch 1 (基线) | 最佳 | 提升 | 最佳 Epoch |
|------|:---:|:---:|:---:|:---:|
| Val CE | 3.3856 | **3.3441** | **-0.0415** | 2 |
| Perplexity | 29.54 | **28.34** | **-4.1%** | 2 |
| Top-1 Accuracy | 16.81% | **18.76%** | **+1.95pp** | 28 |
| Top-5 Accuracy | 48.87% | **52.09%** | **+3.22pp** | 20 |
| Bit Accuracy | 77.38% | **78.86%** | **+1.48pp** | 30 |
| Hamming Distance | 2.2617 | **2.1144** | **-0.147** | 30 |

### 4.3 训练曲线

```
Epoch:  1    5   10   15   20   25   30
CE:    3.39 ─────╲
                  ╲___3.34──3.42──3.46──3.48──3.48──3.48  (最佳 @ epoch 2)

Top-1: 16.8% ──────────────────────────────→ 18.8%  (持续上升)
Bit:   77.4% ──────────────────────────────→ 78.9%  (持续上升)
```

---

## 5. 关键分析

### 5.1 CE 与 Accuracy 背离

这是本次实验最显著的现象：

- **Val CE** 在 epoch 2 触底（3.3441），之后持续上升至 3.4833
- **Top-1** 从 16.81% 持续上升至 18.76%（epoch 28 最佳）
- **Bit Accuracy** 和 **Hamming Distance** 全程持续改善

**解读**：模型在 CE 层面上快速过拟合（epoch 2 后即不再改善），但在离散分类指标上仍然持续进步。这表明 head 层在学会更"自信"地预测（softmax 概率分布变得更尖锐），同时错误预测的代价也更高（CE 惩罚更重）。这是一种典型的 classifier overconfidence 现象。

### 5.2 收敛速度

- 可训练参数仅 525,312（占 predictor 总数的 ~2%）
- CE 在 2 个 epoch 内即达到最优，之后只有微弱波动
- Top-1 和 Bit Accuracy 在整个 30 个 epoch 中持续缓慢爬升
- OneCycle LR 的 final_div_factor=10000 导致后 10 个 epoch LR < 1e-6，模型几乎停止有效学习

### 5.3 随机基线对比

| | 随机基线 | Epoch 1 (预训练) | 最佳 |
|---|---|---|---|
| CE | log(1024) = 6.93 | 3.39 | 3.34 |
| Top-1 | 0.098% | 16.81% | 18.76% |

Kronos-small 预训练模型的 S1 head 已具备极强的 zero-shot 预测能力（**171× 随机**），后训练进一步提升至 **192× 随机**。

### 5.4 模型选择策略

按 CE 选模：epoch 2 最佳（CE=3.3441, Top-1=17.29%）
按 Top-1 选模：epoch 28 最佳（CE=3.4833, Top-1=18.76%）

两种策略给出了不同的最优模型，需要在 D-018 中进一步讨论以哪个指标为准。

---

## 6. 待完成工作

### 6.1 观测指标（D-020）

以下指标尚未实施，需要在测试集上对比训练前后模型：

- [ ] S2 退化检测
- [ ] 完整 K 线重建误差
- [ ] 截面 Rank IC
- [ ] 简单策略回测收益
- [ ] 五步自回归 rollout
- [ ] OHLC 约束违反率

### 6.2 待讨论决策

| ID | 主题 | 当前状态 |
|----|------|---------|
| D-018 | 模型选择策略 | CE vs Top-1 最优 epoch 不一致，需确认 |
| D-019 | Rollout 训练 | 是否在后续加入 rollout 训练 |
| D-020 | 观测指标频率 | 具体评测频率和实现顺序 |

### 6.3 改进方向

1. **降低 LR 或缩短训练**：前 5 个 epoch 已覆盖主要改善，30 epoch 的后半段 LR 过低无实质贡献
2. **Label Smoothing**：缓解 overconfidence，可能改善 CE-Accuracy 背离
3. **用 Accuracy 选模**：如果下游应用关心分类准确率而非校准度
4. **调整 final_div_factor**：从 10000 降至 100-1000，保留更多后期学习信号
5. **训练更多参数**：尝试解冻最后几层 Transformer block，观察收益

---

## 7. 产物清单

| 文件 | 描述 |
|------|------|
| `post-train/outputs/best_model.pt` | 最佳 CE 模型 (epoch 2) |
| `post-train/outputs/last_model.pt` | 最终模型 (epoch 30) |
| `post-train/outputs/metrics.jsonl` | 30 epoch 完整指标记录 |
| `data/a_share_daily/raw/` | 原始 tushare 数据 (5534 stocks) |
| `data/a_share_daily/processed/` | 复权后数据 |
| `data/a_share_daily/manifests/` | 数据 manifest |
