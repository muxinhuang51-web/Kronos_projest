# S1 Post-training Decisions

这里记录已经确认的边界和仍需讨论的实现细节。每个决定应包含原因、备选方案、
验证方式和确认日期。

## 已确认

| ID | 主题 | 决定 |
|---|---|---|
| D-001 | 数据域 | 使用全 A 股日线数据 |
| D-002 | 训练目标 | 强化粗 token `s1` 的 next-token 预测 |
| D-003 | 工程位置 | 独立放在 `post-train/`，不直接改写原训练链路 |
| D-004 | 样本定义 | 使用方案 A：一个 context 只监督最后一个 next-token `s1` |
| D-005 | 优化边界 | 训练与选模只优化 s1；其他模型和下游变化仅作为实验影响观测 |
| D-006 | 归一化 | OHLCVA 六维分别使用 context-only mean/std，target 复用同一统计量，`ddof=0`、`eps=1e-5`、裁剪到 `[-5, 5]` |
| D-007 | 数据跨度 | 使用 2016-01-01 至实验开始前最新完整交易日的全 A 股日线 |
| D-008 | 历史股票池 | 使用 point-in-time 动态股票池，包含上市、暂停上市和已退市证券，不按当前状态回筛历史 |
| D-009 | 复权方式 | OHLC 使用 `raw price × 当日 adj_factor` 的后复权口径；volume 与 amount 保持原始口径 |
| D-010 | 停牌与日历 | 不前向填充；target 必须是 context 末日之后的下一市场交易日；context 内部允许保留历史日期缺口 |
| D-011 | 数据快照 | 下载、校验、处理后冻结 Parquet 数据快照和 manifest，再运行全部对照实验 |
| D-012 | 时间切分 | Train 2016–2023，Validation 2024，Test 2025 至最新完整交易日，归属只由 target 日期决定 |
| D-013 | Tokenizer | 整个实验永久冻结官方 tokenizer，不联合训练、不替换 checkpoint、不改变量化器或 token 位宽 |
| D-014 | Predictor 参数范围 | 采用 Head-only，只训练 `head.proj_s1` 的 weight 和 bias，预期可训练参数 525,312，其余 predictor 参数全部冻结 |
| D-015 | 数据采样 | 从有效 `(ts_code, target_date)` 窗口均匀有放回抽样，与官方 `finetune/dataset.py` 一致；每 epoch 抽 100,000 次训练样本，验证使用 seed 100 固定生成的 20,000 次抽样序列；允许重复，不做股票、日期、token、regime 或困难样本重加权 |
| D-016 | 训练损失 | 只对最后位置 s1 使用标准 Cross Entropy；类别等权、`reduction=mean`、无 label smoothing、无辅助损失 |
| D-017 | 优化设置 | 沿用官方 `finetune/` predictor：FP32、AdamW、LR `4e-5`、betas `(0.9, 0.95)`、weight decay `0.1`、每 GPU batch 50、无梯度累积、30 epoch、OneCycle cosine（3% warmup、div factor 10、final div factor 10000）、梯度裁剪 3、seed 100、2 workers；每 epoch 验证一次并每 100 step 记录一次 |

## 实验影响观测

下列内容不进入训练损失，也不参与 checkpoint 选择，但需要对比官方基线、
训练前模型和训练后模型，以观察 s1 专项训练产生的副作用或外溢影响：

- s2 是否退化；
- 完整 K 线误差；
- Rank IC 和策略收益；
- 多步交易价值；
- OHLC 合法性。

归一化阶段额外记录近零标准差比例与 target 裁剪率，它们只用于数据诊断。

## 待讨论

| ID | 主题 | 当前草案 | 需要确认的问题 |
|---|---|---|---|
| D-018 | 模型选择 | 最低验证 s1 CE | 在 s1 指标内部如何处理 CE、Top-1 等关系 |
| D-019 | Rollout | 一步训练，额外评测五步生成 | 是否在后期加入 rollout 训练 |
| D-020 | 影响观测 | 记录 s2、K 线、Rank IC、策略、多步和 OHLC 变化 | 具体评测频率与实现顺序 |

## 决策模板

```text
ID:
日期:
问题:
最终决定:
备选方案:
选择原因:
验证方法:
影响文件:
```
