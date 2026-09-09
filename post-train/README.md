# Kronos S1 Post-training

本目录用于全 A 股日线上的粗 token（`s1`）next-token 后训练。

当前阶段只建立工程边界和决策入口，不实现训练细节。后续每确认一项设计，
再在对应模块中补充实现与测试。

## 目录结构

```text
post-train/
├── configs/                     # 实验配置；未确认项保留 draft 标记
├── docs/                        # 方案决策、实验约束和变更记录
├── post_train/                  # 可复用 Python 包
│   ├── data/                    # 数据读取、时间切分和 next-token 样本
│   ├── evaluation/              # s1 token 与完整预测评测
│   ├── modeling/                # s1 训练入口和参数冻结策略
│   ├── training/                # 优化器、训练循环和 checkpoint
│   └── utils/                   # 随机种子、日志和实验元数据
├── scripts/                     # 命令行入口
├── tests/                       # 防泄漏、跨股票和样本对齐测试
└── outputs/                     # 本地实验产物，不提交模型和日志
```

## 设计原则

1. 数据按股票分组，任何窗口不得跨越 `ts_code`。
2. Train/Validation/Test 按 target 日期切分，禁止随机切分重叠窗口。
3. Tokenizer revision、数据哈希、切分日期和代码版本必须写入实验元数据。
4. 训练逻辑与评测逻辑分离，官方 zero-shot 始终作为固定基线。
5. 每个尚未确定的关键设计记录在 `docs/DECISIONS.md`。
6. 本实验只优化 s1 next-token；s2、完整 K 线和下游表现仅作为影响观测。
7. 原始行情下载、数据处理和模型训练相互隔离；训练只能读取已冻结的数据快照。

## 计划中的命令

```bash
python post-train/scripts/train_s1_next_token.py \
  --config post-train/configs/s1_next_token_baseline.yaml

python post-train/scripts/evaluate_s1_next_token.py \
  --config post-train/configs/s1_next_token_baseline.yaml
```

命令目前仅为占位入口，需在方案确认后逐步实现。
