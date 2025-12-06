# CDM数据生成与风险预测工作流程

## 快速开始 ⚡

### 方式1: 一键运行 (推荐)

```bash
# Windows
quick_start_cdm.bat

# Linux/Mac
chmod +x quick_start_cdm.sh
./quick_start_cdm.sh
```

### 方式2: 分步执行

```bash
# 步骤1: 从Kelvins数据生成CDM
python generate_cdm_data.py --mode kelvins --input_csv ./dataset/train_data.csv

# 步骤2: 预处理数据
python preprocess_cdm_data.py --input_csv ./processed_cdms/kelvins_cdms.csv

# 步骤3: 训练模型
python main.py --model TRANSFORMER --epoch 150 --batch_size 256

# 步骤4: 测试模型
python main.py --model TRANSFORMER --only_test
```

---

## 详细文档 📚

完整操作手册请查看: **[CDM_GENERATION_AND_RISK_PREDICTION_GUIDE.md](./CDM_GENERATION_AND_RISK_PREDICTION_GUIDE.md)**

---

## 项目结构 📁

```
Collision_kelvin/
├── kessler/                          # Kessler库源码
│   ├── kessler/
│   │   ├── cdm.py                   # CDM数据结构
│   │   ├── event.py                 # 事件管理
│   │   ├── data.py                  # 数据加载
│   │   ├── model.py                 # 概率模型
│   │   └── nn.py                    # LSTM预测器
│   └── docs/                        # Kessler文档
│
├── dataset/                          # 数据集目录
│   ├── train_data.csv               # 原始训练数据
│   ├── test_data.csv                # 原始测试数据
│   └── processed_train_data.csv     # 预处理后的数据
│
├── models/                           # 模型定义
│   ├── lstm.py                      # LSTM模型
│   ├── transformer.py               # Transformer模型
│   ├── cdea.py                      # CDEA模型
│   └── bayes_lstm.py                # Bayesian LSTM
│
├── utils/                            # 工具函数
│   ├── dataset.py                   # 数据集类
│   ├── solver.py                    # 训练器
│   └── metrics.py                   # 评估指标
│
├── results/                          # 结果输出
│   ├── checkpoint/                  # 模型检查点
│   └── logs/                        # 训练日志
│
├── generate_cdm_data.py             # CDM数据生成脚本 ⭐
├── preprocess_cdm_data.py           # 数据预处理脚本 ⭐
├── main.py                          # 主训练脚本
├── quick_start_cdm.bat              # 快速启动脚本 ⭐
└── CDM_GENERATION_AND_RISK_PREDICTION_GUIDE.md  # 完整指南 ⭐
```

---

## 核心脚本说明 🔧

### 1. generate_cdm_data.py

**功能**: 生成CDM数据

**使用方法**:

```bash
# 从Kelvins数据生成 (推荐)
python generate_cdm_data.py --mode kelvins --input_csv ./dataset/train_data.csv --num_events 1000

# 生成合成数据 (实验性)
python generate_cdm_data.py --mode synthetic --num_events 500 --save_format csv
```

**参数说明**:
- `--mode`: 生成模式 (`kelvins` 或 `synthetic`)
- `--num_events`: 事件数量
- `--input_csv`: Kelvins数据路径
- `--output_dir`: 输出目录
- `--save_format`: 保存格式 (`csv`, `kvn`, `both`)

### 2. preprocess_cdm_data.py

**功能**: 数据预处理与特征工程

**使用方法**:

```bash
python preprocess_cdm_data.py \
    --input_csv ./processed_cdms/kelvins_cdms.csv \
    --output_csv ./dataset/processed_train_data.csv \
    --normalization_method standard \
    --outlier_method iqr
```

**参数说明**:
- `--input_csv`: 输入CSV文件
- `--output_csv`: 输出CSV文件
- `--missing_strategy`: 缺失值处理 (`drop`, `mean`, `median`)
- `--outlier_method`: 异常值检测 (`iqr`, `zscore`)
- `--normalization_method`: 标准化方法 (`standard`, `robust`, `minmax`)
- `--no_feature_engineering`: 禁用特征工程

**生成的文件**:
- `processed_data.csv`: 预处理后的数据
- `scaler.pkl`: 标准化器 (用于预测时)
- `preprocessing_config.json`: 预处理配置

### 3. main.py

**功能**: 模型训练与测试

**使用方法**:

```bash
# 训练TRANSFORMER模型
python main.py --model TRANSFORMER --epoch 150 --batch_size 256 --lr 0.0005

# 训练LSTM模型
python main.py --model LSTM --hidden_size 64 --num_layers 3 --dropout 0.5

# 仅测试
python main.py --model TRANSFORMER --only_test
```

**支持的模型**:
- `LSTM`: 长短期记忆网络
- `TRANSFORMER`: Transformer编码器
- `CDEA`: 交叉维度增强注意力
- `BAYES_LSTM`: 贝叶斯LSTM

---

## 工作流程图 🔄

```
┌─────────────────────────────────────────────────────────────┐
│                     原始数据                                 │
│              (Kelvins CSV / 合成数据)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Step 1: CDM数据生成                             │
│         (generate_cdm_data.py)                               │
│  • 从Kelvins数据转换为CDM格式                                │
│  • 或使用Conjunction模型生成合成数据                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Step 2: 数据预处理                              │
│         (preprocess_cdm_data.py)                             │
│  • 特征选择与工程                                            │
│  • 缺失值处理                                                │
│  • 异常值移除                                                │
│  • 特征标准化                                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Step 3: 模型训练                                │
│              (main.py)                                       │
│  • 数据加载与划分                                            │
│  • 模型初始化                                                │
│  • 训练循环 (Early Stopping)                                 │
│  • 模型保存                                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Step 4: 模型评估                                │
│              (main.py --only_test)                           │
│  • 加载最佳模型                                              │
│  • 测试集评估                                                │
│  • 指标计算 (MSE, MAE, R², etc.)                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Step 5: 风险预测                                │
│  • 加载训练好的模型                                          │
│  • 对新CDM数据进行预测                                       │
│  • 输出碰撞风险评分                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 数据格式说明 📊

### 输入数据格式 (Kelvins CSV)

```csv
event_id,time_to_tca,miss_distance,relative_speed,...
0,6.5,1234.56,7890.12,...
0,5.8,1100.23,7850.45,...
1,7.0,2345.67,8123.89,...
```

### CDM数据格式

包含以下关键字段:
- **时间信息**: `CREATION_DATE`, `TCA`, `__DAYS_TO_TCA`
- **相对运动**: `MISS_DISTANCE`, `RELATIVE_SPEED`, `RELATIVE_POSITION_*`, `RELATIVE_VELOCITY_*`
- **物体状态**: `OBJECT1_X/Y/Z`, `OBJECT1_X_DOT/Y_DOT/Z_DOT`
- **不确定性**: `OBJECT1_CR_R`, `OBJECT1_CT_T`, `OBJECT1_CN_N`, etc.
- **目标变量**: `COLLISION_PROBABILITY` 或 `risk`

### 预处理后数据格式

- 所有特征已标准化 (均值0, 标准差1)
- 添加了工程特征 (相对距离、不确定性度量等)
- 按`event_id`分组，每个事件包含多个时间步的CDM

---

## 模型性能指标 📈

训练完成后，会在 `./results/` 目录生成:

1. **checkpoint/**: 模型权重文件
2. **logs/**: 训练日志
3. **metrics CSV**: 包含以下指标
   - MSE (均方误差)
   - RMSE (均方根误差)
   - MAE (平均绝对误差)
   - R² (决定系数)
   - Pearson相关系数

---

## 常见问题 ❓

### Q1: Kessler库安装失败?

```bash
# 尝试本地安装
cd kessler
pip install -e .

# 或直接安装依赖
pip install pyro-ppl dsgp4 skyfield torch pandas numpy matplotlib
```

### Q2: 生成的CDM数据为空?

- 检查输入CSV文件是否包含必要的列
- 尝试增加 `--num_events` 参数
- 查看日志中的错误信息

### Q3: 训练时显存不足?

```bash
# 减小batch size
python main.py --model TRANSFORMER --batch_size 128

# 或减小模型大小
python main.py --model TRANSFORMER --d_model 64 --num_layers_tf 2
```

### Q4: 如何使用自己的数据?

1. 将数据转换为Kelvins CSV格式
2. 运行 `generate_cdm_data.py --mode kelvins --input_csv your_data.csv`
3. 按正常流程预处理和训练

---

## 进阶使用 🚀

### 自定义特征工程

编辑 `preprocess_cdm_data.py` 中的 `engineer_features()` 函数:

```python
def engineer_features(df):
    # 添加你的自定义特征
    df['custom_feature'] = df['MISS_DISTANCE'] / df['RELATIVE_SPEED']
    return df, ['custom_feature']
```

### 模型集成

```python
# 训练多个模型
python main.py --model LSTM --itr 5
python main.py --model TRANSFORMER --itr 5

# 使用集成预测
predictions = (lstm_pred + transformer_pred) / 2
```

### 超参数调优

使用 `optuna` 或网格搜索:

```python
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
    # ... 训练并返回验证损失
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

---

## 参考资源 📚

- **Kessler官方文档**: https://kesslerlib.github.io/kessler/
- **CCSDS CDM标准**: https://public.ccsds.org/Pubs/508x0b1e2c1.pdf
- **Kelvins竞赛**: https://kelvins.esa.int/satellite-collision-avoidance-challenge/
- **相关论文**: 见 `kessler/README.md`

---

## 贡献与支持 🤝

如有问题或建议，欢迎:
- 提交 GitHub Issue
- 发送邮件至项目维护者
- 查看完整指南: `CDM_GENERATION_AND_RISK_PREDICTION_GUIDE.md`

---

**祝你的碰撞风险预测项目成功！** 🛰️✨
