# 🚀 CDM数据生成与风险预测 - 快速开始指南

## 📋 已为你创建的文件

我已经为你创建了完整的CDM数据生成和风险预测工作流程。以下是新增的文件：

### 核心文件 ⭐

1. **CDM_GENERATION_AND_RISK_PREDICTION_GUIDE.md** - 完整操作手册
   - 详细的步骤说明
   - 代码示例
   - 参数解释
   - 常见问题解答

2. **generate_cdm_data.py** - CDM数据生成脚本
   - 从Kelvins数据转换为CDM格式
   - 支持合成数据生成（实验性）
   - 灵活的输出格式（CSV/KVN）

3. **preprocess_cdm_data.py** - 数据预处理脚本
   - 特征选择与工程
   - 缺失值处理
   - 异常值移除
   - 特征标准化

4. **README_CDM_WORKFLOW.md** - 工作流程说明
   - 项目结构
   - 快速开始
   - 详细文档索引

5. **quick_start_cdm.bat** - 一键启动脚本（Windows）
   - 自动执行完整流程
   - 错误检测
   - 进度提示

6. **test_cdm_workflow.py** - 工作流程测试
   - 验证环境配置
   - 测试各个模块
   - 端到端测试

---

## 🎯 三种使用方式

### 方式1: 一键运行（最简单）⚡

```bash
# Windows用户
quick_start_cdm.bat

# 这将自动完成:
# 1. 检查依赖
# 2. 生成CDM数据
# 3. 预处理数据
# 4. 训练模型
# 5. 测试模型
```

### 方式2: 分步执行（推荐学习）📚

```bash
# 步骤1: 测试环境
python test_cdm_workflow.py

# 步骤2: 生成CDM数据（从Kelvins数据）
python generate_cdm_data.py --mode kelvins --input_csv ./dataset/train_data.csv --num_events 1000

# 步骤3: 预处理数据
python preprocess_cdm_data.py --input_csv ./processed_cdms/kelvins_cdms.csv --output_csv ./dataset/processed_train_data.csv

# 步骤4: 训练模型
python main.py --model TRANSFORMER --epoch 150 --batch_size 256 --lr 0.0005

# 步骤5: 测试模型
python main.py --model TRANSFORMER --only_test
```

### 方式3: 自定义流程（高级用户）🔧

参考 `CDM_GENERATION_AND_RISK_PREDICTION_GUIDE.md` 中的详细说明，根据你的需求自定义每个步骤。

---

## 📊 工作流程图

```
原始数据 (train_data.csv)
    ↓
[generate_cdm_data.py]
    ↓
CDM格式数据 (kelvins_cdms.csv)
    ↓
[preprocess_cdm_data.py]
    ↓
预处理数据 (processed_train_data.csv)
    ↓
[main.py --model TRANSFORMER]
    ↓
训练好的模型 (checkpoint/)
    ↓
[main.py --only_test]
    ↓
风险预测结果 (results/)
```

---

## 🔍 第一次使用？按这个顺序操作

### Step 0: 环境检查

```bash
# 测试环境是否正确配置
python test_cdm_workflow.py
```

如果所有测试通过，继续下一步。如果有失败，根据提示修复问题。

### Step 1: 了解你的数据

```bash
# 查看原始数据
python data_describe.py
```

### Step 2: 生成CDM数据

```bash
# 从Kelvins数据生成CDM（推荐从这里开始）
python generate_cdm_data.py --mode kelvins --input_csv ./dataset/train_data.csv --output_dir ./processed_cdms/ --num_events 1000
```

**输出**: `./processed_cdms/kelvins_cdms.csv`

### Step 3: 预处理数据

```bash
# 预处理CDM数据
python preprocess_cdm_data.py --input_csv ./processed_cdms/kelvins_cdms.csv --output_csv ./dataset/processed_train_data.csv
```

**输出**: 
- `./dataset/processed_train_data.csv` - 预处理后的数据
- `./scaler.pkl` - 标准化器
- `./preprocessing_config.json` - 配置文件

### Step 4: 训练模型

```bash
# 使用Transformer模型训练
python main.py --model TRANSFORMER --epoch 150 --batch_size 256 --lr 0.0005 --data_path ./dataset/ --dataset_train processed_train_data.csv
```

**输出**: `./results/checkpoint/` - 模型权重

### Step 5: 评估模型

```bash
# 测试模型性能
python main.py --model TRANSFORMER --only_test --data_path ./dataset/ --dataset_test processed_train_data.csv
```

**输出**: 控制台打印评估指标

---

## 📁 重要文件位置

```
Collision_kelvin/
├── dataset/
│   ├── train_data.csv              # 原始训练数据
│   └── processed_train_data.csv    # 预处理后的数据 [生成]
│
├── processed_cdms/
│   └── kelvins_cdms.csv            # CDM格式数据 [生成]
│
├── results/
│   ├── checkpoint/                 # 模型权重 [生成]
│   └── logs/                       # 训练日志 [生成]
│
├── scaler.pkl                      # 标准化器 [生成]
├── preprocessing_config.json       # 预处理配置 [生成]
│
└── 文档/
    ├── CDM_GENERATION_AND_RISK_PREDICTION_GUIDE.md  # 完整指南
    ├── README_CDM_WORKFLOW.md                       # 工作流程说明
    └── QUICK_START_SUMMARY.md                       # 本文件
```

---

## 🎓 学习路径建议

### 初学者路径

1. **阅读**: `README_CDM_WORKFLOW.md` - 了解整体架构
2. **运行**: `test_cdm_workflow.py` - 验证环境
3. **执行**: `quick_start_cdm.bat` - 体验完整流程
4. **学习**: `CDM_GENERATION_AND_RISK_PREDICTION_GUIDE.md` - 深入理解

### 进阶用户路径

1. **自定义特征工程**: 修改 `preprocess_cdm_data.py` 中的 `engineer_features()`
2. **调整模型架构**: 修改 `models/` 中的模型定义
3. **优化超参数**: 使用网格搜索或贝叶斯优化
4. **集成多个模型**: 训练多个模型并集成预测

### 研究者路径

1. **生成合成数据**: 使用 `--mode synthetic` 生成大量训练数据
2. **实验不同的概率模型**: 修改 `kessler/model.py` 中的 `Conjunction` 类
3. **开发新的评估指标**: 在 `utils/metrics.py` 中添加自定义指标
4. **发表论文**: 引用Kessler相关论文

---

## 🔧 常用命令速查

### 数据生成

```bash
# 从Kelvins数据生成
python generate_cdm_data.py --mode kelvins --input_csv ./dataset/train_data.csv

# 生成合成数据（实验性）
python generate_cdm_data.py --mode synthetic --num_events 500

# 同时保存CSV和KVN格式
python generate_cdm_data.py --mode kelvins --save_format both
```

### 数据预处理

```bash
# 标准预处理
python preprocess_cdm_data.py --input_csv ./processed_cdms/kelvins_cdms.csv

# 使用鲁棒标准化（对异常值更稳健）
python preprocess_cdm_data.py --normalization_method robust

# 不进行特征工程
python preprocess_cdm_data.py --no_feature_engineering
```

### 模型训练

```bash
# LSTM模型
python main.py --model LSTM --hidden_size 64 --num_layers 3 --dropout 0.5

# Transformer模型
python main.py --model TRANSFORMER --d_model 128 --nhead 4 --num_layers_tf 3

# CDEA模型
python main.py --model CDEA --cdea_layers 10 --cdea_heads_row 4

# Bayesian LSTM
python main.py --model BAYES_LSTM --bayes_hidden_size 256
```

### 模型评估

```bash
# 仅测试
python main.py --model TRANSFORMER --only_test

# 多次运行取平均
python main.py --model TRANSFORMER --itr 5
```

---

## 💡 提示与技巧

### 提示1: 从小数据集开始

```bash
# 先用少量数据测试流程
python generate_cdm_data.py --mode kelvins --num_events 100
python main.py --model TRANSFORMER --epoch 10
```

### 提示2: 使用GPU加速

```bash
# 确保CUDA可用
python -c "import torch; print(torch.cuda.is_available())"

# 训练时会自动使用GPU（如果可用）
python main.py --model TRANSFORMER --use_gpu True --device 0
```

### 提示3: 监控训练过程

```bash
# 在另一个终端运行
python monitor_training.py
```

### 提示4: 保存中间结果

所有脚本都会自动保存中间结果，如果某步失败，可以从上一步的输出继续。

### 提示5: 查看详细日志

```bash
# 训练日志
cat ./results/logs/training.log

# 或使用Python查看
python analyze_results.py
```

---

## ❓ 遇到问题？

### 问题1: Kessler库导入失败

```bash
cd kessler
pip install -e .
```

### 问题2: 生成的CDM数据为空

- 检查输入CSV是否包含必要的列（event_id, time_to_tca等）
- 尝试增加 `--num_events` 参数
- 查看控制台的错误信息

### 问题3: 训练时显存不足

```bash
# 减小batch size
python main.py --model TRANSFORMER --batch_size 128

# 或减小模型大小
python main.py --model TRANSFORMER --d_model 64 --num_layers_tf 2
```

### 问题4: 预测结果不理想

1. **增加训练数据**: 生成更多CDM事件
2. **特征工程**: 添加更多物理意义的特征
3. **调整超参数**: 使用网格搜索找到最佳参数
4. **模型集成**: 组合多个模型的预测

### 更多帮助

- 查看完整指南: `CDM_GENERATION_AND_RISK_PREDICTION_GUIDE.md`
- 运行测试: `python test_cdm_workflow.py`
- 查看示例: `example_usage.py`

---

## 📈 预期结果

完成整个流程后，你应该得到:

1. ✅ 预处理的CDM数据集
2. ✅ 训练好的风险预测模型
3. ✅ 模型评估指标（MSE, MAE, R²等）
4. ✅ 可用于新数据预测的模型权重

**典型性能指标**（取决于数据质量和模型选择）:
- MSE: 0.001 - 0.01
- MAE: 0.01 - 0.05
- R²: 0.7 - 0.9
- Pearson相关系数: 0.8 - 0.95

---

## 🎉 下一步

完成基础流程后，你可以:

1. **优化模型**: 调整超参数，尝试不同的模型架构
2. **增强数据**: 生成更多合成数据，进行数据增强
3. **部署模型**: 创建API服务，实时预测碰撞风险
4. **可视化结果**: 绘制风险演化曲线，不确定性分析
5. **发表成果**: 整理实验结果，撰写论文或报告

---

## 📚 相关资源

- **Kessler文档**: https://kesslerlib.github.io/kessler/
- **CCSDS CDM标准**: https://public.ccsds.org/Pubs/508x0b1e2c1.pdf
- **Kelvins竞赛**: https://kelvins.esa.int/satellite-collision-avoidance-challenge/
- **项目GitHub**: (你的项目地址)

---

**祝你的碰撞风险预测项目成功！** 🛰️✨

如有任何问题，请查看完整指南或运行测试脚本诊断问题。
