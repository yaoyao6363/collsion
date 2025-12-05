# 🎯 训练配置推荐

## 📊 问题分析

你的模型出现了**过拟合**问题：
- 训练损失：3.06
- 验证损失：39.46（是训练损失的 13倍）
- R²: 0.548（中等）
- F2: 0.868（还可以）

---

## 🔧 推荐配置

### **配置1: 平衡模型（推荐）**

```bash
python main.py \
    --model LSTM \
    --hidden_size 96 \
    --num_layers 3 \
    --dropout 0.4 \
    --batch_size 64 \
    --lr 0.0001 \
    --patience 10 \
    --epoch 200
```

**预期效果**：
- 参数量：~200K
- Train/Valid Loss 比例：< 5
- R² > 0.6

---

### **配置2: 使用 RankLoss 优化 F2-score**

```bash
python main.py \
    --model LSTM \
    --hidden_size 96 \
    --num_layers 3 \
    --dropout 0.4 \
    --batch_size 64 \
    --lr 0.0001 \
    --lambda_rank 0.1 \
    --patience 10 \
    --epoch 200
```

**预期效果**：
- F2-score 提升 5-10%
- 高风险样本召回率提升

---

### **配置3: 更强的正则化**

```bash
python main.py \
    --model LSTM \
    --hidden_size 96 \
    --num_layers 3 \
    --dropout 0.5 \
    --batch_size 64 \
    --lr 0.00005 \
    --patience 15 \
    --epoch 200
```

**适用场景**：如果配置1仍然过拟合

---

### **配置4: 数据增强（使用更多数据）**

```bash
python main.py \
    --model LSTM \
    --hidden_size 128 \
    --num_layers 4 \
    --dropout 0.3 \
    --batch_size 128 \
    --lr 0.0001 \
    --patience 10 \
    --epoch 200
```

**适用场景**：如果你有更多训练数据

---

## 📈 超参数调优指南

### **1. 模型容量（hidden_size, num_layers）**

| 配置 | hidden_size | num_layers | 参数量 | 适用场景 |
|------|-------------|------------|--------|---------|
| 小 | 64 | 2 | ~50K | 数据少，防止过拟合 |
| **中（推荐）** | **96** | **3** | **~200K** | **平衡性能和泛化** |
| 大 | 128 | 4 | ~500K | 数据多，追求性能 |

### **2. Dropout**

| 值 | 效果 | 适用场景 |
|----|------|---------|
| 0.2 | 弱正则化 | 模型欠拟合 |
| 0.3 | 中等正则化 | 标准配置 |
| **0.4（推荐）** | **强正则化** | **防止过拟合** |
| 0.5 | 很强正则化 | 严重过拟合 |

### **3. Learning Rate**

| 值 | 效果 | 适用场景 |
|----|------|---------|
| 0.001 | 快速收敛 | 初期探索 |
| **0.0001（推荐）** | **稳定训练** | **标准配置** |
| 0.00005 | 慢速精细 | 微调阶段 |

### **4. Batch Size**

| 值 | 效果 | 内存 | 泛化能力 |
|----|------|------|---------|
| 32 | 噪声大，正则化强 | 低 | 好 |
| **64（推荐）** | **平衡** | **中** | **好** |
| 128 | 稳定，正则化弱 | 高 | 中 |

---

## 🎯 训练策略

### **阶段1: 快速验证（5 epochs）**

```bash
python main.py --model LSTM --epoch 5
```

**目标**：验证代码正常运行，观察初步趋势

---

### **阶段2: 基线训练（50 epochs）**

```bash
python main.py --model LSTM --epoch 50
```

**目标**：
- 建立基线性能
- 观察过拟合程度
- 决定是否需要调整超参数

---

### **阶段3: 完整训练（200 epochs + early stopping）**

```bash
python main.py --model LSTM --epoch 200 --patience 10
```

**目标**：
- 获得最佳性能
- Early stopping 防止过拟合

---

### **阶段4: 使用高级损失函数**

```bash
# LDS + RankLoss
python main.py --model LSTM --lambda_rank 0.1 --epoch 200

# LDS + RankLoss + SupCR
python main.py --model LSTM --lambda_rank 0.1 --use_supcr --lambda_sup 0.5 --epoch 200
```

**目标**：
- 优化 F2-score
- 提升高风险样本性能

---

## 🔍 监控指标

### **健康的训练应该满足**：

✅ **Train/Valid Loss 比例 < 5**
- 当前：39.46 / 3.06 = 12.9（过拟合）
- 目标：< 5

✅ **R² > 0.7**
- 当前：0.548
- 目标：> 0.7

✅ **F2-score > 0.85**
- 当前：0.868（还可以）
- 目标：> 0.90

✅ **验证损失持续下降**
- 如果验证损失不再下降，说明模型已收敛

---

## 💡 调试技巧

### **1. 可视化训练曲线**

在 `solver.py` 中添加：

```python
import matplotlib.pyplot as plt

# 在 train() 方法结束后
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.legend()
plt.savefig(f'{self.model_path}/loss_curve.png')
```

### **2. 监控梯度**

```python
# 在训练循环中
for name, param in self.model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.4f}")
```

### **3. 检查数据分布**

```python
# 在 _get_loader() 后
print(f"Train: min={y_train.min():.2f}, max={y_train.max():.2f}, mean={y_train.mean():.2f}")
print(f"Valid: min={y_val.min():.2f}, max={y_val.max():.2f}, mean={y_val.mean():.2f}")
```

---

## 🚀 立即开始

```bash
# 1. 使用推荐配置
cd C:\Users\tmpzh\Desktop\kelvin\Collision_kelvin
python main.py --model LSTM --epoch 50

# 2. 观察结果
cat results/checkpoint/LSTM_43_0/result.txt

# 3. 如果仍然过拟合，增加 dropout
python main.py --model LSTM --dropout 0.5 --epoch 50

# 4. 如果欠拟合，增加模型容量
python main.py --model LSTM --hidden_size 128 --num_layers 4 --epoch 50
```

---

**记住**：机器学习是迭代过程，需要多次实验才能找到最佳配置！
