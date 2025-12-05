# 🎯 超参数优化完整指南

## 📊 当前状态分析

### **问题诊断**
1. ✅ **最佳模型表现不错**: R²=0.653, F2=0.875
2. ⚠️ **仍然过拟合**: 损失比例 12.08x (目标 < 5x)
3. ❌ **训练不稳定**: 3次运行中只有1次成功
4. ✅ **RankLoss 有效**: F2-score 从 0.868 提升到 0.875

---

## 🔧 优化方案（按优先级）

### **方案1: 稳定训练（最优先）**

**问题**: 训练不稳定，有些运行完全失败

**解决方案**:
```bash
# 1. 增加学习率预热
# 2. 使用梯度裁剪
# 3. 调整初始化
```

**修改 `main.py`**:
```python
# 降低初始学习率
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')

# 增加批次大小（更稳定）
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
```

**运行**:
```bash
python main.py --model LSTM --lr 0.00005 --batch_size 128 --epoch 50
```

---

### **方案2: 减少过拟合（推荐）**

**目标**: 将损失比例从 12x 降到 < 5x

#### **2.1 增强正则化**

```bash
# 配置A: 更强的 dropout
python main.py --model LSTM \
    --hidden_size 96 \
    --num_layers 3 \
    --dropout 0.5 \
    --lambda_rank 0.1 \
    --epoch 50

# 配置B: 减小模型 + 强 dropout
python main.py --model LSTM \
    --hidden_size 80 \
    --num_layers 3 \
    --dropout 0.5 \
    --lambda_rank 0.1 \
    --epoch 50

# 配置C: 更小的模型
python main.py --model LSTM \
    --hidden_size 64 \
    --num_layers 2 \
    --dropout 0.5 \
    --lambda_rank 0.1 \
    --epoch 50
```

#### **2.2 数据增强（推荐）**

在 `utils/solver.py` 中添加：

```python
# 在 _get_loader 方法中
from torch.utils.data import WeightedRandomSampler

# 对高风险样本进行过采样
high_risk_mask = y_train > -6
sample_weights = np.ones(len(y_train))
sample_weights[high_risk_mask] = 3.0  # 高风险样本权重3倍

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(y_train),
    replacement=True
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=self.args.batch_size, 
    sampler=sampler,  # 使用采样器
    drop_last=False
)
```

#### **2.3 早停策略**

```bash
# 更激进的早停
python main.py --model LSTM \
    --patience 5 \
    --lambda_rank 0.1 \
    --epoch 200
```

---

### **方案3: 优化 RankLoss 参数**

**当前**: `lambda_rank=0.1` 已经有效

**尝试不同权重**:

```bash
# 更强的排序约束
python main.py --model LSTM --lambda_rank 0.15 --epoch 50

# 更弱的排序约束
python main.py --model LSTM --lambda_rank 0.05 --epoch 50

# 添加 margin
python main.py --model LSTM --lambda_rank 0.1 --rank_margin 0.5 --epoch 50
```

---

### **方案4: 尝试 SupConRegressionLoss**

**要求**: 模型已支持 `return_feat=True`

```bash
# LDS + RankLoss + SupCR
python main.py --model LSTM \
    --lambda_rank 0.1 \
    --use_supcr \
    --lambda_sup 0.3 \
    --sup_temp 0.1 \
    --sup_sigma 2.0 \
    --epoch 50
```

**注意**: SupCR 会增加训练时间和内存消耗

---

### **方案5: 学习率调度**

修改 `utils/solver.py`:

```python
# 当前
self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)

# 改为更激进的调度
self.scheduler = ReduceLROnPlateau(
    self.optimizer, 
    mode='min', 
    factor=0.3,      # 从 0.5 改为 0.3
    patience=5,      # 从 3 改为 5
    min_lr=1e-6      # 添加最小学习率
)
```

---

### **方案6: 尝试其他模型**

```bash
# Transformer（可能更稳定）
python main.py --model TRANSFORMER --lambda_rank 0.1 --epoch 50

# CDEA（专为卫星碰撞设计）
python main.py --model CDEA --lambda_rank 0.1 --epoch 50

# BayesLSTM（带不确定性估计）
python main.py --model BAYES_LSTM --lambda_rank 0.1 --epoch 50
```

---

## 📋 推荐的实验计划

### **第1周: 稳定性优化**

```bash
# Day 1-2: 降低学习率
python main.py --model LSTM --lr 0.00005 --batch_size 128 --lambda_rank 0.1 --epoch 50

# Day 3-4: 增强正则化
python main.py --model LSTM --dropout 0.5 --lambda_rank 0.1 --epoch 50

# Day 5-7: 调整模型大小
python main.py --model LSTM --hidden_size 80 --num_layers 3 --dropout 0.5 --lambda_rank 0.1 --epoch 50
```

### **第2周: 高级优化**

```bash
# Day 1-2: 优化 RankLoss
python main.py --model LSTM --lambda_rank 0.15 --rank_margin 0.5 --epoch 50

# Day 3-5: 添加 SupCR
python main.py --model LSTM --lambda_rank 0.1 --use_supcr --lambda_sup 0.3 --epoch 50

# Day 6-7: 尝试其他模型
python main.py --model CDEA --lambda_rank 0.1 --epoch 50
```

---

## 🎯 目标指标

| 指标 | 当前 | 目标 | 优秀 |
|------|------|------|------|
| **R²** | 0.653 | > 0.70 | > 0.80 |
| **F2-score** | 0.875 | > 0.88 | > 0.92 |
| **损失比例** | 12.08x | < 5x | < 3x |
| **训练稳定性** | 33% | 100% | 100% |

---

## 💡 调试技巧

### **1. 监控梯度**

在 `utils/solver.py` 的训练循环中添加：

```python
# 在 loss.backward() 后
if e % 10 == 0:  # 每10个epoch检查一次
    for name, param in self.model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 10:  # 梯度爆炸
                print(f"⚠️  梯度过大: {name} = {grad_norm:.2f}")
            elif grad_norm < 1e-5:  # 梯度消失
                print(f"⚠️  梯度过小: {name} = {grad_norm:.2e}")
```

### **2. 添加梯度裁剪**

```python
# 在 loss.backward() 后, optimizer.step() 前
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

### **3. 可视化权重分布**

```python
import matplotlib.pyplot as plt

# 在 _get_loader 后
plt.figure(figsize=(10, 5))
plt.hist(y_train, bins=50, alpha=0.5, label='Training Labels')
plt.hist(y_val, bins=50, alpha=0.5, label='Validation Labels')
plt.xlabel('Risk Value')
plt.ylabel('Count')
plt.legend()
plt.savefig('label_distribution.png')
print("✅ 标签分布已保存: label_distribution.png")
```

---

## 🚀 立即开始

**推荐配置（最稳定）**:

```bash
cd C:\Users\tmpzh\Desktop\kelvin\Collision_kelvin

python main.py \
    --model LSTM \
    --hidden_size 80 \
    --num_layers 3 \
    --dropout 0.5 \
    --batch_size 128 \
    --lr 0.00005 \
    --lambda_rank 0.1 \
    --patience 10 \
    --epoch 100
```

**运行后分析**:

```bash
python analyze_results.py
```

---

## 📊 超参数速查表

| 参数 | 当前值 | 推荐范围 | 说明 |
|------|--------|---------|------|
| `hidden_size` | 96 | 64-96 | 越小越不容易过拟合 |
| `num_layers` | 3 | 2-3 | 2层更稳定 |
| `dropout` | 0.4 | 0.4-0.6 | 越大正则化越强 |
| `batch_size` | 64 | 64-128 | 越大越稳定 |
| `lr` | 0.0001 | 0.00005-0.0001 | 越小越稳定 |
| `lambda_rank` | 0.1 | 0.05-0.2 | 当前值不错 |
| `patience` | 10 | 5-15 | 根据收敛速度调整 |

---

**记住**: 机器学习是实验科学，需要多次尝试才能找到最佳配置！
