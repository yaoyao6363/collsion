# 🎯 多任务学习（Multi-Task Learning）实施总结

## 📊 概述

成功实施了**多任务学习**，让 Transformer 模型同时预测：
1. **Risk**（碰撞风险）- 主任务
2. **Miss Distance**（最近接距离）- 辅助任务（物理约束）

---

## ✅ 已完成的修改

### **Step 1: 修改 `models/risk_transformer.py`** ✅

#### **添加两个预测头**
```python
# 主分支：预测 Log Risk
self.head_risk = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.ReLU(),
    nn.Linear(d_model // 2, 1)
)

# 物理分支：预测 Log Miss Distance
self.head_dist = nn.Sequential(
    nn.Linear(d_model, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
```

#### **修改 forward 方法**
```python
def forward(self, x, return_feat=False):
    # ... 特征提取 ...
    feat = self.pooling(out)
    feat = self.dropout(feat)
    
    # 输出两个预测值
    pred_risk = self.head_risk(feat)
    pred_dist = self.head_dist(feat)
    
    if return_feat:
        return pred_risk, feat
        
    # 训练时返回两个预测值，测试时只返回 Risk
    if self.training:
        return pred_risk, pred_dist  # 训练: (risk, distance)
    else:
        return pred_risk              # 测试: 只返回 risk
```

---

### **Step 2: 修改 `utils/dataset.py`** ✅

#### **返回两列标签**
```python
# 获取 miss_distance 的索引
miss_distance_output = DataUtils.output_features.index('miss_distance')

# 构造两列标签 [risk, miss_distance]
train_y_risk = np.array([train[:, regression_output].ravel() 
                         for train in train_y]).reshape(-1, 1)
train_y_dist = np.array([train[:, miss_distance_output].ravel() 
                         for train in train_y]).reshape(-1, 1)
train_y_r = np.hstack([train_y_risk, train_y_dist])  # (N, 2)
```

**数据形状**:
- 之前: `train_y_r.shape = (N,)` 或 `(N, 1)`
- 现在: `train_y_r.shape = (N, 2)` → `[risk, miss_distance]`

---

### **Step 3: 修改 `utils/solver.py`** ✅

#### **分离两个标签**
```python
def _process_batch(self, seq_x, seq_y, weights):
    # seq_y: (B, 2) -> [risk, miss_distance]
    if seq_y.dim() == 2 and seq_y.shape[1] == 2:
        true_risk = seq_y[:, 0:1]  # (B, 1)
        true_dist = seq_y[:, 1:2]  # (B, 1)
    else:
        # 兼容旧版本（只有 risk）
        true_risk = seq_y
        true_dist = None
```

#### **处理模型输出**
```python
model_output = self.model(seq_x)

# 处理多任务输出
if isinstance(model_output, tuple) and len(model_output) == 2:
    pred, pred_dist = model_output  # Transformer 训练时返回两个值
else:
    pred = model_output
```

#### **计算多个损失**
```python
# 1. Risk Loss (带 LDS 权重)
loss_mse = self.criterion_mse(pred, true_risk, weights)

# 2. Physics Loss (预测距离的误差，不用加权)
loss_dist = 0
if pred_dist is not None and true_dist is not None:
    loss_dist = F.mse_loss(pred_dist, true_dist)

# 3. RankLoss
loss_rank = self.criterion_rank(pred, true_risk)

# 4. SupCR Loss
loss_sup = self.criterion_sup(features, true_risk)

# 组合 Loss (0.1 是物理约束的权重)
total_loss = loss_mse + 0.1*lambda_rank*loss_rank + 0.1*lambda_sup*loss_sup + 0.1*loss_dist
```

---

## 🎯 多任务学习的优势

### **1. 物理约束**
- Miss Distance 和 Risk 有物理关联
- 距离越小 → 风险越高
- 模型学习这种物理规律

### **2. 正则化效果**
- 辅助任务防止过拟合
- 共享特征提取层
- 提升主任务泛化能力

### **3. 数据利用**
- 充分利用 Miss Distance 信息
- 不需要额外标注
- 数据中已有该字段

---

## 📊 损失函数权重

| 损失项 | 权重 | 说明 |
|--------|------|------|
| **MSE (Risk)** | 1.0 | 主任务，带 LDS 权重 |
| **Physics (Distance)** | 0.1 | 物理约束，辅助任务 |
| **RankLoss** | 0.1 | 优化排序 |
| **SupCR** | 0.1 | 对比学习 |

**总损失**:
```
Total = MSE + 0.1×RankLoss + 0.1×SupCR + 0.1×Physics
```

---

## 🔬 技术细节

### **为什么 Physics Loss 权重是 0.1？**

1. **主任务优先**: Risk 预测是主要目标
2. **辅助作用**: Distance 只是物理约束
3. **经验值**: 0.05-0.2 都是合理范围

### **训练 vs 测试的区别**

```python
# 训练时
if self.training:
    return pred_risk, pred_dist  # 两个输出

# 测试时
else:
    return pred_risk  # 只返回 risk
```

**原因**:
- 训练时需要计算 Physics Loss
- 测试时只关心 Risk 预测
- 保持接口兼容性

---

## 📈 预期效果

### **性能提升**
| 指标 | 单任务 | 多任务 | 提升 |
|------|--------|--------|------|
| **R²** | 0.70 | **0.72-0.75** | +2-5% |
| **F2-score** | 0.93 | **0.94-0.96** | +1-3% |
| **损失比例** | 6-8x | **5-7x** | ✅ 更好 |

### **为什么会提升？**

1. **物理先验知识**: 模型学习 Risk-Distance 关系
2. **特征共享**: 两个任务共享 Encoder
3. **正则化**: 防止过拟合到 Risk 上

---

## 🚀 使用方法

### **训练 Transformer（多任务）**
```bash
python main.py --model TRANSFORMER --epoch 150
```

**默认配置**:
```python
d_model = 64
batch_size = 256
lr = 0.0005
lambda_rank = 0.1
use_supcr = True
lambda_sup = 0.1
physics_weight = 0.1  # 自动应用
```

### **对比单任务 vs 多任务**

#### **LSTM（单任务）**
```bash
python main.py --model LSTM --epoch 100
```

#### **Transformer（多任务）**
```bash
python main.py --model TRANSFORMER --epoch 150
```

#### **分析对比**
```bash
python analyze_results.py
```

---

## 🔧 调优建议

### **如果 Physics Loss 太大**
```python
# 在 solver.py 中调整权重
total_loss = loss_mse + ... + 0.05 * loss_dist  # 从 0.1 改为 0.05
```

### **如果 Physics Loss 太小**
```python
# 增大权重
total_loss = loss_mse + ... + 0.2 * loss_dist  # 从 0.1 改为 0.2
```

### **监控 Physics Loss**

在训练时添加打印：
```python
# 在 solver.py 的训练循环中
if e % 10 == 0:
    print(f"Physics Loss: {loss_dist:.4f}")
```

---

## 📊 架构对比

### **单任务学习**
```
Input → Encoder → Pooling → Head → Risk
```

### **多任务学习**
```
                    ┌→ Head_Risk → Risk
Input → Encoder → Pooling ┤
                    └→ Head_Dist → Distance
```

**共享部分**: Encoder + Pooling  
**独立部分**: 两个预测头

---

## 🎓 理论基础

### **多任务学习的数学表示**

```
L_total = L_risk + λ_physics * L_distance

其中:
L_risk = MSE(pred_risk, true_risk)
L_distance = MSE(pred_dist, true_dist)
λ_physics = 0.1
```

### **物理关系**

```
Risk ∝ 1 / Distance²

即:
log(Risk) ≈ -2 * log(Distance) + C
```

模型通过学习这种关系，提升预测准确性。

---

## 🔍 验证方法

### **1. 检查数据形状**
```python
# 在 solver.py 的 _get_loader 后添加
print(f"y_train shape: {y_train.shape}")  # 应该是 (N, 2)
print(f"y_train[:5]: {y_train[:5]}")      # 查看前5个样本
```

### **2. 检查模型输出**
```python
# 在 _process_batch 中添加
print(f"Model output type: {type(model_output)}")
if isinstance(model_output, tuple):
    print(f"pred_risk shape: {pred.shape}")
    print(f"pred_dist shape: {pred_dist.shape}")
```

### **3. 检查损失值**
```python
# 在训练循环中添加
print(f"Loss MSE: {loss_mse:.4f}")
print(f"Loss Dist: {loss_dist:.4f}")
print(f"Loss Rank: {loss_rank:.4f}")
print(f"Loss Sup: {loss_sup:.4f}")
```

---

## ⚠️ 注意事项

### **1. 兼容性**
- ✅ LSTM 仍然使用单任务（只预测 Risk）
- ✅ Transformer 使用多任务
- ✅ 代码自动检测并兼容

### **2. 数据要求**
- ✅ `miss_distance` 必须在数据中存在
- ✅ 已在 `DataUtils.output_features` 中定义
- ✅ 数据预处理已包含该字段

### **3. 训练稳定性**
- ⚠️ 如果 Physics Loss 震荡，降低权重
- ⚠️ 如果收敛慢，可能需要调整学习率
- ⚠️ 监控两个任务的损失平衡

---

## 🎯 下一步优化（可选）

### **1. 动态权重**
```python
# 根据训练阶段调整 Physics Loss 权重
epoch_ratio = e / self.args.epoch
physics_weight = 0.1 * (1 - epoch_ratio)  # 逐渐减小
```

### **2. 不确定性估计**
```python
# 预测 Distance 的不确定性
self.head_dist_var = nn.Linear(d_model, 1)  # 预测方差
```

### **3. 添加更多物理约束**
```python
# 例如：相对速度、轨道倾角等
self.head_velocity = nn.Linear(d_model, 1)
```

---

## 📝 总结

### **关键改进**
1. ✅ Transformer 支持多任务学习
2. ✅ 数据集返回两列标签
3. ✅ Solver 正确处理多任务输出
4. ✅ 物理约束作为辅助任务

### **预期收益**
- R² 提升 2-5%
- F2-score 提升 1-3%
- 更好的泛化能力
- 物理可解释性

### **立即开始**
```bash
cd C:\Users\tmpzh\Desktop\kelvin\Collision_kelvin
python main.py --model TRANSFORMER --epoch 150
```

训练完成后查看结果：
```bash
python analyze_results.py
```

---

**多任务学习是一个强大的技术，特别适合有物理关联的任务！** 🎉
