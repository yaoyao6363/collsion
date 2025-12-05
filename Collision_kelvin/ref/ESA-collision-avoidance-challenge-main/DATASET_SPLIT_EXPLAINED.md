# 数据集划分说明

## 📊 测试集划分方式

这个项目使用**自定义的分层划分策略**来确保训练集和测试集的类别分布一致。

---

## 🎯 划分比例

### 默认配置
```python
train_size = 0.85      # 85% 训练集
validation_size = 0.05 # 5% 验证集
test_size = 0.15       # 15% 测试集
```

### 实际使用
```python
# 在 get_train_test_dataset 方法中
train_data, test_data, val_data = self.custom_train_test_split(data_frame)
test_data = pd.concat([test_data, val_data])  # 合并验证集和测试集

# 最终比例
训练集: 85%
测试集: 15% (包含原始的 test + validation)
```

---

## 🔍 分层划分策略

### 为什么需要分层？

数据集存在**严重的类别不平衡**：
- **高风险事件** (risk_category=0): 约 3.9% (49个样本)
- **低风险事件** (risk_category=1): 约 96.1% (1213个样本)

如果随机划分，可能导致：
- ❌ 测试集中高风险样本太少
- ❌ 训练集和测试集分布不一致
- ❌ 模型评估不准确

### 分层划分逻辑

```python
def custom_train_test_split(self, data_frame, train_size=0.85, ...):
    # 1. 按风险类别分组
    grouped = data_frame.groupby('risk_category')['event_id'].unique()
    
    # 2. 对每个类别分别划分
    # 高风险事件: 85% 训练, 15% 测试
    # 低风险事件: 85% 训练, 15% 测试
    
    # 3. 确保每个类别的比例一致
    train_test_events = grouped['event_id'].apply(
        lambda g: DataUtils.split_array(g, test_size=test_size)
    )
```

---

## 📈 划分流程图

```
原始数据集 (所有 CDM 记录)
    ↓
按 event_id 分组
    ↓
按 risk_category 分层
    ↓
┌─────────────────────────────────────┐
│  高风险事件 (risk_category=0)      │
│  ├─ 85% → 训练集                    │
│  └─ 15% → 测试集                    │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│  低风险事件 (risk_category=1)      │
│  ├─ 85% → 训练集                    │
│  └─ 15% → 测试集                    │
└─────────────────────────────────────┘
    ↓
合并同类数据
    ↓
最终训练集 + 最终测试集
```

---

## 🎲 随机种子

```python
random_state = 42  # 固定随机种子，确保结果可复现
```

**作用**：
- ✅ 每次运行得到相同的训练/测试划分
- ✅ 结果可复现
- ✅ 便于对比不同模型

---

## 📊 实际测试集统计

从你的运行结果可以看到：

### 测试集大小
```
总样本数: 1262
  - 高风险 (0): 49 个
  - 低风险 (1): 1213 个
```

### 类别分布
```
高风险占比: 49/1262 ≈ 3.88%
低风险占比: 1213/1262 ≈ 96.12%
```

---

## 🔄 数据处理流程

### 1. 事件级别划分
```python
# 按 event_id 划分（不是按单个 CDM）
train_events = [event_1, event_3, event_5, ...]  # 85% 的事件
test_events = [event_2, event_4, event_6, ...]   # 15% 的事件
```

**重要**：划分是基于**事件 (event_id)**，而不是单个 CDM 记录。

### 2. 提取对应数据
```python
train_data_frame = data_frame[data_frame['event_id'].isin(train_events)]
test_data_frame = data_frame[data_frame['event_id'].isin(test_events)]
```

### 3. 输入输出序列划分
```python
# 对每个事件，划分输入序列和输出序列
train_x, train_y = self.split_input_output_seq(train_data)
test_x, test_y = self.split_input_output_seq(test_data)
```

---

## 🎯 输入输出序列逻辑

### 时间线示例

```
事件的 CDM 历史记录（按时间倒序）:
┌────────────────────────────────────────────────────────┐
│ TCA-7天 → TCA-5天 → TCA-3天 → TCA-2天 → TCA-1天 → TCA │
│   CDM1     CDM2      CDM3      CDM4      CDM5     CDM6 │
└────────────────────────────────────────────────────────┘
         ↑                         ↑
    输入序列                   输出序列
  (历史 CDM)               (ground truth)
```

### Ground Truth 选择
```python
# Ground truth 是 TCA-2天 的 CDM
# 条件: time_to_tca = 2 或 time_to_tca = Min(time_to_tca > 2)
```

### 输入序列
- **包含**：ground truth 之前的所有 CDM
- **特征**：包括历史风险值和其他特征
- **数量**：最近的 n_CDMs 个（默认 n_CDMs=2）

### 输出序列
- **目标**：ground truth 时刻的风险值和风险类别
- **用于**：模型训练和评估

---

## 🔢 特征划分

### 分类特征 (Classification Features)
用于训练 **RandomForestClassifier**：
```python
calssification_features = [...]  # 特定的分类特征
train_x_c = train_x[:, calssification_features]
train_y_c = train_y[:, 'risk_category']  # 0 或 1
```

### 回归特征 (Regression Features)
用于训练 **RandomForestRegressor**：
```python
regression_features = [...]  # 特定的回归特征
train_x_r = train_x[:, regression_features]
train_y_r = train_y[:, 'risk']  # 对数尺度的风险值
```

---

## 📦 最终数据结构

### 训练集
```python
train_dataset = (
    (train_x_c, train_y_c),  # 分类训练数据
    (train_x_r, train_y_r)   # 回归训练数据
)
```

### 测试集
```python
test_dataset = (
    (test_x_c, test_y_c),  # 分类测试数据
    (test_x_r, test_y_r)   # 回归测试数据
)
```

---

## 🎓 关键特点总结

### 1. 分层划分
✅ **按风险类别分层**，确保训练集和测试集的类别分布一致

### 2. 事件级别划分
✅ **按事件 (event_id) 划分**，同一事件的所有 CDM 要么全在训练集，要么全在测试集

### 3. 时序逻辑
✅ **考虑时间顺序**，输入序列是历史 CDM，输出序列是 ground truth

### 4. 可复现性
✅ **固定随机种子** (random_state=42)，确保结果可复现

### 5. 类别不平衡处理
✅ **分层采样**保持原始分布，但仍需注意类别不平衡问题

---

## 📊 测试集样本分布

从评估结果可以看到：

```
Classification Report:
              precision    recall  f1-score   support

           0       0.70      0.80      0.74        49   ← 高风险测试样本
           1       0.99      0.99      0.99      1213  ← 低风险测试样本

    accuracy                           0.98      1262  ← 总测试样本
```

### 解读
- **测试集总数**: 1262 个样本
- **高风险**: 49 个 (3.88%)
- **低风险**: 1213 个 (96.12%)
- **类别比例**: 与训练集保持一致（分层划分的效果）

---

## 🔍 验证划分是否正确

### 检查类别分布
```python
# 训练集
print(f"训练集高风险占比: {(train_y_c == 0).sum() / len(train_y_c):.2%}")
print(f"训练集低风险占比: {(train_y_c == 1).sum() / len(train_y_c):.2%}")

# 测试集
print(f"测试集高风险占比: {(test_y_c == 0).sum() / len(test_y_c):.2%}")
print(f"测试集低风险占比: {(test_y_c == 1).sum() / len(test_y_c):.2%}")
```

### 检查事件不重叠
```python
# 确保训练集和测试集的事件 ID 不重叠
train_events = set(train_data['event_id'].unique())
test_events = set(test_data['event_id'].unique())
assert len(train_events & test_events) == 0, "训练集和测试集有重叠！"
```

---

## 💡 与主项目的对比

| 特性 | ESA项目 | 主项目 (Collision_kelvin) |
|------|---------|---------------------------|
| **划分方式** | 自定义分层划分 | 标准划分 |
| **划分单位** | 事件 (event_id) | 样本 |
| **类别平衡** | 分层保持分布 | 可能不平衡 |
| **时序逻辑** | 考虑 CDM 时序 | 标准时序 |
| **验证集** | 合并到测试集 | 独立验证集 |

---

## 🎯 总结

### 测试集特点

1. **大小**: 约 15% 的数据（1262个样本）
2. **划分方式**: 按事件 ID 分层划分
3. **类别分布**: 与训练集保持一致（高风险 ~4%, 低风险 ~96%）
4. **可复现**: 固定随机种子 (random_state=42)
5. **时序性**: 考虑 CDM 的时间顺序

### 关键代码位置

- **划分函数**: `ML_dataset_creation.py` → `custom_train_test_split()`
- **数据准备**: `ML_dataset_creation.py` → `get_train_test_dataset()`
- **调用位置**: `predict_risk.py` → `get_train_test_dataset()`

---

**创建日期**: 2025-12-04  
**版本**: 1.0
