# 评估指标更新说明

## 修改内容

已在 `utils/solver.py` 中的 `compute_metrics` 函数添加 **F2-score** 指标。

## 当前支持的评估指标

### 1. 回归指标
- **MSE** (Mean Squared Error): 均方误差
- **RMSE** (Root Mean Squared Error): 均方根误差  
- **MAE** (Mean Absolute Error): 平均绝对误差
- **MAPE** (Mean Absolute Percentage Error): 平均绝对百分比误差
- **R²** (R-squared): 决定系数

### 2. 分类指标（基于阈值）
- **F2-score**: F-beta分数（beta=2），更重视召回率
  - 阈值默认为 `-6`（对数尺度）
  - 风险值 >= -6 视为高风险
  - 风险值 < -6 视为低风险

## F2-score 计算逻辑

```python
# 1. 将连续的风险值转换为二分类
pred_mask = (preds >= threshold).long()    # 预测的高/低风险
target_mask = (targets >= threshold).long() # 真实的高/低风险

# 2. 计算混淆矩阵
TP = 预测高风险 & 真实高风险
FP = 预测高风险 & 真实低风险  
FN = 预测低风险 & 真实高风险

# 3. 计算 F2-score (beta=2)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F2 = 5 * (precision * recall) / (4 * precision + recall)
```

## 为什么使用 F2-score？

在卫星碰撞规避场景中：
- **高召回率至关重要**：不能漏检高风险事件
- **F2-score (beta=2)**：召回率的权重是精确率的2倍
- **符合航天安全原则**："宁可信其有，不可信其无"

## 输出示例

### 训练过程
```
Epoch: 10 | Cost: 12.345678s || Train Loss: 0.123456 Valid Loss: 0.234567
--> MSE: 1.23456 | MAE: 0.98765 | RMSE: 1.11111 | MAPE: 15.67890% | R^2: 0.85432 | F2: 0.78901
```

### 测试结果
```
Test Loss: 0.234567 | 
--> MSE: 1.23456 | MAE: 0.98765 | RMSE: 1.11111 | MAPE: 15.67890% | R^2: 0.85432 | F2: 0.78901
```

### CSV 文件 (metrics.csv)
每个 epoch 的指标会保存到 CSV 文件，包含以下列：
- epoch
- cost_sec
- train_loss
- valid_loss
- MSE
- MAE
- RMSE
- MAPE
- R2
- **F2** (新增)

## 自定义阈值

如需修改风险阈值，可在调用时传入参数：

```python
# 默认阈值 -6
metrics = compute_metrics(preds, targets)

# 自定义阈值 -8
metrics = compute_metrics(preds, targets, threshold=-8)
```

## 修改的文件

- `utils/solver.py`
  - `compute_metrics()` 函数：添加 F2-score 计算
  - `train()` 方法：更新打印和保存逻辑
  - `test()` 方法：更新打印和保存逻辑

## 兼容性

✅ 向后兼容：所有原有指标保持不变  
✅ 新增指标：F2-score 自动计算并输出  
✅ 无需修改调用代码：自动应用到所有模型（LSTM, Transformer, CDEA, Bayes_LSTM）
