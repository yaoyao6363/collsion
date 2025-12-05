# 评估指标对比总结

## 两个项目的指标增强

已成功为两个项目添加完整的评估指标体系：

### 1. ESA-collision-avoidance-challenge-main
**路径**: `C:\Users\tmpzh\Desktop\kelvin\Collision_kelvin\ref\ESA-collision-avoidance-challenge-main\`

### 2. Collision_kelvin (主项目)
**路径**: `C:\Users\tmpzh\Desktop\kelvin\Collision_kelvin\`

---

## 评估指标对比

| 指标 | ESA项目 | 主项目 | 说明 |
|------|---------|--------|------|
| **MSE** | ✅ | ✅ | 均方误差 |
| **RMSE** | ✅ | ✅ | 均方根误差 |
| **MAE** | ✅ | ✅ | 平均绝对误差 |
| **MAPE** | ✅ | ✅ | 平均绝对百分比误差 |
| **R²** | ✅ | ✅ | 决定系数 |
| **F2-score** | ✅ | ✅ | F2分数（beta=2） |
| **MSE/F2 Ratio** | ✅ | ❌ | ESA原始竞赛指标 |

---

## ESA项目评估结果

### 当前性能指标
```
MSE:   25.64  - 预测误差平方的平均值
RMSE:  5.06   - 均方根误差（对数尺度）
MAE:   2.67   - 平均绝对误差
MAPE:  18.43% - 相对误差百分比
R²:    0.72   - 模型解释了72%的方差
F2:    0.52   - 高风险检测F2分数
```

### 分类性能
```
高风险类别 (0):
  - Precision: 0.70 (70%预测准确)
  - Recall:    0.80 (80%真实高风险被检出)
  - F1-score:  0.74

低风险类别 (1):
  - Precision: 0.99
  - Recall:    0.99
  - F1-score:  0.99

总体准确率: 98%
```

### 数据分布
- 高风险样本: 49 (3.9%)
- 低风险样本: 1213 (96.1%)
- **严重类别不平衡**

---

## 主项目 (Collision_kelvin) 修改

### 修改文件
`utils/solver.py` - `compute_metrics()` 函数

### 新增功能
1. **F2-score 计算**: 基于阈值-6的分类指标
2. **自动输出**: 训练和测试时自动显示所有指标
3. **CSV保存**: 每个epoch的指标保存到metrics.csv

### 输出格式
```python
Epoch: 10 | Cost: 12.34s || Train Loss: 0.123 Valid Loss: 0.234
--> MSE: 1.234 | MAE: 0.987 | RMSE: 1.111 | MAPE: 15.67% | R²: 0.854 | F2: 0.789
```

---

## ESA项目修改

### 修改文件
`predict_risk.py` - `evaluate_final_risk()` 方法

### 新增功能
1. **完整指标计算**: MSE, RMSE, MAE, MAPE, R², F2
2. **美化输出**: 分类显示回归指标、分类指标、综合指标
3. **文件保存**: 
   - `evaluation_results.txt`: 完整文本报告
   - `evaluation_metrics.csv`: CSV格式指标

### 输出格式
```
======================================================================
                    MODEL EVALUATION RESULTS
======================================================================

[Regression Metrics]
  MSE  (Mean Squared Error):           25.636549
  RMSE (Root Mean Squared Error):      5.063255
  MAE  (Mean Absolute Error):          2.673074
  MAPE (Mean Absolute % Error):        18.4318%
  R²   (Coefficient of Determination): 0.719479

[Classification Metric]
  F2-score (beta=2, threshold=-6):  0.520833

[Combined Metric]
  MSE/F2 Ratio (Original):             36.694378
```

---

## 技术实现对比

### ESA项目 (sklearn-based)
```python
# 使用 sklearn 和 numpy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, fbeta_score
import numpy as np

def evaluate_final_risk(self, gt_risk, pred_risk):
    mse = mean_squared_error(gt_risk, pred_risk)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(gt_risk, pred_risk)
    mape = np.mean(np.abs((gt_risk - pred_risk) / (gt_risk + 1e-8))) * 100
    r2 = r2_score(gt_risk, pred_risk)
    
    gt_mask = gt_risk >= THRESHOLD
    pred_mask = pred_risk >= THRESHOLD
    f2 = fbeta_score(gt_mask, pred_mask, beta=2)
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 
            'MAPE': mape, 'R2': r2, 'F2': f2}
```

### 主项目 (PyTorch-based)
```python
# 使用 PyTorch
import torch

def compute_metrics(preds, targets, threshold=-6):
    preds = torch.as_tensor(preds).view(-1).double()
    targets = torch.as_tensor(targets).view(-1).double()
    
    # 回归指标
    diff = preds - targets
    mse = torch.mean(diff ** 2)
    rmse = torch.sqrt(torch.clamp(mse, min=0.0))
    mae = torch.mean(torch.abs(diff))
    mape = torch.mean(torch.abs(diff / (targets + 1e-8))) * 100
    
    ss_res = torch.sum(diff ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2) + 1e-8
    r2 = 1 - ss_res / ss_tot
    
    # F2-score
    pred_mask = (preds >= threshold).long()
    target_mask = (targets >= threshold).long()
    
    tp = torch.sum((pred_mask == 1) & (target_mask == 1)).float()
    fp = torch.sum((pred_mask == 1) & (target_mask == 0)).float()
    fn = torch.sum((pred_mask == 0) & (target_mask == 1)).float()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f2 = 5 * (precision * recall) / (4 * precision + recall + 1e-8)
    
    return {'MSE': float(mse.item()), 'RMSE': float(rmse.item()), 
            'MAE': float(mae.item()), 'MAPE': float(mape.item()),
            'R2': float(r2.item()), 'F2': float(f2.item())}
```

---

## 关键差异

### 1. 框架
- **ESA项目**: sklearn + numpy (传统机器学习)
- **主项目**: PyTorch (深度学习)

### 2. 模型
- **ESA项目**: RandomForest (分类器 + 回归器)
- **主项目**: LSTM / Transformer / CDEA / Bayes_LSTM

### 3. 评估方式
- **ESA项目**: 一次性评估（训练后测试）
- **主项目**: 每个epoch评估（实时监控）

### 4. 输出
- **ESA项目**: 控制台 + TXT + CSV
- **主项目**: 控制台 + CSV (metrics.csv)

---

## 使用建议

### ESA项目
```bash
cd C:\Users\tmpzh\Desktop\kelvin\Collision_kelvin\ref\ESA-collision-avoidance-challenge-main
python predict_risk.py
# 查看: evaluation_results.txt, evaluation_metrics.csv
```

### 主项目
```bash
cd C:\Users\tmpzh\Desktop\kelvin\Collision_kelvin
python main.py --model LSTM
# 查看: results/checkpoint/*/metrics.csv
```

---

## 性能分析建议

### 如何判断模型好坏？

#### 回归性能
- **R² > 0.7**: 良好
- **MAPE < 20%**: 可接受
- **RMSE**: 越小越好（与数据尺度相关）

#### 分类性能
- **F2 > 0.6**: 良好（对于不平衡数据）
- **Recall > 0.8**: 高风险检出率高（关键）
- **Precision > 0.7**: 误报率可接受

#### ESA项目当前表现
- ✅ R² = 0.72 (良好)
- ⚠️ MAPE = 18.43% (接近临界)
- ⚠️ F2 = 0.52 (需要改进)
- ✅ Recall = 0.80 (高风险检出率良好)

---

## 改进方向

### ESA项目
1. **解决类别不平衡**: 
   - 使用SMOTE过采样
   - 调整类别权重
   - 集成学习方法

2. **提升F2-score**:
   - 优化分类器阈值
   - 增加高风险样本特征
   - 尝试代价敏感学习

3. **降低MAPE**:
   - 特征工程
   - 集成多个回归器
   - 异常值处理

### 主项目
1. **模型对比**: 使用新指标对比不同模型性能
2. **超参优化**: 基于F2-score调整超参数
3. **早停策略**: 考虑使用F2作为早停指标

---

## 文件清单

### ESA项目新增文件
- ✅ `evaluation_results.txt` - 完整评估报告
- ✅ `evaluation_metrics.csv` - CSV格式指标
- ✅ `METRICS_ENHANCEMENT.md` - 详细说明文档

### 主项目新增文件
- ✅ `METRICS_UPDATE.md` - 指标更新说明
- ✅ `test_metrics.py` - 指标测试脚本

### 修改的文件
- ✅ ESA项目: `predict_risk.py`
- ✅ 主项目: `utils/solver.py`

---

## 总结

✅ **两个项目都已成功添加完整的评估指标体系**  
✅ **包含6个核心指标: MSE, RMSE, MAE, MAPE, R², F2**  
✅ **自动保存评估结果到文件**  
✅ **输出格式美观、易读**  
✅ **完全向后兼容，无需修改现有代码**

现在你可以全面评估和对比不同模型的性能！🎉
