# 评估指标增强说明

## 更新概述

已对 `predict_risk.py` 进行增强，添加了完整的评估指标体系。

---

## 新增评估指标

### 原有指标
- ✓ **MSE/F2 Ratio**: 原始评估指标（高风险样本的MSE除以F2-score）

### 新增回归指标
- ✅ **MSE** (Mean Squared Error): 均方误差 - 衡量预测值与真实值的平方差
- ✅ **RMSE** (Root Mean Squared Error): 均方根误差 - MSE的平方根，与原始数据同量纲
- ✅ **MAE** (Mean Absolute Error): 平均绝对误差 - 对异常值不敏感
- ✅ **MAPE** (Mean Absolute Percentage Error): 平均绝对百分比误差 - 相对误差百分比
- ✅ **R²** (Coefficient of Determination): 决定系数 - 模型拟合优度（0-1之间，越接近1越好）

### 新增分类指标
- ✅ **F2-score**: F-beta分数（beta=2）- 更重视召回率的分类指标

---

## 输出格式

### 控制台输出
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

======================================================================
[Classification Report]
======================================================================
              precision    recall  f1-score   support

           0       0.70      0.80      0.74        49
           1       0.99      0.99      0.99      1213

    accuracy                           0.98      1262
   macro avg       0.84      0.89      0.87      1262
weighted avg       0.98      0.98      0.98      1262

======================================================================
```

### 文件输出

#### 1. `evaluation_results.txt`
完整的评估结果文本文件，格式与控制台输出相同。

#### 2. `evaluation_metrics.csv`
CSV格式的指标文件，便于后续分析和对比：

| MSE | RMSE | MAE | MAPE | R2 | F2 | MSE_F2_Ratio |
|-----|------|-----|------|----|----|--------------|
| 25.636549 | 5.063255 | 2.673074 | 18.4318 | 0.719479 | 0.520833 | 36.694378 |

---

## 指标解读

### 回归指标分析

#### MSE = 25.64
- 预测误差的平方平均值
- 对大误差更敏感（平方放大效应）

#### RMSE = 5.06
- 与风险值同量纲（对数尺度）
- 表示平均预测偏差约为5个对数单位

#### MAE = 2.67
- 平均绝对误差约2.67个对数单位
- 比RMSE小，说明存在一些较大的异常误差

#### MAPE = 18.43%
- 相对误差约18.43%
- 对于风险预测来说是可接受的范围

#### R² = 0.72
- 模型解释了72%的方差
- 表明模型拟合效果良好

### 分类指标分析

#### F2-score = 0.52
- 基于阈值-6的二分类性能
- Beta=2 更重视召回率（避免漏检高风险）
- 相对较低可能是因为：
  - 高风险样本数量少（49个 vs 1213个）
  - 类别严重不平衡

### 综合指标

#### MSE/F2 Ratio = 36.69
- 原始评估指标
- 平衡了回归精度和分类性能
- 值越小越好

---

## 代码修改详情

### 1. 导入新模块
```python
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
```

### 2. 增强 `evaluate_final_risk` 方法
```python
def evaluate_final_risk(self, gt_risk, pred_risk) -> dict:
    # 计算所有回归指标
    mse = mean_squared_error(gt_risk, pred_risk)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(gt_risk, pred_risk)
    mape = np.mean(np.abs((gt_risk - pred_risk) / (gt_risk + 1e-8))) * 100
    r2 = r2_score(gt_risk, pred_risk)
    
    # 计算F2-score
    gt_mask = gt_risk >= THRESHOLD
    pred_mask = pred_risk >= THRESHOLD
    f2 = fbeta_score(gt_mask, pred_mask, beta=2)
    
    # 返回完整指标字典
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'F2': f2,
        'MSE_F2_Ratio': mse_high_risk / f2
    }
```

### 3. 新增 `_save_metrics_to_file` 方法
自动保存评估结果到文件：
- `evaluation_results.txt`: 完整文本报告
- `evaluation_metrics.csv`: CSV格式指标

### 4. 更新 `run_pipe_line` 方法
- 美化输出格式
- 调用文件保存功能

---

## 使用方法

### 运行评估
```bash
cd C:\Users\tmpzh\Desktop\kelvin\Collision_kelvin\ref\ESA-collision-avoidance-challenge-main
python predict_risk.py
```

### 查看结果
1. **控制台**: 实时查看完整评估结果
2. **evaluation_results.txt**: 保存的文本报告
3. **evaluation_metrics.csv**: 用于Excel/Python分析的CSV文件

---

## 指标对比建议

### 与其他模型对比时关注：
1. **R²**: 整体拟合优度
2. **RMSE/MAE**: 预测精度
3. **F2-score**: 高风险检测能力
4. **MSE/F2 Ratio**: 综合性能（原始竞赛指标）

### 模型改进方向：
- 如果 **R² < 0.7**: 考虑增加模型复杂度或特征工程
- 如果 **F2 < 0.6**: 需要改进高风险样本的识别能力
- 如果 **MAPE > 20%**: 预测相对误差较大，需要优化

---

## 技术细节

### F2-score 计算
```python
# 阈值分类
high_risk_pred = (pred_risk >= -6)
high_risk_true = (gt_risk >= -6)

# F2-score (beta=2, 召回率权重是精确率的2倍)
F2 = 5 * (precision * recall) / (4 * precision + recall)
```

### MAPE 计算
```python
# 避免除零，添加小常数
MAPE = mean(|gt - pred| / (|gt| + 1e-8)) * 100%
```

---

## 兼容性说明

✅ **完全向后兼容**: 保留了原有的 MSE/F2 Ratio 指标  
✅ **无需修改调用**: 自动应用到现有流程  
✅ **自动保存**: 结果自动保存到文件，无需手动操作

---

## 更新日期
2025-12-04

## 作者
Cascade AI Assistant
