# 如何运行 ESA 碰撞规避项目

## ⚠️ 重要提示

**必须从项目目录运行**，否则会出现 `FileNotFoundError` 错误！

---

## 🚀 三种运行方法

### 方法 1: 使用批处理脚本（最简单）✨

**Windows 批处理文件**：
```bash
# 双击运行
run.bat
```

或在任意目录的终端中：
```bash
C:\Users\tmpzh\Desktop\kelvin\Collision_kelvin\ref\ESA-collision-avoidance-challenge-main\run.bat
```

---

### 方法 2: 使用 PowerShell 脚本

```powershell
# 在任意目录运行
powershell -ExecutionPolicy Bypass -File "C:\Users\tmpzh\Desktop\kelvin\Collision_kelvin\ref\ESA-collision-avoidance-challenge-main\run.ps1"
```

或先切换到项目目录：
```powershell
cd C:\Users\tmpzh\Desktop\kelvin\Collision_kelvin\ref\ESA-collision-avoidance-challenge-main
.\run.ps1
```

---

### 方法 3: 手动切换目录（传统方法）

```powershell
# 1. 切换到项目目录
cd C:\Users\tmpzh\Desktop\kelvin\Collision_kelvin\ref\ESA-collision-avoidance-challenge-main

# 2. 运行 Python 脚本
python predict_risk.py
```

---

## ❌ 错误示例（不要这样做）

```powershell
# ❌ 错误：从其他目录运行
cd C:\Users\tmpzh\Desktop\kelvin
python Collision_kelvin\ref\ESA-collision-avoidance-challenge-main\predict_risk.py
# 结果：FileNotFoundError: 'C:\Users\tmpzh\Desktop\kelvin\Dataset\train_data.csv'
```

**原因**：程序使用 `os.getcwd()` 获取当前工作目录，如果不在项目目录下运行，路径会错误。

---

## 📊 运行结果

成功运行后，你会看到：

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

✓ Evaluation results saved to: evaluation_results.txt
✓ Metrics saved to CSV: evaluation_metrics.csv
```

---

## 📁 输出文件位置

运行成功后，会在项目目录生成：

```
C:\Users\tmpzh\Desktop\kelvin\Collision_kelvin\ref\ESA-collision-avoidance-challenge-main\
├── evaluation_results.txt  ← 完整评估报告
├── evaluation_metrics.csv  ← CSV格式指标
└── data_plots\             ← 数据可视化图表（如果生成）
```

---

## 🔧 故障排查

### 问题 1: FileNotFoundError

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: 
'C:\\Users\\tmpzh\\Desktop\\kelvin\\Dataset\\train_data.csv'
```

**解决方法**：
- ✅ 确保从项目目录运行（使用上面的方法 1、2 或 3）
- ✅ 使用提供的 `run.bat` 或 `run.ps1` 脚本

---

### 问题 2: 权限错误（PowerShell）

**错误信息**：
```
无法加载文件 run.ps1，因为在此系统上禁止运行脚本
```

**解决方法**：
```powershell
# 使用 -ExecutionPolicy Bypass 参数
powershell -ExecutionPolicy Bypass -File run.ps1
```

---

### 问题 3: Python 未找到

**错误信息**：
```
'python' 不是内部或外部命令
```

**解决方法**：
```powershell
# 使用完整路径
d:/ProgramData/anaconda3/python.exe predict_risk.py

# 或激活 conda 环境
conda activate base
python predict_risk.py
```

---

## 📝 快速参考卡片

### 最简单的运行方式（推荐）

```bash
# 方法 A: 双击运行
run.bat

# 方法 B: 命令行运行
cd C:\Users\tmpzh\Desktop\kelvin\Collision_kelvin\ref\ESA-collision-avoidance-challenge-main
python predict_risk.py
```

### 查看结果

```powershell
# 查看文本报告
type evaluation_results.txt

# 查看 CSV 数据
type evaluation_metrics.csv

# 在 Excel 中打开
start evaluation_metrics.csv
```

---

## 🎯 关键点总结

1. ✅ **必须从项目目录运行**
2. ✅ **使用提供的脚本最简单**（`run.bat` 或 `run.ps1`）
3. ✅ **结果自动保存到文件**
4. ✅ **支持任何 Python 环境**（base、sg2ada 等）

---

## 📞 需要帮助？

如果遇到问题，请检查：
1. 当前工作目录是否正确
2. Dataset 文件夹是否存在
3. train_data.csv 是否在 Dataset 文件夹中
4. Python 环境是否正确配置

---

**创建日期**: 2025-12-04  
**版本**: 1.0
