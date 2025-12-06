# 🎯 当前训练配置

## 📋 完整配置参数

### **模型架构**
```python
model = 'LSTM'
hidden_size = 64      # ⬇️ 减小模型容量
num_layers = 3        # 保持
dropout = 0.5         # ✅ 强正则化
```

### **损失函数**
```python
# LDS + Weighted MSE (自动启用)
lambda_rank = 0.1     # ✅ RankLoss
use_supcr = True      # ✅ SupConRegressionLoss (新启用)
lambda_sup = 0.1      # ✅ SupCR 权重
sup_temp = 0.1        # 对比学习温度
sup_sigma = 2.0       # 标签相似度
```

### **训练参数**
```python
epoch = 200           # 最大训练轮数
patience = 15         # ⬆️ 增加早停耐心
batch_size = 128      # ⬆️ 增大批次（更稳定）
lr = 0.001            # ⬆️ 提高学习率 (10倍)
weight_decay = 1e-3   # L2 正则化
```

### **数据增强**
```python
augment = True        # ✅ 训练集噪声注入
noise_std = 0.02      # 噪声强度
```

---

## 🔧 关键变化说明

### **1. 启用 SupConRegressionLoss** ⭐
```
use_supcr: False → True
lambda_sup: 0.0 → 0.1
```

**效果**:
- ✅ 利用特征空间的对比学习
- ✅ 相似标签的样本特征更接近
- ✅ 预期提升 R² 和 F2-score

### **2. 减小模型容量**
```
hidden_size: 80 → 64
```

**效果**:
- ✅ 减少参数量约 20%
- ✅ 降低过拟合风险
- ✅ 训练更快

### **3. 增大批次大小**
```
batch_size: 64 → 128
```

**效果**:
- ✅ 更稳定的梯度估计
- ✅ 减少训练时间
- ✅ 更平滑的收敛

### **4. 提高学习率**
```
lr: 0.0001 → 0.001 (10倍)
```

**效果**:
- ✅ 更快的收敛速度
- ⚠️ 需要配合更大的 batch_size
- ⚠️ 可能需要调整（如果不稳定）

### **5. 增加早停耐心**
```
patience: 10 → 15
```

**效果**:
- ✅ 给模型更多时间优化
- ✅ 避免过早停止
- ✅ 配合 SupCR 需要更多 epoch

---

## 📊 预期效果

| 指标 | 之前 | 预期 | 目标 |
|------|------|------|------|
| **R²** | 0.660 | **0.68-0.72** | > 0.70 |
| **F2-score** | 0.920 | **0.93-0.95** | > 0.88 |
| **损失比例** | 10.79x | **5-7x** | < 5x |
| **训练时间/epoch** | 0.41s | **0.35s** | - |

---

## ⚠️ 潜在风险

### **风险1: 学习率过高**
```
lr = 0.001 (提高了 10 倍)
```

**症状**:
- 训练损失震荡
- 验证损失不下降
- 梯度爆炸

**解决方案**:
```bash
# 如果训练不稳定，降低学习率
python main.py --lr 0.0005
# 或
python main.py --lr 0.0001
```

### **风险2: SupCR 增加计算开销**
```
use_supcr = True
```

**症状**:
- 训练时间显著增加
- 内存占用增大

**解决方案**:
```bash
# 如果内存不足，减小批次
python main.py --batch_size 64
# 或关闭 SupCR
python main.py --no-use_supcr --lambda_sup 0.0
```

---

## 🚀 运行命令

### **使用默认配置（推荐）**
```bash
cd C:\Users\tmpzh\Desktop\kelvin\Collision_kelvin
python main.py --model LSTM
```

### **显式指定所有参数**
```bash
python main.py \
    --model LSTM \
    --epoch 200 \
    --patience 15 \
    --batch_size 128 \
    --hidden_size 64 \
    --dropout 0.5 \
    --lambda_rank 0.1 \
    --use_supcr \
    --lambda_sup 0.1 \
    --lr 0.001
```

### **如果训练不稳定，使用保守配置**
```bash
python main.py \
    --model LSTM \
    --lr 0.0005 \
    --batch_size 64 \
    --patience 10
```

---

## 📈 监控重点

### **训练开始时（前10个epoch）**
观察指标:
- 训练损失是否平稳下降
- 验证损失是否跟随
- 是否出现 NaN 或 Inf

**健康信号**:
```
Epoch 1:  Train Loss: 15.2 | Valid Loss: 18.5
Epoch 5:  Train Loss: 8.3  | Valid Loss: 12.1
Epoch 10: Train Loss: 5.6  | Valid Loss: 9.2
```

**异常信号**:
```
Epoch 1:  Train Loss: 150.2 | Valid Loss: 185.5  ❌ 学习率太高
Epoch 5:  Train Loss: NaN    | Valid Loss: NaN    ❌ 梯度爆炸
Epoch 10: Train Loss: 15.1  | Valid Loss: 15.3   ❌ 学习太慢
```

### **训练中期（epoch 50-100）**
观察指标:
- 损失比例是否 < 5x
- R² 是否持续提升
- F2-score 是否 > 0.90

### **训练后期（epoch 100+）**
观察指标:
- 早停是否正常触发
- 最佳模型在哪个 epoch
- 是否过拟合

---

## 🔬 技术细节

### **SupConRegressionLoss 工作原理**
```python
# 1. 提取特征
features = model(x, return_feat=True)  # (B, feature_dim)

# 2. 计算标签相似度
sim_matrix = exp(-|y_i - y_j|^2 / (2 * sigma^2))

# 3. 对比损失
loss = -log(
    sum(sim * exp(feat_i · feat_j / temp)) / 
    sum(exp(feat_i · feat_k / temp))
)
```

**效果**:
- 相似标签的样本在特征空间中聚集
- 不同标签的样本在特征空间中分离
- 提升模型的判别能力

### **组合损失函数**
```python
total_loss = loss_mse + 
             lambda_rank * loss_rank + 
             lambda_sup * loss_supcr

# 当前配置:
# total_loss = MSE + 0.1 * RankLoss + 0.1 * SupCR
```

---

## 📊 与之前配置的对比

| 参数 | 第2轮 | 第3轮 | 第4轮（当前） |
|------|-------|-------|--------------|
| hidden_size | 96 | 80 | **64** ⬇️ |
| batch_size | 64 | 64 | **128** ⬆️ |
| lr | 0.0001 | 0.0001 | **0.001** ⬆️ |
| patience | 10 | 10 | **15** ⬆️ |
| use_supcr | ❌ | ❌ | **✅** |
| lambda_sup | 0.0 | 0.0 | **0.1** |
| augment | ❌ | ✅ | **✅** |
| weight_decay | 1e-4 | 1e-3 | **1e-3** |

---

## 🎯 成功标准

### **最低要求（必须达到）**
- ✅ 训练稳定（不出现 NaN）
- ✅ R² > 0.65
- ✅ F2-score > 0.90
- ✅ 损失比例 < 8x

### **良好表现（期望达到）**
- ✅ R² > 0.70
- ✅ F2-score > 0.92
- ✅ 损失比例 < 5x

### **优秀表现（最佳情况）**
- ✅ R² > 0.75
- ✅ F2-score > 0.95
- ✅ 损失比例 < 3x

---

## 📝 训练完成后

### **1. 分析结果**
```bash
python analyze_results.py
```

### **2. 查看对比**
```bash
# 打开生成的文件
notepad results\checkpoint\comparison.csv
```

### **3. 查看训练曲线**
```bash
# 图片位置
results\checkpoint\LSTM_43_0\training_curve.png
```

### **4. 如果效果不好**

**情况A: 训练不稳定（损失震荡）**
```bash
# 降低学习率
python main.py --lr 0.0005
```

**情况B: 仍然过拟合（损失比例 > 8x）**
```bash
# 进一步减小模型
python main.py --hidden_size 48 --num_layers 2
```

**情况C: 欠拟合（R² < 0.60）**
```bash
# 增大模型
python main.py --hidden_size 80 --dropout 0.4
```

---

## 🎓 配置哲学

### **这个配置的设计思路**

1. **小模型 + 强正则化** → 防止过拟合
2. **大批次 + 高学习率** → 快速稳定收敛
3. **多损失函数组合** → 全方位优化
4. **数据增强** → 提升泛化能力
5. **长耐心** → 给 SupCR 足够时间

### **适用场景**
- ✅ 数据量中等（~6000 样本）
- ✅ 存在过拟合问题
- ✅ 需要优化排序和分类性能
- ✅ 有 GPU 支持

---

**当前状态**: 配置已更新，准备开始训练 🚀

**预计训练时间**: 15-25 分钟（200 epochs，可能早停）

**下一步**: 运行 `python main.py --model LSTM`
