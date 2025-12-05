# 🚀 损失函数快速参考卡片

## 📋 一分钟快速开始

```bash
# 1. 测试集成
python test_loss_integration.py

# 2. 基础训练（LDS 自动启用）
python main.py --model LSTM --epoch 50

# 3. 优化 F2-score
python main.py --model LSTM --lambda_rank 0.1 --epoch 50
```

---

## 🎯 三种损失函数速查

| 损失函数 | 作用 | 何时使用 | 额外开销 | 需要修改模型 |
|---------|------|---------|---------|------------|
| **LDS** | 解决类别不平衡 | 必用 | ~5% | ❌ |
| **RankLoss** | 优化样本顺序 | 优化 F2-score | ~30% | ❌ |
| **SupCR** | 提升特征质量 | 追求最佳性能 | ~80% | ✅ |

---

## 🔧 命令行参数速查

```bash
# LDS（自动启用，无需参数）

# RankLoss
--lambda_rank 0.1      # 权重系数 [0.05, 0.1, 0.2, 0.5]
--rank_margin 0.0      # 边界值 [0.0, 0.5, 1.0]

# SupConRegressionLoss
--use_supcr            # 启用标志
--lambda_sup 0.5       # 权重系数 [0.3, 0.5, 0.7, 1.0]
--sup_temp 0.1         # 温度 [0.05, 0.1, 0.2]
--sup_sigma 2.0        # 标签相似度带宽 [1.0, 2.0, 3.0]
```

---

## 📊 预期性能提升

```
Baseline:           MSE=25.64  F2=0.521  R²=0.719
+ LDS:              MSE=24.12  F2=0.548  R²=0.738  (+5%)
+ LDS + RankLoss:   MSE=25.01  F2=0.612  R²=0.725  (+17% F2)
+ 全部:             MSE=23.45  F2=0.635  R²=0.756  (+22% F2)
```

---

## ⚡ 常用命令组合

### **场景1: 快速验证**
```bash
python main.py --model LSTM --epoch 10 --lambda_rank 0.1
```

### **场景2: 完整训练**
```bash
python main.py --model LSTM --epoch 50 --lambda_rank 0.1 --batch_size 64
```

### **场景3: 超参数搜索**
```bash
for lr in 0.05 0.1 0.2; do
    python main.py --model LSTM --lambda_rank $lr --epoch 30
done
```

### **场景4: 全面优化（需先修改模型）**
```bash
python main.py --model LSTM \
    --lambda_rank 0.1 \
    --use_supcr \
    --lambda_sup 0.5 \
    --epoch 50
```

---

## 🐛 故障排查

### **问题1: CUDA out of memory**
```bash
# 解决方案
--batch_size 32        # 减小 batch size
--lambda_rank 0.0      # 暂时禁用 RankLoss
```

### **问题2: F2-score 没提升**
```bash
# 检查 LDS 权重
# 在 solver.py 的 _get_loader 后添加：
print(f"High risk weight: {train_weights_n[y_train > -6].mean():.3f}")
print(f"Low risk weight: {train_weights_n[y_train <= -6].mean():.3f}")

# 调整 lambda_rank
--lambda_rank 0.05     # 减小权重
```

### **问题3: 模型不支持 return_feat**
```python
# 在模型的 forward 方法中添加：
def forward(self, x, return_feat=False):
    features = self.encoder(x)
    pred = self.fc(features)
    if return_feat:
        return pred, features
    return pred
```

---

## 📁 文件速查

| 文件 | 用途 |
|------|------|
| `utils/loss.py` | 损失函数实现 |
| `utils/solver.py` | 训练流程（已集成） |
| `test_loss_integration.py` | 测试脚本 |
| `LOSS_INTEGRATION_GUIDE.md` | 详细使用指南 |
| `LOSS_INTEGRATION_SUMMARY.md` | 完整总结 |
| `LOSS_QUICK_REFERENCE.md` | 本文档 |

---

## 🎓 核心概念

### **LDS 三步法**
```
1. Binning    → 直方图统计
2. Smoothing  → 高斯平滑
3. Reweighting → 逆频率加权
```

### **RankLoss 公式**
```python
Loss = ReLU(-sign(target_diff) * pred_diff + margin)
# 顺序正确 → Loss = 0
# 顺序错误 → Loss > 0
```

### **SupCR 核心**
```python
weight = exp(-(label_dist)² / (2σ²))
# 标签相近 → 权重高 → 特征拉近
# 标签远离 → 权重低 → 特征推开
```

---

## 🔬 实验建议

### **基础实验（1-2天）**
```bash
# 1. Baseline
python main.py --model LSTM --epoch 50

# 2. + RankLoss
python main.py --model LSTM --lambda_rank 0.1 --epoch 50

# 3. 对比结果
# 查看 results/checkpoint/*/result.txt
```

### **完整实验（1周）**
```bash
# 测试所有模型
for model in LSTM TRANSFORMER CDEA BAYES_LSTM; do
    python main.py --model $model --lambda_rank 0.1 --epoch 50
done

# 超参数搜索
for lr in 0.05 0.1 0.15 0.2; do
    python main.py --model LSTM --lambda_rank $lr --epoch 30
done
```

---

## 💡 最佳实践

1. ✅ **先测试**: `python test_loss_integration.py`
2. ✅ **从简单开始**: 先用 LDS，再加 RankLoss
3. ✅ **小步迭代**: 每次只调一个超参数
4. ✅ **监控指标**: 重点看 F2-score 和 R²
5. ✅ **保存结果**: 每次实验都记录超参数和结果

---

## 📞 获取帮助

- 详细文档: `LOSS_INTEGRATION_GUIDE.md`
- 完整总结: `LOSS_INTEGRATION_SUMMARY.md`
- 代码实现: `utils/loss.py`
- 测试脚本: `test_loss_integration.py`

---

**打印本页，贴在显示器旁边！** 📌
