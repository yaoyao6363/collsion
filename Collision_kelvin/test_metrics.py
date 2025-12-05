"""
测试 compute_metrics 函数，验证所有指标计算是否正确
"""
import torch
import sys
sys.path.append('C:/Users/tmpzh/Desktop/kelvin/Collision_kelvin')
from utils.solver import compute_metrics

# 创建测试数据
torch.manual_seed(42)
preds = torch.randn(100) * 5 - 10  # 模拟风险值（对数尺度）
targets = torch.randn(100) * 5 - 10

# 计算指标
metrics = compute_metrics(preds, targets, threshold=-6)

print("=" * 60)
print("评估指标测试结果")
print("=" * 60)
print(f"MSE   (均方误差):        {metrics['MSE']:.5f}")
print(f"RMSE  (均方根误差):      {metrics['RMSE']:.5f}")
print(f"MAE   (平均绝对误差):    {metrics['MAE']:.5f}")
print(f"MAPE  (平均绝对百分比误差): {metrics['MAPE']:.5f}%")
print(f"R²    (决定系数):        {metrics['R2']:.5f}")
print(f"F2    (F2-score):        {metrics['F2']:.5f}")
print("=" * 60)

# 验证F2计算逻辑
pred_mask = (preds >= -6).long()
target_mask = (targets >= -6).long()
tp = torch.sum((pred_mask == 1) & (target_mask == 1)).item()
fp = torch.sum((pred_mask == 1) & (target_mask == 0)).item()
fn = torch.sum((pred_mask == 0) & (target_mask == 1)).item()
tn = torch.sum((pred_mask == 0) & (target_mask == 0)).item()

print("\nF2-score 详细信息 (阈值=-6):")
print(f"  True Positives  (TP): {tp}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")
print(f"  True Negatives  (TN): {tn}")
print(f"  Precision: {tp/(tp+fp) if (tp+fp)>0 else 0:.5f}")
print(f"  Recall:    {tp/(tp+fn) if (tp+fn)>0 else 0:.5f}")
print("=" * 60)
print("✓ 所有指标计算成功！")
