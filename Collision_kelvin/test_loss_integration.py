"""
测试损失函数集成是否正确
运行: python test_loss_integration.py
"""

import torch
import numpy as np
from utils.loss import calculate_lds_weights, WeightedMSELoss, RankLoss, SupConRegressionLoss

def test_lds_weights():
    """测试 LDS 权重计算"""
    print("=" * 60)
    print("测试 1: LDS 权重计算")
    print("=" * 60)
    
    # 模拟不平衡的风险分布
    # 大部分样本是低风险 (-30 到 -10)，少数是高风险 (-6 到 -2)
    low_risk = np.random.uniform(-30, -10, 950)
    high_risk = np.random.uniform(-6, -2, 50)
    targets = np.concatenate([low_risk, high_risk])
    np.random.shuffle(targets)
    
    print(f"样本总数: {len(targets)}")
    print(f"高风险样本 (>-6): {np.sum(targets > -6)} ({np.sum(targets > -6)/len(targets)*100:.1f}%)")
    print(f"低风险样本 (<-6): {np.sum(targets <= -6)} ({np.sum(targets <= -6)/len(targets)*100:.1f}%)")
    
    # 计算 LDS 权重
    weights = calculate_lds_weights(targets, n_bins=100, kernel='gaussian', ks=5, sigma=2)
    
    print(f"\nLDS 权重统计:")
    print(f"  Min: {weights.min():.3f}")
    print(f"  Max: {weights.max():.3f}")
    print(f"  Mean: {weights.mean():.3f}")
    print(f"  Std: {weights.std():.3f}")
    
    # 验证高风险样本是否获得更高权重
    high_risk_mask = targets > -6
    avg_weight_high = weights[high_risk_mask].mean()
    avg_weight_low = weights[~high_risk_mask].mean()
    
    print(f"\n权重对比:")
    print(f"  高风险样本平均权重: {avg_weight_high:.3f}")
    print(f"  低风险样本平均权重: {avg_weight_low:.3f}")
    print(f"  权重比: {avg_weight_high / avg_weight_low:.2f}x")
    
    if avg_weight_high > avg_weight_low:
        print("✅ 测试通过: 高风险样本获得了更高权重")
    else:
        print("❌ 测试失败: 权重分配不符合预期")
    
    return weights

def test_weighted_mse():
    """测试加权 MSE 损失"""
    print("\n" + "=" * 60)
    print("测试 2: WeightedMSELoss")
    print("=" * 60)
    
    criterion = WeightedMSELoss()
    
    # 创建测试数据
    pred = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    target = torch.tensor([[1.5], [2.5], [3.5], [4.5]])
    weights = torch.tensor([[1.0], [1.0], [5.0], [5.0]])  # 后两个样本权重更高
    
    # 不加权的 MSE
    loss_unweighted = criterion(pred, target, weights=None)
    
    # 加权的 MSE
    loss_weighted = criterion(pred, target, weights=weights)
    
    print(f"不加权 MSE: {loss_unweighted.item():.4f}")
    print(f"加权 MSE: {loss_weighted.item():.4f}")
    
    # 加权后的损失应该更大（因为误差大的样本权重高）
    if loss_weighted > loss_unweighted:
        print("✅ 测试通过: 加权机制正常工作")
    else:
        print("⚠️  警告: 加权效果不明显")
    
    # 测试梯度
    pred.requires_grad = True
    loss = criterion(pred, target, weights=weights)
    loss.backward()
    
    if pred.grad is not None:
        print(f"梯度计算正常: {pred.grad.shape}")
        print("✅ 测试通过: 梯度反向传播正常")
    else:
        print("❌ 测试失败: 梯度为 None")

def test_rank_loss():
    """测试排序损失"""
    print("\n" + "=" * 60)
    print("测试 3: RankLoss")
    print("=" * 60)
    
    criterion = RankLoss(margin=0.0)
    
    # 场景1: 顺序完全正确
    pred_correct = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    target = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    loss_correct = criterion(pred_correct, target)
    print(f"顺序完全正确的损失: {loss_correct.item():.4f}")
    
    # 场景2: 顺序完全错误
    pred_wrong = torch.tensor([[4.0], [3.0], [2.0], [1.0]])
    loss_wrong = criterion(pred_wrong, target)
    print(f"顺序完全错误的损失: {loss_wrong.item():.4f}")
    
    # 场景3: 部分正确
    pred_partial = torch.tensor([[1.0], [3.0], [2.0], [4.0]])
    loss_partial = criterion(pred_partial, target)
    print(f"顺序部分正确的损失: {loss_partial.item():.4f}")
    
    if loss_wrong > loss_partial > loss_correct:
        print("✅ 测试通过: RankLoss 正确惩罚顺序错误")
    else:
        print("❌ 测试失败: 损失大小关系不符合预期")
    
    # 测试梯度
    pred_partial.requires_grad = True
    loss = criterion(pred_partial, target)
    loss.backward()
    
    if pred_partial.grad is not None:
        print(f"梯度计算正常: {pred_partial.grad.shape}")
        print("✅ 测试通过: 梯度反向传播正常")
    else:
        print("❌ 测试失败: 梯度为 None")

def test_supcr_loss():
    """测试监督对比损失"""
    print("\n" + "=" * 60)
    print("测试 4: SupConRegressionLoss")
    print("=" * 60)
    
    criterion = SupConRegressionLoss(temperature=0.1, sigma=2.0)
    
    # 创建测试数据
    batch_size = 8
    feature_dim = 64
    
    # 模拟特征向量
    features = torch.randn(batch_size, feature_dim)
    
    # 创建标签：前4个相似，后4个相似
    labels = torch.tensor([
        [-5.0], [-5.5], [-5.2], [-5.3],  # 高风险组
        [-25.0], [-24.5], [-25.2], [-24.8]  # 低风险组
    ])
    
    # 计算损失
    loss = criterion(features, labels)
    print(f"SupCR 损失: {loss.item():.4f}")
    
    if loss.item() > 0:
        print("✅ 测试通过: SupCR 损失计算正常")
    else:
        print("❌ 测试失败: 损失为负或零")
    
    # 测试梯度
    features.requires_grad = True
    loss = criterion(features, labels)
    loss.backward()
    
    if features.grad is not None:
        print(f"梯度计算正常: {features.grad.shape}")
        print("✅ 测试通过: 梯度反向传播正常")
    else:
        print("❌ 测试失败: 梯度为 None")

def test_combined_loss():
    """测试组合损失"""
    print("\n" + "=" * 60)
    print("测试 5: 组合损失（模拟训练场景）")
    print("=" * 60)
    
    # 初始化损失函数
    criterion_mse = WeightedMSELoss()
    criterion_rank = RankLoss(margin=0.0)
    criterion_sup = SupConRegressionLoss(temperature=0.1, sigma=2.0)
    
    # 模拟一个 batch
    batch_size = 16
    feature_dim = 64
    
    features = torch.randn(batch_size, feature_dim, requires_grad=True)
    pred = torch.randn(batch_size, 1, requires_grad=True)
    target = torch.randn(batch_size, 1)
    weights = torch.ones(batch_size, 1)
    
    # 计算各个损失
    loss_mse = criterion_mse(pred, target, weights)
    loss_rank = criterion_rank(pred, target)
    loss_sup = criterion_sup(features, target)
    
    # 组合损失
    lambda_rank = 0.1
    lambda_sup = 0.5
    total_loss = loss_mse + lambda_rank * loss_rank + lambda_sup * loss_sup
    
    print(f"MSE Loss: {loss_mse.item():.4f}")
    print(f"Rank Loss: {loss_rank.item():.4f}")
    print(f"SupCR Loss: {loss_sup.item():.4f}")
    print(f"Total Loss: {total_loss.item():.4f}")
    
    # 测试反向传播
    total_loss.backward()
    
    if pred.grad is not None and features.grad is not None:
        print("✅ 测试通过: 组合损失的梯度反向传播正常")
    else:
        print("❌ 测试失败: 梯度为 None")

def test_dataset_integration():
    """测试 Dataset 集成"""
    print("\n" + "=" * 60)
    print("测试 6: Dataset 集成")
    print("=" * 60)
    
    from utils.solver import RegressionSequenceDataset
    
    # 创建测试数据
    data = np.random.randn(100, 10, 20)  # (样本数, 序列长度, 特征维度)
    targets = np.random.randn(100)
    weights = np.random.rand(100)
    
    # 测试带权重的 Dataset
    dataset_with_weights = RegressionSequenceDataset(data, targets, weights=weights)
    x, y, w = dataset_with_weights[0]
    
    print(f"数据形状: {x.shape}")
    print(f"标签形状: {y.shape}")
    print(f"权重形状: {w.shape}")
    
    if x.shape == (10, 20) and y.shape == (1,) and w.shape == (1,):
        print("✅ 测试通过: 带权重的 Dataset 返回正确")
    else:
        print("❌ 测试失败: 返回形状不正确")
    
    # 测试不带权重的 Dataset
    dataset_no_weights = RegressionSequenceDataset(data, targets, weights=None)
    x, y, w = dataset_no_weights[0]
    
    if w.item() == 1.0:
        print("✅ 测试通过: 无权重时默认为 1.0")
    else:
        print("❌ 测试失败: 默认权重不正确")

def main():
    """运行所有测试"""
    print("\n" + "🚀" * 30)
    print("开始测试损失函数集成")
    print("🚀" * 30 + "\n")
    
    try:
        # 测试 1: LDS 权重
        test_lds_weights()
        
        # 测试 2: 加权 MSE
        test_weighted_mse()
        
        # 测试 3: 排序损失
        test_rank_loss()
        
        # 测试 4: 监督对比损失
        test_supcr_loss()
        
        # 测试 5: 组合损失
        test_combined_loss()
        
        # 测试 6: Dataset 集成
        test_dataset_integration()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成！")
        print("=" * 60)
        print("\n建议:")
        print("1. 如果所有测试通过，可以开始训练")
        print("2. 运行: python main.py --model LSTM --lambda_rank 0.1 --epoch 5")
        print("3. 查看 LOSS_INTEGRATION_GUIDE.md 了解详细使用方法")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
