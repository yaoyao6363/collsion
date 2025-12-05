"""
损失函数使用示例
演示如何在训练脚本中使用新的损失函数

运行: python example_usage.py
"""

import torch
import numpy as np
from utils.loss import calculate_lds_weights, WeightedMSELoss, RankLoss, SupConRegressionLoss

def example_1_lds_weights():
    """示例1: 计算 LDS 权重"""
    print("=" * 60)
    print("示例 1: 计算 LDS 权重")
    print("=" * 60)
    
    # 模拟训练集的风险值
    # 实际使用时，这应该是 y_train
    train_targets = np.random.uniform(-30, -2, 1000)
    
    # 计算 LDS 权重
    weights = calculate_lds_weights(
        train_targets,
        n_bins=100,        # 直方图分箱数
        kernel='gaussian', # 平滑核类型
        ks=5,              # 核窗口大小
        sigma=2            # 高斯带宽
    )
    
    print(f"训练样本数: {len(train_targets)}")
    print(f"权重统计:")
    print(f"  Min: {weights.min():.3f}")
    print(f"  Max: {weights.max():.3f}")
    print(f"  Mean: {weights.mean():.3f}")
    print(f"  Std: {weights.std():.3f}")
    
    # 验证高风险样本是否获得更高权重
    high_risk_mask = train_targets > -6
    print(f"\n高风险样本 (>-6): {high_risk_mask.sum()} 个")
    print(f"高风险样本平均权重: {weights[high_risk_mask].mean():.3f}")
    print(f"低风险样本平均权重: {weights[~high_risk_mask].mean():.3f}")
    
    return weights

def example_2_weighted_mse():
    """示例2: 使用加权 MSE"""
    print("\n" + "=" * 60)
    print("示例 2: 使用加权 MSE")
    print("=" * 60)
    
    # 初始化损失函数
    criterion = WeightedMSELoss()
    
    # 模拟一个 batch 的数据
    batch_size = 32
    pred = torch.randn(batch_size, 1)
    target = torch.randn(batch_size, 1)
    weights = torch.rand(batch_size, 1)  # 从 LDS 计算得到
    
    # 计算损失
    loss = criterion(pred, target, weights)
    
    print(f"Batch size: {batch_size}")
    print(f"Loss: {loss.item():.4f}")
    
    # 对比不加权的损失
    loss_unweighted = criterion(pred, target, weights=None)
    print(f"Unweighted loss: {loss_unweighted.item():.4f}")
    print(f"Difference: {abs(loss.item() - loss_unweighted.item()):.4f}")

def example_3_rank_loss():
    """示例3: 使用排序损失"""
    print("\n" + "=" * 60)
    print("示例 3: 使用排序损失")
    print("=" * 60)
    
    # 初始化损失函数
    criterion = RankLoss(margin=0.0)
    
    # 模拟预测和真实值
    pred = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    target = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    
    # 计算损失
    loss = criterion(pred, target)
    
    print(f"预测值: {pred.squeeze().tolist()}")
    print(f"真实值: {target.squeeze().tolist()}")
    print(f"RankLoss: {loss.item():.4f}")
    
    # 测试顺序错误的情况
    pred_wrong = torch.tensor([[4.0], [3.0], [2.0], [1.0]])
    loss_wrong = criterion(pred_wrong, target)
    print(f"\n顺序错误时的损失: {loss_wrong.item():.4f}")
    print(f"损失增加: {loss_wrong.item() - loss.item():.4f}")

def example_4_supcr_loss():
    """示例4: 使用监督对比损失"""
    print("\n" + "=" * 60)
    print("示例 4: 使用监督对比损失")
    print("=" * 60)
    
    # 初始化损失函数
    criterion = SupConRegressionLoss(temperature=0.1, sigma=2.0)
    
    # 模拟特征向量（从模型的倒数第二层获得）
    batch_size = 16
    feature_dim = 64
    features = torch.randn(batch_size, feature_dim)
    labels = torch.randn(batch_size, 1)
    
    # 计算损失
    loss = criterion(features, labels)
    
    print(f"Batch size: {batch_size}")
    print(f"Feature dim: {feature_dim}")
    print(f"SupCR Loss: {loss.item():.4f}")

def example_5_combined_loss():
    """示例5: 组合使用多个损失"""
    print("\n" + "=" * 60)
    print("示例 5: 组合使用多个损失")
    print("=" * 60)
    
    # 初始化所有损失函数
    criterion_mse = WeightedMSELoss()
    criterion_rank = RankLoss(margin=0.0)
    criterion_sup = SupConRegressionLoss(temperature=0.1, sigma=2.0)
    
    # 模拟一个 batch
    batch_size = 32
    feature_dim = 64
    
    features = torch.randn(batch_size, feature_dim)
    pred = torch.randn(batch_size, 1)
    target = torch.randn(batch_size, 1)
    weights = torch.ones(batch_size, 1)
    
    # 计算各个损失
    loss_mse = criterion_mse(pred, target, weights)
    loss_rank = criterion_rank(pred, target)
    loss_sup = criterion_sup(features, target)
    
    # 设置权重系数
    lambda_rank = 0.1
    lambda_sup = 0.5
    
    # 组合损失
    total_loss = loss_mse + lambda_rank * loss_rank + lambda_sup * loss_sup
    
    print(f"MSE Loss:    {loss_mse.item():.4f}")
    print(f"Rank Loss:   {loss_rank.item():.4f} (× {lambda_rank})")
    print(f"SupCR Loss:  {loss_sup.item():.4f} (× {lambda_sup})")
    print(f"Total Loss:  {total_loss.item():.4f}")
    
    # 显示各个损失的贡献
    mse_contrib = loss_mse.item() / total_loss.item() * 100
    rank_contrib = (lambda_rank * loss_rank.item()) / total_loss.item() * 100
    sup_contrib = (lambda_sup * loss_sup.item()) / total_loss.item() * 100
    
    print(f"\n损失贡献:")
    print(f"  MSE:   {mse_contrib:.1f}%")
    print(f"  Rank:  {rank_contrib:.1f}%")
    print(f"  SupCR: {sup_contrib:.1f}%")

def example_6_training_loop():
    """示例6: 完整的训练循环示例"""
    print("\n" + "=" * 60)
    print("示例 6: 完整的训练循环")
    print("=" * 60)
    
    print("""
    # 伪代码：在 solver.py 中的使用方式
    
    class Solver:
        def __init__(self, args):
            # 1. 初始化损失函数
            self.criterion_mse = WeightedMSELoss()
            self.criterion_rank = RankLoss(margin=args.rank_margin)
            self.criterion_sup = SupConRegressionLoss(
                temperature=args.sup_temp,
                sigma=args.sup_sigma
            )
            
            # 2. 设置权重系数
            self.lambda_rank = args.lambda_rank
            self.lambda_sup = args.lambda_sup
            self.use_supcr = args.use_supcr
        
        def _get_loader(self):
            # 3. 计算 LDS 权重（只在训练集）
            train_weights = calculate_lds_weights(y_train)
            
            # 4. 创建 Dataset（传入权重）
            train_dataset = RegressionSequenceDataset(
                x_train, y_train, weights=train_weights
            )
            val_dataset = RegressionSequenceDataset(
                x_val, y_val, weights=None  # 验证集不加权
            )
        
        def _process_batch(self, seq_x, seq_y, weights):
            # 5. 前向传播
            if self.use_supcr:
                pred, features = self.model(seq_x, return_feat=True)
            else:
                pred = self.model(seq_x)
                features = None
            
            # 6. 计算各个损失
            loss_mse = self.criterion_mse(pred, seq_y, weights)
            
            loss_rank = 0
            if self.lambda_rank > 0:
                loss_rank = self.criterion_rank(pred, seq_y)
            
            loss_sup = 0
            if self.use_supcr and features is not None:
                loss_sup = self.criterion_sup(features, seq_y)
            
            # 7. 组合损失
            total_loss = (loss_mse + 
                         self.lambda_rank * loss_rank + 
                         self.lambda_sup * loss_sup)
            
            return total_loss, pred, seq_y
        
        def train(self):
            for epoch in range(self.args.epoch):
                for (seq_x, seq_y, weights) in self.train_loader:
                    # 8. 训练步骤
                    self.optimizer.zero_grad()
                    loss, _, _ = self._process_batch(seq_x, seq_y, weights)
                    loss.backward()
                    self.optimizer.step()
    """)

def example_7_hyperparameter_tuning():
    """示例7: 超参数调优建议"""
    print("\n" + "=" * 60)
    print("示例 7: 超参数调优建议")
    print("=" * 60)
    
    print("""
    # 推荐的超参数搜索空间
    
    1. LDS 参数（通常不需要调整）:
       n_bins: 100 (固定)
       kernel: 'gaussian' (固定)
       ks: 5 (固定)
       sigma: [1.0, 2.0, 3.0]  # 可选调整
    
    2. RankLoss 参数:
       lambda_rank: [0.0, 0.05, 0.1, 0.15, 0.2, 0.5]
       rank_margin: [0.0, 0.5, 1.0]  # 通常 0.0 即可
    
    3. SupConRegressionLoss 参数:
       lambda_sup: [0.0, 0.3, 0.5, 0.7, 1.0]
       sup_temp: [0.05, 0.1, 0.2, 0.5]
       sup_sigma: [1.0, 2.0, 3.0, 5.0]
    
    # 调优策略
    
    策略1: 网格搜索（适合小规模）
    for lambda_rank in [0.05, 0.1, 0.2]:
        for lambda_sup in [0.3, 0.5, 0.7]:
            train_model(lambda_rank, lambda_sup)
    
    策略2: 随机搜索（适合大规模）
    for i in range(50):
        lambda_rank = random.uniform(0.0, 0.5)
        lambda_sup = random.uniform(0.0, 1.0)
        train_model(lambda_rank, lambda_sup)
    
    策略3: 贝叶斯优化（推荐，使用 Optuna）
    import optuna
    
    def objective(trial):
        lambda_rank = trial.suggest_float('lambda_rank', 0.0, 0.5)
        lambda_sup = trial.suggest_float('lambda_sup', 0.0, 1.0)
        sup_temp = trial.suggest_float('sup_temp', 0.05, 0.5)
        
        # 训练并返回验证集 F2-score
        f2_score = train_and_evaluate(lambda_rank, lambda_sup, sup_temp)
        return f2_score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print("Best hyperparameters:", study.best_params)
    """)

def main():
    """运行所有示例"""
    print("\n" + "🎯" * 30)
    print("损失函数使用示例")
    print("🎯" * 30 + "\n")
    
    # 运行所有示例
    example_1_lds_weights()
    example_2_weighted_mse()
    example_3_rank_loss()
    example_4_supcr_loss()
    example_5_combined_loss()
    example_6_training_loop()
    example_7_hyperparameter_tuning()
    
    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成！")
    print("=" * 60)
    print("\n下一步:")
    print("1. 运行测试: python test_loss_integration.py")
    print("2. 查看文档: LOSS_INTEGRATION_GUIDE.md")
    print("3. 开始训练: python main.py --model LSTM --lambda_rank 0.1")

if __name__ == "__main__":
    main()
