"""
测试所有模型是否正确支持 return_feat 参数
运行: python test_model_modifications.py
"""

import torch
import argparse
from models import RiskLSTM, RiskTransformer, BayesLSTM
from models.cdea_regressor import CDEARegressor

def create_test_args():
    """创建测试用的 args"""
    args = argparse.Namespace()
    
    # LSTM 参数
    args.hidden_size = 128
    args.num_layers = 2
    args.dropout = 0.1
    
    # Transformer 参数
    args.d_model = 128
    args.nhead = 4
    args.num_layers_tf = 2
    args.dim_ff = 256
    args.tf_dropout = 0.1
    
    # BayesLSTM 参数
    args.bayes_hidden_size = 256
    args.bayes_num_layers = 2
    args.bayes_dropout = 0.2
    
    # CDEA 参数
    args.cdea_layers = 2
    args.cdea_heads_row = 4
    args.cdea_heads_col = 4
    args.cdea_k_neighbors = 8
    args.n_latest_cdms = 13
    
    return args

def test_model(model_class, model_name, input_size=20, test_cdea=False):
    """
    测试模型是否支持 return_feat
    
    Args:
        model_class: 模型类
        model_name: 模型名称
        input_size: 输入特征维度
        test_cdea: 是否测试 CDEA（需要额外参数）
    """
    print(f"\n{'='*70}")
    print(f"测试 {model_name}")
    print('='*70)
    
    try:
        # 创建模型
        args = create_test_args()
        model = model_class(args, input_size)
        model.eval()
        
        # 创建测试数据
        batch_size = 4
        seq_len = 12
        x = torch.randn(batch_size, seq_len, input_size)
        
        print(f"输入形状: {x.shape}")
        
        # 测试1: 不返回特征（默认行为）
        print("\n[测试 1] 不返回特征 (return_feat=False)")
        if test_cdea:
            pred = model(x, return_feat=False)
        else:
            pred = model(x, return_feat=False)
        
        print(f"  ✓ 预测值形状: {pred.shape}")
        assert pred.shape == (batch_size, 1), f"预测值形状错误: 期望 ({batch_size}, 1), 实际 {pred.shape}"
        print(f"  ✓ 预测值范围: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
        
        # 测试2: 返回特征
        print("\n[测试 2] 返回特征 (return_feat=True)")
        if test_cdea:
            result = model(x, return_feat=True)
        else:
            result = model(x, return_feat=True)
        
        # 检查返回值
        if isinstance(result, tuple) and len(result) == 2:
            pred, feat = result
            print(f"  ✓ 返回类型: tuple (pred, feat)")
            print(f"  ✓ 预测值形状: {pred.shape}")
            print(f"  ✓ 特征向量形状: {feat.shape}")
            
            # 验证形状
            assert pred.shape == (batch_size, 1), f"预测值形状错误"
            assert feat.dim() == 2, f"特征应该是2维的，实际是 {feat.dim()} 维"
            assert feat.shape[0] == batch_size, f"特征的 batch 维度错误"
            
            feature_dim = feat.shape[1]
            print(f"  ✓ 特征维度: {feature_dim}")
            print(f"  ✓ 特征范围: [{feat.min().item():.3f}, {feat.max().item():.3f}]")
            
            # 验证特征不是全零
            assert feat.abs().sum() > 0, "特征向量全为零"
            print(f"  ✓ 特征向量非零")
            
        else:
            raise AssertionError(f"return_feat=True 时应该返回 (pred, feat)，实际返回: {type(result)}")
        
        # 测试3: 梯度反向传播
        print("\n[测试 3] 梯度反向传播")
        x_grad = torch.randn(batch_size, seq_len, input_size, requires_grad=True)
        
        if test_cdea:
            pred, feat = model(x_grad, return_feat=True)
        else:
            pred, feat = model(x_grad, return_feat=True)
        
        # 对预测值求和并反向传播
        loss = pred.sum()
        loss.backward()
        
        assert x_grad.grad is not None, "输入梯度为 None"
        print(f"  ✓ 输入梯度形状: {x_grad.grad.shape}")
        print(f"  ✓ 输入梯度范数: {x_grad.grad.norm().item():.6f}")
        
        # 检查模型参数梯度
        param_with_grad = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_with_grad += 1
        
        print(f"  ✓ 有梯度的参数数量: {param_with_grad}")
        
        print(f"\n✅ {model_name} 所有测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ {model_name} 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_models():
    """测试所有模型"""
    print("\n" + "🚀" * 35)
    print("开始测试所有模型的 return_feat 功能")
    print("🚀" * 35)
    
    input_size = 20
    results = {}
    
    # 测试 RiskLSTM
    results['RiskLSTM'] = test_model(RiskLSTM, "RiskLSTM", input_size)
    
    # 测试 RiskTransformer
    results['RiskTransformer'] = test_model(RiskTransformer, "RiskTransformer", input_size)
    
    # 测试 BayesLSTM
    results['BayesLSTM'] = test_model(BayesLSTM, "BayesLSTM", input_size)
    
    # 测试 CDEARegressor
    results['CDEARegressor'] = test_model(CDEARegressor, "CDEARegressor", input_size, test_cdea=True)
    
    # 打印总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    for model_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{model_name:20s} : {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "🎉" * 35)
        print("✅ 所有模型测试通过！")
        print("🎉" * 35)
        print("\n下一步:")
        print("1. 运行完整集成测试: python test_loss_integration.py")
        print("2. 开始训练: python main.py --model LSTM --lambda_rank 0.1 --epoch 10")
    else:
        print("\n" + "⚠️ " * 35)
        print("❌ 部分模型测试失败，请检查上面的错误信息")
        print("⚠️ " * 35)
    
    return all_passed

def test_compatibility():
    """测试向后兼容性（不传 return_feat 参数）"""
    print("\n" + "="*70)
    print("测试向后兼容性")
    print("="*70)
    
    try:
        args = create_test_args()
        input_size = 20
        batch_size = 4
        seq_len = 12
        x = torch.randn(batch_size, seq_len, input_size)
        
        # 测试所有模型的默认行为（不传 return_feat）
        models = [
            (RiskLSTM, "RiskLSTM"),
            (RiskTransformer, "RiskTransformer"),
            (BayesLSTM, "BayesLSTM"),
            (CDEARegressor, "CDEARegressor"),
        ]
        
        for model_class, model_name in models:
            model = model_class(args, input_size)
            model.eval()
            
            # 不传 return_feat 参数（默认行为）
            pred = model(x)
            
            assert pred.shape == (batch_size, 1), f"{model_name}: 默认行为返回形状错误"
            print(f"  ✓ {model_name}: 默认行为正常")
        
        print("\n✅ 向后兼容性测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 向后兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("\n" + "="*70)
    print("模型修改验证脚本")
    print("="*70)
    print("\n本脚本将测试:")
    print("1. 所有模型是否支持 return_feat=True")
    print("2. 返回的特征向量形状是否正确")
    print("3. 梯度反向传播是否正常")
    print("4. 向后兼容性（不传 return_feat 参数）")
    
    # 测试所有模型
    all_passed = test_all_models()
    
    # 测试向后兼容性
    compat_passed = test_compatibility()
    
    # 最终结果
    if all_passed and compat_passed:
        print("\n" + "🎊" * 35)
        print("✅ 所有测试通过！模型修改成功！")
        print("🎊" * 35)
        print("\n你现在可以:")
        print("1. 使用 SupConRegressionLoss 训练模型")
        print("2. 运行: python main.py --model LSTM --use_supcr --lambda_sup 0.5")
        return 0
    else:
        print("\n" + "❌" * 35)
        print("部分测试失败，请检查错误信息")
        print("❌" * 35)
        return 1

if __name__ == "__main__":
    exit(main())
