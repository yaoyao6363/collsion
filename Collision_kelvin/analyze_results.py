"""
分析训练结果
运行: python analyze_results.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_training_results(checkpoint_dir='./results/checkpoint'):
    """分析所有训练结果"""
    
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"❌ 目录不存在: {checkpoint_dir}")
        return
    
    # 查找所有 metrics.csv
    metrics_files = list(checkpoint_path.glob('*/metrics.csv'))
    
    if not metrics_files:
        print("❌ 没有找到训练结果")
        return
    
    print("=" * 80)
    print("📊 训练结果分析")
    print("=" * 80)
    print(f"找到 {len(metrics_files)} 个训练结果\n")
    
    all_results = []
    
    for metrics_file in sorted(metrics_files):
        model_name = metrics_file.parent.name
        
        try:
            df = pd.read_csv(metrics_file)
            
            if len(df) == 0:
                continue
            
            # 获取最佳和最终结果
            best_idx = df['valid_loss'].idxmin()
            best = df.iloc[best_idx]
            final = df.iloc[-1]
            
            # 计算统计信息
            train_loss_ratio = final['valid_loss'] / final['train_loss']
            
            result = {
                'model': model_name,
                'total_epochs': len(df),
                'best_epoch': best['epoch'],
                'final_train_loss': final['train_loss'],
                'final_valid_loss': final['valid_loss'],
                'loss_ratio': train_loss_ratio,
                'best_MSE': best['MSE'],
                'best_MAE': best['MAE'],
                'best_RMSE': best['RMSE'],
                'best_MAPE': best['MAPE'],
                'best_R2': best['R2'],
                'best_F2': best['F2'],
                'final_R2': final['R2'],
                'final_F2': final['F2'],
            }
            
            all_results.append(result)
            
            # 打印详细信息
            print(f"🔹 {model_name}")
            print(f"   训练轮数: {len(df)} epochs")
            print(f"   最佳 epoch: {best['epoch']}")
            print(f"   最终训练损失: {final['train_loss']:.4f}")
            print(f"   最终验证损失: {final['valid_loss']:.4f}")
            print(f"   损失比例: {train_loss_ratio:.2f}x", end="")
            
            if train_loss_ratio < 3:
                print(" ✅ 健康")
            elif train_loss_ratio < 5:
                print(" ⚠️  轻微过拟合")
            elif train_loss_ratio < 10:
                print(" ⚠️  过拟合")
            else:
                print(" ❌ 严重过拟合")
            
            print(f"\n   📈 最佳性能 (epoch {best['epoch']}):")
            print(f"      MSE: {best['MSE']:.4f} | MAE: {best['MAE']:.4f} | RMSE: {best['RMSE']:.4f}")
            print(f"      MAPE: {best['MAPE']:.2f}% | R²: {best['R2']:.4f} | F2: {best['F2']:.4f}")
            
            print(f"\n   📊 最终性能 (epoch {final['epoch']}):")
            print(f"      R²: {final['R2']:.4f} | F2: {final['F2']:.4f}")
            
            # 绘制训练曲线
            plot_training_curve(df, model_name, checkpoint_path / model_name)
            
            print()
            
        except Exception as e:
            print(f"❌ 分析 {metrics_file} 失败: {e}\n")
    
    # 生成对比表格
    if all_results:
        print("=" * 80)
        print("📋 性能对比")
        print("=" * 80)
        
        results_df = pd.DataFrame(all_results)
        
        # 按 R² 排序
        results_df = results_df.sort_values('best_R2', ascending=False)
        
        print("\n按 R² 排序:")
        print(results_df[['model', 'best_R2', 'best_F2', 'best_MSE', 'loss_ratio']].to_string(index=False))
        
        print("\n按 F2 排序:")
        results_df_f2 = results_df.sort_values('best_F2', ascending=False)
        print(results_df_f2[['model', 'best_F2', 'best_R2', 'best_MSE', 'loss_ratio']].to_string(index=False))
        
        # 保存对比结果
        results_df.to_csv(checkpoint_path / 'comparison.csv', index=False)
        print(f"\n✅ 对比结果已保存到: {checkpoint_path / 'comparison.csv'}")

def plot_training_curve(df, model_name, save_dir):
    """绘制训练曲线"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name} 训练曲线', fontsize=16)
    
    # 1. 损失曲线
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax.plot(df['epoch'], df['valid_loss'], label='Valid Loss', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('训练/验证损失')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. R² 曲线
    ax = axes[0, 1]
    ax.plot(df['epoch'], df['R2'], label='R²', marker='o', markersize=3, color='green')
    ax.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='目标 (0.7)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R²')
    ax.set_title('R² Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. F2 曲线
    ax = axes[1, 0]
    ax.plot(df['epoch'], df['F2'], label='F2-score', marker='o', markersize=3, color='orange')
    ax.axhline(y=0.85, color='r', linestyle='--', alpha=0.5, label='目标 (0.85)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F2-score')
    ax.set_title('F2-score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. MSE/MAE/RMSE
    ax = axes[1, 1]
    ax.plot(df['epoch'], df['MSE'], label='MSE', marker='o', markersize=3)
    ax.plot(df['epoch'], df['MAE'], label='MAE', marker='s', markersize=3)
    ax.plot(df['epoch'], df['RMSE'], label='RMSE', marker='^', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.set_title('误差指标')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = save_dir / 'training_curve.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 训练曲线已保存: {save_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='分析训练结果')
    parser.add_argument('--dir', type=str, default='./results/checkpoint', 
                        help='checkpoint 目录')
    
    args = parser.parse_args()
    
    analyze_training_results(args.dir)
