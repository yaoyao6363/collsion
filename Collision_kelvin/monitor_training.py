"""
实时监控训练进度
运行: python monitor_training.py
"""

import os
import time
import pandas as pd
from pathlib import Path

def monitor_training(checkpoint_dir='./results/checkpoint', interval=5):
    """
    监控训练进度
    
    Args:
        checkpoint_dir: checkpoint 目录
        interval: 检查间隔（秒）
    """
    print("🔍 开始监控训练进度...")
    print(f"检查目录: {checkpoint_dir}")
    print(f"刷新间隔: {interval}秒")
    print("按 Ctrl+C 停止监控\n")
    
    last_metrics = {}
    
    try:
        while True:
            # 查找所有 metrics.csv 文件
            checkpoint_path = Path(checkpoint_dir)
            if not checkpoint_path.exists():
                print(f"⚠️  目录不存在: {checkpoint_dir}")
                time.sleep(interval)
                continue
            
            metrics_files = list(checkpoint_path.glob('*/metrics.csv'))
            
            if not metrics_files:
                print("⏳ 等待训练开始...")
                time.sleep(interval)
                continue
            
            # 清屏（Windows）
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("=" * 80)
            print("📊 训练监控面板")
            print("=" * 80)
            print(f"更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            for metrics_file in sorted(metrics_files):
                model_name = metrics_file.parent.name
                
                try:
                    df = pd.read_csv(metrics_file)
                    
                    if len(df) == 0:
                        continue
                    
                    # 获取最新的指标
                    latest = df.iloc[-1]
                    
                    # 计算训练/验证损失比例
                    loss_ratio = latest['valid_loss'] / latest['train_loss'] if latest['train_loss'] > 0 else float('inf')
                    
                    # 判断训练状态
                    if loss_ratio < 3:
                        status = "✅ 健康"
                    elif loss_ratio < 5:
                        status = "⚠️  轻微过拟合"
                    elif loss_ratio < 10:
                        status = "⚠️  过拟合"
                    else:
                        status = "❌ 严重过拟合"
                    
                    print(f"🔹 模型: {model_name}")
                    print(f"   Epoch: {latest['epoch']}/{df['epoch'].max()}")
                    print(f"   训练损失: {latest['train_loss']:.4f}")
                    print(f"   验证损失: {latest['valid_loss']:.4f}")
                    print(f"   损失比例: {loss_ratio:.2f}x - {status}")
                    print(f"   MSE: {latest['MSE']:.4f} | MAE: {latest['MAE']:.4f} | RMSE: {latest['RMSE']:.4f}")
                    print(f"   MAPE: {latest['MAPE']:.2f}% | R²: {latest['R2']:.4f} | F2: {latest['F2']:.4f}")
                    
                    # 检查是否有改善
                    if model_name in last_metrics:
                        prev = last_metrics[model_name]
                        
                        # R² 改善
                        if latest['R2'] > prev['R2']:
                            print(f"   📈 R² 提升: {prev['R2']:.4f} → {latest['R2']:.4f} (+{latest['R2']-prev['R2']:.4f})")
                        
                        # F2 改善
                        if latest['F2'] > prev['F2']:
                            print(f"   📈 F2 提升: {prev['F2']:.4f} → {latest['F2']:.4f} (+{latest['F2']-prev['F2']:.4f})")
                        
                        # 验证损失下降
                        if latest['valid_loss'] < prev['valid_loss']:
                            print(f"   📉 验证损失下降: {prev['valid_loss']:.4f} → {latest['valid_loss']:.4f}")
                    
                    last_metrics[model_name] = latest
                    print()
                    
                except Exception as e:
                    print(f"⚠️  读取 {metrics_file} 失败: {e}\n")
            
            print("=" * 80)
            print(f"下次更新: {interval}秒后...")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n✅ 监控已停止")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='监控训练进度')
    parser.add_argument('--dir', type=str, default='./results/checkpoint', 
                        help='checkpoint 目录')
    parser.add_argument('--interval', type=int, default=5, 
                        help='刷新间隔（秒）')
    
    args = parser.parse_args()
    
    monitor_training(args.dir, args.interval)
