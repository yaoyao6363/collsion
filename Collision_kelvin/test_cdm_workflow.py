"""
CDM工作流程测试脚本
验证数据生成、预处理和模型训练的完整流程
"""
import os
import sys
import pandas as pd
import numpy as np

def test_kessler_import():
    """测试Kessler库导入"""
    print("=" * 60)
    print("测试1: Kessler库导入")
    print("=" * 60)
    
    try:
        sys.path.insert(0, './kessler')
        from kessler import CDM, Event, EventDataset
        from kessler.data import kelvins_to_event_dataset
        print("✓ Kessler库导入成功")
        print(f"  - CDM类: {CDM}")
        print(f"  - Event类: {Event}")
        print(f"  - EventDataset类: {EventDataset}")
        return True
    except Exception as e:
        print(f"✗ Kessler库导入失败: {e}")
        return False


def test_data_loading():
    """测试数据加载"""
    print("\n" + "=" * 60)
    print("测试2: 数据加载")
    print("=" * 60)
    
    train_file = './dataset/train_data.csv'
    
    if not os.path.exists(train_file):
        print(f"✗ 训练数据文件不存在: {train_file}")
        return False
    
    try:
        df = pd.read_csv(train_file)
        print(f"✓ 数据加载成功")
        print(f"  - 数据形状: {df.shape}")
        print(f"  - 列数: {len(df.columns)}")
        print(f"  - 前5列: {list(df.columns[:5])}")
        
        # 检查关键列
        required_cols = ['event_id', 'time_to_tca', 'miss_distance']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠ 缺少关键列: {missing_cols}")
        else:
            print(f"✓ 所有关键列都存在")
        
        return True
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False


def test_cdm_generation():
    """测试CDM数据生成"""
    print("\n" + "=" * 60)
    print("测试3: CDM数据生成 (小规模测试)")
    print("=" * 60)
    
    try:
        sys.path.insert(0, './kessler')
        from kessler.data import kelvins_to_event_dataset
        
        # 只生成少量事件进行测试
        print("正在从Kelvins数据生成CDM (10个事件)...")
        event_dataset = kelvins_to_event_dataset(
            file_name='./dataset/train_data.csv',
            num_events=10,
            remove_outliers=True
        )
        
        print(f"✓ CDM生成成功")
        print(f"  - 事件数: {len(event_dataset)}")
        print(f"  - 平均每事件CDM数: {event_dataset.event_lengths_mean:.2f}")
        print(f"  - CDM数范围: {event_dataset.event_lengths_min} - {event_dataset.event_lengths_max}")
        
        # 转换为DataFrame
        df = event_dataset.to_dataframe()
        print(f"  - DataFrame形状: {df.shape}")
        
        return True
    except Exception as e:
        print(f"✗ CDM生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing():
    """测试数据预处理"""
    print("\n" + "=" * 60)
    print("测试4: 数据预处理")
    print("=" * 60)
    
    # 创建测试数据
    test_data = {
        'event_id': [0, 0, 1, 1],
        '__DAYS_TO_TCA': [7.0, 6.0, 7.0, 5.0],
        'MISS_DISTANCE': [1000, 900, 1200, 800],
        'RELATIVE_SPEED': [7800, 7850, 7900, 7950],
        'RELATIVE_POSITION_R': [100, 90, 110, 85],
        'RELATIVE_POSITION_T': [200, 180, 220, 170],
        'RELATIVE_POSITION_N': [50, 45, 55, 40],
        'RELATIVE_VELOCITY_R': [10, 12, 11, 13],
        'RELATIVE_VELOCITY_T': [20, 22, 21, 23],
        'RELATIVE_VELOCITY_N': [5, 6, 5.5, 6.5],
        'OBJECT1_CR_R': [100, 110, 105, 115],
        'OBJECT1_CT_T': [200, 210, 205, 215],
        'OBJECT1_CN_N': [50, 55, 52, 58],
        'COLLISION_PROBABILITY': [0.001, 0.002, 0.0015, 0.003]
    }
    
    df = pd.DataFrame(test_data)
    
    try:
        from sklearn.preprocessing import StandardScaler
        
        # 特征工程
        df['relative_distance'] = np.sqrt(
            df['RELATIVE_POSITION_R']**2 + 
            df['RELATIVE_POSITION_T']**2 + 
            df['RELATIVE_POSITION_N']**2
        )
        
        df['object1_uncertainty'] = np.sqrt(
            df['OBJECT1_CR_R'] + df['OBJECT1_CT_T'] + df['OBJECT1_CN_N']
        )
        
        # 标准化
        feature_cols = ['MISS_DISTANCE', 'RELATIVE_SPEED', 'relative_distance', 'object1_uncertainty']
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        
        print(f"✓ 数据预处理成功")
        print(f"  - 原始特征数: {len(test_data.keys())}")
        print(f"  - 工程特征数: 2")
        print(f"  - 标准化特征数: {len(feature_cols)}")
        print(f"  - 最终数据形状: {df.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 数据预处理失败: {e}")
        return False


def test_model_import():
    """测试模型导入"""
    print("\n" + "=" * 60)
    print("测试5: 模型导入")
    print("=" * 60)
    
    try:
        from models.lstm import LSTMModel
        from models.transformer import TransformerModel
        
        print(f"✓ 模型导入成功")
        print(f"  - LSTM模型: {LSTMModel}")
        print(f"  - Transformer模型: {TransformerModel}")
        
        # 测试模型实例化
        import torch
        lstm = LSTMModel(input_size=10, hidden_size=32, num_layers=2)
        transformer = TransformerModel(input_size=10, d_model=64, nhead=2, num_layers=2)
        
        print(f"✓ 模型实例化成功")
        print(f"  - LSTM参数数: {sum(p.numel() for p in lstm.parameters()):,}")
        print(f"  - Transformer参数数: {sum(p.numel() for p in transformer.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"✗ 模型导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_class():
    """测试数据集类"""
    print("\n" + "=" * 60)
    print("测试6: 数据集类")
    print("=" * 60)
    
    try:
        from utils.dataset import CDMDataset
        import torch
        
        # 检查是否有处理后的数据
        if os.path.exists('./dataset/train_data.csv'):
            dataset = CDMDataset(
                data_path='./dataset/',
                dataset_name='train_data.csv',
                n_latest_cdms=13,
                is_train=True
            )
            
            print(f"✓ 数据集类初始化成功")
            print(f"  - 数据集大小: {len(dataset)}")
            print(f"  - 特征维度: {dataset.feature_dim}")
            
            # 测试数据加载
            if len(dataset) > 0:
                data, target, mask = dataset[0]
                print(f"  - 单个样本形状: {data.shape}")
                print(f"  - 目标形状: {target.shape}")
                print(f"  - 掩码形状: {mask.shape}")
            
            return True
        else:
            print("⚠ 训练数据不存在，跳过数据集类测试")
            return True
            
    except Exception as e:
        print(f"✗ 数据集类测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_workflow():
    """测试完整工作流程"""
    print("\n" + "=" * 60)
    print("测试7: 完整工作流程 (端到端)")
    print("=" * 60)
    
    try:
        # 1. 数据生成 (使用小样本)
        print("\n[1/4] 生成CDM数据...")
        sys.path.insert(0, './kessler')
        from kessler.data import kelvins_to_event_dataset
        
        event_dataset = kelvins_to_event_dataset(
            file_name='./dataset/train_data.csv',
            num_events=5,
            remove_outliers=True
        )
        print(f"  ✓ 生成了 {len(event_dataset)} 个事件")
        
        # 2. 转换为DataFrame
        print("\n[2/4] 转换为DataFrame...")
        df = event_dataset.to_dataframe()
        event_ids = []
        for i, event in enumerate(event_dataset):
            event_ids.extend([i] * len(event))
        df['event_id'] = event_ids
        print(f"  ✓ DataFrame形状: {df.shape}")
        
        # 3. 预处理
        print("\n[3/4] 预处理数据...")
        from sklearn.preprocessing import StandardScaler
        
        feature_cols = ['MISS_DISTANCE', 'RELATIVE_SPEED']
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        if feature_cols:
            scaler = StandardScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            print(f"  ✓ 标准化了 {len(feature_cols)} 个特征")
        
        # 4. 保存测试数据
        print("\n[4/4] 保存测试数据...")
        test_output = './test_workflow_output.csv'
        df.to_csv(test_output, index=False)
        print(f"  ✓ 数据已保存至: {test_output}")
        
        print("\n✓ 完整工作流程测试成功!")
        return True
        
    except Exception as e:
        print(f"\n✗ 完整工作流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("CDM工作流程测试套件")
    print("=" * 60)
    
    tests = [
        ("Kessler库导入", test_kessler_import),
        ("数据加载", test_data_loading),
        ("CDM数据生成", test_cdm_generation),
        ("数据预处理", test_preprocessing),
        ("模型导入", test_model_import),
        ("数据集类", test_dataset_class),
        ("完整工作流程", test_full_workflow),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ 测试 '{name}' 出现异常: {e}")
            results.append((name, False))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")
    
    print("\n" + "=" * 60)
    print(f"总计: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 所有测试通过! 你可以开始使用CDM工作流程了。")
        print("\n下一步:")
        print("1. 运行 quick_start_cdm.bat 开始完整流程")
        print("2. 或查看 README_CDM_WORKFLOW.md 了解详细说明")
    else:
        print("\n⚠ 部分测试失败，请检查错误信息并修复问题。")
        print("\n常见问题:")
        print("1. 确保已安装Kessler库: cd kessler && pip install -e .")
        print("2. 确保数据文件存在: ./dataset/train_data.csv")
        print("3. 检查Python版本 >= 3.9")


if __name__ == '__main__':
    main()
