"""
CDM数据预处理脚本
将原始CDM数据转换为适合机器学习模型的格式
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle
import json


def select_features():
    """
    定义用于风险预测的特征列表
    """
    feature_columns = [
        # 时间特征
        '__CREATION_DATE', '__TCA', '__DAYS_TO_TCA',
        
        # 相对运动特征
        'MISS_DISTANCE', 'RELATIVE_SPEED',
        'RELATIVE_POSITION_R', 'RELATIVE_POSITION_T', 'RELATIVE_POSITION_N',
        'RELATIVE_VELOCITY_R', 'RELATIVE_VELOCITY_T', 'RELATIVE_VELOCITY_N',
        
        # 目标物体状态 (OBJECT1)
        'OBJECT1_X', 'OBJECT1_Y', 'OBJECT1_Z',
        'OBJECT1_X_DOT', 'OBJECT1_Y_DOT', 'OBJECT1_Z_DOT',
        
        # 目标物体协方差矩阵 (位置不确定性)
        'OBJECT1_CR_R', 'OBJECT1_CT_R', 'OBJECT1_CT_T',
        'OBJECT1_CN_R', 'OBJECT1_CN_T', 'OBJECT1_CN_N',
        
        # 目标物体协方差矩阵 (速度不确定性)
        'OBJECT1_CRDOT_R', 'OBJECT1_CRDOT_T', 'OBJECT1_CRDOT_N', 'OBJECT1_CRDOT_RDOT',
        'OBJECT1_CTDOT_R', 'OBJECT1_CTDOT_T', 'OBJECT1_CTDOT_N', 'OBJECT1_CTDOT_RDOT', 'OBJECT1_CTDOT_TDOT',
        'OBJECT1_CNDOT_R', 'OBJECT1_CNDOT_T', 'OBJECT1_CNDOT_N', 'OBJECT1_CNDOT_RDOT', 'OBJECT1_CNDOT_TDOT', 'OBJECT1_CNDOT_NDOT',
        
        # 追踪物体状态 (OBJECT2)
        'OBJECT2_X', 'OBJECT2_Y', 'OBJECT2_Z',
        'OBJECT2_X_DOT', 'OBJECT2_Y_DOT', 'OBJECT2_Z_DOT',
        
        # 追踪物体协方差矩阵 (位置不确定性)
        'OBJECT2_CR_R', 'OBJECT2_CT_R', 'OBJECT2_CT_T',
        'OBJECT2_CN_R', 'OBJECT2_CN_T', 'OBJECT2_CN_N',
        
        # 追踪物体协方差矩阵 (速度不确定性)
        'OBJECT2_CRDOT_R', 'OBJECT2_CRDOT_T', 'OBJECT2_CRDOT_N', 'OBJECT2_CRDOT_RDOT',
        'OBJECT2_CTDOT_R', 'OBJECT2_CTDOT_T', 'OBJECT2_CTDOT_N', 'OBJECT2_CTDOT_RDOT', 'OBJECT2_CTDOT_TDOT',
        'OBJECT2_CNDOT_R', 'OBJECT2_CNDOT_T', 'OBJECT2_CNDOT_N', 'OBJECT2_CNDOT_RDOT', 'OBJECT2_CNDOT_TDOT', 'OBJECT2_CNDOT_NDOT',
    ]
    
    return feature_columns


def engineer_features(df):
    """
    特征工程: 创建派生特征
    
    Args:
        df: 输入DataFrame
    
    Returns:
        df: 添加了新特征的DataFrame
        new_features: 新特征名称列表
    """
    new_features = []
    
    # 1. 相对距离 (3D欧氏距离)
    if all(col in df.columns for col in ['RELATIVE_POSITION_R', 'RELATIVE_POSITION_T', 'RELATIVE_POSITION_N']):
        df['relative_distance_3d'] = np.sqrt(
            df['RELATIVE_POSITION_R']**2 + 
            df['RELATIVE_POSITION_T']**2 + 
            df['RELATIVE_POSITION_N']**2
        )
        new_features.append('relative_distance_3d')
    
    # 2. 相对速度大小
    if all(col in df.columns for col in ['RELATIVE_VELOCITY_R', 'RELATIVE_VELOCITY_T', 'RELATIVE_VELOCITY_N']):
        df['relative_velocity_magnitude'] = np.sqrt(
            df['RELATIVE_VELOCITY_R']**2 + 
            df['RELATIVE_VELOCITY_T']**2 + 
            df['RELATIVE_VELOCITY_N']**2
        )
        new_features.append('relative_velocity_magnitude')
    
    # 3. 碰撞时间 (基于距离和速度的简单估计)
    if 'relative_distance_3d' in df.columns and 'relative_velocity_magnitude' in df.columns:
        df['time_to_collision_estimate'] = df['relative_distance_3d'] / (df['relative_velocity_magnitude'] + 1e-10)
        new_features.append('time_to_collision_estimate')
    
    # 4. 目标物体位置不确定性 (协方差矩阵对角元素的和)
    if all(col in df.columns for col in ['OBJECT1_CR_R', 'OBJECT1_CT_T', 'OBJECT1_CN_N']):
        df['object1_position_uncertainty'] = np.sqrt(
            df['OBJECT1_CR_R'] + df['OBJECT1_CT_T'] + df['OBJECT1_CN_N']
        )
        new_features.append('object1_position_uncertainty')
    
    # 5. 目标物体速度不确定性
    if all(col in df.columns for col in ['OBJECT1_CRDOT_RDOT', 'OBJECT1_CTDOT_TDOT', 'OBJECT1_CNDOT_NDOT']):
        df['object1_velocity_uncertainty'] = np.sqrt(
            df['OBJECT1_CRDOT_RDOT'] + df['OBJECT1_CTDOT_TDOT'] + df['OBJECT1_CNDOT_NDOT']
        )
        new_features.append('object1_velocity_uncertainty')
    
    # 6. 追踪物体位置不确定性
    if all(col in df.columns for col in ['OBJECT2_CR_R', 'OBJECT2_CT_T', 'OBJECT2_CN_N']):
        df['object2_position_uncertainty'] = np.sqrt(
            df['OBJECT2_CR_R'] + df['OBJECT2_CT_T'] + df['OBJECT2_CN_N']
        )
        new_features.append('object2_position_uncertainty')
    
    # 7. 追踪物体速度不确定性
    if all(col in df.columns for col in ['OBJECT2_CRDOT_RDOT', 'OBJECT2_CTDOT_TDOT', 'OBJECT2_CNDOT_NDOT']):
        df['object2_velocity_uncertainty'] = np.sqrt(
            df['OBJECT2_CRDOT_RDOT'] + df['OBJECT2_CTDOT_TDOT'] + df['OBJECT2_CNDOT_NDOT']
        )
        new_features.append('object2_velocity_uncertainty')
    
    # 8. 总不确定性 (两个物体的不确定性之和)
    if 'object1_position_uncertainty' in df.columns and 'object2_position_uncertainty' in df.columns:
        df['total_position_uncertainty'] = df['object1_position_uncertainty'] + df['object2_position_uncertainty']
        new_features.append('total_position_uncertainty')
    
    # 9. 相对速度方向特征 (径向、切向、法向分量的比例)
    if 'relative_velocity_magnitude' in df.columns:
        df['velocity_radial_ratio'] = df['RELATIVE_VELOCITY_R'] / (df['relative_velocity_magnitude'] + 1e-10)
        df['velocity_tangential_ratio'] = df['RELATIVE_VELOCITY_T'] / (df['relative_velocity_magnitude'] + 1e-10)
        df['velocity_normal_ratio'] = df['RELATIVE_VELOCITY_N'] / (df['relative_velocity_magnitude'] + 1e-10)
        new_features.extend(['velocity_radial_ratio', 'velocity_tangential_ratio', 'velocity_normal_ratio'])
    
    # 10. 距离变化率 (相对速度在径向方向的投影)
    if all(col in df.columns for col in ['RELATIVE_POSITION_R', 'RELATIVE_VELOCITY_R', 'relative_distance_3d']):
        df['distance_change_rate'] = (
            df['RELATIVE_POSITION_R'] * df['RELATIVE_VELOCITY_R'] +
            df['RELATIVE_POSITION_T'] * df['RELATIVE_VELOCITY_T'] +
            df['RELATIVE_POSITION_N'] * df['RELATIVE_VELOCITY_N']
        ) / (df['relative_distance_3d'] + 1e-10)
        new_features.append('distance_change_rate')
    
    print(f"创建了 {len(new_features)} 个新特征")
    return df, new_features


def handle_missing_values(df, strategy='drop'):
    """
    处理缺失值
    
    Args:
        df: 输入DataFrame
        strategy: 处理策略 ('drop', 'mean', 'median', 'forward_fill')
    
    Returns:
        df: 处理后的DataFrame
    """
    missing_count = df.isnull().sum().sum()
    print(f"缺失值总数: {missing_count}")
    
    if missing_count == 0:
        return df
    
    if strategy == 'drop':
        df = df.dropna()
        print(f"删除缺失值后剩余 {len(df)} 行")
    elif strategy == 'mean':
        df = df.fillna(df.mean())
    elif strategy == 'median':
        df = df.fillna(df.median())
    elif strategy == 'forward_fill':
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df


def remove_outliers(df, feature_columns, method='iqr', threshold=3.0):
    """
    移除异常值
    
    Args:
        df: 输入DataFrame
        feature_columns: 要检查异常值的特征列
        method: 方法 ('iqr', 'zscore')
        threshold: 阈值
    
    Returns:
        df: 移除异常值后的DataFrame
    """
    original_len = len(df)
    
    if method == 'iqr':
        # 使用四分位距 (IQR) 方法
        for col in feature_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    elif method == 'zscore':
        # 使用Z-score方法
        from scipy import stats
        for col in feature_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                z_scores = np.abs(stats.zscore(df[col]))
                df = df[z_scores < threshold]
    
    removed = original_len - len(df)
    print(f"移除了 {removed} 个异常值 ({removed/original_len*100:.2f}%)")
    
    return df


def normalize_features(df, feature_columns, method='standard', save_scaler=True, scaler_path='./scaler.pkl'):
    """
    特征标准化/归一化
    
    Args:
        df: 输入DataFrame
        feature_columns: 要标准化的特征列
        method: 标准化方法 ('standard', 'robust', 'minmax')
        save_scaler: 是否保存scaler
        scaler_path: scaler保存路径
    
    Returns:
        df: 标准化后的DataFrame
        scaler: 拟合的scaler对象
    """
    # 只对存在的数值列进行标准化
    valid_columns = [col for col in feature_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    df[valid_columns] = scaler.fit_transform(df[valid_columns])
    
    if save_scaler:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler已保存至: {scaler_path}")
    
    return df, scaler


def preprocess_cdm_for_risk_prediction(
    input_csv='./synthetic_cdms/synthetic_cdms.csv',
    output_csv='./processed_data.csv',
    config_path='./preprocessing_config.json',
    scaler_path='./scaler.pkl',
    target_column='COLLISION_PROBABILITY',
    missing_strategy='drop',
    outlier_method='iqr',
    outlier_threshold=3.0,
    normalization_method='standard',
    add_engineered_features=True
):
    """
    完整的CDM数据预处理流程
    
    Args:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
        config_path: 配置文件保存路径
        scaler_path: Scaler保存路径
        target_column: 目标变量列名
        missing_strategy: 缺失值处理策略
        outlier_method: 异常值检测方法
        outlier_threshold: 异常值阈值
        normalization_method: 标准化方法
        add_engineered_features: 是否添加工程特征
    
    Returns:
        df: 预处理后的DataFrame
        feature_columns: 最终的特征列表
    """
    print("=" * 60)
    print("CDM数据预处理")
    print("=" * 60)
    print(f"输入文件: {input_csv}")
    print(f"输出文件: {output_csv}")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/7] 加载数据...")
    df = pd.read_csv(input_csv)
    print(f"原始数据形状: {df.shape}")
    print(f"列数: {len(df.columns)}")
    
    # 2. 选择基础特征
    print("\n[2/7] 选择特征...")
    base_features = select_features()
    # 只保留存在的列
    base_features = [col for col in base_features if col in df.columns]
    print(f"基础特征数: {len(base_features)}")
    
    # 3. 特征工程
    if add_engineered_features:
        print("\n[3/7] 特征工程...")
        df, new_features = engineer_features(df)
        all_features = base_features + new_features
    else:
        print("\n[3/7] 跳过特征工程")
        all_features = base_features
    
    print(f"总特征数: {len(all_features)}")
    
    # 4. 处理缺失值
    print(f"\n[4/7] 处理缺失值 (策略: {missing_strategy})...")
    df = handle_missing_values(df, strategy=missing_strategy)
    
    # 5. 移除异常值
    print(f"\n[5/7] 移除异常值 (方法: {outlier_method}, 阈值: {outlier_threshold})...")
    df = remove_outliers(df, all_features, method=outlier_method, threshold=outlier_threshold)
    
    # 6. 标准化
    print(f"\n[6/7] 特征标准化 (方法: {normalization_method})...")
    df, scaler = normalize_features(
        df, all_features, 
        method=normalization_method, 
        save_scaler=True, 
        scaler_path=scaler_path
    )
    
    # 7. 保存处理后的数据
    print("\n[7/7] 保存数据...")
    
    # 确保event_id列存在
    if 'event_id' not in df.columns:
        print("警告: 未找到event_id列，将创建默认event_id")
        df['event_id'] = range(len(df))
    
    # 保存DataFrame
    df.to_csv(output_csv, index=False)
    print(f"数据已保存至: {output_csv}")
    
    # 保存配置
    config = {
        'input_csv': input_csv,
        'output_csv': output_csv,
        'feature_columns': all_features,
        'base_features': base_features,
        'engineered_features': new_features if add_engineered_features else [],
        'target_column': target_column,
        'missing_strategy': missing_strategy,
        'outlier_method': outlier_method,
        'outlier_threshold': outlier_threshold,
        'normalization_method': normalization_method,
        'original_shape': list(pd.read_csv(input_csv).shape),
        'processed_shape': list(df.shape),
        'scaler_path': scaler_path
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"配置已保存至: {config_path}")
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("预处理完成!")
    print("=" * 60)
    print(f"最终数据形状: {df.shape}")
    print(f"特征数: {len(all_features)}")
    print(f"事件数: {df['event_id'].nunique()}")
    if target_column in df.columns:
        print(f"目标变量 ({target_column}) 统计:")
        print(df[target_column].describe())
    print("=" * 60)
    
    return df, all_features


def main():
    parser = argparse.ArgumentParser(description='CDM数据预处理')
    parser.add_argument('--input_csv', type=str, default='./synthetic_cdms/synthetic_cdms.csv',
                        help='输入CSV文件路径')
    parser.add_argument('--output_csv', type=str, default='./processed_data.csv',
                        help='输出CSV文件路径')
    parser.add_argument('--config_path', type=str, default='./preprocessing_config.json',
                        help='配置文件保存路径')
    parser.add_argument('--scaler_path', type=str, default='./scaler.pkl',
                        help='Scaler保存路径')
    parser.add_argument('--target_column', type=str, default='COLLISION_PROBABILITY',
                        help='目标变量列名')
    parser.add_argument('--missing_strategy', type=str, default='drop',
                        choices=['drop', 'mean', 'median', 'forward_fill'],
                        help='缺失值处理策略')
    parser.add_argument('--outlier_method', type=str, default='iqr',
                        choices=['iqr', 'zscore'],
                        help='异常值检测方法')
    parser.add_argument('--outlier_threshold', type=float, default=3.0,
                        help='异常值阈值')
    parser.add_argument('--normalization_method', type=str, default='standard',
                        choices=['standard', 'robust', 'minmax'],
                        help='标准化方法')
    parser.add_argument('--no_feature_engineering', action='store_true',
                        help='不进行特征工程')
    
    args = parser.parse_args()
    
    df, features = preprocess_cdm_for_risk_prediction(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        config_path=args.config_path,
        scaler_path=args.scaler_path,
        target_column=args.target_column,
        missing_strategy=args.missing_strategy,
        outlier_method=args.outlier_method,
        outlier_threshold=args.outlier_threshold,
        normalization_method=args.normalization_method,
        add_engineered_features=not args.no_feature_engineering
    )


if __name__ == '__main__':
    main()
