"""
CDM数据生成脚本
使用Kessler库的Conjunction类生成合成的碰撞数据消息
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import pyro
from datetime import datetime

# 添加kessler路径
sys.path.insert(0, './kessler')

from kessler import Conjunction, CDM, Event, EventDataset
from kessler import util


def generate_single_conjunction_event(conj_model, event_id=0, verbose=False):
    """
    生成单个碰撞事件
    
    Args:
        conj_model: Conjunction模型实例
        event_id: 事件ID
        verbose: 是否打印详细信息
    
    Returns:
        Event对象或None
    """
    try:
        # 使用Pyro的Importance采样生成一个场景
        trace = pyro.infer.Importance(conj_model.forward, num_samples=1).run()
        
        # 检查是否生成了有效的碰撞
        if 'conj' not in trace.nodes:
            if verbose:
                print(f"Event {event_id}: No conjunction generated")
            return None
        
        conj_flag = trace.nodes['conj']['value']
        if not conj_flag:
            if verbose:
                print(f"Event {event_id}: Conjunction flag is False")
            return None
        
        # 从trace中提取CDM数据
        # 注意: 这里需要根据Conjunction.forward的实际实现来提取CDM
        # 以下是示例代码，可能需要根据实际情况调整
        
        cdms = []
        # 如果Conjunction.forward返回了CDM列表，从trace中提取
        if 'cdms' in trace.nodes:
            cdms = trace.nodes['cdms']['value']
        
        if not cdms:
            if verbose:
                print(f"Event {event_id}: No CDMs generated")
            return None
        
        event = Event(cdms=cdms)
        if verbose:
            print(f"Event {event_id}: Generated {len(cdms)} CDMs")
        
        return event
        
    except Exception as e:
        if verbose:
            print(f"Event {event_id}: Error - {str(e)}")
        return None


def generate_synthetic_cdm_dataset(
    num_events=1000,
    output_dir='./synthetic_cdms/',
    save_format='csv',
    time0=58991.90384230018,
    max_duration_days=7.0,
    miss_dist_threshold=5e3,
    mc_samples=100,
    verbose=True
):
    """
    批量生成合成CDM数据集
    
    Args:
        num_events: 要生成的事件数量
        output_dir: 输出目录
        save_format: 保存格式 ('kvn', 'csv', 或 'both')
        time0: 起始时间 (MJD)
        max_duration_days: 模拟持续时间
        miss_dist_threshold: 碰撞距离阈值 (米)
        mc_samples: 蒙特卡洛采样数
        verbose: 是否打印详细信息
    
    Returns:
        EventDataset对象
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("CDM数据生成器")
    print("=" * 60)
    print(f"目标事件数: {num_events}")
    print(f"输出目录: {output_dir}")
    print(f"保存格式: {save_format}")
    print(f"起始时间 (MJD): {time0}")
    print(f"模拟时长: {max_duration_days} 天")
    print(f"碰撞阈值: {miss_dist_threshold} 米")
    print("=" * 60)
    
    # 初始化Conjunction模型
    conj = Conjunction(
        time0=time0,
        max_duration_days=max_duration_days,
        time_resolution=6e5,
        time_upsample_factor=100,
        miss_dist_threshold=miss_dist_threshold,
        mc_samples=mc_samples,
        mc_upsample_factor=100,
        cdm_update_every_hours=8.0,
        collision_threshold=70,
        pc_method='MC',
        up_method='MC'
    )
    
    events = []
    successful_events = 0
    
    # 使用进度条
    util.progress_bar_init('生成CDM事件', num_events, '事件')
    
    for i in range(num_events):
        util.progress_bar_update(i)
        
        event = generate_single_conjunction_event(conj, event_id=i, verbose=False)
        
        if event is not None and len(event) > 0:
            events.append(event)
            successful_events += 1
            
            # 保存为KVN格式
            if save_format in ['kvn', 'both']:
                for j, cdm in enumerate(event):
                    filename = os.path.join(output_dir, f'event{i}_{j}.kvn')
                    cdm.save(filename)
    
    util.progress_bar_end()
    
    # 创建EventDataset
    event_dataset = EventDataset(events=events)
    
    print(f"\n成功生成 {successful_events}/{num_events} 个事件")
    print(f"总CDM数: {sum(len(e) for e in events)}")
    
    # 保存为CSV格式
    if save_format in ['csv', 'both'] and len(events) > 0:
        print("\n转换为CSV格式...")
        df = event_dataset.to_dataframe()
        
        # 添加event_id列
        event_ids = []
        for i, event in enumerate(events):
            event_ids.extend([i] * len(event))
        df['event_id'] = event_ids
        
        csv_path = os.path.join(output_dir, 'synthetic_cdms.csv')
        df.to_csv(csv_path, index=False)
        print(f"CSV文件已保存: {csv_path}")
        print(f"数据形状: {df.shape}")
        print(f"特征数: {len(df.columns)}")
    
    return event_dataset


def generate_from_kelvins_data(
    input_csv='./dataset/train_data.csv',
    output_dir='./processed_cdms/',
    num_events=None,
    remove_outliers=True
):
    """
    从Kelvins竞赛数据生成CDM数据集
    
    Args:
        input_csv: 输入CSV文件路径
        output_dir: 输出目录
        num_events: 要处理的事件数 (None表示全部)
        remove_outliers: 是否移除异常值
    
    Returns:
        EventDataset对象
    """
    print("=" * 60)
    print("从Kelvins数据生成CDM")
    print("=" * 60)
    print(f"输入文件: {input_csv}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用kessler的数据加载函数
    from kessler.data import kelvins_to_event_dataset
    
    event_dataset = kelvins_to_event_dataset(
        file_name=input_csv,
        num_events=num_events,
        remove_outliers=remove_outliers,
        drop_features=['c_rcs_estimate', 't_rcs_estimate']
    )
    
    # 保存为CSV
    df = event_dataset.to_dataframe()
    
    # 添加event_id
    event_ids = []
    for i, event in enumerate(event_dataset):
        event_ids.extend([i] * len(event))
    df['event_id'] = event_ids
    
    output_csv = os.path.join(output_dir, 'kelvins_cdms.csv')
    df.to_csv(output_csv, index=False)
    
    print(f"\n处理完成!")
    print(f"事件数: {len(event_dataset)}")
    print(f"总CDM数: {len(df)}")
    print(f"输出文件: {output_csv}")
    
    return event_dataset


def main():
    parser = argparse.ArgumentParser(description='CDM数据生成器')
    parser.add_argument('--mode', type=str, default='synthetic', 
                        choices=['synthetic', 'kelvins'],
                        help='生成模式: synthetic(合成数据) 或 kelvins(从Kelvins数据转换)')
    parser.add_argument('--num_events', type=int, default=1000,
                        help='要生成的事件数量')
    parser.add_argument('--output_dir', type=str, default='./synthetic_cdms/',
                        help='输出目录')
    parser.add_argument('--save_format', type=str, default='csv',
                        choices=['kvn', 'csv', 'both'],
                        help='保存格式')
    parser.add_argument('--input_csv', type=str, default='./dataset/train_data.csv',
                        help='Kelvins模式下的输入CSV文件')
    parser.add_argument('--time0', type=float, default=58991.90384230018,
                        help='起始时间 (MJD)')
    parser.add_argument('--max_duration_days', type=float, default=7.0,
                        help='模拟持续时间 (天)')
    parser.add_argument('--miss_dist_threshold', type=float, default=5000.0,
                        help='碰撞距离阈值 (米)')
    parser.add_argument('--mc_samples', type=int, default=100,
                        help='蒙特卡洛采样数')
    
    args = parser.parse_args()
    
    if args.mode == 'synthetic':
        dataset = generate_synthetic_cdm_dataset(
            num_events=args.num_events,
            output_dir=args.output_dir,
            save_format=args.save_format,
            time0=args.time0,
            max_duration_days=args.max_duration_days,
            miss_dist_threshold=args.miss_dist_threshold,
            mc_samples=args.mc_samples
        )
    elif args.mode == 'kelvins':
        dataset = generate_from_kelvins_data(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            num_events=args.num_events
        )
    
    print("\n" + "=" * 60)
    print("数据生成完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
