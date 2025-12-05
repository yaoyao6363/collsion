import os
import warnings
import pandas as pd
import numpy as np
from sklearn import preprocessing

# Suppress all warnings
warnings.filterwarnings("ignore")

# 将风险值划分为不同的风险等级
def split_risk_range(df):
    df['risk_range'] = -1
    # Assign risk ranges based on conditions
    df.loc[df['risk'] > -6, 'risk_range'] = 0
    df.loc[(df['risk'] <= -6) & (df['risk'] >= -20), 'risk_range'] = 1
    df.loc[df['risk'] <= -20, 'risk_range'] = 1
    df.loc[df['risk'] == -30, 'risk_range'] = 1
    return df

# 删除TCA前2天没有数据的事件
def remove_shorter_events(df):
    grouped = df.groupby('event_id')
    valid_event_ids = grouped.filter(lambda group: (group['time_to_tca'] > 2).any() and
                                                   (group['time_to_tca'] <= 2).any())['event_id'].unique()
    df_filtered = df[df['event_id'].isin(valid_event_ids)]
    print(f'原始数据集事件数: {df["event_id"].nunique()}, 过滤后事件数: {len(valid_event_ids)}')

    # 符合事件“0” ，“不符合事件”1
    events_in_filtered = df_filtered['event_id'].unique()
    df['short_event'] = 1
    df.loc[df['event_id'].isin(events_in_filtered), 'short_event'] = 0

    return df_filtered

# 为数据集中的每个事件分配风险类别
def assign_risk_category(data_frame):
    data_frame['risk_category'] = -1  # 风险分类，初始化为 -1

    chosen_indices = (
        data_frame[data_frame['time_to_tca'] >= 2].groupby('event_id')['time_to_tca'].idxmin()  # Find max index where time_to_tca <= 2
        .fillna(data_frame[data_frame['time_to_tca'] > 2].groupby('event_id')['time_to_tca'].idxmin())# Fill NaNs with min index where time_to_tca < 2
        )  # # 找到每个事件的“time_to_tca”等于 2 或下一个小于 2 的最小值的行索引

    # chosen_indices = chosen_indices.dropna().astype(int)  # Drop NaNs and convert to int

    chosen_rows = data_frame.loc[chosen_indices] # 使用选定的索引更新“risk_category”并删除不必要的行

    chosen_rows = chosen_rows.set_index('event_id')  # 将 chosen_rows 与 event_id 对齐，以确保长度匹配
    data_frame = data_frame.set_index('event_id')

    # The risk range of the grounf truth will be used as risk category
    # Update 'risk_category' based on chosen rows
    data_frame['risk_category'] = chosen_rows['risk_range']

    data_frame = data_frame.reset_index(level=0)
    grouped_data = data_frame.groupby('event_id')

    def filter_rows(group):
        Two_days_before_tca = group[group['time_to_tca'] > 2]
        return group[group['time_to_tca'] >= Two_days_before_tca['time_to_tca'].min()]

    filtered_data_frame = grouped_data.apply(filter_rows)

    return filtered_data_frame.reset_index(drop=True)

# 为数据集中的每个事件分配风险变化率。
def assign_risk_change_rate(data_frame):
    data_frame['risk_change_rate'] = data_frame.groupby('event_id')['risk'].diff() / data_frame.groupby('event_id')['time_to_tca'].diff()
    data_frame['risk_change_rate'].fillna(0, inplace=True)
    return data_frame

# 处理缺失值
def handle_missing_values(dataframe):
    dataframe['c_time_lastob_end'].fillna(-1, inplace=True)
    dataframe['c_time_lastob_start'].fillna(-1, inplace=True)   # 指定列，填充缺失值为-1
    dataframe = dataframe.dropna(axis = 1)  # 其余删除含有缺失值的列
    dataframe.to_feather('processed_dataset.feather')
    return dataframe

# 数据预处理
def pre_data(file):
    df = pd.read_csv(file)
    encoder = preprocessing.LabelEncoder()
    df['c_object_type'] = encoder.fit_transform(df['c_object_type'])

    df = split_risk_range(df)  # 风险分级 'risk_range'
    df = remove_shorter_events(df) # 删除TCA前2天且2天内没有数据的事件
    df = assign_risk_category(df)  # 分配风险类别 'risk_category'
    df = assign_risk_change_rate(df)  # 分配风险变化率
    df = handle_missing_values(df)  # 处理缺失值

    return df

