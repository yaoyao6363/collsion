import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./dataset/train_data.csv')

# 数据形状和前5行
print("数据形状:", df.shape)
print("\n前5行数据:")
print(df.head())

# 检查缺失值
print("\n缺失值统计:")
missing = df.isnull().mean() * 100
missing = missing[missing > 0].sort_values(ascending=False)

# 缺失值热图（采样前1000行以简化显示）
plt.figure(figsize=(12, 8))
plt.imshow(df.iloc[:2000].isnull(), cmap='viridis', aspect='auto')
plt.title('缺失值热图 (紫色: 无缺失, 黄色: 缺失) - 前1000行')
plt.ylabel('行索引')
plt.xlabel('特征索引')
plt.colorbar(label='缺失 (1) / 无 (0)')
plt.tight_layout()
plt.show()

# 高风险与低风险分布可视化
event_col = 'event_id'
latest_risks = df.loc[df.groupby(event_col)['time_to_tca'].idxmin()]['risk']
# 统计区间
bins = np.arange(np.floor(latest_risks.min()), 1, 1)  # 从最小 risk（e.g., -30）到 0，按整数划分
hist, bin_edges = np.histogram(latest_risks, bins=bins)
labels = [f'[{int(edge)}, {int(edge+1)})' for edge in bin_edges[:-1]]
# 高风险阈值
high_risk_threshold = -4

# 方法1: 对数 y 轴柱状图
# 统计区间
bins = np.arange(np.floor(latest_risks.min()), 1, 1)  # 从最小 risk 到 0，按整数划分
hist, bin_edges = np.histogram(latest_risks, bins=bins)

# 区间标签
labels = [f'[{int(edge)}, {int(edge+1)})' for edge in bin_edges[:-1]]

# 颜色区分
colors = ['red' if edge > high_risk_threshold else 'green' for edge in bin_edges[:-1]]

# 绘制
plt.figure(figsize=(12, 6))
plt.bar(labels, hist, color=colors, edgecolor='black', alpha=0.7)
plt.yscale('log')  # 对数 y 轴
plt.title('按 Risk 区间划分的事件分布 (log10 尺度, 对数 y 轴)')
plt.xlabel('Risk 区间')
plt.ylabel('事件数量 (对数尺度)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加阈值线
if high_risk_threshold in bin_edges[:-1]:
    high_risk_index = np.where(np.array(bin_edges[:-1]) == high_risk_threshold)[0][0]
    plt.axvline(x=high_risk_index, color='black', linestyle='--', label='高风险阈值 (-4)')
    plt.legend()

plt.tight_layout()
plt.savefig('risk_log_bar.png')  # 保存为 PNG
plt.show()

# 方法2: ECDF 图（备用）
plt.figure(figsize=(10, 6))
sns.ecdfplot(latest_risks, color='blue')
plt.title('最终 Risk 值的累积分布函数 (ECDF)')
plt.xlabel('Risk (log10)')
plt.ylabel('累积比例')
plt.axvline(high_risk_threshold, color='red', linestyle='--', label='高风险阈值 (-4)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('risk_ecdf.png')  # 保存为 PNG
plt.show()

# 打印区间统计
print("\n各区间事件数量：")
for label, count in zip(labels, hist):
    print(f"{label}: {count} 事件")
