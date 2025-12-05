import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d

# ==========================================
# 1. LDS (Label Distribution Smoothing) 权重计算工具
# ==========================================

def get_lds_kernel_window(kernel, ks, sigma):
    """
    生成用于平滑的一维核窗口
    """
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base = np.zeros(ks)
        base[half_ks] = 1
        window = gaussian_filter1d(base, sigma=sigma)
    elif kernel == 'triang':
        window = np.hstack([np.arange(1, half_ks + 1), np.arange(half_ks + 1, 0, -1)])
    else:  # laplace
        window = np.exp(-np.abs(np.arange(ks) - half_ks) / sigma)
        
    return window / window.sum()

def calculate_lds_weights(targets, n_bins=100, kernel='gaussian', ks=5, sigma=2):
    """
    计算 LDS 权重:
    1. Binning: 对连续标签进行直方图统计
    2. Smoothing: 对频率分布进行高斯平滑
    3. Reweighting: 取平滑后频率的倒数作为权重
    
    Args:
        targets (np.array or torch.Tensor): 训练集的所有真实标签 (risk values)
        n_bins (int): 直方图的分箱数量
        kernel (str): 平滑核类型 ('gaussian', 'triang', 'laplace')
        ks (int): 核窗口大小 (Kernel Size)
        sigma (float): 高斯核的标准差 (Bandwidth)
        
    Returns:
        weights (np.ndarray): 每个样本对应的权重，形状与 targets 相同
    """
    # 确保输入是 numpy 数组并展平
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    targets = targets.flatten()
    
    # 1. 估计分布 (Histogram)
    hist, bin_edges = np.histogram(targets, bins=n_bins, density=True)
    
    # 获取每个样本所属的 bin 索引
    # np.digitize 返回 1-based 索引，减 1 变为 0-based
    # clip 防止浮点数边界误差导致的索引越界
    bin_indices = np.digitize(targets, bin_edges[1:-1]) 
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # 2. 平滑分布 (Smoothing)
    window = get_lds_kernel_window(kernel, ks, sigma)
    smoothed_hist = np.convolve(hist, window, mode='same')
    
    # 3. 计算权重 (Inverse weighting)
    # 加上极小值避免除以 0
    smoothed_hist = np.maximum(smoothed_hist, 1e-8)
    bin_weights = 1.0 / smoothed_hist
    
    # 归一化权重，使其均值为 1，避免改变 Loss 的量级
    bin_weights = bin_weights / np.mean(bin_weights)
    
    # 映射回每个样本
    sample_weights = bin_weights[bin_indices]
    
    return sample_weights


# ==========================================
# 2. 改进的损失函数模块
# ==========================================

class WeightedMSELoss(nn.Module):
    """
    支持样本权重的均方误差损失 (Weighted Mean Squared Error)
    用于配合 LDS 权重使用。
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, weights=None):
        """
        Args:
            pred: 模型预测值 (B, 1)
            target: 真实值 (B, 1)
            weights: 样本权重 (B,) 或 (B, 1)，可选
        """
        # 确保维度一致
        if pred.dim() == 1: pred = pred.view(-1, 1)
        if target.dim() == 1: target = target.view(-1, 1)
            
        loss = (pred - target) ** 2
        
        if weights is not None:
            # 确保权重在正确的设备上
            if weights.device != loss.device:
                weights = weights.to(loss.device)
            # 调整权重形状以进行广播 (B,) -> (B, 1)
            loss = loss * weights.view(-1, 1)
            
        return loss.mean()


class RankLoss(nn.Module):
    """
    成对排序损失 (Pairwise Ranking Loss)
    物理意义：如果真实风险 A > B，那么预测风险也应该 A > B。
    这有助于模型学习样本间的相对顺序，对于 F2-score 等基于阈值的指标提升显著。
    """
    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin

    def forward(self, pred, target):
        """
        Args:
            pred: 模型预测值 (B, 1)
            target: 真实值 (B, 1)
        """
        # 展平
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)
        
        # 构建所有两两组合的差异矩阵 (B, B)
        # pred_diff[i][j] = pred[i] - pred[j]
        pred_diff = pred - pred.t()
        target_diff = target - target.t()
        
        # 指示矩阵 S_ij:
        # 1  如果 target[i] > target[j]
        # -1 如果 target[i] < target[j]
        # 0  如果 target[i] == target[j]
        s_ij = torch.sign(target_diff)
        
        # 损失计算: relu( -S_ij * (pred[i] - pred[j]) + margin )
        # 如果顺序正确 (S_ij 和 pred_diff 同号)，第一项为负，Loss 为 0 (或 margin)
        # 如果顺序错误，第一项为正，产生 Loss
        loss = torch.relu(-s_ij * pred_diff + self.margin)
        
        # Mask 掉 target 相同的对 (对角线和真实值相等的样本对不计算 Loss)
        mask = (target_diff.abs() > 1e-6).float()
        
        # 计算平均 Loss (只考虑有效的 pair)
        return (loss * mask).sum() / (mask.sum() + 1e-8)


class SupConRegressionLoss(nn.Module):
    """
    回归任务的监督对比损失 (Supervised Contrastive Loss for Regression - SupCR)
    原理：在特征空间中，根据标签的相似度来拉近/推开样本。
    Risk 值相近的样本，其 Feature Embedding 应该相似；Risk 差异大的，Feature 应该远离。
    """
    def __init__(self, temperature=0.1, sigma=1.0):
        """
        Args:
            temperature: 控制 logits 的缩放
            sigma: 用于计算标签相似度的高斯核带宽。sigma 越小，对标签差异越敏感。
        """
        super().__init__()
        self.temperature = temperature
        self.sigma = sigma

    def forward(self, features, labels):
        """
        Args:
            features: 模型的特征层输出 (B, feature_dim)，通常是 MLP 倒数第二层
            labels: 真实风险值 (B, 1)
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 1. 特征归一化 (Cosine Similarity 需要)
        features = F.normalize(features, dim=1)
        
        # 2. 计算特征相似度矩阵 (B, B)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        # 数值稳定性：减去每行的最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 3. 构建正样本权重 Mask (基于标签距离的软权重)
        if labels.dim() == 1:
            labels = labels.view(-1, 1)
        
        # 计算标签距离矩阵 (B, B)
        label_dist = torch.abs(labels - labels.T)
        
        # 使用高斯核将距离转换为相似度权重 [0, 1]
        # 距离越近(0)，权重越接近 1；距离越远，权重越接近 0
        weights = torch.exp(- (label_dist ** 2) / (2 * self.sigma ** 2))
        
        # 移除自我对比 (对角线设为 0)
        logits_mask = torch.scatter(
            torch.ones(batch_size, batch_size, device=device),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0
        )
        weights = weights * logits_mask

        # 4. 计算对比损失 (InfoNCE 变体)
        # 分母：所有样本的 exp(sim) 之和
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # 分子：加权平均的正样本对的 log 概率
        # weights.sum(1) 是每个样本的"正样本总权重"
        mean_log_prob_pos = (weights * log_prob).sum(1) / (weights.sum(1) + 1e-8)

        loss = - mean_log_prob_pos
        return loss.mean()
