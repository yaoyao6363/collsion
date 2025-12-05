import torch
from torch import nn


class RiskLSTM(nn.Module):
    """简单的多层 LSTM 回归模型"""

    def __init__(self, args, input_size: int):
        super().__init__()
        hidden_size = getattr(args, "hidden_size", 128)
        num_layers = getattr(args, "num_layers", 2)
        dropout = getattr(args, "dropout", 0.1)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, return_feat=False) -> torch.Tensor:
        """
        Args:
            x: 输入序列 (B, seq_len, input_size)
            return_feat: 是否返回特征向量（用于 SupCR）
        
        Returns:
            pred: 预测值 (B, 1)
            feat: 特征向量 (B, hidden_size)，仅当 return_feat=True 时返回
        """
        out, _ = self.lstm(x)
        
        # 提取特征 (Embedding)
        # out[:, -1, :] 是最后一个时间步的隐藏状态，作为样本的高维特征表示
        feat = self.dropout(out[:, -1, :])
        
        pred = self.head(feat)
        
        # 根据参数返回
        if return_feat:
            return pred, feat
        return pred

