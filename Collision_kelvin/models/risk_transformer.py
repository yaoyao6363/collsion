# models/risk_transformer.py
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    标准正弦/余弦位置编码，适配 batch_first=True
    输入/输出: (B, T, d_model)
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                # (T, d)
        position = torch.arange(0, max_len).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (d/2,)

        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维

        pe = pe.unsqueeze(0)  # (1, T, d)
        self.register_buffer("pe", pe)  # 不当作参数训练，但会随模型一起保存/加载

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]
    

class RiskTransformer(nn.Module):
    """
    用 TransformerEncoder 做时间序列回归
    输入:  (B, T, input_size)
    输出:  (B, 1)
    """
    def __init__(self, args, input_size: int):
        super().__init__()

        # 从 args 里拿 Transformer 的超参，如果没有就给默认值
        d_model = getattr(args, "d_model", 128)
        nhead = getattr(args, "nhead", 4)
        num_layers = getattr(args, "num_layers_tf", 2)
        dim_feedforward = getattr(args, "dim_ff", 256)
        dropout = getattr(args, "tf_dropout", 0.1)

        # 把原始特征映射到 d_model 维度（兼容你现在的 channel 维度）
        self.input_proj = nn.Linear(input_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # 非常重要：让输入形状支持 (B, T, d_model)
            activation="gelu",
            norm_first=True,    # 一般更稳定
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)  # 回归到 1 维

    def forward(self, x: torch.Tensor, return_feat=False) -> torch.Tensor:
        """
        Args:
            x: 输入序列 (B, T, input_size)
            return_feat: 是否返回特征向量（用于 SupCR）
        
        Returns:
            pred: 预测值 (B, 1)
            feat: 特征向量 (B, d_model)，仅当 return_feat=True 时返回
        """
        # 1) 线性投影到 d_model
        x = self.input_proj(x)      # (B, T, d_model)

        # 2) 加位置编码
        x = self.pos_encoder(x)     # (B, T, d_model)

        # 3) 过 TransformerEncoder
        x = self.encoder(x)         # (B, T, d_model)

        # 4) 取最后一个时间步的特征（作为样本的高维特征表示）
        feat = x[:, -1, :]          # (B, d_model)
        feat = self.dropout(feat)

        # 5) 回归头
        pred = self.head(feat)      # (B, 1)
        
        # 根据参数返回
        if return_feat:
            return pred, feat
        return pred
