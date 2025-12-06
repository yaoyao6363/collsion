import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    标准正弦/余弦位置编码
    """
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # (1, T, D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1), :]

class AttentionPooling(nn.Module):
    """
    学习每个时间步的权重，进行加权求和。
    这比单纯取最后一个时间步更能利用全序列信息。
    """
    def __init__(self, d_model):
        super().__init__()
        self.attention_weights = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: (B, T, D)
        # scores: (B, T, 1)
        scores = self.attention_weights(x)
        weights = F.softmax(scores, dim=1)
        
        # 加权求和: (B, D)
        out = torch.sum(x * weights, dim=1)
        return out

class RiskTransformer(nn.Module):
    """
    Baseline: Physics-Aware Encoder-only Transformer
    """
    def __init__(self, args, input_size: int):
        super().__init__()

        # === 参数配置 (针对小样本优化) ===
        d_model = getattr(args, "d_model", 64)       # 建议设小一点，防止过拟合
        nhead = getattr(args, "nhead", 4)            # 4头注意力足够
        num_layers = getattr(args, "num_layers_tf", 2) # 1-2层即可
        dim_feedforward = getattr(args, "dim_ff", 128)
        dropout = getattr(args, "tf_dropout", 0.2)

        # 1. Embedding: 将物理特征映射到高维
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        # 2. Transformer Encoder (核心)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True, # 关键: 输入为 (Batch, Seq, Feature)
            norm_first=True   # Pre-LN 结构，训练更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. Pooling 策略 (改进点)
        # 相比于只取最后一步，Attention Pooling 能自动寻找“最关键的那条预警”
        self.pooling = AttentionPooling(d_model)

        # 4. 预测头（多任务学习）
        self.dropout = nn.Dropout(dropout)
        
        # 主分支：预测 Log Risk
        self.head_risk = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # === 新增：物理分支 ===
        # 预测 Log Miss Distance（物理约束）
        self.head_dist = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor, return_feat=False) -> torch.Tensor:
        """
        x: (B, T, input_size)
        """
        # [Batch, Seq, Dim]
        x = self.input_proj(x)
        x = self.layer_norm(x)
        x = self.pos_encoder(x)

        # Encoder 提取时序特征
        # out: (B, T, D)
        out = self.encoder(x)

        # === 关键改进：聚合特征 ===
        # 使用 Attention Pooling 替代 out[:, -1, :]
        # feat: (B, D)
        feat = self.pooling(out)
        
        feat = self.dropout(feat)

        # 输出两个预测值
        pred_risk = self.head_risk(feat)
        pred_dist = self.head_dist(feat)
        
        if return_feat:
            return pred_risk, feat
            
        # 训练时返回两个预测值，测试时只返回 Risk
        if self.training:
            return pred_risk, pred_dist
        else:
            return pred_risk