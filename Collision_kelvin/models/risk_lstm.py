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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        pred = self.head(out)
        return pred

