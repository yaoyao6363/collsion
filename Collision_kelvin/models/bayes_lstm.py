#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BayesLSTM: Bayesian LSTM for CDM-based collision risk regression

复现思路：
- 使用堆叠 LSTM（默认 2 层，每层 256 单元）
- 在 LSTM 层和后续全连接层中使用 dropout（默认 0.2）
- 训练时正常前向 + MSE 回归
- 测试的不确定性估计：保持 model.train()，多次前向传播 (Monte Carlo Dropout)

接口对齐：
- __init__(args, channel): channel = 每个时间步的特征数
- forward(seq_x): seq_x 形状 (B, T, channel)，输出 (B, 1)
"""

from typing import Tuple
import torch
import torch.nn as nn


class BayesLSTM(nn.Module):
    def __init__(self, args, channel: int):
        super().__init__()

        # 从命令行参数中读取超参数，没有就用论文里的默认值
        hidden_size = getattr(args, "bayes_hidden_size", 256)
        num_layers = getattr(args, "bayes_num_layers", 2)
        dropout = getattr(args, "bayes_dropout", 0.2)

        self.input_dim = channel
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout

        # 核心：堆叠 LSTM + dropout
        # PyTorch 的 LSTM dropout = 各层之间的“变分 dropout”，时间维上 mask 保持不变，
        # 这和 Gal & Ghahramani 的 Bayesian LSTM 思路是一致的。
        self.lstm = nn.LSTM(
            input_size=channel,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,   # 输入 (B, T, C)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 额外的全连接前 dropout（论文里也是“所有权重层都用 dropout，除了输出层”）
        self.pre_fc_dropout = nn.Dropout(p=dropout)

        # 回归头：hidden -> hidden -> 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, seq_x: torch.Tensor) -> torch.Tensor:
        """
        标准前向传播（单次采样）
        输入:
            seq_x: (B, T, C)  CDM 序列特征
        输出:
            pred: (B, 1)      对最终 log10(risk) 的预测
        """
        # LSTM 输出：out: (B, T, H), h_n: (num_layers, B, H)
        out, (h_n, c_n) = self.lstm(seq_x)

        # 使用最后一层的最后一个时间步的隐藏状态，作为序列表示
        # h_n 形状: (num_layers, B, H) -> 取最后一层: (B, H)
        h_last = h_n[-1]  # (B, hidden_size)

        # FC 前再做一次 dropout
        h_last = self.pre_fc_dropout(h_last)

        pred = self.fc(h_last)  # (B, 1)
        return pred

    @torch.no_grad()
    def mc_predict(
        self,
        seq_x: torch.Tensor,
        mc_samples: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout 推理（多次采样）：
        - 保持 dropout 激活（model.train() 状态），多次 forward
        - 返回预测均值和标准差

        输入:
            seq_x: (B, T, C)
            mc_samples: 采样次数 K

        输出:
            mean: (B, 1)  K 次采样的均值
            std:  (B, 1)  K 次采样的标准差
        """
        was_training = self.training
        # 为了启用 dropout，强制切到 train 模式
        self.train()

        preds = []
        for _ in range(mc_samples):
            preds.append(self.forward(seq_x))  # (B, 1)

        preds = torch.stack(preds, dim=0)      # (K, B, 1)
        mean = preds.mean(dim=0)              # (B, 1)
        std = preds.std(dim=0)                # (B, 1)

        # 还原之前的模式
        if not was_training:
            self.eval()

        return mean, std
