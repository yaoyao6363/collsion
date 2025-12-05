import typing

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from dataset import balanced_datasets


class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 rnn_type: str, device: torch.device, dropout: float = 0):
        super(RNN, self).__init__()
        rnn_modules = {
            'rnn': nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout).to(device),
            'gru': nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout).to(device),
            'lstm': nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout).to(device),
        }
        self.rnn = rnn_modules[rnn_type.lower()]
        self.hat = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.PReLU(),

            nn.Linear(128, 256),
            nn.PReLU(),

            nn.Linear(256, 64),
            nn.PReLU(),

            nn.Linear(64, 1),
            # nn.Sigmoid()
        ).to(device)
        self.optimizer = Adam(self.parameters())
        self.loss = nn.MSELoss()
        self.device = device

    def forward(self, samples: typing.List[balanced_datasets.Sample],
                mode: str) -> \
            typing.Union[torch.Tensor, typing.Tuple[np.ndarray, np.ndarray]]:
        outputs = torch.zeros((len(samples))).to(self.device)
        targets = torch.zeros((len(samples))).to(self.device)
        for i, sample in enumerate(samples):
            features, risk, _ = sample.get_tensors()
            features = features.to(self.device)
            risk = risk.to(self.device)
            if len(risk.shape) == 0:
                risk = torch.tensor([risk.item()]).to(self.device)
            event = torch.cat(
                [torch.unsqueeze(torch.unsqueeze(risk, 0), -1), features], 2)
            event = event[:, :-1]
            y_hat = self.rnn(event)[0][:, -1]
            y_hat = self.hat(y_hat)
            outputs[i] = y_hat
            targets[i] = risk[-1]
        if mode.lower() == 'train':
            return self.weighted_mse_loss(outputs, targets)
        elif mode.lower() == 'test':
            return outputs.cpu().numpy(), targets.cpu().numpy()

    def weighted_mse_loss(self, y_hat: torch.Tensor, y: torch.Tensor,
                          threshold: int = -6):
        numpy_y = y.clone().detach().cpu().numpy()
        where = np.where(numpy_y < threshold)[0]
        weight = np.ones((numpy_y.shape[0]))
        weight[where] = weight[where] * 0.1
        weight = torch.from_numpy(weight).type(torch.FloatTensor).to(
            self.device)
        return torch.mean(weight * (y_hat - y) ** 2)
