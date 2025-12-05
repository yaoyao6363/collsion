from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

from dataset.balanced_datasets import Sample

BATCH = 0
OUTPUT = 0
SEQ_LEN = 1
LAST_LAYER = -1
FIRST_TIME_STEP = 0


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 25,
                 num_layers: int = 2):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.hat = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param input: [BATCH_SIZE, SEQUENCE_LENGTH, NO_OF_FEATURES]
        :return: [BATCH_SIZE, SEQUENCE_LENGTH], ([BATCH_SIZE, HIDDEN_SIZE], [BATCH_SIZE, HIDDEN_SIZE])
        """
        lstm_output, (h_n, c_n) = self.lstm(input)
        sequence_length = lstm_output.shape[SEQ_LEN]
        predicted_risks = torch.zeros((1, sequence_length))
        for time_step_hidden_vector_index in range(sequence_length):
            hat_output = self.hat(
                lstm_output[0, time_step_hidden_vector_index])
            predicted_risks[0, time_step_hidden_vector_index] = hat_output
        return predicted_risks, h_n[LAST_LAYER, :], c_n[LAST_LAYER, :]


class Decoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 25, ):
        super(Decoder, self).__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.hat = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, input_tcas: Tensor, last_risk: Tensor,
                last_encoder_h_state: Tensor, last_encoder_c_state: Tensor):
        """
        :param input_tcas: [BATCH_SIZE, SEQUENCE_LENGTH]
        :param last_risk: [BATCH_SIZE, 1]
        :param last_encoder_h_state: [BATCH_SIZE, HIDDEN_SIZE]
        :param last_encoder_c_state: [BATCH_SIZE, HIDDEN_SIZE]
        :return: predicted risks [BATCH_SIZE, SEQUENCE_LENGTH]
        """
        sequence_length = input_tcas.shape[SEQ_LEN]
        batch_size = input_tcas.shape[BATCH]
        predicted_risks = torch.zeros((batch_size, sequence_length))
        first_input = torch.cat(
            [last_risk.unsqueeze(dim=0),
             input_tcas[:, FIRST_TIME_STEP].unsqueeze(dim=0)],
            dim=1)
        hx, cx = self.cell(first_input,
                           (last_encoder_h_state, last_encoder_c_state))
        predicted_risk = self.hat(hx)
        predicted_risks[:, FIRST_TIME_STEP] = predicted_risk
        for time_step in range(1, sequence_length):
            cell_input = torch.cat(
                [predicted_risk, input_tcas[:, time_step].unsqueeze(dim=0)],
                dim=1)
            hx, cx = self.cell(cell_input, (hx, cx))
            predicted_risk = self.hat(hx)
            predicted_risks[:, time_step] = predicted_risk
        return predicted_risks


class S2S(nn.Module):
    def __init__(self, encoder_input_size: int, decoder_input_size: int,
                 hidden_size: int = 50, encoder_num_layers: int = 2):
        super(S2S, self).__init__()
        self.encoder = Encoder(encoder_input_size, hidden_size,
                               encoder_num_layers)
        self.decoder = Decoder(decoder_input_size, hidden_size)
        self.optimizer = Adam(self.parameters())
        self.loss = nn.MSELoss()

    def forward(self, batch: List[Sample], mode: str):
        if mode == 'train':
            outputs = torch.zeros((sum([len(item.risk) for item in batch])))
            targets_for_all = torch.zeros(
                (sum([len(item.risk) for item in batch])))
            last_insert = 0
            i = 0
            for sample in batch:
                features, risk, _ = sample.get_tensors()
                sequence_length = len(risk)
                tcas = features[:, :, sample.time_to_tca_id]
                sample_horizon = torch.argmin(
                    torch.abs(torch.abs(tcas) - 0.3)) + 1
                encoder_input = features[:, :sample_horizon, :]
                encoder_input = torch.cat([torch.unsqueeze(
                    torch.unsqueeze((risk.clone() - (-30)) / (0 - (-30)), 0),
                    -1), encoder_input], 2)
                decoder_tcas = features[:, sample_horizon:,
                               sample.time_to_tca_id]
                encoder_predicted_risks, h_n, c_n = self.encoder(encoder_input)
                decoder_predicted_risks = self.decoder(decoder_tcas,
                                                       encoder_predicted_risks[
                                                       :, -1], h_n, c_n)
                decoder_predicted_risks = decoder_predicted_risks.squeeze() if decoder_predicted_risks.shape != torch.Size(
                    [1, 1]) else decoder_predicted_risks.reshape([1])
                encoder_predicted_risks = encoder_predicted_risks.squeeze() if encoder_predicted_risks.shape != torch.Size(
                    [1, 1]) else encoder_predicted_risks.reshape([1])
                outputs[
                last_insert: last_insert + sequence_length] = torch.cat(
                    [encoder_predicted_risks, decoder_predicted_risks], dim=0)
                targets_for_all[
                last_insert: last_insert + sequence_length] = risk
                last_insert += sequence_length
            return self.loss(outputs, targets_for_all)
        else:
            predictions = []
            trues = []
            for i, sample in enumerate(batch):
                features, risk, _ = sample.get_tensors()
                last_tca = sample.features[-2, sample.time_to_tca_id]
                encoder_input = features[:, :-1, :]
                encoder_input = torch.cat([torch.unsqueeze(
                    torch.unsqueeze((risk.clone()[:-1] - (-30)) / (0 - (-30)),
                                    0), -1), encoder_input], 2)
                decoder_tcas = torch.flip(torch.arange(start=0.0, end=last_tca,
                                                       step=0.02),
                                          dims=[0]).unsqueeze(dim=0)
                encoder_predicted_risks, h_n, c_n = self.encoder(
                    encoder_input)
                decoder_predicted_risks = self.decoder(decoder_tcas,
                                                       encoder_predicted_risks[
                                                       :, -1], h_n, c_n)
                predictions.append(decoder_predicted_risks.squeeze()[-1])
                trues.append(risk[-1])
            return np.array(predictions), np.array(trues)

    def weighted_mse_loss(self, y_hat: torch.Tensor, y: torch.Tensor,
                          threshold: int = -6):
        numpy_y = y.clone().detach().cpu().numpy()
        numpy_y = numpy_y * (0 - (-30)) + (-30)
        where = np.where(numpy_y < threshold)[0]
        weight = np.ones((numpy_y.shape[0]))
        weight[where] = weight[where] * 0.1
        weight = torch.from_numpy(weight).type(torch.FloatTensor)
        return torch.mean(weight * (y_hat - y) ** 2)
