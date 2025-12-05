import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.nn import Module
from torch.utils import data
from tqdm import tqdm

from dataset import balanced_datasets
from models import rnn
from models import utils
from models.utils import save_metrics


class Pipeline(object):
    def __init__(self, model: Module, tr_dataloader: data.DataLoader,
                 patience: int, path: str):
        assert isinstance(model, Module)
        assert hasattr(model, 'optimizer')
        self._model = model
        self._tr_dataloader = tr_dataloader
        self._patience = patience
        self._path = path
        self._artifacts = {
            'mse': [],
            'f2_score': [],
            'test_mse': [],
            'score': []
        }

    def _train_epoch(self):
        print('Training model...')
        loss = []
        for sample in self._tr_dataloader:
            self._model.zero_grad()
            error = self._model.forward(sample, mode='train')
            error.backward()
            self._model.optimizer.step()
            loss.append(error.clone().detach().cpu().numpy())
        self._artifacts['mse'].append(np.mean(loss))
        print('Training mse: {}'.format(self._artifacts['mse'][-1]))

    def check_progress(self) -> bool:
        return min(self._artifacts['score'][:-self._patience]) > min(
            self._artifacts['score'][-self._patience:])

    def _save_model(self):
        print('Saving improvement...')
        torch.save({
            'state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._model.optimizer.state_dict(),
        }, os.path.join(self._path, 'checkpoint'))

    def train(self, epochs: int, test_dataloader: data.DataLoader):
        n_tests = 0
        for i in range(epochs):
            self._train_epoch()
            if i % 25 == 0:
                self.test(test_dataloader)
                n_tests += 1
                if n_tests > self._patience:
                    if self.check_progress():
                        self._save_model()

    def test(self, dataloader: data.DataLoader):
        print('Predicting...')
        assert isinstance(dataloader, data.DataLoader)
        outputs, targets = [], []
        with torch.no_grad():
            for sample in tqdm(dataloader, total=len(dataloader)):
                y_hat, y = self._model.forward(sample, mode='test')
                outputs.append(y_hat)
                targets.append(y)
        outputs, targets = np.hstack(outputs), \
                           np.hstack(targets)
        df = np.vstack([outputs, targets]).swapaxes(0, 1)
        df = pd.DataFrame(data=df, columns=['outputs', 'targets'])
        df.to_csv(os.path.join(self._path, 'val_outputs.csv'),
                  index_label='event_id')
        f2_score = self.f2_score(y_hat=outputs, y=targets)
        mse = self.mse(y_hat=outputs, y=targets)
        self._artifacts['f2_score'].append(f2_score)
        print('F2 score: {}'.format(self._artifacts['f2_score'][-1]))
        self._artifacts['test_mse'].append(mse)
        print('Test mse: {}'.format(self._artifacts['test_mse'][-1]))
        self._artifacts['score'].append(mse / f2_score)
        print('Score: {}'.format(self._artifacts['score'][-1]))

    def save_artifacts(self):
        save_metrics(self._path, self._artifacts)

    @staticmethod
    def infer_model(model: Module, scaler: str, model_path: str,
                    test_df_path: str):
        artifacts = torch.load(model_path)
        model.load_state_dict(artifacts['state_dict'])
        model.optimizer.load_state_dict(artifacts['optimizer_state_dict'])
        df = pd.read_csv(test_df_path)
        events = []
        for event in df['event_id'].unique():
            events.append(df[df['event_id'] == event])
        dataloader = data.DataLoader(
            balanced_datasets.KelvinsDataset(events, scaler), shuffle=False,
            batch_size=1,
            collate_fn=utils.custom_collate)
        print('Inference...')
        assert isinstance(dataloader, data.DataLoader)
        outputs, targets = [], []
        with torch.no_grad():
            for sample in tqdm(dataloader, total=len(dataloader)):
                y_hat, y = model.forward(sample, mode='infer')
                outputs.append(y_hat)
                targets.append(y)
        outputs, targets = np.hstack(outputs), \
                           np.hstack(targets)
        dest_path = os.path.dirname(test_df_path)
        out_df = {'event_id': list(range(len(outputs))),
                  'predicted_risk': outputs}
        out_df = pd.DataFrame(out_df)
        out_df.to_csv(os.path.join(dest_path, 'submission.csv'), index=False)

    @staticmethod
    def f2_score(y_hat: np.ndarray, y: np.ndarray, beta=2,
                 threshold=-6) -> float:
        y_hat = y_hat >= threshold
        y = y >= threshold

        tp = (y_hat & y).sum()
        fp = (y_hat & (~y)).sum()
        fn = ((~y_hat) & y).sum()
        print('True positives: {}'.format(tp))
        print('False positive: {}'.format(fp))
        print('False negative: {}'.format(fn))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f2 = (1 + beta ** 2) * precision * recall / (
                beta ** 2 * precision + recall + 1e-12)
        if f2 == 0:
            f2 = 1e-8
        return f2

    @staticmethod
    def mse(y_hat: np.ndarray, y: np.ndarray, threshold=-6) -> float:
        elements = np.where(y >= threshold)[0]
        y_hat = y_hat[elements]
        y = y[elements]
        loss = (np.square(y_hat - y)).mean(axis=0)
        return loss


if __name__ == '__main__':
    with open('', 'rb') as file:
        tr = pickle.load(file)
    with open('', 'rb') as file:
        test = pickle.load(file)
    tr_dataset = balanced_datasets.KelvinsDataset(tr, '')
    test_dataset = balanced_datasets.KelvinsDataset(test, '')
    device = torch.device('cuda:0') if torch.cuda.is_available() \
        else torch.device('cpu')
    model_ = rnn.RNN(int(tr_dataset[0].features.shape[1] + 1), 36, 2, 'gru',
                     device).to(device)
    train_dataloader = data.DataLoader(tr_dataset, batch_size=1,
                                       collate_fn=utils.custom_collate,
                                       shuffle=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=1,
                                      collate_fn=utils.custom_collate,
                                      shuffle=True)
    print('Train dataset length: {}\nTest dataset length: {}'.format(
        len(tr_dataset), len(test_dataset)))
    pipeline = Pipeline(model=model_, tr_dataloader=train_dataloader,
                        path='', patience=1)
    pipeline.train(510, test_dataloader)
    pipeline.save_artifacts()
    Pipeline.infer_model(model_, '', '', '')
