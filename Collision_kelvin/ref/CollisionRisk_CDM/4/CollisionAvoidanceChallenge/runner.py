import os
import pickle

import pandas as pd
import torch
from torch.utils import data

from dataset import balanced_datasets
from models import s2s, utils
from models.pipeline import Pipeline

if __name__ == '__main__':
    artifacts_path = ''
    os.makedirs(artifacts_path, exist_ok=True)
    with open('', 'rb') as file:
        tr = pickle.load(file)
    with open('', 'rb') as file:
        test = pickle.load(file)

    scaler = balanced_datasets.StandardScaler(pd.read_csv(''))
    tr_dataset = balanced_datasets.KelvinsDataset(tr, scaler)
    test_dataset = balanced_datasets.KelvinsDataset(test, scaler)
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = s2s.S2S(tr_dataset._samples[0].features.shape[1] + 1, 2)
    train_dataloader = data.DataLoader(tr_dataset, batch_size=10,
                                       collate_fn=utils.custom_collate,
                                       shuffle=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=200,
                                      collate_fn=utils.custom_collate,
                                      shuffle=False)
    print('Train dataset length: {}\nTest dataset length: {}'.format(
        len(tr_dataset), len(test_dataset)))
    pipeline = Pipeline(model=model, tr_dataloader=train_dataloader,
                        path=artifacts_path, patience=100)
    pipeline.train(1010, test_dataloader)
    pipeline.save_artifacts()
