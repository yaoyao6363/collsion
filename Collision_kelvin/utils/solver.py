import os
import gc
from time import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.data_utils import DataUtils
from models import RiskLSTM, RiskTransformer,BayesLSTM
from models.cdea_regressor import CDEARegressor
from utils.data_preprocessor import pre_data
from utils.dataset import CreateDataset
from utils.stoper import Stopper


class RegressionSequenceDataset(Dataset):
    """封装后的时序数据集，返回 (seq, target)"""

    def __init__(self, data: np.ndarray, targets: np.ndarray):
        targets = targets.reshape(-1, 1).astype(np.float32)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def compute_metrics(preds, targets, threshold=-6):
    """
    计算回归指标和基于阈值的F2-score
    
    Args:
        preds: 预测值
        targets: 真实值
        threshold: 风险阈值（对数尺度），默认-6，用于计算F2-score
    
    Returns:
        dict: 包含MSE, RMSE, MAE, MAPE, R2, F2的指标字典
    """
    if not isinstance(preds, torch.Tensor):
        preds = torch.as_tensor(preds)
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets)
    preds = preds.view(-1).double()
    targets = targets.view(-1).double()
    
    # 回归指标
    diff = preds - targets
    mse = torch.mean(diff ** 2)
    mae = torch.mean(torch.abs(diff))
    rmse = torch.sqrt(torch.clamp(mse, min=0.0))
    mape = torch.mean(torch.abs(diff / (targets + 1e-8))) * 100
    ss_res = torch.sum(diff ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2) + 1e-8
    r2 = 1 - ss_res / ss_tot
    
    # F2-score (基于阈值的分类指标)
    # 将风险值转换为二分类：>= threshold 为高风险(1)，< threshold 为低风险(0)
    pred_mask = (preds >= threshold).long()
    target_mask = (targets >= threshold).long()
    
    # 计算 TP, FP, FN
    tp = torch.sum((pred_mask == 1) & (target_mask == 1)).float()
    fp = torch.sum((pred_mask == 1) & (target_mask == 0)).float()
    fn = torch.sum((pred_mask == 0) & (target_mask == 1)).float()
    
    # F2-score: beta=2 更重视召回率
    beta = 2.0
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-8)
    
    return {
        'MSE': float(mse.item()),
        'RMSE': float(rmse.item()),
        'MAE': float(mae.item()),
        'MAPE': float(mape.item()),
        'R2': float(r2.item()),
        'F2': float(f2.item())
    }


model_dict = {
    'LSTM': RiskLSTM,
    'TRANSFORMER': RiskTransformer,
    'CDEA': CDEARegressor,
    'BAYES_LSTM':BayesLSTM
}

class Solver:
    def __init__(self, args, setting):
        self.args = args
        self.setting = setting
        self.device = self._acquire_device()
        self.model_path = self._make_dirs()
        self.create_dataset = CreateDataset(args)
        self.train_loader, self.valid_loader, self.test_loader = self._get_loader()
        self.channel = self.train_loader.dataset.data.shape[-1]
        self.model = model_dict[self.args.model](self.args, self.channel).to(self.device)

        # ==== 物理特征索引（在 regression_features 里的位置） ====
        # regression_features 的顺序就是 feature_dim 的顺序
        # 用已有的物理特征来构造邻域：
        # - miss_distance: 最近接距离
        # - relative_position_n/r/t: 相对位置在 NRT 坐标下的三个分量
        # - t_j2k_inc: 目标轨道倾角
        self.regression_feature_names = DataUtils.regression_features

        # 你想用于物理邻域的特征名：
        self.phys_feat_names = [
        'miss_distance',
        'relative_position_n',
        'relative_position_r',
        'relative_position_t',
        't_j2k_inc',
        ]

        self.phys_feat_indices = []
        for name in self.phys_feat_names:
            if name in self.regression_feature_names:
                self.phys_feat_indices.append(self.regression_feature_names.index(name))
            else:
                raise ValueError(f"物理特征 {name} 不在 DataUtils.regression_features 里，请检查命名。")

        # 打印一下，方便确认
        print("Physical features for neighbor attention:",
              {n: i for n, i in zip(self.phys_feat_names, self.phys_feat_indices)})

        # 计算模型参数量并转换为 MB
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        param_size_bytes = total_params * 4  # 每个参数 4 字节（float32）
        param_size_mb = param_size_bytes / (1024 ** 2)  # 转换为 MB
        print(f"Total trainable parameters: {total_params:,}, Size: {param_size_mb:.2f} MB")

        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        self.stopper = Stopper(patience=self.args.patience, path=self.model_path + '/' + 'checkpoint.pth')
        self.criterion = nn.MSELoss()

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device(f"cuda:{self.args.device}")
            print(f"Use GPU: cuda:{self.args.device}")
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _make_dirs(self):
        model_path = os.path.join(self.args.save_path, self.setting)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return model_path

    def _save_metrics_to_csv(self, metrics):
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(self.model_path, 'metrics.csv'), index=False)

    def _inverse_standardize_tensor(self, data_norm, mean, std):
        mean = torch.tensor(mean, device=data_norm.device, dtype=data_norm.dtype)
        std = torch.tensor(std, device=data_norm.device, dtype=data_norm.dtype)
        return data_norm * std + mean

    def _denormalize_tensor(self, data_norm, scaler):
        min_val = torch.tensor(scaler.min_, device=data_norm.device, dtype=data_norm.dtype)
        scale = torch.tensor(scaler.scale_, device=data_norm.device, dtype=data_norm.dtype)
        feature_range_min = torch.tensor(scaler.feature_range[0], device=data_norm.device, dtype=data_norm.dtype)
        return (data_norm - feature_range_min) / scale + min_val

    def _get_loader(self):
        file_train = os.path.join(self.args.data_path, self.args.dataset_train)
        df = pre_data(file_train)
        _, regression_dataset = self.create_dataset.get_train_test_dataset(df)
        train_x_r, train_y_r, test_x_r, test_y_r = regression_dataset

        seq_len = self.create_dataset.n_latest_cdms - 1
        if seq_len <= 0:
            raise ValueError("n_latest_cdms 需大于 1，才能构建时间序列样本。")

        feature_dim = train_x_r.shape[1] // seq_len
        train_x_r = train_x_r.reshape(-1, seq_len, feature_dim)
        test_x_r = test_x_r.reshape(-1, seq_len, feature_dim)

        train_ratio, val_ratio, _ = self.args.ratio_split
        ratio_sum = train_ratio + val_ratio
        if ratio_sum <= 0:
            raise ValueError('ratio_split 中的前两项需要为正数，以便划分训练/验证集。')
        split_ratio = val_ratio / ratio_sum
        x_train, x_val, y_train, y_val = train_test_split(
            train_x_r, train_y_r, test_size=split_ratio, random_state=self.args.seed, shuffle=True
        )

        x_train, normalized_others = self._normalize_sequences(x_train, [x_val, test_x_r])
        x_val, x_test = normalized_others

        train_dataset = RegressionSequenceDataset(x_train, y_train)
        val_dataset = RegressionSequenceDataset(x_val, y_val)
        test_dataset = RegressionSequenceDataset(x_test, test_y_r)

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False)

        return train_loader, val_loader, test_loader

    def _normalize_sequences(self, train_array, other_arrays):
        mean = train_array.mean(axis=(0, 1), keepdims=True)
        std = train_array.std(axis=(0, 1), keepdims=True)
        std[std < 1e-6] = 1e-6
        norm_train = (train_array - mean) / std
        normalized_others = [(array - mean) / std for array in other_arrays]
        self.feature_mean = mean
        self.feature_std = std
        return norm_train, normalized_others

    def _process_batch(self, seq_x, seq_y):
        seq_x = seq_x.float().to(self.device)
        seq_y = seq_y.float().to(self.device)

        if self.args.model == 'CDEA':
            # seq_x: (B, seq_len, feature_dim)
            # 取最后一个时间步的特征作为“当前 TCA 前刻”的物理状态
            last_step = seq_x[:, -1, :]                          # (B, feature_dim)
            phys_feat = last_step[:, self.phys_feat_indices]     # (B, P)

            pred = self.model(seq_x, phys_feat=phys_feat)
        else:
            pred = self.model(seq_x)

        loss = self.criterion(pred, seq_y)
        return loss, pred, seq_y


    ####################
    '''train + val'''
    ####################
    def train(self):
        i = 0
        result_metrics = []
        for e in range(self.args.epoch):
            self.current_epoch = e  # 记录当前 epoch
            start = time()

            self.model.train()
            train_loss = []
            for (seq_x, seq_y) in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                loss, _, _ = self._process_batch(seq_x, seq_y)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                valid_loss = []
                pred_list, true_list = [], []
                for (seq_x, seq_y) in tqdm(self.valid_loader):
                    loss, pred, target = self._process_batch(seq_x, seq_y)
                    valid_loss.append(loss.item())
                    true_list.append(target.detach().cpu())
                    pred_list.append(pred.detach().cpu())

            train_loss, valid_loss = np.mean(train_loss), np.mean(valid_loss)

            # 更新学习率调度器
            self.scheduler.step(valid_loss)

            preds_cat = torch.cat(pred_list, dim=0)
            trues_cat = torch.cat(true_list, dim=0)
            res = compute_metrics(preds_cat, trues_cat)

            end = time()

            # 打印当前epoch的评估指标
            print(
                f"Epoch: {e + 1} | Cost: {end - start:.6f}s || Train Loss: {train_loss:.6f} Valid Loss: {valid_loss:.6f}\n"
                f"--> MSE: {res['MSE']:.5f} | MAE: {res['MAE']:.5f} | RMSE: {res['RMSE']:.5f} | "
                f"MAPE: {res['MAPE']:.5f}% | R^2: {res['R2']:.5f} | F2: {res['F2']:.5f}\n")

            # 保存每轮结果
            result_metrics.append({
                'epoch': e + 1,
                'cost_sec': end - start,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'MSE': res['MSE'],
                'MAE': res['MAE'],
                'RMSE': res['RMSE'],
                'MAPE': res['MAPE'],
                'R2': res['R2'],
                'F2': res['F2']
            })

            # 早停机制
            self.stopper(valid_loss, self.model)
            i = i + 1
            if self.stopper.early_stop:
                print(f"Early stopping at epoch {e} with validation loss: {valid_loss:.4f}")
                break

            torch.cuda.empty_cache()
            gc.collect()
        return i, result_metrics

    ####################
    '''test'''
    ####################
    def test(self):
        self.model.load_state_dict(torch.load(self.model_path + '/' + 'checkpoint.pth', map_location=self.device))
        with torch.no_grad():
            self.model.eval()
            test_loss = []
            pred_list, true_list = [], []
            for (seq_x, seq_y) in tqdm(self.test_loader):
                loss, pred, target = self._process_batch(seq_x, seq_y)
                test_loss.append(loss.item())
                pred_list.append(pred.detach().cpu())
                true_list.append(target.detach().cpu())

        test_loss = np.mean(test_loss)
        res = compute_metrics(torch.cat(pred_list, dim=0), torch.cat(true_list, dim=0))
        print(
            f"Test Loss: {test_loss:.6f} | "
            f"--> MSE: {res['MSE']:.5f} | MAE: {res['MAE']:.5f} | RMSE: {res['RMSE']:.5f} | "
            f"MAPE: {res['MAPE']:.5f}% | R^2: {res['R2']:.5f} | F2: {res['F2']:.5f}\n")

        # 清理内存
        torch.cuda.empty_cache()  # 释放 GPU 缓存
        gc.collect()  # 触发垃圾回收

        # 保存测试结果
        with open(self.model_path + '/' + 'result.txt', 'w', encoding='utf-8') as f:
            f.write(
                f"Test Loss: {test_loss:.6f} | "
                f"--> MSE: {res['MSE']:.5f} | MAE: {res['MAE']:.5f} | RMSE: {res['RMSE']:.5f} | "
                f"MAPE: {res['MAPE']:.5f}% | R^2: {res['R2']:.5f} | F2: {res['F2']:.5f}\n")
            f.write('\n')

        return res, []

    

