import random

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM


# SELECTED_FEATURES = ['risk', 'max_risk_scaling', 'max_risk_estimate', 'miss_distance', 'mahalanobis_distance', 'c_sigma_t', 'c_sigma_rdot', 'relative_position_n', 'c_obs_available', 'c_obs_used', 'relative_position_t', 'c_time_lastob_start', 'c_time_lastob_end', 'c_cr_area_over_mass', 'c_sigma_r', 'c_cd_area_over_mass', 'c_rcs_estimate', 'c_cndot_n', 'c_sigma_n', 't_h_apo', 't_j2k_sma', 'relative_position_r', 'c_h_per', 'c_sedr', 'c_sigma_tdot', 'c_ctdot_r', 'c_recommended_od_span', 't_h_per', 't_j2k_inc', 't_rcs_estimate', 'c_crdot_t', 'c_sigma_ndot', 'relative_velocity_n', 'c_actual_od_span', 't_sedr', 'c_j2k_sma', 'relative_velocity_t', 'c_j2k_inc', 'relative_speed', 'relative_velocity_r']
# SELECTED_FEATURES = ['time_to_tca', 'risk', 'max_risk_scaling', 'max_risk_estimate', 'miss_distance', 'mahalanobis_distance']
# SELECTED_FEATURES = ['time_to_tca', 'risk', 'max_risk_scaling', 'max_risk_estimate', 'miss_distance', 'mahalanobis_distance', 'c_sigma_t', 'c_sigma_rdot', 'relative_position_n', 'c_obs_available']

def f2_score(y_hat: np.ndarray, y: np.ndarray, beta=2, threshold=-6,
             model_treshold=-7) -> float:
    y_hat = y_hat.astype(bool)
    y = y.astype(bool)
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


def mse_score(y_hat: np.ndarray, y: np.ndarray, threshold=-6) -> float:
    elements = np.where(y >= threshold)[0]
    y_hat = y_hat[elements]
    y = y[elements]
    loss = (np.square(y_hat - y)).mean(axis=0)
    return loss


def cross_validation(model, X, y, cv=10):
    samples_in_test = int(len(X) / cv)
    indexes = np.arange(len(X))
    np.random.seed(0)
    random.Random(0).shuffle(indexes)
    f2_scores = []
    mse_scores = []
    scores = []
    for fold in range(0, cv):
        test_indexes = indexes[
                       fold * samples_in_test:fold * samples_in_test + samples_in_test]
        train_indexes = np.asarray(
            indexes[:fold * samples_in_test].tolist() + indexes[
                                                        fold * samples_in_test + samples_in_test:].tolist())
        assert len(
            list(set(train_indexes.tolist() + test_indexes.tolist()))) == \
               X.shape[0]
        X_train = X[train_indexes]
        X_test = X[test_indexes]
        y_train = y[train_indexes]
        y_test = y[test_indexes]
        model.fit(X_train, y_train)
        predicts = model.predict_proba(X_test)
        predicts = (predicts[:, 1] >= 0.5).astype(int)
        f2 = f2_score(predicts, y_test)
        mse = mse_score(predicts, y_test)
        f2_scores.append(f2)
        mse_scores.append(mse)
        scores.append(mse / f2)
        print('F2: {}, MSE: {}, Score: {}'.format(f2, mse, mse / f2))
    print('Avg f2: {} Avg mse: {} Avg score: {}'.format(np.mean(f2_scores),
                                                        np.mean(mse_scores),
                                                        np.mean(scores)))


test_df = pd.read_csv('')

df_scaler = pd.read_csv('').drop(columns=['event_id', 'mission_id', 'c_object_type'])
scaler = MinMaxScaler()
scaler.fit(df_scaler)
test_samples = []
for sample_id in test_df['event_id'].unique():
    sample = pd.DataFrame(
        test_df[test_df['event_id'] == sample_id].iloc[-1]).transpose()
    test_samples.append(sample)
df = pd.read_csv('')
y = df['y'].to_numpy()
df = df.drop(
    columns=['event_id', 'mission_id', 'last_cdm_risk', 'delta_p', 'y'])
X = df.to_numpy()
X = scaler.transform(X)
pca = PCA(n_components=2)
X = pca.fit_transform(X)
X_train = X[y == 0]
X_test = X[y == 1]
np.random.seed(0)
np.random.shuffle(X_train)
samples_to_take = 50000
X_train_stripped = X_train[:samples_to_take]
X_test = np.concatenate(
    [X_test, X_train[samples_to_take:samples_to_take + 4 * len(X_test)]])
model = OneClassSVM(random_state=0, nu=0.5, gamma='auto')
model.fit(X_train_stripped)
predicts = model.predict(X_test)
ones = np.ones((len(y[y == 1])))
zeros = np.zeros((len(X_test) - len(ones)))
y_true = np.concatenate([ones, zeros])
predicts[predicts == 1] = 0
predicts[predicts == -1] = 1
f2 = f2_score(predicts, y_true)
print('F2: {}'.format(f2))
test_samples = pd.concat(test_samples)
test_samples = test_samples.drop(columns=['event_id'])
x_test = scaler.transform(test_samples)
x_test = pca.transform(x_test)
predicts = model.predict(x_test)
predicts = predicts.astype(float)
predicts[predicts == -1] = -5.0
predicts[predicts == 1] = -6.01
pd.DataFrame(predicts, columns=['predicted_risk']).to_csv('',
                                                          index_label='event_id')
