import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# SELECTED_FEATURES = ['risk', 'max_risk_scaling', 'max_risk_estimate', 'miss_distance', 'mahalanobis_distance', 'c_sigma_t', 'c_sigma_rdot', 'relative_position_n', 'c_obs_available', 'c_obs_used', 'relative_position_t', 'c_time_lastob_start', 'c_time_lastob_end', 'c_cr_area_over_mass', 'c_sigma_r', 'c_cd_area_over_mass', 'c_rcs_estimate', 'c_cndot_n', 'c_sigma_n', 't_h_apo', 't_j2k_sma', 'relative_position_r', 'c_h_per', 'c_sedr', 'c_sigma_tdot', 'c_ctdot_r', 'c_recommended_od_span', 't_h_per', 't_j2k_inc', 't_rcs_estimate', 'c_crdot_t', 'c_sigma_ndot', 'relative_velocity_n', 'c_actual_od_span', 't_sedr', 'c_j2k_sma', 'relative_velocity_t', 'c_j2k_inc', 'relative_speed', 'relative_velocity_r']
SELECTED_FEATURES = ['risk', 'time_to_tca', 'max_risk_scaling',
                     'max_risk_estimate', 'miss_distance',
                     'mahalanobis_distance']


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
scaler = MinMaxScaler()
test_samples = []
for sample_id in test_df['event_id'].unique():
    sample = pd.DataFrame(
        test_df[test_df['event_id'] == sample_id].iloc[-1]).transpose()
    test_samples.append(sample)

df = pd.read_csv('')
y = df['y'].to_numpy()
df = df.drop(columns=['event_id', 'y', 'x_position_covariance_det', ])
X = df.to_numpy()
model = RandomForestClassifier(bootstrap=True, class_weight=None,
                               criterion='entropy',
                               max_depth=None, max_features='auto',
                               max_leaf_nodes=None,
                               min_impurity_decrease=0.0,
                               min_impurity_split=None,
                               min_samples_leaf=5, min_samples_split=20,
                               min_weight_fraction_leaf=0.0, n_estimators=100,
                               n_jobs=None, oob_score=False, random_state=0,
                               verbose=0,
                               warm_start=False)
cross_validation(model, X, y, cv=5)
model.fit(X, y)
test_samples = pd.concat(test_samples)
test_samples = test_samples.drop(
    columns=['event_id', 'x_position_covariance_det'])
x_test = test_samples.to_numpy()
predicts = model.predict_proba(x_test)
predicts = (predicts[:, 1] >= 0.45).astype(int)
binarized = []
for value in predicts:
    if value == 1:
        binarized.append(-5)
    else:
        binarized.append(-6.01)
pd.DataFrame(binarized, columns=['predicted_risk']).to_csv('',
                                                           index_label='event_id')
