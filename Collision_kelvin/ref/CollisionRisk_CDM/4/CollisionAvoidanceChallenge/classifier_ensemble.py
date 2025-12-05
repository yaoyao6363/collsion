import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# SELECTED_FEATURES = ['time_to_tca', 'risk', 'max_risk_estimate', 'max_risk_scaling', 'relative_speed',
#                      'relative_velocity_t', 'relative_velocity_n', 't_j2k_ecc', 'c_h_per', 'geocentric_latitude',
#                      't_j2k_inc', 'c_actual_od_span', 'c_j2k_inc', 'c_cndot_n', 'azimuth', 'c_sigma_ndot']
SELECTED_FEATURES = ['risk', 'time_to_tca', 'max_risk_scaling',
                     'max_risk_estimate', 'miss_distance',
                     'mahalanobis_distance', 'relative_speed']
FOLDS = 10


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
        f2_scores.append(f2)
        print('F2: {}'.format(f2))
    print('Avg f2: {}'.format(np.mean(f2_scores)))
    return np.mean(f2_scores)


test_df = pd.read_csv('')

df_scaler = pd.read_csv('').drop(columns=['event_id', 'mission_id'])
df_scaler = df_scaler[SELECTED_FEATURES]
scaler = MinMaxScaler()
scaler.fit(df_scaler)
test_samples = []
for sample_id in test_df['event_id'].unique():
    sample = pd.DataFrame(
        test_df[test_df['event_id'] == sample_id].iloc[-1]).transpose()
    test_samples.append(sample)
f2_folds = []
models = []
for fold in range(FOLDS):
    df = pd.read_csv(''.format(fold))
    y = df['last_cdm_risk'].to_numpy()
    y[y >= -6] = 1
    y[y < -6] = 0
    df = df.drop(
        columns=['event_id', 'mission_id', 'last_cdm_risk', 'delta_p'])
    df = df[SELECTED_FEATURES]
    X = df.to_numpy()
    model = RandomForestClassifier(bootstrap=True, class_weight=None,
                                   criterion='entropy',
                                   max_depth=None, max_features='auto',
                                   max_leaf_nodes=None,
                                   min_impurity_decrease=0.0,
                                   min_impurity_split=None,
                                   min_samples_leaf=5, min_samples_split=20,
                                   min_weight_fraction_leaf=0.0,
                                   n_estimators=100,
                                   n_jobs=None, oob_score=False,
                                   random_state=0, verbose=0,
                                   warm_start=False)
    f2_folds.append(cross_validation(model, X, y, cv=5))
    print('Training model on full dataset')
    models.append(model.fit(X, y))
print("Average across {} datasets: {}".format(FOLDS, np.mean(f2_folds)))
test_samples = pd.concat(test_samples)
test_samples = test_samples[SELECTED_FEATURES]
x_test = test_samples.to_numpy()
all_models_predicts = []
for model in models:
    all_models_predicts.append(model.predict_proba(x_test)[:, 1])
all_models_predicts = np.vstack(all_models_predicts)
all_models_predicts = np.average(all_models_predicts, axis=0)
predicts = []
for i, value in enumerate(all_models_predicts):
    if value >= 0.5:
        predicts.append(-5)
    else:
        predicts.append(-6.01)
pd.DataFrame(predicts, columns=['predicted_risk']).to_csv('',
                                                          index_label='event_id')
