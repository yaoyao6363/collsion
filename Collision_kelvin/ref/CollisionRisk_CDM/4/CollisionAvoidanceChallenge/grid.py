import random

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# SELECTED_FEATURES = ['time_to_tca', 'risk', 'max_risk_estimate', 'max_risk_scaling', 'relative_speed',
#                      'relative_velocity_t', 'relative_velocity_n', 't_j2k_ecc', 'c_h_per', 'geocentric_latitude',
#                      't_j2k_inc', 'c_actual_od_span', 'c_j2k_inc', 'c_cndot_n', 'azimuth', 'c_sigma_ndot']
SELECTED_FEATURES = ['risk', 'max_risk_scaling', 'max_risk_estimate',
                     'miss_distance', 'mahalanobis_distance']


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
        predicts = (predicts[:, 1] >= 0.35).astype(int)
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

df_scaler = pd.read_csv('').drop(
    columns=['event_id'])
df_scaler = df_scaler[SELECTED_FEATURES]
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
df = df[SELECTED_FEATURES]
X = df.to_numpy()
X = scaler.transform(X)

X, y = SMOTETomek(random_state=0).fit_sample(X, y)

ftwo_score = make_scorer(fbeta_score, beta=2)
model = RandomForestClassifier(random_state=0)

param_grid = {'n_estimators': [50, 100, 150, 200],
              'criterion': ['gini', 'entropy'],
              'min_samples_leaf': [5, 7, 10, 15, 20],
              'min_samples_split': [2, 5, 20, 25, 30, 35, 40]}

search = GridSearchCV(estimator=model, param_grid=param_grid,
                      scoring=ftwo_score, n_jobs=4, cv=5, verbose=1)
search.fit(X, y)
print(search.best_estimator_)
results = search.cv_results_
results_df = pd.DataFrame(results)
results_df.to_csv('', index=False)
