import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# There are options to select different combinations of features:
# SELECTED_FEATURES = ['risk', 'max_risk_scaling', 'max_risk_estimate', 'miss_distance', 'mahalanobis_distance', 'c_sigma_t', 'c_sigma_rdot', 'relative_position_n', 'c_obs_available', 'c_obs_used', 'relative_position_t', 'c_time_lastob_start', 'c_time_lastob_end', 'c_cr_area_over_mass', 'c_sigma_r', 'c_cd_area_over_mass', 'c_rcs_estimate', 'c_cndot_n', 'c_sigma_n', 't_h_apo', 't_j2k_sma', 'relative_position_r', 'c_h_per', 'c_sedr', 'c_sigma_tdot', 'c_ctdot_r', 'c_recommended_od_span', 't_h_per', 't_j2k_inc', 't_rcs_estimate', 'c_crdot_t', 'c_sigma_ndot', 'relative_velocity_n', 'c_actual_od_span', 't_sedr', 'c_j2k_sma', 'relative_velocity_t', 'c_j2k_inc', 'relative_speed', 'relative_velocity_r']
# SELECTED_FEATURES = ['risk', 'max_risk_scaling', 'max_risk_estimate', 'miss_distance', 'mahalanobis_distance', 'c_sigma_t', 'c_sigma_rdot', 'relative_position_n', 'c_obs_available']
SELECTED_FEATURES = ['risk', 'max_risk_scaling', 'max_risk_estimate',
                     'miss_distance', 'mahalanobis_distance']
# SELECTED_FEATURES = ['risk', 'max_risk_scaling']

df_scaler = pd.read_csv('').drop(columns=['event_id'])
df_scaler = df_scaler[SELECTED_FEATURES]
scaler = MinMaxScaler()
scaler.fit(df_scaler)

df = pd.read_csv('')
y = df['last_cdm_risk'].to_numpy()
ones = np.ones([int(len(y) / 2)])
zeros = np.zeros([int(len(y) / 2)])
for i, value in enumerate(y):
    if value >= -6:
        y[i] = 1
    else:
        y[i] = 0
y[:int(len(y) / 2)] = ones
y[int(len(y) / 2):] = zeros
df = df.drop(columns=['event_id', 'mission_id', 'last_cdm_risk', 'delta_p'])
df = df[SELECTED_FEATURES]
X = df.to_numpy()
X = scaler.transform(X)

smote = SMOTETomek(random_state=0)
X, y = smote.fit_sample(X, y)

X = PCA(n_components=2).fit_transform(X)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', marker='x')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', marker='x')
plt.show()

print(np.unique(y, return_counts=True))
smote = SMOTE(random_state=0)
X_res, y_res = smote.fit_sample(X, y)
print(np.unique(y_res, return_counts=True))
