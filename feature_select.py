import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
from boruta import BorutaPy

useful_columns = {}

training_df = pd.read_csv('clean_df/adjust_etf_df_span25_0615.csv')

for i in range(0, 5):
    rf = RandomForestRegressor(max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators=20, verbose=1, random_state=1, max_iter=10)
    not_use_columns = ['code_num', 'monday_idx', 'target_ratio_+0', 'target_ratio_+1', 'target_ratio_+2', \
                 'target_ratio_+3', 'target_ratio_+4']
    use_columns = list( set(training_df.columns.tolist()) - set(not_use_columns) )
    feat_selector.fit(training_df[use_columns].values, training_df['target_ratio_+'+str(i)].values)

    print('target_ratio_+'+str(i))
    print(np.array(use_columns)[feat_selector.support_])
    print(np.array(use_columns)[feat_selector.support_weak_])

    useful_columns['target_ratio_+'+str(i)] = list(np.array(use_columns)[feat_selector.support_])+list(np.array(use_columns)[feat_selector.support_weak_])

file = open('etf_boruta.pkl', 'wb')
pickle.dump(useful_columns, file)
file.close()
