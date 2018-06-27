import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from pprint import pprint

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import matplotlib.pyplot as plt

adjust_etf_df = pd.read_csv('data/taetfp.csv', encoding='big5-hkscs')
adjust_normal_df = pd.read_csv('data/tasharep.csv', encoding='big5-hkscs')

adjust_etf_df['Date'] = pd.to_datetime(adjust_etf_df['日期'], format='%Y%m%d')
adjust_normal_df['Date'] = pd.to_datetime(adjust_normal_df['日期'], format='%Y%m%d')

def clean_data_taiwan(df):
    df = df.copy()

    df['weekday'] = df.Date.dt.weekday+1

    df['成交張數(張)'] = df['成交張數(張)'].str.replace(',', '').astype(int)
    df['volume_adj'] = df['成交張數(張)']
    df.loc[df['volume_adj']==0, 'volume_adj'] = 1

    if type(df['開盤價(元)'][0]) != np.float64:
        df['開盤價(元)'] = df['開盤價(元)'].astype(str)
        df['開盤價(元)'] = df['開盤價(元)'].str.replace(',', '').astype(float)

        df['最高價(元)'] = df['最高價(元)'].astype(str)
        df['最高價(元)'] = df['最高價(元)'].str.replace(',', '').astype(float)

        df['最低價(元)'] = df['最低價(元)'].astype(str)
        df['最低價(元)'] = df['最低價(元)'].str.replace(',', '').astype(float)

        df['收盤價(元)'] = df['收盤價(元)'].astype(str)
        df['收盤價(元)'] = df['收盤價(元)'].str.replace(',', '').astype(float)

    df['close_shift1'] = df.groupby('代碼')['收盤價(元)'].shift(1)
    df['volume_adj_shift1'] = df.groupby('代碼')['volume_adj'].shift(1)

    df['close_diff'] = df.groupby('代碼')['收盤價(元)'].diff()
    df['volume_adj_diff'] = df.groupby('代碼')['volume_adj'].diff()

    df = df.dropna()

    df['close_ratio'] = df['close_diff'] / df['close_shift1']
    df['volume_adj_ratio'] = df['volume_adj_diff'] / df['volume_adj_shift1']

    return df

adjust_etf_df = clean_data_taiwan(adjust_etf_df)
adjust_normal_df = clean_data_taiwan(adjust_normal_df)

adjust_etf_df = adjust_etf_df.reset_index()
adjust_normal_df = adjust_normal_df.reset_index()

etf_codes = adjust_etf_df['代碼'].unique().tolist()
normal_codes = adjust_normal_df['代碼'].unique().tolist()

etf_100_correlation = {}

for etf_code in etf_codes:
    temp_etf_df = adjust_etf_df[adjust_etf_df['代碼']==etf_code].copy()

    correlation_dict = {}
    for normal_code in normal_codes:
        temp_normal_df = adjust_normal_df[adjust_normal_df['代碼']==normal_code].copy()

        temp_merge_df = temp_etf_df[['Date', '收盤價(元)']].merge(temp_normal_df[['Date', '收盤價(元)']], on='Date')
        temp_merge_df = temp_merge_df.dropna()

        temp_corrcoef = np.corrcoef(temp_merge_df['收盤價(元)_x'], temp_merge_df['收盤價(元)_y'])[0][1]

        correlation_dict[normal_code] = temp_corrcoef

    top_100_code = [i[0] for i in sorted(correlation_dict.items(), key=lambda x:x[1], reverse=True)][:100]
    etf_100_correlation[etf_code] = top_100_code


training_etf_df = pd.read_csv('clean_df/adjust_etf_df_span25_0615.csv')
training_df = pd.read_csv('clean_df/adjust_normal_df_span25_0615.csv')

not_use_columns = ['code_num', 'monday_idx', 'target_ratio_+0', 'target_ratio_+1', 'target_ratio_+2', \
                 'target_ratio_+3', 'target_ratio_+4']
use_columns = list( set(training_df.columns.tolist()) - set(not_use_columns) )

etf_boruta = pickle.load(open("etf_boruta.pkl", "rb"))

gb_rs = {}
for etf_code in tqdm(etf_codes):
    gb_rs[etf_code] = []
    for i in range(0, 5):
        gb_r = GradientBoostingRegressor(max_depth=8, min_samples_leaf=9, subsample=0.8, learning_rate=0.02, n_estimators=80, verbose=1)
        training_df[training_df['code_num'].isin(etf_100_correlation[etf_code])].copy()
        gb_r.fit(temp_training_df[etf_boruta['target_ratio_+'+str(i)]], \
                temp_training_df['target_ratio_+'+str(i)])
        gb_rs[etf_code].append(gb_r)

predict_etf_df = pd.read_csv('clean_df/predict_df_0615.csv')
etf_predicted = []
for idx, data in tqdm(predict_etf_df.iterrows(), total=predict_etf_df.shape[0]):
    temp_row = []
    for i in range(0, 5):
        temp_row.append( \
            gb_rs[data['code_num']][i].predict(data[etf_boruta['target_ratio_+'+str(i)]].values.reshape(1, -1))[0]
        )

    etf_predicted.append(temp_row)

real_price_predicted = []
for idx, predicts in enumerate(etf_predicted):
    monday_idx = predict_etf_df.iloc[idx]['monday_idx']
    last_price = adjust_etf_df.loc[monday_idx-1, '收盤價(元)']
    predict_prices = []
    for idx2, predict in enumerate(predicts):
        predict_prices.append(last_price+last_price*predict)

    real_price_predicted.append(predict_prices)

# output
result_df = pd.DataFrame(columns=['ETFid', 'Mon_ud', 'Mon_cprice', 'Tue_ud', 'Tue_cprice', 'Wed_ud', 'Wed_cprice', \
                                 'Thu_ud', 'Thu_cprice', 'Fri_ud', 'Fri_cprice'])
count = 0
for idx, data in predict_etf_df.iterrows():
    monday_idx = predict_etf_df.iloc[idx]['monday_idx']
    last_price = adjust_etf_df.loc[monday_idx-1, '收盤價(元)']
    predict_prices = real_price_predicted[count]
    result_df = result_df.append({'ETFid': str(int(data['code_num'])).zfill(4), \
                                  'Mon_ud': int(np.sign(predict_prices[0]-last_price)), 'Mon_cprice': predict_prices[0], \
                                  'Tue_ud': int(np.sign(predict_prices[1]-predict_prices[0])), 'Tue_cprice': predict_prices[1], \
                                  'Wed_ud': int(np.sign(predict_prices[2]-predict_prices[1])), 'Wed_cprice': predict_prices[2], \
                                  'Thu_ud': int(np.sign(predict_prices[3]-predict_prices[2])), 'Thu_cprice': predict_prices[3], \
                                  'Fri_ud': int(np.sign(predict_prices[4]-predict_prices[3])), 'Fri_cprice': predict_prices[4]}, \
                                ignore_index=True)
    count = count + 1
