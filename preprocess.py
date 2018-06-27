import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

adjust_etf_df = pd.read_csv('data/taetfp.csv', encoding='big5-hkscs')
adjust_normal_df = pd.read_csv('data/tasharep.csv', encoding='big5-hkscs')

dji_df = pd.read_csv('outside_data/^DJI.csv')
gspc_df = pd.read_csv('outside_data/^GSPC.csv')
ixic_df = pd.read_csv('outside_data/^IXIC.csv')
nya_df = pd.read_csv('outside_data/^NYA.csv')
sox_df = pd.read_csv('outside_data/^SOX.csv')

vix_df = pd.read_csv('outside_data/^VIX.csv')
bnd_df = pd.read_csv('outside_data/BND.csv')
fxi_df = pd.read_csv('outside_data/FXI.csv')
iau_df = pd.read_csv('outside_data/IAU.csv')
ief_df = pd.read_csv('outside_data/IEF.csv')
shy_df = pd.read_csv('outside_data/SHY.csv')
tlt_df = pd.read_csv('outside_data/TLT.csv')
uso_df = pd.read_csv('outside_data/USO.csv')
vgk_df = pd.read_csv('outside_data/VGK.csv')
vt_df = pd.read_csv('outside_data/VT.csv')

dji_df['Date'] = pd.to_datetime(dji_df['Date'])
gspc_df['Date'] = pd.to_datetime(gspc_df['Date'])
ixic_df['Date'] = pd.to_datetime(ixic_df['Date'])
nya_df['Date'] = pd.to_datetime(nya_df['Date'])
sox_df['Date'] = pd.to_datetime(sox_df['Date'])

vix_df['Date'] = pd.to_datetime(vix_df['Date'])
bnd_df['Date'] = pd.to_datetime(bnd_df['Date'])
fxi_df['Date'] = pd.to_datetime(fxi_df['Date'])
iau_df['Date'] = pd.to_datetime(iau_df['Date'])
ief_df['Date'] = pd.to_datetime(ief_df['Date'])
shy_df['Date'] = pd.to_datetime(shy_df['Date'])
tlt_df['Date'] = pd.to_datetime(tlt_df['Date'])
uso_df['Date'] = pd.to_datetime(uso_df['Date'])
vgk_df['Date'] = pd.to_datetime(vgk_df['Date'])
vt_df['Date'] = pd.to_datetime(vt_df['Date'])

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

def clean_data_us_stock(df):
    df = df.copy()

    df['weekday'] = df.Date.dt.weekday+1

    df['volume_adj'] = df['Volume']
    df.loc[df['volume_adj']==0, 'volume_adj'] = 1

    df['close_shift1'] = df['Adj Close'].shift(1)
    df['volume_adj_shift1'] = df['volume_adj'].shift(1)

    df['close_diff'] = df['Adj Close'].diff()
    df['volume_adj_diff'] = df['volume_adj'].diff()

    df = df.dropna()

    df['close_ratio'] = df['close_diff'] / df['close_shift1']
    df['volume_adj_ratio'] = df['volume_adj_diff'] / df['volume_adj_shift1']

    return df

adjust_etf_df = clean_data_taiwan(adjust_etf_df)
adjust_normal_df = clean_data_taiwan(adjust_normal_df)
adjust_etf_df = adjust_etf_df.reset_index()
adjust_normal_df = adjust_normal_df.reset_index()

dji_df = clean_data_us_stock(dji_df)
dji_df = dji_df.reset_index()

gspc_df = clean_data_us_stock(gspc_df)
gspc_df = gspc_df.reset_index()

ixic_df = clean_data_us_stock(ixic_df)
ixic_df = ixic_df.reset_index()

nya_df = clean_data_us_stock(nya_df)
nya_df = nya_df.reset_index()

sox_df = clean_data_us_stock(sox_df)
sox_df = sox_df.reset_index()


vix_df = clean_data_us_stock(vix_df)
vix_df = vix_df.reset_index()

bnd_df = clean_data_us_stock(bnd_df)
bnd_df = bnd_df.reset_index()

fxi_df = clean_data_us_stock(fxi_df)
fxi_df = fxi_df.reset_index()

iau_df = clean_data_us_stock(iau_df)
iau_df = iau_df.reset_index()

ief_df = clean_data_us_stock(ief_df)
ief_df = ief_df.reset_index()


shy_df = clean_data_us_stock(shy_df)
shy_df = shy_df.reset_index()

tlt_df = clean_data_us_stock(tlt_df)
tlt_df = tlt_df.reset_index()

uso_df = clean_data_us_stock(uso_df)
uso_df = uso_df.reset_index()

vgk_df = clean_data_us_stock(vgk_df)
vgk_df = vgk_df.reset_index()

vt_df = clean_data_us_stock(vt_df)
vt_df = vt_df.reset_index()

twd_df = pd.read_csv('outside_data/twd.csv')
twd_df['Date'] = pd.to_datetime(twd_df['日期'], format='%Y/%m/%d')

def clean_data_twd(df):
    df = df.copy()

    df['weekday'] = df.Date.dt.weekday+1

    df['美元／新台幣_shift1'] = df['美元／新台幣'].shift(1)
    df['歐元／美元_shift1'] = df['歐元／美元'].shift(1)
    df['美元／日幣_shift1'] = df['美元／日幣'].shift(1)
    df['英鎊／美元_shift1'] = df['英鎊／美元'].shift(1)
    df['澳幣／美元_shift1'] = df['澳幣／美元'].shift(1)
    df['美元／港幣_shift1'] = df['美元／港幣'].shift(1)
    df['美元／人民幣_shift1'] = df['美元／人民幣'].shift(1)

    df['美元／新台幣_diff'] = df['美元／新台幣'].diff()
    df['歐元／美元_diff'] = df['歐元／美元'].diff()
    df['美元／日幣_diff'] = df['美元／日幣'].diff()
    df['英鎊／美元_diff'] = df['英鎊／美元'].diff()
    df['澳幣／美元_diff'] = df['澳幣／美元'].diff()
    df['美元／港幣_diff'] = df['美元／港幣'].diff()
    df['美元／人民幣_diff'] = df['美元／人民幣'].diff()

    df = df.dropna()

    df['美元／新台幣_ratio'] = df['美元／新台幣_diff'] / df['美元／新台幣_shift1']
    df['歐元／美元_ratio'] = df['歐元／美元_diff'] / df['歐元／美元_shift1']
    df['美元／日幣_ratio'] = df['美元／日幣_diff'] / df['美元／日幣_shift1']
    df['英鎊／美元_ratio'] = df['英鎊／美元_diff'] / df['英鎊／美元_shift1']
    df['澳幣／美元_ratio'] = df['澳幣／美元_diff'] / df['澳幣／美元_shift1']
    df['美元／港幣_ratio'] = df['美元／港幣_diff'] / df['美元／港幣_shift1']
    df['美元／人民幣_ratio'] = df['美元／人民幣_diff'] / df['美元／人民幣_shift1']

    return df

twd_df = clean_data_twd(twd_df)
twd_df = twd_df.reset_index()

adjust_etf_moday_df = adjust_etf_df[adjust_etf_df['weekday']==1]

def get_info_from_us_stock(df, the_date, pre_info_span, name):
    minus_day = 0
    while df[df['Date']==the_date-timedelta(days=minus_day)].shape[0]<=0:
        minus_day = minus_day + 1

    the_idx = df[df['Date']==the_date-timedelta(days=minus_day)].iloc[0].name

    temp_row = {}
    for i in range(pre_info_span, 0, -1):
        temp_row['close_ratio_'+name+'_-'+str(i)] = df.loc[the_idx-i, 'close_ratio']

    return temp_row

def get_info_from_twd(df, the_date, pre_info_span):
    minus_day = 0
    while df[df['Date']==the_date-timedelta(days=minus_day)].shape[0]<=0:
        minus_day = minus_day + 1

    the_idx = df[df['Date']==the_date-timedelta(days=minus_day)].iloc[0].name

    temp_row = {}
    twd_list = ['美元／新台幣', '歐元／美元', '美元／日幣', '英鎊／美元', '澳幣／美元', '美元／港幣', '美元／人民幣']
    for i in range(pre_info_span, 0, -1):
        for twd_name in twd_list:
            temp_row[twd_name+'_ratio_-'+str(i)] = df.loc[the_idx-i, twd_name+'_ratio']

    return temp_row

training_etf_df = pd.DataFrame()
pre_info_span = 25
for idx, data in tqdm(adjust_etf_moday_df.iterrows(), total=adjust_etf_moday_df.shape[0]):
    if idx-pre_info_span >= 0 and idx+4 < adjust_etf_df.shape[0]:
        if (adjust_etf_df.loc[idx-pre_info_span, '代碼']==data['代碼']) and (adjust_etf_df.loc[idx+4, '代碼']==data['代碼']) and \
            (adjust_etf_df.loc[idx+4, 'weekday']==5):

            last_price = adjust_etf_df.loc[idx-1, '收盤價(元)']
            temp_row = {}
            temp_row['monday_idx'] = idx
            temp_row['code_num'] = adjust_etf_df.loc[idx-1, '代碼']
            for i in range(pre_info_span, 0, -1):
                temp_row['close_ratio_-'+str(i)] = adjust_etf_df.loc[idx-i, 'close_ratio']

            temp_row.update( get_info_from_us_stock(dji_df, data['Date'], pre_info_span, 'dji') )
            temp_row.update( get_info_from_us_stock(gspc_df, data['Date'], pre_info_span, 'gspc') )
            temp_row.update( get_info_from_us_stock(ixic_df, data['Date'], pre_info_span, 'ixic') )
            temp_row.update( get_info_from_us_stock(nya_df, data['Date'], pre_info_span, 'nya') )
            temp_row.update( get_info_from_us_stock(sox_df, data['Date'], pre_info_span, 'sox') )
            temp_row.update( get_info_from_us_stock(vix_df, data['Date'], pre_info_span, 'vix') )
            temp_row.update( get_info_from_us_stock(bnd_df, data['Date'], pre_info_span, 'bnd') )
            temp_row.update( get_info_from_us_stock(fxi_df, data['Date'], pre_info_span, 'fxi') )
            temp_row.update( get_info_from_us_stock(iau_df, data['Date'], pre_info_span, 'iau') )
            temp_row.update( get_info_from_us_stock(ief_df, data['Date'], pre_info_span, 'ief') )
            temp_row.update( get_info_from_us_stock(shy_df, data['Date'], pre_info_span, 'shy') )
            temp_row.update( get_info_from_us_stock(tlt_df, data['Date'], pre_info_span, 'tlt') )
            temp_row.update( get_info_from_us_stock(uso_df, data['Date'], pre_info_span, 'uso') )
            temp_row.update( get_info_from_us_stock(vgk_df, data['Date'], pre_info_span, 'vgk') )
            temp_row.update( get_info_from_us_stock(vt_df, data['Date'], pre_info_span, 'vt') )

            temp_row.update( get_info_from_twd(twd_df, data['Date'], pre_info_span) )

            for i in range(0, 5):
                temp_row['target_ratio_+'+str(i)] = (adjust_etf_df.loc[idx+i, '收盤價(元)']-last_price)/last_price
            training_etf_df = training_etf_df.append(temp_row, ignore_index=True)

training_etf_df.to_csv('clean_df/adjust_etf_df_span25_0615.csv', index=False)

adjust_normal_moday_df = adjust_normal_df[adjust_normal_df['weekday']==1]
cache_time_df = pd.DataFrame(columns=['date'])

training_normal_dict = {}
pre_info_span = 25
for idx, data in tqdm(adjust_normal_moday_df.iterrows(), total=adjust_normal_moday_df.shape[0]):
    if idx-pre_info_span >= 0 and idx+4 < adjust_normal_df.shape[0]:
        if (adjust_normal_df.loc[idx-pre_info_span, '代碼']==data['代碼']) and (adjust_normal_df.loc[idx+4, '代碼']==data['代碼']) and \
            (adjust_normal_df.loc[idx+4, 'weekday']==5):

            last_price = adjust_normal_df.loc[idx-1, '收盤價(元)']
            temp_row = {}
            temp_row['monday_idx'] = idx
            temp_row['code_num'] = adjust_normal_df.loc[idx-1, '代碼']
            for i in range(pre_info_span, 0, -1):
                temp_row['close_ratio_-'+str(i)] = adjust_normal_df.loc[idx-i, 'close_ratio']

            if cache_time_df[cache_time_df['date']==data['Date']].shape[0]==0:
                other_data_row = {}
                other_data_row.update( get_info_from_us_stock(dji_df, data['Date'], pre_info_span, 'dji') )
                other_data_row.update( get_info_from_us_stock(gspc_df, data['Date'], pre_info_span, 'gspc') )
                other_data_row.update( get_info_from_us_stock(ixic_df, data['Date'], pre_info_span, 'ixic') )
                other_data_row.update( get_info_from_us_stock(nya_df, data['Date'], pre_info_span, 'nya') )
                other_data_row.update( get_info_from_us_stock(sox_df, data['Date'], pre_info_span, 'sox') )
                other_data_row.update( get_info_from_us_stock(vix_df, data['Date'], pre_info_span, 'vix') )
                other_data_row.update( get_info_from_us_stock(bnd_df, data['Date'], pre_info_span, 'bnd') )
                other_data_row.update( get_info_from_us_stock(fxi_df, data['Date'], pre_info_span, 'fxi') )
                other_data_row.update( get_info_from_us_stock(iau_df, data['Date'], pre_info_span, 'iau') )
                other_data_row.update( get_info_from_us_stock(ief_df, data['Date'], pre_info_span, 'ief') )
                other_data_row.update( get_info_from_us_stock(shy_df, data['Date'], pre_info_span, 'shy') )
                other_data_row.update( get_info_from_us_stock(tlt_df, data['Date'], pre_info_span, 'tlt') )
                other_data_row.update( get_info_from_us_stock(uso_df, data['Date'], pre_info_span, 'uso') )
                other_data_row.update( get_info_from_us_stock(vgk_df, data['Date'], pre_info_span, 'vgk') )
                other_data_row.update( get_info_from_us_stock(vt_df, data['Date'], pre_info_span, 'vt') )

                other_data_row.update( get_info_from_twd(twd_df, data['Date'], pre_info_span) )

                cache_time_df = cache_time_df.append(dict({'date': data['Date']}, **other_data_row), ignore_index=True)
            else:
                other_data_row = cache_time_df[cache_time_df['date']==data['Date']].iloc[0]
                del other_data_row['date']

            temp_row.update(other_data_row)

            for i in range(0, 5):
                temp_row['target_ratio_+'+str(i)] = (adjust_normal_df.loc[idx+i, '收盤價(元)']-last_price)/last_price

            if len(training_normal_dict)==0:
                for key, value in temp_row.items():
                    training_normal_dict[key] = []

            for key, value in temp_row.items():
                training_normal_dict[key].append(value)

training_normal_df = pd.DataFrame.from_dict(training_normal_dict)
training_normal_df.to_csv('clean_df/adjust_normal_df_span25_0615.csv', index=False)

def get_info_from_us_stock_last(df, the_date, pre_info_span, name):
    the_idx = df.iloc[df.shape[0]-1].name

    temp_row = {}
    for i in range(pre_info_span, 0, -1):
        temp_row['close_ratio_'+name+'_-'+str(i)] = df.loc[the_idx-i+1, 'close_ratio']

    return temp_row

def get_info_from_twd_last(df, the_date, pre_info_span):
    the_idx = df.iloc[df.shape[0]-1].name

    temp_row = {}
    twd_list = ['美元／新台幣', '歐元／美元', '美元／日幣', '英鎊／美元', '澳幣／美元', '美元／港幣', '美元／人民幣']
    for i in range(pre_info_span, 0, -1):
        for twd_name in twd_list:
            temp_row[twd_name+'_ratio_-'+str(i)] = df.loc[the_idx-i+1, twd_name+'_ratio']

    return temp_row

pre_info_span = 25
predict_etf_df = pd.DataFrame()
for idx, data in adjust_etf_df.groupby('代碼').last().iterrows():
    idx = adjust_etf_df[adjust_etf_df['index']==data['index']].iloc[0].name+1

    last_price = adjust_etf_df.loc[idx-1, '收盤價(元)']
    temp_row = {}
    temp_row['monday_idx'] = idx
    temp_row['code_num'] = adjust_etf_df.loc[idx-1, '代碼']
    for i in range(pre_info_span, 0, -1):
        temp_row['close_ratio_-'+str(i)] = adjust_etf_df.loc[idx-i, 'close_ratio']

        temp_row.update( get_info_from_us_stock_last(dji_df, data['Date']+timedelta(days=3), pre_info_span, 'dji') )
        temp_row.update( get_info_from_us_stock_last(gspc_df, data['Date']+timedelta(days=3), pre_info_span, 'gspc') )
        temp_row.update( get_info_from_us_stock_last(ixic_df, data['Date']+timedelta(days=3), pre_info_span, 'ixic') )
        temp_row.update( get_info_from_us_stock_last(nya_df, data['Date']+timedelta(days=3), pre_info_span, 'nya') )
        temp_row.update( get_info_from_us_stock_last(sox_df, data['Date']+timedelta(days=3), pre_info_span, 'sox') )
        temp_row.update( get_info_from_us_stock_last(vix_df, data['Date']+timedelta(days=3), pre_info_span, 'vix') )
        temp_row.update( get_info_from_us_stock_last(bnd_df, data['Date']+timedelta(days=3), pre_info_span, 'bnd') )
        temp_row.update( get_info_from_us_stock_last(fxi_df, data['Date']+timedelta(days=3), pre_info_span, 'fxi') )
        temp_row.update( get_info_from_us_stock_last(iau_df, data['Date']+timedelta(days=3), pre_info_span, 'iau') )
        temp_row.update( get_info_from_us_stock_last(ief_df, data['Date']+timedelta(days=3), pre_info_span, 'ief') )
        temp_row.update( get_info_from_us_stock_last(shy_df, data['Date']+timedelta(days=3), pre_info_span, 'shy') )
        temp_row.update( get_info_from_us_stock_last(tlt_df, data['Date']+timedelta(days=3), pre_info_span, 'tlt') )
        temp_row.update( get_info_from_us_stock_last(uso_df, data['Date']+timedelta(days=3), pre_info_span, 'uso') )
        temp_row.update( get_info_from_us_stock_last(vgk_df, data['Date']+timedelta(days=3), pre_info_span, 'vgk') )
        temp_row.update( get_info_from_us_stock_last(vt_df, data['Date']+timedelta(days=3), pre_info_span, 'vt') )

        temp_row.update( get_info_from_twd_last(twd_df, data['Date']+timedelta(days=3), pre_info_span) )

    predict_etf_df = predict_etf_df.append(temp_row, ignore_index=True)

predict_etf_df.to_csv('clean_df/predict_df_0615.csv', index=False)

result_df = pd.DataFrame()
for i in range(2013, 2019):
    result_df = result_df.append(pd.read_csv('outside_data/'+str(i)+'_twd.csv', encoding='big5'), ignore_index=True)
result_df.to_csv('outside_data/twd.csv', index=False)
