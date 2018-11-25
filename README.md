# TBrain 台灣ETF價格預測競賽
https://tbrain.trendmicro.com.tw/Competitions/Details/2

## 程式運作順序

preprocess.py

feature_select.py

train_model.py


## 結果
Rank: 4/487


## 特徵
國外各式股市指數漲跌（From: http://finance.yahoo.com/ ）、外匯漲跌（From: http://www.taifex.com.tw/chinese/3/3_5.asp ）、與 ETF 關聯性最高的前100名個股漲跌。取25天。

## 模型
用Boruta做特徵選取， cv 找最佳參數，以 Gradient Boosting 的方式預測各個 ETF 個天的漲跌幅。
