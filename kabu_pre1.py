#!/usr/bin/env python
# coding: utf-8

# In[7]:


# 上場企業４２３社の２０年分を学習
# 終値前日比1.03%以上であるかを２クラス分類
# データ比１１：１を１：１に調整（減らした）
# StandardScaler accuracy
import numpy as np
import pandas as pd
from sklearn import *
import seaborn as sns
from sklearn.model_selection import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings
import mglearn
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import glob

# 実行上問題ない注意は非表示にする
warnings.filterwarnings('ignore') 


# In[8]:


# data/kabu1フォルダ内にあるcsvファイルの一覧を取得
files = glob.glob("data/kabu1/*.csv")


# In[9]:


# 説明変数となる行列X, 被説明変数となるy2を作成
base = 100 
day_ago = 3
num_sihyou = 8
reset =True
# すべてのCSVファイルから得微量作成
for file in files:
    temp = pd.read_csv(file, header=0, encoding='cp932')
    temp = temp[['日付','始値', '高値','安値','終値','5日平均','25日平均','75日平均','出来高']]
    temp= temp.iloc[::-1]#上下反対に
    temp2 = np.array(temp)
    
    # 前日比を出すためにbase日後からのデータを取得
    temp3 = np.zeros((len(temp2)-base, num_sihyou))
    temp3[0:len(temp3), 0] = temp2[base:len(temp2), 4] / temp2[base-1:len(temp2)-1, 4]
    temp3[0:len(temp3), 1] = temp2[base:len(temp2), 1] / temp2[base:len(temp2), 4]
    temp3[0:len(temp3), 2] = temp2[base:len(temp2), 2] / temp2[base:len(temp2), 4]
    temp3[0:len(temp3), 3] = temp2[base:len(temp2), 3] / temp2[base:len(temp2), 4]
    temp3[0:len(temp3), 4] = temp2[base:len(temp2), 5].astype(np.float) / temp2[base:len(temp2), 4].astype(np.float)
    temp3[0:len(temp3), 5] = temp2[base:len(temp2), 6].astype(np.float) / temp2[base:len(temp2), 4].astype(np.float)
    temp3[0:len(temp3), 6] = temp2[base:len(temp2), 7].astype(np.float) / temp2[base:len(temp2), 4].astype(np.float)
    temp3[0:len(temp3), 7] = temp2[base:len(temp2), 8].astype(np.float) / temp2[base-1:len(temp2)-1, 8].astype(np.float)
    
    # tempX : 現在の企業のデータ
    tempX = np.zeros((len(temp3), day_ago*num_sihyou))
    
    # 日にちごとに横向きに（day_ago）分並べる
    # sckit-learnは過去の情報を学習できないので、複数日（day_ago）分を特微量に加える必要がある
    # 注：tempX[0:day_ago]分は欠如データが生まれる
    for s in range(0, num_sihyou): 
        for i in range(0, day_ago):
            tempX[i:len(temp3), day_ago*s+i] = temp3[0:len(temp3)-i,s]
             
    # Xに追加
    # X : すべての企業のデータ
    # tempX[0:day_ago]分は削除
    if reset:
        X = tempX[day_ago:]
        reset = False
    else:
        X = np.concatenate((X, tempX[day_ago:]), axis=0)

# 何日後を値段の差を予測するのか
pre_day = 1
# y : pre_day後の終値/当日終値
y = np.zeros(len(X))
y[0:len(y)-pre_day] = X[pre_day:len(X),0]
X = X[:-pre_day]
y = y[:-pre_day]

up_rate =1.03

# データを一旦分別
X_0 = X[y<=up_rate]
X_1 = X[y>up_rate]
y_0 = y[y<=up_rate]
y_1 = y[y>up_rate]

# X_0をX_1とほぼ同じ数にする
X_drop, X_t, y_drop, y_t = train_test_split(X_0, y_0, test_size=0.09, random_state=0)

# 分別したデータの結合
X_ = np.concatenate((X_1, X_t), axis=0)
y_ = np.concatenate((y_1, y_t))


# In[10]:


# 確認
# 何もしないときの比率
print("X.shape: ", X.shape)

print("yの割合")
# yc：翌日の終値/当日の終値がup_rateより上か
yc = np.zeros(len(y))
for i in range(0, len(yc)):
    if y[i] <= up_rate:
        yc[i] = 0
    else:
        yc[i] = 1

pd_yc = pd.DataFrame(yc)
print(pd_yc[0].value_counts())

'''出力結果
X.shape:  (1734326, 24)
yの割合
0.0    1594644
1.0     139682
Name: 0, dtype: int64
'''


# In[11]:


# 確認
# X_0をX_1の数を調整後の比率
print("X_.shape: ", X_.shape)

print("y_の割合")
# yc：翌日の終値/当日の終値がup_rateより上か
yc_ = np.zeros(len(y_))
for i in range(0, len(yc_)):
    if y_[i] <= up_rate:
        yc_[i] = 0
    else:
        yc_[i] = 1

pd_yc_ = pd.DataFrame(yc_)
print(pd_yc_[0].value_counts())

'''出力結果
X_.shape:  (283200, 24)
y_の割合
0.0    143518
1.0    139682
Name: 0, dtype: int64
'''


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X_, y_, random_state=0)

# y_train_,y_test2：翌日の終値/当日の終値がup_rateより上か
y_train2 = np.zeros(len(y_train))
for i in range(0, len(y_train2)):
    if y_train[i] <= up_rate:
        y_train2[i] = 0
    else:
        y_train2[i] = 1
        
y_test2 = np.zeros(len(y_test))
for i in range(0, len(y_test2)):
    if y_test[i] <= up_rate:
        y_test2[i] = 0
    else:
        y_test2[i] = 1


# In[13]:


print("start time: ", datetime.now().strftime("%H:%M:%S"))
pipe = Pipeline([('scaler', MinMaxScaler()), ('classifier', MLPClassifier(max_iter=200000, alpha=0.001, hidden_layer_sizes=(1,)))])
param_grid = {'scaler': [MinMaxScaler(), StandardScaler(), None]}

grid = GridSearchCV(pipe, param_grid=param_grid, n_jobs=1, cv=2 ,return_train_score=False, scoring="accuracy")
grid.fit(X_train, y_train2)

print(grid.cv_results_['mean_test_score'])
print("Best parameters: ", grid.best_params_)
print("grid best score, ", grid.best_score_)
print("Test set score: {:.2f}".format(grid.score(X_test, y_test2)))

# 混同行列で確認
conf = confusion_matrix(y_test2, grid.predict(X_test))
print(conf)
print("Test set precision score(再現率): {:.2f}".format(conf[1,1]/(conf[0,1]+conf[1,1])))

print("over time: ", datetime.now().strftime("%H:%M:%S"))

'''出力結果
start time:  17:38:06
[0.56806497 0.63490113 0.60404426]
Best parameters:  {'scaler': StandardScaler(copy=True, with_mean=True, with_std=True)}
grid best score,  0.6349011299435028
Test set score: 0.64
[[27496  8471]
 [17331 17502]]
Test set precision score(再現率): 0.67
over time:  17:39:10
'''


# In[14]:


# シュミレーション（株価終値前日比＋３％が約２日に１回起こるので正確な結果ではない）
# 予測結果の合計を計算（空売り無し）
# 上がると予測したら終値で買い,翌日の終値で売ったと想定：掛け金☓翌日の上昇値
# tray_day日間
try_day = 50
y_pred = grid.predict(X_test)
for j in range(0, 5):
    a=random.randrange(len(y_test2)-try_day)
    X_test_try = X_test[a:a+try_day]
    y_test2_try = y_test2[a:a+try_day]
    y_test_try = y_test[a:a+try_day]
    y_pred_try = y_pred[a:a+try_day]
    
    c_ = 0
    win_c = 0
    money = 10000

    # 予測結果の総和グラフを描く
    total_return = np.zeros(len(y_test2_try))
    for i in range(0, try_day): 
        if y_pred_try[i] == 1:
            money = money*y_test_try[i]
            c_ +=1
            if y_test_try[i] >= 1:
                win_c +=1
            
        total_return[i] = money
    
    # 混同行列で確認
    conf = confusion_matrix(y_test2_try, grid.predict(X_test_try))
    
    # 上昇予測回数が０のときは、勝率、再現率を９９９にする
    if c_==0:
        win_score=999
    else:
        win_score = win_c / c_
        
    if conf[0,1]==0 and conf[1,1]==0:
        pre_score=999
    else:
        pre_score = conf[1,1]/(conf[0,1]+conf[1,1])

    print("投資結果：10000 円 → %1.3lf" %money, "円", "(買い回数：%1.3lf" %c_, "勝ち：%1.3lf" %win_c, "勝率：%1.3lf" %win_score, "「３％上昇」再現率：%1.3lf" %pre_score, "精度：%1.3lf" %grid.score(X_test_try, y_test2_try), ")") 

plt.figure(figsize=(15, 2))
plt.plot(total_return)

'''出力結果
投資結果：10000 円 → 14808.749 円 (買い回数：17.000 勝ち：12.000 勝率：0.706 「３％上昇」再現率：0.529 精度：0.540 )
投資結果：10000 円 → 31046.279 円 (買い回数：19.000 勝ち：17.000 勝率：0.895 「３％上昇」再現率：0.737 精度：0.580 )
投資結果：10000 円 → 30692.962 円 (買い回数：20.000 勝ち：17.000 勝率：0.850 「３％上昇」再現率：0.800 精度：0.780 )
投資結果：10000 円 → 18082.973 円 (買い回数：15.000 勝ち：14.000 勝率：0.933 「３％上昇」再現率：0.733 精度：0.600 )
投資結果：10000 円 → 15254.049 円 (買い回数：16.000 勝ち：14.000 勝率：0.875 「３％上昇」再現率：0.688 精度：0.580 )
'''


# In[17]:


# 学習データに無い企業でシュミレーション
filestest = glob.glob("data/kabu2/*.csv") ##モデル作成時とは別のファイル
base = 100
day_ago = 3
num_sihyou = 8
reset =True
for file in filestest:
    temp = pd.read_csv(file, header=0, encoding='cp932')
    temp = temp[['日付','始値', '高値','安値','終値','5日平均','25日平均','75日平均','出来高']]
    temp= temp.iloc[::-1]#上下反対
    temp2 = np.array(temp)
    temp3 = np.zeros((len(temp2)-base, num_sihyou))
    temp3[0:len(temp3), 0] = temp2[base:len(temp2), 4] / temp2[base-1:len(temp2)-1, 4]
    temp3[0:len(temp3), 1] = temp2[base:len(temp2), 1] / temp2[base:len(temp2), 4]
    temp3[0:len(temp3), 2] = temp2[base:len(temp2), 2] / temp2[base:len(temp2), 4]
    temp3[0:len(temp3), 3] = temp2[base:len(temp2), 3] / temp2[base:len(temp2), 4]
    temp3[0:len(temp3), 4] = temp2[base:len(temp2), 5].astype(np.float) / temp2[base:len(temp2), 4].astype(np.float)
    temp3[0:len(temp3), 5] = temp2[base:len(temp2), 6].astype(np.float) / temp2[base:len(temp2), 4].astype(np.float)
    temp3[0:len(temp3), 6] = temp2[base:len(temp2), 7].astype(np.float) / temp2[base:len(temp2), 4].astype(np.float)
    temp3[0:len(temp3), 7] = temp2[base:len(temp2), 8].astype(np.float) / temp2[base-1:len(temp2)-1, 8].astype(np.float)
    
        
    # 説明変数となる行列Xtを作成
    tempX = np.zeros((len(temp3), day_ago*num_sihyou))
    for s in range(0, num_sihyou): # 日にちごとに横向きに並べる
        for i in range(0, day_ago):
            tempX[i:len(temp3), day_ago*s+i] = temp3[0:len(temp3)-i,s]
            
    if reset:
        Xt = tempX[day_ago:]
        reset = False
    else:
        Xt= np.concatenate((Xt, tempX[day_ago:]), axis=0)
        
# 被説明変数となる Y = pre_day後の終値/当日終値 を作成
yt = np.zeros(len(Xt))
# 何日後を値段の差を予測するのか
pre_day = 1
yt[0:len(yt)-pre_day] = Xt[pre_day:len(Xt),0]
Xt = Xt[:-pre_day]
yt = yt[:-pre_day]
print(Xt.shape)

up_rate =1.03
yt2 = np.zeros(len(yt))
for i in range(0, len(yt2)):
    if yt[i] <= up_rate:
        yt2[i] = 0
    else:
        yt2[i] = 1

'''出力結果
(4188, 24)
'''


# In[18]:


# 予測結果の合計を計算（空売り無し。買いだけ）
# 上がると予測したら終値で買い,翌日の終値で売ったと想定：掛け金☓翌日の上昇値
# ランダムに日付を選び、tray_day日間運用
try_day = 100
print("全体精度: {:.2f}".format(grid.score(Xt, yt2)))
yt_pred = grid.predict(Xt)
for j in range(0, 5):
    a=random.randrange(len(yt2)-try_day)
    Xt_try = Xt[a:a+try_day]
    yt2_try = yt2[a:a+try_day]
    yt_try = yt[a:a+try_day]
    yt_pred_try = yt_pred[a:a+try_day]
    
    c_ = 0
    win_c = 0
    money = 10000

    # 予測結果の総和グラフを描くーーーーーーーーー
    total_return = np.zeros(len(yt_try))
    for i in range(0, try_day):
        if yt_pred_try[i] == 1:
            money = money*yt_try[i]
            c_ +=1    
            if yt_try[i] >= 1:
                win_c +=1
                   
            
        total_return[i] = money

    # 混同行列で確認
    conf = confusion_matrix(yt2_try, grid.predict(Xt_try))
    # 上昇予測回数が０のときは、勝率、再現率を９９９にする
    if c_==0:
        win_score=999
    else:
        win_score = win_c / c_
        
    if conf[0,1]==0 and conf[1,1]==0:
        pre_score=999
    else:
        pre_score = conf[1,1]/(conf[0,1]+conf[1,1])
    print("投資結果：10000 円 → %1.3lf" %money, "円", "(買い回数：%1.3lf" %c_, "勝ち：%1.3lf" %win_c, "勝率：%1.3lf" %win_score, "「３％上昇」再現率：%1.3lf" %pre_score, "精度：%1.3lf" %grid.score(Xt_try, yt2_try), ")") 

# 最後のシュミレーションだけ可視化
print(conf)
plt.figure(figsize=(15, 2))
plt.plot(total_return)

'''出力結果
全体精度: 0.61
投資結果：10000 円 → 9488.770 円 (買い回数：14.000 勝ち：6.000 勝率：0.429 「３％上昇」再現率：0.143 精度：0.830 )
投資結果：10000 円 → 10305.978 円 (買い回数：22.000 勝ち：11.000 勝率：0.500 「３％上昇」再現率：0.091 精度：0.770 )
投資結果：10000 円 → 12049.243 円 (買い回数：43.000 勝ち：26.000 勝率：0.605 「３％上昇」再現率：0.186 精度：0.580 )
投資結果：10000 円 → 11230.923 円 (買い回数：23.000 勝ち：14.000 勝率：0.609 「３％上昇」再現率：0.217 精度：0.780 )
投資結果：10000 円 → 10230.065 円 (買い回数：3.000 勝ち：2.000 勝率：0.667 「３％上昇」再現率：0.000 精度：0.940 )
[[94  3]
 [ 3  0]]
'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




