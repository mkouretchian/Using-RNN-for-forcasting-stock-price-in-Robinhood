# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:44:54 2019

@author: Mahdi Kouretchian
"""

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from scipy import signal
import pickle
import pandas as pd
from pandas_datareader import data
plt.rcParams["figure.figsize"] = (22, 9)
basket = ['EAT', 'TWTR', 'TTNP', 'SBUX', 'BAC', 'SHLDQ', 'STM', 'ACB', 'GREK']
df = data.DataReader(basket, 'robinhood')
df.head()
orig = df.copy()
df=df.reset_index()
#####  Feature Engineering ######
X_scalers = {}
y_scalers = {}
for stock in basket:
    for col in ('close_price', 'high_price', 'low_price', 'open_price', 'volume'):
        df[col] = df[col].astype(float)
        df.loc[df['symbol'] == stock, col] = signal.detrend(df[df['symbol'] == stock][col])
    df.loc[df['symbol'] == stock, 'mean_close_price_2'] = df.loc[df['symbol'] == stock, 'close_price'].rolling(window=2).mean()
    df.loc[df['symbol'] == stock, 'mean_close_price_3'] = df.loc[df['symbol'] == stock, 'close_price'].rolling(window=3).mean()
    df.loc[df['symbol'] == stock, 'std_close_price_2'] = df.loc[df['symbol'] == stock, 'close_price'].rolling(window=2).std()
    df.loc[df['symbol'] == stock, 'std_close_price_3'] = df.loc[df['symbol'] == stock, 'close_price'].rolling(window=3).std()
    
X_scalers = {stock:{} for stock in basket}
y_scalers = {}

##### Ploting the close price for EAT symbol ######

plt.plot(df[df['symbol'] == 'EAT']['close_price'])



####  Creating date time feature ####

as_date = df['begins_at'].dt
df['dayofweek'] = as_date.dayofweek
df['quarter'] = as_date.quarter
df['weekofyear'] = as_date.weekofyear

df = df.drop(['begins_at', 'interpolated', 'session'], axis=1)
df = df.dropna(axis=0) # Due to window, first two rows now contain nans
df = df.reset_index(drop=True)

for stock in basket:
    df = df.drop(df.index[len(df[df['symbol'] == stock]) - 1], axis=0)
    outliers = abs(df[df['symbol'] == stock]['tomo_gain']) < df[df['symbol'] == stock]['tomo_gain'].std() * 3
    df[df['symbol'] == stock] = df[df['symbol'] == stock].loc[:, :][outliers]
    df = df.drop(df[df['symbol'] == stock].iloc[-1].name) # get rid of last because next is a different stock
    pre_y = df[df['symbol'] == stock]['tomo_gain'].values
    y_scalers[stock] = make_pipeline(StandardScaler(), MinMaxScaler(feature_range=(-1, 1)))
    for col in ('close_price', 'high_price', 'low_price', 'open_price', 'volume', 'mean_close_price_2', \
               'mean_close_price_3', 'std_close_price_2', 'std_close_price_3', 'yday_gain'):
        pre_x = df[df['symbol'] == stock][col]
        X_scalers[stock][col] = make_pipeline(StandardScaler(), MinMaxScaler(feature_range=(-1, 1)))
        df.loc[df['symbol'] == stock, col] = X_scalers[stock][col].fit_transform(pre_x.values.reshape(-1,1))
    df.loc[df['symbol'] == stock, 'tomo_gain'] = y_scalers[stock].fit_transform(pre_y.reshape(-1, 1)).reshape(-1)
    
    
##### Save the feature scalars  #######
    
pickle.dump(X_scalers, open('x_scalers.pkl', 'wb'))
pickle.dump(y_scalers, open('y_scalers.pkl', 'wb'))

num_df_cols = df.shape[1] - 1 + len(basket) - 1


###### Model training  #######
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(num_df_cols, input_shape=(1, num_df_cols)))
model.add(tf.keras.layers.LSTM(64))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

X = df.drop(['tomo_gain', 'symbol'], axis=1)
y = df['tomo_gain']

dummies = pd.get_dummies(df['symbol'], columns=['symbol'])
X = np.append(X, dummies.values, axis=1)
X.shape
X = np.reshape(X, (-1, 1, num_df_cols))


####### Sanity check ########
plt.subplot(1,2,1)
plt.plot(y)
plt.subplot(1,2,2)
plt.plot(df['tomo_gain'].values)


###### Fit the base model ########

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train.values.reshape(-1,1), batch_size=64, epochs=1000, verbose=0)
model.fit(X_train, y_train.values.reshape(-1,1), batch_size=32, epochs=6)
model.evaluate(X_test, y_test)

####### Sanity check ########
model.reset_states()
plt.plot(model.predict(X))


plt.plot(y)



plt.plot(y_scalers['EAT'].inverse_transform(model.predict(X[100:120])))



plt.plot(y[100:120])




def pad_stock(symbol):
    dumies = np.zeros(len(basket))
    dumies[list(dummies.columns.values).index(symbol)] = 1.
    return dumies


pad_stock('TWTR')

model.save('market_model.h5')


#### Pop off the model head and add a different one that we will finetune per stock ####

model.layers

model.pop()
model.pop()


##### Freeze the 2 older dense layers #####

model.layers[0].trainable = False
model.layers[3].trainable = False


model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1))

basket_dfs = {}
specific_models = {}
for stock in basket:
    basket_dfs[stock] = df[df['symbol'] == stock]
    specific_models[stock] = tf.keras.models.clone_model(model)
    specific_models[stock].set_weights(model.get_weights())
    

specific_models['EAT'].layers


for stock in basket:
    specific_models[stock].compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    

Xes = {}
ys = {}
for stock in basket:
    repeated_dummies = pad_stock(stock).reshape(1,-1).repeat(len(basket_dfs[stock]),axis=0)
    Xes[stock] = np.append(basket_dfs[stock].drop(['tomo_gain', 'symbol'], axis=1).values, repeated_dummies, axis=1)
    Xes[stock] = np.reshape(Xes[stock], (-1, 1, num_df_cols))
    ys[stock] = basket_dfs[stock]['tomo_gain'].values.reshape(-1,1)
    
    
Xes_train, ys_train, Xes_test, ys_test, best_model_scores, best_model = {}, {}, {}, {}, {}, {}



for stock in basket:
    best_model_scores[stock] = 1e6
    
    
    
for stock in basket:
    Xes_train[stock] = Xes[stock][:-5]
    ys_train[stock] = ys[stock][:-5]
    Xes_test[stock] = Xes[stock][-5:]
    ys_test[stock] = ys[stock][-5:]
    for i in range(8):
        specific_models[stock].fit(Xes_train[stock], ys_train[stock], batch_size=64, epochs=100, verbose=0)
        specific_models[stock].fit(Xes_train[stock], ys_train[stock], batch_size=16, epochs=30, verbose=0)
        specific_models[stock].fit(Xes_train[stock], ys_train[stock], batch_size=1, epochs=1, verbose=0)
        evaluation = specific_models[stock].evaluate(Xes_test[stock], ys_test[stock])[0]
        if evaluation < best_model_scores[stock]:
            best_model_scores[stock] = evaluation
            print('now saving {} because it was the best with eval score {}'.format(stock, evaluation))
            best_model[stock] = tf.keras.models.clone_model(specific_models[stock])
            best_model[stock].set_weights(specific_models[stock].get_weights())
            best_model[stock].compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        else:
            print('did not save {} because it did not improve with eval score {}'.format(stock, evaluation))
            
            
            
###### Make some prediction for tomorrow  #######
            

for stock in basket:
    today = df[df['symbol'] == stock].iloc[-1].drop(['tomo_gain', 'symbol'])
    today = np.append(today, pad_stock(stock))
    specific_models[stock].reset_states()
    pred = specific_models[stock].predict(np.reshape(today, (-1, 1, num_df_cols)))
    pred = y_scalers[stock].inverse_transform(pred)
    print("Stock {}, pred: {}".format(stock, np.asscalar(pred)))
    

###### Save the fine tuned models ########
    
for stock, model in specific_models.items():
    model.save('finetuned_{}.h5'.format(stock)
    
    

#######  With the models built, run this to generate new predictions for the most recent day ######
    
X_scalers = pickle.load(open('x_scalers.pkl', 'rb'))
y_scalers = pickle.load(open('y_scalers.pkl', 'rb'))


specific_models = {}
for stock in basket:
    specific_models[stock] = tf.keras.models.load_model('finetuned_{}.h5'.format(stock))
    
    
    
new_day = data.DataReader(basket, 'robinhood')

new_day.tail()


new_day = new_day.reset_index()

for stock in basket:
    for col in ('close_price', 'high_price', 'low_price', 'open_price', 'volume'):
        new_day[col] = new_day[col].astype(float)
        new_day.loc[new_day['symbol'] == stock, col] = signal.detrend(new_day[new_day['symbol'] == stock][col])
    new_day.loc[new_day['symbol'] == stock, 'mean_close_price_2'] = new_day.loc[new_day['symbol'] == stock, 'close_price'].rolling(window=2).mean()
    new_day.loc[new_day['symbol'] == stock, 'mean_close_price_3'] = new_day.loc[new_day['symbol'] == stock, 'close_price'].rolling(window=3).mean()
    new_day.loc[new_day['symbol'] == stock, 'std_close_price_2'] = new_day.loc[new_day['symbol'] == stock, 'close_price'].rolling(window=2).std()
    new_day.loc[new_day['symbol'] == stock, 'std_close_price_3'] = new_day.loc[new_day['symbol'] == stock, 'close_price'].rolling(window=3).std()
    
    
new_day['tomo_gain'] = new_day['close_price'].shift(-1) - new_day['close_price']
new_day['yday_gain'] = new_day['tomo_gain'].shift(1)


as_date = new_day['begins_at'].dt
new_day['dayofweek'] = as_date.dayofweek
new_day['quarter'] = as_date.quarter
new_day['weekofyear'] = as_date.weekofyear
new_day = new_day.drop(['begins_at', 'interpolated', 'session'], axis=1)
new_day = new_day.dropna(axis=0)
new_day = new_day.reset_index(drop=True)
for stock in basket:
    new_day = new_day.drop(new_day.index[len(new_day[new_day['symbol'] == stock]) - 1], axis=0)
    outliers = abs(new_day[new_day['symbol'] == stock]['tomo_gain']) < new_day[new_day['symbol'] == stock]['tomo_gain'].std() * 3
    new_day[new_day['symbol'] == stock] = new_day[new_day['symbol'] == stock].loc[:, :][outliers]
    new_day = new_day.drop(new_day[new_day['symbol'] == stock].iloc[-1].name)
    for col in ('close_price', 'high_price', 'low_price', 'open_price', 'volume', 'mean_close_price_2', \
               'mean_close_price_3', 'std_close_price_2', 'std_close_price_3', 'yday_gain'):
        pre_x = new_day[new_day['symbol'] == stock][col]
        new_day.loc[new_day['symbol'] == stock, col] = X_scalers[stock][col].transform(pre_x.values.reshape(-1,1))
        
        

new_day = new_day.dropna(axis=0)


dummies = pd.get_dummies(new_day['symbol'], columns=['symbol'])
num_df_cols = new_day.shape[1] - 1 + len(basket) - 1
def pad_stock(symbol):
    dumdums = np.zeros(len(basket))
    dumdums[list(dummies.columns.values).index(symbol)] = 1.
    return dumdums


###### Today: This should be better ########
    

for stock in basket:
    today = new_day[new_day['symbol'] == stock].iloc[-8:-1].drop(['tomo_gain', 'symbol'], axis=1)
    today = np.append(today, pad_stock(stock).reshape(-1,9).repeat(7,axis=0))
    specific_models[stock].reset_states()
    pred = specific_models[stock].predict(np.reshape(today, (-1, 1, num_df_cols)))
    pred = y_scalers[stock].inverse_transform(pred)
    print("Stock {}, pred: {}".format(stock, np.asscalar(pred[-1])))


