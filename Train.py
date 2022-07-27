#%% import libraries
import os
import pickle
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt


from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, Input
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.metrics import mean_absolute_error, mean_squared_error
from DeepLearnModule import ModelHist_plot,Time_eval,Time_eval_inverse,subplot
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization, Bidirectional
#%% Constant
CSV_PATH_TRAIN = os.path.join(os.getcwd(),'Data','cases_malaysia_train.csv')
CSV_PATH_TEST = os.path.join(os.getcwd(),'Data','cases_malaysia_test.csv')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'Models', 'Model.h5')

df_train = pd.read_csv(CSV_PATH_TRAIN)
df_test = pd.read_csv(CSV_PATH_TEST)

#%% Data inspection
#check data on both test and train
df_train.info() # cases_new is in object
df_test.info() # 99 new cases instead of 100

df_train.head()
df_train.isna().sum()
msno.bar(df_train)
df_train.describe().T
# a lot of NaNs on last 7 columns

print(df_test[df_test.isna().any(axis=1)])
df_test['cases_new'].iloc[58:63]
df_test.isna()
df_test.isna().sum()
msno.bar(df_test)
df_test.describe().T
# have 1 NaN on test data
#%%# data cleaning, wrangling
# change cases_new column on train data from 'object' to 'numeric'
df_train['cases_new'] = pd.to_numeric(df_train['cases_new'], errors='coerce')
df_train.isna().sum()

# impute or handle missing data in test and train data using interpolation
df_train['cases_new'] = df_train['cases_new'].interpolate(method='linear')
df_test['cases_new'] = df_test['cases_new'].interpolate(method='linear')
df_test.head()
#%% Data visualization
# Train data
col = 'cases_new'
title1 = 'Covid-19 Daily New Cases (Dec 2021 - Mar 2022)'
title2 = 'Covid-19 Daily New Cases (Jan 2020 - Dec 2021)'
ylab = 'New cases'
xlab1 = 'Days since 25 Jan 2020'
xlab2 = 'Days since 5 Dec 2021'
subplot(df_train,df_test,col,title1,title2,ylab,xlab1,xlab2)

#%% Features selection
X = df_train['cases_new'] #1 feature
mms = MinMaxScaler()
X = mms.fit_transform(np.expand_dims(X,axis=-1))

#pickle save after fit_transform
MMS_PATH_X = os.path.join(os.getcwd(), 'Models', 'mms.pkl')
with open(MMS_PATH_X, 'wb') as file:
    pickle.dump(mms,file)

#%%
win_size = 30
X_train = []
y_train = []

for i in range(win_size,len(X)):
    X_train.append(X[i-win_size:i])
    y_train.append(X[i])

X_train = np.array(X_train)
y_train = np.array(y_train)
#%%

dataset_cat = pd.concat((df_train['cases_new'],df_test['cases_new']))
# Method
length_days = win_size +len(df_test)
tot_input = dataset_cat[-length_days:]

# Features selection
Xtest = df_test['cases_new'] #1 feature
Xtest = mms.transform(np.expand_dims(tot_input,axis=-1))

X_test = []
y_test = []

for i in range(win_size,len(Xtest)):
    X_test.append(Xtest[i-win_size:i])
    y_test.append(Xtest[i])

X_test = np.array(X_test)
y_test = np.array(y_test)
#%%
input_shape = np.shape(X_train)[1:]
model = Sequential()
model.add(Input(shape=(input_shape)))
model.add(LSTM(128,return_sequences=(True))) # true because passing to other lstm layer, need to retain 3d shape
model.add(Dropout(0.3))
model.add(LSTM(128,return_sequences=(True))) # true because passing to other lstm layer, need to retain 3d shape
model.add(Dropout(0.3))
model.add(LSTM(128,return_sequences=(True))) # true because passing to other lstm layer, need to retain 3d shape
model.add(Dropout(0.2))
model.add(LSTM(128,return_sequences=(True))) # true because passing to other lstm layer, need to retain 3d shape
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(1,activation='linear'))
model.summary()

plot_model(model,show_shapes=True,show_layer_names=True)

#%%
model.compile(optimizer='adam',loss='mse',metrics=['mean_absolute_percentage_error','mse'])
LOGS_PATH = os.path.join(os.getcwd(),'Logs',datetime.datetime.now().
                    strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)
# early_callback = EarlyStopping(monitor='val_loss',patience=3)
#%%
hist = model.fit(X_train,y_train,epochs=50,
                    callbacks=[tensorboard_callback,],
                    validation_data=(X_test,y_test))

#%%
print(hist.history.keys())

ModelHist_plot(hist,'mse','val_mse','training MSE','val_mse')
ModelHist_plot(hist,'mean_absolute_percentage_error',
                'val_mean_absolute_percentage_error',
                'mean_absolute_percentage_error',
                'val_mean_absolute_percentage_error')
#%%
predicted_cases_new = model.predict(X_test)

Time_eval(y_test,predicted_cases_new,ylab='Cases new')

Time_eval_inverse(y_test=y_test,mms=mms,predicted =predicted_cases_new,
                                ylab='Cases new')


pred_cases = mms.inverse_transform(predicted_cases_new)
actual_cases = mms.inverse_transform(y_test)

mape = mean_absolute_percentage_error(actual_cases, pred_cases)

print(f'MAPE is {np.round(mape*100,2)}%')

model.save(MODEL_SAVE_PATH)
# %%
