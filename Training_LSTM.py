import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.metrics import MeanAbsoluteError
from sklearn.metrics import r2_score

df=pd.read_csv("Nifty50.csv")
df=df.iloc[::-1, ::-1]
df.reset_index(inplace=True)

Nifty50_X = pd.DataFrame({'Date': pd.to_datetime(df['Date'])})
Nifty50_Y = df[['Nifty 50']]

timesplit= TimeSeriesSplit(n_splits=20)

for train_index, test_index in timesplit.split(Nifty50_Y):
        train, test = Nifty50_Y[:len(train_index)], Nifty50_Y[len(train_index): (len(train_index)+len(test_index))]


scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.fit_transform(test)

x_train = []
y_train = [] 

for i in range(100, train_scaled.shape[0]):
    x_train.append(train_scaled[i-100: i])
    y_train.append(train_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train) 

# model
model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences=True
              ,input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))

model.summary()

# compile and fit
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=[MeanAbsoluteError()])
model.fit(x_train, y_train,epochs = 50)

model.save('keras_model.h5')

x_test = []
y_test = []
for i in range(100, test_scaled.shape[0]):
   x_test.append(test_scaled[i-100: i])
   y_test.append(test_scaled[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
y_pred = model.predict(x_test)

y_test_ori = scaler.inverse_transform(pd.DataFrame(y_test))
y_pred_ori = scaler.inverse_transform(pd.DataFrame(y_pred))

R2_score_nift50 = r2_score(y_test_ori, y_pred_ori)
print(R2_score_nift50)

