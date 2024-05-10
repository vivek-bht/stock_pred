import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import datetime
from datetime import timedelta

def Get_values(selected_date):
    df = pd.read_csv('Nifty50.csv')
    lastDate = datetime.strptime(df['Date'][0], '%Y-%m-%d')
    counter = (selected_date - lastDate).days
    df.reset_index(inplace=True)
    test = df[['Nifty 50']][:100]
    test = test.iloc[::-1]

    scaler = MinMaxScaler(feature_range=(0,1)) 
    x_test_before = scaler.fit_transform(test)
    model = load_model('keras_model.h5')
    if selected_date <= lastDate:
        date_str = selected_date.strftime('%Y-%m-%d')
        filtered_df = df[df['Date'] <= date_str]
        return filtered_df['Nifty 50'].values[0] if not filtered_df.empty else None
    while lastDate <= selected_date:
        x_test = []
        x_test.append(x_test_before[:100])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
        predicted_price = model.predict(x_test)
        lastDate += timedelta(days=1)
        x_test_before = np.append(x_test_before, predicted_price[-1: , 0:1], axis=0)
        x_test_before = x_test_before[1:, :]
    predicted_prices_df = pd.DataFrame(x_test_before)
    y_pred_ori = scaler.inverse_transform(predicted_prices_df)
    return y_pred_ori[-counter:]

