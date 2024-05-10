import pandas as pd
import csv
import os
from datetime import datetime, timedelta
import yfinance as yf

start = ''
end = ''

def create_csv(filename, data, stock):
    if data is None:
        print('Cannot write data as the var is empty')
        return
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Date', stock])
        for index, row in data.iterrows():
            writer.writerow([index, row[None]])


def dataFetch(stock):
    temp = yf.Ticker(stock)
    hist = temp.history(start=start, end=end)
    hist = hist.loc[:, hist.columns.intersection(['Close'])]
    hist = hist.iloc[::-1]
    hist.columns = [None]
    hist.index = [datetime.strftime(idx, "%Y-%m-%d") for idx in hist.index]
    return hist


def checker(stock, stockName):
    global start
    global end
    if not os.path.exists('Nifty50.csv'):
        begin = 2007
        start='2007-04-01'
        end='2008-04-01'
        today = datetime.now().year
        df = pd.DataFrame()
        for i in range(today - begin):
            temp = dataFetch(stock)
            df = pd.concat([temp, df], axis=0)
            start = end
            end_date = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=365)
            end = end_date.strftime("%Y-%m-%d")
        return df
    else:
        df = pd.read_csv('Nifty50.csv')
        start = df['Date'][0]
        start = datetime.strptime(start, "%Y-%m-%d") + timedelta(days=1)
        end = datetime.strptime(datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d") + timedelta(days=1)
        if end >= start:
            value = dataFetch(stock)
            if not value.empty:
                if value[None].iloc[-1] == df[stockName].iloc[0]:
                    print('its the same date')
                else:
                    df.set_index('Date', inplace=True)
                    df.columns = [None]
                    value = pd.concat([value, df], axis=0)
                    return value
            else:
                print('data not available')
                return None
        else:
            print('Invalid asking')
            return None


# data for nifty 50
a = checker("%5ENSEI", 'Nifty 50')
create_csv('Nifty50.csv',a, 'Nifty 50')
