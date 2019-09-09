import timeit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments
import configparser
from matplotlib import pyplot as plt
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib import gridspec
from matplotlib.dates import DateFormatter
from stockstats import StockDataFrame

path = '../data/'

config = configparser.ConfigParser()
config.read('../config/config.ini')
accountID = config['oanda']['account_id']
access_token = config['oanda']['api_key']
api = API(access_token=access_token, environment="practice")

dates = pd.date_range(start='2016-01-01', end='2017-01-01', freq='B')

h5 = pd.HDFStore(path+'backtest_data.h5', 'w')

instrument = 'EUR_USD'

df = pd.DataFrame()
print('Retriving data for %s' %instrument)

for i in range (0, len(dates)-1):
	d1 = str(dates[i]).replace(' ', 'T')
	d2 = str(dates[i+1]).replace(' ', 'T')

	try:
		data = oanda.get_history(instrument = instrument, start = d1, end = d2, aligmentTimezone = 'Europe/Berlin', granularity = 'M1')
		df = df.append(pd.DataFrame(data['candels']))
	except:
		pass

	if i%25 == 0:
		print ('%02d')

index = pd.DatetimeIndex(df['time'], tz = 'UTC').tz_convert('Europe/Berlin')

df.index = index
df['Close'].plot(figsize = (18,12))

formatter = DateFormatter('%H:%M')

data = pd.read_csv('data/EURUSD.csv')
data = data.set_index('Time')

stock = StockDataFrame.retype(pd.read_csv('data/EURUSD.csv'), index_column='time')

stock['sma20'] = stock.get('open_20_sma')
stock['sma50'] = stock.get('close_50_sma')
stock['sma200'] = stock.get('close_200_sma')
stock['macd'] = stock.get('macds')
stock['rsi20'] = stock.get('rsi_20')
stock['boll'] = stock.get('boll')
stock['boll_ub'] = stock.get('boll_ub')
stock['boll_lb'] = stock.get('boll_lb')
stock['cci'] = stock.get('cci_20')
stock['so'] = stock.get('kdjk')

fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, gridspec_kw={'height_ratios':[2,1,1]})
df1 = stock[['close', 'sma50', 'sma200']].iloc[4500:5000]
df2 = stock[['macd']].iloc[4500:5000]
df3 = stock[['so']].iloc[4500:5000]
df1.plot(ax=axes[0], figsize=(18,12))
df2.plot(ax=axes[1])
df3.plot(ax=axes[2])

stock[['close', 'sma50', 'sma200', 'macd', 'so']].to_csv('data/EURUSD_indicators1.csv')

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios':[2,1]})
df1 = stock[['close', 'boll_ub', 'boll_lb']].iloc[4500:5000]
df2 = stock[['rsi20']].iloc[4500:5000]
df1.plot(ax=axes[0], figsize=(18,12))
df2.plot(ax=axes[1])

stock[['close', 'boll_ub', 'boll_lb', 'rsi20']].to_csv('data/EURUSD_indicators2.csv')

stock[['close']].to_csv('data/EURUSD_clean.csv')

path = '../data/'

config = configparser.ConfigParser()
config.read('../config/config.ini')
accountID = config['oanda']['account_id']
access_token = config['oanda']['api_key']
api = API(access_token=access_token, environment="practice")

dates = pd.date_range(start='2016-01-01', end='2017-01-01', freq='B')

h5 = pd.HDFStore(path+'backtest_data.h5', 'w')

instrument = 'EUR_USD'

df = pd.DataFrame()
print('Retriving data for %s' %instrument)

for i in range (0, len(dates)-1):
	d1 = str(dates[i]).replace(' ', 'T')
	d2 = str(dates[i+1]).replace(' ', 'T')

	try:
		data = oanda.get_history(instrument = instrument, start = d1, end = d2, aligmentTimezone = 'Europe/Berlin', granularity = 'M1')
		df = df.append(pd.DataFrame(data['candels']))
	except:
		pass

	if i%25 == 0:
		print ('%02d')

index = pd.DatetimeIndex(df['time'], tz = 'UTC').tz_convert('Europe/Berlin')

df.index = index
df['Close'].plot(figsize = (18,12))

formatter = DateFormatter('%H:%M')

data = pd.read_csv('data/EURUSD.csv')
data = data.set_index('Time')

stock = StockDataFrame.retype(pd.read_csv('data/EURUSD.csv'), index_column='time')

stock['sma20'] = stock.get('open_20_sma')
stock['sma50'] = stock.get('close_50_sma')
stock['sma200'] = stock.get('close_200_sma')
stock['macd'] = stock.get('macds')
stock['rsi20'] = stock.get('rsi_20')
stock['boll'] = stock.get('boll')
stock['boll_ub'] = stock.get('boll_ub')
stock['boll_lb'] = stock.get('boll_lb')
stock['cci'] = stock.get('cci_20')
stock['so'] = stock.get('kdjk')

fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, gridspec_kw={'height_ratios':[2,1,1]})
df1 = stock[['close', 'sma50', 'sma200']].iloc[4500:5000]
df2 = stock[['macd']].iloc[4500:5000]
df3 = stock[['so']].iloc[4500:5000]
df1.plot(ax=axes[0], figsize=(18,12))
df2.plot(ax=axes[1])
df3.plot(ax=axes[2])

stock[['close', 'sma50', 'sma200', 'macd', 'so']].to_csv('data/EURUSD_indicators1.csv')

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios':[2,1]})
df1 = stock[['close', 'boll_ub', 'boll_lb']].iloc[4500:5000]
df2 = stock[['rsi20']].iloc[4500:5000]
df1.plot(ax=axes[0], figsize=(18,12))
df2.plot(ax=axes[1])

stock[['close', 'boll_ub', 'boll_lb', 'rsi20']].to_csv('data/EURUSD_indicators2.csv')

stock[['close']].to_csv('data/EURUSD_clean.csv')