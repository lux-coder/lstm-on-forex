import timeit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.utils import plot_model
from sklearn import preprocessing
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

class Stock:
    def __init__(self,balance=10000,price = 1.0,hold = 0):        
        self.balance = balance
        self.price = price
        self.hold = hold
        self.total = balance + price * hold

    def update(self,price=0.0):
        self.price = price
        self.total = self.balance + self.hold*price

    def buy(self):
        price = self.price
        inc_hold = np.floor(self.balance/price)
        self.hold +=  inc_hold
        self.balance -= inc_hold*price

    def sell(self):
        price = self.price
        hold = self.hold
        self.balance += hold*price
        self.hold = 0

    def __str__(self):
        return 'Trading:\ncode = %s\nbalance = %d\nprice = %f\nhold = %d\ntotal = %d'%(self.balance,self.price,self.hold,self.total)
    
start_time = timeit.default_timer()

def train_test_split(data,SEQ_LENGTH = 25,test_prop=0.3):    

    data = data.sort_index()
    ntrain = int(len(data) *(1-test_prop))
    predictors = data.columns[1:]
    print(predictors)    
    data_pred = data[predictors] #/norms
    num_attr = data_pred.shape[1]
    
    result = np.empty((len(data) - SEQ_LENGTH - 1, SEQ_LENGTH, num_attr))
    y = np.empty(len(data) - SEQ_LENGTH - 1)
    yopen = np.empty(len(data) - SEQ_LENGTH - 1)

    for index in range(len(data) - SEQ_LENGTH - 1):
        result[index, :, :] = data_pred[index: index + SEQ_LENGTH]
        y[index] = data.iloc[index + SEQ_LENGTH + 1].close
        yopen[index] = data.iloc[index + SEQ_LENGTH + 1].Open
    
    xtrain = result[:ntrain, :, :]
    ytrain = y[:ntrain]
    xtest = result[ntrain:, :, :]
    ytest = y[ntrain:]
    ytest_open = yopen[ntrain:]
   
    return xtrain, xtest, ytrain, ytest, ytest_open

def train_model(xtrain,ytrain,SEQ_LENGTH=25,N_HIDDEN=256):    
    num_attr = xtrain.shape[2]
    model = Sequential()
    model.add(LSTM(N_HIDDEN, return_sequences=True, stateful=True, activation='tanh', batch_input_shape=(5, SEQ_LENGTH, num_attr)))
    #model.add(BatchNormalization())    
    #model.add(LSTM(N_HIDDEN, return_sequences=True, stateful=True, activation='hard_sigmoid'))
    model.add(LSTM(N_HIDDEN, return_sequences=False, stateful=True, activation='hard_sigmoid'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))    
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])  ## optimizer = 'rmsprop'
    model.fit(xtrain, ytrain, batch_size=5, epochs=1, validation_split=0.1, verbose=1)
    model.summary()
    plot_model(model, to_file='model2.png', show_layer_names=True, show_shapes=True)
    return model

def predict(model,xtest):
    predicted = model.predict(xtest, batch_size=5)
    return predicted

def policy(xtest,ytest,ytest_open,model):
    ypred = model.predict(xtest)
    
    xnow = xtest[0]
    price = xnow[-1,2]
    stock = Stock(price=price)
    pred_price = ypred[0,0]
    totals = [stock.total]

    for i in range(1,len(xtest)):
        price_open = ytest_open[i]
        price_close = ytest[i]
        stock.update(price=price_open)
        pred_price_now = ypred[i,0]
        if pred_price_now < pred_price:
            stock.buy()
        else:
            stock.sell()
        pred_price = pred_price_now
        stock.update(price=price_close)
        totals.append(stock.total)

    plt.figure(figsize=(18,12))
    plt.plot(totals)
    plt.title('Wealth curve')
    plt.show()
    return totals


data = pd.read_csv('EURUSD_indicators4.csv')
data = data.set_index('time')

#scaler = preprocessing.StandardScaler()
#xdata = scaler.fit_transform(data)

df = pd.DataFrame(data)
print('Data shape:', data.shape)
#print('XData shape:', xdata.shape)

print(data.head())

xtrain, xtest, ytrain, ytest, ytest_open = train_test_split(data)

print('xtrain.shape',xtrain.shape)
print('xtest.shape', xtest.shape)
print('ytrain.shape', ytrain.shape)
print('ytest.shape', ytest.shape)

model = train_model(xtrain,ytrain)

predicted_tr = model.predict(xtrain)

plot_predicted_tr = pd.DataFrame(predicted_tr)
print(plot_predicted_tr.head())
plt.figure(figsize=(18,12))
plt.plot(ytrain, label='true values')
plt.plot(predicted_tr, label='predicted  values')
plt.legend()
plt.title('train data')
plt.show()

predicted_test = model.predict(xtest)
plt.figure(figsize=(18,12))
plt.plot(ytest, label='true values')
plt.plot(predicted_test, label='predicted  values')
plt.legend()
plt.title('test data')
plt.show()

elapsed = np.round(timeit.default_timer() - start_time, decimals = 2)
print('Completed in: ', elapsed)

wealth = policy(xtest, ytest, ytest_open, model)