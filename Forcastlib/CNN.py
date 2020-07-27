import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from wandb.keras import WandbCallback
import math
import os
import wandb
from datetime import datetime
model = tf.keras.models
layers = tf.keras.layers
wandb.init(project="high-low-peak", group='session_1')



def train_process(data, peaks, train_ratio=0.8):
    peaks_len = math.ceil(len(data)/peaks)
    data = data[:peaks_len]
    scaler = MinMaxScaler(feature_range=(0,1))
    training_len = math.ceil(len(data)*train_ratio)
    print(data)
    data = scaler.fit_transform(data.reshape(-1,1))
    train_data = data[:training_len]

    train_x = []
    train_y = np.array([])

    for i in range(0, len(train_data)-peaks-1):
        p = []
        for j in range(0, peaks-1):
            p.append(train_data[i+j])
        train_x.append(p)
        train_y = np.append(train_y, train_data[i+peaks+1])
    

    train_len = math.ceil(len(train_x)/peaks)
    train_x = train_x[:train_len]
    train_x = np.array(train_x).reshape((-1,peaks))
    train_y = train_y[:train_len]

    test_data = data[training_len:]

    test_x = []
    test_y = np.array([])

    for i in range(0, len(test_data)-peaks-1):
        p = []
        for j in range(0, peaks):
            p.append(test_data[i+j])
        test_x.append(p)
        test_y = np.append(test_y, test_data[i+peaks+1])

    test_len = math.ceil(len(test_x)/peaks)
    test_x = test_x[:test_len]
    test_y = test_y[:test_len]
    test_x = np.array(test_x).reshape((-1,peaks))

    return train_x, train_y, test_x, test_y, scaler

def date_diff(dates):
    diff = np.array([])
    for i in range(0, len(dates)-1):
        z = dates[i+1] - dates[i]
        z = z.days
        diff = np.append(diff, z)
    return diff

def train_process_v2(data, train_ratio=0.8):
    dates = data['Date'].values
    data = data['Value'].values

    dates = pd.to_datetime(dates)
    diff = date_diff(dates)

    scaler_dates = MinMaxScaler(feature_range=(0,1))
    scaler_dates.fit(diff.reshape((-1, 1)))
    scaled_dates = scaler_dates.transform(diff.reshape((-1,1)))
    scaled_dates = np.reshape(scaled_dates, (-1))

    training_len = math.ceil(len(data)*train_ratio)
    print(data)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(data.reshape((-1, 1)))
    data = scaler.transform(data.reshape((-1, 1)))
    data = np.reshape(data, (-1))

    train_data = data[:training_len]
    train_dates = scaled_dates[:training_len]

    train_x = []
    train_y = np.array([])

    for i in range(0, len(train_data)-2):
        train_x.append([train_data[i],train_data[i+1],train_dates[i]])
        train_y = np.append(train_y, [train_data[i+2], train_dates[i+1]])
    
    train_x = np.array(train_x).reshape((-1,3))
    train_y = np.reshape(train_y, (-1,2))

    test_data = data[training_len:]
    test_dates = scaled_dates[training_len:]

    test_x = []
    test_y = np.array([])

    for i in range(0, len(test_data)-2):
        test_x.append([test_data[i],test_data[i+1], test_dates[i]])
        test_y = np.append(test_y, [test_data[i+2], test_dates[i+1]])
    
    test_x = np.array(test_x).reshape((-1,3))
    test_y = np.reshape(test_y, (-1, 2))

    return train_x, train_y, test_x, test_y, scaler, scaler_dates
      
def pred_process(data, peaks):
    scaler = MinMaxScaler(feature_range=(0,1))
    data = scaler.fit_transform(np.array(data).reshape((-1,1)))
    data = data.reshape((-1, peaks))
    return data, scaler

def pred_process_v2(data, scaler_dates, scaler):
    diff = datetime.strptime(data[1][1][2:], '%y-%m-%d') - datetime.strptime(data[0][1][2:], '%y-%m-%d')
    diff = diff.days
    print(data)
    scaled_dates = scaler_dates.transform(np.array(diff).reshape((-1,1))).reshape((-1))

    scaled_values = scaler.transform(np.array([data[0][0], data[1][0]]).reshape((-1,1))).reshape(-1)
    return [scaled_values[0], scaled_values[1], scaled_dates[0]]


class HighLowAI(object):
    def __init__(self, layer_n, layer_size,peaks, name="test"):
        self.layer_n = layer_n
        self.layer_size = layer_size
        self.name = name
        self.peaks = peaks

        self.model = model.Sequential()
        self.model.add(layers.Dense(peaks))
        for i in range(self.layer_n):
            self.model.add(layers.Dense(layer_size, activation = 'relu'))
        self.model.add(layers.Dense(1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def run(self,data, epoch=8):
        X_train, y_train, X_test, y_test, scaler = train_process(data, self.peaks)

        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch, callbacks=[WandbCallback()]) 

        self.model.save(os.path.join(wandb.run.dir, "model.h5"))

    
    def predict(self, data):
        data, scaler = pred_process(data, self.peaks)
        pred = self.model.predict(data)
        return scaler.inverse_transform(pred)


class HighLowAI_v2(object):
    def __init__(self, layer_n, layer_size, name="test"):
        self.layer_n = layer_n
        self.layer_size = layer_size
        self.name = name

        self.model = model.Sequential()
        self.model.add(layers.Dense(3))
        for i in range(self.layer_n):
            self.model.add(layers.Dense(layer_size, activation = 'relu'))
        self.model.add(layers.Dense(2))

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def run(self,data, epoch=8):
        X_train, y_train, X_test, y_test, scaler, scaler_dates = train_process_v2(data)
        self.scaler = scaler
        self.scaler_dates = scaler_dates
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch, callbacks=[WandbCallback()]) 

        self.model.save(os.path.join(wandb.run.dir, "model.h5"))

    
    def predict(self, data):
        data = pred_process_v2(data, self.scaler_dates, self.scaler)
        pred = self.model.predict([data])
        df = pd.DataFrame(pred, columns=['Value', 'Date'])
        scaled = self.scaler.inverse_transform(df['Value'].values.reshape((-1,1)))
        scaled_dates = self.scaler_dates.inverse_transform(df['Date'].values.reshape((-1,1)))
        return np.transpose(np.array([scaled, scaled_dates]))

