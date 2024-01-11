import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.python.client import device_lib

def train_model(data, scaler):
    # Нормализуем данные с помощью MinMaxScaler-а
    scaled_data = scaler.transform(data)

    # Создадим наборы данных для обучения модели
    X, y = [], []
    for i in range(len(scaled_data) - 20):
        X.append(scaled_data[i:i+20])
        y.append(scaled_data[i+20])
    X, y = np.array(X), np.array(y)

    # Спроектируем модель
    # Количество выходных нейронов равно количеству значений,
    # которое модель вернет по инференсу
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(X, y, epochs=50, verbose=0)
    return model, history

def get_prediction(data, scaler, model):
    '''
    Ожидает "сырые" данные, т.е. колонку или dataframe с одной колонкой
    '''
    x = data.reshape(-1, 1)
    scaled_x = scaler.transform(x)
    pred = model.predict(scaled_x[np.newaxis], verbose=0)
    unscaled_pred = scaler.inverse_transform(pred)[0]
    return unscaled_pred

def generate_predictions(data: np.array, count, scaler, model):
    WINDOW_WIDTH = 20

    window = data[:WINDOW_WIDTH]
    all_predictions = []
    for i in range(count):
        pred = get_prediction(window, scaler, model)
        all_predictions.append(pred[0])
        window = np.append(window[1:], pred[0])
    all_predictions = np.array(all_predictions)
    return all_predictions