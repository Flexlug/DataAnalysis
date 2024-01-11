import streamlit as st
from utils.dl import generate_predictions, get_prediction, train_model
from utils.loader_mosecom import download_pol2_df, preprocess_pol2_df
from utils.loader_mosdata import download_pol1_df, merge_pol1_pol2, preprocess_pol1_df, process_pdk
from utils.loader_taxi import download_taxi_df, preprocess_taxi_df
from utils.loader_contrib import count_contrib_df
from utils.moscow import get_moscow, get_ao_list
from stqdm import stqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.python.client import device_lib

@st.cache_data
def get_contrib():
    df = download_pol1_df()
    pol_df = preprocess_pol1_df(df, get_moscow())
    df2 = download_pol2_df()
    pol2_df = preprocess_pol2_df(df2)
    whole_df = merge_pol1_pol2(pol_df, pol2_df)
    df3 = download_taxi_df()
    taxi_df = preprocess_taxi_df(df3)
    return count_contrib_df(taxi_df, whole_df)

@st.cache_data
def get_scaler(col):
    col = np.array(col).reshape(-1, 1)
    scaler = MinMaxScaler()
    _ = scaler.fit_transform(col)
    return scaler

@st.cache_data
def train_models(contrib_df, _scaler):
    ao_models = {}
    for ao in stqdm(get_ao_list(), desc='Обучение моделей'):
        data = contrib_df[contrib_df.NAME_AO == ao][['ContributionValue']]
        model, history = train_model(np.array(data), _scaler)
        ao_models[ao] = (model, history)

    return ao_models
 
@st.cache_data
def get_predictions(contrib_df, _models):
    ao_preds = {}
    for ao in stqdm(get_ao_list(), desc='Инференс'):
        data = contrib_df[contrib_df.NAME_AO == ao]['ContributionValue']
        inp = np.array(data)
        preds = generate_predictions(np.array(inp)[-20:], 20, scaler, _models[ao][0])
        ao_preds[ao] = preds
    return ao_preds

def plot_results(X, y, ao, plot_column, title):
    min_date = X['Period'].max()
    pd1 = pd.Series(
        list(
            pd.date_range(
                start=min_date, 
                periods=20,
                freq='MS'
            )
        )
    )
    pd2 = pd.Series(y)
    y_df = pd.concat([pd1, pd2], axis=1)
    y_df.columns = ['Period', plot_column]

    # Визуализация предсказаний
    fig, ax = plt.subplots()
    ax.plot(X['Period'], X[plot_column], label='Train')
    ax.plot(y_df['Period'], y_df[plot_column], label='Prediction', color='Red')
    ax.set_title(f'{title} - {ao}')
    ax.legend()
    return fig

contrib_df = get_contrib().drop('geometry', axis=1)
col = contrib_df['ContributionValue']
scaler = get_scaler(col)
models = train_models(contrib_df, scaler)
ao_predictions = get_predictions(contrib_df, models)

aos = get_ao_list()
selected_ao = st.select_slider(label='Автономный округ', options=aos, key='taxi_slider')
index = aos.index(selected_ao)
data = contrib_df[contrib_df.NAME_AO == selected_ao][['Period', 'ContributionValue']]
fig = plot_results(data, ao_predictions[selected_ao], selected_ao, 'ContributionValue', 'LSTM')
st.pyplot(fig)