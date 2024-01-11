import streamlit as st
from utils.ml import train_test_split_df, train_sarimax, get_prediction_for
from utils.loader_mosecom import download_pol2_df, preprocess_pol2_df
from utils.loader_mosdata import download_pol1_df, merge_pol1_pol2, preprocess_pol1_df, process_pdk
from utils.loader_taxi import download_taxi_df, preprocess_taxi_df
from utils.loader_contrib import count_contrib_df
from utils.moscow import get_moscow, get_ao_list
from stqdm import stqdm
import matplotlib.pyplot as plt

@st.cache_data
def get_contrib():
    df = download_pol1_df()
    pol_df = preprocess_pol1_df(df, get_moscow())
    df2 = download_pol2_df()
    pol2_df = preprocess_pol2_df(df2)
    whole_df = merge_pol1_pol2(pol_df, pol2_df)
    df3 = download_taxi_df()
    taxi_df = preprocess_taxi_df(df3)
    return (whole_df, taxi_df, count_contrib_df(taxi_df, whole_df))

@st.cache_data
def get_taxi_split():
    return train_test_split_df(get_contrib()[1])

@st.cache_data
def get_whole_split():
    return train_test_split_df(get_contrib()[0])

@st.cache_data
def train_models_taxi():
    whole_df, taxi_df, contrib_df = get_contrib()

    taxi_df_train, taxi_df_test = train_test_split_df(taxi_df)

    sarimax_taxi_models = []
    sarimax_taxi_predicitons = []

    for ao in stqdm(get_ao_list(), desc='Обучение моделей'):
        ao_df = taxi_df_train[taxi_df_train['NAME_AO'] == ao][['Period', 'CarCapacity']].copy()
        result = train_sarimax(ao_df)
        sarimax_taxi_models.append(result)
        
    sarimax_taxi_predicitons = [get_prediction_for(taxi_df_test, model) for model in sarimax_taxi_models]
    return sarimax_taxi_predicitons

@st.cache_data
def train_models_whole():
    whole_df, taxi_df, contrib_df = get_contrib()

    whole_df_train, whole_df_test = train_test_split_df(whole_df)

    sarimax_pol_models = []
    sarimax_pol_predicitons = []

    for ao in stqdm(get_ao_list(), desc='Обучение моделей'):
        ao_df = whole_df_train[whole_df_train['NAME_AO'] == ao][['Period', 'MonthlyAverage']].copy()
        result = train_sarimax(ao_df)
        sarimax_pol_models.append(result)
        
    sarimax_pol_predicitons = [get_prediction_for(whole_df_test, model) for model in sarimax_pol_models]
    return sarimax_pol_predicitons

def plot_results(INDEX, pol_predictions_sarimax, df_train, df_test, plot_column, title):
    prediction_sarimax = pol_predictions_sarimax[INDEX]
    ao = get_ao_list()[INDEX]

    df_train = df_train[df_train['NAME_AO'] == ao]
    df_test = df_test[df_test['NAME_AO'] == ao]

    # Визуализация предсказаний
    fig, ax = plt.subplots()
    ax.plot(df_train['Period'], df_train[plot_column], label='Train')
    ax.plot(df_test['Period'], df_test[plot_column], label='Test')
    ax.plot(prediction_sarimax, label='Predicted SARIMAX', color='Red')
    ax.set_title(f'{title} - {ao}')
    ax.legend()
    return fig

#contrib_df_train, contrib_df_test = train_test_split_df(contrib_df)

col1, col2 = st.columns([1, 1])

whole_df, taxi_df, contrib_df = get_contrib()

with col1:
    taxi_predictions = train_models_taxi()
    taxi_df_train, taxi_df_test = get_taxi_split()
    aos = get_ao_list()
    selected_ao = st.select_slider(label='Автономный округ', options=aos, key='taxi_slider')
    index = aos.index(selected_ao)
    fig = plot_results(
        index,
        taxi_predictions,
        taxi_df_train,
        taxi_df_test,
        'CarCapacity',
        'Парковки такси'
    )
    st.pyplot(fig)

with col2:
    whole_predictions = train_models_whole()
    whole_df_train, whole_df_test = get_whole_split()
    aos = get_ao_list()
    selected_ao = st.select_slider(label='Автономный округ', options=aos, key='whole_slider')
    index = aos.index(selected_ao)
    fig = plot_results(
        index,
        whole_predictions,
        whole_df_train,
        whole_df_test,
        'MonthlyAverage',
        'Загрязнение воздуха'
    )
    st.pyplot(fig)