import os
from os import path
from tqdm import tqdm
import pandas as pd
import numpy as np
import geopandas as gpd
import requests
import zipfile
import folium
import itertools
from matplotlib import pyplot as plt
import seaborn as sns
from shapely import ops
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from geopy.geocoders import Nominatim
from shapely.geometry import Point
from datetime import datetime
from lxml import etree
import json
import streamlit as st
from utils.parsers_mosdata import download_dataset, get_version_history
from utils.moscow import get_moscow, get_ao_list


def get_api_key():
    return os.environ['API_KEY']

@st.cache_resource
def download_taxi_df():
    versions = get_version_history(621)[0]['releases']
    all_taxi_dfs = []
    for version in versions:
        vNum = version['versionNum']
        rNum = version['releaseNum']
        vDate = version['releaseDate']

        date = np.datetime64(datetime.strptime(vDate, "%Y-%m-%d %H:%M:%S"))

        raw_taxi_df = download_dataset(
            dataset_id=621, 
            release_number=rNum,
            version_number=vNum,
            api_key=get_api_key())
        raw_taxi_df['Period'] = [date for i in raw_taxi_df.index]
        all_taxi_dfs.append(raw_taxi_df)
    
    return pd.concat(all_taxi_dfs)

@st.cache_resource
def preprocess_taxi_df(raw_taxi_df):
    taxi_columns = ['AdmArea', 'CarCapacity', 'Period']
    taxi_df = raw_taxi_df.sort_values(by='Period').reset_index()[taxi_columns]
    taxi_df = taxi_df \
        .groupby(['Period', 'AdmArea'])['CarCapacity'] \
        .sum() \
        .reset_index()
    taxi_df['NAME_AO'] = taxi_df['AdmArea'].apply(lambda x: x.split()[0].capitalize())
    taxi_df = taxi_df.drop('AdmArea', axis=1)
    taxi_df = gpd.GeoDataFrame(pd.merge(taxi_df, get_moscow(), on='NAME_AO'))

    min_per = taxi_df['Period'].min()
    max_per = taxi_df['Period'].max() + pd.DateOffset(months = 1)

    min_per = datetime(year = min_per.year, month = min_per.month, day = 1)
    max_per = datetime(year = max_per.year, month = max_per.month, day = 1)

    date_range = pd.date_range(
        start = min_per,
        end = max_per,
        freq='MS'
    )

    rows = []

    for i in range(len(date_range) - 1):
        date1 = date_range[i]
        date2 = date_range[i + 1]
        df = taxi_df[(taxi_df.Period >= date1) & (taxi_df.Period <= date2)]
        if len(df) == 0:
            continue

        for ao in get_ao_list():
            ao_df = df[df.NAME_AO == ao.capitalize()]
            if len(ao_df) == 0:
                continue
            row = ao_df.iloc[0]
            row['Period'] = date1
            rows.append(row)

    new_taxi_df = gpd.GeoDataFrame(rows, geometry='geometry', crs=taxi_df.crs).reset_index().sort_values(by='Period')
    taxi_df = new_taxi_df.copy().reset_index().sort_values(by='Period')
    return taxi_df