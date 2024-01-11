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

from utils.parsers_mosdata import download_dataset

class CachedGeolocator:
    cache = {}
    geolocator = Nominatim(user_agent='pollution_streamlit')

    def geocode_location(self, raw_loc):
        loc_arr = raw_loc.split(',')
        if 'строение' in loc_arr[-1] or 'корпус' in loc_arr[-1]:
            loc_arr = loc_arr[:-1]
        loc = ''.join(loc_arr)
        loc = loc.replace('дом', '')
        loc = loc.replace('город Москва', '')
        if loc in self.cache:
            return self.cache[loc]
        else:
            geo_loc = self.geolocator.geocode(loc)
            self.cache[loc] = geo_loc
        return self.cache[loc]

def get_api_key():
    return os.environ['API_KEY']

@st.cache_resource
def download_pol1_df():
    return download_dataset(2453, get_api_key())

def get_allowed_parameters():
    return ['Оксид углерода']

@st.cache_data
def process_pdk(_pol_dff):
    parameters_pdks_raw = {}
    raw_parameters_set = set(_pol_dff['Parameter'])

    # Соберем показания ПДК для всех вещест
    for param in raw_parameters_set:
        all_pdks = _pol_dff[_pol_dff.Parameter == param]['MonthlyAveragePDKss']
        min_pdk = _pol_dff[_pol_dff.Parameter == param]['MonthlyAveragePDKss'].min()
        max_pdk = _pol_dff[_pol_dff.Parameter == param]['MonthlyAveragePDKss'].max()
        parameters_pdks_raw[param] = (min_pdk, max_pdk)

    # Уберем записи, где ПДК не указан
    def filter_dict(fdict, by_key = lambda x: True, by_value = lambda x: True):
        dict_items = list(fdict.items())
        filtered_items = list(filter(lambda x: by_key(x[0]) and by_value(x[1]), dict_items))
        return dict((k, v) for k, v in filtered_items)
    _parameters_pdks = filter_dict(parameters_pdks_raw, by_value=lambda x: not np.isnan(x[0]) and not np.isnan(x[1]))

    # Унифицируем ПДК
    for key, val in _parameters_pdks.items():
        avg = (val[0] + val[1]) / 2
        if np.isnan(avg):
            avg = 0.1
        _parameters_pdks[key] = avg

    return _parameters_pdks

@st.cache_resource
def get_geolocator():
    return CachedGeolocator()

@st.cache_data
def preprocess_pol1_df(_raw_pol_df, _moscow):
    # Уберем лишние колонки
    pol_df_columns = ['StationName', 'Period', 'SurveillanceZoneCharacteristics', 'Location', 'Parameter', 'MonthlyAverage', 'MonthlyAveragePDKss']
    pol_df = _raw_pol_df[pol_df_columns]

    # Преобразуем колонку с датами к datetime64
    pol_df['Period'] = pd.to_datetime(pol_df['Period'], format='%m.%Y')

    # Оставим информацию только о требуемых веществах
    allowed_papameters = get_allowed_parameters()
    pol_df = pol_df[pol_df['Parameter'].isin(allowed_papameters)]

    # Геокодировка адресов
    geo = get_geolocator()
    pol_df['FullLocation'] = 'Москва, ' + pol_df['Location']
    pol_df['GeoLocation'] = pol_df['FullLocation'].apply(geo.geocode_location)
    print(f"Всего адресов: {len(pol_df['FullLocation'])}")
    pol_df = pol_df[~(pol_df['GeoLocation'].isnull() | pol_df['GeoLocation'].isna())]

    # Найдем для каждой станции соответствующий автономный округ
    pol_df['Latitude'] = pol_df['GeoLocation'].apply(lambda loc: loc.latitude if loc else None)
    pol_df['Longitude'] = pol_df['GeoLocation'].apply(lambda loc: loc.longitude if loc else None)

    geometry = [Point(lon, lat) for lon, lat in zip(pol_df['Longitude'], pol_df['Latitude'])]
    gdf_points = gpd.GeoDataFrame(pol_df, geometry=geometry, crs=_moscow.crs)
    pol_df = gpd.sjoin(gdf_points, _moscow)

    # Возьмём среднее от станций, находящихся в одном автономном округе
    pol_df = pol_df.groupby(['Period', 'Parameter', 'NAME_AO'])['MonthlyAverage'].mean().reset_index()
    pol_df = pol_df.sort_values(by='Period')
    pol_df = gpd.GeoDataFrame(pd.merge(pol_df, _moscow, on='NAME_AO'), geometry='geometry')
    pol_df.head(3)

    return pol_df

@st.cache_data
def merge_pol1_pol2(_pol_df, _pol_df2):
    min_avaliable = _pol_df2['Period'].min()
    whole_df = _pol_df[_pol_df.Period < min_avaliable]
    whole_df = pd.concat([whole_df, _pol_df2]).sort_values(by='Period').reset_index()

    return whole_df