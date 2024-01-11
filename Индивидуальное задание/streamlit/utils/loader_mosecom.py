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

from utils.parsers_mosecom import get_stations_url_list, get_df_for_station
from utils.moscow import get_moscow, get_moscow_exclude

@st.cache_resource
def download_pol2_df():
    aos = get_stations_url_list()
    names_aos = list(aos.keys())
    pol_df2 = []
    for ao in tqdm(names_aos, desc='Пасинг станций автономных округов'):
        for station_url in aos[ao]:
            df = get_df_for_station(station_url)
            
            df['NAME_AO'] = ao.split(' ')[0].capitalize()
            pol_df2.append(df)
    return pd.concat(pol_df2)


def preprocess_pol2_df(pol_df2):
    pol_df2 = pol_df2[~(pol_df2['MonthlyAverage'].isna() | pol_df2['MonthlyAverage'].isna())]
    pol_df2 = pol_df2[~pol_df2.NAME_AO.isin(get_moscow_exclude())]
    pol_df2 = pol_df2[pol_df2.Parameter == 'CO']
    pol2_to_pol_param_dict = {
        'CO': 'Оксид углерода',
    }
    pol_df2['Parameter'] = pol_df2['Parameter'].apply(lambda x: pol2_to_pol_param_dict[x])
    pol_df2 = pol_df2 \
        .groupby(['Period', 'Parameter', 'NAME_AO'])['MonthlyAverage'] \
        .mean() \
        .reset_index() \
        .sort_values(by='Period')
    pol_df2 = gpd.GeoDataFrame(pd.merge(pol_df2, get_moscow(), on='NAME_AO'), geometry='geometry')
    return pol_df2