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

from utils.moscow import get_ao_list

import streamlit as st

def count_contrib_df(taxi_df, whole_df):
    rows = []

    min_per = taxi_df['Period'].min()
    max_per = taxi_df['Period'].max() + pd.DateOffset(months = 1)
    min_per = datetime(year = min_per.year, month = min_per.month, day = 1)
    max_per = datetime(year = max_per.year, month = max_per.month, day = 1)
    date_range = pd.date_range(
        start = min_per,
        end = max_per,
        freq='MS'
    )

    for i in range(len(date_range) - 1):
        date1 = date_range[i]
        date2 = date_range[i + 1]

        date_whole_df = whole_df[(whole_df.Period >= date1) & (whole_df.Period <= date2)]
        date_taxi_df = taxi_df[(taxi_df.Period >= date1) & (taxi_df.Period <= date2)]

        prev_pol_vals = { key: whole_df[whole_df.NAME_AO == key].iloc[0]['MonthlyAverage'] for key in get_ao_list() }
        prev_taxi_vals = { key: taxi_df[taxi_df.NAME_AO == key].iloc[0]['CarCapacity'] for key in get_ao_list() }
        prev_geo_vals = { key: taxi_df[taxi_df.NAME_AO == key].iloc[0]['geometry'] for key in get_ao_list() }

        for ao in get_ao_list():
            ao_whole_df = date_whole_df[date_whole_df.NAME_AO == ao.capitalize()]
            ao_taxi_df = date_taxi_df[date_taxi_df.NAME_AO == ao.capitalize()]

            if not (len(ao_whole_df) == 0 or len(ao_taxi_df) == 0):
                whole_value = ao_whole_df.iloc[0]['MonthlyAverage']
                taxi_value = ao_taxi_df.iloc[0]['CarCapacity']

                prev_pol_vals[ao] = whole_value
                prev_taxi_vals[ao] = taxi_value
            else:
                whole_value = prev_pol_vals[ao]
                taxi_value = prev_taxi_vals[ao]
            contrib_value = whole_value / taxi_value
            new_contrib_row = pd.Series()
            new_contrib_row['ContributionValue'] = contrib_value
            new_contrib_row['NAME_AO'] = ao
            new_contrib_row['Period'] = date1
            new_contrib_row['geometry'] = prev_geo_vals[ao]
            rows.append(new_contrib_row)

    contrib_df = gpd.GeoDataFrame(rows, geometry='geometry', crs=taxi_df.crs)
    return contrib_df