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

def get_moscow_columns(): 
    return ['NAME_AO', 'geometry']

def get_moscow_exclude():
    return ['Троицкий', 'Новомосковский', 'Зеленоградский']

@st.cache_data
def get_ao_list():
    return list(set(get_moscow()['NAME_AO']))

@st.cache_data
def get_moscow():
    moscow_columns = get_moscow_columns()
    moscow_exclude = get_moscow_exclude()

    # Считывание shape-файла
    raw_moscow = gpd.read_file('./moscow/mo.shp')
    raw_moscow = raw_moscow[moscow_columns]

    # Получение списка регионов
    global ao_list
    ao_list = list(set(raw_moscow['NAME_AO']))

    # Объединение регионов
    new_pols = []
    for ao in ao_list:
        regs = raw_moscow[raw_moscow['NAME_AO'] == ao]['geometry']
        regs_list = list(regs)

        unary_pol = ops.unary_union(regs_list)
        new_pols.append([ao, unary_pol])

    moscow = gpd.GeoDataFrame(
        new_pols, 
        columns=moscow_columns, 
        geometry='geometry', 
        crs=raw_moscow.crs
    )

    moscow = moscow[~moscow['NAME_AO'].isin(moscow_exclude)]
    moscow['NAME_AO'] = moscow['NAME_AO'].apply(lambda x: x.capitalize())
    ao_list = list(set(moscow['NAME_AO']))

    return moscow