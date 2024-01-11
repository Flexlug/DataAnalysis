import os
from os import path
from stqdm import stqdm
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

@st.cache_data
def get_dataset_info(dataset_id, api_key):
    base_url = f"https://apidata.mos.ru/v1/datasets/{dataset_id}"
    params = {
        'api_key': api_key
    }

    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Ошибка запроса. Статус код: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {e}")
        return None

@st.cache_data
def get_dataset_version(dataset_id, api_key):
    base_url = f"https://apidata.mos.ru/v1/datasets/{dataset_id}/version"
    params = {
        'api_key': api_key
    }

    try:
        response = requests.get(base_url, params=params)

        # Проверка статуса ответа
        if response.status_code == 200:
            # Возврат десериализованного JSON объекта
            return response.json()
        else:
            print(f"Ошибка запроса. Статус код: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {e}")
        return None
    
@st.cache_data
def get_version_history(dataset_id):
    base_url = f"https://data.mos.ru/odata/catalog/versions"
    params = {
        'idDataset': dataset_id
    }

    try:
        response = requests.get(base_url, params=params)

        # Проверка статуса ответа
        if response.status_code == 200:
            # Возврат десериализованного JSON объекта
            return response.json()
        else:
            print(f"Ошибка запроса. Статус код: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {e}")
        return None

@st.cache_data
def get_dataset_rows(dataset_id, version_number, release_number, api_key, skip=0):
    base_url = f"https://apidata.mos.ru/v1/datasets/{dataset_id}/rows"
    params = {
        'versionNumber': version_number,
        'releaseNumber': release_number,
        '$top': 500,
        '$skip': skip,
        'api_key': api_key,
    }

    try:
        response = requests.get(base_url, params=params)
        # Проверка статуса ответа
        if response.status_code == 200:
            # Возврат десериализованного JSON объекта
            return response.json()
        else:
            print(f"Ошибка запроса. Статус код: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {e}")
        return None

@st.cache_resource
def download_dataset(dataset_id, api_key, version_number = None, release_number = None):
    info = get_dataset_info(dataset_id, api_key)
    row_count = info['ItemsCount']

    if version_number is None or release_number is None:
        last_version = get_dataset_version(dataset_id, api_key)
        version_number = last_version['VersionNumber']
        release_number = last_version['ReleaseNumber']

    all_rows = []
    for offset in stqdm(range(0, row_count, 500), desc=f'Downloading dataset v{version_number}.{release_number}'):
        raw_rows = get_dataset_rows(
            dataset_id,
            version_number, 
            release_number, 
            api_key=api_key,
            skip=offset
        )
        rows = list(map(lambda x: x['Cells'], raw_rows))
        all_rows.extend(rows)
    df = pd.DataFrame(data=all_rows)
    return df