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

def get_stations_url_list():
    ALL_STATIONS_URL = 'https://mosecom.mos.ru/stations/'
    ITEMS_XPATH = '//div[@class="allstan-item"]'
    AO_XPATH = './div[@class="allstan-item-slide"]/h4/text()'
    STATIONS_LIST_XPATH = './div[@class="allstan-item-table"]/div[contains(@class, "allstan-item-row")]/div[@class="row-title"]/a/@href'

    params = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:45.0) Gecko/20100101 Firefox/45.0'
    }

    try:
        response = requests.get(ALL_STATIONS_URL, params=params)

        # Проверка статуса ответа
        if response.status_code == 200:
            parser = etree.HTMLParser()
            tree = etree.fromstring(response.text, parser)
            aos_nodes = tree.xpath(ITEMS_XPATH)
            aos = {}
            for ao_node in aos_nodes:
                ao_name = ao_node.xpath(AO_XPATH)[0]
                stations_urls = ao_node.xpath(STATIONS_LIST_XPATH)
                aos[ao_name] = stations_urls
            return aos

        else:
            print(f"Ошибка запроса. Статус код: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {e}")
        return None
    
def get_df_for_station(url: str):
    SCRIPT_XPATH = '//script[contains(text(), "AirCharts.init")]/text()'

    params = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:45.0) Gecko/20100101 Firefox/45.0'
    }

    try:
        response = requests.get(url, params=params)

        # Проверка статуса ответа
        if response.status_code == 200:
            parser = etree.HTMLParser()
            tree = etree.fromstring(response.text, parser)
            script_node = tree.xpath(SCRIPT_XPATH)[0]
            start, end = (script_node.find('{'), script_node.rfind('}'))
            script_node = script_node[start:end+1]

            name = str(url).strip('/').split('/')[-1]
            js = json.loads('[' + script_node + ']')

            data = []
            js_data = js[0]['units']['y']
            for key in list(js_data.keys()):
                for param in list(js_data.keys()):
                    arr = js_data[param]['data']
                    data.append([
                       [timestamp, name, param, value] for timestamp, value in arr
                    ])

            merged_data = list(itertools.chain(*data))
            df = pd.DataFrame(merged_data, columns=['Period', 'NAME_AO', 'Parameter', 'MonthlyAverage'])
            df['Period'] = pd.to_datetime(df['Period'], unit='ms')
            return df
        else:
            print(f"Ошибка запроса. Статус код: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {e}")
        return None