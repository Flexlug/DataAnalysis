import streamlit as st

import branca.colormap
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import branca.colormap
import geopandas as gpd
import folium
import pandas as pd
from utils.loader_mosecom import download_pol2_df, preprocess_pol2_df
from utils.loader_mosdata import download_pol1_df, merge_pol1_pol2, preprocess_pol1_df, process_pdk
from utils.loader_taxi import download_taxi_df, preprocess_taxi_df
from utils.loader_contrib import count_contrib_df
from utils.moscow import get_moscow, get_ao_list

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

_contrib_df = get_contrib()

max_contrib = _contrib_df['ContributionValue'].max()

green_color = '#00FF00'
red_color = '#FF0000'

def style_fn(feature):
    value = feature['properties']['ContributionValue']
    cm = branca.colormap.LinearColormap([green_color, red_color], vmin=0, vmax=max_contrib)
    color_value = cm(value)
    ss = {
        "fillColor": color_value,
        "fillOpacity": 0.8,
        "weight": 0.8,
        "color": color_value,
    }
    return ss


@st.cache_data
def generate_plot(_contrib_df):
    fig, ax = plt.subplots()
    for ao in get_ao_list():
        df = _contrib_df[_contrib_df.NAME_AO == ao]
        ax.plot(df['Period'], df['ContributionValue'], label=ao)
    ax.set_title('Удельный вклад парковочных мест по выбросам угарного газа')
    ax.legend()
    return fig

@st.cache_data
def generate_contrib_geojsons(_contrib_df):
    geojsons = {}

    for time in list(set(_contrib_df['Period'])):
        folium_df = _contrib_df[_contrib_df.Period == time].drop('Period', axis=1)
        geojson = folium.GeoJson(
            name='Удельный вклад',
            data=folium_df,
            style_function=style_fn,
            tooltip=folium.features.GeoJsonTooltip(["ContributionValue", "NAME_AO"]),
            show=True
        )
        geojsons[time] = geojson
    return geojsons

def main():
    st.title('Удельный вклад парковочных мест по выбросам угарного газа')
    st.markdown(
    '''
    Мы введем признак: **Удельный вклад парковочных мест по выбросам угарного газа**

    Рассчитывать его будем по формуле:
    ''')
    st.latex(r'''ContributionValue = \frac{MonthlyAverage}{CarCapacity}''')

    st.markdown(
    '''
    Индекс рассчитывается для каждого региона отдельно
    ''')

    m = folium.Map([55.755864, 37.617698], zoom_start=10)
    geojsons = generate_contrib_geojsons(_contrib_df)
    keys = sorted(list(geojsons.keys()))
    selected_date = st.select_slider(label='Дата', options=keys)
    selected_index = keys.index(selected_date)
    geojsons[keys[selected_index]].add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m)
    st.divider()
    st.pyplot(generate_plot(_contrib_df))
    st.divider()
    st.dataframe(_contrib_df.drop('geometry', axis=1))

if __name__ == "__main__":
    main()