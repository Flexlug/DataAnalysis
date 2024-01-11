import streamlit as st

import branca.colormap
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
from streamlit_folium import st_folium

from utils.loader_taxi import download_taxi_df, preprocess_taxi_df
from utils.moscow import get_moscow, get_ao_list

moscow = get_moscow()
df3 = download_taxi_df()
_taxi_df = preprocess_taxi_df(df3)
max_parking_places = _taxi_df['CarCapacity'].max()

green_color = '#00FF00'
red_color = '#FF0000'

def style_fn(feature):
    value = feature['properties']['CarCapacity']
    cm = branca.colormap.LinearColormap([green_color, red_color], vmin=0, vmax=max_parking_places)
    color_value = cm(value)
    ss = {
        "fillColor": color_value,
        "fillOpacity": 0.8,
        "weight": 0.8,
        "color": color_value,
    }
    return ss

@st.cache_data
def generate_plot(_taxi_df):
    fig, ax = plt.subplots()
    for ao in get_ao_list():
        df = _taxi_df[_taxi_df.NAME_AO == ao]
        ax.plot(df['Period'], df['CarCapacity'], label=ao)
    ax.set_title('Количество парковочных мест для такси по автономным округам')
    ax.legend()
    return fig

@st.cache_data
def generate_pol3_geojsons(_whole_df):
    geojsons = {}

    for time in list(set(_taxi_df['Period'])):
        folium_df = _taxi_df[_taxi_df.Period == time].drop('Period', axis=1)
        geojson = folium.GeoJson(
            name='Парковки',
            data=folium_df,
            style_function=style_fn,
            tooltip=folium.features.GeoJsonTooltip(["CarCapacity", "NAME_AO"]),
            show=True
        )
        geojsons[time] = geojson
    return geojsons

def main():
    st.title('Данные по парковочным местам')
    st.write('Данные спаршены с портала mosecom.mos.ru')

    m = folium.Map([55.755864, 37.617698], zoom_start=10)
    geojsons = generate_pol3_geojsons(_taxi_df)
    keys = sorted(list(geojsons.keys()))
    selected_date = st.select_slider(label='Дата', options=keys)
    selected_index = keys.index(selected_date)
    geojsons[keys[selected_index]].add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m)
    st.divider()
    st.pyplot(generate_plot(_taxi_df))

if __name__ == "__main__":
    main()