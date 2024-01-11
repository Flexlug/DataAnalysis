import streamlit as st

import branca.colormap
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
from streamlit_folium import st_folium

from utils.loader_mosdata import preprocess_pol1_df, download_pol1_df, process_pdk, merge_pol1_pol2
from utils.loader_mosecom import download_pol2_df, preprocess_pol2_df
from utils.moscow import get_moscow, get_ao_list

moscow = get_moscow()
df = download_pol1_df()
pdks = process_pdk(df)
_pol_df = preprocess_pol1_df(df, moscow)
df2 = download_pol2_df()
_pol_df2 = preprocess_pol2_df(df2)
_whole_df = merge_pol1_pol2(_pol_df, _pol_df2)

green_color = '#00FF00'
red_color = '#FF0000'

def style_fn(_feature):
    parameter = _feature['properties']['Parameter']
    value = _feature['properties']['MonthlyAverage']
    cm = branca.colormap.LinearColormap([green_color, red_color], vmin=0, vmax=pdks[parameter])
    color_value = cm(value)
    ss = {
        "fillColor": color_value,
        "fillOpacity": 0.8,
        "weight": 0.8,
        "color": color_value,
    }
    return ss

@st.cache_data
def generate_plot(_whole_df):
    fig, ax = plt.subplots()
    for ao in get_ao_list():
        df = _whole_df[_whole_df.NAME_AO == ao]
        ax.plot(df['Period'], df['MonthlyAverage'], label=ao)
    ax.set_title('Показания угарного газа по автономным округам')
    ax.legend()
    return fig

@st.cache_data
def generate_pol2_geojsons(_whole_df, pdks):
    geojsons = {}

    for time in list(set(_whole_df['Period'])):
        df = _whole_df[_whole_df['Period'] == time]
        geojsons[time] = []
        for param in pdks.keys():
            folium_df = df[df.Parameter == param].drop('Period', axis=1)
            if len(folium_df.Parameter) == 0:
                continue
            geojson = folium.GeoJson(
                name=param,
                data=folium_df,
                style_function=style_fn,
                tooltip=folium.features.GeoJsonTooltip(["Parameter", "NAME_AO", "MonthlyAverage"]),
                show=True
            )
            geojsons[time].append(geojson)
    return geojsons

def main():
    st.title('Объединенные ежемесячные данные по угарному газу')
    st.write('Данные спаршены с портала mosecom.mos.ru и объединены с предыдущими')

    m = folium.Map([55.755864, 37.617698], zoom_start=10)
    geojsons = generate_pol2_geojsons(_whole_df, pdks)
    keys = sorted(list(geojsons.keys()))
    selected_date = st.select_slider(label='Дата', options=keys)
    selected_index = keys.index(selected_date)
    for json in geojsons[keys[selected_index]]:
        json.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m)
    st.divider()
    st.pyplot(generate_plot(_whole_df))

if __name__ == "__main__":
    main()