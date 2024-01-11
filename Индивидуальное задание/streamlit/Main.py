import streamlit as st

from utils.loader_mosecom import download_pol2_df, preprocess_pol2_df
from utils.loader_mosdata import download_pol1_df, merge_pol1_pol2, preprocess_pol1_df, process_pdk
from utils.loader_taxi import download_taxi_df, preprocess_taxi_df
from utils.moscow import get_moscow

st.set_page_config(page_title='Главная страница')

st.title('Главная страница')
st.markdown(
'''
# Задача

Прогнозирование **доли угарного газа на парковочное место в Москве** на автономный округ. Для этого мы будем использовать данные из портала открытых данных и портала мосэкомониторинга:
- [Парковки такси](https://data.mos.ru/opendata/621)
- [Среднемесячные показатели загрязнения атмосферного воздуха](https://data.mos.ru/opendata/2453/) - исторические данные;
- [Среднемесячные показатели загрязнения атмосферного воздуха](https://mosecom.mos.ru) - за последний год

# Алгоритм работы

1. Загрузка и предобработка shape-файла с автономными округами Московской области;
2. Закачка и предобработка среднемесяных показателей с data.mos.ru;
3. Закачка и предобработка среднемесяных показателей с mosecom.mos.ru;
4. Объединение данных и их визуализация;
5. Закачка и предобработка данных о парковках. Визуализация;
6. Инициализация, обучение моделей машинного обучения. Визуализация результатов;
7. Инициализация, обучение моделей глубокого обучения. Визуализация результатов;
''')

with st.status('Загрузка данных. Пожалуйста, подождите...', expanded=True) as status:
    # Вызываем вхолостую функции, требующие "прогрева"
    st.write('Загрузка shape-файла Москвы')
    moscow = get_moscow()

    st.write('Загрузка датасета из data.mos.ru по угарному газу')
    df = download_pol1_df()

    st.write('Рассет ПДК')
    pdks = process_pdk(df)

    st.write('Предобработка датасета по угарному газу')
    pol_df = preprocess_pol1_df(df, moscow)

    st.write('Загрузка датасета из mosecom.mos.ru по угарному газу')
    df2 = download_pol2_df()
    
    st.write('Предобработка датасета по угарному газу')
    pol2_df = preprocess_pol2_df(df2)

    st.write('Объединение датасетов')
    whole_df = merge_pol1_pol2(pol_df, pol2_df)

    st.write('Загрузка датасета из data.mos.ru по такси')
    df3 = download_taxi_df()
    
    st.write('Предобработка датасета по такси')
    taxi_df = preprocess_taxi_df(df3)

    status.update(label='Данные загружены', state='complete', expanded=False)
