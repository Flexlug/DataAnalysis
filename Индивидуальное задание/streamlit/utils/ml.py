import matplotlib.pyplot
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils.moscow import get_ao_list

def train_test_split_df(df):
    split_index = int(0.8 * len(df))

    df_train = df.iloc[:split_index]
    df_test = df.iloc[split_index:]

    for ao in get_ao_list():
        df = df_train[df_train.NAME_AO == ao]

    for ao in get_ao_list():
        df = df_test[df_test.NAME_AO == ao]

    return (df_train, df_test)

def train_sarimax(ao_df):
    ao_df.set_index('Period', inplace=True)
    ao_df = ao_df.asfreq('MS')
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)

    model = SARIMAX(ao_df, order=order, seasonal_order=seasonal_order)
    result = model.fit(disp=False)

    return result

def get_prediction_for(test_df, model):
    start = test_df['Period'].min()
    end = test_df['Period'].max()
    predictions = model.get_prediction(start=start, end=end, dynamic=False)
    return predictions.predicted_mean