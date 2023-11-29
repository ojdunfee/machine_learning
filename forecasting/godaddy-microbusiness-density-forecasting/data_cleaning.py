import pandas as pd
import numpy as np
import requests
from census import utils

def get_alternative_data(census_codes, value_cols, start_year, end_year, api=False):
    df = None
    for census_code, value_col in zip(census_codes, value_cols):
        data = utils.get_census_data(census_code, value_col, start_year=start_year, end_year=end_year, api=api)
        if df is None:
            df = data
        else:
            df = pd.merge(df, data, on=['date','cfips'], how='inner')
    return df


def get_educational_ratios(df):
    for col in ['bachelors','professional','masters']:
        df[col] = df[col] / df['population']
    return df

def generate_series(df, group='cfips'):
    for _, frame in df.groupby(group):
        frame = frame.sort_values(by='date', ascending=True)
        frame['series'] = np.arange(1,frame.shape[0] + 1)
        frame['series'] = frame['series'].astype(float)
        yield frame

if __name__ == '__main__':
    path = 'forecasting/godaddy-microbusiness-density-forecasting/data'
    train = pd.read_csv(f'{path}/train.csv', converters={'first_day_of_month':np.datetime64}).rename(columns={'first_day_of_month':'date'})
    alt_data = get_alternative_data(['B01001_001E','B15003_023E','B15003_025E','B15003_024E'],
                ['population','bachelors','professional','masters'], start_year=2017, end_year=2021, api=False)
    alt_data = get_educational_ratios(alt_data)
    alt_data.to_csv(f'{path}/alt.csv', index=False)
    df = pd.merge(train, alt_data, on=['date','cfips'], how='inner')
    df = pd.concat(list(generate_series(df)), ignore_index=True)
    

    

    