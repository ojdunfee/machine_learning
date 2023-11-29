import pandas as pd
import numpy as np
import requests


def _get_data(census_code, value_col, start_year=2017, end_year=2021, api=False):
    for year in list(range(start_year, end_year)):
        url = f'https://api.census.gov/data/{year}/acs/acs5'
        if api:
            params = {
                'get': census_code,
                'for': 'county:*',
                'in': 'state:*',
                'key': '0fb360338440ce44424a0e782480c309cb8fbf6e'
            }
        else:
            params = {
                'get': census_code,
                'for': 'county:*',
                'in': 'state:*'
            }

        response = requests.get(url, params=params)
        data = response.json()

        df = pd.DataFrame(data[1:], columns=data[0])

        df[census_code] = df[census_code].astype(int)

        df['cfips'] = df['state'].astype(str).str.zfill(2) + df['county'].astype(str).str.zfill(3)
        df['cfips'] = df['cfips'].astype(int)

        df = df.rename(columns={census_code:value_col})

        df['date'] = pd.to_datetime(f'{year}-01-01')

        yield df


def _census_to_monthly(df, value_col, params={'method':'linear'}):
    for cfips, frame in df.groupby('cfips'):
        frame = frame[['date',value_col]].set_index('date').resample('M').sum().replace(0, np.nan)
        frame = frame.interpolate(**params)
        frame['cfips'] = cfips
        frame = frame.reset_index()
        frame['date'] = frame['date'].to_numpy().astype('datetime64[M]')
        yield frame


def get_census_data(census_code, value_col, start_year=2017, end_year=2021, api=False, params={'method':'linear'}):
    data = pd.concat(list(_get_data(census_code, value_col, start_year, end_year, api=api)))
    data = pd.concat(list(_census_to_monthly(data, value_col, params)))
    return data




if __name__ == "__main__":
    age_data = get_census_data('AGE', 'age', api=False)
    print(age_data)