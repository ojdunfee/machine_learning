import pandas as pd
import numpy as np

from data_cleaning import get_alternative_data, get_educational_ratios, generate_series
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import make_column_transformer, TransformedTargetRegressor
from xgboost import XGBRegressor

def get_data(path='forecasting/godaddy-microbusiness-density-forecasting/data'):
    train = pd.read_csv(f'{path}/train.csv', converters={'first_day_of_month':np.datetime64}).rename(columns={'first_day_of_month':'date'})
    alt_data = get_alternative_data(['B01001_001E','B15003_023E','B15003_025E','B15003_024E'],
                ['population','bachelors','professional','masters'], start_year=2017, end_year=2021, api=False)
    alt_data = get_educational_ratios(alt_data)
    df = pd.merge(train, alt_data, on=['date','cfips'], how='inner')
    df = pd.concat(list(generate_series(df)), ignore_index=True)
    return df

def get_model(df, target):
    df = df[['cfips','state','date','microbusiness_density','bachelors','professional','masters','series']].copy()
    df.set_index('date', inplace=True)

    categorical_features = df.select_dtypes(('O', int)).columns
    numeric_features = [col for col in df.select_dtypes(float).columns if col != target]

    ct = make_column_transformer(
        (OneHotEncoder(sparse=True, handle_unknown='ignore'), categorical_features),
        (MinMaxScaler(), numeric_features)
    )

    model = XGBRegressor(n_estimators=300, eta=0.2, max_depth=11)

    pipe = make_pipeline(ct, model)

    return pipe

if __name__ == '__main__':
    df = get_data()
    
    df.dropna(how='any', inplace=True)
    model = get_model(df, 'microbusiness_density')

    X, y = df.drop('microbusiness_density', axis=1), df['microbusiness_density']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f'R2: {r2_score(y_test, y_pred)}')
    print(f'MAE: {mean_absolute_error(y_test, y_pred)}')

    
