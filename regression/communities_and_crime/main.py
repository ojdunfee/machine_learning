import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer

import optuna



class Model:

    def __init__(self, data, target, train=True):
        self.data = data.df
        self.encoder_dict = data.encoder_dict
        self.X = self.data.drop(target, axis=1)
        self.y = self.data[target]
        self.train = train

    


class Data:

    def __init__(self, data_file, info_file):
        self.data_file = data_file
        self.info_file = info_file
        self.df, self.encoder_dict = self.clean_data(data_file, info_file)

    def load_data(self, filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                yield line.strip('\n').split(',')

    def keep_cols(self, df, thresh=.1):
        for i in range(len(df.isna().sum())):
            pct_missing = df.isna().sum()[i] / df.shape[0]
            if pct_missing < thresh:
                yield df.isna().sum().index[i]

    def get_df_info(self, df, filename):
        cols = []
        data_types = {}
        for item in list(self.load_data(filename)):
            if item[0].startswith('@attribute'):
                col = item[0].split()[1]
                d_type = item[0].split()[-1]
                cols.append(col)
                if d_type in data_types.keys():
                    data_types[d_type].append(col)
                else:
                    data_types[d_type] = [col]
        
        df.columns = cols
        df = df.replace('?', np.nan)

        for k in data_types:
            if k == 'numeric':
                df[data_types[k]] = df[data_types[k]].astype(float)
            if k == 'string':
                df[data_types[k]] = df[data_types[k]].astype(str)

        return df

    def transform_strings(self, df):
        D = {}
        for col in df.select_dtypes('O').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            D[col] = le
        
        return df, D

    def clean_data(self, data_file, info_file):
        df = pd.DataFrame(list(self.load_data(data_file)))
        df = self.get_df_info(df, info_file)
        df, encoder_dict = self.transform_strings(df)

        return df, encoder_dict