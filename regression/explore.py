import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import wrangle, split_scale
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split



data = wrangle.wrangle_telco()
X = data.drop(columns='total_charges').set_index('customer_id')
y = data.total_charges
scaler = StandardScaler().fit(data)

def months_to_years(df, column_name, year_title = 'years'):
    df[year_title] = [int(c / 12) for c in df[column_name]]
    return df



