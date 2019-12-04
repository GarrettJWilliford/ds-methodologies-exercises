import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/Users/garrettwilliford/Desktop/codeup-data-science/ds-methodologies-exercises/outliers/lemonade.csv')



def anomaly_detect(data, column, std = 3, return_column = False):
    zscores = pd.Series((data[column] - data[column].mean()) / data[column].std())
    zscores = pd.Series((data[column] - data[column].mean()) / data[column].std())
    if return_column:
        data[column + '_stand_dev'] = [int(z) for z in zscores]
        return data
    return data[zscores.abs() > std]




def get_lower_and_upper_bounds(series, multiply):
    data['quartile'] = pd.qcut(data['Sales'], 4, labels=False)
    q3 = series.quantile(.75)
    q1 = series.quantile(.25)
    iqr = q3 - q1
    lower_bound = q1 - multiply * iqr
    upper_bound = q1 + multiply * iqr
    return lowe_bound, upper_bound


    
    
#df.Temperature = np.where(df.Temperature == 212,200,df.Temperature)


data = anomaly_detect(data, 'Sales', return_column = True)
print(data.quantile())
print(get_lower_and_upper_bounds(data))