from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def data_wrangle():
    return pd.read_csv('time_dataframe.csv')




data = data_wrangle()

time_split = TimeSeriesSplit(max_train_size=None, n_splits=5)
for train_index, test_index in time_split.split(data):
    print("TRAIN:", train_index, "TEST:", test_index)


print(time_split)



