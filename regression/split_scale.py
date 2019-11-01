import ethropic_encryption as e
import dbtools as d
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
from dbtools import get_db_url
import math
import pickle





def split_data(inp, train = .8, r_state = 123):
    train, test = train_test_split(inp, train_size = train, random_state = r_state)
    return train, test


def standard_scaler(inp, copy = True, mean = True, std = True):
    return StandardScaler(copy = copy, with_mean = mean, with_std = std).fit(inp)

def uniform_scaler(inp, n_q = 100, r_state = 123, copy = True):
    return QuantileTransformer(n_quantiles = n_q, output_distribution = 'uniform', \
                               random_state = r_state, copy=True).fit(inp)

def max_min_scaler(inp, frange = (0, 1), copy = True):
    return MinMaxScaler(copy=copy, feature_range=(frange[0], frange[1])).fit(inp)


def scale_inverse(inp, q_range = (25.0, 75.0), copy = True, centering = True, scaling = True):
    return RobustScaler(quantile_range=(q_range[0],q_range[1]), copy=True, with_centering=centering, with_scaling=scaling).fit(inp)

def gaussian_scaler(inp, standerd = True, copy = True, meth = 0):
    methods = ['yeo-johnson', 'box-cox']
    return PowerTransformer(method=methods[meth], standardize=standerd, copy=copy).fit(inp)

