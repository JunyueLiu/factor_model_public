import pandas as pd
import numpy as np

from data_management.dataIO import market_data


def standard_upper_needle(daily_market: pd.DataFrame, period: int = 5):
    needle = daily_market['adj_high'] - np.maximum(daily_market['adj_open'], daily_market['adj_close'])
    mean = needle.groupby(level=1).rolling(period).mean().droplevel(0).sort_index()
    f = needle / mean
    f.name = 'standard_upper_needle_{}'.format(period)
    return f


def standard_lower_needle(daily_market: pd.DataFrame, period: int = 5):
    needle = np.minimum(daily_market['adj_open'], daily_market['adj_close']) - daily_market['adj_low']
    mean = needle.groupby(level=1).rolling(period).mean().droplevel(0).sort_index()
    f = needle / mean
    f.name = 'standard_lower_needle_{}'.format(period)
    return f


def upper_needle_mean(daily_market: pd.DataFrame, standard_period: int = 5, windows=20):
    sn = standard_upper_needle(daily_market, standard_period)
    f = sn.groupny(level=1).rolling(windows).mean().droplevel(0).sort_index()
    f.name = 'upper_needle_mean_{}_{}'.format(standard_period, windows)
    return f


def upper_needle_std(daily_market: pd.DataFrame, standard_period: int = 5, windows=20):
    sn = standard_upper_needle(daily_market, standard_period)
    f = sn.groupny(level=1).rolling(windows).std().droplevel(0).sort_index()
    f.name = 'upper_needle_std_{}_{}'.format(standard_period, windows)
    return f


def lower_needle_mean(daily_market: pd.DataFrame, standard_period: int = 5, windows=20):
    sn = standard_lower_needle(daily_market, standard_period)
    f = sn.groupny(level=1).rolling(windows).mean().droplevel(0).sort_index()
    f.name = 'lower_needle_mean_{}_{}'.format(standard_period, windows)
    return f


def lower_needle_std(daily_market: pd.DataFrame, standard_period: int = 5, windows=20):
    sn = standard_lower_needle(daily_market, standard_period)
    f = sn.groupny(level=1).rolling(windows).std().droplevel(0).sort_index()
    f.name = 'lower_needle_std_{}_{}'.format(standard_period, windows)
    return f

