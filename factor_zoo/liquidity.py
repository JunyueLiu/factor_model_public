# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:58:39 2022

@author: jingrui
"""

import inspect
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.regression.rolling import RollingOLS
from talib import abstract
from tqdm import tqdm

from arctic import Arctic
from data_management.dataIO import market_data
from data_management.keeper.ZooKeeper import ZooKeeper
from data_management.pandas_utils.parallel import time_series_parallel_apply
from factor_zoo.factor_operator.alpha101_operator import rank, ts_rank
from factor_zoo.utils import rolling_regression_alpha
from technical_analysis.momentum import AROON, CCI
from technical_analysis.overlap import EMA, SMA, KAMA, MAMA, T3
from technical_analysis.statistic_function import LINEARREG

def tradedvalue_rank(daily_market: pd.DataFrame, period: int = 60):
    """
    过去60日平均交易额
    :param daily_market:
    :return:
    """
    tradedvalue = (daily_market['money']).unstack()
    mean_tradedvalue = tradedvalue.rolling(period, min_periods=1).mean()
    mean_tradedvalue[tradedvalue.isna()] = float('nan')
    mean_tradedvalue = mean_tradedvalue.dropna(how='all', axis=0)
    rank_factor = pd.DataFrame(rank(mean_tradedvalue.values), index=mean_tradedvalue.index, columns=mean_tradedvalue.columns).stack()
    factor = 1 - rank_factor
    factor.name = 'tradedvalue_{}D_rank'.format(period)
    return factor

def tradedvalue_raw(daily_market: pd.DataFrame, period: int = 60):
    """
    过去60日平均交易额
    :param daily_market:
    :return:
    """
    tradedvalue = (daily_market['money']).unstack()
    mean_tradedvalue = tradedvalue.rolling(period, min_periods=1).mean()
    mean_tradedvalue[tradedvalue.isna()] = float('nan')
    mean_tradedvalue = mean_tradedvalue.dropna(how='all', axis=0)
    factor = mean_tradedvalue.stack()
    factor.name = 'tradedvalue_raw_{}D'.format(period)
    return factor


if __name__ == '__main__':
    zookeeper_config_path = '../cfg/factor_keeper_setting.ini'
    data_config_path = '../cfg/data_input.ini'
    keeper = ZooKeeper(zookeeper_config_path)
    start_date = '2020-01-01'
    
    daily_market = market_data.get_bars(adjust=True, eod_time_adjust=False, add_limit=False, start_date=start_date,
                                        config_path=data_config_path)
    
    # we use the logged prices for price momentum
    f = tradedvalue_rank(daily_market, period=60)
    category = 'radar'
    keeper.save_factor_value(category, f, source_code=inspect.getsource(tradedvalue_rank),
                              comment='Citi Radar Traded Value Raw Data', source='in-house', author='zjr',
                              to_arctic = False
                              )
    
    # save a local copy in .csv file
    df_f = f.unstack()
    df_f = df_f.dropna(how='all', axis=0)
    keeper.save_factor_value_csv(category, df_f, f.name)    
    