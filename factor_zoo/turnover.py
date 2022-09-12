import inspect
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_management.dataIO import market_data
from data_management.dataIO.trade_data import get_trade, TradeTable
from data_management.keeper.ZooKeeper import ZooKeeper
from factor_zoo.factor_operator.ideal_operation import ideal_mean_high_minus_low
from factor_zoo.utils import fundamental_preprocess_daily, combine_market_with_fundamental

tqdm.pandas()


def turnover(daily_market, capital_change, trading_date) -> pd.Series:
    """

    :param daily_market:
    :param capital_change:
    :param trading_date:
    :return:
    """
    daily = daily_market.loc[pd.to_datetime(trading_date[0]):, :]
    daily = daily.loc[:min(pd.to_datetime(trading_date[-1]), daily.index.get_level_values(level=0)[-1]), :]
    cc = fundamental_preprocess_daily(capital_change, trading_date, True)
    shares = cc['share_rmb']
    merged = combine_market_with_fundamental(daily, shares)
    # vol is in 100 shares, share_rmb is in 100,000 shares. Final result is in percentage.
    factor = merged['vol'] / merged['share_rmb']
    factor.name = 'turnover'
    return factor


def turnover_daily_basic(daily_basic: pd.DataFrame) -> pd.Series:
    factor = daily_basic['turnover_rate']
    factor.name = 'turnover_daily_basic'
    return factor


def turnover_float_daily_basic(daily_basic: pd.DataFrame) -> pd.Series:
    factor = daily_basic['turnover_rate_f']
    factor.name = 'turnover_float_daily_basic'
    return factor


def log_turnover(market, capital_change, trading_date) -> pd.Series:
    """
    Log of trading volume divided by shares outstanding
    :return:
    """
    turnover_factor = turnover(market, capital_change, trading_date)
    factor = pd.Series(np.log(turnover_factor), index=turnover_factor.index, name='log_turnover')
    return factor


def log_turnover_daily_basic(daily_basic: pd.DataFrame) -> pd.Series:
    turnover_factor = turnover_daily_basic(daily_basic)
    return pd.Series(np.log(turnover_factor), index=turnover_factor.index, name='log_turnover_daily_basic')


def log_turnover_float_daily_basic(daily_basic: pd.DataFrame) -> pd.Series:
    turnover_factor = turnover_float_daily_basic(daily_basic)
    return pd.Series(np.log(turnover_factor), index=turnover_factor.index, name='log_turnover_float_daily_basic')


def abnormal_turnover_daily_basic(daily_basic: pd.DataFrame, short_period: int = 21,
                                  long_period: int = 252) -> pd.Series:
    """
    definition: 在t月末，异常换手率的定义为过去21个交易日的平均换手率和过去252个交易日的平均换手率的比值
    :param daily_basic:
    :param short_period:
    :param long_period:
    :return:
    """
    tqdm.pandas()
    turnover_factor = turnover_daily_basic(daily_basic)
    short_period_avg = turnover_factor.groupby(level=1).progress_apply(lambda x_: x_.rolling(short_period).mean())
    long_period_avg = turnover_factor.groupby(level=1).progress_apply(lambda x_: x_.rolling(long_period).mean())
    factor = short_period_avg / long_period_avg
    factor = factor.round(4)
    factor.name = 'abnormal_turnover_daily_basic_{}_{}'.format(short_period, long_period)
    return factor


def ideal_turnover_daily_basic(daily_market: Union[pd.DataFrame, pd.Series],
                               daily_basic: Union[pd.DataFrame, pd.Series],
                               d: int = 20, percentage: float = 0.25) -> pd.Series:
    """
    (1) 对选定股票，回溯取其最近20个交易日数据，计算股票每日换手率
    (2) 选择收盘价较高的25%有效交易日，计算振幅均值得到高价振幅因子V_high
    (3) 选择收盘价较低的25%有效交易日，计算振幅均值得到低价振幅因子V_low
    (4) 将高价振幅因子V_high与低价振幅因子V_low作差，得到理想振幅因子 V = V_high - V_low
    :return:
    """
    num = int(d * percentage)

    def func(df):
        arr = np.empty(len(df))
        arr[:] = np.NaN
        close = df['adj_close'].values
        turnover = df['turnover_rate_f'].values
        for i in range(0, len(df) - d + 1):
            index = np.argsort(close[i: i + d])
            v_low = np.mean(turnover[i: i + d][index[:num]])
            v_high = np.mean(turnover[i: i + d][index[-num:]])
            arr[i + d - 1] = v_high - v_low
        return pd.Series(arr, index=df.index)

    if isinstance(daily_market, pd.DataFrame):
        adj_close = daily_market['adj_close'].to_frame()
    else:
        adj_close = daily_market.to_frame('adj_close')

    if isinstance(daily_basic, pd.DataFrame):
        turnover_rate_f = daily_basic['turnover_rate_f'].to_frame()
    else:
        turnover_rate_f = daily_basic.to_frame('turnover_rate_f')

    data = adj_close.join(turnover_rate_f)
    # todo go parallel
    factor = data.groupby(level=1).progress_apply(func)
    if factor.index.nlevels == 3:
        factor = factor.droplevel(0).sort_index()
    else:
        factor = factor.swaplevel(0, 1).sort_index()
    factor.name = 'ideal_turnover_daily_basic_{}_{}'.format(d, percentage)
    return factor

def ideal_turnover_daily_basic_v2(daily_market: Union[pd.DataFrame, pd.Series],
                               daily_basic: Union[pd.DataFrame, pd.Series],
                               d: int = 20, percentage: float = 0.25) -> pd.Series:
    """
    (1) 对选定股票，回溯取其最近20个交易日数据，计算股票每日换手率
    (2) 选择收盘价较高的25%有效交易日，计算振幅均值得到高价振幅因子V_high
    (3) 选择收盘价较低的25%有效交易日，计算振幅均值得到低价振幅因子V_low
    (4) 将高价振幅因子V_high与低价振幅因子V_low作差，得到理想振幅因子 V = V_high - V_low
    :return:
    """

    if isinstance(daily_market, pd.DataFrame):
        adj_close = daily_market['adj_close'].to_frame()
    else:
        adj_close = daily_market.to_frame('adj_close')

    if isinstance(daily_basic, pd.DataFrame):
        turnover_rate_f = daily_basic['turnover_rate_f'].to_frame()
    else:
        turnover_rate_f = daily_basic.to_frame('turnover_rate_f')

    data = adj_close.join(turnover_rate_f)
    adj_close = data['adj_close'].unstack()
    turnover_rate = data['turnover_rate_f'].unstack()
    idx = adj_close.index
    cols = adj_close.columns
    factor = ideal_mean_high_minus_low(adj_close.values, turnover_rate.values, d, percentage)
    factor = pd.DataFrame(factor, idx, cols).stack().sort_index()
    factor.name = 'ideal_turnover_daily_basic_v2_{}_{}'.format(d, percentage)
    return factor


def turnover_volatility(daily_basic: pd.DataFrame, d: int = 20):
    """

    :param daily_basic:
    :param d:
    :return:
    """
    tqdm.pandas()
    turnover_factor = turnover_daily_basic(daily_basic)
    turnover_std = turnover_factor.groupby(level=1).progress_apply(lambda x_: x_.rolling(d).std())
    turnover_std.name = 'turnover_volatility_{}'.format(d)
    return turnover_std


def turnover_skew(daily_basic: pd.DataFrame, d: int = 20):
    """

    :param daily_basic:
    :param d:
    :return:
    """
    tqdm.pandas()
    turnover_factor = turnover_daily_basic(daily_basic)
    factor = turnover_factor.groupby(level=1).progress_apply(lambda x_: x_.rolling(d).skew())
    factor.name = 'turnover_skew_{}'.format(d)
    return factor


def turnover_kurt(daily_basic: pd.DataFrame, d: int = 20):
    """

    :param daily_basic:
    :param d:
    :return:
    """
    tqdm.pandas()
    turnover_factor = turnover_daily_basic(daily_basic)
    factor = turnover_factor.groupby(level=1).progress_apply(lambda x_: x_.rolling(d).kurt())
    factor.name = 'turnover_kurt_{}'.format(d)
    return factor

