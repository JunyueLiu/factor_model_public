import numpy as np
import pandas as pd
import tqdm

from data_management.dataIO import market_data
from factor_zoo.factor_operator.alpha101_operator import returns
from factor_zoo.factor_operator.basic_operator import var, positive_var, negative_var
from factor_zoo.utils import open_to_close_high_low_limit


def risk_variance(daily_market: pd.DataFrame, window: int) -> pd.Series:

    close_ = daily_market['adj_close']
    unstack_ = close_.unstack()
    ret = returns(unstack_.values)
    factor = var(ret, window)
    factor = pd.DataFrame(factor, index=unstack_.index, columns=unstack_.columns).stack()
    factor.name = 'risk_variance_{}'.format(window)
    return factor


def risk_gainvariance(daily_market: pd.DataFrame, window: int) -> pd.Series:
    """

    :param close_:
    :param window:
    :return:
    """
    close_ = daily_market['adj_close']
    unstack_ = close_.unstack()
    ret = returns(unstack_.values)
    factor = positive_var(ret, window)
    factor = pd.DataFrame(factor, index=unstack_.index, columns=unstack_.columns).stack()
    factor.name = 'risk_gainvariance_{}'.format(window)
    return factor


def risk_lossvariance(daily_market: pd.DataFrame, window: int) -> pd.Series:
    """

    :param close_:
    :param window:
    :return:
    """
    close_ = daily_market['adj_close']
    unstack_ = close_.unstack()
    ret = returns(unstack_.values)
    factor = negative_var(ret, window)
    factor = pd.DataFrame(factor, index=unstack_.index, columns=unstack_.columns).stack()
    factor.name = 'risk_lossvariance_{}'.format(window)
    return factor


def lottery_demand(daily_market, d=5, rolling_d=20, price_var='adj_close'):
    """
    Bali, Turan G., Stephen J. Brown, Scott Murray and Yi Tang, 2017,
    "A lottery demand-based explanation of the beta anomaly", Journal of Financial and Quantitative Analysis, Forthcoming.

    The average of the five highest daily returns of the stock in a given month
    :return:
    """
    tqdm.tqdm.pandas()

    def _avg_highest(x: pd.Series, window: int, top: int):
        return x.rolling(window).agg(lambda s: np.mean(np.sort(s)[-top:]))

    unstack_ = daily_market[price_var].unstack()
    ret = unstack_.pct_change().stack().sort_index()
    factor = ret.groupby(level=1).progress_apply(lambda x: _avg_highest(x, rolling_d, d))
    if factor.index.nlevels == 3:
        factor = factor.droplevel(0).sort_index()

    factor.name = 'lottery_demand_{}_{}'.format(d, rolling_d)
    return factor


def ideal_oscillation(daily_market: pd.DataFrame,
                      d: int = 20, percentage: float = 0.25) -> pd.Series:
    """


    步骤 1 对选定股票 S，回溯取其最近 N 个交易日（这里选择 N=20）的数据；
    步骤 2 计算股票 S 每日的振幅 （最高价/最低价-1）；
    步骤 3 选择收盘价较高的λ（比如 40%）有效交易日，计算振幅均值得到高价振幅
    因子 V_high(λ)；
    步骤 4 选择收盘价较低的λ（比如 40%）有效交易日，计算振幅均值得到低价振幅
    因子 V_low(λ)。

    需要说明的是，有效交易日是指剔除停牌和一字涨跌停后的交易日。若股票 S 在
    最近 20 个交易日内，有效交易日天数小于 10 日，则股票 S 当日因子值设为空值。


    (1) 对选定股票，回溯取其最近20个交易日数据，计算股票每日振幅（最高价/最低价-1）
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
        osc = df['osc'].values
        limit_up_down = df['-up_down'].values
        paused = df['paused'].values

        for i in range(0, len(df) - d + 1):
            if paused[i: i + d].sum() + limit_up_down[i: i + d].sum() > int(0.5 * d):
                arr[i + d - 1] = np.nan
            else:
                index = np.argsort(close[i: i + d])
                v_low = np.mean(osc[i: i + d][index[:num]])
                v_high = np.mean(osc[i: i + d][index[-num:]])
                arr[i + d - 1] = v_high - v_low
        return pd.Series(arr, df.index)

    data = daily_market.copy()
    data['osc'] = data['adj_high'] / data['adj_low'] - 1
    data['-up_down'] = open_to_close_high_low_limit(data)
    data = data[['adj_close', 'osc', '-up_down', 'paused']]
    factor = data.groupby(level=1).progress_apply(func)
    if factor.index.nlevels == 3:
        factor = factor.droplevel(0).sort_index()
    else:
        factor = factor.swaplevel(0, 1).sort_index()
    factor.name = 'ideal_oscillation_{}_{}'.format(d, percentage)
    return factor


