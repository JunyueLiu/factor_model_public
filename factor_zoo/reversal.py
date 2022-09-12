import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from arctic import Arctic
from data_management.keeper.ZooKeeper import ZooKeeper
from data_management.pandas_utils.parallel import time_series_parallel_apply
from factor_zoo.money_flow import cash_inflow_strength3_large, cash_inflow_strength3_small
from factor_zoo.utils import rolling_regression_residual

tqdm.pandas()


def short_term_reverse(daily_market: pd.DataFrame, period: int = 20):
    close = daily_market['adj_close'].unstack()  # type: pd.DataFrame
    close_start = close.shift(period)
    factor = (close / close_start - 1).stack()
    factor.name = 'short_term_reverse_{}'.format(period)
    return factor


def ideal_reverse(daily_market: pd.DataFrame,
                  ts_moneyflow: pd.DataFrame,
                  d: int = 20,
                  percentage: float = 0.5)  -> pd.Series:
    num = int(d * percentage)

    def func(df):
        arr = np.empty(len(df))
        arr[:] = np.NaN
        ret = df['pct'].values
        lg_pct = df['lg_pct'].values
        for i in range(0, len(df) - d + 1):
            index = np.argsort(lg_pct[i: i + d])
            m_low = np.sum(ret[i: i + d][index[:num]])
            m_high = np.sum(ret[i: i + d][index[-num:]])
            arr[i + d - 1] = m_high - m_low
        return pd.Series(arr, index=df.index)

    daily_market = daily_market[['adj_close', 'money']]
    large_amount = (ts_moneyflow['buy_elg_amount'] + ts_moneyflow['sell_elg_amount']) * 10_000
    data = daily_market.join(large_amount.to_frame('large_amount'))
    data['lg_pct'] = data['large_amount'] / data['money']
    data['pct'] = data['adj_close'].groupby(level=1).pct_change(1)
    data = data[['pct', 'lg_pct']]
    factor = data.groupby(level=1).progress_apply(func)
    if factor.index.nlevels == 3:
        factor = factor.droplevel(0).sort_index()
    else:
        factor = factor.swaplevel(0, 1).sort_index()
    factor.name = 'ideal_reverse_{}_{}'.format(d, percentage)
    return factor


def ideal_reverse2(money_flow: pd.DataFrame,
                   d: int = 20,
                   percentage: float = 0.5,
                   split_para='net_pct_main'
                   )  -> pd.Series:
    num = int(d * percentage)

    def func(df):
        arr = np.empty(len(df))
        arr[:] = np.NaN
        ret = df['change_pct'].values / 100
        lg_pct = df[split_para].values
        for i in range(0, len(df) - d + 1):
            index = np.argsort(lg_pct[i: i + d])
            m_low = np.sum(ret[i: i + d][index[:num]])
            m_high = np.sum(ret[i: i + d][index[-num:]])
            arr[i + d - 1] = m_high - m_low
        return pd.Series(arr, index=df.index)

    data = money_flow[['change_pct', split_para]]
    factor = data.groupby(level=1).progress_apply(func)
    if factor.index.nlevels == 3:
        factor = factor.droplevel(0).sort_index()
    else:
        factor = factor.swaplevel(0, 1).sort_index()
    factor.name = 'ideal_reverse2_{}_{}_{}'.format(split_para, d, percentage)
    return factor


def moneyflow_large_residual_reversal(tushare_moneyflow: pd.DataFrame,
                                      daily_market: pd.DataFrame,
                                      window: int) -> pd.Series:
    """

    :param tushare_moneyflow:
    :param close:
    :param window:
    :return:
    """
    s = cash_inflow_strength3_large(tushare_moneyflow, window)
    ret = short_term_reverse(daily_market, window)
    merged_data = ret.to_frame('ret').join(s)
    roll_reg = partial(rolling_regression_residual, y_col='ret', X_cols=[s.name], N=window)
    factor = time_series_parallel_apply(merged_data, roll_reg)
    if factor.index.nlevels == 3:
        factor = factor.droplevel(0).sort_index()
    factor.name = 'moneyflow_large_residual_reversal_{}'.format(window)
    return factor


def moneyflow_small_residual_reversal(tushare_moneyflow: pd.DataFrame,
                                      daily_market: pd.DataFrame,
                                      window: int) -> pd.Series:
    """

    :param tushare_moneyflow:
    :param close:
    :param window:
    :return:
    """
    s = cash_inflow_strength3_small(tushare_moneyflow, window)
    ret = short_term_reverse(daily_market, window)
    merged_data = ret.to_frame('ret').join(s)
    roll_reg = partial(rolling_regression_residual, y_col='ret', X_cols=[s.name], N=window)
    factor = time_series_parallel_apply(merged_data, roll_reg)
    if factor.index.nlevels == 3:
        factor = factor.droplevel(0).sort_index()
    factor.name = 'moneyflow_small_residual_reversal_{}'.format(window)
    return factor


def _read_cal_10_close_ret(i: str, store: Arctic) -> pd.Series:
    df = store['30m'].read(i).data

    close_10 = df[df.index.time == datetime.time(10, 0)]['close']
    close = df[df.index.time == datetime.time(15, 0)]['close']
    if len(close) == 1:
        return pd.Series()

    intraday_ret = np.log(close.values / close_10.values)
    intraday_ret = pd.DataFrame(intraday_ret, index=pd.to_datetime(close.index.date))
    intraday_ret['code'] = i
    intraday_ret = intraday_ret.set_index('code', append=True).squeeze()
    intraday_ret.index.names = ['date', 'code']
    return intraday_ret


def intraday_segmented_reversal(store: Arctic, n=20) -> pd.Series:
    """
    20200901-华安证券-市场微观结构剖析之九：昼夜分离，隔夜跳空与日内反转选股因子
    𝑅𝑒𝑡10:00−𝑡𝑜−𝑐𝑙𝑜𝑠𝑒 = ∑𝑙𝑛 (𝐶𝑙𝑜𝑠𝑒_𝑡 / 𝑃𝑟𝑖𝑐𝑒_𝑡_10:00)
    Parameters
    ----------
    store
    n

    Returns
    -------

    """
    lib = store['30m']
    instruments = lib.list_symbols()
    func = partial(_read_cal_10_close_ret, store=store)
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(func, instruments, chunksize=1),
                            total=len(instruments)))

    f = pd.concat(results).sort_index()
    f = f.unstack().rolling(n).mean().stack().sort_index()
    f.name = 'intraday_segmented_reversal_{}'.format(n)
    return f

