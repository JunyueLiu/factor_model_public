import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import bottleneck as bn
import tqdm

from arctic import Arctic
import warnings

warnings.filterwarnings('ignore')


def _s(df):
    """

    Parameters
    ----------
    df

    Returns
    -------

    """
    return (df['pct_change'].abs() / df['money'] ** 0.5).fillna(0)


def _s1(df):
    """
    S1 = V 分钟成交量
    Parameters
    ----------
    df

    Returns
    -------

    """
    return df['money']


def _s2(df, windows):
    """
    S2 = rank(|R|) + rank(V) 分钟涨跌幅绝对值分位排名与分钟成交量分位排名之和
    Parameters
    ----------
    df

    Returns
    -------

    """
    r_rolling_rank = bn.move_rank(df['pct_change'].abs(), windows)
    v_rolling_rank = bn.move_rank(df['money'], windows)
    return r_rolling_rank + v_rolling_rank


def _s3(df):
    """
     S3 = |R| / ln(V) 分钟涨跌幅绝对值除以分钟成交量对数值
    Parameters
    ----------
    df

    Returns
    -------

    """
    return (df['pct_change'].abs() / np.log(df['money'])).fillna(0)


def _read_transform_S(i: str, lib, n, p, s_function):
    df = lib.read(i).data
    windows = n * 240
    if len(df) < windows:
        return pd.Series()

    df = df[['close', 'code', 'money', 'volume']]
    df.loc[:, 'pct_change'] = df['close'].pct_change()
    idx = df.index
    eod = idx[idx.time == datetime.time(15, 0)]

    s = s_function(df)

    money = df['money'].fillna(0).values
    volume = df['volume'].fillna(0).values
    m = bn.move_sum(money, windows)
    v = bn.move_sum(volume, windows)
    vwap_all = m / v
    s_rolling_rank = bn.move_rank(s, windows) + 1
    mask = (s_rolling_rank >= (1 - p)).astype(int)
    smart_m = bn.move_sum(money * mask, windows)
    smart_v = bn.move_sum(volume * mask, windows)
    vwap_smart = smart_m / smart_v
    f = pd.Series(vwap_smart / vwap_all, index=idx)
    f = f.loc[eod]
    f.index = pd.to_datetime(f.index.date)
    f.index = pd.MultiIndex.from_product([f.index, [i]], names=['date', 'code'])

    return f


def smart_money_v1(store: Arctic, n=10, p=0.2):
    """
    步骤 1 对选定股票，回溯取其过去 10 个交易日的分钟行情数据；
    步骤 2 构造指标St = |Rt|/√Vt，其中𝑅𝑡为第 t 分钟涨跌幅，𝑉𝑡为第 t 分钟成交量；
    步骤 3 将分钟数据按照指标St从大到小进行排序，取成交量累积占比前 20%的分钟，视为聪明钱交易；
    步骤 4 计算聪明钱交易的成交量加权平均价VWAPsmart；
    步骤 5 计算所有交易的成交量加权平均价VWAPall；
    步骤 6 聪明钱因子Q = VWAPsmart/VWAPall。
    Parameters
    ----------
    store
    n
    p

    Returns
    -------

    """
    lib = store['1m']
    instruments = lib.list_symbols()
    func = partial(_read_transform_S, store=store, n=n, p=p, s_function=_s)
    with ProcessPoolExecutor() as executor:
        results = list(tqdm.tqdm(executor.map(func, instruments, chunksize=1), total=len(instruments)))
    f = pd.concat(results).sort_index()
    f.name = 'smart_money_v1_{}_{}'.format(n, p)
    return f


def smart_money_v2_s1(store: Arctic, n=10, p=0.2):
    """
    步骤 1 对选定股票，回溯取其过去 10 个交易日的分钟行情数据；
    步骤 2 构造指标St = |Rt|/√Vt，其中𝑅𝑡为第 t 分钟涨跌幅，𝑉𝑡为第 t 分钟成交量；
    步骤 3 将分钟数据按照指标St从大到小进行排序，取成交量累积占比前 20%的分钟，视为聪明钱交易；
    步骤 4 计算聪明钱交易的成交量加权平均价VWAPsmart；
    步骤 5 计算所有交易的成交量加权平均价VWAPall；
    步骤 6 聪明钱因子Q = VWAPsmart/VWAPall。
    Parameters
    ----------
    store
    n
    p

    Returns
    -------

    """
    lib = store['1m']
    instruments = lib.list_symbols()
    func = partial(_read_transform_S, store=store, n=n, p=p, s_function=_s1)
    with ProcessPoolExecutor() as executor:
        results = list(tqdm.tqdm(executor.map(func, instruments, chunksize=1), total=len(instruments)))
    f = pd.concat(results).sort_index()
    f.name = 'smart_money_v2_s1_{}_{}'.format(n, p)
    return f


def smart_money_v2_s2(store: Arctic, n=10, p=0.2):
    """
    步骤 1 对选定股票，回溯取其过去 10 个交易日的分钟行情数据；
    步骤 2 构造指标St = |Rt|/√Vt，其中𝑅𝑡为第 t 分钟涨跌幅，𝑉𝑡为第 t 分钟成交量；
    步骤 3 将分钟数据按照指标St从大到小进行排序，取成交量累积占比前 20%的分钟，视为聪明钱交易；
    步骤 4 计算聪明钱交易的成交量加权平均价VWAPsmart；
    步骤 5 计算所有交易的成交量加权平均价VWAPall；
    步骤 6 聪明钱因子Q = VWAPsmart/VWAPall。
    Parameters
    ----------
    store
    n
    p

    Returns
    -------

    """
    lib = store['1m']
    instruments = lib.list_symbols()
    func = partial(_read_transform_S, store=store, n=n, p=p, s_function=partial(_s2, windows=n * 240))
    with ProcessPoolExecutor() as executor:
        results = list(tqdm.tqdm(executor.map(func, instruments, chunksize=1), total=len(instruments)))
    f = pd.concat(results).sort_index()
    f.name = 'smart_money_v2_s2_{}_{}'.format(n, p)
    return f


def smart_money_v2_s3(store: Arctic, n=10, p=0.2):
    """
    步骤 1 对选定股票，回溯取其过去 10 个交易日的分钟行情数据；
    步骤 2 构造指标St = |Rt|/√Vt，其中𝑅𝑡为第 t 分钟涨跌幅，𝑉𝑡为第 t 分钟成交量；
    步骤 3 将分钟数据按照指标St从大到小进行排序，取成交量累积占比前 20%的分钟，视为聪明钱交易；
    步骤 4 计算聪明钱交易的成交量加权平均价VWAPsmart；
    步骤 5 计算所有交易的成交量加权平均价VWAPall；
    步骤 6 聪明钱因子Q = VWAPsmart/VWAPall。
    Parameters
    ----------
    store
    n
    p

    Returns
    -------

    """
    lib = store['1m']
    instruments = lib.list_symbols()
    func = partial(_read_transform_S, lib=lib, n=n, p=p, s_function=_s3)
    with ProcessPoolExecutor() as executor:
        results = list(tqdm.tqdm(executor.map(func, instruments, chunksize=10), total=len(instruments)))
    f = pd.concat(results).sort_index()
    f.name = 'smart_money_v2_s3_{}_{}'.format(n, p)
    return f

