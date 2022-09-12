import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.regression.rolling import RollingOLS
from tqdm import tqdm

from data_management.dataIO import market_data
from data_management.keeper.ZooKeeper import ZooKeeper
from factor_zoo.technical import sar
from factor_zoo.cta_factors.oscillator import rsi
from technical_analysis.overlap import SMA, EMA, HT_TRENDLINE


def rsrs(daily_market: pd.DataFrame, n: int, m: int):
    tqdm.pandas()

    def single_stock_rsrs(ohlc, N: int, M: int):
        if len(ohlc) < max(N, M):
            return pd.Series(np.nan, ohlc.index)
        endog = ohlc['adj_high'].values
        exog = sm.add_constant(ohlc['adj_low'])
        rols = RollingOLS(endog, exog, window=N)
        rres = rols.fit()
        beta = rres.params['adj_low']
        r = beta.rolling(window=M)
        m = r.mean()
        s = r.std(ddof=0)
        z = (beta - m) / s
        z = pd.Series(norm.cdf(z), index=z.index)
        z = (z - 0.5) / 0.5
        return z

    f = daily_market.groupby(level=1).progress_apply(lambda x: single_stock_rsrs(x, n, m))

    f = f.droplevel(0).sort_index()
    f.name = 'rsrs_{}_{}'.format(n, m)

    return f


def pivot_low(daily_market: pd.DataFrame):
    """

    Parameters
    ----------
    daily_market

    Returns
    -------

    """

    adj_close = daily_market['adj_close']
    adj_open = daily_market['adj_open']
    price = np.minimum(adj_open, adj_close)
    price_1 = price.groupby(level=1).shift(1)
    price_2 = price.groupby(level=1).shift(-1)

    local_min = pd.Series(np.where((price_1 >= price) & (price_2 >= price), price, np.nan), price.index)
    support = adj_close.unstack() / local_min.unstack().fillna(method='ffill') - 1  # type: pd.Series
    support = support.stack().sort_index()
    support.name = 'pivot_low'
    return support


def pivot_high(daily_market: pd.DataFrame):
    """

    Parameters
    ----------
    daily_market

    Returns
    -------

    """
    adj_close = daily_market['adj_close']
    adj_open = daily_market['adj_open']
    price = np.maximum(adj_open, adj_close)
    price_1 = price.groupby(level=1).shift(1)
    price_2 = price.groupby(level=1).shift(-1)

    local_max = pd.Series(np.where((price_1 <= price) & (price_2 <= price), price, np.nan), price.index)
    support = local_max.unstack().fillna(method='ffill') / adj_close.unstack() - 1  # type: pd.Series
    support = support.stack().sort_index()
    support.name = 'pivot_high'
    return support


def nsar(daily_market: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2):
    f = sar(daily_market, acceleration, maximum)
    f = daily_market['adj_close'] / f - 1
    f.name = 'nsar_{}_{}'.format(acceleration, maximum)
    return f


def rsi_overbought_resistance(daily_market: pd.DataFrame, period: int = 14, overbought_threshold: int = 80):
    """

    Parameters
    ----------
    daily_market
    period
    overbought_threshold

    Returns
    -------

    """
    f = rsi(daily_market, period) * 100
    adj_close = daily_market['adj_close']
    adj_open = daily_market['adj_open']
    price = np.maximum(adj_open, adj_close)  # type: pd.Series
    f = pd.Series(np.where(f >= overbought_threshold, price, np.nan), index=price.index)
    f = f.unstack().fillna(method='ffill') / adj_close.unstack() - 1
    f = f.stack().sort_index()
    f.name = 'rsi_overbought_resistance_{}_{}'.format(period, overbought_threshold)
    return f


def rsi_oversold_support(daily_market: pd.DataFrame, period: int = 14, oversold_threshold: int = 20):
    """

    Parameters
    ----------
    daily_market
    period
    oversold_threshold

    Returns
    -------

    """
    f = rsi(daily_market, period) * 100
    adj_close = daily_market['adj_close']
    adj_open = daily_market['adj_open']
    price = np.minimum(adj_open, adj_close)  # type: pd.Series
    f = pd.Series(np.where(f <= oversold_threshold, price, np.nan), index=price.index)
    f = adj_close.unstack() / f.unstack().fillna(method='ffill') - 1
    f = f.stack().sort_index()
    f.name = 'rsi_overbought_support_{}_{}'.format(period, oversold_threshold)
    return f


def ma_support_resistance(daily_market: pd.DataFrame, period: int = 14):
    ma = daily_market.groupby(level=1).apply(lambda x: SMA(x, period, price_type='adj_close')).droplevel(0).sort_index()
    msr = daily_market['adj_close'] / ma - 1
    msr.name = 'ma_support_resistance_{}'.format(period)
    return msr


def ema_support_resistance(daily_market: pd.DataFrame, period: int = 14):
    ma = daily_market.groupby(level=1).apply(lambda x: EMA(x, period, price_type='adj_close')).droplevel(0).sort_index()
    msr = daily_market['adj_close'] / ma - 1
    msr.name = 'ema_support_resistance_{}'.format(period)
    return msr


def ht_support_resistance(daily_market: pd.DataFrame):
    ma = daily_market.groupby(level=1).apply(lambda x: HT_TRENDLINE(x, price_type='adj_close')).droplevel(
        0).sort_index()
    msr = daily_market['adj_close'] / ma - 1
    msr.name = 'ht_support_resistance'
    return msr


def sar_support():
    pass


def supertrend_support_resistance(daily_market: pd.DataFrame):
    pass

