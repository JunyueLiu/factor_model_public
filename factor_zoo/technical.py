import bottleneck as bn
import pandas as pd
from talib import abstract
from tqdm import tqdm

from factor_zoo.factor_transform import sine_transform

tqdm.pandas()


def atr(daily_market: pd.DataFrame, timeperiod=14) -> pd.Series:
    """
    http://www.tadoc.org/indicator/ATR.htm
    :param daily_market:
    :param timeperiod:
    :return:
    """
    prices = ['adj_high', 'adj_low', 'adj_close']
    indicator = abstract.Function('ATR')
    factor = daily_market.groupby(level=1). \
        progress_apply(lambda x: indicator(x, timeperiod=timeperiod, prices=prices)).droplevel(0).sort_index()
    factor.name = 'atr_{}'.format(timeperiod)
    return factor


def natr(daily_market: pd.DataFrame, timeperiod=14) -> pd.Series:
    """
    :param daily_market:
    :param timeperiod:
    :return:
    """
    prices = ['adj_high', 'adj_low', 'adj_close']
    indicator = abstract.Function('NATR')
    factor = daily_market.groupby(level=1). \
        progress_apply(lambda x: indicator(x, timeperiod=timeperiod, prices=prices)).droplevel(0).sort_index()
    factor.name = 'natr_{}'.format(timeperiod)
    return factor


def obv(daily_market: pd.DataFrame) -> pd.Series:
    """
    http://www.tadoc.org/indicator/OBV.htm
    :param daily_market:
    :return:
    """
    prices = ['adj_close', 'money']
    indicator = abstract.Function('OBV')
    factor = daily_market.groupby(level=1). \
        progress_apply(lambda x: indicator(x, x['money'], price=prices[0], prices=[prices[1]])).droplevel(
        0).sort_index()
    factor.name = 'obv'
    return factor


def sar(daily_market: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
    prices = ['adj_high', 'adj_low']
    indicator = abstract.Function('SAR')
    factor = daily_market.groupby(level=1).progress_apply(
        lambda x: indicator(x, acceleration=acceleration, maximum=maximum, prices=prices)).droplevel(0).sort_index()
    factor.name = 'sar_{}_{}'.format(acceleration, maximum)
    return factor


def ma_diff_std_multiplier_sin(daily_market: pd.DataFrame, period=20):
    close = daily_market['adj_close'].unstack()
    ma = bn.move_mean(close, period, axis=0)
    std = bn.move_std(close, period, axis=0)
    factor = (close - ma) / std
    factor = factor.clip(-3, 3).stack()
    factor = sine_transform(factor).round(4)
    factor.name = 'ma_diff_std_multiplier_sin_{}'.format(period)
    return factor
