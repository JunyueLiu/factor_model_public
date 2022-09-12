import pandas as pd
from talib import abstract
from tqdm import tqdm

from data_management.dataIO.binance import get_um_bars, Freq

tqdm.pandas()


def adxr(daily_market: pd.DataFrame, timeperiod: int = 14) -> pd.Series:
    """
    norm
    1-100
    http://www.tadoc.org/indicator/ADX.htm
    :param daily_market:
    :param timeperiod:
    :return:
    """
    prices = ['adj_high', 'adj_low', 'adj_close']
    indicator = abstract.Function('ADXR')
    factor = daily_market.groupby(level=1) \
        .progress_apply(lambda x: indicator(x, timeperiod=timeperiod, prices=prices)).droplevel(0).sort_index()
    factor = factor / 100
    factor.name = 'adxr_{}'.format(timeperiod)
    return factor


def adx(daily_market: pd.DataFrame, timeperiod: int = 14) -> pd.Series:
    """
    norm
    1-100
    http://www.tadoc.org/indicator/ADX.htm
    :param daily_market:
    :param timeperiod:
    :return:
    """
    prices = ['adj_high', 'adj_low', 'adj_close']
    indicator = abstract.Function('ADX')
    factor = daily_market.groupby(level=1) \
        .progress_apply(lambda x: indicator(x, timeperiod=timeperiod, prices=prices)).droplevel(0).sort_index()
    factor = factor / 100
    factor.name = 'adx_{}'.format(timeperiod)
    return factor


def apo_norm(daily_market, fastperiod: int = 12, slowperiod: int = 26, price_type='adj_close'):
    """
    APO = Shorter Period EMA – Longer Period EMA
    Parameters
    ----------
    daily_market
    fastperiod
    slowperiod

    Returns
    -------

    """
    indicator = abstract.Function('APO')
    factor = daily_market.groupby(level=1) \
        .progress_apply(
        lambda x: indicator(x, fastperiod=fastperiod, slowperiod=slowperiod, matype=0, price=price_type)).droplevel(
        0).sort_index()
    factor = factor / daily_market['adj_close']
    factor.name = 'apo_norm_{}_{}_{}'.format(price_type, fastperiod, slowperiod)
    return factor


def bop(daily_market: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    (close - open) / (high - low)
    -1 to 1 norm
    http://www.tadoc.org/indicator/BOP.htm
    :param daily_market:
    :return:
    """
    factor = (daily_market['adj_close'] - daily_market['adj_open']) / (
            daily_market['adj_high'] - daily_market['adj_low'] + 0.01)
    factor = factor.groupby(level=1).rolling(window).mean().droplevel(0).sort_index()
    factor.name = 'BOP_{}'.format(window)
    return factor


def cci(daily_market: pd.DataFrame, timeperiod=14) -> pd.Series:
    """
    http://www.tadoc.org/indicator/CCI.htm
    https://school.stockcharts.com/doku.php?id=technical_indicators:commodity_channel_index_cci
    :param daily_market:
    :param timeperiod:
    :return:
    """
    prices = ['adj_high', 'adj_low', 'adj_close']
    indicator = abstract.Function('CCI')
    factor = daily_market.groupby(level=1). \
        progress_apply(lambda x: indicator(x, timeperiod=timeperiod, prices=prices)).droplevel(0).sort_index()
    factor = factor / 100
    factor.name = 'cci_{}'.format(timeperiod)
    return factor


def cmo(daily_market: pd.DataFrame, timeperiod=14, price_type='adj_close') -> pd.Series:
    """
    http://www.tadoc.org/indicator/CMO.htm
    https://www.investopedia.com/terms/c/chandemomentumoscillator.asp
    :param daily_market:
    :param timeperiod:
    :return:

    Parameters
    ----------
    price_type
    """
    indicator = abstract.Function('CMO')
    factor = daily_market.groupby(level=1). \
        progress_apply(lambda x: indicator(x, timeperiod=timeperiod, price=price_type)).droplevel(0).sort_index()
    factor = factor / 100
    factor.name = 'cmo_{}_{}'.format(price_type, timeperiod)
    return factor


def macdhist(daily_market: pd.DataFrame, fastperiod=12, slowperiod=26, signalperiod=9,
             price_type='adj_close') -> pd.Series:
    """

    Parameters
    ----------
    daily_market
    fastperiod
    slowperiod
    signalperiod

    Returns
    -------

    """
    indicator = abstract.Function('MACD')
    factor = daily_market.groupby(level=1).progress_apply(
        lambda x: indicator(x, fastperiod=fastperiod,
                            slowperiod=slowperiod,
                            signalperiod=signalperiod,
                            price=price_type))
    factor = factor['macdhist']
    factor.name = 'macdhist_{}_{}_{}_{}'.format(price_type, fastperiod, slowperiod, signalperiod)
    return factor


def mfi(daily_market: pd.DataFrame, timeperiod=14) -> pd.Series:
    """
    http://www.tadoc.org/indicator/MFI.htm
    https://school.stockcharts.com/doku.php?id=technical_indicators:money_flow_index_mfi
    :param daily_market:
    :param timeperiod:
    :return:
    """
    prices = ['adj_high', 'adj_low', 'adj_close', 'money']

    indicator = abstract.Function('MFI')
    factor = daily_market.groupby(level=1). \
        progress_apply(lambda x: indicator(x, timeperiod=timeperiod, prices=prices)).droplevel(0).sort_index()
    factor = factor / 100
    factor.name = 'mfi_{}'.format(timeperiod)
    return factor


def ppo(daily_market: pd.DataFrame, fastperiod=12, slowperiod=26, matype=0, price_type='adj_close') -> pd.Series:
    """
    http://www.tadoc.org/indicator/PPO.htm
    https://www.investopedia.com/terms/p/ppo.asp

    :param daily_market:
    :param fastperiod:
    :param slowperiod:
    :param matype:
    :return:

    Parameters
    ----------
    """

    indicator = abstract.Function('PPO')
    factor = daily_market.groupby(level=1). \
        progress_apply(lambda x:
                       indicator(x, astperiod=fastperiod, slowperiod=slowperiod, matype=matype, price=price_type)) \
        .droplevel(0).sort_index()
    factor.name = 'ppo_{}_{}_{}_{}'.format(price_type, fastperiod, slowperiod, matype)
    return factor


def rsi(daily_market: pd.DataFrame, period: int = 14, price_type='adj_close'):
    """

    Parameters
    ----------
    daily_market
    period
    price_type

    Returns
    -------

    """
    indicator = abstract.Function('RSI')
    factor = daily_market.groupby(level=1). \
        progress_apply(lambda x: indicator(x, timeperiod=period, price=price_type)).droplevel(0).sort_index()
    factor = factor / 100
    factor.name = 'rsi_{}_{}'.format(price_type, period)
    return factor


def slowk(daily_market: pd.DataFrame,
          fastk_period=5, slowk_period=3,
          slowk_matype=0, slowd_period=3,
          slowd_matype=0) -> pd.Series:
    """
    http://www.tadoc.org/indicator/STOCH.htm
    :param daily_market:
    :param fastk_period:
    :param slowk_period:
    :param slowk_matype:
    :param slowd_period:
    :param slowd_matype:
    :return:
    """
    prices = ['adj_high', 'adj_low', 'adj_close']
    indicator = abstract.Function('STOCH')
    factor = daily_market.groupby(level=1). \
        progress_apply(lambda x: indicator(x, fastk_period=fastk_period,
                                           slowk_period=slowk_period,
                                           slowk_matype=slowk_matype,
                                           slowd_period=slowd_period,
                                           slowd_matype=slowd_matype,
                                           prices=prices
                                           ))

    factor = factor['slowk'] / 100
    factor.name = 'slowk_{}_{}_{}_{}_{}'.format(fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype)
    return factor


def slowd(daily_market: pd.DataFrame, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3,
          slowd_matype=0) -> pd.Series:
    """
    http://www.tadoc.org/indicator/STOCH.htm
    :param daily_market:
    :param fastk_period:
    :param slowk_period:
    :param slowk_matype:
    :param slowd_period:
    :param slowd_matype:
    :return:
    """
    prices = ['adj_high', 'adj_low', 'adj_close']
    indicator = abstract.Function('STOCH')
    factor = daily_market.groupby(level=1). \
        progress_apply(lambda x: indicator(x, fastk_period=fastk_period,
                                           slowk_period=slowk_period,
                                           slowk_matype=slowk_matype,
                                           slowd_period=slowd_period,
                                           slowd_matype=slowd_matype,
                                           prices=prices))

    factor = factor['slowd'] / 100
    factor.name = 'slowd_{}_{}_{}_{}_{}'.format(fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype)
    return factor


def willr(daily_market: pd.DataFrame, timeperiod=14) -> pd.Series:
    """
    :param daily_market:
    :param timeperiod:
    :return:
    """
    prices = ['adj_high', 'adj_low', 'adj_close']
    indicator = abstract.Function('WILLR')
    factor = daily_market.groupby(level=1). \
        progress_apply(lambda x: indicator(x, timeperiod=timeperiod, prices=prices)).droplevel(0).sort_index()
    factor = factor / 100
    factor.name = 'willr_{}'.format(timeperiod)
    return factor


def ad(daily_market: pd.DataFrame) -> pd.Series:
    """
    http://www.tadoc.org/indicator/AD.htm
    :param daily_market:
    :return:
    """
    prices = ['adj_high', 'adj_low', 'adj_close', 'money']
    indicator = abstract.Function('AD')
    factor = daily_market.groupby(level=1). \
        progress_apply(lambda x: indicator(x, prices=prices)).droplevel(0).sort_index()
    factor.name = 'ad'
    return factor


def adosc(daily_market: pd.DataFrame, fastperiod=3, slowperiod=10) -> pd.Series:
    """
    http://www.tadoc.org/indicator/ADOSC.htm
    :param daily_market:
    :return:
    """
    prices = ['adj_high', 'adj_low', 'adj_close', 'money']
    indicator = abstract.Function('ADOSC')
    factor = daily_market.groupby(level=1). \
        progress_apply(lambda x: indicator(x, fastperiod=fastperiod, slowperiod=slowperiod, prices=prices)) \
        .droplevel(0).sort_index()
    factor.name = 'adosc_{}_{}'.format(fastperiod, slowperiod)
    return factor


if __name__ == '__main__':
    data = get_um_bars(['BTCUSDT', 'ETHUSDT'], '2021-01-01', freq=Freq.h4)
    f = adxr(data, 14)
