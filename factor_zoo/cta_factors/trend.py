import pandas as pd

from data_management.dataIO.binance import get_um_bars, Freq
from technical_analysis.overlap import *


def nsma(df: pd.DataFrame, period: int, price_type: str = 'close') -> pd.Series:
    ind = df.groupby(level=1).apply(lambda x: SMA(x, period=period, price_type=price_type)).droplevel(0).sort_index()
    f = df[price_type] / ind
    f.name = 'nsma_{}_{}'.format(period, price_type)
    return f


def nema(df: pd.DataFrame, period: int, price_type: str = 'close') -> pd.Series:
    ind = df.groupby(level=1).apply(lambda x: EMA(x, period=period, price_type=price_type)).droplevel(0).sort_index()
    f = df[price_type] / ind
    f.name = 'nema_{}_{}'.format(period, price_type)
    return f


def ndema(df: pd.DataFrame, period: int, price_type: str = 'close') -> pd.Series:
    ind = df.groupby(level=1).apply(lambda x: DEMA(x, period=period, price_type=price_type)).droplevel(0).sort_index()
    f = df[price_type] / ind
    f.name = 'ndema_{}_{}'.format(period, price_type)
    return f


def nht_trendline(df: pd.DataFrame, price_type: str = 'close') -> pd.Series:
    ind = df.groupby(level=1).apply(lambda x: HT_TRENDLINE(x, price_type=price_type)).droplevel(0).sort_index()
    f = df[price_type] / ind
    f.name = 'nht_trendline_{}'.format(price_type)
    return f


def nkama(df: pd.DataFrame, period: int, price_type: str = 'close') -> pd.Series:
    ind = df.groupby(level=1).apply(lambda x: KAMA(x, period=period, price_type=price_type)).droplevel(0).sort_index()
    f = df[price_type] / ind
    f.name = 'nkama_{}_{}'.format(period, price_type)
    return f


def nsar(df, acceleration: float = 0.02, maximum: float = 0.2, prices=('high', 'low')):
    ind = df.groupby(level=1).apply(
        lambda x: SAR(x, acceleration=acceleration, maximum=maximum, prices=prices)).droplevel(0).sort_index()
    f = df['close'] / ind
    f.name = 'nsar_{}_{}_{}'.format(acceleration, maximum, prices)
    return f


def nt3(df: pd.DataFrame, period: int, vfactor, price_type: str = 'close') -> pd.Series:
    ind = df.groupby(level=1).apply(lambda x: T3(x, period=period, vfactor=vfactor, price_type=price_type)) \
        .droplevel(0).sort_index()
    f = df[price_type] / ind
    f.name = 'nt3_{}_{}'.format(period, price_type)
    return f


def ntema(df: pd.DataFrame, period: int, price_type: str = 'close') -> pd.Series:
    ind = df.groupby(level=1).apply(lambda x: TEMA(x, period=period, price_type=price_type)).droplevel(0).sort_index()
    f = df[price_type] / ind
    f.name = 'ntema_{}_{}'.format(period, price_type)
    return f


def nwma(df: pd.DataFrame, period: int, price_type: str = 'close') -> pd.Series:
    ind = df.groupby(level=1).apply(lambda x: WMA(x, period=period, price_type=price_type)).droplevel(0).sort_index()
    f = df[price_type] / ind
    f.name = 'nwma_{}_{}'.format(period, price_type)
    return f


if __name__ == '__main__':
    data = get_um_bars(['BTCUSDT', 'ETHUSDT'], '2021-01-01', freq=Freq.h4)
    nsma(data, 5, 'close')
