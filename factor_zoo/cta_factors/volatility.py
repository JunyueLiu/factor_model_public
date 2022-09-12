import pandas as pd

from data_management.dataIO.binance import get_um_bars, Freq
from technical_analysis.volatility import *


def vol(df: pd.DataFrame, period, price_type: str = 'close'):
    ret = df[price_type].groupby(level=1).pct_change()
    f = ret.groupby(level=1).rolling(period).std().droplevel(0).sort_index()
    f.name = 'vol_{}_{}'.format(period, price_type)
    return f


def natr(df: pd.DataFrame, period, prices=('high', 'low', 'close')):
    f = df.groupby(level=1).apply(lambda x: NATR(x, period=period, prices=list(prices))).droplevel(0).sort_index()
    f.name = 'natr_{}_{}'.format(period, '_'.join(prices))
    return f


if __name__ == '__main__':
    data = get_um_bars(['BTCUSDT', 'ETHUSDT'], '2021-01-01', freq=Freq.h4)
    # f = natr(data, 14)
    vol(data, 14)
