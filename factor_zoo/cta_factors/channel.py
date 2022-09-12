import pandas as pd

from data_management.dataIO.binance import get_um_bars, Freq
from technical_analysis.customization import DUALTHRUST
from technical_analysis.overlap import BBANDS, EMA
from technical_analysis.volatility import ATR


def keltner_channels(bars: pd.DataFrame, *, period=14, n=2):
    atr = bars.groupby(level=1).apply(lambda x: ATR(x, period=period)).droplevel(0).sort_index()
    atr.name = 'atr'
    ema = bars.groupby(level=1).apply(lambda x: EMA(x, period=period)).droplevel(0).sort_index()
    ema.name = 'ema'
    df = pd.concat([atr, ema, bars['close']], axis=1)
    df['keltner_upper'] = (df['ema'] + n * df['atr']) / df['close']
    df['keltner_lower'] = (df['ema'] - n * df['atr']) / df['close']
    return df[['keltner_upper', 'keltner_lower']]


def bolling_band(bars: pd.DataFrame, *, period=14, n=2) -> pd.DataFrame:
    n = float(n)
    bands = bars.groupby(level=1).apply(lambda x: BBANDS(x, period=period, nbdevup=n, nbdevdn=n)).sort_index()
    for c in bands:
        bands[c] = bands[c] / bars['close']
    return bands.rename(columns={c: 'bolling_band_{}'.format(c) for c in bands.columns})


def dual_thrust(bars: pd.DataFrame, *, period=14, k1=0.1, k2=0.1) -> pd.DataFrame:
    bands = bars.groupby(level=1).apply(lambda x: DUALTHRUST(x, period, k1, k2))
    for c in bands:
        bands[c] = bands[c] / bars['close']
    return bands.rename(columns={c: 'dual_thrust_{}'.format(c) for c in bands.columns})


if __name__ == '__main__':
    data = get_um_bars(['BTCUSDT', 'ETHUSDT'], '2021-01-01', freq=Freq.h4)
    bolling_band(data, period=24, n=2)
    # dual_thrust(data)
    # keltner_channels(data, period=14, n=2)
