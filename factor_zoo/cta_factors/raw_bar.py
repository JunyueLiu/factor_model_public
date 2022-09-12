import pandas as pd

from data_management.dataIO.binance import get_um_bars, Freq


def norm_ohlc(df: pd.DataFrame):
    df['pre_close'] = df.groupby(level=1)['close'].shift(1)
    for c in ['open', 'high', 'low', 'close']:
        df['norm_{}'.format(c)] = df[c] / df['pre_close'] - 1
    return df[['norm_open', 'norm_high', 'norm_low', 'norm_close']]


if __name__ == '__main__':
    data = get_um_bars(['BTCUSDT', 'ETHUSDT'], '2021-01-01', freq=Freq.h4)
    # norm_ohlc(data)
    df = norm_ohlc(data)
