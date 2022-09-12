from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd
from tqdm import tqdm

from arctic import Arctic


def _read_cal_dma_ret(i: str, store: Arctic, lib_name, short_period, long_period) -> pd.Series:
    df = store[lib_name].read(i).data
    adj_close = df['close'] * df['factor']
    ma_short = adj_close.rolling(short_period).mean()
    ma_long = adj_close.rolling(long_period).mean()
    ret = adj_close.pct_change()
    masked = (ma_short >= ma_long).shift(1)
    cta_ret = ret * masked
    cta_ret.name = i
    cta_ret = cta_ret.resample('D').sum()
    return cta_ret


def forward_double_ma_ret(store: Arctic, lib_name: str, windows_day: int, short_period, long_period):
    lib = store[lib_name]
    instruments = lib.list_symbols()
    func = partial(_read_cal_dma_ret, store=store, lib_name=lib_name, short_period=short_period,
                   long_period=long_period)
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(func, instruments, chunksize=1),
                            total=len(instruments)))
    label = pd.concat(results, axis=1)
    label = label[~(label == 0).all(axis=1)]
    label = label.rolling(windows_day).sum().shift(-windows_day)
    label_end = pd.Series(label.index, index=label.index, name='label_end').shift(-windows_day)
    label = label.stack().sort_index()
    label_end = label.to_frame().join(label_end)['label_end']
    label.name = 'forward_double_ma_ret_{}_D_{}_{}_{}'.format(windows_day, lib_name, short_period, long_period)
    label_end.name = '{}_label_end'.format(label.name)
    return label, label_end


if __name__ == '__main__':
    # _read_cal_dma_ret('000001.SZ', Arctic('localhost'), '15m', 8, 16)
    f, label_end = forward_double_ma_ret(Arctic('localhost'), '15m', 5, 8, 15)
