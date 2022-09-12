from typing import Tuple

import numpy as np
import pandas as pd
import tqdm
from numba import njit

from data_management.dataIO.binance import get_um_bars, Freq


@njit
def generate_image(ohlcv: np.ndarray,
                   ohlc_height: int = 51,
                   vol_height: int = 12) -> np.ndarray:
    ll = ohlcv[:, 0:4].min()
    hh = ohlcv[:, 0:4].max()

    for c in [0, 1, 2, 3]:  # open, high, low, close
        ohlcv[:, c] = (ohlcv[:, c] - ll) / (hh - ll)
        ohlcv[:, c] = np.rint((1 - ohlcv[:, c]) * (ohlc_height - 1))  # .astype(int)

    # vol
    if ohlcv.shape[-1] > 4:
        ohlcv[:, 4] = ohlcv[:, 4] / ohlcv[:, 4].max()
        ohlcv[:, 4] = np.rint(ohlc_height + 1 + vol_height - (ohlcv[:, 4] * vol_height))  # .astype(int)

    # ma
    if ohlcv.shape[-1] > 5:
        ohlcv[:, 5] = (ohlcv[:, 5] - ll) / (hh - ll)
        ohlcv[:, 5] = np.rint((1 - ohlcv[:, 5]) * (ohlc_height - 1))

    ohlcv = ohlcv.astype(np.int_)
    img = np.zeros((ohlc_height + vol_height + 1, len(ohlcv) * 3))
    for i in range(len(ohlcv)):
        row = ohlcv[i]
        img[row[0], i * 3] = 255
        img[row[1]: row[2] + 1, i * 3 + 1] = 255
        img[row[3], i * 3 + 2] = 255
        if ohlcv.shape[-1] > 4:
            img[row[4]:, i * 3 + 1] = 255
        if ohlcv.shape[-1] > 5:
            if i == 0:
                img[row[5], 1] = 255
            else:
                row_prev = ohlcv[i - 1]
                ma_prev = row_prev[5]
                ma_curr = row[5]
                for j in [1, 2, 3]:
                    ma_temp = ma_prev + (ma_curr - ma_prev) / 3.0 * j
                    img[int(ma_temp), i * 3 - 2 + j] = 255
    return img


def generate_single_asset_img(ohlcv: np.ndarray,
                              idx: np.ndarray,
                              lookback: int,
                              ohlc_height: int = 51,
                              vol_height: int = 12,
                              ) -> Tuple:
    l = list()
    id = list()
    for i in range(lookback, len(ohlcv)):
        img = generate_image(ohlcv[i - lookback:i], ohlc_height, vol_height, )
        l.append(img)
        id.append(idx[i])
    return id, l


def generate_all_img_features(bar: pd.DataFrame, lookback=20,
                              ohlc_height: int = 51,
                              vol_height: int = 12,
                              ohlc: Tuple[str] = ('open', 'high', 'low', 'close'),
                              vol_col: str = 'volume',
                              ma=None
                              ):
    idx = []
    data = []
    codes = bar.index.get_level_values(1).drop_duplicates()
    col = list(ohlc) + [vol_col]
    if ma:
        col += [ma]
    bar = bar[col]

    for c in tqdm.tqdm(codes):
        bar_df = bar.loc[slice(None), c, :]
        bar_array = bar_df.values
        idx_ = bar_df.index.values
        id, imgs = generate_single_asset_img(bar_array, idx_, lookback, ohlc_height, vol_height)
        idx.extend(id)
        data.extend(imgs)
    series = pd.Series(data, pd.MultiIndex.from_tuples(idx, names=['date', 'code']), name='img')
    series = series.sort_index()
    return series


if __name__ == '__main__':
    bar_data = get_um_bars(freq=Freq.h4)
    # img = generate_image(bar_data.iloc[0:20], vol_col='money')
    data = generate_all_img_features(bar_data)
