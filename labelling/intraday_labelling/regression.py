from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from arctic import Arctic
from data_management.dataIO.component_data import IndustryCategory
from factor_circus.preprocessing import CapIndNeuFactorScaler


def _read_cal_intraday_forward_sharpe(i: str, store: Arctic, freq, m) -> pd.Series:
    df = store[freq].read(i).data
    close = df['close'] * df['factor']
    ret = close.pct_change()
    if len(close) == 1:
        return pd.Series()
    sharpe = ret.rolling(m).mean() / ret.rolling(m).std()
    sharpe = sharpe.shift(-m)
    sharpe = sharpe.resample('D').last().dropna()
    sharpe.name = i
    return sharpe


def intraday_sharpe(store: Arctic, freq: str, windows: int = 5) -> Tuple[pd.Series, pd.Series]:
    lib = store[freq]
    instruments = lib.list_symbols()
    if freq == '15m':
        m = windows * 16
    else:
        raise NotImplementedError

    func = partial(_read_cal_intraday_forward_sharpe, store=store, freq=freq, m=m)
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(func, instruments, chunksize=1),
                            total=len(instruments)))

    label = pd.concat(results, axis=1)
    mean = label.mean(axis=1)
    std = label.std(axis=1)
    label = (label.subtract(mean, axis=0)).div(std, axis=0)
    label_end = pd.Series(label.index, index=label.index, name='label_end').shift(-windows)
    label_end = label_end.fillna(0)
    label = label.stack().sort_index()
    label = label.map(lambda x: '%.4f' % x).astype(float)
    label_end = label.to_frame().join(label_end)['label_end']
    label.name = 'forward_intraday_sharpe_{}_{}_D'.format(freq, windows)
    label_end.name = '{}_label_end'.format(label.name)
    label.index.names = ['date', 'code']
    label_end.index.names = ['date', 'code']
    return label, label_end








def intraday_sharpe_cap_industry_neutralized(store: Arctic, freq: str,
                                             config_path: str,
                                             windows: int = 5) -> Tuple[pd.Series, pd.Series]:
    label, label_end = intraday_sharpe(store, freq, windows)
    neu_label = CapIndNeuFactorScaler(CapIndNeuFactorScaler.Method.WinsorizationStandardScaler,
                                      config_path, IndustryCategory.sw_l1, 0.05
                                      ).fit_transform(label)
    neu_label.name = 'intraday_sharpe_cap_industry_neutralized_{}_{}_D'.format(freq, windows)
    label_end.name = '{}_label_end'.format(neu_label.name)
    return neu_label, label_end


if __name__ == '__main__':
    # intraday_sharpe(Arctic('localhost'), '15m', 5)
    config_path = '../../cfg/data_input.ini'
    start_date = '2013-01-01'
    local_cfg_path = '../cfg/factor_keeper_setting.ini'
    # daily_basic = get_trade(TradeTable.daily_basic, config_path=config_path)
    intraday_sharpe_cap_industry_neutralized(Arctic('localhost'), '15m', config_path, 5)
