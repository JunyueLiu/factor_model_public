import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd
from tqdm import tqdm

from arctic import Arctic
from data_management.dataIO.binance import get_um_bars, Freq


def _read_cal_apl20(i: str, store: Arctic) -> pd.Series:
    df = store['5m'].read(i).data

    last_20_amount = df[df.index.time > datetime.time(14, 40)]['money']
    last_20_amount = last_20_amount.resample('D').sum()
    daily_amount = df['money'].resample('D').sum()
    f = (last_20_amount / daily_amount).dropna()
    f.name = i
    return f


def apl(store: Arctic, n=20) -> pd.Series:
    """
    高频视角下成交额蕴藏的 Alpha：市场微观结构剖析之七
    Parameters
    ----------
    store
    n

    Returns
    -------

    """
    lib = store['5m']
    instruments = lib.list_symbols()
    func = partial(_read_cal_apl20, store=store)
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(func, instruments, chunksize=1),
                            total=len(instruments)))

    f = pd.concat(results, axis=1)
    f = f.rolling(n).mean().stack().sort_index()
    f.index.names = ['date', 'code']
    f.name = 'apl_{}'.format(n)
    return f


def inflow_volume(df: pd.DataFrame) -> pd.Series:
    f = df['taker_buy_quote_asset_volume'] / df['quote_asset_volume']
    f.name = 'inflow_volume'
    return f


def inflow_volume_mean(df: pd.DataFrame, *, n: int = 20) -> pd.Series:
    f = inflow_volume(df)
    f = f.groupby(level=1).rolling(n).mean().droplevel(0).sort_index()
    f.name = 'inflow_volume_mean_{}'.format(n)
    return f


if __name__ == '__main__':
    # _read_cal_apl20('000001.SZ', Arctic('localhost'))
    # apl(Arctic('localhost'), 20)
    data = get_um_bars(freq=Freq.h4)
