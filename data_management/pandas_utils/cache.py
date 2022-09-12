from typing import Callable, List

import pandas as pd
from tqdm import tqdm

from data_management.cache_janitor.cache import tmp_cache


# @tmp_cache
def panel_df_join(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    data = df1.unstack().join(df2.unstack())
    data = data.stack()
    return data


def panel_df_concat(dfs: List[pd.DataFrame]):
    # for df in dfs:
    dfs = [df.unstack() for df in dfs]
    data = pd.concat(dfs, axis=1)
    data = data.stack()
    return data


def df_merge_asof(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    res = []
    for ticker, data in df1.groupby(level=1):
        try:
            data2 = df2.loc[(slice(None), ticker), :]
            merged = pd.merge_asof(data2.droplevel(1), data.droplevel(1), left_index=True, right_index=True)
            merged['code'] = ticker
            merged = merged.set_index('code', append=True)
            res.append(merged)
        except KeyError:
            pass
    if len(res) == 0:
        return pd.DataFrame()
    data = pd.concat(res)
    return data


@tmp_cache
def series_join(series1: pd.Series, series2: pd.Series):
    if not series1.name:
        if not series2.name:
            return series1.to_frame('series1').join(series2.to_frame('series2'))
        else:
            return series1.to_frame('series1').join(series2)
    else:
        if not series2.name:
            return series1.to_frame().join(series2.to_frame('series2'))
        else:
            return series1.to_frame().join(series2)


@tmp_cache
def series_cross_sectional_apply(series: pd.Series, func: Callable):
    tqdm.pandas(desc='Cross Sectional Apply using {}'.format(func.__func__.__name__))
    return series.groupby(level=0).progress_apply(func)


@tmp_cache
def series_time_series_apply(series: pd.Series, func: Callable):
    tqdm.pandas(desc='Time Series Apply using {}'.format(func.__func__.__name__))
    return series.groupby(level=1).progress_apply(func)


@tmp_cache
def df_cross_sectional_apply(df: pd.DataFrame, func: Callable):
    try:
        tqdm.pandas(desc='Cross Sectional Apply using {}'.format(func.__func__.__name__))
    except:
        tqdm.pandas(desc='Cross Sectional Apply')
    return df.groupby(level=0).progress_apply(func)


@tmp_cache
def df_time_series_apply(df: pd.DataFrame, func: Callable):
    try:
        tqdm.pandas(desc='Time Series Apply using {}'.format(func.__func__.__name__))
    except:
        tqdm.pandas(desc='Time Series Apply')
    return df.groupby(level=1).progress_apply(func)


@tmp_cache
def df_grouper_apply(df: pd.DataFrame, grouper, func: Callable):
    try:
        tqdm.pandas(desc='Cross Sectional grouper Apply using {}'.format(func.__func__.__name__))
    except:
        tqdm.pandas(desc='Cross Sectional grouper Apply')
    return df.groupby(by=grouper).progress_apply(func)


# @tmp_cache
def cross_sectional_resample(series: pd.Series, offset):
    return series.groupby(level=1) \
        .resample(offset, level=0, label='right', closed='right') \
        .last().swaplevel(0, 1).sort_index()
