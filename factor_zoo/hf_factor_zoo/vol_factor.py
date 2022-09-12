import pandas as pd

from arctic import Arctic
from factor_zoo.hf_factor_zoo.agg_factor import row_corr, row_spearman_corr
from factor_zoo.hf_factor_zoo.intraday_operator import get_min_data_df, intraday_standard_money, intraday_ret
from technical_analysis.momentum import RSI


def intraday_ret_volume_corr(df: pd.DataFrame, ):
    min_df = get_min_data_df(df).set_index('code', append=True).unstack(level=1)
    ret = intraday_ret(min_df['close'].values)[:, 1:]
    standard_money = intraday_standard_money(min_df).values[:, 1:]
    cor = row_corr(ret, standard_money)
    f = pd.Series(cor, index=min_df.index, name='intraday_ret_volume_corr')
    return f


def intraday_ret_volume_spearman_corr(df: pd.DataFrame, ):
    min_df = get_min_data_df(df).set_index('code', append=True).unstack(level=1)
    ret = intraday_ret(min_df['close'].values)[:, 1:]
    standard_money = intraday_standard_money(min_df).values[:, 1:]
    cor = row_spearman_corr(ret, standard_money)
    f = pd.Series(cor, index=min_df.index, name='intraday_ret_volume_spearman_corr')
    return f


def intraday_rsi_volume_spearman_corr(df: pd.DataFrame, n: int = 14):
    min_df = get_min_data_df(df).set_index('code', append=True).unstack(level=1)
    intraday_rsi = min_df['close'].apply(lambda x: RSI(x, n), axis=1)
    intraday_rsi = pd.DataFrame(intraday_rsi.to_list(), columns=min_df['close'].columns, index=min_df.index)
    intraday_rsi = intraday_rsi.dropna(axis=1, how='all')
    standard_money = intraday_standard_money(min_df)
    standard_money = standard_money[intraday_rsi.columns]
    cor = row_spearman_corr(intraday_rsi.values, standard_money.values)
    f = pd.Series(cor, index=min_df.index, name='intraday_rsi_volume_spearman_corr_{}'.format(n))
    return f





if __name__ == '__main__':
    store = Arctic('localhost')
    df = store['1m'].read('000001.SZ').data
    # min_df = get_min_data_df(df).set_index('code', append=True).unstack(level=1)
    # ret = intraday_ret(min_df['close'].values)[:, 1:]
    # standard_money = intraday_standard_money(min_df).values[:, 1:]
    # cor = row_corr(ret, standard_money)
    intraday_rsi_volume_spearman_corr(df, 14)
