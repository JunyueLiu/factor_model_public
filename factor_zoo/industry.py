from functools import partial

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from data_management.pandas_utils.parallel import time_series_parallel_apply
from factor_zoo.utils import rolling_regression_beta, rolling_regression_alpha


def industry_dummy(industry_dict) -> pd.DataFrame:
    """

    :param industry_dict:
    :return:
    """
    series = pd.DataFrame(industry_dict).T.stack().explode().swaplevel(0, 1)
    series.name = 'code'
    series.index.names = ["date", 'industry_code']
    df = series.reset_index()
    df = df.dropna()  # type: pd.DataFrame
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date', 'code']).sort_index()
    # todo could one code is to match two industry
    # df = df[~df.index.duplicated(keep='last')]
    # df = df.sort_index()
    onehot_model = OneHotEncoder()
    data = onehot_model.fit_transform(df).toarray()
    industry_df = pd.DataFrame(data, index=df.index, columns=onehot_model.categories_[0])
    return industry_df


def industry_category(industry_dict: dict, industry_name=None) -> pd.Series:
    """

    :param industry_dict:
    :return:
    """
    series = pd.DataFrame(industry_dict).T.stack().explode().swaplevel(0, 1)
    series.name = 'code'
    series.index.names = ["date", 'industry_code']
    df = series.reset_index()
    df = df.dropna()  # type: pd.DataFrame
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date', 'code']).sort_index()
    series = df['industry_code']
    series = series[~series.index.duplicated(keep='last')]
    if industry_name:
        series.name = industry_name

    return series


def industry_rolling_beta(daily_market: pd.DataFrame,
                          industry_daily_market: pd.DataFrame,
                          industry_dict: dict,
                          window
                          ):
    industry_ret = industry_daily_market['change_pct'] / 100
    industry_ret.index.names = ['date', 'sw_l1']
    stock_ret = daily_market['adj_close'].unstack().pct_change().stack()
    category = industry_category(industry_dict, 'sw_l1')
    merged = stock_ret.to_frame('ret').join(category)
    merged = merged.set_index('sw_l1', append=True).join(industry_ret.to_frame('ind_ret'))
    merged = merged.sort_index().droplevel(1)
    merged = merged[~merged.index.duplicated()]
    roll_reg = partial(rolling_regression_beta, y_col='ret', X_cols=['ind_ret'], N=window)
    factor = time_series_parallel_apply(merged, roll_reg)
    factor = factor.sort_index()
    factor.name = 'industry_rolling_beta_sw_l1_{}'.format(window)
    return factor


def industry_rolling_alpha(daily_market: pd.DataFrame,
                           industry_daily_market: pd.DataFrame,
                           industry_dict: dict,
                           window
                           ):
    industry_ret = industry_daily_market['change_pct'] / 100
    industry_ret.index.names = ['date', 'sw_l1']
    stock_ret = daily_market['adj_close'].unstack().pct_change().stack()
    category = industry_category(industry_dict, 'sw_l1')
    merged = stock_ret.to_frame('ret').join(category)
    merged = merged.set_index('sw_l1', append=True).join(industry_ret.to_frame('ind_ret'))
    merged = merged.sort_index().droplevel(1)
    merged = merged[~merged.index.duplicated()]
    roll_reg = partial(rolling_regression_alpha, y_col='ret', X_cols=['ind_ret'], N=window)
    factor = time_series_parallel_apply(merged, roll_reg)
    factor = factor.sort_index()
    factor.name = 'industry_rolling_alpha_sw_l1_{}'.format(window)
    return factor


def industry_net_pct_main_avg():
    pass
