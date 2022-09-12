import os
import pandas as pd

from factor_zoo.utils import get_last_trading_of_month_dict


def load_single_category_factors(fold_path: str, trading_date=None) -> pd.DataFrame:
    """

    :param fold_path:
    :return:
    """
    ff = os.listdir(fold_path)
    data = pd.DataFrame()
    for f in ff:
        if f.endswith('parquet'):
            df = pd.read_parquet(os.path.join(fold_path, f))
            if trading_date is not None:
                df = df.sort_index()
                df = df.loc[trading_date]
            print('load successfully', f)
            data = pd.concat([data, df], axis=1)
    return data


def load_multi_category_factors(paths: list, trading_date=None) -> pd.DataFrame:
    for p in paths:
        if not os.path.exists(p):
            raise IOError('Not exist {}'.format(p))

    data = pd.DataFrame()
    for path in paths:
        if os.path.isdir(path):
            df = load_single_category_factors(path, trading_date)
        elif path.endswith('parquet'):
            df = pd.read_parquet(path)
            if trading_date is not None:
                df = df.loc[trading_date]
            print('load successfully', path)
        else:
            raise ValueError('cannot read {}'.format(path))
        # to deal with duplicates
        df = df[~df.index.duplicated(keep='last')]
        df.index.names = ['date', 'code']
        data = pd.concat([data, df], axis=1)

    return data

def get_end_of_month_trading_date(trading_list: list):
    trading_list.sort()
    end_month = list(set(get_last_trading_of_month_dict(trading_list).values()))
    end_month.sort()
    trading_dates = pd.to_datetime(end_month)
    return trading_dates


def get_forward_returns_columns(columns):
    return [c for c in columns if c.startswith('forward_')]


def calculate_forward_returns(data: pd.DataFrame, periods: list, price_key='close') -> pd.DataFrame:
    if price_key == 'ret':
        if type(data.index) == pd.MultiIndex:
            data['price'] = (data['ret'] + 1).groupby(level=1).cumprod()
        else:
            data['price'] = (data['ret'] + 1).cumprod()
        price_key = 'price'

    returns = pd.DataFrame(index=data.index)
    for period in periods:
        if type(data.index) == pd.MultiIndex:
            def multi_index_forward_returns(df: pd.DataFrame):
                return df[price_key].pct_change(periods=period).shift(-period)

            tmp = data.groupby(level=1).apply(multi_index_forward_returns).droplevel(0)
            returns['forward_' + str(period) + '_period_return'] = tmp
        else:
            returns['forward_' + str(period) + '_period_return'] = data[price_key].pct_change(periods=period).shift(
                -period)
    returns.index.names = data.index.names
    return returns


def forward_returns_to_category(factor_data, bins=5, grouper=None):
    """

    :param factor_data:
    :param bin:
    :return:
    """

    if not grouper:
        grouper = factor_data.index.get_level_values(0)

    def quantile_calc(x, _bins):
        quantile_factor = pd.qcut(x, _bins, labels=False, duplicates='drop') + 1
        return quantile_factor

    cols = get_forward_returns_columns(factor_data.columns)
    forward_ret_quantile = factor_data.groupby(grouper)[cols] \
        .transform(quantile_calc, bins)  # type: pd.DataFrame
    forward_ret_quantile = forward_ret_quantile.rename(columns={k: k.replace('return', 'group') for k in cols})
    forward_ret_quantile.index.names = factor_data.index.names
    return forward_ret_quantile


def calculate_forward_returns_group(data: pd.DataFrame, periods: list, bins=5, price_key='close') -> pd.DataFrame:
    """

    :param data:
    :param periods:
    :param bins:
    :param price_key:
    :return:
    """
    if price_key == 'ret':
        if type(data.index) == pd.MultiIndex:
            data['price'] = (data['ret'] + 1).groupby(level=1).cumprod()
        else:
            data['price'] = (data['ret'] + 1).cumprod()
        price_key = 'price'

    returns = pd.DataFrame(index=data.index)
    for period in periods:
        if type(data.index) == pd.MultiIndex:
            def multi_index_forward_returns(df: pd.DataFrame):
                x = df[price_key].pct_change(periods=period).shift(-period)
                quantile_factor = pd.qcut(x, bins, labels=False, duplicates='drop') + 1
                return quantile_factor

            tmp = data.groupby(level=1).apply(multi_index_forward_returns).droplevel(0)
            returns['forward_' + str(period) + '_period_group'] = tmp
        else:
            tmp = data[price_key].pct_change(periods=period).shift(
                -period)
            returns['forward_' + str(period) + '_period_group'] = pd.qcut(tmp, bins, labels=False,
                                                                          duplicates='drop') + 1
    returns.index.names = data.index.names
    return returns


def calculate_demarket_forward_returns(data: pd.DataFrame, periods: list, price_key='close',
                                       market_price_key='benchmark_close',
                                       benchmark_name='benchmark_change_pct') -> pd.DataFrame:
    """

    :param data:
    :param periods:
    :param price_key:
    :param market_price_key:
    :param benchmark_name:
    :return:
    """
    data = data.copy()
    if price_key == 'ret':
        data['price'] = (data['ret'] + 1).groupby(level=1).cumprod()
        price_key = 'price'

    if 'close' in market_price_key:
        raise NotImplementedError('Not support right now. Since stock industry could change')

    returns = pd.DataFrame(index=data.index)
    for period in periods:
        def multi_index_forward_returns(df: pd.DataFrame):
            stock_fret = df[price_key].pct_change(periods=period).shift(-period)
            indus_fret = df[market_price_key].shift(-period)
            return stock_fret - indus_fret

        tmp = data.groupby(level=1).apply(multi_index_forward_returns).droplevel(0)
        returns['forward_{}_period_de_{}_return'.format(period, benchmark_name)] = tmp
    returns.index.names = data.index.names
    return returns


def demean_forward_returns(factor_data, grouper=None):
    """
    Convert forward returns to returns relative to mean
    period wise all-universe or group returns.
    group-wise normalization incorporates the assumption of a
    group neutral portfolio constraint and thus allows allows the
    factor to be evaluated across groups.

    For example, if AAPL 5 period return is 0.1% and mean 5 period
    return for the Technology stocks in our universe was 0.5% in the
    same period, the group adjusted 5 period return for AAPL in this
    period is -0.4%.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        Forward returns indexed by date and asset.
        Separate column for each forward return window.
    grouper : list
        If True, demean according to group.

    Returns
    -------
    adjusted_forward_returns : pd.DataFrame - MultiIndex
        DataFrame of the same format as the input, but with each
        security's returns normalized by group.
    """

    factor_data = factor_data.copy()

    if not grouper:
        grouper = factor_data.index.get_level_values(0)

    cols = get_forward_returns_columns(factor_data.columns)
    factor_data[cols] = factor_data.groupby(grouper)[cols] \
        .transform(lambda x: x - x.mean())

    return factor_data


