import numpy as np
import pandas as pd

from factor_testing.utils import get_end_of_month_trading_date
from factor_zoo.utils import combine_market_with_fundamental, fundamental_preprocess, load_pickle, \
    date_to_last_trading_date_of_month, get_last_trading_of_month_dict, get_next_trading_date_dict


def market_cap(market, capital_change, trading_date) -> pd.Series:
    cc = fundamental_preprocess(capital_change, trading_date, ['share_total'], 'change_date')
    merged = combine_market_with_fundamental(market, cc)
    market['capital'] = market['close'] * merged['share_total'] * 10000
    factor = market['capital']
    factor.name = 'market_cap'
    return factor


def market_cap_2(daily_basic: pd.DataFrame) -> pd.Series:
    """
    :param daily_basic:
    :return:
    """
    mrk_cap = daily_basic['total_mv'] * 10000  # type: pd.Series
    mrk_cap = mrk_cap.unstack().fillna(method='ffill').stack().sort_index()
    mrk_cap.name = 'market_cap_2'
    return mrk_cap


def float_market_cap(market, capital_change, trading_date) -> pd.Series:
    cc = fundamental_preprocess(capital_change, trading_date, ['share_trade_total'], 'change_date')
    merged = combine_market_with_fundamental(market, cc)
    market['capital'] = market['close'] * merged['share_trade_total'] * 10000
    factor = market['capital']
    factor.name = 'float_market_cap'
    return factor


def float_market_cap_2(daily_basic: pd.DataFrame) -> pd.Series:
    """

    :param daily_basic:
    :return:
    """
    mrk_cap = daily_basic['circ_mv'] * 10000  # type: pd.Series
    mrk_cap = mrk_cap.unstack().fillna(method='ffill').stack().sort_index()
    mrk_cap.name = 'float_market_cap_2'
    return mrk_cap


def size(market, capital_change, trading_date) -> pd.Series:
    cap = market_cap(market, capital_change, trading_date)
    return pd.Series(np.log(cap), index=cap.index, name='size')


def size_2(daily_basic: pd.DataFrame) -> pd.Series:
    cap = market_cap_2(daily_basic)
    return pd.Series(np.log(cap), index=cap.index, name='size_2')


def float_size(market, capital_change, trading_date) -> pd.Series:
    cap = float_market_cap(market, capital_change, trading_date)
    return pd.Series(np.log(cap), index=cap.index, name='float_size')


def float_size_2(daily_basic: pd.DataFrame) -> pd.Series:
    cap = float_market_cap_2(daily_basic)
    return pd.Series(np.log(cap), index=cap.index, name='float_size_2')


def SMB():
    pass


def avg_market_cap(market, capital_change, trading_date, resample='M') -> pd.Series:
    """

    :param market:
    :param capital_change:
    :param trading_date:
    :param resample:
    :return:
    """
    factor = market_cap(market, capital_change, trading_date).to_frame().reset_index()
    factor = factor.groupby(by=['instCode']).resample(resample, on='date').agg({'market_cap': 'mean',
                                                                                'date': 'last'}).droplevel(1)
    next_trading_date = get_next_trading_date_dict(trading_date)
    last_trading_of_month = get_last_trading_of_month_dict(trading_date)
    factor = factor.dropna(subset=['date'])
    factor['date'] = factor['date'].apply(
        lambda x: date_to_last_trading_date_of_month(x, trading_date, next_trading_date, last_trading_of_month))
    factor = factor.set_index('date', append=True).swaplevel(0, 1).sort_index()['market_cap']
    factor.name = 'avg_market_cap'
    return factor

