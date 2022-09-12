import pandas as pd
from talib import abstract

from data_management.dataIO import market_data


def rsrs():
    pass


def keltner_channels(daily_market: pd.DataFrame):
    pass


def supertrend(daily_market: pd.DataFrame):
    pass


def dualthrust(daily_market: pd.DataFrame, n, ):
    hh = daily_market['adj_high'].groupby(level=1).rolling(n).max()
    ll = daily_market['adj_low'].groupby(level=1).rolling(n).min()


def double_ma(daily_market: pd.DataFrame, short: int, long: int):
    indicator = abstract.Function('MA')
    ma_short = daily_market.groupby(level=1).apply(
        lambda x: indicator(x, timeperiod=short, price='adj_close')).droplevel(0).sort_index()
    ma_long = daily_market.groupby(level=1).apply(lambda x: indicator(x, timeperiod=long, price='adj_close')) \
        .droplevel(0).sort_index()
    f = (ma_short >= ma_long).astype(int)
    f.name = 'double_ma_{}_{}'.format(short, long)
    return f


def triple_moving_average(daily_market: pd.DataFrame, short: int, mid: int, long: int):
    indicator = abstract.Function('MA')
    ma_short = daily_market.groupby(level=1).apply(
        lambda x: indicator(x, timeperiod=short, price='adj_close')).droplevel(0).sort_index()
    ma_mid = daily_market.groupby(level=1).apply(
        lambda x: indicator(x, timeperiod=mid, price='adj_close')).droplevel(0).sort_index()
    ma_long = daily_market.groupby(level=1).apply(lambda x: indicator(x, timeperiod=long, price='adj_close')) \
        .droplevel(0).sort_index()
    f = (daily_market['adj_close'] >= ma_short).astype(int) + \
        (ma_short >= ma_mid).astype(int) + (ma_mid >= ma_long).astype(int)
    f = f / 3
    f.name = 'triple_moving_average_{}_{}_{}'.format(short, mid, long)
    return f


def cci_strategy():
    pass


if __name__ == '__main__':
    start_date = '2013-01-01'
    data_config_path = '../../cfg/data_input.ini'
    data = market_data.get_bars(
        cols=('open', 'high', 'low', 'close', 'volume', 'money', 'factor'),
        adjust=True, eod_time_adjust=False, add_limit=False, start_date=start_date,
        config_path=data_config_path)
    # double_ma(data, 5, 10)
    triple_moving_average(data, 5, 10, 20)
