import numpy as np
import pandas as pd
import bottleneck as bn

from data_management.dataIO import market_data
from data_management.dataIO.market_data import Freq
from data_management.dataIO.trade_data import TradeTable, get_trade
from factor_zoo.factor_operator.alpha101_operator import returns


def get_min_data_df(min_data: pd.DataFrame) -> pd.DataFrame:
    """

    Parameters
    ----------
    min_data

    Returns
    -------

    """
    min_data.loc[:, 'time'] = min_data.index.get_level_values(0).time
    min_data.loc[:, 'date'] = min_data.index.get_level_values(0).date
    min_data = min_data.set_index(['date', 'time'])
    return min_data


def intraday_standard_price(min_data: pd.DataFrame) -> pd.DataFrame:
    """

    Parameters
    ----------
    min_data

    Returns
    -------

    """
    high = min_data['high'].max(axis=1)
    low = min_data['low'].min(axis=1)
    close = min_data['close']
    df = close.sub(low, axis=0).div(high - low + 0.0001, axis=0)
    return df


def intraday_standard_money(min_data: pd.DataFrame) -> pd.DataFrame:
    """

    Parameters
    ----------
    min_data

    Returns
    -------

    """
    money = min_data['money']
    money = money.div(money.sum(axis=1), axis=0)
    return money


def intraday_turnover(min_data: pd.DataFrame, daily_basic: pd.DataFrame) -> pd.DataFrame:
    turnover = daily_basic['turnover_rate_f'].droplevel(1)
    standard_money = intraday_standard_money(min_data)
    turnover = standard_money.mul(turnover, axis=0)
    return turnover


def intraday_ret(x: np.ndarray) -> np.ndarray:
    x = x.T
    ret = returns(x)
    return np.nan_to_num(ret.T, posinf=0.0, neginf=0.0)


def select_open_session(x: np.ndarray) -> np.ndarray:
    x = x[:, :30]
    return x


def select_intra_session(x: np.ndarray) -> np.ndarray:
    x = x[:, 30:-30]
    return x


def select_close_session(x: np.ndarray) -> np.ndarray:
    x = x[:, -30:]
    return x


def select_high(price: np.ndarray):
    median = bn.nanmedian(price, axis=1)
    masked = np.where(price.T >= median, 1, np.nan).T
    return masked


def select_low(price: np.ndarray):
    median = bn.nanmedian(price, axis=1)
    masked = np.where(price.T < median, 1, np.nan).T
    return masked


def smart_money_S():
    pass


def smart_money_S1():
    pass


def smart_money_S2():
    pass


def smart_money_S3():
    pass


if __name__ == '__main__':
    config_path = '../../cfg/data_input.ini'
    arctic_config_path = '../../cfg/data_input_arctic.ini'
    min_data = market_data.get_bars('000001.SZ', start_date='2020-01-01', freq=Freq.m1, config_path=arctic_config_path,
                                    eod_time_adjust=False, verbose=0
                                    )
    # daily_basic = get_trade(TradeTable.daily_basic, code='000001.SZ', start_date='2020-01-01',
    #                         cols=['turnover_rate_f'],
    #                         config_path=arctic_config_path, verbose=0)
    # sp = intraday_standard_price(min_data)
    # open_session = select_open_session(sp.values)
    # intraday_ret(sp.values)
