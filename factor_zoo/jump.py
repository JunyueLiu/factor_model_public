import pandas as pd
import numpy as np

from data_management.dataIO import market_data


def abs_ret_night(daily_market: pd.DataFrame, n=20):
    """
    20200901-华安证券-市场微观结构剖析之九：昼夜分离，隔夜跳空与日内反转选股因子
    𝑎𝑏𝑠𝑅𝑒𝑡𝑛𝑖𝑔ℎ𝑡 = ∑𝑎𝑏𝑠 (𝑙𝑛 (𝑂𝑝𝑒𝑛𝑡/𝐶𝑙𝑜𝑠𝑒𝑡−1))

    Parameters
    ----------
    daily_market
    n

    Returns
    -------

    """
    close = daily_market['adj_close'].unstack()
    close_1 = close.shift(1)
    open_ = daily_market['adj_open'].unstack()
    log_ret = np.log(open_ / close_1)
    abs_log_ret = np.abs(log_ret)
    f = abs_log_ret.rolling(n).mean()
    f = f.stack()
    f.name = 'abs_ret_night_{}'.format(n)
    return f



if __name__ == '__main__':
    config_path = '../cfg/data_input.ini'
    start_date = '2013-01-01'
    daily_market = market_data.get_bars(adjust=True, eod_time_adjust=False, add_limit=True, start_date=start_date,
                                        config_path=config_path)
    abs_ret_night(daily_market)