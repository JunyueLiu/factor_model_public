import datetime
from typing import List, Union

import numpy as np
import pandas as pd


def cal_surprise_magnitude(expected_quarter_net_profit: pd.Series,
                           net_profit_quarterly: pd.Series,
                           ):
    #                             单季度实际净利润 - 单季度预期净利润
    #           单季度超预期幅度 = ------------------------------------
    #                                   abs(单季度预期净利润)
    """
        The implementation is not shown in open source version
        """


def resample_surprise_magnitude(surprise_magnitude: pd.Series, trading_dates):
    """
    The implementation is not shown in open source version
    """


def cal_surprise_num(valid_analyst_forecast: pd.DataFrame, surprise_used_quarter_net_profit: pd.DataFrame,
                     minimum_cover_num):
    """
        The implementation is not shown in open source version
        """


def tf_surprise_30(analyst_forecast: pd.DataFrame,
                   income_statement: pd.DataFrame,
                   fin_forecast: pd.DataFrame,
                   quick_fin: pd.DataFrame,
                   mkt_cap: pd.Series,
                   money: pd.Series,
                   industry_dict,
                   trading_dates: Union[List[datetime.datetime.date], np.ndarray],
                   minimum_cover_num=3,
                   max_lookback_days=60):
    """
        The implementation is not shown in open source version
        """
