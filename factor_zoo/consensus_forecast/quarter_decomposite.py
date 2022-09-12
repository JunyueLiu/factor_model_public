import datetime
from functools import partial
from typing import List, Union, Dict, Any

import numpy as np
import pandas as pd

from data_management.cache_janitor.cache import tmp_cache
from factor_zoo.utils import parallelize_on_rows


def get_last_year_forward_net_profit(x: pd.Series, income_statement_np):
    """
        The implementation is not shown in open source version
        """


def get_income_statement_origin_net_profit(x: pd.Series, income_statement_cum_net_profit: pd.DataFrame):
    """
        The implementation is not shown in open source version
        """


@tmp_cache
def get_last_quarter_cum_net_profit(quarterly_net_profit: pd.DataFrame, income_statement_cum_net_profit: pd.DataFrame):
    """
        The implementation is not shown in open source version
        """


def cal_expected_quarter_net_profit(np_consensus_forecast: pd.Series,
                                    last_quarter_cum_net_profit: pd.Series,
                                    last_year_forward_quarter: pd.Series,
                                    last_year_quarter_net_profit: pd.Series):
    """
    The implementation is not shown in open source version
    """


def cal_surprise_magnitude(expected_quarter_net_profit: float,
                           net_profit_quarterly: float,
                           ):
    #                             单季度实际净利润 - 单季度预期净利润
    #           单季度超预期幅度 = ------------------------------------
    #                                   abs(单季度预期净利润)
    """
    The implementation is not shown in open source version
    """


def cal_analyst_quarter_forecast(key: Dict[str, Any],
                                 analyst_forecast,
                                 max_lookback_days: int = 60
                                 ):
    #                                                       Fnetpro
    # pub_date   code      forecast_year Brokern
    # 2001-01-01 000006.SZ 2001          大鹏证券有限责任公司  1.191500e+08
    #                      2002          大鹏证券有限责任公司  1.138400e+08
    #            000631.SZ 2002          大鹏证券有限责任公司  4.956225e+07
    #            600089.SH 2001          大鹏证券有限责任公司  1.127900e+08
    #                      2002          大鹏证券有限责任公司  1.458600e+08
    """
    The implementation is not shown in open source version
    """


@tmp_cache
def get_valid_analyst_forecast_df(quarterly_net_profit: pd.DataFrame,
                                  analyst_forecast: pd.DataFrame,
                                  max_lookback_days
                                  ):
    """
    The implementation is not shown in open source version
    """


@tmp_cache
def get_last_year_forward_quarter_net_profit_df(quarterly_net_profit: pd.DataFrame, income_statement_np: pd.DataFrame):
    f = partial(get_last_year_forward_net_profit, income_statement_np=income_statement_np.groupby(level=1))
    return parallelize_on_rows(quarterly_net_profit, f, 8)


def get_last_year_whole_year_net_profit(quarterly_net_profit: pd.DataFrame, income_statement_np: pd.DataFrame):
    pass


def get_quarter(s: pd.Series):
    return s.apply(lambda x:
                   '{}S1'.format(x.year) if x.month == 3 else
                   '{}S2'.format(x.year) if x.month == 6 else
                   '{}S3'.format(x.year) if x.month == 9 else
                   '{}S4'.format(x.year),
                   )


def get_last_year_q4(s: pd.Series):
    return s.apply(lambda x:
                   '{}S4'.format(int(x[:4]) - 1)
                   )


@tmp_cache
def decompose_analyst_forecast(analyst_forecast: pd.DataFrame,
                               income_statement: pd.DataFrame,
                               fin_forecast: pd.DataFrame,
                               quick_fin: pd.DataFrame,
                               trading_dates: Union[List[datetime.date], np.ndarray],
                               income_statement_net_profit_name='np_parent_company_owners',
                               quick_fin_net_profit_name='NetProfPareComp',
                               minimum_cover_num=3,
                               max_lookback_days=60
                               ):
    """
        The implementation is not shown in open source version
    """
