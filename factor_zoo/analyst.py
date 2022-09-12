import datetime
import inspect
import warnings
from concurrent.futures.process import ProcessPoolExecutor
from enum import Enum
from functools import partial
from typing import Tuple, Optional, Union, List

import numpy as np
import pandas as pd
import tqdm

from data_management.cache_janitor.cache import tmp_cache
from data_management.dataIO.component_data import IndustryCategory
from data_management.dataIO.exotic_data import get_exotic, Exotic
from data_management.dataIO.fundamental_data import get_fundamental, Fundamental
from data_management.dataIO.trading_calendar import trading_dates_offsets, get_trading_date, Market
from data_management.keeper.ZooKeeper import ZooKeeper
from factor_circus.preprocessing import ListedSelector, NotSTSelector, MarketCapSelector, IndNeuFactorScaler, \
    IndRankScoreFactorScaler
from factor_zoo.consensus_forecast.wind import eps_fwd12M
from factor_zoo.utils import cal_net_profit_report

tqdm.tqdm.pandas()

warnings.filterwarnings('ignore')


class AdjMethod(Enum):
    DIVIDE4 = 'divide4'
    FORWARD_EQUITY = 'forward_equity'
    HISTORICAL_SEASONALITY = 'historical_seasonality'


def cal_suprise_value(net_profit_quarterly, _forecast_net_profit):
    std = _forecast_net_profit.std()
    if std > 0:
        f = (net_profit_quarterly - _forecast_net_profit.mean()) / std
    else:
        f = (net_profit_quarterly - _forecast_net_profit.mean()) / 0.1
    return f


def surprise_logic(i: int,
                   real: pd.DataFrame,
                   income_statement2: pd.DataFrame,
                   analyst_forecast: pd.core.groupby.generic.DataFrameGroupBy,
                   income_statement_quarterly_net_profit: pd.DataFrame,
                   adjust_method: AdjMethod = AdjMethod.DIVIDE4) -> dict:
    """

    :param i:
    :param real:
    :param income_statement2:
    :param analyst_forecast:
    :param income_statement_quarterly_net_profit:
    :param adjust_method:
    :return:
    """
    """ 
    The implementation is not shown in open source version   
    """


@tmp_cache
def _prepare_statements_xy(income_statement: pd.DataFrame,
                           fin_forecast: pd.DataFrame,
                           quick_fin: pd.DataFrame,
                           analyst_forecast: pd.DataFrame,
                           trading_dates: Union[List[datetime.date], np.ndarray],
                           income_statement_net_profit_name='np_parent_company_owners',
                           quick_fin_net_profit_name='NetProfPareComp',
                           ):
    """
    :param income_statement:
    :param fin_forecast:
    :param quick_fin:
    :param analyst_forecast:
    :param trading_dates:
    :return:
    quarterly_net_profit.head()
    Out[2]:
                             quarter  net_profit_quarterly  priority  \
    pub_date   code
    2016-12-30 300599.SZ  2015S4           37306000.95         1
               300599.SZ  2016S3           29208987.31         1
    2017-01-03 300595.SZ  2015S4           22756100.08         1
               300595.SZ  2016S3           44116369.65         1
               300597.SZ  2015S4           21999432.32         1
                         natural_year_start last_yearly_report last_quarter  \
    pub_date   code
    2016-12-30 300599.SZ         2016-01-01             2014S4          NaN
               300599.SZ         2016-01-01             2015S4       2015S4
    2017-01-03 300595.SZ         2017-01-01             2014S4          NaN
               300595.SZ         2017-01-01             2015S4       2015S4
               300597.SZ         2017-01-01             2014S4          NaN
                               code last_quarter_report_pub_date  \
    pub_date   code
    2016-12-30 300599.SZ  300599.SZ                          NaT
               300599.SZ  300599.SZ                   2016-12-30
    2017-01-03 300595.SZ  300595.SZ                          NaT
               300595.SZ  300595.SZ                   2017-01-03
               300597.SZ  300597.SZ                          NaT
                         last_yearly_report_pub_date  fiscal_year  fiscal_quarter
    pub_date   code
    2016-12-30 300599.SZ                         NaT         2015               4
               300599.SZ                  2016-12-30         2016               3
    2017-01-03 300595.SZ                         NaT         2015               4
               300595.SZ                  2017-01-03         2016               3
               300597.SZ                         NaT         2015               4


    Out[3]: analyst_forecast.head()
                                         start_date   end_date  InstitutionID  \
    pub_date   code      fiscal_year
    2001-01-01 000006.SZ 2001        2001-01-01 2001-12-31       103962.0
                         2002        2002-01-01 2002-12-31       103962.0
               000631.SZ 2002        2002-01-01 2002-12-31       103962.0
               600089.SH 2001        2001-01-01 2001-12-31       103962.0
                         2002        2002-01-01 2002-12-31       103962.0
                                          Fnetpro  forecast_year
    pub_date   code      fiscal_year
    2001-01-01 000006.SZ 2001         119150000.0           2001
                         2002         113840000.0           2002
               000631.SZ 2002          49562253.0           2002
               600089.SH 2001         112790000.0           2001
                         2002         145860000.0           2002


    income_statement.head()
    Out[5]:
                                                       net_profit
    code      fiscal_year fiscal_quarter pub_date
    000001.SZ 2015        4              2017-03-17  2.186500e+10
              2016        1              2017-04-22  6.086000e+09
                          2              2017-08-11  1.229200e+10
                          3              2017-10-21  1.871900e+10
                          4              2017-03-17  2.259900e+10

    income_statement_quarterly_net_profit.head()
    Out[6]:
                                                     net_profit_quarterly quarter  \
    pub_date   code      fiscal_year fiscal_quarter
    2016-12-30 300599.SZ 2016        3                        29208987.31  2016S3
                         2015        4                        37306000.95  2015S4
    2017-01-03 300595.SZ 2015        4                        22756100.08  2015S4
                         2016        3                        44116369.65  2016S3
               300597.SZ 2016        3                        22525569.00  2016S3
                                                     priority
    pub_date   code      fiscal_year fiscal_quarter
    2016-12-30 300599.SZ 2016        3                      1
                         2015        4                      1
    2017-01-03 300595.SZ 2015        4                      1
                         2016        3                      1
               300597.SZ 2016        3                      1


    """

    """ 
        The implementation is not shown in open source version   
        """


def priority_algo(x: pd.DataFrame):
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    else:
        return x[0]


@tmp_cache
def alsue(income_statement: pd.DataFrame,
          fin_forecast: pd.DataFrame,
          quick_fin: pd.DataFrame,
          analyst_forecast: pd.DataFrame,
          trading_dates: Union[List[datetime.date], np.ndarray],
          adjust_method: AdjMethod = AdjMethod.DIVIDE4,
          parallel=False,
          resample: Optional[str] = None,
          debug=True,
          half_life_days=0
          ) -> pd.Series:
    """

    :param income_statement:
    :param fin_forecast:
    :param quick_fin:
    :param analyst_forecast:
    :param trading_dates:
    :param adjust_method:
    :param parallel:
    :param resample:
    :param debug:
    :param half_life_days:
    :return:
    """
    """ 
        The implementation is not shown in open source version   
        """


def alsue0(income_statement: pd.DataFrame,
           fin_forecast: pd.DataFrame,
           quick_fin: pd.DataFrame,
           analyst_forecast: pd.DataFrame,
           trading_dates: Union[List[datetime.date], np.ndarray],
           parallel=False,
           resample: Optional[str] = None,
           ) -> pd.Series:
    """
    Using divide 4 to calculate quarterly forecast.

    :param income_statement:
    :param fin_forecast:
    :param quick_fin:
    :param analyst_forecast:
    :param trading_dates:
    :param parallel:
    :param resample:
    :return:
    """
    factor = alsue(income_statement, fin_forecast, quick_fin,
                   analyst_forecast, trading_dates, AdjMethod.DIVIDE4,
                   parallel, resample)
    factor.name = 'alsue0'
    return factor


def alsue1(income_statement: pd.DataFrame,
           fin_forecast: pd.DataFrame,
           quick_fin: pd.DataFrame,
           analyst_forecast: pd.DataFrame,
           trading_dates: Union[List[datetime.date], np.ndarray],
           parallel=False,
           resample: Optional[str] = None,
           ):
    """

    :param income_statement:
    :param fin_forecast:
    :param quick_fin:
    :param analyst_forecast:
    :param trading_dates:
    :param parallel:
    :param resample:
    :return:
    """
    factor = alsue(income_statement, fin_forecast, quick_fin,
                   analyst_forecast, trading_dates, AdjMethod.HISTORICAL_SEASONALITY,
                   parallel, resample, debug=False)
    factor.name = 'alsue1'
    return factor


def alsue0_decay(income_statement: pd.DataFrame,
                 fin_forecast: pd.DataFrame,
                 quick_fin: pd.DataFrame,
                 analyst_forecast: pd.DataFrame,
                 trading_dates: Union[List[datetime.date], np.ndarray],
                 parallel=False,
                 half_life_days=20
                 ):
    """

    :param income_statement:
    :param fin_forecast:
    :param quick_fin:
    :param analyst_forecast:
    :param trading_dates:
    :param parallel:
    :param half_life_days:
    :return:
    """
    factor = alsue(income_statement, fin_forecast, quick_fin,
                   analyst_forecast, trading_dates, AdjMethod.DIVIDE4,
                   parallel, 'M', debug=False, half_life_days=half_life_days)
    factor.name = 'alsue0_decay_{}'.format(half_life_days)
    return factor


def eps_fwd12M_R3M(analyst_forecast: pd.DataFrame, trading_dates, income_statement):
    eps_fwd12m = eps_fwd12M(analyst_forecast, trading_dates, parallel=True, end=datetime.datetime.now().strftime('%Y-%m-%d'),
                            income_statement=income_statement
                            )
    eps_fwd12m = eps_fwd12m.unstack()
    r3m = eps_fwd12m - eps_fwd12m.shift(60)
    r3m = r3m.stack()
    r3m.name = 'EPS_Fwd12M_R3M'
    return r3m


def profit_double_surprise_v1(alsue_factor: pd.Series,
                              eps_fwd12M_R3M_factor: pd.Series,
                              data_input_path: str,
                              alsue_num_select=100,
                              eps_num_select=25
                              ):
    alsue_factor = ListedSelector(data_input_path, 180).fit_transform(alsue_factor)
    alsue_factor = NotSTSelector(data_input_path).fit_transform(alsue_factor)
    alsue_factor = alsue_factor.groupby(level=0).nlargest(alsue_num_select).droplevel(0)
    f = alsue_factor.to_frame().join(eps_fwd12M_R3M_factor).dropna()
    f = f.groupby(level=0)['EPS_Fwd12M_R3M'].nlargest(eps_num_select).droplevel(0)
    f.name = 'profit_double_surprise_v1_{}_{}'.format(alsue_num_select, eps_num_select)
    return f


def profit_double_surprise_v2(alsue_factor: pd.Series,
                              eps_fwd12M_R3M_factor: pd.Series,
                              data_input_path: str,
                              alsue_num_select=100,
                              eps_num_select=25,
                              cap_q=0.2,
                              ):
    alsue_factor = ListedSelector(data_input_path, 180).fit_transform(alsue_factor)
    alsue_factor = NotSTSelector(data_input_path).fit_transform(alsue_factor)
    alsue_factor = MarketCapSelector(data_input_path, cap_q).fit_transform(alsue_factor)
    dt = alsue_factor.index.get_level_values(0)
    dt = [d for d in dt if d.month in [1, 2, 3, 4, 7, 8, 10]]
    alsue_factor = alsue_factor[alsue_factor.index.get_level_values(0).isin(dt)]
    alsue_factor = IndNeuFactorScaler(IndNeuFactorScaler.Method.DemeanScaler, data_input_path,
                                      IndustryCategory.sw_l1).fit_transform(alsue_factor)
    alsue_factor = alsue_factor.groupby(level=0).nlargest(alsue_num_select).droplevel(0)
    f = alsue_factor.to_frame().join(eps_fwd12M_R3M_factor).dropna()
    f = f.groupby(level=0)['EPS_Fwd12M_R3M'].nlargest(eps_num_select).droplevel(0)
    f.name = 'profit_double_surprise_v2_{}_{}_{}'.format(alsue_num_select, eps_num_select, cap_q)
    return f


def profit_double_surprise_v3(alsue_factor: pd.Series,
                              eps_fwd12M_R3M_factor: pd.Series,
                              data_input_path: str,
                              alsue_num_select=100,
                              eps_num_select=25,
                              cap_q=0.2,
                              ):
    alsue_factor = ListedSelector(data_input_path, 180).fit_transform(alsue_factor)
    alsue_factor = NotSTSelector(data_input_path).fit_transform(alsue_factor)
    alsue_factor = MarketCapSelector(data_input_path, cap_q).fit_transform(alsue_factor)
    dt = alsue_factor.index.get_level_values(0)
    dt = [d for d in dt if d.month in [1, 2, 3, 4, 7, 8, 10]]
    alsue_factor = alsue_factor[alsue_factor.index.get_level_values(0).isin(dt)]

    alsue_factor = IndNeuFactorScaler(IndNeuFactorScaler.Method.WinsorizationStandardScaler, data_input_path,
                                      IndustryCategory.sw_l1, 0.05).fit_transform(alsue_factor)
    alsue_factor = alsue_factor.groupby(level=0).nlargest(alsue_num_select).droplevel(0)
    f = alsue_factor.to_frame().join(eps_fwd12M_R3M_factor).dropna()
    f = f.groupby(level=0)['EPS_Fwd12M_R3M'].nlargest(eps_num_select).droplevel(0)
    f.name = 'profit_double_surprise_v3_{}_{}_{}'.format(alsue_num_select, eps_num_select, cap_q)
    return f


def profit_double_surprise_v4(alsue_factor: pd.Series,
                              eps_fwd12M_R3M_factor: pd.Series,
                              data_input_path: str,
                              eps_num_select=25,
                              cap_q=0.2,
                              ):
    alsue_factor = ListedSelector(data_input_path, 180).fit_transform(alsue_factor)
    alsue_factor = NotSTSelector(data_input_path).fit_transform(alsue_factor)
    alsue_factor = MarketCapSelector(data_input_path, cap_q).fit_transform(alsue_factor)
    dt = alsue_factor.index.get_level_values(0)
    dt = [d for d in dt if d.month in [1, 2, 3, 4, 7, 8, 10]]
    alsue_factor = alsue_factor[alsue_factor.index.get_level_values(0).isin(dt)]
    # alsue_factor = alsue_factor[alsue_factor > 0]
    alsue_factor = IndRankScoreFactorScaler(data_input_path, IndustryCategory.sw_l1).fit_transform(alsue_factor)
    f = alsue_factor.to_frame().join(eps_fwd12M_R3M_factor).dropna()
    f['EPS_Fwd12M_R3M'] = IndRankScoreFactorScaler(data_input_path, IndustryCategory.sw_l1).fit_transform(f['EPS_Fwd12M_R3M'])
    f = f['alsue0_decay_20'] + f['EPS_Fwd12M_R3M']
    f = f.groupby(level=0).nlargest(eps_num_select).droplevel(0)
    f.name = 'profit_double_surprise_v4_{}_{}'.format(eps_num_select, cap_q)
    return f


