from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
import tqdm

from data_management.cache_janitor.cache import tmp_cache
from data_management.dataIO.exotic_data import get_exotic, Exotic
from data_management.dataIO.fundamental_data import get_fundamental, Fundamental
from data_management.dataIO.trading_calendar import get_trading_date, Market
from factor_zoo.utils import is_annual_report, get_end_trading_date_dict


def preprocess_analyst_forecast(analyst_forecast: pd.DataFrame) -> pd.DataFrame:
    """
    returns DataFrame with this columns
    'ReportID',
    'start_date',
    'code',
    'pub_date',
    'end_date',
    'AnanmID',
    'Ananm',
    'InstitutionID',
    'Brokern',
    'Feps',
    'Fpe',
    'Fnetpro',
    'Febit',
    'Febitda',
    'Fturnover',
    'Fcfps',
    'FBPS',
    'FROA',
    'FROE',
    'FPB',
    'FTotalAssetsTurnover',
    'forecast_year',
    'report_wise_fwd12M_weights'
    Parameters
    ----------
    analyst_forecast

    Returns
    -------

    """
    # have no idea why InstitutionID will have nan but Brokern have value. The value looks correct.
    #                code   pub_date start_date   end_date  \
    # 3182  000001.SZ 2021-02-02 2021-01-01 2021-12-31
    # 3242  000001.SZ 2021-04-21 2021-01-01 2021-12-31
    # 3257  000001.SZ 2021-04-21 2022-01-01 2022-12-31
    # 3334  000001.SZ 2021-08-20 2021-01-01 2021-12-31
    # 3336  000001.SZ 2021-08-20 2022-01-01 2022-12-31
    #                                      AnanmID       Ananm    ReportID  \
    # 3182  30000000000000092889,30434917,30587468  戴志锋,邓美君,贾靖  23496829.0
    # 3242  30000000000000092889,30434917,30587468  戴志锋,邓美君,贾靖  25487642.0
    # 3257  30000000000000092889,30434917,30587468  戴志锋,邓美君,贾靖  25487642.0
    # 3334  30000000000000092889,30434917,30587468  戴志锋,邓美君,贾靖  30058635.0
    # 3336  30000000000000092889,30434917,30587468  戴志锋,邓美君,贾靖  30058635.0
    #       InstitutionID     Brokern  Feps    Fpe       Fnetpro         Febit  \
    # 3182       104104.0  中泰证券股份有限公司  1.62  15.11  3.240700e+10  4.154700e+10
    # 3242       104104.0  中泰证券股份有限公司  1.73  12.57  3.435200e+10  4.404100e+10
    # 3257       104104.0  中泰证券股份有限公司  1.99  10.91  3.944400e+10  5.056900e+10
    # 3334            NaN  中泰证券股份有限公司  1.90  12.28           NaN  4.832000e+10
    # 3336            NaN  中泰证券股份有限公司  2.25  10.36           NaN  5.704300e+10
    # analyst_forecast = analyst_forecast.set_index(['code', 'pub_date', 'start_date', 'end_date', 'Brokern']).sort_index()
    # 将 start_date 转换为FY1， FY2
    # 7) FY1，FY2，FY3
    # FY1 定义为最近预测年度，即以个股的年报实际披露日为界，当年盈利公布之日，当年
    # 数据会被（原）次年数据取代。算头不算尾原则. 举例：300144.SZ，2013-11-14，2013
    # 年报还未披露，此时研究员预测的年度有 2013 年,2014 年，2015 年, 站在 2013-11-14，
    # 该个股已经公布的最大报告期为 2012 年报（300144.sz 的 2012 年报已经于 20130227 披
    # 露，定义为 FY0），因此自实际披露日（含）至今，对 2013 年报的预测为 FY1，对 2014
    # 年的预测为 FY2,对 2015 年的预测为 FY3
    analyst_forecast['forecast_year'] = 1
    analyst_forecast = analyst_forecast.set_index(['ReportID', 'start_date']).sort_index()
    analyst_forecast['forecast_year'] = analyst_forecast.groupby(level=0)['forecast_year'].cumsum()
    # analyst_forecast.forecast_year.value_counts()
    # Out[2]:
    # 1     565909
    # 2     560847
    # 3     485500
    # 4       5767
    # 5       3102
    # 6       1026
    # 7        886
    # 8        736
    # 9        576
    # 10       523
    # 11        83
    # 12        25
    # 13        12
    # 14         9
    # 15         9
    # 16         7

    """ 
    The implementation is not shown in open source version   
    """

def get_estimated_statement_pub_date(income_statement: pd.DataFrame) -> pd.Series:
    """
    The implementation is not shown in open source version
    """



def get_valid_sample(raw: pd.DataFrame, date: pd.Timestamp, days_valid: int = 180):
    days = (date - raw['pub_date']).dt.days
    raw = raw[(days >= 0) & (days <= days_valid)]
    valid_report_id = raw.groupby(['code', 'Brokern'])['ReportID'].last()
    raw = raw[raw['ReportID'].isin(valid_report_id)]
    # also should consider drop those invalid forecast (after actual report is out, may still have data in this forecast)

    return raw


def single_period_fwd12M(d: pd.Timestamp, analyst_forecast: pd.DataFrame, estimated_statement_pub_date, col: str, day_valid: int):
    """
    The implementation is not shown in open source version
    """


@tmp_cache
def eps_fwd12M(analyst_forecast: pd.DataFrame,
               trading_dates, *,
               day_valid: int = 180,
               start: str = None,
               end: str = None,
               parallel: bool = False,
               end_month=False,
               income_statement: Optional[pd.DataFrame] = None,
               estimated_statement_pub_date: Optional[pd.Series] = None
               ):
    """
    The implementation is not shown in open source version
    """

