import datetime
import multiprocessing
import os
import pickle
import re
from enum import Enum
from functools import partial
from typing import Union, Dict, List, Optional

import numpy as np
import pandas as pd
import tqdm
from pandas._libs.tslibs.offsets import CBMonthEnd, CustomBusinessDay
from statsmodels import api as sm
from statsmodels.regression.rolling import RollingOLS

from data_management.cache_janitor.cache import tmp_cache
from data_management.dataIO.trading_calendar import trading_dates_offsets

tqdm.tqdm.pandas()


def EBIT(income_statement: pd.DataFrame, bank=False):
    """
    一般企业及其他：（营业总收入-营业税金及附加）-（营业成本+利息支出+手续费及佣金支出+销售费用+管理费用+研发费用+坏账损失+存货跌价损失）+其他收益
    银行业：（营业总收入-营业税金及附加）-（营业成本+管理费用+研发费用+坏账损失+存货跌价损失）
       其中：利息支出为列示在利润表营业成本之后的科目，并非财务费用附注中的利息支出
    :param income_statement:
    :return:
    """
    if bank:
        raise NotImplementedError
    else:
        income_statement = income_statement.fillna(0)
        ebit = (income_statement['total_operating_revenue'] - income_statement['operating_tax_surcharges']) \
               - (income_statement['operating_cost'] + income_statement['interest_expense']
                  + income_statement['commission_expense'] + income_statement['sale_expense']
                  + income_statement['administration_expense']  # todo short for 研发费用 (报表附注)+坏账损失(报表附注)+存货跌价损失（报表附注）
                  ) + income_statement['other_earnings']
    ebit.name = 'EBIT'
    return ebit


def EBIT_ttm(income_statement_TTM: pd.DataFrame, bank=False):
    """
    一般企业及其他：（营业总收入-营业税金及附加）-（营业成本+利息支出+手续费及佣金支出+销售费用+管理费用+研发费用+坏账损失+存货跌价损失）+其他收益
    银行业：（营业总收入-营业税金及附加）-（营业成本+管理费用+研发费用+坏账损失+存货跌价损失）
       其中：利息支出为列示在利润表营业成本之后的科目，并非财务费用附注中的利息支出
    :param income_statement:
    :return:
    """
    if bank:
        raise NotImplementedError
    else:
        income_statement_TTM = income_statement_TTM.fillna(0)
        ebit = (income_statement_TTM['total_operating_revenue_TTM'] - income_statement_TTM[
            'operating_tax_surcharges_TTM']) \
               - (income_statement_TTM['operating_cost_TTM'] + income_statement_TTM['interest_expense_TTM']
                  + income_statement_TTM['commission_expense_TTM'] + income_statement_TTM['sale_expense_TTM']
                  + income_statement_TTM['administration_expense_TTM']
                  # todo short for 研发费用 (报表附注)+坏账损失(报表附注)+存货跌价损失（报表附注）
                  ) + income_statement_TTM['other_earnings_TTM']
    ebit.name = 'EBIT_TTM'
    return ebit


def EBIT_bottom_up(income_statement, bank=False):
    """
    【算法】
    利润总额+利息费用（不含资本化利息支出）
    如果财务报告中公布了财务费用明细，则“利息费用=（利息支出-财务费用明细.利息资本化金额）-利息收入”；
    如果财务报告中未公布财务费用明细，则以“利润表.财务费用”替代。一般而言，中期报告和年度报告中会公布财务费用明细。
    :param cashflow_statement:
    :return:
    """
    if bank:
        raise NotImplementedError
    else:
        income_statement = income_statement.fillna(0)
        ebit = income_statement['total_profit'] + (income_statement['interest_expense']
            # - UNKNOWN[利息资本化金额]
            ) - income_statement['interest_income']
        ebit.name = 'EBIT_Bottom_up'
    return ebit


def EBITD(income_statement: pd.DataFrame, cashflow_statement, bank=False):
    """
    :param income_statement:
    :param cashflow_statement:
    :return:
    """
    if bank:
        raise NotImplementedError
    else:
        income_statement = income_statement.fillna(0)
        cashflow_statement = cashflow_statement.fillna(0)
        ebitd = (income_statement['total_operating_revenue'] - income_statement['operating_tax_surcharges']) \
                - (income_statement['operating_cost'] + income_statement['interest_expense']
                   + income_statement['commission_expense'] + income_statement['sale_expense']
                   + income_statement['administration_expense']  # todo short for 研发费用 (报表附注)+坏账损失(报表附注)+存货跌价损失（报表附注）
                   ) \
                + (cashflow_statement['fixed_assets_depreciation']) + income_statement['other_earnings']
        ebitd.name = 'EBITD'
    return ebitd


def EBITDA(income_statement: pd.DataFrame, cashflow_statement, bank=False):
    """
    一般企业及其他：（营业总收入-营业税金及附加）-（营业成本+利息支出+手续费及佣金支出+销售费用+管理费用+研发费用+坏账损失+存货跌价损失）
        +（固定资产折旧、油气资产折耗、生产性生物资产折旧）+无形资产摊销+长期待摊费用摊销）+其他收益
    银行业：（营业总收入-营业税金及附加）-（营业成本+管理费用+坏账损失+存货跌价损失+研发费用）+（固定资产折旧、油气资产折耗、生产性生物资产折旧+无形资产摊销+长期待摊费用摊销）
    其中：利息支出为列示在利润表营业成本之后的科目，并非财务费用附注中的利息支出
    :param income_statement:
    :param cashflow_statement:
    :return:
    """
    if bank:
        raise NotImplementedError
    else:
        income_statement = income_statement.fillna(0)
        cashflow_statement = cashflow_statement.fillna(0)
        ebitda = (income_statement['total_operating_revenue'] - income_statement['operating_tax_surcharges']) \
                 - (income_statement['operating_cost'] + income_statement['interest_expense']
                    + income_statement['commission_expense'] + income_statement['sale_expense']
                    + income_statement['administration_expense']  # todo short for 研发费用 (报表附注)+坏账损失(报表附注)+存货跌价损失（报表附注）
                    ) \
                 + (cashflow_statement['fixed_assets_depreciation'] + cashflow_statement[
            'intangible_assets_amortization']
                    + cashflow_statement['defferred_expense_amortization']) + income_statement['other_earnings']
        ebitda.name = 'EBITDA'
    return ebitda


def transform_wind_code(code: str):
    """

    :param code:
    :return:
    """
    if code.endswith('.HK'):
        return code
    elif code.startswith('6'):
        return code.split('.')[0] + '.SH'
    elif code.startswith('3'):
        return code.split('.')[0] + '.SZ'
    elif code.startswith('0'):
        return code.split('.')[0] + '.SZ'


def get_next_trading_date_dict(trading_date: list):
    """
    Get a dict mapping any date to next trading date
    :param trading_date:
    :return:
    """
    trading_date = pd.to_datetime(trading_date)
    start = trading_date[0]
    end = trading_date[-1]
    date = pd.date_range(start, end=end, freq='D')
    normal_date = pd.DataFrame(index=date)
    trading_date_df = pd.DataFrame(trading_date, index=trading_date, columns=['next_trading_date'])
    next_trading_date = normal_date \
        .join(trading_date_df).fillna(method='bfill') \
        .shift(-1) \
        .to_dict()['next_trading_date']  # type: dict
    return next_trading_date


def no_trading_date_to_next(date, trading_date: set, next_trading_date: dict):
    """
    date to trading date
    :param date:
    :param trading_date:
    :param next_trading_date:
    :return:
    """
    if date in trading_date:
        return date
    else:
        if date in next_trading_date:
            return next_trading_date[date]
        else:
            l = list(trading_date)
            l.sort()
            return l[0]


def get_last_trading_of_month_dict(trading_date: list):
    """
    Get a dict mapping any date to end of month trading date
    :param trading_date:
    :return:
    """
    trading_date = pd.to_datetime(trading_date)
    trading_date_df = pd.DataFrame(trading_date, index=trading_date, columns=['last_trading_date_month'])
    last_trading_date_of_month = trading_date_df.groupby(pd.Grouper(freq='M')).last()
    last_trading_date_of_month.reset_index(inplace=True)
    last_trading_date_of_month['index'] = last_trading_date_of_month['last_trading_date_month']
    last_trading_date_of_month.set_index('last_trading_date_month', inplace=True)
    last_trading_date_of_month = trading_date_df \
        .join(last_trading_date_of_month) \
        .fillna(method='bfill') \
        .drop(columns=['last_trading_date_month']) \
        .to_dict()['index']
    return last_trading_date_of_month


def get_end_trading_date_dict(trading_date: np.ndarray, freq='MS'):
    """
    Get a dict mapping any date to end of frequency trading date
    :param trading_date:
    :param freq:
    :return:
    """
    trading_date = pd.to_datetime(trading_date)
    dates = pd.date_range(trading_date[0], trading_date[-1])
    trading_date_series = pd.Series(trading_date, index=trading_date, name='end_trading_date')
    last_trading_date = trading_date_series.resample(freq, label='right', closed='right').last()
    last_trading_date = last_trading_date.reindex(dates).fillna(method='bfill')
    return last_trading_date.to_dict()


def date_to_last_trading_date_of_month(date, trading_date: list, next_trading_date: dict,
                                       last_trading_date_of_month: dict):
    """

    :param date:
    :param trading_date:
    :param next_trading_date:
    :param last_trading_date_of_month:
    :return:
    """
    if date not in trading_date:
        date = next_trading_date[date]
    return last_trading_date_of_month[date]


def is_annual_report(start_date: pd.Series, end_date: pd.Series, report_date: pd.Series, report_type: pd.Series):
    """

    :param start_date:
    :param end_date:
    :param report_date:
    :param report_type:
    :return:
    """
    return (start_date.dt.year == end_date.dt.year) & \
           (end_date.dt.is_year_end) & \
           (start_date.dt.is_year_start) & \
           (end_date == report_date) & \
           (report_type == 0)


def is_first_quarter_report(start_date: pd.Series, end_date: pd.Series, report_date: pd.Series, report_type: pd.Series):
    """

    :param start_date:
    :param end_date:
    :param report_date:
    :param report_type:
    :return:
    """
    return (start_date.dt.year == end_date.dt.year) & \
           (end_date.dt.is_quarter_end) & \
           (end_date.dt.month == 3) & \
           (start_date.dt.is_year_start) & \
           (end_date == report_date) & \
           (report_type == 0)


def is_half_year_report(start_date: pd.Series, end_date: pd.Series, report_date: pd.Series, report_type: pd.Series):
    return (start_date.dt.year == end_date.dt.year) & \
           (end_date.dt.is_quarter_end) & \
           (end_date.dt.month == 6) & \
           (start_date.dt.is_year_start) & \
           (end_date == report_date) & \
           (report_type == 0)


def is_third_quarter_report(start_date: pd.Series, end_date: pd.Series, report_date: pd.Series, report_type: pd.Series):
    return (start_date.dt.year == end_date.dt.year) & \
           (end_date.dt.is_quarter_end) & \
           (end_date.dt.month == 9) & \
           (start_date.dt.is_year_start) & \
           (end_date == report_date) & \
           (report_type == 0)


def to_last_quarter(series):
    if series['is_half']:
        return series['end_date'].replace(month=3, day=31)
    elif series['is_quarter3']:
        return series['end_date'].replace(month=6, day=30)
    elif series['is_annual_report']:
        return series['end_date'].replace(month=9, day=30)
    else:
        return np.nan


def four_previous_report_avg_func(x, col: str, df: pd.DataFrame):
    """

    :param x:
    :param col:
    :param df:
    :return:
    """
    if x['is_annual_report']:
        start_end_tuple = ((x['start_date'], x['this_year_quarter1_end']),
                           (x['start_date'], x['this_year_half_end']),
                           (x['start_date'], x['this_year_quarter3_end']),
                           (x['start_date'], x['end_date']),
                           )

    elif x['is_quarter1']:
        start_end_tuple = ((x['last_year_start'], x['last_year_quarter1_end']),
                           (x['last_year_start'], x['last_year_half_end']),
                           (x['last_year_start'], x['last_year_quarter3_end']),
                           (x['start_date'], x['end_date']),
                           )
    elif x['is_half']:
        start_end_tuple = ((x['last_year_start'], x['last_year_half_end']),
                           (x['last_year_start'], x['last_year_quarter3_end']),
                           (x['start_date'], x['this_year_quarter1_end']),
                           (x['start_date'], x['end_date']),
                           )
    elif x['is_quarter3']:
        start_end_tuple = ((x['last_year_start'], x['last_year_quarter3_end']),
                           (x['start_date'], x['this_year_quarter1_end']),
                           (x['start_date'], x['this_year_half_end']),
                           (x['start_date'], x['end_date']),
                           )
    else:
        return np.nan
    look_back = df.loc[x['code']]
    res = []
    for t in start_end_tuple:
        try:
            res.append(look_back.loc[t].loc[:x['pub_date']].iloc[-1][col])
        except:
            return np.nan
    return np.mean(res)


def two_avg_func(x, col, df):
    start_end_tuple = ((x['last_year_start'], x['last_year_end']),
                       (x['start_date'], x['end_date']),
                       )
    look_back = df.loc[x['code']]
    res = []
    for t in start_end_tuple:
        try:
            res.append(look_back.loc[t].loc[:x['pub_date']].iloc[-1][col])
        except:
            return np.nan
    return np.mean(res)


def other_TTM_func(x: pd.Series, col, df):
    if x['is_annual_report']:
        return x[col]
    else:
        # last year annual report
        start_end_tuple = ((x['last_year_start'], x['last_year_end']),
                           (x['last_year_start'], x['last_year_annual_end']),
                           (x['start_date'], x['end_date']),
                           )
        look_back = df.loc[x['code']]
        res = []
        # 当前报告期 + 上年年报报告期 - 上年同比报告期
        for i in range(len(start_end_tuple)):
            try:
                if i == 0:
                    res.append(- look_back.loc[start_end_tuple[i]].loc[:x['pub_date']].iloc[-1][col])
                else:
                    res.append(look_back.loc[start_end_tuple[i]].loc[:x['pub_date']].iloc[-1][col])
            except:
                if x['is_quarter1']:
                    return x[col] * 4
                elif x['is_half']:
                    return x[col] * 2
                elif x['is_quarter3']:
                    return (x[col] * 4) / 3
                else:
                    return np.nan
        return np.sum(res)


def single_quarter_func(x: pd.Series, col: str, df):
    if 'report_type' in x and x['report_type'] == 1:
        return np.nan

    if x['is_quarter1']:
        return x[col]
    else:
        try:
            res = df.get_group((x['code'], x['start_date'], x['last_quarter_end'])). \
                      droplevel([0, 1, 2]).loc[:x['pub_date']].iloc[-1][col]
            return x[col] - res
        except:
            return np.nan


def parallelize(data, func, num_of_processes=8):
    start = datetime.datetime.now()
    data_split = np.array_split(data, num_of_processes)
    pool = multiprocessing.Pool(num_of_processes, initializer=print('initial parallel pool...'))
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    print("Finish. Time used: {}s".format((datetime.datetime.now() - start).seconds))
    return data


def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)


def parallelize_on_rows(data, func, num_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)


def fundamental_preprocess_daily(data: pd.DataFrame, trading_date: list, drop_duplicate=True,
                                 second_date_col='change_date'):
    """
    To deal with daily level data.

    :param data:
    :param trading_date:
    :param drop_duplicate:
    :param second_date_col:
    :return:
    """
    data['pub_date'] = pd.to_datetime(data['pub_date'])
    d = get_next_trading_date_dict(trading_date)
    data['pub_date'] = data['pub_date'].apply(lambda x: no_trading_date_to_next(x, set(trading_date), d))

    # transform wind code

    if second_date_col is not None:
        data.sort_values(by=['pub_date', 'code', second_date_col], inplace=True)
    if drop_duplicate:
        data.drop_duplicates(subset=['pub_date', 'code'], keep='last', inplace=True)
    # set index
    data.set_index(['pub_date', 'code'], inplace=True)  # type:pd.DataFrame
    return data


class SheetType(Enum):
    BalanceSheet = 'balance_sheet'
    IncomeStatement = 'income_statement'
    CashflowStatement = 'cashflow_statement'


@tmp_cache
def fundamental_preprocess(data: pd.DataFrame,
                           trading_date: np.ndarray,
                           use_features: list,
                           *,
                           second_date_col: str or None = None,
                           resample_freq: Optional[Union[str, CBMonthEnd, CustomBusinessDay]] = None,
                           drop_duplicate: bool = True,
                           ttm_transformation: bool = False,
                           sheet_type: Optional[SheetType] = None,
                           balance_sheet_ttm_formula=None,
                           quarterly: bool = False):
    """
    from raw financial statement data to transform to point-to-market data.
    Also, it provide TTM transformation of data.
    For balance sheet, TTM has three way: ['latest', 'four_avg', 'two_avg']
    The TTM columns will have suffix of _TTM_latest, _TTM_4avg, _TTM_2avg
    For others, TTM is calcualted by 当前报告期 + 上年年报报告期 - 上年同比报告期
    The TTM columns will have suffix of _TTM

                          col1 col2 col3
    pub_date   code
    2015-08-31 300413.SZ    ... ... ...
    :param data:
    :param trading_date:
    :param use_features:
    :param second_date_col:
    :param resample_freq:
    :param ttm_transformation:
    :param sheet_type:
    :param balance_sheet_ttm_formula:
    :return:
    """
    """ 
    The implementation is not shown in open source version   
    """


def report_avg(data: pd.DataFrame):
    """
    To deal with 去年同期
    :param data:
    :return:
    """
    assert isinstance(data.index, pd.MultiIndex)

    def avg_func(x):
        if len(x) == 2:
            return np.mean(x)
        else:
            return np.nan

    avg = data.groupby(level=[0, 1]).agg(avg_func)
    return avg


def report_delta(data: pd.DataFrame):
    """
    To deal with 去年同期
    :param data:
    :return:
    """
    assert isinstance(data.index, pd.MultiIndex)

    def delta_func(x):
        if len(x) == 2:
            return x[-1] - x[0]
        else:
            return np.nan

    delta = data.groupby(level=[0, 1]).agg(delta_func)
    return delta


def get_latest_info_by_date(df: pd.DataFrame, start_pd: pd.Timestamp):
    """
    :param df:
    :param start_pd:
    :return:
    """
    names = df.index.names
    df = df.sort_index(level=[1, 0]).reset_index()
    df[names[0]] = df[names[0]].apply(
        lambda x: x if x > start_pd else start_pd)
    df = df.drop_duplicates(subset=names, keep='last')
    return df.set_index(names).sort_index()


def combine_fundamental_with_fundamental(fundamental_data1: pd.DataFrame or pd.Series,
                                         fundamental_data2: pd.DataFrame or pd.Series,
                                         start=None, end=None,
                                         universe: pd.Series or pd.DataFrame = None,
                                         ) -> pd.DataFrame:
    """

    :param fundamental_data1:
    :param fundamental_data2:
    :param start:
    :param end:
    :param universe:
    :return:
    """
    assert isinstance(fundamental_data1.index, pd.MultiIndex)
    assert isinstance(fundamental_data2.index, pd.MultiIndex)
    if isinstance(fundamental_data1, pd.Series):
        fundamental_data1 = fundamental_data1.to_frame()

    if isinstance(fundamental_data2, pd.Series):
        fundamental_data2 = fundamental_data2.to_frame()

    if start is not None:
        start_pd = pd.to_datetime(start)

        # transform the fundamental_data that contains data it should have know at the start date
        fundamental_data1 = get_latest_info_by_date(fundamental_data1, start_pd)
        fundamental_data2 = get_latest_info_by_date(fundamental_data2, start_pd)
        if universe is not None:
            universe = universe.loc[start_pd:]

    if end is not None:
        end_pd = pd.to_datetime(end)
        fundamental_data1 = fundamental_data1.loc[:end_pd]
        fundamental_data2 = fundamental_data2.loc[:end_pd]
        if universe is not None:
            universe = universe.loc[:end_pd]

    if universe is not None:
        fundamental_data1 = fundamental_data1.loc[(slice(None), universe.index.get_level_values(1).unique()), :]
        fundamental_data2 = fundamental_data2.loc[(slice(None), universe.index.get_level_values(1).unique()), :]
    fundamental_data2.index.names = fundamental_data1.index.names
    merge_df = fundamental_data1.join(fundamental_data2, how='outer', lsuffix='l_')
    merge_df = merge_df.groupby(level=1).fillna(method='ffill')
    return merge_df


def combine_market_with_fundamental(market_data: pd.DataFrame or pd.Series,
                                    fundamental_data: pd.DataFrame or pd.Series,
                                    limit=None,
                                    ) -> pd.DataFrame:
    assert isinstance(market_data.index, pd.MultiIndex)
    assert isinstance(fundamental_data.index, pd.MultiIndex)
    if isinstance(fundamental_data, pd.Series):
        fundamental_data = fundamental_data.to_frame()

    if isinstance(market_data, pd.Series):
        market_data = market_data.to_frame()

    # names = fundamental_data.index.names
    # join the data
    fundamental_data.index.names = market_data.index.names

    merge_df = market_data.join(fundamental_data)
    ## this is to make sure no future information is in the joined df
    merge_df = merge_df.groupby(level=1).fillna(method='ffill', limit=limit)
    ## if the raw_factor data is not included the data, could be nan
    return merge_df


def combine_market_with_market(market_low_resolution: pd.DataFrame or pd.Series,
                               market_high_resolution: pd.DataFrame or pd.Series):
    assert isinstance(market_low_resolution.index, pd.MultiIndex)
    assert isinstance(market_high_resolution.index, pd.MultiIndex)
    if isinstance(market_low_resolution, pd.Series):
        market_low_resolution = market_low_resolution.to_frame()

    if isinstance(market_high_resolution, pd.Series):
        market_high_resolution = market_high_resolution.to_frame()

    # names = fundamental_data.index.names
    # join the data
    market_low_resolution.index.names = market_high_resolution.index.names

    merge_df = market_low_resolution.join(market_high_resolution)
    return merge_df


def combine_market_with_market_benchmark_return(market_data: pd.DataFrame,
                                                benchmark_data: pd.DataFrame,
                                                benchmark_col_keep: list or None = None,
                                                benchmark_name: str = 'benchmark', ):
    """

    :param market_data:
    :param benchmark_data:
    :param benchmark_col_keep:
    :param benchmark_name:
    :return:
    """
    if benchmark_data.index.nlevels == 2:
        benchmark_data_ = benchmark_data.droplevel(1)
    else:
        benchmark_data_ = benchmark_data.copy()
    if benchmark_col_keep:
        benchmark_data_ = benchmark_data_[benchmark_col_keep]
    benchmark_data_ = benchmark_data_.rename(columns={c: benchmark_name + '_' + c for c in benchmark_data.columns})

    return market_data.join(benchmark_data_)


def combine_market_with_industry_benchmark_return(market_data: pd.DataFrame,
                                                  industry_category: pd.DataFrame,
                                                  benchmark_data: pd.DataFrame,
                                                  benchmark_col_keep: list or None = None,
                                                  benchmark_name: str = 'benchmark', ):
    """

    :param market_data:
    :param industry_category:
    :param benchmark_data:
    :param benchmark_col_keep:
    :param benchmark_name:
    :return:
    """
    merged = market_data.join(industry_category)
    merged = merged.dropna()
    merged = merged.set_index('industry_code', append=True)

    if not isinstance(benchmark_data.index, pd.MultiIndex):
        if not isinstance(benchmark_data.index, pd.DatetimeIndex):
            benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
            benchmark_data = benchmark_data.rename(columns={'code': 'industry_code'})
            benchmark_data['industry_code'] = benchmark_data['industry_code'].astype(str)
            benchmark_data = benchmark_data.set_index(['date', 'industry_code'])

    if benchmark_col_keep:
        benchmark_data = benchmark_data[benchmark_col_keep]

    benchmark_data = benchmark_data.rename(columns={c: benchmark_name + '_' + c for c in benchmark_data.columns})
    merged = merged.join(benchmark_data)
    merged = merged.droplevel(1).sort_index()
    return merged


def market_filter_in(data: Union[pd.DataFrame, pd.Series],
                     component: Union[Dict[pd.Timestamp, List[str]],
                                      List[Dict[pd.Timestamp, List[str]]]]) \
        -> Union[pd.Series, pd.DataFrame]:
    """
    component is a dict with key is date, values is list of stocks
    or list of that

    {datetime.datetime(2013, 7, 31, 0, 0):
    ['000006.XSHE', '000021.XSHE', '000028.XSHE', '000031.XSHE', '000042.XSHE', '000050.XSHE', '000066.XSHE', ...],...}
    :param data:
    :param component:
    :return:
    """
    if isinstance(component, list):
        universe_tuple = [(k, c) for un in component for k, v in un.items() for c in v if
                          len(v) > 0]
        idx = pd.MultiIndex.from_tuples(universe_tuple, names=data.index.names).sort_values()
    else:
        universe_tuple = [(k, c) for k, v in component.items() for c in v if len(v) > 0]
        idx = pd.MultiIndex.from_tuples(universe_tuple, names=data.index.names).sort_values()

    idx = idx.drop_duplicates()

    tickers = data.index.get_level_values(1).drop_duplicates().sort_values()
    masked = pd.Series(1, idx).unstack().reindex(columns=tickers)
    new_data = (data.unstack() * masked).stack().sort_index()
    if isinstance(data, pd.Series):
        new_data.name = data.name
    new_data.index.names = data.index.names
    return new_data


def market_filter_out(market_data: pd.DataFrame, component: dict or list):
    """
        component is a dict with key is date, values is list of stocks
        or list of that

        {datetime.datetime(2013, 7, 31, 0, 0):
        ['000006.XSHE', '000021.XSHE', '000028.XSHE', '000031.XSHE', '000042.XSHE', '000050.XSHE', '000066.XSHE', ...],...}
        :param market_data:
        :param component:
        :return:
        """
    if isinstance(component, list):
        universe_tuple = [(k, c) for un in component for k, v in un.items() for c in v if
                          len(v) > 0]
        idx = pd.MultiIndex.from_tuples(universe_tuple).sort_values()
    else:
        universe_tuple = [(k, c) for k, v in component.items() for c in v if len(v) > 0]
        idx = pd.MultiIndex.from_tuples(universe_tuple).sort_values()
    return market_data.loc[market_data.index.difference(idx)]


def save_pickle(obj, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('successfully save to {}'.format(path))


def load_pickle(path):
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)
    print('successfully load from {}'.format(path))
    return obj


def make_field_mapping(excel_path):
    bs = pd.read_excel(excel_path, sheet_name='balance sheet', index_col=1)['字段名称'].to_dict()
    cf = pd.read_excel(excel_path, sheet_name='cashflow_statement', index_col=1)['字段名称'].to_dict()
    income = pd.read_excel(excel_path, sheet_name='income_statement', index_col=1)['字段名称'].to_dict()
    return bs, cf, income


def code_generator(financial_statement_formula: str, balance_sheet_field, income_statement_field, cashflow_field,
                   note_field, ttm_suffix: str or None = None):
    # sample formula
    # （营业总收入-营业税金及附加）-（营业成本+利息支出+手续费及佣金支出+销售费用+管理费用+研发费用+坏账损失+存货跌价损失）
    #         +（固定资产折旧、油气资产折耗、生产性生物资产折旧）+无形资产摊销+长期待摊费用摊销）+其他收益
    financial_statement_formula = financial_statement_formula.strip()
    splited = re.split('（|）|\)|\(|\+|\-|\*|/', financial_statement_formula)
    result = ''
    for i in range(len(splited)):
        s = splited[i]
        if s == '':
            continue
        else:
            while financial_statement_formula.startswith(s) is False:
                char = financial_statement_formula[0]
                if char == '（':
                    result += '('
                elif char == '）':
                    result += ')'
                else:
                    result += char
                financial_statement_formula = financial_statement_formula[1:]
                result += ' '

            if s in balance_sheet_field:
                result += 'balance_sheet' + "['" + balance_sheet_field[s]
                if ttm_suffix:
                    result += ttm_suffix
                result += "']"

            elif s in income_statement_field:
                result += 'income_statement' + "['" + income_statement_field[s]
                if ttm_suffix:
                    result += ttm_suffix
                result += "']"
            elif s in cashflow_field:
                result += 'cashflow_statement' + "['" + cashflow_field[s]
                if ttm_suffix:
                    result += ttm_suffix
                result += "']"
            elif note_field is not None and s in note_field:
                result += note_field
            else:
                result += 'UNKNOWN' + '[' + s + ']'
            financial_statement_formula = financial_statement_formula.replace(s, '')
        # print(financial_statement_formula)
        result += ' '
    return result


def cal_all_balance_sheet_TTM(balance_sheet: pd.DataFrame, trading_list):
    all_features = ['cash_equivalents', 'trading_assets', 'bill_receivable', 'account_receivable', 'advance_payment',
                    'other_receivable', 'affiliated_company_receivable', 'interest_receivable', 'dividend_receivable',
                    'inventories', 'expendable_biological_asset', 'non_current_asset_in_one_year',
                    'total_current_assets', 'hold_for_sale_assets', 'hold_to_maturity_investments',
                    'longterm_receivable_account', 'longterm_equity_invest', 'investment_property', 'fixed_assets',
                    'constru_in_process', 'construction_materials', 'fixed_assets_liquidation', 'biological_assets',
                    'oil_gas_assets', 'intangible_assets', 'development_expenditure', 'good_will',
                    'long_deferred_expense', 'deferred_tax_assets', 'total_non_current_assets', 'total_assets',
                    'shortterm_loan', 'trading_liability', 'notes_payable', 'accounts_payable', 'advance_peceipts',
                    'salaries_payable', 'taxs_payable', 'interest_payable', 'dividend_payable', 'other_payable',
                    'affiliated_company_payable', 'non_current_liability_in_one_year', 'total_current_liability',
                    'longterm_loan', 'bonds_payable', 'longterm_account_payable', 'specific_account_payable',
                    'estimate_liability', 'deferred_tax_liability', 'total_non_current_liability', 'total_liability',
                    'paidin_capital', 'capital_reserve_fund', 'specific_reserves', 'surplus_reserve_fund',
                    'treasury_stock', 'retained_profit', 'equities_parent_company_owners', 'minority_interests',
                    'foreign_currency_report_conv_diff', 'irregular_item_adjustment', 'total_owner_equities',
                    'total_sheet_owner_equities', 'other_comprehensive_income', 'deferred_earning', 'settlement_provi',
                    'lend_capital', 'loan_and_advance_current_assets', 'derivative_financial_asset',
                    'insurance_receivables', 'reinsurance_receivables', 'reinsurance_contract_reserves_receivable',
                    'bought_sellback_assets', 'hold_sale_asset', 'loan_and_advance_noncurrent_assets',
                    'borrowing_from_centralbank', 'deposit_in_interbank', 'borrowing_capital',
                    'derivative_financial_liability', 'sold_buyback_secu_proceeds', 'commission_payable',
                    'reinsurance_payables', 'insurance_contract_reserves', 'proxy_secu_proceeds',
                    'receivings_from_vicariously_sold_securities', 'hold_sale_liability', 'estimate_liability_current',
                    'deferred_earning_current', 'preferred_shares_noncurrent', 'pepertual_liability_noncurrent',
                    'longterm_salaries_payable', 'other_equity_tools', 'preferred_shares_equity',
                    'pepertual_liability_equity']
    # all_features = ['cash_equivalents']
    for type in ['latest', 'four_avg', 'two_avg']:
        balance_sheet_ = balance_sheet.copy()
        all_features_ = all_features.copy()
        bs = fundamental_preprocess(balance_sheet_, trading_list, all_features_, second_date_col='end_date',
                                    ttm_transformation=True, sheet_type='bs',
                                    balance_sheet_ttm_formula=type)  # type: pd.DataFrame
        bs.to_parquet('../data/balance_sheet_TTM_' + type + '.parquet')


def cal_all_income_statement_TTM(income_statement: pd.DataFrame, trading_list):
    all_features = ['total_operating_revenue', 'operating_revenue', 'total_operating_cost', 'operating_cost',
                    'operating_tax_surcharges', 'sale_expense', 'administration_expense', 'exploration_expense',
                    'financial_expense', 'asset_impairment_loss', 'fair_value_variable_income', 'investment_income',
                    'invest_income_associates', 'exchange_income', 'other_items_influenced_income', 'operating_profit',
                    'subsidy_income', 'non_operating_revenue', 'non_operating_expense',
                    'disposal_loss_non_current_liability', 'other_items_influenced_profit', 'total_profit',
                    'income_tax', 'other_items_influenced_net_profit', 'net_profit', 'np_parent_company_owners',
                    'minority_profit', 'eps', 'basic_eps', 'diluted_eps', 'other_composite_income',
                    'total_composite_income', 'ci_parent_company_owners', 'ci_minority_owners', 'interest_income',
                    'premiums_earned', 'commission_income', 'interest_expense', 'commission_expense',
                    'refunded_premiums', 'net_pay_insurance_claims', 'withdraw_insurance_contract_reserve',
                    'policy_dividend_payout', 'reinsurance_cost', 'non_current_asset_disposed', 'other_earnings']
    # all_features = ['total_operating_revenue', 'operating_revenue']
    # income_statement_ = income_statement.copy()
    # all_features_ = all_features.copy()
    # income = fundamental_preprocess(income_statement_, trading_list, all_features_, second_date_col='end_date',
    #                                 ttm_transformation=True, sheet_type='income')  # type: pd.DataFrame
    # print(income)
    # income.to_csv('../data/income_statement_TTM.csv')
    # print('finish income statement', datetime.datetime.now())

    for feature in all_features:
        cs = fundamental_preprocess(income_statement, trading_list, [feature], second_date_col='end_date',
                                    ttm_transformation=True, sheet_type='income')  # type: pd.DataFrame
        cs.to_pickle(os.path.join('../data/income_ttm', feature + '_TTM.pickle'))
    print('finish income statement', datetime.datetime.now())


def cal_all_cashflow_statement_TTM(cashflow_statement: pd.DataFrame, trading_list):
    all_features = ['goods_sale_and_service_render_cash', 'tax_levy_refund', 'subtotal_operate_cash_inflow',
                    'goods_and_services_cash_paid', 'staff_behalf_paid', 'tax_payments',
                    'subtotal_operate_cash_outflow', 'net_operate_cash_flow', 'invest_withdrawal_cash',
                    'invest_proceeds', 'fix_intan_other_asset_dispo_cash', 'net_cash_deal_subcompany',
                    'subtotal_invest_cash_inflow', 'fix_intan_other_asset_acqui_cash', 'invest_cash_paid',
                    'impawned_loan_net_increase', 'net_cash_from_sub_company', 'subtotal_invest_cash_outflow',
                    'net_invest_cash_flow', 'cash_from_invest', 'cash_from_borrowing', 'cash_from_bonds_issue',
                    'subtotal_finance_cash_inflow', 'borrowing_repayment', 'dividend_interest_payment',
                    'subtotal_finance_cash_outflow', 'net_finance_cash_flow', 'exchange_rate_change_effect',
                    'other_reason_effect_cash', 'cash_equivalent_increase', 'cash_equivalents_at_beginning',
                    'cash_and_equivalents_at_end', 'net_profit', 'assets_depreciation_reserves',
                    'fixed_assets_depreciation', 'intangible_assets_amortization', 'defferred_expense_amortization',
                    'fix_intan_other_asset_dispo_loss', 'fixed_asset_scrap_loss', 'fair_value_change_loss',
                    'financial_cost', 'invest_loss', 'deffered_tax_asset_decrease', 'deffered_tax_liability_increase',
                    'inventory_decrease', 'operate_receivables_decrease', 'operate_payable_increase', 'others',
                    'net_operate_cash_flow_indirect', 'debt_to_capital', 'cbs_expiring_in_one_year',
                    'financial_lease_fixed_assets', 'cash_at_end', 'cash_at_beginning', 'equivalents_at_end',
                    'equivalents_at_beginning', 'other_reason_effect_cash_indirect',
                    'cash_equivalent_increase_indirect', 'net_deposit_increase', 'net_borrowing_from_central_bank',
                    'net_borrowing_from_finance_co', 'net_original_insurance_cash',
                    'net_cash_received_from_reinsurance_business', 'net_insurer_deposit_investment',
                    'net_deal_trading_assets', 'interest_and_commission_cashin', 'net_increase_in_placements',
                    'net_buyback', 'net_loan_and_advance_increase', 'net_deposit_in_cb_and_ib',
                    'original_compensation_paid', 'handling_charges_and_commission', 'policy_dividend_cash_paid',
                    'cash_from_mino_s_invest_sub', 'proceeds_from_sub_to_mino_s', 'investment_property_depreciation']

    # this has problem! Don't know why. Save nothing after the
    # all_features_ = all_features.copy()
    # cs = fundamental_preprocess(cashflow_statement, trading_list, all_features_, second_date_col='end_date',
    #                             ttm_transformation=True, sheet_type='cashflow')  # type: pd.DataFrame
    # cs.to_csv('../data/cashflow_statement_TTM.csv')
    # print('finish cashflow statement', datetime.datetime.now())
    for feature in all_features:
        cs = fundamental_preprocess(cashflow_statement, trading_list, [feature], second_date_col='end_date',
                                    ttm_transformation=True, sheet_type='cashflow')  # type: pd.DataFrame
        cs.to_pickle(os.path.join('../data/cashflow_ttm', feature + '_TTM.pickle'))
    print('finish cashflow statement', datetime.datetime.now())


def cal_all_income_statement_quarterly(income_statement: pd.DataFrame, trading_list):
    all_features = ['total_operating_revenue', 'operating_revenue', 'total_operating_cost', 'operating_cost',
                    'operating_tax_surcharges', 'sale_expense', 'administration_expense', 'exploration_expense',
                    'financial_expense', 'asset_impairment_loss', 'fair_value_variable_income', 'investment_income',
                    'invest_income_associates', 'exchange_income', 'other_items_influenced_income', 'operating_profit',
                    'subsidy_income', 'non_operating_revenue', 'non_operating_expense',
                    'disposal_loss_non_current_liability', 'other_items_influenced_profit', 'total_profit',
                    'income_tax', 'other_items_influenced_net_profit', 'net_profit', 'np_parent_company_owners',
                    'minority_profit', 'eps', 'basic_eps', 'diluted_eps', 'other_composite_income',
                    'total_composite_income', 'ci_parent_company_owners', 'ci_minority_owners', 'interest_income',
                    'premiums_earned', 'commission_income', 'interest_expense', 'commission_expense',
                    'refunded_premiums', 'net_pay_insurance_claims', 'withdraw_insurance_contract_reserve',
                    'policy_dividend_payout', 'reinsurance_cost', 'non_current_asset_disposed', 'other_earnings']

    income_statement = income_statement.fillna(0)
    for feature in all_features:
        ic = fundamental_preprocess(income_statement, trading_list, [feature], second_date_col='end_date',
                                    quarterly=True, sheet_type='income')  # type: pd.DataFrame
        ic.to_pickle(os.path.join('../data/income_quarter', feature + '_quarterly.pickle'))
    print('finish income statement', datetime.datetime.now())


def cal_all_cashflow_statement_quarterly(cashflow_statement: pd.DataFrame, trading_list):
    all_features = ['goods_sale_and_service_render_cash', 'tax_levy_refund', 'subtotal_operate_cash_inflow',
                    'goods_and_services_cash_paid', 'staff_behalf_paid', 'tax_payments',
                    'subtotal_operate_cash_outflow', 'net_operate_cash_flow', 'invest_withdrawal_cash',
                    'invest_proceeds', 'fix_intan_other_asset_dispo_cash', 'net_cash_deal_subcompany',
                    'subtotal_invest_cash_inflow', 'fix_intan_other_asset_acqui_cash', 'invest_cash_paid',
                    'impawned_loan_net_increase', 'net_cash_from_sub_company', 'subtotal_invest_cash_outflow',
                    'net_invest_cash_flow', 'cash_from_invest', 'cash_from_borrowing', 'cash_from_bonds_issue',
                    'subtotal_finance_cash_inflow', 'borrowing_repayment', 'dividend_interest_payment',
                    'subtotal_finance_cash_outflow', 'net_finance_cash_flow', 'exchange_rate_change_effect',
                    'other_reason_effect_cash', 'cash_equivalent_increase', 'cash_equivalents_at_beginning',
                    'cash_and_equivalents_at_end', 'net_profit', 'assets_depreciation_reserves',
                    'fixed_assets_depreciation', 'intangible_assets_amortization', 'defferred_expense_amortization',
                    'fix_intan_other_asset_dispo_loss', 'fixed_asset_scrap_loss', 'fair_value_change_loss',
                    'financial_cost', 'invest_loss', 'deffered_tax_asset_decrease', 'deffered_tax_liability_increase',
                    'inventory_decrease', 'operate_receivables_decrease', 'operate_payable_increase', 'others',
                    'net_operate_cash_flow_indirect', 'debt_to_capital', 'cbs_expiring_in_one_year',
                    'financial_lease_fixed_assets', 'cash_at_end', 'cash_at_beginning', 'equivalents_at_end',
                    'equivalents_at_beginning', 'other_reason_effect_cash_indirect',
                    'cash_equivalent_increase_indirect', 'net_deposit_increase', 'net_borrowing_from_central_bank',
                    'net_borrowing_from_finance_co', 'net_original_insurance_cash',
                    'net_cash_received_from_reinsurance_business', 'net_insurer_deposit_investment',
                    'net_deal_trading_assets', 'interest_and_commission_cashin', 'net_increase_in_placements',
                    'net_buyback', 'net_loan_and_advance_increase', 'net_deposit_in_cb_and_ib',
                    'original_compensation_paid', 'handling_charges_and_commission', 'policy_dividend_cash_paid',
                    'cash_from_mino_s_invest_sub', 'proceeds_from_sub_to_mino_s', 'investment_property_depreciation']
    # all_features = ['goods_sale_and_service_render_cash',]
    cashflow_statement = cashflow_statement.fillna(0)
    for feature in all_features:
        cs = fundamental_preprocess(cashflow_statement, trading_list, [feature], second_date_col='end_date',
                                    quarterly=True, sheet_type='cashflow')  # type: pd.DataFrame
        cs.to_pickle(os.path.join('../data/cashflow_quarter', feature + '_quarterly.pickle'))
    print('finish cashflow statement', datetime.datetime.now())


def last_quarter(quarter_str: str):
    year = int(quarter_str[:4])
    quarter = int(quarter_str[-1])
    if quarter == 1:
        year -= 1
        quarter = 4
    else:
        quarter -= 1
    return '{}S{}'.format(year, quarter)


def last_year_quarter(quarter_str: str):
    year = int(quarter_str[:4])
    quarter = int(quarter_str[-1])
    return '{}S{}'.format(year - 1, quarter)


def previous_quarters(quarter_str: str):
    year = int(quarter_str[:4])
    quarter = int(quarter_str[-1])
    return ['{}S{}'.format(year, i) for i in range(1, quarter)]


def forward_quarter(quarter_str: str):
    year = int(quarter_str[:4])
    quarter = int(quarter_str[-1])
    return ['{}S{}'.format(year - 1, i) for i in range(1, 5)]


def last_year_forward_quarters(quarter_str: str):
    year = int(quarter_str[:4])
    quarter = int(quarter_str[-1])
    return ['{}S{}'.format(year - 1, i) for i in range(1, quarter)]


def cal_net_profit_report(income_statement: pd.DataFrame,
                          trading_dates,
                          *,
                          fin_forecast: Optional[pd.DataFrame] = None,
                          quick_fin: Optional[pd.DataFrame] = None,
                          income_statement_net_profit_name='np_parent_company_owners',
                          quick_fin_net_profit_name='NetProfPareComp'
                          ):
    """

    Parameters
    ----------
    income_statement
    fin_forecast
    quick_fin
    trading_dates
    income_statement_net_profit_name

    Returns
    -------

    """
    offsets = trading_dates_offsets(trading_dates, 'D')
    results = []
    income_statement_quarterly_net_profit = fundamental_preprocess(income_statement, trading_dates,
                                                                   [income_statement_net_profit_name],
                                                                   quarterly=True,
                                                                   sheet_type=SheetType.IncomeStatement)
    income_statement_quarterly_net_profit = income_statement_quarterly_net_profit \
        .rename(columns={'{}_quarterly'.format(income_statement_net_profit_name): 'net_profit_quarterly'})
    income_statement_quarterly_net_profit.loc[:, 'report_type'] = 'income_statement'
    income_statement_quarterly_net_profit = shift_one_pub_date(income_statement_quarterly_net_profit, offsets)
    results.append(income_statement_quarterly_net_profit)
    #                       net_profit_quarterly quarter
    # pub_date   code
    # 2001-03-31 000011.SZ          -14170000.00  2001S1
    #            000025.SZ           -6728503.59  2001S1
    #            000025.SZ          -16521912.13  2000S1
    #            000030.SZ           -7451500.00  2001S1
    #            000411.SZ           -9550000.00  2001S1

    if fin_forecast is not None:
        fin_forecast_quarterly_net_profit = cal_fin_forecast_quarter_net_profit(fin_forecast, income_statement,
                                                                                trading_dates,
                                                                                income_statement_net_profit_name=income_statement_net_profit_name,
                                                                                shift_offset=offsets
                                                                                )
        results.append(fin_forecast_quarterly_net_profit)

    if quick_fin is not None:
        quick_fin_net_profit = cal_quick_fin_quarter_net_profit(quick_fin, income_statement, trading_dates,
                                                                income_statement_net_profit_name=income_statement_net_profit_name,
                                                                shift_offset=offsets,
                                                                quick_fin_net_profit_name=quick_fin_net_profit_name)
        results.append(quick_fin_net_profit)

    quarterly_net_profit = pd.concat(results).sort_index()
    return quarterly_net_profit


def prepare_quarterly_net_profit_income_statement(income_statement: pd.DataFrame,
                                                  income_statement_net_profit_name='np_parent_company_owners'
                                                  ):
    income_statement = income_statement[
        ['code', 'start_date', 'end_date', 'pub_date', 'report_date', 'report_type', income_statement_net_profit_name]]
    income_statement = income_statement.rename(columns={income_statement_net_profit_name: 'net_profit'})
    income_statement['start_date'] = pd.to_datetime(income_statement['start_date'])
    income_statement['end_date'] = pd.to_datetime(income_statement['end_date'])
    income_statement['pub_date'] = pd.to_datetime(income_statement['pub_date'])
    income_statement['report_date'] = pd.to_datetime(income_statement['report_date'])
    return income_statement


def cal_quick_fin_quarter_net_profit(quick_fin: pd.DataFrame,
                                     income_statement: pd.DataFrame,
                                     trading_dates, *,
                                     income_statement_net_profit_name='np_parent_company_owners',
                                     quick_fin_net_profit_name='NetProfPareComp',
                                     shift_offset: Optional[Union[pd.tseries.offsets.CustomBusinessDay,
                                                                  pd.tseries.offsets.CustomBusinessMonthEnd]] = None
                                     ):
    income_statement = prepare_quarterly_net_profit_income_statement(income_statement, income_statement_net_profit_name)
    # 去除同一天的错误报告.
    #         公司于2017年4月14日披露《2016年度业绩快报》，“一、2016年度主要财务数据和指标”。表格“单位：元”。现更正为，表格“单位：万元”。
    #     2017-04-14 000863.SZ         -4.493582e+08  2016S4 三湘印象:2016年度业绩快报(已取消) http://data.eastmoney.com/notices/detail/000863/AN201704130502662476.html
    #                000863.SZ          2.558966e+08  2016S4 三湘印象:2016年度业绩快报(更新后) http://data.eastmoney.com/notices/detail/000863/AN201704140505704055.html
    """ 
    The implementation is not shown in open source version   
    """

def cal_fin_forecast_quarter_net_profit(fin_forecast: pd.DataFrame,
                                        income_statement: pd.DataFrame,
                                        trading_dates, *,
                                        income_statement_net_profit_name='np_parent_company_owners',
                                        shift_offset: Optional[Union[pd.tseries.offsets.CustomBusinessDay,
                                                                     pd.tseries.offsets.CustomBusinessMonthEnd]] = None
                                        ):
    """
    The implementation is not shown in open source version
    """


def _cal_last_year_quarter_value(x: pd.Series, income_statement_quarterly: pd.core.groupby.generic.DataFrameGroupBy,
                                 target_col: str):
    # net_profit_quarterly         -14170000.0
    # quarter                           2001S1
    # report_type             income_statement
    # last_quarter                      2000S4
    # last_year_quarter                 2000S1
    # fiscal_year                         2001
    # fiscal_quarter                         1
    pub_date, code = x.name
    try:
        data_available = income_statement_quarterly.get_group((code, x['last_year_quarter'])) \
                             .droplevel([0, 1]).loc[:pub_date]
        if data_available.empty:
            return np.nan
        else:
            return data_available[target_col].iloc[-1]
    except KeyError:
        return np.nan


@tmp_cache
def get_last_year_quarterly_value(target_quarterly: pd.DataFrame,
                                  income_statement_quarterly: pd.DataFrame,
                                  target_col: str
                                  ):
    income_statement_quarterly = \
        income_statement_quarterly.set_index('quarter', append=True).reorder_levels([1, 2, 0]).sort_index()

    target_quarterly['last_quarter'] = target_quarterly['quarter'].apply(last_quarter)
    target_quarterly['last_year_quarter'] = target_quarterly['quarter'].apply(last_year_quarter)

    func = partial(_cal_last_year_quarter_value,
                   income_statement_quarterly=income_statement_quarterly.groupby(level=[0, 1]),
                   target_col=target_col)
    return parallelize_on_rows(target_quarterly, func)


def _cal_last_quarter_value(x: pd.Series, income_statement_quarterly: pd.DataFrame, target_col: str):
    # net_profit_quarterly         -14170000.0
    # quarter                           2001S1
    # report_type             income_statement
    # last_quarter                      2000S4
    # last_year_quarter                 2000S1
    # fiscal_year                         2001
    # fiscal_quarter                         1
    pub_date, code = x.name
    try:
        data_available = income_statement_quarterly.loc[code, x['last_quarter'], :pub_date]
        if data_available.empty:
            return np.nan
        else:
            return data_available[target_col].iloc[-1]
    except KeyError:
        return np.nan


@tmp_cache
def get_last_quarter_quarterly_value(target_quarterly: pd.DataFrame,
                                     income_statement_quarterly: pd.DataFrame,
                                     target_col: str):
    income_statement_quarterly = \
        income_statement_quarterly.set_index('quarter', append=True).reorder_levels([1, 2, 0]).sort_index()

    target_quarterly['last_quarter'] = target_quarterly['quarter'].apply(last_quarter)

    func = partial(_cal_last_quarter_value, income_statement_quarterly=income_statement_quarterly,
                   target_col=target_col)
    return parallelize_on_rows(target_quarterly, func)


def factor_fill_status_index(factor, status):
    factor = factor.reindex(status.index)
    factor = factor.groupby(level=1).fillna(method='ffill')
    return factor


def shift_one_pub_date(df: pd.DataFrame, offset) -> pd.DataFrame:
    df = df.reset_index(0)
    df['pub_date'] = df['pub_date'] - offset
    df = df.set_index('pub_date', append=True).swaplevel(0, 1).sort_index()
    return df


def open_to_close_high_low_limit(df: pd.DataFrame) -> np.ndarray:
    return np.where((df['high'] == df['low']) &
                    ((df['close'] == df['high_limit']) |
                     (df['close'] == df['low_limit'])
                     ), True, False
                    )


def open_to_close_high_limit(df: pd.DataFrame) -> np.ndarray:
    return np.where((df['high'] == df['low']) & (df['close'] == df['high_limit'])
                    , True, False
                    )


def open_to_close_low_limit(df: pd.DataFrame) -> np.ndarray:
    return np.where((df['high'] == df['low']) & (df['close'] == df['low_limit'])
                    , True, False
                    )


def rolling_regression_residual(data, y_col, X_cols, N):
    if len(data) < N:
        return pd.Series(np.nan, data.index)

    endog = data[y_col].values
    x = data[X_cols].values
    if np.isnan(x).all():
        return pd.Series(np.nan, data.index)
    exog = sm.add_constant(data[X_cols].values)
    rols = RollingOLS(endog, exog, window=N)
    rres = rols.fit(cov_type='HCCM')
    res = endog - np.sum(exog * rres.params, axis=1)
    return pd.Series(res, data.index)


def rolling_regression_alpha(data, y_col, X_cols, N):
    if len(data) < N:
        return pd.Series(np.nan, data.index)

    endog = data[y_col].values
    x = data[X_cols].values
    if np.isnan(x).all():
        return pd.Series(np.nan, data.index)
    exog = sm.add_constant(x)
    rols = RollingOLS(endog, exog, window=N)
    rres = rols.fit(cov_type='HCCM')
    res = rres.params[:, 0]
    return pd.Series(res, data.index)


def rolling_regression_beta(data, y_col, X_cols, N, beta_index=0):
    if len(data) < N:
        return pd.Series(np.nan, data.index)

    endog = data[y_col].values
    x = data[X_cols].values
    if np.isnan(x).all():
        return pd.Series(np.nan, data.index)
    exog = sm.add_constant(x)
    rols = RollingOLS(endog, exog, window=N)
    rres = rols.fit(cov_type='HCCM')
    res = rres.params[:, beta_index + 1]
    return pd.Series(res, data.index)
