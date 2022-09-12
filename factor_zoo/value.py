import numpy as np
import pandas as pd

from data_management.dataIO.fundamental_data import get_fundamental, Fundamental
from data_management.dataIO.trade_data import get_trade, TradeTable
from data_management.dataIO.trading_calendar import get_trading_date, Market, trading_dates_offsets
from factor_zoo.size import float_market_cap_2
from factor_zoo.utils import fundamental_preprocess, combine_fundamental_with_fundamental, \
    combine_market_with_fundamental, EBIT, get_next_trading_date_dict, get_last_trading_of_month_dict, \
    date_to_last_trading_date_of_month, report_delta

pd.set_option('max_columns', None)


def book_to_market_ratio_daily_basic(daily_basic: pd.DataFrame) -> pd.Series:
    """

    :param daily_basic:
    :return:
    """
    factor = daily_basic['pb']
    factor.name = 'book_to_market_ratio_daily_basic'
    return factor


def EM(daily_basic: pd.DataFrame, income_statement, trading_date) -> pd.Series:
    """
    Magic Formula
    EBIT / cap
    :return:
    """
    # todo
    raise NotImplementedError('Problem in calculating EBIT')
    income_statement['EBIT'] = EBIT(income_statement)
    income = fundamental_preprocess(income_statement, trading_date, ['EBIT'],
                                    'end_date')

    merged = combine_market_with_fundamental(daily_basic, income)
    # merged['capital'] = merged['close'] * merged['share_total'] * 10000  # total_share is in 10 thousand shares

    factor = merged['EBIT'] / merged['capital']
    factor.name = 'EM'
    return factor


def pe_ttm(daily_basic: pd.DataFrame) -> pd.Series:
    """
    分子=最近交易日收盘价*最新普通股总股数 分母=归属母公司股东的净利润(TTM)*最近交易日转换汇率(记帐本位币转换为交易币种)  返回=分子／分母
    :return:
    """
    # cap = market_cap(market, capital_change, trading_date)
    # cap.name = 'cap'
    # cap = cap.dropna()
    # if not income_statement_statement_is_ttm:
    #     np_parent_company_owners = fundamental_preprocess(income_statement, trading_date, ['np_parent_company_owners'],
    #                                                       'end_date', ttm_transformation=True, sheet_type='income')
    # else:
    #     np_parent_company_owners = income_statement['np_parent_company_owners_TTM']
    #
    # merged = combine_fundamental_with_fundamental(cap, np_parent_company_owners)
    # factor = merged['cap'] / merged['np_parent_company_owners_TTM']
    # factor.name = 'pe_ttm'
    # factor = factor.dropna()
    # return factor

    factor = daily_basic['pe_ttm']
    factor.name = 'pe_ttm_daily_basic'
    return factor


# def pb(market, capital_change, balance_sheet: pd.DataFrame, trading_date) -> pd.Series:
#     """
#     总市值2／指定日最新报告期股东权益(不含少数股东权益、优先股及永续债)
#     :param market:
#     :param capital_change:
#     :param trading_date:
#     :return:
#     """
#     cap = market_cap(market, capital_change, trading_date)
#     cap.name = 'cap'
#     cap = cap.dropna()
#     balance_sheet = balance_sheet.fillna(0)
#     bs = fundamental_preprocess(balance_sheet, trading_date,
#                                 ['total_owner_equities', 'minority_interests', 'other_equity_tools',
#                                  'pepertual_liability_equity'], 'end_date')
#
#     merged = combine_fundamental_with_fundamental(cap, bs)
#     factor = merged['cap'] / (
#             merged['total_owner_equities']
#             - merged['minority_interests']
#             - merged['other_equity_tools']
#             - merged['pepertual_liability_equity'])
#     factor.name = 'pb'
#     factor = factor.dropna()
#     return factor


# def avg_pb(market, capital_change, balance_sheet: pd.DataFrame, trading_date, resample='M') -> pd.Series:
#     """
#
#     :param market:
#     :param capital_change:
#     :param balance_sheet:
#     :param trading_date:
#     :param resample:
#     :return:
#     """
#     factor = pb(market, capital_change, balance_sheet, trading_date).to_frame().reset_index()
#     factor = factor.groupby(by=['instCode']).resample(resample, on='date').agg({'pb': 'mean',
#                                                                                 'date': 'last'}).droplevel(1)
#     next_trading_date = get_next_trading_date_dict(trading_date)
#     last_trading_of_month = get_last_trading_of_month_dict(trading_date)
#     factor = factor.dropna(subset=['date'])
#     factor['date'] = factor['date'].apply(
#         lambda x: date_to_last_trading_date_of_month(x, trading_date, next_trading_date, last_trading_of_month))
#     factor = factor.set_index('date', append=True).swaplevel(0, 1).sort_index()['pb']
#     factor.name = 'avg_pb'
#     return factor


def pcf_opera(daily_basic: pd.DataFrame,
              cash_flow_statement: pd.DataFrame,
              trading_date: np.ndarray,
              cashflow_statement_statement_is_ttm=False):
    """
    总市值2/经营活动现金净流量TTM

    :param market:
    :param capital_change:
    :param cash_flow_statement:
    :param trading_date:
    :return:
    """
    cap = float_market_cap_2(daily_basic)
    cap.name = 'cap'
    cap = cap.dropna()
    offset = trading_dates_offsets(trading_date, 'D')
    if not cashflow_statement_statement_is_ttm:
        net_operate_cash_flow = fundamental_preprocess(cash_flow_statement,
                                                       trading_date,
                                                       ['net_operate_cash_flow'],
                                                       'end_date',
                                                       resample_freq=offset,
                                                       ttm_transformation=True,
                                                       sheet_type='cashflow')
    else:
        net_operate_cash_flow = cash_flow_statement['net_operate_cash_flow_TTM']

    merged = combine_fundamental_with_fundamental(cap, net_operate_cash_flow)
    factor = merged['cap'] / merged['net_operate_cash_flow_TTM']
    factor.name = 'pcf_opera'
    factor = factor.dropna()
    return factor


def pcf_cash(daily_basic: pd.DataFrame,
             balance_sheet: pd.DataFrame,
             trading_date: np.ndarray):
    """
    分子=最近交易日收盘价*最新普通股总股数
    分母=现金及现金等价物净增加额(TTM)*最近交易日转换汇率(记帐本位币转换为交易币种)
    返回=分子／分母
    """
    cap = float_market_cap_2(daily_basic)
    cap.name = 'cap'
    cap = cap.dropna()

    offset = trading_dates_offsets(trading_date, 'D')
    bs = fundamental_preprocess(balance_sheet, trading_date, ['cash_equivalents'], 'end_date',
                                resample_freq=offset, drop_duplicate=False)
    delta_cash_equivalents = report_delta(bs)
    delta_cash_equivalents.columns = ['delta_cash_equivalents']

    merged = combine_fundamental_with_fundamental(cap, delta_cash_equivalents)
    factor = merged['cap'] / merged['delta_cash_equivalents']
    factor.name = 'pcf_cash'
    factor = factor.dropna()
    return factor


# def did_yield(market, cash_flow_statement, trading_date, cashflow_statement_statement_is_ttm=False):
#     """
#
#     :param market:
#     :param cash_flow_statement:
#     :param trading_date:
#     :param cashflow_statement_statement_is_ttm:
#     :return:
#     """
#     cap = market_cap(market, capital_change, trading_date)
#     cap.name = 'cap'
#     cap = cap.dropna()
#     if not cashflow_statement_statement_is_ttm:
#         dividend_interest_payment = fundamental_preprocess(cash_flow_statement, trading_date,
#                                                            ['dividend_interest_payment'],
#                                                            'end_date', ttm_transformation=True, sheet_type='cashflow')
#     else:
#         dividend_interest_payment = cash_flow_statement['dividend_interest_payment_TTM']
#
#     merged = combine_fundamental_with_fundamental(cap, dividend_interest_payment)
#     factor = merged['dividend_interest_payment_TTM'] / merged['cap']
#     factor.name = 'did_yield'
#     factor = factor.dropna()
#     return factor

