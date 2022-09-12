import pandas as pd
import numpy as np

from factor_zoo.utils import fundamental_preprocess, load_pickle


"""
Troublesome:
半年报和季报不一定会公布这些项目，导致大量空值。。。
fillna处理后依旧大量空值

"""
def quick_ratio(balance_sheet: pd.DataFrame, trading_date) -> pd.Series:
    """
    速动比率	速动比率=(流动资产合计-存货)/ 流动负债合计
    :param balance_sheet:
    :return:
    """
    bs = fundamental_preprocess(balance_sheet, trading_date,
                                ['total_current_assets', 'inventories',
                                 'total_current_liability'], 'end_date',
                                )
    # bs = bs.fillna(0)

    factor = (bs['total_current_assets'] - bs['inventories']) / bs['total_current_liability']
    factor.name = 'quick_ratio'
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.groupby(level=1).fillna(method='ffill')
    factor = factor.dropna()
    return factor


def intangible_asset_ratio(balance_sheet, trading_date) -> pd.Series:
    """
    无形资产比率=(无形资产+研发支出+商誉)/总资产
    :return:
    """

    bs = fundamental_preprocess(balance_sheet, trading_date,
                                ['intangible_assets', 'development_expenditure',
                                 'good_will', 'total_assets'], 'end_date',
                                )
    bs = bs.fillna(0)
    factor = (bs['intangible_assets'] +
              bs['development_expenditure'] +
              bs['good_will']) / bs['total_assets']
    factor.name = 'intangible_asset_ratio'
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.groupby(level=1).fillna(method='ffill')
    factor = factor.dropna()
    return factor


def long_debt_to_working_capital_ratio(balance_sheet, trading_date) -> pd.Series:
    """

    :return:
    """
    bs = fundamental_preprocess(balance_sheet, trading_date,
                                ['cash_equivalents', 'account_receivable',
                                 'accounts_payable', 'inventories',
                                 'longterm_loan'], 'end_date',
                                )
    bs = bs.fillna(0)
    factor = bs['longterm_loan']/(bs['cash_equivalents'] +
              bs['account_receivable'] +
              bs['accounts_payable'] - bs['inventories'])
    factor.name = 'long_debt_to_working_capital_ratio'
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.groupby(level=1).fillna(method='ffill')
    factor = factor.dropna()
    return factor


def long_debt_to_asset_ratio(balance_sheet, trading_date) -> pd.Series:
    """
    long_debt_to_asset_ratio	长期借款与资产总计之比	长期借款与资产总计之比=长期借款/总资产
    :return:
    """
    bs = fundamental_preprocess(balance_sheet, trading_date,
                                ['longterm_loan', 'total_assets'], 'end_date',
                                )
    bs = bs.fillna(0)
    factor = bs['longterm_loan']/ bs['total_assets']
    factor.name = 'long_debt_to_asset_ratio'
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.groupby(level=1).fillna(method='ffill')
    factor = factor.dropna()
    return factor


def debt_to_tangible_equity_ratio(balance_sheet, trading_date) -> pd.Series:
    """
    有形净值债务率	负债合计/有形净值 其中有形净值=股东权益-无形资产净值，无形资产净值= 商誉+无形资产
    :return:
    """
    bs = fundamental_preprocess(balance_sheet, trading_date,
                                ['total_liability',
                                 'total_owner_equities',
                                 'good_will', 'intangible_assets'], 'end_date',
                                )
    bs = bs.fillna(0)
    factor = bs['total_liability'] / \
             (bs['total_owner_equities'] - (bs['good_will'] + bs['intangible_assets']))
    factor.name = 'debt_to_tangible_equity_ratio'
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.groupby(level=1).fillna(method='ffill')
    factor = factor.dropna()
    return factor


