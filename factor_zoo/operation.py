import pandas as pd
from factor_zoo.size import market_cap
from factor_zoo.utils import code_generator, fundamental_preprocess, combine_fundamental_with_fundamental, \
    combine_market_with_fundamental, load_pickle, make_field_mapping, report_avg, EBIT


def accounts_payable_turnover_days(balance_sheet: pd.DataFrame, income_statement, trading_date,
                                   income_statement_ttm=False) -> pd.Series:
    """
    accounts_payable_turnover_days	应付账款周转天数	应付账款周转天数 = 360 / 应付账款周转率
    :return:
    """
    accounts_payable = fundamental_preprocess(balance_sheet, trading_date, ['accounts_payable'], 'end_date',
                                              drop_duplicate=False)
    accounts_payable = report_avg(accounts_payable)
    if not income_statement_ttm:
        total_operating_cost = fundamental_preprocess(income_statement, trading_date, ['operating_cost'],
                                                      ttm_transformation=True, sheet_type='income')
    else:
        total_operating_cost = income_statement['operating_cost_TTM']

    merged = combine_fundamental_with_fundamental(accounts_payable, total_operating_cost)

    factor = 360 * merged['accounts_payable'] / merged['operating_cost_TTM']
    factor.name = 'accounts_payable_turnover_days'
    factor = factor.dropna()
    return factor


def account_receivable_turnover_days(balance_sheet: pd.DataFrame, income_statement, trading_date,
                                     income_statement_ttm=False) -> pd.Series:
    """
    account_receivable_turnover_days	应收账款周转天数	应收账款周转天数=360/应收账款周转率
    :return:
    """
    account_receivable = fundamental_preprocess(balance_sheet, trading_date, ['account_receivable'], 'end_date',
                                                drop_duplicate=False)
    account_receivable = report_avg(account_receivable)
    if not income_statement_ttm:
        total_operating_revenue = fundamental_preprocess(income_statement, trading_date, ['total_operating_revenue'],
                                                         ttm_transformation=True, sheet_type='income')
    else:
        total_operating_revenue = income_statement['total_operating_revenue_TTM']

    merged = combine_fundamental_with_fundamental(account_receivable, total_operating_revenue)

    factor = 360 * merged['account_receivable'] / merged['total_operating_revenue_TTM']
    factor.name = 'account_receivable_turnover_days'
    factor = factor.dropna()
    return factor


def inventory_turnover_days(balance_sheet: pd.DataFrame, income_statement, trading_date,
                            income_statement_ttm=False) -> pd.Series:
    """
    inventory_turnover_days	存货周转天数	存货周转天数=360/存货周转率
    :return:
    """
    inventories = fundamental_preprocess(balance_sheet, trading_date, ['inventories'], 'end_date',
                                         drop_duplicate=False)
    inventories = report_avg(inventories)
    if not income_statement_ttm:
        total_operating_cost = fundamental_preprocess(income_statement, trading_date, ['operating_cost'],
                                                      ttm_transformation=True, sheet_type='income')
    else:
        total_operating_cost = income_statement['operating_cost_TTM']

    merged = combine_fundamental_with_fundamental(inventories, total_operating_cost)

    factor = 360 * merged['inventories'] / merged['operating_cost_TTM']
    factor.name = 'accounts_payable_turnover_days'
    factor = factor.dropna()
    return factor


def operating_cycle(balance_sheet: pd.DataFrame, income_statement, trading_date,
                    income_statement_ttm=False) -> pd.Series:
    """
    应收账款周转天数+存货周转天数
    :return:
    """
    f1 = account_receivable_turnover_days(balance_sheet, income_statement, trading_date,
                                          income_statement_ttm=income_statement_ttm)
    f2 = inventory_turnover_days(balance_sheet, income_statement, trading_date,
                                 income_statement_ttm=income_statement_ttm)
    factor = f1 + f2
    factor.name = 'operating_cycle'
    factor = factor.dropna()
    return factor


def inventory_turnover_rate(balance_sheet: pd.DataFrame, income_statement, trading_date,
                            income_statement_ttm=False) -> pd.Series:
    """
    inventory_turnover_rate	存货周转率	存货周转率=营业成本（TTM）/存货
    :return:
    """
    inventories = fundamental_preprocess(balance_sheet, trading_date, ['inventories'], 'end_date',
                                         drop_duplicate=False)
    inventories = report_avg(inventories)
    if not income_statement_ttm:
        total_operating_cost = fundamental_preprocess(income_statement, trading_date, ['operating_cost'],
                                                      ttm_transformation=True, sheet_type='income')
    else:
        total_operating_cost = income_statement['operating_cost_TTM']

    merged = combine_fundamental_with_fundamental(inventories, total_operating_cost)

    factor = merged['operating_cost_TTM'] / merged['inventories']
    factor.name = 'inventory_turnover_rate'
    factor = factor.dropna()
    return factor


def total_asset_turnover_rate(balance_sheet: pd.DataFrame, income_statement, trading_date,
                              income_statement_ttm=False) -> pd.Series:
    """
    total_asset_turnover_rate	总资产周转率	总资产周转率=营业收入(ttm)/总资产
    :return:
    """
    total_assets = fundamental_preprocess(balance_sheet, trading_date, ['total_assets'], 'end_date',
                                          drop_duplicate=False)
    total_assets = report_avg(total_assets)
    if not income_statement_ttm:
        total_operating_cost = fundamental_preprocess(income_statement, trading_date, ['operating_cost'],
                                                      ttm_transformation=True, sheet_type='income')
    else:
        total_operating_cost = income_statement['operating_cost_TTM']

    merged = combine_fundamental_with_fundamental(total_assets, total_operating_cost)

    factor = merged['operating_cost_TTM'] / merged['total_assets']
    factor.name = 'total_asset_turnover_rate'
    factor = factor.dropna()
    return factor

