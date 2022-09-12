import pandas as pd
import numpy as np

from data_management.dataIO.fundamental_data import get_fundamental, Fundamental
from factor_zoo.utils import fundamental_preprocess, combine_fundamental_with_fundamental, load_pickle


def np_parent_to_cash(cashflow_statment, income_statement, trading_date,
                      cashflow_statment_is_ttm=False, income_statement_statement_is_ttm=False) -> pd.Series:
    """
    【释义】
    参见算法


    【算法】
        净利润现金含量=经营活动产生的现金流量净额/归属于母公司所有者的净利润*100%


    【参数】
        报告期


    【来源】
    Wind计算


    :param cashflow_statment:
    :param income_statement:
    :param trading_date:
    :param cashflow_statment_is_ttm:
    :param income_statement_statement_is_ttm:
    :return:
    """

    if cashflow_statment_is_ttm is False:
        net_operate_cash_flow = fundamental_preprocess(cashflow_statment, trading_date,
                                                       ['net_operate_cash_flow'],
                                                       second_date_col='end_date', ttm_transformation=True,
                                                       sheet_type='cashflow_statment')
    else:
        net_operate_cash_flow = cashflow_statment['net_operate_cash_flow_TTM']

    if income_statement_statement_is_ttm is False:
        operating_revenue = fundamental_preprocess(income_statement, trading_date, ['np_parent_company_owners'],
                                                   second_date_col='end_date', ttm_transformation=True,
                                                   sheet_type='income')
    else:
        operating_revenue = income_statement['np_parent_company_owners_TTM']

    fundamental = combine_fundamental_with_fundamental(net_operate_cash_flow, operating_revenue)
    fundamental = fundamental.fillna(0)

    factor = fundamental['net_operate_cash_flow_TTM'] / fundamental['np_parent_company_owners_TTM']
    factor.name = 'np_parent_to_cash'
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.groupby(level=1).fillna(method='ffill')
    factor = factor.dropna()
    return factor


def capital_expense_to_depreciation_and_amortization(cashflow_statment, trading_date,
                                                  cashflow_statment_is_ttm=False) -> pd.Series:
    """
    【释义】
    本指标通常用来衡量企业为长期发展所发生的投资水平和趋势。
    通常连续观察若干报告期，如果投资高于折旧的程度正在下降，可能意味着企业管理策略开始趋于保守，相反则可能意味着企业管理趋于激进。
    该指标还可以结合净资产收益率(ROE)和投入资本回报率使用，如果ROE很低，而该比率却很高，那么很可能用于固定资产的投资是无效的，这将损害企业的价值。


    【算法】
        购建固定资产、无形资产和其他长期资产支付的现金／(固定资产折旧、油气资产折耗、生产性生物资产折旧+无形资产摊销+长期待摊费用摊销)


    【参数】
        报告期

    :return:
    """
    if cashflow_statment_is_ttm is False:
        cashflow_ttm = fundamental_preprocess(cashflow_statment, trading_date,
                                              ['fix_intan_other_asset_acqui_cash',
                                               'fixed_assets_depreciation',
                                               'intangible_assets_amortization',
                                               'defferred_expense_amortization'],
                                              second_date_col='end_date', ttm_transformation=True,
                                              sheet_type='cashflow_statment')
    else:
        cashflow_ttm = cashflow_statment[['fix_intan_other_asset_acqui_cash_TTM',
                                          'fixed_assets_depreciation_TTM',
                                          'intangible_assets_amortization_TTM',
                                          'defferred_expense_amortization_TTM']]
    cashflow_ttm = cashflow_ttm.fillna(0)

    factor = cashflow_ttm['fix_intan_other_asset_acqui_cash_TTM'] / \
             (cashflow_ttm['fixed_assets_depreciation_TTM']
              + cashflow_ttm['intangible_assets_amortization_TTM']
              + cashflow_ttm['defferred_expense_amortization_TTM'])
    factor.name = 'capital_expense_to_depreciation_and_amoration'
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.groupby(level=1).fillna(method='ffill')
    factor = factor.dropna()
    return factor


def cash_to_sales_ttm(cashflow_statment, income_statement, trading_date,
                      cashflow_statment_is_ttm=False, income_statement_statement_is_ttm=False) -> pd.Series:
    """

    【释义】
        参见算法


    【算法】
        销售商品提供劳务收到的现金(TTM)／营业收入(TTM)


    【参数】
        报告期


    【来源】
        Wind计算

        :return:
    """
    if cashflow_statment_is_ttm is False:
        goods_sale_and_service_render_cash = fundamental_preprocess(cashflow_statment, trading_date,
                                                                    ['goods_sale_and_service_render_cash'],
                                                                    second_date_col='end_date', ttm_transformation=True,
                                                                    sheet_type='cashflow_statment')
    else:
        goods_sale_and_service_render_cash = cashflow_statment['goods_sale_and_service_render_cash_TTM']

    if income_statement_statement_is_ttm is False:
        operating_revenue = fundamental_preprocess(income_statement, trading_date, ['operating_revenue'],
                                                   second_date_col='end_date', ttm_transformation=True,
                                                   sheet_type='income')
    else:
        operating_revenue = income_statement['operating_revenue_TTM']

    fundamental = combine_fundamental_with_fundamental(goods_sale_and_service_render_cash, operating_revenue)
    fundamental = fundamental.fillna(0)

    factor = fundamental['goods_sale_and_service_render_cash_TTM'] / fundamental['operating_revenue_TTM']
    factor.name = 'cash_to_sales_ttm'
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.groupby(level=1).fillna(method='ffill')
    factor = factor.dropna()
    return factor


def operation_cashflow_to_operating_revenue_ttm(cashflow_statment, income_statement, trading_date,
                                                cashflow_statment_is_ttm=False,
                                                income_statement_statement_is_ttm=False) -> pd.Series:
    """

    【释义】
        参见算法


    【算法】
        销售商品提供劳务收到的现金(TTM)／营业收入(TTM)


    【参数】
        报告期


    【来源】
        Wind计算

        :return:
    """
    if cashflow_statment_is_ttm is False:
        net_operate_cash_flow = fundamental_preprocess(cashflow_statment, trading_date,
                                                       ['net_operate_cash_flow'],
                                                       second_date_col='end_date', ttm_transformation=True,
                                                       sheet_type='cashflow_statment')
    else:
        net_operate_cash_flow = cashflow_statment['net_operate_cash_flow_TTM']

    if income_statement_statement_is_ttm is False:
        operating_revenue = fundamental_preprocess(income_statement, trading_date, ['operating_revenue'],
                                                   second_date_col='end_date', ttm_transformation=True,
                                                   sheet_type='income')
    else:
        operating_revenue = income_statement['operating_revenue_TTM']

    fundamental = combine_fundamental_with_fundamental(net_operate_cash_flow, operating_revenue)
    fundamental = fundamental.fillna(0)

    factor = fundamental['net_operate_cash_flow_TTM'] / fundamental['operating_revenue_TTM']
    factor.name = 'operation_cashflow_to_operating_revenue_ttm'
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.groupby(level=1).fillna(method='ffill')
    factor = factor.dropna()
    return factor


def operation_cashflow_to_operating_net_profit_ttm(cashflow_statment, income_statement, trading_date,
                                                   cashflow_statment_is_ttm=False,
                                                   income_statement_statement_is_ttm=False) -> pd.Series:
    """

        【释义】
        参见算法


    【算法】
        经营活动产生的现金流量净额(TTM)／经营活动净收益(TTM)
        经营活动净收益等于营业总收入减去营业总成本


    【参数】
        报告期


    【来源】
        Wind计算


        :return:
    """
    if cashflow_statment_is_ttm is False:
        net_operate_cash_flow = fundamental_preprocess(cashflow_statment, trading_date,
                                                       ['net_operate_cash_flow'],
                                                       second_date_col='end_date', ttm_transformation=True,
                                                       sheet_type='cashflow_statment')
    else:
        net_operate_cash_flow = cashflow_statment['net_operate_cash_flow_TTM']

    if income_statement_statement_is_ttm is False:
        net_profit = fundamental_preprocess(income_statement, trading_date,
                                            ['total_operating_revenue', 'total_operating_cost'],
                                            second_date_col='end_date', ttm_transformation=True,
                                            sheet_type='income')
    else:
        net_profit = income_statement[['total_operating_revenue_TTM', 'total_operating_cost_TTM']]

    fundamental = combine_fundamental_with_fundamental(net_operate_cash_flow, net_profit)
    fundamental = fundamental.fillna(0)

    factor = fundamental['net_operate_cash_flow_TTM'] / \
             (fundamental['total_operating_revenue_TTM'] - fundamental['total_operating_cost_TTM'])
    factor.name = 'operation_cashflow_to_operating_net_profit_ttm'
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.groupby(level=1).fillna(method='ffill')
    factor = factor.dropna()
    return factor


def operation_cashflow_to_operating_profit_ttm(cashflow_statment, income_statement, trading_date,
                                               cashflow_statment_is_ttm=False,
                                               income_statement_statement_is_ttm=False) -> pd.Series:
    """

        【释义】
        参见算法


    【算法】
        经营活动产生的现金流量净额（TTM)/营业利润(TTM)*100％


    【参数】
        报告期


    【来源】
        Wind计算

        :return:
    """
    if cashflow_statment_is_ttm is False:
        net_operate_cash_flow = fundamental_preprocess(cashflow_statment, trading_date,
                                                       ['net_operate_cash_flow'],
                                                       second_date_col='end_date', ttm_transformation=True,
                                                       sheet_type='cashflow_statment')
    else:
        net_operate_cash_flow = cashflow_statment['net_operate_cash_flow_TTM']

    if income_statement_statement_is_ttm is False:
        operating_revenue = fundamental_preprocess(income_statement, trading_date, ['operating_profit'],
                                                   second_date_col='end_date', ttm_transformation=True,
                                                   sheet_type='income')
    else:
        operating_revenue = income_statement['operating_profit_TTM']

    fundamental = combine_fundamental_with_fundamental(net_operate_cash_flow, operating_revenue)
    fundamental = fundamental.fillna(0)

    factor = fundamental['net_operate_cash_flow_TTM'] / fundamental['operating_profit_TTM']
    factor.name = 'operation_cashflow_to_operating_profit_ttm'
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.groupby(level=1).fillna(method='ffill')
    factor = factor.dropna()
    return factor


def operate_cashflow_ratio_ttm(cashflow_statment, trading_date,
                               cashflow_statment_is_ttm=False) -> pd.Series:
    """
    【释义】
    反映经营活动产生的净现金流占总净现金流中的比率，是“主营业务比率”的现金流量修正。


    【算法】
        经营活动产生的现金流量净额／（经营活动产生的现金流量净额+投资活动产生的现金流量净额+筹资活动产生的现金流量净额）*100%


    【参数】
        报告期


    【来源】
        Wind计算

    :param cashflow_statment:
    :param trading_date:
    :param cashflow_statment_is_ttm:
    :return:
    """

    if cashflow_statment_is_ttm is False:
        cashflow_ttm = fundamental_preprocess(cashflow_statment, trading_date,
                                              ['net_operate_cash_flow', 'net_invest_cash_flow',
                                               'net_finance_cash_flow'],
                                              second_date_col='end_date', ttm_transformation=True,
                                              sheet_type='cashflow_statment')
    else:
        cashflow_ttm = cashflow_statment[['net_operate_cash_flow_TTM',
                                          'net_invest_cash_flow_TTM',
                                          'net_finance_cash_flow_TTM']]
    cashflow_ttm = cashflow_ttm.fillna(0)

    factor = cashflow_ttm['net_operate_cash_flow_TTM'] / \
             (cashflow_ttm['net_operate_cash_flow_TTM']
              + cashflow_ttm['net_invest_cash_flow_TTM']
              + cashflow_ttm['net_finance_cash_flow_TTM'])
    factor.name = 'operate_cashflow_ratio_ttm'
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.groupby(level=1).fillna(method='ffill')
    factor = factor.dropna()
    return factor


def invest_cashflow_ratio_ttm(cashflow_statment, trading_date,
                              cashflow_statment_is_ttm=False) -> pd.Series:
    """
    【释义】
    反映投资活动产生的净现金流占总净现金流中的比率。


    【算法】
        投资活动产生的现金流量净额／（经营活动产生的现金流量净额+投资活动产生的现金流量净额+筹资活动产生的现金流量净额）*100%


    【参数】
        报告期

    :param cashflow_statment:
    :param trading_date:
    :param cashflow_statment_is_ttm:
    :return:
    """

    if cashflow_statment_is_ttm is False:
        cashflow_ttm = fundamental_preprocess(cashflow_statment, trading_date,
                                              ['net_operate_cash_flow', 'net_invest_cash_flow',
                                               'net_finance_cash_flow'],
                                              second_date_col='end_date', ttm_transformation=True,
                                              sheet_type='cashflow_statment')
    else:
        cashflow_ttm = cashflow_statment[['net_operate_cash_flow_TTM',
                                          'net_invest_cash_flow_TTM',
                                          'net_finance_cash_flow_TTM']]
    cashflow_ttm = cashflow_ttm.fillna(0)

    factor = cashflow_ttm['net_invest_cash_flow_TTM'] / \
             (cashflow_ttm['net_operate_cash_flow_TTM']
              + cashflow_ttm['net_invest_cash_flow_TTM']
              + cashflow_ttm['net_finance_cash_flow_TTM'])
    factor.name = 'invest_cashflow_ratio_ttm'
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.groupby(level=1).fillna(method='ffill')
    factor = factor.dropna()
    return factor


def finance_cashflow_ratio_ttm(cashflow_statment, trading_date,
                               cashflow_statment_is_ttm=False) -> pd.Series:
    """
    【释义】
    反映筹资活动产生的净现金流占总净现金流中的比率。


    【算法】
        筹资活动产生的现金流量净额／（经营活动产生的现金流量净额+投资活动产生的现金流量净额+筹资活动产生的现金流量净额）*100%


    【参数】
        报告期

    :param cashflow_statment:
    :param trading_date:
    :param cashflow_statment_is_ttm:
    :return:
    """

    if cashflow_statment_is_ttm is False:
        cashflow_ttm = fundamental_preprocess(cashflow_statment, trading_date,
                                              ['net_operate_cash_flow', 'net_invest_cash_flow',
                                               'net_finance_cash_flow'],
                                              second_date_col='end_date', ttm_transformation=True,
                                              sheet_type='cashflow_statment')
    else:
        cashflow_ttm = cashflow_statment[['net_operate_cash_flow_TTM',
                                          'net_invest_cash_flow_TTM',
                                          'net_finance_cash_flow_TTM']]
    cashflow_ttm = cashflow_ttm.fillna(0)

    factor = cashflow_ttm['net_finance_cash_flow_TTM'] / \
             (cashflow_ttm['net_operate_cash_flow_TTM']
              + cashflow_ttm['net_invest_cash_flow_TTM']
              + cashflow_ttm['net_finance_cash_flow_TTM'])
    factor.name = 'finance_cashflow_ratio_ttm'
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.groupby(level=1).fillna(method='ffill')
    factor = factor.dropna()
    return factor


def cash_flow_adequacy_ratio(cashflow_statment, trading_date,
                               cashflow_statment_is_ttm=False) -> pd.Series:
    """
    【释义】
    反映企业经营产生的现金满足资本支出、存货增加和发放现金股利的能力


    【算法】
        经营活动产生的现金流量净额／（购建固定资产、无形资产和其他长期资产支付的现金-存货的减少+分配股利、利润或偿付利息支付的现金），
        其中：购建固定资产、无形资产和其他长期资产支付的现金、存货的减少、分配股利、利润或偿付利息支付的现金相关数据均来自现金流量表
        注：该指标不适合金融类公司


    【参数】
        报告期
    :return:
    """
    if cashflow_statment_is_ttm is False:
        cashflow_ttm = fundamental_preprocess(cashflow_statment, trading_date,
                                              ['net_operate_cash_flow',
                                               'fix_intan_other_asset_acqui_cash',
                                               'inventory_decrease',
                                               'dividend_interest_payment'],
                                              second_date_col='end_date', ttm_transformation=True,
                                              sheet_type='cashflow_statment')
    else:
        cashflow_ttm = cashflow_statment[['net_operate_cash_flow_TTM',
                                          'fix_intan_other_asset_acqui_cash_TTM',
                                          'inventory_decrease_TTM',
                                          'dividend_interest_payment_TTM']]
    cashflow_ttm = cashflow_ttm.fillna(0)

    factor = cashflow_ttm['net_operate_cash_flow_TTM'] / \
             (cashflow_ttm['fix_intan_other_asset_acqui_cash_TTM']
              - cashflow_ttm['inventory_decrease_TTM']
              + cashflow_ttm['dividend_interest_payment_TTM'])
    factor.name = 'cash_flow_adequacy_ratio'
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.groupby(level=1).fillna(method='ffill')
    factor = factor.dropna()
    return factor



def operating_cash_flow_to_operating_profit_ratio(cashflow_statment, income_statement, trading_date,
                                               cashflow_statment_is_ttm=False,
                                               income_statement_statement_is_ttm=False) -> pd.Series:
    """
    【释义】
    分析会计收益和现金净流量的比例关系，评价收益质量。


    【算法】
        经营活动产生的现金流量净额／
        （净利润+资产减值准备+
        固定资产折旧、油气资产折旧、生产性生物资产折旧+无形资产摊销+长期待摊费用摊销+待摊费用减少+预提费用增加+
        处置固定资产、无形资产和其他长期资产的损失+
        固定资产报废损失+公允价值变动损失+财务费用+投资损失+递延所得税资产减少+递延所得税负债增加）
          注：该指标不适合金融类公司


    【参数】
        报告期

    :return:
    """
    raise NotImplementedError('missing some columns')
    if cashflow_statment_is_ttm is False:
        cashflow_ttm = fundamental_preprocess(cashflow_statment, trading_date,
                                              ['net_operate_cash_flow',
                                               'fix_intan_other_asset_acqui_cash',
                                               'assets_depreciation_reserves',
                                               'fixed_assets_depreciation',
                                               # todo UNKNOWN[待摊费用减少] + UNKNOWN[预提费用增加]
                                               'defferred_expense_amortization',
                                               'dividend_interest_payment'],
                                              second_date_col='end_date', ttm_transformation=True,
                                              sheet_type='cashflow_statment')
    else:
        cashflow_ttm = cashflow_statment[['net_operate_cash_flow_TTM',
                                               'fix_intan_other_asset_acqui_cash_TTM',
                                               'assets_depreciation_reserves_TTM',
                                               'fixed_assets_depreciation_TTM',
                                               # todo UNKNOWN[待摊费用减少] + UNKNOWN[预提费用增加]
                                               'defferred_expense_amortization_TTM',
                                               'dividend_interest_payment_TTM']]

    if income_statement_statement_is_ttm is False:
        net_profit = fundamental_preprocess(income_statement, trading_date,
                                            ['net_profit',
                                             'intangible_assets_amortization',
                                             'defferred_expense_amortization',
                                             'fix_intan_other_asset_dispo_loss',
                                             'fix_intan_other_asset_dispo_loss',
                                             'fixed_asset_scrap_loss',
                                             'fair_value_change_loss',
                                             'financial_cost',
                                             'invest_loss',
                                             'deffered_tax_asset_decrease',
                                             'deffered_tax_liability_increase'],
                                                   second_date_col='end_date', ttm_transformation=True,
                                                   sheet_type='income')
    else:
        net_profit = income_statement[['net_profit_TTM',
                                             'intangible_assets_amortization_TTM',
                                             'defferred_expense_amortization_TTM',
                                             'fix_intan_other_asset_dispo_loss_TTM',
                                             'fix_intan_other_asset_dispo_loss_TTM',
                                             'fixed_asset_scrap_loss_TTM',
                                             'fair_value_change_loss_TTM',
                                             'financial_cost_TTM',
                                             'invest_loss_TTM',
                                             'deffered_tax_asset_decrease_TTM',
                                             'deffered_tax_liability_increase_TTM']]




    fundamental = combine_fundamental_with_fundamental(cashflow_ttm, net_profit)

    factor = fundamental['net_operate_cash_flow_TTM'] /  \
    (fundamental['net_profit_TTM'] + fundamental['assets_depreciation_reserves_TTM'] +
     fundamental['assets_depreciation_reserves'] +
    fundamental['intangible_assets_amortization_TTM'] +
    fundamental['defferred_expense_amortization_TTM'] +
     # todo  UNKNOWN[待摊费用减少] + UNKNOWN[预提费用增加] +
    fundamental['fix_intan_other_asset_dispo_loss_TTM'] +
    fundamental['fixed_asset_scrap_loss_TTM']
     + fundamental['fair_value_change_loss_TTM']
     + fundamental['financial_cost_TTM']
     + fundamental['invest_loss_TTM']
     + fundamental['deffered_tax_asset_decrease_TTM']
     + fundamental['deffered_tax_liability_increase_TTM'])

    factor.name = 'operating_cash_flow_to_operating_profit_ratio'
    return factor


def recovery_ratio_cash_to_total_assets(cashflow_statment, balance_sheet, trading_date,
                                               cashflow_statment_is_ttm=False,
                                               ):
    """

    【释义】
        反映企业资产产生现金的能力，其值越大越好。


    【算法】
        经营活动产生的现金流量净额／期末资产总额*100%


    【参数】
        报告期


    【来源】
        Wind计算

    :return:
    """
    if cashflow_statment_is_ttm is False:
        net_operate_cash_flow = fundamental_preprocess(cashflow_statment, trading_date,
                                                       ['net_operate_cash_flow'],
                                                       second_date_col='end_date', ttm_transformation=True,
                                                       sheet_type='cashflow_statment')
    else:
        net_operate_cash_flow = cashflow_statment['net_operate_cash_flow_TTM']
    balance_sheet = fundamental_preprocess(balance_sheet, trading_date, ['total_assets'],  second_date_col='end_date')
    fundamental = combine_fundamental_with_fundamental(net_operate_cash_flow, balance_sheet)
    fundamental = fundamental.fillna(0)

    factor = fundamental['net_operate_cash_flow_TTM'] / fundamental['total_assets']
    factor.name = 'recovery_ratio_cash_to_total_assets'
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.groupby(level=1).fillna(method='ffill')
    factor = factor.dropna()
    return factor


def cash_flow_interest_coverage_ratio(cashflow_statment, trading_date,
                               cashflow_statment_is_ttm=False) -> pd.Series:
    """

    【释义】
    反映公司支付现金股利的能力


    【算法】
        经营活动产生的现金流量净额／支付普通股股利


    【参数】
        报告期


    【来源】
        Wind计算


    :return:
    """
    if cashflow_statment_is_ttm is False:
        cashflow_ttm = fundamental_preprocess(cashflow_statment, trading_date,
                                              ['net_operate_cash_flow', 'dividend_interest_payment'],
                                              second_date_col='end_date', ttm_transformation=True,
                                              sheet_type='cashflow_statment')
    else:
        cashflow_ttm = cashflow_statment[['net_operate_cash_flow_TTM', 'dividend_interest_payment_TTM']]

    cashflow_ttm = cashflow_ttm.fillna(0)

    factor = cashflow_ttm['net_operate_cash_flow_TTM'] / cashflow_ttm['dividend_interest_payment_TTM']
    factor.name = 'cash_flow_interest_coverage_ratio'
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.groupby(level=1).fillna(method='ffill')
    factor = factor.dropna()
    return factor


def cfo_to_ev():
    pass


