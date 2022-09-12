import pandas as pd

from factor_zoo.utils import fundamental_preprocess, combine_fundamental_with_fundamental, \
    report_avg, EBIT, EBIT_ttm, \
    factor_fill_status_index


def ps_ttm(daily_basic: pd.DataFrame) -> pd.Series:
    """
    :return:
    """
    factor = daily_basic['ps_ttm']
    factor.name = 'ps_ttm'
    return factor


def roa1(balance_sheet,
         income_statement,
         trading_date,
         income_statement_statement_is_ttm=False,
         status: None or pd.DataFrame = None) -> pd.Series:
    """
    总资产报酬率   息税前利润*2／(期初总资产+期末总资产)*100%
    :param balance_sheet:
    :param income_statement:
    :param trading_date:
    :param income_statement_statement_is_ttm:
    :param market_cap:
    :return:
    """
    bs = fundamental_preprocess(balance_sheet, trading_date, ['total_assets'], 'end_date', drop_duplicate=False)
    bs = report_avg(bs)

    if not income_statement_statement_is_ttm:
        income_statement['EBIT'] = EBIT(income_statement)
        ebit = fundamental_preprocess(income_statement, trading_date, ['EBIT'], 'end_date', ttm_transformation=True,
                                      sheet_type='income')
    else:
        income_statement['EBIT_TTM'] = EBIT_ttm(income_statement)
        ebit = fundamental_preprocess(income_statement, trading_date, ['EBIT_TTM'], 'end_date')

    merged = combine_fundamental_with_fundamental(bs, ebit)
    factor = merged['EBIT_TTM'] * 100 / merged['total_assets']
    factor.name = 'roa1'
    if status is not None:
        factor = factor_fill_status_index(factor, status)
    factor = factor.dropna()
    return factor


def roa_ttm(balance_sheet,
            income_statement,
            trading_date,
            income_statement_statement_is_ttm=False,
            status: None or pd.DataFrame = None) -> pd.Series:
    """
    总资产净利率  净利润*2／(期初总资产+期末总资产)*100%
    :param balance_sheet:
    :param income_statement:
    :param trading_date:
    :return:
    """
    bs = fundamental_preprocess(balance_sheet, trading_date, ['total_assets'], 'end_date', drop_duplicate=False)
    bs = report_avg(bs)
    if not income_statement_statement_is_ttm:
        net_profit = fundamental_preprocess(income_statement, trading_date, ['net_profit'], 'end_date',
                                            ttm_transformation=True,
                                            sheet_type='income')
    else:
        net_profit = income_statement['net_profit_TTM']
    merged = combine_fundamental_with_fundamental(bs, net_profit)
    factor = merged['net_profit_TTM'] * 100 / merged['total_assets']
    factor.name = 'roa_ttm'
    if status is not None:
        factor = factor_fill_status_index(factor, status)
    factor = factor.dropna()
    return factor


def roe(balance_sheet,
        income_statement,
        trading_date,
        income_statement_statement_is_ttm=False,
        status: None or pd.DataFrame = None) -> pd.Series:
    """

    :param balance_sheet:
    :param income_statement:
    :param trading_date:
    :return:
    """
    bs = fundamental_preprocess(balance_sheet, trading_date, ['equities_parent_company_owners'], 'end_date',
                                drop_duplicate=False)
    bs = report_avg(bs)
    if not income_statement_statement_is_ttm:
        np_parent_company_owners = fundamental_preprocess(income_statement, trading_date, ['np_parent_company_owners'],
                                                          'end_date', ttm_transformation=True, sheet_type='income')
    else:
        np_parent_company_owners = income_statement['np_parent_company_owners_TTM']
    merged = combine_fundamental_with_fundamental(bs, np_parent_company_owners)
    factor = merged['np_parent_company_owners_TTM'] * 100 / merged['equities_parent_company_owners']
    factor.name = 'roe'
    if status is not None:
        factor = factor_fill_status_index(factor, status)
    factor = factor.dropna()
    return factor


def rotc(balance_sheet: pd.DataFrame,
         income_statement,
         trading_date,
         income_statement_statement_is_ttm=False,
         status: None or pd.DataFrame = None) -> pd.Series:
    """
    神奇公式 return on tangible capital

    EBIT /（Net Working Capital + Net Fixed Assets）
     Net Working Capital + Net Fixed Assets，即净营运资金和固定资产净额之和。
     具体的，ROC 的定义为最新一季息税前利润与上一季资本的比值，这里上一季资本等于 该季的流动资产 – 流动负债 + 固定资产 – 固定资产折旧。



    无息流动负债=应付账款(TTM)+预收款项(TTM)+应付职工薪酬(TTM)+应交税费(TTM)+其他应付款(TTM)+应付利息(TTM)+应付股利(TTM)
    +其他流动负债(TTM)+递延收益(TTM) （这里的递延收益项还有点拿不准）

    无息非流动负债=非流动负债合计(TTM)-长期借款(TTM)-应付债券(TTM)



    净运营资本 = 应收账款(TTM)+其他应收款(TTM)+预付款项(TTM)+存货(TTM)-无息流动负债
    :return:
    """
    balance_sheet = balance_sheet.fillna(0)
    bs = fundamental_preprocess(balance_sheet, trading_date,
                                ['account_receivable',
                                 'other_receivable',
                                 'advance_payment',
                                 'inventories',
                                 'accounts_payable',
                                 'advance_peceipts',
                                 'salaries_payable',
                                 'taxs_payable',
                                 'interest_payable',
                                 'dividend_payable',
                                 'other_payable'], 'end_date', drop_duplicate=True)

    net_working_capital = (bs['account_receivable'] + bs['other_receivable']
                           + bs['advance_payment'] + bs['inventories']) \
                          - (bs['accounts_payable'] + bs['advance_peceipts'] + bs['salaries_payable'] + bs['taxs_payable']
                           + bs['interest_payable'] + bs['dividend_payable'] + bs['other_payable'])
    net_working_capital.name = 'net_working_capital'



    if not income_statement_statement_is_ttm:
        income_statement['EBIT'] = EBIT(income_statement)
        ebit = fundamental_preprocess(income_statement, trading_date, ['EBIT'], 'end_date', ttm_transformation=True,
                                      sheet_type='income')
    else:
        income_statement['EBIT_TTM'] = EBIT_ttm(income_statement)
        ebit = fundamental_preprocess(income_statement, trading_date, ['EBIT_TTM'], 'end_date')

    merged = combine_fundamental_with_fundamental(ebit, net_working_capital)
    factor = merged['EBIT_TTM'] / merged['net_working_capital']
    factor.name = 'rotc'
    if status is not None:
        factor = factor_fill_status_index(factor, status)
    factor = factor.dropna()
    return factor


def net_profit_to_total_operate_revenue_ttm():
    # 净利润与营业总收入之比=净利润（TTM）/营业总收入（TTM）
    pass


def basic_eps(income_statement,
              trading_date,
              income_statement_statement_is_ttm=False,
              status: None or pd.DataFrame = None) -> pd.Series:
    if not income_statement_statement_is_ttm:
        factor = fundamental_preprocess(income_statement, trading_date, ['basic_eps'],
                                            'end_date', ttm_transformation=True, sheet_type='income')
    else:
        factor = income_statement['basic_eps_TTM']
    factor.name = 'basic_eps'
    if status is not None:
        factor = factor_fill_status_index(factor, status)
    factor = factor.dropna()
    return factor


def eps_ttm(capital_change, income_statement,
            trading_date,
            income_statement_statement_is_ttm=False,
            status: None or pd.DataFrame = None) -> pd.Series:
    cc = fundamental_preprocess(capital_change, trading_date, ['share_total'], 'change_date')
    if not income_statement_statement_is_ttm:
        np_parent_company_owners = fundamental_preprocess(income_statement, trading_date, ['np_parent_company_owners'],
                                                          'end_date', ttm_transformation=True, sheet_type='income')
    else:
        np_parent_company_owners = income_statement['np_parent_company_owners_TTM']

    merged = combine_fundamental_with_fundamental(cc, np_parent_company_owners)
    factor = merged['np_parent_company_owners_TTM'] / (merged['share_total'] * 10000)
    factor.name = 'eps_ttm'
    if status is not None:
        factor = factor_fill_status_index(factor, status)
    factor = factor.dropna()
    return factor


