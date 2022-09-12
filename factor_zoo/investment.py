import pandas as pd
from factor_zoo.size import market_cap
from factor_zoo.utils import code_generator, fundamental_preprocess, combine_fundamental_with_fundamental, \
    combine_market_with_fundamental, load_pickle, make_field_mapping, report_delta


def invest(balance_sheet: pd.DataFrame, trading_date):
    """

    :param balance_sheet:
    :param trading_date:
    :return:
    """
    bs = fundamental_preprocess(balance_sheet, trading_date, ['total_assets'], 'end_date', drop_duplicate=False)
    delta_asset = report_delta(bs)
    delta_asset.columns = ['delta_asset']
    bs = bs.groupby(level=[0, 1]).last()
    bs = bs.join(delta_asset)
    factor = bs['delta_asset'] / bs['total_assets']
    factor.name = 'invest'
    return factor

def invest_cash_paid(cashflow_statement, cashflow_statement_is_ttm=False):
    pass



def RD_intensity():
    """
    Lin, Ji-Chai and Yanzhi Wang, 2016, "The R&D Premium and Takeover Risk", Accounting Review 91, 955-971.

    Long firms in the top and short in the bottom R&D intensity quantile that is defined as R&D expenditure divided by size
    :return:
    """
    pass

