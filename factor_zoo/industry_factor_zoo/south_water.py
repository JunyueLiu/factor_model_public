from typing import Dict

import pandas as pd
from sklearn.linear_model import LinearRegression

from data_management.dataIO import market_data
from data_management.dataIO.component_data import get_industry_component, IndustryCategory
from data_management.dataIO.exotic_data import get_exotic, Exotic
from data_management.dataIO.trade_data import get_trade, TradeTable
from data_management.keeper.ZooKeeper import ZooKeeper
from factor_zoo.industry import industry_category


def hk_holding_size(daily_market: pd.DataFrame, hk_holding: pd.DataFrame,
                    industry_dict: Dict, industry_name: IndustryCategory
                    ) -> pd.Series:
    share_num = hk_holding['share_number'].unstack()
    close = daily_market['close'].unstack()
    holding_size = (share_num * close).fillna(method='ffill')
    holding_size.columns = pd.MultiIndex.from_product([['holding'], holding_size.columns])

    cat = industry_category(industry_dict)
    cat = cat.to_frame().unstack()
    df = holding_size.join(cat)
    df = df.stack()
    df['holding'] = df['holding'].fillna(0)
    df = df.set_index('industry_code', append=True)
    f = df.groupby(level=[0, 2])['holding'].sum()
    f = f.sort_index()
    f.name = 'hk_holding_size_{}'.format(industry_name.name)
    return f


def hk_holding_size_mv(daily_market: pd.DataFrame, hk_holding: pd.DataFrame,
                       industry_dict: Dict, industry_name: IndustryCategory, n=20
                       ) -> pd.Series:
    f = hk_holding_size(daily_market, hk_holding, industry_dict, industry_name)
    f = f.unstack().rolling(n).mean().stack()
    f.name = 'hk_holding_size_mv_{}_{}'.format(industry_name.name, n)
    return f


def hk_holding_percentage(daily_basic: pd.DataFrame,
                          hk_holding: pd.DataFrame,
                          industry_dict: Dict,
                          industry_name: IndustryCategory
                          ) -> pd.Series:
    # data = daily_basic[['total_share', 'float_share']].unstack()

    hk_holding_cap = daily_basic['total_share'].unstack() \
                     * hk_holding['share_ratio'].unstack() / 100 \
                     * daily_basic['close'].unstack()
    hk_holding_cap.columns = pd.MultiIndex.from_product([['hk_holding_cap'], hk_holding_cap.columns])

    df = hk_holding_cap.join(daily_basic['circ_mv'].to_frame().unstack())
    cat = industry_category(industry_dict)
    cat = cat.to_frame().unstack()

    df = df.join(cat).stack()
    df['hk_holding_cap'] = df['hk_holding_cap'].fillna(0)
    df = df.dropna()
    df = df.set_index('industry_code', append=True)
    df = df.groupby(level=[0, 2]).sum()
    f = df['hk_holding_cap'] / df['circ_mv']
    f.name = 'hk_holding_percentage_{}'.format(industry_name.name)
    return f


def hk_holding_percentage_mv(daily_basic: pd.DataFrame,
                             hk_holding: pd.DataFrame,
                             industry_dict: Dict,
                             industry_name: IndustryCategory,
                             n=20
                             ) -> pd.Series:
    f = hk_holding_percentage(daily_basic, hk_holding, industry_dict, industry_name)
    f = f.unstack().rolling(n).mean().stack()
    f.name = 'hk_holding_percentage_mv_{}_{}'.format(industry_name.name, n)
    return f


def hk_trade_amount(daily_market: pd.DataFrame,
                    hk_holding: pd.DataFrame,
                    industry_dict: Dict,
                    industry_name: IndustryCategory) -> pd.Series:
    share_num = hk_holding['share_number'].unstack()
    vwap = (daily_market['money'] / daily_market['volume']).unstack()
    diff_share_num = share_num.diff().abs()

    trade_amount = diff_share_num * vwap
    trade_amount.columns = pd.MultiIndex.from_product([['trade_amount'], trade_amount.columns])

    cat = industry_category(industry_dict)
    cat = cat.to_frame().unstack()
    df = trade_amount.join(cat)
    df = df.stack()
    df['trade_amount'] = df['trade_amount'].fillna(0)
    df = df.set_index('industry_code', append=True)
    f = df.groupby(level=[0, 2])['trade_amount'].sum()
    f.name = 'hk_trade_amount_{}'.format(industry_name.name)
    return f


def hk_trade_amount_mv(daily_market: pd.DataFrame,
                       hk_holding: pd.DataFrame,
                       industry_dict: Dict,
                       industry_name: IndustryCategory,
                       n=20
                       ) -> pd.Series:
    f = hk_trade_amount(daily_market, hk_holding, industry_dict, industry_name)
    f = f.unstack().rolling(n).sum().stack()
    f.name = 'hk_trade_amount_mv_{}_{}'.format(industry_name.name, n)
    return f


def hk_trade_amount_percentage(daily_market: pd.DataFrame,
                               hk_holding: pd.DataFrame,
                               industry_dict: Dict,
                               industry_name: IndustryCategory):
    tm = hk_trade_amount(daily_market, hk_holding, industry_dict, industry_name)
    tm = tm.unstack()
    tm.columns = pd.MultiIndex.from_product([['trade_amount'], tm.columns])

    money = daily_market['money'].to_frame().unstack()
    cat = industry_category(industry_dict)
    cat = cat.to_frame().unstack()
    df = money.join(cat)
    df = df.stack()
    df = df.set_index('industry_code', append=True)
    total_money = df.groupby(level=[0, 2]).sum().unstack()
    df = total_money.join(tm).stack()
    f = df['trade_amount'] / df['money']
    f.name = 'hk_trade_amount_percentage_{}'.format(industry_name.name)
    return f


def hk_trade_amount_percentage_mv(daily_market: pd.DataFrame,
                                  hk_holding: pd.DataFrame,
                                  industry_dict: Dict,
                                  industry_name: IndustryCategory,
                                  n=20
                                  ) -> pd.Series:
    f = hk_trade_amount_percentage(daily_market, hk_holding, industry_dict, industry_name)
    f = f.unstack().rolling(n).mean().stack()
    f.name = 'hk_trade_amount_percentage_mv_{}_{}'.format(industry_name.name, n)
    return f


def hk_net_inflow(daily_market: pd.DataFrame,
                  hk_holding: pd.DataFrame,
                  industry_dict: Dict,
                  industry_name: IndustryCategory) -> pd.Series:
    share_num = hk_holding['share_number'].unstack()
    vwap = (daily_market['money'] / daily_market['volume']).unstack()
    diff_share_num = share_num.diff()

    trade_amount = diff_share_num * vwap
    trade_amount.columns = pd.MultiIndex.from_product([['trade_amount'], trade_amount.columns])

    cat = industry_category(industry_dict)
    cat = cat.to_frame().unstack()
    df = trade_amount.join(cat)
    df = df.stack()
    df['trade_amount'] = df['trade_amount'].fillna(0)
    df = df.set_index('industry_code', append=True)
    f = df.groupby(level=[0, 2])['trade_amount'].sum()
    f.name = 'hk_net_inflow_{}'.format(industry_name.name)
    return f


def hk_net_inflow_mv(daily_market: pd.DataFrame,
                     hk_holding: pd.DataFrame,
                     industry_dict: Dict,
                     industry_name: IndustryCategory,
                     n=20
                     ) -> pd.Series:
    f = hk_net_inflow(daily_market, hk_holding, industry_dict, industry_name)
    f = f.unstack().rolling(n).mean().stack()
    f.name = 'hk_net_inflow_mv_{}_{}'.format(industry_name.name, n)
    return f


def hk_inflow_ratio(daily_market: pd.DataFrame,
                    daily_basic: pd.DataFrame,
                    hk_holding: pd.DataFrame,
                    industry_dict: Dict,
                    industry_name: IndustryCategory) -> pd.Series:
    circ_mv = daily_basic['circ_mv'].to_frame().unstack()
    net_inflow = hk_net_inflow(daily_market, hk_holding, industry_dict, industry_name)
    net_inflow = net_inflow.unstack()
    net_inflow.columns = pd.MultiIndex.from_product([['net_inflow'], net_inflow.columns])

    cat = industry_category(industry_dict)
    cat = cat.to_frame().unstack()
    df = circ_mv.join(cat).stack().dropna()
    df = df.set_index('industry_code', append=True)
    industry_liq_mv = df.groupby(level=[0, 2]).sum().unstack()
    df = industry_liq_mv.join(net_inflow)
    df = df.stack()
    f = df['net_inflow'] / df['circ_mv']
    f.name = 'hk_inflow_ratio_{}'.format(industry_name.name)
    return f


def hk_inflow_ratio_mv(daily_market: pd.DataFrame,
                       daily_basic: pd.DataFrame,
                       hk_holding: pd.DataFrame,
                       industry_dict: Dict,
                       industry_name: IndustryCategory,
                       n=20
                       ) -> pd.Series:
    f = hk_inflow_ratio(daily_market, daily_basic, hk_holding, industry_dict, industry_name)
    f = f.unstack().rolling(n).mean().stack()
    f.name = 'hk_inflow_ratio_mv_{}_{}'.format(industry_name.name, n)
    return f


def hk_inflow_holding_residual(daily_market: pd.DataFrame,
                               daily_basic: pd.DataFrame,
                               hk_holding: pd.DataFrame,
                               industry_dict: Dict,
                               industry_name: IndustryCategory,
                               n1=20, n2=60):
    holding = hk_holding_size_mv(daily_market, hk_holding, industry_dict, industry_name, n1)
    inflow = hk_inflow_ratio_mv(daily_market, daily_basic, hk_holding, industry_dict, industry_name, n2)

    df = holding.to_frame().unstack().join(inflow.to_frame().unstack())
    df = df.stack()
    df = df.dropna()

    def _f(x):
        ols = LinearRegression().fit(x[[holding.name]], x[inflow.name])
        res = x[inflow.name] - ols.predict(x[[holding.name]])
        return res

    f = df.groupby(level=0).apply(_f)
    f = f.droplevel(0).sort_index()
    f.name = 'hk_inflow_holding_residual_{}_{}_{}'.format(industry_name.name, n1, n2)
    return f


