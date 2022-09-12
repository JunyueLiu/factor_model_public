from typing import Dict

import pandas as pd

from data_management.dataIO import market_data
from data_management.dataIO.component_data import get_industry_component, IndustryCategory
from data_management.dataIO.trade_data import TradeTable, get_trade
from data_management.keeper.ZooKeeper import ZooKeeper
from factor_zoo.industry import industry_category
from factor_zoo.industry_factor_zoo.utils import industry_mean
from factor_zoo.money_flow import avg_net_amount_l


def elg_rush(tushare_moneyflow: pd.DataFrame, daily_basic: pd.DataFrame, industry_dict: Dict,
             industry_name: IndustryCategory, n=20):
    elg_inflow = (tushare_moneyflow['buy_elg_amount'] - tushare_moneyflow['sell_elg_amount']).to_frame(
        'elg_inflow').unstack()
    cir_mv = daily_basic['circ_mv'].to_frame().unstack()
    cat = industry_category(industry_dict).to_frame().unstack()
    data = elg_inflow.join(cir_mv).join(cat)
    data = data.stack().dropna()
    data = data.set_index('industry_code', append=True)
    data = data.groupby(level=[0, 2]).sum()
    rate = data['elg_inflow'] / data['circ_mv']
    f = rate.groupby(level=1).rolling(n).mean().droplevel(0).dropna().sort_index()
    f.index.names = ['date', 'code']
    f.name = 'elg_rush_{}_{}'.format(industry_name.name, n)
    return f


def small_order_exit(tushare_moenyflow: pd.DataFrame, daily_makret: pd.DataFrame, industry_dict: Dict,
                     industry_name: IndustryCategory, n=20):
    small_outflow = (tushare_moenyflow['sell_sm_amount'] - tushare_moenyflow['buy_sm_amount']) \
        .to_frame('small_outflow').unstack()
    money = daily_makret['money'].to_frame().unstack()
    cat = industry_category(industry_dict).to_frame().unstack()
    data = small_outflow.join(money).join(cat)
    data = data.stack().dropna()
    data = data.set_index('industry_code', append=True)
    data = data.groupby(level=[0, 2]).sum()
    rate = data['small_outflow'] / data['money']
    f = rate.groupby(level=1).rolling(n).mean().droplevel(0).dropna().sort_index()
    f.index.names = ['date', 'code']
    f.name = 'small_order_exit_{}_{}'.format(industry_name.name, n)
    return f


def avg_net_amount_l_industry_mean(money_flow, n=20):
    f = avg_net_amount_l(money_flow, n)
    return industry_mean(f)

