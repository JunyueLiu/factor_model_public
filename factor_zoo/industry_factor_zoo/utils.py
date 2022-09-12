from typing import Dict

import pandas as pd

from data_management.dataIO.component_data import IndustryCategory, get_industry_component
from data_management.dataIO.trade_data import TradeTable, get_trade
from factor_zoo.industry import industry_category
from factor_zoo.money_flow import avg_net_amount_l


def transform_data(factor: pd.Series, industry_dict: Dict):
    cat = industry_category(industry_dict)
    data = factor.to_frame().unstack().join(cat.to_frame().unstack())
    data = data.stack().dropna()
    data = data.set_index('industry_code', append=True)
    return data.squeeze()


def industry_mean(factor: pd.Series, industry_dict: Dict, industry_name: IndustryCategory):
    data = transform_data(factor, industry_dict)
    f = data.groupby(level=[0, 2]).mean()
    f.name = '{}_{}_mean'.format(factor.name, industry_name.name)
    return f


def industry_std(factor: pd.Series, industry_dict: Dict, industry_name: IndustryCategory):
    data = transform_data(factor, industry_dict)
    f = data.groupby(level=[0, 2]).std()
    f.name = '{}_{}_std'.format(factor.name, industry_name.name)
    return f


def industry_median(factor: pd.Series, industry_dict: Dict, industry_name: IndustryCategory):
    data = transform_data(factor, industry_dict)
    f = data.groupby(level=[0, 2]).median()
    f.name = '{}_{}_median'.format(factor.name, industry_name.name)
    return f


def industry_skew(factor: pd.Series, industry_dict: Dict, industry_name: IndustryCategory):
    data = transform_data(factor, industry_dict)
    f = data.groupby(level=[0, 2]).skew()
    f.name = '{}_{}_skew'.format(factor.name, industry_name.name)
    return f


def industry_kurt(factor: pd.Series, industry_dict: Dict, industry_name: IndustryCategory):
    data = transform_data(factor, industry_dict)
    f = data.groupby(level=[0, 2]).transform(lambda x: x.kurt()).droplevel(1).drop_duplicates().sort_index()
    f.name = '{}_{}_kurt'.format(factor.name, industry_name.name)
    return f


def industry_cap_mean(factor: pd.Series, daily_basic: pd.DataFrame, industry_dict: Dict,
                      industry_name: IndustryCategory):
    cap = daily_basic['circ_mv']
    data = transform_data(factor, industry_dict)
    data = data.to_frame().reset_index(level=2).unstack()
    data = data.join(cap.to_frame().unstack()).stack().dropna()
    data = data.set_index('industry_code', append=True)
    f = data.groupby(level=[0, 2]).apply(lambda x: (x[factor.name] * x['circ_mv']).sum() / x['circ_mv'].sum())
    f.name = '{}_{}_cap_mean'.format(factor.name, industry_name.name)
    return f


if __name__ == '__main__':
    money_flow = get_trade(TradeTable.money_flow, start_date='2021-10-01', )
    daily_basic = get_trade(TradeTable.daily_basic, start_date='2021-10-01')
    f = avg_net_amount_l(money_flow, 20)
    industry_dict = get_industry_component(IndustryCategory.sw_l1)
    industry_kurt(f, industry_dict, IndustryCategory.sw_l1)
    # industry_cap_mean(f, daily_basic, industry_dict, IndustryCategory.sw_l1)
