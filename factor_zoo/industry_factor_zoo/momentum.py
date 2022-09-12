from typing import Dict

import pandas as pd

from data_management.dataIO import market_data
from data_management.dataIO.component_data import get_bars, IndustryCategory, get_industry_component
from data_management.dataIO.market_data import Freq
from data_management.keeper.ZooKeeper import ZooKeeper
from factor_zoo.industry import industry_category


def golden_rule_industry_mom(industry_daily: pd.DataFrame, n: int = 20) -> pd.Series:
    """

    Parameters
    ----------
    industry_daily
    n

    Returns
    -------

    """
    intraday_ret = (industry_daily['close'] - industry_daily['open']) / industry_daily['open']
    industry_daily['prev_close'] = industry_daily.groupby(level=1)['close'].shift(1)
    overnight_ret = (industry_daily['open'] - industry_daily['prev_close']) / industry_daily['prev_close']
    intraday_ret = intraday_ret.groupby(level=1).rolling(n).sum().droplevel(0).sort_index()
    overnight_ret = overnight_ret.groupby(level=1).rolling(n).sum().droplevel(0).sort_index()

    r1 = intraday_ret.groupby(level=0).rank()
    r2 = overnight_ret.groupby(level=0).rank(ascending=False)
    score = r1 + r2
    score = score.groupby(level=0).transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    score.name = 'golden_rule_industry_mom_{}'.format(n)
    return score


def lead_lag_industry_mom(daily_market: pd.DataFrame, industry_dict: Dict, industry_name: IndustryCategory, n: int,
                          r: float) -> pd.Series:
    """

    Parameters
    ----------
    daily_market
    industry_dict
    industry_name
    n
    r

    Returns
    -------

    """
    cat = industry_category(industry_dict)
    df = daily_market[['adj_close', 'money']]
    ret = df['adj_close'].to_frame('ret').unstack().pct_change()
    mean_ret = ret.rolling(n).mean()
    cum_money = df['money'].to_frame('cum_money').unstack().rolling(n).sum()
    data = mean_ret.join(cum_money).join(cat.to_frame().unstack())
    data = data.stack().dropna()
    data = data.set_index('industry_code', append=True)

    data = data.sort_values(['date', 'industry_code', 'cum_money'], ascending=[True, True, False])
    data['rate'] = data.groupby(level=[0, 2])['cum_money'].transform(lambda x: x.cumsum() / x.sum())
    data['is_dragon'] = (data['rate'] <= r).astype(int).replace(0, -1)

    f = data['ret'] * data['is_dragon']
    f = f.groupby(level=[0, 2]).sum()
    f.name = 'lead_lag_{}_mom_{}_{}'.format(industry_name.name, n, r)
    f.index.names = ['date', 'code']
    return f

