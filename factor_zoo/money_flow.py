import inspect
from typing import Union

import pandas as pd
import numpy as np
from tqdm import tqdm

from data_management.dataIO import market_data
from data_management.dataIO.trade_data import get_trade, TradeTable
from data_management.keeper.ZooKeeper import ZooKeeper

tqdm.pandas()


def net_pct_main(money_flow: pd.DataFrame) -> pd.Series:
    """
    主力净额 = 超大单净额 + 大单净额
    主力净占比 = 主力净额 / 成交额
    :param money_flow:
    :return:
    """
    return money_flow['net_pct_main']


def net_pct_xl(money_flow: pd.DataFrame) -> pd.Series:
    """
    超大单：大于等于50万股或者100万元的成交单
    超大单净占比 = 超大单净额 / 成交额
    :param money_flow:
    :param d:
    :return:
    """
    return money_flow['net_pct_xl']


def net_pct_l(money_flow: pd.DataFrame) -> pd.Series:
    """
    大单：大于等于10万股或者20万元且小于50万股或者100万元的成交单
    大单净占比 = 大单净额 / 成交额
    :return:
    """
    return money_flow['net_pct_l']


def net_pct_m(money_flow: pd.DataFrame) -> pd.Series:
    """
    中单：大于等于2万股或者4万元且小于10万股或者20万元的成交单
    中单净占比 = 中单净额 / 成交额
    :param money_flow:
    :param d:
    :return:
    """
    return money_flow['net_pct_m']


def net_pct_s(money_flow: pd.DataFrame) -> pd.Series:
    """
    小单：小于2万股或者4万元的成交单
    小单净占比 = 小单净额 / 成交额
    :param money_flow:
    :param d:
    :return:
    """
    return money_flow['net_pct_s']


def net_pct_main_avg(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """
    :param money_flow:
    :return:
    """
    mf = net_pct_main(money_flow)
    agg_mf = mf.groupby(level=1).progress_apply(lambda x: x.rolling(d).mean())
    agg_mf.name = 'net_pct_main_avg_{}'.format(d)
    return agg_mf


def net_pct_main_std(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    mf = net_pct_main(money_flow)
    agg_mf = mf.groupby(level=1).progress_apply(lambda x: x.rolling(d).std())
    agg_mf.name = 'net_pct_main_std_{}'.format(d)
    return agg_mf


def net_pct_main_z_score(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """

    mean = net_pct_main_avg(money_flow, d)
    std = net_pct_main_std(money_flow, d)
    factor = mean / std
    factor.name = 'net_pct_main_z_score_{}'.format(d)
    return factor


def net_pct_main_sum(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    mf = net_pct_main(money_flow)
    agg_mf = mf.groupby(level=1).progress_apply(lambda x: x.rolling(d).sum())
    agg_mf.name = 'net_pct_main_sum_{}'.format(d)
    return agg_mf


def delta_net_pct_main_sum(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    agg_mf = net_pct_main_sum(money_flow, d)
    diff = agg_mf.groupby(level=1).diff()
    diff.name = 'delta_net_pct_main_sum_{}'.format(d)
    return diff


def net_pct_xl_avg(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    mf = net_pct_xl(money_flow)
    agg_mf = mf.groupby(level=1).progress_apply(lambda x: x.rolling(d).mean())
    agg_mf.name = 'net_pct_xl_avg_{}'.format(d)
    return agg_mf


def net_pct_xl_std(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param trading_date:
    :param d:
    :return:
    """
    mf = net_pct_xl(money_flow)
    agg_mf = mf.groupby(level=1).progress_apply(lambda x: x.rolling(d).std())
    agg_mf.name = 'net_pct_xl_std_{}'.format(d)
    return agg_mf


def net_pct_xl_z_score(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param trading_date:
    :param d:
    :return:
    """
    mean = net_pct_xl_avg(money_flow, d)
    std = net_pct_xl_std(money_flow, d)
    factor = mean / std
    factor.name = 'net_pct_xl_z_score_{}'.format(d)
    return factor


def net_pct_xl_sum(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param trading_date:
    :param d:
    :return:
    """
    mf = net_pct_xl(money_flow)
    agg_mf = mf.groupby(level=1).progress_apply(lambda x: x.rolling(d).sum())
    agg_mf.name = 'net_pct_xl_sum_{}'.format(d)
    return agg_mf


def delta_net_pct_xl_sum(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    agg_mf = net_pct_xl_sum(money_flow, d)
    diff = agg_mf.groupby(level=1).diff()
    diff.name = 'delta_net_pct_xl_sum_{}'.format(d)
    return diff


def net_pct_l_avg(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    mf = net_pct_l(money_flow)
    agg_mf = mf.groupby(level=1).progress_apply(lambda x: x.rolling(d).mean())
    agg_mf.name = 'net_pct_l_avg_{}'.format(d)
    return agg_mf


def net_pct_l_std(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    mf = net_pct_l(money_flow)
    agg_mf = mf.groupby(level=1).progress_apply(lambda x: x.rolling(d).std())
    agg_mf.name = 'net_pct_l_std_{}'.format(d)
    return agg_mf


def net_pct_l_z_score(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    mean = net_pct_l_avg(money_flow, d)
    std = net_pct_l_std(money_flow, d)
    factor = mean / std
    factor.name = 'net_pct_l_z_score_{}'.format(d)
    return factor


def net_pct_l_sum(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    mf = net_pct_l(money_flow)
    agg_mf = mf.groupby(level=1).progress_apply(lambda x: x.rolling(d).sum())
    agg_mf.name = 'net_pct_l_sum_{}'.format(d)
    return agg_mf


def delta_net_pct_l_sum(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    agg_mf = net_pct_l_sum(money_flow, d)
    diff = agg_mf.groupby(level=1).diff()
    diff.name = 'delta_net_pct_l_sum_{}'.format(d)
    return diff


def net_pct_m_avg(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    mf = net_pct_m(money_flow)
    agg_mf = mf.groupby(level=1).progress_apply(lambda x: x.rolling(d).mean())
    agg_mf.name = 'net_pct_m_avg_{}'.format(d)
    return agg_mf


def net_pct_m_std(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    mf = net_pct_m(money_flow)
    agg_mf = mf.groupby(level=1).progress_apply(lambda x: x.rolling(d).std())
    agg_mf.name = 'net_pct_m_std_{}'.format(d)
    return agg_mf


def net_pct_m_z_score(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    mean = net_pct_m_avg(money_flow, d)
    std = net_pct_m_std(money_flow, d)
    factor = mean / std
    factor.name = 'net_pct_m_z_score_{}'.format(d)
    return factor


def net_pct_m_sum(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    mf = net_pct_m(money_flow)
    agg_mf = mf.groupby(level=1).progress_apply(lambda x: x.rolling(d).sum())
    agg_mf.name = 'net_pct_m_sum_{}'.format(d)
    return agg_mf


def delta_net_pct_m_sum(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    agg_mf = net_pct_m_sum(money_flow, d)
    diff = agg_mf.groupby(level=1).diff()
    diff.name = 'delta_net_pct_m_sum_{}'.format(d)
    return diff


def net_pct_s_avg(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    mf = net_pct_s(money_flow)
    agg_mf = mf.groupby(level=1).progress_apply(lambda x: x.rolling(d).mean())
    agg_mf.name = 'net_pct_s_avg_{}'.format(d)
    return agg_mf


def net_pct_s_std(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    mf = net_pct_s(money_flow)
    agg_mf = mf.groupby(level=1).progress_apply(lambda x: x.rolling(d).std())
    agg_mf.name = 'net_pct_s_std_{}'.format(d)
    return agg_mf


def net_pct_s_z_score(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    mean = net_pct_s_avg(money_flow, d)
    std = net_pct_s_std(money_flow, d)
    factor = mean / std
    factor.name = 'net_pct_s_z_score_{}'.format(d)
    return factor


def net_pct_s_sum(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    mf = net_pct_s(money_flow)
    agg_mf = mf.groupby(level=1).progress_apply(lambda x: x.rolling(d).sum())
    agg_mf.name = 'net_pct_s_sum_{}'.format(d)
    return agg_mf


def delta_net_pct_s_sum(money_flow: pd.DataFrame, d: int = 20) -> pd.Series:
    """

    :param money_flow:
    :param d:
    :return:
    """
    agg_mf = net_pct_s_sum(money_flow, d)
    diff = agg_mf.groupby(level=1).diff()
    diff.name = 'delta_net_pct_s_sum_{}'.format(d)
    return diff


def avg_net_amount_main(money_flow: pd.DataFrame, d: int = 20):
    """

    :param money_flow:
    :param d:
    :return:
    """
    factor = money_flow['net_amount_main'].groupby(level=1).progress_apply(lambda x: x.rolling(d).mean())
    factor.name = 'avg_net_amount_main_{}'.format(d)
    return factor


def avg_net_amount_xl(money_flow: pd.DataFrame, d: int = 20):
    """

    :param money_flow:
    :param d:
    :return:
    """
    factor = money_flow['net_amount_xl'].groupby(level=1).progress_apply(lambda x: x.rolling(d).mean())
    factor.name = 'avg_net_amount_xl_{}'.format(d)
    return factor


def avg_net_amount_l(money_flow: pd.DataFrame, d: int = 20):
    """

    :param money_flow:
    :param d:
    :return:
    """
    factor = money_flow['net_amount_l'].groupby(level=1).progress_apply(lambda x: x.rolling(d).mean())
    factor.name = 'avg_net_amount_l_{}'.format(d)
    return factor


def avg_net_amount_m(money_flow: pd.DataFrame, d: int = 20):
    """

    :param money_flow:
    :param d:
    :return:
    """
    factor = money_flow['net_amount_m'].groupby(level=1).progress_apply(lambda x: x.rolling(d).mean())
    factor.name = 'avg_net_amount_m_{}'.format(d)
    return factor


def avg_net_amount_s(money_flow: pd.DataFrame, d: int = 20):
    """

    :param money_flow:
    :param d:
    :return:
    """
    factor = money_flow['net_amount_s'].groupby(level=1).progress_apply(lambda x: x.rolling(d).mean())
    factor.name = 'avg_net_amount_s_{}'.format(d)
    return factor


def ideal_main_moneyflow(daily_market: Union[pd.DataFrame, pd.Series],
                         money_flow: Union[pd.DataFrame, pd.Series],
                         d: int = 20, percentage: float = 0.25):
    """
    参考ideal_turnover改造而成
    (1) 对选定股票，回溯取其最近20个交易日数据，计算股票main pct
    (2) 选择收盘价较高的25%有效交易日，计算振幅均值得到高价振幅因子V_high
    (3) 选择收盘价较低的25%有效交易日，计算振幅均值得到低价振幅因子V_low
    (4) 将高价振幅因子V_high与低价振幅因子V_low作差，得到理想振幅因子 V = V_high - V_low
    :return:
    """
    num = int(d * percentage)

    def func(df):
        arr = np.empty(len(df))
        arr[:] = np.NaN
        close = df['adj_close'].values
        turnover = df['net_pct_main'].values
        for i in range(0, len(df) - d + 1):
            index = np.argsort(close[i: i + d])
            v_low = np.mean(turnover[i: i + d][index[:num]])
            v_high = np.mean(turnover[i: i + d][index[-num:]])
            arr[i + d - 1] = v_high - v_low
        return pd.Series(arr, index=df.index)

    if isinstance(daily_market, pd.DataFrame):
        adj_close = daily_market['adj_close'].to_frame()
    else:
        adj_close = daily_market.to_frame('adj_close')

    mf = net_pct_main(money_flow)

    data = adj_close.join(mf)
    # todo go parallel
    factor = data.groupby(level=1).progress_apply(func)
    if factor.index.nlevels == 3:
        factor = factor.droplevel(0).sort_index()
    else:
        factor = factor.swaplevel(0, 1).sort_index()
    factor.name = 'ideal_main_moneyflow_{}_{}'.format(d, percentage)
    return factor


def cash_inflow_strength2_large(tushare_moneyflow: pd.DataFrame, window: int) -> pd.Series:
    """

    :param tushare_moneyflow:
    :param window:
    :return:
    """
    diff_ = tushare_moneyflow['buy_lg_amount'] - tushare_moneyflow['sell_lg_amount']
    sum_ = tushare_moneyflow['buy_lg_amount'] + tushare_moneyflow['sell_lg_amount']
    factor = diff_.groupby(level=1).rolling(window).sum() / sum_.groupby(level=1).rolling(window).sum()
    if factor.index.nlevels == 3:
        factor = factor.droplevel(0).sort_index()
    factor.name = 'cash_inflow_strength2_large_{}'.format(window)
    return factor


def cash_inflow_strength3_large(tushare_moneyflow: pd.DataFrame, window: int) -> pd.Series:
    """

    :param tushare_moneyflow:
    :param window:
    :return:
    """
    diff_ = tushare_moneyflow['buy_lg_amount'] - tushare_moneyflow['sell_lg_amount']
    abs_diff = diff_.abs()
    factor = diff_.groupby(level=1).rolling(window).sum() / abs_diff.groupby(level=1).rolling(window).sum()
    if factor.index.nlevels == 3:
        factor = factor.droplevel(0).sort_index()
    factor.name = 'cash_inflow_strength3_large_{}'.format(window)
    return factor


def cash_inflow_strength2_small(tushare_moneyflow: pd.DataFrame, window: int) -> pd.Series:
    """

    :param tushare_moneyflow:
    :param window:
    :return:
    """
    diff_ = tushare_moneyflow['buy_sm_amount'] - tushare_moneyflow['sell_sm_amount']
    sum_ = tushare_moneyflow['buy_sm_amount'] + tushare_moneyflow['sell_sm_amount']
    factor = diff_.groupby(level=1).rolling(window).sum() / sum_.groupby(level=1).rolling(window).sum()
    if factor.index.nlevels == 3:
        factor = factor.droplevel(0).sort_index()
    factor.name = 'cash_inflow_strength2_small_{}'.format(window)
    return factor


def cash_inflow_strength3_small(tushare_moneyflow: pd.DataFrame, window: int) -> pd.Series:
    """

    :param tushare_moneyflow:
    :param window:
    :return:
    """
    diff_ = tushare_moneyflow['buy_sm_amount'] - tushare_moneyflow['sell_sm_amount']
    abs_diff = diff_.abs()
    factor = diff_.groupby(level=1).rolling(window).sum() / abs_diff.groupby(level=1).rolling(window).sum()
    if factor.index.nlevels == 3:
        factor = factor.droplevel(0).sort_index()
    factor.name = 'cash_inflow_strength3_small_{}'.format(window)
    return factor


