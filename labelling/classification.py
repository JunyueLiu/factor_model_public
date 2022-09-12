from typing import Tuple, Union

import numpy as np
import pandas as pd

from data_management.dataIO import market_data
from data_management.dataIO.component_data import get_index_component, IndexTicker
from data_management.dataIO.market_data import Freq
from data_management.dataIO.trading_calendar import trading_dates_offsets, get_trading_date, Market
from data_management.keeper.LabelGuardian import LabelGuardian
from factor_zoo.utils import market_filter_in
from labelling.regreesion import forward_ret_labeler, forward_excess_ret_labler, sharpe_labeler, calmar_labeler, \
    forward_path_dependent_ret_labeler, hold_eliminate_paused_and_limit


def quantile_label(x: pd.Series, bins):
    quantile_factor = pd.qcut(x, bins, labels=False, duplicates='drop') + 1
    return quantile_factor


def forward_ret_bit_labeler(daily_market: pd.DataFrame, windows,
                            price_type='adj_open', add_label_end=True, ):
    if add_label_end:
        label, label_end = forward_ret_labeler(daily_market, windows, price_type, add_label_end=add_label_end)
    else:
        label = forward_ret_labeler(daily_market, windows, price_type, add_label_end=add_label_end)
    label = (label > 0).astype(int)
    if add_label_end:
        label_end.name = label.name + '_label_end'
        return label, label_end
    else:
        return label


def forward_ret_group_labeler(daily_market: pd.DataFrame,
                              windows,
                              price_type='adj_open',
                              bins: int = 5,
                              add_label_end=True,
                              universe=None,
                              universe_name=''
                              ) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """

    Parameters
    ----------
    daily_market
    windows
    price_type
    bins
    add_label_end
    universe
    universe_name

    Returns
    -------

    """
    if add_label_end:
        label, label_end = forward_ret_labeler(daily_market, windows, price_type, add_label_end=add_label_end)
    else:
        label = forward_ret_labeler(daily_market, windows, price_type, add_label_end=add_label_end)
    if universe:
        label = market_filter_in(label, universe)
        if add_label_end:
            label_end = market_filter_in(label_end, universe)
    label = label.groupby(level=0).transform(lambda x: quantile_label(x, bins))
    if universe:
        label.name = 'forward_{}_D_{}_ret_{}_{}_group'.format(windows, price_type, bins, universe_name)
    else:
        label.name = 'forward_{}_{}_ret_{}_group'.format(windows, price_type, bins)
    if add_label_end:
        label_end.name = label.name + '_label_end'
        return label, label_end
    else:
        return label


def forward_demean_ret_group_labeler(daily_market: pd.DataFrame,
                                     windows,
                                     trading_date,
                                     resample='M',
                                     bins: int = 5,
                                     universe=None,
                                     universe_name=''
                                     ) -> pd.Series:
    label = forward_ret_labeler(daily_market, windows, trading_date, resample)

    label = label.groupby(level=0).transform(lambda x: x - x.mean())
    if universe:
        label = market_filter_in(label, universe)
    label = label.groupby(level=0).transform(lambda x: quantile_label(x, bins))
    label.name = 'forward_{}_{}_demean_ret_{}_{}_group'.format(windows, resample, bins, universe_name)
    return label


def forward_excess_ret_group_labler(daily_market: pd.DataFrame,
                                    benchmark: pd.DataFrame,
                                    windows,
                                    trading_date,
                                    resample='M',
                                    bins: int = 5,
                                    universe=None,
                                    universe_name=''

                                    ) -> pd.Series:
    label = forward_excess_ret_labler(daily_market, benchmark, windows, trading_date, resample)
    if universe:
        label = market_filter_in(label, universe)
    label = label.groupby(level=0).transform(lambda x: quantile_label(x, bins))
    label.name = 'forward_{}_{}_excess_ret_{}_{}_group'.format(windows, resample, bins, universe_name)
    return label


def sharpe_group_labeler(daily_market: pd.DataFrame, windows, trading_date, resample='M', bins=5,
                         universe=None,
                         universe_name=''):
    label = sharpe_labeler(daily_market, windows, trading_date, resample)
    if universe:
        label = market_filter_in(label, universe)
    label = label.groupby(level=0).transform(lambda x: quantile_label(x, bins))
    label.name = 'forward_{}_{}_sharpe_{}_{}_group'.format(windows, resample, bins, universe_name)
    return label


def calmar_group_labeler(daily_market: pd.DataFrame,
                         windows,
                         trading_date,
                         resample='M', bins: int = 5,
                         universe=None,
                         universe_name=''):
    label = calmar_labeler(daily_market, windows, trading_date, resample)
    if universe:
        label = market_filter_in(label, universe)
    label = label.groupby(level=0).transform(lambda x: quantile_label(x, bins))
    label.name = 'forward_{}_{}_calmar_{}_{}_group'.format(windows, resample, bins, universe_name)
    return label


def triple_barrier_labeler(daily_market: pd.DataFrame,
                           windows, trading_date=None, resample='M', up=0.2, down=0.05):
    offsets = trading_dates_offsets(trading_date, resample)

    def tbl(series: pd.Series, up, down):
        # returns = series.droplevel(1).pct_change(1).dropna()
        arr = series.values
        if resample == 'M' and len(arr) <= 10:
            return np.nan
        elif resample == 'D' and len(arr) <= int(0.4 * windows):
            return np.nan
        else:
            start = arr[0]
            max_p = arr.max()
            min_p = arr.min()
            if max_p >= start * (1 + up) and min_p <= start * (1 - down):
                argmax = arr.argmax()
                argmin = arr.argmin()
                if argmax <= argmin:
                    return 2
                else:
                    return 0
            elif max_p >= start * (1 + up):
                return 2
            elif min_p <= start * (1 - down):
                return 0
            else:
                return 1

    label = daily_market['close'].groupby(level=1). \
        resample(windows * offsets, level=0, label='left', closed='right').agg(lambda x: tbl(x, up, down))

    label.name = 'forward_{}_{}_tbl_{}_{}_group'.format(windows, resample, up, down)
    return label


def forward_path_dependent_group_labeler(daily_market: pd.DataFrame,
                                         windows,
                                         trading_date,
                                         path_dependent_func,
                                         resample: str,
                                         bins: int = 5,
                                         universe: dict = None,
                                         universe_name: str = None,
                                         **path_dependent_func_kwargs
                                         ):
    label = forward_path_dependent_ret_labeler(daily_market, windows, trading_date, path_dependent_func, resample,
                                               **path_dependent_func_kwargs)
    if universe:
        if not universe_name:
            raise ValueError('must specify universe_name')

        label = market_filter_in(label, universe)
    label = label.groupby(level=0).transform(lambda x: quantile_label(x, bins))
    label.name = 'forward_{}_{}_{}_{}_{}_group'.format(windows, resample,
                                                       path_dependent_func.__name__, bins,
                                                       universe_name)
    return label


def forward_buy_and_hold_zz500_group_labeler(daily_market: pd.DataFrame,
                                             windows,
                                             trading_date,
                                             resample: str,
                                             bins: int = 5,
                                             universe_dict=None
                                             ) -> pd.Series:
    label = forward_path_dependent_group_labeler(daily_market, windows, trading_date,
                                                 hold_eliminate_paused_and_limit, resample,
                                                 bins, universe_dict, 'zz500')
    return label


if __name__ == '__main__':
    config_path = '../cfg/data_input.ini'
    daily_market = market_data.get_bars(start_date='2014-01-01', freq=Freq.D1, config_path=config_path,
                                        eod_time_adjust=False)
    trading_list = get_trading_date(Market.AShares, '2014-01-01', config_path=config_path)
    zz500 = get_index_component(IndexTicker.zz500, config_path=config_path)
    l = forward_buy_and_hold_zz500_group_labeler(daily_market, 1, trading_list, 'W-2', 5, zz500)
    guardian = LabelGuardian('../cfg/label_guardian_setting.ini')
    guardian.save_label_values('classification', l, True)
