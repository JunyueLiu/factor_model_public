from functools import partial
from typing import Dict

import pandas as pd
import numpy as np

from data_management.dataIO import index_data, market_data
from data_management.dataIO.component_data import get_index_component
from data_management.dataIO.index_data import IndexTicker
from data_management.pandas_utils.parallel import time_series_parallel_apply
from factor_zoo.utils import rolling_regression_beta


def rolling_beta(daily_market: pd.DataFrame, index_daily: pd.DataFrame, window=20):
    """

    Parameters
    ----------
    daily_market
    index_daily
    window

    Returns
    -------

    """
    index_code = index_daily['code'].iloc[0]
    stock_ret = daily_market['adj_close'].unstack().pct_change().stack()
    index_ret = index_daily['close'].pct_change()
    merged = stock_ret.to_frame('ret').join(index_ret.to_frame('index_ret'))

    roll_reg = partial(rolling_regression_beta, y_col='ret', X_cols=['index_ret'], N=window)
    factor = time_series_parallel_apply(merged, roll_reg)
    factor = factor.sort_index()
    factor.name = 'rolling_beta_{}_{}'.format(index_code, window)
    return factor


def is_component(daily_market: pd.DataFrame, component_dict: Dict, universe_name: str) -> pd.Series:

    def _is_component_logic(x, u_dict):

        date = x.index.get_level_values(0)[0]
        comp = set(u_dict.get(date, []))
        if len(comp) > 0:
            return pd.Series([c in comp for c in x.index.get_level_values(1)],
                             index=x.index.get_level_values(1))
        else:
            return pd.Series(False, index=x.index.get_level_values(1))

    factor = daily_market.groupby(level=0).apply(lambda x: _is_component_logic(x, component_dict))
    factor = factor.astype(int)
    factor.name = 'is_{}_component'.format(universe_name)
    return factor


