from typing import Tuple

import pandas as pd
import numpy as np

from arctic import Arctic
from data_management.dataIO.component_data import get_index_component, IndexTicker
from factor_zoo.beta import is_component
from labelling.intraday_labelling.regression import intraday_sharpe, intraday_sharpe_cap_industry_neutralized


def intraday_sharpe_group(store: Arctic, freq: str, windows: int = 5, bins: int = 5) -> Tuple[pd.Series, pd.Series]:
    sharpe, end = intraday_sharpe(store, freq, windows)
    sharpe = sharpe.groupby(level=0).transform(lambda x: pd.qcut(x, bins, labels=False, duplicates='drop') + 1)
    sharpe.name = 'forward_intraday_sharpe_{}_{}_D_{}_group'.format(freq, windows, bins)
    end.name = '{}_label_end'.format(sharpe.name)
    return sharpe, end


def intraday_sharpe_cap_industry_neutralized_group(store: Arctic, freq: str, config_path: str, windows: int = 5,
                                                   bins: int = 5) -> Tuple[pd.Series, pd.Series]:
    sharpe, end = intraday_sharpe_cap_industry_neutralized(store, freq, config_path, windows)
    sharpe = sharpe.groupby(level=0).transform(lambda x: pd.qcut(x, bins, labels=False, duplicates='drop') + 1)
    sharpe.name = 'forward_intraday_sharpe_cap_industry_neutralized_group_{}_{}_D_{}_group'.format(freq, windows, bins)
    end.name = '{}_label_end'.format(sharpe.name)
    return sharpe, end


def intraday_sharpe_component_group(store: Arctic, freq: str, config_path: str, windows: int = 5,
                                    bins: int = 5):
    sharpe, end = intraday_sharpe(store, freq, windows)
    data = sharpe.to_frame()
    data['csi300'] = is_component(data, get_index_component(IndexTicker.csi300, config_path=config_path), 'csi300')
    data['zz500'] = is_component(data, get_index_component(IndexTicker.zz500, config_path=config_path), 'zz500')
    data['zz1000'] = is_component(data, get_index_component(IndexTicker.zz1000, config_path=config_path), 'zz1000')
    data['component'] = np.where(data['csi300'] == 1, 'csi300', np.where(data['zz500'] == 1, 'zz500', np.where(data['zz1000'] == 1, 'zz1000', 'other')) )
    data = data.set_index(['component'], append=True)
    l = data.groupby(level=[0, 2])[sharpe.name].transform(
        lambda x: pd.qcut(x, bins, labels=False, duplicates='drop') + 1)
    l = l.droplevel(2).sort_index()
    l.name = 'forward_intraday_sharpe_component_group_{}_{}D_{}'.format(freq, windows, bins)
    end.name = '{}_label_end'.format(l.name)
    return l, end




if __name__ == '__main__':
    # intraday_sharpe_group(Arctic('localhost'), '15m', 20)
    intraday_sharpe_component_group(Arctic('localhost'), '15m', '../../cfg/data_input.ini')
