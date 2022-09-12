import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd
import tqdm

from arctic import Arctic
from data_management.dataIO.component_data import get_bars
from factor_zoo.industry_factor_zoo.changepoint_detection import get_single_asset_changepoint_detection


def _f(c, groups, store, lookback_window_length, use_kM_hyp_to_initialise_kC=True):
    lib_name = 'cpd_{}_{}'.format(lookback_window_length, use_kM_hyp_to_initialise_kC)
    res = get_single_asset_changepoint_detection(groups.get_group(c), lookback_window_length, '2010-01-01',
                                                 use_kM_hyp_to_initialise_kC=use_kM_hyp_to_initialise_kC)
    if not store.library_exists(lib_name):
        store.initialize_library(lib_name)
    last_update = datetime.datetime.now()
    store[lib_name].delete(c)
    store[lib_name].append(c, res,
                           metadata={'last_update': last_update, 'source': 'in-house-calculation'},
                           prune_previous_version=False, upsert=True)


def cal_save(store: Arctic, df: pd.DataFrame, lookback_window_length: int,
             use_kM_hyp_to_initialise_kC=True):
    codes = df.index.get_level_values(1).drop_duplicates()
    groups = df.groupby(level=1)
    f = partial(_f,
                groups=groups, store=store,
                lookback_window_length=lookback_window_length, use_kM_hyp_to_initialise_kC=use_kM_hyp_to_initialise_kC
                )
    with ProcessPoolExecutor(5) as executor:
        list(tqdm.tqdm(executor.map(f, codes, chunksize=1), total=len(codes)))


if __name__ == '__main__':
    df = get_bars()
    cal_save(Arctic('localhost'), df, 21, )
