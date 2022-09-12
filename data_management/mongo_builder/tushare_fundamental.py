

import datetime
import warnings
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial

import jqdatasdk
import pandas as pd
import numpy as np
import tqdm
import tushare as ts
from jqdatasdk import get_all_securities

from arctic import Arctic, VERSION_STORE
from arctic.exceptions import NoDataFoundException
from data_management.mongo_builder.decorator import joinquant_retry, tushare_retry
from data_management.mongo_builder.utils import market_batch_insert_df_to_arctic, market_batch_update_df_to_arctic, \
    fundamental_batch_insert_df_to_arctic
from data_management.tokens import jq_user, jq_password, token


@tushare_retry
def get_tushare_quick_fin(ts_code: str, pro):
    df = pro.express(ts_code=ts_code)
    # ['code', 'start_date', 'end_date', 'pub_date',
    #  'StockName', 'NumQuiTraFinReport',
    #  'TotOpeReve', 'TotOpeReveLast', 'RatTotOpeReve', 'OpeProf',
    #  'OpeProfLast', 'RatOpeProf', 'TotProf', 'TotProfLast', 'RatTotProf',
    #  'NetProfPareComp', 'NetProfPareCompLast', 'RatNetProfPareComp',
    #  'NetProf', 'NetProfLast', 'RatNetProf', 'EPS', 'EPSLast', 'RatEPS',
    #  'RetuEqui', 'RetuEquiLast', 'RatRetuEqui', 'TotAsse', 'TotAsseBegin',
    #  'RatTotAsse', 'EquiPareComp', 'EquiPareCompBegin', 'RatEquiPareComp',
    #  'SharehEqui', 'SharehEquiBegin', 'RatSharehEqui', 'NetAsseShaPareComp',
    #  'NetAsseShaPareCompBegin', 'RatNetAsseShaPareComp', 'NetAsseSha',
    #  'NetAsseShaBegin', 'RatNetAsseSha', 'ReaModiFinResu']
    df = df.rename(columns={'ts_code': 'code',
                            'ann_date': 'pub_date',
                            })
    df['pub_date'] = pd.to_datetime(df['pub_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df['start_date'] = df['end_date'].apply(lambda x: x.replace(month=1, day=1))
    df = df.fillna(value=np.nan)
    df['bps'] = df['bps'].astype(float)
    df['perf_summary'] = df['perf_summary'].astype(str)
    return df



def build_quick_fin(arctic_store):
    # tickers = get_all_ticker()
    MAX_WORKER = 10
    ts.set_token(token)
    pro = ts.pro_api()
    jqdatasdk.auth(jq_user, jq_password)
    df = get_all_securities(types=['stock'], date=None)  # type: pd.DataFrame
    df.index = [c.replace('XSHE', 'SZ').replace('XSHG', 'SH') for c in df.index]
    codes = df.index
    func = partial(get_tushare_quick_fin, pro=pro)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        result = list(tqdm.tqdm(executor.map(func, codes), total=len(codes)))
    data = pd.concat(result, ignore_index=True)
    # for c in data.columns:
    #     if c in ['code', 'date']:
    #         continue
    #     data[c] = data[c].astype(float)
    data = data.sort_values(['code', 'pub_date', 'start_date', 'end_date'])
    data = data.reset_index(drop=True)
    fundamental_batch_insert_df_to_arctic(data, arctic_store, 'tushare_quick_fin')

if __name__ == '__main__':
    build_quick_fin(Arctic('localhost'))