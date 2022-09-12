import datetime
from collections import OrderedDict
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial

import jqdatasdk
import pandas as pd
import tqdm

from arctic import Arctic
from data_management.mongo_builder.decorator import joinquant_retry
from data_management.tokens import jq_user, jq_password

MAX_WORKER = 3


@joinquant_retry
def get_index_stock(date, jq_index_symbol):
    stocks = jqdatasdk.get_index_stocks(jq_index_symbol, date)
    stocks = [ticker.replace('XSHE', 'SZ').replace('XSHG', 'SH') for ticker in stocks]
    return pd.to_datetime(date), stocks


@joinquant_retry
def get_index_weights(date, jq_index_symbol):
    df = jqdatasdk.get_index_weights(jq_index_symbol, date)  # type: pd.DataFrame
    df = df.reset_index()
    df['index'] = df['index'].apply(lambda x: x.replace('XSHE', 'SZ').replace('XSHG', 'SH'))
    df = df.rename(columns={'index': 'code'})
    return df


def build_index_stocks(arctic_store, index_symbol: str, ):
    jq_index_symbol = index_symbol.replace('SZ', 'XSHE') if 'SZ' in index_symbol else index_symbol.replace('SH', 'XSHG')
    jqdatasdk.auth(jq_user, jq_password)
    trading_date = jqdatasdk.get_all_trade_days()
    trading_date = trading_date[trading_date <= datetime.datetime.now().date()]

    func = partial(get_index_stock, jq_index_symbol=jq_index_symbol)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        result = list(tqdm.tqdm(executor.map(func, trading_date), total=len(trading_date)))
    executor.shutdown()
    d = {r[0]: r[1] for r in result}
    d = OrderedDict(sorted(d.items()))
    last_update = list(d.keys())[-1]
    if arctic_store.library_exists('index_stocks'):
        arctic_store['index_stocks'].delete(index_symbol)
        arctic_store['index_stocks'].write(index_symbol, d,
                                           metadata={'last_update': last_update, 'source': 'joinquant'})
    else:
        arctic_store.initialize_library('index_stocks')
        arctic_store['index_stocks'].write(index_symbol, d,
                                           metadata={'last_update': last_update, 'source': 'joinquant'})


def build_index_weights(arctic_store, index_symbol: str, ):
    jq_index_symbol = index_symbol.replace('SZ', 'XSHE') if 'SZ' in index_symbol else index_symbol.replace('SH', 'XSHG')
    jqdatasdk.auth(jq_user, jq_password)
    trading_date = jqdatasdk.get_all_trade_days()
    trading_date = trading_date[trading_date <= datetime.datetime.now().date()]

    func = partial(get_index_weights, jq_index_symbol=jq_index_symbol)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        result = list(tqdm.tqdm(executor.map(func, trading_date), total=len(trading_date)))
    executor.shutdown()
    # d = {r[0]: r[1] for r in result}
    # d = OrderedDict(sorted(d.items()))
    data = pd.concat(result, ignore_index=True)
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index(['date', 'code']).sort_index()
    data = data[~data.index.duplicated(keep='first')]
    last_update = data.index.get_level_values(0).drop_duplicates().sort_values()[-1]
    if arctic_store.library_exists('index_weights'):
        arctic_store['index_weights'].delete(index_symbol)
        arctic_store['index_weights'].append(index_symbol, data,
                                             metadata={'last_update': last_update, 'source': 'joinquant'},
                                             prune_previous_version=False, upsert=True)
    else:
        arctic_store.initialize_library('index_weights')
        arctic_store['index_weights'].append(index_symbol, data,
                                             metadata={'last_update': last_update, 'source': 'joinquant'},
                                             prune_previous_version=False, upsert=True)

def build_bbg_index_weights(arctic_store: Arctic, csv_path: str, index_symbol):
    # XIN9
    df = pd.read_csv(csv_path, index_col=0)
    df.columns = df.columns.str.replace(' CG', '.SH').str.replace(' CS', '.SZ')
    df.index = pd.to_datetime(df.index)
    df = df.stack().to_frame('weight')
    df.index.names = ['date', 'code']
    last_update = df.index.get_level_values(0).drop_duplicates().sort_values()[-1]
    arctic_store['index_weights'].append(index_symbol, df,
                                         metadata={'last_update': last_update, 'source': 'bbg'},
                                         prune_previous_version=False, upsert=True)



def update_index_stocks(arctic_store, index_symbol: str):
    jq_index_symbol = index_symbol.replace('SZ', 'XSHE') if 'SZ' in index_symbol else index_symbol.replace('SH', 'XSHG')
    jqdatasdk.auth(jq_user, jq_password)
    last_update = arctic_store['index_stocks'].read_metadata(index_symbol).metadata['last_update'].date()

    trading_date = jqdatasdk.get_all_trade_days()
    trading_date = trading_date[(trading_date > last_update) & (trading_date <= datetime.datetime.now().date())]

    old_d = arctic_store['index_stocks'].read(index_symbol).data

    func = partial(get_index_stock, jq_index_symbol=jq_index_symbol)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        result = list(tqdm.tqdm(executor.map(func, trading_date), total=len(trading_date)))
    executor.shutdown()
    d = {r[0]: r[1] for r in result}
    d = OrderedDict(sorted(d.items()))
    try:
        last_update = list(d.keys())[-1]
        d = OrderedDict(sorted({**old_d, **d}.items()))

        arctic_store['index_stocks'].write(index_symbol, d,
                                           metadata={'last_update': last_update, 'source': 'joinquant'})
    except IndexError:
        pass


def update_index_weights(arctic_store, index_symbol: str, ):
    jq_index_symbol = index_symbol.replace('SZ', 'XSHE') if 'SZ' in index_symbol else index_symbol.replace('SH', 'XSHG')
    jqdatasdk.auth(jq_user, jq_password)
    last_update = arctic_store['index_weights'].read_metadata(index_symbol).metadata['last_update'].date()

    trading_date = jqdatasdk.get_all_trade_days()
    trading_date = trading_date[(trading_date > last_update) & (trading_date <= datetime.datetime.now().date())]

    func = partial(get_index_weights, jq_index_symbol=jq_index_symbol)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        result = list(tqdm.tqdm(executor.map(func, trading_date), total=len(trading_date)))
    executor.shutdown()
    data = pd.concat(result, ignore_index=True)
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index(['date', 'code']).sort_index()
    data = data[~data.index.duplicated(keep='first')]
    data = data.loc[pd.to_datetime(last_update) + pd.Timedelta(seconds=1):]
    try:
        last_update = data.index.get_level_values(0).drop_duplicates().sort_values()[-1]
        arctic_store['index_weights'].append(index_symbol, data,
                                             metadata={'last_update': last_update, 'source': 'joinquant'},
                                             prune_previous_version=False, upsert=True)
    except IndexError:
        pass


if __name__ == '__main__':
    store = Arctic('localhost')
    # build_index_stocks(store, '000300.SH')
    # build_index_stocks(store, '000905.SH')
    # build_index_stocks(store, '000906.SH')
    # build_index_stocks(store, '000852.SH')
    # build_index_stocks(store, '399004.SZ')
    # build_index_stocks(store, '000016.SH')
    # update_index_stocks(store, '000300.SH')
    # update_index_stocks(store, '000905.SH')
    # update_index_stocks(store, '000906.SH')
    # update_index_stocks(store, '000852.SH')

    # build_index_weights(store, '000300.SH')
    # build_index_weights(store, '000905.SH')
    # build_index_weights(store, '000906.SH')
    # build_index_weights(store, '000852.SH')
    # build_index_weights(store, '000016.SH')
    # build_index_weights(store, '399004.SZ')

    # update_index_weights(store, '000300.SH')
    # update_index_weights(store, '000905.SH')
    # update_index_weights(store, '000906.SH')
    # update_index_weights(store, '000906.SH')
    # update_index_weights(store, '000852.SH')
    build_bbg_index_weights(store, '../../data/csv/XIN9I_Weights.csv', 'XIN9')
