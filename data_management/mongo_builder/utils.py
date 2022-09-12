import datetime
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial

import tqdm


def market_batch_insert_df_to_arctic(data, arctic_store, collection_name, source='tushare'):
    MAX_WORKER = 10
    data = data.sort_values('code')
    codes = data['code'].drop_duplicates().sort_values().to_list()

    def _insert(code, data_, arctic_store_):
        df = data_[data_['code'] == code]
        df = df.sort_index()
        last_update = df.index[-1]
        if arctic_store_.library_exists(collection_name):
            arctic_store_[collection_name].delete(code)
            arctic_store_[collection_name].append(code, df, metadata={'last_update': last_update,
                                                                      'source': source},
                                                  prune_previous_version=False, upsert=True)
        else:
            arctic_store_.initialize_library(collection_name)
            arctic_store_[collection_name].append(code, df, metadata={'last_update': last_update,
                                                                      'source': source},
                                                  prune_previous_version=False, upsert=True)

    func = partial(_insert, data_=data, arctic_store_=arctic_store)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, codes), total=len(codes)))


def market_batch_update_df_to_arctic(data, arctic_store, collection_name, source='tushare'):
    MAX_WORKER = 10
    data = data.sort_values('code')
    codes = data['code'].drop_duplicates().sort_values().to_list()

    def _insert(code, data_, arctic_store_):
        df = data_[data_['code'] == code]
        df = df.sort_index()
        last_update = df.index[-1]
        arctic_store_[collection_name].append(code, df, metadata={'last_update': last_update,
                                                                  'source': source},
                                              prune_previous_version=False, upsert=True)

    func = partial(_insert, data_=data, arctic_store_=arctic_store)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, codes), total=len(codes)))


def fundamental_batch_insert_df_to_arctic(data, arctic_store, collection_name, source='csmar'):
    MAX_WORKER = 10
    data = data.sort_values('code')
    codes = data['code'].drop_duplicates().sort_values().to_list()

    def _insert(code):
        df = data[data['code'] == code]
        # df = df.sort_index()
        if len(df) == 0:
            return
        df = df.sort_values('pub_date')
        last_update = df['pub_date'].iloc[-1]
        if arctic_store.library_exists(collection_name):
            arctic_store[collection_name].delete(code)
            arctic_store[collection_name].append(code, df, metadata={'last_update': last_update,
                                                                     'source': source},
                                                 prune_previous_version=False, upsert=True)
        else:
            arctic_store.initialize_library(collection_name)
            arctic_store[collection_name].append(code, df, metadata={'last_update': last_update,
                                                                     'source': source},
                                                 prune_previous_version=False, upsert=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(_insert, codes), total=len(codes)))

def get_latest_last_update_date(arctic_store, collection):
    last_update_date = datetime.datetime.fromtimestamp(0)
    for s in arctic_store[collection].list_symbols():
        tmp = arctic_store[collection].read_metadata(s).metadata['last_update']
        last_update_date = max(last_update_date, tmp)
    return last_update_date