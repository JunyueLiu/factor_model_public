import configparser
import json
import pickle
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any

import pandas as pd
import tqdm

from arctic import Arctic
from arctic.arctic import APPLICATION_NAME
from arctic.exceptions import NoDataFoundException


def get_arctic_store(arctic_config: configparser.ConfigParser) -> Arctic:
    mongo_host = arctic_config['Arctic'].get('mongo_host', 'localhost')
    app_name = arctic_config['Arctic'].get('app_name', APPLICATION_NAME)
    username = arctic_config['Arctic'].get('username', None)
    password = arctic_config['Arctic'].get('password', None)

    if username is None and password is None:
        store = Arctic(mongo_host, app_name)
    elif username is not None and password is not None:
        store = Arctic(mongo_host, app_name, username=username, password=password)
    else:
        raise ValueError('Must provide both username and password.')
    return store


def read_arctic_version_store(store: Arctic, library, instruments, max_workers=10, set_index=True, verbose=1, date_range=None) -> pd.DataFrame:
    def _read(symbol):
        try:
            df = store[library].read(symbol, date_range=date_range).data
            if 'code' not in df:
                df['code'] = symbol
            return df

        except NoDataFoundException:
            return pd.DataFrame()

    if instruments is None:
        instruments = store[library].list_symbols()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if verbose == 1:
            results = list(tqdm.tqdm(executor.map(_read, instruments), total=len(instruments)))
        else:
            results = list(executor.map(_read, instruments))
    data = pd.concat(results)
    if data.empty:
        return data
    if set_index:
        data = data.set_index('code', append=True)
        data = data.sort_index()
    return data


def read_pickle(path: str) -> Any:
    with open(path, 'rb') as handle:
        dic = pickle.load(handle)
        return dic


def save_pickle(d: Any, path: str):
    with open(path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_json(path: str) -> Any:
    with open(path, 'rb') as handle:
        dic = json.load(handle)
        return dic
