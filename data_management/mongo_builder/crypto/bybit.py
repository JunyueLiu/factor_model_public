import datetime
import json
import logging
import re
import time
import urllib
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from urllib.error import HTTPError

import pandas as pd
import requests
import tqdm

from arctic import Arctic, TICK_STORE
from arctic.exceptions import NoDataFoundException, OverlappingDataException
from data_management.mongo_builder.crypto import SymbolType

HEADERS = headers = {
    "content-type": "text/plain charset=UTF-8",
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'accept-encoding': 'gzip, deflate, br',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36'
}


def get_all_symbols_info(symbol_type: SymbolType) -> pd.DataFrame:
    if symbol_type == SymbolType.SPOT:
        response = urllib.request.urlopen("https://api.bybit.com/spot/v1/symbols").read()
    else:
        response = urllib.request.urlopen("https://api.bybit.com/v2/public/symbols").read()
    data = json.loads(response)['result']
    data = pd.DataFrame(data)
    if symbol_type == SymbolType.USDT_PERP:
        data = data[data['quote_currency'] == 'USDT']
    elif symbol_type == SymbolType.INVERSE_PERP:
        data = data[data['quote_currency'] == 'USD']
    return data


def get_all_symbol(symbol_type: SymbolType, quote_asset=None):
    data = get_all_symbols_info(symbol_type)
    if quote_asset:
        data = data[data['quote_currency'] == quote_asset]
    symbols = data['name'].to_list()
    return symbols


from bs4 import BeautifulSoup


def get_all_url(symbol: str):
    # https://public.bybit.com/trading/BTCUSDT/
    r = urllib.request.urlopen('https://public.bybit.com/trading/{}/'.format(symbol)).read()
    soup = BeautifulSoup(r, 'html.parser')
    all_zips = soup.findAll('a')
    urls = ['https://public.bybit.com/trading/{}/{}'.format(symbol, a.get('href')) for a in all_zips]
    return urls


def get_all_update_url(symbol: str, start_date: str):
    urls = get_all_url(symbol)
    urls = [u for u in urls if urls if re.search(r'\d{4}-\d{2}-\d{2}', u).group() >= start_date]
    return urls


def get_trades(url: str):
    while True:
        try:
            data = pd.read_csv(url)
            data['timestamp'] = pd.to_datetime(data['timestamp'] * 1_000, unit='ms', utc=True)
            data = data.set_index('timestamp').sort_index()
            if len(data) == 0:
                return pd.DataFrame()
            return data
        except HTTPError as error:
            if error.code == 403:
                time.sleep(100)
            elif error.code == 404:
                return pd.DataFrame()
            else:
                print(error.code, url)


def get_all_trades(symbol: str):
    urls = get_all_url(symbol)
    data = []
    for url in urls:
        df = get_trades(url)
        data.append(df)
    data = pd.concat(data)
    return data


def get_funding_rate_history(symbol: str, start_date=None):
    base_url = 'https://api2.bybit.com/linear/funding-rate/list'

    page = 1
    last_page = 2
    data = []
    while page <= last_page:
        url = '{}?symbol={}&date=&export=false&page={}'.format(base_url, symbol, page)
        while True:
            req = requests.get(url, headers=HEADERS)
            if req.status_code == 200:
                res = req.json()['result']
                last_page = int(res['last_page'])
                df = pd.DataFrame(res['data'])
                data.append(df)
                page += 1
                break
            elif req.status_code == 403:
                time.sleep(10)
                logging.error(req.content)
            else:
                logging.error('{} {}'.format(req.status_code, req.content))
        if start_date is not None and pd.to_datetime(df['time'].min()) <= pd.to_datetime(start_date):
            break
    data = pd.concat(data)
    data['time'] = pd.to_datetime(data['time'])
    data = data.rename(columns={'time': 'timestamp', 'value': 'funding_rate'})
    data = data.set_index('timestamp').sort_index()
    return data


def build_funding_rate(arctic_store: Arctic, symbol_type: SymbolType):
    def _get_insert_funding_rate(symbol, symbol_type: SymbolType, arctic_store):
        lib_name = 'bybit_{}.funding_rate'.format(symbol_type.value)
        if symbol_type == SymbolType.USDT_PERP:
            funding_rates = get_funding_rate_history(symbol)
        else:
            raise NotImplemented

        common_symbol = symbol.replace('USDT', '_USDT')
        last_update = funding_rates.index[-1]
        if arctic_store.library_exists(lib_name):
            arctic_store[lib_name].delete(common_symbol)
            arctic_store[lib_name].append(common_symbol, funding_rates,
                                          metadata={'last_update': last_update, 'source': 'bybit'},
                                          prune_previous_version=False, upsert=True)
        else:
            arctic_store.initialize_library(lib_name)
            arctic_store[lib_name].append(common_symbol, funding_rates,
                                          metadata={'last_update': last_update, 'source': 'bybit'},
                                          prune_previous_version=False, upsert=True)

    symbols = get_all_symbol(symbol_type)
    for s in tqdm.tqdm(symbols):
        _get_insert_funding_rate(s, symbol_type=symbol_type, arctic_store=arctic_store)
        time.sleep(1)


def update_funding_rate(symbol, symbol_type: SymbolType, arctic_store):
    lib_name = 'bybit_{}.funding_rate'.format(symbol_type.value)
    common_symbol = symbol.replace('USDT', '_USDT')
    # last_update = funding_rates.index[-1]
    if symbol_type != SymbolType.USDT_PERP:
        raise NotImplemented
    try:
        max_date = arctic_store[lib_name].read_metadata(common_symbol).metadata['last_update']
        funding_rates = get_funding_rate_history(symbol, max_date)
        funding_rates = funding_rates.loc[max_date + datetime.timedelta(seconds=1):]
    except NoDataFoundException:
        funding_rates = get_funding_rate_history(symbol)
    if len(funding_rates) == 0:
        return

    last_update = funding_rates.index[-1]
    arctic_store[lib_name].append(common_symbol, funding_rates,
                                  metadata={'last_update': last_update, 'source': 'bybit'},
                                  prune_previous_version=False, upsert=True)


def build_trades_db(arctic_store: Arctic, symbol_type: SymbolType, quote_asset=None):
    def _get_insert_trades(symbol, symbol_type: SymbolType, arctic_store):
        lib_name = 'bybit_{}.trades'.format(symbol_type.value)
        urls = get_all_url(symbol)
        common_symbol = symbol.replace('USDT', '_USDT')
        for url in tqdm.tqdm(urls, desc=symbol):
            df = get_trades(url)
            if not arctic_store.library_exists(lib_name):
                arctic_store.initialize_library(lib_name, lib_type=TICK_STORE)
                arctic_store.set_quota(lib_name, 0)
            try:
                arctic_store[lib_name].write(common_symbol, df, metadata={'source': 'bybit'})
            except OverlappingDataException:
                pass

    symbols = get_all_symbol(symbol_type, quote_asset)

    func = partial(_get_insert_trades,
                   symbol_type=symbol_type, arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(tqdm.tqdm(executor.map(func, symbols), total=len(symbols)))


if __name__ == '__main__':
    # df = get_funding_rate_history('BTCUSDT')
    store = Arctic('localhost')
    # build_funding_rate(store, SymbolType.USDT_PERP)
    build_trades_db(store, SymbolType.USDT_PERP, )