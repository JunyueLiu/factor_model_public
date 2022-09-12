import datetime
import json
import time
import urllib

import pandas as pd
import requests
import tqdm

from arctic import Arctic
from data_management.mongo_builder.crypto import SymbolType


def get_all_symbols_info(symbol_type: SymbolType) -> pd.DataFrame:
    if symbol_type == SymbolType.SPOT:
        response = urllib.request.urlopen("https://www.okex.com/api/v5/public/instruments?instType=SPOT").read()
    elif symbol_type in [SymbolType.USDT_PERP, SymbolType.INVERSE_PERP]:
        response = urllib.request.urlopen("https://www.okex.com/api/v5/public/instruments?instType=SWAP").read()
    else:
        raise NotImplementedError
    data = json.loads(response)['data']
    data = pd.DataFrame(data)
    if symbol_type == SymbolType.USDT_PERP:
        data = data[data['ctValCcy'] != 'USD']
    elif symbol_type == SymbolType.INVERSE_PERP:
        data = data[data['ctValCcy'] == 'USD']

    return data


def get_all_symbol(symbol_type: SymbolType):
    data = get_all_symbols_info(symbol_type)
    symbols = data['instId'].to_list()
    return symbols


def to_common_symbol(okex_symbol: str):
    if 'SWAP' in okex_symbol:
        return okex_symbol.replace('-SWAP', '').replace('-', '_')
    else:
        return okex_symbol.replace('-', '_')


def get_funding_rate_history(symbol: str):
    base_url = 'https://www.okex.com/api/v5/public/funding-rate-history'
    headers = {
        "Content-Type": "application/json",
        "accept-language": "en-US,en",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"}
    after = int(datetime.datetime.now().timestamp() * 1000)
    data = []
    while True:
        url = '{}?instId={}&after={}&limit=100'.format(base_url, symbol, after)
        req = requests.get(url, headers=headers)
        if req.status_code == 200:
            res = req.json()['data']
            df = pd.DataFrame(res)
            if df.empty:
                break
            after = df['fundingTime'].iloc[-1]
            data.append(df)
        else:
            print(req.status_code)
            time.sleep(1)
    data = pd.concat(data)
    data['fundingTime'] = pd.to_datetime(data['fundingTime'], unit='ms')
    data['realizedRate'] = data['realizedRate'].astype(float)
    data = data.rename(columns={'fundingTime': 'timestamp', 'realizedRate': 'funding_rate'})
    data = data[['timestamp', 'funding_rate']]
    data = data.set_index('timestamp').sort_index()
    return data


def update_funding_rate_db(arctic_store: Arctic, symbol_type: SymbolType):
    def _update_insert_funding_rate(symbol, symbol_type: SymbolType, arctic_store):
        lib_name = 'okx_{}.funding_rate'.format(symbol_type.value)
        if symbol_type == SymbolType.USDT_PERP:
            funding_rates = get_funding_rate_history(symbol)
        else:
            raise NotImplemented

        common_symbol = to_common_symbol(symbol)
        last_update = funding_rates.index[-1]
        if not arctic_store.library_exists(lib_name):
            arctic_store.initialize_library(lib_name)

        if arctic_store.get_library(lib_name).has_symbol(common_symbol):
            max_date = arctic_store[lib_name].read_metadata(common_symbol).metadata['last_update']
            funding_rates = funding_rates.loc[max_date + datetime.timedelta(seconds=1):]
            arctic_store[lib_name].append(common_symbol, funding_rates,
                                          metadata={'last_update': last_update, 'source': 'okx'},
                                          prune_previous_version=False, upsert=True)
        else:
            arctic_store[lib_name].append(common_symbol, funding_rates,
                                          metadata={'last_update': last_update, 'source': 'okx'},
                                          prune_previous_version=False, upsert=True)

    symbols = get_all_symbol(symbol_type)
    for s in tqdm.tqdm(symbols):
        _update_insert_funding_rate(s, symbol_type, arctic_store)


if __name__ == '__main__':
    # df = get_funding_rate_history('BTC-USDT-SWAP')
    update_funding_rate_db(Arctic('localhost'), SymbolType.USDT_PERP)