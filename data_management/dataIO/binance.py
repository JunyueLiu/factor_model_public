import datetime
from enum import Enum
from typing import Optional, List, Union

import ccxt
import pandas as pd
import requests

from arctic import Arctic
from arctic.date import DateRange
from data_management.dataIO.utils import read_arctic_version_store


class Freq(Enum):
    m1 = '1m'
    m15 = '15m'
    h4 = '4h'


def get_um_symbol_info(perp=True):
    binance = ccxt.binanceusdm()
    d = binance.fapiPublicGetExchangeInfo()
    df = pd.DataFrame(d['symbols'])
    if perp:
        df = df[df['contractType'] == 'PERPETUAL']
    return df


def get_um_bars(code: Optional[List[str]] = None,
                start_date: Optional[Union[str, datetime.datetime]] = None,
                end_date: Optional[Union[str, datetime.datetime]] = None,
                freq: Freq = Freq.m1,
                ) -> pd.DataFrame:
    store = Arctic('localhost')
    lib_name = 'binance_um.{}'.format(freq.value)
    date_range = DateRange(start_date, end_date)
    if code:
        data = read_arctic_version_store(store, lib_name, code, min(len(code), 10), date_range=date_range)
    else:
        data = read_arctic_version_store(store, lib_name, code, 10, date_range=date_range)

    for c in ['open', 'high', 'low', 'close']:
        data[c] = data[c].astype(float)
        data['adj_{}'.format(c)] = data[c]

    data['volume'] = data['volume'].astype(float)
    data['quote_asset_volume'] = data['quote_asset_volume'].astype(float)
    data['number_of_trades'] = data['number_of_trades'].astype(float)
    data['taker_buy_base_asset_volume'] = data['taker_buy_base_asset_volume'].astype(float)
    data['taker_buy_quote_asset_volume'] = data['taker_buy_quote_asset_volume'].astype(float)

    data['money'] = data['quote_asset_volume']
    data['factor'] = 1
    data.index.names = ['date', 'code']
    return data


def get_tags():
    # https://www.binance.com/exchange-api/v2/public/asset-service/product/get-products
    r = requests.get('https://www.binance.com/exchange-api/v2/public/asset-service/product/get-products')
    while True:
        if r.status_code == 200:
            df = pd.DataFrame(r.json()['data'])
            df = df[['s', 'tags']]
            df = df.set_index('s')
            df = df.explode('tags').dropna()
            df = df.reset_index().set_index('tags').sort_index().squeeze()
            d = df.groupby(level=0).apply(list).to_dict()
            return d
        else:
            pass


def get_live_um_bar(code, freq: Freq, limit=500):
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
               'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    binance = ccxt.binance()
    data = binance.fapiPublicGetKlines(params={'symbol': code, 'interval': freq.value, 'limit': limit})
    bars = pd.DataFrame(data, columns=columns)
    bars['open_time'] = pd.to_datetime(bars['open_time'], utc=True, unit='ms').dt.tz_localize(None)
    bars['close_time'] = pd.to_datetime(bars['close_time'], utc=True, unit='ms').dt.tz_localize(None)
    bars['code'] = code
    bars = bars.set_index(['open_time', 'code'])

    for c in ['open', 'high', 'low', 'close']:
        bars[c] = bars[c].astype(float)
        bars['adj_{}'.format(c)] = bars[c]

    for c in ['quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
        bars[c] = bars[c].astype(float)

    bars['money'] = bars['quote_asset_volume']
    bars['factor'] = 1
    bars['adj_ohlc4'] = (bars['adj_open'] + bars['adj_high'] + bars['adj_low'] + bars[
        'adj_close']) / 4
    bars['adj_ohl3'] = (bars['adj_open'] + bars['adj_high'] + bars['adj_low']) / 3
    bars['adj_ohc3'] = (bars['adj_open'] + bars['adj_high'] + bars['adj_close']) / 3
    bars['adj_hl2'] = (bars['adj_high'] + bars['adj_low']) / 2
    bars['adj_oc2'] = (bars['adj_open'] + bars['adj_close']) / 4
    bars.index.names = ['date', 'code']
    return bars


if __name__ == '__main__':
    # store = Arctic('localhost')
    # get_um_bars(['BTCUSDT', 'ETHUSDT'], '2021-01-01')
    # get_tags()
    # df = get_live_um_bar('BTCUSDT', Freq.h4, 1500)
    get_um_symbol_info()
