import logging
from concurrent.futures import ThreadPoolExecutor

import ccxt
import pandas as pd
import requests
import tqdm

from arctic import Arctic
from arctic.exceptions import NoDataFoundException


def get_all_um_perp_symbol_info():
    r = requests.get('https://fapi.binance.com/fapi/v1/exchangeInfo')
    df = pd.DataFrame(r.json()['symbols'])
    return df


def get_um_monthly_avaiable(symbol: str, freq):
    URL = 'https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/futures/um/monthly/klines/{}/{}/'.format(
        symbol, freq)
    try:
        df = pd.read_xml(URL)

        available = df['Key'].dropna().sort_values()
        available = available.str.extract(r'([\d]{4}-[\d]{2})').drop_duplicates().squeeze().to_list()
        return available
    except:
        print(URL, 'has not data.')
        return []


def get_um_daily_avaiable(symbol: str, freq):
    URL = 'https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/futures/um/daily/klines/{}/{}/'.format(
        symbol, freq)
    try:
        df = pd.read_xml(URL)

        available = df['Key'].dropna().sort_values()
        available = available.str.extract(r'([\d]{4}-[\d]{2}-[\d]{2})').drop_duplicates().squeeze().to_list()
        return available
    except:
        print(URL, 'has not data.')
        return []


def get_um_monthly_bar(symbol, freq, month: str):
    # months = get_um_monthly_avaiable(symbol, freq)
    # days = get_um_daily_avaiable(symbol, freq)
    # days = [d for d in days if d > months]
    URL = 'https://data.binance.vision/data/futures/um/monthly/klines/{}/{}/{}-{}-{}.zip'.format(symbol, freq, symbol,
                                                                                                 freq, month)
    while True:
        try:
            df = pd.read_csv(URL, header=None)
            break
        except Exception as e:
            print(e, URL)

    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
               'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    df.columns = columns
    df['open_time'] = pd.to_datetime(df['open_time'], utc=True, unit='ms').dt.tz_localize(None)
    df['close_time'] = pd.to_datetime(df['close_time'], utc=True, unit='ms').dt.tz_localize(None)
    return df


def get_um_daily_bar(symbol, freq, day: str):
    URL = 'https://data.binance.vision/data/futures/um/daily/klines/{}/{}/{}-{}-{}.zip'.format(symbol, freq, symbol,
                                                                                               freq, day)
    while True:
        try:
            df = pd.read_csv(URL, header=None)
            break
        except Exception as e:
            if e.code == 404:
                return pd.DataFrame()
            print(e, URL)

    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
               'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    df.columns = columns
    df['open_time'] = pd.to_datetime(df['open_time'], utc=True, unit='ms').dt.tz_localize(None)
    df['close_time'] = pd.to_datetime(df['close_time'], utc=True, unit='ms').dt.tz_localize(None)
    return df


def get_um_bar(symbol, freq, start_date, end_date=None):
    binance = ccxt.binance()

    start_date = int(pd.to_datetime(start_date, utc=True).timestamp() * 1000)
    paras = {
        'symbol': symbol, 'interval': freq,
        'startTime': start_date,
        # 'limit': 1500
    }
    if end_date is not None:
        end_date = pd.to_datetime(end_date, utc=True).timestamp()
        paras['endTime'] = end_date
    data = binance.fapiPublicGetKlines(params=paras)
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
               'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    bars = pd.DataFrame(data, columns=columns)
    bars['open_time'] = pd.to_datetime(bars['open_time'], utc=True, unit='ms').dt.tz_localize(None)
    bars['close_time'] = pd.to_datetime(bars['close_time'], utc=True, unit='ms').dt.tz_localize(None)
    return bars


def build_um_bar_df(arctic_store: Arctic, freq: str):
    symbols_df = get_all_um_perp_symbol_info()
    symbols_df = symbols_df[symbols_df['quoteAsset'] == 'USDT']
    symbols_df = symbols_df[symbols_df['contractType'] == 'PERPETUAL']
    for s in symbols_df['symbol']:
        months = get_um_monthly_avaiable(s, freq)
        data = []
        for m in tqdm.tqdm(months):
            df = get_um_monthly_bar(s, freq, m)
            data.append(df)
        if len(data) == 0:
            continue
        data = pd.concat(data)
        data = data.set_index('open_time')
        data = data.sort_index()
        unit = 'binance_um.{}'.format(freq)
        last_update = data.index[-1]
        if arctic_store.library_exists(unit):
            arctic_store[unit].delete(s)
            arctic_store[unit].append(s, data,
                                      metadata={'last_update': last_update,
                                                'source': 'binance'},
                                      prune_previous_version=False, upsert=True)
        else:
            arctic_store.initialize_library(unit)
            arctic_store[unit].append(s, data,
                                      metadata={'last_update': last_update, 'source': 'binance'},
                                      prune_previous_version=False, upsert=True)
        print('finish...{}'.format(s))


def update_um_bar_db(arctic_store: Arctic, freq: str):
    symbols_df = get_all_um_perp_symbol_info()
    symbols_df = symbols_df[symbols_df['quoteAsset'] == 'USDT']
    symbols_df = symbols_df[symbols_df['contractType'] == 'PERPETUAL']
    unit = 'binance_um.{}'.format(freq)
    for s in tqdm.tqdm(symbols_df['symbol']):
        try:
            metadata = arctic_store[unit].read_metadata(s).metadata
            last_update = metadata['last_update']
            days = pd.date_range(last_update.strftime('%Y-%m-%d'), pd.Timestamp.now(), freq='D')[1:]
            days = days.strftime('%Y-%m-%d').to_list()
        except:
            days = get_um_daily_avaiable(s, freq)
        data = []
        for d in days:
            df = get_um_daily_bar(s, freq, d)
            data.append(df)
        if len(data) == 0:
            continue
        data = pd.concat(data)
        data = data.set_index('open_time')
        unit = 'binance_um.{}'.format(freq)
        last_update = data.index[-1]
        arctic_store[unit].append(s, data,
                                  metadata={'last_update': last_update,
                                            'source': 'binance'},
                                  prune_previous_version=False, upsert=True)


def update_um_bar_db2(arctic_store: Arctic, freq: str, logger):
    symbols_df = get_all_um_perp_symbol_info()
    symbols_df = symbols_df[symbols_df['quoteAsset'] == 'USDT']
    symbols_df = symbols_df[symbols_df['contractType'] == 'PERPETUAL']


    def _update(s):
        unit = 'binance_um.{}'.format(freq)
        try:
            metadata = arctic_store[unit].read_metadata(s).metadata
            last_update = metadata['last_update']
            data = get_um_bar(s, freq, last_update)
            data = data.set_index('open_time')
            data = data.loc[pd.to_datetime(last_update) + pd.Timedelta(seconds=1):]
        except NoDataFoundException:
            data = get_um_bar(s, freq, '2021-01-01')
            data = data.set_index('open_time')
        except Exception as e:
            print(e)
            return
        if len(data) == 0:
            return
        unit = 'binance_um.{}'.format(freq)
        last_update = data.index[-1]
        logger.info('last update time for {} is {}'.format(s, last_update))
        arctic_store[unit].append(s, data,
                                  metadata={'last_update': last_update,
                                            'source': 'binance'},
                                  prune_previous_version=False, upsert=True)

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm.tqdm(executor.map(_update, symbols_df['symbol']), total=len(symbols_df['symbol'])))
    executor.shutdown()

if __name__ == '__main__':
    # get_um_monthly_avaiable("BTCUSDT", '1m')
    # get_um_daily_avaiable('BTCUSDT', '1m')
    # get_usdt_bar('BTCUSDT', '1m')
    # get_all_um_perp_symbol_info()
    # build_um_bar_df(Arctic('localhost'), '1h')
    # build_um_bar_df(Arctic('localhost'), '4h')
    # update_um_bar_db(Arctic('localhost'), '1m')
    # build_um_bar_df(Arctic('localhost'), '15m')
    # update_um_bar_db(Arctic('localhost'), '4h')
    logger = logging.getLogger('update')
    store = Arctic('localhost')
    update_um_bar_db2(store, '15m', logger)
