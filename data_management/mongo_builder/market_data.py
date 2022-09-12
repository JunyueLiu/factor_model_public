import datetime
import logging
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial

import jqdatasdk
import numpy as np
import pandas as pd
import tqdm
from jqdatasdk import get_all_securities

from arctic import Arctic
from arctic.exceptions import NoDataFoundException
from data_management.mongo_builder.decorator import joinquant_retry
from data_management.tokens import *

MAX_WORKER = 3


def get_instrument_list(arctic_store):
    jqdatasdk.auth(jq_user, jq_password)
    df = get_all_securities(types=['stock', 'index'], date=None)  # type: pd.DataFrame
    df.index = [c.replace('XSHE', 'SZ').replace('XSHG', 'SH') for c in df.index]
    last_update = datetime.datetime.now()
    if arctic_store.library_exists('instrument_list'):
        arctic_store['instrument_list'].delete('A')
        arctic_store['instrument_list'].append('A', df, metadata={'last_update': last_update, 'source': 'joinquant'},
                                               prune_previous_version=False, upsert=True)
    else:
        arctic_store.initialize_library('instrument_list')
        arctic_store['instrument_list'].append('A', df, metadata={'last_update': last_update, 'source': 'joinquant'},
                                               prune_previous_version=False, upsert=True)


@joinquant_retry
def get_bar(jq_code: str, num: int, unit: str, include_now=False):
    df = jqdatasdk.get_bars(jq_code, num, unit=unit,
                            fields=['date', 'open', 'high', 'low', 'close', 'volume', 'money', 'factor'],
                            include_now=include_now
                            )
    normal_code = jq_code.replace('XSHE', 'SZ').replace('XSHG', 'SH')
    df['code'] = normal_code
    df['date'] = pd.to_datetime(df['date'])  # type: pd.DataFrame
    df = df.set_index('date')
    return df


def update_and_insert_bar(jq_code: str, num: int, unit: str, arctic_store: Arctic, trading_date):
    normal_code = jq_code.replace('XSHE', 'SZ').replace('XSHG', 'SH')
    # trading_date = jqdatasdk.get_all_trade_days()
    include_now = True
    try:
        last_update = arctic_store[unit].read_metadata(normal_code).metadata['last_update']
        next_trading_dates = trading_date[(trading_date > last_update.date()) &
                                          (trading_date <= datetime.datetime.now().date())]
        days = len(next_trading_dates)
        if '1d' == unit:
            num = days
        elif '1m' == unit:
            num = 4 * 60 * days
        elif '5m' == unit:
            num = 4 * 12 * days
        elif '15m' == unit:
            num = 4 * 4 * days
        elif '30m' == unit:
            num = 8 * days
        elif '60m' == unit:
            num = 4 * days
        elif '1w' == unit:
            week_now = last_update.isocalendar()[1]
            week_next_day = next_trading_dates[-1].isocalendar()[1]
            if week_now < week_next_day:
                num = week_next_day - week_now + 1
            else:
                return
        elif '1M' == unit:
            month_last_update = last_update.month
            month_next = next_trading_dates[0].month
            month_today = next_trading_dates[-1].month
            month_tomorrow = trading_date[np.where(trading_date <= datetime.datetime.now().date())[0][-1] + 1].month
            if month_today == month_next:
                if month_tomorrow == month_today:
                    return
                else:
                    num = 1
            else:
                if month_tomorrow == month_today:
                    num = month_today - month_last_update - 1
                    include_now = False
                else:
                    num = month_today - month_last_update
        else:
            raise NotImplementedError

        if num <= 0:
            return
        df = get_bar(jq_code, num, unit, include_now=include_now)
        if df is None or len(df) == 0:
            print('no data for {} please check.'.format(normal_code))
            print('Possible reason: delisted, not yet listed, not listed enough time')
            return
        df = df.loc[last_update + pd.Timedelta(seconds=1):]
    except NoDataFoundException:
        df = get_bar(jq_code, num, unit)
        if df is None or len(df) == 0:
            print('no data for {} please check.'.format(normal_code))
            print('Possible reason: delisted, not yet listed, not listed enough time')
            return
    if df is None or len(df) == 0:
        return
    last_update = df.index[-1]
    arctic_store[unit].append(normal_code, df, metadata={'last_update': last_update, 'source': 'joinquant'},
                              prune_previous_version=False, upsert=True)


def get_and_insert_bar(jq_code: str, num: int, unit: str, arctic_store: Arctic):
    normal_code = jq_code.replace('XSHE', 'SZ').replace('XSHG', 'SH')
    df = get_bar(jq_code, num, unit)
    if df is None or len(df) == 0:
        print('no data for {} please check.'.format(normal_code))
        print('Possible reason: delisted, not yet listed, not listed enough time')
        return
    last_update = df.index[-1]
    if arctic_store.library_exists(unit):
        arctic_store[unit].delete(normal_code)
        arctic_store[unit].append(normal_code, df, metadata={'last_update': last_update, 'source': 'joinquant'},
                                  prune_previous_version=False, upsert=True)
    else:
        arctic_store.initialize_library(unit)
        arctic_store[unit].append(normal_code, df, metadata={'last_update': last_update, 'source': 'joinquant'},
                                  prune_previous_version=False, upsert=True)


def get_and_insert_bar2(jq_code: str, num: int, unit: str, arctic_store: Arctic):
    normal_code = jq_code.replace('XSHE', 'SZ').replace('XSHG', 'SH')
    if arctic_store.library_exists(unit) and normal_code in arctic_store[unit].list_symbols():
        return

    df = get_bar(jq_code, num, unit)
    if df is None or len(df) == 0:
        print('no data for {} please check.'.format(normal_code))
        print('Possible reason: delisted, not yet listed, not listed enough time')
        return
    last_update = df.index[-1]
    if arctic_store.library_exists(unit):
        arctic_store[unit].append(normal_code, df, metadata={'last_update': last_update, 'source': 'joinquant'},
                                  prune_previous_version=False, upsert=True)
    else:
        arctic_store.initialize_library(unit)
        arctic_store[unit].append(normal_code, df, metadata={'last_update': last_update, 'source': 'joinquant'},
                                  prune_previous_version=False, upsert=True)


def build_1M_ohlc_db(arctic_store: Arctic, instrument_type: str):
    """

    :param arctic_store:
    :return:
    """
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities([instrument_type])  # type: pd.DataFrame
    func = partial(get_and_insert_bar, num=250, unit='1M', arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()


def build_1w_ohlc_db(arctic_store: Arctic, instrument_type: str):
    """

    :param arctic_store:
    :return:
    """
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities([instrument_type])  # type: pd.DataFrame
    func = partial(get_and_insert_bar, num=1000, unit='1w', arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()


def build_1d_ohlc_db(arctic_store: Arctic, instrument_type: str):
    """

    :param arctic_store:
    :return:
    """
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities([instrument_type])  # type: pd.DataFrame
    # 4000 is set because only have 2005 to today data, which less than 4000 bars.
    func = partial(get_and_insert_bar, num=4000, unit='1d', arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()


def build_bbg_csv_ohlc_db(arctic_store: Arctic, csv_path, code: str, unit: str):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['code'] = code
    df.columns = df.columns.str.lower()
    df = df.set_index('date')
    df = df.sort_index()
    last_update = df.index[-1]
    if arctic_store.library_exists(unit):
        arctic_store[unit].delete(code)
        arctic_store[unit].append(code, df, metadata={'last_update': last_update, 'source': 'csv'},
                                  prune_previous_version=False, upsert=True)


def build_60m_ohlc_db(arctic_store: Arctic, instrument_type):
    """
    :param arctic_store:
    :return:
    """
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities([instrument_type])  # type: pd.DataFrame
    func = partial(get_and_insert_bar, num=4000 * 4, unit='60m', arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()


def build_30m_ohlc_db(arctic_store: Arctic, instrument_type):
    """
    :param arctic_store:
    :return:
    """
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities([instrument_type])  # type: pd.DataFrame
    func = partial(get_and_insert_bar, num=4000 * 8, unit='30m', arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()


def build_15m_ohlc_db(arctic_store: Arctic, instrument_type):
    """
    :param arctic_store:
    :return:
    """
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities([instrument_type])  # type: pd.DataFrame
    func = partial(get_and_insert_bar, num=4000 * 16, unit='15m', arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()


def build_5m_ohlc_db(arctic_store: Arctic, instrument_type):
    """
    :param arctic_store:
    :return:
    """
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities([instrument_type])  # type: pd.DataFrame
    store.set_quota('5m', 0)
    func = partial(get_and_insert_bar2, num=4000 * 48, unit='5m', arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()


def build_1m_ohlc_db(arctic_store: Arctic, instrument_type):
    """
    :param arctic_store:
    :return:
    """
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities([instrument_type])  # type: pd.DataFrame
    store.set_quota('1m', 0)
    func = partial(get_and_insert_bar2, num=4000 * 240, unit='1m', arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()


def update_1d_ohlc_db(arctic_store: Arctic, instrument_type: str):
    """

    :param arctic_store:
    :return:
    """
    now = datetime.datetime.now()
    # if now.date().weekday() < 5 and (now.hour < 15 or (now.hour == 15 and now.minute == 0)):
    #     print('Must run end of date')
    #     return

    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities([instrument_type])  # type: pd.DataFrame
    trading_date = jqdatasdk.get_all_trade_days()
    func = partial(update_and_insert_bar, num=4000,
                   unit='1d', arctic_store=arctic_store, trading_date=trading_date)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()
    logging.info('finish...{}'.format('update_1d_ohlc_db'))


def update_60m_ohlc_db(arctic_store: Arctic, instrument_type: str):
    """

    :param arctic_store:
    :return:
    """
    now = datetime.datetime.now()
    if now.date().weekday() < 5 and (now.hour < 15 or (now.hour == 15 and now.minute == 0)):
        print('Must run end of date')
        return
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities([instrument_type])  # type: pd.DataFrame
    trading_date = jqdatasdk.get_all_trade_days()
    func = partial(update_and_insert_bar, num=4000 * 4,
                   unit='60m', arctic_store=arctic_store, trading_date=trading_date)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()
    logging.info('finish...{}'.format('update_60m_ohlc_db'))

def update_30m_ohlc_db(arctic_store: Arctic, instrument_type: str):
    """

    :param arctic_store:
    :return:
    """
    now = datetime.datetime.now()
    if now.date().weekday() < 5 and (now.hour < 15 or (now.hour == 15 and now.minute == 0)):
        print('Must run end of date')
        return
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities([instrument_type])  # type: pd.DataFrame
    trading_date = jqdatasdk.get_all_trade_days()
    func = partial(update_and_insert_bar, num=4000 * 8,
                   unit='30m', arctic_store=arctic_store, trading_date=trading_date)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()
    logging.info('finish...{}'.format('update_30m_ohlc_db'))


def update_15m_ohlc_db(arctic_store: Arctic, instrument_type: str):
    """

    :param arctic_store:
    :return:
    """
    now = datetime.datetime.now()
    if now.date().weekday() < 5 and (now.hour < 15 or (now.hour == 15 and now.minute == 0)):
        print('Must run end of date')
        return
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities([instrument_type])  # type: pd.DataFrame
    trading_date = jqdatasdk.get_all_trade_days()
    arctic_store.set_quota('15m', 0)
    func = partial(update_and_insert_bar, num=4000 * 4 * 4,
                   unit='15m', arctic_store=arctic_store, trading_date=trading_date)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()
    logging.info('finish...{}'.format('update_15m_ohlc_db'))


def update_5m_ohlc_db(arctic_store: Arctic, instrument_type: str):
    """

    :param arctic_store:
    :return:
    """
    now = datetime.datetime.now()
    if now.date().weekday() < 5 and (now.hour < 15 or (now.hour == 15 and now.minute == 0)):
        print('Must run end of date')
        return
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities([instrument_type])  # type: pd.DataFrame
    trading_date = jqdatasdk.get_all_trade_days()
    arctic_store.set_quota('5m', 0)
    func = partial(update_and_insert_bar, num=4000 * 4 * 12,
                   unit='5m', arctic_store=arctic_store, trading_date=trading_date)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()
    logging.info('finish...{}'.format('update_5m_ohlc_db'))


def update_1m_ohlc_db(arctic_store: Arctic, instrument_type: str):
    """

    :param arctic_store:
    :return:
    """
    now = datetime.datetime.now()
    if now.date().weekday() < 5 and (now.hour < 15 or (now.hour == 15 and now.minute == 0)):
        print('Must run end of date')
        return
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities([instrument_type])  # type: pd.DataFrame
    trading_date = jqdatasdk.get_all_trade_days()
    arctic_store.set_quota('1m', 0)
    func = partial(update_and_insert_bar, num=4000 * 4 * 60,
                   unit='1m', arctic_store=arctic_store, trading_date=trading_date)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()
    logging.info('finish...{}'.format('update_1m_ohlc_db'))


def update_1w_ohlc_db(arctic_store: Arctic, instrument_type: str):
    """

    :param arctic_store:
    :return:
    """
    now = datetime.datetime.now()
    if (now.date().weekday() == 5 and (now.hour < 15 or (now.hour == 15 and now.minute == 0))) \
            or now.date().weekday() < 5:
        print('Must run end of Friday market close')
        return
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities([instrument_type])  # type: pd.DataFrame
    trading_date = jqdatasdk.get_all_trade_days()
    func = partial(update_and_insert_bar, num=4000 / 5,
                   unit='1w', arctic_store=arctic_store, trading_date=trading_date)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()
    print('finish...{}'.format('update_1w_ohlc_db'))


def update_1M_ohlc_db(arctic_store: Arctic, instrument_type: str):
    """

    :param arctic_store:
    :return:
    """
    now = datetime.datetime.now()
    if now.date().weekday() < 5 and (now.hour < 15 or (now.hour == 15 and now.minute == 0)):
        print('Must run end of date')
        return
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities([instrument_type])  # type: pd.DataFrame
    trading_date = jqdatasdk.get_all_trade_days()

    func = partial(update_and_insert_bar, num=4000 / 5,
                   unit='1M', arctic_store=arctic_store, trading_date=trading_date)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()


if __name__ == '__main__':
    store = Arctic('localhost')
    # get_instrument_list(store)
    # build_1M_ohlc_db(store, 'stock')
    # build_1w_ohlc_db(store, 'stock')
    # build_1d_ohlc_db(store, 'stock')
    # build_60m_ohlc_db(store, 'stock')
    # build_30m_ohlc_db(store, 'stock')
    # build_5m_ohlc_db(store, 'stock')
    # build_1m_ohlc_db(store, 'stock')
    # build_15m_ohlc_db(store, 'stock')
    # build_1M_ohlc_db(store, 'index')
    # build_1w_ohlc_db(store, 'index')
    # build_1d_ohlc_db(store, 'index')
    # build_60m_ohlc_db(store, 'index')
    # update_1d_ohlc_db(store, 'stock')
    # update_1w_ohlc_db(store, 'stock')
    # update_1M_ohlc_db(store, 'stock')
    # update_1M_ohlc_db(store, 'index')
    # update_5m_ohlc_db(store, 'stock')
    # update_15m_ohlc_db(store, 'stock')
    # update_1m_ohlc_db(store, 'stock')
    update_30m_ohlc_db(store, 'stock')
    # build_bbg_csv_ohlc_db(store, '../../data/csv/HistoricalPrices.csv', 'XIN9', '1d')
