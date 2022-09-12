import datetime
import logging
from collections import OrderedDict
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import jqdatasdk
import tqdm
from arctic import Arctic

from data_management.mongo_builder.decorator import joinquant_retry
from data_management.tokens import jq_user, jq_password
from jqdatasdk import finance, query

MAX_WORKER = 3


@joinquant_retry
def get_industry_stocks(date, industry_code):
    stocks = jqdatasdk.get_industry_stocks(industry_code, date)
    stocks = [s.replace('XSHE', 'SZ').replace('XSHG', 'SH') for s in stocks]
    return pd.to_datetime(date), stocks


@joinquant_retry
def get_sw1_market_data(code):
    res = []
    df = finance.run_query(query(finance.SW1_DAILY_PRICE)
                           .filter(finance.SW1_DAILY_PRICE.code == code))
    df['date'] = pd.to_datetime(df['date'])  # type: pd.DataFrame
    df = df.set_index('date')
    end = df.index[-1].to_pydatetime()
    res.append(df)
    while True:
        df = finance.run_query(query(finance.SW1_DAILY_PRICE).filter(finance.SW1_DAILY_PRICE.code == code).filter(
            finance.SW1_DAILY_PRICE.date > end))
        if df.empty:
            break
        df['date'] = pd.to_datetime(df['date'])  # type: pd.DataFrame
        df = df.set_index('date')
        end = df.index[-1].to_pydatetime()
        res.append(df)
    df = pd.concat(res)
    return df

@joinquant_retry
def update_sw1_market_data(code, end):
    df = finance.run_query(query(finance.SW1_DAILY_PRICE).filter(finance.SW1_DAILY_PRICE.code == code).filter(
        finance.SW1_DAILY_PRICE.date > end))
    df['date'] = pd.to_datetime(df['date'])  # type: pd.DataFrame
    df = df.set_index('date')
    return df


def build_industry(arctic_store: Arctic):
    jqdatasdk.auth(jq_user, jq_password)
    for i in ['zjw', 'sw_l1', 'sw_l2', 'sw_l3']:
        ind = jqdatasdk.get_industries(i)  # type: pd.DataFrame
        last_update = datetime.datetime.now()
        if arctic_store.library_exists('industries'):
            arctic_store['industries'].delete(i)
            arctic_store['industries'].append(i, ind,
                                              metadata={'last_update': last_update, 'source': 'joinquant'},
                                              prune_previous_version=False, upsert=True)
        else:
            arctic_store.initialize_library('industries')
            arctic_store['industries'].append(i, ind,
                                              metadata={'last_update': last_update, 'source': 'joinquant'},
                                              prune_previous_version=False, upsert=True)


def build_industry_stocks(arctic_store, industry):
    jqdatasdk.auth(jq_user, jq_password)
    trading_date = jqdatasdk.get_all_trade_days()
    trading_date = trading_date[trading_date <= datetime.datetime.now().date()]

    ind = jqdatasdk.get_industries(industry)

    for industry_code, row in ind.iterrows():
        library_name = '{}_stocks'.format(industry)
        if arctic_store.library_exists(library_name) and industry_code in arctic_store[library_name].list_symbols():
            continue
        trading_date2 = trading_date[trading_date >= row.start_date]
        func = partial(get_industry_stocks, industry_code=industry_code)
        print('start...', industry_code, row['name'])
        with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
            result = list(tqdm.tqdm(executor.map(func, trading_date2), total=len(trading_date2)))
        executor.shutdown()
        d = {r[0]: r[1] for r in result}
        d = OrderedDict(sorted(d.items()))
        last_update = list(d.keys())[-1]
        if arctic_store.library_exists(library_name):
            # arctic_store[library_name].delete(industry_code)
            arctic_store[library_name].write(industry_code, d,
                                             metadata={'last_update': last_update, 'source': 'joinquant',
                                                       'Chinese': row['name'],
                                                       })
        else:
            arctic_store.initialize_library(library_name)
            arctic_store[library_name].write(industry_code, d,
                                             metadata={'last_update': last_update, 'source': 'joinquant',
                                                       'Chinese': row['name'],
                                                       })
        print('finish...', industry_code, row['name'])


def build_sw_l1_1d(arctic_store):
    jqdatasdk.auth(jq_user, jq_password)
    ind = jqdatasdk.get_industries('sw_l1')
    for industry_code in tqdm.tqdm(ind.index):
        df = get_sw1_market_data(industry_code)
        last_update = df.index[-1]
        if arctic_store.library_exists('sw_l1_1d'):
            arctic_store['sw_l1_1d'].delete(industry_code)
            arctic_store['sw_l1_1d'].append(industry_code, df,
                                            metadata={'last_update': last_update, 'source': 'joinquant'},
                                            prune_previous_version=False, upsert=True)
        else:
            arctic_store.initialize_library('sw_l1_1d')
            arctic_store['sw_l1_1d'].append(industry_code, df,
                                            metadata={'last_update': last_update, 'source': 'joinquant'},
                                            prune_previous_version=False, upsert=True)


def update_sw_l1_1d(arctic_store):
    jqdatasdk.auth(jq_user, jq_password)
    ind = jqdatasdk.get_industries('sw_l1')
    for industry_code in tqdm.tqdm(ind.index):
        last_update = arctic_store['sw_l1_1d'].read_metadata(industry_code).metadata['last_update']
        df = update_sw1_market_data(industry_code, last_update)
        if len(df) > 0:
            last_update = df.index[-1]
            arctic_store['sw_l1_1d'].append(industry_code, df,
                                            metadata={'last_update': last_update, 'source': 'joinquant'},
                                            prune_previous_version=False, upsert=True)


def update_industry(arctic_store: Arctic):
    build_industry(arctic_store)


def update_industry_stocks(arctic_store, industry):
    jqdatasdk.auth(jq_user, jq_password)
    trading_date = jqdatasdk.get_all_trade_days()
    trading_date = trading_date[trading_date <= datetime.datetime.now().date()]

    trading_date = trading_date[trading_date <= datetime.datetime.now().date()]

    ind = jqdatasdk.get_industries(industry)

    for industry_code, row in ind.iterrows():
        library_name = '{}_stocks'.format(industry)
        if industry_code in arctic_store[library_name].list_symbols():
            last_update = arctic_store[library_name].read(industry_code).metadata['last_update'].date()
            trading_date2 = trading_date[trading_date > last_update]
            old_d = arctic_store[library_name].read(industry_code).data
        else:
            trading_date2 = trading_date[trading_date >= row.start_date]
            old_d = None

        func = partial(get_industry_stocks, industry_code=industry_code)
        logging.info('start...', industry_code, row['name'])
        with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
            result = list(tqdm.tqdm(executor.map(func, trading_date2), total=len(trading_date2)))
        executor.shutdown()
        d = {r[0]: r[1] for r in result}
        d = OrderedDict(sorted(d.items()))
        if len(d) == 0:
            continue
        last_update = list(d.keys())[-1]
        if old_d:
            d = OrderedDict(sorted({**old_d, **d}.items()))

        arctic_store[library_name].write(industry_code, d,
                                         metadata={'last_update': last_update, 'source': 'joinquant',
                                                   'Chinese': row['name'],
                                                   })
        logging.info('finish update...', industry_code, row['name'])


if __name__ == '__main__':
    store = Arctic('localhost')
    # build_industry(store)
    # update_industry(store)
    # build_industry_stocks(store, 'sw_l1')
    # build_industry_stocks(store, 'sw_l2')
    # build_industry_stocks(store, 'sw_l3')
    # build_industry_stocks(store, 'zjw')
    # build_sw_l1_1d(store)
    # update_industry_stocks(store, 'sw_l1')
    # update_industry_stocks(store, 'sw_l2')
    # update_industry_stocks(store, 'sw_l3')
    # update_industry_stocks(store, 'zjw')
    update_sw_l1_1d(store)