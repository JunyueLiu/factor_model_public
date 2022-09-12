import datetime
import logging
import warnings
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial

import jqdatasdk
import pandas as pd
import tqdm
import tushare as ts

from arctic import Arctic, VERSION_STORE
from arctic.exceptions import NoDataFoundException
from data_management.mongo_builder.decorator import joinquant_retry, tushare_retry
from data_management.mongo_builder.utils import market_batch_insert_df_to_arctic, market_batch_update_df_to_arctic, \
    get_latest_last_update_date
from data_management.tokens import jq_user, jq_password, token

MAX_WORKER = 3


def cast_type(df: pd.DataFrame):
    df['sec_code'] = df['sec_code'].apply(lambda x: x.replace('XSHE', 'SZ').replace('XSHG', 'SH'))
    df = df.rename(columns={'sec_code': 'code'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.sort_index()
    return df


def cast_limit_type(df: pd.DataFrame, code):
    df['code'] = code
    df.index.names = ['date']
    # df.index = pd.to_datetime(df.index)
    return df


@joinquant_retry
def get_and_insert_moneyflow(jq_code, arctic_store):
    normal_code = jq_code.replace('XSHE', 'SZ').replace('XSHG', 'SH')
    df = jqdatasdk.get_money_flow(jq_code, '2010-01-01', end_date=datetime.datetime.now())
    if df is None or len(df) == 0:
        print('no data for {} please check.'.format(normal_code))
        print('Possible reason: delisted, not yet listed, not listed enough time')
        return

    df = cast_type(df)
    last_update = df.index[-1]
    if arctic_store.library_exists('money_flow'):
        arctic_store['money_flow'].delete(normal_code)
        arctic_store['money_flow'].append(normal_code, df, metadata={'last_update': last_update,
                                                                     'source': 'joinquant'},
                                          prune_previous_version=False, upsert=True)
    else:
        arctic_store.initialize_library('money_flow')
        arctic_store['money_flow'].append(normal_code, df, metadata={'last_update': last_update,
                                                                     'source': 'joinquant'},
                                          prune_previous_version=False, upsert=True)


@joinquant_retry
def update_and_insert_moneyflow(jq_code, arctic_store):
    normal_code = jq_code.replace('XSHE', 'SZ').replace('XSHG', 'SH')
    try:
        last_update = arctic_store['money_flow'].read_metadata(normal_code).metadata['last_update']
    except NoDataFoundException:
        last_update = '2010-01-01'

    df = jqdatasdk.get_money_flow(jq_code, last_update, end_date=datetime.datetime.now())
    if df is None or len(df) == 0:
        print('no data update for {} please check.'.format(normal_code))
        return

    df = cast_type(df)
    df = df.loc[pd.to_datetime(last_update) + pd.Timedelta(seconds=1):]
    if len(df) == 0:
        return
    last_update = df.index[-1]
    arctic_store['money_flow'].append(normal_code, df,
                                      metadata={'last_update': last_update, 'source': 'joinquant'},
                                      prune_previous_version=False, upsert=True)


@tushare_retry
def get_daily_basic(trade_date: str, pro):
    df = pro.daily_basic(trade_date=trade_date)  # type: pd.DataFrame
    df = df.rename(columns={'ts_code': 'code', 'trade_date': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df['code'] = df['code'].astype(str)
    return df


@tushare_retry
def get_tushare_moneyflow(trade_date: str, pro):
    df = pro.moneyflow(trade_date=trade_date)
    df = df.rename(columns={'ts_code': 'code', 'trade_date': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df['code'] = df['code'].astype(str)
    return df


@tushare_retry
def get_tushare_stock_limit(trade_date, pro):
    warnings.warn("tushare_stock_limit is deprecated; use joinquant.", DeprecationWarning)
    df = pro.stk_limit(trade_date=trade_date)
    df = df.rename(columns={'ts_code': 'code', 'trade_date': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df['code'] = df['code'].astype(str)
    return df


@joinquant_retry
def get_and_insert_joinquant_stock_limit(jq_code, arctic_store: Arctic):
    normal_code = jq_code.replace('XSHE', 'SZ').replace('XSHG', 'SH')
    df = jqdatasdk.get_price(jq_code, '2005-01-01', end_date=datetime.datetime.now(),
                             fields=('high_limit', 'low_limit', 'paused'),
                             panel=False, fq='none'
                             )
    df = cast_limit_type(df, normal_code)
    if df is None or len(df) == 0:
        print('no data for {} please check.'.format(normal_code))
        print('Possible reason: delisted, not yet listed, not listed enough time')
        return

    # df = cast_type(df)
    last_update = df.index[-1]
    if arctic_store.library_exists('stock_limit'):
        arctic_store['stock_limit'].delete(normal_code)
        arctic_store['stock_limit'].append(normal_code, df, metadata={'last_update': last_update,
                                                                      'source': 'joinquant'},
                                           prune_previous_version=False, upsert=True)
    else:
        arctic_store.initialize_library('stock_limit', lib_type=VERSION_STORE)
        arctic_store['stock_limit'].append(normal_code, df, metadata={'last_update': last_update,
                                                                      'source': 'joinquant'},
                                           prune_previous_version=False, upsert=True)

@joinquant_retry
def update_and_insert_joinquant_stock_limit(jq_code, arctic_store: Arctic):
    normal_code = jq_code.replace('XSHE', 'SZ').replace('XSHG', 'SH')

    try:
        last_update = arctic_store['stock_limit'].read_metadata(normal_code).metadata['last_update'] + pd.Timedelta(days=1)
    except NoDataFoundException:
        last_update = '2005-01-01'

    df = jqdatasdk.get_price(jq_code, last_update, end_date=datetime.datetime.now(),
                             fields=('high_limit', 'low_limit', 'paused'),
                             panel=False, fq='none'
                             )
    df = cast_limit_type(df, normal_code)
    if df is None or len(df) == 0:
        print('no data update for {} please check.'.format(normal_code))
        return

    last_update = df.index[-1]
    arctic_store['stock_limit'].append(normal_code, df, metadata={'last_update': last_update,
                                                                  'source': 'joinquant'},
                                       prune_previous_version=False, upsert=True)




def build_moneyflow(arctic_store):
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities(['stock'])
    func = partial(get_and_insert_moneyflow, arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()


def build_daily_basic(arctic_store):
    # tickers = get_all_ticker()
    MAX_WORKER = 10
    ts.set_token(token)
    pro = ts.pro_api()
    jqdatasdk.auth(jq_user, jq_password)
    trading_date = jqdatasdk.get_all_trade_days()
    trading_date = trading_date[trading_date <= datetime.datetime.now().date()]
    trading_date = [d.strftime('%Y%m%d') for d in trading_date]
    func = partial(get_daily_basic, pro=pro)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        result = list(tqdm.tqdm(executor.map(func, trading_date), total=len(trading_date)))
    data = pd.concat(result, ignore_index=True)
    for c in data.columns:
        if c in ['code', 'date']:
            continue
        data[c] = data[c].astype(float)
    data = data.set_index('date')
    market_batch_insert_df_to_arctic(data, arctic_store, 'daily_basic')


def build_tushare_moneyflow(arctic_store):
    # tickers = get_all_ticker()
    MAX_WORKER = 10
    ts.set_token(token)
    pro = ts.pro_api()
    jqdatasdk.auth(jq_user, jq_password)
    trading_date = jqdatasdk.get_all_trade_days()
    trading_date = trading_date[(trading_date <= datetime.datetime.now().date()) &
                                (trading_date >= pd.to_datetime('2010-01-01').date())]
    trading_date = [d.strftime('%Y%m%d') for d in trading_date]
    func = partial(get_tushare_moneyflow, pro=pro)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        result = list(tqdm.tqdm(executor.map(func, trading_date), total=len(trading_date)))
    data = pd.concat(result, ignore_index=True)
    for c in data.columns:
        if c in ['code', 'date']:
            continue
        data[c] = data[c].astype(float)
    data = data.set_index('date')
    market_batch_insert_df_to_arctic(data, arctic_store, 'tushare_moneyflow')


def build_limit(arctic_store):
    # tickers = get_all_ticker()
    warnings.warn("build_limit is deprecated; use build_joinquant_limit.", warnings.DeprecationWarning)
    MAX_WORKER = 10
    ts.set_token(token)
    pro = ts.pro_api()
    jqdatasdk.auth(jq_user, jq_password)
    trading_date = jqdatasdk.get_all_trade_days()
    trading_date = trading_date[trading_date <= datetime.datetime.now().date()]
    trading_date = [d.strftime('%Y%m%d') for d in trading_date]
    func = partial(get_tushare_stock_limit, pro=pro)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        result = list(tqdm.tqdm(executor.map(func, trading_date), total=len(trading_date)))
    data = pd.concat(result, ignore_index=True)
    for c in data.columns:
        if c in ['code', 'date']:
            continue
        data[c] = data[c].astype(float)
    data = data.set_index('date')
    market_batch_insert_df_to_arctic(data, arctic_store, 'stock_limit')


def build_joinquant_limit(arctic_store: Arctic):
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities(['stock'])
    func = partial(get_and_insert_joinquant_stock_limit, arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()


def build_ST(arctic_store: Arctic):
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities(['stock'])
    data = jqdatasdk.get_extras('is_st', instruments.index.to_list(), start_date='2005-01-01')
    last_update = data.index[-1]
    for c in tqdm.tqdm(data):
        df = data[c].loc[instruments.loc[c, 'start_date']:].to_frame('is_st')  # type: pd.Series
        code = c.replace('XSHE', 'SZ').replace('XSHG', 'SH')
        df['code'] = code
        if arctic_store.library_exists('ST'):
            arctic_store['ST'].delete(code)
            arctic_store['ST'].append(code, df, metadata={'last_update': last_update,
                                                          'source': 'joinquant'},
                                      prune_previous_version=False, upsert=True)
        else:
            arctic_store.initialize_library('ST')
            arctic_store['ST'].append(code, df, metadata={'last_update': last_update,
                                                          'source': 'joinquant'},
                                      prune_previous_version=False, upsert=True)


def update_moneyflow(arctic_store):
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities(['stock'])
    logging.info("update_moneyflow")
    func = partial(update_and_insert_moneyflow, arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()
    last_update = get_latest_last_update_date(arctic_store, "money_flow")
    logging.info("{} latest update {}".format("money_flow", last_update))


def update_daily_basic(arctic_store):
    MAX_WORKER = 10
    logging.info("update_daily_basic")
    ts.set_token(token)
    pro = ts.pro_api()
    jqdatasdk.auth(jq_user, jq_password)
    trading_date = jqdatasdk.get_all_trade_days()
    lib = arctic_store['daily_basic']
    last_update_dates = [lib.read(c).metadata['last_update']
                         for c in lib.list_symbols()]
    last_update = max(last_update_dates)
    last_update = last_update.date()
    trading_date = trading_date[(trading_date > last_update) & (trading_date <= datetime.datetime.now().date())]
    trading_date = [d.strftime('%Y%m%d') for d in trading_date]
    func = partial(get_daily_basic, pro=pro)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        result = list(tqdm.tqdm(executor.map(func, trading_date), total=len(trading_date)))
    if len(result) == 0:
        return

    data = pd.concat(result, ignore_index=True)
    for c in data.columns:
        if c in ['code', 'date']:
            continue
        data[c] = data[c].astype(float)
    data = data.set_index('date')
    market_batch_update_df_to_arctic(data, arctic_store, 'daily_basic')
    last_update = get_latest_last_update_date(arctic_store, "daily_basic")
    logging.info("{} latest update {}".format("daily_basic", last_update))


def update_tushare_moneyflow(arctic_store):
    # tickers = get_all_ticker()
    logging.info("update_tushare_moneyflow")
    MAX_WORKER = 10
    ts.set_token(token)
    pro = ts.pro_api()
    jqdatasdk.auth(jq_user, jq_password)
    trading_date = jqdatasdk.get_all_trade_days()
    lib = arctic_store['tushare_moneyflow']
    last_update_dates = [lib.read(c).metadata['last_update']
                         for c in lib.list_symbols()]
    last_update = max(last_update_dates)
    last_update = last_update.date()
    trading_date = trading_date[(trading_date > last_update) & (trading_date <= datetime.datetime.now().date())]

    trading_date = [d.strftime('%Y%m%d') for d in trading_date]
    func = partial(get_tushare_moneyflow, pro=pro)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        result = list(tqdm.tqdm(executor.map(func, trading_date), total=len(trading_date)))
    data = pd.concat(result, ignore_index=True)
    for c in data.columns:
        if c in ['code', 'date']:
            continue
        data[c] = data[c].astype(float)
    data = data.set_index('date')
    market_batch_update_df_to_arctic(data, arctic_store, 'tushare_moneyflow')
    last_update = get_latest_last_update_date(arctic_store, "tushare_moneyflow")
    logging.info("{} latest update {}".format("tushare_moneyflow", last_update))


def update_limit(arctic_store):
    warnings.warn("update_limit is deprecated; use joinquant.", DeprecationWarning)
    MAX_WORKER = 10
    ts.set_token(token)
    pro = ts.pro_api()
    jqdatasdk.auth(jq_user, jq_password)
    trading_date = jqdatasdk.get_all_trade_days()
    lib = arctic_store['stock_limit']
    last_update_dates = [lib.read(c).metadata['last_update']
                         for c in lib.list_symbols()]
    last_update = max(last_update_dates)
    last_update = last_update.date()
    trading_date = trading_date[(trading_date > last_update) & (trading_date <= datetime.datetime.now().date())]
    trading_date = [d.strftime('%Y%m%d') for d in trading_date]
    func = partial(get_tushare_stock_limit, pro=pro)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        result = list(tqdm.tqdm(executor.map(func, trading_date), total=len(trading_date)))
    data = pd.concat(result, ignore_index=True)
    for c in data.columns:
        if c in ['code', 'date']:
            continue
        data[c] = data[c].astype(float)
    data = data.set_index('date')
    market_batch_update_df_to_arctic(data, arctic_store, 'stock_limit')


def update_joinquant_limit(arctic_store):
    jqdatasdk.auth(jq_user, jq_password)
    logging.info("update_joinquant_limit")
    instruments = jqdatasdk.get_all_securities(['stock'])
    func = partial(update_and_insert_joinquant_stock_limit, arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()
    last_update = get_latest_last_update_date(arctic_store, "stock_limit")
    logging.info("{} latest update {}".format("stock_limit", last_update))


if __name__ == '__main__':
    store = Arctic('localhost')
    # arctic_config_path = '../../cfg/arctic_hk.ini'
    # arctic_config = configparser.ConfigParser()
    # arctic_config.read(os.path.abspath(arctic_config_path))
    # store = get_arctic_store(arctic_config)
    # arctic_config_path = '../../cfg/arctic_hk.ini'
    # arctic_config = configparser.ConfigParser()
    # arctic_config.read(os.path.abspath(arctic_config_path))
    # store = get_arctic_store(arctic_config)
    # build_moneyflow(store)
    # update_moneyflow(store)
    # build_daily_basic(store)
    # update_daily_basic(store)
    # build_tushare_moneyflow(store)
    # update_tushare_moneyflow(store)
    # build_limit(store)
    # update_limit(store)
    build_ST(store)
    # build_joinquant_limit(store)
    # update_joinquant_limit(store)
