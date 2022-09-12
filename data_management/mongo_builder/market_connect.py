import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import jqdatasdk
import pandas as pd
import tqdm
import tushare as ts
from jqdatasdk import finance, query

from arctic import Arctic
from arctic.exceptions import NoDataFoundException
from data_management.mongo_builder.decorator import joinquant_retry
from data_management.tokens import jq_user, jq_password, token


# jqdatasdk.auth(jq_user, jq_password)
# df = finance.run_query(query(finance.STK_HK_HOLD_INFO).filter(finance.STK_HK_HOLD_INFO.link_id == 310001))
# print(df)


@joinquant_retry
def get_hk_holding(jq_code, arctic_store):
    # 310001	沪股通
    # 310002	深股通
    normal_code = jq_code.replace('XSHE', 'SZ').replace('XSHG', 'SH')
    if jq_code.endswith('SH'):

        df = finance.run_query(query(finance.STK_HK_HOLD_INFO)
                               .filter(finance.STK_HK_HOLD_INFO.link_id == 310001)
                               .filter(finance.STK_HK_HOLD_INFO.code == jq_code)
                               )
    else:
        df = finance.run_query(query(finance.STK_HK_HOLD_INFO)
                               .filter(finance.STK_HK_HOLD_INFO.link_id == 310002)
                               .filter(finance.STK_HK_HOLD_INFO.code == jq_code)
                               )
    if len(df) == 0:
        return

    df = df[['day', 'share_number', 'share_ratio']]
    df['code'] = normal_code
    df = df.rename(columns={'day': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date'])
    last_update = df.index[-1]
    if arctic_store.library_exists('hk_holding'):
        arctic_store['hk_holding'].delete(normal_code)
        arctic_store['hk_holding'].append(normal_code, df,
                                          metadata={'last_update': last_update,
                                                    'source': 'joinquant'},
                                          prune_previous_version=False, upsert=True)
    else:
        arctic_store.initialize_library('hk_holding')
        arctic_store['hk_holding'].append(normal_code, df, metadata={'last_update': last_update,
                                                                     'source': 'joinquant'},
                                          prune_previous_version=False, upsert=True)


@joinquant_retry
def get_update_hk_holding(jq_code, arctic_store):
    normal_code = jq_code.replace('XSHE', 'SZ').replace('XSHG', 'SH')
    try:
        last_update = arctic_store['hk_holding'].read_metadata(normal_code).metadata['last_update']
        if jq_code.endswith('SH'):
            df = finance.run_query(query(finance.STK_HK_HOLD_INFO)
                                   .filter(finance.STK_HK_HOLD_INFO.link_id == 310001)
                                   .filter(finance.STK_HK_HOLD_INFO.code == jq_code)
                                   .filter(finance.STK_HK_HOLD_INFO.day > last_update)
                                   )
        else:
            df = finance.run_query(query(finance.STK_HK_HOLD_INFO)
                                   .filter(finance.STK_HK_HOLD_INFO.link_id == 310002)
                                   .filter(finance.STK_HK_HOLD_INFO.code == jq_code)
                                   .filter(finance.STK_HK_HOLD_INFO.day > last_update)
                                   )
    except NoDataFoundException:
        if jq_code.endswith('SH'):
            df = finance.run_query(query(finance.STK_HK_HOLD_INFO)
                                   .filter(finance.STK_HK_HOLD_INFO.link_id == 310001)
                                   .filter(finance.STK_HK_HOLD_INFO.code == jq_code)
                                   )
        else:
            df = finance.run_query(query(finance.STK_HK_HOLD_INFO)
                                   .filter(finance.STK_HK_HOLD_INFO.link_id == 310002)
                                   .filter(finance.STK_HK_HOLD_INFO.code == jq_code)
                                   )
    if len(df) == 0:
        return
    df = df[['day', 'share_number', 'share_ratio']]
    df['code'] = normal_code
    df = df.rename(columns={'day': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date'])
    last_update = df.index[-1]

    arctic_store['hk_holding'].append(normal_code, df, metadata={'last_update': last_update,
                                                                 'source': 'joinquant'},
                                      prune_previous_version=False, upsert=True)


def build_hk_holding(arctic_store):
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities(['stock'])
    func = partial(get_hk_holding, arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()


def update_hk_holding(arctic_store):
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities(['stock'])
    func = partial(get_update_hk_holding, arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=1) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()


def build_hk_market_moneyflow(arctic_store):
    ts.set_token(token)
    pro = ts.pro_api()
    data = []
    year = datetime.datetime.now().strftime('%Y')
    for i in range(2014, int(year), 3):
        df = pro.ggt_daily(start_date='{}0101'.format(i), end_date='{}0101'.format(i + 3))
        data.append(df)
        time.sleep(30)
    data.append(pro.ggt_daily(start_date='{}0101'.format(year)))

    data = pd.concat(data)
    data['trade_date'] = pd.to_datetime(data['trade_date'])
    data = data.rename(columns={'trade_date': 'date'})
    data = data.set_index('date')
    data = data.sort_index()
    data = data[~data.index.duplicated()]
    last_update = data.index[-1]
    if arctic_store.library_exists('hk_market_moneyflow'):
        arctic_store['hk_market_moneyflow'].delete('hk_market_moneyflow')
        arctic_store['hk_market_moneyflow'].append('hk_market_moneyflow', data,
                                                   metadata={'last_update': last_update,
                                                             'source': 'tushare'},
                                                   prune_previous_version=False, upsert=True)
    else:
        arctic_store.initialize_library('hk_market_moneyflow')
        arctic_store['hk_market_moneyflow'].append('hk_market_moneyflow', data, metadata={'last_update': last_update,
                                                                                          'source': 'tushare'},
                                                   prune_previous_version=False, upsert=True)

def update_hk_market_moneyflow(arctic_store):
    ts.set_token(token)
    pro = ts.pro_api()
    last_update = arctic_store['hk_market_moneyflow'].read_metadata('hk_market_moneyflow').metadata['last_update']
    data = pro.ggt_daily(start_date=(last_update + datetime.timedelta(days=1)).strftime('%Y%m%d'))
    data['trade_date'] = pd.to_datetime(data['trade_date'])
    data = data.rename(columns={'trade_date': 'date'})
    data = data.set_index('date')
    data = data.sort_index()
    data = data[~data.index.duplicated()]
    if len(data) == 1:
        return
    last_update = data.index[-1]
    arctic_store['hk_market_moneyflow'].append('hk_market_moneyflow', data,
                                               metadata={'last_update': last_update,
                                                         'source': 'tushare'},
                                               prune_previous_version=False, upsert=True)


if __name__ == '__main__':
    # df = get_hk_holding(datetime.datetime.now().date())
    store = Arctic('localhost')
    # build_hk_holding(store)
    # get_hk_holding('000001.XSHE')
    # build_hk_market_moneyflow(store)
    # update_hk_holding(store)
    update_hk_market_moneyflow(store)