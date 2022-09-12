import datetime
import logging
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import jqdatasdk
import tqdm
from arctic import Arctic
from arctic.exceptions import NoDataFoundException
from jqdatasdk import finance, query
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime

from data_management.mongo_builder.decorator import joinquant_retry
from data_management.mongo_builder.utils import get_latest_last_update_date
from data_management.tokens import jq_user, jq_password

MAX_WORKER = 3


@joinquant_retry
def run_finance_query(q) -> pd.DataFrame:
    return finance.run_query(q)


def cast_type(df: pd.DataFrame):
    df = df.fillna(value=np.nan)
    df['code'] = df['code'].apply(lambda x: x.replace('XSHE', 'SZ').replace('XSHG', 'SH'))
    for c in df:
        if df[c].dtype == object:
            if 'date' in c:
                try:
                    df[c] = pd.to_datetime(df[c])
                except OutOfBoundsDatetime:
                    df[c] = pd.to_datetime(df[c], errors='coerce')
            else:
                df[c] = df[c].astype(str)
        if c == 'unpledged_date':
            df['unpledged_date'] = pd.to_datetime(df['unpledged_date'])
        elif c == 'unfrozen_date':
            df['unfrozen_date'] = pd.to_datetime(df['unfrozen_date'])

    return df


def get_and_insert_fundamental(jq_code, collection: str, arctic_store):
    """

    :param jq_code:
    :param collection:
    :param arctic_store:
    :return:
    """
    normal_code = jq_code.replace('XSHE', 'SZ').replace('XSHG', 'SH')
    q = query(eval('finance.{}'.format(collection))) \
        .filter(eval('finance.{}.code'.format(collection)) == jq_code)
    df = run_finance_query(q)
    if df is None or len(df) == 0:
        print('no data for {} please check.'.format(normal_code))
        print('Possible reason: delisted, not yet listed, not listed enough time')
        return

    df = cast_type(df)
    if 'end_date' in df:
        df = df.sort_values(['end_date', 'pub_date'])
    elif 'change_date' in df:
        if 'pub_date' in df:
            df = df.sort_values(['change_date', 'pub_date'])
        else:
            df = df.sort_values(['change_date'])
    else:
        df = df.sort_values('pub_date')
    if 'pub_date' in df:
        last_update = df['pub_date'].iloc[-1]
    else:
        last_update = datetime.datetime.now()
    if arctic_store.library_exists(collection):
        arctic_store[collection].delete(normal_code)
        arctic_store[collection].append(normal_code, df, metadata={'last_update': last_update, 'source': 'joinquant'},
                                        prune_previous_version=False, upsert=True)
    else:
        arctic_store.initialize_library(collection)
        arctic_store[collection].append(normal_code, df, metadata={'last_update': last_update, 'source': 'joinquant'},
                                        prune_previous_version=False, upsert=True)


def update_and_insert_fundamental(jq_code: str, collection: str, arctic_store: Arctic):
    """

    :param jq_code:
    :param collection:
    :param arctic_store:
    :return:
    """
    normal_code = jq_code.replace('XSHE', 'SZ').replace('XSHG', 'SH')
    try:
        last_update = arctic_store[collection].read_metadata(normal_code).metadata['last_update']
        q = query(eval('finance.{}'.format(collection))) \
            .filter(eval('finance.{}.code'.format(collection)) == jq_code,
                    eval('finance.{}.pub_date'.format(collection)) > last_update
                    )

    except NoDataFoundException:
        q = query(eval('finance.{}'.format(collection))) \
            .filter(eval('finance.{}.code'.format(collection)) == jq_code)
    df = run_finance_query(q)
    if df is None or len(df) == 0:
        # print('no update for {}.'.format(normal_code))
        return

    df = cast_type(df)
    if 'end_date' in df:
        df = df.sort_values(['end_date', 'pub_date'])
    elif 'change_date' in df:
        df = df.sort_values(['change_date', 'pub_date'])
    else:
        df = df.sort_values('pub_date')
    last_update = df['pub_date'].iloc[-1]
    arctic_store[collection].append(normal_code, df, metadata={'last_update': last_update, 'source': 'joinquant'},
                                    prune_previous_version=False, upsert=True)


def build_securities_info(arctic_store):
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities(['stock'])  # type: pd.DataFrame
    instruments.index = [c.replace('XSHE', 'SZ').replace('XSHG', 'SH') for c in instruments.index]
    last_update = datetime.datetime.now()
    if arctic_store.library_exists('securities_info'):
        arctic_store['securities_info'].delete('stock')
        arctic_store['securities_info'].append('stock', instruments,
                                               metadata={'last_update': last_update, 'source': 'joinquant'},
                                               prune_previous_version=False, upsert=True)
    else:
        arctic_store.initialize_library('securities_info')
        arctic_store['securities_info'].append('stock', instruments,
                                               metadata={'last_update': last_update, 'source': 'joinquant'},
                                               prune_previous_version=False, upsert=True)


def build_fundamental(arctic_store, jq_table_name):
    jqdatasdk.auth(jq_user, jq_password)
    finance.__getattr__('STK_FIN_FORCAST')
    finance_tables = [c for c in finance._DBTable__table_names if c.startswith('STK')]
    if jq_table_name not in finance_tables:
        raise ValueError('jq_table_name must in {}'.format(finance_tables))
    instruments = jqdatasdk.get_all_securities(['stock'])
    func = partial(get_and_insert_fundamental, collection=jq_table_name, arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()


def update_fundamental(arctic_store, jq_table_name):
    logging.info('update_fundamental: {}'.format(jq_table_name))
    jqdatasdk.auth(jq_user, jq_password)
    finance.__getattr__('STK_FIN_FORCAST')
    finance_tables = [c for c in finance._DBTable__table_names if c.startswith('STK')]
    if jq_table_name not in finance_tables:
        raise ValueError('jq_table_name must in {}'.format(finance_tables))

    instruments = jqdatasdk.get_all_securities(['stock'])
    func = partial(update_and_insert_fundamental, collection=jq_table_name, arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        list(tqdm.tqdm(executor.map(func, instruments.index), total=len(instruments.index)))
    executor.shutdown()
    last_update = get_latest_last_update_date(arctic_store, jq_table_name)
    logging.info("{} latest update {}".format(jq_table_name, last_update))

def build_fin_forcast(arctic_store):
    build_fundamental(arctic_store, 'STK_FIN_FORCAST')


def build_balance_sheet(arctic_store):
    build_fundamental(arctic_store, 'STK_BALANCE_SHEET')


def build_cashflow_statement(arctic_store):
    build_fundamental(arctic_store, 'STK_CASHFLOW_STATEMENT')


def build_income_statement(arctic_store):
    build_fundamental(arctic_store, 'STK_INCOME_STATEMENT')


def build_shareholder_top10(arctic_store):
    build_fundamental(arctic_store, 'STK_SHAREHOLDER_TOP10')


def build_shareholder_floating_top10(arctic_store):
    build_fundamental(arctic_store, 'STK_SHAREHOLDER_FLOATING_TOP10')


def build_capital_change(arctic_store):
    build_fundamental(arctic_store, 'STK_CAPITAL_CHANGE')


def build_shares_pledge(arctic_store):
    build_fundamental(arctic_store, 'STK_SHARES_PLEDGE')


def build_shares_frozen(arctic_store):
    build_fundamental(arctic_store, 'STK_SHARES_FROZEN')


def build_holder_num(arctic_store):
    build_fundamental(arctic_store, 'STK_HOLDER_NUM')


def build_shareholders_share_change(arctic_store):
    build_fundamental(arctic_store, 'STK_SHAREHOLDERS_SHARE_CHANGE')


def build_limited_shares_list(arctic_store):
    build_fundamental(arctic_store, 'STK_LIMITED_SHARES_LIST')


def build_limited_shares_unlimited(arctic_store):
    build_fundamental(arctic_store, 'STK_LIMITED_SHARES_UNLIMIT')


def update_fin_forcast(arctic_store):
    update_fundamental(arctic_store, 'STK_FIN_FORCAST')


def update_balance_sheet(arctic_store):
    update_fundamental(arctic_store, 'STK_BALANCE_SHEET')


def update_cashflow_statement(arctic_store):
    update_fundamental(arctic_store, 'STK_CASHFLOW_STATEMENT')


def update_income_statement(arctic_store):
    update_fundamental(arctic_store, 'STK_INCOME_STATEMENT')


def update_shareholder_top10(arctic_store):
    update_fundamental(arctic_store, 'STK_SHAREHOLDER_TOP10')


def update_shareholder_floating_top10(arctic_store):
    update_fundamental(arctic_store, 'STK_SHAREHOLDER_FLOATING_TOP10')


def update_capital_change(arctic_store):
    update_fundamental(arctic_store, 'STK_CAPITAL_CHANGE')


def update_shares_pledge(arctic_store):
    update_fundamental(arctic_store, 'STK_SHARES_PLEDGE')


def update_shares_frozen(arctic_store):
    update_fundamental(arctic_store, 'STK_SHARES_FROZEN')


def update_holder_num(arctic_store):
    update_fundamental(arctic_store, 'STK_HOLDER_NUM')


def update_shareholders_share_change(arctic_store):
    update_fundamental(arctic_store, 'STK_SHAREHOLDERS_SHARE_CHANGE')


def update_limited_shares_list(arctic_store):
    update_fundamental(arctic_store, 'STK_LIMITED_SHARES_LIST')


def update_limited_shares_unlimited(arctic_store):
    update_fundamental(arctic_store, 'STK_LIMITED_SHARES_UNLIMIT')


if __name__ == '__main__':
    store = Arctic('localhost')
    # build_securities_info(store)
    # build_fin_forcast(store)
    # build_balance_sheet(store)
    # build_cashflow_statement(store)
    # build_income_statement(store)
    # build_capital_change(store)
    # build_shareholder_top10(store)
    # build_shareholder_floating_top10(store)
    # build_holder_num(store)
    # build_shareholders_share_change(store)
    # build_shares_pledge(store)
    # build_shares_frozen(store)
    # build_limited_shares_list(store)
    # build_limited_shares_unlimited(store)

    # update_fin_forcast(store)
    # update_balance_sheet(store  )
    # update_cashflow_statement(store)
    # update_income_statement(store)
    # update_capital_change(store)
    # update_shareholder_top10(store)
    # update_shareholder_floating_top10(store)
    # update_holder_num(store)
    # update_shareholders_share_change(store)
    # update_shares_pledge(store)  # todo cannot run this
    # update_shares_frozen(store) # todo cannot run this
    # update_limited_shares_list(store)
    # update_limited_shares_unlimited(store)
