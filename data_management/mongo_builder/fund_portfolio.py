from concurrent.futures import ThreadPoolExecutor
from functools import partial

import jqdatasdk
import pandas as pd
import tqdm

from arctic import Arctic
from data_management.mongo_builder.decorator import joinquant_retry
from data_management.tokens import jq_user, jq_password
from jqdatasdk import finance, query


def cast_type(df: pd.DataFrame):
    df['code'] = df['code'].astype(str)
    df['symbol'] = df['symbol'].astype(str)
    df['name'] = df['name'].astype(str)
    df['report_type'] = df['report_type'].astype(str)
    df['period_start'] = pd.to_datetime(df['period_start'])
    df['period_end'] = pd.to_datetime(df['period_end'])
    df['pub_date'] = pd.to_datetime(df['pub_date'])
    df = df.sort_index()
    return df


@joinquant_retry
def get_fund_and_insert_portfolio(fund_code: str, arctic_store):
    q = query(finance.FUND_PORTFOLIO_STOCK) \
        .filter(finance.FUND_PORTFOLIO_STOCK.code == fund_code)
    df = finance.run_query(q)
    if df is None or len(df) == 0:
        print('no data for {} please check.'.format(fund_code))
        return

    df = cast_type(df)
    last_update = df['pub_date'].iloc[-1]
    if arctic_store.library_exists('fund_portfolio'):
        arctic_store['fund_portfolio'].delete(fund_code)
        arctic_store['fund_portfolio'].append(fund_code, df, metadata={'last_update': last_update,
                                                                       'source': 'joinquant'},
                                              prune_previous_version=False, upsert=True)
    else:
        arctic_store.initialize_library('fund_portfolio')
        arctic_store['fund_portfolio'].append(fund_code, df, metadata={'last_update': last_update,
                                                                       'source': 'joinquant'},
                                              prune_previous_version=False, upsert=True)


def build_stock_fund_portfolio(arctic_store):
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities(['stock_fund'])
    tickers = [c.replace('.OF', '') for c in instruments.index.to_list()]
    func = partial(get_fund_and_insert_portfolio, arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(tqdm.tqdm(executor.map(func, tickers), total=len(tickers)))
    executor.shutdown()


def build_mixture_fund_portfolio(arctic_store):
    jqdatasdk.auth(jq_user, jq_password)
    instruments = jqdatasdk.get_all_securities(['mixture_fund'])
    tickers = [c.replace('.OF', '') for c in instruments.index.to_list()]
    func = partial(get_fund_and_insert_portfolio, arctic_store=arctic_store)
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(tqdm.tqdm(executor.map(func, tickers), total=len(tickers)))
    executor.shutdown()


if __name__ == '__main__':
    # build_stock_fund_portfolio(Arctic('localhost'))
    build_mixture_fund_portfolio(Arctic('localhost'))
