import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import tqdm

from arctic import Arctic
from arctic.date import DateRange
from arctic.exceptions import NoDataFoundException
from data_management.dataIO import market_data
from data_management.dataIO.market_data import Freq
from factor_zoo.factor_operator.alpha101_operator import delta
from factor_zoo.hf_factor_zoo.agg_factor import *
from factor_zoo.hf_factor_zoo.intraday_operator import get_min_data_df, intraday_standard_price, intraday_ret, \
    intraday_standard_money
from factor_zoo.hf_factor_zoo.xy_intraday import _single_stock_intraday_mixture_gaussian


def get_min_bars(ticker, start_date, data_config_path):
    return market_data.get_bars(ticker,
                                start_date=start_date, freq=Freq.m1,
                                config_path=data_config_path, adjust=False,
                                eod_time_adjust=False, verbose=0,
                                )


def get_intraday_operator(df: pd.DataFrame, turnover: pd.Series):
    idx = df.index
    operators = {}
    standard_price = intraday_standard_price(df).values
    standard_money = intraday_standard_money(df)
    # This fill is just temperately fixed for missing data
    swap = standard_money.droplevel(1).divide(turnover.reindex(idx.get_level_values(0)).fillna(method='ffill'),
                                              axis=0).values
    ret = intraday_ret(df['close'].values)

    corrpriceswap = row_corr(standard_price, swap)
    operators['corrpriceswap'] = corrpriceswap
    deltaswap = delta(swap.T, 1).T
    corretdeltaswap = row_corr(ret[:, 1:], deltaswap[:, 1:])
    operators['corretdeltaswap'] = corretdeltaswap

    # retavg
    operators['retavg'] = avg(ret)
    operators['open_retavg'] = open_avg(ret)
    operators['intra_retavg'] = intra_avg(ret)
    operators['close_retavg'] = close_avg(ret)
    operators['high_retavg'] = high_avg(standard_price, ret)
    operators['low_retavg'] = low_avg(standard_price, ret)
    # retstd
    operators['retstd'] = std(ret)
    operators['open_retstd'] = open_std(ret)
    operators['intra_retstd'] = intra_std(ret)
    operators['close_retstd'] = close_std(ret)
    operators['high_retstd'] = high_std(standard_price, ret)
    operators['low_retstd'] = low_std(standard_price, ret)
    # retskew
    operators['retskew'] = skew(ret)
    operators['open_retskew'] = open_skew(ret)
    operators['intra_retskew'] = intra_skew(ret)
    operators['close_retskew'] = close_skew(ret)
    operators['high_retskew'] = high_skew(standard_price, ret)
    operators['low_retskew'] = low_skew(standard_price, ret)
    # retkurt
    operators['retkurt'] = kurt(ret)
    operators['open_retkurt'] = open_kurt(ret)
    operators['intra_retkurt'] = intra_kurt(ret)
    operators['close_retkurt'] = close_kurt(ret)
    operators['high_retkurt'] = high_kurt(standard_price, ret)
    operators['low_retkurt'] = low_kurt(standard_price, ret)

    # priceavg
    operators['priceavg'] = avg(standard_price)
    operators['open_priceavg'] = open_avg(standard_price)
    operators['intra_priceavg'] = intra_avg(standard_price)
    operators['close_priceavg'] = close_avg(standard_price)
    operators['high_priceavg'] = high_avg(standard_price, standard_price)
    operators['low_priceavg'] = low_avg(standard_price, standard_price)
    # pricestd
    operators['pricestd'] = std(standard_price)
    operators['open_pricestd'] = open_std(standard_price)
    operators['intra_pricestd'] = intra_std(standard_price)
    operators['close_pricestd'] = close_std(standard_price)
    operators['high_pricestd'] = high_std(standard_price, standard_price)
    operators['low_pricestd'] = low_std(standard_price, standard_price)
    # priceskew
    operators['priceskew'] = skew(standard_price)
    operators['open_priceskew'] = open_skew(standard_price)
    operators['intra_priceskew'] = intra_skew(standard_price)
    operators['close_priceskew'] = close_skew(standard_price)
    operators['high_priceskew'] = high_skew(standard_price, standard_price)
    operators['low_priceskew'] = low_skew(standard_price, standard_price)
    # pricekurt
    operators['pricekurt'] = kurt(standard_price)
    operators['open_pricekurt'] = open_kurt(standard_price)
    operators['intra_pricekurt'] = intra_kurt(standard_price)
    operators['close_pricekurt'] = close_kurt(standard_price)
    operators['high_pricekurt'] = high_kurt(standard_price, standard_price)
    operators['low_pricekurt'] = high_kurt(standard_price, standard_price)

    # swapavg
    operators['swapavg'] = avg(swap)
    operators['open_swapavg'] = open_avg(swap)
    operators['intra_swapavg'] = intra_avg(swap)
    operators['close_swapavg'] = close_avg(swap)
    operators['high_swapavg'] = high_avg(standard_price, swap)
    operators['low_swapavg'] = low_avg(standard_price, swap)
    # swapstd
    operators['swapstd'] = std(swap)
    operators['open_swapstd'] = open_std(swap)
    operators['intra_swapstd'] = intra_std(swap)
    operators['close_swapstd'] = close_std(swap)
    operators['high_swapstd'] = high_std(standard_price, swap)
    operators['low_swapstd'] = low_std(standard_price, swap)
    # swapskew
    operators['swapskew'] = skew(swap)
    operators['open_swapskew'] = open_skew(swap)
    operators['intra_swapskew'] = intra_skew(swap)
    operators['close_swapskew'] = close_skew(swap)
    operators['high_swapskew'] = high_skew(standard_price, swap)
    operators['low_swapskew'] = low_skew(standard_price, swap)
    # swapkurt
    operators['swapkurt'] = kurt(swap)
    operators['open_swapkurt'] = open_kurt(swap)
    operators['intra_swapkurt'] = intra_kurt(swap)
    operators['close_swapkurt'] = close_kurt(swap)
    operators['high_swapkurt'] = high_kurt(standard_price, swap)
    operators['low_swapkurt'] = low_kurt(standard_price, swap)
    operators = pd.DataFrame(operators, index=idx).replace([-np.inf, np.inf], np.nan).astype(float).reset_index(level=1)
    operators.index = pd.to_datetime(operators.index)
    return operators


def _read_transform_citic(i: str, store: Arctic):
    df = store['1m'].read(i).data
    df = df[['open', 'high', 'low', 'close', 'code', 'money']]
    df = get_min_data_df(df).set_index('code', append=True).unstack(level=1)
    try:
        turnover = store['daily_basic'].read(i).data['turnover_rate_f']
    except NoDataFoundException as e:
        print(e)
        return

    # idx = df.index
    # operators = {}
    # standard_price = intraday_standard_price(df).values
    # standard_money = intraday_standard_money(df)
    # # This fill is just temperately fixed for missing data
    # swap = standard_money.droplevel(1).divide(turnover.reindex(idx.get_level_values(0)).fillna(method='ffill'),
    #                                           axis=0).values
    # ret = intraday_ret(df['close'].values)
    #
    # corrpriceswap = row_corr(standard_price, swap)
    # operators['corrpriceswap'] = corrpriceswap
    # deltaswap = delta(swap.T, 1).T
    # corretdeltaswap = row_corr(ret[:, 1:], deltaswap[:, 1:])
    # operators['corretdeltaswap'] = corretdeltaswap
    #
    # # retavg
    # operators['retavg'] = avg(ret)
    # operators['open_retavg'] = open_avg(ret)
    # operators['intra_retavg'] = intra_avg(ret)
    # operators['close_retavg'] = close_avg(ret)
    # operators['high_retavg'] = high_avg(standard_price, ret)
    # operators['low_retavg'] = low_avg(standard_price, ret)
    # # retstd
    # operators['retstd'] = std(ret)
    # operators['open_retstd'] = open_std(ret)
    # operators['intra_retstd'] = intra_std(ret)
    # operators['close_retstd'] = close_std(ret)
    # operators['high_retstd'] = high_std(standard_price, ret)
    # operators['low_retstd'] = low_std(standard_price, ret)
    # # retskew
    # operators['retskew'] = skew(ret)
    # operators['open_retskew'] = open_skew(ret)
    # operators['intra_retskew'] = intra_skew(ret)
    # operators['close_retskew'] = close_skew(ret)
    # operators['high_retskew'] = high_skew(standard_price, ret)
    # operators['low_retskew'] = low_skew(standard_price, ret)
    # # retkurt
    # operators['retkurt'] = kurt(ret)
    # operators['open_retkurt'] = open_kurt(ret)
    # operators['intra_retkurt'] = intra_kurt(ret)
    # operators['close_retkurt'] = close_kurt(ret)
    # operators['high_retkurt'] = high_kurt(standard_price, ret)
    # operators['low_retkurt'] = low_kurt(standard_price, ret)
    #
    # # priceavg
    # operators['priceavg'] = avg(standard_price)
    # operators['open_priceavg'] = open_avg(standard_price)
    # operators['intra_priceavg'] = intra_avg(standard_price)
    # operators['close_priceavg'] = close_avg(standard_price)
    # operators['high_priceavg'] = high_avg(standard_price, standard_price)
    # operators['low_priceavg'] = low_avg(standard_price, standard_price)
    # # pricestd
    # operators['pricestd'] = std(standard_price)
    # operators['open_pricestd'] = open_std(standard_price)
    # operators['intra_pricestd'] = intra_std(standard_price)
    # operators['close_pricestd'] = close_std(standard_price)
    # operators['high_pricestd'] = high_std(standard_price, standard_price)
    # operators['low_pricestd'] = low_std(standard_price, standard_price)
    # # priceskew
    # operators['priceskew'] = skew(standard_price)
    # operators['open_priceskew'] = open_skew(standard_price)
    # operators['intra_priceskew'] = intra_skew(standard_price)
    # operators['close_priceskew'] = close_skew(standard_price)
    # operators['high_priceskew'] = high_skew(standard_price, standard_price)
    # operators['low_priceskew'] = low_skew(standard_price, standard_price)
    # # pricekurt
    # operators['pricekurt'] = kurt(standard_price)
    # operators['open_pricekurt'] = open_kurt(standard_price)
    # operators['intra_pricekurt'] = intra_kurt(standard_price)
    # operators['close_pricekurt'] = close_kurt(standard_price)
    # operators['high_pricekurt'] = high_kurt(standard_price, standard_price)
    # operators['low_pricekurt'] = high_kurt(standard_price, standard_price)
    #
    # # swapavg
    # operators['swapavg'] = avg(swap)
    # operators['open_swapavg'] = open_avg(swap)
    # operators['intra_swapavg'] = intra_avg(swap)
    # operators['close_swapavg'] = close_avg(swap)
    # operators['high_swapavg'] = high_avg(standard_price, swap)
    # operators['low_swapavg'] = low_avg(standard_price, swap)
    # # swapstd
    # operators['swapstd'] = std(swap)
    # operators['open_swapstd'] = open_std(swap)
    # operators['intra_swapstd'] = intra_std(swap)
    # operators['close_swapstd'] = close_std(swap)
    # operators['high_swapstd'] = high_std(standard_price, swap)
    # operators['low_swapstd'] = low_std(standard_price, swap)
    # # swapskew
    # operators['swapskew'] = skew(swap)
    # operators['open_swapskew'] = open_skew(swap)
    # operators['intra_swapskew'] = intra_skew(swap)
    # operators['close_swapskew'] = close_skew(swap)
    # operators['high_swapskew'] = high_skew(standard_price, swap)
    # operators['low_swapskew'] = low_skew(standard_price, swap)
    # # swapkurt
    # operators['swapkurt'] = kurt(swap)
    # operators['open_swapkurt'] = open_kurt(swap)
    # operators['intra_swapkurt'] = intra_kurt(swap)
    # operators['close_swapkurt'] = close_kurt(swap)
    # operators['high_swapkurt'] = high_kurt(standard_price, swap)
    # operators['low_swapkurt'] = low_kurt(standard_price, swap)
    # operators = pd.DataFrame(operators, index=idx).replace([-np.inf, np.inf], np.nan).astype(float).reset_index(level=1)
    # operators.index = pd.to_datetime(operators.index)
    operators = get_intraday_operator(df, turnover)
    if not store.library_exists('citic_hf_basic_operator'):
        store.initialize_library('citic_hf_basic_operator')
    last_update = datetime.datetime.now()
    store['citic_hf_basic_operator'].delete(i)
    store['citic_hf_basic_operator'].append(i, operators,
                                            metadata={'last_update': last_update, 'source': 'in-house-calculation'},
                                            prune_previous_version=False, upsert=True)


def _read_transform_xy_gmm(i: str, store: Arctic):
    df = store['5m'].read(i).data
    df = df[['close', 'code']]
    df = get_min_data_df(df)
    df = df.set_index('code', append=True).unstack(level=1)

    ret = pd.DataFrame(intraday_ret(df['close'].values), index=df.index)
    ret = ret[ret.columns[1:]]
    ret = ret.rename(lambda x: pd.to_datetime(x), level=0, axis=0)
    data = _single_stock_intraday_mixture_gaussian(ret)

    if not store.library_exists('xy_gmm'):
        store.initialize_library('xy_gmm')
    last_update = datetime.datetime.now()
    store['xy_gmm'].delete(i)
    store['xy_gmm'].append(i, data,
                           metadata={'last_update': last_update, 'source': 'in-house-calculation'},
                           prune_previous_version=False, upsert=True)


def _read_transform_xy_gmm_1m(i: str, store: Arctic):
    df = store['1m'].read(i).data
    df = df[['close', 'code']]
    df = get_min_data_df(df)
    df = df.set_index('code', append=True).unstack(level=1)

    ret = pd.DataFrame(intraday_ret(df['close'].values), index=df.index)
    ret = ret[ret.columns[1:]]
    ret = ret.rename(lambda x: pd.to_datetime(x), level=0, axis=0)
    data = _single_stock_intraday_mixture_gaussian(ret)

    if not store.library_exists('xy_gmm_1m'):
        store.initialize_library('xy_gmm_1m')
    last_update = datetime.datetime.now()
    store['xy_gmm_1m'].delete(i)
    store['xy_gmm_1m'].append(i, data,
                              metadata={'last_update': last_update, 'source': 'in-house-calculation'},
                              prune_previous_version=False, upsert=True)


def _read_transform_xy_gmm_5m_rolling(i: str, store: Arctic):
    df = store['1m'].read(i).data
    df = df[['close', 'code']]
    df = get_min_data_df(df)
    df = df.set_index('code', append=True).unstack(level=1)

    ret = df['close'].pct_change(5, axis=1)
    ret = ret[ret.columns[5:]]
    ret = ret.rename(lambda x: pd.to_datetime(x), level=0, axis=0)
    data = _single_stock_intraday_mixture_gaussian(ret)

    if not store.library_exists('xy_gmm_5m_rolling'):
        store.initialize_library('xy_gmm_5m_rolling')
    last_update = datetime.datetime.now()
    store['xy_gmm_5m_rolling'].delete(i)
    store['xy_gmm_5m_rolling'].append(i, data,
                              metadata={'last_update': last_update, 'source': 'in-house-calculation'},
                              prune_previous_version=False, upsert=True)


def _read_update_citic(i: str, store: Arctic):
    try:
        last = store['citic_hf_basic_operator'].read(i).data.index[-1] + pd.Timedelta(days=1)
    except:
        last = None
    date_range = DateRange(last, None)

    df = store['1m'].read(i, date_range=date_range).data
    df = df[['open', 'high', 'low', 'close', 'code', 'money']]
    df = get_min_data_df(df).set_index('code', append=True).unstack(level=1)
    try:
        turnover = store['daily_basic'].read(i, date_range=date_range).data['turnover_rate_f']
    except NoDataFoundException as e:
        print(e)
        return
    if df.empty:
        return

    operators = get_intraday_operator(df, turnover)
    last_update = datetime.datetime.now()
    store.set_quota('citic_hf_basic_operator', 0)
    store['citic_hf_basic_operator'].append(i, operators,
                                            metadata={'last_update': last_update, 'source': 'in-house-calculation'},
                                            prune_previous_version=False, upsert=True)


def _read_transform_smart_money(i: str, store: Arctic):
    df = store['1m'].read(i).data
    df = df[['close', 'code', 'money', 'volume']]
    df['pct_change'] = df['close'].pct_change()
    df['S'] = df['pct_change'].abs() / df['money'] ** 0.5


def cal_save_basic_factor_matrix():
    store = Arctic('localhost')
    lib = store['1m']
    instruments = lib.list_symbols()
    func = partial(_read_transform_citic, store=store)
    with ProcessPoolExecutor() as executor:
        list(tqdm.tqdm(executor.map(func, instruments, chunksize=1), total=len(instruments)))


def update_basic_factor_matrix(store):
    lib = store['1m']
    instruments = lib.list_symbols()
    func = partial(_read_update_citic, store=store)
    with ProcessPoolExecutor() as executor:
        list(tqdm.tqdm(executor.map(func, instruments, chunksize=1), total=len(instruments)))


def cal_save_smart_money_matrix():
    store = Arctic('localhost')
    lib = store['1m']
    instruments = lib.list_symbols()
    func = partial(_read_transform_smart_money, store=store)
    with ProcessPoolExecutor() as executor:
        list(tqdm.tqdm(executor.map(func, instruments, chunksize=1), total=len(instruments)))


def cal_save_gmm():
    store = Arctic('localhost')
    lib = store['5m']
    instruments = lib.list_symbols()
    func = partial(_read_transform_xy_gmm, store=store)
    with ProcessPoolExecutor() as executor:
        list(tqdm.tqdm(executor.map(func, instruments, chunksize=1), total=len(instruments)))

def cal_save_gmm_5m_rolling():
    store = Arctic('localhost')
    lib = store['1m']
    instruments = lib.list_symbols()
    func = partial(_read_transform_xy_gmm_5m_rolling, store=store)
    with ProcessPoolExecutor() as executor:
        list(tqdm.tqdm(executor.map(func, instruments, chunksize=1), total=len(instruments)))


def cal_save_gmm_1m():
    store = Arctic('localhost')
    lib = store['1m']
    instruments = lib.list_symbols()
    func = partial(_read_transform_xy_gmm_1m, store=store)
    with ProcessPoolExecutor() as executor:
        list(tqdm.tqdm(executor.map(func, instruments, chunksize=1), total=len(instruments)))


if __name__ == '__main__':
    local_cfg_path = '../cfg/factor_keeper_setting.ini'

    # missing for someday 000760.SZ, 000939.SZ, 002604.SZ, 300216.SZ, 689009.SH
    # No data found for 600849.SH in library arctic.daily_basic 600849.SH 601313.SH

    # _read_transform_citic('600849.SH', Arctic('localhost'))
    # _read_transform_smart_money('000001.SZ', Arctic('localhost'))
    _read_transform_xy_gmm_5m_rolling('000001.SZ', Arctic('localhost'))