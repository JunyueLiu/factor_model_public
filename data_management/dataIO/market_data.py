import configparser
import datetime
import os
from enum import Enum
from typing import List, Optional, Union, Tuple

import pandas as pd

from data_management.dataIO.utils import get_arctic_store, read_arctic_version_store
from data_management.pandas_utils.cache import panel_df_join, df_merge_asof


class Freq(Enum):
    m1 = 'm1'
    m5 = 'm5'
    m15 = 'm15'
    m30 = 'm30'
    m60 = "m60"
    D1 = "D1"
    W1 = "W1"
    M1 = "M1"


def get_bars(code: Optional[Union[str, List[str]]] = None,
             start_date: Optional[Union[str, datetime.datetime]] = None,
             end_date: Optional[Union[str, datetime.datetime]] = None,
             cols: Tuple = ('open', 'high', 'low', 'close', 'money', 'factor'),
             freq: Freq = Freq.D1,
             config_path: str = '../../cfg/data_input.ini',
             verbose: int = 1,
             adjust: bool = True,
             index: bool = True,
             eod_time_adjust: bool = True,
             add_limit: bool = True
             ) -> pd.DataFrame:
    config = configparser.ConfigParser()
    config.read(config_path)
    if 'Source' not in config.sections():
        raise ValueError('Source section not in config. PLease check config path and config file')

    market_data_source = config['Source']['market_data']
    project_deploy_path = config['Source']['project_deploy_path']
    config2 = configparser.ConfigParser()
    config2.read(os.path.join(project_deploy_path, 'cfg', market_data_source), encoding='utf-8')
    if len(config2.sections()) == 0:
        raise ValueError('Empty config. PLease check config path and config file for {}'
                         .format(os.path.join(project_deploy_path, 'cfg', market_data_source)))
    cols = list(cols)

    start_time = datetime.datetime.now()
    PRICE_COLS = config2['Market'].get('price_cols', 'open,high,low,close').split(',')
    FACTOR_COL = config2['Market'].get('adj_factor_col', 'factor')
    if FACTOR_COL not in cols and adjust:
        cols.append(FACTOR_COL)

    daily_data_path = config2['Market']['daily_data']
    limit_data_path = config2['Market']['limit_data']
    weekly_data_path = config2['Market']['weekly_data']
    monthly_data_path = config2['Market']['monthly_data']
    # ============== add more table here ==============

    if 'Arctic' in config2.sections():
        store = get_arctic_store(config2)
        m1_data_path = config2['Market']['m1_data']
        m5_data_path = config2['Market']['m5_data']
        m15_data_path = config2['Market']['m15_data']
        m30_data_path = config2['Market']['m30_data']
        m60_data_path = config2['Market']['m60_data']
        MAX_WORKER = int(config2['Arctic'].get('MAX_WORKER', 10))
        if freq == Freq.D1:
            library = daily_data_path
        elif freq == Freq.M1:
            library = monthly_data_path
        elif freq == Freq.W1:
            library = weekly_data_path
        elif freq == Freq.m1:
            library = m1_data_path
        elif freq == Freq.m5:
            library = m5_data_path
        elif freq == Freq.m15:
            library = m15_data_path
        elif freq == Freq.m30:
            library = m30_data_path
        elif freq == Freq.m60:
            library = m60_data_path
        # ============== add more table here ==============
        else:
            raise NotImplementedError
        if code is None:
            instruments = store['instrument_list'].read('A').data
            instruments = instruments[instruments['type'] == 'stock']
            instruments = instruments.index.to_list()
            instruments = list(set(instruments).intersection(store[library].list_symbols()))
        else:
            if isinstance(code, str):
                instruments = [code]
            elif isinstance(code, list):
                instruments = code
            else:
                raise ValueError('code must be type of str or list, but {} is given.'.format(type(code)))

        data = read_arctic_version_store(store, library, instruments, MAX_WORKER, verbose=verbose)
        if add_limit and freq not in [Freq.W1, Freq.M1]:
            limit_data = read_arctic_version_store(store, limit_data_path, instruments, MAX_WORKER, verbose=verbose)

    elif 'Local' in config2.sections():
        data_folder = os.path.join(project_deploy_path, config2['Local']['relative_path'],
                                   config2['Local']['Market']
                                   )
        if not os.path.exists(data_folder):
            print('relative path not found. Try to find the absolute_path.')
            data_folder = os.path.join(config2['Local']['absolute_path'], config2['Local']['Market'])
            if not os.path.exists(data_folder):
                raise ValueError('data folder path: {} not exists'.format(data_folder))
        if freq == Freq.D1:
            path = os.path.join(data_folder, daily_data_path)
            if add_limit:
                limit_data_path = os.path.join(data_folder, limit_data_path)
        elif freq == Freq.W1:
            path = os.path.join(data_folder, weekly_data_path)
        elif freq == Freq.M1:
            path = os.path.join(data_folder, monthly_data_path)
        # ============== add more table here ==============
        else:
            raise NotImplementedError

        if path.endswith('.parquet'):
            data = pd.read_parquet(path, columns=cols)
        else:
            raise NotImplementedError('Currently only support parquet')

        if add_limit and freq == Freq.D1:
            if path.endswith('.parquet'):
                limit_data = pd.read_parquet(limit_data_path)
        if code is not None:
            if isinstance(code, (str, list)):
                data = data.loc[(slice(None), code), :]
            else:
                raise ValueError('code must be type of str or list, but {} is given.'.format(type(code)))
    else:
        raise ValueError()

    data = data.sort_index()
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        data = data.loc[start_date:]
        if add_limit and freq not in [Freq.W1, Freq.M1]:
            limit_data = limit_data.loc[start_date:]

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        data = data.loc[:end_date]
        if add_limit and freq == Freq.D1:
            limit_data = limit_data.loc[:end_date]

    if len(data) == 0:
        return pd.DataFrame()

    if adjust:
        for price in PRICE_COLS:
            if price in cols:
                data['adj_{}'.format(price)] = data[price] * data[FACTOR_COL]
                cols.append('adj_{}'.format(price))

    data = data[list(cols)]
    data = data.sort_index()
    data.index.names = ['date', 'code']
    data = data[~data.index.duplicated()]  # may have some problem with data source

    if add_limit:
        if freq == Freq.D1:
            limit_data = limit_data[~limit_data.index.duplicated()]
            data = panel_df_join(limit_data, data)
            data = data[cols + ['high_limit', 'low_limit', 'paused']]
        elif freq not in [Freq.W1, Freq.M1]:
            limit_data = limit_data[~limit_data.index.duplicated()]
            data = df_merge_asof(limit_data, data)

    if eod_time_adjust:
        if freq in [Freq.D1, Freq.W1, Freq.M1]:
            data = data.reset_index(level=0, drop=False)
            data['date'] = data['date'] + pd.Timedelta(hours=15)
            data = data.set_index('date', append=True)
            data = data.swaplevel(0, 1)

    data = data.dropna(how='all')

    if index is False:
        data = data.reset_index(drop=False)

    if verbose == 1:
        end_time = datetime.datetime.now()
        print('finish. Time used: {} seconds'.format((end_time - start_time).seconds))
        print(data.info())
    elif verbose == 2:
        end_time = datetime.datetime.now()
        print('finish. Time used: {} seconds'.format((end_time - start_time).seconds))
        # todo
    return data


def get_limit_and_paused(code: Optional[Union[str, List[str]]] = None,
                         start_date: Optional[Union[str, datetime.datetime]] = None,
                         end_date: Optional[Union[str, datetime.datetime]] = None,
                         config_path: str = '../../cfg/data_input.ini', ):
    bars = get_bars(code, start_date, end_date, config_path=config_path,
                    cols=('close',),
                    add_limit=True, eod_time_adjust=False, adjust=False)
    bars['close_high_limit'] = (bars['close'] == bars['high_limit']).astype(int)
    bars['close_low_limit'] = (bars['close'] == bars['low_limit']).astype(int)
    return bars[['paused', 'close_high_limit', 'close_low_limit']]


def get_intraday_bars(code: str,
                      freq: Freq,
                      start_date: Optional[Union[str, datetime.datetime]] = None,
                      end_date: Optional[Union[str, datetime.datetime]] = None,
                      adjust: bool = True,
                      verbose=1,
                      config_path: str = '../../cfg/data_input.ini'
                      ):
    config = configparser.ConfigParser()
    config.read(config_path)
    if 'Source' not in config.sections():
        raise ValueError('Source section not in config. PLease check config path and config file')

    market_data_source = config['Source']['market_data']
    project_deploy_path = config['Source']['project_deploy_path']
    config2 = configparser.ConfigParser()
    config2.read(os.path.join(project_deploy_path, 'cfg', market_data_source), encoding='utf-8')
    if len(config2.sections()) == 0:
        raise ValueError('Empty config. PLease check config path and config file for {}'
                         .format(os.path.join(project_deploy_path, 'cfg', market_data_source)))

    start_time = datetime.datetime.now()
    PRICE_COLS = config2['Market'].get('price_cols', 'open,high,low,close').split(',')
    # ============== add more table here ==============

    if 'Arctic' in config2.sections():
        store = get_arctic_store(config2)
        m1_data_path = config2['Market']['m1_data']
        m5_data_path = config2['Market']['m5_data']
        m15_data_path = config2['Market']['m15_data']
        m30_data_path = config2['Market']['m30_data']
        m60_data_path = config2['Market']['m60_data']
        MAX_WORKER = int(config2['Arctic'].get('MAX_WORKER', 10))
        if freq == Freq.m1:
            library = m1_data_path
        elif freq == Freq.m5:
            library = m5_data_path
        elif freq == Freq.m15:
            library = m15_data_path
        elif freq == Freq.m30:
            library = m30_data_path
        elif freq == Freq.m60:
            library = m60_data_path
        # ============== add more table here ==============
        else:
            raise NotImplementedError

        instruments = [code]
        data = read_arctic_version_store(store, library, instruments, MAX_WORKER, verbose=verbose)
    elif 'Local' in config2.sections():
        data_folder = os.path.join(project_deploy_path, config2['Local']['relative_path'],
                                   config2['Local']['Market']
                                   )
        if not os.path.exists(data_folder):
            print('relative path not found. Try to find the absolute_path.')
            data_folder = os.path.join(config2['Local']['absolute_path'], config2['Local']['Market'])
            if not os.path.exists(data_folder):
                raise ValueError('data folder path: {} not exists'.format(data_folder))

        m5_data_path = config2['Market']['m5_data']
        if freq == Freq.m5:
            path = os.path.join(data_folder, m5_data_path, '{}.parquet'.format(code))
        else:
            raise NotImplementedError

        if path.endswith('.parquet'):
            data = pd.read_parquet(path)
        else:
            raise NotImplementedError('Currently only support parquet')

    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        data = data.loc[start_date:]

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        data = data.loc[:end_date]

    if len(data) == 0:
        return pd.DataFrame()

    if adjust:
        for price in PRICE_COLS:
            data['adj_{}'.format(price)] = data[price] * data[price]

    return data


if __name__ == '__main__':
    data = get_bars(start_date='2022-01-01', freq=Freq.D1, eod_time_adjust=False, add_limit=True)
    # data = get_limit_and_paused()
    # get_intraday_bars('000001.SZ', Freq.m5)
