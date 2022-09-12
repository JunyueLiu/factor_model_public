import configparser
import datetime
import os
from enum import Enum
from typing import Optional, List, Union

import pandas as pd

from data_management.dataIO.utils import get_arctic_store, read_arctic_version_store


class TradeTable(Enum):
    daily_basic = 'daily_basic'
    money_flow = 'money_flow'
    tushare_moneyflow = 'tushare_moneyflow'
    ST = 'ST'


def get_trade(table: TradeTable,
              code: Optional[Union[str or List[str]]] = None,
              start_date: Optional[Union[str, datetime.datetime]] = None,
              end_date: Optional[Union[str, datetime.datetime]] = None,
              config_path: str = '../../cfg/data_input.ini',
              cols: Optional[List[str]] = None,
              index: bool = True,
              verbose=1):
    """

    :param table:
    :param code:
    :param start_date:
    :param end_date:
    :param config_path:
    :param cols:
    :param index:
    :param verbose:
    :return:
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    if 'Source' not in config.sections():
        raise ValueError('Source section not in config. PLease check config path and config file')

    trade_data_source = config['Source']['fundamental_data']
    project_deploy_path = config['Source']['project_deploy_path']
    config2 = configparser.ConfigParser()
    config2.read(os.path.join(project_deploy_path, 'cfg', trade_data_source), encoding='utf-8')
    if len(config2.sections()) == 0:
        raise ValueError('Empty config. PLease check config path and config file for {}'
                         .format(os.path.join(project_deploy_path, 'cfg', trade_data_source)))

    start_time = datetime.datetime.now()
    daily_basic_path = config2['Trade']['daily_basic']
    money_flow_path = config2['Trade']['money_flow']
    tushare_moneyflow_path = config2['Trade']['tushare_moneyflow']
    st_path = config2['Trade']['ST']

    # ============== add more table here ============
    if 'Arctic' in config2.sections():
        store = get_arctic_store(config2)
        MAX_WORKER = int(config2['Arctic'].get('MAX_WORKER', 10))
        if table == TradeTable.daily_basic:
            library = daily_basic_path
        elif table == TradeTable.money_flow:
            library = money_flow_path
        elif table == TradeTable.tushare_moneyflow:
            library = tushare_moneyflow_path
        elif table == TradeTable.ST:
            library = st_path
        # ============== add more table here ==============
        else:
            raise NotImplementedError
        if code is None:
            instruments = store[library].list_symbols()
        else:
            if isinstance(code, str):
                instruments = [code]
            elif isinstance(code, list):
                instruments = code
            else:
                raise ValueError('code must be type of str or list, but {} is given.'.format(type(code)))
        data = read_arctic_version_store(store, library, instruments, MAX_WORKER, verbose=0)
    elif 'Local' in config2.sections():
        data_folder = os.path.join(project_deploy_path, config2['Local']['relative_path'],
                                   config2['Local']['Trade']
                                   )
        if not os.path.exists(data_folder):
            print('relative path not found. Try to find the absolute_path.')
            data_folder = os.path.join(config2['Local']['absolute_path'], config2['Local']['Trade'])
            if not os.path.exists(data_folder):
                raise ValueError('data folder path: {} not exists'.format(data_folder))

        if TradeTable.money_flow == table:
            path = os.path.join(data_folder, money_flow_path)
        elif TradeTable.daily_basic == table:
            path = os.path.join(data_folder, daily_basic_path)
        elif TradeTable.tushare_moneyflow == table:
            path = os.path.join(data_folder, tushare_moneyflow_path)
        elif TradeTable.ST == table:
            path = os.path.join(data_folder, st_path)
        # ============== add more table here ============
        else:
            raise NotImplementedError

        if path.endswith('.parquet'):
            data = pd.read_parquet(path)
        else:
            raise NotImplementedError('Currently only support parquet')
        if code is not None:
            if isinstance(code, (str, list)):
                data = data.loc[(slice(None), code), :]
            else:
                raise ValueError('code must be type of str or list, but {} is given.'.format(type(code)))

    else:
        raise NotImplementedError

    if not cols:
        cols = data.columns

    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        data = data.loc[start_date:]

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        data = data.loc[:end_date]

    data = data[cols]
    data = data.sort_index()
    data.index.names = ['date', 'code']

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


if __name__ == '__main__':
    # data = get_trade(TradeTable.money_flow, code='000001.SZ')
    data = get_trade(TradeTable.daily_basic, start_date='2022-01-01')