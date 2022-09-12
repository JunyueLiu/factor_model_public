import configparser
import datetime
import os
from enum import Enum
from typing import Optional, Union, Tuple

import pandas as pd

from data_management.dataIO.utils import get_arctic_store, read_arctic_version_store


class IndexTicker(Enum):
    sz50 = '000016.SH'
    sz180 = '000010.SH'
    csi300 = '000300.SH'
    zz500 = '000905.SH'
    zz800 = '000906.SH'
    zz1000 = '000852.SH'
    szr100 = '399004.SZ'
    ft_A50 = 'XIN9'


class Freq(Enum):
    D1 = "D1"
    W1 = "W1"
    M1 = "M1"


def get_bars(code: IndexTicker,
             start_date: Optional[Union[str, datetime.datetime]] = None,
             end_date: Optional[Union[str, datetime.datetime]] = None,
             cols: Tuple = ('code', 'open', 'high', 'low', 'close', 'money'),
             freq: Freq = Freq.D1,
             config_path: str = '../../cfg/data_input.ini',
             verbose: int = 1,
             index: bool = True,
             eod_time_adjust: bool = True,
             ) -> pd.DataFrame:
    config = configparser.ConfigParser()
    config.read(config_path)
    if 'Source' not in config.sections():
        raise ValueError('Source section not in config. PLease check config path and config file')

    market_data_source = config['Source']['index_data']
    project_deploy_path = config['Source']['project_deploy_path']
    config2 = configparser.ConfigParser()
    config2.read(os.path.join(project_deploy_path, 'cfg', market_data_source), encoding='utf-8')
    if len(config2.sections()) == 0:
        raise ValueError('Empty config. PLease check config path and config file for {}'
                         .format(os.path.join(project_deploy_path, 'cfg', market_data_source)))

    start_time = datetime.datetime.now()
    daily_data_path = config2['Index']['daily_data']
    weekly_data_path = config2['Index']['weekly_data']
    monthly_data_path = config2['Index']['monthly_data']
    # ============== add more table here ==============

    if 'Arctic' in config2.sections():

        store = get_arctic_store(config2)
        MAX_WORKER = int(config2['Arctic'].get('MAX_WORKER', 10))
        if freq == Freq.D1:
            library = daily_data_path
        elif freq == Freq.M1:
            library = monthly_data_path
        elif freq == Freq.W1:
            library = weekly_data_path
        # ============== add more table here ==============
        else:
            raise NotImplementedError
        data = read_arctic_version_store(store, library, [code], MAX_WORKER)

    elif 'Local' in config2.sections():

        data_folder = os.path.join(project_deploy_path, config2['Local']['relative_path'],
                                   config2['Local']['Index']
                                   )
        if not os.path.exists(data_folder):
            print('relative path not found. Try to find the absolute_path.')
            data_folder = os.path.join(config2['Local']['absolute_path'], config2['Local']['Index'])
            if not os.path.exists(data_folder):
                raise ValueError('data folder path: {} not exists'.format(data_folder))
        if freq == Freq.D1:
            path = os.path.join(data_folder, daily_data_path, config2['Index'][code.name])
        elif freq == Freq.W1:
            path = os.path.join(data_folder, weekly_data_path, config2['Index'][code.name])
        elif freq == Freq.M1:
            path = os.path.join(data_folder, monthly_data_path, config2['Index'][code.name])
        # ============== add more table here ==============
        else:
            raise NotImplementedError

        if path.endswith('.parquet'):
            data = pd.read_parquet(path)
        else:
            raise NotImplementedError('Currently only support parquet')

    else:
        raise ValueError()

    cols = list(cols)

    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        data = data.loc[start_date:]

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        data = data.loc[:end_date]

    data = data[list(cols)]
    data = data.sort_index()
    data.index.names = ['date']

    if eod_time_adjust:
        if freq in [Freq.D1, Freq.W1, Freq.M1]:
            data = data.reset_index(drop=False)
            data['date'] = data['date'] + pd.Timedelta(hours=15)
            data = data.set_index('date')

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
    data = get_bars(IndexTicker.sz50)
