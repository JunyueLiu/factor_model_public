import configparser
import datetime
import os
from enum import Enum
from typing import Optional, List, Union

import pandas as pd

from data_management.dataIO.utils import get_arctic_store, read_arctic_version_store


class Exotic(Enum):
    analyst_forecast = 'analyst_forecast'
    citic_hf_basic_operator = 'citic_hf_basic_operator'
    hk_holding = 'hk_holding'
    announcement = 'announcement'
    hk_market_moneyflow = 'hk_market_moneyflow'
    xy_gmm = 'xy_gmm'
    xy_gmm_1m = 'xy_gmm_1m'
    xy_gmm_5m_rolling = 'xy_gmm_5m_rolling'


def get_exotic(table: Exotic,
               code: Optional[Union[str or List[str]]] = None,
               pub_start_date: Optional[Union[str, datetime.datetime]] = None,
               pub_end_date: Optional[Union[str, datetime.datetime]] = None,
               config_path: str = '../../cfg/data_input.ini',
               cols: Optional[List[str]] = None,
               verbose=1):
    config = configparser.ConfigParser()
    config.read(config_path)
    if 'Source' not in config.sections():
        raise ValueError('Source section not in config. PLease check config path and config file')

    exotic_data_source = config['Source']['exotic_data']
    project_deploy_path = config['Source']['project_deploy_path']
    config2 = configparser.ConfigParser()
    config2.read(os.path.join(project_deploy_path, 'cfg', exotic_data_source), encoding='utf-8')
    if len(config2.sections()) == 0:
        raise ValueError('Empty config. PLease check config path and config file for {}'
                         .format(os.path.join(project_deploy_path, 'cfg', exotic_data_source)))

    start_time = datetime.datetime.now()
    analyst_forecast_path = config2['Exotic']['analyst_forecast']
    citic_hf_basic_operator_path = config2['Exotic']['citic_hf_basic_operator']
    hk_holding_path = config2['Exotic']['hk_holding']
    announcement_path = config2['Exotic']['announcement']
    hk_market_moneyflow = config2['HK_market_moneyflow']['hk_market_moneyflow']
    xy_gmm_path = config2['Exotic']['xy_gmm']
    xy_gmm_1m_path = config2['Exotic']['xy_gmm_1m']
    xy_gmm_5m_rolling_path = config2['Exotic']['xy_gmm_5m_rolling']
    # ============== add more table here ============

    if 'Arctic' in config2.sections():
        store = get_arctic_store(config2)
        MAX_WORKER = int(config2['Arctic'].get('MAX_WORKER', 10))
        if table == Exotic.analyst_forecast:
            library = analyst_forecast_path
        elif table == Exotic.citic_hf_basic_operator:
            library = citic_hf_basic_operator_path
        elif table == Exotic.hk_holding:
            library = hk_holding_path
        elif table == Exotic.announcement:
            library = announcement_path
        elif table == Exotic.hk_market_moneyflow:
            library = config2['HK_market_moneyflow']['hk_market_moneyflow']
        # ============== add more table here ============
        else:
            raise NotImplementedError
        if code:
            if isinstance(code, str):
                instruments = [code]
            elif isinstance(code, list):
                instruments = code
            else:
                raise ValueError('code must be type of str or list, but {} is given.'.format(type(code)))
        else:
            instruments = list(store[library].list_symbols())

        if table == Exotic.analyst_forecast:
            data = read_arctic_version_store(store, library, instruments, MAX_WORKER, False)
        elif table == Exotic.hk_market_moneyflow:
            data = read_arctic_version_store(store, library, ['hk_market_moneyflow'], 1, False)

        else:
            data = read_arctic_version_store(store, library, instruments, MAX_WORKER)

    elif 'Local' in config2.sections():
        data_folder = os.path.join(project_deploy_path, config2['Local']['relative_path'],
                                   config2['Local']['Exotic']
                                   )
        if not os.path.exists(data_folder):
            print('relative path not found. Try to find the absolute_path.')
            data_folder = os.path.join(config2['Local']['absolute_path'], config2['Local']['Exotic'])
            if not os.path.exists(data_folder):
                raise ValueError('data folder path: {} not exists'.format(data_folder))

        if Exotic.analyst_forecast == table:
            path = os.path.join(data_folder, analyst_forecast_path)
        elif table == Exotic.citic_hf_basic_operator:
            path = os.path.join(data_folder, citic_hf_basic_operator_path)
        elif table == Exotic.hk_holding:
            path = os.path.join(data_folder, hk_holding_path)
        elif table == Exotic.announcement:
            path = os.path.join(data_folder, announcement_path)
        elif table == Exotic.hk_market_moneyflow:
            path = os.path.join(project_deploy_path, config2['Local']['relative_path'],
                                'HK_market_moneyflow', 'hk_market_moneyflow.parquet')
        elif table == Exotic.xy_gmm:
            path = os.path.join(data_folder, xy_gmm_path)
        elif table == Exotic.xy_gmm_1m:
            path = os.path.join(data_folder, xy_gmm_1m_path)
        elif table == Exotic.xy_gmm_5m_rolling:
            path = os.path.join(data_folder, xy_gmm_5m_rolling_path)
        # ============== add more table here ============
        else:
            raise NotImplementedError

        if path.endswith('.parquet'):
            if cols:
                data = pd.read_parquet(path, columns=cols + ['code'])
            else:
                data = pd.read_parquet(path)
        else:
            raise NotImplementedError('Currently only support parquet')

        if code is not None:
            if isinstance(code, list):
                data = data[data['code'].isin(code)]
            elif isinstance(code, str):
                data = data[data['code'] == code]
            else:
                raise ValueError('code must be type of str or list, but {} is given.'.format(type(code)))
        if table in [Exotic.hk_holding, Exotic.citic_hf_basic_operator, Exotic.announcement]:
            data.index = pd.to_datetime(data.index)
            data = data.set_index('code', append=True)
    else:
        raise ValueError()

    if pub_start_date is not None:
        start_date = pd.to_datetime(pub_start_date)
        if 'pub_date' in data:
            data = data[data['pub_date'] >= start_date]
        else:
            data = data.sort_index().loc[start_date:]

    if pub_end_date is not None:
        end_date = pd.to_datetime(pub_end_date)
        if 'pub_date' in data:
            data = data[data['pub_date'] < end_date]
        else:
            data = data.sort_index().loc[:end_date]

    if cols:
        data = data[cols]
    if table == Exotic.analyst_forecast:
        data = data.sort_values(['code', 'pub_date'])
        data = data.reset_index(drop=True)
    else:
        data = data.sort_index()
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
    data = get_exotic(Exotic.xy_gmm_5m_rolling)
