import configparser
import datetime
import os
from collections import defaultdict
from enum import Enum
from typing import Optional, Union, List, Dict

import pandas as pd

from data_management.dataIO.trading_calendar import get_trading_date, Market
from data_management.dataIO.utils import get_arctic_store, read_pickle


class IndexTicker(Enum):
    sz50 = '000016.SH'
    sz180 = '000010.SH'
    csi300 = '000300.SH'
    zz500 = '000905.SH'
    zz800 = '000906.SH'
    zz1000 = '000852.SH'
    szr100 = '399004.SZ'


class IndustryCategory(Enum):
    sw_l1 = 'sw_l1_stocks'
    sw_l2 = 'sw_l2_stocks'
    sw_l3 = 'sw_l3_stocks'
    zjw = 'zjw_stocks'


class Freq(Enum):
    D1 = "D1"


class Exchange(Enum):
    Shenzhen = 'SZ'
    Shanghai = 'SH'


def get_index_component(code: IndexTicker,
                        date: Optional[Union[str, datetime.datetime,
                                             datetime.date, pd.Timestamp, List]] = None,
                        config_path: str = '../../cfg/data_input.ini',
                        ) -> Dict[pd.Timestamp, List[str]]:
    config = configparser.ConfigParser()
    config.read(config_path)
    if 'Source' not in config.sections():
        raise ValueError('Source section not in config. PLease check config path and config file')

    component_data_source = config['Source']['component_data']
    project_deploy_path = config['Source']['project_deploy_path']
    config2 = configparser.ConfigParser()
    config2.read(os.path.join(project_deploy_path, 'cfg', component_data_source), encoding='utf-8')
    if len(config2.sections()) == 0:
        raise ValueError('Empty config. PLease check config path and config file for {}'
                         .format(os.path.join(project_deploy_path, 'cfg', component_data_source)))

    if 'Arctic' in config2.sections():
        store = get_arctic_store(config2)
        dic = store['index_stocks'].read(code.value).data
    elif 'Local' in config2.sections():
        data_folder = os.path.join(project_deploy_path, config2['Local']['relative_path'],
                                   'index_stocks'
                                   )
        if not os.path.exists(data_folder):
            print('relative path not found. Try to find the absolute_path.')
            data_folder = os.path.join(config2['Local']['absolute_path'], 'index_stocks')
            if not os.path.exists(data_folder):
                raise ValueError('data folder path: {} not exists'.format(data_folder))

        path = os.path.join(data_folder, code.value + '.pickle')
        dic = read_pickle(path)
    else:
        raise NotImplementedError

    if date is not None:
        if isinstance(date, list):
            date = pd.to_datetime(date)
            new_d = {d: dic.get(d) for d in date}
        else:
            date = pd.to_datetime(date)
            new_d = {date: dic.get(date)}
        dic = new_d

    return dic


def get_index_weights(code: IndexTicker,
                      config_path: str = '../../cfg/data_input.ini',
                      ) -> pd.DataFrame:
    config = configparser.ConfigParser()
    config.read(config_path)
    if 'Source' not in config.sections():
        raise ValueError('Source section not in config. PLease check config path and config file')

    component_data_source = config['Source']['component_data']
    project_deploy_path = config['Source']['project_deploy_path']
    config2 = configparser.ConfigParser()
    config2.read(os.path.join(project_deploy_path, 'cfg', component_data_source), encoding='utf8')
    if len(config2.sections()) == 0:
        raise ValueError('Empty config. PLease check config path and config file for {}'
                         .format(os.path.join(project_deploy_path, 'cfg', component_data_source)))

    if 'Arctic' in config2.sections():
        store = get_arctic_store(config2)
        df = store['index_weights'].read(code.value).data
    elif 'Local' in config2.sections():
        data_folder = os.path.join(project_deploy_path, config2['Local']['relative_path'],
                                   'index_weights'
                                   )
        if not os.path.exists(data_folder):
            print('relative path not found. Try to find the absolute_path.')
            data_folder = os.path.join(config2['Local']['absolute_path'], 'index_weights')
            if not os.path.exists(data_folder):
                raise ValueError('data folder path: {} not exists'.format(data_folder))

        path = os.path.join(data_folder, code.value + '.parquet')
        df = pd.read_parquet(path)
    else:
        raise NotImplementedError

    return df


def get_industry_info(category: IndustryCategory, config_path: str = '../../cfg/data_input.ini', ):
    config = configparser.ConfigParser()
    config.read(config_path)
    if 'Source' not in config.sections():
        raise ValueError('Source section not in config. PLease check config path and config file')

    component_data_source = config['Source']['component_data']
    project_deploy_path = config['Source']['project_deploy_path']
    config2 = configparser.ConfigParser()
    config2.read(os.path.join(project_deploy_path, 'cfg', component_data_source), encoding='utf8')
    if len(config2.sections()) == 0:
        raise ValueError('Empty config. PLease check config path and config file for {}'
                         .format(os.path.join(project_deploy_path, 'cfg', component_data_source)))

    if 'Arctic' in config2.sections():
        store = get_arctic_store(config2)
        data = store['industries'].read(category.name).data
        return data
    elif 'Local' in config2.sections():
        data_folder = os.path.join(project_deploy_path, config2['Local']['relative_path'],
                                   'industries'
                                   )
        if not os.path.exists(data_folder):
            print('relative path not found. Try to find the absolute_path.')
            data_folder = os.path.join(config2['Local']['absolute_path'], 'industries')
            if not os.path.exists(data_folder):
                raise ValueError('data folder path: {} not exists'.format(data_folder))

        data = pd.read_csv(os.path.join(data_folder, category.name + '.csv'), index_col=0)
        return data


def get_industry_component(category: IndustryCategory,
                           industry_code: Optional[Union[str, List[str]]] = None,
                           date: Optional[Union[str, datetime.datetime, datetime.date, List]] = None,
                           config_path: str = '../../cfg/data_input.ini',
                           ) -> Dict[str, Dict[pd.Timestamp, List[str]]]:
    config = configparser.ConfigParser()
    config.read(config_path)
    if 'Source' not in config.sections():
        raise ValueError('Source section not in config. PLease check config path and config file')

    component_data_source = config['Source']['component_data']
    project_deploy_path = config['Source']['project_deploy_path']
    config2 = configparser.ConfigParser()
    config2.read(os.path.join(project_deploy_path, 'cfg', component_data_source), encoding='utf-8')
    if len(config2.sections()) == 0:
        raise ValueError('Empty config. PLease check config path and config file for {}'
                         .format(os.path.join(project_deploy_path, 'cfg', component_data_source)))

    path = category.value
    dic = {}
    if 'Arctic' in config2.sections():
        store = get_arctic_store(config2)
        if industry_code is None:
            for code in store[path].list_symbols():
                d = store[path].read(code).data
                dic[code] = d
        elif isinstance(industry_code, list):
            for code in industry_code:
                d = store[path].read(code).data
                dic[code] = d
        elif isinstance(industry_code, str):
            d = store[path].read(industry_code).data
            dic[industry_code] = d

    elif 'Local' in config2.sections():
        data_folder = os.path.join(project_deploy_path, config2['Local']['relative_path'],
                                   'industry_stocks', path
                                   )
        if not os.path.exists(data_folder):
            print('relative path not found. Try to find the absolute_path.')
            data_folder = os.path.join(config2['Local']['absolute_path'], 'industry_stocks', path)
            if not os.path.exists(data_folder):
                raise ValueError('data folder path: {} not exists'.format(data_folder))
        dic = {}
        if industry_code is None:
            for code in os.listdir(data_folder):
                d = read_pickle(os.path.join(data_folder, code))
                dic[code.replace('.pickle', '')] = d
        elif isinstance(industry_code, list):
            for code in industry_code:
                d = read_pickle(os.path.join(data_folder, code + '.pickle'))
                dic[code] = d
        elif isinstance(industry_code, str):
            d = read_pickle(os.path.join(data_folder, industry_code + '.pickle'))
            dic[industry_code] = d

    else:
        raise NotImplementedError

    if date is not None:
        if isinstance(date, list):
            date = pd.to_datetime(date)
            new_d = {i: {d: data.get(d) for d in date} for i, data in dic.items()}
        else:
            date = pd.to_datetime(date)
            new_d = {i: {date: data.get(date)} for i, data in dic.items()}
        dic = new_d

    return dic


def get_north_connect_component(date: Optional[Union[str, datetime.datetime, datetime.date,
                                                     List]] = None,
                                config_path: str = '../../cfg/data_input.ini'
                                ) -> Dict[str, Dict[pd.Timestamp, List[str]]]:
    config = configparser.ConfigParser()
    config.read(config_path)
    if 'Source' not in config.sections():
        raise ValueError('Source section not in config. PLease check config path and config file')

    component_data_source = config['Source']['component_data']
    project_deploy_path = config['Source']['project_deploy_path']
    config2 = configparser.ConfigParser()
    config2.read(os.path.join(project_deploy_path, 'cfg', component_data_source), encoding='utf-8')
    if len(config2.sections()) == 0:
        raise ValueError('Empty config. PLease check config path and config file for {}'
                         .format(os.path.join(project_deploy_path, 'cfg', component_data_source)))

    #     沪港通2014117开通，深港通20161205开通
    if 'Arctic' in config2.sections():
        store = get_arctic_store(config2)
        market_connect = store['MarketConnect'].read('AShares').data

    elif 'Local' in config2.sections():
        data_folder = os.path.join(project_deploy_path, config2['Local']['relative_path'],
                                   config2['Component']['market_connect']
                                   )
        if not os.path.exists(data_folder):
            print('relative path not found. Try to find the absolute_path.')
            data_folder = os.path.join(config2['Local']['absolute_path'], config2['Component']['market_connect'])
            if not os.path.exists(data_folder):
                raise ValueError('data folder path: {} not exists'.format(data_folder))
        market_connect = pd.read_csv(os.path.join(data_folder, 'AShares.csv'), index_col=0, parse_dates=['date'])

    trading_dates = get_trading_date(Market.AShares, start_date='2014-11-07',
                                     config_path=
                                     os.path.join(project_deploy_path, 'cfg', 'data_input.ini'))
    trading_dates = pd.to_datetime(trading_dates)
    trading_dates = trading_dates[trading_dates <= datetime.datetime.now()]
    dic = defaultdict(list)
    for code, x in market_connect.groupby('code'):
        if len(x) == 1:
            start_date = x.iloc[0].date
            for td in trading_dates:
                if td >= start_date:
                    dic[td].append(code)
        else:
            start_date = []
            end_date = []
            for _, row in x.iterrows():
                if 'eligible for sell only' in row.Change:
                    if 'Transfer' in row.Change:
                        end_date.append(row.date)
                    elif 'Addition (' in row.Change:
                        start_date.append(row.date)
                    elif 'Addition to' in row.Change:
                        end_date.append(row.date)
                elif 'Addition' in row.Change:
                    start_date.append(row.date)
                elif 'Removal' in row.Change:
                    end_date.append(row.date)
                elif 'Buy orders suspended' in row.Change:
                    end_date.append(row.date)
                elif 'Buy orders resumed' in row.Change:
                    start_date.append(row.date)
                elif 'Code and Stock Name are changed':
                    end_date.append(row.date)

            if len(end_date) == 0:
                for td in trading_dates:
                    if td >= start_date[0]:
                        dic[td].append(code)
            else:
                for e, s in zip(end_date, start_date):
                    for td in trading_dates:
                        if s <= td < e:
                            dic[td].append(code)
    dic = dict(sorted(dic.items()))
    if date is not None:
        if isinstance(date, list):
            date = pd.to_datetime(date)
            new_d = {i: data for i, data in dic.items() for d in date if d == i}
        else:
            date = pd.to_datetime(date)
            new_d = {i: {date: data.get(date)} for i, data in dic.items()}
        dic = new_d
    return dic


def get_bars(industry_code: Optional[Union[str, List[str]]] = None,
             industry_category=IndustryCategory.sw_l1,
             freq: Freq = Freq.D1,
             start_date: Optional[Union[str, datetime.datetime]] = None,
             end_date: Optional[Union[str, datetime.datetime]] = None,
             config_path: str = '../../cfg/data_input.ini', ):
    # todo
    if industry_category != IndustryCategory.sw_l1:
        raise ValueError('only have sw1')

    if freq != Freq.D1:
        raise ValueError('only have daily data')

    config = configparser.ConfigParser()
    config.read(config_path)
    if 'Source' not in config.sections():
        raise ValueError('Source section not in config. PLease check config path and config file')

    component_data_source = config['Source']['component_data']
    project_deploy_path = config['Source']['project_deploy_path']
    config2 = configparser.ConfigParser()
    config2.read(os.path.join(project_deploy_path, 'cfg', component_data_source), encoding='utf-8')
    if len(config2.sections()) == 0:
        raise ValueError('Empty config. PLease check config path and config file for {}'
                         .format(os.path.join(project_deploy_path, 'cfg', component_data_source)))

    if 'Arctic' in config2.sections():
        store = get_arctic_store(config2)
        path = industry_category.value

        results = []
        if industry_code is None:
            for code in store[path].list_symbols():
                df = store[path].read(code).data
                results.append(df)
        else:
            for code in industry_code:
                df = store[path].read(code).data
                results.append(df)
        data = pd.concat(results)
        data = data.set_index('code', append=True).sort_index()

    elif 'Local' in config2.sections():
        data_path = os.path.join(project_deploy_path,
                                 config2['Local']['relative_path'],
                                 'sw_l1_1d', '1d.parquet')
        if not os.path.exists(data_path):
            print('relative path not found. Try to find the absolute_path.')
            data_path = os.path.join(config2['Local']['absolute_path'], 'sw_l1_1d', '1d.parquet')
        if not os.path.exists(data_path):
            raise ValueError('data folder path: {} not exists'.format(data_path))
        data = pd.read_parquet(data_path)
        if industry_code is not None:
            data = data.loc[:, industry_code, :]

    else:
        raise NotImplementedError

    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        data = data.loc[start_date:]

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        data = data.loc[:end_date]

    for c in ['open', 'high', 'low', 'close']:
        data['adj_' + c] = data[c]

    data['factor'] = 1
    return data


if __name__ == '__main__':
    # d = get_index_component(IndexTicker.csi300)
    # d = get_industry_component(IndustryCategory.sw_l1)
    info = get_industry_info(IndustryCategory.sw_l1)
    # d = get_north_connect_component()
    # df = get_bars()
