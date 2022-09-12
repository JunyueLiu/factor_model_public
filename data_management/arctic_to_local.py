import configparser
import os
import pickle
from collections import Callable
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import pandas as pd
import tqdm

from data_management.dataIO.utils import get_arctic_store, save_pickle


def download_fundamental_from_arctic(arctic_obj, library, MAX_WORKER=5) -> pd.DataFrame:
    def _read(symbol):
        return arctic_obj[library].read(symbol).data

    if library == 'securities_info':
        data = _read('stock')
        return data

    instruments = arctic_obj[library].list_symbols()
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        results = list(tqdm.tqdm(executor.map(_read, instruments), total=len(instruments)))
    data = pd.concat(results, ignore_index=True)
    data = data.sort_values(['code', 'pub_date'])
    data = data.reset_index(drop=True)
    data = data.fillna(np.nan)
    data = data.replace('None', np.nan)
    for c in data:
        if 'date' in c:
            try:
                data[c] = pd.to_datetime(data[c])
            except:
                data[c] = data[c].astype(str)
        elif c in ['company_id', 'company_name', 'code', 'a_code', 'b_code', 'h_code']:
            data[c] = data[c].astype(str)
        elif c in ['report_type', 'source_id', 'report_type_id', 'type_id', 'type']:
            try:
                data[c] = data[c].astype(int)
            except:
                data[c] = data[c].astype(str)
        elif data[c].dtypes == object:
            try:
                data[c] = data[c].astype(float)
            except:
                data[c] = data[c].astype(str)

    return data


def download_market_data_from_arctic(arctic_obj, library='1d', ticker_type='stock', MAX_WORKER=5) -> pd.DataFrame:
    def _read(symbol):
        return arctic_obj[library].read(symbol).data

    instruments = arctic_obj['instrument_list'].read('A').data
    instruments = instruments[instruments['type'] == ticker_type]
    instruments = instruments.index.to_list()
    instruments = list(set(instruments).intersection(arctic_obj[library].list_symbols()))

    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        results = list(tqdm.tqdm(executor.map(_read, instruments), total=len(instruments)))
    data = pd.concat(results)
    data = data.set_index('code', append=True)
    data = data.sort_index()
    return data


def download_trade_data_from_arctic(arctic_obj, library, MAX_WORKER=5) -> pd.DataFrame:
    def _read(symbol):
        return arctic_obj[library].read(symbol).data

    instruments = arctic_obj[library].list_symbols()
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        results = list(tqdm.tqdm(executor.map(_read, instruments), total=len(instruments)))
    data = pd.concat(results)
    data = data.set_index('code', append=True)
    data = data.sort_index()
    return data


def download_index_data_from_arctic(arctic_obj, index_symbol, library='1d', MAX_WORKER=5) -> pd.DataFrame:
    def _read(symbol):
        return arctic_obj[library].read(symbol).data

    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        results = list(tqdm.tqdm(executor.map(_read, [index_symbol]), total=len([index_symbol])))
    data = pd.concat(results, )
    # data = data.set_index('code', append=True)
    data = data.sort_index()
    return data


def download_exotic_data_from_arctic(arctic_obj, library, MAX_WORKER=5) -> pd.DataFrame:
    def _read(symbol):
        return arctic_obj[library].read(symbol).data

    instruments = arctic_obj[library].list_symbols()
    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        results = list(tqdm.tqdm(executor.map(_read, instruments), total=len(instruments)))
    data = pd.concat(results)
    return data


def _sections_check(arctic_config, local_config, sections):
    if len(arctic_config.sections()) == 0:
        raise ValueError("arctic_config is empty. Please check path, {}".format(arctic_config_path))

    if len(local_config.sections()) == 0:
        raise ValueError("local_config is empty. Please check path, {}".format(local_config_path))

    if sections not in arctic_config.sections():
        raise ValueError('arctic_config doesn\'t contain {} section'.format(sections))

    if sections not in local_config.sections():
        raise ValueError('local_config doesn\'t contain {} section'.format(sections))

    for key in local_config[sections]:
        if key not in arctic_config[sections]:
            raise ValueError('{} in local config but not in arctic config. Check arctic config'.format(key))


def _download_sections(arctic_config, local_config, section, download_function: Callable, **kwargs):
    store = get_arctic_store(arctic_config)
    MAX_WORKER = int(arctic_config['Arctic'].get('MAX_WORKER', 10))

    data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../',
                                               local_config['Local']['relative_path']))
    if not os.path.exists(data_folder):
        data_folder = local_config['Local']['absolute_path']
        if not os.path.exists(data_folder):
            raise ValueError('data folder path: {} not exists'.format(data_folder))

    for key in local_config[section]:
        library = arctic_config[section][key]
        if not store.library_exists(library):
            continue
        print('Start download {}'.format(key))

        if not os.path.exists(os.path.join(data_folder, local_config['Local'].get(section, ''))):
            os.makedirs(os.path.join(data_folder, local_config['Local'].get(section, '')))
        local_save_path = os.path.join(data_folder, local_config['Local'].get(section, ''),
                                       local_config[section][key]
                                       )
        print('Will be downloaded to {}'.format(local_save_path))
        if library == '5m':
            continue

        data = download_function(arctic_obj=store, library=library, MAX_WORKER=MAX_WORKER, **kwargs)
        if local_save_path.endswith('.parquet'):
            data.to_parquet(local_save_path)
        elif local_save_path.endswith('.pickle'):
            with open(local_save_path, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        elif local_save_path.endswith('.csv'):
            data.to_csv(local_save_path)
        else:
            raise NotImplementedError
        print('Finish download {}'.format(key))


def _download_index_section(arctic_config, local_config):
    store = get_arctic_store(arctic_config)
    MAX_WORKER = int(arctic_config['Arctic'].get('MAX_WORKER', 10))

    data_folder = os.path.join('../', local_config['Local']['relative_path'])
    if not os.path.exists(data_folder):
        data_folder = local_config['Local']['absolute_path']
        if not os.path.exists(data_folder):
            raise ValueError('data folder path: {} not exists'.format(data_folder))

    index_targets = []
    for key in arctic_config['Index']:
        library = arctic_config['Index'][key]
        if not store.library_exists(library):
            index_targets.append((key, arctic_config['Index'][key]))

    for key in local_config['Index']:
        library = arctic_config['Index'][key]
        if not store.library_exists(library):
            continue

        for index in index_targets:
            if not os.path.exists(os.path.join(data_folder, local_config['Local'].get('Index', ''),
                                               local_config['Index'][key]
                                               )):
                os.makedirs(os.path.join(data_folder, local_config['Local'].get('Index', ''),
                                         local_config['Index'][key]
                                         ))
            local_save_path = os.path.join(data_folder, local_config['Local'].get('Index', ''),
                                           local_config['Index'][key],
                                           local_config['Index'][index[0]]
                                           )
            print('Will be downloaded to {}'.format(local_save_path))
            try:
                data = download_index_data_from_arctic(store, index[1], library, MAX_WORKER)
                if local_save_path.endswith('.parquet'):
                    data.to_parquet(local_save_path)
                else:
                    raise ValueError
                print('Finish download {}/{}'.format(key, index))
            except:
                pass


def _download_component_section(arctic_config, local_config):
    store = get_arctic_store(arctic_config)
    # MAX_WORKER = int(arctic_config['Arctic'].get('MAX_WORKER', 10))

    data_folder = os.path.join('../', local_config['Local']['relative_path'])
    if not os.path.exists(data_folder):
        data_folder = local_config['Local']['absolute_path']
        if not os.path.exists(data_folder):
            raise ValueError('data folder path: {} not exists'.format(data_folder))

    if 'index' in local_config['Component']:
        indexes = []
        for key in local_config['Index']:
            index_ticker = arctic_config['Index'][key]
            if not store.library_exists(index_ticker):
                indexes.append(index_ticker)

        for ticker in indexes:
            try:
                d = store['index_stocks'].read(ticker).data

                local_save_path = os.path.join(data_folder,
                                               local_config['Component']['index'])
                if not os.path.exists(local_save_path):
                    os.makedirs(local_save_path)
                local_save_path = os.path.join(local_save_path, ticker + '.pickle')
                with open(local_save_path, 'wb') as handle:
                    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except:
                pass

        for ticker in indexes:
            df = store['index_weights'].read(ticker).data  # type: pd.DataFrame

            local_save_path = os.path.join(data_folder,
                                           local_config['Component']['index_weights'])
            if not os.path.exists(local_save_path):
                os.makedirs(local_save_path)
            local_save_path = os.path.join(local_save_path, ticker + '.parquet')
            df.to_parquet(local_save_path)

    if 'industry' in local_config['Component']:
        industry_libraries = [c + '_stocks' for c in local_config['Industry']]
        for lib in industry_libraries:
            symbols = store[lib].list_symbols()
            for s in tqdm.tqdm(symbols):
                d = store[lib].read(s).data
                local_save_path = os.path.join(data_folder,
                                               local_config['Component']['industry'], lib)
                if not os.path.exists(local_save_path):
                    os.makedirs(local_save_path)
                local_save_path = os.path.join(local_save_path, s + '.pickle')
                with open(local_save_path, 'wb') as handle:
                    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if 'market_connect' in local_config['Component']:
        symbols = store['MarketConnect'].list_symbols()
        for s in tqdm.tqdm(symbols):
            df = store['MarketConnect'].read(s).data  # type: pd.DataFrame
            local_save_path = os.path.join(data_folder,
                                           local_config['Component']['market_connect'])
            if not os.path.exists(local_save_path):
                os.makedirs(local_save_path)
            local_save_path = os.path.join(local_save_path, s + '.csv')
            # with open(local_save_path, 'wb') as handle:
            #     pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
            df.to_csv(local_save_path)

    if 'sw_l1_1d' in local_config['Component']:
        library = arctic_config['Component']['sw_l1_1d']

        def _read(symbol):
            return store[library].read(symbol).data

        instruments = store[library].list_symbols()
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(tqdm.tqdm(executor.map(_read, instruments), total=len(instruments)))
        data = pd.concat(results)
        data = data.set_index('code', append=True).sort_index()
        local_save_path = os.path.join(data_folder,
                                       local_config['Component']['sw_l1_1d'])
        if not os.path.exists(local_save_path):
            os.makedirs(local_save_path)
        local_save_path = os.path.join(local_save_path, '1d' + '.parquet')
        data.to_parquet(local_save_path)


def _download_industries_section(arctic_config, local_config):
    store = get_arctic_store(arctic_config)
    # MAX_WORKER = int(arctic_config['Arctic'].get('MAX_WORKER', 10))

    data_folder = os.path.join('../', local_config['Local']['relative_path'])
    if not os.path.exists(data_folder):
        data_folder = local_config['Local']['absolute_path']
        if not os.path.exists(data_folder):
            raise ValueError('data folder path: {} not exists'.format(data_folder))
    data_folder = os.path.join(data_folder, 'industries')
    os.makedirs(data_folder, exist_ok=True)
    for key in local_config['Industry']:
        data = store['industries'].read(key).data  # type: pd.DataFrame
        data.to_csv(os.path.join(data_folder, key + '.csv'), )


def _download_calendar_section(arctic_config, local_config):
    store = get_arctic_store(arctic_config)

    data_folder = os.path.join('../', local_config['Local']['relative_path'])
    if not os.path.exists(data_folder):
        data_folder = local_config['Local']['absolute_path']
        if not os.path.exists(data_folder):
            raise ValueError('data folder path: {} not exists'.format(data_folder))
    data_folder = os.path.join(data_folder, 'calendar')
    os.makedirs(data_folder, exist_ok=True)
    for key in local_config['Calendar']:
        data = store['trading_calendar'].read(arctic_config['Calendar'][key]).data  # type: np.ndarray
        local_save_path = os.path.join(data_folder, local_config['Calendar'][key])
        save_pickle(data, local_save_path)

def _download_intraday(arctic_config, local_config, name):
    store = get_arctic_store(arctic_config)

    data_folder = os.path.join('../', local_config['Local']['relative_path'])
    if not os.path.exists(data_folder):
        data_folder = local_config['Local']['absolute_path']
        if not os.path.exists(data_folder):
            raise ValueError('data folder path: {} not exists'.format(data_folder))


    data_folder = os.path.join(data_folder, 'market', local_config['Market'][name])
    os.makedirs(data_folder, exist_ok=True)
    lib = arctic_config['Market'][name]

    instruments = store['instrument_list'].read('A').data
    instruments = instruments[instruments['type'] == 'stock']
    instruments = instruments.index.to_list()
    instruments = list(set(instruments).intersection(store[lib].list_symbols()))
    library = store.get_library(lib)
    for symbol in tqdm.tqdm(instruments):
        df = library.read(symbol).data
        df.to_parquet(os.path.join(data_folder, '{}.parquet'.format(symbol)))


def fundamental_arctic_to_local(arctic_config_path: str, local_config_path: str):
    """

    :param arctic_config_path:
    :param local_config_path:
    :return:
    """
    arctic_config = configparser.ConfigParser()
    arctic_config.read(os.path.abspath(arctic_config_path), encoding='utf-8')
    local_config = configparser.ConfigParser()
    local_config.read(os.path.abspath(local_config_path), encoding='utf-8')
    _sections_check(arctic_config, local_config, 'Fundamental')
    _download_sections(arctic_config, local_config, 'Fundamental', download_fundamental_from_arctic)


def market_arctic_to_local(arctic_config_path: str, local_config_path: str):
    """

    :param arctic_config_path:
    :param local_config_path:
    :return:
    """
    arctic_config = configparser.ConfigParser()
    arctic_config.read(os.path.abspath(arctic_config_path), encoding='utf-8')
    local_config = configparser.ConfigParser()
    local_config.read(os.path.abspath(local_config_path), encoding='utf-8')
    _sections_check(arctic_config, local_config, 'Market')
    _download_sections(arctic_config, local_config, 'Market', download_market_data_from_arctic)


def intraday_arctic_to_local(arctic_config_path: str, local_config_path: str, ):
    """

    :param arctic_config_path:
    :param local_config_path:
    :return:
    """
    arctic_config = configparser.ConfigParser()
    arctic_config.read(os.path.abspath(arctic_config_path), encoding='utf-8')
    local_config = configparser.ConfigParser()
    local_config.read(os.path.abspath(local_config_path), encoding='utf-8')

    # _download_sections(arctic_config, local_config, 'Market', download_market_data_from_arctic)
    _download_intraday(arctic_config, local_config, 'm5_data')



def trade_arctic_to_local(arctic_config_path: str, local_config_path: str):
    """

    :param arctic_config_path:
    :param local_config_path:
    :return:
    """
    arctic_config = configparser.ConfigParser()
    arctic_config.read(os.path.abspath(arctic_config_path), encoding='utf-8')
    local_config = configparser.ConfigParser()
    local_config.read(os.path.abspath(local_config_path), encoding='utf-8')
    _sections_check(arctic_config, local_config, 'Trade')
    _download_sections(arctic_config, local_config, 'Trade', download_trade_data_from_arctic)


def index_arctic_to_local(arctic_config_path: str, local_config_path: str):
    """

    :param arctic_config_path:
    :param local_config_path:
    :return:
    """
    arctic_config = configparser.ConfigParser()
    arctic_config.read(os.path.abspath(arctic_config_path), encoding='utf-8')
    local_config = configparser.ConfigParser()
    local_config.read(os.path.abspath(local_config_path), encoding='utf-8')
    _sections_check(arctic_config, local_config, 'Index')
    _download_index_section(arctic_config, local_config)


def component_arctic_to_local(arctic_config_path: str, local_config_path: str):
    """

    :param arctic_config_path:
    :param local_config_path:
    :return:
    """
    arctic_config = configparser.ConfigParser()
    arctic_config.read(os.path.abspath(arctic_config_path), encoding='utf-8')
    local_config = configparser.ConfigParser()
    local_config.read(os.path.abspath(local_config_path), encoding='utf-8')
    _sections_check(arctic_config, local_config, 'Component')
    _sections_check(arctic_config, local_config, 'Industry')
    _sections_check(arctic_config, local_config, 'Index')
    _download_component_section(arctic_config, local_config)


def industries_arctic_to_local(arctic_config_path: str, local_config_path: str):
    arctic_config = configparser.ConfigParser()
    arctic_config.read(os.path.abspath(arctic_config_path), encoding='utf-8')
    local_config = configparser.ConfigParser()
    local_config.read(os.path.abspath(local_config_path), encoding='utf-8')
    _sections_check(arctic_config, local_config, 'Industry')
    _download_industries_section(arctic_config, local_config)


def exotic_arctic_to_local(arctic_config_path: str, local_config_path: str):
    arctic_config = configparser.ConfigParser()
    arctic_config.read(os.path.abspath(arctic_config_path), encoding='utf-8')
    local_config = configparser.ConfigParser()
    local_config.read(os.path.abspath(local_config_path), encoding='utf-8')
    _sections_check(arctic_config, local_config, 'Exotic')
    _download_sections(arctic_config, local_config, 'Exotic', download_exotic_data_from_arctic)


def calendar_arctic_to_local(arctic_config_path: str, local_config_path: str):
    arctic_config = configparser.ConfigParser()
    arctic_config.read(os.path.abspath(arctic_config_path), encoding='utf-8')
    local_config = configparser.ConfigParser()
    local_config.read(os.path.abspath(local_config_path), encoding='utf-8')
    _sections_check(arctic_config, local_config, 'Calendar')
    _download_calendar_section(arctic_config, local_config)


def hk_makret_moneyflow_arctic_to_local(arctic_config_path: str, local_config_path: str):
    arctic_config = configparser.ConfigParser()
    arctic_config.read(os.path.abspath(arctic_config_path), encoding='utf-8')
    local_config = configparser.ConfigParser()
    local_config.read(os.path.abspath(local_config_path), encoding='utf-8')
    _sections_check(arctic_config, local_config, 'HK_market_moneyflow')
    store = get_arctic_store(arctic_config)
    data_folder = os.path.join('../', local_config['Local']['relative_path'])
    if not os.path.exists(data_folder):
        data_folder = local_config['Local']['absolute_path']
        if not os.path.exists(data_folder):
            raise ValueError('data folder path: {} not exists'.format(data_folder))
    data_folder = os.path.join(data_folder, 'HK_market_moneyflow')
    os.makedirs(data_folder, exist_ok=True)
    for key in local_config['HK_market_moneyflow']:
        data = store['hk_market_moneyflow'].read('hk_market_moneyflow').data  # type: pd.DataFrame
        local_save_path = os.path.join(data_folder, local_config['HK_market_moneyflow'][key])
        data.to_parquet(local_save_path)


if __name__ == '__main__':
    arctic_config_path = '../cfg/arctic_local.ini'
    local_config_path = '../cfg/local_data.ini'
    # fundamental_arctic_to_local(arctic_config_path, local_config_path)
    # market_arctic_to_local(arctic_config_path, local_config_path)
    # trade_arctic_to_local(arctic_config_path, local_config_path)
    # index_arctic_to_local(arctic_config_path, local_config_path)
    # component_arctic_to_local(arctic_config_path, local_config_path)
    # industries_arctic_to_local(arctic_config_path, local_config_path)
    # exotic_arctic_to_local(arctic_config_path, local_config_path)
    # calendar_arctic_to_local(arctic_config_path, local_config_path)
    # hk_makret_moneyflow_arctic_to_local(arctic_config_path, local_config_path)
    # _download_intraday(arctic_config_path, local_config_path, 'm5_data')
    intraday_arctic_to_local(arctic_config_path, local_config_path)