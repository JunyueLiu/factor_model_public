import configparser
import datetime
import os
from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd
from pandas.tseries.offsets import CBMonthEnd, CustomBusinessDay

from data_management.dataIO.utils import get_arctic_store, read_pickle


class Market(Enum):
    AShares = 'Ashares'


def get_trading_date(market: Market,
                     start_date: Optional[Union[str, datetime.datetime]] = None,
                     end_date: Optional[Union[str, datetime.datetime]] = None,
                     config_path: str = '../../cfg/data_input.ini', ) -> np.ndarray:
    config = configparser.ConfigParser()
    config.read(config_path)
    if 'Source' not in config.sections():
        raise ValueError('Source section not in config. PLease check config path and config file')

    data_source = config['Source']['calendar_data']
    project_deploy_path = config['Source']['project_deploy_path']
    config2 = configparser.ConfigParser()
    config2.read(os.path.join(project_deploy_path, 'cfg', data_source), encoding='utf-8')
    if len(config2.sections()) == 0:
        raise ValueError('Empty config. PLease check config path and config file for {}'
                         .format(os.path.join(project_deploy_path, 'cfg', data_source)))

    ashares_path = config2['Calendar']['AShares']
    # ============== add more table here ============
    if 'Arctic' in config2.sections():
        store = get_arctic_store(config2)
        data = store['trading_calendar'].read(ashares_path).data

    elif 'Local' in config2.sections():
        data_folder = os.path.join(project_deploy_path, config2['Local']['relative_path'],
                                   config2['Local']['Calendar']
                                   )
        if not os.path.exists(data_folder):
            print('relative path not found. Try to find the absolute_path.')
            data_folder = os.path.join(config2['Local']['absolute_path'], config2['Local']['Calendar'])
            if not os.path.exists(data_folder):
                raise ValueError('data folder path: {} not exists'.format(data_folder))

        if Market.AShares == market:
            path = os.path.join(data_folder, ashares_path)
        # ============== add more table here ============
        else:
            raise NotImplementedError

        data = read_pickle(path)
    else:
        raise NotImplementedError

    if start_date is not None:
        start_date = pd.to_datetime(start_date).to_pydatetime().date()
        data = data[data >= start_date]

    if end_date is not None:
        end_date = pd.to_datetime(end_date).to_pydatetime().date()
        data = data[data < end_date]

    return data


def trading_dates_offsets(trading_date: np.ndarray, resample: str) -> Union[pd.tseries.offsets.CustomBusinessDay,
                                                                            pd.tseries.offsets.CustomBusinessMonthEnd]:
    if resample not in ['D', 'M', 'M-Mid', 'W-1', 'W-2', 'W-3', 'W-4', 'W-5', 'BW']:
        raise ValueError('resample must in {}'.format(['D', 'M', 'M-Mid', 'W-1', 'W-2', 'W-3', 'W-4', 'W-5', 'BW']))

    def _num_week_trading_date_func(td, trading_weekday, non_td):
        num_week = [w.isocalendar()[:2] for w in td]
        df = pd.DataFrame(num_week, index=pd.to_datetime(td), columns=['year', 'num_week'])
        df['num_week_trading_dates'] = 1
        df['num_week_trading_dates'] = df.groupby(['year', 'num_week'])['num_week_trading_dates'].cumsum()
        hd = df[df['num_week_trading_dates'] != trading_weekday].index
        non_trading = non_td.append(hd).sort_values()
        return non_trading

    def _not_mid_trading_date_func(td, non_td):
        df = pd.DataFrame(pd.to_datetime(td), columns=['td'])
        df['month'] = df['td'].dt.strftime('%Y-%m')
        mid_day = df.groupby('month')['td'].apply(lambda x: x.iloc[len(x) // 2])
        non_trading = non_td.append(pd.Index(df[~df['td'].isin(mid_day)]['td'])).sort_values()
        return non_trading

    def _not_mid_month_end_trading_date_func(td, non_td):
        df = pd.DataFrame(pd.to_datetime(td), columns=['td'])
        df['month'] = df['td'].dt.strftime('%Y-%m')
        mid_day = df.groupby('month')['td'].apply(lambda x: x.iloc[len(x) // 2])
        last_day = df.groupby('month')['td'].last()
        non_trading = non_td.append(pd.Index(df[~df['td'].isin(mid_day)]['td']).difference(pd.Index(last_day.values)))\
            .sort_values().drop_duplicates()
        return non_trading

    days = pd.date_range(start=trading_date[0], end=trading_date[-1], freq='D')
    non_trading = days.difference(trading_date)
    if resample == 'M':
        non_trading = days.difference(trading_date)
        offsets = CBMonthEnd(holidays=non_trading)
    elif resample == 'M-Mid':
        non_trading = _not_mid_trading_date_func(trading_date, non_trading)
        offsets = CustomBusinessDay(holidays=non_trading)
    elif resample == 'D':
        non_trading = days.difference(trading_date)
        offsets = CustomBusinessDay(holidays=non_trading)
    elif resample == 'W-1':
        non_trading = _num_week_trading_date_func(trading_date, 1, non_trading)
        offsets = CustomBusinessDay(holidays=non_trading)
    elif resample == 'W-2':
        non_trading = _num_week_trading_date_func(trading_date, 2, non_trading)
        offsets = CustomBusinessDay(holidays=non_trading)
    elif resample == 'W-3':
        non_trading = _num_week_trading_date_func(trading_date, 3, non_trading)
        offsets = CustomBusinessDay(holidays=non_trading)
    elif resample == 'W-4':
        non_trading = _num_week_trading_date_func(trading_date, 4, non_trading)
        offsets = CustomBusinessDay(holidays=non_trading)
    elif resample == 'W-5':
        non_trading = _num_week_trading_date_func(trading_date, 5, non_trading)
        offsets = CustomBusinessDay(holidays=non_trading)
    elif resample == 'BW':
        non_trading = _not_mid_month_end_trading_date_func(trading_date, non_trading)
        offsets = CustomBusinessDay(holidays=non_trading)
    else:
        raise NotImplementedError
    return offsets


if __name__ == '__main__':
    td = get_trading_date(Market.AShares, '2015-01-01', config_path='../../cfg/data_input_arctic.ini')
    offset = trading_dates_offsets(td, 'BW')
