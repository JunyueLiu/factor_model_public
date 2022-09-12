import configparser
import datetime
import os
from enum import Enum
from typing import Optional, List, Union

import pandas as pd

from data_management.dataIO.utils import get_arctic_store, read_arctic_version_store


class Fundamental(Enum):
    balance_sheet = 'balance_sheet'
    income_statement = 'income_statement'
    cashflow_statement = 'cashflow_statement'
    fin_forecast = 'fin_forecast'
    quick_fin = 'quick_fin'


class ReportPeriod(Enum):
    first_season = 'FS'
    half_year = 'HY'
    third_season = 'TS'
    whole_year = 'WY'


class ReportType(Enum):
    current = 0
    previous = 1


report_period_end_date_dict = {
    ReportPeriod.first_season: (3, 31),
    ReportPeriod.half_year: (6, 30),
    ReportPeriod.third_season: (9, 30),
    ReportPeriod.whole_year: (12, 31),

}


def get_stock_info(config_path: str = '../../cfg/data_input.ini'):
    config = configparser.ConfigParser()
    config.read(config_path)
    if 'Source' not in config.sections():
        raise ValueError('Source section not in config. PLease check config path and config file')

    fundamental_data_source = config['Source']['fundamental_data']
    project_deploy_path = config['Source']['project_deploy_path']
    config2 = configparser.ConfigParser()
    config2.read(os.path.join(project_deploy_path, 'cfg', fundamental_data_source), encoding='utf-8')

    if 'Arctic' in config2.sections():
        store = get_arctic_store(config2)
        MAX_WORKER = int(config2['Arctic'].get('MAX_WORKER', 10))
        library = 'securities_info'
        data = store[library].read('stock').data



    elif 'Local' in config2.sections():
        data_folder = os.path.join(project_deploy_path, config2['Local']['relative_path'],
                                   config2['Local']['Fundamental']
                                   )
        path = os.path.join(data_folder, 'securities_info.csv')
        data = pd.read_csv(path, parse_dates=['start_date', 'end_date'], index_col=0)
        data = data.convert_dtypes()
    else:
        raise NotImplementedError

    return data


def _correct_fin_forcast_data(fin_forecast: pd.DataFrame):
    # id 642205
    # https://pdf.dfcfw.com/pdf/H2_AN201308280004279263_1.pdf?1377710470000.pdf page 36
    # Out[25]:
    #                                                               27411
    # id                                                           642205
    # company_id                                                300003281
    # code                                                      002142.SZ
    # name                                                     宁波银行股份有限公司
    # end_date                                        2013-09-30 00:00:00
    # report_type_id                                               304003
    # report_type                                                   三季度预告
    # pub_date                                        2013-08-28 00:00:00 <- Wrong should be 2013-08-29 00:00:00
    # type_id                                                      305002
    # type                                                           业绩预增
    # profit_min                                              3.66808e+09
    # profit_max                                              4.00154e+09
    # profit_last                                             3.33462e+09
    # profit_ratio_min                                                 10
    # profit_ratio_max                                                 20
    # content           预计公司2013年01-09月归属于上市公司股东的净利润为3,668,079千元-4,001...
    # net_profit                                              3.83481e+09
    fin_forecast.loc[fin_forecast[(fin_forecast.id == 642205)].index, 'pub_date'] += pd.Timedelta(days=1)
    return fin_forecast


def get_fundamental(table: Fundamental,
                    code: Optional[Union[str or List[str]]] = None,
                    pub_start_date: Optional[Union[str, datetime.datetime]] = None,
                    pub_end_date: Optional[Union[str, datetime.datetime]] = None,
                    fiscal_year: Optional[Union[int or List[int]]] = None,
                    report_period: Optional[ReportPeriod] = None,
                    config_path: str = '../../cfg/data_input.ini',
                    cols: Optional[List[str]] = None,
                    report_type: Optional[ReportType] = None,
                    verbose=1):
    config = configparser.ConfigParser()
    config.read(config_path)
    if 'Source' not in config.sections():
        raise ValueError('Source section not in config. PLease check config path and config file')

    fundamental_data_source = config['Source']['fundamental_data']
    project_deploy_path = config['Source']['project_deploy_path']
    config2 = configparser.ConfigParser()
    config2.read(os.path.join(project_deploy_path, 'cfg', fundamental_data_source), encoding='utf-8')
    if len(config2.sections()) == 0:
        raise ValueError('Empty config. PLease check config path and config file for {}'
                         .format(os.path.join(project_deploy_path, 'cfg', fundamental_data_source)))

    start_time = datetime.datetime.now()
    balance_sheet_path = config2['Fundamental']['balance_sheet']
    income_statement_path = config2['Fundamental']['income_statement']
    cashflow_statement_path = config2['Fundamental']['cashflow_statement']
    fin_forecast_path = config2['Fundamental']['fin_forecast']
    quick_fin_path = config2['Fundamental']['quick_fin']
    # ============== add more table here ============

    if 'Arctic' in config2.sections():
        store = get_arctic_store(config2)
        MAX_WORKER = int(config2['Arctic'].get('MAX_WORKER', 10))
        if table == Fundamental.balance_sheet:
            library = balance_sheet_path
        elif table == Fundamental.income_statement:
            library = income_statement_path
        elif table == Fundamental.cashflow_statement:
            library = cashflow_statement_path
        elif table == Fundamental.fin_forecast:
            library = fin_forecast_path
        elif table == Fundamental.quick_fin:
            library = quick_fin_path
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
        data = read_arctic_version_store(store, library, instruments, MAX_WORKER, False)


    elif 'Local' in config2.sections():
        data_folder = os.path.join(project_deploy_path, config2['Local']['relative_path'],
                                   config2['Local']['Fundamental']
                                   )
        if not os.path.exists(data_folder):
            print('relative path not found. Try to find the absolute_path.')
            data_folder = os.path.join(config2['Local']['absolute_path'], config2['Local']['Fundamental'])
            if not os.path.exists(data_folder):
                raise ValueError('data folder path: {} not exists'.format(data_folder))

        if Fundamental.balance_sheet == table:
            path = os.path.join(data_folder, balance_sheet_path)
        elif Fundamental.income_statement == table:
            path = os.path.join(data_folder, income_statement_path)
        elif Fundamental.cashflow_statement == table:
            path = os.path.join(data_folder, cashflow_statement_path)
        elif Fundamental.fin_forecast == table:
            path = os.path.join(data_folder, fin_forecast_path)
        elif Fundamental.quick_fin == table:
            path = os.path.join(data_folder, quick_fin_path)
        # ============== add more table here ============
        else:
            raise NotImplementedError

        if path.endswith('.parquet'):
            data = pd.read_parquet(path)
        else:
            raise NotImplementedError('Currently only support parquet')

    else:
        raise ValueError()

    if table == Fundamental.fin_forecast:
        data = _correct_fin_forcast_data(data)

    if code is not None:
        if isinstance(code, list):
            data = data[data['code'].isin(code)]
        elif isinstance(code, str):
            data = data[data['code'] == code]
        else:
            raise ValueError('code must be type of str or list, but {} is given.'.format(type(code)))

    if pub_start_date is not None:
        start_date = pd.to_datetime(pub_start_date)
        data = data[data['pub_date'] >= start_date]

    if pub_end_date is not None:
        end_date = pd.to_datetime(pub_end_date)
        data = data[data['pub_date'] < end_date]

    if cols:
        c = []
        for d in ['start_date', 'end_date', 'pub_date', 'report_date', 'code', 'report_type', 'source']:
            if d in data:
                c.append(d)
        cols = c + cols
        data = data[cols]

    if fiscal_year:
        if isinstance(fiscal_year, int):
            data = data[(data['end_date'].dt.year == fiscal_year)]
        elif isinstance(fiscal_year, list):
            data = data[data['end_date'].dt.year.isin(fiscal_year)]
        else:
            raise NotImplementedError

    if report_period:
        if report_period == ReportPeriod.first_season:
            data = data[data['end_date'].dt.month == report_period_end_date_dict[ReportPeriod.first_season][0]]
        elif report_period == ReportPeriod.half_year:
            data = data[data['end_date'].dt.month == report_period_end_date_dict[ReportPeriod.half_year][0]]
        elif report_period == ReportPeriod.third_season:
            data = data[data['end_date'].dt.month == report_period_end_date_dict[ReportPeriod.third_season][0]]
        elif report_period == ReportPeriod.whole_year:
            data = data[data['end_date'].dt.month == report_period_end_date_dict[ReportPeriod.whole_year][0]]
        else:
            raise ValueError('report_period must be one of {}'.format(ReportPeriod.__dict__.get('_member_names_')))

    if table in [Fundamental.quick_fin, Fundamental.fin_forecast]:
        print('quick_fin and fin_forecast not support report_type filter')
        report_type = None

    if report_type:
        if report_type == ReportType.current:
            data = data[data['report_type'] == ReportType.current.value]
        elif report_type == ReportType.previous:
            data = data[data['report_type'] == ReportType.previous.value]
        else:
            raise ValueError('report_type must be one of {}'.format(ReportType.__dict__.get('_member_names_')))

    data = data.sort_values(['code', 'pub_date'])
    data = data.reset_index(drop=True)
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
    # data = get_fundamental(Fundamental.fin_forecast, ['000001.SZ', '600519.SH'],
    #                        report_period=ReportPeriod.whole_year, fiscal_year=[2020, 2019],
    #                        report_type=ReportType.current)
    # info = get_stock_info()

    data = get_fundamental(Fundamental.quick_fin)
