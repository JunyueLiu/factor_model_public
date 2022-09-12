import os
from typing import Callable

import pandas as pd
import tqdm
from arctic import Arctic

from data_management.mongo_builder.utils import market_batch_insert_df_to_arctic, fundamental_batch_insert_df_to_arctic


def _get_dtype(data_folder) -> dict:
    field_type_df = pd.read_excel(os.path.join(data_folder, '字段说明.xlsx'), engine='openpyxl')
    field_type = {}
    for i, row in field_type_df.iterrows():
        key = row['字段']
        type = row['数据类型']
        if type == 'Nvarchar':
            type = 'str'
        elif type == 'decimal':
            type = 'float'
        # elif type == ''
        field_type[key] = type
    return field_type


def _prepare_csmar_data(data_folder: str, func: Callable, *args, **kwargs):
    dfs = []
    folders = [os.path.join(data_folder, p) for p in os.listdir(data_folder) if
               os.path.isdir(os.path.join(data_folder, p))]
    for f in tqdm.tqdm(folders):
        files = [os.path.join(f, p) for p in os.listdir(f) if os.path.isfile(os.path.join(f, p)) and p.endswith('.csv')]
        for csv in files:
            df = func(csv, *args, **kwargs)
            dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    return data


def _prepare_analyst_forecast_data(data_folder: str) -> pd.DataFrame:
    field_type = _get_dtype(data_folder)

    def f(csv, ):
        df = pd.read_csv(csv, dtype={'Stkcd': str})
        df = df[df['Stkcd'].apply(lambda x: x[0] in ['0', '3', '6'])]  # only keep 主板、中小创、科创板
        df['Stkcd'] = df['Stkcd'].apply(lambda x: x + '.SZ' if x[0] in ['0', '3'] else x + '.SH')
        if 'Fenddt' in df:
            # This is for resolve a bug in data.
            # see the report, the '3022-12-31' should be '2023-12-31'
            # 方正
            # 20210330-方正证券-扬农化工（600486.SH）：优嘉四期稳步推进，成长确定性较强.pdf
            df['Fenddt'] = df['Fenddt'].replace('3022-12-31', '2023-12-31')
            #  研究报告：中银国际-冀东水泥-000401-华北龙头整合至尾声，区域需求提升受益明显-210927
            df['Fenddt'] = df['Fenddt'].replace('2421-12-31', '2021-12-31')
            df['Fenddt'] = df['Fenddt'].replace('2422-12-31', '2022-12-31')

        for field, data_type in field_type.items():
            if field in df:
                if data_type == 'Datetime':
                    df[field] = pd.to_datetime(df[field])
                else:
                    df[field] = df[field].astype(data_type)
            else:
                raise ValueError('please check field_type dict')
        return df

    data = _prepare_csmar_data(data_folder, f)
    data = data.sort_values(['Stkcd', 'Rptdt', 'Fenddt'])
    data = data.drop_duplicates()
    data = data.rename(columns={'Stkcd': 'code', 'Rptdt': 'pub_date', 'Fenddt': 'end_date'})
    data['start_date'] = data['end_date'].apply(lambda x: x.replace(month=1, day=1))
    data = data.reset_index(drop=True)
    data = data[['code', 'pub_date', 'start_date', 'end_date',
                 'AnanmID', 'Ananm', 'ReportID',
                 'InstitutionID', 'Brokern', 'Feps', 'Fpe', 'Fnetpro', 'Febit',
                 'Febitda', 'Fturnover', 'Fcfps', 'FBPS', 'FROA', 'FROE', 'FPB',
                 'FTotalAssetsTurnover']]
    return data


def _prepare_quick_fin(data_folder: str) -> pd.DataFrame:
    # field_type = _get_dtype(data_folder)

    def f(csv, ):
        df = pd.read_csv(csv, dtype={'StockCode': str}, error_bad_lines=False)
        df = df[df['StockCode'].apply(lambda x: x[0] in ['0', '3', '6'])]  # only keep 主板、中小创、科创板
        df['StockCode'] = df['StockCode'].apply(lambda x: x + '.SZ' if x[0] in ['0', '3'] else x + '.SH')
        for c in ['PubliDate', 'AccPeriod']:
            df[c] = pd.to_datetime(df[c])
        df['ReaModiFinResu'] = df['ReaModiFinResu'].astype(str)
        return df

    data = _prepare_csmar_data(data_folder, f)

    data = data.rename(columns={'StockCode': 'code', 'PubliDate': 'pub_date', 'AccPeriod': 'end_date'})
    data['start_date'] = data['end_date'].apply(lambda x: x.replace(month=1, day=1))
    data = data.sort_values(['code', 'pub_date', 'start_date', 'end_date'])
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    data = data[['code', 'start_date', 'end_date', 'pub_date',
                 'StockName', 'NumQuiTraFinReport',
                 'TotOpeReve', 'TotOpeReveLast', 'RatTotOpeReve', 'OpeProf',
                 'OpeProfLast', 'RatOpeProf', 'TotProf', 'TotProfLast', 'RatTotProf',
                 'NetProfPareComp', 'NetProfPareCompLast', 'RatNetProfPareComp',
                 'NetProf', 'NetProfLast', 'RatNetProf', 'EPS', 'EPSLast', 'RatEPS',
                 'RetuEqui', 'RetuEquiLast', 'RatRetuEqui', 'TotAsse', 'TotAsseBegin',
                 'RatTotAsse', 'EquiPareComp', 'EquiPareCompBegin', 'RatEquiPareComp',
                 'SharehEqui', 'SharehEquiBegin', 'RatSharehEqui', 'NetAsseShaPareComp',
                 'NetAsseShaPareCompBegin', 'RatNetAsseShaPareComp', 'NetAsseSha',
                 'NetAsseShaBegin', 'RatNetAsseSha', 'ReaModiFinResu']]
    return data


def build_analyst_forecast(arctic_store: Arctic, analyst_forecast_path):
    data = _prepare_analyst_forecast_data(analyst_forecast_path)
    fundamental_batch_insert_df_to_arctic(data, arctic_store, 'analyst_forecast', 'csmar')

def build_quick_fin(arctic_store: Arctic, quick_fin_path):
    data = _prepare_quick_fin(quick_fin_path)
    fundamental_batch_insert_df_to_arctic(data, arctic_store, 'quick_fin', 'csmar')


if __name__ == '__main__':
    store = Arctic('localhost')
    analyst_forecast_path = '../../data/csmar/raw/analyst_forecast'
    # analyst_forecast = _prepare_analyst_forecast_data(analyst_forecast_path)
    build_analyst_forecast(store, analyst_forecast_path)
    quick_fin_path = '../../data/csmar/raw/quick_fin'
    # quick_fin = _prepare_quick_fin(quick_fin_path)
    build_quick_fin(store, quick_fin_path)