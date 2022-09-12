import os
from typing import Callable

import pandas as pd
import tqdm
from data_management.keeper.ZooKeeper import ZooKeeper


def _prepare_csmar_data(data_folder: str, func: Callable, *args, **kwargs):
    dfs = []
    folders = [os.path.join(data_folder, p) for p in os.listdir(data_folder) if
               os.path.isdir(os.path.join(data_folder, p))]
    for f in tqdm.tqdm(folders):
        files = [os.path.join(f, p) for p in os.listdir(f) if os.path.isfile(os.path.join(f, p)) and p.endswith('.csv')]
        for csv in files:
            df = func(csv, *args, **kwargs)
            dfs.append(df)
    data = pd.concat(dfs)
    return data

def read_fama_french_csv(csv_path):
    """
     # MarkettypeID [股票市场类型编码] - P9705：创业板;P9706：综合A股市场（不包含科创板、创业板）;P9707：综合B股市场;P9709：综合A股和创业板; P9710：综合AB股和创业板；P9711：科创板；P9712：综合A股和科创板；P9713：综合AB股和科创板；P9714：综合A股和创业板和科创板；P9715：综合AB股和创业板和科创板。
    # TradingDate [交易日期] - 以YYYY-MM-DD表示
    # RiskPremium1 [市场风险溢价因子(流通市值加权)] - 考虑现金红利再投资的日市场回报率(流通市值加权平均法)与日度化无风险利率之差（央行公布三月定存基准利率）。
    # RiskPremium2 [市场风险溢价因子(总市值加权)] - 考虑现金红利再投资的日市场回报率(总市值加权平均法)与日度化无风险利率之差（央行公布三月定存基准利率）。
    # SMB1 [市值因子(流通市值加权)] - 小盘股组合和大盘股组合的收益率之差，组合划分基于FAMA 2*3组合划分方法。组合日收益率的计算采用流通市值加权计算。
    # SMB2 [市值因子(总市值加权)] - 小盘股组合和大盘股组合的收益率之差，组合划分基于FAMA 2*3组合划分方法。组合日收益率的计算采用总市值加权计算。
    # HML1 [账面市值比因子(流通市值加权)] - 高账面市值比组合和低账面市值比组合的收益率之差，组合划分基于FAMA 2*3组合划分方法。组合投资收益率的计算采用流通市值加权。
    # HML2 [账面市值比因子(总市值加权)] - 高账面市值比组合和低账面市值比组合的收益率之差，组合划分基于FAMA 2*3组合划分方法。组合投资收益率的计算采用总市值加权。
    :param csv_path:
    :return:
    """
    df = pd.read_csv(csv_path)
    df = df[df['MarkettypeID'] == 'P9714']
    df['TradingDate'] = pd.to_datetime(df['TradingDate'])
    df = df.rename(columns={'TradingDate': 'date'})
    df = df.set_index('date',)
    return df

def fama_french_risk_premium(data_folder):
    df = _prepare_csmar_data(data_folder, read_fama_french_csv)
    df = df.loc['2005-01-01':]
    series = df['RiskPremium1']
    series.name = 'risk_premium'
    return series

def fama_french_SMB(data_folder):
    df = _prepare_csmar_data(data_folder, read_fama_french_csv)
    df = df.loc['2005-01-01':]
    series = df['SMB1']
    series.name = 'SMB'
    return series

def fama_french_HML(data_folder):
    df = _prepare_csmar_data(data_folder, read_fama_french_csv)
    df = df.loc['2005-01-01':]
    series = df['HML1']
    series.name = 'HML'
    return series



if __name__ == '__main__':
    data_path = '../data/csmar/raw/fama_french_3'
    aliyun_cfg_path = '../cfg/factor_keeper_setting_hk.ini'
    local_cfg_path = '../cfg/factor_keeper_setting.ini'

    keeper = ZooKeeper(local_cfg_path)
    smb = fama_french_SMB(data_path)
    keeper.save_factor_value('FAMA-FRENCH', smb, comment='Fama-French SMB factor', data_source='csmar')
    hml = fama_french_HML(data_path)
    keeper.save_factor_value('FAMA-FRENCH', hml, comment='Fama-French HML factor', data_source='csmar')
    rp = fama_french_risk_premium(data_path)
    keeper.save_factor_value('FAMA-FRENCH', rp, comment='Fama-French Risk premium factor', data_source='csmar')