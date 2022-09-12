from collections import defaultdict

import pandas as pd
import tushare as ts
import yfinance as yf
import os
import tqdm
import time
import datetime
import jqdatasdk
import pickle
from jqdatasdk import finance, sup
from jqdatasdk import query

from data_management.tokens import *
from factor_zoo.utils import get_last_trading_of_month_dict, transform_wind_code, save_pickle, load_pickle
from pandas.tseries.offsets import MonthEnd


ts.set_token(token)
pro = ts.pro_api()

jqdatasdk.auth(jq_user, jq_password)


def download_trade_days(path='../data/trading_days.pickle'):
    trading_date = jqdatasdk.get_all_trade_days()
    save_pickle(trading_date, path)

def get_trade_days():
    trading_date = jqdatasdk.get_all_trade_days()
    return trading_date

def get_index_component(index_ticker):
    """
    CSI300 : 399300

    :param index_ticker:
    :return:
    """
    # 最新成分
    # next page function
    # todo
    raise NotImplementedError
    df = pd.read_html('http://stock.jrj.com.cn/share,sz399300,nzxcf.shtml')[4]

    df = df[['股票代码', '纳入时间']]
    df['股票代码'] = df['股票代码'].apply(lambda x: f"{x:0>6}")
    pass
    # http: // stock.jrj.com.cn / share, sz399300, nzxcf.shtml
    # 'http://stock.jrj.com.cn/share,sz399300,nlscf.shtml'


def get_market_data(ticker, freq='M', code_rename='instCode', trade_date_rename='date', start_date=None, end_date=None):
    df = ts.pro_bar(ts_code=ticker, asset='E', freq=freq, adj='qfq', start_date=start_date, end_date=end_date)  # type: pd.DataFrame
    if df is None:
        print(ticker, 'return None')
        return pd.DataFrame(columns=['close', 'open', 'high', 'low', 'pre_close', 'change', 'pct_chg', 'vol', 'amount'])
    if 'trade_time' in df:
        df.rename(columns={'ts_code': code_rename, 'trade_time': trade_date_rename}, inplace=True)
    else:
        df.rename(columns={'ts_code': code_rename, 'trade_date': trade_date_rename}, inplace=True)
    df[trade_date_rename] = pd.to_datetime(df[trade_date_rename])
    df = df.set_index([trade_date_rename, code_rename]).sort_index()
    return df


def get_limit(ticker, code_rename='instCode', start_year='20050101', end_year=None):
    # if end_year is None:
    #     end_year = datetime.datetime.now().strftime('%Y%m%d')
    # year = pd.date_range(start_year, end_year, freq='YS').strftime('%Y%m%d').to_list()
    # year.append(pd.to_datetime(end_year))
    #
    df = pro.stk_limit(ts_code=ticker, start_date=start_year)
    df.rename(columns={'ts_code': code_rename}, inplace=True)
    data = df.set_index(['trade_date', code_rename]).sort_index()
    return data


# def download_balance(ticker,  code_rename='instCode', trade_date_rename='date'):
#
#     df = pro.balancesheet(ts_code=ticker) # type: pd.DataFrame
#     if df is None:
#         print(ticker, 'return None')
#         return pd.DataFrame()
#     # df.rename(columns={'ts_code': code_rename, 'trade_date': trade_date_rename}, inplace=True)
#     # df[trade_date_rename] = pd.to_datetime(df[trade_date_rename])
#     # df = df.set_index([trade_date_rename, code_rename]).sort_index()
#     return df

def get_suspend_data(ticker, code_rename='instCode'):
    df = pro.suspend(ts_code=ticker)  # type:pd.DataFrame
    df['suspend_date'] = pd.to_datetime(df['suspend_date'])
    df['resume_date'] = pd.to_datetime(df['resume_date'])
    df.rename(columns={'ts_code': code_rename}, inplace=True)
    df.set_index(['suspend_date', code_rename], inplace=True)
    return df

def get_all_ticker(code_rename='instCode', local=None):
    """

    list_status	str	上市状态： L上市 D退市 P暂停上市
    :return:
    """
    ticker = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,list_date,delist_date')
    ticker1 = pro.stock_basic(exchange='', list_status='D', fields='ts_code,symbol,name,list_date,delist_date')
    ticker2 = pro.stock_basic(exchange='', list_status='P', fields='ts_code,symbol,name,list_date,delist_date')
    data = pd.concat([ticker, ticker1, ticker2])
    data.rename(columns={'ts_code': code_rename}, inplace=True)
    data['list_date'] = pd.to_datetime(data['list_date'])
    data['delist_date'] = pd.to_datetime(data['delist_date'])

    data = data.set_index(code_rename).sort_index()
    return data


def download_index_market_data(index_code, data_folder: str = r'../data/market', freq='M', code_rename='instCode',
                               trade_date_rename='date'):
    df = ts.pro_bar(ts_code=index_code, asset='I', freq=freq, adj='qfq')  # type: pd.DataFrame
    df.rename(columns={'ts_code': code_rename, 'trade_date': trade_date_rename}, inplace=True)
    df[trade_date_rename] = pd.to_datetime(df[trade_date_rename])
    data = df.set_index([trade_date_rename, code_rename]).sort_index()
    data.to_parquet(os.path.join(data_folder, '{}_{}_data.parquet'.format(index_code, freq)))


def download_all_market(data_folder: str = r'../data', freq='M', code_rename='instCode', trade_date_rename='date'):
    """

    :param data_folder:
    :return:
    """
    tickers = get_all_ticker(code_rename)
    tickers.to_csv(os.path.join(data_folder, 'all_tickers.csv'))
    dfs = []
    for ticker in tqdm.tqdm(tickers.index):
        try:
            df = get_market_data(ticker, freq=freq, code_rename=code_rename, trade_date_rename=trade_date_rename)
            dfs.append(df)
            # time.sleep(0.5)  # 每分钟只能调取120次
        except:
            print('download {} fails'.format(ticker))
    data = pd.concat(dfs)
    data.sort_index(inplace=True)
    data.to_parquet(os.path.join(data_folder, 'all_{}_data.parquet'.format(freq)))
    # data.to_csv(os.path.join(data_folder, 'all_{}_data.csv'.format(freq)))

def download_component_min_data(components:dict,
                                component_name:str,
                                data_folder: str = r'../data',
                                freq='15min', code_rename='instCode',
                                trade_date_rename='date',
                                start_date='2017-01-01'):
    keys = sorted(components)
    pd_start = pd.to_datetime(start_date)
    dfs = []
    for key in tqdm.tqdm(keys):
        pd_key_date = pd.to_datetime(key)
        if pd_key_date < pd_start:
            pass

        data_start = (pd.to_datetime(key) + pd.Timedelta(days=1, hours=9)).strftime('%Y-%m-%d %H:%M:%S')
        data_end = (pd.to_datetime(key) + pd.Timedelta(hours=15) + MonthEnd(1)).strftime('%Y-%m-%d %H:%M:%S')
        for ticker in tqdm.tqdm(components[key]):
            t = transform_wind_code(ticker)
            df = get_market_data(t, freq, code_rename=code_rename,
                                trade_date_rename=trade_date_rename,
                                 start_date=data_start, end_date=data_end)
            dfs.append(df)
    data = pd.concat(dfs)
    data.to_parquet(os.path.join(data_folder, '{}_{}_data.parquet'.format(component_name,freq)))



    pass


def download_all_market_bht_schema(data_folder: str = r'../data', freq='M', code_rename='instCode',
                                   trade_date_rename='date', test=False, start_date=None):
    """

    :param data_folder:
    :return:
    """
    tickers = get_all_ticker(code_rename)
    # tickers.to_csv(os.path.join(data_folder, 'all_tickers.csv'))
    dfs = []
    # 12 dataSource
    # CREATE TABLE `dailydata` (
    #   `instCode` varchar(50) NOT NULL,
    #   `date` datetime NOT NULL,
    #   `frame` int NOT NULL,
    #   `open` decimal(20,6) DEFAULT '0.000000',
    #   `high` decimal(20,6) DEFAULT '0.000000',
    #   `low` decimal(20,6) DEFAULT '0.000000',
    #   `close` decimal(20,6) DEFAULT '0.000000',
    #   `volume` decimal(20,6) DEFAULT '0.000000',
    #   `amount` decimal(20,6) DEFAULT '0.000000',
    #   `chg` decimal(20,6) DEFAULT NULL,
    #   `pctChg` decimal(20,6) DEFAULT NULL,
    #   `vwap` decimal(20,6) DEFAULT NULL,
    #   `dataSrcId` varchar(50) NOT NULL,
    #   `timestamp` datetime DEFAULT NULL,
    #   `ccyId` varchar(10) DEFAULT 'CNY',
    #   PRIMARY KEY (`instCode`,`date`,`frame`,`dataSrcId`),
    #   KEY `index_dailydata` (`frame`,`date`,`dataSrcId`)
    # ) ENGINE=InnoDB DEFAULT CHARSET=utf8
    if freq == 'D':
        frame = 1440
    elif freq == '1min':
        frame = 1
    elif freq == '15min':
        frame = 15
    elif freq == 'M':
        frame = 43200
    if test:
        tickers = tickers.iloc[:10]

    for ticker in tqdm.tqdm(tickers.index):
        try:
            df = get_market_data(ticker, freq=freq, code_rename=code_rename, trade_date_rename=trade_date_rename, start_date=start_date)
            dfs.append(df)
            # time.sleep(0.5)  # 每分钟只能调取120次
        except:
            print('download {} fails'.format(ticker))
    data = pd.concat(dfs)
    data = data.sort_index().reset_index()
    data = data.rename(columns={'change': 'chg', 'pct_chg': 'pctChg', 'vol': 'volume'})
    if frame >= 1440:
        data['date'] = data['date'].apply(lambda x: x.replace(hour=15))


    data['frame'] = frame
    data['vwap'] = 0.0
    data['dataSrcId'] = 'DS00012'
    data['timestamp'] = datetime.datetime.now()
    data['ccyId'] = 'CNY'
    data = data[['instCode', 'date', 'frame',
                 'open', 'high', 'low', 'close', 'volume', 'amount', 'chg', 'pctChg',
                 'vwap', 'dataSrcId', 'timestamp', 'ccyId']]
    # print(data)
    path = os.path.join(data_folder, 'all_{}_data_bht_schema_{}.parquet'.format(freq, datetime.datetime.now().strftime('%Y-%m-%d')))
    data.to_parquet(path)
    return path


def download_all_suspend(data_folder: str = r'data', code_rename='instCode'):
    """

    :param data_folder:
    :return:
    """
    tickers = get_all_ticker(code_rename)
    # tickers.to_csv(os.path.join(data_folder, 'all_tickers.csv'))
    dfs = []
    for ticker in tqdm.tqdm(tickers.index):
        try:
            df = get_suspend_data(ticker, code_rename=code_rename)
            dfs.append(df)
            time.sleep(0.5)  # 每分钟只能调取120次
        except:
            print('download {} fails'.format(ticker))
    data = pd.concat(dfs)
    data.sort_index(inplace=True)
    data.to_pickle(os.path.join(data_folder, 'all_suspend_data.pickle'))
    data.to_csv(os.path.join(data_folder, 'all_suspend_data.csv'))


def download_all_limit(data_folder: str = r'../data', code_rename='instCode'):
    """

    :param data_folder:
    :return:
    """
    tickers = get_all_ticker(code_rename)
    # tickers.to_csv(os.path.join(data_folder, 'all_tickers.csv'))
    dfs = []
    for ticker in tqdm.tqdm(tickers.index):
        try:
            df = get_limit(ticker, code_rename=code_rename)
            dfs.append(df)
            time.sleep(0.5)  # 每分钟只能调取120次
        except:
            print('download {} fails'.format(ticker))
    data = pd.concat(dfs)
    data.sort_index(inplace=True)
    data.to_parquet(os.path.join(data_folder, 'all_limit_data.parquet'))
    data.to_csv(os.path.join(data_folder, 'all_limit_data.csv'))


def download_balance_sheet(code, data_folder=r'../data/joinquant/balance_sheet'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    q = query(finance.STK_BALANCE_SHEET).filter(finance.STK_BALANCE_SHEET.code == code)
    df = finance.run_query(q)
    df.to_csv(os.path.join(data_folder, code + '.csv'))


def download_all_balance_sheet(code_list: list, data_folder=r'../data/joinquant/balance_sheet'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    for code in tqdm.tqdm(code_list):
        download_balance_sheet(code, data_folder=data_folder)


def download_income_statement(code, data_folder=r'../data/joinquant/income_statement'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    q = query(finance.STK_INCOME_STATEMENT).filter(finance.STK_INCOME_STATEMENT.code == code)
    df = finance.run_query(q)
    df.to_csv(os.path.join(data_folder, code + '.csv'))


def download_all_income_statement(code_list: list, data_folder=r'../data/joinquant/income_statement'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    for code in tqdm.tqdm(code_list):
        download_income_statement(code, data_folder=data_folder)


def download_cashflow_statement(code, data_folder=r'../data/joinquant/cashflow_statement'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    q = query(finance.STK_CASHFLOW_STATEMENT).filter(finance.STK_CASHFLOW_STATEMENT.code == code)
    df = finance.run_query(q)
    df.to_csv(os.path.join(data_folder, code + '.csv'))


def download_all_cashflow_statement(code_list: list, data_folder=r'../data/joinquant/cashflow_statement'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    for code in tqdm.tqdm(code_list):
        download_cashflow_statement(code, data_folder=data_folder)


def download_capital_change(code, data_folder=r'../data/joinquant/capital_change'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    q = query(finance.STK_CAPITAL_CHANGE).filter(finance.STK_CAPITAL_CHANGE.code == code)
    df = finance.run_query(q)
    df.to_csv(os.path.join(data_folder, code + '.csv'))


def download_all_capital_change(code_list: list, data_folder='../data/joinquant/capital_change'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    for code in tqdm.tqdm(code_list):
        download_capital_change(code, data_folder=data_folder)


def download_supplement(code, data_folder=r'../data/joinquant/supplement'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    q = query(sup.STK_FINANCE_SUPPLEMENT).filter(sup.STK_FINANCE_SUPPLEMENT.code == code)
    df = sup.run_query(q)
    df.to_csv(os.path.join(data_folder, code + '.csv'))


def download_all_supplement(code_list: list, data_folder='../data/joinquant/supplement'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    for code in tqdm.tqdm(code_list):
        download_supplement(code, data_folder=data_folder)


def download_shareholder_floating_top10(code, data_folder=r'../data/joinquant/shareholder_floating_top10'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    q = query(finance.STK_SHAREHOLDER_FLOATING_TOP10).filter(finance.STK_SHAREHOLDER_FLOATING_TOP10.code == code)
    df = finance.run_query(q)
    df.to_csv(os.path.join(data_folder, code + '.csv'))


def download_all_shareholder_floating_top10(code_list: list,
                                            data_folder='../data/joinquant/shareholder_floating_top10'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    for code in tqdm.tqdm(code_list):
        download_shareholder_floating_top10(code, data_folder=data_folder)


def download_shares_pledge(code, data_folder=r'../data/joinquant/shares_pledge'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    q = query(finance.STK_SHARES_PLEDGE).filter(finance.STK_SHARES_PLEDGE.code == code)
    df = finance.run_query(q)
    df.to_csv(os.path.join(data_folder, code + '.csv'))


def download_all_shares_pledge(code_list: list, data_folder='../data/joinquant/shares_pledge'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    for code in tqdm.tqdm(code_list):
        download_shares_pledge(code, data_folder=data_folder)


def download_holder_num(code, data_folder=r'../data/joinquant/holder_num'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    q = query(finance.STK_HOLDER_NUM).filter(finance.STK_HOLDER_NUM.code == code)
    df = finance.run_query(q)
    df.to_csv(os.path.join(data_folder, code + '.csv'))


def download_all_shareholders_share_change(code_list: list, data_folder='../datajoinquant/shareholders_share_change'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    for code in tqdm.tqdm(code_list):
        download_shareholders_share_change(code, data_folder=data_folder)


def download_shareholders_share_change(code, data_folder=r'../data/joinquant/shareholders_share_change'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    q = query(finance.STK_SHAREHOLDERS_SHARE_CHANGE).filter(finance.STK_SHAREHOLDERS_SHARE_CHANGE.code == code)
    df = finance.run_query(q)
    df.to_csv(os.path.join(data_folder, code + '.csv'))


def download_all_holder_num(code_list: list, data_folder='../data/joinquant/holder_num'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    for code in tqdm.tqdm(code_list):
        download_holder_num(code, data_folder=data_folder)


def download_all_money_flow(code_list, data_folder='../data/joinquant/money_flow'):
    if os.path.exists(data_folder) is False:
        os.makedirs(data_folder)
    end = datetime.datetime.now()
    for code in tqdm.tqdm(code_list):
        df = jqdatasdk.get_money_flow(code, '2010-01-01', end)
        df.to_csv(os.path.join(data_folder, code + '.csv'))

def get_component(index_symbol, date):
    return jqdatasdk.get_index_stocks(index_symbol, date=date)


def download_component(index_symbol, trading_date: list, path='../data/universe'):
    component = {}
    if not os.path.exists(path):
        os.makedirs(path)
    for date in tqdm.tqdm(trading_date):
        component[date] = get_component(index_symbol, date)
    save_pickle(component, os.path.join(path, index_symbol + '.pickle'))
    # return component


def download_industries_component(trading_date: list, path='../data', issuer='zjh'):
    if issuer == 'zjh':
        ll = [{'行业代码': 'A01', '行业名称': '农业'},
              {'行业代码': 'A02', '行业名称': '林业'},
              {'行业代码': 'A03', '行业名称': '畜牧业'},
              {'行业代码': 'A04', '行业名称': '渔业'},
              {'行业代码': 'A05', '行业名称': '农、林、牧、渔服务业'},
              {'行业代码': 'B06', '行业名称': '煤炭开采和洗选业'},
              {'行业代码': 'B07', '行业名称': '石油和天然气开采业'},
              {'行业代码': 'B08', '行业名称': '黑色金属矿采选业'},
              {'行业代码': 'B09', '行业名称': '有色金属矿采选业'},
              {'行业代码': 'B10', '行业名称': '非金属矿采选业'},
              {'行业代码': 'B11', '行业名称': '开采辅助活动'},
              {'行业代码': 'B12', '行业名称': '其他采矿业'},
              {'行业代码': 'C13', '行业名称': '农副食品加工业'},
              {'行业代码': 'C14', '行业名称': '食品制造业'},
              {'行业代码': 'C15', '行业名称': '酒、饮料和精制茶制造业'},
              {'行业代码': 'C16', '行业名称': '烟草制品业'},
              {'行业代码': 'C17', '行业名称': '纺织业'},
              {'行业代码': 'C18', '行业名称': '纺织服装、服饰业'},
              {'行业代码': 'C19', '行业名称': '皮革、毛皮、羽毛及其制品和制鞋业'},
              {'行业代码': 'C20', '行业名称': '木材加工和木、竹、藤、棕、草制品业'},
              {'行业代码': 'C21', '行业名称': '家具制造业'},
              {'行业代码': 'C22', '行业名称': '造纸和纸制品业'},
              {'行业代码': 'C23', '行业名称': '印刷和记录媒介复制业'},
              {'行业代码': 'C24', '行业名称': '文教、工美、体育和娱乐用品制造业'},
              {'行业代码': 'C25', '行业名称': '石油加工、炼焦和核燃料加工业'},
              {'行业代码': 'C26', '行业名称': '化学原料和化学制品制造业'},
              {'行业代码': 'C27', '行业名称': '医药制造业'},
              {'行业代码': 'C28', '行业名称': '化学纤维制造业'},
              {'行业代码': 'C29', '行业名称': '橡胶和塑料制品业'},
              {'行业代码': 'C30', '行业名称': '非金属矿物制品业'},
              {'行业代码': 'C31', '行业名称': '黑色金属冶炼和压延加工业'},
              {'行业代码': 'C32', '行业名称': '有色金属冶炼和压延加工业'},
              {'行业代码': 'C33', '行业名称': '金属制品业'},
              {'行业代码': 'C34', '行业名称': '通用设备制造业'},
              {'行业代码': 'C35', '行业名称': '专用设备制造业'},
              {'行业代码': 'C36', '行业名称': '汽车制造业'},
              {'行业代码': 'C37', '行业名称': '铁路、船舶、航空航天和其他运输设备制造业'},
              {'行业代码': 'C38', '行业名称': '电气机械和器材制造业'},
              {'行业代码': 'C39', '行业名称': '计算机、通信和其他电子设备制造业'},
              {'行业代码': 'C40', '行业名称': '仪器仪表制造业'},
              {'行业代码': 'C41', '行业名称': '其他制造业'},
              {'行业代码': 'C42', '行业名称': '废弃资源综合利用业'},
              {'行业代码': 'C43', '行业名称': '金属制品、机械和设备修理业'},
              {'行业代码': 'D44', '行业名称': '电力、热力生产和供应业'},
              {'行业代码': 'D45', '行业名称': '燃气生产和供应业'},
              {'行业代码': 'D46', '行业名称': '水的生产和供应业'},
              {'行业代码': 'E47', '行业名称': '房屋建筑业'},
              {'行业代码': 'E48', '行业名称': '土木工程建筑业'},
              {'行业代码': 'E49', '行业名称': '建筑安装业'},
              {'行业代码': 'E50', '行业名称': '建筑装饰和其他建筑业'},
              {'行业代码': 'F51', '行业名称': '批发业'},
              {'行业代码': 'F52', '行业名称': '零售业'},
              {'行业代码': 'G53', '行业名称': '铁路运输业'},
              {'行业代码': 'G54', '行业名称': '道路运输业'},
              {'行业代码': 'G55', '行业名称': '水上运输业'},
              {'行业代码': 'G56', '行业名称': '航空运输业'},
              {'行业代码': 'G57', '行业名称': '管道运输业'},
              {'行业代码': 'G58', '行业名称': '装卸搬运和运输代理业'},
              {'行业代码': 'G59', '行业名称': '仓储业'},
              {'行业代码': 'G60', '行业名称': '邮政业'},
              {'行业代码': 'H61', '行业名称': '住宿业'},
              {'行业代码': 'H62', '行业名称': '餐饮业'},
              {'行业代码': 'I63', '行业名称': '电信、广播电视和卫星传输服务'},
              {'行业代码': 'I64', '行业名称': '互联网和相关服务'},
              {'行业代码': 'I65', '行业名称': '软件和信息技术服务业'},
              {'行业代码': 'J66', '行业名称': '货币金融服务'},
              {'行业代码': 'J67', '行业名称': '资本市场服务'},
              {'行业代码': 'J68', '行业名称': '保险业'},
              {'行业代码': 'J69', '行业名称': '其他金融业'},
              {'行业代码': 'K70', '行业名称': '房地产业'},
              {'行业代码': 'L71', '行业名称': '租赁业'},
              {'行业代码': 'L72', '行业名称': '商务服务业'},
              {'行业代码': 'M73', '行业名称': '研究和试验发展'},
              {'行业代码': 'M74', '行业名称': '专业技术服务业'},
              {'行业代码': 'M75', '行业名称': '科技推广和应用服务业'},
              {'行业代码': 'N76', '行业名称': '水利管理业'},
              {'行业代码': 'N77', '行业名称': '生态保护和环境治理业'},
              {'行业代码': 'N78', '行业名称': '公共设施管理业'},
              {'行业代码': 'O79', '行业名称': '居民服务业'},
              {'行业代码': 'O80', '行业名称': '机动车、电子产品和日用产品修理业'},
              {'行业代码': 'O81', '行业名称': '其他服务业'},
              {'行业代码': 'P82', '行业名称': '教育'},
              {'行业代码': 'Q83', '行业名称': '卫生'},
              {'行业代码': 'Q84', '行业名称': '社会工作'},
              {'行业代码': 'R85', '行业名称': '新闻和出版业'},
              {'行业代码': 'R86', '行业名称': '广播、电视、电影和影视录音制作业'},
              {'行业代码': 'R87', '行业名称': '文化艺术业'},
              {'行业代码': 'R88', '行业名称': '体育'},
              {'行业代码': 'R89', '行业名称': '娱乐业'},
              {'行业代码': 'S90', '行业名称': '综合'}]
    elif issuer == 'sw1':
        ll = [{'行业代码': '801010', '行业名称': '农林牧渔I'},
              {'行业代码': '801020', '行业名称': '采掘I'},
              {'行业代码': '801030', '行业名称': '化工I'},
              {'行业代码': '801040', '行业名称': '钢铁I'},
              {'行业代码': '801050', '行业名称': '有色金属I'},
              {'行业代码': '801060', '行业名称': '建筑建材I(于2014年02月21日废弃)'},
              {'行业代码': '801070', '行业名称': '机械设备I(于2014年02月21日废弃)'},
              {'行业代码': '801080', '行业名称': '电子I'},
              {'行业代码': '801090', '行业名称': '交运设备I(于2014年02月21日废弃)'},
              {'行业代码': '801100', '行业名称': '信息设备I(于2014年02月21日废弃)'},
              {'行业代码': '801110', '行业名称': '家用电器I'},
              {'行业代码': '801120', '行业名称': '食品饮料I'},
              {'行业代码': '801130', '行业名称': '纺织服装I'},
              {'行业代码': '801140', '行业名称': '轻工制造I'},
              {'行业代码': '801150', '行业名称': '医药生物I'},
              {'行业代码': '801160', '行业名称': '公用事业I'},
              {'行业代码': '801170', '行业名称': '交通运输I'},
              {'行业代码': '801180', '行业名称': '房地产I'},
              {'行业代码': '801190', '行业名称': '金融服务I(于2014年02月21日废弃)'},
              {'行业代码': '801200', '行业名称': '商业贸易I'},
              {'行业代码': '801210', '行业名称': '休闲服务I'},
              {'行业代码': '801220', '行业名称': '信息服务I(于2014年02月21日废弃)'},
              {'行业代码': '801230', '行业名称': '综合I'},
              {'行业代码': '801710', '行业名称': '建筑材料I'},
              {'行业代码': '801720', '行业名称': '建筑装饰I'},
              {'行业代码': '801730', '行业名称': '电气设备I'},
              {'行业代码': '801740', '行业名称': '国防军工I'},
              {'行业代码': '801750', '行业名称': '计算机I'},
              {'行业代码': '801760', '行业名称': '传媒I'},
              {'行业代码': '801770', '行业名称': '通信I'},
              {'行业代码': '801780', '行业名称': '银行I'},
              {'行业代码': '801790', '行业名称': '非银金融I'},
              {'行业代码': '801880', '行业名称': '汽车I'},
              {'行业代码': '801890', '行业名称': '机械设备I'}]
    elif issuer == 'sw2':
        ll = [{'行业代码': '801011', '行业名称': '林业II'},
              {'行业代码': '801012', '行业名称': '农产品加工II'},
              {'行业代码': '801013', '行业名称': '农业综合II'},
              {'行业代码': '801014', '行业名称': '饲料II'},
              {'行业代码': '801015', '行业名称': '渔业II'},
              {'行业代码': '801016', '行业名称': '种植业II'},
              {'行业代码': '801017', '行业名称': '畜禽养殖II'},
              {'行业代码': '801018', '行业名称': '动物保健II'},
              {'行业代码': '801021', '行业名称': '煤炭开采II'},
              {'行业代码': '801022', '行业名称': '其他采掘II'},
              {'行业代码': '801023', '行业名称': '石油开采II'},
              {'行业代码': '801024', '行业名称': '采掘服务II'},
              {'行业代码': '801031', '行业名称': '化工新材料II(于2014年02月21日废弃)'},
              {'行业代码': '801032', '行业名称': '化学纤维II'},
              {'行业代码': '801033', '行业名称': '化学原料II'},
              {'行业代码': '801034', '行业名称': '化学制品II'},
              {'行业代码': '801035', '行业名称': '石油化工II'},
              {'行业代码': '801036', '行业名称': '塑料II'},
              {'行业代码': '801037', '行业名称': '橡胶II'},
              {'行业代码': '801041', '行业名称': '钢铁II'},
              {'行业代码': '801042', '行业名称': '金属制品II(于2008年06月02日废弃)'},
              {'行业代码': '801051', '行业名称': '金属非金属新材料II'},
              {'行业代码': '801052', '行业名称': '有色金属冶炼II(于2014年02月21日废弃)'},
              {'行业代码': '801053', '行业名称': '黄金II'},
              {'行业代码': '801054', '行业名称': '稀有金属II'},
              {'行业代码': '801055', '行业名称': '工业金属II'},
              {'行业代码': '801061', '行业名称': '建筑材料II(于2014年02月21日废弃)'},
              {'行业代码': '801062', '行业名称': '建筑装饰II(于2014年02月21日废弃)'},
              {'行业代码': '801071', '行业名称': '电气设备II(于2014年02月21日废弃)'},
              {'行业代码': '801072', '行业名称': '通用机械II'},
              {'行业代码': '801073', '行业名称': '仪器仪表II'},
              {'行业代码': '801074', '行业名称': '专用设备II'},
              {'行业代码': '801075', '行业名称': '金属制品II'},
              {'行业代码': '801076', '行业名称': '运输设备II'},
              {'行业代码': '801081', '行业名称': '半导体II'},
              {'行业代码': '801082', '行业名称': '其他电子II'},
              {'行业代码': '801083', '行业名称': '元件II'},
              {'行业代码': '801084', '行业名称': '光学光电子II'},
              {'行业代码': '801085', '行业名称': '电子制造II'},
              {'行业代码': '801091', '行业名称': '非汽车交运设备II(于2014年02月21日废弃)'},
              {'行业代码': '801092', '行业名称': '汽车服务II'},
              {'行业代码': '801093', '行业名称': '汽车零部件II'},
              {'行业代码': '801094', '行业名称': '汽车整车II'},
              {'行业代码': '801101', '行业名称': '计算机设备II'},
              {'行业代码': '801102', '行业名称': '通信设备II'},
              {'行业代码': '801111', '行业名称': '白色家电II'},
              {'行业代码': '801112', '行业名称': '视听器材II'},
              {'行业代码': '801121', '行业名称': '食品加工II(于2011年10月10日废弃)'},
              {'行业代码': '801122', '行业名称': '食品制造II(于2011年10月10日废弃)'},
              {'行业代码': '801123', '行业名称': '饮料制造II'},
              {'行业代码': '801124', '行业名称': '食品加工II'},
              {'行业代码': '801131', '行业名称': '纺织制造II'},
              {'行业代码': '801132', '行业名称': '服装家纺II'},
              {'行业代码': '801141', '行业名称': '包装印刷II'},
              {'行业代码': '801142', '行业名称': '家用轻工II'},
              {'行业代码': '801143', '行业名称': '造纸II'},
              {'行业代码': '801144', '行业名称': '其他轻工制造II'},
              {'行业代码': '801151', '行业名称': '化学制药II'},
              {'行业代码': '801152', '行业名称': '生物制品II'},
              {'行业代码': '801153', '行业名称': '医疗器械II'},
              {'行业代码': '801154', '行业名称': '医药商业II'},
              {'行业代码': '801155', '行业名称': '中药II'},
              {'行业代码': '801156', '行业名称': '医疗服务II'},
              {'行业代码': '801161', '行业名称': '电力II'},
              {'行业代码': '801162', '行业名称': '环保工程及服务II'},
              {'行业代码': '801163', '行业名称': '燃气II'},
              {'行业代码': '801164', '行业名称': '水务II'},
              {'行业代码': '801171', '行业名称': '港口II'},
              {'行业代码': '801172', '行业名称': '公交II'},
              {'行业代码': '801173', '行业名称': '航空运输II'},
              {'行业代码': '801174', '行业名称': '机场II'},
              {'行业代码': '801175', '行业名称': '高速公路II'},
              {'行业代码': '801176', '行业名称': '航运II'},
              {'行业代码': '801177', '行业名称': '铁路运输II'},
              {'行业代码': '801178', '行业名称': '物流II'},
              {'行业代码': '801181', '行业名称': '房地产开发II'},
              {'行业代码': '801182', '行业名称': '园区开发II'},
              {'行业代码': '801191', '行业名称': '多元金融II'},
              {'行业代码': '801192', '行业名称': '银行II'},
              {'行业代码': '801193', '行业名称': '证券II'},
              {'行业代码': '801194', '行业名称': '保险II'},
              {'行业代码': '801201', '行业名称': '零售II(于2014年02月21日废弃)'},
              {'行业代码': '801202', '行业名称': '贸易II'},
              {'行业代码': '801203', '行业名称': '一般零售II'},
              {'行业代码': '801204', '行业名称': '专业零售II'},
              {'行业代码': '801205', '行业名称': '商业物业经营II'},
              {'行业代码': '801211', '行业名称': '餐饮II'},
              {'行业代码': '801212', '行业名称': '景点II'},
              {'行业代码': '801213', '行业名称': '酒店II'},
              {'行业代码': '801214', '行业名称': '旅游综合II'},
              {'行业代码': '801215', '行业名称': '其他休闲服务II'},
              {'行业代码': '801221', '行业名称': '传媒II(于2014年02月21日废弃)'},
              {'行业代码': '801222', '行业名称': '计算机应用II'},
              {'行业代码': '801223', '行业名称': '通信运营II'},
              {'行业代码': '801224', '行业名称': '网络服务II(于2014年02月21日废弃)'},
              {'行业代码': '801231', '行业名称': '综合II'},
              {'行业代码': '801711', '行业名称': '水泥制造II'},
              {'行业代码': '801712', '行业名称': '玻璃制造II'},
              {'行业代码': '801713', '行业名称': '其他建材II'},
              {'行业代码': '801721', '行业名称': '房屋建设II'},
              {'行业代码': '801722', '行业名称': '装修装饰II'},
              {'行业代码': '801723', '行业名称': '基础建设II'},
              {'行业代码': '801724', '行业名称': '专业工程II'},
              {'行业代码': '801725', '行业名称': '园林工程II'},
              {'行业代码': '801731', '行业名称': '电机II'},
              {'行业代码': '801732', '行业名称': '电气自动化设备II'},
              {'行业代码': '801733', '行业名称': '电源设备II'},
              {'行业代码': '801734', '行业名称': '高低压设备II'},
              {'行业代码': '801741', '行业名称': '航天装备II'},
              {'行业代码': '801742', '行业名称': '航空装备II'},
              {'行业代码': '801743', '行业名称': '地面兵装II'},
              {'行业代码': '801744', '行业名称': '船舶制造II'},
              {'行业代码': '801751', '行业名称': '营销传播II'},
              {'行业代码': '801752', '行业名称': '互联网传媒II'},
              {'行业代码': '801761', '行业名称': '文化传媒II'},
              {'行业代码': '801881', '行业名称': '其他交运设备II'}]
    elif issuer == 'sw3':
        ll = [{'行业代码': '850111', '行业名称': '种子生产III'},
              {'行业代码': '850112', '行业名称': '粮食种植III'},
              {'行业代码': '850113', '行业名称': '其他种植业III'},
              {'行业代码': '850121', '行业名称': '海洋捕捞III'},
              {'行业代码': '850122', '行业名称': '水产养殖III'},
              {'行业代码': '850131', '行业名称': '林业III'},
              {'行业代码': '850141', '行业名称': '饲料III'},
              {'行业代码': '850151', '行业名称': '果蔬加工III'},
              {'行业代码': '850152', '行业名称': '粮油加工III'},
              {'行业代码': '850153', '行业名称': '畜禽加工III(于2011年10月10日废弃)'},
              {'行业代码': '850154', '行业名称': '其他农产品加工III'},
              {'行业代码': '850161', '行业名称': '农业综合III'},
              {'行业代码': '850171', '行业名称': '畜禽养殖III'},
              {'行业代码': '850181', '行业名称': '动物保健III'},
              {'行业代码': '850211', '行业名称': '石油开采III'},
              {'行业代码': '850221', '行业名称': '煤炭开采III'},
              {'行业代码': '850222', '行业名称': '焦炭加工III'},
              {'行业代码': '850231', '行业名称': '其他采掘III'},
              {'行业代码': '850241', '行业名称': '油气钻采服务III'},
              {'行业代码': '850242', '行业名称': '其他采掘服务III'},
              {'行业代码': '850311', '行业名称': '石油加工III'},
              {'行业代码': '850313', '行业名称': '石油贸易III'},
              {'行业代码': '850321', '行业名称': '纯碱III'},
              {'行业代码': '850322', '行业名称': '氯碱III'},
              {'行业代码': '850323', '行业名称': '无机盐III'},
              {'行业代码': '850324', '行业名称': '其他化学原料III'},
              {'行业代码': '850331', '行业名称': '氮肥III'},
              {'行业代码': '850332', '行业名称': '磷肥III'},
              {'行业代码': '850333', '行业名称': '农药III'},
              {'行业代码': '850334', '行业名称': '日用化学产品III'},
              {'行业代码': '850335', '行业名称': '涂料油漆油墨制造III'},
              {'行业代码': '850336', '行业名称': '钾肥III'},
              {'行业代码': '850337', '行业名称': '民爆用品III'},
              {'行业代码': '850338', '行业名称': '纺织化学用品III'},
              {'行业代码': '850339', '行业名称': '其他化学制品III'},
              {'行业代码': '850341', '行业名称': '涤纶III'},
              {'行业代码': '850342', '行业名称': '维纶III'},
              {'行业代码': '850343', '行业名称': '粘胶III'},
              {'行业代码': '850344', '行业名称': '其他纤维III'},
              {'行业代码': '850345', '行业名称': '氨纶III'},
              {'行业代码': '850351', '行业名称': '其他塑料制品III'},
              {'行业代码': '850352', '行业名称': '合成革III'},
              {'行业代码': '850353', '行业名称': '改性塑料III'},
              {'行业代码': '850361', '行业名称': '轮胎III'},
              {'行业代码': '850362', '行业名称': '其他橡胶制品III'},
              {'行业代码': '850363', '行业名称': '炭黑III'},
              {'行业代码': '850371', '行业名称': '其他化工新材料III(于2014年02月21日废弃)'},
              {'行业代码': '850372', '行业名称': '聚氨酯III'},
              {'行业代码': '850373', '行业名称': '玻纤III'},
              {'行业代码': '850381', '行业名称': '复合肥III'},
              {'行业代码': '850382', '行业名称': '氟化工及制冷剂III'},
              {'行业代码': '850383', '行业名称': '磷化工及磷酸盐III'},
              {'行业代码': '850411', '行业名称': '普钢III'},
              {'行业代码': '850412', '行业名称': '特钢III'},
              {'行业代码': '850421', '行业名称': '金属制品III(于2008年06月02日废弃)'},
              {'行业代码': '850511', '行业名称': '有色金属冶炼III(于2007年07月02日废弃)'},
              {'行业代码': '850512', '行业名称': '铝III(于2014年02月21日废弃)'},
              {'行业代码': '850513', '行业名称': '铜III(于2014年02月21日废弃)'},
              {'行业代码': '850514', '行业名称': '铅锌III(于2014年02月21日废弃)'},
              {'行业代码': '850515', '行业名称': '黄金III(于2014年02月21日废弃)'},
              {'行业代码': '850516', '行业名称': '小金属III(于2014年02月21日废弃)'},
              {'行业代码': '850521', '行业名称': '金属新材料III'},
              {'行业代码': '850522', '行业名称': '磁性材料III'},
              {'行业代码': '850523', '行业名称': '非金属新材料III'},
              {'行业代码': '850531', '行业名称': '黄金III'},
              {'行业代码': '850541', '行业名称': '稀土III'},
              {'行业代码': '850542', '行业名称': '钨III'},
              {'行业代码': '850543', '行业名称': '锂III'},
              {'行业代码': '850544', '行业名称': '其他稀有小金属III'},
              {'行业代码': '850551', '行业名称': '铝III'},
              {'行业代码': '850552', '行业名称': '铜III'},
              {'行业代码': '850553', '行业名称': '铅锌III'},
              {'行业代码': '850611', '行业名称': '玻璃制造III'},
              {'行业代码': '850612', '行业名称': '水泥制造III'},
              {'行业代码': '850613', '行业名称': '陶瓷制造III(于2014年02月21日废弃)'},
              {'行业代码': '850614', '行业名称': '其他建材III'},
              {'行业代码': '850615', '行业名称': '耐火材料III'},
              {'行业代码': '850616', '行业名称': '管材III'},
              {'行业代码': '850621', '行业名称': '建筑施工III(于2011年10月10日废弃)'},
              {'行业代码': '850622', '行业名称': '装饰园林III(于2014年02月21日废弃)'},
              {'行业代码': '850623', '行业名称': '房屋建设III'},
              {'行业代码': '850624', '行业名称': '基础建设III(于2014年02月21日废弃)'},
              {'行业代码': '850625', '行业名称': '专业工程III(于2014年02月21日废弃)'},
              {'行业代码': '850711', '行业名称': '机床工具III'},
              {'行业代码': '850712', '行业名称': '机械基础件III'},
              {'行业代码': '850713', '行业名称': '磨具磨料III'},
              {'行业代码': '850714', '行业名称': '内燃机III'},
              {'行业代码': '850715', '行业名称': '制冷空调设备III'},
              {'行业代码': '850716', '行业名称': '其它通用机械III'},
              {'行业代码': '850721', '行业名称': '纺织服装设备III'},
              {'行业代码': '850722', '行业名称': '工程机械III'},
              {'行业代码': '850723', '行业名称': '农用机械III'},
              {'行业代码': '850724', '行业名称': '重型机械III'},
              {'行业代码': '850725', '行业名称': '冶金矿采化工设备III'},
              {'行业代码': '850726', '行业名称': '印刷包装机械III'},
              {'行业代码': '850727', '行业名称': '其它专用机械III'},
              {'行业代码': '850728', '行业名称': '楼宇设备III'},
              {'行业代码': '850729', '行业名称': '环保设备III'},
              {'行业代码': '850731', '行业名称': '仪器仪表III'},
              {'行业代码': '850741', '行业名称': '电机III'},
              {'行业代码': '850742', '行业名称': '电气自控设备III(于2014年02月21日废弃)'},
              {'行业代码': '850743', '行业名称': '电源设备III(于2014年02月21日废弃)'},
              {'行业代码': '850744', '行业名称': '输变电设备III(于2014年02月21日废弃)'},
              {'行业代码': '850745', '行业名称': '其他电力设备III(于2014年02月21日废弃)'},
              {'行业代码': '850751', '行业名称': '金属制品III'},
              {'行业代码': '850811', '行业名称': '集成电路III'},
              {'行业代码': '850812', '行业名称': '分立器件III'},
              {'行业代码': '850813', '行业名称': '半导体材料III'},
              {'行业代码': '850821', '行业名称': '元件III(于2011年10月10日废弃)'},
              {'行业代码': '850822', '行业名称': '印制电路板III'},
              {'行业代码': '850823', '行业名称': '被动元件III'},
              {'行业代码': '850831', '行业名称': '显示器件III'},
              {'行业代码': '850832', '行业名称': 'LEDIII'},
              {'行业代码': '850833', '行业名称': '光学元件III'},
              {'行业代码': '850841', '行业名称': '其他电子III'},
              {'行业代码': '850851', '行业名称': '电子系统组装III'},
              {'行业代码': '850852', '行业名称': '电子零部件制造III'},
              {'行业代码': '850911', '行业名称': '乘用车III'},
              {'行业代码': '850912', '行业名称': '商用载货车III'},
              {'行业代码': '850913', '行业名称': '商用载客车III'},
              {'行业代码': '850914', '行业名称': '专用汽车III(于2014年02月21日废弃)'},
              {'行业代码': '850921', '行业名称': '汽车零部件III'},
              {'行业代码': '850931', '行业名称': '摩托车III(于2014年02月21日废弃)'},
              {'行业代码': '850932', '行业名称': '农机设备III(于2011年10月10日废弃)'},
              {'行业代码': '850933', '行业名称': '其他交运设备III(于2007年07月02日废弃)'},
              {'行业代码': '850934', '行业名称': '航空航天设备III(于2014年02月21日废弃)'},
              {'行业代码': '850935', '行业名称': '船舶制造III'},
              {'行业代码': '850936', '行业名称': '铁路设备III'},
              {'行业代码': '850937', '行业名称': '其他交运设备III(于2014年02月21日废弃)'},
              {'行业代码': '850941', '行业名称': '汽车服务III'},
              {'行业代码': '850942', '行业名称': '其他交运设备服务III(于2014年02月21日废弃)'},
              {'行业代码': '851011', '行业名称': '交换设备III(于2011年10月10日废弃)'},
              {'行业代码': '851012', '行业名称': '终端设备III'},
              {'行业代码': '851013', '行业名称': '通信传输设备III'},
              {'行业代码': '851014', '行业名称': '通信配套服务III'},
              {'行业代码': '851021', '行业名称': '计算机设备III'},
              {'行业代码': '851111', '行业名称': '冰箱III'},
              {'行业代码': '851112', '行业名称': '空调III'},
              {'行业代码': '851113', '行业名称': '洗衣机III'},
              {'行业代码': '851114', '行业名称': '小家电III'},
              {'行业代码': '851115', '行业名称': '家电零部件III'},
              {'行业代码': '851121', '行业名称': '彩电III'},
              {'行业代码': '851122', '行业名称': '其它视听器材III'},
              {'行业代码': '851211', '行业名称': '肉制品III(于2011年10月10日废弃)'},
              {'行业代码': '851212', '行业名称': '制糖III(于2011年10月10日废弃)'},
              {'行业代码': '851213', '行业名称': '食品综合III(于2011年10月10日废弃)'},
              {'行业代码': '851221', '行业名称': '调味发酵品III(于2011年10月10日废弃)'},
              {'行业代码': '851222', '行业名称': '乳品III(于2011年10月10日废弃)'},
              {'行业代码': '851231', '行业名称': '白酒III'},
              {'行业代码': '851232', '行业名称': '啤酒III'},
              {'行业代码': '851233', '行业名称': '其他酒类III'},
              {'行业代码': '851234', '行业名称': '软饮料III'},
              {'行业代码': '851235', '行业名称': '葡萄酒III'},
              {'行业代码': '851236', '行业名称': '黄酒III'},
              {'行业代码': '851241', '行业名称': '肉制品III'},
              {'行业代码': '851242', '行业名称': '调味发酵品III'},
              {'行业代码': '851243', '行业名称': '乳品III'},
              {'行业代码': '851244', '行业名称': '食品综合III'},
              {'行业代码': '851311', '行业名称': '毛纺III'},
              {'行业代码': '851312', '行业名称': '棉纺III'},
              {'行业代码': '851313', '行业名称': '丝绸III'},
              {'行业代码': '851314', '行业名称': '印染III'},
              {'行业代码': '851315', '行业名称': '辅料III'},
              {'行业代码': '851316', '行业名称': '其他纺织III'},
              {'行业代码': '851321', '行业名称': '服装III(于2011年10月10日废弃)'},
              {'行业代码': '851322', '行业名称': '男装III'},
              {'行业代码': '851323', '行业名称': '女装III'},
              {'行业代码': '851324', '行业名称': '休闲服装III'},
              {'行业代码': '851325', '行业名称': '鞋帽III'},
              {'行业代码': '851326', '行业名称': '家纺III'},
              {'行业代码': '851327', '行业名称': '其他服装III'},
              {'行业代码': '851411', '行业名称': '造纸III'},
              {'行业代码': '851421', '行业名称': '包装印刷III'},
              {'行业代码': '851431', '行业名称': '其他轻工制造III(于2011年10月10日废弃)'},
              {'行业代码': '851432', '行业名称': '家具III'},
              {'行业代码': '851433', '行业名称': '其他家用轻工III'},
              {'行业代码': '851434', '行业名称': '珠宝首饰III'},
              {'行业代码': '851435', '行业名称': '文娱用品III'},
              {'行业代码': '851441', '行业名称': '其他轻工制造III'},
              {'行业代码': '851511', '行业名称': '化学原料药III'},
              {'行业代码': '851512', '行业名称': '化学制剂III'},
              {'行业代码': '851513', '行业名称': '化学药III(于2008年06月02日废弃)'},
              {'行业代码': '851521', '行业名称': '中药III'},
              {'行业代码': '851531', '行业名称': '生物制品III'},
              {'行业代码': '851541', '行业名称': '医药商业III'},
              {'行业代码': '851551', '行业名称': '医疗器械III'},
              {'行业代码': '851561', '行业名称': '医疗服务III'},
              {'行业代码': '851611', '行业名称': '火电III'},
              {'行业代码': '851612', '行业名称': '水电III'},
              {'行业代码': '851613', '行业名称': '燃机发电III'},
              {'行业代码': '851614', '行业名称': '热电III'},
              {'行业代码': '851615', '行业名称': '新能源发电III'},
              {'行业代码': '851621', '行业名称': '水务III'},
              {'行业代码': '851631', '行业名称': '燃气III'},
              {'行业代码': '851641', '行业名称': '环保工程及服务III'},
              {'行业代码': '851711', '行业名称': '港口III'},
              {'行业代码': '851721', '行业名称': '公交III'},
              {'行业代码': '851731', '行业名称': '高速公路III'},
              {'行业代码': '851741', '行业名称': '航空运输III'},
              {'行业代码': '851751', '行业名称': '机场III'},
              {'行业代码': '851761', '行业名称': '航运III'},
              {'行业代码': '851771', '行业名称': '铁路运输III'},
              {'行业代码': '851781', '行业名称': '物流III'},
              {'行业代码': '851811', '行业名称': '房地产开发III'},
              {'行业代码': '851821', '行业名称': '园区开发III'},
              {'行业代码': '851911', '行业名称': '银行III'},
              {'行业代码': '851921', '行业名称': '多元金融III'},
              {'行业代码': '851931', '行业名称': '证券III'},
              {'行业代码': '851941', '行业名称': '保险III'},
              {'行业代码': '852011', '行业名称': '百货零售III(于2014年02月21日废弃)'},
              {'行业代码': '852012', '行业名称': '专业连锁III(于2014年02月21日废弃)'},
              {'行业代码': '852013', '行业名称': '商业物业经营III(于2014年02月21日废弃)'},
              {'行业代码': '852021', '行业名称': '贸易III'},
              {'行业代码': '852031', '行业名称': '百货III'},
              {'行业代码': '852032', '行业名称': '超市III'},
              {'行业代码': '852033', '行业名称': '多业态零售III'},
              {'行业代码': '852041', '行业名称': '专业连锁III'},
              {'行业代码': '852051', '行业名称': '一般物业经营III'},
              {'行业代码': '852052', '行业名称': '专业市场III'},
              {'行业代码': '852111', '行业名称': '人工景点III'},
              {'行业代码': '852112', '行业名称': '自然景点III'},
              {'行业代码': '852121', '行业名称': '酒店III'},
              {'行业代码': '852131', '行业名称': '旅游综合III'},
              {'行业代码': '852141', '行业名称': '餐饮III'},
              {'行业代码': '852151', '行业名称': '其他休闲服务III'},
              {'行业代码': '852211', '行业名称': '通信运营III'},
              {'行业代码': '852221', '行业名称': '互联网信息服务III'},
              {'行业代码': '852222', '行业名称': '移动互联网服务III'},
              {'行业代码': '852223', '行业名称': '其他互联网服务III'},
              {'行业代码': '852224', '行业名称': '有线电视网络III'},
              {'行业代码': '852225', '行业名称': '软件开发III'},
              {'行业代码': '852226', '行业名称': 'IT服务III'},
              {'行业代码': '852231', '行业名称': '计算机应用III(于2008年06月02日废弃)'},
              {'行业代码': '852232', '行业名称': '软件开发及服务III(于2014年02月21日废弃)'},
              {'行业代码': '852233', '行业名称': '系统集成III(于2014年02月21日废弃)'},
              {'行业代码': '852241', '行业名称': '平面媒体III'},
              {'行业代码': '852242', '行业名称': '影视动漫III'},
              {'行业代码': '852243', '行业名称': '营销服务III'},
              {'行业代码': '852244', '行业名称': '其他文化传媒III'},
              {'行业代码': '852311', '行业名称': '综合III'},
              {'行业代码': '857221', '行业名称': '装修装饰III'},
              {'行业代码': '857231', '行业名称': '城轨建设III'},
              {'行业代码': '857232', '行业名称': '路桥施工III'},
              {'行业代码': '857233', '行业名称': '水利工程III'},
              {'行业代码': '857234', '行业名称': '铁路建设III'},
              {'行业代码': '857235', '行业名称': '其他基础建设III'},
              {'行业代码': '857241', '行业名称': '钢结构III'},
              {'行业代码': '857242', '行业名称': '化学工程III'},
              {'行业代码': '857243', '行业名称': '国际工程承包III'},
              {'行业代码': '857244', '行业名称': '其他专业工程III'},
              {'行业代码': '857251', '行业名称': '园林工程III'},
              {'行业代码': '857321', '行业名称': '电网自动化III'},
              {'行业代码': '857322', '行业名称': '工控自动化III'},
              {'行业代码': '857323', '行业名称': '计量仪表III'},
              {'行业代码': '857331', '行业名称': '综合电力设备商III'},
              {'行业代码': '857332', '行业名称': '风电设备III'},
              {'行业代码': '857333', '行业名称': '光伏设备III'},
              {'行业代码': '857334', '行业名称': '火电设备III'},
              {'行业代码': '857335', '行业名称': '储能设备III'},
              {'行业代码': '857336', '行业名称': '其它电源设备III'},
              {'行业代码': '857341', '行业名称': '高压设备III'},
              {'行业代码': '857342', '行业名称': '中压设备III'},
              {'行业代码': '857343', '行业名称': '低压设备III'},
              {'行业代码': '857344', '行业名称': '线缆部件及其他III'},
              {'行业代码': '857411', '行业名称': '航天装备III'},
              {'行业代码': '857421', '行业名称': '航空装备III'},
              {'行业代码': '857431', '行业名称': '地面兵装III'},
              {'行业代码': '858811', '行业名称': '其他交运设备III'}]
    else:
        raise ValueError

    component = defaultdict(dict)

    for l in ll:
        code = l['行业代码']
        for date in tqdm.tqdm(trading_date):
            component[date][code] = jqdatasdk.get_industry_stocks(code, date=date)
    # print(component)
    save_pickle(component, os.path.join(path, issuer + '_industry' + '.pickle'))


def get_sw1_daily_data(date):
    df = finance.run_query(query(finance.SW1_DAILY_PRICE)
                           .filter(finance.SW1_DAILY_PRICE.date == date))
    return df


def download_all_sw1_daily_data(trading_date: list, data_folder='../data/market'):
    dfs = []
    for d in tqdm.tqdm(trading_date):
        if d < datetime.date.today():
            df = get_sw1_daily_data(d)
            dfs.append(df)
    data = pd.concat(dfs)  # type: pd.DataFrame

    data.to_parquet(os.path.join(data_folder, 'SW1_D_data.parquet'))


# def download_market_ret(index_symbol_list: list, freq='M', code_rename='instCode', trade_date_rename='date'):
#     pass
#     for index in index_symbol_list:
#         ticker = yf.ticker(index)

def load_jointquant_fundamental(path: str):
    df = pd.read_parquet(path)
    df['code'] = df['code'].apply(lambda x: x.split('.')[0])
    df['pub_date'] = pd.to_datetime(df['pub_date'])
    df.set_index(['pub_date', 'code'], inplace=True)
    df.sort_index(inplace=True)
    return df


def download_daily_basic(data_folder='../data/daily_basic'):
    tickers = get_all_ticker()
    dfs = []
    for t in tqdm.tqdm(tickers.index):
        df = pro.daily_basic(ts_code=t)
        dfs.append(df)
    data = pd.concat(dfs)
    data = data.rename(columns={'ts_code': 'code', 'trade_date': 'date'})
    data['date'] = pd.to_datetime(data['date'])  # type: pd.DataFrame
    data = data.set_index(['date', 'code'])
    data.to_parquet(os.path.join(data_folder, 'daily_basic.parquet'))

def download_margin_stocks(trading_dates, data_path='../data/margin.pickle'):
    d = {}
    for date in tqdm.tqdm(trading_dates):
        d[date] = jqdatasdk.get_marginsec_stocks(date)
    save_pickle(d, data_path)





if __name__ == '__main__':
    # data = download_market_data('300104.SZ',)
    # download_all_and_save()
    # ticker = '000001.XSHE'
    # download_capital_change(ticker)
    # download_balance_sheet(ticker)
    # download_income_statement(ticker)
    # download_cashflow_statement(ticker)
    # bs = download_balance(ticker)
    # download_all_suspend()
    # download_all_suspend()
    download_trade_days()
    trading_list = load_pickle(r'../data/trading_days.pickle')
    trading_list.sort()
    end_month = list(set(get_last_trading_of_month_dict(trading_list).values()))
    end_month.sort()
    # end_month = [m.to_pydatetime() for m in end_month]
    # csi300 = '000300.XSHG'
    # ZZ500 = '000905.XSHG'
    # download_component(csi300, end_month)
    # download_component(ZZ500, end_month)
    # download_all_market(freq='D')
    # download_industries_component(trading_date=end_month, issuer='sw1')
    # codes = jqdatasdk.get_all_securities(types=['stock']).index
    # codes2 = codes[codes > '6003968.XSHG']
    # download_all_supplement(codes)
    # download_all_shareholder_floating_top10(codes)
    # download_all_shares_pledge(codes)
    # download_all_holder_num(codes)
    # download_all_money_flow(codes2)
    # download_all_shareholders_share_change(codes)
    # download_index_market_data('000300.SH', freq='D')
    # download_all_suspend()
    # download_all_limit()
    # download_all_sw1_daily_data(trading_list)
    # download_index_market_data('000906.SH', freq='D')
    # download_index_market_data('000906.SH', freq='M')
    # download_trade_days()
    # download_daily_basic()
    # download_margin_stocks(end_month)
    # df = get_market_data('000001.SZ', freq='15min', start_date='2019-09-01 09:00:00')
    zz_500 = load_pickle('../data/universe/000905.XSHG.pickle')
    download_component_min_data(zz_500, 'zz500')