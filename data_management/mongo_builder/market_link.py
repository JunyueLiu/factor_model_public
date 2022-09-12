import datetime

import pandas as pd
import jqdatasdk
from arctic import Arctic

from data_management.mongo_builder.fundamental import build_fundamental


def build_marketlink(arctic_store):
    """
    joinquant 的沪港通深港通数据有问题， 深港通会禁止买入某些要求不达标的股票and会暂停某时段

    :param arctic_store:
    :return:
    """
    build_fundamental(arctic_store, 'STK_EL_CONST_CHANGE')


def build_marketlink_hkex(arctic_store):
    sh_connect_url = 'https://www.hkex.com.hk/-/media/HKEX-Market/Mutual-Market/Stock-Connect/Eligible-Stocks/' \
                     'View-All-Eligible-Securities_xls/Change_of_SSE_Securities_Lists.xls?la=en'
    sh_changes = pd.read_excel(sh_connect_url, header=3, index_col=0)
    sh_changes['SSE Stock Code'] = sh_changes['SSE Stock Code'].apply(lambda x: str(x) + '.SH')
    sh_changes = sh_changes.rename(columns={'SSE Stock Code': 'code', 'Effective Date': 'date'})
    sh_changes = sh_changes.sort_values(['date', 'code'], ignore_index=True)
    # sh_changes.Change.value_counts()
    # Addition                                                                                                                1016
    # Addition to List of Eligible SSE Securities for Margin Trading and List of Eligible SSE Securities for Short Selling     627
    # Transfer to List of Special SSE Securities/Special China Connect Securities (stocks eligible for sell only)              522
    # Addition (from List of Special SSE Securities/Special China Connect Securities (stocks eligible for sell only))           97
    # Removal                                                                                                                   19
    # Remove from List of Eligible SSE Securities for Margin Trading and List of Eligible SSE Securities for Short Selling       2
    # Buy orders suspended                                                                                                       1
    # Addition to List of Special SSE Securities/Special China Connect Securities (stocks eligible for sell only)                1
    # Buy orders resumed                                                                                                         1
    # SSE Stock Code and Stock Name are changed from 601313 and SJEC respectively                                                1
    # SSE Stock Code and Stock Name are changed to 601360 and 360 SECURITY TECHNOLOGY respectively                               1
    # Name: Change, dtype: int64
    sz_connect_url = 'https://www.hkex.com.hk/-/media/HKEX-Market/Mutual-Market/Stock-Connect/Eligible-Stocks/' \
                     'View-All-Eligible-Securities_xls/Change_of_SZSE_Securities_Lists.xls?la=en'
    sz_changes = pd.read_excel(sz_connect_url, header=3, index_col=0)
    sz_changes['SZSE Stock Code'] = sz_changes['SZSE Stock Code'].apply(lambda x: str(x).rjust(6, '0') + '.SZ')
    sz_changes = sz_changes.rename(columns={'SZSE Stock Code': 'code', 'Effective Date': 'date'})
    sz_changes = sz_changes.sort_values(['date', 'code'], ignore_index=True)
    # sz_changes.Change.value_counts()
    # Addition                                                                                                                  1404
    # Transfer to List of Special SZSE Securities/Special China Connect Securities (stocks eligible for sell only)               809
    # Addition to List of Eligible SZSE Securities for Margin Trading and List of Eligible SZSE Securities for Short Selling     343
    # Addition (from List of Special SZSE Securities/Special China Connect Securities (stocks eligible for sell only))           257
    # Remove from List of Eligible SZSE Securities for Margin Trading and List of Eligible SZSE Securities for Short Selling      26
    # Removal                                                                                                                     23
    # Buy orders suspended                                                                                                        12
    # Buy orders resumed                                                                                                           8
    # SZSE Stock Code and Stock Name are changed to 1872 and CHINA MERCHANTS PORT GROUP respectively                               1
    # SZSE Stock Code and Stock Name are changed from 22 and SHENZHEN CHIWAN WHARF HOLDINGS respectively                           1
    # SZSE Stock Code and Stock Name are changed from 000043 and AVIC SUNDA HOLDING respectively                                   1
    # SZSE Stock Code and Stock Name are changed to 001914 and CHINA MERCHANTS PPTY OPERATION&SERVICE respectively                 1
    market_connect = sh_changes.append(sz_changes).reindex()
    last_update = datetime.datetime.now()
    if arctic_store.library_exists('MarketConnect'):
        arctic_store['MarketConnect'].delete('AShares')
        arctic_store['MarketConnect'].append('AShares', market_connect,
                                             metadata={'last_update': last_update, 'source': 'hkex'},
                                             prune_previous_version=False, upsert=True)
    else:
        arctic_store.initialize_library('MarketConnect')
        arctic_store['MarketConnect'].append('AShares', market_connect,
                                             metadata={'last_update': last_update, 'source': 'hkex'},
                                             prune_previous_version=False, upsert=True)


if __name__ == '__main__':
    store = Arctic('localhost')
    # build_marketlink(store)
    # jqdatasdk.auth(jq_user, jq_password)
    build_marketlink_hkex(store)
