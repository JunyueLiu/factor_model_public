import time

import jqdatasdk
import pandas as pd
import requests
import tqdm
from jqdatasdk import get_all_securities

from arctic import Arctic
from data_management.tokens import jq_password, jq_user


def get_announcement(code: str, page):
    # https://np-anotice-stock.eastmoney.com/api/security/ann?page_size=100&page_index=20&ann_type=A&stock_list=600496
    symbol = code.split('.')[0]
    url = 'https://np-anotice-stock.eastmoney.com/api/security/ann' \
          '?page_size=100&&page_index={}&ann_type=A&stock_list={}'.format(page, symbol)
    header = {'User-Agent':
                  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'}
    while True:
        try:
            r = requests.get(url, headers=header)
            if r.status_code == 200:
                res = r.json()
                if res['success'] == 1:
                    # res['data']['total_hits']
                    data = pd.DataFrame(res['data']['list'])
                    if len(data) == 0:
                        return None
                    data['info'] = data['columns'].apply(lambda x: ','.join([c['column_name'] for c in x]))
                    data = data[['art_code', 'info', 'display_time', 'eiTime', 'notice_date', 'title']]
                    data['code'] = code
                    return data
                else:
                    return None
            else:
                print(r.status_code, url)
                time.sleep(10)
        except Exception as e:
            print(e.__traceback__)
            time.sleep(10)




def get_announcement_detail(code: str, eastmoney_ann_id):
    # http://data.eastmoney.com/notices/detail/689009/AN202112301537524985.html
    pass


def build_announcement(arctic_store):
    jqdatasdk.auth(jq_user, jq_password)
    info = get_all_securities(types=['stock'], date=None)  # type: pd.DataFrame
    info.index = [c.replace('XSHE', 'SZ').replace('XSHG', 'SH') for c in info.index]
    for code in tqdm.tqdm(info.index):
        i = 1
        lib = 'announcement'
        # if arctic_store.library_exists(lib) and arctic_store[lib].has_symbol(code):
        #     continue
        data = []
        while True:
            df = get_announcement(code, i)
            if df is None:
                break
            data.append(df)
            i += 1
            # time.sleep(0.1)
        if len(data) == 0:
            print(code, ' has not data')
            continue
        data = pd.concat(data)
        data = data.rename(columns={'notice_date': 'pub_date'})
        data = data.set_index('pub_date').sort_index()
        lib = 'announcement'
        last_update = data.index[-1]
        if arctic_store.library_exists(lib):
            arctic_store[lib].delete(code)
            arctic_store[lib].append(code, data, metadata={'last_update': last_update, 'source': 'eastmoney'},
                                     prune_previous_version=False, upsert=True)
        else:
            arctic_store.initialize_library(lib)
            arctic_store[lib].append(code, data, metadata={'last_update': last_update, 'source': 'eastmoney'},
                                     prune_previous_version=False, upsert=True)


if __name__ == '__main__':
    # get_announcement('000001.SZ', 18)
    build_announcement(Arctic('localhost'))
