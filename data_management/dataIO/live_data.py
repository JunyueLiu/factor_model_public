import pandas as pd
import akshare as ak
import requests


async def get_intraday_data(code: str):
    c, ex = code.split('.')
    if ex == 'SH':
        url = 'https://img1.money.126.net/data/hs/time/today/0{}.json'.format(c)
    elif ex == 'SZ':
        url = 'https://img1.money.126.net/data/hs/time/today/1{}.json'.format(c)
    else:
        raise ValueError
    r = requests.get(url)
    if r.status_code == 200:
        data = pd.DataFrame(r.json()['data'], columns=['time', 'price', 'avg_price', 'volume'])
        data['time'] = pd.to_datetime(r.json()['date'] + ' ' +data['time'])
        data['code'] = code
        data = data.set_index(['time', 'code'])
        return data
    else:
        return pd.DataFrame()


if __name__ == '__main__':
    data = get_intraday_data('600519.SH')