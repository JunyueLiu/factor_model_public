import ccxt
import pandas as pd
import requests


def bybit_symbol_to_coin(symbol: str):
    return symbol.replace("USDT", '').replace("USD", '')


def get_symbol_info(usdt=True):
    # https://api.bybit.com/v2/public/tickers
    r = requests.get('https://api.bybit.com/v2/public/symbols')
    while True:
        if r.status_code == 200:
            df = pd.DataFrame(r.json()['result'])
            break
    if usdt:
        df = df[df['name'].str.endswith("USDT")]
    df['tick_size'] = df['price_filter'].apply(lambda x: float(x['tick_size']))
    df['min_trading_qty'] = df['lot_size_filter'].apply(lambda x: float(x['min_trading_qty']))

    return df

def get_bars(code, interval, start_time):
    bybit = ccxt.bybit()
    start = int(pd.to_datetime(start_time, utc=True).timestamp())

    results = bybit.public_linear_get_kline(params={'symbol': code, 'interval': interval, 'from': start})
    df = pd.DataFrame(results['result'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='s')
    for c in ['volume', 'open', 'high', 'low', 'close', 'turnover']:
        df[c] = df[c].astype(float)
    df = df.set_index('open_time')
    return df






if __name__ == '__main__':
    # info = get_symbol_info()
    # bybit = ccxt.bybit()
    get_bars('BTCUSDT', 240, '2022-05-21 04:00:00')