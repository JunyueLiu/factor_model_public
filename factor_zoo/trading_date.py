import pandas as pd

from data_management.dataIO import market_data


def day_of_week(market_data: pd.DataFrame):
    market_data['date'] = market_data.index.get_level_values(0)
    factor = market_data['date'].dt.day_of_week
    factor.name = 'day_of_week'
    return factor


def day_of_month(market_data: pd.DataFrame):
    date = market_data.index.get_level_values(0).drop_duplicates()
    days = pd.Series(1, date)
    c = days.groupby(pd.Grouper(freq='M')).cumsum()
    c = c.to_frame('day_of_month')
    c['group'] = c.index.strftime('%Y-%m')
    days_of_month = c.groupby(by='group')['day_of_month'].max()
    days_of_month.name = 'days_of_month'
    c = c.join(days_of_month, on='group')
    factor = c['day_of_month'] / c['days_of_month']
    factor.name = 'day_of_month'
    factor = market_data.join(factor)['day_of_month']
    return factor


if __name__ == '__main__':
    start_date = '2013-01-01'
    data_config_path = '../cfg/data_input.ini'
    data = market_data.get_bars(
        cols=('open', 'high', 'low', 'close', 'volume', 'money', 'factor'),
        adjust=True, eod_time_adjust=False, add_limit=False, start_date=start_date,
        config_path=data_config_path)
    # day_of_week(data)
    day_of_month(data)
