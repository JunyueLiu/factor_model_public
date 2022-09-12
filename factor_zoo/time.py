import pandas as pd

def session_category(market_data: pd.DataFrame) -> pd.Series:
    market_data['hour'] = market_data.index.get_level_values(0).hour
    session = market_data['hour'].apply(lambda x: 0 if 0 <= x < 8 else 1 if 8 <= x < 16 else 2)
    session.name = 'session'
    return session


def weekday_category(market_data: pd.DataFrame) -> pd.Series:
    market_data['weekday'] = market_data.index.get_level_values(0).weekday
    f = market_data['weekday']
    f.name = 'weekday'
    return f