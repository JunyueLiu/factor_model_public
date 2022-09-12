import pandas as pd
from sklearn.utils import shuffle

from data_management.dataIO.trading_calendar import get_trading_date, Market, trading_dates_offsets


class TimeSeriesSplit:

    def __init__(self, end_train_date: str,
                 embargo: int,
                 offset: str,
                 data_input_config_path: str,
                 market: Market = Market.AShares,
                 shuffle=True,
                 random_state=0,
                 ):
        self.data_input_config_path = data_input_config_path
        self.embargo = embargo
        trading_dates = get_trading_date(market, config_path=self.data_input_config_path)
        self.offset = trading_dates_offsets(trading_dates, offset)
        self.end_train_date = pd.to_datetime(end_train_date)
        self.start_test_date = self.end_train_date + embargo * self.offset
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: pd.DataFrame, y: pd.Series):
        X_cols = X.columns
        y_col = y.name
        data = X.join(y)
        train_data = data.loc[: self.end_train_date]
        if self.shuffle:
            train_data = shuffle(train_data, self.random_state)
        test_data = data.loc[self.start_test_date:]
        return train_data[X_cols], train_data[y_col], test_data[X_cols], test_data[y_col]
