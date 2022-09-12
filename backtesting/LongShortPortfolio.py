import pandas as pd
from plotly import graph_objs as go

from backtesting.Account import Account
from graph.backtest_graph import net_value_line


class LongShortPortfolio:

    def __init__(self, long_account: Account, short_account: Account):
        self.long_account = long_account
        self.short_account = short_account

    def get_return_plot(self, start_date=None,
                        end_date=None) -> go.Figure:
        long_cap = self.long_account.get_account_info()['cap']
        short_cap = self.short_account.get_account_info()['cap']
        if start_date is not None:
            long_cap = long_cap.loc[pd.to_datetime(start_date):]
            short_cap = short_cap.loc[pd.to_datetime(start_date):]

        if end_date is not None:
            long_cap = long_cap.loc[:pd.to_datetime(end_date)]
            short_cap = short_cap.loc[:pd.to_datetime(end_date)]

        long_short_cap = (long_cap.pct_change() - short_cap.pct_change()).cumsum()
        fig = go.Figure()
        fig.add_trace(net_value_line(long_short_cap, color=None))
        return fig
