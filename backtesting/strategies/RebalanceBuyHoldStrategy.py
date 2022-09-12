from decimal import Decimal
from typing import List, Dict

import pandas as pd

from backtesting.Account import Account
from backtesting.strategies.Strategy import AbstractStrategy


class RebalanceBuyHoldStrategy(AbstractStrategy):

    def __init__(self, stock_selections: pd.Series, force_rebalance=False, timing=1):
        super().__init__()
        self.stock_selections = stock_selections.sort_index()
        self.force_rebalance = force_rebalance
        self.rebalance_dates = set(stock_selections.index.get_level_values(0).drop_duplicates())
        self.target_stocks_weights = None
        self.holding_paused = []
        self.timing = timing

    def on_init(self, account: Account, trading_dates, offset, **kwargs):
        super().on_init(account, trading_dates, offset, **kwargs)

    def before_market_open(self, date, paused: List[str], **kwargs):
        last_day = date - self.offset
        holding = self.account.get_holding()
        self.holding_paused = list(set(holding.keys()).intersection(paused))
        if last_day in self.rebalance_dates:
            target_stocks_weights = self.stock_selections.loc[last_day]
            self.cancel_order(set(target_stocks_weights.index).intersection(paused))
            # idx = set(target_stocks_weights.index).difference(paused) \
            #     .union(set(holding.keys()).intersection(target_stocks_weights.index))
            # target_stocks_weights = target_stocks_weights.loc[idx]
            # target_stocks_weights = target_stocks_weights / target_stocks_weights.sum()
            # self.target_stocks_weights = target_stocks_weights

    def on_eod_bar(self, date, bars: pd.DataFrame):
        if date in self.rebalance_dates:
            target_stocks_weights = self.stock_selections[date]
            holding = self.account.get_holding().copy()
            trade_vol = self.cal_opposite(holding)
            # todo could be problematic because of pause
            # should subtract those paused
            if isinstance(self.timing, pd.Series):
                t = self.timing.loc[:date].values[-1]
            else:
                t = self.timing

            cap = (self.account.get_cash() + self.account.get_equity()) * t
            for ticker in self.holding_paused:
                cap -= self.account.get_holding_equity().get(ticker, 0.0)
            target_vol = {}
            for ticker, weight in target_stocks_weights.items():
                try:
                    if ticker in holding and self.force_rebalance is False:
                        vol = holding[ticker]
                        target_vol[ticker] = vol
                    else:
                        price = bars.loc[ticker, 'adj_close']
                        amount = cap * weight
                        vol = int(amount / price)
                        target_vol[ticker] = vol
                except KeyError:
                    if ticker in holding:
                        vol = holding[ticker]
                        target_vol[ticker] = vol

            trade_vol = self.map_reduce_dict(trade_vol, target_vol)

            self.cancel_order(set(self.account.get_holding().keys()))
            self.place_order(trade_vol)
        elif (date - self.offset) in self.rebalance_dates:
            # date after rebalance date logic
            ticker_set = set(self.stock_selections.loc[date - self.offset].index)
            self.cancel_order(ticker_set)
        elif (date + self.offset) in self.rebalance_dates:
            # date before rebalance date logic
            # could close all position before rebalance
            pass

    def on_order(self):
        pass

    def cal_opposite(self, d: Dict[str, float]):
        return {k: -v for k, v in d.items()}

    def map_reduce_dict(self, d1: Dict, d2: Dict):
        d = {}
        for x in set(d1).union(d2):
            v = d1.get(x, 0) + d2.get(x, 0)
            if v != 0:
                d[x] = v
        return d


class CryptoBuyAndHold(RebalanceBuyHoldStrategy):
    def on_init(self, account: Account, trading_dates, **kwargs):
        super().on_init(account, trading_dates, None)

    def before_market_open(self, date, paused: List[str], **kwargs):
        pass

    def on_eod_bar(self, date, bars: pd.DataFrame):
        if date in self.rebalance_dates:
            target_stocks_weights = self.stock_selections[date]
            holding = self.account.get_holding().copy()
            trade_vol = self.cal_opposite(holding)
            # should subtract those paused
            if isinstance(self.timing, pd.Series):
                t = self.timing.loc[:date].values[-1]
            else:
                t = self.timing


            cap = (self.account.get_cash() + self.account.get_equity()) * t
            target_vol = {}
            for ticker, weight in target_stocks_weights.items():
                try:
                    if ticker in holding and self.force_rebalance is False:
                        vol = holding[ticker]
                        target_vol[ticker] = vol
                    else:
                        price = bars.loc[ticker, 'adj_close']
                        amount = cap * weight
                        vol = Decimal(amount) / Decimal(price)
                        target_vol[ticker] = float(vol)
                except KeyError:
                    if ticker in holding:
                        vol = holding[ticker]
                        target_vol[ticker] = vol

            trade_vol = self.map_reduce_dict(trade_vol, target_vol)
            self.cancel_order(set(self.account.get_holding().keys()))
            self.place_order(trade_vol)

    def map_reduce_dict(self, d1: Dict, d2: Dict):
        d = {}
        for x in set(d1).union(d2):
            v = Decimal(d1.get(x, 0)) + Decimal(d2.get(x, 0))
            v = float(v)
            if v != 0:
                d[x] = v
        return d
