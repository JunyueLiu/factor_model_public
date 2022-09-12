from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd

from backtesting.strategies.RebalanceBuyHoldStrategy import RebalanceBuyHoldStrategy
from data_management.dataIO.market_data import get_bars, Freq
from technical_analysis.overlap import SAR


class EarlyExitPolicy(ABC):
    pass


class CutLossPolicy(EarlyExitPolicy):

    def __init__(self):
        self.cut_loss_price: Dict[str, float] = {}

    @abstractmethod
    def cal_cut_loss_price(self, date, code: str, entry_price: float):
        pass

    def update_cut_loss_price(self, code: str, price: float):
        self.cut_loss_price[code] = price

    def update_trailing_cut_loss_price(self, date, code, price):
        pass

    def clear_cut_loss_price(self):
        self.cut_loss_price.clear()

    @abstractmethod
    def check_cut_loss(self, date, code: str, price: float) -> bool:
        pass


class FixedCutLossPolicy(CutLossPolicy):

    def __init__(self, fixed_percentage: float):
        super(FixedCutLossPolicy, self).__init__()
        self.fixed_percentage = fixed_percentage

    def cal_cut_loss_price(self, date, code: str, entry_price: float):
        self.update_cut_loss_price(code, entry_price * (1 - self.fixed_percentage))

    def check_cut_loss(self, date, code: str, price: float) -> bool:
        cut_loss_price = self.cut_loss_price.get(code, -1)
        if cut_loss_price == -1:
            return False
        elif price <= cut_loss_price:
            return True
        return False


class SarCutLossPolicy(CutLossPolicy):
    def __init__(self, market_data: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2):
        super(SarCutLossPolicy, self).__init__()
        market_data = market_data.dropna()
        sar = market_data.groupby(level=1) \
            .apply(lambda x: SAR(x, acceleration, maximum, ['adj_high', 'adj_low'])).droplevel(0).sort_index()
        self.sar = sar
        self.entry_prices = {}

    def update_trailing_cut_loss_price(self, date, code, price):
        pass

    def cal_cut_loss_price(self, date, code: str, entry_price: float):
        self.entry_prices[code] = entry_price


    def check_cut_loss(self, date, code: str, price: float) -> bool:
        pass


class RebalanceCutLossStrategy(RebalanceBuyHoldStrategy):
    def __init__(self, stock_selections: pd.Series,
                 cutLoss: CutLossPolicy,
                 beta_timing: Optional[pd.Series] = None):
        super().__init__(stock_selections)
        self.cutLoss = cutLoss
        self.beta_timing = beta_timing

    def get_timing_weights(self, date):
        if self.beta_timing is None:
            return 1
        return self.beta_timing.loc[:date].iloc[-1]

    def on_eod_bar(self, date, bars: pd.DataFrame):
        timing = self.get_timing_weights(date)
        if date in self.rebalance_dates:
            target_stocks_weights = self.stock_selections[date]
            holding = self.account.get_holding().copy()
            trade_vol = self.cal_opposite(holding)
            # todo could be problematic because of pause
            # should subtract those paused
            cap = (self.account.get_cash() + self.account.get_equity())
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
                        amount = cap * weight * timing
                        vol = int(amount / price)
                        target_vol[ticker] = vol
                        self.cutLoss.cal_cut_loss_price(date, ticker, price)
                except KeyError:
                    if ticker in holding:
                        vol = holding[ticker]
                        target_vol[ticker] = vol

            trade_vol = self.map_reduce_dict(trade_vol, target_vol)

            self.cancel_order(set(self.account.get_holding().keys()))
            self.place_order(trade_vol)
        elif (date + self.offset) in self.rebalance_dates:
            # date before rebalance date logic
            # could close all position before rebalance
            pass
        elif (date - self.offset) in self.rebalance_dates:
            # date after rebalance date logic
            ticker_set = set(self.stock_selections.loc[date - self.offset].index)
            self.cancel_order(ticker_set)
        else:
            holding = self.account.get_holding().copy()
            if len(holding) > 0:
                cut_loss_tickers = set()
                trade_vol = {}
                for code, volume in holding.items():
                    try:
                        price = bars.loc[code, 'adj_close']
                        if self.cutLoss.check_cut_loss(date, code, price):
                            cut_loss_tickers.add(code)
                            trade_vol[code] = - volume
                        else:
                            self.cutLoss.update_trailing_cut_loss_price(date, code, price)
                    except KeyError:
                        pass
                self.cancel_order(cut_loss_tickers)
                self.place_order(trade_vol)


if __name__ == '__main__':
    data_input_path = '../../cfg/data_input.ini'
    market_data = get_bars(start_date='2021-01-01', freq=Freq.D1,
                           cols=('open', 'high', 'low', 'close', 'volume', 'money', 'factor'),
                           add_limit=True,
                           adjust=True,
                           eod_time_adjust=False,
                           config_path=data_input_path)
    SarCutLossPolicy(market_data)
