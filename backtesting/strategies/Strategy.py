from abc import ABC, abstractmethod
from typing import Dict, Set, List

from backtesting.Account import Account


class AbstractStrategy(ABC):

    def __init__(self):
        self.account = None
        self.trading_dates = None
        self.offset = None

    def on_init(self, account: Account, trading_dates, offset, **kwargs):
        self.account = account
        self.trading_dates = trading_dates
        self.offset = offset

    @abstractmethod
    def before_market_open(self, date, paused: List[str], **kwargs):
        pass

    @abstractmethod
    def on_eod_bar(self, date, bars):
        pass

    @abstractmethod
    def on_order(self):
        pass

    def _map_reduce_dict(self, d1: Dict, d2: Dict):
        d = {}
        for x in set(d1).union(d2):
            v = d1.get(x, 0) + d2.get(x, 0)
            if v != 0:
                d[x] = v
        return d

    def place_order(self, orders: Dict[str, float]):
        unfilled_order = self._map_reduce_dict(self.account.get_unfilled_orders(), orders)
        self.account.set_unfilled_orders(unfilled_order)

    def cancel_order(self, cancel_tickers: Set[str]):
        d = {k: v for k, v in self.account.get_unfilled_orders().items() if k not in cancel_tickers}
        self.account.set_unfilled_orders(d)


