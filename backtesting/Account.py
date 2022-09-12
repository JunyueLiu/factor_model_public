from typing import List, Dict

import numpy as np
import pandas as pd
from attr import dataclass


@dataclass
class Account:
    dates: List[pd.Timestamp]
    cash: List[float]
    equity: List[float]
    holding: List[Dict[str, float]]
    holding_equity: List[Dict[str, float]]
    fee: List[float]
    unfilled_orders = dict()
    records: List[Dict]

    def add(self, date: pd.Timestamp, cash: float, equity: float, holding: Dict[str, float],
            holding_equity: Dict[str, float], fee: float, record: List):
        self.dates.append(date)
        self.cash.append(cash)
        self.equity.append(equity)
        self.holding.append(holding)
        self.holding_equity.append(holding_equity)
        self.fee.append(fee)
        self.records.extend(record)

    def get_snapshot(self, date):
        history_holding = self.get_history_holding()
        snapshot = history_holding.loc[pd.to_datetime(date)]
        return snapshot

    def get_cash(self):
        return self.cash[-1]

    def get_holding(self) -> Dict[str, float]:
        return self.holding[-1]

    def get_holding_equity(self) -> Dict[str, float]:
        return self.holding_equity[-1]

    def get_equity(self):
        return self.equity[-1]

    def get_unfilled_orders(self):
        return self.unfilled_orders

    def set_unfilled_orders(self, unfilled_orders):
        self.unfilled_orders = unfilled_orders

    def get_history_holding(self) -> pd.Series:
        series = pd.DataFrame(self.holding, index=self.dates).stack().sort_index()
        series.name = 'holding'
        return series

    def get_account_info(self) -> pd.DataFrame:
        df = pd.DataFrame({'cash': self.cash, 'equity': self.equity, 'fee': self.fee}, index=self.dates)
        df['cap'] = df['cash'] + df['equity']
        return df

    def get_full_trade_records(self):
        records = pd.DataFrame(self.records)
        records = records.sort_values(['date', 'code'])
        records['cum_volume'] = records.groupby('code')['volume'].cumsum()
        records['cashflow'] = - records['amount'] - records['fee']
        records['stock_cum_cashflow'] = records.groupby('code')['cashflow'].cumsum()
        records['trade_id'] = records['cum_volume'].apply(lambda x: 1 if x == 0 else np.nan)
        records['trade_id'] = records.groupby(by='code')['trade_id'].cumsum()
        records['trade_id'] = records.groupby(by='code')['trade_id'].fillna(method='bfill')
        records['trade_cum_cashflow'] = records.groupby(by=['code', 'trade_id'])['cashflow'].cumsum()
        records['cost'] = - records['trade_cum_cashflow'] / records['cum_volume']
        return records
