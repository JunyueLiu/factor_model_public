import datetime
import logging
import sys
from decimal import Decimal
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtesting.Account import Account
from backtesting.strategies.RebalanceBuyHoldStrategy import RebalanceBuyHoldStrategy
from backtesting.strategies.Strategy import AbstractStrategy
from data_management.dataIO.market_data import get_bars, Freq
from data_management.dataIO.trading_calendar import trading_dates_offsets, get_trading_date, Market
from data_management.keeper.ZooKeeper import ZooKeeper


class TradeAlgo(Enum):
    OPEN = 'OPEN'
    CLOSE = 'CLOSE'
    HIGH = 'HIGH'
    LOW = 'LOW'
    VWAP = 'VWAP'
    TWAP = 'TWAP'
    OHLC4 = 'OHLC4'
    OC2 = 'OC2'
    OHC3 = 'OHC3'


def map_reduce_dict(d1: Dict, d2: Dict):
    d = {}
    for x in set(d1).union(d2):
        v = Decimal(d1.get(x, 0)) + Decimal(d2.get(x, 0))
        v = float(v)
        if v != 0:
            d[x] = v
    return d


def check_tradable(snapshot: pd.Series, trade_direction, algo: TradeAlgo, logger=None):
    if snapshot['paused'] == 1:
        if logger:
            logger.warn('{} {}'.format(snapshot.name, 'Paused...untradable'))
        else:
            print(snapshot.name, 'Paused...untradable')
        return False
    if snapshot['high'] == snapshot['low']:
        if snapshot['high'] == snapshot['high_limit'] and trade_direction > 0:
            if logger:
                logger.warn('{} {}'.format(snapshot.name, '-- high limit...untradable'))
            else:
                print(snapshot.name, '-- high limit...untradable')
            return False
        elif snapshot['low'] == snapshot['low_limit'] and trade_direction < 0:
            if logger:
                logger.warn('{} {}'.format(snapshot.name, '-- low limit...untradable'))
            else:
                print(snapshot.name, '-- low limit...untradable')
            return False
    elif algo == TradeAlgo.CLOSE:
        if snapshot['close'] == snapshot['high_limit'] and trade_direction > 0:
            if logger:
                logger.warn('{} {}'.format(snapshot.name, 'close on high limit...untradable'))
            else:
                print(snapshot.name, 'close on high limit...untradable')
            return False
        elif snapshot['close'] == snapshot['low_limit'] and trade_direction < 0:
            if logger:
                logger.warn('{} {}'.format(snapshot.name, 'close on low limit...untradable'))
            else:
                print('{} {}'.format(snapshot.name, 'close on low limit...untradable'))
            return False

    return True


def cal_opposite(d: Dict[str, float]):
    return {k: -v for k, v in d.items()}


def get_algo_trade_price(snapshot: pd.Series, algo: TradeAlgo):
    if algo == TradeAlgo.OPEN:
        return snapshot['adj_open']
    elif algo == TradeAlgo.CLOSE:
        return snapshot['adj_close']
    elif algo == TradeAlgo.HIGH:
        return snapshot['adj_high']
    elif algo == TradeAlgo.LOW:
        return snapshot['adj_low']
    elif algo == TradeAlgo.VWAP:
        vwap = snapshot['money'] / snapshot['volume']
        vwap *= snapshot['factor']
        return vwap
    elif algo == TradeAlgo.OHLC4:
        return (snapshot['adj_open'] + snapshot['adj_high'] + snapshot['adj_low'] + snapshot['adj_close']) / 4
    elif algo == TradeAlgo.OHC3:
        return (snapshot['adj_open'] + snapshot['adj_high'] + snapshot['adj_close']) / 3
    elif algo == TradeAlgo.OC2:
        return (snapshot['adj_open'] + snapshot['adj_close']) / 2
    else:
        raise NotImplementedError


def cal_equity(close: pd.Series, eod_holding_vol: Dict[str, float], last_day_holding_amount: Dict[str, float]):
    if len(eod_holding_vol) == 0:
        return 0
    cap = 0
    for ticker, vol in eod_holding_vol.items():
        try:
            cap += close[ticker] * vol
        except KeyError:
            cap += last_day_holding_amount[ticker]
    return cap


def cal_holding(close: pd.Series, eod_holding_vol: Dict[str, float], last_day_holding_amount: Dict[str, float]):
    res = {}
    for ticker, vol in eod_holding_vol.items():
        try:
            equity = close[ticker] * vol
        except KeyError:
            equity = last_day_holding_amount[ticker]
        res[ticker] = equity
    return res


def match_unfill_orders(date: pd.Timestamp,
                        unfilled_order: Dict[str, int],
                        current_market: pd.DataFrame,
                        long_algo: TradeAlgo,
                        short_algo: TradeAlgo,
                        long_fee_rate: float,
                        short_fee_rate: float,
                        cash_available: float,
                        holding_vol: Dict[str, int],
                        logger=None,
                        has_tradable_check=True
                        ):
    unmatched_order = {}
    cash_inflow = 0
    trading_fee = 0
    trade_vol = {}
    records = []
    unfilled_order = dict(sorted(unfilled_order.items(), key=lambda item: item[1]))
    long_count = 0
    short_count = 0
    for ticker, volume in unfilled_order.items():
        try:
            snapshot = current_market.loc[ticker]
            if (not has_tradable_check) or \
                    (has_tradable_check and
                     check_tradable(snapshot, volume, long_algo if volume > 0 else short_algo, logger)):
                # trade logic
                # + for long, - for short
                # + decrease cash, - add cash
                # decrease for fee
                # first update in term of vol, then in amount
                if volume > 0:
                    if long_algo == TradeAlgo.OPEN:
                        trade_date = date + pd.Timedelta(hours=9, minutes=30)
                    elif long_algo == TradeAlgo.CLOSE:
                        trade_date = date + pd.Timedelta(hours=15)
                    else:
                        trade_date = date

                    price = get_algo_trade_price(snapshot, long_algo)
                    amount = volume * price
                    if amount > cash_available:
                        logger.info('{}: insufficient funds. want to trade {} but only have cash {}'
                                    .format(ticker, amount, cash_available))
                        new_volume = int(cash_available / price)
                        amount = new_volume * price
                        logger.info('{}: adjust to new volume {} from {}'.format(ticker, new_volume, volume))
                        volume = new_volume

                    cash_inflow -= amount
                    cash_available -= amount
                    trade_vol[ticker] = volume
                    fee = amount * long_fee_rate
                    trading_fee += fee
                    records.append(
                        {'date': date, 'code': ticker, 'trade_price': price, 'volume': volume, 'amount': amount,
                         'fee': fee})
                    if logger:
                        if ticker not in holding_vol:
                            logger.info(
                                '{} long {} {} shares(${}) @{} fee:{} count:{}'.format(trade_date, ticker, volume,
                                                                                       amount,
                                                                                       price,
                                                                                       fee, long_count))
                        else:
                            logger.info(
                                '{} expand long {} {} shares(${}) @{} fee:{} count:{}'.format(trade_date, ticker,
                                                                                              volume,
                                                                                              amount,
                                                                                              price,
                                                                                              fee, long_count))
                        long_count += 1
                elif volume < 0:
                    if short_algo == TradeAlgo.OPEN:
                        trade_date = date + pd.Timedelta(hours=9, minutes=30)
                    elif short_algo == TradeAlgo.CLOSE:
                        trade_date = date + pd.Timedelta(hours=15)
                    else:
                        trade_date = date

                    if (holding_vol[ticker] + volume) < 0:
                        if logger:
                            logger.warn('Current version not support margin trade...You can only sell what you have. '
                                        'Adjust to {} from {} for {}'.format(volume, - holding_vol[ticker], ticker))
                        volume = - holding_vol[ticker]

                    price = get_algo_trade_price(snapshot, short_algo)
                    amount = volume * price
                    cash_inflow -= amount
                    cash_available -= amount
                    trade_vol[ticker] = volume
                    fee = - amount * short_fee_rate
                    trading_fee += fee
                    records.append(
                        {'date': date, 'code': ticker, 'trade_price': price, 'volume': volume, 'amount': amount,
                         'fee': fee})
                    if logger:
                        if ticker not in holding_vol:
                            logger.info(
                                '{} short {} {} shares(${}) @{} fee:{} count:{}'.format(trade_date, ticker, volume,
                                                                                        amount,
                                                                                        price,
                                                                                        fee, short_count))
                        else:
                            logger.info(
                                '{} sell {} {} shares(${}) @{} fee:{} count:{}'.format(trade_date, ticker, volume,
                                                                                       amount,
                                                                                       price,
                                                                                       fee, short_count))
                    short_count += 1
                else:
                    print('Trade 0 amount')
            else:
                unmatched_order[ticker] = volume
        except KeyError:
            unmatched_order[ticker] = volume
    return trade_vol, cash_inflow, trading_fee, unmatched_order, records


def crypto_match_unfill_orders(date: pd.Timestamp,
                               unfilled_order: Dict[str, float],
                               current_market: pd.DataFrame,
                               long_algo: TradeAlgo,
                               short_algo: TradeAlgo,
                               long_fee_rate: float,
                               short_fee_rate: float,
                               cash_available: float,
                               holding_vol: Dict[str, float],
                               logger=None,
                               ):
    unmatched_order = {}
    cash_inflow = 0
    trading_fee = 0
    trade_vol = {}
    records = []
    unfilled_order = dict(sorted(unfilled_order.items(), key=lambda item: item[1]))
    long_count = 0
    short_count = 0
    for ticker, volume in unfilled_order.items():
        try:
            snapshot = current_market.loc[ticker]
            # trade logic
            # + for long, - for short
            # + decrease cash, - add cash
            # decrease for fee
            # first update in term of vol, then in amount
            trade_date = date
            if volume > 0:
                price = get_algo_trade_price(snapshot, long_algo)
                amount = volume * price
                cash_inflow -= amount
                cash_available -= amount
                trade_vol[ticker] = volume
                fee = amount * long_fee_rate
                trading_fee += fee
                records.append(
                    {'date': date, 'code': ticker, 'trade_price': price, 'volume': volume, 'amount': amount,
                     'fee': fee})
                if logger:
                    if ticker not in holding_vol:
                        logger.info(
                            '{} long {} {} shares(${}) @{} fee:{} count:{}'.format(trade_date, ticker, volume,
                                                                                   amount,
                                                                                   price,
                                                                                   fee, long_count))
                    else:
                        logger.info(
                            '{} expand long {} {} shares(${}) @{} fee:{} count:{}'.format(trade_date, ticker,
                                                                                          volume,
                                                                                          amount,
                                                                                          price,
                                                                                          fee, long_count))
                    long_count += 1
            elif volume < 0:

                price = get_algo_trade_price(snapshot, short_algo)
                amount = volume * price
                cash_inflow -= amount
                cash_available -= amount
                trade_vol[ticker] = volume
                fee = - amount * short_fee_rate
                trading_fee += fee
                records.append(
                    {'date': date, 'code': ticker, 'trade_price': price, 'volume': volume, 'amount': amount,
                     'fee': fee})
                if logger:
                    if ticker not in holding_vol:
                        logger.info(
                            '{} short {} {} shares(${}) @{} fee:{} count:{}'.format(trade_date, ticker, volume,
                                                                                    amount,
                                                                                    price,
                                                                                    fee, short_count))
                    else:
                        logger.info(
                            '{} sell {} {} shares(${}) @{} fee:{} count:{}'.format(trade_date, ticker, volume,
                                                                                   amount,
                                                                                   price,
                                                                                   fee, short_count))
                short_count += 1
            else:
                print('Trade 0 amount')
        except KeyError:
            unmatched_order[ticker] = volume
    return trade_vol, cash_inflow, trading_fee, unmatched_order, records


def backtest(market_data: pd.DataFrame,
             strategy: AbstractStrategy,
             trading_dates: np.ndarray,
             start_date, end_date,
             *,
             initial_capital: float = 1_000_000,
             long_algo: TradeAlgo = TradeAlgo.OPEN,
             short_algo: TradeAlgo = TradeAlgo.OPEN,
             long_fee_rate: float = 0.0002,
             short_fee_rate: float = 0.0012,
             account: Optional[Account] = None,
             end_settlement: bool = True,
             securities_info: Optional[pd.DataFrame] = None,
             **strategy_paras
             ):
    """

    :param market_data:
    :param strategy:
    :param trading_dates:
    :param start_date:
    :param end_date:
    :param initial_capital:
    :param long_algo:
    :param short_algo:
    :param long_fee_rate:
    :param short_fee_rate:
    :param account:
    :param strategy_paras:
    :return:
    """
    offset = trading_dates_offsets(trading_dates, 'D')
    market_data['adj_close'] = market_data['adj_close'].groupby(level=1).fillna(method='ffill')
    start_date = pd.to_datetime(start_date).to_pydatetime().date()
    end_date = pd.to_datetime(end_date).to_pydatetime().date()
    trading_dates = trading_dates[trading_dates >= start_date]
    trading_dates = trading_dates[trading_dates <= end_date]
    trading_dates = pd.to_datetime(trading_dates)
    last_trading_dates = trading_dates[-1]

    log_filename = datetime.datetime.now().strftime('../logs/' + type(strategy).__name__ + "_%Y-%m-%d_%H_%M_%S.log")
    file_handler = logging.FileHandler(filename=log_filename)
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG,
                        format='|%(levelname)s|%(asctime)s|%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=handlers)
    logger = logging.getLogger(type(strategy).__name__)
    if account is None:
        account = Account([trading_dates[0] - offset], [initial_capital], [0], [{}], [{}], [0], [])
    strategy.on_init(account, trading_dates, offset, **strategy_paras)
    for trading_date in trading_dates:
        print(trading_date)
        # before trading session
        current_market = market_data.loc[trading_date]  # type: pd.DataFrame
        paused = current_market[current_market['paused'] == 1].index.to_list()
        strategy.before_market_open(trading_date, paused)
        # in the trading session
        # matching the unfilled order in the view of EOD
        unfilled_order = account.get_unfilled_orders()
        current_holding = account.get_holding()
        last_day_holding_equity = account.get_holding_equity()
        trade_vol, cash_inflow, trading_fee, unmatched_order, record = \
            match_unfill_orders(trading_date, unfilled_order, current_market,
                                long_algo, short_algo,
                                long_fee_rate, short_fee_rate,
                                account.get_cash(),
                                current_holding,
                                logger=logger)

        cash = account.get_cash() + cash_inflow - trading_fee
        eod_holding_vol = map_reduce_dict(current_holding, trade_vol)
        account.set_unfilled_orders(unmatched_order)
        if unmatched_order:
            logger.info('{} eod unmatched order: {}'.format(trading_date + pd.Timedelta(hours=15), unmatched_order))

        # end trade date. Force to convert to cash
        if end_settlement and trading_date == last_trading_dates:
            unfilled_order = {k: -v for k, v in account.get_holding().items()}
            trade_vol, cash_inflow, trading_fee1, unmatched_order, record1 = \
                match_unfill_orders(trading_date, unfilled_order, current_market,
                                    TradeAlgo.CLOSE, TradeAlgo.CLOSE,
                                    long_fee_rate, short_fee_rate, cash,
                                    eod_holding_vol,
                                    logger=logger)
            record.extend(record1)
            trading_fee += trading_fee1
            cash = cash + cash_inflow - trading_fee1
            eod_holding_vol = map_reduce_dict(eod_holding_vol, trade_vol)
            account.set_unfilled_orders(unmatched_order)

        # calculate eod holding
        equity = cal_equity(current_market['adj_close'], eod_holding_vol, last_day_holding_equity)
        holding_equity = cal_holding(current_market['adj_close'], eod_holding_vol, last_day_holding_equity)
        if securities_info is not None:
            # deal with delisting
            today_delist = securities_info[securities_info['end_date'] == trading_date].index.to_list()
            for c in today_delist:
                if c in eod_holding_vol:
                    equity_value = holding_equity[c]
                    cash += equity_value
                    equity -= equity_value
                    del holding_equity[c]
                    del eod_holding_vol[c]
                    try:
                        del unmatched_order[c]
                        account.set_unfilled_orders(unmatched_order)
                    except KeyError:
                        pass
        if logger:
            logger.info('{} EOD holding: {}'.format(trading_date + pd.Timedelta(hours=15), eod_holding_vol))
        logger.info('{} EOD cash: {}, equity: {}'.format(trading_date + pd.Timedelta(hours=15), cash, equity))
        account.add(trading_date, cash, equity, eod_holding_vol, holding_equity, trading_fee, record)
        strategy.on_eod_bar(trading_date, current_market[current_market['paused'] == 0])
    return account


def crypto_backtest(market_data: pd.DataFrame,
                    strategy: AbstractStrategy,
                    start_date, end_date,
                    *,
                    initial_capital: float = 100_000,
                    long_algo: TradeAlgo = TradeAlgo.OPEN,
                    short_algo: TradeAlgo = TradeAlgo.OPEN,
                    long_fee_rate: float = 0.0006,
                    short_fee_rate: float = 0.0006,
                    account: Optional[Account] = None,
                    end_settlement: bool = True,
                    logger=None,
                    **strategy_paras
                    ):
    """

    :param market_data:
    :param strategy:
    :param trading_dates:
    :param start_date:
    :param end_date:
    :param initial_capital:
    :param long_algo:
    :param short_algo:
    :param long_fee_rate:
    :param short_fee_rate:
    :param account:
    :param strategy_paras:
    :return:
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    market_data = market_data.loc[start_date: end_date]
    trading_dates = market_data.index.get_level_values(0).drop_duplicates()
    last_trading_dates = trading_dates[-1]
    if logger is None:
        log_filename = datetime.datetime.now().strftime('../logs/' + type(strategy).__name__ + "_%Y-%m-%d_%H_%M_%S.log")
        file_handler = logging.FileHandler(filename=log_filename)
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers = [file_handler, stdout_handler]
        logging.basicConfig(level=logging.DEBUG,
                            format='|%(levelname)s|%(asctime)s|%(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=handlers)
        logger = logging.getLogger(type(strategy).__name__)
    if account is None:
        account = Account([trading_dates[0] - pd.Timedelta(minutes=1)], [initial_capital], [0], [{}], [{}], [0], [])
    strategy.on_init(account, trading_dates, **strategy_paras)
    for trading_date in trading_dates:
        print(trading_date)
        # before trading session
        current_market = market_data.loc[trading_date]  # type: pd.DataFrame
        # in the trading session
        # matching the unfilled order in the view of EOD
        unfilled_order = account.get_unfilled_orders()
        current_holding = account.get_holding()
        last_day_holding_equity = account.get_holding_equity()
        trade_vol, cash_inflow, trading_fee, unmatched_order, record = \
            crypto_match_unfill_orders(trading_date, unfilled_order, current_market,
                                       long_algo, short_algo,
                                       long_fee_rate, short_fee_rate,
                                       account.get_cash(),
                                       current_holding,
                                       logger=logger)

        cash = account.get_cash() + cash_inflow - trading_fee
        eod_holding_vol = map_reduce_dict(current_holding, trade_vol)
        account.set_unfilled_orders(unmatched_order)
        if unmatched_order:
            logger.info('{} unmatched order: {}'.format(trading_date, unmatched_order))

        # end trade date. Force to convert to cash
        if end_settlement and trading_date == last_trading_dates:
            unfilled_order = {k: -v for k, v in account.get_holding().items()}
            trade_vol, cash_inflow, trading_fee1, unmatched_order, record1 = \
                crypto_match_unfill_orders(trading_date, unfilled_order, current_market,
                                           TradeAlgo.CLOSE, TradeAlgo.CLOSE,
                                           long_fee_rate, short_fee_rate, cash,
                                           eod_holding_vol,
                                           logger=logger)
            record.extend(record1)
            trading_fee += trading_fee1
            cash = cash + cash_inflow - trading_fee1
            eod_holding_vol = map_reduce_dict(eod_holding_vol, trade_vol)
            account.set_unfilled_orders(unmatched_order)

        # calculate eod holding
        equity = cal_equity(current_market['adj_close'], eod_holding_vol, last_day_holding_equity)
        holding_equity = cal_holding(current_market['adj_close'], eod_holding_vol, last_day_holding_equity)

        if logger:
            logger.info('{} EOD holding: {}'.format(trading_date, eod_holding_vol))
        logger.info('{} EOD cash: {}, equity: {}. Total asset: {}'.format(trading_date, cash, equity, cash + equity))
        account.add(trading_date, cash, equity, eod_holding_vol, holding_equity, trading_fee, record)
        strategy.on_eod_bar(trading_date, current_market)
    return account


if __name__ == '__main__':
    data_input_path = '../cfg/data_input.ini'
    local_cfg_path = '../cfg/factor_keeper_setting.ini'
    keeper2 = ZooKeeper(local_cfg_path)
    start_date = '2020-01-01'
    end_date = '2021-08-01'
    factor, _ = keeper2.get_factor_values('analyst', 'alsue0', start_date, end_date)
    nl = factor.groupby(level=0).nlargest(100).droplevel(0)
    selection = pd.Series(1, nl.index)
    selection = selection.groupby(level=0).transform(lambda x: x / len(x))
    td = get_trading_date(Market.AShares, '2015-01-01', config_path=data_input_path)
    market_data = get_bars(start_date=start_date, end_date=end_date, freq=Freq.D1,
                           cols=('open', 'high', 'low', 'close', 'volume', 'money', 'factor'),
                           add_limit=True,
                           adjust=True,
                           eod_time_adjust=False,
                           config_path=data_input_path)
    strategy = RebalanceBuyHoldStrategy(selection)
    account = backtest(market_data, strategy, td, start_date, end_date)
    df = account.get_full_trade_records()
    df.to_excel('records.xlsx')
