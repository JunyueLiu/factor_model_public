import inspect
from enum import Enum

import numba
import numpy as np
import talib

from data_management.cache_janitor.cache import tmp_cache
from data_management.dataIO import market_data, binance
from data_management.dataIO.binance import Freq
from data_management.dataIO.component_data import get_industry_component, IndustryCategory, get_bars
from data_management.dataIO.trade_data import TradeTable, get_trade
from data_management.pandas_utils.cache import panel_df_concat
from factor_zoo.industry import industry_category


class PriceVar(Enum):
    adj_open = 'adj_open'
    adj_high = 'adj_high'
    adj_low = 'adj_low'
    adj_close = 'adj_close'
    adj_ohlc4 = 'adj_ohlc4'
    adj_ohl3 = 'adj_ohl3'
    adj_ohc3 = 'adj_ohc3'
    adj_hl2 = 'adj_hl2'
    adj_oc2 = 'adj_oc2'

    @staticmethod
    def map(i: int):
        n = i % PriceVar.get_size()
        return PriceVar(PriceVar._member_names_[n])

    @staticmethod
    def get_size():
        return len(PriceVar._member_names_)

    @staticmethod
    def get_i(price_var):
        return PriceVar._member_names_.index(price_var.value)


class IndClass(Enum):
    sector = 'sw1'
    industries = 'sw2'
    subindustries = 'sw3'

    @staticmethod
    def map(i: int):
        n = i % IndClass.get_size()
        return IndClass.__members__[IndClass._member_names_[n]]

    @staticmethod
    def get_size():
        return len(IndClass._member_names_)

    @staticmethod
    def get_i(ind_class):
        return IndClass._member_names_.index(ind_class.name)


def get_paras_string(args: inspect.ArgInfo):
    s = '&'.join(['{}={}'.format(k, args.locals.get(k)) for k in args.args
                  if isinstance(args.locals.get(k), (float, int, str, PriceVar, IndClass))])
    if len(s) > 80:
        s = '&'.join(['{}'.format(args.locals.get(k)) for k in args.args
                      if isinstance(args.locals.get(k), (float, int, str, PriceVar, IndClass))])
    return s


@tmp_cache
def get_alpha101_data_input(data_config_path: str, start_date=None, end_date=None):
    daily_data = market_data.get_bars(
        cols=('open', 'high', 'low', 'close', 'volume', 'money', 'factor'),
        adjust=True, eod_time_adjust=False, add_limit=False, start_date=start_date,
        config_path=data_config_path)
    daily_data['adj_ohlc4'] = (daily_data['adj_open'] + daily_data['adj_high'] + daily_data['adj_low'] + daily_data[
        'adj_close']) / 4
    daily_data['adj_ohl3'] = (daily_data['adj_open'] + daily_data['adj_high'] + daily_data['adj_low']) / 3
    daily_data['adj_ohc3'] = (daily_data['adj_open'] + daily_data['adj_high'] + daily_data['adj_close']) / 3
    daily_data['adj_hl2'] = (daily_data['adj_high'] + daily_data['adj_low']) / 2
    daily_data['adj_oc2'] = (daily_data['adj_open'] + daily_data['adj_close']) / 4
    daily_basic = get_trade(TradeTable.daily_basic, cols=['total_mv', 'circ_mv'],
                            start_date=start_date, end_date=end_date,
                            config_path=data_config_path
                            )
    trading_dates = daily_data.index.get_level_values(0).drop_duplicates().tolist()
    sw1 = get_industry_component(IndustryCategory.sw_l1, date=trading_dates,
                                 config_path=data_config_path
                                 )
    sw2 = get_industry_component(IndustryCategory.sw_l2, date=trading_dates,
                                 config_path=data_config_path
                                 )
    sw3 = get_industry_component(IndustryCategory.sw_l3, date=trading_dates,
                                 config_path=data_config_path
                                 )
    sw1_ind = industry_category(sw1, 'sw1')
    sw2_ind = industry_category(sw2, 'sw2')
    sw3_ind = industry_category(sw3, 'sw3')
    sw3_ind = sw3_ind[~sw3_ind.index.duplicated()]
    data = panel_df_concat([daily_data, daily_basic, sw1_ind.to_frame(), sw2_ind.to_frame(), sw3_ind.to_frame()])
    data = data.dropna().sort_index()
    return data


def get_alpha101_industry_data_input(data_config_path: str, start_date=None, end_date=None):
    daily_data = get_bars(start_date=start_date, end_date=end_date, config_path=data_config_path)
    daily_data['adj_ohlc4'] = (daily_data['adj_open'] + daily_data['adj_high'] + daily_data['adj_low'] + daily_data[
        'adj_close']) / 4
    daily_data['adj_ohl3'] = (daily_data['adj_open'] + daily_data['adj_high'] + daily_data['adj_low']) / 3
    daily_data['adj_ohc3'] = (daily_data['adj_open'] + daily_data['adj_high'] + daily_data['adj_close']) / 3
    daily_data['adj_hl2'] = (daily_data['adj_high'] + daily_data['adj_low']) / 2
    daily_data['adj_oc2'] = (daily_data['adj_open'] + daily_data['adj_close']) / 4
    return daily_data


def get_crypto_alpha101_data_input(start_date=None, end_date=None, freq=Freq.h4):
    bars = binance.get_um_bars(start_date=start_date, end_date=end_date, freq=freq)
    bars['adj_ohlc4'] = (bars['adj_open'] + bars['adj_high'] + bars['adj_low'] + bars[
        'adj_close']) / 4
    bars['adj_ohl3'] = (bars['adj_open'] + bars['adj_high'] + bars['adj_low']) / 3
    bars['adj_ohc3'] = (bars['adj_open'] + bars['adj_high'] + bars['adj_close']) / 3
    bars['adj_hl2'] = (bars['adj_high'] + bars['adj_low']) / 2
    bars['adj_oc2'] = (bars['adj_open'] + bars['adj_close']) / 4
    return bars


@numba.njit
def nans(arr, dtype=np.float64) -> np.ndarray:
    a = np.empty_like(arr, dtype)
    a.fill(np.nan)
    return a


@numba.njit
def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def talib_corr(one_d_array_x: np.ndarray, one_d_array_y: np.ndarray, d: int):
    """
    The implementation is not shown in open source version
    """
    raise NotImplementedError


def add_cap():
    pass


if __name__ == '__main__':
    v = PriceVar.get_i(PriceVar.adj_low)
