import functools
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Union, Tuple

import numpy as np
import pandas as pd
from quantstats.stats import calmar
from tqdm import tqdm

from data_management.dataIO import market_data, index_data
from data_management.dataIO.fundamental_data import get_stock_info
from data_management.dataIO.index_data import IndexTicker
from data_management.dataIO.market_data import Freq
from data_management.dataIO.trading_calendar import trading_dates_offsets
from data_management.pandas_utils.parallel import time_series_parallel_apply
from factor_testing.performance import sharpe_ratio
from labelling.utils import pre_adjust_price

tqdm.pandas()

"""
more rigorous method to label forward return
more objective to selection

"""


def forward_ret_labeler(daily_market: pd.DataFrame,
                        windows,
                        price_type='adj_open',
                        add_label_end=True
                        ) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """

    :param daily_market:
    :param windows:
    :return:
    """

    if 'close' in price_type:
        label = daily_market[price_type].unstack().pct_change(windows).shift(-windows)
        label_end = pd.Series(label.index, index=label.index, name='label_end').shift(-windows)
        label = label.stack().sort_index()
        label_end = label.to_frame().join(label_end)['label_end']
    elif 'open' in price_type:
        tradable = pd.Series(np.where(
            (daily_market['open'] == daily_market['low_limit']) | (daily_market['open'] == daily_market['high_limit']),
            np.nan, 1), index=daily_market.index)
        tradable = tradable.unstack()
        label = daily_market[price_type].unstack().pct_change(windows).shift(-windows)
        label = (label * tradable).shift(-1)
        label_end = pd.Series(label.index, index=label.index, name='label_end').shift(-windows - 1)
        label = label.stack().sort_index()
        label_end = label.to_frame().join(label_end)['label_end']
    else:
        raise NotImplementedError
    label = label.dropna()
    label = label.map(lambda x: '%.4f' % x).astype(float)
    label.name = 'forward_{}_D_{}_ret'.format(windows, price_type)
    if add_label_end:
        label_end.name = label.name + '_label_end'
        return label, label_end
    else:
        return label


def forward_demean_ret_labeler(daily_market: pd.DataFrame, windows,
                               price_type='adj_open', clip_quantile=None):
    """

    :param daily_market:
    :param windows:
    :param trading_date:
    :param resample:
    :return:
    """

    label, end = forward_ret_labeler(daily_market, windows, price_type, True)
    label = label.groupby(level=0).transform(lambda x: x - x.mean())
    label = label.map(lambda x: '%.4f' % x).astype(float)
    label.name = 'forward_{}_demean_ret'.format(windows)
    if clip_quantile:
        lower = label.quantile(clip_quantile)
        upper = label.quantile(1 - clip_quantile)
        label = label.clip(lower, upper)
        label.name = 'forward_{}_demean_{}_ret'.format(windows, clip_quantile)

    return label, end


def forward_excess_ret_labler(daily_market: pd.DataFrame,
                              benchmark: pd.DataFrame,
                              windows: int,
                              price_type='adj_open',
                              add_label_end=True) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """

    Parameters
    ----------
    daily_market
    benchmark
    windows
    price_type
    add_label_end

    Returns
    -------

    """
    benchmark_name = benchmark.code.iloc[0] if 'code' in benchmark else 'benchmark'
    # benchmark_name = benchmark_name.replace('.', '')
    try:
        benchmark_name = IndexTicker(benchmark_name).name
    except:
        pass

    benchmark = benchmark[price_type.replace('adj_', '')].pct_change(windows).shift(-windows)
    if 'open' in price_type:
        benchmark = benchmark.shift(-1)
    benchmark.name = 'forward_{}_D_{}_ret'.format(windows, benchmark_name)

    label, label_end = forward_ret_labeler(daily_market, windows, price_type, True)
    label = label.to_frame().join(benchmark)

    label = label['forward_{}_D_{}_ret'.format(windows, price_type)] - \
            label['forward_{}_D_{}_ret'.format(windows, benchmark_name)]
    label = label.map(lambda x: '%.4f' % x).astype(float)
    label.name = 'forward_{}_D_{}_excess_{}_ret'.format(windows, price_type, benchmark_name)
    if add_label_end:
        label_end.name = label.name + '_label_end'
        return label, label_end
    else:
        return label


def forward_excess_industry_ret_labler(daily_market: pd.DataFrame,
                                       benchmark: pd.DataFrame,
                                       windows: int,
                                       price_type='adj_open',
                                       add_label_end=True) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    pass


def sharpe_labeler(daily_market: pd.DataFrame, windows, trading_date, resample, adjust=True) -> pd.Series:
    offsets = trading_dates_offsets(trading_date, resample)

    def sharpe(series):
        returns = series.pct_change(1).dropna()
        if resample == 'M' and len(returns) <= 10:
            return np.nan
        elif resample == 'D' and len(returns) <= int(0.4 * windows):
            return np.nan
        else:
            return sharpe_ratio(returns)

    if adjust:
        daily_market = pre_adjust_price(daily_market)

    label = daily_market['close'].groupby(level=1). \
        resample(windows * offsets, level=0, label='left', closed='right').agg(sharpe)
    label = label.swaplevel(1, 0).sort_index()
    label = label.dropna()
    label.name = 'forward_{}_{}_sharpe'.format(windows, resample)
    return label


def calmar_labeler(daily_market: pd.DataFrame, windows, trading_date, resample, adjust=True) -> pd.Series:
    offsets = trading_dates_offsets(trading_date, resample)

    def calmar_(series):
        returns = series.droplevel(1).pct_change(1).dropna()
        if resample == 'M' and len(returns) <= 10:
            return np.nan
        elif resample == 'D' and len(returns) <= int(0.4 * windows):
            return np.nan
        else:
            return calmar(returns)

    if adjust:
        daily_market = pre_adjust_price(daily_market)

    label = daily_market['close'].groupby(level=1). \
        resample(windows * offsets, level=0, label='left', closed='right').agg(calmar_)
    label = label.swaplevel(1, 0).sort_index()
    label = label.dropna()
    label.name = 'forward_{}_{}_calmar'.format(windows, resample)
    return label


def triple_barrier_reg_labeler(daily_market: pd.DataFrame,
                               windows, trading_date=None, resample='M', up=0.2, down=0.05) -> pd.Series:
    offsets = trading_dates_offsets(trading_date, resample)

    def tbl(series: pd.Series, up, down):
        arr = series.values

        if resample == 'M' and len(arr) <= 10:
            return np.nan
        elif resample == 'D' and len(arr) <= int(0.4 * windows):
            return np.nan
        else:
            start = arr[0]
            max_p = arr.max()
            min_p = arr.min()
            if max_p >= start * (1 + up) and min_p <= start * (1 - down):
                argmax = arr.argmax()
                argmin = arr.argmin()
                if argmax <= argmin:
                    return up
                else:
                    return -down
            elif max_p >= start * (1 + up):
                return up
            elif min_p <= start * (1 - down):
                return -down
            else:
                return (arr[-1] - arr[0]) / arr[0]

    label = daily_market['close'].groupby(level=1). \
        resample(windows * offsets, level=0, label='left', closed='right').agg(lambda x: tbl(x, up, down))
    label = label.swaplevel(1, 0).sort_index()
    label = label.dropna()
    label.name = 'forward_{}_{}_tbl_{}_{}_ret'.format(windows, resample, up, down)
    return label


def forward_path_dependent_ret_labeler(daily_market: pd.DataFrame,
                                       windows,
                                       trading_date,
                                       path_dependent_func:
                                       Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]],
                                       resample='M',
                                       add_label_end=True,
                                       **path_dependent_func_kwargs
                                       ) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """

    :param daily_market:
    :param windows:
    :param trading_date:
    :param path_dependent_func: 
    :param resample:
    :return:
    """

    offsets = trading_dates_offsets(trading_date, resample)

    grouper = [daily_market.index.get_level_values(1),
               pd.Grouper(freq=windows * offsets, level=0, closed='right', label='left')]

    label = daily_market.groupby(grouper).progress_apply(path_dependent_func, path_dependent_func_kwargs)
    label = label.swaplevel(1, 0).sort_index()
    label = label.map(lambda x: '%.4f' % x).astype(float)
    label.name = 'forward_{}_{}_{}_ret'.format(windows, resample, path_dependent_func.__name__)
    if add_label_end:
        end_datetime = daily_market.groupby(grouper).apply(end_label_datetime).swaplevel(1, 0).sort_index()
        end_datetime.name = label.name + '_label_end'
        return label, end_datetime
    else:
        return label


def forward_rolling_path_dependent_ret_labeler(daily_market: pd.DataFrame,
                                               windows,
                                               trading_date,
                                               path_dependent_func:
                                               Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]],
                                               add_label_end=True,
                                               **path_dependent_func_kwargs
                                               ) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """

    :param daily_market:
    :param windows:
    :param trading_date:
    :param path_dependent_func: 
    :return:
    """
    offsets = trading_dates_offsets(trading_date, 'D')

    fun = functools.partial(rolling_windows_path, windows=windows,
                            offsets=offsets,
                            path_dependent_func=path_dependent_func,
                            path_dependent_func_kwargs=path_dependent_func_kwargs)

    label_end = time_series_parallel_apply(daily_market, fun, chuncksize=50)
    label_end = label_end.sort_index()
    label = label_end['label']
    label = label.replace(pd.NaT, np.nan).map(lambda x: '%.4f' % x).astype(float)
    label.name = 'forward_rolling_{}_D_{}_ret'.format(windows, path_dependent_func.__name__)
    if add_label_end:
        end_datetime = label_end['label_end']
        end_datetime.name = label.name + '_label_end'
        return label, end_datetime
    else:
        return label


def forward_min_rolling_path_dependent_ret_labeler(data_config_path,
                                                   bar_windows,
                                                   freq: Freq,
                                                   trading_date,
                                                   path_dependent_func:
                                                   Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]],
                                                   start_date='2013-01-01',
                                                   add_label_end=True,
                                                   **path_dependent_func_kwargs):
    offsets = trading_dates_offsets(trading_date, 'D')
    stock_info = get_stock_info(config_path=data_config_path)
    stock_info = stock_info[stock_info.end_date > pd.to_datetime(start_date)]
    func = functools.partial(get_data_and_cal_result, start_date=start_date,
                             data_config_path=data_config_path, freq=freq, windows=bar_windows,
                             path_dependent_func=path_dependent_func, offsets=offsets)
    tickers = stock_info.index.to_list()
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(func, tickers, chunksize=5), total=len(tickers)))

    label_end = pd.concat(results).sort_index()
    label = label_end['label']
    label = label.replace(pd.NaT, np.nan).map(lambda x: '%.4f' % x).astype(float)
    label.name = 'forward_min_rolling_{}_{}_{}'.format(bar_windows, freq.value, path_dependent_func.__name__)
    if add_label_end:
        end_datetime = label_end['label_end']
        end_datetime.name = label.name + '_label_end'
        return label, end_datetime
    else:
        return label


def get_data_and_cal_result(ticker, start_date, data_config_path, freq, windows,
                            path_dependent_func, offsets=None, **path_dependent_func_kwargs):
    bars = market_data.get_bars(ticker,
                                start_date=start_date,
                                config_path=data_config_path, freq=freq, verbose=0)
    if bars.empty:
        return pd.DataFrame()

    label = rolling_windows_path(bars, windows, path_dependent_func, **path_dependent_func_kwargs)
    if len(label) == 0:
        return pd.DataFrame()

    if offsets is not None:
        label = label.resample(offsets, level=0).first()
    label['code'] = ticker
    label = label.set_index('code', append=True)
    label.index.names = ['date', 'code']
    return label


def rolling_windows_path(df: pd.DataFrame, windows,
                         path_dependent_func, offsets=None,
                         **path_dependent_func_kwargs):
    res = {}
    df_small = df[['open', 'high', 'low', 'close', 'money', 'factor', 'high_limit', 'low_limit',
                   'paused']]
    array = df_small.values
    idx_array = df_small.index.get_level_values(0).values
    ticker = df.index.get_level_values(1)[0]
    for i in range(windows, len(array)):
        obs = array[i - windows: i]
        r = path_dependent_func(obs, **path_dependent_func_kwargs)
        end = end_label_datetime(idx_array[i - windows: i])
        label_start = pd.to_datetime(idx_array[i - windows])
        if offsets is not None:
            res[(label_start - offsets, ticker)] = {'label': r, 'label_end': end}
        else:
            res[(label_start, ticker)] = {'label': r, 'label_end': end}

    res = pd.DataFrame(res).T
    return res


def buy_and_hold(x: pd.DataFrame):
    # MultiIndex DataFrame
    open_px = x['open'].iloc[0]
    entry_px = open_px
    exit_px = x['close'].iloc[-1]
    return (exit_px - entry_px) / entry_px
    # elif isinstance(x, np.ndarray):
    #     #  df[['open', 'high', 'low', 'close', 'money', 'factor', 'high_limit', 'low_limit',
    #     #        'paused']]
    #     open_px = x[0, 0]
    #     close_px = x[-1, 3]
    #     entry_px = open_px * x[0, 5]
    #     exit_px = close_px * x[-1, 5]
    #     return (exit_px - entry_px) / entry_px


def hold_eliminate_paused_and_limit(x: pd.DataFrame):
    # MultiIndex DataFrame
    if isinstance(x, pd.DataFrame):
        open_px = x['open'].iloc[0]
        if x['paused'].iloc[0] == 1:
            return np.nan
        elif (open_px == x['high_limit'].iloc[0]) or (open_px == x['low_limit'].iloc[0]):
            return np.nan
        entry_px = open_px * x['factor'].iloc[0]
        exit_px = x['close'].iloc[-1] * x['factor'].iloc[-1]
        return (exit_px - entry_px) / entry_px
    elif isinstance(x, np.ndarray):
        #  df[['open', 'high', 'low', 'close', 'money', 'factor', 'high_limit', 'low_limit',
        #        'paused']]
        open_px = x[0, 0]
        paused = x[0, -1]
        high_limit = x[0, 6]
        low_limit = x[0, 7]
        close_px = x[-1, 3]
        if paused == 1:
            return np.nan
        elif (open_px == high_limit) or (open_px == low_limit):
            return np.nan
        entry_px = open_px * x[0, 5]
        exit_px = close_px * x[-1, 5]
        return (exit_px - entry_px) / entry_px


def hold_eliminate_paused_and_limit_sharpe(x):
    if isinstance(x, pd.DataFrame):
        raise NotImplementedError
    elif isinstance(x, np.ndarray):
        #  df[['open', 'high', 'low', 'close', 'money', 'factor', 'high_limit', 'low_limit',
        #        'paused']]
        open_px = x[:, 0]
        paused = x[:, -1]
        high_limit = x[:, 6]
        low_limit = x[:, 7]
        close_px = x[:, 3]
        if paused[0] == 1:
            return np.nan
        elif (open_px[0] == high_limit[0]) or (open_px[0] == low_limit[0]):
            return np.nan
        ret = (close_px[1:] - close_px[:-1]) / close_px[:-1]
        mask = np.where(
            (close_px[1:] == close_px[:-1]) * ((close_px[1:] == high_limit[1:]) | (close_px[1:] == low_limit[1:])),
            np.nan, 1)
        ret = ret * mask
        non_annual_sharpe = np.nanmean(ret) / (np.nanstd(ret) + 0.00001)
        return non_annual_sharpe


def buy_hold_ret_path(x: pd.DataFrame):
    return x['close'].pct_change().droplevel(1)


def end_label_datetime(x: pd.DataFrame):
    if isinstance(x, pd.DataFrame):
        return x.index.get_level_values(0)[-1]
    elif isinstance(x, np.ndarray):
        return x[-1]


def forward_atr_adjusted_ret():
    pass


if __name__ == '__main__':
    config_path = '../cfg/data_input.ini'
    arctic_config_path = '../cfg/data_input_arctic.ini'
    market_data = market_data.get_bars(start_date='2014-01-01', freq=Freq.D1, config_path=config_path,
                                       adjust=True,
                                       eod_time_adjust=False)
    benchmark_data = index_data.get_bars(IndexTicker.csi300, config_path=config_path, eod_time_adjust=False)
    # trading_list = get_trading_date(Market.AShares, config_path=config_path)
    # trading_dates = get_end_of_month_trading_date(trading_list)
    # trading_dates = trading_dates[trading_dates < pd.to_datetime('2020-10-01')]
    # l = forward_path_dependent_ret_labeler(market_data, 1, trading_list, hold_eliminate_paused_and_limit, 'M')
    # guardian = LabelGuardian('../cfg/label_guardian_setting.ini')
    # guardian.save_label_values('regression', l, False)
    # l, end = forward_rolling_path_dependent_ret_labeler(market_data, 5, trading_list, hold_eliminate_paused_and_limit)
    # forward_min_rolling_path_dependent_ret_labeler(arctic_config_path, 20, Freq.m60, trading_list,
    #                                                hold_eliminate_paused_and_limit, )
    # forward_min_rolling_path_dependent_ret_labeler(arctic_config_path, 20, Freq.m60, trading_list,
    #                                                hold_eliminate_paused_and_limit_sharpe)

    # l = forward_path_dependent_ret_labeler(market_data, 1, trading_list, hold_return_eliminate_paused_and_limit, 'W-3')

    # forward_path_dependent_ret_labeler(market_data, 1, trading_list, buy_hold_ret_path, 'M')
    # l, end = forward_ret_labeler(market_data, 5)
    forward_excess_ret_labler(market_data, benchmark_data, 5)
