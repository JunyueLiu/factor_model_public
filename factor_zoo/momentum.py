from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arctic import Arctic
from sklearn.linear_model import LinearRegression
from statsmodels.regression.rolling import RollingOLS
from talib import abstract
from tqdm import tqdm

from data_management.pandas_utils.parallel import time_series_parallel_apply
from factor_zoo.factor_operator.alpha101_operator import rank, ts_rank
from factor_zoo.utils import rolling_regression_alpha
from technical_analysis.momentum import AROON, CCI
from technical_analysis.overlap import EMA, SMA, KAMA, MAMA, T3
from technical_analysis.statistic_function import LINEARREG


# cross sectional momentum

def mom(daily_market: pd.DataFrame, period=252, skip_period=21) -> pd.Series:
    close = daily_market['adj_close'].unstack()  # type: pd.DataFrame
    close_start = close.shift(period)
    close_end = close.shift(skip_period)
    factor = (close_end / close_start - 1).stack()
    factor.name = 'mom_{}_{}'.format(period, skip_period)
    return factor


def long_side_momentum(daily_market: pd.DataFrame, d=160, percentage=0.7):
    data = daily_market[['adj_high', 'adj_low', 'adj_close']]
    data['osc'] = (data['adj_high'] / data['adj_low']) - 1
    data['ret'] = data.groupby(level=1)['adj_close'].pct_change()
    num = int(d * percentage)

    def func(df):
        arr = np.empty(len(df))
        arr[:] = np.NaN
        osc = df['osc'].values
        ret = df['ret'].values
        for i in range(0, len(df) - d + 1):
            index = np.argsort(osc[i: i + d])
            arr[i + d - 1] = np.sum(ret[i: i + d][index[:num]])
        return pd.Series(arr, df.index)

    tqdm.pandas(desc='long_side_momentum')
    f = data.groupby(level=1).progress_apply(func)
    if f.index.nlevels == 3:
        f = f.droplevel(0).sort_index()
    else:
        f = f.swaplevel(0, 1).sort_index()
    f.name = 'long_side_momentum_{}_{}'.format(d, percentage)
    return f


def long_side_momentum_v2(daily_market: pd.DataFrame, d=160, d2=20, percentage=0.7):
    data = daily_market[['adj_high', 'adj_low', 'adj_close']]
    data['osc'] = (data['adj_high'] / data['adj_low']) - 1
    data['ret'] = data.groupby(level=1)['adj_close'].pct_change()
    num = int(d * percentage)

    def func(df):
        arr = np.empty(len(df))
        arr[:] = np.NaN
        osc = df['osc'].values
        ret = df['ret'].values
        for i in range(0, len(df) - d + 1):
            index = np.argsort(osc[i + d2: i + d])
            arr[i + d - 1] = np.sum(ret[i + d2: i + d][index[:num]])
        return pd.Series(arr, df.index)

    tqdm.pandas(desc='long_side_momentum_v2')
    f = data.groupby(level=1).progress_apply(func)
    if f.index.nlevels == 3:
        f = f.droplevel(0).sort_index()
    else:
        f = f.swaplevel(0, 1).sort_index()
    f.name = 'long_side_momentum_v2_{}_{}_{}'.format(d, d2, percentage)
    return f


def long_side_momentum_v3(daily_market: pd.DataFrame, d=160, d2=20, percentage=0.7):
    data = daily_market[['adj_high', 'adj_low', 'adj_close']]
    data['osc'] = (data['adj_high'] / data['adj_low']) - 1
    data['ret'] = data.groupby(level=1)['adj_close'].pct_change()
    num = int(d * percentage)

    def func(df):
        arr = np.empty(len(df))
        arr[:] = np.NaN
        osc = df['osc'].values
        ret = df['ret'].values
        for i in range(0, len(df) - d + 1):
            index = np.argsort(osc[i: i + d])
            arr[i + d - 1] = np.sum(ret[i: i + d][index[:num]]) - np.sum(ret[i: i + d][-d2:])
        return pd.Series(arr, df.index)

    tqdm.pandas(desc='long_side_momentum_v3')
    f = data.groupby(level=1).progress_apply(func)
    if f.index.nlevels == 3:
        f = f.droplevel(0).sort_index()
    else:
        f = f.swaplevel(0, 1).sort_index()
    f.name = 'long_side_momentum_v3_{}_{}_{}'.format(d, d2, percentage)
    return f


def long_side_momentum_v4(daily_market: pd.DataFrame, *, d: int = 160, d1: int = 20, percentage: float = 0.7,
                          b: float = 1) -> pd.Series:
    data = daily_market[['adj_high', 'adj_low', 'adj_close']]
    data['osc'] = (data['adj_high'] / data['adj_low']) - 1
    data['ret'] = data.groupby(level=1)['adj_close'].pct_change()
    num = int(d * percentage)

    def func(df):
        arr = np.empty(len(df))
        arr[:] = np.NaN
        osc = df['osc'].values
        ret = df['ret'].values
        for i in range(0, len(df) - d + 1):
            index = np.argsort(osc[i: i + d])
            arr[i + d - 1] = np.mean(ret[i: i + d][index[:num]]) + b * np.mean(ret[i: i + d][-d1:])
        return pd.Series(arr, df.index)

    tqdm.pandas(desc='long_side_momentum_v4')
    f = data.groupby(level=1).progress_apply(func)
    if f.index.nlevels == 3:
        f = f.droplevel(0).sort_index()
    else:
        f = f.swaplevel(0, 1).sort_index()
    f.name = 'long_side_momentum_v4_{}_{}_{}_{}'.format(d, d1, percentage, b)
    return f


def _read_cal_long_mom_min(i: str, store: Arctic, d, percentage) -> pd.Series:
    df = store['15m'].read(i).data

    data = df[['adj_high', 'adj_low', 'adj_close']]
    data['osc'] = (data['adj_high'] / data['adj_low']) - 1
    data['ret'] = data['adj_close'].pct_change()

    num = int(d * percentage)

    def func(df):
        arr = np.empty(len(df))
        arr[:] = np.NaN
        osc = df['osc'].values
        ret = df['ret'].values
        for i in range(0, len(df) - d + 1):
            index = np.argsort(osc[i: i + d])
            arr[i + d - 1] = np.mean(ret[i: i + d][index[:num]])
        return pd.Series(arr, df.index)

    return func(data)


def long_side_momentum_15min(store: Arctic, d=160, percentage=0.7):
    lib = store['15m']
    instruments = lib.list_symbols()

    # data =
    # data['osc'] = (data['adj_high'] / data['adj_low']) - 1
    # data['ret'] = data.groupby(level=1)['adj_close'].pct_change()
    func = partial(_read_cal_long_mom_min, store=store)
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(func, instruments, chunksize=1),
                            total=len(instruments)))


# ----- time series momentum -------

def arron_up(daily_market: pd.DataFrame, period=25) -> pd.Series:
    """
    Aroon(上升)=[(计算期天数-最高价后的天数)/计算期天数]*100
    :param daily_market:
    :param period:
    :return:
    """
    tqdm.pandas(desc='arron_up')
    factor = daily_market.groupby(level=1).progress_apply(lambda x: AROON(x, period, ['adj_high', 'adj_low']))
    factor = factor['aroonup'] / 100
    factor.name = 'arron_up_{}'.format(period)
    return factor


def arron_down(daily_market: pd.DataFrame, period=25) -> pd.Series:
    """
    Aroon(下降)=[(计算期天数-最低价后的天数)/计算期天数]*100
    :param daily_market:
    :param period:
    :return:
    """
    tqdm.pandas(desc='arron_down')
    factor = daily_market.groupby(level=1).progress_apply(lambda x: AROON(x, period, ['adj_high', 'adj_low']))
    factor = factor['aroondown'] / 100
    factor.name = 'arron_down_{}'.format(period)
    return factor


def bull_power(daily_market: pd.DataFrame, period=13) -> pd.Series:
    """
    (最高价-EMA(close,13)) / close

    :param daily_market:
    :return:
    """
    tqdm.pandas(desc='bull_power')
    ema = daily_market.groupby(level=1).progress_apply(lambda x: EMA(x, period, 'adj_close')) \
        .droplevel(0).sort_index()
    factor = (daily_market['adj_high'] - ema) / daily_market['adj_close']
    factor.name = 'bull_power_{}'.format(13)
    return factor


def bear_power(daily_market: pd.DataFrame, period=13) -> pd.Series:
    """
    bear_power	空头力道	(最低价-EMA(close,13)) / close
    :param daily_market:
    :param period:
    :return:
    """
    tqdm.pandas(desc='bear_power')
    ema = daily_market.groupby(level=1).progress_apply(lambda x: EMA(x, period, 'adj_close')).droplevel(0).sort_index()
    factor = (daily_market['adj_low'] - ema) / daily_market['adj_close']
    factor.name = 'bear_power_{}'.format(13)
    return factor


def bias_sma(daily_market: pd.DataFrame, period=5) -> pd.Series:
    """

    :param daily_market:
    :param period:
    :return:
    """
    tqdm.pandas(desc='bias_sma')
    ma = daily_market.groupby(level=1).progress_apply(lambda x: SMA(x, period, 'adj_close')) \
        .droplevel(0).sort_index()
    factor = (daily_market['adj_close'] - ma) * 100 / ma
    factor.name = 'bias_sma_{}'.format(period)
    return factor


def bias_ema(daily_market: pd.DataFrame, *, period: int = 5) -> pd.Series:
    """

    :param daily_market:
    :param period:
    :return:
    """
    tqdm.pandas(desc='bias_ema')
    ma = daily_market.groupby(level=1).progress_apply(lambda x: EMA(x, period, 'adj_close')) \
        .droplevel(0).sort_index()
    factor = (daily_market['adj_close'] - ma) / ma
    factor.name = 'bias_ema_{}'.format(period)
    return factor


def bias_kama(daily_market: pd.DataFrame, period=5) -> pd.Series:
    """

    :param daily_market:
    :param period:
    :return:
    """
    tqdm.pandas(desc='bias_kama')
    ma = daily_market.groupby(level=1).progress_apply(lambda x: KAMA(x, period, 'adj_close')) \
        .droplevel(0).sort_index()
    factor = (daily_market['adj_close'] - ma) / ma
    factor.name = 'bias_kama_{}'.format(period)
    return factor


def bias_mama(daily_market: pd.DataFrame, fastlimit=0.5,
              slowlimit=0.05
              ) -> pd.Series:
    """

    :param daily_market:
    :param fastlimit:
    :param slowlimit:
    :return:
    """
    tqdm.pandas(desc='bias_mama')
    ma = daily_market.groupby(level=1) \
        .apply(lambda x: MAMA(x, fastlimit, slowlimit, 'adj_close')).droplevel(0).sort_index()
    factor = (daily_market['adj_close'] - ma) / ma
    factor.name = 'bias_mama_{}_{}'.format(fastlimit, fastlimit)
    return factor


def bias_t3(daily_market: pd.DataFrame, period=5, vfactor=0.7) -> pd.Series:
    """

    :param daily_market:
    :param period:
    :return:
    """
    tqdm.pandas(desc='bias_t3')
    ma = daily_market.groupby(level=1).progress_apply(lambda x: T3(x, period, vfactor, 'adj_close')) \
        .droplevel(0).sort_index()
    factor = (daily_market['adj_close'] - ma) / ma
    factor.name = 'bias_t3_{}'.format(period)
    return factor


def cci(daily_market: pd.DataFrame, period=10) -> pd.Series:
    """
    :param daily_market:
    :return:
    """
    factor = daily_market.groupby(level=1).apply(lambda x: CCI(x, period, ['adj_high', 'adj_low', 'adj_close'])) \
        .droplevel(0).sort_index()
    factor.name = 'cci_{}'.format(period)
    return factor


def _trend_regression(data, y_col, window):
    if len(data) < window:
        return pd.Series(np.nan, data.index)
    endog = data[y_col].values
    x = np.arange(0, len(data))
    exog = sm.add_constant(x)
    rols = RollingOLS(endog, exog, window=window)
    rres = rols.fit(cov_type='HCCM')
    return pd.Series(rres.tvalues[:, 1], data.index)


def trend_t_stats(daily_market: pd.DataFrame, d=10):
    roll_reg = partial(_trend_regression, y_col='adj_close', window=d)
    factor = time_series_parallel_apply(daily_market, roll_reg).sort_index()
    factor.name = 'trend_t_stats_{}'.format(d)
    return factor


def CR20():
    pass


def fifty_two_week_close_rank(daily_market: pd.DataFrame) -> pd.Series:
    unstack = daily_market['adj_close'].unstack()

    factor = ts_rank(unstack.values, 250)
    factor = pd.DataFrame(factor, index=unstack.index, columns=unstack.columns).stack()
    factor.name = 'fifty_two_week_close_rank'
    return factor


def MASS():
    pass


def PLRC(daily_market: pd.DataFrame, period=12) -> pd.Series:
    """

    :param daily_market:
    :param period:
    :return:
    """
    factor = daily_market.groupby(level=1) \
        .apply(lambda x: LINEARREG(x, period, 'adj_close')).droplevel(0).sort_index()
    factor.name = 'PLRC_{}'.format(period)
    return factor


def ret_rank(daily_market: pd.DataFrame, period=20) -> pd.Series:
    """

    :param daily_market:
    :param period:
    :return:
    """
    close = daily_market['adj_close'].unstack()  # type: pd.DataFrame
    close_start = close.shift(period)
    ror = (close - close_start) / close_start
    rank_factor = pd.DataFrame(rank(ror.values), index=ror.index, columns=ror.columns).stack()
    factor = 1 - rank_factor
    factor.name = 'ret_rank_{}'.format(period)
    return factor


def ROC(daily_market: pd.DataFrame, period=12):
    """

    :param daily_market:
    :param period:
    :return:
    """

    indicator = abstract.Function('ROC')
    factor = daily_market.groupby(level=1) \
        .apply(lambda x: indicator(x, timeperiod=period)).droplevel(0).sort_index()
    factor.name = 'ROC_{}'.format(period)
    return factor


def single_day_VPT(daily_market: pd.DataFrame, window=12) -> pd.Series:
    """

    :param daily_market:
    :return:
    """
    close = daily_market['adj_close'].unstack()  # type: pd.DataFrame
    amount = daily_market['money']
    ret = close.pct_change()
    factor = (amount * ret).rolling(window).mean()
    factor.name = 'single_day_VPT_{}'.format(window)
    return factor


def volume_ret(daily_market: pd.DataFrame, period: int = 20):
    """
    当日交易量 / 过去20日交易量MEAN * 过去20日收益率MEAN
    :param daily_market:
    :return:
    """
    ret = daily_market['adj_close'].unstack().pct_change()
    amount = daily_market['money'].unstack()
    mean_ret = ret.rolling(period).mean()
    mean_amount = amount.rolling(period).mean()
    factor = amount / mean_amount * mean_ret
    factor.name = 'volume_ret_{}'.format(period)
    return factor


def alpha_momentum(daily_market: pd.DataFrame,
                   risk_premium: pd.Series,
                   smb: pd.Series,
                   hml: pd.Series,
                   look_back: int, ):
    """
    由于股票收益可以分解为特质和公共部分，原始的价格动量受到公共部分影响较大。
    比如说，如果市场超额收益为正，那么动量策略倾向于买入贝塔较大卖出贝塔较小的股票；
    如果市场继续向上，动量策略能够保持盈利，相反如果市场掉头向下，那么动量策略大概率会被拖累。
    基于上面的逻辑，Hühn and Scholz(2018)提出了阿尔法动量
    Hühn, H. L., & Scholz, H. (2018). Alpha momentum and price momentum. International Journal of Financial Studies, 6(2), 49.
    :param daily_market:
    :param risk_premium:
    :param smb:
    :param hml:
    :param look_back:
    :return:
    """
    close = daily_market['adj_close']
    ret = close.groupby(level=1).pct_change()
    three_factors = pd.concat([risk_premium, hml, smb], axis=1)
    data = ret.to_frame('ret').join(three_factors)
    roll_reg = partial(rolling_regression_alpha, y_col='ret', X_cols=['risk_premium', 'HML', 'SMB'], N=look_back)
    factor = time_series_parallel_apply(data, roll_reg)
    if factor.index.nlevels == 3:
        factor = factor.droplevel(0)
    factor = factor.sort_index()
    factor.name = 'alpha_momentum_{}'.format(look_back)
    return factor


def _trend_convexity_regression(y, look_back):
    if len(y) < look_back:
        return pd.Series(np.nan, index=y.index)

    X = np.arange(0, look_back)
    X = np.append([X], [X ** 2], axis=0).T
    res = [np.nan] * (look_back - 1)
    model = LinearRegression()
    for i in range(look_back, len(y) + 1):
        ols = model.fit(X, y[i - look_back: i])
        beta, r = ols.coef_
        if beta >= 0:
            if r >= 0:
                tmp = 1
            else:
                tmp = 0.5
        else:
            if r >= 0:
                tmp = -0.5
            else:
                tmp = -1
        res.append(tmp)
    return pd.Series(res, index=y.index)


def acceleration_momentum(daily_market: pd.DataFrame, look_back: int):
    """
    Investor Attention, Visual Price Pattern, and Momentum Investing

    :param daily_market:
    :param look_back:
    :return:
    """
    tqdm.pandas()
    close = daily_market['adj_close']
    roll_reg = partial(_trend_convexity_regression, look_back=look_back)
    factor = time_series_parallel_apply(close, roll_reg)
    factor = factor.sort_index()
    factor.name = 'acceleration_momentum_{}'.format(look_back)
    return factor


def volume_momentum(daily_market: pd.DataFrame, period_short: int = 20, period_long: int = 60):
    """
    short term average volume/ long term average volume
    :param daily_market:
    :return:
    """
    volume = daily_market['money'] / daily_market['adj_close']
    volume = volume.unstack()
    mean_volume_short = volume.rolling(period_short).mean()
    mean_volume_long = volume.rolling(period_long).mean()
    factor = mean_volume_short / mean_volume_long
    factor = factor.stack()
    factor.name = 'volume_momentum_{}_{}'.format(period_short, period_long)
    return factor


def price_zscores(daily_market: pd.DataFrame, period=20) -> pd.Series:
    """
    (close-EMA(close,20)) / std(close,20)

    :param daily_market:
    :return:
    """
    daily_market_local = daily_market['adj_close'].unstack()
    daily_market_rolling = daily_market_local.rolling(window=period)
    mean = daily_market_rolling.mean()
    std = daily_market_rolling.std(ddof=0)

    factor = (daily_market_local - mean) / std
    factor = factor.dropna(how='all', axis=0)
    factor = factor.stack()
    factor.name = 'price_zscore_{}'.format(period)
    return factor
