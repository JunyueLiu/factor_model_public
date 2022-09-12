import math
from typing import Union

import bottleneck as bn
import numpy as np

from factor_zoo.factor_operator.utils import nans, shift, talib_corr


def returns(x: np.ndarray) -> np.ndarray:
    f = delta(x, 1) / shift(x, 1)
    return f


def vwap(volume: np.ndarray, amount: np.ndarray, factor: np.ndarray) -> np.ndarray:
    f = (amount / volume) * factor
    return f


def adv(amount: np.ndarray, d: int) -> np.ndarray:
    f = bn.move_mean(amount, d, min_count=max(1, d // 2), axis=0)
    return f


def abs(x: np.ndarray) -> np.ndarray:
    f = np.abs(x)
    return f


def rank(x: np.ndarray) -> np.ndarray:
    _rank = bn.nanrankdata(x, axis=1)
    _rank_max = np.nanmax(_rank, axis=1)
    f = ((1 / _rank_max) * _rank.T).T
    return f


def delay(x: np.ndarray, n: int):
    return shift(x, n)


# @numba.njit
def corrcoef(a: np.ndarray, b: np.ndarray, c: int) -> np.ndarray:
    corr_matrix = np.corrcoef(a, b, False)
    corr = corr_matrix[0:c, -c:]
    corr = np.diag(corr)
    return corr


def correlation(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    assert d > 0

    arr = nans(x)
    # talib approach doesn't work
    # for i in range(x.shape[1]):
    #     arr[:, i] = talib_corr(x[:, i], y[:, i], d)

    # too slow
    # r, c = x.shape
    # for j in tqdm.tqdm(range(d, r)):
    #     # what we need is diagonal of upper right matrix
    #     # c[0:20, -20:]
    #     # corr_matrix = np.corrcoef(x[j - d:j, :], y[j - d:j, :], False)
    #     corr = corrcoef(x[j - d:j, :], y[j - d:j, :], c)
    #     arr[j, :] = corr
    #     # print(j / r * 100)

    for i in range(x.shape[1]):
        arr[:, i] = talib_corr(x[:, i], y[:, i], d)
    return arr


def covariance(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    assert d > 0
    if isinstance(d, float):
        d = math.floor(d)
    f = correlation(x, y, d) * stddev(x, d) * stddev(y, d)
    return f


def delta(x: np.ndarray, d: int) -> np.ndarray:
    assert d > 0
    f = nans(x)
    f[d:, :] = np.diff(x, d, axis=0)
    return f


def scale(x: np.ndarray, a: int = 1) -> np.ndarray:
    m = a * np.nanmean(np.abs(x), axis=1, keepdims=True)
    f = x / m
    return f


def signedpower(x: np.ndarray, a: Union[float, np.ndarray]) -> np.ndarray:
    f = np.power(np.abs(x), a) * np.sign(x)
    return f


# @numba.njit
def decay_linear(x: np.ndarray, d: int) -> np.ndarray:
    """
    decay_linear(x, d) = weighted moving average over the past d days
    with linearly decaying weights d, d – 1, ..., 1 (rescaled to sum up to 1)
    :param x:
    :param d:
    :return:
    """
    if isinstance(d, float):
        d = math.floor(d)

    f = nans(x)
    weights = np.arange(1, d + 1)
    weights = weights / weights.sum()
    for i in range(d - 1, x.shape[0]):
        f[i, :] = np.sum(weights[:, None] * x[(i - d + 1): i + 1, :], axis=0)
    return f


def indneutralize(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    indneutralize(x, g) = x cross-sectionally neutralized against groups g (subindustries, industries, sectors, etc.),
    i.e., x is cross-sectionally demeaned within each group g

    :param x:
    :param g:
    :return:
    """
    ind = np.unique(g.tolist())
    for i in range(len(ind)):
        a = g == ind[i]
        ind_mean = np.nanmean(x * a, axis=1, keepdims=True)
        x = x - ind_mean
    return x


# def ts_operation(x: np.ndarray, d: int or float, operation) -> np.ndarray:
#     """
#     ts_{O}(x, d) = operator O applied across the time-series for the past d days;
#     non-integer number of days d is converted to floor(d)
#     :param x:
#     :param d:
#     :param operation:
#     :return:
#     """
#     if isinstance(d, float):
#         d = math.floor(d)
#     return x.groupby(level=1).rolling(d).apply(operation)


# 
# def ts_min(x: np.ndarray, d: int or float) -> np.ndarray:
#     """
#     ts_min(x, d) = time-series min over the past d days
#     :param x:
#     :param d:
#     :return:
#     """
#     if isinstance(d, float):
#         d = math.floor(d)
#     return x.groupby(level=1).rolling(d).min().droplevel(0).sort_index()


def ts_min(x: np.ndarray, d: int or float) -> np.ndarray:
    """
    ts_min(x, d) = time-series min over the past d days
    :param x:
    :param d:
    :return:
    """
    if isinstance(d, float):
        d = math.floor(d)
    f = bn.move_min(x, d, min_count=max(1, d // 2), axis=0)
    return f


def ts_max(x: np.ndarray, d: int or float) -> np.ndarray:
    """
    ts_max(x, d) = time-series max over the past d days
    :param x:
    :param d:
    :return:
    """
    if isinstance(d, float):
        d = math.floor(d)
    f = bn.move_max(x, d, min_count=max(1, d // 2), axis=0)
    return f


def ts_argmax(x: np.ndarray, d: int or float) -> np.ndarray:
    """
    ts_argmax(x, d) = which day ts_max(x, d) occurred on
    :param x:
    :param d:
    :return:
    """
    if isinstance(d, float):
        d = math.floor(d)
    f = bn.move_argmax(x, d, min_count=max(1, d // 2), axis=0) / d
    return f


def ts_argmin(x: np.ndarray, d: int or float) -> np.ndarray:
    """
    ts_argmin(x, d) = which day ts_min(x, d) occurred on
    :param x:
    :param d:
    :return:
    """
    if isinstance(d, float):
        d = math.floor(d)
    f = bn.move_argmin(x, d, min_count=max(1, d // 2), axis=0) / d
    return f


def ts_rank(x: np.ndarray, d: int or float) -> np.ndarray:
    """
    ts_rank(x, d) = time-series rank in the past d days
    :param x:
    :param d:
    :return:
    """
    if isinstance(d, float):
        d = math.floor(d)

    f = (bn.move_rank(x, d, min_count=max(1, d // 2), axis=0) + 1) / 2
    return f


def min(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """

    :param y:
    :param x:
    :return:
    """
    return np.minimum(x, y)


def max(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    :param y:
    :param x:
    :return:
    """
    return np.maximum(x, y)


def sum(x: np.ndarray, d: int or float) -> np.ndarray:
    """
    :param x:
    :param d:
    :return:
    """
    if isinstance(d, float):
        d = math.floor(d)
    f = bn.move_sum(x, d, min_count=max(1, d // 2), axis=0)
    return f


def product(x: np.ndarray, d: int or float) -> np.ndarray:
    """
    product(x, d) = time-series product over the past d days
    :param x:
    :param d:
    :return:
    """
    if isinstance(d, float):
        d = math.floor(d)

    log_x = np.log(x)
    f = bn.move_sum(log_x, d, min_count=max(1, d // 2), axis=0)
    f = np.exp(f)
    return f


def stddev(x: np.ndarray, d: int or float) -> np.ndarray:
    """
    stddev(x, d) = moving time-series standard deviation over the past d days
    :param x:
    :param d:
    :return:
    """
    if isinstance(d, float):
        d = math.floor(d)

    f = bn.move_std(x, d, min_count=max(1, d // 2), axis=0)
    return f
