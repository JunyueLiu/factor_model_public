import bottleneck as bn
import numpy as np

from factor_zoo.factor_operator import alpha101_operator


def cs_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def cs_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a - b


def cs_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def cs_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / b


def cs_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)


def cs_max(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)


def cs_sqrt(a: np.ndarray) -> np.ndarray:
    return a ** 0.5


def cs_curt(a: np.ndarray) -> np.ndarray:
    return a ** (1 / 3)


def cs_square(a: np.ndarray) -> np.ndarray:
    return a ** 2


def cs_cube(a: np.ndarray) -> np.ndarray:
    return a ** 3


def cs_log(a: np.ndarray) -> np.ndarray:
    return np.log(a)


def ts_min(a: np.ndarray, n: int) -> np.ndarray:
    return alpha101_operator.ts_min(a, n)


def ts_mean(a: np.ndarray, n: int) -> np.ndarray:
    f = bn.move_mean(a, n, axis=0)
    return f


def ts_max(a: np.ndarray, n: int) -> np.ndarray:
    return alpha101_operator.ts_max(a, n)


def ts_std(a, n: int) -> np.ndarray:
    f = bn.move_std(a, n, axis=0)
    return f


def ts_median(a: np.ndarray, n: int) -> np.ndarray:
    f = bn.move_median(a, n, axis=0)
    return f


def ts_stable(a: np.ndarray, n: int) -> np.ndarray:
    return ts_mean(a, n) / ts_std(a, n)


def ts_minmaxnorm(a: np.ndarray, n: int) -> np.ndarray:
    hh = ts_max(a, n)
    ll = ts_min(a, n)
    return (a - ll) / (hh - ll)


def ts_meanstdnorm(a: np.ndarray, n: int) -> np.ndarray:
    return (a - ts_mean(a, n)) / ts_std(a, n)


def ts_delta(a: np.ndarray, n: int) -> np.ndarray:
    return a - ts_mean(a, n)


def ts_corr(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    return alpha101_operator.correlation(a, b, n)
