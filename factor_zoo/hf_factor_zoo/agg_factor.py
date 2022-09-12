import numpy as np
import pandas as pd
from pandas._libs.tslibs.offsets import Day
from scipy import stats

from factor_zoo.hf_factor_zoo.intraday_operator import select_open_session, select_close_session, select_intra_session, \
    select_high, select_low


def avg(f: np.ndarray) -> np.ndarray:
    f = np.nanmean(f, axis=1)
    return f


def open_avg(f: np.ndarray) -> np.ndarray:
    f = select_open_session(f)
    f = np.nanmean(f, axis=1)
    return f


def intra_avg(f: np.ndarray) -> np.ndarray:
    f = select_intra_session(f)
    f = np.nanmean(f, axis=1)
    return f


def close_avg(f: np.ndarray) -> np.ndarray:
    f = select_close_session(f)
    f = np.nanmean(f, axis=1)
    return f


def std(f: np.ndarray) -> np.ndarray:
    f = np.nanstd(f, axis=1)
    return f


def open_std(f: np.ndarray) -> np.ndarray:
    f = select_open_session(f)
    f = np.nanstd(f, axis=1)
    return f


def intra_std(f: np.ndarray) -> np.ndarray:
    f = select_intra_session(f)
    f = np.nanstd(f, axis=1)
    return f


def close_std(f: np.ndarray) -> np.ndarray:
    f = select_close_session(f)
    f = np.nanstd(f, axis=1)
    return f


def skew(f: np.ndarray) -> np.ndarray:
    f = stats.skew(f, axis=1, nan_policy='omit')
    return f


def open_skew(f: np.ndarray) -> np.ndarray:
    f = select_open_session(f)
    f = stats.skew(f, axis=1, nan_policy='omit')
    return f


def intra_skew(f: np.ndarray) -> np.ndarray:
    f = select_intra_session(f)
    f = stats.skew(f, axis=1, nan_policy='omit')
    return f


def close_skew(f: np.ndarray) -> np.ndarray:
    f = select_close_session(f)
    f = stats.skew(f, axis=1, nan_policy='omit')
    return f


def kurt(f: np.ndarray) -> np.ndarray:
    f = stats.kurtosis(f, axis=1, nan_policy='omit')
    return f


def open_kurt(f: np.ndarray) -> np.ndarray:
    f = select_open_session(f)
    f = stats.kurtosis(f, axis=1, nan_policy='omit')
    return f


def intra_kurt(f: np.ndarray) -> np.ndarray:
    f = select_intra_session(f)
    f = stats.kurtosis(f, axis=1, nan_policy='omit')
    return f


def close_kurt(f: np.ndarray) -> np.ndarray:
    f = select_close_session(f)
    f = stats.kurtosis(f, axis=1, nan_policy='omit')
    return f


def high_avg(price: np.ndarray, f: np.ndarray) -> np.ndarray:
    masked = select_high(price)
    f = np.nanmean(masked * f, axis=1)
    return f


def low_avg(price: np.ndarray, f: np.ndarray) -> np.ndarray:
    masked = select_low(price)
    f = np.nanmean(masked * f, axis=1)
    return f


def high_std(price: np.ndarray, f: np.ndarray) -> np.ndarray:
    masked = select_high(price)
    f = np.nanstd(masked * f, axis=1)
    return f


def low_std(price: np.ndarray, f: np.ndarray) -> np.ndarray:
    masked = select_low(price)
    f = np.nanstd(masked * f, axis=1)
    return f


def high_skew(price: np.ndarray, f: np.ndarray) -> np.ndarray:
    masked = select_high(price)
    f = stats.skew(masked * f, axis=1, nan_policy='omit')
    return f


def low_skew(price: np.ndarray, f: np.ndarray) -> np.ndarray:
    masked = select_low(price)
    f = stats.skew(masked * f, axis=1, nan_policy='omit')
    return f


def high_kurt(price: np.ndarray, f: np.ndarray) -> np.ndarray:
    masked = select_high(price)
    f = stats.kurtosis(masked * f, axis=1, nan_policy='omit')
    return f


def low_kurt(price: np.ndarray, f: np.ndarray) -> np.ndarray:
    masked = select_low(price)
    f = stats.kurtosis(masked * f, axis=1, nan_policy='omit')
    return f


def row_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(x, y)
    c = np.diag(corr[x.shape[0]:, :x.shape[0]])
    return c


def row_spearman_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    correlation, pvalue = stats.spearmanr(x, y, axis=1)
    c = np.diag(correlation[x.shape[0]:, :x.shape[0]])
    return c


def hf_to_lf_mv_mean(hf_factor: pd.Series, *, n: int, offset: Day):
    pass
