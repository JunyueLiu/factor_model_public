import numpy as np
import bottleneck as bn
from typing import Tuple


def high_low_rank_factor(sort_factor: np.ndarray, agg_factor: np.ndarray, lookback: int, threshold: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    rank = (bn.move_rank(sort_factor, lookback, min_count=max(1, lookback // 2), axis=0) + 1) / 2
    high_rank = rank > (1 - threshold)
    low_rank = rank < threshold
    high_factor = agg_factor * high_rank
    low_factor = agg_factor * low_rank
    return high_factor, low_factor


def ideal_sum_high_minus_low(sort_factor: np.ndarray, agg_factor: np.ndarray, lookback: int,
                             threshold: float) -> np.ndarray:
    hf, lf = high_low_rank_factor(sort_factor, agg_factor, lookback, threshold)
    hf_sum = bn.move_sum(hf, lookback, min_count=max(1, lookback // 2), axis=0)
    lf_sum = bn.move_sum(lf, lookback, min_count=max(1, lookback // 2), axis=0)
    return hf_sum - lf_sum


def ideal_mean_high_minus_low(sort_factor: np.ndarray, agg_factor: np.ndarray, lookback: int,
                              threshold: float) -> np.ndarray:
    hf, lf = high_low_rank_factor(sort_factor, agg_factor, lookback, threshold)
    hf_mean = bn.move_sum(hf, lookback, min_count=max(1, lookback // 2), axis=0) / int(lookback * threshold)
    lf_mean = bn.move_sum(lf, lookback, min_count=max(1, lookback // 2), axis=0) / int(lookback * threshold)
    return hf_mean - lf_mean
