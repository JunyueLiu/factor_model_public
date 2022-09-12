import numpy as np
import tqdm
import bottleneck as bn

from factor_zoo.factor_operator.utils import nans
import talib

def sma(x: np.ndarray, n: int, m: int) -> np.ndarray:
    k = m / n
    ta_lib_period = int(2 / k - 1)
    arr = nans(x)
    for i in tqdm.tqdm(range(x.shape[1]), desc='sma'):
        arr[:, i] = talib.EMA(x[:, i], ta_lib_period)
    return arr

def mean(x: np.ndarray, n:int) -> np.ndarray:
    return bn.move_mean(x, n, axis=0)