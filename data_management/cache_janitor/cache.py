import hashlib
import inspect
import os
import time
from functools import wraps
from typing import Callable
import tempfile

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

from data_management.dataIO.utils import read_pickle, save_pickle

TEMP_PATH = tempfile.gettempdir()
VALID_SEC = 60 * 60 * 24 * 7


def hash_dataframe(df: pd.DataFrame) -> str:
    df_hash = hash_pandas_object(df)
    values_hash = hashlib.sha256(df_hash.values.tobytes()).hexdigest()
    idx_hash = hashlib.sha256(df.index.values.tobytes()).hexdigest()
    col_hash = hashlib.sha256(df.columns.values.tobytes()).hexdigest()
    hash_str = idx_hash + col_hash + values_hash
    return hashlib.sha256(hash_str.encode('utf-8')).hexdigest()


def hash_series(series: pd.Series):
    series_hash = hash_pandas_object(series)
    values_hash = hashlib.sha256(series_hash.values.tobytes()).hexdigest()
    idx_hash = hashlib.sha256(series.index.values.tobytes()).hexdigest()
    name_hash = hashlib.sha256(series.name.encode('utf-8')).hexdigest()
    hash_str = idx_hash + name_hash + values_hash
    return hashlib.sha256(hash_str.encode('utf-8')).hexdigest()


def function_hash(func: Callable, *args, **kwargs):
    source_code = inspect.getsource(func)
    source_code_hash = hashlib.sha256(source_code.encode('utf-8')).hexdigest()
    arg_hash = ''
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            arg_hash += hash_dataframe(arg)
        elif isinstance(arg, pd.Series):
            arg_hash += hash_series(arg)
        elif isinstance(arg, np.ndarray):
            arg_hash += hashlib.sha256(arg.tobytes()).hexdigest()
        elif isinstance(arg, Callable):
            arg_hash += hashlib.sha256(func.__func__.__name__.tobytes()).hexdigest()
        else:
            arg_hash += hashlib.sha256(str(arg).encode('utf-8')).hexdigest()
    kwarg_hash = ''
    for k, v in kwargs.items():
        key_hash = hashlib.sha256(str(k).encode('utf-8')).hexdigest()
        kwarg_hash += key_hash
        if isinstance(v, pd.DataFrame):
            kwarg_hash += hash_dataframe(v)
        elif isinstance(v, pd.Series):
            kwarg_hash += hash_series(v)
        elif isinstance(v, np.ndarray):
            kwarg_hash += hashlib.sha256(v.tobytes()).hexdigest()
        else:
            kwarg_hash += hashlib.sha256(str(v).encode('utf-8')).hexdigest()

    hash_str = source_code_hash + arg_hash + kwarg_hash
    return hashlib.sha256(hash_str.encode('utf-8')).hexdigest()


def tmp_cache(f):
    os.makedirs(TEMP_PATH, exist_ok=True)

    @wraps(f)
    def f_cache(*args, **kwargs):
        h = function_hash(f, args, kwargs)
        save_path = os.path.join(TEMP_PATH, h)
        if os.path.exists(os.path.join(TEMP_PATH, h)):
            save_time = os.path.getmtime(save_path)
            now_time = time.time()
            alive_sec = int(now_time - save_time)
            if alive_sec <= VALID_SEC:
                # read results
                print('In the cache and valid, load directly from {}'.format(os.path.join(TEMP_PATH, h)))
                res = read_pickle(save_path)
                return res
        print('Not in cache or not valid, calculate the results.')
        res = f(*args, **kwargs)
        save_pickle(res, save_path)
        return res

    return f_cache
