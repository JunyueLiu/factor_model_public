import itertools
import multiprocessing
import time
from concurrent.futures.process import ProcessPoolExecutor
from typing import Callable, Tuple, Union

import pandas as pd
import tqdm


def groupby_parallel(groupby_df: pd.core.groupby.DataFrameGroupBy,
                     func: Callable[[Tuple[str, pd.DataFrame]], Union[pd.DataFrame, pd.Series]],
                     num_cpus: int = multiprocessing.cpu_count() - 1,
                     logger: Callable[[str], None] = print) -> pd.DataFrame:
    """Performs a Pandas groupby operation in parallel.
    Example usage:
        import pandas as pd
        df = pd.DataFrame({'A': [0, 1], 'B': [100, 200]})
        df.groupby(df.groupby('A'), lambda row: row['B'].sum())
    Authors: Tamas Nagy and Douglas Myers-Turnbull
    """
    start = time.time()
    logger("\nUsing {} CPUs in parallel...".format(num_cpus))
    with multiprocessing.Pool(num_cpus) as pool:
        queue = multiprocessing.Manager().Queue()
        result = pool.starmap_async(func, [(name, group) for name, group in groupby_df])
        cycler = itertools.cycle('\|/―')
        while not result.ready():
            logger("Percent complete: {:.0%} {}".format(queue.qsize() / len(groupby_df), next(cycler)), end="\r")
            time.sleep(0.4)
        got = result.get()
    logger("\nProcessed {} rows in {:.1f}s".format(len(got), time.time() - start))
    return pd.concat(got)


def time_series_parallel_apply(df, func, chuncksize=10):
    group = [d for n, d in df.groupby(level=1)]
    with ProcessPoolExecutor() as executor:
        results = list(tqdm.tqdm(executor.map(func, group, chunksize=chuncksize), total=len(group)))
    df = pd.concat(results)
    return df
