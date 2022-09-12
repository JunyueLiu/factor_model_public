import pandas as pd

from tqdm import tqdm
tqdm.pandas()


def pre_adjust_price(origin_df: pd.DataFrame, adj_factor: pd.Series = None,
                     fixed_time: None or int or pd.Timestamp or str = None,
                     PRICE_COLS=('open', 'close', 'high', 'low'),
                     FACTOR_COL='factor'
                     ) -> pd.DataFrame:
    # PRICE_COLS = ['open', 'close', 'high', 'low']
    FORMAT = lambda x: '%.4f' % x
    # if adj_factor is None:
    #     return origin_df

    origin_cols = origin_df.columns
    # if adj_factor is None:
    #     adj_factor = origin_df[FACTOR_COL]

    if origin_df.index.nlevels == 1:
        if adj_factor is not None and adj_factor.index.ndim != 1:
            raise ValueError('adj_factor must have only one level of index')
        if adj_factor is not None:
            origin_df = origin_df.sort_index()
            adj_factor = adj_factor.sort_index()
            data = pd.merge_asof(origin_df, adj_factor.to_frame(FACTOR_COL), left_index=True, right_index=True)
        else:
            data = origin_df.copy()

        if fixed_time is None:
            factor = float(data[FACTOR_COL][-1])
        elif isinstance(fixed_time, int):
            if fixed_time > 0:
                fixed_time = - fixed_time
            factor = float(data[FACTOR_COL][fixed_time])
        elif isinstance(fixed_time, str):
            idx = data[FACTOR_COL].loc[data.index >= fixed_time].index[0]
            factor = data[FACTOR_COL].loc[idx]
        elif isinstance(fixed_time, pd.Timestamp):
            idx = data[FACTOR_COL].loc[data.index >= fixed_time].index[0]
            factor = data[FACTOR_COL].loc[idx]
        else:
            raise NotImplementedError
        for col in PRICE_COLS:
            data[col] = data[col] * data[FACTOR_COL] / factor
            data[col] = data[col].map(FORMAT)
            data[col] = data[col].astype(float)
    elif origin_df.index.nlevels == 2:
        if adj_factor is not None and adj_factor.index.ndim != 2:
            raise ValueError('adj_factor must have only two level of index')
        if adj_factor is not None:
            origin_df = origin_df.sort_index()
            adj_factor = adj_factor.sort_index()
            data = pd.merge_asof(origin_df, adj_factor.to_frame(FACTOR_COL), left_index=True, right_index=True)
        else:
            data = origin_df.copy()

        def func(x, fixed_time):
            if fixed_time is None:
                factor = float(x[FACTOR_COL][-1])
            elif isinstance(fixed_time, int):
                if fixed_time > 0:
                    fixed_time = - fixed_time
                factor = float(x[FACTOR_COL][fixed_time])
            elif isinstance(fixed_time, str):
                try:
                    idx = x[FACTOR_COL].loc[x.index >= fixed_time].index[0]
                    factor = x[FACTOR_COL].loc[idx]
                except:
                    factor = float(x[FACTOR_COL][-1])
            elif isinstance(fixed_time, pd.Timestamp):
                try:
                    idx = x[FACTOR_COL].loc[x.index >= fixed_time].index[0]
                    factor = x[FACTOR_COL].loc[idx]
                except:
                    factor = float(x[FACTOR_COL][-1])
            else:
                raise NotImplementedError
            return factor

        divisor = data.groupby(level=1).progress_apply(lambda x: func(x, fixed_time))
        for col in PRICE_COLS:
            data[col] = data[col] * data[FACTOR_COL] / divisor
            data[col] = data[col].map(FORMAT)
            data[col] = data[col].astype(float)

    else:
        raise NotImplementedError
    return data[origin_cols]
