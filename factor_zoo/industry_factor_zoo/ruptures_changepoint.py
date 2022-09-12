import pandas as pd
import ruptures as rpt

from data_management.dataIO.component_data import get_bars


def rbf_changepoint(df: pd.DataFrame, n: int = 20):
    ret = df['change_pct'] / 100

    def _f(signal):
        result = rpt.KernelCPD(kernel="rbf", ).fit_predict(signal.values, 1)
        return result[0]

    f = ret.groupby(level=1).rolling(n).apply(_f)
    f = f.droplevel(0).sort_index() / n
    f.name = 'rbf_changepoint_{}'.format(n)
    return f


if __name__ == '__main__':
    df = get_bars(start_date='2013-01-01')
    rbf_changepoint(df)
