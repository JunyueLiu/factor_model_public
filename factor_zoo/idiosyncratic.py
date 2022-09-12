from functools import partial

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from data_management.cache_janitor.cache import tmp_cache
from data_management.dataIO import market_data
from data_management.keeper.ZooKeeper import ZooKeeper
from data_management.pandas_utils.parallel import time_series_parallel_apply
from factor_zoo.factor_transform import cosine_transform


def idiosyncratic_volatility(daily_market: pd.DataFrame,
                             risk_premium: pd.Series,
                             smb: pd.Series,
                             hml: pd.Series,
                             look_back: int,
                             std_window: int
                             ):
    """

    :param daily_market:
    :param risk_premium:
    :param smb:
    :param hml:
    :param look_back:
    :param std_window:
    :return:
    """
    residual = fama_french_hotpot(daily_market, risk_premium, smb, hml, look_back)
    idiosyncratic_volatility = residual.groupby(level=1).rolling(std_window).std()
    idiosyncratic_volatility.name = 'idiosyncratic_volatility_{}_{}'.format(look_back, std_window)
    return idiosyncratic_volatility


def rolling_regression(data, y_col, X_cols, N):
    if len(data) < N:
        return pd.Series(np.nan, data.index)

    endog = data[y_col].values
    exog = sm.add_constant(data[X_cols].values)
    rols = RollingOLS(endog, exog, window=N)
    rres = rols.fit(cov_type='HCCM')
    res = endog - np.sum(exog * rres.params, axis=1)
    return pd.Series(res, data.index)

@tmp_cache
def fama_french_hotpot(daily_market: pd.DataFrame,
                       risk_premium: pd.Series,
                       smb: pd.Series,
                       hml: pd.Series,
                       look_back: int
                       ):
    close = daily_market['adj_close']
    ret = close.unstack().pct_change().stack().sort_index()
    three_factors = pd.concat([risk_premium, hml, smb], axis=1)
    data = ret.to_frame('ret').join(three_factors)
    roll_reg = partial(rolling_regression, y_col='ret', X_cols=['risk_premium', 'HML', 'SMB'], N=look_back)

    factor = time_series_parallel_apply(data, roll_reg)
    if factor.index.nlevels == 3:
        factor = factor.droplevel(0).sort_index()
    factor.name = 'fama_french_hotpot_{}'.format(look_back)
    return factor


def fama_french_hotpot_quadratic(daily_market: pd.DataFrame,
                                 risk_premium: pd.Series,
                                 smb: pd.Series,
                                 hml: pd.Series,
                                 look_back: int):
    f = fama_french_hotpot(daily_market, risk_premium, smb, hml, look_back)
    f = - (f - f.groupby(level=0).mean()) ** 2 + 1
    f.name = '{}_quadratic'.format(f.name)
    return f


def fama_french_hotpot_cosine(daily_market: pd.DataFrame,
                              risk_premium: pd.Series,
                              smb: pd.Series,
                              hml: pd.Series,
                              look_back: int):
    f = fama_french_hotpot(daily_market, risk_premium, smb, hml, look_back)
    f = cosine_transform(f)
    return f


def residual_momentum(daily_market: pd.DataFrame,
                      risk_premium: pd.Series,
                      smb: pd.Series,
                      hml: pd.Series,
                      look_back: int,
                      smooth_period: int,
                      skip_period: int
                      ):
    assert smooth_period > skip_period

    def smooth(x: pd.Series):
        used = x[:-skip_period]
        return used.mean() / used.std()

    f = fama_french_hotpot(daily_market, risk_premium, smb, hml, look_back)
    f = f.groupby(level=1).rolling(smooth_period).apply(smooth)
    f = f.droplevel(0).sort_index()
    f.name = 'residual_momentum_{}_{}_{}'.format(look_back, smooth_period, skip_period)
    return f


if __name__ == '__main__':
    zookeeper_config_path = '../cfg/factor_keeper_setting.ini'
    data_config_path = '../cfg/data_input.ini'
    keeper = ZooKeeper(zookeeper_config_path)
    start_date = '2013-01-01'
    hml, _ = keeper.get_factor_values('FAMA-FRENCH', 'HML', )
    smb , _= keeper.get_factor_values('FAMA-FRENCH', 'SMB')
    risk_premium, _ = keeper.get_factor_values('FAMA-FRENCH', 'risk_premium')
    daily_market = market_data.get_bars(adjust=True, eod_time_adjust=False, add_limit=False, start_date=start_date,
                                        config_path=data_config_path)
    # f = fama_french_hotpot(daily_market, risk_premium, smb, hml, 20)
    f = residual_momentum(daily_market, risk_premium, smb, hml, 120, 40, 5)
