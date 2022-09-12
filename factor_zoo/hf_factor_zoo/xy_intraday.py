from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import tqdm
from sklearn.mixture import GaussianMixture

from data_management.dataIO.exotic_data import Exotic, get_exotic
from data_management.dataIO.market_data import get_intraday_bars, Freq
from factor_zoo.hf_factor_zoo.intraday_operator import get_min_data_df, intraday_ret


def _single_stock_intraday_mixture_gaussian(ret: pd.DataFrame):
    """

    Parameters
    ----------
    ret

    Returns
    -------

    """
    w = []
    mu = []
    sigma = []
    dt = ret.index
    params = dict(
        tol=1e-10, random_state=42, max_iter=2000, weights_init=[0.3, 0.7],
        means_init=[[0], [0]], reg_covar=1e-12, init_params='random'

    )
    model = GaussianMixture(2, tol=1e-10, random_state=42, max_iter=2000, weights_init=[0.3, 0.7],
                            means_init=[[0], [0]], reg_covar=1e-12, init_params='random')
    for i in range(0, len(ret)):
        r = ret.iloc[i]
        model.set_params(**params)
        model = model.fit(r.values.reshape(-1, 1))
        w.append(model.weights_)
        mu.append(model.means_.reshape(-1))
        sigma.append(model.covariances_.reshape(-1))
    data = pd.DataFrame(np.concatenate([w, mu], axis=1), index=dt, columns=['w1', 'w2', 'mu1', 'mu2'])
    data['w_i'] = np.maximum(data['w1'], data['w2'])
    data['w_j'] = np.minimum(data['w1'], data['w2'])

    data['mu_i'] = data.apply(lambda x: x['mu2'] if x['w1'] < x['w2'] else x['mu1'], axis=1)
    data['mu_j'] = data.apply(lambda x: x['mu1'] if x['w1'] < x['w2'] else x['mu2'], axis=1)
    return data


def single_stock_gmm_factors(symbol, config_path, start_date='2010-01-01'):
    try:
        df = get_intraday_bars(symbol, Freq.m5, start_date=start_date,
                               config_path=config_path)
    except:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df = get_min_data_df(df).set_index('code', append=True).unstack(level=1)
    ret = pd.DataFrame(intraday_ret(df['close'].values), index=df.index)
    ret = ret[ret.columns[1:]]

    data = _single_stock_intraday_mixture_gaussian(ret)
    return data


def single_stock_gmm_factors_rolling(symbol, config_path, n=5, start_date='2010-01-01'):
    try:
        df = get_intraday_bars(symbol, Freq.m5, start_date=start_date,
                               config_path=config_path)
    except:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df = get_min_data_df(df).set_index('code', append=True).unstack(level=1)
    ret = pd.DataFrame(intraday_ret(df['close'].values), index=df.index)
    ret = ret[ret.columns[1:]]

    data = _single_stock_intraday_mixture_gaussian(ret)
    return data


def gmm_factors(stock_info: pd.DataFrame, config_path: str, start_date='2010-01-01'):
    symbols = stock_info[stock_info['type'] == 'stock'].index
    func = partial(single_stock_gmm_factors, config_path=config_path, start_date=start_date)
    with ProcessPoolExecutor() as executor:
        res = list(tqdm.tqdm(executor.map(func, symbols, chunksize=10), total=len(symbols)))
    data = pd.concat(res)
    return data


def _single_stock_intraday_stats(ret):
    dt = ret.index

    # intraday_skew =

    for i in range(0, len(ret)):
        r = ret.iloc[i]


def single_stock_stats_factors(symbol, config_path):
    try:
        df = get_intraday_bars(symbol, Freq.m5, start_date='2010-01-01',
                               config_path=config_path)
    except:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df = get_min_data_df(df).set_index('code', append=True).unstack(level=1)
    ret = pd.DataFrame(intraday_ret(df['close'].values), index=df.index)
    ret = ret[ret.columns[1:]]
    _single_stock_intraday_stats(ret)


def stats_factors(stock_info: pd.DataFrame, config_path: str):
    symbols = stock_info[stock_info['type'] == 'stock'].index


def gmm_mean(config_path: str,
             start_date: str,
             end_date: str):
    data = get_exotic(Exotic.xy_gmm,
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    f = data['mu_j']
    f.name = 'gmm_mean'
    return f


def gmm_mean2wgt(config_path: str,
                 start_date: str,
                 end_date: str):
    data = get_exotic(Exotic.xy_gmm,
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    f = data['mu_j'] / data['w_j']
    f.name = 'gmm_mean2wgt'
    return f


def gmm_meandif(config_path: str,
                start_date: str,
                end_date: str):
    data = get_exotic(Exotic.xy_gmm,
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    f = data['mu_i'] - data['mu_j']
    f.name = 'gmm_meandif'
    return f


def gmm_meandif2wgtdif(config_path: str,
                       start_date: str,
                       end_date: str):
    data = get_exotic(Exotic.xy_gmm,
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    f = (data['mu_i'] - data['mu_j']) / (data['w_i'] - data['w_j'])
    f.name = 'gmm_meandif2wgtdif'
    return f


def gmm_1m_mean(config_path: str,
                start_date: str,
                end_date: str):
    data = get_exotic(Exotic.xy_gmm_1m,
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    f = data['mu_j']
    f.name = 'gmm_1m_mean'
    return f


def gmm_1m_mean2wgt(config_path: str,
                    start_date: str,
                    end_date: str):
    data = get_exotic(Exotic.xy_gmm_1m,
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    f = data['mu_j'] / data['w_j']
    f.name = 'gmm_1m_mean2wgt'
    return f


def gmm_1m_meandif(config_path: str,
                   start_date: str,
                   end_date: str):
    data = get_exotic(Exotic.xy_gmm_1m,
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    f = data['mu_i'] - data['mu_j']
    f.name = 'gmm_1m_meandif'
    return f


def gmm_1m_meandif2wgtdif(config_path: str,
                          start_date: str,
                          end_date: str):
    data = get_exotic(Exotic.xy_gmm_1m,
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    f = (data['mu_i'] - data['mu_j']) / (data['w_i'] - data['w_j'])
    f.name = 'gmm_1m_meandif2wgtdif'
    return f


def gmm_5m_rolling_mean(config_path: str,
                        start_date: str,
                        end_date: str):
    data = get_exotic(Exotic.xy_gmm_5m_rolling,
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    f = data['mu_j']
    f.name = 'gmm_5m_rolling_mean'
    return f


def gmm_5m_rolling_mean2wgt(config_path: str,
                            start_date: str,
                            end_date: str):
    data = get_exotic(Exotic.xy_gmm_5m_rolling,
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    f = data['mu_j'] / data['w_j']
    f.name = 'gmm_5m_rolling_mean2wgt'
    return f


def gmm_5m_rolling_meandif(config_path: str,
                           start_date: str,
                           end_date: str):
    data = get_exotic(Exotic.xy_gmm_5m_rolling,
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    f = data['mu_i'] - data['mu_j']
    f.name = 'gmm_5m_rolling_meandif'
    return f


def gmm_5m_rolling_meandif2wgtdif(config_path: str,
                                  start_date: str,
                                  end_date: str):
    data = get_exotic(Exotic.xy_gmm_5m_rolling,
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    f = (data['mu_i'] - data['mu_j']) / (data['w_i'] - data['w_j'])
    f.name = 'gmm_5m_rolling_meandif2wgtdif'
    return f


def gmm_5m_rolling_mean_mv(config_path: str,
                           start_date: str,
                           end_date: str,
                           n: int = 20
                           ):
    data = get_exotic(Exotic.xy_gmm_5m_rolling,
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    f = data['mu_j'].groupby(level=1).rolling(n).mean().droplevel(0).sort_index()
    f.name = 'gmm_5m_rolling_mean_{}'.format(n)
    return f


def gmm_5m_rolling_mean2wgt_mv(config_path: str,
                               start_date: str,
                               end_date: str,
                               n: int = 20
                               ):
    data = get_exotic(Exotic.xy_gmm_5m_rolling,
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    f = (data['mu_j'] / data['w_j']).groupby(level=1).rolling(n).mean().droplevel(0).sort_index()
    f.name = 'gmm_5m_rolling_mean2wgt_{}'.format(n)
    return f


def gmm_5m_rolling_meandif_mv(config_path: str,
                              start_date: str,
                              end_date: str,
                              n: int = 20
                              ):
    data = get_exotic(Exotic.xy_gmm_5m_rolling,
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    f = (data['mu_i'] - data['mu_j']).groupby(level=1).rolling(n).mean().droplevel(0).sort_index()
    f.name = 'gmm_5m_rolling_meandif_{}'.format(n)
    return f


def gmm_5m_rolling_meandif2wgtdif_mv(config_path: str,
                                     start_date: str,
                                     end_date: str,
                                     n: int = 20
                                     ):
    data = get_exotic(Exotic.xy_gmm_5m_rolling,
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    f = ((data['mu_i'] - data['mu_j']) / (data['w_i'] - data['w_j'])).groupby(level=1).rolling(n).mean().droplevel(0).sort_index()
    f.name = 'gmm_5m_rolling_meandif2wgtdif_{}'.format(n)
    return f


if __name__ == '__main__':
    # df = get_intraday_bars('000001.SZ', Freq.m5)
    # single_stock_gmm_factors('000001.SZ', '../../cfg/data_input.ini')
    # stock_info = get_stock_info('../../cfg/data_input.ini')
    # gmm_factors(stock_info, '../../cfg/data_input.ini')
    # single_stock_stats_factors('000001.SZ', '../../cfg/data_input.ini')
    # gmm_mean('../../cfg/data_input.ini', '2022-01-01', '2022-06-01')
    gmm_5m_rolling_mean_mv('../../cfg/data_input.ini', '2022-01-01', '2022-06-01', 20)
