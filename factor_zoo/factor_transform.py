import re

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from data_management.keeper.ZooKeeper import ZooKeeper


# from factor_zoo.utils import rank


def winsorize_factor(series: pd.Series, limits: float or tuple):
    """

    :param series:
    :param limits:
    :return:
    """
    assert isinstance(series.index, pd.MultiIndex)

    def scipy_winsorize(x):
        x_w = winsorize(x, limits=limits, nan_policy='omit')
        return pd.Series(x_w, x.index)

    return series.groupby(level=0).apply(scipy_winsorize)


def normalized_factor(series: pd.Series) -> pd.Series:
    """

    :param series:
    :return:
    """
    assert isinstance(series.index, pd.MultiIndex)

    def sklearn_norm(x):
        normal_scaler = preprocessing.StandardScaler()
        x_scaled = normal_scaler.fit_transform(x.values.reshape(-1, 1))
        return pd.Series(x_scaled[:, 0], x.index)

    return series.groupby(level=0).apply(sklearn_norm)


def winsorize_and_normal_factor(series: pd.Series, limits: float or tuple):
    """

    :param series:
    :param limits:
    :return:
    """
    assert isinstance(series.index, pd.MultiIndex)

    def winsorize_and_norm(x):
        normal_scaler = preprocessing.StandardScaler()
        x_w = winsorize(x, limits=limits, nan_policy='omit')
        x_scaled = normal_scaler.fit_transform(x_w.reshape(-1, 1))
        return pd.Series(x_scaled[:, 0], x.index)

    return series.groupby(level=0).apply(winsorize_and_norm)


def industry_neutralize(df: pd.DataFrame,
                        neutralize_factor_name: str,
                        group_col: str,
                        method: str = 'norm',
                        winsorize_limits: None or int or tuple = None) -> pd.Series:
    """

    :param df:
    :param neutralize_factor_name:
    :param group_col:
    :param method:
    :param winsorize_limits:
    :return:
    """
    methods = ['norm', 'demean', 'winsorize', 'winsorize_norm']
    if method not in methods:
        raise ValueError('method must be one of {}, but {} is given.'.format(methods, method))

    if group_col not in df.columns:
        raise KeyError('{} not in df.columns {}, industry label must in columns'.format(group_col, df.columns))

    if (method == 'winsorize' or method == 'winsorize_norm') and winsorize_limits is None:
        raise ValueError('{} must fill'.format(winsorize_limits))

    grouper = [group_col, df.index.get_level_values(0)]

    def neutralize(x: pd.DataFrame):
        if method == 'norm':
            neu_factor = (x[neutralize_factor_name] - x[neutralize_factor_name].mean()) / x[
                neutralize_factor_name].std()
            neu_factor = neu_factor.to_frame()
        elif method == 'demean':
            neu_factor = x[neutralize_factor_name] - x[neutralize_factor_name].mean()
            neu_factor = neu_factor.to_frame()
        elif method == 'winsorize':
            neu_factor = winsorize(x[neutralize_factor_name], limits=winsorize_limits, nan_policy='omit')
            neu_factor = pd.Series(neu_factor, index=x.index, name=neutralize_factor_name)
        elif method == 'winsorize_norm':
            neu_factor = winsorize(x[neutralize_factor_name], limits=winsorize_limits, nan_policy='omit')
            neu_factor = pd.Series(neu_factor, index=x.index, name=neutralize_factor_name)
            neu_factor = (neu_factor - neu_factor.mean()) / neu_factor.std()
            neu_factor = neu_factor.to_frame()
        else:
            raise NotImplementedError
        return neu_factor

    df = df.groupby(by=grouper).apply(neutralize)[neutralize_factor_name]
    return df


def cap_industry_neutralize(df: pd.DataFrame,
                            neutralize_factor_name: str,
                            industry_col: str,
                            cap_col: str,
                            method: str = 'norm',
                            winsorize_limits: None or int or tuple = None) -> pd.Series:
    df = df.copy()
    methods = ['norm', 'winsorize', 'winsorize_norm']
    if method not in methods:
        raise ValueError('method must be one of {}, but {} is given.'.format(methods, method))

    if industry_col not in df.columns:
        raise KeyError('{} not in df.columns {}, industry label must in columns'.format(industry_col, df.columns))

    if (method == 'winsorize' or method == 'winsorize_norm') and winsorize_limits is None:
        raise ValueError('{} must fill'.format(winsorize_limits))

    onehot_model = OneHotEncoder()
    df[industry_col] = df[industry_col].fillna('unknown_industry')
    data = onehot_model.fit_transform(df[[industry_col]]).toarray()
    industry_df = pd.DataFrame(data, index=df.index, columns=onehot_model.categories_[0])
    df[cap_col] = np.log(df[cap_col])
    df = df.join(industry_df)
    regressor = industry_df.columns.to_list() + [cap_col]

    def neutralize(x: pd.DataFrame):
        if method == 'norm':
            f = (x[neutralize_factor_name] - x[neutralize_factor_name].mean()) / x[
                neutralize_factor_name].std()
        elif method == 'winsorize':
            f = winsorize(x[neutralize_factor_name], limits=winsorize_limits, nan_policy='omit').data
        elif method == 'winsorize_norm':
            neu_factor = winsorize(x[neutralize_factor_name], limits=winsorize_limits, nan_policy='omit').data
            f = (neu_factor - neu_factor.mean()) / neu_factor.std()
        else:
            raise NotImplementedError
        f = np.nan_to_num(f, nan=0)
        x[cap_col] = (x[cap_col] - x[cap_col].mean()) / x[cap_col].std()
        x[cap_col] = x[cap_col].fillna(0)
        ols = LinearRegression().fit(x[regressor], f)
        res = f - ols.predict(x[regressor])
        res = pd.Series(res, index=x.index.get_level_values(1), name=neutralize_factor_name)
        return res

    series = df.groupby(level=0).apply(neutralize)
    return series


def minmax_factor(series: pd.Series):
    """

    :param series:
    :return:
    """
    assert isinstance(series.index, pd.MultiIndex)

    def sklearn_minmax(x):
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x.values.reshape(-1, 1))
        return pd.Series(x_scaled[:, 0], x.index)

    return series.groupby(level=0).apply(sklearn_minmax)


def factor_pca(factors: pd.DataFrame, n_components: int or float or None or str = 0.8,
               svd_solver: str = 'auto') -> pd.DataFrame:
    """

    :param factors:
    :param n_components:
    :param svd_solver:
    :return:
    """
    assert isinstance(factors.index, pd.MultiIndex)

    def sklearn_pca(x):
        pca = PCA(n_components, svd_solver=svd_solver)
        x_pca = pca.fit_transform(x)
        return pd.DataFrame(x_pca, index=x.index)

    return factors.groupby(level=0).apply(sklearn_pca)


# def factors_score(factors: pd.DataFrame, factors_names: list or None = None, normalized=False):
#     """
#     Mainly for multiple financial statement indicator
#     :param factors:
#     :param factors_names:
#     :param normalized:
#     :return:
#     """
#     if factors_names is None:
#         factors_names = factors.columns
#
#     factors_rank = factors.copy()[factors_names]
#     for name in factors_names:
#         factors_rank[name] = rank(factors_rank[name])
#
#     score = factors_rank.sum(axis=1)
#     if normalized:
#         score = minmax_factor(score)
#     return score


def rolling_minmax_factor(series: pd.Series, d: int, scale=1.0, center=True):
    """
    mainly use in the technical indicator
    :param series:
    :return:
    """
    assert isinstance(series.index, pd.MultiIndex)

    def minmax_center(x):
        maxx = np.max(x)
        minx = np.min(x)
        return scale * ((x[-1] - minx) / (maxx - minx) - 0.5)

    def minmax(x):
        maxx = np.max(x)
        minx = np.min(x)
        return scale * (x[-1] - minx) / (maxx - minx)

    if center:
        return series.groupby(level=1).apply(lambda x: x.rolling(d).apply(minmax_center))
    else:
        return series.groupby(level=1).apply(lambda x: x.rolling(d).apply(minmax))


def mid_de_extreme(series: pd.Series, n=5):
    D_M = series.median()
    D_M1 = (series - D_M).abs().median()

    def replace(x):
        if x > D_M + n * D_M1:
            return D_M + n * D_M1
        elif x < D_M - n * D_M1:
            return D_M - n * D_M1
        else:
            return x

    series = series.apply(replace)
    return series


def median_absolute_deviation(factor: pd.Series, n):
    return factor.groupby(level=0).apply(lambda x: mid_de_extreme(x, n))


def mv_industry_neutralization(factor: pd.Series, market_cap: pd.Series, industry_dummy: pd.DataFrame):
    regressor = ['market_cap'] + industry_dummy.columns.to_list()

    def ind_size_neutralization(x: pd.DataFrame):
        ols = LinearRegression().fit(x[regressor], x['factor'])
        res = x['factor'] - ols.predict(x[regressor])
        return res

    factor = factor.unstack()
    industry_dummy = industry_dummy.unstack()
    market_cap = market_cap.unstack()
    idx = factor.index.intersection(industry_dummy.index).intersection(market_cap.index)
    industry_dummy = industry_dummy.loc[idx].stack()
    market_cap = market_cap.loc[idx].stack()
    factor = factor.loc[idx].stack()

    market_cap = np.log(market_cap)
    market_cap = median_absolute_deviation(market_cap, 3.1483)
    market_cap = normalized_factor(market_cap)
    factors = factor.to_frame('factor').join(market_cap.to_frame('market_cap')).join(industry_dummy).dropna()
    tqdm.pandas()
    f = factors.groupby(level=0).progress_apply(ind_size_neutralization).droplevel(0).sort_index()
    f = f.round(4)
    return f


def huatai_factor_transformation(factors: pd.DataFrame, factor_name: str, industry_col: str = 'industry_code',
                                 size_col: str = 'size'):
    """
    2． 特征提取和预处理：
    1) 每个自然月的最后一个交易日，计算82个因子暴露度，作为样本的原始特征，
    因子池如图表10 和图表11 所示。
    2) 中位数去极值：设第T期某因子在所有个股上的暴露度序列为𝐷𝑖，𝐷𝑀为该序列
    中位数，𝐷𝑀1为序列|𝐷𝑖 − 𝐷𝑀|的中位数，则将序列𝐷𝑖中所有大于𝐷𝑀 + 5𝐷𝑀1的数
    重设为𝐷𝑀 + 5𝐷𝑀1，将序列𝐷𝑖中所有小于𝐷𝑀 − 5𝐷𝑀1的数重设为𝐷𝑀 − 5𝐷𝑀1；
    3) 缺失值处理：得到新的因子暴露度序列后，将因子暴露度缺失的地方设为中信一
    级行业相同个股的平均值；
    4) 行业市值中性化：将填充缺失值后的因子暴露度对行业哑变量和取对数后的市值
    做线性回归，取残差作为新的因子暴露度；
    5) 标准化：将中性化处理后的因子暴露度序列减去其现在的均值、除以其标准差，
    得到一个新的近似服从N(0, 1)分布的序列。
    :param factors:
    :param factor_name:
    :param industry_col:
    :param size_col:
    :return:
    """
    industry_dummy_col = [c for c in factors.columns if len(re.findall('(\d{6}){1}|([A-Z]{1}\d{2}){1}', c))]
    regressor = industry_dummy_col + [size_col]
    factors = factors[[factor_name, industry_col] + regressor]

    #  中位数去极值
    factors[factor_name] = factors[factor_name].groupby(level=0).apply(mid_de_extreme)

    # 缺失值处理
    factors = factors.set_index(industry_col, append=True)
    factors[factor_name] = factors[factor_name].groupby(level=[0, 2]).apply(lambda x: x.fillna(x.mean()))
    ## drop nan in the sample
    ## The reason comes from industry dummy and size factor
    idx = factors.index.droplevel(-1)
    factors = factors.dropna()

    # 行业市值中性化
    def ind_size_neutralization(x: pd.DataFrame):
        # X = sm.add_constant(x[regressor].values)  # constant is not added by default
        # model = sm.OLS(x[factor_name].values, X, missing='none')
        # result = model.fit()
        #
        # return result.resid
        # the size and dummy cannot have nan!
        ols = LinearRegression().fit(x[regressor], x[factor_name])
        res = x[factor_name] - ols.predict(x[regressor])
        return res

    f = factors.groupby(level=0).apply(ind_size_neutralization).droplevel([0, -1])

    # 标准化
    f = normalized_factor(f)  # type: pd.Series
    f = f.reindex(idx).sort_index()
    return f


def cosine_transform(factors: pd.Series, method='minmax'):
    tqdm.pandas()

    def cos_f(x: pd.Series):
        if method == 'minmax':
            x_min = x.min()
            x_max = x.max()
            x_ = 2 * ((x - x_min) / (x_max - x_min) - 0.5)
            x_ = np.pi * x_
            return np.cos(x_)
        else:
            raise NotImplementedError

    f = factors.groupby(level=1).progress_apply(cos_f)
    f.name = 'cos_trans({},{})'.format(factors.name, method)
    return f


def sine_transform(factors: pd.Series, method='minmax'):
    tqdm.pandas()

    def cos_f(x: pd.Series):
        if method == 'minmax':
            x_min = x.min()
            x_max = x.max()
            x_ = 2 * ((x - x_min) / (x_max - x_min) - 0.5)
            x_ = np.pi * x_
            return np.sin(x_)
        else:
            raise NotImplementedError

    f = factors.groupby(level=1).progress_apply(cos_f)
    f.name = 'sin_trans({},{})'.format(factors.name, method)
    return f


if __name__ == '__main__':
    local_cfg_path = '../cfg/factor_keeper_setting.ini'
    keeper = ZooKeeper(local_cfg_path)
    value, _ = keeper.get_factor_values('idiosyncratic', 'fama_french_hotpot_120')
    f = cosine_transform(value)
