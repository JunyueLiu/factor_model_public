from abc import abstractmethod
from enum import Enum
from functools import partial
from typing import Tuple, Union, Dict, List

import bottleneck as bn
import numpy as np
import pandas as pd
import tqdm
from scipy.stats.mstats_basic import winsorize
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from data_management.dataIO import market_data
from data_management.dataIO.component_data import IndustryCategory, get_industry_component, get_index_component, \
    IndexTicker, get_north_connect_component
from data_management.dataIO.fundamental_data import get_stock_info
from data_management.dataIO.market_data import get_limit_and_paused
from data_management.dataIO.trade_data import get_trade, TradeTable
from data_management.pandas_utils.cache import series_cross_sectional_apply, series_time_series_apply, \
    panel_df_join
from factor_circus import FactorProcesser
from factor_zoo.industry import industry_category
from factor_zoo.size import float_size_2


class FactorScaler(FactorProcesser):

    @abstractmethod
    def fit_transform(self, X, y=None, **fit_params):
        pass


class FactorPadder(FactorProcesser):
    @abstractmethod
    def fit_transform(self, X, y=None, **fit_params):
        pass


class FactorNANProcesser(FactorProcesser):
    @abstractmethod
    def fit_transform(self, X, y=None, **fit_params):
        pass


class UniverseSelector(FactorProcesser):
    def __init__(self):
        self.component = None
        self.universe_name = ''

    def filter_in(self, X):
        """
        This is same with factor_zoo/utils
        :param X:
        :return:
        """
        if isinstance(self.component, list):
            universe_tuple = [(k, c) for un in self.component for k, v in un.items() for c in v if
                              len(v) > 0]
        else:
            universe_tuple = [(k, c) for k, v in self.component.items() for c in v if len(v) > 0]
        idx = pd.MultiIndex.from_tuples(universe_tuple, names=X.index.names).drop_duplicates()
        tickers = X.index.get_level_values(1).drop_duplicates().sort_values()
        factor_name = X.name
        masked = pd.Series(1, idx).unstack().reindex(columns=tickers)
        data = (X.unstack() * masked).stack()
        # data = data.reindex(idx)
        data.name = factor_name
        data.index.names = X.index.names
        data.name = factor_name + '_' + self.universe_name
        return data

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        return self.filter_in(X)


# ===========================================================
# FactorScaler
# ===========================================================

class WinsorizationFactorScaler(FactorScaler):
    def __init__(self, limits: Union[Tuple[float, float], float]):
        self.limits = limits

    def _scipy_winsorize(self, X):
        x_w = winsorize(X, limits=self.limits, nan_policy='omit')
        return pd.Series(x_w, X.index)

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        return series_cross_sectional_apply(X, self._scipy_winsorize).sort_index()


class MidDeExtremeFactorScaler(FactorScaler):
    """
    中位数去极值：设第T期某因子在所有个股上的暴露度序列为𝐷𝑖，𝐷𝑀为该序列
    中位数，𝐷𝑀1为序列|𝐷𝑖 − 𝐷𝑀|的中位数，则将序列𝐷𝑖中所有大于𝐷𝑀 + 5𝐷𝑀1的数
    重设为𝐷𝑀 + 5𝐷𝑀1，将序列𝐷𝑖中所有小于𝐷𝑀 − 5𝐷𝑀1的数重设为𝐷𝑀 − 5𝐷𝑀1；

    """

    def __init__(self, multiplier):
        self.multiplier = multiplier

    def _mid_de_extreme(self, series: pd.Series):
        D_M = series.median()
        D_M1 = (series - D_M).abs().median()
        upper = D_M + self.multiplier * D_M1
        lower = D_M - self.multiplier * D_M1
        s = np.where(series > upper, upper,
                     np.where(series < lower, lower, series),
                     )
        return s

    def fit_transform(self, X, y=None, **fit_params):
        f = X.groupby(level=0).transform(self._mid_de_extreme)
        return f


class StandardFactorScaler(FactorScaler):

    def _sklearn_norm(self, X):
        normal_scaler = preprocessing.StandardScaler()
        x_scaled = normal_scaler.fit_transform(X.values.reshape(-1, 1))
        return pd.Series(x_scaled[:, 0], X.index)

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        return series_cross_sectional_apply(X, self._sklearn_norm).sort_index()


class MinMaxFactorScaler(FactorScaler):
    def _sklearn_minmax(self, X):
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(X.values.reshape(-1, 1))
        return pd.Series(x_scaled[:, 0], X.index)

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        return series_cross_sectional_apply(X, self._sklearn_minmax).sort_index()


class IndNeuFactorScaler(FactorScaler):
    class Method(Enum):
        DemeanScaler = 'Demean'
        StandardScaler = 'Standard'
        WinsorizationScaler = 'Winsorization'
        WinsorizationStandardScaler = 'Winsorization_Standard'

    def __init__(self, method: Method,
                 industry_data_input_path,
                 ind_category: IndustryCategory,
                 winsorize_limits=None):
        self.method = method
        self.factor_name = None
        self.industry_data_input_path = industry_data_input_path
        self.ind_category = ind_category
        self.winsorize_limits = winsorize_limits
        if self.method in [self.Method.WinsorizationScaler, self.Method.WinsorizationStandardScaler]:
            assert winsorize_limits is not None
        industry_dict = get_industry_component(ind_category, config_path=industry_data_input_path)
        self.cat = industry_category(industry_dict).astype(str)
        self.cat.name = self.ind_category.name

    def _neutralize(self, X: pd.Series, **paras):
        if self.method == self.Method.StandardScaler:
            neu_factor = (X - X.mean()) / X.std()
        elif self.method == self.Method.DemeanScaler:
            neu_factor = X - X.mean()
        elif self.method == self.Method.WinsorizationScaler:
            winsorize_limits = paras['winsorize_limits']
            neu_factor = winsorize(X, limits=winsorize_limits, nan_policy='omit')
        elif self.method == self.Method.WinsorizationStandardScaler:
            winsorize_limits = paras['winsorize_limits']
            neu_factor = winsorize(X, limits=winsorize_limits, nan_policy='omit')
            neu_factor = (neu_factor - np.nanmean(neu_factor)) / np.nanstd(neu_factor)
        else:
            raise NotImplementedError
        return neu_factor

    def fit_transform(self, X: pd.Series, y=None,
                      **fit_params):
        # todo to test it
        self.factor_name = X.name
        # factor_data = X.to_frame().join(self.cat)
        factor_data = panel_df_join(X.to_frame(), self.cat.to_frame())
        grouper = [self.ind_category.name, factor_data.index.get_level_values(0)]
        func = partial(self._neutralize, winsorize_limits=self.winsorize_limits)
        f = factor_data.groupby(by=grouper)[self.factor_name].transform(func)
        return f


class CapIndNeuFactorScaler(FactorScaler):
    class Method(Enum):
        StandardScaler = 'Standard'
        WinsorizationScaler = 'Winsorization'
        WinsorizationStandardScaler = 'Winsorization_Standard'

    def __init__(self, method: Method,
                 industry_data_input_path,
                 ind_category: IndustryCategory,
                 winsorize_limits=None):
        self.method = method
        self.factor_name = None
        self.industry_data_input_path = industry_data_input_path
        self.ind_category = ind_category
        self.winsorize_limits = winsorize_limits
        if self.method in [self.Method.WinsorizationScaler, self.Method.WinsorizationStandardScaler]:
            assert winsorize_limits is not None
        industry_dict = get_industry_component(ind_category, config_path=industry_data_input_path)
        self.cat = industry_category(industry_dict).astype(str)
        self.cat.name = self.ind_category.name
        self.daily_basic = get_trade(TradeTable.daily_basic, cols=['circ_mv'], config_path=industry_data_input_path)

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        neutralize_factor_name = X.name
        size = float_size_2(self.daily_basic)
        cap_col = size.name
        data = panel_df_join(X.to_frame(), size.to_frame())
        data = panel_df_join(data, self.cat.to_frame())

        onehot_model = OneHotEncoder()
        data[self.cat.name] = data[self.cat.name].fillna('unknown_industry')
        df = onehot_model.fit_transform(data[[self.cat.name]]).toarray()
        industry_df = pd.DataFrame(df, index=data.index, columns=onehot_model.categories_[0])
        regressor = industry_df.columns.to_list() + [size.name]
        data = panel_df_join(data, industry_df)

        def neutralize(x: pd.DataFrame):
            if self.method == self.Method.StandardScaler:
                f = (x[neutralize_factor_name] - x[neutralize_factor_name].mean()) / x[neutralize_factor_name].std()
            elif self.method == self.Method.WinsorizationScaler:
                f = winsorize(x[neutralize_factor_name], limits=self.winsorize_limits, nan_policy='omit').data
            elif self.method == self.Method.WinsorizationStandardScaler:
                neu_factor = winsorize(x[neutralize_factor_name], limits=self.winsorize_limits, nan_policy='omit').data
                f = (neu_factor - np.nanmean(neu_factor)) / np.nanstd(neu_factor)
            else:
                raise NotImplementedError
            f = np.nan_to_num(f, nan=0)
            x[cap_col] = (x[cap_col] - x[cap_col].mean()) / x[cap_col].std()
            x[cap_col] = x[cap_col].fillna(0)
            ols = LinearRegression().fit(x[regressor], f)
            res = f - ols.predict(x[regressor])
            res = pd.Series(res, index=x.index.get_level_values(1), name=neutralize_factor_name)
            return res

        tqdm.tqdm.pandas()
        series = data.groupby(level=0).progress_apply(neutralize)
        series.name = neutralize_factor_name
        return series


class RollingMaxAbsFactorScaler(FactorScaler):

    def __init__(self, d):
        self.d = d

    def _minmax(self, X):
        maxx = X.rolling(self.d).max()
        minx = X.rolling(self.d).min()
        return 2 * (X - minx) / (maxx - minx) - 1

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        return series_time_series_apply(X, self._minmax)


class RankScoreFactorScaler(FactorScaler):

    def __init__(self):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        # todo to test it
        self.input_check(X)
        unstack_X = X.unstack()
        unstack_X = unstack_X.astype(float)
        rank = bn.nanrankdata(unstack_X, axis=1)
        rank = pd.DataFrame(rank, index=unstack_X.index, columns=unstack_X.columns)
        rank = rank.div(rank.max(axis=1), axis=0)
        rank = rank.stack().sort_index()
        rank.index.names = X.index.names
        rank.name = X.name
        return rank


class IndRankScoreFactorScaler(FactorScaler):

    def __init__(self,
                 industry_data_input_path,
                 ind_category: IndustryCategory):
        self.industry_data_input_path = industry_data_input_path
        self.ind_category = ind_category
        industry_dict = get_industry_component(ind_category, config_path=industry_data_input_path)
        self.cat = industry_category(industry_dict).astype(str)
        self.cat.name = self.ind_category.name

    def fit_transform(self, X, y=None, **fit_params):
        # there is problem of only one stock in the industries
        self.input_check(X)
        factor_name = X.name
        X = panel_df_join(X.to_frame(), self.cat.to_frame()).dropna()
        unstack_X = X[factor_name].unstack()
        idx = unstack_X.index
        cols = unstack_X.columns
        unstack_X = unstack_X.values
        new_X = np.zeros(unstack_X.shape)
        g = X[self.cat.name].unstack().values

        ind = np.unique(g.tolist())
        for i in range(len(ind)):
            a = np.where(g == ind[i], 1, np.nan)
            rank = bn.nanrankdata(unstack_X * a, axis=1)
            max_rank = np.nanmax(rank, axis=1)
            norm_rank = np.nan_to_num((rank.T / max_rank).T)
            new_X += norm_rank
        f = pd.DataFrame(new_X, idx, cols).stack().sort_index()
        f.name = factor_name
        return f


class ScoreFactorScaler(FactorScaler):

    def __init__(self, n):
        self.n = n

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        f = X.groupby(level=0).transform(
            lambda x: pd.qcut(x, self.n, labels=list(range(self.n)), duplicates='drop')).astype(float)
        # f.name = X.name
        return f


# ===========================================================
# FactorPadder
# ===========================================================
class FactorForwardPadder(FactorPadder):

    def __init__(self, offset):
        self.offset = offset

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        name = X.name
        f_ = X.unstack()
        f_ = f_.fillna(np.Inf).resample(self.offset).last().fillna(method='ffill')
        f_ = f_[f_ < np.Inf]
        f0 = f_.stack()
        f0.name = name
        return f0


class FactorDecayPadder(FactorPadder):

    def __init__(self, half_life_days):
        self.half_life_days = half_life_days

    # def _linear_decay_algo(self, x: pd.Series):
    #     name = x.name
    #     x = x.to_frame()
    #     x['dt'] = x.index.get_level_values(0)
    #     x['diff'] = x['dt'] - x['dt'].shift(1)
    #     x['diff'] = x.apply(lambda e: e['diff'].days if pd.isna(e[name]) else np.nan, axis=1)
    #     x['decay_factor'] = (2 ** (-x['diff'] / self.half_life_days))
    #     x['g'] = pd.isna(x['decay_factor']).cumsum()
    #     x['decay_factor'] = x['decay_factor'].fillna(1)
    #     x['decay_factor'] = x.groupby(by='g')['decay_factor'].cumprod()
    #     x[name] = x[name].fillna(method='ffill')
    #     x[name] = x[name] * x['decay_factor']
    #     return x[name]

    def fit_transform(self, X: pd.Series, y=None, **fit_params):
        self.input_check(X)
        unstack_X = X.unstack()
        nans = unstack_X.isna()
        decay_factor = nans.cumsum() - nans.cumsum().where(~nans).ffill().fillna(0).astype(int)
        decay_factor = 2 ** (-decay_factor / self.half_life_days)
        filled = unstack_X.fillna(method='ffill')
        df = filled * decay_factor
        series = df.stack()
        series.index.names = X.index.names
        series.name = X.name
        return series


# ===========================================================
# FactorStats
# ===========================================================
class FactorMeanProcesser(FactorProcesser):

    def fit_transform(self, X, y=None, **fit_params):
        factor_name = X.name
        f = X.groupby(level=0).transform(lambda x: x.mean())
        f.name = '{}_mean'.format(factor_name)
        return f


class FactorMedianProcesser(FactorProcesser):

    def fit_transform(self, X, y=None, **fit_params):
        factor_name = X.name
        f = X.groupby(level=0).transform(lambda x: x.median())
        f.name = '{}_median'.format(factor_name)
        return f


class FactorStdProcesser(FactorProcesser):

    def fit_transform(self, X, y=None, **fit_params):
        factor_name = X.name
        f = X.groupby(level=0).transform(lambda x: x.std())
        f.name = '{}_std'.format(factor_name)
        return f


class FactorSkewProcesser(FactorProcesser):

    def fit_transform(self, X, y=None, **fit_params):
        factor_name = X.name
        f = X.groupby(level=0).transform(lambda x: x.skew())
        f.name = '{}_skew'.format(factor_name)
        return f


class FactorKurtProcesser(FactorProcesser):

    def fit_transform(self, X, y=None, **fit_params):
        factor_name = X.name
        f = X.groupby(level=0).transform(lambda x: x.kurtosis())
        f.name = '{}_skew'.format(factor_name)
        return f


# ===========================================================
# Factor lag factor
# ===========================================================
class FactorLagProcesser(FactorProcesser):

    def __init__(self, lag: int) -> None:
        self.lag = lag

    def fit_transform(self, X, y=None, **fit_params):
        factor_name = X.name
        f = X.groupby(level=1).shift(self.lag)
        f.name = '{}_lag_{}'.format(factor_name, self.lag)
        return f


# ===========================================================
# FactorNANProcesser
# ===========================================================


class LimitNanProcesser(FactorNANProcesser):

    def __init__(self, limit_data_input_path, start_date=None, end_date=None):
        """

        :param limit_data_input_path:
        :param start_date:
        :param end_date:
        """
        limit_and_paused = get_limit_and_paused(
            start_date=start_date, end_date=end_date,
            config_path=limit_data_input_path)
        self.limit_and_paused = limit_and_paused.replace(1, np.nan).replace(0, 1)

    def fit_transform(self, X: pd.Series, y=None, **fit_params):
        self.input_check(X)
        merged = X.to_frame().join(self.limit_and_paused)
        res = merged.prod(axis=1, skipna=False)
        res.name = X.name
        return res


class FillNanIndMedianProcesser(FactorNANProcesser):
    def __init__(self, industry_data_input_path,
                 ind_category: IndustryCategory):
        industry_dict = get_industry_component(ind_category, config_path=industry_data_input_path)
        self.cat = industry_category(industry_dict).astype(str)
        self.factor_name = None

    def _fillna_with_industry_median(self, X):
        return X[self.factor_name].fillna(X[self.factor_name].median())

    def fit_transform(self, X: pd.Series, y=None, **fit_params):
        self.factor_name = X.name
        if not X.isna().any():
            return X
        factor_data = panel_df_join(X.to_frame(), self.cat.to_frame())
        grouper = [self.cat.name, factor_data.index.get_level_values(0)]
        f = factor_data.groupby(by=grouper)[self.factor_name] \
            .transform(lambda x: x.fillna(x.median()))
        f.name = self.factor_name
        return f


class FillNanIndMeanProcesser(FactorNANProcesser):
    def __init__(self, industry_data_input_path,
                 ind_category: IndustryCategory):
        industry_dict = get_industry_component(ind_category, config_path=industry_data_input_path)
        self.cat = industry_category(industry_dict).astype(str)
        self.factor_name = None

    # def _fillna_with_industry_mean(self, X):
    #     return X[self.factor_name].fillna(X[self.factor_name].mean())

    def fit_transform(self, X: pd.Series, y=None, **fit_params):
        self.factor_name = X.name
        factor_data = X.to_frame().join(self.cat)
        grouper = [self.cat.name, factor_data.index.get_level_values(0)]
        f = factor_data.groupby(by=grouper)[self.factor_name] \
            .transform(lambda x: x.fillna(x.mean()))
        f.name = self.factor_name
        return f


class FillNanMeanProcesser(FactorNANProcesser):
    def __init__(self, industry_data_input_path,
                 ind_category: IndustryCategory):
        industry_dict = get_industry_component(ind_category, config_path=industry_data_input_path)
        self.cat = industry_category(industry_dict).astype(str)
        self.factor_name = None

    def _fillna_with_mean(self, x):
        return x.fillna(x.mean())

    def fit_transform(self, X: pd.Series, y=None, **fit_params):
        self.factor_name = X.name
        f = X.groupby(level=0).transform(lambda x: x.fillna(x.mean()))
        return f


class FillNanMedianProcesser(FactorNANProcesser):
    def __init__(self):
        self.factor_name = None

    # def _fillna_with_median(self, x):
    #     return x.fillna(x.median())

    def fit_transform(self, X: pd.Series, y=None, **fit_params):
        self.factor_name = X.name
        if not X.isna().any():
            return X
        f = X.groupby(level=0).transform(lambda x: x.fillna(x.median()))
        return f


class FillNanIndMedianStandardizedProcesser(FactorProcesser):
    def __init__(self, industry_data_input_path,
                 ind_category: IndustryCategory):
        industry_dict = get_industry_component(ind_category, config_path=industry_data_input_path)
        self.cat = industry_category(industry_dict).astype(str)
        self.factor_name = None

        # def _fillna_with_industry_mean(self, X):
        #     return X[self.factor_name].fillna(X[self.factor_name].mean())

    def _fillna_standardized(self, x):
        x = x.fillna(x.median())
        return (x - x.mean()) / x.std()

    def fit_transform(self, X: pd.Series, y=None, **fit_params):
        self.factor_name = X.name
        factor_data = panel_df_join(X.to_frame(), self.cat.to_frame())
        factor_data = factor_data[~factor_data[self.factor_name].isna()]
        grouper = [self.cat.name, factor_data.index.get_level_values(0)]
        f = factor_data.groupby(by=grouper)[self.factor_name] \
            .transform(lambda x: x.fillna(x.mean()))
        f.name = self.factor_name
        return f


class CategoryProcesser(FactorProcesser):

    def fit_transform(self, X, y=None, **fit_params):
        X = X.astype('category')
        X = X.cat.codes
        return X


# ===========================================================
# UniverseSelector
# Will drop the non-universe samples
# ===========================================================

class SZ50Selector(UniverseSelector):
    def __init__(self, universe_data_input_path: str):
        super().__init__()
        self.component = get_index_component(IndexTicker.sz50, config_path=universe_data_input_path)
        self.universe_name = 'sz50'

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        return self.filter_in(X)


class SZR100Selector(UniverseSelector):
    def __init__(self, universe_data_input_path: str):
        super().__init__()
        self.component = get_index_component(IndexTicker.szr100, config_path=universe_data_input_path)
        self.universe_name = 'szr100'

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        return self.filter_in(X)


class CSI300Selector(UniverseSelector):

    def __init__(self, universe_data_input_path: str):
        super().__init__()
        self.component = get_index_component(IndexTicker.csi300, config_path=universe_data_input_path)
        self.universe_name = 'csi300'

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        return self.filter_in(X)


class CSI300_SZZR100Selector(UniverseSelector):

    def __init__(self, universe_data_input_path: str):
        super().__init__()
        csi = get_index_component(IndexTicker.csi300, config_path=universe_data_input_path)
        szr100 = get_index_component(IndexTicker.szr100, config_path=universe_data_input_path)
        self.component = [csi, szr100]

        self.universe_name = 'csi300_szr100'


class ZZ500Selector(UniverseSelector):

    def __init__(self, universe_data_input_path: str):
        super().__init__()
        self.component = get_index_component(IndexTicker.zz500, config_path=universe_data_input_path)
        self.universe_name = 'zz500'

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        return self.filter_in(X)


class ZZ800Selector(UniverseSelector):

    def __init__(self, universe_data_input_path: str):
        super().__init__()
        self.component = get_index_component(IndexTicker.zz800, config_path=universe_data_input_path)
        self.universe_name = 'zz800'

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        return self.filter_in(X)


class ZZ1000Selector(UniverseSelector):

    def __init__(self, universe_data_input_path: str):
        super().__init__()
        self.component = get_index_component(IndexTicker.zz1000, config_path=universe_data_input_path)
        self.universe_name = 'zz1000'

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        return self.filter_in(X)


class MarketConnectSelector(UniverseSelector):

    def __init__(self, universe_data_input_path: str):
        super().__init__()
        self.component = get_north_connect_component(config_path=universe_data_input_path)
        self.universe_name = 'market_connect'

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        return self.filter_in(X)


class ListedSelector(UniverseSelector):
    def __init__(self, universe_data_input_path: str, days: int):
        super().__init__()
        self.info = get_stock_info(universe_data_input_path)
        self.universe_name = 'listed_{}'.format(days)
        self.days = days

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        listed_time = self.info['start_date']
        listed_time.index.names = ['code']
        merged = X.to_frame().join(listed_time)
        merged['days'] = merged.index.get_level_values(0) - merged['start_date']
        merged = merged[merged['days'].dt.days > self.days]
        return merged[X.name].sort_index()


class MarketCapSelector(UniverseSelector):
    def __init__(self, universe_data_input_path: str, top_quantile: float):
        super().__init__()
        self.daily_basic = get_trade(TradeTable.daily_basic, config_path=universe_data_input_path,
                                     cols=['total_mv', 'circ_mv'])
        self.universe_name = 'market_cap_top_{}'.format(top_quantile)
        self.top_quantile = top_quantile

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        unstack = self.daily_basic['circ_mv'].unstack()
        q = unstack.quantile(self.top_quantile, axis=1)
        masked = unstack.ge(q, axis=0).astype(int).replace(0, np.nan)
        s = (X.unstack() * masked).stack().sort_index()
        s.name = X.name
        return s


class CryptoVolumeSelector(UniverseSelector):
    def __init__(self, bar_data: pd.DataFrame, top_quantile: float, rolling_n=1):
        super().__init__()
        self.money = bar_data['money']
        self.universe_name = 'volume_top_{}'.format(top_quantile)
        self.top_quantile = top_quantile
        self.rolling_n = rolling_n

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        unstack = self.money.unstack()
        unstack = unstack.rolling(self.rolling_n).sum()
        q = unstack.quantile(self.top_quantile, axis=1)
        masked = unstack.ge(q, axis=0).astype(int).replace(0, np.nan)
        s = (X.unstack() * masked).stack().sort_index()
        s.name = X.name
        return s


class BlackListSelector(UniverseSelector):
    def __init__(self, black_list):
        super(BlackListSelector, self).__init__()
        self.black_list = black_list

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        return X[~X.index.get_level_values(1).isin(self.black_list)]


class PriceSelector(UniverseSelector):
    def __init__(self, universe_data_input_path: str, price_low_bound: float, price_upper_bound: float):
        super().__init__()
        self.close = market_data.get_bars(config_path=universe_data_input_path,
                                          cols=('close',),
                                          eod_time_adjust=False,
                                          add_limit=False,
                                          adjust=False)
        self.universe_name = 'close_between_{}_{}'.format(price_low_bound, price_upper_bound)
        self.price_low_bound = price_low_bound
        self.price_upper_bound = price_upper_bound

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        masked = self.close['close'] \
            .between(self.price_low_bound, self.price_upper_bound).astype(int).replace(0, np.nan).unstack()
        s = (X.unstack() * masked).stack().sort_index()
        s.name = X.name
        return s


class PausedSelector(UniverseSelector):
    def __init__(self, universe_data_input_path: str, lookback: int, minimum_trading_days: int):
        super(PausedSelector, self).__init__()
        self.paused = market_data.get_limit_and_paused(config_path=universe_data_input_path)['paused'].unstack()
        self.lookback = lookback
        self.minimum_trading_days = minimum_trading_days

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        sum_paused_days = self.paused.rolling(self.lookback).sum()
        masked = sum_paused_days <= (self.lookback - self.minimum_trading_days)
        s = (X.unstack() * masked).stack().sort_index()
        s.name = X.name
        return s


class NotSTSelector(UniverseSelector):
    def __init__(self, universe_data_input_path):
        super(NotSTSelector, self).__init__()
        self.st = get_trade(TradeTable.ST, config_path=universe_data_input_path)['is_st']

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        masked = self.st.unstack()
        masked = masked.replace(True, np.nan).replace(False, 1)
        s = (X.unstack() * masked).stack().sort_index()
        s.name = X.name
        return s


class CustomizedUniverseSelector(UniverseSelector):

    def __init__(self, universe: Union[pd.MultiIndex, Dict[pd.Timestamp, List[str]]]):
        super().__init__()
        self.universe = universe

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        X_unstack = X.unstack()
        idx = X_unstack.index
        if isinstance(self.universe, pd.MultiIndex):
            masked = pd.Series(1, self.universe).unstack().fillna(0)
            masked = masked.reindex(idx.union(masked.index))
            masked = masked.fillna(method='ffill').replace(0, np.nan)
            s = (X_unstack * masked).stack().sort_index()
            s.name = X.name
            return s
        else:
            raise NotImplementedError


class CryptoUniverseSelector(UniverseSelector):
    def __init__(self, universe: List[str]):
        super().__init__()
        self.universe = universe

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        X_unstack = X.unstack()
        selected = X_unstack.columns.intersection(self.universe)
        X_unstack = X_unstack[selected]
        s = X_unstack.stack().sort_index()
        s.name = X.name
        return s


class CryptoPriceSelector(UniverseSelector):
    def __init__(self, bar: pd.DataFrame, price_low_bound: float, price_upper_bound: float):
        super().__init__()
        self.bar = bar
        self.universe_name = 'close_between_{}_{}'.format(price_low_bound, price_upper_bound)
        self.price_low_bound = price_low_bound
        self.price_upper_bound = price_upper_bound

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        masked = self.bar['close'] \
            .between(self.price_low_bound, self.price_upper_bound).astype(int).replace(0, np.nan).unstack()
        s = (X.unstack() * masked).stack().sort_index()
        s.name = X.name
        return s


class PipelineSelector(UniverseSelector):

    def __init__(self, selectors: List[UniverseSelector]):
        super(PipelineSelector, self).__init__()
        self.selectors = selectors

    def fit_transform(self, X, y=None, **fit_params):
        idx = X.index
        mask = pd.Series(1, index=idx, name='mask')
        for s in self.selectors:
            mask = s.fit_transform(mask)
        X = X.loc[idx.intersection(mask.index)]
        return X


# ===========================================================
# UniverseSelector
# based on traded value
# ===========================================================
class VolumeSelector(UniverseSelector):
    def __init__(self, universe_data_input_path: str, top_quantile: float):
        super().__init__()
        self.money = market_data.get_bars(config_path=universe_data_input_path,
                                          cols=('money',), eod_time_adjust=False, add_limit=False, adjust=False)
        self.universe_name = 'volume_top_{}'.format(top_quantile)
        self.top_quantile = top_quantile

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        unstack = self.money['money'].unstack()
        q = unstack.quantile(self.top_quantile, axis=1)
        masked = unstack.ge(q, axis=0).astype(int).replace(0, np.nan)
        s = (X.unstack() * masked).stack().sort_index()
        s.name = X.name
        return s


class TradedValueQuantileSelector(UniverseSelector):
    def __init__(self, df_tradedvalue: pd.DataFrame, top_quantile: float):
        super().__init__()
        q = df_tradedvalue.quantile(self.top_quantile, axis=1)
        masked = df_tradedvalue.ge(q, axis=0).astype(int).replace(0, np.nan)
        self.universe = masked.stack(dropna=True)
        self.universe_name = 'tradedvalue_top_{}'.format(top_quantile)

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)

        s = X[X.index.isin(self.universe.index.to_list())].sort_index()
        s.name = X.name
        return s


class TradedValueRawSelector(UniverseSelector):
    def __init__(self, df_tradedvalue: pd.DataFrame, threshold: float):
        super().__init__()
        masked = (df_tradedvalue >= threshold).astype(int).replace(0, np.nan)
        self.universe = masked.stack(dropna=True)
        self.universe_name = 'tradedvalue_threshold_{}'.format(threshold)

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)

        s = X[X.index.isin(self.universe.index.to_list())].sort_index()
        s.name = X.name
        return s
