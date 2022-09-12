from typing import Callable, Optional, List

import numpy as np
import pandas as pd

from data_management.dataIO.component_data import get_index_weights, IndexTicker, get_industry_component, \
    IndustryCategory
from data_management.dataIO.trade_data import get_trade, TradeTable
from data_management.pandas_utils.cache import panel_df_join
from factor_circus import FactorProcesser
from factor_zoo.industry import industry_category


class FactorAppender(FactorProcesser):

    def fit_transform(self, X: List[pd.Series], y=None, **fit_params):
        f = pd.concat(X)

        return f


class TopNSelector(FactorProcesser):
    def __init__(self, n):
        self.n = n

    def fit_transform(self, X: pd.Series, y=None, **fit_params):
        self.input_check(X)
        f = X.groupby(level=0).nlargest(self.n).droplevel(0)
        f.name = '{}_top_{}'.format(X.name, self.n)
        return f


class TopQuantileSelector(FactorProcesser):
    def __init__(self, q):
        self.q = q

    def fit_transform(self, X: pd.Series, y=None, **fit_params):
        self.input_check(X)
        unstack_X = X.unstack()
        quantile = unstack_X.quantile(1 - self.q, axis=1)
        masked = unstack_X.gt(quantile, axis=0).astype(int).replace(0, np.nan)
        unstack_X = unstack_X * masked

        f = unstack_X.stack()
        f.name = '{}_top_q_{}'.format(X.name, self.q)
        return f


class SectorTopNSelector(FactorProcesser):

    def __init__(self, industry_data_input_path: str,
                 ind_category: IndustryCategory,
                 n: int):
        self.n = n
        self.ind_category = ind_category
        industry_dict = get_industry_component(ind_category, config_path=industry_data_input_path)
        self.cat = industry_category(industry_dict).astype(str)

    def fit_transform(self, X: pd.Series, y=None, **fit_params):
        self.factor_name = X.name
        factor_data = panel_df_join(X.to_frame(), self.cat.to_frame())
        factor_data = factor_data[~factor_data[self.factor_name].isna()]
        grouper = [self.cat.name, factor_data.index.get_level_values(0)]
        f = factor_data.groupby(by=grouper)[self.factor_name].nlargest(self.n).droplevel([0, 1]).sort_index()
        f.name = '{}_{}_top_{}'.format(X.name, self.ind_category.name, self.n)
        return f


class SectorTopOverUnderWeightSelector(FactorProcesser):

    def __init__(self, industry_data_input_path: str,
                 ind_category: IndustryCategory,
                 industry_score: pd.Series,
                 multiplier=1
                 ):
        self.ind_category = ind_category
        industry_dict = get_industry_component(ind_category, config_path=industry_data_input_path)
        self.cat = industry_category(industry_dict).astype(str)
        self.industry_score = industry_score
        self.multiplier = multiplier

    def fit_transform(self, X, y=None, **fit_params):
        def _f(x):
            top_n = int(x[self.industry_score.name].fillna(0).iloc[0] * self.multiplier)
            return x[self.factor_name].iloc[:top_n]

        self.factor_name = X.name
        factor_data = panel_df_join(X.to_frame(), self.cat.to_frame())
        factor_data = factor_data[~factor_data[self.factor_name].isna()]
        factor_data = factor_data.set_index(self.cat.name, append=True)
        self.industry_score.index.names = ['date', self.cat.name]
        factor_data = factor_data.join(self.industry_score)
        factor_data = factor_data.sort_values(['date', self.cat.name, self.factor_name], ascending=[True, True, False])
        f = factor_data.groupby(level=[0, 1]).apply(_f)
        f = f.droplevel([1, 2, 3])
        f = f.sort_index()

        return f


class SectorNormScore(FactorProcesser):
    def __init__(self, beta):
        self.beta = beta

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        self.factor_name = X.name
        f = X.groupby(level=0).transform(lambda x: x / (x.abs().sum() * self.beta))
        return f


class SectorFixedBound(FactorProcesser):

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        self.factor_name = X.name
        bound = X.to_frame()
        bound['lower'] = X
        bound['upper'] = X
        return bound[['lower', 'upper']]


class SectorScoreBound(FactorProcesser):
    def __init__(self, labels, use_group: True):
        self.labels = labels
        self.use_group = use_group

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        self.factor_name = X.name
        f = X.groupby(level=0).transform(lambda x: pd.qcut(x, self.labels))
        bound = X.to_frame()
        bound['lower'] = f.array.left
        bound['upper'] = f.array.right

        if self.use_group:
            return bound[['lower', 'upper']]
        else:
            l = list(range(len(self.labels) - 1))
            g = X.groupby(level=0).transform(lambda x: pd.qcut(x, self.labels, l))
            bound['group'] = g
            bound['upper'] = bound.apply(
                lambda x: x['upper'] if x['group'] not in [l[0], l[-1]] else x[self.factor_name], axis=1)
            bound['lower'] = bound.apply(
                lambda x: x['lower'] if x['group'] not in [l[0], l[-1]] else x[self.factor_name],
                axis=1)
            return bound[['lower', 'upper']]


class AddSectorScore(FactorProcesser):
    def __init__(self, sector_score: pd.Series, beta, industry_category_name: IndustryCategory, config_path: str):
        self.sector_score = sector_score
        self.beta = beta
        self.industry_category = industry_category_name
        industry_dict = get_industry_component(self.industry_category, config_path=config_path)
        self.cat = industry_category(industry_dict).to_frame().unstack()

    def fit_transform(self, X, y=None, **fit_params):
        # component = get_industry_component(self.industry_category)
        self.input_check(X)
        factor_name = X.name
        f = X.to_frame().unstack().join(self.cat).stack().dropna()
        f = f.set_index('industry_code', append=True)
        score = self.sector_score
        score.index.names = ['date', 'industry_code']
        f = f.join(score)
        f = f[factor_name] + f[score.name] * self.beta
        f = f.droplevel(1).sort_index()
        f.name = factor_name
        return f


class BottomNSelector(FactorProcesser):
    def __init__(self, n):
        self.n = n

    def fit_transform(self, X: pd.Series, y=None, **fit_params):
        self.input_check(X)
        f = X.groupby(level=0).nsmallest(self.n).droplevel(0)
        f.name = '{}_bottom_{}'.format(X.name, self.n)
        return f


class BottomQuantileSelector(FactorProcesser):
    def __init__(self, q):
        self.q = q

    def fit_transform(self, X: pd.Series, y=None, **fit_params):
        self.input_check(X)
        unstack_X = X.unstack()
        quantile = unstack_X.quantile(self.q, axis=1)
        masked = unstack_X.lt(quantile, axis=0).astype(int).replace(0, np.nan)
        unstack_X = unstack_X * masked

        f = unstack_X.stack()
        f.name = '{}_bottom_q_{}'.format(X.name, self.q)
        return f


class EqualWeightPortfolioGenerator(FactorProcesser):

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        weight = pd.Series(1, X.index)
        weight = weight.groupby(level=0).transform(lambda x: x / len(x))
        weight.name = '{}_equal_weight_portfolio'.format(X.name)
        return weight


class FactorWeightPortfolioGenerator(FactorProcesser):

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        weight = pd.Series(1, X.index)
        weight = weight.groupby(level=0).transform(lambda x: x / x.abs().sum())
        weight.name = '{}_equal_weight_portfolio'.format(X.name)
        return weight


class CapWeightPortfolioGenerator(FactorProcesser):

    def __init__(self, daily_basic_path: str, use_total=True,
                 func: Optional[Callable[[pd.Series], pd.Series]] = None) -> None:
        daily_basic = get_trade(TradeTable.daily_basic,
                                cols=['circ_mv', 'total_mv'],
                                config_path=daily_basic_path)
        self.use_total = use_total
        if use_total:
            self.cap = daily_basic['total_mv']
        else:
            self.cap = daily_basic['circ_mv']

        if func:
            self.func = func
            self.cap = func(self.cap)
        else:
            self.func = None

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        weight = self.cap.reindex(X.index)
        for i, v in weight.iteritems():
            if pd.isna(v):
                try:
                    v = self.cap.loc[slice(None, i[0]), i[1]].iloc[-1]
                    weight.loc[i] = v
                except:
                    pass
        weight.name = '{}_{}{}_weight_portfolio'.format(X.name,
                                                        '{}_'.format(self.func.__name__) if self.func else '',
                                                        'total_cap' if self.use_total else 'circ_cap'
                                                        )
        return weight


class IndexWeightPortfolioGenerator(FactorProcesser):

    def __init__(self, index_ticker: IndexTicker, index_weights_data_input: str):
        self.index_ticker = index_ticker.value
        self.weights = get_index_weights(index_ticker, index_weights_data_input)['weight']

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        weight = self.weights.reindex(X.index)
        for i, v in weight.iteritems():
            if pd.isna(v):
                try:
                    w = self.weights.loc[slice(None, i[0]), i[1]]
                    if (i[0] - w.index[-1][0]).days < 40:
                        weight.loc[i] = w.iloc[-1]
                    else:
                        weight.loc[i] = 0
                except:
                    weight.loc[i] = 0
        weight = weight.groupby(level=0).transform(lambda x: x / x.sum())
        weight.name = '{}_{}_weight_portfolio'.format(X.name, self.index_ticker)
        return weight


class SectorRotationAdjuster(FactorProcesser):

    def __init__(self, sector_weights: pd.Series,
                 sector_category_used: IndustryCategory,
                 industry_component_data_input: str
                 ):
        self.sector_weights = sector_weights
        self.industry_component = get_industry_component(sector_category_used,
                                                         config_path=industry_component_data_input)

    def fit_transform(self, X, y=None, **fit_params):
        self.input_check(X)
        # todo


class IndexStyleAdjuster(FactorProcesser):

    def fit_transform(self, X, y=None, **fit_params):
        raise NotImplementedError


class BetaTimingAdjuster(FactorProcesser):

    def __init__(self):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        raise NotImplementedError


class OptimizedPortfolioGenerator(FactorProcesser):

    def fit_transform(self, X, y=None, **fit_params):
        raise NotImplementedError


class IndustryBoundGenerator(FactorProcesser):

    def __init__(self, r: float, n=2):
        self.r = r
        self.n = n

    def fit_transform(self, X, y=None, **fit_params):
        if self.n == 2:
            bound = X.groupby(level=0).transform(lambda x: pd.qcut(x, 2, labels=[-1, 1])).astype(
                float).to_frame('mid')
            bound['lower'] = bound['mid'].apply(lambda x: 0 if x == 1 else -self.r)
            bound['upper'] = bound['mid'].apply(lambda x: self.r if x == 1 else 0)
        elif self.n == 3:
            bound = X.groupby(level=0).transform(lambda x: pd.qcut(x, 3, labels=[-1, 0, 1])).astype(
                float).to_frame('mid')
            bound['lower'] = bound['mid'].apply(lambda x: self.r / 2 if x == 1 else -self.r / 2 if x == 0 else -self.r)
            bound['upper'] = bound['mid'].apply(lambda x: self.r if x == 1 else self.r / 2 if x == 0 else -self.r / 2)
        else:
            raise NotImplementedError
        return bound[['lower', 'upper']]
