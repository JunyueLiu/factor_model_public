from typing import Dict, Union, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
import tqdm
from cvxpy.settings import SOLVERS

from factor_circus.preprocessing import UniverseSelector
from factor_zoo.industry import industry_dummy, industry_category
from portfolio_builder.BaseOptimizer import BaseOptimizer


class EnhancedIndexOptimizer(BaseOptimizer):
    def __init__(self,
                 index_weights: pd.DataFrame,
                 industry_dict: Dict,
                 *,
                 styles_factors: Optional[Union[pd.Series, pd.DataFrame]] = None,
                 benchmark_bound: Union[Tuple, Dict[str, Tuple[float, float]], pd.DataFrame] = (-0.01, 0.01),
                 industry_bound: Union[Tuple, Dict[str, Tuple[float, float]], pd.DataFrame] = (-0.01, 0.01),
                 max_turnover: Union[float, pd.Series] = 0.25,
                 position_sum: Union[float, pd.Series] = 1.0,
                 universe_selector: Optional[UniverseSelector] = None,
                 clip=0.01,
                 verbose=False,
                 ):
        self.index_weights = index_weights['weight'] / 100
        self.industry_dict = industry_dict
        self.styles_factors = styles_factors
        self.benchmark_bound = benchmark_bound
        self.industry_bound = industry_bound
        self.max_turnover = max_turnover
        self.position_sum = position_sum
        self.universe_selector = universe_selector
        self.objective = None
        self.a = None
        self.w = None
        self.wb = None
        self.problem = None
        self.ind_lower = None
        self.ind_upper = None
        self.wh = None
        self.cp_pos_sum = None
        self.cp_max_turnover = None
        self.cp_benchmark_lower = None
        self.cp_benchmark_upper = None
        self.clip = clip
        self.verbose = verbose

        self.res = None

    def validate_industry_exposure(self, industry_info: pd.DataFrame):
        # todo
        if self.res is None:
            print("haven't been optimized yet")
            return

        industry = industry_category(self.industry_dict).unstack()
        industry.columns = pd.MultiIndex.from_product([['category'], industry.columns])
        index_weights = self.index_weights.unstack()
        index_weights.columns = pd.MultiIndex.from_product([['index'], index_weights.columns])
        df = self.res.unstack()
        df.columns = pd.MultiIndex.from_product([[self.res.name], df.columns])
        df = df.join(index_weights).join(industry)
        print(df)
        df = df.stack()
        df = df.set_index('category', append=True)
        df = df.fillna(0)
        exposure = df.groupby(level=[0, 2])['diff'].sum()

    def fit_transform(self, factor: pd.Series,
                      risk_matrix=None,
                      **fit_params):

        alpha = factor.unstack()
        alpha.columns = pd.MultiIndex.from_product([[factor.name], alpha.columns])

        industry = industry_dummy(self.industry_dict)
        industry = industry.loc[alpha.index]
        if self.universe_selector is not None:
            masked = pd.Series(1, index=industry.index, name='masked_universe')
            masked = self.universe_selector.fit_transform(masked)
            industry = industry.loc[masked.index]

        industry_cols = industry.columns
        industry = industry.unstack()
        index_weights = self.index_weights.unstack()
        index_weights.columns = pd.MultiIndex.from_product([['index'], index_weights.columns])
        index_weights = index_weights.loc[:alpha.index[0]].iloc[[-1]].append(index_weights.loc[alpha.index[0]:])
        index_weights = index_weights[~index_weights.index.duplicated()]
        idx = alpha.index
        idx = idx.append(index_weights.index).drop_duplicates().sort_values()
        index_weights = index_weights.fillna(0).reindex(idx).fillna(method='ffill')
        data = alpha.join(index_weights).join(industry)

        if self.styles_factors is not None:
            styles_cols = self.styles_factors.columns
        else:
            styles_cols = []

        last_weight = None
        res = []

        for i in tqdm.tqdm(range(len(data))):
            df = data.iloc[i].unstack(level=0)
            # fill nan alpha to minimum number to not be selected
            df[factor.name] = df[factor.name].fillna(df[factor.name].min() - df[factor.name].std())
            df = df.fillna(0)
            if last_weight is None:
                df['last_weight'] = df['index']
            else:
                df['last_weight'] = last_weight
            w = self.single_period_opt(df, factor.name, 'index', industry_cols, styles_cols, data.index[i])
            res.append(w)
            w.name = data.index[i]
            last_weight = w

        res = pd.concat(res, axis=1).stack().swaplevel(0, 1).sort_index()
        res = res[res > self.clip]
        # res = res.groupby(level=0).transform(lambda x: (x / x.sum()).round(4))
        res.name = 'opt_{}'.format(factor.name)
        res.index.names = ['date', 'code']
        self.res = res
        return res

    def single_period_opt(self, data: pd.DataFrame, alpha_col: str,
                          wb_col,
                          industry_cols: list,
                          style_cols: list, dt):
        # https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/applications/portfolio_optimization.ipynb#scrollTo=_s1N2gOaWEIM
        # todo how to deal with stock not in universe better way
        a = data[[alpha_col]].values
        industry_matrix = data[industry_cols].values.T
        if self.objective is None:
            self.a = cp.Parameter((len(data), 1), name='A')
            self.w = cp.Variable(len(a))
            self.wb = cp.Parameter(len(data), name='wb')
            self.a.value = a
            self.objective = cp.Maximize(cp.sum(self.a.T @ self.w))
            self.ind_lower = cp.Parameter(industry_matrix.shape[0], name='ind_lower')
            self.ind_upper = cp.Parameter(industry_matrix.shape[0], name='ind_upper')
            self.wh = cp.Parameter(len(data), name='w_holding')
            self.cp_pos_sum = cp.Parameter(name='pos_sum')
            self.cp_max_turnover = cp.Parameter(name='max_turnover')
            self.cp_benchmark_lower = cp.Parameter(name='benchmark_lower')
            self.cp_benchmark_upper = cp.Parameter(name='benchmark_upper')

        if isinstance(self.industry_bound, Tuple):
            self.ind_lower.value = np.ones(industry_matrix.shape[0]) * self.industry_bound[0]
            self.ind_upper.value = np.ones(industry_matrix.shape[0]) * self.industry_bound[1]
        elif isinstance(self.industry_bound, pd.DataFrame):
            idx = self.industry_bound.index.get_level_values(0)
            dt_ = idx[idx <= dt][-1]
            ind_bound = self.industry_bound.loc[:dt_].droplevel(0)
            ind_bound = ind_bound[~ind_bound.index.duplicated()]
            ind_bound = ind_bound.reindex(industry_cols).fillna(0)
            self.ind_lower.value = ind_bound['lower'].values
            self.ind_upper.value = ind_bound['upper'].values
        else:
            raise NotImplementedError

        if isinstance(self.benchmark_bound, Tuple):
            benchmark_lower_bound = self.benchmark_bound[0]
            benchmark_upper_bound = self.benchmark_bound[1]
        elif isinstance(self.benchmark_bound, pd.DataFrame):

            idx = self.benchmark_bound.index.get_level_values(0)
            dt_ = idx[idx <= dt][-1]
            benchmark_bound = self.benchmark_bound.loc[dt_]
            benchmark_lower_bound = benchmark_bound['lower']
            benchmark_upper_bound = benchmark_bound['upper']
        else:
            raise NotImplementedError

        if isinstance(self.position_sum, (float, int)):
            position_sum = self.position_sum
        else:
            position_sum = self.position_sum.loc[:dt].values[-1]

        if isinstance(self.max_turnover, (float, int)):
            max_turnover = self.max_turnover
        else:
            max_turnover = self.max_turnover.loc[:dt].values[-1]

        w = self.w
        wb = self.wb
        ind_lower = self.ind_lower
        ind_upper = self.ind_upper
        wh = self.wh

        self.a.value = a
        wb.value = data[wb_col].values
        wh.value = data['last_weight'].values
        self.cp_pos_sum.value = position_sum
        self.cp_max_turnover.value = max_turnover
        self.cp_benchmark_lower.value = benchmark_lower_bound
        self.cp_benchmark_upper.value = benchmark_upper_bound

        constraints = [cp.sum(w) == self.cp_pos_sum,
                       w >= 0,
                       cp.multiply(w, (1 / self.clip)) >= 0,
                       self.cp_benchmark_lower <= (w - wb), (w - wb) <= self.cp_benchmark_upper,
                       ind_lower <= industry_matrix @ (w - wb), industry_matrix @ (w - wb) <= ind_upper,
                       cp.sum(cp.abs(w - wh)) <= self.cp_max_turnover
                       ]
        if len(style_cols) > 0:
            pass

        if self.problem is None:
            self.problem = cp.Problem(self.objective, constraints)
        # else:
        #     if not isinstance(self.position_sum, (float, int)) or \
        #             not isinstance(self.benchmark_bound, Tuple) or \
        #             not isinstance(self.max_turnover, (float, int)) or \
        #             not isinstance(self.industry_bound, Tuple):
        #         self.problem = cp.Problem(self.objective, constraints)

        for solver in SOLVERS:
            try:
                self.problem.solve(solver)
                if self.problem.status == 'optimal':
                    break
                else:
                    print('{} return {}. try next solver'.format(solver, self.problem.status))

            except Exception as e:
                # self.problem.solve(cp.SCS, verbose=self.verbose)
                print(e.__traceback__)

        if self.problem.status == 'infeasible':
            raise ValueError()

        return pd.Series(w.value, data.index).round(4)
