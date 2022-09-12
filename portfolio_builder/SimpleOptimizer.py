from typing import Tuple, Union

import cvxpy as cp
import pandas as pd
import tqdm

from portfolio_builder.BaseOptimizer import BaseOptimizer


class SimpleOptimizer(BaseOptimizer):
    def __init__(self,
                 position_bound: Tuple[float, float] = (-0.01, 0.01),
                 max_turnover: Union[float, pd.Series] = 0.25,
                 clip: float = 0.01,
                 w_sum: Union[float, pd.Series] = 0.0
                 ):
        self.position_bound = position_bound
        self.max_turnover = max_turnover
        self.clip = clip
        self.w_sum = w_sum
        self.objective = None
        self.a = None
        self.w = None
        self.problem = None
        self.wh = None

    def fit_transform(self, factor: pd.Series,
                      **fit_params):
        alpha = factor.unstack()
        alpha.columns = pd.MultiIndex.from_product([[factor.name], alpha.columns])

        data = alpha

        last_weight = None
        res = []

        for i in tqdm.tqdm(range(len(data))):
            df = data.iloc[i].unstack(level=0).fillna(0)
            if last_weight is None:
                df['last_weight'] = 0
            else:
                df['last_weight'] = last_weight
            w = self.single_period_opt(df, factor.name, data.index[i])
            res.append(w)
            w.name = data.index[i]
            last_weight = w

        res = pd.concat(res, axis=1).stack().swaplevel(0, 1).sort_index()
        res = res[res.abs() > self.clip]
        # res = res.groupby(level=0).transform(lambda x: (x / x.abs().sum()).round(4))
        res.name = 'opt_{}'.format(factor.name)
        return res

    def single_period_opt(self, data: pd.DataFrame, alpha_col: str, dt):
        # https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/applications/portfolio_optimization.ipynb#scrollTo=_s1N2gOaWEIM
        a = data[[alpha_col]].values
        if self.objective is None:
            self.a = cp.Parameter((len(data), 1), name='A')
            self.w = cp.Variable(len(a))
            self.wb = cp.Parameter(len(data), name='wb')
            self.a.value = a
            self.objective = cp.Maximize(cp.sum(self.a.T @ self.w))
            self.wh = cp.Parameter(len(data), name='w_holding')
        w = self.w
        wh = self.wh

        self.a.value = a
        wh.value = data['last_weight'].values

        constraints = []
        if isinstance(self.w_sum, pd.Series):
            constraints.append(cp.sum(w) == self.w_sum.loc[dt])
            # constraints.append(cp.sum(cp.abs(w)) == 1)
        else:
            constraints.append(cp.sum(w) == self.w_sum)
            # constraints.append(cp.sum(cp.abs(w)) == 1)

        if isinstance(self.max_turnover, pd.Series):
            constraints += [
                cp.sum(cp.abs(w - wh)) <= self.max_turnover.loc[dt]
            ]
        else:
            constraints += [
                cp.sum(cp.abs(w - wh)) <= self.max_turnover
            ]

        constraints += [self.position_bound[0] <= w, w <= self.position_bound[1]]

        if self.problem is None:
            self.problem = cp.Problem(self.objective, constraints)
            self.problem.solve()
        else:
            self.problem.solve()
        r = pd.Series(w.value, data.index)
        r = r / r.abs().sum()
        if isinstance(self.w_sum, pd.Series):
            r = (self.w_sum.loc[dt] / r.sum()) * r
        r = r.round(4)
        return r