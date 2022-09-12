from abc import ABC, abstractmethod

import pandas as pd
import cvxpy

class BaseOptimizer(ABC):
    @abstractmethod
    def fit_transform(self, factor: pd.Series, **fit_params):
        pass
