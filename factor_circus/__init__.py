from abc import ABC, abstractmethod

import pandas as pd
from pandas import DatetimeIndex


class FactorProcesser(ABC):
    def input_check(self, X):
        assert isinstance(X, pd.Series)
        assert isinstance(X.index, pd.MultiIndex)
        names = X.index.names
        assert names[0] == 'date' and names[1] == 'code'
        assert isinstance(X.index.get_level_values(0), DatetimeIndex)

    @abstractmethod
    def fit_transform(self, X, y=None, **fit_params):
        pass
