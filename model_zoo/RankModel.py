import numpy as np
import pandas as pd

from model_zoo import BaseModel


class RankModel(BaseModel):

    def predict(self, X):
        if isinstance(X, np.ndarray):
            raise NotImplementedError
        elif isinstance(X, pd.Series):
            raise NotImplementedError
        elif isinstance(X, pd.DataFrame):
            ranks = X.rank() / len(X)
            return ranks.mean(axis=1)
        else:
            raise NotImplementedError