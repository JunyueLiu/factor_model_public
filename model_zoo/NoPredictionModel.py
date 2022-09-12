from model_zoo import BaseModel
import numpy as np
import pandas as pd


class NoPredictionModel(BaseModel):

    def predict(self, X):
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                return X
        elif isinstance(X, pd.Series):
            return X.values
        elif isinstance(X, pd.DataFrame):
            if X.values.shape[-1] == 1:
                return X.values[:, 0]
        else:
            raise NotImplementedError

class NegativeNoPredictionModel(BaseModel):

    def predict(self, X):
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                return -X
        elif isinstance(X, pd.Series):
            return -X.values
        elif isinstance(X, pd.DataFrame):
            if X.values.shape[-1] == 1:
                return -X.values[:, 0]
        else:
            raise NotImplementedError