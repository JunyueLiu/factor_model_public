import numpy as np
import pandas as pd
from scipy import stats

from model_zoo.BaseModel import BaseModel


class ICWeightModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.feature_weights = None
        self.feature_names = None

    def is_classifier(self):
        return False

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones_like(X.columns)
        # todo sample weight
        ic = stats.spearmanr(X.values, y, axis=0)[0][-1, :-1]
        ic = np.nan_to_num(ic, 0)
        ic = ic / np.abs(ic).sum()
        self.feature_weights = ic
        self.feature_names = X.columns
        return self

    def predict(self, X):
        return (X * self.feature_weights).sum(axis=1)

    def predict_proba(self, X):
        pass

    def get_feature_importance(self):
        return np.abs(self.feature_weights)
