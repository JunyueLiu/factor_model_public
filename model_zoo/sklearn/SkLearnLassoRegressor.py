import pandas as pd
from sklearn.linear_model import Lasso

from model_zoo.BaseModel import BaseModel


class SKLearnLassoRegressor(BaseModel):
    _estimator_type = 'regressor'

    def __init__(self, alpha=1.0,
                 *,
                 fit_intercept=True,
                 normalize="deprecated",
                 precompute=False,
                 copy_X=True,
                 max_iter=1000,
                 tol=1e-4,
                 warm_start=False,
                 positive=False,
                 random_state=None,
                 selection="cyclic", ):
        super().__init__()
        lasso = Lasso(alpha,
                      fit_intercept=fit_intercept, normalize=normalize,
                      precompute=precompute,
                      copy_X=copy_X,
                      max_iter=max_iter,
                      tol=tol,
                      warm_start=warm_start,
                      positive=positive,
                      random_state=random_state,
                      selection=selection
                      )
        self.model = lasso

    def is_classifier(self):
        return False

    def fit(self, X, y, sample_weight=None):
        return self.model.fit(X, y, sample_weight)

    def predict(self, X):
        y = self.model.predict(X)
        if isinstance(X, (pd.DataFrame, pd.Series)):
            y = pd.Series(y, index=X.index)
        return y

    def predict_proba(self, X):
        pass

    def get_feature_importance(self):
        return self.model.coef_
