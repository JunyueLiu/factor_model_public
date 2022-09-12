import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from model_zoo.BaseModel import BaseModel


class SKLearnRandomForestClassifier(BaseModel):
    _estimator_type = 'classifier'

    def __init__(self, n_estimators=100, *,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None):
        super().__init__()
        random_forest = RandomForestClassifier(n_estimators=n_estimators,
                                               criterion=criterion,
                                               max_depth=max_depth,
                                               min_samples_split=min_samples_split,
                                               min_samples_leaf=min_samples_leaf,
                                               min_weight_fraction_leaf=min_weight_fraction_leaf,
                                               max_features=max_features,
                                               max_leaf_nodes=max_leaf_nodes,
                                               min_impurity_decrease=min_impurity_decrease,
                                               bootstrap=bootstrap,
                                               oob_score=oob_score,
                                               n_jobs=n_jobs,
                                               random_state=random_state,
                                               verbose=verbose,
                                               warm_start=warm_start,
                                               class_weight=class_weight,
                                               ccp_alpha=ccp_alpha,
                                               max_samples=max_samples)
        self.model = random_forest

    def is_classifier(self):
        return True

    def fit(self, X, y, sample_weight=None):
        return self.model.fit(X, y, sample_weight)

    def predict(self, X):
        y = self.model.predict(X)
        if isinstance(X, (pd.DataFrame, pd.Series)):
            y = pd.Series(y, index=X.index)
        return y

    def predict_proba(self, X):
        y = self.model.predict_proba(X)
        if isinstance(X, (pd.DataFrame, pd.Series)):
            y = pd.DataFrame(y, index=X.index)
            y.columns = self.model.classes_
        return y

    def get_feature_importance(self):
        return self.model.feature_importances_

    def save_model(self):
        super().save_model()

    def load_model(self):
        super().load_model()

    @property
    def estimator_type(self):
        return super().estimator_type()
