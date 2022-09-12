from abc import ABC, abstractmethod


class BaseModel(ABC):

    _estimator_type = None

    def __init__(self):
        pass

    @abstractmethod
    def is_classifier(self):
        pass

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    @abstractmethod
    def get_feature_importance(self):
        pass

    def save_model(self, save_path: str, model_name: str):
        pass

    def load_model(self, path):
        pass

