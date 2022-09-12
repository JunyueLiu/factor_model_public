import numpy as np
import pandas as pd


def predict_label_to_df(y: np.ndarray, index: pd.Index):
    return pd.Series(y, index=index)


