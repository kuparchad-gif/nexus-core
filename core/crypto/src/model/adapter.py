from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

try:
    import lightgbm as lgb
except Exception:
    lgb = None

@dataclass
class Prediction:
    y_hat: np.ndarray
    meta: dict

class BaseAdapter:
    def fit(self, X: pd.DataFrame, y: pd.Series): ...
    def predict(self, X: pd.DataFrame) -> Prediction: ...
    def save(self, path: str): ...
    @classmethod
    def load(cls, path: str): ...

class NoModelAdapter(BaseAdapter):
    """Baseline that echoes a transformed momentum signal as prediction."""
    def fit(self, X, y): 
        return self
    def predict(self, X):
        s = X.get("sma_diff", pd.Series(0.0, index=X.index)).values
        return Prediction(y_hat=s, meta={})

class LightGBMAdapter(BaseAdapter):
    def __init__(self, params: Optional[dict] = None):
        if lgb is None:
            raise RuntimeError("lightgbm not installed")
        self.params = params or {"objective": "regression", "metric": "l2", "learning_rate": 0.05, "num_leaves": 63}
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        dtrain = lgb.Dataset(X.values, label=y.values)
        self.model = lgb.train(self.params, dtrain, num_boost_round=400)
        return self

    def predict(self, X: pd.DataFrame) -> Prediction:
        y = self.model.predict(X.values)
        return Prediction(y_hat=y, meta={"features": list(X.columns)})

    def save(self, path: str):
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str):
        m = cls()
        m.model = lgb.Booster(model_file=path)
        return m
