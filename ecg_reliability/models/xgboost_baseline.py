from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class MultiLabelXGBoost:
    def __init__(self, **model_kwargs) -> None:
        self.model_kwargs = model_kwargs
        self.models: list[object] = []

    def _new_estimator(self):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("xgboost is required for the baseline model.") from exc
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **self.model_kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> MultiLabelXGBoost:
        self.models = []
        for class_idx in range(y.shape[1]):
            target = y[:, class_idx]
            if np.unique(target).size < 2:
                model = {"constant_probability": float(target[0])}
            else:
                model = self._new_estimator()
                model.fit(X, target)
            self.models.append(model)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = []
        for model in self.models:
            if isinstance(model, dict) and "constant_probability" in model:
                probs.append(np.full(X.shape[0], model["constant_probability"], dtype=np.float32))
            else:
                probs.append(model.predict_proba(X)[:, 1])
        return np.stack(probs, axis=1)

    def predict_logits(self, X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        probs = np.clip(self.predict_proba(X), eps, 1.0 - eps)
        return np.log(probs / (1.0 - probs))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> MultiLabelXGBoost:
        with Path(path).open("rb") as handle:
            return pickle.load(handle)
