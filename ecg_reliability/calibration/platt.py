from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConstantCalibrator:
    value: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.value, dtype=np.float32)


@dataclass
class PlattScaler:
    models: list[object] = field(default_factory=list)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> PlattScaler:
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("scikit-learn is required for Platt scaling.") from exc
        self.models = []
        for class_idx in range(labels.shape[1]):
            target = labels[:, class_idx]
            if np.unique(target).size < 2:
                model = ConstantCalibrator(float(target[0]))
            else:
                model = LogisticRegression(max_iter=1000)
                model.fit(logits[:, [class_idx]], target)
            self.models.append(model)
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        probs = []
        for idx, model in enumerate(self.models):
            if isinstance(model, ConstantCalibrator):
                probs.append(model.predict_proba(logits[:, [idx]]))
            else:
                probs.append(model.predict_proba(logits[:, [idx]])[:, 1])
        return np.stack(probs, axis=1)
