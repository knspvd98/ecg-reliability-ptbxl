from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ecg_reliability.calibration.platt import ConstantCalibrator
from ecg_reliability.evaluation.thresholding import sigmoid


@dataclass
class IsotonicScaler:
    models: list[object] = field(default_factory=list)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> IsotonicScaler:
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("scikit-learn is required for isotonic regression.") from exc
        self.models = []
        raw_probs = sigmoid(logits)
        for class_idx in range(labels.shape[1]):
            target = labels[:, class_idx]
            if np.unique(target).size < 2:
                model = ConstantCalibrator(float(target[0]))
            else:
                model = IsotonicRegression(out_of_bounds="clip")
                model.fit(raw_probs[:, class_idx], target)
            self.models.append(model)
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        raw_probs = sigmoid(logits)
        probs = []
        for idx, model in enumerate(self.models):
            if isinstance(model, ConstantCalibrator):
                probs.append(model.predict_proba(raw_probs[:, [idx]]))
            else:
                probs.append(model.predict(raw_probs[:, idx]))
        return np.stack(probs, axis=1)
