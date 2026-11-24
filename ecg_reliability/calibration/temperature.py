from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ecg_reliability.evaluation.thresholding import sigmoid


@dataclass
class TemperatureScaler:
    vector: bool = False
    max_iter: int = 200
    lr: float = 0.05
    temperature: np.ndarray | None = None

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> TemperatureScaler:
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        init_shape = (logits.shape[1],) if self.vector else (1,)
        temperature = torch.nn.Parameter(torch.ones(init_shape, dtype=torch.float32))
        optimizer = torch.optim.LBFGS([temperature], lr=self.lr, max_iter=self.max_iter)
        criterion = torch.nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            scaled = logits_t / torch.clamp(temperature, min=1e-3)
            loss = criterion(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = temperature.detach().cpu().numpy()
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        if self.temperature is None:
            raise RuntimeError("TemperatureScaler must be fit before prediction.")
        return sigmoid(logits / np.clip(self.temperature, 1e-3, None))
