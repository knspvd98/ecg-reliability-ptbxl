from __future__ import annotations

import torch


def compute_gradient_x_input(model, inputs: torch.Tensor, target_class: int) -> torch.Tensor:
    inputs = inputs.clone().detach().requires_grad_(True)
    logits = model(inputs)
    score = logits[:, target_class].sum()
    score.backward()
    attribution = inputs.grad * inputs
    return attribution.detach()
