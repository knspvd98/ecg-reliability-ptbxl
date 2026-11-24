from __future__ import annotations

import torch


def compute_integrated_gradients(model, inputs: torch.Tensor, target_class: int, steps: int = 32) -> torch.Tensor:
    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("captum is required for Integrated Gradients.") from exc
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(inputs)
    attributions = ig.attribute(inputs, baselines=baseline, target=target_class, n_steps=steps)
    return attributions.detach()
