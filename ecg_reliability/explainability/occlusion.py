from __future__ import annotations

import torch


def compute_occlusion(
    model,
    inputs: torch.Tensor,
    target_class: int,
    sliding_window_shapes: tuple[int, int] = (1, 50),
    strides: tuple[int, int] = (1, 25),
) -> torch.Tensor:
    try:
        from captum.attr import Occlusion
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("captum is required for occlusion attribution.") from exc
    occlusion = Occlusion(model)
    attributions = occlusion.attribute(
        inputs,
        target=target_class,
        sliding_window_shapes=sliding_window_shapes,
        strides=strides,
    )
    return attributions.detach()
