from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from ecg_reliability.explainability.stability import cosine_similarity, spearman_similarity


def ig_completeness_check(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    attributions: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor | None = None,
) -> dict[str, float]:
    """Verify the Integrated Gradients completeness axiom: sum(IG) ≈ f(x) - f(baseline)."""
    model.eval()
    base = baseline if baseline is not None else torch.zeros_like(inputs)
    with torch.no_grad():
        fx = float(model(inputs)[:, target_class].sum().detach().cpu())
        fb = float(model(base)[:, target_class].sum().detach().cpu())
    target_delta = fx - fb
    attr_sum = float(attributions.detach().cpu().sum())
    abs_error = abs(attr_sum - target_delta)
    rel_error = abs_error / max(abs(target_delta), 1e-6)
    return {
        "fx_logit": fx,
        "baseline_logit": fb,
        "logit_delta": target_delta,
        "attribution_sum": attr_sum,
        "abs_error": float(abs_error),
        "rel_error": float(rel_error),
    }


def model_parameter_randomization_check(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    attr_fn: Callable[[torch.nn.Module, torch.Tensor, int], torch.Tensor],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Adebayo et al. (2018) parameter randomization sanity check.

    Compares the attribution map of the trained model against a weight-randomized copy
    (both final linear head randomization and full-model randomization). A method sensitive
    to learned model parameters should exhibit low rank/cosine similarity after randomization.
    """
    trained_attr = attr_fn(model, inputs, target_class).detach().cpu().numpy()[0]

    # 1. Final layer randomization
    head_rand = copy.deepcopy(model)
    linear_layers = [m for m in head_rand.modules() if isinstance(m, torch.nn.Linear)]
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        if linear_layers:
            linear_layers[-1].reset_parameters()
        head_rand.eval()
        head_attr = attr_fn(head_rand, inputs, target_class).detach().cpu().numpy()[0]

        # 2. Full parameter randomization
        full_rand = copy.deepcopy(model)
        for module in full_rand.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        full_rand.eval()
        full_attr = attr_fn(full_rand, inputs, target_class).detach().cpu().numpy()[0]
    finally:
        torch.random.set_rng_state(rng_state)

    head_spearman = spearman_similarity(trained_attr, head_attr)
    head_cosine = cosine_similarity(trained_attr, head_attr)
    full_spearman = spearman_similarity(trained_attr, full_attr)
    full_cosine = cosine_similarity(trained_attr, full_attr)

    return {
        "head_randomized_spearman": float(head_spearman),
        "head_randomized_cosine": float(head_cosine),
        "full_randomized_spearman": float(full_spearman),
        "full_randomized_cosine": float(full_cosine),
        "passes_sensitivity_check": bool(abs(full_spearman) < 0.5 and abs(full_cosine) < 0.5),
        "interpretation": (
            "Lower similarity between trained and parameter-randomized attributions indicates "
            "the method reflects learned model weights rather than input-only edge features."
        ),
    }


def faithfulness_deletion_insertion(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    attributions: np.ndarray,
    target_class: int,
    *,
    steps: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Measure attribution faithfulness via progressive deletion and insertion curves.

    Replaces the top-k highest-|attribution| input coordinates with zero baseline
    and compares the drop in target-class probability against a random-mask control.
    """
    model.eval()
    x_np = inputs.detach().cpu().numpy()[0]
    attr_abs = np.abs(np.asarray(attributions, dtype=np.float32)).reshape(-1)
    n_elements = attr_abs.size
    ranked_indices = np.argsort(-attr_abs)
    rng = np.random.default_rng(seed)
    random_indices = rng.permutation(n_elements)
    device = inputs.device

    fractions = np.linspace(0.0, 1.0, max(int(steps), 2) + 1)
    deletion_probs: list[float] = []
    random_deletion_probs: list[float] = []
    insertion_probs: list[float] = []

    with torch.no_grad():
        for frac in fractions:
            k = int(round(float(frac) * n_elements))

            del_x = x_np.reshape(-1).copy()
            if k > 0:
                del_x[ranked_indices[:k]] = 0.0
            del_t = torch.from_numpy(del_x.reshape(1, *x_np.shape)).to(device=device, dtype=torch.float32)
            deletion_probs.append(float(torch.sigmoid(model(del_t)[0, target_class]).cpu()))

            rand_x = x_np.reshape(-1).copy()
            if k > 0:
                rand_x[random_indices[:k]] = 0.0
            rand_t = torch.from_numpy(rand_x.reshape(1, *x_np.shape)).to(device=device, dtype=torch.float32)
            random_deletion_probs.append(float(torch.sigmoid(model(rand_t)[0, target_class]).cpu()))

            ins_x = np.zeros_like(x_np.reshape(-1))
            if k > 0:
                ins_x[ranked_indices[:k]] = x_np.reshape(-1)[ranked_indices[:k]]
            ins_t = torch.from_numpy(ins_x.reshape(1, *x_np.shape)).to(device=device, dtype=torch.float32)
            insertion_probs.append(float(torch.sigmoid(model(ins_t)[0, target_class]).cpu()))

    del_auc = float(np.trapezoid(deletion_probs, fractions))
    rand_del_auc = float(np.trapezoid(random_deletion_probs, fractions))
    ins_auc = float(np.trapezoid(insertion_probs, fractions))
    return {
        "fractions": [float(f) for f in fractions],
        "deletion_probs": deletion_probs,
        "random_deletion_probs": random_deletion_probs,
        "insertion_probs": insertion_probs,
        "deletion_auc": del_auc,
        "random_deletion_auc": rand_del_auc,
        "insertion_auc": ins_auc,
        "faithfulness_margin": float(rand_del_auc - del_auc),
        "interpretation": (
            "Lower deletion AUC relative to random deletion (positive faithfulness_margin) and higher "
            "insertion AUC indicate that high-attribution samples carry stronger predictive signal."
        ),
    }


def run_attribution_sanity_suite(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    attributions: torch.Tensor,
    target_class: int,
    attr_fn: Callable[[torch.nn.Module, torch.Tensor, int], torch.Tensor],
    *,
    method_name: str = "integrated_gradients",
    steps: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute completeness, randomization sensitivity, and deletion/insertion faithfulness checks."""
    attr_np = attributions.detach().cpu().numpy()[0]
    completeness = (
        ig_completeness_check(model, inputs, attributions, target_class)
        if method_name == "integrated_gradients"
        else None
    )
    randomization = model_parameter_randomization_check(model, inputs, target_class, attr_fn, seed=seed)
    faithfulness = faithfulness_deletion_insertion(model, inputs, attr_np, target_class, steps=steps, seed=seed)
    return {
        "method": method_name,
        "target_class": int(target_class),
        "completeness": completeness,
        "parameter_randomization": randomization,
        "faithfulness": faithfulness,
        "limitations": [
            "Post-hoc attributions measure local model sensitivity, not physiological causality or clinical diagnostic proof.",
            "Zero-baseline masking assumes zero represents absence of ECG morphology, which is an approximation under per-lead z-normalization.",
            "Deletion curves can push inputs off the training manifold; compare against random-deletion controls.",
        ],
    }
