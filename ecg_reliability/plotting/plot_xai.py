from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_ecg_with_attribution(signal: np.ndarray, attribution: np.ndarray, output_path: str | Path, dpi: int = 200) -> None:
    fig, axes = plt.subplots(signal.shape[0], 1, figsize=(12, 14), sharex=True, dpi=dpi)
    time_axis = np.arange(signal.shape[1])
    if signal.shape[0] == 1:
        axes = [axes]
    for idx, axis in enumerate(axes):
        axis.plot(time_axis, signal[idx], color="black", linewidth=0.8)
        axis.fill_between(time_axis, 0.0, attribution[idx], color="#C00000", alpha=0.25)
        axis.set_ylabel(f"L{idx + 1}")
    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
