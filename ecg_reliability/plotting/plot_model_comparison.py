from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_model_comparison(metrics_df: pd.DataFrame, output_path: str | Path, dpi: int = 200) -> None:
    summary = metrics_df.pivot(index="model", columns="metric", values="value").reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
    summary.plot(x="model", y="macro_auroc", kind="bar", ax=axes[0], color="#5B9BD5")
    summary.plot(x="model", y="macro_auprc", kind="bar", ax=axes[1], color="#ED7D31")
    axes[0].set_title("Macro AUROC")
    axes[1].set_title("Macro AUPRC")
    for axis in axes:
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
