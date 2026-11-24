from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ecg_reliability.data.labels import TARGET_LABELS


def plot_label_distribution(index_df: pd.DataFrame, output_path: str | Path, dpi: int = 200) -> None:
    counts = index_df[TARGET_LABELS].sum().reindex(TARGET_LABELS)
    plt.figure(figsize=(8, 5), dpi=dpi)
    counts.plot(kind="bar", color="#4472C4")
    plt.ylabel("Count")
    plt.title("PTB-XL Diagnostic Superclass Distribution")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
