from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_expected_harm_vs_coverage(curves: list[dict], output_path: str | Path, dpi: int = 200) -> None:
    coverage = [item["coverage"] for item in curves]
    harm = [item["expected_harm_with_abstain"] for item in curves]
    plt.figure(figsize=(8, 5), dpi=dpi)
    plt.plot(coverage, harm, marker="o", color="#70AD47")
    plt.xlabel("Coverage")
    plt.ylabel("Expected Harm")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
