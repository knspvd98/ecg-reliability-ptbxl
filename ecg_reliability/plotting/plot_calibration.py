from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_reliability_diagram(bins: list[dict], output_path: str | Path, title: str, dpi: int = 200) -> None:
    centers = [0.5 * (item["bin_left"] + item["bin_right"]) for item in bins]
    accuracies = [item["accuracy"] for item in bins]
    plt.figure(figsize=(6, 6), dpi=dpi)
    plt.plot([0, 1], [0, 1], "--", color="black")
    plt.plot(centers, accuracies, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_reliability_comparison(raw_bins: list[dict], calibrated_bins: list[dict], output_path: str | Path, dpi: int = 200) -> None:
    raw_centers = [0.5 * (item["bin_left"] + item["bin_right"]) for item in raw_bins]
    raw_acc = [item["accuracy"] for item in raw_bins]
    cal_centers = [0.5 * (item["bin_left"] + item["bin_right"]) for item in calibrated_bins]
    cal_acc = [item["accuracy"] for item in calibrated_bins]
    plt.figure(figsize=(6, 6), dpi=dpi)
    plt.plot([0, 1], [0, 1], "--", color="black", label="Ideal")
    plt.plot(raw_centers, raw_acc, marker="o", label="Raw")
    plt.plot(cal_centers, cal_acc, marker="s", label="Calibrated")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_risk_coverage(curves: list[dict], output_path: str | Path, dpi: int = 200) -> None:
    coverage = [item["coverage"] for item in curves]
    risk = [item["accepted_risk"] for item in curves]
    harm = [item["expected_harm_with_abstain"] for item in curves]
    plt.figure(figsize=(8, 5), dpi=dpi)
    plt.plot(coverage, risk, marker="o", label="Selective risk")
    plt.plot(coverage, harm, marker="s", label="Expected harm")
    plt.xlabel("Coverage")
    plt.ylabel("Risk / Harm")
    plt.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
