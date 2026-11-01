from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def generate_research_charts(
    research: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    """Create publication-ready benchmark and ablation charts."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(research.get("benchmark_table", []))
    created: list[Path] = []

    if not rows.empty:
        labels = rows["model_name"].astype(str).tolist()
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
        axes[0].bar(labels, rows["latency_ms"], color="#2563eb")
        axes[0].set_title("Latency (ms) - lower is better")
        axes[0].set_ylabel("Milliseconds")
        axes[1].bar(labels, rows["vram_mb"], color="#14b8a6")
        axes[1].set_title("Peak VRAM (MB) - lower is better")
        axes[1].set_ylabel("MB")
        for axis in axes:
            axis.tick_params(axis="x", labelrotation=28)
            axis.grid(axis="y", alpha=0.25)
        path = destination / "benchmark_efficiency.png"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        created.append(path)

    ablations = pd.DataFrame(research.get("ablation_study", []))
    if not ablations.empty:
        figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        axis.plot(
            ablations["configuration"],
            ablations["primary_metric"],
            marker="o",
            linewidth=2,
            label="Primary metric",
        )
        axis.set_title("Ablation trade-off")
        axis.set_ylabel("Primary metric")
        axis.tick_params(axis="x", labelrotation=28)
        axis.grid(alpha=0.25)
        axis.legend()
        path = destination / "ablation_tradeoff.png"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        created.append(path)

    return created
