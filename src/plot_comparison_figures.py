#!/usr/bin/env python3
"""Recreate reward comparison figures from their saved numeric data."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Sequence

from plot_style import STYLE


COMPARISON_PLOT_STYLE = {
    **STYLE,
    "font.size": 12,
    "axes.labelsize": 20,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
}


def load_reward_series(
    csv_path: str | Path, split: str
) -> tuple[list[float], list[float]]:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    regular_key = f"regular_{split}_reward"
    guided_key = f"ga_guided_{split}_reward"
    regular: list[float] = []
    guided: list[float] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row[regular_key]:
                regular.append(float(row[regular_key]))
            if row[guided_key]:
                guided.append(float(row[guided_key]))
    return regular, guided


def plot_reward_comparison(
    regular: Sequence[float],
    guided: Sequence[float],
    output_path: str | Path,
) -> Path:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "comparison-plots"),
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_path)
    with matplotlib.rc_context(COMPARISON_PLOT_STYLE):
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 6))
        axes[0].plot(regular, color="blue", label="Regular DDQN")
        axes[1].plot(guided, color="red", label="GA guided DDQN")
        for axis in axes:
            axis.set_ylabel("Reward")
            axis.legend(loc="best")
        axes[1].set_xlabel("Scenario")
        figure.tight_layout()
        figure.savefig(destination, dpi=200, bbox_inches="tight")
        plt.close(figure)
    return destination


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    csv_path = root / "results" / "reward_comparison.csv"
    image_root = root / "results" / "images"
    for split, filename in (
        ("test", "reward_comparison.png"),
        ("train", "reward_train_comparison.png"),
    ):
        regular, guided = load_reward_series(csv_path, split)
        plot_reward_comparison(regular, guided, image_root / filename)


if __name__ == "__main__":
    main()
