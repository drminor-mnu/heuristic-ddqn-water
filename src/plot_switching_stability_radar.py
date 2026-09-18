#!/usr/bin/env python3
"""Radar comparison for Regular and switching-focused GA-guided DDQN."""

from __future__ import annotations

import csv
import io
import os
import tempfile
from pathlib import Path

import numpy as np

from repeat_heuristic_benchmark import parse_legacy_result


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REGULAR = (
    RESULTS / "dqn_gru_regular_w0_5_w0_5_w0_0_w0_0_min2_tr0_6_acttype0.csv"
)
GUIDED = (
    RESULTS
    / "dqn_guided_GA_w0_15_w0_6_w0_15_w0_1_ga0_6_min2_tr0_6_acttype0.csv"
)
OLD_GUIDED = (
    RESULTS
    / "dqn_guided_GA_w0_25_w0_4_w0_25_w0_1_ga0_6_min2_tr0_6_acttype0.csv"
)
OUTPUT = RESULTS / "dqn_regular_vs_ga_guided_switching_radar.png"
SUMMARY = RESULTS / "dqn_regular_vs_ga_guided_switching_radar_summary.csv"
LABEL_RADIUS = 1.12


def relative_cost_scores(costs: np.ndarray) -> np.ndarray:
    values = np.asarray(costs, dtype=float)
    if values.ndim != 2 or np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("costs must be a finite non-negative matrix")
    best = values.min(axis=0)
    scores = np.ones_like(values)
    for column in range(values.shape[1]):
        if best[column] == 0:
            scores[:, column] = (values[:, column] == 0).astype(float)
        else:
            scores[:, column] = best[column] / values[:, column]
    return scores


def _weighted_mean(block: np.ndarray, counts: np.ndarray) -> float:
    return float(np.sum(block * counts) / np.sum(counts))


def raw_metrics(path: Path) -> dict[str, float]:
    blocks = parse_legacy_result(path).blocks
    counts = blocks[0]
    pump100 = _weighted_mean(blocks[3], counts)
    pump170 = _weighted_mean(blocks[4], counts)
    return {
        "Water Level": _weighted_mean(blocks[1], counts),
        "Pump Switching": _weighted_mean(blocks[2], counts),
        "Energy Use Proxy": 100.0 * pump100 + 170.0 * pump170,
        "Dry Running": _weighted_mean(blocks[11], counts),
    }


def create_radar() -> tuple[Path, Path]:
    if not REGULAR.is_file() or not GUIDED.is_file():
        raise FileNotFoundError("regular or switching-focused result CSV is missing")
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "switching-stability-radar-mpl"),
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("Regular DQN", "Switching-focused GA-guided DQN")
    raw = [raw_metrics(REGULAR), raw_metrics(GUIDED)]
    cost_keys = ("Pump Switching", "Energy Use Proxy", "Dry Running")
    relative = relative_cost_scores(np.asarray([
        [row[key] for key in cost_keys] for row in raw
    ]))
    level_scores = np.clip(
        (10.0 - np.asarray([row["Water Level"] for row in raw]))
        / (10.0 - 4.7),
        0.0, 1.0,
    )
    scores = np.column_stack((level_scores, relative))
    labels = (
        "Water Level Safety", "Switching Stability",
        "Energy Efficiency", "Dry-running Safety",
    )
    raw_keys = (
        "Water Level", "Pump Switching", "Energy Use Proxy", "Dry Running",
    )
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "legend.fontsize": 12,
    })
    fig, axis = plt.subplots(
        figsize=(9.5, 9.5), subplot_kw={"projection": "polar"}
    )
    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)
    colors = ("tab:blue", "tab:orange")
    for index, name in enumerate(names):
        values = np.r_[scores[index], scores[index, 0]]
        axis.plot(
            closed_angles, values, linewidth=2.5,
            marker="o", color=colors[index], label=name,
        )
        axis.fill(closed_angles, values, color=colors[index], alpha=0.14)

    axis.set_xticks(angles)
    axis.set_xticklabels([])
    alignments = (
        ("center", "bottom"), ("left", "center"),
        ("center", "top"), ("right", "center"),
    )
    for angle, label, (horizontal, vertical) in zip(
        angles, labels, alignments
    ):
        axis.text(
            angle, LABEL_RADIUS, label,
            ha=horizontal, va=vertical,
            fontsize=16, fontweight="semibold", clip_on=False,
        )
    axis.set_ylim(0, 1)
    axis.set_yticks((0.2, 0.4, 0.6, 0.8, 1.0))
    axis.set_yticklabels(("0.2", "0.4", "0.6", "0.8", "1.0"))
    fig.suptitle(
        "Operational Performance Comparison\n(Higher Score Is Better)",
        y=0.97, fontsize=18, fontweight="bold",
    )
    axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16),
        ncol=2, frameon=True,
    )
    fig.subplots_adjust(left=0.17, right=0.83, bottom=0.23, top=0.78)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    OUTPUT.write_bytes(buffer.getvalue())
    plt.close(fig)

    old_switching = (
        raw_metrics(OLD_GUIDED)["Pump Switching"]
        if OLD_GUIDED.is_file() else float("nan")
    )
    with SUMMARY.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "model", "metric", "raw_value", "score",
            "old_ga_guided_switching",
        ))
        for model_index, name in enumerate(names):
            for metric_index, (label, key) in enumerate(zip(labels, raw_keys)):
                writer.writerow((
                    name, label, f"{raw[model_index][key]:.6f}",
                    f"{scores[model_index, metric_index]:.6f}",
                    f"{old_switching:.6f}" if key == "Pump Switching" else "",
                ))
    return OUTPUT, SUMMARY


if __name__ == "__main__":
    print("\n".join(str(path) for path in create_radar()))
