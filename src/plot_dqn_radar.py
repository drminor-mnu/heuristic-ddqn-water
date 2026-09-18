#!/usr/bin/env python3
"""Create an English-labelled DQN operational-performance radar chart."""

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
REGULAR = RESULTS / "dqn_gru_regular_w0_5_w0_5_w0_0_w0_0_min2_tr0_6_acttype0.csv"
GUIDED = RESULTS / "dqn_guided_GA_w0_25_w0_4_w0_25_w0_1_ga0_6_min2_tr0_6_acttype0.csv"
OUTPUT = RESULTS / "dqn_regular_vs_ga_guided_radar.png"
SUMMARY = RESULTS / "dqn_regular_vs_ga_guided_radar_summary.csv"


def relative_cost_scores(costs: np.ndarray) -> np.ndarray:
    values = np.asarray(costs, dtype=float)
    if values.ndim != 2 or np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("costs must be a finite non-negative model-by-metric matrix")
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
    level = _weighted_mean(blocks[1], counts)
    switching = _weighted_mean(blocks[2], counts)
    pump100 = _weighted_mean(blocks[3], counts)
    pump170 = _weighted_mean(blocks[4], counts)
    dry_running = _weighted_mean(blocks[11], counts)
    return {
        "Water Level": level,
        "Pump Switching": switching,
        "Energy Use Proxy": 100.0 * pump100 + 170.0 * pump170,
        "Dry Running": dry_running,
    }


def create_radar() -> tuple[Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dqn-radar-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("Regular DQN", "GA-guided DQN")
    raw = [raw_metrics(REGULAR), raw_metrics(GUIDED)]
    cost_keys = ("Pump Switching", "Energy Use Proxy", "Dry Running")
    relative = relative_cost_scores(np.asarray([
        [row[key] for key in cost_keys] for row in raw
    ]))
    level_scores = np.clip(
        (10.0 - np.asarray([row["Water Level"] for row in raw])) / (10.0 - 4.7),
        0.0, 1.0,
    )
    scores = np.column_stack((level_scores, relative))
    labels = ("Water Level Safety", "Switching Stability", "Energy Efficiency", "Dry-running Safety")

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    fig, axis = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"projection": "polar"})
    colors = ("tab:blue", "tab:orange")
    for index, name in enumerate(names):
        values = np.r_[scores[index], scores[index, 0]]
        axis.plot(closed_angles, values, linewidth=2, color=colors[index], label=name)
        axis.fill(closed_angles, values, color=colors[index], alpha=.15)
    axis.set_xticks(angles, labels)
    axis.set_ylim(0, 1)
    axis.set_yticks((.2, .4, .6, .8, 1.0))
    axis.set_yticklabels(("0.2", "0.4", "0.6", "0.8", "1.0"))
    axis.set_title("Operational Performance Comparison\n(Higher Score Is Better)", pad=24)
    axis.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12))
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    OUTPUT.write_bytes(buffer.getvalue())
    plt.close(fig)

    with SUMMARY.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "metric", "raw_value", "score"))
        raw_keys = ("Water Level", "Pump Switching", "Energy Use Proxy", "Dry Running")
        for model_index, name in enumerate(names):
            for metric_index, (label, key) in enumerate(zip(labels, raw_keys)):
                writer.writerow((name, label, f"{raw[model_index][key]:.6f}",
                                 f"{scores[model_index, metric_index]:.6f}"))
    return OUTPUT, SUMMARY


if __name__ == "__main__":
    print("\n".join(str(path) for path in create_radar()))
