#!/usr/bin/env python3
"""Five-run test-only runtime benchmark over the complete source population."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

import data_paths
from benchmark_test_runtime import (
    METHOD_CONFIGS, RESULTS, ROOT, WORKERS, benchmark_method, stratified_order,
)


FULL_SAMPLE_SIZES = tuple(range(500, 5001, 500))
RUN_SEEDS = (101, 202, 303, 404, 505)
RUN_DIR = RESULTS / "runtime_full_5runs"
RAW_PATH = RESULTS / "test_runtime_full_500_5000_5runs_raw.csv"
SUMMARY_PATH = RESULTS / "test_runtime_full_500_5000_5runs_summary.csv"
FIGURE_PATH = RESULTS / "test_runtime_full_500_5000_5runs.png"
METADATA_PATH = RESULTS / "test_runtime_full_500_5000_5runs_metadata.json"
RAW_FIELDS = (
    "method", "run", "seed", "test_scenarios", "elapsed_seconds",
    "seconds_per_scenario", "scenarios_per_second",
)
SUMMARY_FIELDS = (
    "method", "test_scenarios", "runs",
    "mean_elapsed_seconds", "std_elapsed_seconds",
    "mean_seconds_per_scenario", "std_seconds_per_scenario",
    "mean_scenarios_per_second", "std_scenarios_per_second",
)


def load_full_population() -> list[str]:
    # Full 7,440-scenario source population (10 durations, including "0010"),
    # not the 6,696-scenario population (9 durations) used to build the
    # canonical train/test split in data_paths.load_fixed_split() — this
    # benchmark measures runtime scaling over the complete source set, not
    # accuracy on the experimental population. See docs/RESULTS_PROVENANCE.md.
    paths = [
        data_paths.resolve_path(line.strip())
        for line in (ROOT / "data" / "selected_inp_relative.txt")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ordered = stratified_order(paths)
    if len(ordered) != 7440 or len(set(ordered)) != 7440:
        raise ValueError(
            f"expected 7,440 unique source scenarios, got {len(ordered)}"
        )
    return ordered


def summarize_runs(rows: Iterable[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), int(row["test_scenarios"]))].append(row)
    summary = []
    method_index = {method: index for index, method in enumerate(METHOD_CONFIGS)}
    for (method, size), values in sorted(
        groups.items(), key=lambda item: (method_index[item[0][0]], item[0][1])
    ):
        if len(values) != 5:
            raise ValueError(f"{method} at {size} has {len(values)} runs, not 5")
        elapsed = np.asarray(
            [float(value["elapsed_seconds"]) for value in values]
        )
        per_scenario = np.asarray(
            [float(value["seconds_per_scenario"]) for value in values]
        )
        throughput = np.asarray(
            [float(value["scenarios_per_second"]) for value in values]
        )
        summary.append({
            "method": method,
            "test_scenarios": size,
            "runs": len(values),
            "mean_elapsed_seconds": float(elapsed.mean()),
            "std_elapsed_seconds": float(elapsed.std(ddof=1)),
            "mean_seconds_per_scenario": float(per_scenario.mean()),
            "std_seconds_per_scenario": float(per_scenario.std(ddof=1)),
            "mean_scenarios_per_second": float(throughput.mean()),
            "std_scenarios_per_second": float(throughput.std(ddof=1)),
        })
    return summary


def canonicalize_raw_rows(rows: Iterable[dict]) -> list[dict]:
    """Match in-memory values to the precision persisted in the raw CSV."""
    canonical = []
    for row in rows:
        canonical.append({
            key: round(value, 9) if isinstance(value, float) else value
            for key, value in row.items()
        })
    return canonical


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.9f}" if isinstance(value, float) else value
                for key, value in row.items()
            })
    temporary.replace(path)


def _read_run(path: Path, method: str, run: int, seed: int) -> list[dict] | None:
    if not path.is_file():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if len(rows) != len(FULL_SAMPLE_SIZES):
        return None
    parsed = []
    previous = -1.0
    for row, size in zip(rows, FULL_SAMPLE_SIZES):
        value = {
            "method": row["method"],
            "run": int(row["run"]),
            "seed": int(row["seed"]),
            "test_scenarios": int(row["test_scenarios"]),
            "elapsed_seconds": float(row["elapsed_seconds"]),
            "seconds_per_scenario": float(row["seconds_per_scenario"]),
            "scenarios_per_second": float(row["scenarios_per_second"]),
        }
        if (
            value["method"] != method or value["run"] != run
            or value["seed"] != seed or value["test_scenarios"] != size
            or value["elapsed_seconds"] <= previous
        ):
            return None
        previous = value["elapsed_seconds"]
        parsed.append(value)
    return parsed


def _run_path(method: str, run: int, seed: int) -> Path:
    slug = method.lower().replace("-", "_").replace(" ", "_")
    return RUN_DIR / f"{slug}_run_{run}_seed_{seed}.csv"


def _plot(summary: list[dict]) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "runtime-full-five-plot"),
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.titlesize": 20,
        "lines.linewidth": 2.2,
        "lines.markersize": 6,
    })
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for method in METHOD_CONFIGS:
        selected = [row for row in summary if row["method"] == method]
        x = [row["test_scenarios"] for row in selected]
        y = [row["mean_elapsed_seconds"] for row in selected]
        error = [row["std_elapsed_seconds"] for row in selected]
        axes[0].errorbar(
            x, y, yerr=error, marker="o", capsize=4, label=method
        )
        if "DQN" in method:
            axes[1].errorbar(
                x, y, yerr=error, marker="o", capsize=4, label=method
            )
    axes[0].set_title("(a) All Methods", fontweight="bold")
    axes[1].set_title("(b) DQN Inference Detail", fontweight="bold")
    for axis in axes:
        axis.set_xlabel("Number of Test Scenarios")
        axis.set_ylabel("Test Execution Time (s)")
        axis.set_xticks(FULL_SAMPLE_SIZES[::2])
        axis.grid(True, alpha=0.3)
        axis.legend(frameon=True)
    fig.suptitle(
        "Test-only Runtime Scaling over the Full Dataset\n"
        "(Mean ± SD, Five Runs; Training Excluded)",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_combined(raw: list[dict]) -> list[dict]:
    raw = canonicalize_raw_rows(raw)
    _write_csv(RAW_PATH, RAW_FIELDS, raw)
    summary = summarize_runs(raw)
    _write_csv(SUMMARY_PATH, SUMMARY_FIELDS, summary)
    _plot(summary)
    METADATA_PATH.write_text(json.dumps({
        "source_population": 7440,
        "benchmarked_scenarios_per_run": 5000,
        "sample_sizes": FULL_SAMPLE_SIZES,
        "runs": 5,
        "run_seeds": RUN_SEEDS,
        "workers": WORKERS,
        "error_bars": "mean plus/minus one sample standard deviation (ddof=1)",
        "timing_excludes": [
            "training", "process startup", "module imports",
            "checkpoint loading", "CSV and plot writing",
        ],
        "methods": {
            method: {
                key: str(value) if isinstance(value, Path) else value
                for key, value in config.items()
            }
            for method, config in METHOD_CONFIGS.items()
        },
    }, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    paths = load_full_population()[:5000]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    raw = []
    for method in METHOD_CONFIGS:
        for run, seed in enumerate(RUN_SEEDS, start=1):
            path = _run_path(method, run, seed)
            rows = _read_run(path, method, run, seed)
            if rows is None:
                print(f"\n{method}: run {run}/5, seed={seed}")
                measured = benchmark_method(
                    method, paths, workers=WORKERS,
                    sample_sizes=FULL_SAMPLE_SIZES, run_seed=seed,
                )
                rows = [
                    {**row, "run": run, "seed": seed}
                    for row in measured
                ]
                rows = [
                    {key: row[key] for key in RAW_FIELDS}
                    for row in rows
                ]
                _write_csv(path, RAW_FIELDS, rows)
            else:
                print(f"\n{method}: reusing run {run}/5, seed={seed}")
            raw.extend(rows)
    summary = _write_combined(raw)
    print(f"\nraw rows={len(raw)}, summary rows={len(summary)}")
    print(RAW_PATH)
    print(SUMMARY_PATH)
    print(FIGURE_PATH)
    print(METADATA_PATH)


if __name__ == "__main__":
    main()
