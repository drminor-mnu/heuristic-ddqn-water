#!/usr/bin/env python3
"""Benchmark test-only runtime scaling for five pump-control methods."""

from __future__ import annotations

import csv
import json
import os
import random
import re
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import data_paths
from plot_style import STYLE


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DDQN_DETAIL_TITLE = "DDQN Inference Detail"
RUNTIME_PLOT_STYLE = {
    **STYLE,
    "figure.titlesize": 20,
    "axes.titlesize": 20,
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
}
WORKERS = 24
SAMPLE_SIZES = (50, 100, 250, 500, 1000, 1500, 2000, 2700)
METHOD_CONFIGS = {
    "Regular DQN": {
        "kind": "dqn",
        "checkpoint": ROOT / "trained_models"
        / "dqn_gru_regular_w0_5_w0_5_w0_0_w0_0_min2_tr0_6_acttype0.pkl",
        "weights": (0.5, 0.5, 0.0, 0.0),
    },
    "GA-guided DQN": {
        "kind": "dqn",
        "checkpoint": ROOT / "trained_models"
        / "dqn_guided_GA_w0_25_w0_4_w0_25_w0_1_ga0_6_min2_tr0_6_acttype0.pkl",
        "weights": (0.25, 0.4, 0.25, 0.1),
    },
    "PSO-guided DQN": {
        "kind": "dqn",
        "checkpoint": ROOT / "trained_models"
        / "dqn_guided_PSO_w0_25_w0_4_w0_25_w0_1_ga0_6_min2_tr0_6_acttype0.pkl",
        "weights": (0.25, 0.4, 0.25, 0.1),
    },
    "GA": {"kind": "ga", "weights": (0.25, 0.4, 0.25, 0.1)},
    "PSO": {"kind": "pso", "weights": (0.25, 0.4, 0.25, 0.1)},
}

_WORKER_KIND = None
_WORKER_MODEL = None
_WORKER_OPTIMIZER = None
_WORKER_WEIGHTS = None

def _stratum(path: str) -> tuple[int, int]:
    values = re.findall(r"\d+", path.rsplit("/", 1)[-1])
    if len(values) < 2:
        raise ValueError(f"cannot identify year and duration in {path}")
    return int(values[0]), int(values[1])


def stratified_order(paths: Iterable[str]) -> list[str]:
    """Return a deterministic round-robin order over year-duration strata."""
    groups: dict[tuple[int, int], list[str]] = defaultdict(list)
    for path in paths:
        groups[_stratum(path)].append(path)
    for values in groups.values():
        values.sort()
    keys = sorted(groups)
    ordered: list[str] = []
    for index in range(max(map(len, groups.values()), default=0)):
        ordered.extend(groups[key][index] for key in keys if index < len(groups[key]))
    return ordered


def cumulative_rows(
    method: str,
    completion_times: Sequence[float],
    sample_sizes: Sequence[int],
) -> list[dict[str, float | int | str]]:
    rows = []
    for size in sample_sizes:
        if size < 1 or size > len(completion_times):
            raise ValueError(f"invalid sample size {size}")
        elapsed = float(completion_times[size - 1])
        rows.append({
            "method": method,
            "test_scenarios": int(size),
            "elapsed_seconds": elapsed,
            "seconds_per_scenario": elapsed / size,
            "scenarios_per_second": size / elapsed,
        })
    return rows


def scenario_seed(index: int, run_seed: int = 20260724) -> int:
    return (run_seed * 100_000 + index) % (2**32)


def _initialize_worker(config: dict) -> None:
    global _WORKER_KIND, _WORKER_MODEL, _WORKER_OPTIMIZER, _WORKER_WEIGHTS
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/runtime-scaling-mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/runtime-scaling-cache")
    _WORKER_KIND = config["kind"]
    _WORKER_WEIGHTS = list(config["weights"])
    _WORKER_MODEL = None
    _WORKER_OPTIMIZER = None
    if _WORKER_KIND == "dqn":
        import torch
        from dqn_from_demon_v1 import (
            DqnGRU, Actions0, device, hidden_dim, num_layers,
        )
        model = DqnGRU(
            input_dim=5, hidden_dim=hidden_dim,
            output_dim=len(Actions0), num_layers=num_layers,
        ).to(device)
        model.load_state_dict(
            torch.load(config["checkpoint"], map_location=device)
        )
        model.eval()
        _WORKER_MODEL = model
    elif _WORKER_KIND == "ga":
        from genetic_algo import GeneticAlgorithm
        _WORKER_OPTIMIZER = GeneticAlgorithm(
            minutes=2, weights=_WORKER_WEIGHTS, act_type=0
        )
    elif _WORKER_KIND == "pso":
        from pso import ParticleSwarmOptimization
        _WORKER_OPTIMIZER = ParticleSwarmOptimization(
            minutes=2, weights=_WORKER_WEIGHTS, act_type=0
        )
    else:
        raise ValueError(f"unknown method kind {_WORKER_KIND}")


def _warm_worker(_: int) -> int:
    time.sleep(0.05)
    return os.getpid()


def _run_scenario(task: tuple[int, int, str]) -> int:
    run_seed, index, path = task
    seed = scenario_seed(index, run_seed)
    random.seed(seed)
    np.random.seed(seed)
    if _WORKER_KIND == "dqn":
        import torch
        from dqn_from_demon_v1 import test_model
        with torch.no_grad():
            test_model(
                _WORKER_MODEL, path, minutes=2,
                weights=_WORKER_WEIGHTS, act_type=0,
            )
    elif _WORKER_KIND == "ga":
        _WORKER_OPTIMIZER.run_genetic_algo(path)
    else:
        _WORKER_OPTIMIZER.run_pso(path)
    return index


def benchmark_method(
    method: str,
    paths: Sequence[str],
    workers: int = WORKERS,
    sample_sizes: Sequence[int] = SAMPLE_SIZES,
    run_seed: int = 20260724,
) -> list[dict[str, float | int | str]]:
    config = METHOD_CONFIGS[method]
    completion_times = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(config,),
    ) as executor:
        # Submitting one task per worker forces process creation and checkpoint
        # loading before the timed region.
        pids = set(executor.map(_warm_worker, range(workers), chunksize=1))
        while len(pids) < workers:
            pids.update(executor.map(_warm_worker, range(workers), chunksize=1))
        started = time.perf_counter()
        tasks = ((run_seed, index, path) for index, path in enumerate(paths))
        for completed, _ in enumerate(
            executor.map(_run_scenario, tasks, chunksize=1), start=1
        ):
            completion_times.append(time.perf_counter() - started)
            if completed % 50 == 0:
                print(
                    f"\r{method}: {completed}/{len(paths)}",
                    end="", flush=True,
                )
    print()
    return cumulative_rows(method, completion_times, sample_sizes)


def _load_paths() -> list[str]:
    # Legacy cache (kept for reference, no longer read): this snapshot was
    # written by repeat_heuristic_benchmark.py's own split generator with
    # split_seed=20260724, not the canonical seed-42 split below. Filename-only
    # overlap with the seed-42 test set is 41.1% (see docs/DATA_SPLIT_AUDIT.md
    # and docs/RESULTS_PROVENANCE.md), consistent with two independent
    # 2,700-of-6,696 samples rather than the same split.
    # document = json.loads(
    #     (RESULTS / "repeated_heuristics" / "fixed_test_2700.json")
    #     .read_text(encoding="utf-8")
    # )
    # ordered = stratified_order(document["paths"])
    _, _, test_inps = data_paths.load_fixed_split()
    ordered = stratified_order(
        [data_paths.resolve_path(path) for path in test_inps]
    )
    if len(ordered) != 2700 or len(set(ordered)) != 2700:
        raise ValueError("fixed test population must contain 2,700 unique paths")
    return ordered


def runtime_legend_label(method: str) -> str:
    return method.replace("DQN", "DDQN")


def plot_runtime_scaling(rows: Sequence[dict], png_path: Path) -> Path:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "runtime-scaling-plot")
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png_path = Path(png_path)
    with matplotlib.rc_context(RUNTIME_PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for method in METHOD_CONFIGS:
            selected = [row for row in rows if row["method"] == method]
            axes[0].plot(
                [row["test_scenarios"] for row in selected],
                [row["elapsed_seconds"] for row in selected],
                marker="o", label=runtime_legend_label(method),
            )
            if "DQN" in method:
                axes[1].plot(
                    [row["test_scenarios"] for row in selected],
                    [row["elapsed_seconds"] for row in selected],
                    marker="o", label=runtime_legend_label(method),
                )
        axes[0].set_title("All Methods")
        axes[1].set_title(DDQN_DETAIL_TITLE)
        for axis in axes:
            axis.set_xlabel("Number of Test Scenarios")
            axis.set_ylabel("Test Execution Time (seconds)")
            axis.grid(True, alpha=.3)
            axis.legend()
        fig.suptitle("Test-only Runtime Scaling (Training Excluded)")
        fig.tight_layout()
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    return png_path


def _write_outputs(rows: list[dict]) -> tuple[Path, Path, Path]:
    csv_path = RESULTS / "test_runtime_scaling.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "method", "test_scenarios", "elapsed_seconds",
            "seconds_per_scenario", "scenarios_per_second",
        ))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.6f}" if isinstance(value, float) else value
                for key, value in row.items()
            })

    png_path = RESULTS / "test_runtime_scaling.png"
    plot_runtime_scaling(rows, png_path)

    metadata_path = RESULTS / "test_runtime_scaling_metadata.json"
    metadata_path.write_text(json.dumps({
        "workers": WORKERS,
        "sample_sizes": SAMPLE_SIZES,
        "test_population": 2700,
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
    return csv_path, png_path, metadata_path


def main() -> None:
    paths = _load_paths()
    rows = []
    for method in METHOD_CONFIGS:
        print(f"\nBenchmarking {method}")
        rows.extend(benchmark_method(method, paths))
        _write_outputs(rows)
    print("\n".join(map(str, _write_outputs(rows))))


if __name__ == "__main__":
    main()
