#!/usr/bin/env python3
"""Run resumable five-seed GA/PSO benchmarks on one fixed test population."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import data_paths


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RESULTS = ROOT / "results"
RUN_DIR = RESULTS / "repeated_heuristics"
YEARS = ("10", "20", "30", "50", "80", "100")
DURATIONS = ("0060", "0120", "0180", "0240", "0360", "0540", "0720", "1080", "1440")
SEEDS = (101, 202, 303, 404, 505)
WEIGHTS = (0.25, 0.4, 0.25, 0.1)


@dataclass(frozen=True)
class LegacyResult:
    blocks: tuple[np.ndarray, ...]


def parse_legacy_result(path: str | Path) -> LegacyResult:
    groups: list[np.ndarray] = []
    current: list[list[float]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("mean_execution_time_seconds,"):
            continue
        if line:
            current.append([float(value) for value in line.split(",")])
        elif current:
            groups.append(np.asarray(current, dtype=float))
            current = []
    if current:
        groups.append(np.asarray(current, dtype=float))
    if len(groups) != 13:
        raise ValueError(f"{path} contains {len(groups)} legacy blocks instead of 13")
    return LegacyResult(tuple(groups))


def average_results(results: Sequence[LegacyResult]) -> LegacyResult:
    if not results:
        raise ValueError("at least one result is required")
    if any(len(result.blocks) != 13 for result in results):
        raise ValueError("every result must contain 13 blocks")
    means = []
    for index in range(13):
        shapes = {result.blocks[index].shape for result in results}
        if len(shapes) != 1:
            raise ValueError(f"block {index} shapes do not match: {shapes}")
        means.append(np.mean([result.blocks[index] for result in results], axis=0))
    return LegacyResult(tuple(means))


def write_legacy_result(result: LegacyResult, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for block in result.blocks:
        lines.extend(",".join(f"{value:.3f}" for value in row) for row in block)
        lines.append("")
    destination.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return destination


def append_mean_timing(path: str | Path, seconds: float) -> None:
    destination = Path(path)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(f"\nmean_execution_time_seconds,{seconds:.6f}\n")


def _numbers(path: str) -> tuple[str, str]:
    values = re.findall(r"\d+", Path(path).name)
    return values[0], values[1]


def fixed_test_paths(split_seed: int = 20260724) -> tuple[str, ...]:
    selected = [
        line.strip()
        for line in (ROOT / "data" / "selected_inp.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng = random.Random(split_seed)
    test = []
    for year in YEARS:
        for duration in DURATIONS:
            stratum = sorted(
                path for path in selected if _numbers(path) == (year, duration)
            )
            rng.shuffle(stratum)
            train_count = int(len(stratum) * 0.6)
            test.extend(stratum[train_count:])
    rng.shuffle(test)
    if len(test) != 2700 or len(set(test)) != 2700:
        raise ValueError(f"expected 2,700 unique test paths, got {len(test)}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    document = {"split_seed": split_seed, "count": len(test), "paths": test}
    (RUN_DIR / "fixed_test_2700.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return tuple(test)


def _scenario_worker(arguments):
    algorithm, run_seed, index, test_path = arguments
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/water-heuristic-mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/water-heuristic-cache")
    from genetic_algo import GeneticAlgorithm
    from pso import ParticleSwarmOptimization
    from perform_evaluate import count_changes, count_pumps
    from water_gym import Level

    scenario_seed = run_seed * 100_000 + index
    random.seed(scenario_seed)
    np.random.seed(scenario_seed)
    optimizer_class, method = {
        "genetic": (GeneticAlgorithm, "run_genetic_algo"),
        "pso": (ParticleSwarmOptimization, "run_pso"),
    }[algorithm]
    optimizer = optimizer_class(minutes=2, weights=list(WEIGHTS), act_type=0)
    actions, states, _, infos = getattr(optimizer, method)(test_path)
    pump100, pump170, individual = count_pumps(actions, 0)
    highest = max(float(state[-1]) for state in states)
    overpump = float(np.asarray(infos, dtype=np.float32)[:, -1].sum())
    year, duration = _numbers(test_path)
    return (
        year, duration, highest, float(count_changes(actions, 0)),
        float(pump100), float(pump170), np.asarray(individual, dtype=float),
        overpump, float(highest >= Level[-1]),
    )


def _aggregate_scenarios(rows) -> LegacyResult:
    shape = (len(YEARS), len(DURATIONS))
    counts = np.zeros(shape)
    level = np.zeros(shape)
    changes = np.zeros(shape)
    pump100 = np.zeros(shape)
    pump170 = np.zeros(shape)
    individual = np.zeros(shape + (5,))
    overpump = np.zeros(shape)
    flood = np.zeros(shape)
    year_index = {value: index for index, value in enumerate(YEARS)}
    duration_index = {value: index for index, value in enumerate(DURATIONS)}
    for row in rows:
        year, duration, high, switch, p100, p170, pumps, dry, flooded = row
        y, d = year_index[year], duration_index[duration]
        counts[y, d] += 1
        level[y, d] += high
        changes[y, d] += switch
        pump100[y, d] += p100
        pump170[y, d] += p170
        individual[y, d] += pumps
        overpump[y, d] += dry
        flood[y, d] += flooded
    if counts.sum() != 2700 or np.any(counts != 50):
        raise ValueError("scenario aggregation must contain exactly 50 cases per stratum")
    average = lambda values: values / counts
    blocks = [
        counts, average(level), average(changes), average(pump100), average(pump170),
        *(individual[index] / counts[index, :, None] for index in range(len(YEARS))),
        average(overpump), flood,
    ]
    return LegacyResult(tuple(np.asarray(block) for block in blocks))


def _run_one(algorithm: str, seed: int, test_paths: Sequence[str]) -> tuple[Path, float]:
    relative_name = f"repeated_heuristics/{algorithm}_seed_{seed}"
    output = RESULTS / f"{relative_name}.csv"
    timing_path = RUN_DIR / f"{algorithm}_seed_{seed}_timing.json"
    if output.is_file() and timing_path.is_file():
        parse_legacy_result(output)
        elapsed = float(json.loads(timing_path.read_text(encoding="utf-8"))["seconds"])
        return output, elapsed

    random.seed(seed)
    np.random.seed(seed)
    tasks = [(algorithm, seed, index, path) for index, path in enumerate(test_paths)]
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=24) as executor:
        for completed, row in enumerate(executor.map(_scenario_worker, tasks, chunksize=1), start=1):
            rows.append(row)
            if completed % 50 == 0 or completed == len(tasks):
                print(f"\r{algorithm} seed {seed}: {completed}/{len(tasks)}", end="", flush=True)
    print()
    elapsed = time.perf_counter() - started
    write_legacy_result(_aggregate_scenarios(rows), output)
    parse_legacy_result(output)
    timing_path.write_text(
        json.dumps({"algorithm": algorithm, "seed": seed, "seconds": elapsed}, indent=2) + "\n",
        encoding="utf-8",
    )
    return output, elapsed


def run_all() -> dict:
    # fixed_test_paths() (defined above) used its own split_seed=20260724 and
    # is kept for reference; the canonical seed-42 split is used instead so
    # this benchmark's test population matches the rest of the codebase. See
    # docs/RESULTS_PROVENANCE.md for the seed-20260724 vs seed-42 discrepancy.
    _, _, test_inps = data_paths.load_fixed_split()
    split_meta = data_paths.get_split_meta()
    print(f"split={split_meta['split_file']} seed={split_meta['split_seed']}")
    test_paths = tuple(data_paths.resolve_path(path) for path in test_inps)
    summary = {}
    for algorithm in ("genetic", "pso"):
        paths, timings = [], []
        for run_index, seed in enumerate(SEEDS, start=1):
            print(f"\n{algorithm.upper()} run {run_index}/5, seed={seed}")
            path, elapsed = _run_one(algorithm, seed, test_paths)
            paths.append(path)
            timings.append(elapsed)
            print(f"{algorithm} seed {seed} completed in {elapsed:.3f} seconds")
        mean = average_results([parse_legacy_result(path) for path in paths])
        output = RESULTS / f"{algorithm}_w0.25_w0.4_w0.25_w0.1_mean.csv"
        write_legacy_result(mean, output)
        mean_time = float(np.mean(timings))
        append_mean_timing(output, mean_time)
        summary[algorithm] = {
            "output": str(output), "mean_execution_time_seconds": mean_time,
            "run_execution_time_seconds": timings,
        }
    (RUN_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run_all(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
