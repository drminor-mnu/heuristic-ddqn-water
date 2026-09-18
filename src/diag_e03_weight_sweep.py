"""E3 (b)-option test: can any reward-weight choice pull the CURRENT-code
GA-only controller out of the action-0 fixed point?

GA-only has no training -> cheap. Runs evaluate_genetic_algo (real path,
run_genetic_algo -> current genetic_algo.py objective_function) on the first
30 test scenarios of the seed-42 split for several weight vectors.

Manuscript GA-only row (Table 11/12/15/16): max_level 5.783, n_switches 43.66,
n_dryrun_proxy 0.05 -- non-flooding.

Writes results/summary/E03_weight_sweep.csv; raw -> results/E03_weight_sweep/.
"""
from __future__ import annotations
import os
import sys
import csv
import time
import shutil
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.chdir(SRC)

RESULTS = os.path.join(ROOT, "results")
RAW_DIR = os.path.join(RESULTS, "E03_weight_sweep")
OUT_CSV = os.path.join(RESULTS, "summary", "E03_weight_sweep.csv")

YEARS = ["10", "20", "30", "50", "80", "100"]
DURATIONS = ["0060", "0120", "0180", "0240", "0360", "0540", "0720", "1080", "1440"]
N_TEST = 30
MS = dict(max_level_m=5.783, n_switches=43.66, n_dryrun_proxy=0.05)

WEIGHT_SETS = {
    "w_current_0.25_0.4_0.25_0.1": [0.25, 0.4, 0.25, 0.1],   # filename / current
    "w_2crit_1_1_0_0":             [1.0, 1.0, 0.0, 0.0],       # manuscript Regular's 2-criterion
    "w_level_gt_switch_0.4_0.25":  [0.4, 0.25, 0.25, 0.1],     # w1 > w2
    "w_level_dominant_0.6_0.2":    [0.6, 0.2, 0.1, 0.1],
    "w_level_only_1_0_0_0":        [1.0, 0.0, 0.0, 0.0],
}


def run_task(task):
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    tag = task["tag"]
    weights = task["weights"]
    seed = task["seed"]

    random.seed(seed)
    np.random.seed(seed)

    import data_paths
    import perform_evaluate as pe

    _, _, test_all = data_paths.load_fixed_split()
    test_inps = test_all[:N_TEST]

    stage = f"WSWEEP_{tag}_seed{seed}"
    params = {"minutes": 2, "weights": weights, "act_type": 0}
    t0 = time.perf_counter()
    pe.evaluate_genetic_algo(test_inps, YEARS, DURATIONS, seed=seed,
                             model_label=f"GA-only-{tag}", result_file=stage, **params)
    dt = time.perf_counter() - t0

    import pandas as pd
    scen = os.path.join(RESULTS, f"{stage}_scenario_metrics.csv")
    agg = os.path.join(RESULTS, f"{stage}.csv")
    df = pd.read_csv(scen)
    a0 = ((df.n_switches == 0) & (df.n_intervals_100 == 0) & (df.n_intervals_170 == 0))
    res = dict(
        tag=tag, weights="|".join(map(str, weights)), seed=seed, n_test=N_TEST,
        overflow_rate=round(float(df.overflow_flag.mean()), 4),
        mean_max_level_m=round(float(df.max_level_m.mean()), 4),
        min_max_level_m=round(float(df.max_level_m.min()), 4),
        action0_lock_rate=round(float(a0.mean()), 4),
        mean_n_switches=round(float(df.n_switches.mean()), 2),
        mean_n_intervals_100=round(float(df.n_intervals_100.mean()), 2),
        mean_n_intervals_170=round(float(df.n_intervals_170.mean()), 2),
        mean_dryrun_proxy=round(float(df.n_dryrun_proxy.mean()), 4),
        dist_max_level=round(float(df.max_level_m.mean() - MS["max_level_m"]), 3),
        elapsed_s=round(dt, 1),
    )
    os.makedirs(RAW_DIR, exist_ok=True)
    for p in (scen, agg):
        if os.path.exists(p):
            shutil.move(p, os.path.join(RAW_DIR, os.path.basename(p)))
    return res


def main():
    seeds = [1, 2]
    tasks = [dict(tag=t, weights=w, seed=s) for t, w in WEIGHT_SETS.items() for s in seeds]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["tag", "weights", "seed", "n_test", "overflow_rate", "mean_max_level_m",
              "min_max_level_m", "action0_lock_rate", "mean_n_switches",
              "mean_n_intervals_100", "mean_n_intervals_170", "mean_dryrun_proxy",
              "dist_max_level", "elapsed_s"]
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f"tasks={len(tasks)}", flush=True)
    t0 = time.perf_counter()
    n = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            n += 1
            with open(OUT_CSV, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(res)
            print(f">>> {n}/{len(tasks)} {res['tag']:32s} s{res['seed']} "
                  f"overflow={res['overflow_rate']} maxlvl={res['mean_max_level_m']} "
                  f"act0={res['action0_lock_rate']} nsw={res['mean_n_switches']} "
                  f"({res['elapsed_s']}s)", flush=True)

    print(f"\nall done {time.perf_counter()-t0:.0f}s\n")
    import pandas as pd
    df = pd.read_csv(OUT_CSV)
    g = df.groupby("tag").agg(
        overflow=("overflow_rate", "mean"),
        max_level=("mean_max_level_m", "mean"),
        act0=("action0_lock_rate", "mean"),
        n_switches=("mean_n_switches", "mean"),
        dry=("mean_dryrun_proxy", "mean"),
    ).reindex(list(WEIGHT_SETS))
    print("=== GA-only weight sweep (mean over seeds 1,2) ===")
    print(g.to_string())
    print(f"\nmanuscript GA-only target: max_level {MS['max_level_m']}, "
          f"n_switches {MS['n_switches']}, dry-running {MS['n_dryrun_proxy']}, non-flooding")


if __name__ == "__main__":
    main()
