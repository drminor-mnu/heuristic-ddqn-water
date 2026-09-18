"""E6 weight-sensitivity: principled selection of reward weights for the
4-criterion objective (R3-1, R4-6).

Runs on the CURRENT code (vol_reward reverted to current level, Eq. 13).
GA-only + PSO-only, first 30 test scenarios of the seed-42 split, seeds 1,2.
man_policy on the same 30 scenarios as the reference point.

Selection criteria (declared before running):
  1. hard constraint: overflow rate == 0 on all test scenarios
  2. among those, best balance of switching / pump-use / dry-running
  3. reference: man_policy (this run computes it on the same 30 scenarios)

Manuscript GA-only numbers (5.783 / 43.66 / 0.05) are NOT a selection target
-- they came from the pre-Eq.(13) objective and are recorded for reference only.

Writes results/summary/E06_weights.csv; raw -> results/E06_weights/.
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
RAW_DIR = os.path.join(RESULTS, "E06_weights")
OUT_CSV = os.path.join(RESULTS, "summary", "E06_weights.csv")

YEARS = ["10", "20", "30", "50", "80", "100"]
DUR = ["0060", "0120", "0180", "0240", "0360", "0540", "0720", "1080", "1440"]
N_TEST = 30
MS_REF = dict(max_level_m=5.783, n_switches=43.66, n_dryrun_proxy=0.05)  # reference only

WEIGHT_SETS = [
    ("w_0.40_0.25_0.25_0.10", [0.40, 0.25, 0.25, 0.10]),   # minimal w1<->w2 swap
    ("w_0.50_0.25_0.15_0.10", [0.50, 0.25, 0.15, 0.10]),
    ("w_0.45_0.35_0.10_0.10", [0.45, 0.35, 0.10, 0.10]),
    ("w_0.35_0.30_0.25_0.10", [0.35, 0.30, 0.25, 0.10]),   # boundary: w1 slightly > w2
    ("w_0.30_0.30_0.30_0.10", [0.30, 0.30, 0.30, 0.10]),   # boundary: w1 == w2
]
ALGOS = ["GA", "PSO"]
SEEDS = [1, 2]


def summarize(df, extra):
    a0 = ((df.n_switches == 0) & (df.n_intervals_100 == 0) & (df.n_intervals_170 == 0))
    r = dict(
        overflow_rate=round(float(df.overflow_flag.mean()), 4),
        mean_max_level_m=round(float(df.max_level_m.mean()), 4),
        min_max_level_m=round(float(df.max_level_m.min()), 3),
        max_max_level_m=round(float(df.max_level_m.max()), 3),
        action0_lock_rate=round(float(a0.mean()), 4),
        mean_n_switches=round(float(df.n_switches.mean()), 2),
        mean_n_intervals_100=round(float(df.n_intervals_100.mean()), 2),
        mean_n_intervals_170=round(float(df.n_intervals_170.mean()), 2),
        mean_n_dryrun_proxy=round(float(df.n_dryrun_proxy.mean()), 4),
    )
    r.update(extra)
    r["ref_dist_max_level"] = round(r["mean_max_level_m"] - MS_REF["max_level_m"], 3)
    return r


def run_task(task):
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    import data_paths
    import perform_evaluate as pe
    import pandas as pd

    _, _, test_all = data_paths.load_fixed_split()
    test_inps = test_all[:N_TEST]

    kind = task["kind"]
    t0 = time.perf_counter()

    if kind == "man_policy":
        stage = "E06_manpolicy"
        pe.evaluate_man_policy(test_inps, YEARS, DUR, seed=0, model_label="man_policy",
                               minutes=2, weights=[0.25, 0.4, 0.25, 0.1], act_type=0)
        # evaluate_man_policy writes man_policy_w0.25_w0.4_w0.25_w0.1_scenario_metrics.csv
        scen = os.path.join(RESULTS, "man_policy_w0.25_w0.4_w0.25_w0.1_scenario_metrics.csv")
        agg = os.path.join(RESULTS, "man_policy_w0.25_w0.4_w0.25_w0.1.csv")
        df = pd.read_csv(scen)
        res = summarize(df, dict(condition="man_policy_REFERENCE", algo="man_policy",
                                 weights="-", seed=0, elapsed_s=round(time.perf_counter() - t0, 1)))
    else:
        algo = task["algo"]
        tag = task["tag"]
        weights = task["weights"]
        seed = task["seed"]
        random.seed(seed)
        np.random.seed(seed)
        stage = f"E06_{tag}_{algo}_s{seed}"
        params = {"minutes": 2, "weights": weights, "act_type": 0}
        if algo == "GA":
            pe.evaluate_genetic_algo(test_inps, YEARS, DUR, seed=seed,
                                     model_label=f"GA-only-{tag}", result_file=stage, **params)
        else:
            pe.evaluate_pso(test_inps, YEARS, DUR, seed=seed,
                            model_label=f"PSO-only-{tag}", result_file=stage, **params)
        scen = os.path.join(RESULTS, f"{stage}_scenario_metrics.csv")
        agg = os.path.join(RESULTS, f"{stage}.csv")
        df = pd.read_csv(scen)
        res = summarize(df, dict(condition=tag, algo=algo,
                                 weights="|".join(map(str, weights)), seed=seed,
                                 elapsed_s=round(time.perf_counter() - t0, 1)))

    os.makedirs(RAW_DIR, exist_ok=True)
    for p in (scen, agg):
        if os.path.exists(p):
            shutil.move(p, os.path.join(RAW_DIR, os.path.basename(p)))
    return res


def main():
    tasks = [dict(kind="man_policy")]
    for tag, w in WEIGHT_SETS:
        for algo in ALGOS:
            for s in SEEDS:
                tasks.append(dict(kind="opt", tag=tag, algo=algo, weights=w, seed=s))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["condition", "algo", "weights", "seed", "overflow_rate",
              "mean_max_level_m", "min_max_level_m", "max_max_level_m",
              "action0_lock_rate", "mean_n_switches", "mean_n_intervals_100",
              "mean_n_intervals_170", "mean_n_dryrun_proxy", "ref_dist_max_level",
              "elapsed_s"]
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f"tasks={len(tasks)}", flush=True)
    print("SELECTION CRITERIA: (1) overflow_rate == 0  (2) best switching/pump-use/"
          "dry-running balance  (3) ref = man_policy", flush=True)
    t0 = time.perf_counter()
    n = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            n += 1
            with open(OUT_CSV, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(res)
            print(f">>> {n}/{len(tasks)} {res['condition']:24s} {res['algo']:10s} "
                  f"s{res['seed']}  overflow={res['overflow_rate']}  "
                  f"maxlvl={res['mean_max_level_m']}  nsw={res['mean_n_switches']}  "
                  f"i100={res['mean_n_intervals_100']}  dry={res['mean_n_dryrun_proxy']}  "
                  f"({res['elapsed_s']}s)", flush=True)

    print(f"\nall done {time.perf_counter()-t0:.0f}s\n")
    import pandas as pd
    df = pd.read_csv(OUT_CSV)
    opt = df[df.algo != "man_policy"]
    g = opt.groupby(["condition", "algo"]).agg(
        overflow=("overflow_rate", "mean"),
        max_level=("mean_max_level_m", "mean"),
        act0=("action0_lock_rate", "mean"),
        n_switches=("mean_n_switches", "mean"),
        i100=("mean_n_intervals_100", "mean"),
        i170=("mean_n_intervals_170", "mean"),
        dry=("mean_n_dryrun_proxy", "mean"),
    )
    print("=== E6 weight sweep (mean over seeds 1,2) ===")
    print(g.to_string())
    print("\n=== man_policy reference (same 30 scenarios) ===")
    print(df[df.algo == "man_policy"][["mean_max_level_m", "mean_n_switches",
          "mean_n_intervals_100", "mean_n_intervals_170", "mean_n_dryrun_proxy"]].to_string(index=False))


if __name__ == "__main__":
    main()
