"""E6 reinforcement (R3-1, R4-6):
  Part 1 -- larger validation of the adopted weight [0.40,0.25,0.25,0.10] and
           runner-up [0.35,0.30,0.25,0.10]: 3 seeds, 270-scenario stratified
           subsample (5 per (return_period, duration) stratum). Focus: is
           dry-running stable?
  Part 2 -- w1/w2 threshold: w1 in {0.30,0.32,0.34,0.36}, w2 fixed 0.30
           (w3,w4 = 0.30,0.10 so weights sum ~1), 2 seeds, 30 test scenarios.
           Is "w1 > w2" the exact condition or is a margin needed?

vol_reward is the reverted (current-level / Eq.13) form -- current main.
GA-only + PSO-only (shared objective). Writes:
  results/summary/E06_refine_validate.csv
  results/summary/E06_refine_threshold.csv
raw -> results/E06_refine/.
"""
from __future__ import annotations
import os
import re
import sys
import csv
import time
import shutil
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.chdir(SRC)

RESULTS = os.path.join(ROOT, "results")
RAW_DIR = os.path.join(RESULTS, "E06_refine")
OUT_VAL = os.path.join(RESULTS, "summary", "E06_refine_validate.csv")
OUT_THR = os.path.join(RESULTS, "summary", "E06_refine_threshold.csv")

YEARS = ["10", "20", "30", "50", "80", "100"]
DUR = ["0060", "0120", "0180", "0240", "0360", "0540", "0720", "1080", "1440"]
SUBSAMPLE_SEED = 12345          # same as diag_e03_manpolicy.py
PER_STRATUM = 5

VALIDATE_WEIGHTS = [
    ("w_0.40_0.25_0.25_0.10", [0.40, 0.25, 0.25, 0.10]),   # adopted
    ("w_0.35_0.30_0.25_0.10", [0.35, 0.30, 0.25, 0.10]),   # runner-up
]
VALIDATE_SEEDS = [1, 2, 3]

THRESHOLD_W1 = [0.30, 0.32, 0.34, 0.36]                    # w2 fixed at 0.30
THRESHOLD_SEEDS = [1, 2]
N_TEST_THR = 30


def stratified_subsample(test_all):
    strata = defaultdict(list)
    for p in test_all:
        n = re.findall(r"\d+", p.split("/")[-1])
        strata[(n[0], n[1])].append(p)
    rng = np.random.RandomState(SUBSAMPLE_SEED)
    picked = []
    for k in sorted(strata):
        lst = sorted(strata[k])
        idx = rng.choice(len(lst), size=min(PER_STRATUM, len(lst)), replace=False)
        picked.extend(lst[i] for i in idx)
    return picked


def summarize(df):
    a0 = ((df.n_switches == 0) & (df.n_intervals_100 == 0) & (df.n_intervals_170 == 0))
    return dict(
        overflow_rate=round(float(df.overflow_flag.mean()), 4),
        n_overflow=int(df.overflow_flag.sum()),
        n_scen=int(len(df)),
        mean_max_level_m=round(float(df.max_level_m.mean()), 4),
        min_max_level_m=round(float(df.max_level_m.min()), 3),
        max_max_level_m=round(float(df.max_level_m.max()), 3),
        action0_lock_rate=round(float(a0.mean()), 4),
        mean_n_switches=round(float(df.n_switches.mean()), 2),
        mean_n_intervals_100=round(float(df.n_intervals_100.mean()), 2),
        mean_n_intervals_170=round(float(df.n_intervals_170.mean()), 2),
        mean_n_dryrun_proxy=round(float(df.n_dryrun_proxy.mean()), 4),
        max_n_dryrun_proxy=int(df.n_dryrun_proxy.max()),
    )


def run_task(task):
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    import data_paths
    import perform_evaluate as pe
    import pandas as pd

    part = task["part"]
    algo = task["algo"]
    weights = task["weights"]
    seed = task["seed"]
    tag = task["tag"]

    random.seed(seed)
    np.random.seed(seed)

    _, _, test_all = data_paths.load_fixed_split()
    if part == "validate":
        test_inps = stratified_subsample(test_all)
    else:
        test_inps = test_all[:N_TEST_THR]

    stage = f"E06R_{part}_{tag}_{algo}_s{seed}"
    params = {"minutes": 2, "weights": weights, "act_type": 0}
    t0 = time.perf_counter()
    if algo == "GA":
        pe.evaluate_genetic_algo(test_inps, YEARS, DUR, seed=seed,
                                 model_label=f"GA-{tag}", result_file=stage, **params)
    else:
        pe.evaluate_pso(test_inps, YEARS, DUR, seed=seed,
                        model_label=f"PSO-{tag}", result_file=stage, **params)
    scen = os.path.join(RESULTS, f"{stage}_scenario_metrics.csv")
    agg = os.path.join(RESULTS, f"{stage}.csv")
    df = pd.read_csv(scen)
    res = dict(part=part, tag=tag, weights="|".join(map(str, weights)), algo=algo, seed=seed,
               elapsed_s=round(time.perf_counter() - t0, 1))
    res.update(summarize(df))
    os.makedirs(RAW_DIR, exist_ok=True)
    for p in (scen, agg):
        if os.path.exists(p):
            shutil.move(p, os.path.join(RAW_DIR, os.path.basename(p)))
    return res


def main():
    tasks = []
    for tag, w in VALIDATE_WEIGHTS:
        for algo in ("GA", "PSO"):
            for s in VALIDATE_SEEDS:
                tasks.append(dict(part="validate", tag=tag, weights=w, algo=algo, seed=s))
    for w1 in THRESHOLD_W1:
        w = [round(w1, 2), 0.30, 0.30, 0.10]
        tag = f"w1_{w1:.2f}_w2_0.30"
        for algo in ("GA", "PSO"):
            for s in THRESHOLD_SEEDS:
                tasks.append(dict(part="threshold", tag=tag, weights=w, algo=algo, seed=s))

    fields = ["part", "tag", "weights", "algo", "seed", "overflow_rate", "n_overflow",
              "n_scen", "mean_max_level_m", "min_max_level_m", "max_max_level_m",
              "action0_lock_rate", "mean_n_switches", "mean_n_intervals_100",
              "mean_n_intervals_170", "mean_n_dryrun_proxy", "max_n_dryrun_proxy",
              "elapsed_s"]
    for pth in (OUT_VAL, OUT_THR):
        os.makedirs(os.path.dirname(pth), exist_ok=True)
        with open(pth, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f"tasks={len(tasks)} (validate: 270-scen x 3 seeds x 2 weights x GA/PSO; "
          f"threshold: 30-scen x 2 seeds x 4 w1 x GA/PSO)", flush=True)
    t0 = time.perf_counter()
    n = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            n += 1
            out = OUT_VAL if res["part"] == "validate" else OUT_THR
            with open(out, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(res)
            print(f">>> {n}/{len(tasks)} [{res['part']:9s}] {res['tag']:22s} {res['algo']:4s} "
                  f"s{res['seed']}  ovf={res['overflow_rate']} ({res['n_overflow']}/{res['n_scen']})  "
                  f"maxlvl={res['mean_max_level_m']}  nsw={res['mean_n_switches']}  "
                  f"dry_mean={res['mean_n_dryrun_proxy']} dry_max={res['max_n_dryrun_proxy']}  "
                  f"({res['elapsed_s']}s)", flush=True)

    print(f"\nall done {time.perf_counter()-t0:.0f}s\n", flush=True)
    import pandas as pd
    for name, pth in (("PART 1 -- validate (270 scen, 3 seeds)", OUT_VAL),
                      ("PART 2 -- w1/w2 threshold (30 scen, 2 seeds)", OUT_THR)):
        df = pd.read_csv(pth)
        g = df.groupby(["tag", "algo"]).agg(
            overflow=("overflow_rate", "mean"),
            max_level=("mean_max_level_m", "mean"),
            act0=("action0_lock_rate", "mean"),
            n_switches=("mean_n_switches", "mean"),
            i100=("mean_n_intervals_100", "mean"),
            i170=("mean_n_intervals_170", "mean"),
            dry_mean=("mean_n_dryrun_proxy", "mean"),
            dry_spread=("mean_n_dryrun_proxy", lambda s: round(s.max() - s.min(), 3)),
        )
        print(f"\n=== {name} ===")
        print(g.to_string())


if __name__ == "__main__":
    main()
