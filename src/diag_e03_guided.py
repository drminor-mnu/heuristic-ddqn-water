"""E3 diagnostic 3b: does heuristic GUIDANCE let the 4-criterion DDQN escape
the action-0 trap that E00 / D0-D5 showed?

Runs the REAL guided training path (perform_evaluate.evaluate_dpn_guided_GA /
_PSO, i.e. dqn_from_demon_v1._train_guided_by_optimizer) -- no reimplementation.
Only change vs E3: 500 train scenarios (first 500 of the seed-42 fixed split,
same subset D0-D6 used) and 30 test scenarios (first 30), seeds {1,2}.

Params identical to the E3 driver: weights [0.25,0.4,0.25,0.1], gamma 0.1,
ga_start 0.6 -> ga_end 0.0, eps 0.3 -> 0.05, sync_freq 200, batch 20.

Writes results/summary/E03_guided_diag.csv and relocates raw outputs to
results/E03_diag_guided/.
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
RAW_DIR = os.path.join(RESULTS, "E03_diag_guided")
OUT_CSV = os.path.join(RESULTS, "summary", "E03_guided_diag.csv")
TRAINED = os.path.join(ROOT, "trained_models")

YEARS = ["10", "20", "30", "50", "80", "100"]
DURATIONS = ["0060", "0120", "0180", "0240", "0360", "0540", "0720", "1080", "1440"]
BASE = {"epochs": 1, "minutes": 2, "weights": [0.25, 0.4, 0.25, 0.1], "act_type": 0}
GUIDED_EXTRA = {
    "ga_start": 0.6, "ga_end": 0.0, "eps_start": 0.3, "eps_end": 0.05,
    "gamma": 0.1, "batch_size": 20, "learning_rate": 1e-3, "sync_freq": 200,
}
N_TRAIN = 500
N_TEST = 30
LEVEL_MAX = 10.0


def run_task(task):
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    import torch
    torch.set_num_threads(2)

    guide = task["guide"]   # 'GA' or 'PSO'
    seed = task["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import data_paths
    import perform_evaluate as pe

    train_all, _, test_all = data_paths.load_fixed_split()
    train_inps = train_all[:N_TRAIN]
    test_inps = test_all[:N_TEST]

    stage = f"DIAG_{guide}guided_{N_TRAIN}_seed{seed}"
    model_path = os.path.join(TRAINED, f"{stage}.pkl")

    params = dict(BASE)
    params.update(GUIDED_EXTRA)

    t0 = time.perf_counter()
    fn = pe.evaluate_dpn_guided_GA if guide == "GA" else pe.evaluate_dpn_guided_PSO
    fn(train_inps, test_inps, YEARS, DURATIONS, train=True, model_path=model_path,
       seed=seed, model_label=f"{guide}-guided", **params)
    dt = time.perf_counter() - t0

    # raw scenario metrics the function just wrote
    scen_csv = os.path.join(RESULTS, f"{stage}_scenario_metrics.csv")
    agg_csv = os.path.join(RESULTS, f"{stage}.csv")
    import pandas as pd
    df = pd.read_csv(scen_csv)
    a0 = ((df.n_switches == 0) & (df.n_intervals_100 == 0) & (df.n_intervals_170 == 0))
    res = dict(
        guide=guide, seed=seed, n_train=N_TRAIN, n_test=N_TEST,
        overflow_rate=round(float(df.overflow_flag.mean()), 4),
        mean_max_level_m=round(float(df.max_level_m.mean()), 4),
        action0_lock_rate=round(float(a0.mean()), 4),
        mean_n_intervals_100=round(float(df.n_intervals_100.mean()), 2),
        mean_n_intervals_170=round(float(df.n_intervals_170.mean()), 2),
        mean_n_switches=round(float(df.n_switches.mean()), 2),
        mean_dryrun_proxy=round(float(df.n_dryrun_proxy.mean()), 4),
        elapsed_s=round(dt, 1),
    )
    # relocate raw outputs out of results/ top level
    os.makedirs(RAW_DIR, exist_ok=True)
    for p in (scen_csv, agg_csv):
        if os.path.exists(p):
            shutil.move(p, os.path.join(RAW_DIR, os.path.basename(p)))
    return res


def main():
    tasks = [dict(guide=g, seed=s) for g in ("GA", "PSO") for s in (1, 2)]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["guide", "seed", "n_train", "n_test", "overflow_rate", "mean_max_level_m",
              "action0_lock_rate", "mean_n_intervals_100", "mean_n_intervals_170",
              "mean_n_switches", "mean_dryrun_proxy", "elapsed_s"]
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            done += 1
            with open(OUT_CSV, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(res)
            print(f">>> {done}/{len(tasks)}  {res['guide']}-guided seed{res['seed']}  "
                  f"overflow={res['overflow_rate']}  max_level={res['mean_max_level_m']}  "
                  f"act0={res['action0_lock_rate']}  ({res['elapsed_s']}s)", flush=True)

    print(f"\nall done in {time.perf_counter()-t0:.0f}s")
    import pandas as pd
    df = pd.read_csv(OUT_CSV)
    agg = df.groupby("guide").agg(
        overflow_rate=("overflow_rate", "mean"),
        mean_max_level_m=("mean_max_level_m", "mean"),
        action0_lock_rate=("action0_lock_rate", "mean"),
        mean_n_intervals_100=("mean_n_intervals_100", "mean"),
        mean_dryrun_proxy=("mean_dryrun_proxy", "mean"),
    )
    print("\n=== guided diagnostic (mean over seeds 1,2) ===")
    print(agg.to_string())
    print("\nper-seed:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
