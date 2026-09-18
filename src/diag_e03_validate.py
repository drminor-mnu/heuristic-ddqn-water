"""E3 fix validation: all 5 models, small scale, on the REAL entry points,
with the three fixes now default in dqn_from_demon_v1.py (action_list append,
gamma via params, min-max input standardization) + gamma=0.7.

500 train (first 500 of seed-42 fixed split) / 30 test (first 30) / seeds 1,2.
Pass criterion: overflow resolved (target: <= man_policy's 0/30, i.e. 0.0).

Note: GA-only / PSO-only have no training and no DDQN normalization -- gamma and
the standardization fix do not apply to them, and run_genetic_algo / run_pso
already maintain action_list. Their action-0 lock is the E00 section-8 fixed
point (starts at [0], never takes a first non-zero action). Expect them to
stay ~100% flooded; that is the honest degenerate-baseline result, not a
regression.

Writes results/summary/E03_validate.csv; raw outputs -> results/E03_validate/.
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
RAW_DIR = os.path.join(RESULTS, "E03_validate")
OUT_CSV = os.path.join(RESULTS, "summary", "E03_validate2.csv")
TRAINED = os.path.join(ROOT, "trained_models")

YEARS = ["10", "20", "30", "50", "80", "100"]
DURATIONS = ["0060", "0120", "0180", "0240", "0360", "0540", "0720", "1080", "1440"]
WEIGHTS = [0.40, 0.25, 0.25, 0.10]   # E6-adopted (was [0.25,0.4,0.25,0.1])
GAMMA = 0.7
BASE = {"epochs": 1, "minutes": 2, "weights": WEIGHTS, "act_type": 0}
GUIDED_EXTRA = {
    "ga_start": 0.6, "ga_end": 0.0, "eps_start": 0.3, "eps_end": 0.05,
    "batch_size": 20, "learning_rate": 1e-3, "sync_freq": 200,
}
N_TRAIN, N_TEST = 500, 30
MODELS = ["Regular", "GA-guided", "PSO-guided", "GA-only", "PSO-only"]


def run_task(task):
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    import torch
    torch.set_num_threads(2)

    model = task["model"]
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

    stage = f"VALID_{model}_seed{seed}"
    t0 = time.perf_counter()

    if model == "Regular":
        mp = os.path.join(TRAINED, f"{stage}.pkl")
        pe.evaluate_dpn_gru(train_inps, test_inps, YEARS, DURATIONS, train=True,
                            model_path=mp, seed=seed, model_label="Regular",
                            gamma=GAMMA, **BASE)
    elif model in ("GA-guided", "PSO-guided"):
        mp = os.path.join(TRAINED, f"{stage}.pkl")
        params = dict(BASE); params.update(GUIDED_EXTRA); params["gamma"] = GAMMA
        fn = pe.evaluate_dpn_guided_GA if model == "GA-guided" else pe.evaluate_dpn_guided_PSO
        fn(train_inps, test_inps, YEARS, DURATIONS, train=True, model_path=mp,
           seed=seed, model_label=model, **params)
    elif model == "GA-only":
        pe.evaluate_genetic_algo(test_inps, YEARS, DURATIONS, seed=seed,
                                 model_label="GA-only", result_file=stage, **BASE)
    elif model == "PSO-only":
        pe.evaluate_pso(test_inps, YEARS, DURATIONS, seed=seed,
                        model_label="PSO-only", result_file=stage, **BASE)
    dt = time.perf_counter() - t0

    scen_csv = os.path.join(RESULTS, f"{stage}_scenario_metrics.csv")
    agg_csv = os.path.join(RESULTS, f"{stage}.csv")
    import pandas as pd
    df = pd.read_csv(scen_csv)
    a0 = ((df.n_switches == 0) & (df.n_intervals_100 == 0) & (df.n_intervals_170 == 0))
    res = dict(
        model=model, seed=seed, gamma=GAMMA, n_train=N_TRAIN, n_test=N_TEST,
        overflow_rate=round(float(df.overflow_flag.mean()), 4),
        mean_max_level_m=round(float(df.max_level_m.mean()), 4),
        min_max_level_m=round(float(df.max_level_m.min()), 4),
        action0_lock_rate=round(float(a0.mean()), 4),
        mean_n_intervals_100=round(float(df.n_intervals_100.mean()), 2),
        mean_n_intervals_170=round(float(df.n_intervals_170.mean()), 2),
        mean_n_switches=round(float(df.n_switches.mean()), 2),
        mean_dryrun_proxy=round(float(df.n_dryrun_proxy.mean()), 4),
        elapsed_s=round(dt, 1),
    )
    os.makedirs(RAW_DIR, exist_ok=True)
    for p in (scen_csv, agg_csv):
        if os.path.exists(p):
            shutil.move(p, os.path.join(RAW_DIR, os.path.basename(p)))
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = [dict(model=m, seed=s) for m in models for s in seeds]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["model", "seed", "gamma", "n_train", "n_test", "overflow_rate",
              "mean_max_level_m", "min_max_level_m", "action0_lock_rate",
              "mean_n_intervals_100", "mean_n_intervals_170", "mean_n_switches",
              "mean_dryrun_proxy", "elapsed_s"]
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f"tasks={len(tasks)} workers={args.workers} gamma={GAMMA}", flush=True)
    t0 = time.perf_counter()
    n = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            n += 1
            with open(OUT_CSV, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(res)
            print(f">>> {n}/{len(tasks)} {res['model']:12s} seed{res['seed']}  "
                  f"overflow={res['overflow_rate']}  max_level={res['mean_max_level_m']}  "
                  f"act0={res['action0_lock_rate']}  ({res['elapsed_s']}s)", flush=True)

    print(f"\nall done {time.perf_counter()-t0:.0f}s\n")
    import pandas as pd
    df = pd.read_csv(OUT_CSV)
    print(df.sort_values(["model", "seed"]).to_string(index=False))
    print("\n=== mean over seeds ===")
    g = df.groupby("model").agg(
        overflow=("overflow_rate", "mean"),
        max_level=("mean_max_level_m", "mean"),
        act0=("action0_lock_rate", "mean"),
        i100=("mean_n_intervals_100", "mean"),
        dry=("mean_dryrun_proxy", "mean"),
    ).reindex(models)
    print(g.to_string())
    ok = df[df.model.isin(["Regular", "GA-guided", "PSO-guided"])].overflow_rate.max()
    print(f"\nlearned-model max overflow_rate across seeds: {ok}  "
          f"({'PASS' if ok == 0 else 'NOT resolved'})")


if __name__ == "__main__":
    main()
