"""E3 diagnostic 3-3 (real-code-path version).

Runs the GENUINE guided path -- perform_evaluate.evaluate_dpn_guided_GA/_PSO
-> dqn_from_demon_v1._train_guided_by_optimizer / test_model -- with only two
injected changes, both external to the algorithm:

  * gamma : passed via params (the real path already reads params.get('gamma',0.1))
  * input normalization : env var WATER_INPUT_NORM ('sum' default == original,
    'minmax' == candidate fix). Implemented by the _diag_input_norm() hook added
    on branch diag/e3-guided-realpath; 'sum' mode is bit-identical to the
    original x/(x.sum()+1e-5).

No reimplementation of the training loop (unlike diag_e03_guided_fix.py, whose
clone diverged from the original in RNG order and in action-history handling --
the real loop passes a static [0] action_list to the optimizer every step).

Configs (each: GA + PSO, seeds 1,2, 500 train / 30 test):
  ctrl : gamma 0.1, norm sum     -> MUST reproduce diag_e03_guided.py results
  fix  : gamma 0.7, norm minmax  -> the actual step-3 answer

Writes results/summary/E03_guided_realpath.csv  and relocates raw outputs to
results/E03_diag_guided_realpath/.
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
RAW_DIR = os.path.join(RESULTS, "E03_diag_guided_realpath")
OUT_CSV = os.path.join(RESULTS, "summary", "E03_guided_realpath.csv")
TRAINED = os.path.join(ROOT, "trained_models")

YEARS = ["10", "20", "30", "50", "80", "100"]
DURATIONS = ["0060", "0120", "0180", "0240", "0360", "0540", "0720", "1080", "1440"]
BASE = {"epochs": 1, "minutes": 2, "weights": [0.25, 0.4, 0.25, 0.1], "act_type": 0}
GUIDED_EXTRA = {
    "ga_start": 0.6, "ga_end": 0.0, "eps_start": 0.3, "eps_end": 0.05,
    "batch_size": 20, "learning_rate": 1e-3, "sync_freq": 200,
}
N_TRAIN, N_TEST = 500, 30

CONFIGS = {
    "ctrl":       dict(gamma=0.1, norm="sum",    fix_al=False),
    "fix":        dict(gamma=0.7, norm="minmax", fix_al=False),
    # action_list bug fix isolated (original settings otherwise) + combined
    "alfix_orig": dict(gamma=0.1, norm="sum",    fix_al=True),
    "alfix_full": dict(gamma=0.7, norm="minmax", fix_al=True),
}


def run_task(task):
    cfg = task["cfg"]
    guide = task["guide"]
    seed = task["seed"]
    conf = CONFIGS[cfg]

    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["WATER_INPUT_NORM"] = conf["norm"]   # read by _diag_input_norm() at call time
    os.environ["WATER_FIX_ACTIONLIST"] = "1" if conf.get("fix_al") else "0"
    import torch
    torch.set_num_threads(2)

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

    stage = f"DIAGRP_{cfg}_{guide}_{N_TRAIN}_seed{seed}"
    model_path = os.path.join(TRAINED, f"{stage}.pkl")
    params = dict(BASE)
    params.update(GUIDED_EXTRA)
    params["gamma"] = conf["gamma"]

    t0 = time.perf_counter()
    fn = pe.evaluate_dpn_guided_GA if guide == "GA" else pe.evaluate_dpn_guided_PSO
    fn(train_inps, test_inps, YEARS, DURATIONS, train=True, model_path=model_path,
       seed=seed, model_label=f"{guide}-guided-{cfg}", **params)
    dt = time.perf_counter() - t0

    scen_csv = os.path.join(RESULTS, f"{stage}_scenario_metrics.csv")
    agg_csv = os.path.join(RESULTS, f"{stage}.csv")
    import pandas as pd
    df = pd.read_csv(scen_csv)
    a0 = ((df.n_switches == 0) & (df.n_intervals_100 == 0) & (df.n_intervals_170 == 0))
    res = dict(
        config=cfg, gamma=conf["gamma"], norm=conf["norm"], fix_al=int(bool(conf.get("fix_al"))), guide=guide, seed=seed,
        n_train=N_TRAIN, n_test=N_TEST,
        overflow_rate=round(float(df.overflow_flag.mean()), 4),
        mean_max_level_m=round(float(df.max_level_m.mean()), 4),
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
    ap.add_argument("--configs", default="alfix_orig,alfix_full")
    ap.add_argument("--guides", default="GA,PSO")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfgs = [c.strip() for c in args.configs.split(",")]
    guides = [g.strip() for g in args.guides.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = [dict(cfg=c, guide=g, seed=s) for c in cfgs for g in guides for s in seeds]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["config", "gamma", "norm", "fix_al", "guide", "seed", "n_train", "n_test",
              "overflow_rate", "mean_max_level_m", "action0_lock_rate",
              "mean_n_intervals_100", "mean_n_intervals_170", "mean_n_switches",
              "mean_dryrun_proxy", "elapsed_s"]
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f"tasks={len(tasks)} workers={args.workers}", flush=True)
    t0 = time.perf_counter()
    n = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            n += 1
            with open(OUT_CSV, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(res)
            print(f">>> {n}/{len(tasks)} [{res['config']}] {res['guide']} seed{res['seed']} "
                  f"overflow={res['overflow_rate']} max_level={res['mean_max_level_m']} "
                  f"act0={res['action0_lock_rate']} i100={res['mean_n_intervals_100']} "
                  f"({res['elapsed_s']}s)", flush=True)

    print(f"\nall done {time.perf_counter()-t0:.0f}s\n")
    import pandas as pd
    df = pd.read_csv(OUT_CSV)
    print(df.sort_values(["config", "guide", "seed"]).to_string(index=False))
    print("\n=== mean over seeds ===")
    print(df.groupby(["config", "guide"])[
        ["overflow_rate", "mean_max_level_m", "action0_lock_rate",
         "mean_n_intervals_100", "mean_dryrun_proxy"]].mean().to_string())


if __name__ == "__main__":
    main()
