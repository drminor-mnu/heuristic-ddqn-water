"""E6 controlled w1 sensitivity sweep + trade-off figure (R3-1, R4-6).

Single-variable sweep: w1 (level weight) varied, w2 = w3 = 0.25, w4 = 0.10
fixed. GA-only + PSO-only (shared objective), first 30 test scenarios of the
seed-42 split, seeds 1,2. vol_reward = current-level form (Eq. 13), current main.

Purpose: show the trade-off that IS the multi-objective sensitivity result --
raising w1 lowers max_level but raises switching. NOT a defect; the essence of
scalarized multi-objective control.

Selection criteria (no man_policy comparison):
  hard constraint : overflow rate == 0 on all test scenarios
  among those     : the balance point where none of the four metrics
                    (max_level / switching / pump-use / dry-running) is extreme

Writes results/summary/E06_curve.csv and results/summary/E06_tradeoff.png .
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
RAW_DIR = os.path.join(RESULTS, "E06_curve")
OUT_CSV = os.path.join(RESULTS, "summary", "E06_curve.csv")
FIG = os.path.join(RESULTS, "summary", "E06_tradeoff.png")

YEARS = ["10", "20", "30", "50", "80", "100"]
DUR = ["0060", "0120", "0180", "0240", "0360", "0540", "0720", "1080", "1440"]
N_TEST = 30

W2 = W3 = 0.25
W4 = 0.10
W1_GRID = [0.20, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.40, 0.45, 0.50]
ADOPTED_W1 = 0.40
ALGOS = ["GA", "PSO"]
SEEDS = [1, 2]


def run_task(task):
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    import data_paths
    import perform_evaluate as pe
    import pandas as pd

    w1 = task["w1"]
    algo = task["algo"]
    seed = task["seed"]
    weights = [round(w1, 2), W2, W3, W4]

    random.seed(seed)
    np.random.seed(seed)
    _, _, test_all = data_paths.load_fixed_split()
    test_inps = test_all[:N_TEST]

    stage = f"E06C_w1_{w1:.2f}_{algo}_s{seed}"
    params = {"minutes": 2, "weights": weights, "act_type": 0}
    t0 = time.perf_counter()
    if algo == "GA":
        pe.evaluate_genetic_algo(test_inps, YEARS, DUR, seed=seed,
                                 model_label=f"GA-w1{w1}", result_file=stage, **params)
    else:
        pe.evaluate_pso(test_inps, YEARS, DUR, seed=seed,
                        model_label=f"PSO-w1{w1}", result_file=stage, **params)
    scen = os.path.join(RESULTS, f"{stage}_scenario_metrics.csv")
    agg = os.path.join(RESULTS, f"{stage}.csv")
    df = pd.read_csv(scen)
    a0 = ((df.n_switches == 0) & (df.n_intervals_100 == 0) & (df.n_intervals_170 == 0))
    res = dict(
        w1=round(w1, 2), w1_minus_w2=round(w1 - W2, 2), algo=algo, seed=seed,
        weights="|".join(map(str, weights)),
        overflow_rate=round(float(df.overflow_flag.mean()), 4),
        n_overflow=int(df.overflow_flag.sum()),
        mean_max_level_m=round(float(df.max_level_m.mean()), 4),
        action0_lock_rate=round(float(a0.mean()), 4),
        mean_n_switches=round(float(df.n_switches.mean()), 3),
        mean_n_intervals_100=round(float(df.n_intervals_100.mean()), 2),
        mean_n_intervals_170=round(float(df.n_intervals_170.mean()), 2),
        mean_n_dryrun_proxy=round(float(df.n_dryrun_proxy.mean()), 4),
        elapsed_s=round(time.perf_counter() - t0, 1),
    )
    os.makedirs(RAW_DIR, exist_ok=True)
    for p in (scen, agg):
        if os.path.exists(p):
            shutil.move(p, os.path.join(RAW_DIR, os.path.basename(p)))
    return res


def make_figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = df.groupby("w1_minus_w2").agg(
        overflow=("overflow_rate", "mean"),
        max_level=("mean_max_level_m", "mean"),
        n_switches=("mean_n_switches", "mean"),
        i100=("mean_n_intervals_100", "mean"),
        dry=("mean_n_dryrun_proxy", "mean"),
    ).reset_index().sort_values("w1_minus_w2")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = g["w1_minus_w2"].values
    ok = g["overflow"].values == 0

    # max_level (left axis)
    c1 = "#1f5c8b"
    ax1.plot(x, g["max_level"], "o-", color=c1, label="max water level (m)")
    ax1.scatter(x[~ok], g["max_level"].values[~ok], s=140, facecolors="none",
                edgecolors="#c0392b", linewidths=2, zorder=5,
                label="overflow present (constraint violated)")
    ax1.set_xlabel(r"$w_1 - w_2$  (level weight $-$ switching weight)")
    ax1.set_ylabel("mean max water level (m)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.axhline(10.0, ls=":", color="#c0392b", lw=1, alpha=.6)
    ax1.axvline(0.0, ls="--", color="grey", lw=1, alpha=.6)

    # n_switches (right axis)
    ax2 = ax1.twinx()
    c2 = "#b9770e"
    ax2.plot(x, g["n_switches"], "s--", color=c2, label="mean on/off switches")
    ax2.set_ylabel("mean number of on/off switches", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    # adopted marker
    ax1.axvline(ADOPTED_W1 - W2, color="#117a2b", lw=2, alpha=.5)
    ax1.text(ADOPTED_W1 - W2, ax1.get_ylim()[1], "  adopted\n  [0.40,0.25,0.25,0.10]",
             va="top", ha="left", color="#117a2b", fontsize=9)

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="center right", fontsize=8, framealpha=.9)
    ax1.set_title("E6 weight sensitivity: level vs switching trade-off\n"
                  r"($w_2=w_3=0.25$, $w_4=0.10$ fixed; GA-only + PSO-only, 30 scen, 2 seeds)",
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)
    print(f"wrote {FIG}")


def main():
    tasks = [dict(w1=w1, algo=a, seed=s) for w1 in W1_GRID for a in ALGOS for s in SEEDS]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["w1", "w1_minus_w2", "algo", "seed", "weights", "overflow_rate",
              "n_overflow", "mean_max_level_m", "action0_lock_rate",
              "mean_n_switches", "mean_n_intervals_100", "mean_n_intervals_170",
              "mean_n_dryrun_proxy", "elapsed_s"]
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f"tasks={len(tasks)}  w1 grid={W1_GRID}  (w2=w3={W2}, w4={W4})", flush=True)
    t0 = time.perf_counter()
    rows = []
    n = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            rows.append(res)
            n += 1
            with open(OUT_CSV, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(res)
            print(f">>> {n}/{len(tasks)} w1={res['w1']:.2f} (w1-w2={res['w1_minus_w2']:+.2f}) "
                  f"{res['algo']:4s} s{res['seed']}  ovf={res['overflow_rate']} "
                  f"maxlvl={res['mean_max_level_m']:.3f} nsw={res['mean_n_switches']:.1f} "
                  f"dry={res['mean_n_dryrun_proxy']:.3f}  ({res['elapsed_s']}s)", flush=True)

    print(f"\nall done {time.perf_counter()-t0:.0f}s\n", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    g = df.groupby("w1_minus_w2").agg(
        w1=("w1", "first"),
        overflow=("overflow_rate", "mean"),
        max_level=("mean_max_level_m", "mean"),
        n_switches=("mean_n_switches", "mean"),
        i100=("mean_n_intervals_100", "mean"),
        i170=("mean_n_intervals_170", "mean"),
        dry=("mean_n_dryrun_proxy", "mean"),
    ).reset_index().sort_values("w1_minus_w2")
    print("=== E6 controlled w1 sweep (mean over GA/PSO x seeds 1,2) ===")
    print(g.to_string(index=False))
    try:
        make_figure(df)
    except Exception as e:
        print(f"figure step failed: {e}")


if __name__ == "__main__":
    main()
