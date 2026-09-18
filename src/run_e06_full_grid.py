#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E6 full grid (R3-1, R4-6, R2 1st round): >=24 weight combinations on the
w1+w2+w3+w4=1 simplex (including the adopted [0.40,0.25,0.25,0.10] and 4
single-weight-1.0 extreme corners), GA-only/PSO-only (no DQN training,
same pattern as diag_e06_weights.py/diag_e06_refine.py -- reuses
perform_evaluate.evaluate_genetic_algo/evaluate_pso unmodified), 2 seeds
each.

Scale note (explicit reduction, per REVISION_EXPERIMENT_PLAN.md E6's own
allowance for a stratified subsample): uses the first N_TEST=30 fixed-split
test scenarios per combination, not the full 2,700 or a 270-scenario
stratified subsample -- 28 combos x 2 algos x 2 seeds x 30 scenarios was
sized to complete in this session; the plan's suggested "represent 3
combos re-confirmed on full data" step was not done (see
docs/REMAINING_WORK.md). This mirrors the scale already used for
diag_e06_refine.py's "threshold" part (N_TEST_THR=30), not its "validate"
part (270).

Run from src/.
"""
import csv
import os
import random
import re
import shutil
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
RESULTS = os.path.join(ROOT, 'results')
RAW_DIR = os.path.join(RESULTS, 'E06_weights')
OUT_GRID = os.path.join(RESULTS, 'E06_weights', 'grid_results.csv')
SEEDS = [1, 2]
N_TEST = 30
YEARS = ["10", "20", "30", "50", "80", "100"]
DUR = ["0060", "0120", "0180", "0240", "0360", "0540", "0720", "1080", "1440"]
GRID_RNG_SEED = 20260901


def sample_combos():
    """24 interior Dirichlet(alpha=[2,2,2,2]) samples (fixed seed) +
    adopted weight + 4 extreme corners = 29 combos."""
    rng = np.random.RandomState(GRID_RNG_SEED)
    combos = []
    for _ in range(24):
        w = rng.dirichlet([2, 2, 2, 2])
        combos.append([round(float(x), 3) for x in w])
    combos.append([0.40, 0.25, 0.25, 0.10])  # adopted (E3 v2 / E5)
    combos.append([1.0, 0.0, 0.0, 0.0])
    combos.append([0.0, 1.0, 0.0, 0.0])
    combos.append([0.0, 0.0, 1.0, 0.0])
    combos.append([0.0, 0.0, 0.0, 1.0])
    return combos


def summarize(rows):
    n = len(rows)
    overflow = sum(1 for r in rows if float(r['overflow_flag']) > 0)
    return {
        'n_scen': n,
        'overflow_rate': round(overflow / n, 4) if n else float('nan'),
        'mean_max_level_m': round(sum(float(r['max_level_m']) for r in rows) / n, 4),
        'mean_n_switches': round(sum(float(r['n_switches']) for r in rows) / n, 3),
        'mean_pump_use': round(sum(float(r['n_intervals_100']) + float(r['n_intervals_170']) for r in rows) / n, 3),
        'mean_n_dryrun_proxy': round(sum(float(r['n_dryrun_proxy']) for r in rows) / n, 4),
    }


def run_task(task):
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.chdir(SRC)
    import data_paths
    import perform_evaluate as pe

    algo, weights, seed, combo_id = task['algo'], task['weights'], task['seed'], task['combo_id']
    random.seed(seed)
    np.random.seed(seed)

    _, _, test_all = data_paths.load_fixed_split()
    test_inps = test_all[:N_TEST]

    stage = f'E06_grid_{combo_id}_{algo}_s{seed}'
    params = {'minutes': 2, 'weights': weights, 'act_type': 0}
    t0 = time.perf_counter()
    if algo == 'GA':
        pe.evaluate_genetic_algo(test_inps, YEARS, DUR, seed=seed,
                                 model_label=f'GA-{combo_id}', result_file=stage, **params)
    else:
        pe.evaluate_pso(test_inps, YEARS, DUR, seed=seed,
                        model_label=f'PSO-{combo_id}', result_file=stage, **params)
    scen = os.path.join(RESULTS, f'{stage}_scenario_metrics.csv')
    agg = os.path.join(RESULTS, f'{stage}.csv')
    with open(scen) as f:
        rows = list(csv.DictReader(f))
    res = {'combo_id': combo_id, 'w1': weights[0], 'w2': weights[1], 'w3': weights[2],
          'w4': weights[3], 'algo': algo, 'seed': seed, 'elapsed_s': round(time.perf_counter() - t0, 1)}
    res.update(summarize(rows))
    os.makedirs(RAW_DIR, exist_ok=True)
    for p in (scen, agg):
        if os.path.exists(p):
            shutil.move(p, os.path.join(RAW_DIR, os.path.basename(p)))
    return res


def main():
    os.chdir(SRC)
    combos = sample_combos()
    tasks = []
    for i, w in enumerate(combos):
        combo_id = f'c{i:02d}'
        for algo in ('GA', 'PSO'):
            for s in SEEDS:
                tasks.append({'combo_id': combo_id, 'weights': w, 'algo': algo, 'seed': s})

    fields = ['combo_id', 'w1', 'w2', 'w3', 'w4', 'algo', 'seed', 'n_scen',
              'overflow_rate', 'mean_max_level_m', 'mean_n_switches',
              'mean_pump_use', 'mean_n_dryrun_proxy', 'elapsed_s']
    os.makedirs(os.path.dirname(OUT_GRID), exist_ok=True)
    with open(OUT_GRID, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f'{len(combos)} weight combos x 2 algos x {len(SEEDS)} seeds = {len(tasks)} tasks, '
          f'N_TEST={N_TEST} scenarios each', flush=True)
    t0 = time.perf_counter()
    n = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            n += 1
            with open(OUT_GRID, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fields).writerow(res)
            print(f'{n}/{len(tasks)} {res["combo_id"]} w=({res["w1"]},{res["w2"]},{res["w3"]},{res["w4"]}) '
                  f'{res["algo"]} s{res["seed"]} ovf={res["overflow_rate"]} '
                  f'maxlvl={res["mean_max_level_m"]} ({res["elapsed_s"]}s)', flush=True)

    print(f'\nall done, {time.perf_counter()-t0:.0f}s total', flush=True)


if __name__ == '__main__':
    main()
