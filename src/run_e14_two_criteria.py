#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Section 3.1 regeneration: 2-criterion Regular DDQN (w=[0.5,0.5,0,0]),
using the CURRENT (fixed) code path -- Algorithm-3-line-27 action_list
update, gamma=0.7, min-max state normalization, corrected R_level
current-state term (all already in dqn_from_demon_v1.train() /
water_gym.py, no new code needed for this driver).

Only the 2-criterion Regular model needs training. The 4-criterion
GA-guided column reuses E3 v2's existing 5 seeds unchanged
(results/E03_seeds/GA-guided/) -- not retrained here.

Weights w=[0.5, 0.5, 0, 0] confirmed against the manuscript's Table 5
Regular DDQN row (docs/response/SECTION_3_1_REGENERATION.md Sec 0).
w3=w4=0 requires no special handling: water_gym.py's reward() multiplies
each term by its weight with no division or other w-dependent branching
(w1*vol_reward + w2*act_reward + w3*energy_reward + w4*over_pump*...) --
confirmed by reading water_gym.py:241-244 and the surrounding reward()
body. Zero weights simply zero out those two terms.

Same training code path as E3 v2's Regular (dqn_from_demon_v1.train() via
perform_evaluate.evaluate_dpn_gru(train=True, ...)) -- lr/batch_size/
target_update_C/gru_layers/gru_hidden/seq_len are hardcoded inside
train() itself (not passed as params), so reusing the same function call
reproduces them automatically; only `weights` differs from E3 v2 Regular.
Same data split (data_paths.load_fixed_split(), split_seed=42), same 5
seeds (1-5), same epochs=1/minutes=2/act_type=0/gamma=0.7.

Reuses run_e03_seeds.py's config/helpers (GAMMA, YEARS, DURATIONS,
TRAINED_MODELS, atomic-write/DONE-marker helpers) the same way
run_e05_ablation.py / run_e12_datasize.py do -- run_e03_seeds.py's own
WEIGHTS/BASE_PARAMS constants are NOT reused (they are E3 v2's 4-criterion
values) -- this driver defines its own TWO_CRITERION_WEIGHTS.

Output per seed:
    results/E14_two_criteria/Regular/seed{n}/test_metrics.csv
    results/E14_two_criteria/Regular/seed{n}/train_log.csv
    results/E14_two_criteria/Regular/seed{n}/run_meta.json
    results/E14_two_criteria/Regular/seed{n}/DONE

Run from src/. NOT run in this session (training, hours-scale) -- the
user runs it. Estimated cost: E3 v2 Regular's own run_meta.json records
elapsed_train_s ~= 10,230s (~2.84h) per seed for the same code path, same
data size, same network -- this driver is expected to take a similar
~2.8h/seed (2-criterion is not inherently faster; same architecture,
same number of episodes/steps). 5 seeds sequential ~= 14h; with
--workers 2 (CPU/SWMM-bound per run_e03_seeds.py's own note) roughly
half that wall-clock.

Resumable: re-running skips seeds with a valid DONE marker.

Usage:
    nohup python run_e14_two_criteria.py --workers 2 > ../results/_log/E14.out 2>&1 &
    python run_e14_two_criteria.py --max-test 3  # smoke test only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
RESULTS = ROOT / 'results'
E14_DIR = RESULTS / 'E14_two_criteria'
LOG_PATH = RESULTS / '_log' / 'E14.log'

import run_e03_seeds as e3  # noqa: E402 -- reuse config + atomic-write helpers

GAMMA = e3.GAMMA               # 0.7, unchanged from E3 v2
YEARS = e3.YEARS
DURATIONS = e3.DURATIONS
TRAINED_MODELS = e3.TRAINED_MODELS
FULL_TEST_N = e3.FULL_TEST_N
SEEDS_DEFAULT = [1, 2, 3, 4, 5]

# The manuscript's Table 5 "Regular DDQN" row: w1=w2=0.50, w3=w4=0.
TWO_CRITERION_WEIGHTS = [0.50, 0.50, 0.0, 0.0]
BASE_PARAMS = {'epochs': 1, 'minutes': 2, 'weights': TWO_CRITERION_WEIGHTS,
               'act_type': 0, 'gamma': GAMMA}


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def _seed_dir(seed: int) -> Path:
    return E14_DIR / 'Regular' / f'seed{seed}'


def _is_done(seed_dir: Path, expected_test_n) -> bool:
    done_path = seed_dir / 'DONE'
    metrics_path = seed_dir / 'test_metrics.csv'
    if not done_path.exists() or not metrics_path.exists():
        return False
    try:
        done_info = json.loads(done_path.read_text())
        actual_rows = e3._count_csv_rows(metrics_path)
    except (OSError, json.JSONDecodeError):
        return False
    if actual_rows != done_info.get('row_count'):
        return False
    if expected_test_n is not None and actual_rows != expected_test_n:
        return False
    return True


def _run_one(task: dict):
    import os
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    import random

    import numpy as np
    import torch
    torch.set_num_threads(2)

    seed = task['seed']
    max_train = task.get('max_train')
    max_test = task.get('max_test')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import data_paths
    import perform_evaluate as pe

    train_inps, _valid_inps, test_inps = data_paths.load_fixed_split()
    split_meta = data_paths.get_split_meta()
    if max_train is not None:
        train_inps = train_inps[:max_train]
    if max_test is not None:
        test_inps = test_inps[:max_test]

    seed_dir = _seed_dir(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)

    stage = f'E14_Regular2c_seed{seed}'
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    model_path = str(TRAINED_MODELS / f'{stage}.pkl')
    losses, rewards, _test_rewards, elapsed_train = pe.evaluate_dpn_gru(
        train_inps, test_inps, YEARS, DURATIONS, train=True,
        model_path=model_path, seed=seed, model_label='Regular-2criteria',
        **BASE_PARAMS,
    )
    hyperparams = {'lr': 1e-3, 'gamma': GAMMA, 'batch_size': 20,
                   'target_update_C': 10, 'gru_layers': 3, 'gru_hidden': 32,
                   'seq_len': 5}

    finished_at = datetime.now(timezone.utc).isoformat()
    e3._relocate_stage_outputs(stage, seed_dir)
    if losses is not None:
        e3._write_train_log(seed_dir / 'train_log.csv', losses, rewards)

    metrics_path = seed_dir / 'test_metrics.csv'
    if not metrics_path.exists():
        raise RuntimeError(f'{stage}: test_metrics.csv missing after relocation')
    row_count = e3._count_csv_rows(metrics_path)

    meta = {
        'exp_id': 'E14', 'model': 'Regular-2criteria', 'seed': seed,
        'git_commit': e3._git_commit(),
        'started_at': started_at, 'finished_at': finished_at,
        'elapsed_train_s': elapsed_train,
        'hyperparams': hyperparams,
        'reward_weights': {'w1': TWO_CRITERION_WEIGHTS[0], 'w2': TWO_CRITERION_WEIGHTS[1],
                           'w3': TWO_CRITERION_WEIGHTS[2], 'w4': TWO_CRITERION_WEIGHTS[3]},
        'heuristic_params': {},
        'data': {'train_n': len(train_inps), 'test_n': len(test_inps),
                  'split_seed': split_meta.get('split_seed'), 'split_file': split_meta.get('split_file')},
        'versions': {'python': sys.version, 'torch': torch.__version__},
    }
    e3._write_run_meta(seed_dir / 'run_meta.json', meta)
    e3._write_done_marker(seed_dir / 'DONE', row_count)
    return seed, row_count, time.perf_counter() - t0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seeds', type=str, default='1,2,3,4,5')
    p.add_argument('--max-train', type=int, default=None, help='smoke-testing only')
    p.add_argument('--max-test', type=int, default=None, help='smoke-testing only')
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    tasks = []
    for seed in seeds:
        seed_dir = _seed_dir(seed)
        expected_test_n = args.max_test if args.max_test is not None else FULL_TEST_N
        if _is_done(seed_dir, expected_test_n):
            log(f'SKIP seed{seed}: already DONE with {expected_test_n} rows')
            continue
        tasks.append({'seed': seed, 'max_train': args.max_train, 'max_test': args.max_test})

    if not tasks:
        log('nothing to do -- all seeds already DONE')
        return

    log(f'{len(tasks)} pending seed(s), weights={TWO_CRITERION_WEIGHTS}, workers={args.workers}')
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            seed, row_count, elapsed = fut.result()
            log(f'DONE seed{seed}: {row_count} rows in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
