#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E12 minimal execution (R2-1, R2-4, R4-4): repeat ONLY the Abstract's
specific "160 training scenarios -> mean reward 5.401 higher (GA-guided vs
Regular DDQN) over 2,700 paired test scenarios" claim, across seeds -- not
the plan's full grid ({80,160,400,1000,3996} x 3 models x 3 seeds = 45
runs). See docs/REMAINING_WORK.md Sec 3 for why the full grid was not
scoped for this session.

Minimal condition: train_n=160, models={Regular, GA-guided}, seeds={1,2,3}
= 6 new runs. Test set is the FULL fixed-split 2,700 scenarios (unchanged,
matches "2,700 paired test scenarios" in the Abstract) -- only the
TRAINING set is truncated. The 160 training scenarios are the first 160 of
data_paths.load_fixed_split()'s train_inps (deterministic, fixed order) --
the Abstract does not specify how its 160 were chosen either, so this is a
documented assumption, not a reproduction of the original selection.

Reuses run_e03_seeds.py's config/helpers (WEIGHTS, GAMMA, BASE_PARAMS,
GUIDED_EXTRA-equivalent params, atomic-write/DONE-marker helpers) the same
way run_e05_ablation.py does -- run_e03_seeds.py itself is untouched.

Output per (model, seed):
    results/E12_datasize/{model}/n160/seed{n}/test_metrics.csv
    results/E12_datasize/{model}/n160/seed{n}/train_log.csv
    results/E12_datasize/{model}/n160/seed{n}/run_meta.json
    results/E12_datasize/{model}/n160/seed{n}/DONE

Run from src/. NOT run in this session (training, ~hours) -- see the
launch command in docs/REMAINING_WORK.md / this file's __main__ docstring.
Resumable: re-running skips (model, seed) combos with a valid DONE marker.

Usage:
    nohup python run_e12_datasize.py --workers 2 > ../results/_log/E12.out 2>&1 &
    python run_e12_datasize.py --max-test 3 --seeds 1  # smoke test only
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
E12_DIR = RESULTS / 'E12_datasize'
LOG_PATH = RESULTS / '_log' / 'E12.log'
TRAIN_N = 160

import run_e03_seeds as e3  # noqa: E402 -- reuse config + atomic-write helpers

WEIGHTS = e3.WEIGHTS
GAMMA = e3.GAMMA
YEARS = e3.YEARS
DURATIONS = e3.DURATIONS
BASE_PARAMS = e3.BASE_PARAMS
TRAINED_MODELS = e3.TRAINED_MODELS
GUIDED_EXTRA = e3.GUIDED_EXTRA
MODELS = ['Regular', 'GA-guided']


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def _seed_dir(model: str, seed: int) -> Path:
    return E12_DIR / model / f'n{TRAIN_N}' / f'seed{seed}'


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

    model, seed = task['model'], task['seed']
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
    train_inps = train_inps[:TRAIN_N]  # the "160 training scenarios" -- see module docstring
    if max_test is not None:
        test_inps = test_inps[:max_test]

    seed_dir = _seed_dir(model, seed)
    seed_dir.mkdir(parents=True, exist_ok=True)

    stage = f'E12_{model}_n{TRAIN_N}_seed{seed}'
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    if model == 'Regular':
        model_path = str(TRAINED_MODELS / f'{stage}.pkl')
        losses, rewards, _test_rewards, elapsed_train = pe.evaluate_dpn_gru(
            train_inps, test_inps, YEARS, DURATIONS, train=True,
            model_path=model_path, seed=seed, model_label='Regular',
            **BASE_PARAMS,
        )
        hyperparams = {'lr': 1e-3, 'gamma': GAMMA, 'batch_size': 20,
                       'target_update_C': 10, 'gru_layers': 3, 'gru_hidden': 32, 'seq_len': 5}
        heuristic_params = {}
    else:  # GA-guided
        model_path = str(TRAINED_MODELS / f'{stage}.pkl')
        params = dict(BASE_PARAMS)
        params.update(GUIDED_EXTRA)
        losses, rewards, _test_rewards, elapsed_train = pe.evaluate_dpn_guided_GA(
            train_inps, test_inps, YEARS, DURATIONS, train=True,
            model_path=model_path, seed=seed, model_label='GA-guided',
            **params,
        )
        hyperparams = {
            'lr': params['learning_rate'], 'gamma': params['gamma'],
            'batch_size': params['batch_size'], 'target_update_C': params['sync_freq'],
            'min_replay_size': params['batch_size'] * 5,
            'ga_start': params['ga_start'], 'ga_end': params['ga_end'],
            'eps_start': params['eps_start'], 'eps_end': params['eps_end'],
            'gru_layers': 3, 'gru_hidden': 32, 'seq_len': 5,
        }
        heuristic_params = {'type': 'GA', 'population_size': 100, 'num_generations': 100,
                            'crossover_rate': 0.8, 'mutation_rate': 0.01}

    finished_at = datetime.now(timezone.utc).isoformat()
    e3._relocate_stage_outputs(stage, seed_dir)
    if losses is not None:
        e3._write_train_log(seed_dir / 'train_log.csv', losses, rewards)

    metrics_path = seed_dir / 'test_metrics.csv'
    if not metrics_path.exists():
        raise RuntimeError(f'{stage}: test_metrics.csv missing after relocation')
    row_count = e3._count_csv_rows(metrics_path)

    meta = {
        'exp_id': 'E12', 'model': model, 'seed': seed, 'train_n_condition': TRAIN_N,
        'git_commit': e3._git_commit(),
        'started_at': started_at, 'finished_at': finished_at,
        'elapsed_train_s': elapsed_train,
        'hyperparams': hyperparams,
        'reward_weights': {'w1': WEIGHTS[0], 'w2': WEIGHTS[1], 'w3': WEIGHTS[2], 'w4': WEIGHTS[3]},
        'heuristic_params': heuristic_params,
        'data': {'train_n': len(train_inps), 'test_n': len(test_inps),
                  'split_seed': split_meta.get('split_seed'), 'split_file': split_meta.get('split_file')},
        'versions': {'python': sys.version, 'torch': torch.__version__},
    }
    e3._write_run_meta(seed_dir / 'run_meta.json', meta)
    e3._write_done_marker(seed_dir / 'DONE', row_count)
    return model, seed, row_count, time.perf_counter() - t0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seeds', type=str, default='1,2,3')
    p.add_argument('--max-test', type=int, default=None, help='smoke-testing only')
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    tasks = []
    for model in MODELS:
        for seed in seeds:
            seed_dir = _seed_dir(model, seed)
            expected_test_n = args.max_test if args.max_test is not None else e3.FULL_TEST_N
            if _is_done(seed_dir, expected_test_n):
                log(f'SKIP {model} seed{seed}: already DONE with {expected_test_n} rows')
                continue
            tasks.append({'model': model, 'seed': seed, 'max_test': args.max_test})

    if not tasks:
        log('nothing to do -- all combinations already DONE')
        return

    log(f'{len(tasks)} pending task(s), train_n={TRAIN_N}, workers={args.workers}')
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            model, seed, row_count, elapsed = fut.result()
            log(f'DONE {model} seed{seed}: {row_count} rows in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
