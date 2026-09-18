#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E15: GA-guided DDQN (4-criterion, new weights) retrained under the
SAME parallel condition as E14 Regular (--workers 2, default), so that
Table 5's Training Time row for GA-guided becomes directly comparable
to E14 Regular's -- see results/summary/TABLE5_COMPARABILITY.md for why
the existing E3 v2 GA-guided number (measured at --workers 8, inside a
25-combo batch) is not comparable to E14 Regular's (--workers 2).

Every setting besides parallelism is identical to E3 v2's GA-guided run
(reuses run_e03_seeds.py's own WEIGHTS/GAMMA/YEARS/DURATIONS/
GUIDED_EXTRA/BASE_PARAMS and _run_one()'s GA-guided branch logic,
duplicated here rather than imported+called because run_e03_seeds.py's
_run_one() is model-dispatch-shaped and would need editing to isolate
one model -- this repo's no-refactor rule -- so this driver re-expresses
just the GA-guided branch on its own):
  - weights = [0.40, 0.25, 0.25, 0.10] (e3.WEIGHTS)
  - gamma = 0.7, epochs=1, minutes=2, act_type=0 (e3.BASE_PARAMS)
  - ga_start/ga_end/eps_start/eps_end/batch_size/learning_rate/sync_freq
    (e3.GUIDED_EXTRA)
  - same 5 seeds (1-5), same fixed data split (split_seed=42)
  - same per-worker thread caps (OMP/MKL/OPENBLAS=2, torch.set_num_threads(2))

Secondary purpose (reproducibility check, R4-9): since seeds and every
non-parallelism setting match E3 v2 GA-guided exactly, the resulting
test_metrics.csv should reproduce E3 v2's GA-guided test_metrics.csv
seed-for-seed if the pipeline is deterministic under a fixed seed.
results/summary/E15_reproducibility.md (via a separate, not-yet-run
comparison script) checks this after E15 completes.

Output per seed (results/E03_seeds/GA-guided/ is NOT touched or
overwritten):
    results/E15_timing/GA-guided/seed{n}/test_metrics.csv
    results/E15_timing/GA-guided/seed{n}/train_log.csv
    results/E15_timing/GA-guided/seed{n}/run_meta.json
    results/E15_timing/GA-guided/seed{n}/DONE

Resumable: re-running skips seeds with a valid DONE marker (row count
re-verified against an actual recount, same pattern as run_e03_seeds.py
/ run_e14_two_criteria.py).

NOT run in this session (training, hours-scale) -- the user runs it.
See module-level docstring end for the time estimate.

Usage:
    nohup python run_e15_timing.py --workers 2 > ../results/_log/E15.out 2>&1 &
    python run_e15_timing.py --max-test 3  # smoke test only
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
E15_DIR = RESULTS / 'E15_timing'
LOG_PATH = RESULTS / '_log' / 'E15.log'

import run_e03_seeds as e3  # noqa: E402 -- reuse config + atomic-write helpers

GAMMA = e3.GAMMA                 # 0.7, unchanged from E3 v2
YEARS = e3.YEARS
DURATIONS = e3.DURATIONS
TRAINED_MODELS = e3.TRAINED_MODELS
FULL_TEST_N = e3.FULL_TEST_N
WEIGHTS = e3.WEIGHTS             # [0.40, 0.25, 0.25, 0.10], unchanged from E3 v2
BASE_PARAMS = dict(e3.BASE_PARAMS)
GUIDED_EXTRA = dict(e3.GUIDED_EXTRA)
SEEDS_DEFAULT = [1, 2, 3, 4, 5]


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def _seed_dir(seed: int) -> Path:
    return E15_DIR / 'GA-guided' / f'seed{seed}'


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

    stage = f'E15_GA-guided_seed{seed}'
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

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
        'batch_size': params['batch_size'],
        'target_update_C': params['sync_freq'],
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
        'exp_id': 'E15', 'model': 'GA-guided', 'seed': seed,
        'git_commit': e3._git_commit(),
        'started_at': started_at, 'finished_at': finished_at,
        'elapsed_train_s': elapsed_train,
        'hyperparams': hyperparams,
        'reward_weights': {'w1': WEIGHTS[0], 'w2': WEIGHTS[1],
                            'w3': WEIGHTS[2], 'w4': WEIGHTS[3]},
        'heuristic_params': heuristic_params,
        'data': {'train_n': len(train_inps), 'test_n': len(test_inps),
                  'split_seed': split_meta.get('split_seed'), 'split_file': split_meta.get('split_file')},
        'versions': {'python': sys.version, 'torch': torch.__version__},
        'note': 'E15: same settings as E3 v2 GA-guided, run at --workers 2 '
                '(vs E3 v2\'s --workers 8) for Table 5 comparability with E14 Regular. '
                'results/E03_seeds/GA-guided/ was not touched.',
    }
    e3._write_run_meta(seed_dir / 'run_meta.json', meta)
    e3._write_done_marker(seed_dir / 'DONE', row_count)
    return seed, row_count, time.perf_counter() - t0, elapsed_train


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=2,
                   help='default 2, matching E14 Regular\'s condition (see module docstring)')
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

    log(f'{len(tasks)} pending seed(s), weights={WEIGHTS}, workers={args.workers}')
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            seed, row_count, elapsed, elapsed_train = fut.result()
            log(f'DONE seed{seed}: {row_count} rows in {elapsed:.1f}s '
                f'(elapsed_train_s={elapsed_train:.1f})')


if __name__ == '__main__':
    main()

# --- Time estimate (results/summary/TABLE5_COMPARABILITY.md Sec 7-(b-1)) ---
# E3 v2 GA-guided elapsed_train_s at --workers 8 (5-seed mean): 17,678.6s.
# The one directly-measured workers=8->2 scaling factor available (same
# model, Regular, E3 v2 vs E14): 3.46x wall-clock reduction. Applying that
# factor to GA-guided AS AN EXTRAPOLATION (not independently measured for
# GA-guided) gives an estimated ~17,678.6/3.46 = ~5,109s (~1.42h) per seed.
# With --workers 2 and 5 seeds, expect a similar task-assignment pattern to
# E14 Regular (2 seeds running concurrently at a time, the 5th finishing
# alone with less contention) -- E14 Regular's own wall-clock ratio (total
# job wall-clock vs a naive 5x-per-seed serial estimate) was about 2.5x, not
# 5x, because of the 2-worker overlap. Applying that same ~2.5x factor:
# estimated total wall-clock for all 5 GA-guided seeds at workers=2 =~
# 5,109s x 2.5 =~ 12,770s =~ 3.5h. This is an extrapolation built on two
# chained assumptions (workers-scaling factor transfers from Regular to
# GA-guided; wall-clock/per-seed ratio transfers from E14's own run) and is
# not a guarantee.
