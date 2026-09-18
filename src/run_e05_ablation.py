#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5 driver: single-variable online-guidance ablation (A0-A4).

docs/REVISION_EXPERIMENT_PLAN.md E5 (R3-3, R4-5; weights corrected 2026-09-01
to the new weights, see the plan doc). New weights [0.40, 0.25, 0.25, 0.10],
gamma=0.7, split_seed=42 -- identical to E3 v2 -- for every condition; only
the guide-related settings (ga_start, ga_end, optimizer_class) vary.

Conditions (see REVISION_EXPERIMENT_PLAN.md E5 table):
    A0  ga_start=0.0, ga_end=0.0            (no guidance at all)
    A1  ga_start=0.6, ga_end=0.0, RandomGuide (uniform-random action instead
        of GA, same probability schedule as A3 -- isolates "more exploration"
        from "intelligent selection")
    A2  ga_start=0.3, ga_end=0.3            (constant probability, no decay)
    A3  ga_start=0.6, ga_end=0.0, GeneticAlgorithm  (== E3 v2 GA-guided)
    A4_03 / A4_09  ga_start=0.3 / 0.9, ga_end=0.0, GeneticAlgorithm

A0 is NOT reused from E3 v2's "Regular" model: E3 v2's Regular was trained by
dqn_from_demon_v1.train() (epsilon 1.0->0.2 schedule, sync every 10 *steps*,
replay warm-up at len>batch_size) which is a structurally different code path
from _train_guided_by_optimizer (eps_start/eps_end schedule, sync every
sync_freq *updates*, replay warm-up at len>=min_replay_size) that A1-A4 all
go through -- reusing Regular for A0 would confound the guidance variable
with these other differences. A0 is trained fresh via
_train_guided_by_optimizer with ga_start=ga_end=0.0 (the guide branch is then
never taken, but every other setting matches A1-A4 exactly).
A3 IS reused from E3 v2 when available (identical code path and params,
verified against E3 v2's run_meta.json).

This driver calls dqn_from_demon_v1._train_guided_by_optimizer and
perform_evaluate._evaluate_dpn_guided directly with per-condition
optimizer_class/ga_start/ga_end (both already accept these as parameters --
no wrapper functions added, no existing function modified). RandomGuide
(A1) lives in the new, additive random_guide.py.

Reuses run_e03_seeds.py's resume/atomic-write/staging helpers (imported, not
duplicated) the same way run_e04_guided.py does; run_e03_seeds.py itself is
untouched.

Output per (condition, seed):
    results/E05_ablation/{A-id}/seed{n}/test_metrics.csv
    results/E05_ablation/{A-id}/seed{n}/train_log.csv
    results/E05_ablation/{A-id}/seed{n}/run_meta.json
    results/E05_ablation/{A-id}/seed{n}/DONE

Run from src/. Long-running (GA-guided-scale conditions ~3-5h/seed on this
hardware per E3 v2's run_meta.json elapsed_train_s) -- run via tmux/nohup,
not inside an interactive session.

Usage:
    python run_e05_ablation.py --conditions A0,A1 --seeds 1,2,3 --workers 3
    python run_e05_ablation.py --conditions A0 --seeds 1 --max-train 4 --max-test 3   # smoke
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
RESULTS = ROOT / 'results'
E05_DIR = RESULTS / 'E05_ablation'
LOG_PATH = RESULTS / '_log' / 'E05.log'

import run_e03_seeds as e3  # noqa: E402  -- reuse config + atomic-write helpers

WEIGHTS = e3.WEIGHTS          # [0.40, 0.25, 0.25, 0.10] -- confirmed == E3 v2
GAMMA = e3.GAMMA               # 0.7
YEARS = e3.YEARS
DURATIONS = e3.DURATIONS
BASE_PARAMS = e3.BASE_PARAMS
TRAINED_MODELS = e3.TRAINED_MODELS
FULL_TEST_N = e3.FULL_TEST_N

# Shared guidance-schedule settings, identical to E3 v2's GUIDED_EXTRA for
# every condition except ga_start/ga_end (and optimizer_class for A1).
SHARED_GUIDED = {
    'eps_start': 0.3, 'eps_end': 0.05,
    'gamma': GAMMA, 'batch_size': 20,
    'learning_rate': 1e-3, 'sync_freq': 200,
}

# condition -> (optimizer_class_name, optimizer_method, guide_label, ga_start, ga_end)
# optimizer_class resolved lazily inside the worker (after chdir/imports).
CONDITIONS = {
    'A0':    ('GeneticAlgorithm', 'genetic_algorithm', 'GA',     0.0, 0.0),
    'A1':    ('RandomGuide',      'genetic_algorithm', 'Random', 0.6, 0.0),
    'A2':    ('GeneticAlgorithm', 'genetic_algorithm', 'GA',     0.3, 0.3),
    'A3':    ('GeneticAlgorithm', 'genetic_algorithm', 'GA',     0.6, 0.0),
    'A4_03': ('GeneticAlgorithm', 'genetic_algorithm', 'GA',     0.3, 0.0),
    'A4_09': ('GeneticAlgorithm', 'genetic_algorithm', 'GA',     0.9, 0.0),
}


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_dir(cond: str, seed: int) -> Path:
    return E05_DIR / cond / f'seed{seed}'


def _is_done(seed_dir: Path, expected_test_n: int | None) -> bool:
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


def _cleanup_stale_tmp() -> None:
    for base in (RESULTS, TRAINED_MODELS):
        if base.exists():
            for p in base.glob('.E05_*.tmp'):
                try:
                    p.unlink()
                except OSError:
                    pass
    if E05_DIR.exists():
        for p in E05_DIR.rglob('*.tmp'):
            try:
                p.unlink()
            except OSError:
                pass


def _run_one(task: dict):
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.chdir(str(SRC))

    import random

    import numpy as np
    import torch

    torch.set_num_threads(2)

    cond = task['cond']
    seed = task['seed']
    max_train = task.get('max_train')
    max_test = task.get('max_test')
    opt_name, opt_method, guide_label, ga_start, ga_end = CONDITIONS[cond]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import data_paths
    import perform_evaluate as pe
    import dqn_from_demon_v1 as ddqn
    from genetic_algo import GeneticAlgorithm
    from random_guide import RandomGuide

    optimizer_class = {'GeneticAlgorithm': GeneticAlgorithm,
                       'RandomGuide': RandomGuide}[opt_name]

    train_inps, _valid, test_inps = data_paths.load_fixed_split()
    split_meta = data_paths.get_split_meta()
    if max_train is not None:
        train_inps = train_inps[:max_train]
    if max_test is not None:
        test_inps = test_inps[:max_test]

    seed_dir = _seed_dir(cond, seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    stage = f'E05_{cond}_seed{seed}'
    model_path = str(TRAINED_MODELS / f'{stage}.pkl')

    params = dict(BASE_PARAMS)
    params.update(SHARED_GUIDED)
    params['ga_start'] = ga_start
    params['ga_end'] = ga_end

    trainer = functools.partial(
        ddqn._train_guided_by_optimizer,
        optimizer_class=optimizer_class,
        optimizer_method=opt_method,
        guide_label=guide_label,
    )

    started_at = _now_iso()
    t0 = time.perf_counter()
    losses, rewards, _test_rewards, elapsed_train = pe._evaluate_dpn_guided(
        train_inps, test_inps, YEARS, DURATIONS, trainer=trainer,
        guide_label=guide_label, train=True, model_path=model_path,
        seed=seed, model_label=cond, **params,
    )
    finished_at = _now_iso()

    e3._relocate_stage_outputs(stage, seed_dir)
    if losses is not None:
        e3._write_train_log(seed_dir / 'train_log.csv', losses, rewards)

    metrics_path = seed_dir / 'test_metrics.csv'
    if not metrics_path.exists():
        raise RuntimeError(f'{stage}: test_metrics.csv missing after relocation')
    row_count = e3._count_csv_rows(metrics_path)

    meta = {
        'exp_id': 'E05', 'condition': cond, 'seed': seed,
        'git_commit': e3._git_commit(),
        'started_at': started_at, 'finished_at': finished_at,
        'elapsed_train_s': elapsed_train,
        'hyperparams': {
            'lr': params['learning_rate'], 'gamma': params['gamma'],
            'batch_size': params['batch_size'],
            'target_update_C': params['sync_freq'],
            'min_replay_size': params['batch_size'] * 5,
            'ga_start': ga_start, 'ga_end': ga_end,
            'eps_start': params['eps_start'], 'eps_end': params['eps_end'],
            'gru_layers': 3, 'gru_hidden': 32, 'seq_len': 5,
        },
        'reward_weights': {'w1': WEIGHTS[0], 'w2': WEIGHTS[1],
                           'w3': WEIGHTS[2], 'w4': WEIGHTS[3]},
        'heuristic_params': {'type': opt_name, 'guide_label': guide_label},
        'data': {'train_n': len(train_inps), 'test_n': len(test_inps),
                 'split_seed': split_meta.get('split_seed'),
                 'split_file': split_meta.get('split_file')},
        'versions': {'python': sys.version, 'torch': torch.__version__},
    }
    e3._write_run_meta(seed_dir / 'run_meta.json', meta)
    e3._write_done_marker(seed_dir / 'DONE', row_count)
    return cond, seed, row_count, time.perf_counter() - t0


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--conditions', type=str, default='A0,A1,A2,A3,A4_03,A4_09')
    p.add_argument('--seeds', type=str, default='1,2,3')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--max-train', type=int, default=None)
    p.add_argument('--max-test', type=int, default=None)
    return p.parse_args()


def main():
    os.chdir(str(SRC))
    args = parse_args()
    conditions = args.conditions.split(',')
    for c in conditions:
        if c not in CONDITIONS:
            raise SystemExit(f'unknown condition {c!r}, expected one of {list(CONDITIONS)}')
    seeds = [int(s) for s in args.seeds.split(',')]
    expected_test_n = args.max_test if args.max_test is not None else FULL_TEST_N

    _cleanup_stale_tmp()

    tasks = [{'cond': c, 'seed': s, 'max_train': args.max_train,
              'max_test': args.max_test} for c in conditions for s in seeds]
    pending = []
    for t in tasks:
        seed_dir = _seed_dir(t['cond'], t['seed'])
        if _is_done(seed_dir, expected_test_n):
            log(f"skip (already done): {t['cond']} seed{t['seed']}")
        else:
            pending.append(t)
            log(f"output path: {seed_dir}")

    log(f'{len(tasks)} total combos, {len(tasks) - len(pending)} done, '
        f'{len(pending)} to run, workers={args.workers}')
    if not pending:
        log('nothing to do')
        return

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in pending}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                cond, seed, rc, el = fut.result()
                log(f'DONE {cond} seed{seed}: {rc} rows in {el:.1f}s')
            except Exception as exc:  # noqa: BLE001
                log(f"FAILED {t['cond']} seed{t['seed']}: {exc!r}")
    log('run complete')


if __name__ == '__main__':
    main()
