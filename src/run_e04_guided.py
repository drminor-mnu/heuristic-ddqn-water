#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4 driver: ENUM-guided / lookahead-guided DDQN-GRU.

docs/REVISION_EXPERIMENT_PLAN.md ``## E4``.  Trains the guided DDQN with the
exhaustive-enumeration guide (L=1, E4-a) -- and later the L-step lookahead
guides (L in {3,5}, E4-b) -- reusing the E3 driver's config and its
resume / atomic-write / staging machinery so results stay comparable.

Config (weights, gamma, guidance schedule, YEARS/DURATIONS) is imported from
run_e03_seeds so it is byte-for-byte the E3 v2 setup; only the guide optimizer
changes.

Output per (L, seed):
    results/E04_enum/ddqn_by_L/L{L}/seed{n}/test_metrics.csv
    results/E04_enum/ddqn_by_L/L{L}/seed{n}/train_log.csv
    results/E04_enum/ddqn_by_L/L{L}/seed{n}/run_meta.json
    results/E04_enum/ddqn_by_L/L{L}/seed{n}/DONE

Resume: a (L, seed) is skipped only when DONE exists AND its recorded row count
matches a live recount of test_metrics.csv AND that equals the expected test-set
size (a truncated CSV from a killed run does not pass) -- same rule as E3.

Run from src/.  Long runs: use tmux/nohup (manuscript ~3.4 h per guided train).

Usage:
    python run_e04_guided.py --L 1 --seeds 1,2,3 --workers 3
    python run_e04_guided.py --L 1 --seeds 1 --max-train 4 --max-test 3   # smoke
"""
from __future__ import annotations

import argparse
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
E04_DDQN_DIR = RESULTS / 'E04_enum' / 'ddqn_by_L'
LOG_PATH = RESULTS / '_log' / 'E04.log'

import run_e03_seeds as e3  # noqa: E402  -- reuse E3 config + helpers

FULL_TEST_N = e3.FULL_TEST_N
WEIGHTS = e3.WEIGHTS
GAMMA = e3.GAMMA
YEARS = e3.YEARS
DURATIONS = e3.DURATIONS
BASE_PARAMS = e3.BASE_PARAMS
GUIDED_EXTRA = e3.GUIDED_EXTRA
TRAINED_MODELS = e3.TRAINED_MODELS

# L -> (perform_evaluate trainer entry point, guide label). L>1 lands here in
# E4-b once LookaheadGA/EnumL exist.
GUIDE_BY_L = {
    1: ('evaluate_dpn_guided_ENUM', 'ENUM-guided'),
}


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_dir(L: int, seed: int) -> Path:
    return E04_DDQN_DIR / f'L{L}' / f'seed{seed}'


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
            for p in base.glob('.E04g_*.tmp'):
                try:
                    p.unlink()
                except OSError:
                    pass
    if E04_DDQN_DIR.exists():
        for p in E04_DDQN_DIR.rglob('*.tmp'):
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

    L = task['L']
    seed = task['seed']
    max_train = task.get('max_train')
    max_test = task.get('max_test')
    trainer_name, guide_label = GUIDE_BY_L[L]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import data_paths
    import perform_evaluate as pe

    train_inps, _valid, test_inps = data_paths.load_fixed_split()
    split_meta = data_paths.get_split_meta()
    if max_train is not None:
        train_inps = train_inps[:max_train]
    if max_test is not None:
        test_inps = test_inps[:max_test]

    seed_dir = _seed_dir(L, seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    stage = f'E04g_L{L}_seed{seed}'
    model_path = str(TRAINED_MODELS / f'{stage}.pkl')

    params = dict(BASE_PARAMS)
    params.update(GUIDED_EXTRA)

    started_at = _now_iso()
    t0 = time.perf_counter()
    trainer = getattr(pe, trainer_name)
    losses, rewards, _test_rewards, elapsed_train = trainer(
        train_inps, test_inps, YEARS, DURATIONS, train=True,
        model_path=model_path, seed=seed, model_label=guide_label,
        **params,
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
        'exp_id': 'E04', 'sub': 'E4-a' if L == 1 else 'E4-b', 'L': L,
        'model': guide_label, 'seed': seed,
        'git_commit': e3._git_commit(),
        'started_at': started_at, 'finished_at': finished_at,
        'elapsed_train_s': elapsed_train,
        'hyperparams': {
            'lr': params['learning_rate'], 'gamma': params['gamma'],
            'batch_size': params['batch_size'],
            'target_update_C': params['sync_freq'],
            'min_replay_size': params['batch_size'] * 5,
            'ga_start': params['ga_start'], 'ga_end': params['ga_end'],
            'eps_start': params['eps_start'], 'eps_end': params['eps_end'],
            'gru_layers': 3, 'gru_hidden': 32, 'seq_len': 5,
        },
        'reward_weights': {'w1': WEIGHTS[0], 'w2': WEIGHTS[1],
                           'w3': WEIGHTS[2], 'w4': WEIGHTS[3]},
        'heuristic_params': {'type': 'ENUM' if L == 1 else f'lookahead-L{L}',
                             'candidate_space': 6 ** L},
        'data': {'train_n': len(train_inps), 'test_n': len(test_inps),
                 'split_seed': split_meta.get('split_seed'),
                 'split_file': split_meta.get('split_file')},
        'versions': {'python': sys.version, 'torch': torch.__version__},
    }
    e3._write_run_meta(seed_dir / 'run_meta.json', meta)
    e3._write_done_marker(seed_dir / 'DONE', row_count)
    return L, seed, row_count, time.perf_counter() - t0


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--L', type=int, default=1, choices=sorted(GUIDE_BY_L),
                   help='lookahead horizon (1 = ENUM-guided, E4-a)')
    p.add_argument('--seeds', type=str, default='1,2,3')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--max-train', type=int, default=None)
    p.add_argument('--max-test', type=int, default=None)
    return p.parse_args()


def main():
    os.chdir(str(SRC))
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    expected_test_n = args.max_test if args.max_test is not None else FULL_TEST_N

    _cleanup_stale_tmp()

    tasks = [{'L': args.L, 'seed': s, 'max_train': args.max_train,
              'max_test': args.max_test} for s in seeds]
    pending = []
    for t in tasks:
        if _is_done(_seed_dir(t['L'], t['seed']), expected_test_n):
            log(f"skip (already done): L{t['L']} seed{t['seed']}")
        else:
            pending.append(t)

    log(f"L={args.L} guide={GUIDE_BY_L[args.L][1]}: {len(tasks)} combos, "
        f"{len(tasks) - len(pending)} done, {len(pending)} to run, "
        f"workers={args.workers}")
    if not pending:
        log('nothing to do')
        return

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in pending}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                L, seed, rc, el = fut.result()
                log(f'DONE L{L} seed{seed}: {rc} rows in {el:.1f}s')
            except Exception as exc:  # noqa: BLE001
                log(f"FAILED L{t['L']} seed{t['seed']}: {exc!r}")
    log('run complete')


if __name__ == '__main__':
    main()
