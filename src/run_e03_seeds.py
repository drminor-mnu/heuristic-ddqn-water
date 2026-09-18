#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E3 driver: 5 models x N seeds, resumable, atomic-write-safe.

Runs Regular / GA-guided / PSO-guided / GA-only / PSO-only DDQN, each
across --seeds (default 1..5), and writes per (model, seed):

    results/E03_seeds/{model}/seed{n}/test_metrics.csv   (scenario-level raw)
    results/E03_seeds/{model}/seed{n}/train_log.csv      (DQN models only)
    results/E03_seeds/{model}/seed{n}/run_meta.json
    results/E03_seeds/{model}/seed{n}/DONE

Design (docs/REVISION_EXPERIMENT_PLAN.md 2.4.1):
  - Resume unit is one (model, seed) combination. A combo is skipped only
    when DONE exists AND its recorded row count matches an actual recount
    of test_metrics.csv AND that count matches the expected number of test
    scenarios -- a truncated CSV from an interrupted run does not pass.
  - All writes (test_metrics.csv via perform_evaluate.save_scenario_metrics,
    train_log.csv, run_meta.json, DONE) go through atomic_io.atomic_write /
    atomic_rename, so a crash mid-write never leaves a partial file at the
    final path.
  - perform_evaluate's save_results()/save_scenario_metrics() always derive
    their filename from a caller-supplied basename and write flat under
    results/ -- they cannot be told to write into a nested directory. Each
    worker therefore writes to a unique staging name (E03_{model}_seed{n})
    and this script relocates the outputs into the target seed directory
    with atomic_rename() once the run finishes.
  - Each worker seeds random/numpy/torch/torch.cuda and sets
    cudnn.deterministic=True/benchmark=False *before* calling into
    perform_evaluate. perform_evaluate's own `seed=` argument is only a
    CSV label (docs/CODEBASE_MAP.md) -- skipping this step would silently
    turn "5 seeds" into 5 unseeded runs.
  - torch.set_num_threads(2) plus OMP/MKL/OPENBLAS_NUM_THREADS=2 per worker
    process, since the real bottleneck is CPU (WaterGym.reset() runs a
    single-threaded SWMM simulation per scenario), not GPU.

Usage:
    python run_e03_seeds.py --workers 8
    python run_e03_seeds.py --seeds 1,2 --models Regular,GA-guided \
        --max-train 3 --max-test 3   # smoke-test the driver itself
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
E03_DIR = RESULTS / 'E03_seeds'
LOG_PATH = RESULTS / '_log' / 'E03.log'
TRAINED_MODELS = ROOT / 'trained_models'

SEEDS_DEFAULT = [1, 2, 3, 4, 5]
MODELS_DEFAULT = ['Regular', 'GA-guided', 'PSO-guided', 'GA-only', 'PSO-only']
FULL_TEST_N = 2700  # data_paths.load_fixed_split() test split size

# E6-adopted reward weights (was [0.25, 0.4, 0.25, 0.1]). The original had
# w2 (switching) > w1 (level), which fixes the 1-step objective onto action 0
# (see docs/E00 sections 7-8, results/summary/E06_*). w1 must exceed w2 by a
# margin (~+0.04+); +0.15 here keeps GA/PSO-only in the active-control regime.
WEIGHTS = [0.40, 0.25, 0.25, 0.10]
GAMMA = 0.7   # E6-adopted (was hardcoded 0.1 -> effective ~1-step horizon)
YEARS = ['10', '20', '30', '50', '80', '100']
DURATIONS = ['0060', '0120', '0180', '0240', '0360', '0540', '0720', '1080', '1440']

BASE_PARAMS = {'epochs': 1, 'minutes': 2, 'weights': WEIGHTS, 'act_type': 0,
               'gamma': GAMMA}
GUIDED_EXTRA = {
    'ga_start': 0.6, 'ga_end': 0.0,
    'eps_start': 0.3, 'eps_end': 0.05,
    'gamma': GAMMA, 'batch_size': 20,
    'learning_rate': 1e-3, 'sync_freq': 200,
}

# Rough per-run wall-clock estimates for the *full* (untruncated) dataset,
# printed before running so the operator knows what they're starting.
# DQN-training models: manuscript Table 5 (~12,124s for GA-guided). Heuristic-
# only models: extrapolated from an empirical 30-scenario probe spanning
# 60/360/1440 min durations during E0 (docs/E00_HEURISTIC_MYOPIA_FINDING.md),
# ~6.6s/scenario average x 2700 -- both are estimates, not guarantees.
ESTIMATED_SECONDS_FULL = {
    'Regular': 12000,
    'GA-guided': 12124,
    'PSO-guided': 12124,
    'GA-only': 17900,
    'PSO-only': 17900,
}


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# atomic_io-backed writers
# ---------------------------------------------------------------------------

def _count_csv_rows(path: Path) -> int:
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return 0
        return sum(1 for _ in reader)


def _write_train_log(path: Path, losses, rewards) -> None:
    from atomic_io import atomic_write

    def _write(f):
        writer = csv.writer(f)
        writer.writerow(['episode', 'train_loss', 'train_reward'])
        for i, (loss, reward) in enumerate(zip(losses, rewards)):
            writer.writerow([i, loss, reward])

    atomic_write(path, _write, newline='')


def _write_run_meta(path: Path, meta: dict) -> None:
    from atomic_io import atomic_write

    def _write(f):
        json.dump(meta, f, indent=2, default=str)

    atomic_write(path, _write)


def _write_done_marker(path: Path, row_count: int) -> None:
    from atomic_io import atomic_write

    payload = {
        'completed_at': _now_iso(),
        'row_count': row_count,
        'git_commit': _git_commit(),
    }

    def _write(f):
        json.dump(payload, f, indent=2)

    atomic_write(path, _write)


def _relocate_stage_outputs(stage: str, seed_dir: Path) -> None:
    from atomic_io import atomic_rename

    src_agg = RESULTS / f'{stage}.csv'
    src_scenario = RESULTS / f'{stage}_scenario_metrics.csv'
    src_model = TRAINED_MODELS / f'{stage}.pkl'

    if src_scenario.exists():
        atomic_rename(src_scenario, seed_dir / 'test_metrics.csv')
    if src_agg.exists():
        atomic_rename(src_agg, seed_dir / 'year_duration_aggregate.csv')
    if src_model.exists():
        atomic_rename(src_model, seed_dir / 'model.pkl')


# ---------------------------------------------------------------------------
# resume check
# ---------------------------------------------------------------------------

def is_done(model: str, seed: int, expected_test_n: int | None) -> bool:
    seed_dir = E03_DIR / model / f'seed{seed}'
    done_path = seed_dir / 'DONE'
    metrics_path = seed_dir / 'test_metrics.csv'
    if not done_path.exists() or not metrics_path.exists():
        return False
    try:
        done_info = json.loads(done_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    try:
        actual_rows = _count_csv_rows(metrics_path)
    except OSError:
        return False
    if actual_rows != done_info.get('row_count'):
        return False
    if expected_test_n is not None and actual_rows != expected_test_n:
        return False
    return True


def cleanup_stale_tmp() -> list[str]:
    """Remove leftover .tmp files from a killed prior run. Safe: a stale
    staging file for a not-yet-DONE combo gets overwritten on retry anyway;
    this just avoids clutter and stale .tmp files lingering forever."""
    removed = []
    if RESULTS.exists():
        for p in RESULTS.glob('.E03_*.tmp'):
            try:
                p.unlink()
                removed.append(str(p))
            except OSError:
                pass
        for p in TRAINED_MODELS.glob('.E03_*.tmp'):
            try:
                p.unlink()
                removed.append(str(p))
            except OSError:
                pass
    if E03_DIR.exists():
        for p in E03_DIR.rglob('*.tmp'):
            try:
                p.unlink()
                removed.append(str(p))
            except OSError:
                pass
    return removed


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

def _run_one(task: dict):
    """Executes entirely inside a worker process."""
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'

    import random

    import numpy as np
    import torch

    torch.set_num_threads(2)

    model = task['model']
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

    seed_dir = E03_DIR / model / f'seed{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)

    stage = f'E03_{model}_seed{seed}'
    started_at = _now_iso()
    t0 = time.perf_counter()

    losses = rewards = None
    elapsed_train = None
    hyperparams = {}
    heuristic_params = {}

    if model == 'Regular':
        model_path = str(TRAINED_MODELS / f'{stage}.pkl')
        losses, rewards, _test_rewards, elapsed_train = pe.evaluate_dpn_gru(
            train_inps, test_inps, YEARS, DURATIONS, train=True,
            model_path=model_path, seed=seed, model_label='Regular',
            **BASE_PARAMS,
        )
        # dqn_from_demon_v1.train() hardcodes lr/batch/sync; gamma now via params.
        hyperparams = {
            'lr': 1e-3, 'gamma': GAMMA, 'batch_size': 20, 'target_update_C': 10,
            'gru_layers': 3, 'gru_hidden': 32, 'seq_len': 5,
        }
    elif model in ('GA-guided', 'PSO-guided'):
        model_path = str(TRAINED_MODELS / f'{stage}.pkl')
        params = dict(BASE_PARAMS)
        params.update(GUIDED_EXTRA)
        trainer = pe.evaluate_dpn_guided_GA if model == 'GA-guided' else pe.evaluate_dpn_guided_PSO
        losses, rewards, _test_rewards, elapsed_train = trainer(
            train_inps, test_inps, YEARS, DURATIONS, train=True,
            model_path=model_path, seed=seed, model_label=model,
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
        heuristic_params = (
            {'type': 'GA', 'population_size': 100, 'num_generations': 100,
             'crossover_rate': 0.8, 'mutation_rate': 0.01}
            if model == 'GA-guided' else
            {'type': 'PSO', 'swarm_size': 30, 'num_iterations': 100,
             'inertia': 0.7, 'cognitive': 1.5, 'social': 1.5,
             'patience': 10, 'min_improvement': 1e-6}
        )
    elif model == 'GA-only':
        pe.evaluate_genetic_algo(
            test_inps, YEARS, DURATIONS, seed=seed, model_label='GA-only',
            result_file=stage, **BASE_PARAMS,
        )
        heuristic_params = {'type': 'GA', 'population_size': 100,
                             'num_generations': 100, 'crossover_rate': 0.8,
                             'mutation_rate': 0.01}
    elif model == 'PSO-only':
        pe.evaluate_pso(
            test_inps, YEARS, DURATIONS, seed=seed, model_label='PSO-only',
            result_file=stage, **BASE_PARAMS,
        )
        heuristic_params = {'type': 'PSO', 'swarm_size': 30,
                             'num_iterations': 100, 'inertia': 0.7,
                             'cognitive': 1.5, 'social': 1.5,
                             'patience': 10, 'min_improvement': 1e-6}
    else:
        raise ValueError(f'unknown model {model!r}')

    finished_at = _now_iso()
    _relocate_stage_outputs(stage, seed_dir)

    if losses is not None:
        _write_train_log(seed_dir / 'train_log.csv', losses, rewards)

    metrics_path = seed_dir / 'test_metrics.csv'
    if not metrics_path.exists():
        raise RuntimeError(f'{stage}: test_metrics.csv missing after relocation')
    row_count = _count_csv_rows(metrics_path)

    meta = {
        'exp_id': 'E03', 'model': model, 'seed': seed,
        'git_commit': _git_commit(),
        'started_at': started_at, 'finished_at': finished_at,
        'elapsed_train_s': elapsed_train,
        'hyperparams': hyperparams,
        'reward_weights': {'w1': WEIGHTS[0], 'w2': WEIGHTS[1],
                            'w3': WEIGHTS[2], 'w4': WEIGHTS[3]},
        'heuristic_params': heuristic_params,
        'data': {'train_n': len(train_inps), 'test_n': len(test_inps),
                  'split_seed': split_meta.get('split_seed'),
                  'split_file': split_meta.get('split_file')},
        'versions': {'python': sys.version, 'torch': torch.__version__},
    }
    _write_run_meta(seed_dir / 'run_meta.json', meta)
    _write_done_marker(seed_dir / 'DONE', row_count)

    return model, seed, row_count, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=8,
                    help='concurrent (model, seed) runs (default: 8; CPU/SWMM-bound, not GPU)')
    p.add_argument('--seeds', type=str, default=None,
                    help='comma-separated seed list (default: 1,2,3,4,5)')
    p.add_argument('--models', type=str, default=None,
                    help='comma-separated model list (default: all 5)')
    p.add_argument('--max-train', type=int, default=None,
                    help='truncate training scenarios (smoke-testing only)')
    p.add_argument('--max-test', type=int, default=None,
                    help='truncate test scenarios (smoke-testing only)')
    return p.parse_args()


def estimate_runtime(pending, workers):
    total_cpu_s = sum(ESTIMATED_SECONDS_FULL.get(t['model'], 12000)
                       for t in pending)
    wall_s = total_cpu_s / max(1, workers)
    return total_cpu_s, wall_s


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else SEEDS_DEFAULT
    models = args.models.split(',') if args.models else MODELS_DEFAULT
    for m in models:
        if m not in MODELS_DEFAULT:
            raise SystemExit(f'unknown model {m!r}, expected one of {MODELS_DEFAULT}')

    removed = cleanup_stale_tmp()
    if removed:
        log(f'cleaned up {len(removed)} stale .tmp file(s) from a prior interrupted run')

    expected_test_n = args.max_test if args.max_test is not None else FULL_TEST_N

    tasks = [{'model': m, 'seed': s, 'max_train': args.max_train,
              'max_test': args.max_test}
             for m in models for s in seeds]

    pending = []
    for t in tasks:
        if is_done(t['model'], t['seed'], expected_test_n):
            log(f"skip (already done): {t['model']} seed{t['seed']}")
        else:
            pending.append(t)

    log(f'{len(tasks)} total combos, {len(tasks) - len(pending)} already done, '
        f'{len(pending)} to run, workers={args.workers}')

    if pending:
        total_cpu_s, wall_s = estimate_runtime(pending, args.workers)
        log(f'estimated total CPU-time: {total_cpu_s/3600:.1f}h, '
            f'estimated wall time at workers={args.workers}: {wall_s/3600:.1f}h '
            f'(rough -- scheduling is not bin-packed, GA/PSO-only estimate is '
            f'extrapolated from a 30-scenario probe, not measured on the full set)')

    if not pending:
        log('nothing to do')
        return

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_one, t): t for t in pending}
        for future in as_completed(futures):
            t = futures[future]
            try:
                model, seed, row_count, elapsed = future.result()
                log(f'DONE {model} seed{seed}: {row_count} rows in {elapsed:.1f}s')
            except Exception as exc:  # noqa: BLE001 -- log and continue other tasks
                log(f"FAILED {t['model']} seed{t['seed']}: {exc!r}")

    log('run complete')


if __name__ == '__main__':
    main()
