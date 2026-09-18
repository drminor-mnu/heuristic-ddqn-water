#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-a driver: exhaustive-enumeration (ENUM) baseline.

docs/REVISION_EXPERIMENT_PLAN.md ``## E4`` (R1-1, R4-1).  Two parts:

  agreement  -- at every decision step of a stratified test subsample, score
                the 6 feasible actions with the exact 1-step objective (ENUM)
                and, on the *same* state trajectory, also ask GA and PSO what
                they would pick.  Reports:
                  * how often GA / PSO agree with the exact ENUM optimum
                  * per-decision wall-clock for ENUM vs GA vs PSO
                  * the objective gap tau(a_ENUM) - tau(a_GA/PSO)  (>= 0)
                Outputs: results/E04_enum/agreement_rate.csv,
                         results/E04_enum/decision_time_by_L.csv   (L=1 rows),
                         results/E04_enum/agreement_steps_L1.csv   (raw)

  baseline   -- run ENUM as a full-episode controller over the test set and
                write scenario-level raw metrics in the plan 2.2 schema, so
                ENUM can be placed alongside the E3 models.
                Output: results/E04_enum/enum_scenario_metrics.csv

Weights are the E3 v2 set [0.40, 0.25, 0.25, 0.10] (the "live" configuration).
The separate old-manuscript-weights fixed-point analysis (plan E4-§3) is a
different script and is run next, per the approved order E4-a -> E4-§3 -> E4-b.

Determinism: ENUM touches no RNG.  GA/PSO are seeded once (--seed, default 0)
before the agreement sweep so the reported agreement/timing is reproducible.

Run from the src/ directory (perform_evaluate writes '../results/...').

Usage:
    python run_e04_enum.py                      # both parts, full test set
    python run_e04_enum.py --part agreement --per-stratum 3
    python run_e04_enum.py --part baseline --workers 8
    python run_e04_enum.py --per-stratum 1 --limit 12   # smoke
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
RESULTS = ROOT / 'results'
E04_DIR = RESULTS / 'E04_enum'

# E3 v2 reward weights (run_e03_seeds.py). Used for the E4-a "live" comparison.
WEIGHTS = [0.40, 0.25, 0.25, 0.10]
BASE_PARAMS = {'minutes': 2, 'weights': WEIGHTS, 'act_type': 0}

YEARS = ['10', '20', '30', '50', '80', '100']
DURATIONS = ['0060', '0120', '0180', '0240', '0360', '0540', '0720', '1080', '1440']

L1_CANDIDATE_SPACE = 6  # feasible actions with act_type=0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _scenario_key(path: str) -> tuple[str, str]:
    """(return_period, duration) parsed from a scenario filename."""
    nums = re.findall(r'\d+', path.split('/')[-1])
    return nums[0], nums[1]


def stratified_subsample(test_inps, per_stratum: int) -> list[str]:
    """First `per_stratum` scenarios of each (return_period, duration) stratum,
    in the split file's own order (deterministic, no RNG)."""
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in test_inps:
        buckets[_scenario_key(p)].append(p)
    picked = []
    for key in sorted(buckets):
        picked.extend(buckets[key][:per_stratum])
    return picked


def _pct(values, q):
    if not values:
        return ''
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


def write_csv(path: Path, fieldnames, rows) -> None:
    from atomic_io import atomic_write

    def _write(fd):
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    atomic_write(path, _write, newline='')


# ---------------------------------------------------------------------------
# part 1: agreement + per-decision timing (L = 1)
# ---------------------------------------------------------------------------

def run_agreement(per_stratum: int, seed: int, limit: int | None) -> None:
    import numpy as np

    from water_gym import WaterGym
    from genetic_algo import GeneticAlgorithm
    from pso import ParticleSwarmOptimization
    from enum_baseline import EnumSearch
    import data_paths

    _, _, test_inps = data_paths.load_fixed_split()
    scenarios = stratified_subsample(test_inps, per_stratum)
    if limit is not None:
        scenarios = scenarios[:limit]

    np.random.seed(seed)
    import random
    random.seed(seed)

    enum = EnumSearch(**BASE_PARAMS)
    ga = GeneticAlgorithm(**BASE_PARAMS)
    pso = ParticleSwarmOptimization(**BASE_PARAMS)

    step_fields = ['scenario_id', 'return_period', 'duration_min', 'step',
                   'a_enum', 'a_ga', 'a_pso',
                   'tau_enum', 'tau_ga', 'tau_pso',
                   'gap_ga', 'gap_pso',
                   't_enum_s', 't_ga_s', 't_pso_s']
    step_rows = []

    t_start = time.perf_counter()
    for si, inp in enumerate(scenarios):
        rp, dur = _scenario_key(inp)
        scen_id = Path(inp).stem
        gym = WaterGym(inp, minutes=BASE_PARAMS['minutes'],
                       weights=WEIGHTS, act_type=BASE_PARAMS['act_type'])
        state, _r, done, _i = gym.reset()
        action_list = [0]
        step = 0
        while not done:
            next_inflow = gym.outfalls[gym.clock + 1]

            t0 = time.perf_counter()
            scores = enum.score_all_actions(state, action_list, next_inflow)
            a_enum = int(np.argmax(scores))
            t_enum = time.perf_counter() - t0

            t0 = time.perf_counter()
            a_ga = int(ga.genetic_algorithm(state, action_list, next_inflow))
            t_ga = time.perf_counter() - t0

            t0 = time.perf_counter()
            a_pso = int(pso.particle_swarm_optimization(state, action_list, next_inflow))
            t_pso = time.perf_counter() - t0

            tau_enum = float(scores[a_enum])
            tau_ga = float(scores[a_ga])
            tau_pso = float(scores[a_pso])
            step_rows.append({
                'scenario_id': scen_id, 'return_period': rp, 'duration_min': dur,
                'step': step,
                'a_enum': a_enum, 'a_ga': a_ga, 'a_pso': a_pso,
                'tau_enum': tau_enum, 'tau_ga': tau_ga, 'tau_pso': tau_pso,
                'gap_ga': tau_enum - tau_ga, 'gap_pso': tau_enum - tau_pso,
                't_enum_s': t_enum, 't_ga_s': t_ga, 't_pso_s': t_pso,
            })

            action_list.append(a_enum)          # advance on the exact optimum
            state, _r, done, _i = gym.step(a_enum)
            step += 1
        print(f'\r  agreement {si + 1}/{len(scenarios)} ({scen_id})',
              end='', flush=True)
    print()
    elapsed = time.perf_counter() - t_start

    E04_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(E04_DIR / 'agreement_steps_L1.csv', step_fields, step_rows)

    n = len(step_rows)
    summ_fields = ['optimizer', 'n_decisions', 'n_agree_with_enum',
                   'agreement_rate', 'mean_gap', 'max_gap',
                   'mean_decision_s', 'median_decision_s', 'p95_decision_s']
    summ_rows = []
    for opt, akey, gapkey, tkey in (
        ('ENUM', 'a_enum', None, 't_enum_s'),
        ('GA', 'a_ga', 'gap_ga', 't_ga_s'),
        ('PSO', 'a_pso', 'gap_pso', 't_pso_s'),
    ):
        agree = sum(1 for r in step_rows if r[akey] == r['a_enum'])
        gaps = [r[gapkey] for r in step_rows] if gapkey else [0.0] * n
        times = [r[tkey] for r in step_rows]
        summ_rows.append({
            'optimizer': opt, 'n_decisions': n, 'n_agree_with_enum': agree,
            'agreement_rate': (agree / n) if n else '',
            'mean_gap': (sum(gaps) / n) if n else '',
            'max_gap': max(gaps) if gaps else '',
            'mean_decision_s': (sum(times) / n) if n else '',
            'median_decision_s': statistics.median(times) if times else '',
            'p95_decision_s': _pct(times, 0.95),
        })
    write_csv(E04_DIR / 'agreement_rate.csv', summ_fields, summ_rows)

    dt_fields = ['L', 'candidate_space', 'optimizer', 'n_decisions',
                 'mean_decision_s', 'median_decision_s', 'p95_decision_s',
                 'mean_obj_gap', 'agreement_rate']
    dt_rows = [{
        'L': 1, 'candidate_space': L1_CANDIDATE_SPACE, 'optimizer': s['optimizer'],
        'n_decisions': s['n_decisions'],
        'mean_decision_s': s['mean_decision_s'],
        'median_decision_s': s['median_decision_s'],
        'p95_decision_s': s['p95_decision_s'],
        'mean_obj_gap': s['mean_gap'],
        'agreement_rate': s['agreement_rate'],
    } for s in summ_rows]
    write_csv(E04_DIR / 'decision_time_by_L.csv', dt_fields, dt_rows)

    print(f'  {len(scenarios)} scenarios, {n} decisions, {elapsed:.1f}s')
    for s in summ_rows:
        rate = s['agreement_rate']
        gap = s['mean_gap']
        mt = s['mean_decision_s']
        rate_s = f'{rate:.4f}' if isinstance(rate, float) else str(rate)
        gap_s = f'{gap:.6g}' if isinstance(gap, float) else str(gap)
        mt_s = f'{mt * 1e3:.4f} ms' if isinstance(mt, float) else str(mt)
        print(f"    {s['optimizer']:5s} agree_with_ENUM={rate_s:>8}  "
              f"mean_gap={gap_s:>12}  mean_decision={mt_s}")


# ---------------------------------------------------------------------------
# part 2: ENUM full-episode baseline -> scenario-level raw metrics
# ---------------------------------------------------------------------------

def _baseline_chunk(task: dict):
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
    os.chdir(str(SRC))

    import perform_evaluate as pe
    from enum_baseline import EnumSearch

    k = task['k']
    chunk = task['chunk']
    stage = f'E04_enumbase_part{k}'
    pe._evaluate_heuristic(
        chunk, YEARS, DURATIONS,
        optimizer_class=EnumSearch,
        optimize_label=f'ENUM[{k}]',
        run_method='run_genetic_algo',
        result_prefix='enum',
        result_file=stage,
        seed=0,
        model_label='ENUM',
        **BASE_PARAMS,
    )
    return k, stage, len(chunk)


def run_baseline(workers: int, limit: int | None) -> None:
    os.chdir(str(SRC))
    import data_paths

    _, _, test_inps = data_paths.load_fixed_split()
    if limit is not None:
        test_inps = test_inps[:limit]

    workers = max(1, min(workers, len(test_inps)))
    chunks = [test_inps[i::workers] for i in range(workers)]
    tasks = [{'k': i, 'chunk': c} for i, c in enumerate(chunks) if c]

    E04_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_baseline_parts()

    t0 = time.perf_counter()
    part_files = []
    if len(tasks) == 1:
        k, stage, ncnt = _baseline_chunk(tasks[0])
        part_files.append(RESULTS / f'{stage}_scenario_metrics.csv')
        print(f'  chunk {k}: {ncnt} scenarios')
    else:
        with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
            futs = {ex.submit(_baseline_chunk, t): t for t in tasks}
            for fut in as_completed(futs):
                k, stage, ncnt = fut.result()
                part_files.append(RESULTS / f'{stage}_scenario_metrics.csv')
                print(f'  chunk {k}: {ncnt} scenarios done')

    merged = _merge_scenario_parts(part_files)
    out = E04_DIR / 'enum_scenario_metrics.csv'
    from atomic_io import atomic_write

    header = merged[0] if merged else []
    body = merged[1:]

    def _write(fd):
        w = csv.writer(fd)
        w.writerow(header)
        w.writerows(body)

    atomic_write(out, _write, newline='')
    _cleanup_baseline_parts()

    elapsed = time.perf_counter() - t0
    n_over = sum(1 for r in body if r[header.index('overflow_flag')] == '1')
    print(f'  wrote {out.relative_to(ROOT)} ({len(body)} scenarios, '
          f'{n_over} overflow) in {elapsed:.1f}s')


def _merge_scenario_parts(part_files):
    merged = []
    header = None
    for pf in sorted(part_files, key=lambda p: p.name):
        if not pf.exists():
            continue
        with open(pf, newline='') as f:
            rows = list(csv.reader(f))
        if not rows:
            continue
        if header is None:
            header = rows[0]
            merged.append(header)
        merged.extend(rows[1:])
    return merged


def _cleanup_baseline_parts():
    for p in RESULTS.glob('E04_enumbase_part*'):
        try:
            p.unlink()
        except OSError:
            pass
    for p in RESULTS.glob('.E04_enumbase_part*.tmp'):
        try:
            p.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--part', choices=['agreement', 'baseline', 'all'],
                   default='all')
    p.add_argument('--per-stratum', type=int, default=3,
                   help='agreement: scenarios per (return_period x duration) '
                        'stratum (54 strata; default 3 -> 162 scenarios)')
    p.add_argument('--seed', type=int, default=0,
                   help='agreement: RNG seed for GA/PSO (ENUM is RNG-free)')
    p.add_argument('--workers', type=int, default=8,
                   help='baseline: parallel chunks (CPU/SWMM-bound)')
    p.add_argument('--limit', type=int, default=None,
                   help='cap number of scenarios (smoke-testing)')
    return p.parse_args()


def main():
    os.chdir(str(SRC))
    args = parse_args()
    E04_DIR.mkdir(parents=True, exist_ok=True)

    if args.part in ('agreement', 'all'):
        print('[E4-a] agreement + per-decision timing (L=1)')
        run_agreement(args.per_stratum, args.seed, args.limit)

    if args.part in ('baseline', 'all'):
        print('[E4-a] ENUM full-episode baseline -> scenario metrics')
        run_baseline(args.workers, args.limit)


if __name__ == '__main__':
    sys.exit(main())
