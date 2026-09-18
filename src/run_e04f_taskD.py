#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-f Task D-completion: (c) ENUM-const first-action match rate, (d)
ENUM-const full-episode performance, + ENUM-const decision time
(un-batched and vectorised).

"ENUM-const" = the reduced search that only evaluates the 6 constant
sequences (all-0, all-1, ..., all-5) instead of the full n_actions**L tree.
Uses lookahead_enum.LookaheadEnum.score_sequences() (additive method already
added for E4-e Task D(a)(b)) -- no change to plan()/score_sequences().

(c): walked along the TRUE ENUM-L optimal trajectory (the same trajectory
E4-e Task B/saturation_w2_scenarios.csv used), at every decision compare
ENUM-const's argmax-of-6 first action against the true ENUM-L first action.
(d): ENUM-const run as its OWN receding-horizon controller (its own
trajectory, since after one disagreement the visited states can differ).

New weights only, persistence forecast, same 270-scenario subsample as
Task B (E4-e) for L in {2,3,5,6}; L=8 uses the same 18-scenario reduced
subsample as Task B/E4-e.

Run from src/.
"""
from __future__ import annotations

import csv
import itertools
import os
import re
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E04_DIR = ROOT / 'results' / 'E04_enum'

NEW_WEIGHTS = [0.40, 0.25, 0.25, 0.10]
GAMMA_H = 1.0
FORECAST = 'persistence'
L_LIST_FULL = [2, 3, 5, 6]
L_LIST_REDUCED = [8]


def _key(path: str) -> tuple[str, str]:
    n = re.findall(r'\d+', path.split('/')[-1])
    return n[0], n[1]


def load_full_scenarios():
    import data_paths

    ids, order = set(), []
    with open(E04_DIR / 'fixed_point_by_L_scenarios.csv') as f:
        for r in csv.DictReader(f):
            if r['scenario_id'] not in ids:
                ids.add(r['scenario_id'])
                order.append(r['scenario_id'])
    _, _, test = data_paths.load_fixed_split()
    by_stem = {Path(p).stem: p for p in test}
    return [by_stem[s] for s in order]


def load_reduced_scenarios():
    import data_paths
    from collections import defaultdict

    _, _, test = data_paths.load_fixed_split()
    buckets = defaultdict(list)
    for p in test:
        buckets[_key(p)].append(p)
    return [buckets[(rp, d)][0] for rp in ('10', '80')
            for d in ('0060', '0120', '0180', '0240', '0360', '0540', '0720',
                      '1080', '1440')]


def _run_one(inp: str, L_list: list[int]) -> list[dict]:
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    os.chdir(str(SRC))
    import numpy as np
    from sim_swmm import SimSwmm
    from water_gym import Level
    from lookahead_enum import LookaheadEnum, build_forecast

    rains, outfalls, length = SimSwmm(inp, 2, 'gasan').execute()
    rp, dur = _key(inp)
    n_dec = length - 1
    rows = []
    for L in L_list:
        pl = LookaheadEnum(NEW_WEIGHTS, minutes=2, L=L, gamma_h=GAMMA_H)
        const_seqs = np.tile(np.arange(pl.n_actions, dtype=np.int64)[:, None], (1, L))

        # ---- (c) first-action match along the TRUE ENUM-L trajectory ----
        vol, level, prev = 0.0, float(Level[0]), 0
        n_match = 0
        for step in range(n_dec):
            clock = step + 1
            fc = build_forecast(outfalls, clock, length, L, FORECAST)
            true_first = pl.plan(vol, level, prev, fc)
            const_scores = pl.score_sequences(const_seqs, vol, level, prev, fc)
            const_first = int(const_seqs[int(np.argmax(const_scores)), 0])
            if const_first == true_first:
                n_match += 1
            cur_inflow = outfalls[clock]
            new_vol = vol + cur_inflow - pl.pump_rate[true_first] * pl.minutes
            vol = max(new_vol, 0.0)
            level = float(np.interp(vol, pl.Volume, pl.Level))
            prev = true_first
        match_rate = n_match / n_dec if n_dec else float('nan')

        # ---- (d) ENUM-const run as its own controller ----
        vol, level, prev = 0.0, float(Level[0]), 0
        actions = [0]
        max_level = level
        for step in range(n_dec):
            clock = step + 1
            fc = build_forecast(outfalls, clock, length, L, FORECAST)
            const_scores = pl.score_sequences(const_seqs, vol, level, prev, fc)
            a = int(const_seqs[int(np.argmax(const_scores)), 0])
            actions.append(a)
            cur_inflow = outfalls[clock]
            new_vol = vol + cur_inflow - pl.pump_rate[a] * pl.minutes
            vol = max(new_vol, 0.0)
            level = float(np.interp(vol, pl.Volume, pl.Level))
            prev = a
            if level > max_level:
                max_level = level
        real = actions[1:]
        n_switch = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])

        rows.append({
            'scenario_id': Path(inp).stem, 'return_period': rp,
            'duration_min': dur, 'L': L, 'n_decisions': n_dec,
            'first_action_match_rate': round(match_rate, 5),
            'const_max_level_m': round(max_level, 4),
            'const_overflow': int(max_level >= Level[-1]),
            'const_n_switches': n_switch,
        })
    return rows


def measure_timing(L_list):
    """ENUM-const decision time, un-batched and vectorised, one representative
    state (same state used in E4-e Task A calibration for continuity)."""
    import numpy as np
    from genetic_algo import GeneticAlgorithm
    from lookahead_enum import LookaheadEnum

    OLD_OR_NEW = NEW_WEIGHTS
    vol0, level0, prev, inflow = 8000.0, 8.0, 0, 50.0
    rows = []
    for L in L_list:
        fc = [inflow] * L
        # un-batched: literal L sequential objective_function() calls per
        # constant sequence, 6 sequences -- same style as enum_l_deploy.py.
        ga = GeneticAlgorithm(minutes=2, weights=OLD_OR_NEW)
        t0 = time.perf_counter()
        best_total, best_a = -1e18, 0
        for a_const in range(len(ga.Actions)):
            vol, level, hist = vol0, level0, [prev]
            total, g = 0.0, 1.0
            pstate = [0.0, 0.0, 0.0, vol, level]
            for k in range(L):
                pstate[3] = vol
                pstate[-1] = level
                total += g * ga.objective_function(a_const, pstate, hist, fc[k])
                pump_rate = sum(q for q, on in zip(__import__('water_gym').pumpq,
                                                    ga.Actions[a_const]) if on)
                attempted = pump_rate * ga.minutes
                available = vol + fc[k]
                vol = max(available - attempted, 0.0)
                level = ga._volume_to_level(vol)
                hist.append(a_const)
                g *= GAMMA_H
            if total > best_total:
                best_total, best_a = total, a_const
        t_dep = time.perf_counter() - t0

        pl = LookaheadEnum(OLD_OR_NEW, minutes=2, L=L, gamma_h=GAMMA_H)
        const_seqs = np.tile(np.arange(pl.n_actions, dtype=np.int64)[:, None], (1, L))
        t0 = time.perf_counter()
        _ = pl.score_sequences(const_seqs, vol0, level0, prev, fc)
        t_vec = time.perf_counter() - t0

        rows.append({'L': L, 'enum_const_deployed_s': round(t_dep, 6),
                     'enum_const_vectorized_s': round(t_vec, 6)})
    return rows


def main():
    os.chdir(str(SRC))
    full_scenarios = load_full_scenarios()
    reduced_scenarios = load_reduced_scenarios()

    fields = ['scenario_id', 'return_period', 'duration_min', 'L', 'n_decisions',
              'first_action_match_rate', 'const_max_level_m', 'const_overflow',
              'const_n_switches']
    all_rows = []

    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_run_one, s, L_LIST_FULL): s for s in full_scenarios}
        done = 0
        for fut in as_completed(futs):
            all_rows.extend(fut.result())
            done += 1
            print(f'\r[full L<=6] {done}/{len(full_scenarios)} '
                  f'({time.perf_counter() - t0:.0f}s)', end='', flush=True)
    print()

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_run_one, s, L_LIST_REDUCED): s for s in reduced_scenarios}
        done = 0
        for fut in as_completed(futs):
            all_rows.extend(fut.result())
            done += 1
            print(f'\r[reduced L=8] {done}/{len(reduced_scenarios)} '
                  f'({time.perf_counter() - t0:.0f}s)', end='', flush=True)
    print()

    out_path = E04_DIR / 'constant_collapse_w2.csv'
    from atomic_io import atomic_write

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    atomic_write(out_path, _w, newline='')
    print(f'wrote {out_path.relative_to(ROOT)} ({len(all_rows)} rows)')

    timing_rows = measure_timing([2, 3, 5, 6, 8])
    t_fields = ['L', 'enum_const_deployed_s', 'enum_const_vectorized_s']
    t_path = E04_DIR / 'constant_collapse_timing_w2.csv'

    def _wt(fd):
        w = csv.DictWriter(fd, fieldnames=t_fields)
        w.writeheader()
        for r in timing_rows:
            w.writerow(r)

    atomic_write(t_path, _wt, newline='')
    print(f'wrote {t_path.relative_to(ROOT)}')
    for r in timing_rows:
        print(r)

    import statistics as st
    from collections import defaultdict
    by_L = defaultdict(list)
    for r in all_rows:
        by_L[r['L']].append(r)
    print('\nper-L summary:')
    for L in sorted(by_L):
        grp = by_L[L]
        print(f"L={L} n={len(grp)} mean_match_rate={st.mean(r['first_action_match_rate'] for r in grp):.4f} "
              f"mean_const_max_level={st.mean(r['const_max_level_m'] for r in grp):.4f} "
              f"const_overflow_rate={st.mean(r['const_overflow'] for r in grp):.4f}")


if __name__ == '__main__':
    main()
