#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-e Task A: implementation-matched ENUM-L / GA-L / PSO-L comparison.

docs/E04E_REPORT.md Task A. For every method x implementation pair, measures
per-decision wall time, an implementation-independent evaluation count
(objective_function calls, actual for the deployed/un-batched classes, exact
analytic N*L for the vectorised ones since they never call
objective_function() individually), tau_L achieved, first action, and (for
GA-L/PSO-L) the tau_L gap and first-action agreement against the ENUM-L
optimum of the *same* implementation family.

Representative decision states are sampled from real receding-horizon L=6
rollouts (old-manuscript weights) over 3 scenarios spanning short/mid/long
duration, at evenly spaced decision indices -- not synthetic fixed states --
so prev_action, vol and level vary realistically.

GA-L/PSO-L hyperparameters are the unmodified genetic_algo.py/pso.py module
defaults (population_size=100, num_generations=100, mutation_rate=0.01,
crossover_rate=0.8; swarm_size=30, num_iterations=100, inertia=0.7,
cognitive=1.5, social=1.5) -- not scaled with L.

Run from src/. This does not touch genetic_algo.py, pso.py, lookahead_enum.py
or enum_l_deploy.py -- all instrumentation lives in lookahead_ga_pso.py
(_CallCounting mixin, _n_eval_equiv counter) and this script.
"""
from __future__ import annotations

import csv
import os
import re
import statistics as st
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E04_DIR = ROOT / 'results' / 'E04_enum'

OLD_WEIGHTS = [0.25, 0.40, 0.25, 0.10]
L_LIST = [1, 2, 3, 5, 6, 7, 8]
N_REPEATS_STOCHASTIC = 3  # GA-L / PSO-L (both impls); ENUM is deterministic
STATE_SCENARIOS_RP = '50'
STATE_DURATIONS = ['0180', '0540', '1440']  # short / mid / long
STATE_POINTS_PER_SCENARIO = 4


def _key(path: str) -> tuple[str, str]:
    n = re.findall(r'\d+', path.split('/')[-1])
    return n[0], n[1]


def build_states():
    """Sample representative (vol, level, prev_action, clock, outfalls,
    length, scenario_id) decision states from real L=6 receding-horizon
    rollouts (old-manuscript weights)."""
    import data_paths
    from sim_swmm import SimSwmm
    from lookahead_enum import LookaheadEnum, build_forecast

    _, _, test = data_paths.load_fixed_split()
    buckets: dict[tuple, list[str]] = {}
    for p in test:
        buckets.setdefault(_key(p), []).append(p)
    scenario_paths = [buckets[(STATE_SCENARIOS_RP, d)][0] for d in STATE_DURATIONS]

    states = []
    for inp in scenario_paths:
        rains, outfalls, length = SimSwmm(inp, 2, 'gasan').execute()
        pl = LookaheadEnum(OLD_WEIGHTS, minutes=2, L=6, gamma_h=1.0)
        vol, level, prev = 0.0, float(4.7), 0
        n_dec = length - 1
        traj = []
        for step in range(n_dec):
            clock = step + 1
            traj.append((step, clock, vol, level, prev))
            fc = build_forecast(outfalls, clock, length, 6, 'perfect')
            a = pl.plan(vol, level, prev, fc)
            cur_inflow = outfalls[clock]
            new_vol = vol + cur_inflow - pl.pump_rate[a] * 2
            vol = max(new_vol, 0.0)
            import numpy as np
            level = float(np.interp(vol, pl.Volume, pl.Level))
            prev = a
        idxs = [int(round(i * (n_dec - 1) / (STATE_POINTS_PER_SCENARIO - 1)))
                for i in range(STATE_POINTS_PER_SCENARIO)]
        for i in sorted(set(idxs)):
            step, clock, vol_i, level_i, prev_i = traj[i]
            states.append({
                'scenario_id': Path(inp).stem, 'step': step, 'clock': clock,
                'vol': vol_i, 'level': level_i, 'prev_action': prev_i,
                'outfalls': outfalls, 'length': length,
            })
    return states


def measure(state_id, s, L):
    from enum_l_deploy import DeployedLookaheadEnum
    from lookahead_enum import LookaheadEnum, build_forecast
    from lookahead_ga_pso import (DeployedLookaheadGA, DeployedLookaheadPSO,
                                  VectorizedLookaheadGA, VectorizedLookaheadPSO,
                                  _CallCounting)

    class CountedDeployedEnum(_CallCounting, DeployedLookaheadEnum):
        pass

    vol0, level0, prev = s['vol'], s['level'], s['prev_action']
    fc = build_forecast(s['outfalls'], s['clock'], s['length'], L, 'perfect')
    pstate = [0.0, 0.0, 0.0, vol0, level0]
    rows = []

    def _row(method, impl, wall_s, n_calls, first_action, best_total):
        return {
            'state_id': state_id, 'scenario_id': s['scenario_id'],
            'L': L, 'method': method, 'impl': impl,
            'wall_s': wall_s, 'n_obj_calls': n_calls,
            'first_action': first_action, 'best_total': best_total,
        }

    # ENUM -- deterministic, single run each.
    dep = CountedDeployedEnum(minutes=2, weights=OLD_WEIGHTS, L=L, gamma_h=1.0)
    t0 = time.perf_counter()
    a_enum_dep = dep.plan(pstate, [prev], fc)
    t_enum_dep = time.perf_counter() - t0
    rows.append(_row('ENUM', 'deployed', t_enum_dep, dep._n_obj_calls,
                      a_enum_dep, None))

    vec = LookaheadEnum(OLD_WEIGHTS, minutes=2, L=L, gamma_h=1.0)
    t0 = time.perf_counter()
    a_enum_vec, det = vec.plan(vol0, level0, prev, fc, return_detail=True)
    t_enum_vec = time.perf_counter() - t0
    n_actions = vec.n_actions
    rows.append(_row('ENUM', 'vectorized', t_enum_vec, (n_actions ** L) * L,
                      a_enum_vec, det['best_total']))
    enum_tau = det['best_total']

    # GA-L / PSO-L -- stochastic, N_REPEATS_STOCHASTIC repeats each.
    for method, dep_cls, vec_cls in (
        ('GA', DeployedLookaheadGA, VectorizedLookaheadGA),
        ('PSO', DeployedLookaheadPSO, VectorizedLookaheadPSO),
    ):
        for rep in range(N_REPEATS_STOCHASTIC):
            searcher = dep_cls(minutes=2, weights=OLD_WEIGHTS, L=L, gamma_h=1.0)
            t0 = time.perf_counter()
            a, det = searcher.plan(pstate, [prev], fc, return_detail=True)
            dt = time.perf_counter() - t0
            row = _row(method, 'deployed', dt, searcher._n_obj_calls, a,
                      det['best_total'])
            row['rep'] = rep
            row['gap_vs_enum'] = enum_tau - det['best_total']
            row['first_action_match'] = int(a == a_enum_dep)
            rows.append(row)

            searcher_v = vec_cls(OLD_WEIGHTS, minutes=2, L=L, gamma_h=1.0)
            t0 = time.perf_counter()
            av, detv = searcher_v.plan(vol0, level0, prev, fc, return_detail=True)
            dtv = time.perf_counter() - t0
            rowv = _row(method, 'vectorized', dtv, searcher_v._n_eval_equiv, av,
                       detv['best_total'])
            rowv['rep'] = rep
            rowv['gap_vs_enum'] = enum_tau - detv['best_total']
            rowv['first_action_match'] = int(av == a_enum_vec)
            rows.append(rowv)

    return rows


def main():
    os.chdir(str(SRC))
    print('building representative decision states from real L=6 rollouts...')
    states = build_states()
    print(f'{len(states)} states from {len(STATE_DURATIONS)} scenarios '
          f'(return period {STATE_SCENARIOS_RP}yr): '
          + ', '.join(f"{s['scenario_id']}#{s['step']}" for s in states))

    fields = ['state_id', 'scenario_id', 'L', 'method', 'impl', 'wall_s',
              'n_obj_calls', 'first_action', 'best_total', 'rep',
              'gap_vs_enum', 'first_action_match']
    all_rows = []
    E04_DIR.mkdir(parents=True, exist_ok=True)
    out_path = E04_DIR / 'impl_fairness_w2.csv'

    t_start = time.perf_counter()
    for si, s in enumerate(states):
        state_id = f"{s['scenario_id']}_step{s['step']}"
        for L in L_LIST:
            t0 = time.perf_counter()
            rows = measure(state_id, s, L)
            all_rows.extend(rows)
            print(f'  [{si + 1}/{len(states)}] {state_id} L={L} '
                  f'({time.perf_counter() - t0:.1f}s, total elapsed '
                  f'{time.perf_counter() - t_start:.0f}s)')
            _flush(out_path, fields, all_rows)
    print(f'done in {time.perf_counter() - t_start:.0f}s, '
          f'{len(all_rows)} rows -> {out_path.relative_to(ROOT)}')


def _flush(path, fields, rows) -> None:
    from atomic_io import atomic_write

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})

    atomic_write(path, _w, newline='')


if __name__ == '__main__':
    main()
