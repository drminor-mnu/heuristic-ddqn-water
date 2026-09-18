#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-f Task G: does a terminal value term remove the L-vs-max_level reversal
found in E4-e Task C?

    tau_L^term = sum_{k=0..L-1} gamma_h**k * tau(k)  +  gamma_h**L * V_term(level_{t+L})

Two terminal forms:
  (i)  V_term = -c * level / level_max
  (ii) V_term = -c * max(0, level - 8.0) / (10.0 - 8.0)          (near-overflow penalty)

c in {0.5, 1, 2, 5} (c=0 is the E4-e Task B/C baseline, already computed --
reused from results/E04_enum/saturation_w2_scenarios.csv /
saturation_w2_scenarios_L78.csv, not rerun).

Scenario set: identical to E4-e Task B -- the 270-scenario stratified
subsample for L in {1,2,3,5,6} (full), the same 18-scenario reduced subsample
for L in {7,8} (one representative c per form, chosen after the L<=6 sweep).
New weights [0.40, 0.25, 0.25, 0.10] only. Persistence forecast (same as
Task B). Additive: lookahead_enum.LookaheadEnum.plan() is untouched; this
uses the new plan_with_terminal() method added alongside it.

Run from src/.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E04_DIR = ROOT / 'results' / 'E04_enum'

NEW_WEIGHTS = [0.40, 0.25, 0.25, 0.10]
GAMMA_H = 1.0
FORECAST = 'persistence'
SCENARIO_LIST_SOURCE = E04_DIR / 'fixed_point_by_L_scenarios.csv'


def _key(path: str) -> tuple[str, str]:
    n = re.findall(r'\d+', path.split('/')[-1])
    return n[0], n[1]


def load_scenario_ids() -> list[str]:
    ids, order = set(), []
    with open(SCENARIO_LIST_SOURCE) as f:
        for r in csv.DictReader(f):
            if r['scenario_id'] not in ids:
                ids.add(r['scenario_id'])
                order.append(r['scenario_id'])
    return order


def resolve_scenarios(scenario_ids: list[str]) -> list[str]:
    import data_paths
    _, _, test = data_paths.load_fixed_split()
    by_stem = {Path(p).stem: p for p in test}
    return [by_stem[s] for s in scenario_ids]


def make_terminal_fn(form: str, c: float):
    import numpy as np

    if form == 'level':
        return lambda level: -c * level / 10.0
    if form == 'near_overflow':
        return lambda level: -c * np.maximum(0.0, level - 8.0) / 2.0
    raise ValueError(form)


def _run_one(inp: str, L_list: list[int], form: str, c: float) -> list[dict]:
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
    term_fn = make_terminal_fn(form, c)
    out = []
    for L in L_list:
        pl = LookaheadEnum(NEW_WEIGHTS, minutes=2, L=L, gamma_h=GAMMA_H)
        vol, level, prev = 0.0, float(Level[0]), 0
        actions = [0]
        max_level = level
        cum_tau_L = 0.0   # tau-only part (terminal excluded), for Task C comparability
        first_pump = None
        for step in range(n_dec):
            clock = step + 1
            fc = build_forecast(outfalls, clock, length, L, FORECAST)
            first, det = pl.plan_with_terminal(vol, level, prev, fc, term_fn,
                                               return_detail=True)
            cum_tau_L += det['best_total_tau_only']
            actions.append(first)
            if first != 0 and first_pump is None:
                first_pump = step
            cur_inflow = outfalls[clock]
            new_vol = vol + cur_inflow - pl.pump_rate[first] * pl.minutes
            vol = max(new_vol, 0.0)
            level = float(np.interp(vol, pl.Volume, pl.Level))
            prev = first
            if level > max_level:
                max_level = level
        real = actions[1:]
        n_switch = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])
        out.append({
            'scenario_id': Path(inp).stem, 'return_period': rp,
            'duration_min': dur, 'L': L, 'form': form, 'c': c,
            'n_decisions': n_dec, 'max_level_m': round(max_level, 4),
            'overflow': int(max_level >= Level[-1]),
            'n_switches': n_switch,
            'steps_to_first_pump': first_pump if first_pump is not None else '',
            'cum_tau_L': round(cum_tau_L, 6),
        })
    return out


def _flush(path: Path, fields, rows) -> None:
    from atomic_io import atomic_write

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    atomic_write(path, _w, newline='')


FIELDS = ['scenario_id', 'return_period', 'duration_min', 'L', 'form', 'c',
          'n_decisions', 'max_level_m', 'overflow', 'n_switches',
          'steps_to_first_pump', 'cum_tau_L']


def run_sweep(scenarios, L_list, forms, c_list, workers, out_path, label):
    import time

    configs = [(form, c) for form in forms for c in c_list]
    print(f'[{label}] {len(scenarios)} scenarios x {len(configs)} (form,c) x '
          f'{len(L_list)} L values = {len(scenarios) * len(configs)} scenario-config jobs')
    all_rows = []
    t0 = time.perf_counter()
    tasks = [(s, form, c) for s in scenarios for (form, c) in configs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, s, L_list, form, c): (s, form, c)
                for (s, form, c) in tasks}
        done = 0
        for fut in as_completed(futs):
            all_rows.extend(fut.result())
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f'\r  [{label}] {done}/{len(tasks)} '
                      f'({time.perf_counter() - t0:.0f}s)', end='', flush=True)
                _flush(out_path, FIELDS, all_rows)
    print()
    _flush(out_path, FIELDS, all_rows)
    print(f'  [{label}] wrote {out_path.relative_to(ROOT)} ({len(all_rows)} rows) '
          f'in {time.perf_counter() - t0:.0f}s')


def main():
    os.chdir(str(SRC))
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['main', 'highL'], default='main')
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--highL-c', type=float, default=None,
                   help='representative c for the L in {7,8} phase')
    p.add_argument('--highL-forms', type=str, default='level,near_overflow')
    args = p.parse_args()

    scenario_ids = load_scenario_ids()
    scenarios = resolve_scenarios(scenario_ids)

    if args.phase == 'main':
        run_sweep(scenarios, [1, 2, 3, 5, 6], ['level', 'near_overflow'],
                  [0.5, 1.0, 2.0, 5.0], args.workers,
                  E04_DIR / 'terminal_value_w2.csv', 'main')
    else:
        if args.highL_c is None:
            raise SystemExit('--highL-c required for phase=highL')
        import data_paths
        _, _, test = data_paths.load_fixed_split()
        buckets: dict[tuple, list[str]] = defaultdict(list)
        for pth in test:
            buckets[_key(pth)].append(pth)
        red_scenarios = [buckets[(rp, d)][0]
                         for rp in ('10', '80')
                         for d in ('0060', '0120', '0180', '0240', '0360',
                                  '0540', '0720', '1080', '1440')]
        forms = args.highL_forms.split(',')
        run_sweep(red_scenarios, [7, 8], forms, [args.highL_c], args.workers,
                  E04_DIR / 'terminal_value_w2_highL.csv', 'highL')


if __name__ == '__main__':
    main()
