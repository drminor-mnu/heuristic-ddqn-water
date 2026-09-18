#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-e Task B (saturation, new weights) + Task D (constant-sequence collapse)
+ raw for Task C (cumulative tau_L / realized tau_1, per scenario per L).

docs/E04E_REPORT.md. New weights [0.40, 0.25, 0.25, 0.10] only (E3 v2 / "신가중치").
Reuses the *same* 270-scenario stratified subsample as E4-§3
(results/E04_enum/fixed_point_by_L_scenarios.csv scenario_id list) -- does not
regenerate it.

Cost-driven split (measured, see docs/E04E_REPORT.md Task B note): the
vectorised planner costs ~0.83 s/decision at L=8; running the full 270 at
L in {7,8} would take on the order of 6 hours. L in {1,2,3,5,6} (<=2.3
s/decision) runs on the full 270. L in {7,8} runs on an explicitly reduced
n=18 subsample (2 scenarios per duration, spanning 2 return periods) --
deviation from the requested n>=90 floor, documented in the report.

Per (scenario, L) row:
  max_level_m, overflow, n_switches, act0_lock_rate, steps_to_first_pump,
  n_dryrun_proxy                      -- Task B base metrics
  cum_tau_L                           -- sum over decisions of tau_L(best
                                         sequence chosen at that decision);
                                         provably non-decreasing in L (see
                                         report) -- Task C sanity check
  cum_tau1_realized                   -- sum over decisions of the realized
                                         1-step tau of the *executed* action
                                         (not guaranteed monotonic) -- Task C
  frac_const_seq, mean_const_gap,
  frac_const_gap_zero                 -- Task D (L>=2 only)

Run from src/.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E04_DIR = ROOT / 'results' / 'E04_enum'

NEW_WEIGHTS = [0.40, 0.25, 0.25, 0.10]
GAMMA_H = 1.0
FORECAST = 'persistence'          # main value per instructions; perfect is reference
FULL_L = [1, 2, 3, 5, 6]
REDUCED_L = [7, 8]
REDUCED_DURATIONS = ['0060', '0120', '0180', '0240', '0360', '0540', '0720', '1080', '1440']
REDUCED_RETURN_PERIODS = ['10', '80']   # 2 return periods x 9 durations = 18
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
    missing = [s for s in scenario_ids if s not in by_stem]
    if missing:
        raise SystemExit(f'{len(missing)} scenario ids not found in fixed '
                         f'split test set, e.g. {missing[:5]}')
    return [by_stem[s] for s in scenario_ids]


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

        vol, level, prev = 0.0, float(Level[0]), 0
        actions = [0]
        max_level = level
        n_dry = 0
        first_pump = None
        cum_tau_L = 0.0
        cum_tau1_realized = 0.0
        n_const = 0
        const_gaps = []

        for step in range(n_dec):
            clock = step + 1
            fc = build_forecast(outfalls, clock, length, L, FORECAST)
            first, det = pl.plan(vol, level, prev, fc, return_detail=True)
            cum_tau_L += det['best_total']
            if len(set(det['best_seq'])) == 1:
                n_const += 1
            if L >= 2:
                const_scores = pl.score_sequences(const_seqs, vol, level, prev, fc)
                const_gaps.append(det['best_total'] - float(np.max(const_scores)))

            tau1_vec = pl.tau1_per_action(vol, level, prev, float(outfalls[clock]))
            cum_tau1_realized += float(tau1_vec[first])

            actions.append(first)
            if first != 0 and first_pump is None:
                first_pump = step

            cur_inflow = outfalls[clock]
            new_vol = vol + cur_inflow - pl.pump_rate[first] * pl.minutes
            if new_vol < 0.0:
                new_vol = 0.0
                n_dry += 1
            vol = new_vol
            level = float(np.interp(vol, pl.Volume, pl.Level))
            prev = first
            if level > max_level:
                max_level = level

        real = actions[1:]
        n_switch = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])
        rows.append({
            'scenario_id': Path(inp).stem, 'return_period': rp,
            'duration_min': dur, 'L': L, 'gamma_h': GAMMA_H,
            'forecast': FORECAST, 'weights': 'new',
            'n_decisions': n_dec,
            'max_level_m': round(max_level, 4),
            'overflow': int(max_level >= Level[-1]),
            'n_switches': n_switch,
            'act0_lock_rate': round(sum(1 for a in real if a == 0) / n_dec, 5) if n_dec else '',
            'steps_to_first_pump': first_pump if first_pump is not None else '',
            'n_dryrun_proxy': n_dry,
            'cum_tau_L': round(cum_tau_L, 6),
            'cum_tau1_realized': round(cum_tau1_realized, 6),
            'frac_const_seq': round(n_const / n_dec, 5) if (n_dec and L >= 2) else ('' if L < 2 else 1.0),
            'mean_const_gap': (round(sum(const_gaps) / len(const_gaps), 6)
                              if const_gaps else ''),
            'frac_const_gap_zero': (round(sum(1 for g in const_gaps if g <= 1e-9) / len(const_gaps), 5)
                                   if const_gaps else ''),
        })
    return rows


def _flush(path: Path, fields, rows) -> None:
    from atomic_io import atomic_write

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    atomic_write(path, _w, newline='')


def main():
    os.chdir(str(SRC))
    p = argparse.ArgumentParser()
    p.add_argument('--which', choices=['full', 'reduced', 'both'], default='both')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()

    fields = ['scenario_id', 'return_period', 'duration_min', 'L', 'gamma_h',
              'forecast', 'weights', 'n_decisions', 'max_level_m', 'overflow',
              'n_switches', 'act0_lock_rate', 'steps_to_first_pump',
              'n_dryrun_proxy', 'cum_tau_L', 'cum_tau1_realized',
              'frac_const_seq', 'mean_const_gap', 'frac_const_gap_zero']
    E04_DIR.mkdir(parents=True, exist_ok=True)

    if args.which in ('full', 'both'):
        scenario_ids = load_scenario_ids()
        scenarios = resolve_scenarios(scenario_ids)
        print(f'[full] {len(scenarios)} scenarios (E4-§3 subsample, reused), '
              f'L={FULL_L}')
        out_path = E04_DIR / 'saturation_w2_scenarios.csv'
        rows = []
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_one, s, FULL_L): s for s in scenarios}
            done = 0
            for fut in as_completed(futs):
                rows.extend(fut.result())
                done += 1
                print(f'\r  [full] {done}/{len(scenarios)} '
                      f'({time.perf_counter() - t0:.0f}s)', end='', flush=True)
                if done % 20 == 0:
                    _flush(out_path, fields, rows)
        print()
        _flush(out_path, fields, rows)
        print(f'  [full] wrote {out_path.relative_to(ROOT)} ({len(rows)} rows) '
              f'in {time.perf_counter() - t0:.0f}s')

    if args.which in ('reduced', 'both'):
        import data_paths
        _, _, test = data_paths.load_fixed_split()
        buckets: dict[tuple, list[str]] = defaultdict(list)
        for pth in test:
            buckets[_key(pth)].append(pth)
        red_scenarios = [buckets[(rp, d)][0]
                         for rp in REDUCED_RETURN_PERIODS
                         for d in REDUCED_DURATIONS]
        print(f'[reduced] {len(red_scenarios)} scenarios '
              f'(return periods {REDUCED_RETURN_PERIODS} x '
              f'{len(REDUCED_DURATIONS)} durations), L={REDUCED_L} '
              f'-- deviation from n>=90, see report')
        out_path = E04_DIR / 'saturation_w2_scenarios_L78.csv'
        rows = []
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=min(args.workers, len(red_scenarios))) as ex:
            futs = {ex.submit(_run_one, s, REDUCED_L): s for s in red_scenarios}
            done = 0
            for fut in as_completed(futs):
                rows.extend(fut.result())
                done += 1
                print(f'  [reduced] {done}/{len(red_scenarios)} '
                      f'({time.perf_counter() - t0:.0f}s elapsed)')
                _flush(out_path, fields, rows)
        print(f'  [reduced] wrote {out_path.relative_to(ROOT)} ({len(rows)} rows) '
              f'in {time.perf_counter() - t0:.0f}s')


if __name__ == '__main__':
    main()
