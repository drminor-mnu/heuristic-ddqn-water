#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-f Task H: perfect-foresight (A) vs persistence (B) inflow forecast.

E4-e Task B only ran (B) persistence. This runs (A) perfect foresight
(gym.outfalls[clock..clock+L-1], the true SWMM inflow) for L in {2,3,5} on
the *same* 270-scenario subsample as Task B (E4-e), so the two are paired.
New weights [0.40, 0.25, 0.25, 0.10] only, gamma_h=1.0.

Run from src/.
"""
from __future__ import annotations

import csv
import os
import re
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E04_DIR = ROOT / 'results' / 'E04_enum'

NEW_WEIGHTS = [0.40, 0.25, 0.25, 0.10]
GAMMA_H = 1.0
L_LIST = [2, 3, 5]


def _key(path: str) -> tuple[str, str]:
    n = re.findall(r'\d+', path.split('/')[-1])
    return n[0], n[1]


def load_scenarios():
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


def _run_one(inp: str) -> list[dict]:
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    os.chdir(str(SRC))
    import numpy as np
    from sim_swmm import SimSwmm
    from water_gym import Level
    from lookahead_enum import LookaheadEnum, build_forecast, simulate_episode

    rains, outfalls, length = SimSwmm(inp, 2, 'gasan').execute()
    rp, dur = _key(inp)
    rows = []
    for L in L_LIST:
        pl = LookaheadEnum(NEW_WEIGHTS, minutes=2, L=L, gamma_h=GAMMA_H)
        acts, mx, dry, fp = simulate_episode(outfalls, length, pl, 'perfect')
        real = acts[1:]
        n_switch = sum(1 for i in range(1, len(acts)) if acts[i] != acts[i - 1])
        rows.append({
            'scenario_id': Path(inp).stem, 'return_period': rp,
            'duration_min': dur, 'L': L, 'forecast': 'perfect',
            'n_decisions': length - 1, 'max_level_m': round(mx, 4),
            'overflow': int(mx >= Level[-1]), 'n_switches': n_switch,
            'steps_to_first_pump': fp if fp is not None else '',
            'n_dryrun_proxy': dry,
        })
    return rows


def main():
    os.chdir(str(SRC))
    scenarios = load_scenarios()
    print(f'{len(scenarios)} scenarios, L={L_LIST}, forecast=perfect, new weights')

    fields = ['scenario_id', 'return_period', 'duration_min', 'L', 'forecast',
              'n_decisions', 'max_level_m', 'overflow', 'n_switches',
              'steps_to_first_pump', 'n_dryrun_proxy']
    from concurrent.futures import ProcessPoolExecutor, as_completed
    all_rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_run_one, s): s for s in scenarios}
        done = 0
        for fut in as_completed(futs):
            all_rows.extend(fut.result())
            done += 1
            print(f'\r{done}/{len(scenarios)} ({time.perf_counter() - t0:.0f}s)',
                  end='', flush=True)
    print()

    out_path = E04_DIR / 'foresight_compare_w2.csv'
    from atomic_io import atomic_write

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    atomic_write(out_path, _w, newline='')
    print(f'wrote {out_path.relative_to(ROOT)} ({len(all_rows)} rows)')


if __name__ == '__main__':
    main()
