#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-b follow-up: does performance keep improving for L in {6, 7, 8}, or
saturate? Old-manuscript weights, perfect forecast, gamma_h=1.0 only (E4-§3
already showed forecast/gamma_h barely matter at L<=5).

L=6..8 costs too much wall-clock for the full 270-scenario stratified
subsample even with the vectorised planner (L=8 ~= 0.8 s/decision -> hours
over 270 long scenarios), so this uses a smaller, explicitly-labelled set:
one scenario per duration (9 durations) at a single return period, run for
every L in {1..8} so the L=1..5 rows are directly comparable (same scenarios)
to the L=6..8 rows, not just to the separate 270-scenario sweep.

Run from src/.
"""
from __future__ import annotations

import csv
import os
import re
import time
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E04_DIR = ROOT / 'results' / 'E04_enum'

OLD_WEIGHTS = [0.25, 0.40, 0.25, 0.10]
DURATIONS = ['0060', '0120', '0180', '0240', '0360', '0540', '0720', '1080', '1440']
RETURN_PERIOD = '50'
L_LIST = [1, 2, 3, 4, 5, 6, 7, 8]


def main():
    os.chdir(str(SRC))
    import data_paths
    from sim_swmm import SimSwmm
    from water_gym import Level
    from lookahead_enum import LookaheadEnum, simulate_episode

    _, _, test = data_paths.load_fixed_split()
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for p in test:
        n = re.findall(r'\d+', p.split('/')[-1])
        buckets[(n[0], n[1])].append(p)
    scenarios = [buckets[(RETURN_PERIOD, d)][0] for d in DURATIONS]
    print(f'{len(scenarios)} scenarios (return period {RETURN_PERIOD}yr, '
          f'one per duration): {[Path(s).stem for s in scenarios]}')

    fields = ['scenario_id', 'return_period', 'duration_min', 'L', 'gamma_h',
              'forecast', 'n_decisions', 'act0_lock_rate', 'overflow',
              'max_level_m', 'steps_to_first_pump', 'wall_s']
    rows = []
    E04_DIR.mkdir(parents=True, exist_ok=True)
    out_path = E04_DIR / 'fixed_point_by_L_highL.csv'

    for si, inp in enumerate(scenarios):
        rains, outfalls, length = SimSwmm(inp, 2, 'gasan').execute()
        rp, dur = re.findall(r'\d+', inp.split('/')[-1])[:2]
        for L in L_LIST:
            pl = LookaheadEnum(OLD_WEIGHTS, minutes=2, L=L, gamma_h=1.0)
            t0 = time.perf_counter()
            acts, mx, dry, fp = simulate_episode(outfalls, length, pl, 'perfect')
            dt = time.perf_counter() - t0
            real = acts[1:]
            lock = sum(1 for a in real if a == 0) / len(real) if real else 0.0
            row = {
                'scenario_id': Path(inp).stem, 'return_period': rp,
                'duration_min': dur, 'L': L, 'gamma_h': 1.0,
                'forecast': 'perfect', 'n_decisions': len(real),
                'act0_lock_rate': round(lock, 4),
                'overflow': int(mx >= Level[-1]), 'max_level_m': round(mx, 4),
                'steps_to_first_pump': fp if fp is not None else '',
                'wall_s': round(dt, 3),
            }
            rows.append(row)
            print(f'  [{si + 1}/{len(scenarios)}] {row["scenario_id"]} L={L:2d} '
                  f'max_level={row["max_level_m"]:.3f} overflow={row["overflow"]} '
                  f'first_pump={row["steps_to_first_pump"]} wall={dt:.2f}s')
            _flush(out_path, fields, rows)

    print('\n=== per-L mean over the 9 scenarios ===')
    import statistics as st
    for L in L_LIST:
        grp = [r for r in rows if r['L'] == L]
        print(f'  L={L:2d}  mean_max_level={st.mean(r["max_level_m"] for r in grp):.3f}  '
              f'overflow_rate={st.mean(r["overflow"] for r in grp):.3f}  '
              f'mean_lock={st.mean(r["act0_lock_rate"] for r in grp):.3f}  '
              f'total_wall={sum(r["wall_s"] for r in grp):.1f}s')


def _flush(path: Path, fields, rows) -> None:
    from atomic_io import atomic_write

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    atomic_write(path, _w, newline='')


if __name__ == '__main__':
    main()
