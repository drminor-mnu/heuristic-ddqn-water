#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-f Task I-2: clip(., >=0) diagnostics under a real operating policy.

E4-e Task F found clip(.,>=0) firing on 62-78% of *candidate sequence* steps
during ENUM-L's internal branch enumeration (branching, not an executed
trajectory). This checks the same question along an *executed* trajectory:
using ENUM-L2 (the best-performing controller found in Task B, new weights,
persistence forecast) over all 270 scenarios, log vol/level every real step
and see how often/where the basin actually runs (near-)empty.

Run from src/.
"""
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
L = 2
CAPACITY = 16000.0


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


def _run_one(inp: str) -> dict:
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.chdir(str(SRC))
    import numpy as np
    from sim_swmm import SimSwmm
    from lookahead_enum import LookaheadEnum, build_forecast

    rains, outfalls, length = SimSwmm(inp, 2, 'gasan').execute()
    rp, dur = _key(inp)
    pl = LookaheadEnum(NEW_WEIGHTS, minutes=2, L=L, gamma_h=1.0)
    vol, level, prev = 0.0, 4.7, 0
    vols, levels_at_clip = [], []
    n_clip = 0
    n_dec = length - 1
    for step in range(n_dec):
        clock = step + 1
        fc = build_forecast(outfalls, clock, length, L, 'persistence')
        first = pl.plan(vol, level, prev, fc)
        cur_inflow = outfalls[clock]
        attempted = pl.pump_rate[first] * pl.minutes
        available = vol + cur_inflow
        new_vol_raw = available - attempted
        if new_vol_raw < 0:
            n_clip += 1
            levels_at_clip.append(level)
        vol = max(new_vol_raw, 0.0)
        level = float(np.interp(vol, pl.Volume, pl.Level))
        prev = first
        vols.append(vol)

    vols_arr = np.array(vols)
    return {
        'scenario_id': Path(inp).stem, 'return_period': rp, 'duration_min': dur,
        'n_decisions': n_dec, 'n_clip': n_clip,
        'clip_rate': round(n_clip / n_dec, 5) if n_dec else '',
        'vol_mean_frac_cap': round(float(vols_arr.mean()) / CAPACITY, 5),
        'vol_median_frac_cap': round(float(np.median(vols_arr)) / CAPACITY, 5),
        'vol_p10_frac_cap': round(float(np.percentile(vols_arr, 10)) / CAPACITY, 5),
        'vol_p90_frac_cap': round(float(np.percentile(vols_arr, 90)) / CAPACITY, 5),
        'level_at_clip_mean': (round(float(np.mean(levels_at_clip)), 4)
                              if levels_at_clip else ''),
        'level_at_clip_max': (round(float(np.max(levels_at_clip)), 4)
                             if levels_at_clip else ''),
    }


def main():
    os.chdir(str(SRC))
    scenarios = load_scenarios()
    print(f'{len(scenarios)} scenarios, ENUM-L={L}, new weights, persistence')

    fields = ['scenario_id', 'return_period', 'duration_min', 'n_decisions',
              'n_clip', 'clip_rate', 'vol_mean_frac_cap', 'vol_median_frac_cap',
              'vol_p10_frac_cap', 'vol_p90_frac_cap', 'level_at_clip_mean',
              'level_at_clip_max']
    rows = []
    import time
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_run_one, s): s for s in scenarios}
        done = 0
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            print(f'\r{done}/{len(scenarios)} ({time.perf_counter()-t0:.0f}s)',
                  end='', flush=True)
    print()

    out_path = E04_DIR / 'clip_diagnostics_w2.csv'
    from atomic_io import atomic_write

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    atomic_write(out_path, _w, newline='')
    print(f'wrote {out_path.relative_to(ROOT)} ({len(rows)} rows)')

    import statistics as st
    print(f"\noverall: mean_clip_rate={st.mean(r['clip_rate'] for r in rows):.4f} "
          f"mean_vol_frac={st.mean(r['vol_mean_frac_cap'] for r in rows):.4f} "
          f"mean_vol_p90_frac={st.mean(r['vol_p90_frac_cap'] for r in rows):.4f}")
    by_dur = defaultdict(list)
    for r in rows:
        by_dur[r['duration_min']].append(r)
    print("by duration: dur  mean_clip_rate  mean_vol_frac  vol_p90_frac")
    for d in sorted(by_dur):
        g = by_dur[d]
        print(f"  {d}  {st.mean(x['clip_rate'] for x in g):.4f}  "
              f"{st.mean(x['vol_mean_frac_cap'] for x in g):.4f}  "
              f"{st.mean(x['vol_p90_frac_cap'] for x in g):.4f}")
    by_rp = defaultdict(list)
    for r in rows:
        by_rp[r['return_period']].append(r)
    print("by return period: rp  mean_clip_rate  mean_vol_frac")
    for rp in sorted(by_rp):
        g = by_rp[rp]
        print(f"  {rp}  {st.mean(x['clip_rate'] for x in g):.4f}  "
              f"{st.mean(x['vol_mean_frac_cap'] for x in g):.4f}")


if __name__ == '__main__':
    main()
