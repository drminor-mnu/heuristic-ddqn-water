#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-§3 driver: does the 1-step action-0 fixed point dissolve at horizon L>1 ?

docs/REVISION_EXPERIMENT_PLAN.md ``#### E4-§3``.  Old-manuscript weights
[0.25, 0.40, 0.25, 0.10] -- the setting where the 1-step objective locks onto
action 0 (docs/E00_HEURISTIC_MYOPIA_FINDING.md).  No training.

sweep  -- 270-scenario stratified test subsample, L in {1..5},
          gamma_h in {1.0, 0.9}, inflow forecast in {perfect, persistence}.
          Per config: action-0 lock rate, overflow rate, mean max_level,
          steps-to-first-pump.  Reports the escape threshold L* (smallest L
          with 0 overflow) per (gamma_h, forecast).
          -> results/E04_enum/fixed_point_by_L.csv
          -> results/E04_enum/fixed_point_by_L_scenarios.csv
          -> results/E04_enum/fixed_point_by_L_by_duration.csv

probe  -- one scenario: at selected decision steps, tau_1 per action vs
          tau_L(best sequence) vs tau_L(all-zeros), for each L, showing why the
          switch happens at a particular L.
          -> results/E04_enum/horizon_probe_<scenario>.csv

Run from src/.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics as st
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E04_DIR = ROOT / 'results' / 'E04_enum'

OLD_WEIGHTS = [0.25, 0.40, 0.25, 0.10]   # E00 fixed-point setting
L_LIST = [1, 2, 3, 4, 5]
GAMMA_LIST = [1.0, 0.9]
FORECASTS = ['perfect', 'persistence']
MINUTES = 2


def _key(path: str) -> tuple[str, str]:
    n = re.findall(r'\d+', path.split('/')[-1])
    return n[0], n[1]


def stratified_subsample(test_inps, per_stratum: int) -> list[str]:
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in test_inps:
        buckets[_key(p)].append(p)
    picked = []
    for k in sorted(buckets):
        picked.extend(buckets[k][:per_stratum])
    return picked


def _configs():
    """(L, gamma_h, forecast); collapse redundant L=1 duplicates."""
    seen = set()
    for L in L_LIST:
        for g in GAMMA_LIST:
            for fc in FORECASTS:
                tag = ('perfect', 1.0) if L == 1 else (fc, g)
                key = (L,) + tag
                if key in seen:
                    continue
                seen.add(key)
                yield L, tag[1], tag[0]


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def _run_scenario(inp: str):
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    os.chdir(str(SRC))
    import numpy as np
    from sim_swmm import SimSwmm
    from water_gym import Level
    from lookahead_enum import LookaheadEnum, simulate_episode

    rains, outfalls, length = SimSwmm(inp, MINUTES, 'gasan').execute()
    rp, dur = _key(inp)
    out = []
    for L, g, fc in _configs():
        pl = LookaheadEnum(OLD_WEIGHTS, minutes=MINUTES, L=L, gamma_h=g)
        acts, mx, dry, first_pump = simulate_episode(outfalls, length, pl, fc)
        real = acts[1:]
        n = len(real)
        n_switch = sum(1 for i in range(1, len(acts)) if acts[i] != acts[i - 1])
        out.append({
            'scenario_id': Path(inp).stem, 'return_period': rp,
            'duration_min': dur, 'L': L, 'gamma_h': g, 'forecast': fc,
            'n_decisions': n,
            'act0_lock_rate': float(np.mean([a == 0 for a in real])) if n else '',
            'fully_locked': int(all(a == 0 for a in real)) if n else '',
            'overflow': int(mx >= Level[-1]),
            'max_level_m': mx,
            'steps_to_first_pump': first_pump if first_pump is not None else '',
            'n_switches': n_switch,
            'n_dryrun_proxy': dry,
        })
    return out


def run_sweep(per_stratum: int, workers: int, limit: int | None) -> None:
    os.chdir(str(SRC))
    import data_paths
    _, _, test = data_paths.load_fixed_split()
    scenarios = stratified_subsample(test, per_stratum)
    if limit is not None:
        scenarios = scenarios[:limit]

    E04_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    workers = max(1, min(workers, len(scenarios)))
    if workers == 1:
        for i, s in enumerate(scenarios):
            rows.extend(_run_scenario(s))
            print(f'\r  {i + 1}/{len(scenarios)}', end='', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_scenario, s): s for s in scenarios}
            done = 0
            for fut in as_completed(futs):
                rows.extend(fut.result())
                done += 1
                print(f'\r  {done}/{len(scenarios)}', end='', flush=True)
    print()

    scen_fields = ['scenario_id', 'return_period', 'duration_min', 'L',
                   'gamma_h', 'forecast', 'n_decisions', 'act0_lock_rate',
                   'fully_locked', 'overflow', 'max_level_m',
                   'steps_to_first_pump', 'n_switches', 'n_dryrun_proxy']
    _write(E04_DIR / 'fixed_point_by_L_scenarios.csv', scen_fields, rows)

    # ---- aggregate per config ----
    by_cfg: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cfg[(r['L'], r['gamma_h'], r['forecast'])].append(r)

    agg_fields = ['L', 'gamma_h', 'forecast', 'n_scenarios',
                  'mean_act0_lock_rate', 'frac_fully_locked', 'overflow_rate',
                  'mean_max_level_m', 'median_steps_to_first_pump',
                  'n_never_pump', 'mean_n_switches', 'mean_n_dryrun_proxy']
    agg_rows = []
    for (L, g, fc) in sorted(by_cfg):
        grp = by_cfg[(L, g, fc)]
        ns = len(grp)
        firsts = [r['steps_to_first_pump'] for r in grp
                  if r['steps_to_first_pump'] != '']
        agg_rows.append({
            'L': L, 'gamma_h': g, 'forecast': fc, 'n_scenarios': ns,
            'mean_act0_lock_rate': round(st.mean(r['act0_lock_rate'] for r in grp), 5),
            'frac_fully_locked': round(st.mean(r['fully_locked'] for r in grp), 5),
            'overflow_rate': round(st.mean(r['overflow'] for r in grp), 5),
            'mean_max_level_m': round(st.mean(r['max_level_m'] for r in grp), 4),
            'median_steps_to_first_pump': (st.median(firsts) if firsts else ''),
            'n_never_pump': ns - len(firsts),
            'mean_n_switches': round(st.mean(r['n_switches'] for r in grp), 2),
            'mean_n_dryrun_proxy': round(st.mean(r['n_dryrun_proxy'] for r in grp), 3),
        })
    _write(E04_DIR / 'fixed_point_by_L.csv', agg_fields, agg_rows)

    # ---- overflow by duration ----
    dur_fields = ['L', 'gamma_h', 'forecast', 'duration_min', 'n_scenarios',
                  'overflow_rate', 'mean_max_level_m', 'mean_act0_lock_rate']
    dur_rows = []
    by_cfg_dur: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cfg_dur[(r['L'], r['gamma_h'], r['forecast'], r['duration_min'])].append(r)
    for key in sorted(by_cfg_dur):
        L, g, fc, d = key
        grp = by_cfg_dur[key]
        dur_rows.append({
            'L': L, 'gamma_h': g, 'forecast': fc, 'duration_min': d,
            'n_scenarios': len(grp),
            'overflow_rate': round(st.mean(r['overflow'] for r in grp), 4),
            'mean_max_level_m': round(st.mean(r['max_level_m'] for r in grp), 4),
            'mean_act0_lock_rate': round(st.mean(r['act0_lock_rate'] for r in grp), 4),
        })
    _write(E04_DIR / 'fixed_point_by_L_by_duration.csv', dur_fields, dur_rows)

    # ---- L* summary to stdout ----
    print('\n  escape threshold L* (smallest L with 0 overflow on the subsample):')
    for g in GAMMA_LIST:
        for fc in FORECASTS:
            lstar = None
            for L in L_LIST:
                key = (L, 1.0, 'perfect') if L == 1 else (L, g, fc)
                grp = by_cfg.get(key)
                if grp and st.mean(r['overflow'] for r in grp) == 0.0:
                    lstar = L
                    break
            print(f'    gamma_h={g}  forecast={fc:12s} -> L* = {lstar}')
    for row in agg_rows:
        print(f"    L={row['L']} g={row['gamma_h']} {row['forecast']:11s} "
              f"lock={row['mean_act0_lock_rate']:.3f} "
              f"overflow={row['overflow_rate']:.3f} "
              f"max_level={row['mean_max_level_m']:.3f} "
              f"first_pump~{row['median_steps_to_first_pump']}")


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

def run_probe(scenario_substr: str, steps: list[int] | None,
              gamma_h: float, forecast: str) -> None:
    os.chdir(str(SRC))
    import numpy as np
    import data_paths
    from sim_swmm import SimSwmm
    from water_gym import Level
    from lookahead_enum import LookaheadEnum, build_forecast, simulate_episode

    _, _, test = data_paths.load_fixed_split()
    matches = [p for p in test if scenario_substr in p]
    if not matches:
        raise SystemExit(f'no test scenario matches {scenario_substr!r}')
    inp = matches[0]
    rains, outfalls, length = SimSwmm(inp, MINUTES, 'gasan').execute()
    n_dec = length - 1
    print(f'probe {inp}  ({n_dec} decisions), gamma_h={gamma_h}, forecast={forecast}')

    # L=1 trajectory (to get the fixed-point vol/level path and the first
    # step where L=2 diverges).
    pl1 = LookaheadEnum(OLD_WEIGHTS, minutes=MINUTES, L=1, gamma_h=1.0)
    acts1, mx1, dry1, fp1 = simulate_episode(outfalls, length, pl1, 'perfect')

    pl2 = LookaheadEnum(OLD_WEIGHTS, minutes=MINUTES, L=2, gamma_h=gamma_h)
    acts2, mx2, dry2, fp2 = simulate_episode(outfalls, length, pl2, forecast)

    if steps is None:
        cand = sorted({max(0, (fp2 or 1) - 20), max(0, (fp2 or 1) - 1),
                       (fp2 if fp2 is not None else n_dec // 3),
                       min(n_dec - 1, (fp2 or 1) + 20),
                       min(n_dec - 1, n_dec // 2)})
        steps = [s for s in cand if 0 <= s < n_dec]

    # Re-walk the L=1 (fixed-point) water balance to get the state at each
    # probed decision step, then evaluate every L there.
    fields = ['step', 'clock', 'vol', 'level', 'prev_action', 'L',
              'first_action', 'best_seq', 'tau_L_best', 'tau_L_allzero',
              'best_minus_zero', 'tau1_per_action', 'myopic_stays']
    out_rows = []
    vol = 0.0
    level = float(Level[0])
    prev = 0
    probe_set = set(steps)
    for step in range(n_dec):
        clock = step + 1
        if step in probe_set:
            for L in L_LIST:
                pl = LookaheadEnum(OLD_WEIGHTS, minutes=MINUTES, L=L,
                                   gamma_h=gamma_h)
                fc = build_forecast(outfalls, clock, length, L, forecast)
                first, det = pl.plan(vol, level, prev, fc, return_detail=True)
                tau1 = det['tau1_per_action']
                out_rows.append({
                    'step': step, 'clock': clock, 'vol': round(vol, 1),
                    'level': round(level, 4), 'prev_action': prev, 'L': L,
                    'first_action': first,
                    'best_seq': '-'.join(map(str, det['best_seq'])),
                    'tau_L_best': round(det['best_total'], 5),
                    'tau_L_allzero': round(det['all_zero_total'], 5),
                    'best_minus_zero': round(det['best_minus_zero'], 5),
                    'tau1_per_action': '|'.join(f'{x:.4f}' for x in tau1),
                    'myopic_stays': int(int(np.argmax(tau1)) == 0),
                })
        # advance along the L=1 fixed-point path
        a = pl1.plan(vol, level, prev, build_forecast(outfalls, clock, length,
                                                      1, 'perfect'))
        cur_inflow = outfalls[clock]
        new_vol = vol + cur_inflow - pl1.pump_rate[a] * MINUTES
        if new_vol < 0.0:
            new_vol = 0.0
        vol = new_vol
        level = float(np.interp(vol, pl1.Volume, pl1.Level))
        prev = a

    E04_DIR.mkdir(parents=True, exist_ok=True)
    tag = Path(inp).stem
    _write(E04_DIR / f'horizon_probe_{tag}.csv', fields, out_rows)
    print(f'  L=1 overflow={mx1 >= Level[-1]} (max_level {mx1:.3f}), '
          f'L=2 overflow={mx2 >= Level[-1]} (max_level {mx2:.3f}, '
          f'first pump step {fp2})')
    print(f'  wrote results/E04_enum/horizon_probe_{tag}.csv '
          f'({len(out_rows)} rows, steps={steps})')
    for r in out_rows:
        print(f"    step {r['step']:>4} lvl={r['level']:.3f} prev={r['prev_action']} "
              f"| L={r['L']} a*={r['first_action']} "
              f"tauL(best)={r['tau_L_best']:.4f} tauL(0..0)={r['tau_L_allzero']:.4f} "
              f"Δ={r['best_minus_zero']:+.4f} myopic_stays={r['myopic_stays']}")


# ---------------------------------------------------------------------------

def _write(path: Path, fields, rows) -> None:
    from atomic_io import atomic_write

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    atomic_write(path, _w, newline='')


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['sweep', 'probe', 'all'], default='all')
    p.add_argument('--per-stratum', type=int, default=5)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--probe-scenario', type=str, default='1440m',
                   help='substring; first matching test scenario is probed')
    p.add_argument('--probe-steps', type=str, default=None,
                   help='comma-separated decision-step indices (default: auto)')
    p.add_argument('--probe-gamma', type=float, default=1.0)
    p.add_argument('--probe-forecast', choices=FORECASTS, default='perfect')
    return p.parse_args()


def main():
    os.chdir(str(SRC))
    a = parse_args()
    if a.mode in ('sweep', 'all'):
        print('[E4-§3] horizon sweep (old-manuscript weights)')
        run_sweep(a.per_stratum, a.workers, a.limit)
    if a.mode in ('probe', 'all'):
        print('\n[E4-§3] step-by-step tau_L probe')
        steps = ([int(x) for x in a.probe_steps.split(',')]
                 if a.probe_steps else None)
        run_probe(a.probe_scenario, steps, a.probe_gamma, a.probe_forecast)


if __name__ == '__main__':
    main()
