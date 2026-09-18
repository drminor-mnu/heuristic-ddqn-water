#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-e Task E: planner (ENUM-L) vs learned policy (E3 v2 DDQN), join only.

No retraining/re-evaluation. Filters E3 v2's 5-model x 5-seed test_metrics.csv
(results/E03_seeds/{model}/seed{n}/test_metrics.csv, new weights already) down
to the same 270-scenario subsample used in Task B, and places ENUM-L (each L)
next to Regular/GA-guided/PSO-guided/GA-only/PSO-only.

ENUM-L uses the true (persistence-forecast, see Task B) inflow forecast;
none of the 5 DDQN models do -- they act from the GRU-encoded recent
observation history only. Noted in the output and must be repeated in any
caption drawn from it.

Run from src/.
"""
import csv
import os
import statistics as st
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E04_DIR = ROOT / 'results' / 'E04_enum'
E03_DIR = ROOT / 'results' / 'E03_seeds'
MODELS = ['Regular', 'GA-guided', 'PSO-guided', 'GA-only', 'PSO-only']
SEEDS = [1, 2, 3, 4, 5]


def load_scenario_ids():
    ids = set()
    with open(E04_DIR / 'fixed_point_by_L_scenarios.csv') as f:
        for r in csv.DictReader(f):
            ids.add(r['scenario_id'])
    return ids


def main():
    os.chdir(str(SRC))
    scenario_ids = load_scenario_ids()
    print(f'{len(scenario_ids)} target scenario ids (Task B/E4-§3 subsample)')

    # ---- ENUM-L side ----
    enum_by_L: dict[int, list[dict]] = {}
    full_path = E04_DIR / 'saturation_w2_scenarios.csv'
    red_path = E04_DIR / 'saturation_w2_scenarios_L78.csv'
    for path in (full_path, red_path):
        if not path.exists():
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                if r['scenario_id'] not in scenario_ids:
                    continue
                L = int(r['L'])
                enum_by_L.setdefault(L, []).append(r)

    # ---- E3 v2 side: filter to same scenario ids, pool 5 seeds ----
    model_rows: dict[str, list[dict]] = {m: [] for m in MODELS}
    n_present = {}
    for model in MODELS:
        found_scenarios = set()
        for seed in SEEDS:
            path = E03_DIR / model / f'seed{seed}' / 'test_metrics.csv'
            if not path.exists():
                continue
            with open(path) as f:
                for r in csv.DictReader(f):
                    if r['scenario_id'] in scenario_ids:
                        model_rows[model].append(r)
                        found_scenarios.add(r['scenario_id'])
        n_present[model] = len(found_scenarios)

    print('E3 v2 scenario-id overlap with the 270-scenario subsample:')
    for m in MODELS:
        print(f'  {m}: {n_present[m]} / {len(scenario_ids)} scenarios present '
              f'(x {len(SEEDS)} seeds = {len(model_rows[m])} rows)')

    out_fields = ['source', 'model_or_L', 'n_scenarios', 'n_seeds_or_note',
                  'max_level_mean', 'max_level_std', 'n_switches_mean',
                  'n_dryrun_mean', 'overflow_rate']
    out_rows = []

    for model in MODELS:
        rows = model_rows[model]
        if not rows:
            out_rows.append({'source': 'E3v2-DDQN', 'model_or_L': model,
                             'n_scenarios': 0, 'n_seeds_or_note': 'NO DATA',
                             'max_level_mean': '', 'max_level_std': '',
                             'n_switches_mean': '', 'n_dryrun_mean': '',
                             'overflow_rate': ''})
            continue
        ml = [float(r['max_level_m']) for r in rows]
        sw = [float(r['n_switches']) for r in rows]
        dr = [float(r['n_dryrun_proxy']) for r in rows]
        of = [int(r['overflow_flag']) for r in rows]
        out_rows.append({
            'source': 'E3v2-DDQN (no forecast)', 'model_or_L': model,
            'n_scenarios': n_present[model], 'n_seeds_or_note': f'{len(SEEDS)} seeds',
            'max_level_mean': round(st.mean(ml), 4),
            'max_level_std': round(st.pstdev(ml), 4),
            'n_switches_mean': round(st.mean(sw), 3),
            'n_dryrun_mean': round(st.mean(dr), 3),
            'overflow_rate': round(st.mean(of), 5),
        })

    for L in sorted(enum_by_L):
        rows = enum_by_L[L]
        ml = [float(r['max_level_m']) for r in rows]
        sw = [float(r['n_switches']) for r in rows]
        dr = [float(r['n_dryrun_proxy']) for r in rows]
        of = [int(r['overflow']) for r in rows]
        out_rows.append({
            'source': 'ENUM-L (persistence forecast)', 'model_or_L': f'L={L}',
            'n_scenarios': len(rows), 'n_seeds_or_note': 'deterministic (no seed)',
            'max_level_mean': round(st.mean(ml), 4),
            'max_level_std': round(st.pstdev(ml), 4),
            'n_switches_mean': round(st.mean(sw), 3),
            'n_dryrun_mean': round(st.mean(dr), 3),
            'overflow_rate': round(st.mean(of), 5),
        })

    from atomic_io import atomic_write

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=out_fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    out_path = E04_DIR / 'planner_vs_policy_w2.csv'
    atomic_write(out_path, _w, newline='')
    print(f'\nwrote {out_path.relative_to(ROOT)}')
    for r in out_rows:
        print(f"  {r['source']:28s} {r['model_or_L']:12s} n={r['n_scenarios']:>4} "
              f"max_level={r['max_level_mean']!s:>8} overflow={r['overflow_rate']!s:>8}")


if __name__ == '__main__':
    main()
