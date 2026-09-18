#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E6 full-grid analysis: non-dominated set, 4 tradeoff scatter plots,
sensitivity summary. Reads results/E06_weights/grid_results.csv
(produced by run_e06_full_grid.py). No training/search performed here.

Run from src/.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
GRID_CSV = ROOT / 'results' / 'E06_weights' / 'grid_results.csv'
OUT_DIR = ROOT / 'results' / 'summary'
ADOPTED = (0.40, 0.25, 0.25, 0.10)
METRICS = ['mean_max_level_m', 'mean_n_switches', 'mean_pump_use', 'mean_n_dryrun_proxy']
METRIC_LABELS = {'mean_max_level_m': 'max level (m)', 'mean_n_switches': 'switches',
                 'mean_pump_use': 'pump use (intervals)', 'mean_n_dryrun_proxy': 'dry-running proxy'}


def load_and_aggregate():
    rows = list(csv.DictReader(open(GRID_CSV)))
    by_combo = defaultdict(list)
    for r in rows:
        by_combo[r['combo_id']].append(r)

    combos = []
    for combo_id, rs in by_combo.items():
        w = (float(rs[0]['w1']), float(rs[0]['w2']), float(rs[0]['w3']), float(rs[0]['w4']))
        overflow_rate = np.mean([float(r['overflow_rate']) for r in rs])
        entry = {'combo_id': combo_id, 'w1': w[0], 'w2': w[1], 'w3': w[2], 'w4': w[3],
                'overflow_rate': overflow_rate, 'n_runs': len(rs)}
        for m in METRICS:
            entry[m] = np.mean([float(r[m]) for r in rs])
        combos.append(entry)
    return combos


def is_dominated(a, b, metrics):
    """b dominates a if b is <= a on all metrics and < on at least one."""
    le_all = all(b[m] <= a[m] for m in metrics)
    lt_any = any(b[m] < a[m] for m in metrics)
    return le_all and lt_any


def nondominated_set(combos, metrics):
    feasible = [c for c in combos if c['overflow_rate'] == 0.0]
    nd = []
    for a in feasible:
        if not any(is_dominated(a, b, metrics) for b in feasible if b is not a):
            nd.append(a)
    return feasible, nd


def main():
    combos = load_and_aggregate()
    print(f'{len(combos)} weight combinations aggregated (from {GRID_CSV.name})')

    feasible, nd = nondominated_set(combos, METRICS)
    print(f'feasible (overflow_rate==0): {len(feasible)}/{len(combos)}')
    print(f'non-dominated among feasible: {len(nd)}')

    adopted_combo = None
    for c in combos:
        if (round(c['w1'], 2), round(c['w2'], 2), round(c['w3'], 2), round(c['w4'], 2)) == ADOPTED:
            adopted_combo = c
            break
    if adopted_combo:
        in_nd = any(c['combo_id'] == adopted_combo['combo_id'] for c in nd)
        print(f'adopted weight {ADOPTED}: combo_id={adopted_combo["combo_id"]}, '
              f'overflow_rate={adopted_combo["overflow_rate"]}, in non-dominated set: {in_nd}')
    else:
        print(f'adopted weight {ADOPTED} not found among combos (unexpected)')

    # write non-dominated CSV
    fields = ['combo_id', 'w1', 'w2', 'w3', 'w4', 'overflow_rate'] + METRICS
    with open(OUT_DIR / 'E06_nondominated.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields + ['is_adopted'])
        w.writeheader()
        for c in sorted(nd, key=lambda c: c['mean_max_level_m']):
            row = {k: c[k] for k in fields}
            row['is_adopted'] = (adopted_combo is not None and c['combo_id'] == adopted_combo['combo_id'])
            w.writerow(row)
    print(f'wrote results/summary/E06_nondominated.csv ({len(nd)} rows)')

    # 4 tradeoff scatter plots
    pairs = [('mean_max_level_m', 'mean_n_switches'),
             ('mean_max_level_m', 'mean_n_dryrun_proxy'),
             ('mean_pump_use', 'mean_n_dryrun_proxy'),
             ('mean_n_switches', 'mean_pump_use')]
    nd_ids = {c['combo_id'] for c in nd}
    for mx, my in pairs:
        fig, ax = plt.subplots(figsize=(7, 6))
        feas_x = [c[mx] for c in feasible if c['combo_id'] not in nd_ids]
        feas_y = [c[my] for c in feasible if c['combo_id'] not in nd_ids]
        nd_x = [c[mx] for c in feasible if c['combo_id'] in nd_ids]
        nd_y = [c[my] for c in feasible if c['combo_id'] in nd_ids]
        ovf_x = [c[mx] for c in combos if c['overflow_rate'] > 0]
        ovf_y = [c[my] for c in combos if c['overflow_rate'] > 0]
        ax.scatter(ovf_x, ovf_y, c='lightgray', marker='x', label=f'overflow>0 (n={len(ovf_x)})')
        ax.scatter(feas_x, feas_y, c='tab:blue', label=f'feasible, dominated (n={len(feas_x)})')
        ax.scatter(nd_x, nd_y, c='tab:red', marker='*', s=120, label=f'non-dominated (n={len(nd_x)})')
        if adopted_combo:
            ax.scatter([adopted_combo[mx]], [adopted_combo[my]], c='black', marker='D', s=80,
                      label='adopted [0.40,0.25,0.25,0.10]')
        ax.set_xlabel(METRIC_LABELS[mx])
        ax.set_ylabel(METRIC_LABELS[my])
        ax.set_title(f'{METRIC_LABELS[mx]} vs {METRIC_LABELS[my]}')
        ax.legend(fontsize=8)
        fig.tight_layout()
        pair_name = f'{mx.replace("mean_", "")}_{my.replace("mean_", "")}'
        fig.savefig(OUT_DIR / f'E06_tradeoff_{pair_name}.png', dpi=150)
        plt.close(fig)
        print(f'wrote results/summary/E06_tradeoff_{pair_name}.png')

    # sensitivity: |slope| of each metric vs each w_i (simple linear regression), feasible only
    sens_lines = ['# E6 sensitivity summary (|d metric / d w_i|, linear-fit slope over feasible combos)\n']
    sens_lines.append(f'n_feasible={len(feasible)}, n_total_combos={len(combos)}\n')
    sens_lines.append('| metric | vs w1 | vs w2 | vs w3 | vs w4 |')
    sens_lines.append('|---|---|---|---|---|')
    for m in METRICS:
        row = [METRIC_LABELS[m]]
        for wi in ('w1', 'w2', 'w3', 'w4'):
            x = np.array([c[wi] for c in feasible])
            y = np.array([c[m] for c in feasible])
            if len(x) > 2 and x.std() > 0:
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = float('nan')
            row.append(f'{slope:.3f}')
        sens_lines.append('| ' + ' | '.join(row) + ' |')

    (OUT_DIR / 'E06_sensitivity.md').write_text('\n'.join(sens_lines) + '\n')
    print(f'wrote results/summary/E06_sensitivity.md')


if __name__ == '__main__':
    main()
