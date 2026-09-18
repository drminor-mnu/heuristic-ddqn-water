#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E9 item 3 (R1-4): locate the w1 interval where rho(cum_reward,
max_level_m) crosses zero, and check whether F/U/D also vary with w1.
Reuses results/E06_curve/*_scenario_metrics.csv (existing GA/PSO-only
w1-sweep data, no new runs) and the manual Spearman implementation from
run_e09_correlation.py (re-imported, not duplicated).

Run from src/.
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_e09_correlation import spearman

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E06_CURVE_DIR = ROOT / 'results' / 'E06_curve'
ADOPTED_W1 = 0.40


def main():
    by_w1 = defaultdict(list)
    for p in sorted(E06_CURVE_DIR.glob('*_scenario_metrics.csv')):
        m = re.match(r'E06C_w1_([\d.]+)_([A-Z]+)_s(\d+)_scenario_metrics\.csv', p.name)
        if not m:
            continue
        w1 = float(m.group(1))
        with open(p) as f:
            by_w1[w1].extend(list(csv.DictReader(f)))

    w1s = sorted(by_w1)
    metrics = [('max_level_m', 'H (max_level_m)'), ('n_switches', 'F (n_switches)'),
               ('n_dryrun_proxy', 'D (n_dryrun_proxy)')]
    results = {m[0]: [] for m in metrics}
    results['U'] = []
    rows_out = []
    for w1 in w1s:
        rows = by_w1[w1]
        cum_reward = [float(r['cum_reward']) for r in rows]
        for key, label in metrics:
            vals = [float(r[key]) for r in rows]
            rho, lo, hi, n = spearman(cum_reward, vals)
            results[key].append(rho)
            rows_out.append({'w1': w1, 'metric': label, 'n': n, 'rho': rho,
                             'ci_lo': lo, 'ci_hi': hi})
        u_vals = [float(r['n_intervals_100']) + float(r['n_intervals_170']) for r in rows]
        rho, lo, hi, n = spearman(cum_reward, u_vals)
        results['U'].append(rho)
        rows_out.append({'w1': w1, 'metric': 'U (n_intervals_100+170)', 'n': n, 'rho': rho,
                         'ci_lo': lo, 'ci_hi': hi})

    # zero-crossing interval for H
    h_rho = results['max_level_m']
    crossing = None
    for i in range(len(w1s) - 1):
        a, b = h_rho[i], h_rho[i + 1]
        if a == a and b == b and a < 0 <= b:  # a==a filters nan
            frac = -a / (b - a)
            crossing = w1s[i] + frac * (w1s[i + 1] - w1s[i])
            print(f'H rho crosses zero between w1={w1s[i]} (rho={a:.4f}) and '
                  f'w1={w1s[i+1]} (rho={b:.4f}) -- linear-interpolated crossing point: w1~{crossing:.4f}')
    print(f'adopted w1={ADOPTED_W1}: ', end='')
    if ADOPTED_W1 in w1s:
        idx = w1s.index(ADOPTED_W1)
        print(f'rho(H)={h_rho[idx]:.4f} (measured point)')
    else:
        print('not a measured grid point')

    for key, label in metrics:
        print(f'\n{label} vs w1:')
        for w1, rho in zip(w1s, results[key]):
            print(f'  w1={w1}: rho={rho}' if rho == rho else f'  w1={w1}: rho=nan')
    print('\nU (n_intervals_100+170) vs w1:')
    for w1, rho in zip(w1s, results['U']):
        print(f'  w1={w1}: rho={rho}' if rho == rho else f'  w1={w1}: rho=nan')

    # plot
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {'max_level_m': 'tab:red', 'n_switches': 'tab:blue',
             'n_dryrun_proxy': 'tab:green', 'U': 'tab:orange'}
    labels = {'max_level_m': 'H (max_level_m)', 'n_switches': 'F (n_switches)',
             'n_dryrun_proxy': 'D (n_dryrun_proxy)', 'U': 'U (pump use)'}
    for key in ('max_level_m', 'n_switches', 'n_dryrun_proxy', 'U'):
        ax.plot(w1s, results[key], marker='o', label=labels[key], color=colors[key])
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axvline(ADOPTED_W1, color='black', linewidth=0.8, linestyle=':', label=f'adopted w1={ADOPTED_W1}')
    if crossing:
        ax.axvline(crossing, color='tab:red', linewidth=0.8, linestyle=':', alpha=0.5,
                  label=f'H zero-crossing w1~{crossing:.3f}')
    ax.set_xlabel('w1')
    ax.set_ylabel('Spearman rho(cum_reward, metric)')
    ax.set_title('rho(cum_reward, objective) vs w1 (GA/PSO-only, w2=w3=0.30/0.25 grid, no DQN training)')
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_dir = ROOT / 'results' / 'summary'
    fig.savefig(out_dir / 'E09_rho_by_w1.png', dpi=150)
    print(f'\nwrote results/summary/E09_rho_by_w1.png')

    from atomic_io import atomic_write
    fields = ['w1', 'metric', 'n', 'rho', 'ci_lo', 'ci_hi']

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=fields)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    atomic_write(out_dir / 'E09_rho_by_w1_all_metrics.csv', _w, newline='')
    print(f'wrote results/summary/E09_rho_by_w1_all_metrics.csv ({len(rows_out)} rows)')


if __name__ == '__main__':
    main()
