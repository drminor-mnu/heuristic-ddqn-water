#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 11 (new sec 3.5.4, weight sensitivity) -- trade-off scatter of
29 reward-weight combinations: x = mean max water level (m), y = mean
number of pump on/off changes. Non-dominated set (11 of 29) highlighted
with filled markers; the adopted combination [0.40, 0.25, 0.25, 0.10]
(combo c24) labeled; combinations with any overflow (17 of 29) marked
distinctly, since overflow saturates the water-level metric (16 of the
17 hit the simulator's 10 m cap with mean_n_switches = 0 -- an overflow
artifact, not a genuine trade-off point) and would otherwise be
misread as ordinary policies.

No new measurement -- reuses existing E6 outputs unchanged:
  results/E06_weights/grid_results.csv  (29 combos x {GA,PSO} x {seed1,seed2},
    averaged per combo_id here)
  results/summary/E06_nondominated.csv  (11-combo non-dominated set,
    is_adopted flag for c24)

Output: results/summary/figures_final/figure11.png
        results/summary/figure_data/fig11.csv
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
RESULTS = ROOT / 'results'

import sys
sys.path.insert(0, str(SRC))
from plot_style import set_plot_style

SAVE_DPI = 72


def load_grid():
    """Average results/E06_weights/grid_results.csv over {algo, seed} per combo_id."""
    rows = list(csv.DictReader(open(RESULTS / 'E06_weights' / 'grid_results.csv')))
    by_combo = defaultdict(list)
    for r in rows:
        by_combo[r['combo_id']].append(r)

    combos = {}
    for cid, rs in by_combo.items():
        w = (float(rs[0]['w1']), float(rs[0]['w2']), float(rs[0]['w3']), float(rs[0]['w4']))
        overflow_rates = set(r['overflow_rate'] for r in rs)
        mean_level = sum(float(r['mean_max_level_m']) for r in rs) / len(rs)
        mean_switch = sum(float(r['mean_n_switches']) for r in rs) / len(rs)
        combos[cid] = {
            'w': w,
            'overflow_rates_seen': overflow_rates,
            'any_overflow': overflow_rates != {'0.0'},
            'mean_max_level_m': mean_level,
            'mean_n_switches': mean_switch,
            'n_rows': len(rs),
        }
    return combos


def load_nondominated():
    nd = {}
    with open(RESULTS / 'summary' / 'E06_nondominated.csv', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            nd[r['combo_id']] = r['is_adopted'] == 'True'
    return nd


def main():
    combos = load_grid()
    nondominated = load_nondominated()
    assert len(combos) == 29, f'expected 29 combos, found {len(combos)}'
    assert len(nondominated) == 11, f'expected 11 non-dominated combos, found {len(nondominated)}'

    n_overflow = sum(1 for v in combos.values() if v['any_overflow'])
    print(f'{len(combos)} combos total, {n_overflow} with any overflow, '
          f'{len(nondominated)} non-dominated')

    figure_data = RESULTS / 'summary' / 'figure_data'
    figure_data.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS / 'summary' / 'figures_final'
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = figure_data / 'fig11.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['combo_id', 'w1', 'w2', 'w3', 'w4', 'mean_max_level_m',
                    'mean_n_switches', 'any_overflow', 'is_nondominated', 'is_adopted'])
        for cid, v in sorted(combos.items()):
            w.writerow([cid, *v['w'], f"{v['mean_max_level_m']:.4f}",
                        f"{v['mean_n_switches']:.4f}", v['any_overflow'],
                        cid in nondominated, nondominated.get(cid, False)])
    print(f'wrote {csv_path}')

    set_plot_style()
    fig, ax = plt.subplots(figsize=(14, 11))

    # 1. overflow combos (any overflow_rate > 0): distinct marker, drawn first (background)
    ov_x = [v['mean_max_level_m'] for v in combos.values() if v['any_overflow']]
    ov_y = [v['mean_n_switches'] for v in combos.values() if v['any_overflow']]
    ax.scatter(ov_x, ov_y, marker='x', color='dimgray', s=140, linewidths=3,
               label=f'Overflow occurred (n={len(ov_x)})', zorder=2)

    # 2. dominated, no-overflow combos: open circles
    dom_x, dom_y = [], []
    for cid, v in combos.items():
        if not v['any_overflow'] and cid not in nondominated:
            dom_x.append(v['mean_max_level_m'])
            dom_y.append(v['mean_n_switches'])
    ax.scatter(dom_x, dom_y, marker='o', facecolors='none', edgecolors='tab:blue',
               s=90, linewidths=2, label=f'Dominated (n={len(dom_x)})', zorder=3)

    # 3. non-dominated set: filled circles
    nd_x, nd_y, nd_ids = [], [], []
    for cid, is_adopted in nondominated.items():
        v = combos[cid]
        nd_x.append(v['mean_max_level_m'])
        nd_y.append(v['mean_n_switches'])
        nd_ids.append(cid)
    ax.scatter(nd_x, nd_y, marker='o', color='tab:blue', s=140,
               label=f'Non-dominated (n={len(nd_x)})', zorder=4)

    # 4. adopted combination: highlighted + labeled
    adopted_cid = [cid for cid, a in nondominated.items() if a][0]
    av = combos[adopted_cid]
    ax.scatter([av['mean_max_level_m']], [av['mean_n_switches']], marker='*',
               color='tab:red', s=500, edgecolors='black', linewidths=1.5,
               label='Adopted [0.40, 0.25, 0.25, 0.10]', zorder=5)
    ax.annotate('adopted', (av['mean_max_level_m'], av['mean_n_switches']),
                textcoords='offset points', xytext=(15, 15), fontsize=20, fontweight='bold')

    ax.set_xlabel('Maximum water level (m)')
    ax.set_ylabel('Number of pump on/off changes')
    # legend placed outside the axes -- several data points (incl. c28 at
    # (8.39, 298)) occupy the upper-right region where an in-axes legend
    # would otherwise hide them.
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=16, framealpha=0.9)

    fig.savefig(out_dir / 'figure11.png', dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_dir / "figure11.png"}')


if __name__ == '__main__':
    main()
