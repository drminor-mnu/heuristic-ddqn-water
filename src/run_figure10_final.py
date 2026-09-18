#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate Figure 10 for Section 3.4 (R4-12 -- and its 2026-09-08
recurrence in the regenerated draft, see
results/summary/TABLE5_FIGURE10_CONSISTENCY.md). Option C (user decision,
2026-09-09): unify Table 5 and Figure 10 on the same per-decision metric
-- (a) forward pass/heuristic decision + (b) preprocessing + (c)
environment-step overhead, EXCLUDING the once-per-episode SWMM cost --
for all five models, using the new 3-repeat measurement in
results/summary/E10_unified_c_summary.csv (src/run_e10_unified_c.py).
No new measurement performed by this script; it only plots.

Output: results/summary/figures_final/figure10.png
        results/summary/figure_data/fig10.csv
"""
import csv
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
MODELS = ['Regular', 'GA-guided', 'PSO-guided', 'GA-only', 'PSO-only']
COLORS = {'Regular': 'tab:blue', 'GA-guided': 'tab:orange', 'PSO-guided': 'tab:green',
          'GA-only': 'tab:red', 'PSO-only': 'tab:purple'}


def load_summary():
    src = RESULTS / 'summary' / 'E10_unified_c_summary.csv'
    data = {}
    with open(src, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            data[r['condition']] = (float(r['mean_ms_per_decision']), float(r['sd_ms_per_decision']))
    return data, src


def main():
    figure_data = RESULTS / 'summary' / 'figure_data'
    figure_data.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS / 'summary' / 'figures_final'
    out_dir.mkdir(parents=True, exist_ok=True)

    data, src = load_summary()

    csv_path = figure_data / 'fig10.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['model', 'mean_ms_per_decision_c', 'sd_ms_per_decision_c', 'n_repeats', 'source'])
        for m in MODELS:
            mean, sd = data[m]
            w.writerow([m, f'{mean:.4f}', f'{sd:.4f}', 3, str(src)])
    print(f'wrote {csv_path}')

    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    x = list(range(len(MODELS)))
    means = [data[m][0] for m in MODELS]
    sds = [data[m][1] for m in MODELS]
    colors = [COLORS[m] for m in MODELS]
    ax1.bar(x, means, yerr=sds, capsize=6, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(MODELS, rotation=20, ha='right')
    ax1.set_ylabel('Time per decision (ms)')
    ax1.set_title('All five models')

    ddqn_models = ['Regular', 'GA-guided', 'PSO-guided']
    x2 = list(range(len(ddqn_models)))
    means2 = [data[m][0] for m in ddqn_models]
    sds2 = [data[m][1] for m in ddqn_models]
    colors2 = [COLORS[m] for m in ddqn_models]
    ax2.bar(x2, means2, yerr=sds2, capsize=6, color=colors2)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(ddqn_models)
    ax2.set_ylabel('Time per decision (ms)')
    ax2.set_ylim(0, 0.3)
    ax2.set_title('DDQN models (enlarged)')

    fig.tight_layout()
    fig.savefig(out_dir / 'figure10.png', dpi=SAVE_DPI)
    plt.close(fig)
    print(f'wrote {out_dir / "figure10.png"}')
    for m in MODELS:
        print(f'{m}: {data[m][0]:.4f} +- {data[m][1]:.4f} ms/decision')


if __name__ == '__main__':
    main()
