#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate Figure 7 for Section 3.2 -- REPLACED CONTENT (2026-09-08 user
decision): the original Figure 7 ("test reward" trajectory over training
episodes) cannot be reconstructed -- no periodic test-set evaluation was
logged during E03 v2 training (dqn_from_demon_v1.train() has no such
logic; only a single post-training evaluation exists in test_metrics.csv).
Retraining with checkpoint instrumentation was ruled out ("학습 불필요").

New Figure 7 (user-specified, keeps the figure number so Figures 8-10
don't need renumbering): distribution of scenario-level cumulative test
reward (cum_reward) for the three DDQN models under the identical
four-criterion reward, box plot with per-seed means overlaid as points.

Data: results/E03_seeds/{Regular,GA-guided,PSO-guided}/seed{1..5}/
      test_metrics.csv, cum_reward column, 2,700 scenarios/seed
      (5 seeds x 2,700 = 13,500 rows/model).

Output: results/summary/figures_final/figure7.png
        results/summary/figure_data/fig7.csv
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
RESULTS = ROOT / 'results'

import sys
sys.path.insert(0, str(SRC))
from plot_style import set_plot_style

SAVE_DPI = 72
MODELS = ['Regular', 'GA-guided', 'PSO-guided']
LABELS = {'Regular': 'Regular DDQN', 'GA-guided': 'GA-guided DDQN', 'PSO-guided': 'PSO-guided DDQN'}
SEEDS = [1, 2, 3, 4, 5]


def load_cum_reward(model, seed):
    p = RESULTS / 'E03_seeds' / model / f'seed{seed}' / 'test_metrics.csv'
    with open(p) as f:
        return [float(r['cum_reward']) for r in csv.DictReader(f)]


def main():
    figure_data = RESULTS / 'summary' / 'figure_data'
    figure_data.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS / 'summary' / 'figures_final'
    out_dir.mkdir(parents=True, exist_ok=True)

    per_model_seed = {}  # model -> {seed: [values]}
    for m in MODELS:
        per_model_seed[m] = {s: load_cum_reward(m, s) for s in SEEDS}
        n = sum(len(v) for v in per_model_seed[m].values())
        print(f'{m}: {n} scenario-level rows across {len(SEEDS)} seeds')

    # CSV: long format, one row per scenario
    csv_path = figure_data / 'fig7.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['model', 'seed', 'cum_reward'])
        for m in MODELS:
            for s in SEEDS:
                for v in per_model_seed[m][s]:
                    w.writerow([m, s, f'{v:.6f}'])
    print(f'wrote {csv_path}')

    # per-seed means, for the overlaid markers and for a summary CSV
    seed_means_path = figure_data / 'fig7_seed_means.csv'
    with open(seed_means_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['model', 'seed', 'mean_cum_reward'])
        for m in MODELS:
            for s in SEEDS:
                w.writerow([m, s, f'{np.mean(per_model_seed[m][s]):.6f}'])
    print(f'wrote {seed_means_path}')

    # plot
    set_plot_style()
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 12)

    all_values = [np.concatenate([per_model_seed[m][s] for s in SEEDS]) for m in MODELS]
    bp = ax.boxplot(all_values, labels=[LABELS[m] for m in MODELS], showfliers=False,
                     patch_artist=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)

    # overlay per-seed means as points, jittered horizontally
    rng = np.random.default_rng(0)
    for i, m in enumerate(MODELS):
        means = [np.mean(per_model_seed[m][s]) for s in SEEDS]
        x = np.full(len(means), i + 1, dtype=float) + rng.uniform(-0.08, 0.08, len(means))
        ax.scatter(x, means, color='black', zorder=5, s=60, marker='D',
                   label='Per-seed mean' if i == 0 else None)

    ax.set_ylabel('Cumulative test reward')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_dir / 'figure7.png', dpi=SAVE_DPI)
    plt.close(fig)
    print(f'wrote {out_dir / "figure7.png"}')

    for m in MODELS:
        all_vals = np.concatenate([per_model_seed[m][s] for s in SEEDS])
        seed_means = [np.mean(per_model_seed[m][s]) for s in SEEDS]
        print(f'{m}: overall mean={all_vals.mean():.4f}, sd={all_vals.std(ddof=1):.4f}, '
              f'seed means={[f"{v:.4f}" for v in seed_means]}')


if __name__ == '__main__':
    main()
