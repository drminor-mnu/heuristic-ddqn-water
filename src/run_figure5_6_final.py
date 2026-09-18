#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate Figure 5 (training loss) and Figure 6 (training reward) for
Section 3.2 -- Regular vs GA-guided vs PSO-guided DDQN, all under the
identical four-criterion reward (new weights [0.40,0.25,0.25,0.10]).

Extends the style of run_figure1_2_final.py (set_plot_style(), single
overlaid axes, 5-seed mean then 100-episode rolling mean, no title) to a
third model (PSO-guided), per R1-7 (reviewer noted PSO-guided was missing
from Figures 1/5/6 -- added here per user decision; Figure 1 in Section
3.1 explicitly keeps only Regular/GA-guided, see SECTION_3_2_PACKAGE.md).

Data: results/E03_seeds/{Regular,GA-guided,PSO-guided}/seed{1..5}/train_log.csv
(all four-criterion, new weights, same source used for Table 11-16).

Output: results/summary/figures_final/figure5.png, figure6.png
        results/summary/figure_data/fig5.csv, fig6.csv
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
ROLL = 100
MODELS = ['Regular', 'GA-guided', 'PSO-guided']
LABELS = {'Regular': 'Regular DDQN', 'GA-guided': 'GA-guided DDQN', 'PSO-guided': 'PSO-guided DDQN'}


def load_column(model, seed, column):
    p = RESULTS / 'E03_seeds' / model / f'seed{seed}' / 'train_log.csv'
    with open(p) as f:
        return [float(r[column]) for r in csv.DictReader(f)]


def rolling_mean(x, window):
    import numpy as np
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='valid')


def build_series(column):
    import numpy as np
    result = {}
    min_len = None
    for m in MODELS:
        per_seed = [load_column(m, s, column) for s in [1, 2, 3, 4, 5]]
        ml = min(len(r) for r in per_seed)
        min_len = ml if min_len is None else min(min_len, ml)
        result[m] = per_seed
    means = {}
    for m in MODELS:
        arr = np.array([r[:min_len] for r in result[m]])
        means[m] = arr.mean(axis=0)
    smoothed = {m: rolling_mean(means[m], ROLL) for m in MODELS}
    return means, smoothed, min_len


def write_csv(path, means, smoothed, min_len, ylabel):
    smooth_len = len(smoothed['Regular'])
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        header = ['episode']
        for m in MODELS:
            header += [f'{m}_{ylabel}_mean_5seed_raw', f'{m}_{ylabel}_mean_5seed_smoothed_roll100']
        w.writerow(header)
        for i in range(min_len):
            row = [i]
            for m in MODELS:
                row.append(f'{means[m][i]:.6f}')
                row.append(f'{smoothed[m][i]:.6f}' if i < smooth_len else '')
            w.writerow(row)


def plot_curve(smoothed, ylabel, out_path):
    set_plot_style()
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 12)
    for m in MODELS:
        ax.plot(smoothed[m], label=LABELS[m])
    ax.set_xlabel('Episode')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f'wrote {out_path}: final ' + ', '.join(f'{m}={smoothed[m][-1]:.4f}' for m in MODELS))


def main():
    figure_data = RESULTS / 'summary' / 'figure_data'
    figure_data.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS / 'summary' / 'figures_final'
    out_dir.mkdir(parents=True, exist_ok=True)

    means, smoothed, min_len = build_series('train_loss')
    write_csv(figure_data / 'fig5.csv', means, smoothed, min_len, 'loss')
    plot_curve(smoothed, 'Loss', out_dir / 'figure5.png')

    means, smoothed, min_len = build_series('train_reward')
    write_csv(figure_data / 'fig6.csv', means, smoothed, min_len, 'reward')
    plot_curve(smoothed, 'Reward', out_dir / 'figure6.png')


if __name__ == '__main__':
    main()
