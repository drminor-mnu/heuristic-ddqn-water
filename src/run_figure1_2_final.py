#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate Figure 1 (training loss) and Figure 2 (training reward).

No original function reproduces the exact style of
results_org/images/loss_v2.png / rewards_training.png (single overlaid
axes, both series on one plot) -- see
results/summary/FIGURE_DATA_README.md Sec.4 for the provenance search
that established this. This script is newly written to match that
style: set_plot_style() (src/plot_style.py, same styling used by
graph_draw.py/dqn_from_demon_v1.py/perform_evaluate.py) applied, both
series overlaid on one axes (not separate subplots), legend text
"Regular DDQN" / "GA-guided DDQN", xlabel "Episode", ylabel "Loss" /
"Reward" (labels confirmed against the two original images directly).
2026-09-05: xlabel was briefly changed to "Scenario" per a user
instruction, then reverted the same day to "Episode" per a follow-up
instruction -- "Episode" is the final xlabel.
No plot title (2026-09-05: user instruction -- removed, matching the
original two images which also have no title).

Data: results/summary/figure_data/fig1_loss.csv, fig2_reward.csv --
the *_smoothed_roll100 columns (5-seed mean, then 100-episode rolling
mean; see FIGURE_DATA_README.md Sec.1 for how those were produced).

Output: results/summary/figures_final/figure1.png, figure2.png.
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


def load_smoothed(csv_path, reg_col, ga_col):
    episodes, reg, ga = [], [], []
    with open(csv_path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rv, gv = row[reg_col], row[ga_col]
            if rv == '' or gv == '':
                break  # smoothed columns end before the raw columns do
            episodes.append(int(row['episode']))
            reg.append(float(rv))
            ga.append(float(gv))
    return episodes, reg, ga


def plot_curve(episodes, reg, ga, ylabel, out_path):
    set_plot_style()
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 12)
    ax.plot(episodes, reg, label='Regular DDQN')
    ax.plot(episodes, ga, label='GA-guided DDQN')
    ax.set_xlabel('Episode')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f'wrote {out_path}: {len(episodes)} points, '
          f'final Regular={reg[-1]:.4f}, GA-guided={ga[-1]:.4f}')


def main():
    figure_data = RESULTS / 'summary' / 'figure_data'
    out_dir = RESULTS / 'summary' / 'figures_final'
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes, reg_loss, ga_loss = load_smoothed(
        figure_data / 'fig1_loss.csv',
        'regular_loss_mean_5seed_smoothed_roll100',
        'ga_guided_loss_mean_5seed_smoothed_roll100')
    plot_curve(episodes, reg_loss, ga_loss, 'Loss',
               out_dir / 'figure1.png')

    episodes, reg_reward, ga_reward = load_smoothed(
        figure_data / 'fig2_reward.csv',
        'regular_reward_mean_5seed_smoothed_roll100',
        'ga_guided_reward_mean_5seed_smoothed_roll100')
    plot_curve(episodes, reg_reward, ga_reward, 'Reward',
               out_dir / 'figure2.png')


if __name__ == '__main__':
    main()
