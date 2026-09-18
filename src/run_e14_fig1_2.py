#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Section 3.1 regeneration: Figure 1 (training-loss trajectories) and
Figure 2 (training-reward trajectories) of Regular DDQN(2-criterion,
w=[0.5,0.5,0,0]) vs GA-guided DDQN(4-criterion, new weights
[0.40,0.25,0.25,0.10]), "under different reward structures"
(manuscript captions, docs/response/SECTION_3_1_REGENERATION.md Sec 3).

Same rolling-mean-over-5-seeds pattern as run_e05_plot_curves.py (reused
conceptually, not imported -- that script is E5-specific).

Data: results/E14_two_criteria/Regular/seed{1..5}/train_log.csv (NOT
produced yet as of this script's creation -- run_e14_two_criteria.py must
finish first) and results/E03_seeds/GA-guided/seed{1..5}/train_log.csv
(existing, reused).

Run from src/, only after E14 has produced all 5 Regular seeds (or with
--regular-source e3 for a format smoke test, output written to
--out-dir, not the real summary path).
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
RESULTS = ROOT / 'results'
SEEDS = [1, 2, 3, 4, 5]
ROLL = 100


def load_column(dir_path: Path, seed: int, column: str):
    p = dir_path / f'seed{seed}' / 'train_log.csv'
    with open(p) as f:
        return [float(r[column]) for r in csv.DictReader(f)]


def rolling_mean(x, window):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='valid')


def plot_metric(regular_dir, ga_dir, column, ylabel, title, out_path, regular_label):
    fig, ax = plt.subplots(figsize=(14, 8))
    for label, dir_path, color in ((regular_label, regular_dir, 'tab:gray'),
                                    ('GA-guided(4-criterion)', ga_dir, 'tab:blue')):
        per_seed = [load_column(dir_path, s, column) for s in SEEDS]
        min_len = min(len(r) for r in per_seed)
        arr = np.array([r[:min_len] for r in per_seed])
        mean_curve = arr.mean(axis=0)
        smoothed = rolling_mean(mean_curve, ROLL)
        ax.plot(smoothed, label=f'{label} (n={len(per_seed)} seeds)', color=color, linewidth=2)
        print(f'{label}: {len(per_seed)} seeds, {min_len} episodes each, '
              f'final rolling-mean {column} = {smoothed[-1]:.4f}')
    ax.set_xlabel('training episode')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'wrote {out_path}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--regular-source', choices=['e14', 'e3'], default='e14',
                   help='e14 = results/E14_two_criteria/Regular (real 2-criterion run); '
                        'e3 = results/E03_seeds/Regular (4-criterion, SMOKE TEST ONLY)')
    p.add_argument('--out-dir', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.regular_source == 'e14':
        regular_dir = RESULTS / 'E14_two_criteria' / 'Regular'
        regular_label = 'Regular(2-criterion)'
    else:
        regular_dir = RESULTS / 'E03_seeds' / 'Regular'
        regular_label = 'Regular(SMOKE-TEST-4-criterion-stand-in)'
        print('*** SMOKE TEST MODE -- not a valid Figure 1/2 regeneration. ***')
    ga_dir = RESULTS / 'E03_seeds' / 'GA-guided'

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS / 'summary'
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(regular_dir, ga_dir, 'train_loss', f'train loss ({ROLL}-episode rolling mean)',
               'Figure 1 (regenerated): training-loss trajectories, different reward structures',
               out_dir / 'E14_figure1_loss.png', regular_label)
    plot_metric(regular_dir, ga_dir, 'train_reward', f'train reward ({ROLL}-episode rolling mean)',
               'Figure 2 (regenerated): training-reward trajectories, different reward structures',
               out_dir / 'E14_figure2_reward.png', regular_label)


if __name__ == '__main__':
    main()
