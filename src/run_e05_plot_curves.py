#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5 stage-1: learning-curve comparison (A0 / A1 / A3), A1 vs A3 is the key
panel (Q3 -- where do their curves diverge, if at all).

Reads train_log.csv (episode, train_loss, train_reward) written by
run_e05_ablation.py for A0/A1, and results/E03_seeds/GA-guided/seed{n}/
train_log.csv for A3 (reused, not re-run). Averages across the 3 seeds per
condition, plots reward per episode (rolling mean) for all three.

Run from src/, after A0/A1 stage-1 training completes.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E05_DIR = ROOT / 'results' / 'E05_ablation'
E03_DIR = ROOT / 'results' / 'E03_seeds'
SEEDS = [1, 2, 3, 4, 5]
ROLL = 100  # rolling-mean window over episodes


def load_rewards(cond: str):
    per_seed = []
    for seed in SEEDS:
        if cond == 'A3':
            path = E03_DIR / 'GA-guided' / f'seed{seed}' / 'train_log.csv'
        else:
            path = E05_DIR / cond / f'seed{seed}' / 'train_log.csv'
        if not path.exists():
            continue
        with open(path) as f:
            rewards = [float(r['train_reward']) for r in csv.DictReader(f)]
        per_seed.append(rewards)
    return per_seed


def rolling_mean(x, window):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='valid')


def main():
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = {'A0': 'tab:gray', 'A1': 'tab:orange', 'A3': 'tab:blue'}
    any_data = False
    for cond in ('A0', 'A1', 'A3'):
        per_seed = load_rewards(cond)
        if not per_seed:
            print(f'{cond}: no train_log.csv found, skipping')
            continue
        any_data = True
        min_len = min(len(r) for r in per_seed)
        arr = np.array([r[:min_len] for r in per_seed])
        mean_curve = arr.mean(axis=0)
        smoothed = rolling_mean(mean_curve, ROLL)
        ax.plot(smoothed, label=f'{cond} (n={len(per_seed)} seeds)',
               color=colors[cond], linewidth=2)
        print(f'{cond}: {len(per_seed)} seeds, {min_len} episodes each, '
              f'final rolling-mean reward = {smoothed[-1]:.2f}')

    ax.set_xlabel('training episode')
    ax.set_ylabel(f'train reward ({ROLL}-episode rolling mean)')
    ax.set_title('E5 stage 1: A0 vs A1 (random guide) vs A3 (GA-guided)')
    ax.legend()
    fig.tight_layout()

    out_dir = ROOT / 'results' / 'summary'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'E05_learning_curves.png'
    if any_data:
        fig.savefig(out_path, dpi=150)
        print(f'\nwrote {out_path.relative_to(ROOT)}')
    else:
        print('\nno data available yet -- figure not written.')


if __name__ == '__main__':
    main()
