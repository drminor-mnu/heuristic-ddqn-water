#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5 item 3-2 (2026-08-31/09-01): A3 vs A0 early-training-curve comparison,
isolated from A1 -- 5-seed final (originally computed ad hoc for 3 seeds).

Two facts computed per seed, both on the 100-episode rolling-mean train
reward curve (same ROLL=100 convention as run_e05_plot_curves.py):

  1. mean(A3 rolling reward) - mean(A0 rolling reward) over episodes
     [0, 1300) -- the "early-region gap".
  2. episode index at which each of A0's own curve and A3's curve first
     reaches A0's own final level -- defined as the mean of A0's own last
     200 raw episode rewards (matches the original 3-seed ad hoc
     computation this script reproduces/extends to 5 seeds) -- as
     "a0_reach_episode" and "a3_reach_episode" -- and their ratio.

Caveat (unchanged, carried into docs/E05_REPORT.md Sec 7-2 verbatim): this
rolling-mean reward includes the guide's own immediate reward during
guided steps, so it is not direct evidence of learned-policy performance
in isolation from the guide.

Run from src/, after A0 5-seed training and A3 (E3 v2 GA-guided 5-seed,
reused) both exist.
"""
import csv
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E05_DIR = ROOT / 'results' / 'E05_ablation'
E03_DIR = ROOT / 'results' / 'E03_seeds'
SEEDS = [1, 2, 3, 4, 5]
ROLL = 100
EARLY_CUTOFF = 1300


def load_rewards(cond, seed):
    if cond == 'A3':
        path = E03_DIR / 'GA-guided' / f'seed{seed}' / 'train_log.csv'
    else:
        path = E05_DIR / cond / f'seed{seed}' / 'train_log.csv'
    with open(path) as f:
        return [float(r['train_reward']) for r in csv.DictReader(f)]


def rolling_mean(x, window):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='valid')


def first_reach(curve, target):
    """First index (in the rolling-mean curve) at which curve >= target,
    or None if never reached."""
    idx = np.argmax(curve >= target)
    if curve[idx] >= target:
        return int(idx)
    return None


def main():
    rows = []
    for seed in SEEDS:
        a0 = rolling_mean(load_rewards('A0', seed), ROLL)
        a3 = rolling_mean(load_rewards('A3', seed), ROLL)
        n = min(len(a0), len(a3), EARLY_CUTOFF)
        early_gap = float(np.mean(a3[:n]) - np.mean(a0[:n]))

        a0_raw = load_rewards('A0', seed)
        a0_final = float(np.mean(a0_raw[-200:]))  # matches the original
        # 3-seed ad hoc computation's "A0's own final level" definition
        a0_reach = first_reach(a0, a0_final)
        a3_reach = first_reach(a3, a0_final)
        ratio = (a0_reach / a3_reach) if (a0_reach and a3_reach and a3_reach > 0) else None

        row = {
            'seed': seed, 'n_episodes_a0': len(load_rewards('A0', seed)),
            'n_episodes_a3': len(load_rewards('A3', seed)),
            'early_region_episodes': n,
            'early_gap_a3_minus_a0': round(early_gap, 4),
            'a0_final_last200_mean': round(a0_final, 4),
            'a0_reach_episode': a0_reach, 'a3_reach_episode': a3_reach,
            'ratio_a0_over_a3': round(ratio, 3) if ratio else '',
        }
        rows.append(row)
        print(row)

    out_dir = ROOT / 'results' / 'summary'
    from atomic_io import atomic_write

    fields = list(rows[0].keys())

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    atomic_write(out_dir / 'E05_early_curve.csv', _w, newline='')
    print(f'\nwrote results/summary/E05_early_curve.csv ({len(rows)} rows)')

    gaps = [r['early_gap_a3_minus_a0'] for r in rows]
    ratios = [r['ratio_a0_over_a3'] for r in rows if r['ratio_a0_over_a3'] != '']
    print(f'\nmean early_gap (A3-A0, episodes 0-{EARLY_CUTOFF}) over {len(gaps)} seeds: '
          f'{np.mean(gaps):.4f} (range {min(gaps):.4f} to {max(gaps):.4f})')
    if ratios:
        print(f'ratio_a0_over_a3 over {len(ratios)} seeds with both reach-episodes found: '
              f'{ratios} (min {min(ratios):.2f}x, max {max(ratios):.2f}x)')


if __name__ == '__main__':
    main()
