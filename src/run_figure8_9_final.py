#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate Figure 8 (training reward by rainfall group) and Figure 9
(paired test-reward scatter) for Section 3.2's limited-data (E12,
train_n=160) results. Regular vs GA-guided only (E12 was not run for
PSO-guided -- see docs/E12_REPORT.md "실행 범위와 축소 사유").

Figure 8 rainfall-group definition: return period (10/20/30/50/80/100
year), the coarsest categorical axis already used throughout this paper
(e.g. Table 6-15's "Return periods" columns). No original Figure 8 image
or script survives in this repo to confirm the author's original grouping
(docs/response/SECTION_3_2_PACKAGE.md records this as a documented
assumption, not a reproduction of the original method).

Per-episode -> scenario mapping: train_log.csv only stores
(episode, train_loss, train_reward), not which of the 160 training
scenarios was used in that episode. Tracing src/dqn_from_demon_v1.py's
train() shows the SAME data_paths.load_fixed_split() train_inps[:160]
list is shuffled exactly once per run, immediately after
random.seed(seed) in run_e12_datasize.py's _run_one() -- with no other
random.* module calls in between (verified: data_paths.load_fixed_split()
and perform_evaluate.evaluate_dpn_gru()/_evaluate_dpn_guided() do not
touch the `random` module before calling train()). So the exact
per-episode scenario order is reproducible by replaying
random.seed(seed); random.shuffle(train_inps[:160].copy()) -- this
script does exactly that (not a guess).

Figure 9: paired scenario-level test cum_reward, Regular (x) vs
GA-guided (y), diagonal line, seeds 1-3, 2,700 scenarios/seed = 8,100
points. Above/below diagonal counts annotated in-plot per user request.

Output: results/summary/figures_final/figure8.png, figure9.png
        results/summary/figure_data/fig8.csv, fig9.csv
"""
import csv
import random
import re
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
import data_paths

SAVE_DPI = 72
SEEDS = [1, 2, 3]
TRAIN_N = 160
RETURN_PERIODS = [10, 20, 30, 50, 80, 100]


def reconstruct_scenario_order(seed):
    """Replay the exact random.seed(seed) + random.shuffle() sequence used
    in run_e12_datasize.py's _run_one() to recover per-episode scenario
    identity. See module docstring for why this is exact, not a guess."""
    train_inps_full, _valid, _test = data_paths.load_fixed_split()
    train_inps_160 = train_inps_full[:TRAIN_N]
    random.seed(seed)
    ti = train_inps_160.copy()
    random.shuffle(ti)
    return ti  # ti[i] == scenario used at episode i (0-indexed)


def return_period_of(path):
    m = re.search(r'/(\d+)year/', path)
    return int(m.group(1))


# ---------------- Figure 8 ----------------

def build_figure8():
    figure_data = RESULTS / 'summary' / 'figure_data'
    out_dir = RESULTS / 'summary' / 'figures_final'

    # per (model, seed): list of (episode_idx, return_period, train_reward)
    rows = []
    group_sums = {m: {rp: [] for rp in RETURN_PERIODS} for m in ['Regular', 'GA-guided']}

    for seed in SEEDS:
        order = reconstruct_scenario_order(seed)
        rps = [return_period_of(p) for p in order]
        for model in ['Regular', 'GA-guided']:
            p = RESULTS / 'E12_datasize' / model / f'n{TRAIN_N}' / f'seed{seed}' / 'train_log.csv'
            with open(p) as f:
                train_rewards = [float(r['train_reward']) for r in csv.DictReader(f)]
            assert len(train_rewards) == TRAIN_N, f'{model} seed{seed}: {len(train_rewards)} episodes, expected {TRAIN_N}'
            for i, (rp, tr) in enumerate(zip(rps, train_rewards)):
                rows.append([model, seed, i, rp, f'{tr:.6f}'])
                group_sums[model][rp].append(tr)

    csv_path = figure_data / 'fig8.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['model', 'seed', 'episode', 'return_period_year', 'train_reward'])
        w.writerows(rows)
    print(f'wrote {csv_path}: {len(rows)} rows')

    # group means (pooled across seeds and the scenarios in that group)
    group_means = {m: {rp: float(np.mean(group_sums[m][rp])) for rp in RETURN_PERIODS} for m in ['Regular', 'GA-guided']}
    group_sd = {m: {rp: float(np.std(group_sums[m][rp], ddof=1)) if len(group_sums[m][rp]) > 1 else 0.0
                     for rp in RETURN_PERIODS} for m in ['Regular', 'GA-guided']}
    group_n = {m: {rp: len(group_sums[m][rp]) for rp in RETURN_PERIODS} for m in ['Regular', 'GA-guided']}

    summary_path = figure_data / 'fig8_group_summary.csv'
    with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['return_period_year', 'Regular_mean', 'Regular_sd', 'Regular_n',
                    'GA-guided_mean', 'GA-guided_sd', 'GA-guided_n', 'GA_minus_Regular'])
        higher_count = 0
        for rp in RETURN_PERIODS:
            gmean = group_means['GA-guided'][rp]
            rmean = group_means['Regular'][rp]
            diff = gmean - rmean
            if diff > 0:
                higher_count += 1
            w.writerow([rp, f'{rmean:.4f}', f'{group_sd["Regular"][rp]:.4f}', group_n['Regular'][rp],
                        f'{gmean:.4f}', f'{group_sd["GA-guided"][rp]:.4f}', group_n['GA-guided'][rp],
                        f'{diff:.4f}'])
    print(f'wrote {summary_path}')
    print(f'GA-guided higher than Regular in {higher_count}/{len(RETURN_PERIODS)} return-period groups')
    for rp in RETURN_PERIODS:
        print(f'  {rp}yr: Regular={group_means["Regular"][rp]:.3f} (n={group_n["Regular"][rp]}), '
              f'GA-guided={group_means["GA-guided"][rp]:.3f} (n={group_n["GA-guided"][rp]}), '
              f'diff={group_means["GA-guided"][rp]-group_means["Regular"][rp]:+.3f}')

    # plot: grouped bar, mean +- sd
    set_plot_style()
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 10)
    x = np.arange(len(RETURN_PERIODS))
    width = 0.35
    reg_means = [group_means['Regular'][rp] for rp in RETURN_PERIODS]
    reg_sds = [group_sd['Regular'][rp] for rp in RETURN_PERIODS]
    ga_means = [group_means['GA-guided'][rp] for rp in RETURN_PERIODS]
    ga_sds = [group_sd['GA-guided'][rp] for rp in RETURN_PERIODS]
    ax.bar(x - width / 2, reg_means, width, yerr=reg_sds, capsize=4, label='Regular DDQN', color='tab:blue')
    ax.bar(x + width / 2, ga_means, width, yerr=ga_sds, capsize=4, label='GA-guided DDQN', color='tab:orange')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{rp}yr' for rp in RETURN_PERIODS])
    ax.set_xlabel('Return period')
    ax.set_ylabel('Mean training reward')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_dir / 'figure8.png', dpi=SAVE_DPI)
    plt.close(fig)
    print(f'wrote {out_dir / "figure8.png"}')
    return higher_count, len(RETURN_PERIODS)


# ---------------- Figure 9 ----------------

def build_figure9():
    figure_data = RESULTS / 'summary' / 'figure_data'
    out_dir = RESULTS / 'summary' / 'figures_final'

    reg_vals, ga_vals, seeds_col, sid_col = [], [], [], []
    above = below = equal = 0
    for seed in SEEDS:
        reg = {}
        with open(RESULTS / 'E12_datasize' / 'Regular' / f'n{TRAIN_N}' / f'seed{seed}' / 'test_metrics.csv') as f:
            for r in csv.DictReader(f):
                reg[r['scenario_id']] = float(r['cum_reward'])
        ga = {}
        with open(RESULTS / 'E12_datasize' / 'GA-guided' / f'n{TRAIN_N}' / f'seed{seed}' / 'test_metrics.csv') as f:
            for r in csv.DictReader(f):
                ga[r['scenario_id']] = float(r['cum_reward'])
        common = sorted(set(reg) & set(ga))
        for sid in common:
            rv, gv = reg[sid], ga[sid]
            reg_vals.append(rv)
            ga_vals.append(gv)
            seeds_col.append(seed)
            sid_col.append(sid)
            d = gv - rv
            if d > 0:
                above += 1
            elif d < 0:
                below += 1
            else:
                equal += 1

    total = len(reg_vals)
    csv_path = figure_data / 'fig9.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'scenario_id', 'Regular_cum_reward', 'GA-guided_cum_reward', 'diff_GA_minus_Regular'])
        for s, sid, rv, gv in zip(seeds_col, sid_col, reg_vals, ga_vals):
            w.writerow([s, sid, f'{rv:.6f}', f'{gv:.6f}', f'{gv - rv:.6f}'])
    print(f'wrote {csv_path}: {total} rows')
    print(f'above diagonal (GA>Regular): {above} ({100*above/total:.1f}%), '
          f'below diagonal (GA<Regular): {below} ({100*below/total:.1f}%), equal: {equal}')

    set_plot_style()
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 14)
    ax.scatter(reg_vals, ga_vals, s=6, alpha=0.15, color='tab:blue', linewidths=0)
    lo = min(min(reg_vals), min(ga_vals))
    hi = max(max(reg_vals), max(ga_vals))
    ax.plot([lo, hi], [lo, hi], color='black', linewidth=2, linestyle='--', label='y = x')
    ax.set_xlabel('Regular DDQN cumulative test reward')
    ax.set_ylabel('GA-guided DDQN cumulative test reward')
    ax.text(0.03, 0.97, f'Above diagonal: {above:,} ({100*above/total:.1f}%)\n'
                          f'Below diagonal: {below:,} ({100*below/total:.1f}%)',
            transform=ax.transAxes, va='top', ha='left', fontsize=22,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_dir / 'figure9.png', dpi=SAVE_DPI)
    plt.close(fig)
    print(f'wrote {out_dir / "figure9.png"}')
    return above, below, total


if __name__ == '__main__':
    build_figure8()
    build_figure9()
