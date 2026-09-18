#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E12 follow-up: distribution of per-scenario cum_reward differences
(GA-guided(n160) - Regular(n160)), to check why the paired mean diff
(-0.479) and Cliff's delta (+0.0082) have opposite signs.

Run from src/. No training.
"""
import csv
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E12_DIR = ROOT / 'results' / 'E12_datasize'
SEEDS = [1, 2, 3]


def load(model, seed):
    p = E12_DIR / model / 'n160' / f'seed{seed}' / 'test_metrics.csv'
    with open(p) as f:
        return list(csv.DictReader(f))


def parse_scenario(sid):
    # e.g. '20yr_0060m_h225'
    m = re.match(r'(\d+)yr_(\d+)m_h(\d+)', sid)
    if not m:
        return None, None
    return m.group(1), m.group(2)  # return_period, duration


def cliffs_delta(x, y):
    """delta = P(x>y) - P(x<y). delta<0 means x tends to be lower than y
    (Cliff's delta < 0 indicates lower values for the first group)."""
    x = np.asarray(x)
    y = np.asarray(y)
    y_sorted = np.sort(y)
    more = less = 0
    for xi in x:
        lo = np.searchsorted(y_sorted, xi, side='left')   # count of y < xi -> x>y
        hi = np.searchsorted(y_sorted, xi, side='right')  # count of y <= xi
        more += lo                        # x > y pairs
        less += len(y_sorted) - hi        # x < y pairs
    return (more - less) / (len(x) * len(y))


def wilcoxon_p(a, b):
    from scipy.stats import wilcoxon
    try:
        _, p = wilcoxon(a, b)
        return p
    except ValueError:
        return float('nan')


def main():
    rows = []  # each: seed, scenario_id, return_period, duration, ga, reg, diff
    for seed in SEEDS:
        reg = {r['scenario_id']: float(r['cum_reward']) for r in load('Regular', seed)}
        ga = {r['scenario_id']: float(r['cum_reward']) for r in load('GA-guided', seed)}
        common = sorted(set(reg) & set(ga))
        for sid in common:
            rp, dur = parse_scenario(sid)
            diff = ga[sid] - reg[sid]
            rows.append({'seed': seed, 'scenario_id': sid, 'return_period': rp,
                        'duration': dur, 'ga': ga[sid], 'reg': reg[sid], 'diff': diff})

    diffs = [r['diff'] for r in rows]
    n = len(diffs)
    print(f'n = {n}')

    def pct(v, p):
        return float(np.percentile(v, p))

    stats = {
        'n': n, 'mean': st.mean(diffs), 'median': st.median(diffs),
        'p5': pct(diffs, 5), 'p25': pct(diffs, 25), 'p75': pct(diffs, 75),
        'p95': pct(diffs, 95), 'min': min(diffs), 'max': max(diffs),
        'std': st.stdev(diffs),
    }
    print('1. distribution of diffs:', stats)

    # histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diffs, bins=80, color='tab:blue')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(stats['mean'], color='red', linestyle='--', label=f"mean={stats['mean']:.3f}")
    ax.axvline(stats['median'], color='green', linestyle='--', label=f"median={stats['median']:.3f}")
    ax.set_xlabel('cum_reward diff (GA-guided - Regular)')
    ax.set_ylabel('count')
    ax.set_title(f'E12 (n160) per-scenario diff distribution, n={n}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / 'results' / 'summary' / 'E12_diff_dist.png', dpi=150)
    print('wrote results/summary/E12_diff_dist.png')

    # 2. bottom 5% tail characteristics
    sorted_rows = sorted(rows, key=lambda r: r['diff'])
    n_tail = int(round(n * 0.05))
    tail = sorted_rows[:n_tail]
    print(f'\n2. bottom 5% tail: n={len(tail)}, diff range [{tail[0]["diff"]:.3f}, {tail[-1]["diff"]:.3f}]')

    def dist_count(rows_subset, key):
        c = defaultdict(int)
        for r in rows_subset:
            c[r[key]] += 1
        return dict(sorted(c.items()))

    overall_dur = dist_count(rows, 'duration')
    tail_dur = dist_count(tail, 'duration')
    overall_rp = dist_count(rows, 'return_period')
    tail_rp = dist_count(tail, 'return_period')
    print('overall duration dist:', overall_dur)
    print('tail duration dist:', tail_dur)
    print('overall return_period dist:', overall_rp)
    print('tail return_period dist:', tail_rp)

    # same scenario across seeds in tail?
    tail_scenario_ids = [r['scenario_id'] for r in tail]
    sid_counts = defaultdict(int)
    for sid in tail_scenario_ids:
        sid_counts[sid] += 1
    repeated = {sid: c for sid, c in sid_counts.items() if c > 1}
    print(f'\ndistinct scenario_ids in tail: {len(sid_counts)} (out of {len(tail)} tail rows)')
    print(f'scenario_ids appearing in tail for >1 seed: {len(repeated)}')
    print('repeated scenario_ids and counts:', repeated)

    # 3. exclude tail, recompute
    rest = sorted_rows[n_tail:]
    rest_ga = [r['ga'] for r in rest]
    rest_reg = [r['reg'] for r in rest]
    rest_diffs = [r['diff'] for r in rest]
    delta_excl = cliffs_delta(rest_ga, rest_reg)
    p_excl = wilcoxon_p(rest_ga, rest_reg)
    print(f'\n3. excluding bottom 5% (n={len(rest)}): mean diff={st.mean(rest_diffs):.4f}, '
          f"Cliff's delta={delta_excl:.4f}, wilcoxon p={p_excl}")

    all_ga = [r['ga'] for r in rows]
    all_reg = [r['reg'] for r in rows]
    delta_all = cliffs_delta(all_ga, all_reg)
    p_all = wilcoxon_p(all_ga, all_reg)
    print(f'   (full n={n} for comparison): mean diff={stats["mean"]:.4f}, '
          f"Cliff's delta={delta_all:.4f}, wilcoxon p={p_all}")

    # 4. median-based
    print(f'\n4. median diff = {stats["median"]:.4f} (vs mean diff = {stats["mean"]:.4f})')

    # write report
    lines = []
    lines.append('# E12 — per-scenario 차이 분포 분석 (2026-09-02)\n')
    lines.append('Regular(n160) vs GA-guided(n160), 시나리오단위(n=8,100=3시드x2,700).\n')
    lines.append('## 1. 차이(GA-guided - Regular) 분포\n')
    lines.append('| 통계 | 값 |')
    lines.append('|---|---|')
    for k in ['n', 'mean', 'median', 'p5', 'p25', 'p75', 'p95', 'min', 'max', 'std']:
        v = stats[k]
        lines.append(f'| {k} | {v:.4f} |' if isinstance(v, float) else f'| {k} | {v} |')
    lines.append('\n히스토그램: `results/summary/E12_diff_dist.png`\n')

    lines.append('## 2. 하위 5% 시나리오 특성\n')
    lines.append(f'n={len(tail)}, diff 범위 [{tail[0]["diff"]:.4f}, {tail[-1]["diff"]:.4f}]\n')
    lines.append(f'**전체 지속시간 분포**: {overall_dur}\n')
    lines.append(f'**하위5% 지속시간 분포**: {tail_dur}\n')
    lines.append(f'**전체 재현기간 분포**: {overall_rp}\n')
    lines.append(f'**하위5% 재현기간 분포**: {tail_rp}\n')
    lines.append(f'**하위5% 내 고유 scenario_id 수**: {len(sid_counts)} / {len(tail)}행\n')
    lines.append(f'**2개 이상 시드에서 하위5%에 반복 등장한 scenario_id 수**: {len(repeated)}\n')
    if repeated:
        lines.append(f'반복 scenario_id와 등장 횟수: {repeated}\n')

    lines.append('## 3. 하위 5% 제외 후 재계산\n')
    lines.append('| 대상 | n | mean diff | Wilcoxon p | Cliff\'s δ |')
    lines.append('|---|---|---|---|---|')
    lines.append(f'| 전체 | {n} | {stats["mean"]:.4f} | {p_all:.4g} | {delta_all:.4f} |')
    lines.append(f'| 하위5% 제외 | {len(rest)} | {st.mean(rest_diffs):.4f} | {p_excl:.4g} | {delta_excl:.4f} |')

    lines.append('\n## 4. 중앙값 기준 차이\n')
    lines.append(f'median diff = {stats["median"]:.4f} (mean diff = {stats["mean"]:.4f})\n')

    (ROOT / 'results' / 'summary' / 'E12_diff_analysis.md').write_text('\n'.join(lines) + '\n')
    print('\nwrote results/summary/E12_diff_analysis.md')


if __name__ == '__main__':
    main()
