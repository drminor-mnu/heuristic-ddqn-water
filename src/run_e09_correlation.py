#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E9 (R1-4): correlation between cumulative reward (Sigma r, Eq.12 summed
over an episode) and the 4 scenario-level objectives H/F/U/D (Eq.2-6).

Item 1 (main): Spearman rho + 95% CI (Fisher z) on E3 v2's existing
test_metrics.csv, no new runs -- pooled across all 5 models x 5 seeds
(n=67,500) and per model (n=13,500 each).

Item 2 (weight variation): reuses results/E06_curve/*_scenario_metrics.csv
(the existing w1-sweep of GA-only/PSO-only heuristic evaluations, no DQN
training -- see docs/E06_WEIGHT_SENSITIVITY.md) to check whether
rho(cum_reward, max_level_m) strengthens as w1 increases. This is
heuristic-only (tau-driven) data, not trained-DQN-policy data -- labelled
as such throughout.

Item 3 (terminal-reward variant training) is NOT done here -- cancelled,
see docs/REMAINING_WORK.md Sec 0.

Does NOT recompute anything already covered by E4-f Task C (tau_L vs
max_level divergence under lookahead) -- that is a different pair
(heuristic 1-step objective vs outcome under varying L) from this script's
pair (actual cumulative RL reward vs the 4 objectives); Task C is cited,
not reproduced.

Run from src/.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E03_DIR = ROOT / 'results' / 'E03_seeds'
E06_CURVE_DIR = ROOT / 'results' / 'E06_curve'
MODELS = ['Regular', 'GA-guided', 'PSO-guided', 'GA-only', 'PSO-only']
SEEDS = [1, 2, 3, 4, 5]
OBJECTIVES = ['max_level_m', 'n_switches', 'n_dryrun_proxy']  # H, F, D
# U (pump-use proxy) = n_intervals_100 + n_intervals_170, computed below


def spearman(x, y):
    """Spearman rank correlation + 95% CI via Fisher z-transform. No scipy
    dependency (matches run_e05_temporal.py's manual-stats convention)."""
    n = len(x)

    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        ranks = [0.0] * len(v)
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[idx[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    rho = cov / (vx ** 0.5 * vy ** 0.5) if vx > 0 and vy > 0 else float('nan')

    if n > 3 and abs(rho) < 1:
        z = 0.5 * math.log((1 + rho) / (1 - rho))
        se = 1 / math.sqrt(n - 3)
        lo, hi = z - 1.96 * se, z + 1.96 * se
        ci_lo = (math.exp(2 * lo) - 1) / (math.exp(2 * lo) + 1)
        ci_hi = (math.exp(2 * hi) - 1) / (math.exp(2 * hi) + 1)
    else:
        ci_lo = ci_hi = float('nan')
    return rho, ci_lo, ci_hi, n


def load_e03_rows(model, seed):
    p = E03_DIR / model / f'seed{seed}' / 'test_metrics.csv'
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def main():
    print('=== Item 1: cum_reward vs H/F/U/D, E3 v2 test_metrics.csv (no new runs) ===\n')
    all_rows = []
    per_model_rows = defaultdict(list)
    for model in MODELS:
        for seed in SEEDS:
            rows = load_e03_rows(model, seed)
            all_rows.extend(rows)
            per_model_rows[model].extend(rows)

    out_rows = []

    def _corr_block(label, rows):
        cum_reward = [float(r['cum_reward']) for r in rows]
        for obj_label, key in (('H (max_level_m)', 'max_level_m'),
                                ('F (n_switches)', 'n_switches'),
                                ('D (n_dryrun_proxy)', 'n_dryrun_proxy')):
            vals = [float(r[key]) for r in rows]
            rho, lo, hi, n = spearman(cum_reward, vals)
            out_rows.append({'group': label, 'objective': obj_label, 'n': n,
                             'spearman_rho': round(rho, 4),
                             'ci95_lo': round(lo, 4) if lo == lo else '',
                             'ci95_hi': round(hi, 4) if hi == hi else ''})
        u_vals = [float(r['n_intervals_100']) + float(r['n_intervals_170']) for r in rows]
        rho, lo, hi, n = spearman(cum_reward, u_vals)
        out_rows.append({'group': label, 'objective': 'U (n_intervals_100+170)', 'n': n,
                         'spearman_rho': round(rho, 4),
                         'ci95_lo': round(lo, 4) if lo == lo else '',
                         'ci95_hi': round(hi, 4) if hi == hi else ''})
        print(f'{label} (n={len(rows)}): ' +
              ', '.join(f"{r['objective']}={r['spearman_rho']}" for r in out_rows if r['group'] == label))

    _corr_block('pooled (5 models x 5 seeds)', all_rows)
    for model in MODELS:
        _corr_block(model, per_model_rows[model])

    print('\n=== Item 2: rho(cum_reward, max_level_m) vs w1 -- E06_curve GA/PSO-only sweep ===')
    print('(heuristic-only data, tau-driven GA/PSO search, no DQN training -- see docs/E06_WEIGHT_SENSITIVITY.md)\n')

    import re
    by_w1 = defaultdict(list)
    for p in sorted(E06_CURVE_DIR.glob('*_scenario_metrics.csv')):
        m = re.match(r'E06C_w1_([\d.]+)_([A-Z]+)_s(\d+)_scenario_metrics\.csv', p.name)
        if not m:
            continue
        w1 = float(m.group(1))
        with open(p) as f:
            by_w1[w1].extend(list(csv.DictReader(f)))

    w1_rows = []
    for w1 in sorted(by_w1):
        rows = by_w1[w1]
        cum_reward = [float(r['cum_reward']) for r in rows]
        max_level = [float(r['max_level_m']) for r in rows]
        rho, lo, hi, n = spearman(cum_reward, max_level)
        w1_rows.append({'w1': w1, 'n': n, 'spearman_rho_cumreward_maxlevel': round(rho, 4),
                        'ci95_lo': round(lo, 4) if lo == lo else '', 'ci95_hi': round(hi, 4) if hi == hi else ''})
        print(f'  w1={w1}: n={n}, rho={rho:.4f} [{lo:.4f}, {hi:.4f}]' if lo == lo else
              f'  w1={w1}: n={n}, rho={rho:.4f}')

    out_dir = ROOT / 'results' / 'summary'
    from atomic_io import atomic_write

    def _w1(fd):
        w = csv.DictWriter(fd, fieldnames=['group', 'objective', 'n', 'spearman_rho', 'ci95_lo', 'ci95_hi'])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    atomic_write(out_dir / 'E09_correlation.csv', _w1, newline='')

    def _w2(fd):
        w = csv.DictWriter(fd, fieldnames=['w1', 'n', 'spearman_rho_cumreward_maxlevel', 'ci95_lo', 'ci95_hi'])
        w.writeheader()
        for r in w1_rows:
            w.writerow(r)
    atomic_write(out_dir / 'E09_correlation_by_weight.csv', _w2, newline='')

    print(f'\nwrote results/summary/E09_correlation.csv ({len(out_rows)} rows)')
    print(f'wrote results/summary/E09_correlation_by_weight.csv ({len(w1_rows)} rows)')


if __name__ == '__main__':
    main()
