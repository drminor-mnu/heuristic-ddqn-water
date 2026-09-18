#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-f Task I-1: paired Wilcoxon + Cliff's delta + 95% CI for the main
E4-e Task B/C/E comparisons, with n and seed counts on every row.

No new simulation -- reads existing CSVs:
  results/E04_enum/saturation_w2_scenarios.csv        (E4-e Task B, 270, L<=6)
  results/E04_enum/saturation_w2_scenarios_L78.csv     (E4-e Task B, 18, L=7,8)
  results/E03_seeds/{model}/seed{1..5}/test_metrics.csv (E3 v2)

Run from src/.
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E04_DIR = ROOT / 'results' / 'E04_enum'
E03_DIR = ROOT / 'results' / 'E03_seeds'


def cliffs_delta(x, y):
    nx, ny = len(x), len(y)
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (nx * ny)


def bootstrap_ci_mean_diff(x, y, n_boot=5000, seed=0):
    import random
    rng = random.Random(seed)
    diffs = [a - b for a, b in zip(x, y)]
    n = len(diffs)
    boots = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(st.mean(sample))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    return lo, hi


def load_scenario_ids():
    ids, order = set(), []
    with open(E04_DIR / 'fixed_point_by_L_scenarios.csv') as f:
        for r in csv.DictReader(f):
            if r['scenario_id'] not in ids:
                ids.add(r['scenario_id'])
                order.append(r['scenario_id'])
    return order


def load_enum(path):
    by = defaultdict(dict)
    if not Path(path).exists():
        return by
    with open(path) as f:
        for r in csv.DictReader(f):
            by[r['scenario_id']][int(r['L'])] = float(r['max_level_m'])
    return by


def main():
    from scipy.stats import wilcoxon

    by_full = load_enum(E04_DIR / 'saturation_w2_scenarios.csv')
    by_red = load_enum(E04_DIR / 'saturation_w2_scenarios_L78.csv')

    rows = []

    def add_row(label, n, seeds, x, y):
        w, p = wilcoxon(x, y)
        d = cliffs_delta(x, y)
        lo, hi = bootstrap_ci_mean_diff(x, y)
        rows.append({
            'comparison': label, 'n_scenarios': n, 'n_seeds': seeds,
            'mean_diff': round(st.mean(a - b for a, b in zip(x, y)), 4),
            'ci95_lo': round(lo, 4), 'ci95_hi': round(hi, 4),
            'wilcoxon_p': f'{p:.3e}', 'cliffs_delta': round(d, 4),
        })

    # ---- Task B comparisons (ENUM-L, deterministic, 1 "seed") ----
    scen_full = sorted(by_full.keys())
    l1 = [by_full[s][1] for s in scen_full]
    l2 = [by_full[s][2] for s in scen_full]
    l3 = [by_full[s][3] for s in scen_full]
    l5 = [by_full[s][5] for s in scen_full]
    l6 = [by_full[s][6] for s in scen_full]
    add_row('ENUM max_level: L1 vs L2', len(scen_full), 1, l1, l2)
    add_row('ENUM max_level: L1 vs L3', len(scen_full), 1, l1, l3)
    add_row('ENUM max_level: L2 vs L3', len(scen_full), 1, l2, l3)
    add_row('ENUM max_level: L3 vs L5', len(scen_full), 1, l3, l5)
    add_row('ENUM max_level: L5 vs L6', len(scen_full), 1, l5, l6)

    scen_red = sorted(by_red.keys())
    l1r = [by_full[s][1] for s in scen_red]
    l7 = [by_red[s][7] for s in scen_red]
    l8 = [by_red[s][8] for s in scen_red]
    add_row('ENUM max_level: L1 vs L7', len(scen_red), 1, l1r, l7)
    add_row('ENUM max_level: L1 vs L8', len(scen_red), 1, l1r, l8)
    add_row('ENUM max_level: L6 vs L7', len(scen_red), 1,
           [by_full[s][6] for s in scen_red], l7)
    add_row('ENUM max_level: L7 vs L8', len(scen_red), 1, l7, l8)

    # ---- Task E: ENUM-L vs DDQN (R4-10) ----
    # Independent scenario-level pairs (n=270, not pseudo-replicated across
    # seeds): DDQN side is each scenario's mean over its 5 seeds, so the pair
    # count equals the scenario count and Wilcoxon's independence assumption
    # is not violated by repeating the deterministic ENUM value 5x.
    def load_seed_mean(model):
        per_scenario = defaultdict(list)
        for seed in (1, 2, 3, 4, 5):
            path = E03_DIR / model / f'seed{seed}' / 'test_metrics.csv'
            with open(path) as f:
                for r in csv.DictReader(f):
                    if r['scenario_id'] in by_full:
                        per_scenario[r['scenario_id']].append(float(r['max_level_m']))
        return {s: st.mean(v) for s, v in per_scenario.items() if len(v) == 5}

    ga_mean = load_seed_mean('GA-guided')
    reg_mean = load_seed_mean('Regular')
    scen_ga = sorted(s for s in scen_full if s in ga_mean)
    scen_reg = sorted(s for s in scen_full if s in reg_mean)

    add_row('ENUM-L5 vs GA-guided DDQN (max_level, DDQN=5-seed mean/scenario)',
           len(scen_ga), 5,
           [by_full[s][5] for s in scen_ga], [ga_mean[s] for s in scen_ga])
    add_row('ENUM-L2 vs GA-guided DDQN (max_level, DDQN=5-seed mean/scenario)',
           len(scen_ga), 5,
           [by_full[s][2] for s in scen_ga], [ga_mean[s] for s in scen_ga])
    add_row('ENUM-L5 vs Regular DDQN (max_level, DDQN=5-seed mean/scenario)',
           len(scen_reg), 5,
           [by_full[s][5] for s in scen_reg], [reg_mean[s] for s in scen_reg])

    out_path = E04_DIR / 'stats_w2.csv'
    from atomic_io import atomic_write

    def _w(fd):
        fields = ['comparison', 'n_scenarios', 'n_seeds', 'mean_diff',
                  'ci95_lo', 'ci95_hi', 'wilcoxon_p', 'cliffs_delta']
        w = csv.DictWriter(fd, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    atomic_write(out_path, _w, newline='')
    print(f'wrote {out_path.relative_to(ROOT)}')
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
