#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5 item 4 (2026-08-31 instruction): place E3 v2's GA-guided vs Regular
(n=13,500, results/summary/E03_paired_tests.csv) side by side with E5's
A3-A1 and A1-A0 scenario-level rows (results/summary/E05_stage1_tests.csv,
granularity='scenario_level') in one table, for the four metrics
max_level_m / n_switches / n_dryrun_proxy / cum_reward.

Facts only, no interpretation: this script does not label which contrast is
"guidance type" vs "guidance presence" beyond the comparison names already
used elsewhere in this experiment (A3-A1 / A1-A0 / GA-guided-Regular) --
those names are carried through unchanged from run_e05_analyze.py and
docs/E03_FINDINGS.md.

Run from src/, after results/summary/E05_stage1_tests.csv has been
regenerated with 5 seeds (run_e05_analyze.py, SEEDS=[1,2,3,4,5]).
"""
import csv
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
METRICS = ['max_level_m', 'n_switches', 'n_dryrun_proxy', 'cum_reward']


def load_e03_paired():
    path = ROOT / 'results' / 'summary' / 'E03_paired_tests.csv'
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['model_a'] == 'GA-guided' and r['model_b'] == 'Regular' and r['metric'] in METRICS:
                rows[r['metric']] = r
    return rows


def load_e05_scenario_level():
    path = ROOT / 'results' / 'summary' / 'E05_stage1_tests.csv'
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['granularity'] != 'scenario_level':
                continue
            if r['comparison'] not in ('A3-A1 (intelligent selection)',
                                        'A1-A0 (exploration increase)'):
                continue
            if r['metric'] not in METRICS:
                continue
            rows[(r['comparison'], r['metric'])] = r
    return rows


def main():
    e03 = load_e03_paired()
    e05 = load_e05_scenario_level()

    if not e03:
        print('E03_paired_tests.csv: no GA-guided/Regular rows found')
        return
    missing = [m for m in METRICS if m not in e03]
    if missing:
        print(f'E03_paired_tests.csv: missing metrics {missing}')

    out_rows = []
    for m in METRICS:
        if m in e03:
            r = e03[m]
            out_rows.append({
                'source': 'E03_v2', 'comparison': 'GA-guided-Regular',
                'metric': m, 'n_pairs': r['n_pairs'],
                'mean_diff': r['mean_diff'], 'wilcoxon_p': r['p'],
                'holm_p': r['p_holm'], 'cliffs_delta': r['cliffs_delta'],
            })
        for comp in ('A3-A1 (intelligent selection)', 'A1-A0 (exploration increase)'):
            key = (comp, m)
            if key in e05:
                r = e05[key]
                out_rows.append({
                    'source': 'E5_scenario_level', 'comparison': comp,
                    'metric': m, 'n_pairs': r['n_pairs'],
                    'mean_diff': r['mean_diff'], 'wilcoxon_p': r['wilcoxon_p'],
                    'holm_p': r['holm_p'], 'cliffs_delta': r['cliffs_delta'],
                })
            else:
                print(f'E05_stage1_tests.csv: missing scenario_level row for {comp} / {m} '
                      '-- run run_e05_analyze.py with SEEDS=[1,2,3,4,5] first')

    out_dir = ROOT / 'results' / 'summary'
    from atomic_io import atomic_write

    fields = ['source', 'comparison', 'metric', 'n_pairs', 'mean_diff',
              'wilcoxon_p', 'holm_p', 'cliffs_delta']

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    atomic_write(out_dir / 'E05_guidance_contrast.csv', _w, newline='')
    print(f'wrote results/summary/E05_guidance_contrast.csv ({len(out_rows)} rows)')
    for r in out_rows:
        print(r)


if __name__ == '__main__':
    main()
