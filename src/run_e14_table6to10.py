#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Section 3.1 regeneration: Table 6-10 (2-criterion Regular DDQN vs
4-criterion GA-guided DDQN), 9(duration) x 6(return period) cross grid,
5 metrics (max_level_m, n_switches, n_intervals_100, n_intervals_170,
n_dryrun_proxy). No aggregation script for this cross grid existed in
the repo before this file (src/analyze_e03.py only groups by duration
alone or return_period alone) -- this is new, additive.

CI methodology matches analyze_e03.py's aggregate()/ci_row() exactly
(2-stage: per (duration, return_period, seed) mean over scenarios in
that cell, then mean/std(ddof=1)/95% CI via t-distribution df=4 across
the 5 per-seed means) -- duplicated here, not imported, since
analyze_e03.py hardcodes its own ROOT/MODELS/SEEDS for E03 specifically.

Data sources:
  - Regular (2-criterion, w=[0.5,0.5,0,0]): results/E14_two_criteria/Regular/seed{1..5}/
    (produced by src/run_e14_two_criteria.py -- NOT run yet as of this
    script's creation; --regular-source e3 substitutes E3 v2's 4-criterion
    Regular for a smoke test of this script's cross-grid/CI logic only,
    see run_e14_table6to10_smoketest.sh).
  - GA-guided (4-criterion, new weights [0.40,0.25,0.25,0.10] -- NOTE:
    this differs from the manuscript's original Table 6-10 GA-guided
    column, which used the OLD weights [0.25,0.40,0.25,0.10]; this
    script uses E3 v2's GA-guided as instructed by the regeneration plan,
    the weight difference is a known, already-documented fact
    (docs/response/SECTION_3_1_REGENERATION.md), not re-derived here):
    results/E03_seeds/GA-guided/seed{1..5}/ (reused, not retrained).

Run from src/, only after E14 has produced all 5 Regular seeds (or with
--regular-source e3 for a format smoke test).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
RESULTS = ROOT / 'results'
SEEDS = [1, 2, 3, 4, 5]
METRICS = ['max_level_m', 'n_switches', 'n_intervals_100', 'n_intervals_170', 'n_dryrun_proxy']
DURATIONS = [60, 120, 180, 240, 360, 540, 720, 1080, 1440]
RETURN_PERIODS = [10, 20, 30, 50, 80, 100]
T_CRIT_DF4 = stats.t.ppf(0.975, 4)  # matches analyze_e03.py exactly


def load_condition(dir_path: Path, label: str) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        p = dir_path / f'seed{seed}' / 'test_metrics.csv'
        if not p.exists():
            raise FileNotFoundError(f'{label}: missing {p} -- all 5 seeds required')
        df = pd.read_csv(p, dtype={'duration_min': str, 'return_period': str})
        df['seed'] = seed
        df['model'] = label
        df['duration_int'] = df['duration_min'].astype(int)
        df['return_period_int'] = df['return_period'].astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def ci_row(per_seed_means):
    a = np.asarray(per_seed_means, float)
    mean = a.mean()
    sd = a.std(ddof=1) if len(a) > 1 else 0.0
    half = T_CRIT_DF4 * sd / np.sqrt(len(a)) if len(a) > 1 else float('nan')
    return mean, sd, half


def build_grid(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """2-stage aggregation matching analyze_e03.py: per (dur, rp, model, seed)
    mean over scenarios, then mean/std/CI across the 5 seed-level means."""
    stage1 = df.groupby(['duration_int', 'return_period_int', 'model', 'seed'])[metric].mean().reset_index()
    rows = []
    for (dur, rp, model), g in stage1.groupby(['duration_int', 'return_period_int', 'model']):
        mean, sd, half = ci_row(g[metric].values)
        rows.append({'duration_int': dur, 'return_period_int': rp, 'model': model,
                     'n_seeds': g['seed'].nunique(), f'{metric}_mean': round(mean, 4),
                     f'{metric}_std': round(sd, 4),
                     f'{metric}_ci95_half': round(half, 4) if half == half else '',
                     })
    return pd.DataFrame(rows)


def render_markdown(grid: pd.DataFrame, metric: str, title: str) -> str:
    lines = [f'# {title}\n']
    lines.append('| Model | Duration (min) | ' + ' | '.join(str(rp) for rp in RETURN_PERIODS) + ' |')
    lines.append('|---|---|' + '---|' * len(RETURN_PERIODS))
    for dur in DURATIONS:
        for model in sorted(grid['model'].unique()):
            row = [model, str(dur)]
            for rp in RETURN_PERIODS:
                sub = grid[(grid.duration_int == dur) & (grid.return_period_int == rp) & (grid.model == model)]
                if len(sub) == 0:
                    row.append('')
                else:
                    r = sub.iloc[0]
                    row.append(f"{r[f'{metric}_mean']:.3f}±{r[f'{metric}_std']:.3f}")
            lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines) + '\n'


TABLE_TITLES = {
    'max_level_m': 'Table 6 (regenerated). Comparison of maximum water level between the two-criterion DDQN and four-criterion GA-guided DDQN (mean +/- sd over 5 seeds).',
    'n_switches': 'Table 7 (regenerated). Comparison of the mean number of on/off changes (mean +/- sd over 5 seeds).',
    'n_intervals_100': 'Table 8 (regenerated). Comparison of the mean number of operating intervals for the 100 m3/min pumps (mean +/- sd over 5 seeds).',
    'n_intervals_170': 'Table 9 (regenerated). Comparison of the mean number of operating intervals for the 170 m3/min pumps (mean +/- sd over 5 seeds).',
    'n_dryrun_proxy': 'Table 10 (regenerated). Comparison of the mean number of dry-running risk-proxy events (mean +/- sd over 5 seeds).',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--regular-source', choices=['e14', 'e3'], default='e14',
                   help='e14 = results/E14_two_criteria/Regular (real 2-criterion run); '
                        'e3 = results/E03_seeds/Regular (4-criterion, SMOKE TEST ONLY -- '
                        'format/CI-logic validation, not a real 2-criterion comparison)')
    p.add_argument('--out-dir', type=str, default=None,
                   help='override output directory (smoke test uses a scratch dir)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.regular_source == 'e14':
        regular_dir = RESULTS / 'E14_two_criteria' / 'Regular'
        regular_label = 'Regular(2-criterion)'
    else:
        regular_dir = RESULTS / 'E03_seeds' / 'Regular'
        regular_label = 'Regular(SMOKE-TEST-4-criterion-stand-in)'
        print('*** SMOKE TEST MODE: using E3 v2 4-criterion Regular as a '
              'placeholder for E14 2-criterion Regular. Output is NOT a valid '
              'Table 6-10 regeneration -- format/CI-logic check only. ***')

    ga_dir = RESULTS / 'E03_seeds' / 'GA-guided'

    reg_df = load_condition(regular_dir, regular_label)
    ga_df = load_condition(ga_dir, 'GA-guided(4-criterion)')
    df = pd.concat([reg_df, ga_df], ignore_index=True)

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS / 'summary'
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        grid = build_grid(df, metric)
        csv_path = out_dir / f'E14_table_{metric}.csv'
        grid.to_csv(csv_path, index=False)
        md_path = out_dir / f'E14_table_{metric}.md'
        md_path.write_text(render_markdown(grid, metric, TABLE_TITLES[metric]))
        print(f'wrote {csv_path} ({len(grid)} rows), {md_path}')


if __name__ == '__main__':
    main()
