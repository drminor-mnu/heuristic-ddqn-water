#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E15 reproducibility check (R4-9 support): compares
results/E15_timing/GA-guided/seed{1..5}/test_metrics.csv against
results/E03_seeds/GA-guided/seed{1..5}/test_metrics.csv row-for-row
(joined on scenario_id, return_period, duration_min, seed). Both runs
use identical seeds/weights/hyperparameters/data split -- only the
parallel worker count differs (E15: --workers 2, E3 v2: --workers 8,
see results/summary/TABLE5_COMPARABILITY.md) -- so under a fully
deterministic pipeline the two should be identical.

DO NOT RUN before E15 (src/run_e15_timing.py) has produced all 5 seeds
-- this script will raise FileNotFoundError otherwise.

Compared columns (deterministic outcome columns only):
    max_level_m, n_switches, n_intervals_100, n_intervals_170,
    n_dryrun_proxy, cum_reward, overflow_flag

Excluded from the reproducibility judgment (these are wall-clock
measurements taken during evaluation and are expected to differ between
runs regardless of determinism):
    inference_time_s, swmm_time_s, total_time_s

Tolerance (stated explicitly, not tuned after seeing data):
    - overflow_flag, n_switches, n_intervals_100, n_intervals_170: exact
      integer match required (these are counts, not continuous floats).
    - max_level_m, n_dryrun_proxy, cum_reward: floating-point columns,
      compared with numpy.isclose(rtol=1e-9, atol=1e-9) -- i.e.
      effectively exact (float64 round-trip precision), not a loose
      tolerance. This catches true non-determinism (e.g. GPU reduction
      order) while still allowing for float64 serialization round-trip
      noise at the last bit.

Verdict per seed:
    - 완전 일치 (exact match): 0 rows differ on any compared column.
    - 부분 일치 (partial match): >0 but <1% of rows differ on at least
      one compared column.
    - 불일치 (mismatch): >=1% of rows differ.
    (The 1% threshold is stated here before running, not chosen after
    seeing the result.)

Writes results/summary/E15_reproducibility.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
RESULTS = ROOT / 'results'
E15_DIR = RESULTS / 'E15_timing' / 'GA-guided'
E03_DIR = RESULTS / 'E03_seeds' / 'GA-guided'
SEEDS = [1, 2, 3, 4, 5]

KEY_COLS = ['scenario_id', 'return_period', 'duration_min', 'seed']
INT_COLS = ['overflow_flag', 'n_switches', 'n_intervals_100', 'n_intervals_170']
FLOAT_COLS = ['max_level_m', 'n_dryrun_proxy', 'cum_reward']
EXCLUDED_COLS = ['inference_time_s', 'swmm_time_s', 'total_time_s']
PARTIAL_THRESHOLD = 0.01  # 1% of rows differing on >=1 compared column


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={'duration_min': str, 'return_period': str, 'quartile': str})


def compare_seed(seed: int) -> dict:
    e15_path = E15_DIR / f'seed{seed}' / 'test_metrics.csv'
    e03_path = E03_DIR / f'seed{seed}' / 'test_metrics.csv'
    if not e15_path.exists():
        raise FileNotFoundError(f'E15 not complete: {e15_path} missing -- do not run this script before E15 finishes')
    if not e03_path.exists():
        raise FileNotFoundError(f'E3 v2 reference missing: {e03_path}')

    e15 = load(e15_path)
    e03 = load(e03_path)

    merged = e15.merge(e03, on=KEY_COLS, how='outer', suffixes=('_e15', '_e03'), indicator=True)
    unmatched = merged[merged['_merge'] != 'both']
    both = merged[merged['_merge'] == 'both'].copy()

    diff_mask = pd.Series(False, index=both.index)
    col_diff_counts = {}
    for c in INT_COLS:
        d = both[f'{c}_e15'].astype('Int64') != both[f'{c}_e03'].astype('Int64')
        col_diff_counts[c] = int(d.sum())
        diff_mask |= d.fillna(True)
    for c in FLOAT_COLS:
        d = ~np.isclose(both[f'{c}_e15'].astype(float), both[f'{c}_e03'].astype(float),
                         rtol=1e-9, atol=1e-9, equal_nan=True)
        col_diff_counts[c] = int(d.sum())
        diff_mask |= d

    n_rows = len(both)
    n_diff = int(diff_mask.sum())
    frac_diff = n_diff / n_rows if n_rows else float('nan')

    if len(unmatched) > 0:
        verdict = '불일치(키 불일치 -- scenario_id/return_period/duration_min 조합이 다름)'
    elif n_diff == 0:
        verdict = '완전 일치'
    elif frac_diff < PARTIAL_THRESHOLD:
        verdict = '부분 일치'
    else:
        verdict = '불일치'

    diff_rows = both[diff_mask]
    diff_examples = []
    for c in INT_COLS + FLOAT_COLS:
        if col_diff_counts[c] > 0:
            sub = diff_rows[diff_rows[f'{c}_e15'] != diff_rows[f'{c}_e03']] if c in INT_COLS else diff_rows
            if len(sub) > 0:
                deltas = (sub[f'{c}_e15'].astype(float) - sub[f'{c}_e03'].astype(float))
                diff_examples.append({
                    'column': c, 'n_diff': col_diff_counts[c],
                    'max_abs_delta': float(deltas.abs().max()) if len(deltas) else 0.0,
                    'mean_abs_delta': float(deltas.abs().mean()) if len(deltas) else 0.0,
                })

    return {
        'seed': seed, 'n_rows_e15': len(e15), 'n_rows_e03': len(e03),
        'n_unmatched_keys': len(unmatched), 'n_rows_compared': n_rows,
        'n_rows_diff': n_diff, 'frac_diff': frac_diff,
        'verdict': verdict, 'col_diff_counts': col_diff_counts,
        'diff_examples': diff_examples,
    }


def render_report(results: list[dict]) -> str:
    lines = ['# E15 재현성 대조 — E15(workers=2) vs E3 v2 GA-guided(workers=8)\n']
    lines.append('해석·옹호 없이 사실·숫자만 기록한다.\n')
    lines.append(f'\n판정기준: 비교 열(정수) {INT_COLS}, 비교 열(실수, '
                  f'numpy.isclose rtol=1e-9 atol=1e-9) {FLOAT_COLS}. '
                  f'제외 열(측정마다 달라지는 벽시계 시간) {EXCLUDED_COLS}. '
                  f'완전 일치=차이행0, 부분 일치=차이행비율<{PARTIAL_THRESHOLD*100:.0f}%, '
                  f'불일치=그 이상 (기준은 결과를 보기 전에 명시).\n')

    lines.append('\n| seed | 판정 | 비교행수 | 차이행수 | 차이비율 | 키불일치 |')
    lines.append('|---|---|---|---|---|---|')
    for r in results:
        lines.append(f"| {r['seed']} | {r['verdict']} | {r['n_rows_compared']} | "
                      f"{r['n_rows_diff']} | {r['frac_diff']*100:.4f}% | {r['n_unmatched_keys']} |")

    for r in results:
        if r['diff_examples']:
            lines.append(f"\n## seed{r['seed']} 열별 차이 상세\n")
            lines.append('| 열 | 차이행수 | 최대절대차 | 평균절대차 |')
            lines.append('|---|---|---|---|')
            for d in r['diff_examples']:
                lines.append(f"| {d['column']} | {d['n_diff']} | {d['max_abs_delta']:.6g} | {d['mean_abs_delta']:.6g} |")

    n_exact = sum(1 for r in results if r['verdict'] == '완전 일치')
    n_partial = sum(1 for r in results if r['verdict'] == '부분 일치')
    n_mismatch = sum(1 for r in results if r['verdict'] not in ('완전 일치', '부분 일치'))
    lines.append(f"\n## 요약\n\n5시드 중 완전일치 {n_exact}개, 부분일치 {n_partial}개, "
                  f"불일치 {n_mismatch}개.\n")
    return '\n'.join(lines) + '\n'


def main():
    results = [compare_seed(s) for s in SEEDS]
    report = render_report(results)
    out_path = RESULTS / 'summary' / 'E15_reproducibility.md'
    out_path.write_text(report)
    print(f'wrote {out_path}')
    for r in results:
        print(f"seed{r['seed']}: {r['verdict']} ({r['n_rows_diff']}/{r['n_rows_compared']} rows differ)")


if __name__ == '__main__':
    main()
