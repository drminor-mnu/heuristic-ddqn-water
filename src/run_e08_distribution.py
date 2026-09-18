#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E8 item 2 (R1-5): quantify train/test rainfall-distribution similarity.

For every scenario in data/splits/fixed_split_seed42.json, reads the
corresponding cumulative-rainfall .txt file (same (year,duration,h-index)
as the .inp, under data/rainfall/{year}year/) and computes:
  - total rainfall depth (last cumulative value, mm)
  - peak single-interval intensity (max first-difference, mm/interval)
No training, no SWMM -- pure file read + stats. Two-sample KS statistic
computed manually (max |ECDF_train - ECDF_test|), no scipy dependency.

Run from src/.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def txt_path_for_inp(inp_rel_path: str) -> Path:
    # data/gasan/10year/10yr_0720m_h934.inp -> data/rainfall/10year/10yr_0720m_h934.txt
    parts = Path(inp_rel_path).parts  # ('data','gasan','10year','10yr_0720m_h934.inp')
    year_dir = parts[2]
    fname = Path(parts[-1]).stem + '.txt'
    return ROOT / 'data' / 'rainfall' / year_dir / fname


def load_stats(inp_rel_paths):
    totals, peaks = [], []
    missing = 0
    for p in inp_rel_paths:
        tp = txt_path_for_inp(p)
        if not tp.exists():
            missing += 1
            continue
        vals = [float(x) for x in tp.read_text().split()]
        totals.append(vals[-1])
        diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        peaks.append(max(diffs) if diffs else 0.0)
    return totals, peaks, missing


def ks_statistic(a, b):
    """Two-sample KS statistic: max|ECDF_a(x) - ECDF_b(x)| over x in a union b."""
    a_sorted, b_sorted = sorted(a), sorted(b)
    all_vals = sorted(set(a_sorted) | set(b_sorted))
    na, nb = len(a_sorted), len(b_sorted)

    def ecdf(sorted_vals, n, x):
        import bisect
        return bisect.bisect_right(sorted_vals, x) / n

    d = max(abs(ecdf(a_sorted, na, x) - ecdf(b_sorted, nb, x)) for x in all_vals)
    return d


def main():
    split = json.loads((ROOT / 'data' / 'splits' / 'fixed_split_seed42.json').read_text())
    train_inps = split['train_inps']
    test_inps = split['test_inps']

    train_totals, train_peaks, train_missing = load_stats(train_inps)
    test_totals, test_peaks, test_missing = load_stats(test_inps)

    print(f'train: n={len(train_totals)} (missing txt: {train_missing})')
    print(f'test:  n={len(test_totals)} (missing txt: {test_missing})')

    import statistics as st
    for label, tr, te in (('total_depth_mm', train_totals, test_totals),
                          ('peak_interval_mm', train_peaks, test_peaks)):
        d = ks_statistic(tr, te)
        print(f'\n{label}:')
        print(f'  train mean={st.mean(tr):.3f} std={st.stdev(tr):.3f} '
              f'min={min(tr):.3f} max={max(tr):.3f}')
        print(f'  test  mean={st.mean(te):.3f} std={st.stdev(te):.3f} '
              f'min={min(te):.3f} max={max(te):.3f}')
        print(f'  KS statistic D = {d:.5f}')

    out = {
        'n_train': len(train_totals), 'n_test': len(test_totals),
        'total_depth_ks': ks_statistic(train_totals, test_totals),
        'total_depth_train_mean': st.mean(train_totals), 'total_depth_train_std': st.stdev(train_totals),
        'total_depth_test_mean': st.mean(test_totals), 'total_depth_test_std': st.stdev(test_totals),
        'peak_interval_ks': ks_statistic(train_peaks, test_peaks),
        'peak_interval_train_mean': st.mean(train_peaks), 'peak_interval_train_std': st.stdev(train_peaks),
        'peak_interval_test_mean': st.mean(test_peaks), 'peak_interval_test_std': st.stdev(test_peaks),
    }
    out_dir = ROOT / 'results' / 'summary'
    import csv
    from atomic_io import atomic_write

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=list(out.keys()))
        w.writeheader()
        w.writerow({k: round(v, 5) if isinstance(v, float) else v for k, v in out.items()})
    atomic_write(out_dir / 'E08_distribution_shift.csv', _w, newline='')
    print(f'\nwrote results/summary/E08_distribution_shift.csv')


if __name__ == '__main__':
    main()
