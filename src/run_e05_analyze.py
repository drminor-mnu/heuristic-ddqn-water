#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5 stage-1 analysis: A0 / A1 (fresh, results/E05_ablation/) vs A3 (reused
from results/E03_seeds/GA-guided/, E3 v2 -- not re-run, see
docs/REVISION_EXPERIMENT_PLAN.md E5 and the pre-flight check in
docs/E05_REPORT.md).

Computes, per condition (3 seeds each):
  - mean +/- std, 95% CI (t, df=2) for max_level / n_switches / n_intervals_100
    / n_intervals_170 / n_dryrun_proxy / cum_reward / overflow_flag
  - A3-A1 ("intelligent selection" effect) and A1-A0 ("exploration increase"
    effect), paired Wilcoxon + Cliff's delta + Holm correction
  - action-usage histograms for A1 vs A3 (Q4)

Run from src/, after results/E05_ablation/{A0,A1}/seed{1,2,3}/test_metrics.csv
exist.
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E05_DIR = ROOT / 'results' / 'E05_ablation'
E03_DIR = ROOT / 'results' / 'E03_seeds'
SEEDS = [1, 2, 3, 4, 5]
# With n=5 seeds, Wilcoxon signed-rank's minimum attainable two-sided p is
# 2/32 = 0.0625 (all 5 differences same sign) -- still > 0.05 before Holm
# correction, so seed_mean-level significance remains structurally
# unreachable at n=5. Reported as a footnote wherever seed_mean rows are
# shown; scenario_level is the primary granularity, seed_mean a reference.
SEED_MEAN_MIN_P_NOTE = ('n=5 seeds: Wilcoxon two-sided minimum attainable '
                         'p = 2/32 = 0.0625 (pre-Holm); seed_mean-level '
                         'significance is structurally unreachable at this '
                         'n. scenario_level is the primary granularity, '
                         'seed_mean a reference only.')
METRICS = ['max_level_m', 'n_switches', 'n_intervals_100', 'n_intervals_170',
           'n_dryrun_proxy', 'cum_reward', 'overflow_flag']


def cliffs_delta(x, y):
    nx, ny = len(x), len(y)
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (nx * ny)


def load_condition(cond: str) -> dict[int, list[dict]]:
    """{seed: [row, ...]} for a condition, from E05_ablation (A0/A1/A2/A4_*)
    or E03_seeds/GA-guided or Regular (A3 / A0_baseline)."""
    out = {}
    if cond == 'A3':
        for seed in SEEDS:
            path = E03_DIR / 'GA-guided' / f'seed{seed}' / 'test_metrics.csv'
            if path.exists():
                out[seed] = list(csv.DictReader(open(path)))
    elif cond == 'A0_baseline':
        # E3 v2 Regular (dqn_from_demon_v1.train(), NOT
        # _train_guided_by_optimizer -- see docs/E05_REPORT.md Sec 1-1).
        # Compared against A0 to isolate the training-schedule difference
        # (eps schedule / replay warm-up / target-sync) from any guidance
        # effect, since neither condition uses guidance (ga_prob=0 either
        # way) -- R3-3.
        for seed in SEEDS:
            path = E03_DIR / 'Regular' / f'seed{seed}' / 'test_metrics.csv'
            if path.exists():
                out[seed] = list(csv.DictReader(open(path)))
    else:
        for seed in SEEDS:
            path = E05_DIR / cond / f'seed{seed}' / 'test_metrics.csv'
            if path.exists():
                out[seed] = list(csv.DictReader(open(path)))
    return out


def summarize(cond_data: dict[int, list[dict]]) -> dict:
    per_seed_means = defaultdict(list)
    for seed, rows in cond_data.items():
        for m in METRICS:
            per_seed_means[m].append(st.mean(float(r[m]) for r in rows))
    out = {}
    for m in METRICS:
        vals = per_seed_means[m]
        n = len(vals)
        mean = st.mean(vals)
        sd = st.stdev(vals) if n > 1 else 0.0
        if n > 1:
            import math
            t_crit = {2: 4.303, 3: 3.182, 4: 2.776}.get(n - 1, 2.776)
            half = t_crit * sd / math.sqrt(n)
        else:
            half = float('nan')
        out[m] = {'mean': mean, 'std': sd, 'ci95_lo': mean - half, 'ci95_hi': mean + half,
                  'per_seed': vals}
    return out


def paired_test(cond_a: dict, cond_b: dict, label: str):
    from scipy.stats import wilcoxon

    seeds = sorted(set(cond_a) & set(cond_b))
    rows_out = []
    for m in METRICS:
        a = [st.mean(float(r[m]) for r in cond_a[s]) for s in seeds]
        b = [st.mean(float(r[m]) for r in cond_b[s]) for s in seeds]
        if len(a) < 2:
            rows_out.append({'comparison': label, 'metric': m, 'n_seeds': len(a),
                             'mean_diff': (a[0] - b[0]) if a else float('nan'),
                             'wilcoxon_p': '', 'cliffs_delta': ''})
            continue
        if all(x == y for x, y in zip(a, b)):
            # every paired difference is exactly zero (e.g. overflow_flag=0
            # for every seed on both sides) -- scipy's wilcoxon raises
            # rather than returning p=1; record the degenerate case as-is.
            rows_out.append({
                'comparison': label, 'metric': m, 'n_seeds': len(a),
                'mean_diff': 0.0, 'wilcoxon_p': 'n/a (all diffs zero)',
                'cliffs_delta': 0.0,
            })
            continue
        w, p = wilcoxon(a, b)
        rows_out.append({
            'comparison': label, 'metric': m, 'n_seeds': len(a),
            'mean_diff': round(st.mean(x - y for x, y in zip(a, b)), 5),
            'wilcoxon_p': f'{p:.4g}', 'cliffs_delta': round(cliffs_delta(a, b), 4),
        })
    return rows_out


def paired_test_scenario_level(cond_a: dict, cond_b: dict, label: str):
    """Same comparison as paired_test(), but paired on (scenario_id, seed)
    instead of collapsed to one mean per seed -- matches the unit
    docs/E03_FINDINGS.md / results/summary/E03_paired_tests.csv use (n up to
    n_seeds x 2,700). Only seeds present in BOTH conditions are used; only
    scenario_ids present in both conditions for a given seed are paired.
    Reported side-by-side with paired_test()'s seed-mean-level rows in the
    same output table (granularity column) -- the two are not
    interchangeable and must not be cited without that label (see
    docs/E05_REPORT.md Sec 1-4 / Sec 7-3)."""
    from scipy.stats import wilcoxon

    seeds = sorted(set(cond_a) & set(cond_b))
    rows_out = []
    for m in METRICS:
        a, b = [], []
        for s in seeds:
            by_scen_a = {r['scenario_id']: r for r in cond_a[s]}
            by_scen_b = {r['scenario_id']: r for r in cond_b[s]}
            for scen in sorted(set(by_scen_a) & set(by_scen_b)):
                a.append(float(by_scen_a[scen][m]))
                b.append(float(by_scen_b[scen][m]))
        if len(a) < 2:
            rows_out.append({'comparison': label, 'metric': m, 'n_pairs': len(a),
                             'n_seeds': len(seeds), 'mean_diff': float('nan'),
                             'wilcoxon_p': '', 'cliffs_delta': ''})
            continue
        if all(x == y for x, y in zip(a, b)):
            rows_out.append({
                'comparison': label, 'metric': m, 'n_pairs': len(a),
                'n_seeds': len(seeds), 'mean_diff': 0.0,
                'wilcoxon_p': 'n/a (all diffs zero)', 'cliffs_delta': 0.0,
            })
            continue
        w, p = wilcoxon(a, b)
        rows_out.append({
            'comparison': label, 'metric': m, 'n_pairs': len(a),
            'n_seeds': len(seeds),
            'mean_diff': round(st.mean(x - y for x, y in zip(a, b)), 5),
            'wilcoxon_p': f'{p:.4g}', 'cliffs_delta': round(cliffs_delta(a, b), 4),
        })
    return rows_out


def holm_correct(rows, p_key='wilcoxon_p'):
    def _as_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    with_p = [(i, _as_float(r[p_key])) for i, r in enumerate(rows)]
    with_p = [(i, p) for i, p in with_p if p is not None]
    with_p.sort(key=lambda t: t[1])
    m = len(with_p)
    for rank, (i, p) in enumerate(with_p):
        adj = min(1.0, p * (m - rank))
        rows[i]['holm_p'] = round(adj, 5)
    for r in rows:
        r.setdefault('holm_p', '')
    return rows


def action_histogram(cond_data: dict[int, list[dict]]):
    # test_metrics.csv doesn't carry per-step actions; report n_intervals_100/
    # 170 and n_switches as the available action-usage proxies instead.
    out = {}
    for seed, rows in cond_data.items():
        out[seed] = {
            'mean_n_intervals_100': st.mean(float(r['n_intervals_100']) for r in rows),
            'mean_n_intervals_170': st.mean(float(r['n_intervals_170']) for r in rows),
            'mean_n_switches': st.mean(float(r['n_switches']) for r in rows),
        }
    return out


def main():
    conditions = ['A0', 'A1', 'A3', 'A0_baseline']
    data = {c: load_condition(c) for c in conditions}
    for c in conditions:
        print(f'{c}: seeds present = {sorted(data[c].keys())}')

    summary_rows = []
    for c in conditions:
        if not data[c]:
            continue
        s = summarize(data[c])
        for m in METRICS:
            summary_rows.append({
                'condition': c, 'metric': m, 'n_seeds': len(data[c]),
                'mean': round(s[m]['mean'], 5), 'std': round(s[m]['std'], 5),
                'ci95_lo': round(s[m]['ci95_lo'], 5), 'ci95_hi': round(s[m]['ci95_hi'], 5),
            })

    out_dir = ROOT / 'results' / 'summary'
    out_dir.mkdir(parents=True, exist_ok=True)
    from atomic_io import atomic_write

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=['condition', 'metric', 'n_seeds', 'mean',
                                           'std', 'ci95_lo', 'ci95_hi'])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    atomic_write(out_dir / 'E05_ablation.csv', _w, newline='')
    print(f'\nwrote results/summary/E05_ablation.csv ({len(summary_rows)} rows)')

    if data.get('A0') and data.get('A1') and data.get('A3'):
        # granularity='seed_mean': paired on seed (one mean/seed) -- n=n_seeds.
        # granularity='scenario_level': paired on (scenario_id, seed) -- n up
        # to n_seeds x 2,700, the same unit results/summary/E03_paired_tests.csv
        # uses. The two are NOT interchangeable; every row is labelled.
        rows = paired_test(data['A3'], data['A1'], 'A3-A1 (intelligent selection)')
        rows += paired_test(data['A1'], data['A0'], 'A1-A0 (exploration increase)')
        rows += paired_test(data['A3'], data['A0'], 'A3-A0 (guided vs no guidance)')
        for r in rows:
            r['granularity'] = 'seed_mean'
            r['n_pairs'] = r['n_seeds']
            r['note'] = SEED_MEAN_MIN_P_NOTE

        scen_rows = paired_test_scenario_level(data['A3'], data['A1'], 'A3-A1 (intelligent selection)')
        scen_rows += paired_test_scenario_level(data['A1'], data['A0'], 'A1-A0 (exploration increase)')
        scen_rows += paired_test_scenario_level(data['A3'], data['A0'], 'A3-A0 (guided vs no guidance)')
        for r in scen_rows:
            r['granularity'] = 'scenario_level'
            r['note'] = ''

        if data.get('A0_baseline'):
            # A0 (ga_prob=0 via _train_guided_by_optimizer) vs E3 v2 Regular
            # (ga_prob=0 via train()) -- neither uses guidance, so this
            # isolates the *training-schedule* difference (eps schedule,
            # replay warm-up, target-sync -- Sec 1-1) from any guidance
            # effect. Directly feeds R3-3 (source of the improvement).
            r0 = paired_test(data['A0'], data['A0_baseline'],
                             'A0-Regular (training schedule effect, no guidance either side)')
            for r in r0:
                r['granularity'] = 'seed_mean'
                r['n_pairs'] = r['n_seeds']
                r['note'] = SEED_MEAN_MIN_P_NOTE
            rows += r0
            scen0 = paired_test_scenario_level(
                data['A0'], data['A0_baseline'],
                'A0-Regular (training schedule effect, no guidance either side)')
            for r in scen0:
                r['granularity'] = 'scenario_level'
                r['note'] = ''
            scen_rows += scen0
        else:
            print('A0_baseline (E3 v2 Regular) not found -- skipping A0-Regular test')

        # Holm correction is applied within each granularity separately --
        # the two granularities test different null hypotheses (seed-level
        # vs scenario-level pairing) and mixing them into one family would
        # not be a meaningful correction.
        rows = holm_correct(rows)
        scen_rows = holm_correct(scen_rows)
        # scenario_level listed first: it is the primary granularity per
        # the n=5 Wilcoxon-floor note above; seed_mean follows as reference.
        all_rows = scen_rows + rows

        fields = ['granularity', 'comparison', 'metric', 'n_pairs', 'n_seeds',
                  'mean_diff', 'wilcoxon_p', 'holm_p', 'cliffs_delta', 'note']

        def _wt(fd):
            w = csv.DictWriter(fd, fieldnames=fields)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k, '') for k in fields})

        atomic_write(out_dir / 'E05_stage1_tests.csv', _wt, newline='')
        print(f'wrote results/summary/E05_stage1_tests.csv ({len(all_rows)} rows, '
              f'scenario_level (primary) + seed_mean (reference) side by side)')
        for r in all_rows:
            print(r)

        print('\naction-usage proxies (mean per scenario, per seed):')
        for c in ('A1', 'A3'):
            print(f'  {c}:', action_histogram(data[c]))
    else:
        print('\nA0/A1/A3 not all present yet -- stage-1 tests skipped.')


if __name__ == '__main__':
    main()
