#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5 Task 3-1: temporal structure of action sequences -- run lengths,
lag-1 autocorrelation, P(a_t != a_{t-1}).

No per-step "was this step guided" log exists anywhere (train_log.csv is
episode,train_loss,train_reward only; _train_guided_by_optimizer computes
a per-episode guided_count but never persists it; the guided/exogenous
identity of each training step is not recoverable post hoc without
re-running training with new instrumentation, which this script does not
do). Two things ARE computable without training:

  (a) "guide-only" trajectories: RandomGuide / GeneticAlgorithm making
      every decision autonomously over real test scenarios (their existing
      run_genetic_algo(), unmodified) -- the closest available proxy for
      "what does the guide itself produce", NOT the literal in-training
      guided-step subsequence (labelled as such throughout).
  (b) trained-policy trajectories: test_model() forward pass on each
      condition's saved checkpoint (same mechanism as run_e05_action_hist.py)
      -- the policy's own behaviour, no guidance at test time (matches how
      the models are actually evaluated).

Run from src/, after results/E05_ablation/{A0,A1}/seed{1,2,3}/model.pkl and
results/E03_seeds/GA-guided/seed{1,2,3}/model.pkl exist.
"""
import csv
import os
import statistics as st
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E05_DIR = ROOT / 'results' / 'E05_ablation'
E03_DIR = ROOT / 'results' / 'E03_seeds'
NEW_WEIGHTS = [0.40, 0.25, 0.25, 0.10]
SEEDS = [1, 2, 3, 4, 5]
SAMPLE_PER_STRATUM = 2   # same subsample as run_e05_action_hist.py


def stratified_subsample(test_inps, per_stratum):
    import re
    from collections import defaultdict
    buckets = defaultdict(list)
    for p in test_inps:
        n = re.findall(r'\d+', p.split('/')[-1])
        buckets[(n[0], n[1])].append(p)
    picked = []
    for k in sorted(buckets):
        picked.extend(buckets[k][:per_stratum])
    return picked


def _pearson(x, y):
    # statistics.correlation() needs Python >= 3.10; this repo runs 3.9.
    n = len(x)
    if n < 2:
        return float('nan')
    mx, my = st.mean(x), st.mean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx == 0 or vy == 0:
        return float('nan')
    return cov / (vx ** 0.5 * vy ** 0.5)


def run_stats(actions: list[int]) -> dict:
    """actions: a single scenario's action sequence (post-reset, len >= 2)."""
    n = len(actions)
    # run lengths
    runs = []
    cur = 1
    for i in range(1, n):
        if actions[i] == actions[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    # P(change)
    n_change = sum(1 for i in range(1, n) if actions[i] != actions[i - 1])
    p_change = n_change / (n - 1) if n > 1 else float('nan')
    # lag-1 Pearson autocorrelation (action index treated as numeric --
    # defensible here since actions are an ordered pump-capacity ladder)
    if n > 2:
        x = actions[:-1]
        y = actions[1:]
        r = _pearson(x, y)
    else:
        r = float('nan')
    return {'runs': runs, 'p_change': p_change, 'lag1_autocorr': r,
           'n_decisions': n - 1}


def aggregate(scenario_stats: list[dict]) -> dict:
    all_runs = [r for s in scenario_stats for r in s['runs']]
    p_changes = [s['p_change'] for s in scenario_stats if s['p_change'] == s['p_change']]
    autocorrs = [s['lag1_autocorr'] for s in scenario_stats
                if s['lag1_autocorr'] == s['lag1_autocorr']]
    runs_sorted = sorted(all_runs)

    def pct(p):
        if not runs_sorted:
            return float('nan')
        idx = min(len(runs_sorted) - 1, int(round(p * (len(runs_sorted) - 1))))
        return runs_sorted[idx]

    return {
        'n_scenarios': len(scenario_stats),
        'n_runs': len(all_runs),
        'run_len_mean': st.mean(all_runs) if all_runs else float('nan'),
        'run_len_median': pct(0.5),
        'run_len_p90': pct(0.9),
        'p_change_mean': st.mean(p_changes) if p_changes else float('nan'),
        'lag1_autocorr_mean': st.mean(autocorrs) if autocorrs else float('nan'),
    }


def guide_only_stats(scenarios, guide_cls, label):
    from water_gym import Level  # noqa: F401  (import-time sanity only)

    rows = []
    for inp in scenarios:
        g = guide_cls(minutes=2, weights=NEW_WEIGHTS, act_type=0)
        actions, states, rewards, infos = g.run_genetic_algo(inp)
        rows.append(run_stats(actions[1:]))
    agg = aggregate(rows)
    agg['label'] = label
    print(f'  guide-only {label}: {agg}')
    return agg


def policy_stats(scenarios, cond, seeds, label_prefix):
    import dqn_from_demon_v1 as ddqn

    per_seed_aggs = []
    for seed in seeds:
        mp = (E03_DIR / 'GA-guided' / f'seed{seed}' / 'model.pkl' if cond == 'A3'
              else E05_DIR / cond / f'seed{seed}' / 'model.pkl')
        if not mp.exists():
            print(f'  {label_prefix} seed{seed}: model.pkl not found, skipping')
            continue
        rows = []
        for inp in scenarios:
            _, actions, states, rewards, infos = ddqn.test_model(
                None, inp, minutes=2, model_path=str(mp),
                weights=NEW_WEIGHTS, act_type=0)
            rows.append(run_stats(actions[1:]))
        agg = aggregate(rows)
        agg['label'] = f'{label_prefix}_seed{seed}'
        print(f'  policy {label_prefix} seed{seed}: {agg}')
        per_seed_aggs.append(agg)
    return per_seed_aggs


def main():
    os.chdir(str(SRC))
    import data_paths

    _, _, test_inps = data_paths.load_fixed_split()
    scenarios = stratified_subsample(test_inps, SAMPLE_PER_STRATUM)
    print(f'{len(scenarios)} scenarios (stratified, {SAMPLE_PER_STRATUM}/stratum)\n')

    from genetic_algo import GeneticAlgorithm
    from random_guide import RandomGuide

    print('[guide-only trajectories -- proxy for "the guide itself", NOT the '
          'literal in-training guided-step subsequence]')
    guide_rows = [
        guide_only_stats(scenarios, RandomGuide, 'RandomGuide-only (A1 guide)'),
        guide_only_stats(scenarios, GeneticAlgorithm, 'GA-only (A3 guide)'),
    ]

    print('\n[trained-policy trajectories -- test_model() forward pass, no guidance]')
    policy_rows = []
    policy_rows += policy_stats(scenarios, 'A0', SEEDS, 'A0')
    policy_rows += policy_stats(scenarios, 'A1', SEEDS, 'A1')
    policy_rows += policy_stats(scenarios, 'A3', SEEDS, 'A3')

    out_path = ROOT / 'results' / 'summary' / 'E05_temporal.csv'
    fields = ['label', 'n_scenarios', 'n_runs', 'run_len_mean', 'run_len_median',
              'run_len_p90', 'p_change_mean', 'lag1_autocorr_mean']
    from atomic_io import atomic_write

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=fields)
        w.writeheader()
        for r in guide_rows + policy_rows:
            w.writerow({k: r[k] for k in fields})

    atomic_write(out_path, _w, newline='')
    print(f'\nwrote {out_path.relative_to(ROOT)} ({len(guide_rows) + len(policy_rows)} rows)')


if __name__ == '__main__':
    main()
