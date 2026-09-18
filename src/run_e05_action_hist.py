#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5 stage-1 Q4: 6-action usage histogram for A1 vs A3, from saved
checkpoints. test_metrics.csv (E3/E5 2.2-schema) does not carry per-step
action sequences, only aggregates (n_switches, n_intervals_100/170) --
this recovers full per-action counts by loading each seed's model.pkl and
running dqn_from_demon_v1.test_model() (forward pass only, no training) over
a test subsample.

Run from src/, after results/E05_ablation/A1/seed{1,2,3}/model.pkl exist
(A3 reuses results/E03_seeds/GA-guided/seed{1,2,3}/model.pkl).
"""
import csv
import os
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E05_DIR = ROOT / 'results' / 'E05_ablation'
E03_DIR = ROOT / 'results' / 'E03_seeds'
SEEDS = [1, 2, 3, 4, 5]
SAMPLE_PER_STRATUM = 2   # keep this pass cheap; full test set is 2700 scenarios


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


def model_path_for(cond: str, seed: int) -> Path:
    if cond == 'A3':
        return E03_DIR / 'GA-guided' / f'seed{seed}' / 'model.pkl'
    return E05_DIR / cond / f'seed{seed}' / 'model.pkl'


def main():
    os.chdir(str(SRC))
    import data_paths
    import dqn_from_demon_v1 as ddqn

    _, _, test_inps = data_paths.load_fixed_split()
    scenarios = stratified_subsample(test_inps, SAMPLE_PER_STRATUM)
    print(f'{len(scenarios)} scenarios (stratified, {SAMPLE_PER_STRATUM}/stratum)')

    rows = []
    for cond in ('A1', 'A3'):
        for seed in SEEDS:
            mp = model_path_for(cond, seed)
            if not mp.exists():
                print(f'  {cond} seed{seed}: model.pkl not found, skipping')
                continue
            counter = Counter()
            n_dec = 0
            for inp in scenarios:
                _, actions, states, rewards, infos = ddqn.test_model(
                    None, inp, minutes=2, model_path=str(mp),
                    weights=[0.40, 0.25, 0.25, 0.10], act_type=0)
                counter.update(actions[1:])
                n_dec += len(actions) - 1
            hist = {f'action_{a}_frac': round(counter.get(a, 0) / n_dec, 5)
                   for a in range(6)}
            rows.append({'condition': cond, 'seed': seed, 'n_decisions': n_dec, **hist})
            print(f'  {cond} seed{seed}: {hist}')

    if rows:
        out_path = ROOT / 'results' / 'summary' / 'E05_action_histogram.csv'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        from atomic_io import atomic_write

        def _w(fd):
            w = csv.DictWriter(fd, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

        atomic_write(out_path, _w, newline='')
        print(f'\nwrote results/summary/E05_action_histogram.csv ({len(rows)} rows)')
    else:
        print('\nno checkpoints found yet -- nothing written.')


if __name__ == '__main__':
    main()
