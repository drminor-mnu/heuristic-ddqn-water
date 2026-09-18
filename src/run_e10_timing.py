#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E10 -- computation-time decomposition (R4-12, R2-2).

Deliberately does NOT call perform_evaluate.evaluate_dpn_gru(train=False) --
that function has a confirmed pre-existing bug (basic_losses/basic_rewards
referenced at its final return statement are only assigned inside the
`if train:` branch, so any train=False call raises UnboundLocalError; see
docs/REMAINING_WORK.md item 1). No code in this repo's actual results-
generating paths calls it with train=False (run_e03_seeds.py and
diag_e03_validate.py both pass train=True), so E3 v2 is unaffected by this
bug -- it is confirmed dormant, not fixed. Under this project's
no-refactoring rule,
this script works around it instead of patching perform_evaluate.py: it
reuses dqn_from_demon_v1.test_model()'s own loop structure directly,
inserting perf_counter() calls around gym.reset() (the only place SimSwmm
runs, confirmed in water_gym.py -- WaterGym.step() is pure Python/numpy,
no SWMM call) and around each Q-network forward pass.

Reuses existing E3 v2 GA-guided/PSO-guided/Regular seed1 checkpoints --
no training. Evaluates a stratified subsample (not all 2,700 scenarios)
to bound wall-clock cost of this measurement itself.
"""
import csv
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E03_DIR = ROOT / 'results' / 'E03_seeds'
SEED = 1
SAMPLE_PER_STRATUM = 1
NEW_WEIGHTS = [0.40, 0.25, 0.25, 0.10]
DECISION_INTERVAL_S = 120  # 2 minutes, per manuscript's decision cadence


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


def time_dqn_model(model_path, test_inp, minutes=2, weights=NEW_WEIGHTS, act_type=0):
    """Reproduces dqn_from_demon_v1.test_model()'s loop with instrumentation.
    Does not modify or import from a patched copy -- duplicated here only for
    timing instrumentation, matching the existing function's logic line by
    line (verified against dqn_from_demon_v1.py:919-977)."""
    import dqn_from_demon_v1 as ddqn
    from water_gym import WaterGym

    actions = ddqn.Actions0 if act_type == 0 else ddqn.Actions1
    model = ddqn.DqnGRU(input_dim=5, hidden_dim=ddqn.hidden_dim,
                         output_dim=len(actions), num_layers=ddqn.num_layers).to(ddqn.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    gym = WaterGym(test_inp, minutes, weights, act_type=act_type)
    t0 = time.perf_counter()
    init_state_, reward, done, info = gym.reset()
    swmm_time_s = time.perf_counter() - t0

    input_q = deque([], ddqn.nstates)
    for _ in range(ddqn.nstates - 1):
        input_q.append(np.array([0.0 for _ in range(len(init_state_))]))
    state_ = ddqn._input_norm(init_state_)
    input_q.append(state_)
    state = torch.tensor(np.stack(input_q), dtype=torch.float32, device=ddqn.device)

    inference_time_s = 0.0
    n_decisions = 0
    t_total0 = time.perf_counter()
    while True:
        ti = time.perf_counter()
        q_val, _ = model(state.unsqueeze(0))
        q_val_ = torch.Tensor.cpu(q_val).data.numpy()
        action = np.argmax(q_val_)
        inference_time_s += time.perf_counter() - ti
        n_decisions += 1

        state_, reward, done, info = gym.step(action)
        state_ = ddqn._input_norm(state_)
        input_q.append(state_)
        state = torch.tensor(np.stack(input_q), dtype=torch.float32, device=ddqn.device)
        if done:
            break
    total_time_s = time.perf_counter() - t0

    return {
        'swmm_time_s': swmm_time_s,
        'inference_time_s': inference_time_s,
        'total_time_s': total_time_s,
        'n_decisions': n_decisions,
        'ms_per_decision_inference_only': (inference_time_s / n_decisions * 1000) if n_decisions else float('nan'),
    }


def time_heuristic(guide_cls, test_inp, minutes=2, weights=NEW_WEIGHTS, act_type=0):
    """GA/PSO-only: no DQN, no checkpoint -- the guide decides every step via
    its own run_genetic_algo() (also calls WaterGym.reset() once internally)."""
    import time as _t
    g = guide_cls(minutes=minutes, weights=weights, act_type=act_type)
    t0 = _t.perf_counter()
    actions, states, rewards, infos = g.run_genetic_algo(test_inp)
    total_time_s = _t.perf_counter() - t0
    n_decisions = len(actions) - 1
    return {
        'swmm_time_s': '',  # not separated out for GA/PSO-only (single call boundary not instrumented)
        'inference_time_s': '',
        'total_time_s': total_time_s,
        'n_decisions': n_decisions,
        'ms_per_decision_inference_only': (total_time_s / n_decisions * 1000) if n_decisions else float('nan'),
    }


def main():
    import os
    os.chdir(str(SRC))
    import data_paths
    from genetic_algo import GeneticAlgorithm
    from pso import ParticleSwarmOptimization

    _, _, test_inps = data_paths.load_fixed_split()
    scenarios = stratified_subsample(test_inps, SAMPLE_PER_STRATUM)
    print(f'{len(scenarios)} scenarios (stratified, {SAMPLE_PER_STRATUM}/stratum), seed{SEED}')

    rows = []
    for cond in ('Regular', 'GA-guided', 'PSO-guided'):
        mp = E03_DIR / cond / f'seed{SEED}' / 'model.pkl'
        if not mp.exists():
            print(f'{cond}: checkpoint not found at {mp}, skipping')
            continue
        for inp in scenarios:
            r = time_dqn_model(str(mp), inp)
            r['condition'] = cond
            r['scenario'] = Path(inp).stem
            r['margin_x'] = DECISION_INTERVAL_S / (r['total_time_s'] / r['n_decisions'])
            rows.append(r)
        print(f'{cond}: {len(scenarios)} scenarios timed')

    for label, cls in (('GA-only', GeneticAlgorithm), ('PSO-only', ParticleSwarmOptimization)):
        for j, inp in enumerate(scenarios):
            print(f'\r{label} {j+1}/{len(scenarios)}', end='', flush=True)
            r = time_heuristic(cls, inp)
            r['condition'] = label
            r['scenario'] = Path(inp).stem
            r['margin_x'] = DECISION_INTERVAL_S / (r['total_time_s'] / r['n_decisions'])
            rows.append(r)
        print(f'{label}: {len(scenarios)} scenarios timed')

    out_dir = ROOT / 'results' / 'summary'
    from atomic_io import atomic_write

    fields = ['condition', 'scenario', 'n_decisions', 'swmm_time_s', 'inference_time_s',
              'total_time_s', 'ms_per_decision_inference_only', 'margin_x']

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})

    atomic_write(out_dir / 'E10_timing_breakdown.csv', _w, newline='')
    print(f'\nwrote results/summary/E10_timing_breakdown.csv ({len(rows)} rows)')


if __name__ == '__main__':
    main()
