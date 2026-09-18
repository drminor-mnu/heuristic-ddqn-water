#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E10 item 2 (R4-12): decompose per-decision DDQN inference time into
(a) neural-net forward pass only, (b) + state preprocessing (min-max
normalization + 5-step sequence tensor construction), (c) + environment
step overhead (WaterGym.step() -- action decoding, water-balance update,
reward/info computation; CONFIRMED to not call SWMM, see
docs/E04E_REPORT.md / water_gym.py -- SWMM runs only once per episode
inside WaterGym.reset()), and (d) SWMM's one-time per-episode cost
(reset()) as a reference, both as a per-episode total and amortized per
decision.

Reproduces dqn_from_demon_v1.test_model()'s loop line-by-line with finer
timers inserted -- no existing function modified.

Run from src/.
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
DECISION_INTERVAL_S = 120


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


def time_detailed(model_path, test_inp, minutes=2, weights=NEW_WEIGHTS, act_type=0):
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
    swmm_time_s = time.perf_counter() - t0  # (d) one-time per-episode SWMM cost

    input_q = deque([], ddqn.nstates)
    for _ in range(ddqn.nstates - 1):
        input_q.append(np.array([0.0 for _ in range(len(init_state_))]))
    state_ = ddqn._input_norm(init_state_)
    input_q.append(state_)
    state = torch.tensor(np.stack(input_q), dtype=torch.float32, device=ddqn.device)

    # NOTE: no torch.no_grad() here -- dqn_from_demon_v1.test_model() does not
    # use it either (line-by-line fidelity to the actual deployed code path).
    t_a = t_b = t_c = 0.0  # (a) forward pass, (b) preprocessing, (c) env-step overhead
    n_decisions = 0
    while True:
        # (a) forward pass only
        ta0 = time.perf_counter()
        q_val, _ = model(state.unsqueeze(0))
        t_a += time.perf_counter() - ta0

        # action-selection post-processing (counted into (c) below, matches
        # what the real loop does immediately after the forward pass)
        tc0 = time.perf_counter()
        q_val_ = torch.Tensor.cpu(q_val).data.numpy()
        action = np.argmax(q_val_)
        state_, reward, done, info = gym.step(action)  # confirmed SWMM-free
        t_c += time.perf_counter() - tc0

        # (b) state preprocessing: normalization + sequence tensor build
        tb0 = time.perf_counter()
        state_n = ddqn._input_norm(state_)
        input_q.append(state_n)
        state = torch.tensor(np.stack(input_q), dtype=torch.float32, device=ddqn.device)
        t_b += time.perf_counter() - tb0

        n_decisions += 1
        if done:
            break

    return {
        'n_decisions': n_decisions,
        'a_forward_pass_s': t_a,
        'b_preprocessing_s': t_b,
        'c_env_step_overhead_s': t_c,
        'd_swmm_reset_s': swmm_time_s,
        'ab_s': t_a + t_b,
        'abc_s': t_a + t_b + t_c,
        'abc_d_s': t_a + t_b + t_c + swmm_time_s,
    }


def main():
    import os
    os.chdir(str(SRC))
    import data_paths

    _, _, test_inps = data_paths.load_fixed_split()
    scenarios = stratified_subsample(test_inps, SAMPLE_PER_STRATUM)
    print(f'{len(scenarios)} scenarios (stratified, {SAMPLE_PER_STRATUM}/stratum), seed{SEED}')

    rows = []
    for cond in ('Regular', 'GA-guided', 'PSO-guided'):
        mp = E03_DIR / cond / f'seed{SEED}' / 'model.pkl'
        if not mp.exists():
            print(f'{cond}: checkpoint not found, skipping')
            continue
        for inp in scenarios:
            r = time_detailed(str(mp), inp)
            r['condition'] = cond
            r['scenario'] = Path(inp).stem
            rows.append(r)
        print(f'{cond}: {len(scenarios)} scenarios timed')

    out_dir = ROOT / 'results' / 'summary'
    from atomic_io import atomic_write

    fields = ['condition', 'scenario', 'n_decisions', 'a_forward_pass_s', 'b_preprocessing_s',
              'c_env_step_overhead_s', 'd_swmm_reset_s', 'ab_s', 'abc_s', 'abc_d_s']

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})

    atomic_write(out_dir / 'E10_timing_detailed.csv', _w, newline='')
    print(f'\nwrote results/summary/E10_timing_detailed.csv ({len(rows)} rows)')


if __name__ == '__main__':
    main()
