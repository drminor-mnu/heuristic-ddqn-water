#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Section 3.1 regeneration, Table 5 (calculation-time column): same
(a)-(d) decomposition as src/run_e10_timing_detailed.py, applied to the
2-criterion Regular DDQN (E14, w=[0.5,0.5,0,0]) instead of the
4-criterion models. run_e10_timing_detailed.py hardcodes its condition
list to E03_seeds/{Regular,GA-guided,PSO-guided}, so it cannot be reused
directly for E14 -- this is a separate, additive script with identical
methodology (same sample, same seed, same hardware) so the two are
directly comparable.

(a) neural-net forward pass only, (b) + state preprocessing (min-max
normalization + 5-step sequence tensor construction), (c) + environment
step overhead (WaterGym.step(), confirmed SWMM-free), (d) SWMM's
one-time per-episode cost (WaterGym.reset()).

Run from src/, after E14 has produced results/E14_two_criteria/Regular/seed1/model.pkl.
"""
import csv
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E14_DIR = ROOT / 'results' / 'E14_two_criteria'
SEED = 1
SAMPLE_PER_STRATUM = 1  # matches run_e10_timing_detailed.py exactly
TWO_CRITERION_WEIGHTS = [0.50, 0.50, 0.0, 0.0]
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


def time_detailed(model_path, test_inp, minutes=2, weights=TWO_CRITERION_WEIGHTS, act_type=0):
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

    t_a = t_b = t_c = 0.0
    n_decisions = 0
    while True:
        ta0 = time.perf_counter()
        q_val, _ = model(state.unsqueeze(0))
        t_a += time.perf_counter() - ta0

        tc0 = time.perf_counter()
        q_val_ = torch.Tensor.cpu(q_val).data.numpy()
        action = np.argmax(q_val_)
        state_, reward, done, info = gym.step(action)
        t_c += time.perf_counter() - tc0

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

    mp = E14_DIR / 'Regular' / f'seed{SEED}' / 'model.pkl'
    if not mp.exists():
        print(f'checkpoint not found at {mp}')
        return
    rows = []
    for inp in scenarios:
        r = time_detailed(str(mp), inp)
        r['condition'] = 'Regular(2-criterion)'
        r['scenario'] = Path(inp).stem
        rows.append(r)
    print(f'Regular(2-criterion): {len(scenarios)} scenarios timed')

    out_dir = ROOT / 'results' / 'summary'
    from atomic_io import atomic_write

    fields = ['condition', 'scenario', 'n_decisions', 'a_forward_pass_s', 'b_preprocessing_s',
              'c_env_step_overhead_s', 'd_swmm_reset_s', 'ab_s', 'abc_s', 'abc_d_s']

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})

    atomic_write(out_dir / 'E14_timing_detailed.csv', _w, newline='')
    print(f'\nwrote results/summary/E14_timing_detailed.csv ({len(rows)} rows)')

    n_dec_total = sum(r['n_decisions'] for r in rows)
    a = sum(r['a_forward_pass_s'] for r in rows)
    b = sum(r['b_preprocessing_s'] for r in rows)
    c = sum(r['c_env_step_overhead_s'] for r in rows)
    d = sum(r['d_swmm_reset_s'] for r in rows)
    n_ep = len(rows)
    print(f'n_decisions total={n_dec_total}, n_episodes={n_ep}')
    print(f'(a) forward only: {a/n_dec_total*1000:.4f} ms/decision')
    print(f'(a+b) +preprocess: {(a+b)/n_dec_total*1000:.4f} ms/decision')
    print(f'(a+b+c) +env overhead: {(a+b+c)/n_dec_total*1000:.4f} ms/decision')
    print(f'(d) SWMM reset: {d/n_ep:.4f} s/episode = {d/n_dec_total*1000:.4f} ms/decision (amortized)')
    print(f'(a+b+c+d): {(a+b+c+d)/n_dec_total*1000:.4f} ms/decision')
    margin_c = DECISION_INTERVAL_S * 1000 / ((a + b + c) / n_dec_total * 1000)
    margin_d = DECISION_INTERVAL_S * 1000 / ((a + b + c + d) / n_dec_total * 1000)
    print(f'margin vs 120s, (c)-based: {margin_c:.0f}x')
    print(f'margin vs 120s, (c)+(d)-based: {margin_d:.0f}x')


if __name__ == '__main__':
    main()
