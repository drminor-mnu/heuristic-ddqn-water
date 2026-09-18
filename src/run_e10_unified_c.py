#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Table 5 / Figure 10 unification (option C, user decision 2026-09-09,
see results/summary/TABLE5_FIGURE10_CONSISTENCY.md). Measures all five
models -- Regular(E03 v2, 4-criterion), GA-guided, PSO-guided, GA-only,
PSO-only -- plus Regular(E14, 2-criterion, for Table 5) on the SAME
per-decision metric: (a) forward pass / heuristic decision + (b)
preprocessing + (c) environment-step overhead, EXCLUDING the
once-per-episode SWMM cost (d). Repeated 3x for mean +- sd. No existing
evaluation function is modified -- DDQN timing duplicates
run_e10_timing_detailed.py's/run_e14_timing.py's instrumented loop
line-by-line (their own stated pattern); GA/PSO-only timing is obtained
by temporarily monkey-patching WaterGym.reset (restored immediately
after each call) to separate the once-per-episode SWMM cost from the
rest of run_genetic_algo()/run_pso(), without editing genetic_algo.py,
pso.py, or water_gym.py.

BUG FOUND (pre-existing, in src/run_e10_timing.py's time_heuristic(),
NOT modified here): that function calls `g.run_genetic_algo(test_inp)`
unconditionally for both GA-only and PSO-only. ParticleSwarmOptimization
does not override run_genetic_algo() or genetic_algorithm() -- it only
defines run_pso()/particle_swarm_optimization(). So the "PSO-only" rows
in results/summary/E10_timing_breakdown.csv actually measured
GeneticAlgorithm's inherited search method, not PSO. This does NOT
affect any of the paper's substantive PSO-only results (Table 11-16 use
perform_evaluate.evaluate_pso(), which correctly dispatches to
run_pso() via run_method='run_pso') -- it is confined to that one timing
script. Documented in results/summary/TABLE5_FIGURE10_CONSISTENCY.md;
run_e10_timing.py itself is left unmodified (not a function this task
was asked to fix, and silent refactors are avoided in this project).

Run from src/, with the `torch` conda env active.
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
E14_DIR = ROOT / 'results' / 'E14_two_criteria'
SEED = 1
SAMPLE_PER_STRATUM = 1
NEW_WEIGHTS_4C = [0.40, 0.25, 0.25, 0.10]
WEIGHTS_2C = [0.50, 0.50, 0.0, 0.0]
N_REPEATS = 3


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


def time_dqn_c(model_path, test_inp, minutes, weights, act_type=0):
    """(a)+(b)+(c), SWMM excluded. Duplicated line-by-line from
    run_e10_timing_detailed.py's time_detailed() -- see module docstring."""
    import dqn_from_demon_v1 as ddqn
    from water_gym import WaterGym

    actions = ddqn.Actions0 if act_type == 0 else ddqn.Actions1
    model = ddqn.DqnGRU(input_dim=5, hidden_dim=ddqn.hidden_dim,
                         output_dim=len(actions), num_layers=ddqn.num_layers).to(ddqn.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    gym = WaterGym(test_inp, minutes, weights, act_type=act_type)
    init_state_, reward, done, info = gym.reset()  # (d), excluded below

    input_q = deque([], ddqn.nstates)
    for _ in range(ddqn.nstates - 1):
        input_q.append(np.array([0.0 for _ in range(len(init_state_))]))
    state_ = ddqn._input_norm(init_state_)
    input_q.append(state_)
    state = torch.tensor(np.stack(input_q), dtype=torch.float32, device=ddqn.device)

    t_abc = 0.0
    n_decisions = 0
    while True:
        ta0 = time.perf_counter()
        q_val, _ = model(state.unsqueeze(0))
        q_val_ = torch.Tensor.cpu(q_val).data.numpy()
        action = np.argmax(q_val_)
        state_, reward, done, info = gym.step(action)
        state_n = ddqn._input_norm(state_)
        input_q.append(state_n)
        state = torch.tensor(np.stack(input_q), dtype=torch.float32, device=ddqn.device)
        t_abc += time.perf_counter() - ta0
        n_decisions += 1
        if done:
            break

    return t_abc, n_decisions


def time_heuristic_c(instance, run_method_name, test_inp):
    """(a)+(b)+(c)-equivalent for GA-only/PSO-only: total wall-clock of
    the existing run_genetic_algo()/run_pso() call, minus the
    once-per-episode SWMM cost, isolated by temporarily monkey-patching
    WaterGym.reset (restored immediately after, in a finally block) --
    genetic_algo.py/pso.py/water_gym.py are not edited."""
    from water_gym import WaterGym
    original_reset = WaterGym.reset
    captured = {'swmm_s': 0.0}

    def timed_reset(self):
        t0 = time.perf_counter()
        result = original_reset(self)
        captured['swmm_s'] = time.perf_counter() - t0
        return result

    WaterGym.reset = timed_reset
    try:
        t0 = time.perf_counter()
        run_method = getattr(instance, run_method_name)  # correct per-class dispatch
        actions, states, rewards, infos = run_method(test_inp)
        total_s = time.perf_counter() - t0
    finally:
        WaterGym.reset = original_reset

    n_decisions = len(actions) - 1
    abc_s = total_s - captured['swmm_s']
    return abc_s, n_decisions, captured['swmm_s']


def run_one_repeat(scenarios):
    """Returns {condition: (pooled_abc_s, pooled_n_decisions)} for one repeat."""
    import dqn_from_demon_v1  # noqa: F401 (side-effect imports match existing scripts' pattern)
    from genetic_algo import GeneticAlgorithm
    from pso import ParticleSwarmOptimization

    results = {}

    dqn_conditions = [
        ('Regular', E03_DIR / 'Regular' / f'seed{SEED}' / 'model.pkl', NEW_WEIGHTS_4C),
        ('GA-guided', E03_DIR / 'GA-guided' / f'seed{SEED}' / 'model.pkl', NEW_WEIGHTS_4C),
        ('PSO-guided', E03_DIR / 'PSO-guided' / f'seed{SEED}' / 'model.pkl', NEW_WEIGHTS_4C),
        ('Regular(2-criterion,E14)', E14_DIR / 'Regular' / f'seed{SEED}' / 'model.pkl', WEIGHTS_2C),
    ]
    for cond, mp, weights in dqn_conditions:
        if not mp.exists():
            print(f'{cond}: checkpoint not found at {mp}, skipping')
            continue
        abc_sum, n_sum = 0.0, 0
        for inp in scenarios:
            t, n = time_dqn_c(str(mp), inp, minutes=2, weights=weights)
            abc_sum += t
            n_sum += n
        results[cond] = (abc_sum, n_sum)

    heuristic_conditions = [
        ('GA-only', GeneticAlgorithm, 'run_genetic_algo'),
        ('PSO-only', ParticleSwarmOptimization, 'run_pso'),
    ]
    for cond, cls, method_name in heuristic_conditions:
        abc_sum, n_sum, swmm_sum = 0.0, 0, 0.0
        for j, inp in enumerate(scenarios):
            print(f'\r{cond} {j + 1}/{len(scenarios)}', end='', flush=True)
            instance = cls(minutes=2, weights=NEW_WEIGHTS_4C, act_type=0)
            t, n, swmm = time_heuristic_c(instance, method_name, inp)
            abc_sum += t
            n_sum += n
            swmm_sum += swmm
        print()
        results[cond] = (abc_sum, n_sum)
        print(f'{cond}: pooled swmm_time_s={swmm_sum:.3f}s over {len(scenarios)} scenarios '
              f'({swmm_sum / len(scenarios) * 1000:.1f} ms/scenario avg) -- excluded from (c)')

    return results


def main():
    import os
    os.chdir(str(SRC))
    import data_paths

    _, _, test_inps = data_paths.load_fixed_split()
    scenarios = stratified_subsample(test_inps, SAMPLE_PER_STRATUM)
    print(f'{len(scenarios)} scenarios (stratified, {SAMPLE_PER_STRATUM}/stratum), seed{SEED}, '
          f'{N_REPEATS} repeats')

    rows = []  # condition, repeat, abc_s_pooled, n_decisions_pooled, ms_per_decision
    for repeat in range(1, N_REPEATS + 1):
        print(f'\n=== repeat {repeat}/{N_REPEATS} ===')
        res = run_one_repeat(scenarios)
        for cond, (abc_s, n) in res.items():
            ms = abc_s / n * 1000
            rows.append({'condition': cond, 'repeat': repeat, 'abc_s_pooled': abc_s,
                         'n_decisions_pooled': n, 'ms_per_decision': ms})
            print(f'  repeat {repeat} {cond}: {ms:.4f} ms/decision (n={n})')

    out_dir = ROOT / 'results' / 'summary'
    csv_path = out_dir / 'E10_unified_c_repeats.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['condition', 'repeat', 'abc_s_pooled',
                                           'n_decisions_pooled', 'ms_per_decision'])
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {csv_path}')

    # summary mean+-sd across the 3 repeats, per condition
    import statistics
    by_cond = {}
    for r in rows:
        by_cond.setdefault(r['condition'], []).append(r['ms_per_decision'])
    summary_path = out_dir / 'E10_unified_c_summary.csv'
    with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'n_repeats', 'mean_ms_per_decision', 'sd_ms_per_decision', 'values'])
        for cond, vals in by_cond.items():
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            w.writerow([cond, len(vals), f'{mean:.4f}', f'{sd:.4f}',
                        ';'.join(f'{v:.4f}' for v in vals)])
            print(f'{cond}: {mean:.4f} +- {sd:.4f} ms/decision (n={len(vals)} repeats)')
    print(f'wrote {summary_path}')


if __name__ == '__main__':
    main()
