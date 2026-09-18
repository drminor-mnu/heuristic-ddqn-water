#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Section 3.1 regeneration: Figure 3 (2-criterion Regular DDQN operation)
and Figure 4 (4-criterion GA-guided DDQN operation) for a single example
scenario, 30-year return period / 60 min duration -- manuscript captions:
"Operation of the two-criterion Regular DDQN for a 60 min rainfall event
with a 30-year return period" / "...under the same scenario as Figure 3."

The original scenario_id is lost (no raw provenance survives, see
docs/response/SECTION_3_1_REGENERATION.md). These figures illustrate
operating behavior, not statistical representativeness (the manuscript
does not claim the single scenario is representative), so a new scenario
from the same stratum is a like-for-like substitute.

Scenario selection (deterministic, reproducible):
    candidates = sorted(30yr_0060m test scenarios in
                         data/splits/fixed_split_seed42.json), 50 total
    chosen = random.Random(42).choice(candidates)
    -> data/gasan/30year/30yr_0060m_h211.inp (scenario_id: 30yr_0060m_h211)
This selection is fixed here as a constant, not re-randomized on each run.

Run from src/, only after E14 has produced at least one Regular seed (or
with --regular-source e3 for a format smoke test -- NOT a valid
regeneration, output written to --out-dir, not the real summary path).
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
RESULTS = ROOT / 'results'

# Fixed scenario selection -- see module docstring for how this was chosen.
SCENARIO_INP = 'data/gasan/30year/30yr_0060m_h211.inp'
SCENARIO_ID = '30yr_0060m_h211'
NEW_WEIGHTS_2C = [0.50, 0.50, 0.0, 0.0]
NEW_WEIGHTS_4C = [0.40, 0.25, 0.25, 0.10]


def rollout(model_path, weights, act_type=0, minutes=2):
    import dqn_from_demon_v1 as ddqn
    _, actions, states, rewards, infos = ddqn.test_model(
        None, SCENARIO_INP, minutes=minutes, model_path=str(model_path),
        weights=weights, act_type=act_type)
    return actions, states, rewards, infos


def plot_operation(actions, states, title, out_path):
    levels = [s[-1] for s in states]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(levels, color='tab:blue')
    ax1.set_ylabel('basin water level (m)')
    ax1.set_title(title)
    ax2.step(range(len(actions)), actions, where='post', color='tab:orange')
    ax2.set_ylabel('pump action index')
    ax2.set_xlabel('decision step (2 min interval)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'wrote {out_path}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--regular-source', choices=['e14', 'e3'], default='e14')
    p.add_argument('--regular-seed', type=int, default=1)
    p.add_argument('--ga-seed', type=int, default=1)
    p.add_argument('--out-dir', type=str, default=None)
    return p.parse_args()


def main():
    import os
    os.chdir(str(SRC))
    args = parse_args()

    if args.regular_source == 'e14':
        regular_model = RESULTS / 'E14_two_criteria' / 'Regular' / f'seed{args.regular_seed}' / 'model.pkl'
        regular_weights = NEW_WEIGHTS_2C
        regular_label = 'Regular(2-criterion)'
    else:
        regular_model = RESULTS / 'E03_seeds' / 'Regular' / f'seed{args.regular_seed}' / 'model.pkl'
        regular_weights = NEW_WEIGHTS_4C  # E3 v2 Regular is 4-criterion
        regular_label = 'Regular(SMOKE-TEST-4-criterion-stand-in)'
        print('*** SMOKE TEST MODE -- not a valid Figure 3/4 regeneration. ***')

    ga_model = RESULTS / 'E03_seeds' / 'GA-guided' / f'seed{args.ga_seed}' / 'model.pkl'

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS / 'summary'
    out_dir.mkdir(parents=True, exist_ok=True)

    reg_actions, reg_states, _, _ = rollout(regular_model, regular_weights)
    plot_operation(reg_actions, reg_states,
                   f'Figure 3 (regenerated): {regular_label} operation, {SCENARIO_ID}',
                   out_dir / 'E14_figure3_regular_operation.png')

    ga_actions, ga_states, _, _ = rollout(ga_model, NEW_WEIGHTS_4C)
    plot_operation(ga_actions, ga_states,
                   f'Figure 4 (regenerated): GA-guided(4-criterion) operation, {SCENARIO_ID}',
                   out_dir / 'E14_figure4_ga_guided_operation.png')


if __name__ == '__main__':
    main()
