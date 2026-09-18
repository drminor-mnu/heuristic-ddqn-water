#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate Figure 3 (Regular DDQN operation) and Figure 4 (GA-guided
DDQN operation) using the ORIGINAL manuscript plotting function,
graph_draw.plot_individual_case(state_list, action_list), unmodified.

This script only CALLS plot_individual_case() -- it does not alter
graph_draw.py in any way. plot_individual_case() itself ends with
plt.show() (no savefig()); under the Agg backend plt.show() is a
no-op, so the current figure remains available via plt.gcf() right
after the call, which is what we savefig() here.

Action axis: plot_individual_case() displays
action = action_list + 1 (0-5 internal -> 1-6 displayed). 2026-09-05:
briefly replaced with a local, un-offset copy of this function per a
user instruction, then reverted the same day -- the manuscript's 1-6
display convention is correct, confirmed by direct comparison against
results_org/images/figure4.png (Action Type axis runs 1-6 there).

Scenario / models: same as src/run_e14_fig3_4.py (30yr_0060m_h211,
Regular = results/E14_two_criteria/Regular/seed1/model.pkl
(2-criterion, w=[0.5,0.5,0,0]), GA-guided =
results/E03_seeds/GA-guided/seed1/model.pkl (4-criterion, new
weights [0.40,0.25,0.25,0.10])).

Output: results/summary/figures_final/figure3.png, figure4.png.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
RESULTS = ROOT / 'results'

SCENARIO_INP = 'data/gasan/30year/30yr_0060m_h211.inp'
SCENARIO_ID = '30yr_0060m_h211'
NEW_WEIGHTS_2C = [0.50, 0.50, 0.0, 0.0]
NEW_WEIGHTS_4C = [0.40, 0.25, 0.25, 0.10]

# match results_org/images/figure4.png (2131x834 px @ 72.009 dpi)
SAVE_DPI = 72


def main():
    os.chdir(str(SRC))

    import dqn_from_demon_v1 as ddqn
    from graph_draw import plot_individual_case

    out_dir = RESULTS / 'summary' / 'figures_final'
    out_dir.mkdir(parents=True, exist_ok=True)

    regular_model = RESULTS / 'E14_two_criteria' / 'Regular' / 'seed1' / 'model.pkl'
    ga_model = RESULTS / 'E03_seeds' / 'GA-guided' / 'seed1' / 'model.pkl'

    # -- Figure 3: Regular DDQN (2-criterion) --
    _, actions, states, _, _ = ddqn.test_model(
        None, SCENARIO_INP, minutes=2, model_path=str(regular_model),
        weights=NEW_WEIGHTS_2C, act_type=0)
    plot_individual_case(states, actions)  # unmodified original function; ends in plt.show() (no-op under Agg)
    fig3_path = out_dir / 'figure3.png'
    plt.gcf().savefig(fig3_path, dpi=SAVE_DPI)
    plt.close('all')
    print(f'wrote {fig3_path} (scenario={SCENARIO_ID}, model=Regular seed1, '
          f'max_level={max(s[-1] for s in states):.6f}, '
          f'action range(raw 0-5)={min(actions)}-{max(actions)}, '
          f'displayed range(+1)={min(actions)+1}-{max(actions)+1})')

    # -- Figure 4: GA-guided DDQN (4-criterion) --
    _, actions, states, _, _ = ddqn.test_model(
        None, SCENARIO_INP, minutes=2, model_path=str(ga_model),
        weights=NEW_WEIGHTS_4C, act_type=0)
    plot_individual_case(states, actions)
    fig4_path = out_dir / 'figure4.png'
    plt.gcf().savefig(fig4_path, dpi=SAVE_DPI)
    plt.close('all')
    print(f'wrote {fig4_path} (scenario={SCENARIO_ID}, model=GA-guided seed1, '
          f'max_level={max(s[-1] for s in states):.6f}, '
          f'action range(raw 0-5)={min(actions)}-{max(actions)}, '
          f'displayed range(+1)={min(actions)+1}-{max(actions)+1})')


if __name__ == '__main__':
    main()
