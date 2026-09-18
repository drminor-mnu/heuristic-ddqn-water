#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-e Task F-2: action-degeneracy check for the 6 / 12 / 32-combo action
space definitions, using the *current* (unchanged) objective_function tau at
real decision states drawn from actual trajectories.

12-combo (mathematically distinct, (3+1)x(2+1)) and 32-combo (individually
identified, 2^5) action sets are constructed here purely for this scoping
calculation; they are not wired into any evaluation/training path and the
6-combo action space used everywhere else in the repo is unchanged.

Run from src/.
"""
import csv
import itertools
import os
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E04_DIR = ROOT / 'results' / 'E04_enum'

NEW_WEIGHTS = [0.40, 0.25, 0.25, 0.10]


def build_action_sets():
    # 32: every boolean vector of length 5 (individually identified pumps).
    actions32 = [list(bits) for bits in itertools.product([False, True], repeat=5)]
    # 12: mathematically distinct combos -- (n100 on, n170 on) for
    # n100 in 0..3, n170 in 0..2, realised as "first n100 / first n170" (same
    # canonical convention as the repo's Actions0/Actions1).
    actions12 = []
    for n100 in range(4):
        for n170 in range(3):
            v = [False] * 5
            for i in range(n100):
                v[i] = True
            for i in range(n170):
                v[3 + i] = True
            actions12.append(v)
    from water_gym import Actions0
    return actions32, actions12, list(Actions0)


def tau(weights, pumpq, prev_vec, act_vec, vol, level, inflow, minutes=2):
    from water_gym import switching_stability
    pump_rate = sum(q for q, on in zip(pumpq, act_vec) if on)
    qsum = sum(pumpq)
    attempted = pump_rate * minutes
    available = vol + inflow
    excess = 1.0 if attempted > available else 0.0
    actual = min(attempted, available)
    w1, w2, w3, w4 = weights
    vol_reward = (actual / qsum) * (level / 10.0)
    act_reward = switching_stability(prev_vec, act_vec)
    energy_reward = 1.0 - pump_rate / qsum
    return w1 * vol_reward + w2 * act_reward + w3 * energy_reward + w4 * excess * -1.0


def main():
    os.chdir(str(SRC))
    from water_gym import pumpq
    from run_e04e_taskA import build_states  # reuse the same real-state sampler

    actions32, actions12, actions6 = build_action_sets()
    states = build_states()
    print(f'{len(states)} real decision states (reused from Task A)')

    rows = []
    for s in states:
        vol, level, prev6 = s['vol'], s['level'], s['prev_action']
        prev_vec = actions6[prev6]
        inflow = float(s['outfalls'][s['clock']])
        for label, actions in (('6', actions6), ('12', actions12), ('32', actions32)):
            taus = [round(tau(NEW_WEIGHTS, pumpq, prev_vec, a, vol, level, inflow), 10)
                    for a in actions]
            n_unique = len(set(taus))
            rows.append({
                'state_id': f"{s['scenario_id']}_step{s['step']}",
                'action_space': label, 'n_actions': len(actions),
                'n_unique_tau': n_unique,
                'effective_fraction': round(n_unique / len(actions), 4),
            })

    out_path = E04_DIR / 'action_degeneracy_w2.csv'
    from atomic_io import atomic_write

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    atomic_write(out_path, _w, newline='')

    import statistics as st
    print(f'\nwrote {out_path.relative_to(ROOT)}')
    for label in ('6', '12', '32'):
        vals = [r['n_unique_tau'] for r in rows if r['action_space'] == label]
        print(f'  action_space={label:>3}: n_unique_tau mean={st.mean(vals):.2f} '
              f'min={min(vals)} max={max(vals)} (n_actions={vals and rows[0]})')


if __name__ == '__main__':
    main()
