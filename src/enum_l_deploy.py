#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deployable (non-batched) L-step exhaustive enumeration -- timing reference.

E4-b follow-up (2026-08-30): ``lookahead_enum.LookaheadEnum`` scores all
``n_actions ** L`` candidate sequences as one batched numpy array op, which
hides the branching factor inside vectorised arithmetic and is *not* what a
naive "just enumerate the L-step tree" implementation looks like. This module
is that naive implementation: for every candidate sequence it calls the
unmodified, unbatched ``GeneticAlgorithm.objective_function()`` once per
(candidate, step) inside a plain Python loop -- exactly the per-action
scoring GA/PSO/EnumSearch already use at L=1, just repeated over the full
tree. This is the complexity R1-1/R4-1's "just enumerate" argument is about,
and the one that should be quoted against the 120 s decision interval.

Verified against LookaheadEnum (bit-identical best-sequence choice at L<=5 on
a probe state) before being used purely for wall-clock timing -- it is not
used for the episode sweep (too slow at L>=7; see run_e04_horizon.py, which
uses the vectorised planner for the 6^L-blowup-independent metrics: overflow,
max_level, lock rate).
"""
import itertools
import math

import numpy as np

from genetic_algo import GeneticAlgorithm
from water_gym import pumpq


class DeployedLookaheadEnum(GeneticAlgorithm):
    """Exhaustive L-step search, one un-batched objective_function() call per
    (candidate sequence, step) -- O(n_actions**L) Python-level work."""

    def __init__(self, **params):
        super().__init__(**params)
        self.L = params.get('L', 1)
        self.gamma_h = params.get('gamma_h', 1.0)

    def plan(self, state, action_list, inflow_forecast):
        n = len(self.Actions)
        best_total = -math.inf
        best_first = 0
        for seq in itertools.product(range(n), repeat=self.L):
            vol = state[3]
            level = state[-1]
            hist = list(action_list)
            pstate = list(state)
            total = 0.0
            g = 1.0
            for k, a in enumerate(seq):
                pstate[3] = vol
                pstate[-1] = level
                total += g * self.objective_function(a, pstate, hist,
                                                      inflow_forecast[k])
                pump_rate = np.array(pumpq)[self.Actions[a]].sum()
                attempted = pump_rate * self.minutes
                available = vol + inflow_forecast[k]
                vol = max(available - attempted, 0.0)
                level = self._volume_to_level(vol)
                hist.append(a)
                g *= self.gamma_h
            if total > best_total:
                best_total = total
                best_first = seq[0]
        return best_first


if __name__ == '__main__':
    import time

    from lookahead_enum import LookaheadEnum

    OLD_W = [0.25, 0.40, 0.25, 0.10]
    vol0, level0, prev, inflow = 8000.0, 8.0, 0, 50.0

    print('cross-check vs vectorised LookaheadEnum:')
    for L in (1, 2, 3, 4):
        dep = DeployedLookaheadEnum(minutes=2, weights=OLD_W, L=L, gamma_h=1.0)
        vec = LookaheadEnum(OLD_W, minutes=2, L=L, gamma_h=1.0)
        fc = [inflow] * L
        state = [0.0, 0.0, 0.0, vol0, level0]
        a_dep = dep.plan(state, [prev], fc)
        a_vec = vec.plan(vol0, level0, prev, fc)
        print(f'  L={L}: deployed={a_dep}  vectorised={a_vec}  match={a_dep == a_vec}')

    DECISION_BUDGET_S = 120.0
    print(f'\nsingle-decision wall-clock, deployable (unbatched) reference '
          f'(decision budget = {DECISION_BUDGET_S:.0f} s):')
    for L in (1, 2, 3, 4, 5, 6):
        dep = DeployedLookaheadEnum(minutes=2, weights=OLD_W, L=L, gamma_h=1.0)
        fc = [inflow] * L
        state = [0.0, 0.0, 0.0, vol0, level0]
        t0 = time.perf_counter()
        a = dep.plan(state, [prev], fc)
        dt = time.perf_counter() - t0
        n_cand = 6 ** L
        margin = DECISION_BUDGET_S / dt if dt > 0 else float('inf')
        print(f'  L={L}  N=6^L={n_cand:>10,}  time={dt:9.4f}s  '
              f'margin(120s/t)={margin:12,.0f}x  action={a}')
