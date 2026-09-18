#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L-step lookahead exhaustive enumeration over the cumulative 1-step objective.

E4-§3 (docs/REVISION_EXPERIMENT_PLAN.md ``#### E4-§3``): does the action-0
fixed point of the 1-step objective (docs/E00_HEURISTIC_MYOPIA_FINDING.md,
old-manuscript weights [0.25, 0.40, 0.25, 0.10]) dissolve once the planning
horizon is L > 1 ?

At each decision the planner brute-forces all ``n_actions ** L`` action
sequences, scores each with

    tau_L = sum_{k=0..L-1} gamma_h**k * tau(a_k, x_hat_k, hist_k)

where ``tau`` is exactly ``GeneticAlgorithm.objective_function`` and the
predicted states ``x_hat_k`` are rolled forward with the *same* closed-form
water balance the environment uses (WaterGym.step has no SWMM call --
docs/REVISION_EXPERIMENT_PLAN.md E4-b), so there is no model mismatch.  The
only modelling choice is the horizon inflow forecast, supplied by the caller:

  * perfect  -- the true SWMM inflow hydrograph outfalls[clock .. clock+L-1]
  * persistence -- the one-step-ahead inflow held flat across the horizon
                   (so L=1 perfect == L=1 persistence == the deployed 1-step
                   controller / enum_baseline.EnumSearch)

Receding horizon: only the first action of the best sequence is executed, then
the plan is recomputed.

Vectorised: the whole ``n_actions ** L`` sweep is numpy array arithmetic, so
L=5 (7776 sequences) costs ~100 us per decision.  The scalar reference
``objective_function`` is reproduced exactly (verified against
enum_baseline.EnumSearch at L=1).  Ties -> lowest action index (np.argmax),
i.e. the fixed point keeps action 0 on a tie, the conservative choice.
"""
import itertools

import numpy as np

from water_gym import (Actions0, Actions1, Level, Volume, pumpq,
                       switching_stability)

EXCESS_PUMP_PENALTY = -1.0


class LookaheadEnum:
    def __init__(self, weights, minutes: int = 2, act_type: int = 0,
                 L: int = 1, gamma_h: float = 1.0):
        self.weights = list(weights)
        self.w1, self.w2, self.w3, self.w4 = self.weights
        self.minutes = minutes
        self.act_type = act_type
        self.L = int(L)
        self.gamma_h = float(gamma_h)

        self.Actions = Actions0 if act_type == 0 else Actions1
        self.n_actions = len(self.Actions)

        pq = np.asarray(pumpq, dtype=float)
        self.Qsum = float(pq.sum())
        self.pump_rate = np.array(
            [pq[self.Actions[a]].sum() for a in range(self.n_actions)], dtype=float)
        self.attempt = self.pump_rate * self.minutes
        self.energy = 1.0 - self.pump_rate / self.Qsum
        self.SW = np.array(
            [[switching_stability(self.Actions[p], self.Actions[a])
              for a in range(self.n_actions)]
             for p in range(self.n_actions)], dtype=float)

        self.Volume = np.asarray(Volume, dtype=float)
        self.Level = np.asarray(Level, dtype=float)
        self._seq_cache: dict[int, np.ndarray] = {}

    # -- helpers ---------------------------------------------------------------

    def _sequences(self, L: int) -> np.ndarray:
        if L not in self._seq_cache:
            self._seq_cache[L] = np.array(
                list(itertools.product(range(self.n_actions), repeat=L)),
                dtype=np.int64)
        return self._seq_cache[L]

    def _vol2level(self, vol):
        # np.interp gives flat extrapolation at both ends -> matches
        # GeneticAlgorithm._volume_to_level / WaterGym.vol2level for vol >= 0
        # (verified on a dense grid).
        return np.interp(vol, self.Volume, self.Level)

    def tau1_per_action(self, vol0, level0, prev_action, inflow0):
        """The 1-step objective tau for every feasible action at one state."""
        available = vol0 + inflow0
        actual = np.minimum(self.attempt, available)
        excess = (self.attempt > available).astype(float)
        vol_reward = (actual / self.Qsum) * (level0 / self.Level[-1])
        act_reward = self.SW[int(prev_action)]
        return (self.w1 * vol_reward + self.w2 * act_reward
                + self.w3 * self.energy + self.w4 * excess * EXCESS_PUMP_PENALTY)

    # -- planning -----------------------------------------------------------

    def plan(self, vol0, level0, prev_action, inflow_forecast, return_detail=False):
        """Return the first action of the tau_L-optimal length-L sequence."""
        L = len(inflow_forecast)
        S = self._sequences(L)
        N = S.shape[0]

        vol = np.full(N, float(vol0))
        level = np.full(N, float(level0))
        prev = np.full(N, int(prev_action), dtype=np.int64)
        total = np.zeros(N)
        gpow = 1.0
        for k in range(L):
            a = S[:, k]
            inflow_k = float(inflow_forecast[k])
            available = vol + inflow_k
            attempt = self.attempt[a]
            actual = np.minimum(attempt, available)
            excess = (attempt > available).astype(float)
            vol_reward = (actual / self.Qsum) * (level / self.Level[-1])
            act_reward = self.SW[prev, a]
            r = (self.w1 * vol_reward + self.w2 * act_reward
                 + self.w3 * self.energy[a]
                 + self.w4 * excess * EXCESS_PUMP_PENALTY)
            total += gpow * r
            vol = np.maximum(available - attempt, 0.0)
            level = self._vol2level(vol)
            prev = a
            gpow *= self.gamma_h

        best = int(np.argmax(total))
        first = int(S[best, 0])
        if not return_detail:
            return first

        zero_row = int(np.ravel_multi_index((0,) * L, (self.n_actions,) * L))
        return first, {
            'first_action': first,
            'best_seq': S[best].tolist(),
            'best_total': float(total[best]),
            'all_zero_total': float(total[zero_row]),
            'best_minus_zero': float(total[best] - total[zero_row]),
            'tau1_per_action': [float(x) for x in
                                self.tau1_per_action(vol0, level0, prev_action,
                                                     float(inflow_forecast[0]))],
        }

    def score_sequences(self, seqs, vol0, level0, prev_action, inflow_forecast):
        """tau_L for an arbitrary (N, L) candidate-sequence array -- the same
        rollout `plan()` runs over `self._sequences(L)` (all n_actions**L
        sequences), exposed here so callers can score a custom subset (e.g.
        E4-e Task D's 6 constant sequences) without duplicating the loop.
        Additive: `plan()` above is unchanged."""
        seqs = np.asarray(seqs, dtype=np.int64)
        N, L = seqs.shape
        vol = np.full(N, float(vol0))
        level = np.full(N, float(level0))
        prev = np.full(N, int(prev_action), dtype=np.int64)
        total = np.zeros(N)
        gpow = 1.0
        for k in range(L):
            a = seqs[:, k]
            inflow_k = float(inflow_forecast[k])
            available = vol + inflow_k
            attempt = self.attempt[a]
            actual = np.minimum(attempt, available)
            excess = (attempt > available).astype(float)
            vol_reward = (actual / self.Qsum) * (level / self.Level[-1])
            act_reward = self.SW[prev, a]
            r = (self.w1 * vol_reward + self.w2 * act_reward
                 + self.w3 * self.energy[a]
                 + self.w4 * excess * EXCESS_PUMP_PENALTY)
            total += gpow * r
            vol = np.maximum(available - attempt, 0.0)
            level = self._vol2level(vol)
            prev = a
            gpow *= self.gamma_h
        return total

    def plan_with_terminal(self, vol0, level0, prev_action, inflow_forecast,
                           terminal_fn, return_detail=False):
        """E4-f Task G: tau_L^term = tau_L + gamma_h**L * terminal_fn(level_final).

        Same rollout as `plan()` (unchanged, duplicated here) over all
        n_actions**L sequences, plus a terminal value applied to the level
        reached after L steps. `terminal_fn(level_array) -> value_array`.
        Additive -- `plan()` is not touched."""
        L = len(inflow_forecast)
        S = self._sequences(L)
        N = S.shape[0]

        vol = np.full(N, float(vol0))
        level = np.full(N, float(level0))
        prev = np.full(N, int(prev_action), dtype=np.int64)
        total = np.zeros(N)
        gpow = 1.0
        for k in range(L):
            a = S[:, k]
            inflow_k = float(inflow_forecast[k])
            available = vol + inflow_k
            attempt = self.attempt[a]
            actual = np.minimum(attempt, available)
            excess = (attempt > available).astype(float)
            vol_reward = (actual / self.Qsum) * (level / self.Level[-1])
            act_reward = self.SW[prev, a]
            r = (self.w1 * vol_reward + self.w2 * act_reward
                 + self.w3 * self.energy[a]
                 + self.w4 * excess * EXCESS_PUMP_PENALTY)
            total += gpow * r
            vol = np.maximum(available - attempt, 0.0)
            level = self._vol2level(vol)
            prev = a
            gpow *= self.gamma_h

        base_total = total.copy()          # tau_L without the terminal term
        total = total + gpow * terminal_fn(level)

        best = int(np.argmax(total))
        first = int(S[best, 0])
        if not return_detail:
            return first
        return first, {
            'first_action': first,
            'best_seq': S[best].tolist(),
            'best_total_with_terminal': float(total[best]),
            'best_total_tau_only': float(base_total[best]),
            'terminal_value_at_best': float(gpow * terminal_fn(level)[best]),
        }


def build_forecast(outfalls, clock, length, L, mode):
    """Horizon inflow forecast for the decision taken at pre-step clock=`clock`.

    The inflow applied at this step is outfalls[clock]; forecast[k] is the
    inflow k steps further ahead.  Tail is clamped to the last value.
    """
    if mode == 'perfect':
        return [outfalls[min(clock + k, length - 1)] for k in range(L)]
    if mode == 'persistence':
        return [outfalls[min(clock, length - 1)]] * L
    raise ValueError(f'unknown forecast mode {mode!r}')


def simulate_episode(outfalls, length, planner: LookaheadEnum, forecast_mode):
    """Receding-horizon rollout replicating WaterGym.step's water balance.

    Returns (actions, max_level, n_dryrun_proxy, steps_to_first_pump).
    actions[0] is the [0] placeholder used by run_genetic_algo; real decisions
    are actions[1:].
    """
    minutes = planner.minutes
    vol = 0.0
    level = float(Level[0])
    actions = [0]
    max_level = level
    n_dry = 0
    first_pump = None
    n_decisions = length - 1
    for step in range(n_decisions):
        clock = step + 1
        fc = build_forecast(outfalls, clock, length, planner.L, forecast_mode)
        a = planner.plan(vol, level, actions[-1], fc)
        actions.append(a)
        if a != 0 and first_pump is None:
            first_pump = step

        cur_inflow = outfalls[clock]
        cur_outflow = planner.pump_rate[a] * minutes
        new_vol = vol + cur_inflow - cur_outflow
        if new_vol < 0.0:
            new_vol = 0.0
            n_dry += 1
        vol = new_vol
        level = float(np.interp(vol, planner.Volume, planner.Level))
        if level > max_level:
            max_level = level
    return actions, max_level, n_dry, first_pump


if __name__ == '__main__':
    import os
    import sys

    import data_paths
    from sim_swmm import SimSwmm

    OLD_W = [0.25, 0.40, 0.25, 0.10]
    _, _, test = data_paths.load_fixed_split()
    inp = sys.argv[1] if len(sys.argv) > 1 else next(
        p for p in test if '1440m' in p)
    rains, outfalls, length = SimSwmm(inp, 2, 'gasan').execute()
    print(f'{inp}  ({length - 1} decisions)')
    for L in (1, 2, 3, 4, 5):
        pl = LookaheadEnum(OLD_W, minutes=2, L=L, gamma_h=1.0)
        acts, mx, dry, fp = simulate_episode(outfalls, length, pl, 'perfect')
        lock = np.mean([a == 0 for a in acts[1:]])
        print(f'  L={L}  max_level={mx:6.3f}  overflow={mx >= Level[-1]}  '
              f'act0_lock={lock:5.3f}  first_pump_step={fp}  '
              f'hist={np.bincount(acts[1:], minlength=6).tolist()}')
