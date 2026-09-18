#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integer L-genome GA-L / PSO-L over the cumulative tau_L objective.

E4-e Task A (docs/E04E_REPORT.md): a fair implementation-matched comparison
against ENUM-L (enum_l_deploy.DeployedLookaheadEnum / lookahead_enum.
LookaheadEnum) requires GA-L and PSO-L measured under the *same* two
conditions:

  * "deployed" -- un-batched, one call into the unmodified
    GeneticAlgorithm.objective_function() per (individual/particle, step),
    exactly the per-action scoring GA/PSO already do at L=1 in genetic_algo.py
    / pso.py, just repeated over an L-length sequence.
  * "vectorised" -- the population's tau_L is evaluated as one batched numpy
    rollout per generation/iteration (same rollout arithmetic as
    lookahead_enum.LookaheadEnum, duplicated here rather than imported so
    this file is fully additive and touches no existing module).

Hyperparameters (population_size, num_generations, mutation_rate,
crossover_rate, swarm_size, num_iterations, inertia, cognitive, social,
patience, min_improvement) are taken unchanged from genetic_algo.py / pso.py
-- NOT scaled up with L, per instruction. Genome/particle dimensionality is
the only thing that changes with L: each individual/particle is a length-L
sequence of action indices in {0 .. n_actions-1} (no feasibility filter is
needed here, unlike the 5-bool pump-vector GA/PSO, because every integer in
range is already a valid action id).

Nothing in genetic_algo.py, pso.py or lookahead_enum.py is modified.
"""
import math
import random

import numpy as np

from genetic_algo import (GeneticAlgorithm, crossover_rate, mutation_rate,
                          num_generations, population_size)
from water_gym import Level, Volume, pumpq


class _CallCounting:
    """Mixin: count objective_function() calls without touching
    GeneticAlgorithm.objective_function itself. Reset _n_obj_calls=0 before
    each plan() call to get a clean per-decision count."""

    def objective_function(self, *args, **kwargs):
        self._n_obj_calls = getattr(self, '_n_obj_calls', 0) + 1
        return super().objective_function(*args, **kwargs)


# ---------------------------------------------------------------------------
# deployed (un-batched, scalar) GA-L / PSO-L
# ---------------------------------------------------------------------------

def _seq_tau_scalar(searcher, seq, state, action_list, inflow_forecast, gamma_h):
    """tau_L of one candidate sequence via L sequential, unmodified
    GeneticAlgorithm.objective_function() calls (mirrors
    enum_l_deploy.DeployedLookaheadEnum.plan's inner loop)."""
    vol = state[3]
    level = state[-1]
    hist = list(action_list)
    pstate = list(state)
    total = 0.0
    g = 1.0
    for k, a in enumerate(seq):
        pstate[3] = vol
        pstate[-1] = level
        total += g * searcher.objective_function(a, pstate, hist, inflow_forecast[k])
        pump_rate = np.array(pumpq)[searcher.Actions[a]].sum()
        attempted = pump_rate * searcher.minutes
        available = vol + inflow_forecast[k]
        vol = max(available - attempted, 0.0)
        level = searcher._volume_to_level(vol)
        hist.append(a)
        g *= gamma_h
    return total


class DeployedLookaheadGA(_CallCounting, GeneticAlgorithm):
    """Integer L-genome GA over tau_L. Same population_size/num_generations/
    mutation_rate/crossover_rate/patience/min_improvement as
    GeneticAlgorithm.genetic_algorithm() (genetic_algo.py module constants)."""

    def __init__(self, **params):
        super().__init__(**params)
        self.L = params.get('L', 1)
        self.gamma_h = params.get('gamma_h', 1.0)
        self.n_actions = len(self.Actions)

    def plan(self, state, action_list, inflow_forecast, return_detail=False):
        L = self.L
        n = self.n_actions
        patience = 10
        min_improvement = 1e-6
        best_fitness_so_far = -math.inf
        no_improve = 0

        pop = [[random.randrange(n) for _ in range(L)] for _ in range(population_size)]
        best_seq = pop[0]

        for _gen in range(num_generations):
            fitness = [_seq_tau_scalar(self, ind, state, action_list,
                                       inflow_forecast, self.gamma_h)
                       for ind in pop]
            cur_best = max(fitness)
            if cur_best > best_fitness_so_far + min_improvement:
                best_fitness_so_far = cur_best
                best_seq = list(pop[fitness.index(cur_best)])
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

            new_pop = []
            for _ in range(0, population_size, 2):
                p1 = self._tournament(pop, fitness)
                p2 = self._tournament(pop, fitness)
                if L > 1 and random.random() < crossover_rate:
                    cx = random.randint(1, L - 1)
                    c1 = p1[:cx] + p2[cx:]
                    c2 = p2[:cx] + p1[cx:]
                else:
                    c1, c2 = list(p1), list(p2)
                for c in (c1, c2):
                    for i in range(L):
                        if random.random() < mutation_rate:
                            c[i] = random.randrange(n)
                new_pop.extend([c1, c2])
            pop = new_pop[:population_size]

        if not return_detail:
            return best_seq[0]
        return best_seq[0], {'best_seq': best_seq, 'best_total': best_fitness_so_far}

    @staticmethod
    def _tournament(pop, fitness, k=3):
        idxs = random.sample(range(len(pop)), k)
        best_i = max(idxs, key=lambda i: fitness[i])
        return pop[best_i]


class DeployedLookaheadPSO(_CallCounting, GeneticAlgorithm):
    """Integer L-genome (real-valued position, nearest-integer projection)
    PSO over tau_L. Same swarm_size/num_iterations/inertia/cognitive/social/
    patience/min_improvement as pso.py's ParticleSwarmOptimization defaults."""

    def __init__(self, **params):
        super().__init__(**params)
        self.L = params.get('L', 1)
        self.gamma_h = params.get('gamma_h', 1.0)
        self.n_actions = len(self.Actions)
        self.swarm_size = params.get('swarm_size', 30)
        self.num_iterations = params.get('num_iterations', 100)
        self.inertia = params.get('inertia', 0.7)
        self.cognitive = params.get('cognitive', 1.5)
        self.social = params.get('social', 1.5)
        self.patience = params.get('patience', 10)
        self.min_improvement = params.get('min_improvement', 1e-6)

    def _project(self, position):
        hi = self.n_actions - 1
        return [int(min(hi, max(0, round(x)))) for x in position]

    def plan(self, state, action_list, inflow_forecast, return_detail=False):
        L = self.L
        hi = self.n_actions - 1

        positions = [[random.uniform(0, hi) for _ in range(L)]
                     for _ in range(self.swarm_size)]
        velocities = [[random.uniform(-1, 1) for _ in range(L)]
                      for _ in range(self.swarm_size)]
        seqs = [self._project(p) for p in positions]
        fitness = [_seq_tau_scalar(self, s, state, action_list,
                                   inflow_forecast, self.gamma_h)
                   for s in seqs]

        pbest_pos = [list(p) for p in positions]
        pbest_fit = list(fitness)
        gbest_i = int(np.argmax(fitness))
        gbest_pos = list(positions[gbest_i])
        gbest_seq = list(seqs[gbest_i])
        gbest_fit = fitness[gbest_i]
        no_improve = 0

        for _it in range(self.num_iterations):
            for i in range(self.swarm_size):
                for d in range(L):
                    r1, r2 = random.random(), random.random()
                    velocities[i][d] = (
                        self.inertia * velocities[i][d]
                        + self.cognitive * r1 * (pbest_pos[i][d] - positions[i][d])
                        + self.social * r2 * (gbest_pos[d] - positions[i][d]))
                    positions[i][d] = min(hi, max(0, positions[i][d] + velocities[i][d]))
                seq = self._project(positions[i])
                fit = _seq_tau_scalar(self, seq, state, action_list,
                                      inflow_forecast, self.gamma_h)
                seqs[i] = seq
                fitness[i] = fit
                if fit > pbest_fit[i]:
                    pbest_fit[i] = fit
                    pbest_pos[i] = list(positions[i])

            it_best_i = int(np.argmax(fitness))
            if fitness[it_best_i] > gbest_fit + self.min_improvement:
                gbest_fit = fitness[it_best_i]
                gbest_pos = list(positions[it_best_i])
                gbest_seq = list(seqs[it_best_i])
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                break

        if not return_detail:
            return gbest_seq[0]
        return gbest_seq[0], {'best_seq': gbest_seq, 'best_total': gbest_fit}


# ---------------------------------------------------------------------------
# vectorised (batched-fitness) GA-L / PSO-L
# ---------------------------------------------------------------------------

class _VectorRollout:
    """Shared batched tau_L rollout, duplicated from lookahead_enum.LookaheadEnum
    (not imported) so this file makes no assumption about / dependency on
    that module's internals staying unchanged."""

    def _setup_rollout(self, weights, minutes, act_type):
        from water_gym import Actions0, Actions1, switching_stability
        self.w1, self.w2, self.w3, self.w4 = weights
        self.minutes = minutes
        Actions = Actions0 if act_type == 0 else Actions1
        n = len(Actions)
        pq = np.asarray(pumpq, dtype=float)
        self._Qsum = float(pq.sum())
        self._pump_rate = np.array([pq[Actions[a]].sum() for a in range(n)])
        self._attempt = self._pump_rate * minutes
        self._energy = 1.0 - self._pump_rate / self._Qsum
        self._SW = np.array(
            [[switching_stability(Actions[p], Actions[a]) for a in range(n)]
             for p in range(n)], dtype=float)
        self._Volume = np.asarray(Volume, dtype=float)
        self._Level = np.asarray(Level, dtype=float)
        self._n_eval_equiv = 0  # (candidates evaluated) x L, summed over all
                                 # _batch_tau_L calls in the most recent plan()

    def _batch_tau_L(self, seqs, vol0, level0, prev_action, inflow_forecast, gamma_h):
        """seqs: (N, L) int array -> (N,) tau_L array."""
        N, L = seqs.shape
        self._n_eval_equiv += N * L
        vol = np.full(N, float(vol0))
        level = np.full(N, float(level0))
        prev = np.full(N, int(prev_action), dtype=np.int64)
        total = np.zeros(N)
        gpow = 1.0
        for k in range(L):
            a = seqs[:, k]
            inflow_k = float(inflow_forecast[k])
            available = vol + inflow_k
            attempt = self._attempt[a]
            actual = np.minimum(attempt, available)
            excess = (attempt > available).astype(float)
            vol_reward = (actual / self._Qsum) * (level / self._Level[-1])
            act_reward = self._SW[prev, a]
            r = (self.w1 * vol_reward + self.w2 * act_reward
                 + self.w3 * self._energy[a] + self.w4 * excess * -1.0)
            total += gpow * r
            vol = np.maximum(available - attempt, 0.0)
            level = np.interp(vol, self._Volume, self._Level)
            prev = a
            gpow *= gamma_h
        return total


class VectorizedLookaheadGA(_VectorRollout):
    """Integer L-genome GA-L, population's tau_L evaluated as one batched
    numpy rollout per generation (same evaluation count as
    DeployedLookaheadGA, no per-individual Python-level objective_function
    calls). Same GA hyperparameters as genetic_algo.py."""

    def __init__(self, weights, minutes=2, act_type=0, L=1, gamma_h=1.0):
        self._setup_rollout(weights, minutes, act_type)
        self.L = L
        self.gamma_h = gamma_h
        self.n_actions = self._pump_rate.shape[0]

    def plan(self, vol0, level0, prev_action, inflow_forecast, return_detail=False):
        L = self.L
        n = self.n_actions
        self._n_eval_equiv = 0
        rng = np.random.default_rng()
        patience, min_improvement = 10, 1e-6
        pop = rng.integers(0, n, size=(population_size, L))
        best_total, best_seq = -np.inf, pop[0]
        no_improve = 0
        for _gen in range(num_generations):
            fitness = self._batch_tau_L(pop, vol0, level0, prev_action,
                                        inflow_forecast, self.gamma_h)
            i = int(np.argmax(fitness))
            if fitness[i] > best_total + min_improvement:
                best_total = float(fitness[i])
                best_seq = pop[i].copy()
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break
            new_pop = np.empty_like(pop)
            for j in range(0, population_size, 2):
                i1 = rng.choice(population_size, size=3, replace=False)
                i2 = rng.choice(population_size, size=3, replace=False)
                p1 = pop[i1[np.argmax(fitness[i1])]]
                p2 = pop[i2[np.argmax(fitness[i2])]]
                if L > 1 and rng.random() < crossover_rate:
                    cx = rng.integers(1, L)
                    c1 = np.concatenate([p1[:cx], p2[cx:]])
                    c2 = np.concatenate([p2[:cx], p1[cx:]])
                else:
                    c1, c2 = p1.copy(), p2.copy()
                for c in (c1, c2):
                    mut = rng.random(L) < mutation_rate
                    c[mut] = rng.integers(0, n, size=mut.sum())
                new_pop[j] = c1
                if j + 1 < population_size:
                    new_pop[j + 1] = c2
            pop = new_pop
        if not return_detail:
            return int(best_seq[0])
        return int(best_seq[0]), {'best_seq': best_seq.tolist(), 'best_total': best_total}


class VectorizedLookaheadPSO(_VectorRollout):
    """Integer L-genome PSO-L, swarm's tau_L evaluated as one batched numpy
    rollout per iteration. Same PSO hyperparameters as pso.py."""

    def __init__(self, weights, minutes=2, act_type=0, L=1, gamma_h=1.0,
                swarm_size=30, num_iterations=100, inertia=0.7,
                cognitive=1.5, social=1.5, patience=10, min_improvement=1e-6):
        self._setup_rollout(weights, minutes, act_type)
        self.L = L
        self.gamma_h = gamma_h
        self.n_actions = self._pump_rate.shape[0]
        self.swarm_size = swarm_size
        self.num_iterations = num_iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.patience = patience
        self.min_improvement = min_improvement

    def plan(self, vol0, level0, prev_action, inflow_forecast, return_detail=False):
        L = self.L
        hi = self.n_actions - 1
        self._n_eval_equiv = 0
        rng = np.random.default_rng()
        pos = rng.uniform(0, hi, size=(self.swarm_size, L))
        vel = rng.uniform(-1, 1, size=(self.swarm_size, L))
        seqs = np.clip(np.rint(pos), 0, hi).astype(np.int64)
        fitness = self._batch_tau_L(seqs, vol0, level0, prev_action,
                                    inflow_forecast, self.gamma_h)
        pbest_pos = pos.copy()
        pbest_fit = fitness.copy()
        gi = int(np.argmax(fitness))
        gbest_pos, gbest_seq, gbest_fit = pos[gi].copy(), seqs[gi].copy(), float(fitness[gi])
        no_improve = 0
        for _it in range(self.num_iterations):
            r1 = rng.random((self.swarm_size, L))
            r2 = rng.random((self.swarm_size, L))
            vel = (self.inertia * vel
                   + self.cognitive * r1 * (pbest_pos - pos)
                   + self.social * r2 * (gbest_pos - pos))
            pos = np.clip(pos + vel, 0, hi)
            seqs = np.clip(np.rint(pos), 0, hi).astype(np.int64)
            fitness = self._batch_tau_L(seqs, vol0, level0, prev_action,
                                        inflow_forecast, self.gamma_h)
            improved = fitness > pbest_fit
            pbest_pos[improved] = pos[improved]
            pbest_fit[improved] = fitness[improved]
            it_best = int(np.argmax(fitness))
            if fitness[it_best] > gbest_fit + self.min_improvement:
                gbest_fit = float(fitness[it_best])
                gbest_pos = pos[it_best].copy()
                gbest_seq = seqs[it_best].copy()
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                break
        if not return_detail:
            return int(gbest_seq[0])
        return int(gbest_seq[0]), {'best_seq': gbest_seq.tolist(), 'best_total': gbest_fit}


if __name__ == '__main__':
    import time

    from enum_l_deploy import DeployedLookaheadEnum
    from lookahead_enum import LookaheadEnum

    OLD_W = [0.25, 0.40, 0.25, 0.10]
    vol0, level0, prev, inflow = 8000.0, 8.0, 0, 50.0
    state = [0.0, 0.0, 0.0, vol0, level0]

    for L in (1, 2, 3):
        fc = [inflow] * L
        enum_a = LookaheadEnum(OLD_W, minutes=2, L=L, gamma_h=1.0).plan(
            vol0, level0, prev, fc)
        ga_dep = DeployedLookaheadGA(minutes=2, weights=OLD_W, L=L, gamma_h=1.0)
        a_ga = ga_dep.plan(state, [prev], fc)
        ga_vec = VectorizedLookaheadGA(OLD_W, minutes=2, L=L, gamma_h=1.0)
        a_gav = ga_vec.plan(vol0, level0, prev, fc)
        pso_dep = DeployedLookaheadPSO(minutes=2, weights=OLD_W, L=L, gamma_h=1.0)
        a_pso = pso_dep.plan(state, [prev], fc)
        print(f'L={L}: ENUM={enum_a} GA-deploy={a_ga} GA-vec={a_gav} PSO-deploy={a_pso}')
