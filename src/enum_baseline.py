#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exhaustive-enumeration (ENUM) baseline for the 1-step pump-control objective.

E4-a (docs/REVISION_EXPERIMENT_PLAN.md, ``## E4``).  Reviewers R1-1 / R4-1 note
that with only 6 feasible pump combinations, scoring all 6 with the 1-step
objective ``tau`` is faster *and* an exact optimum compared to running GA/PSO.
This class makes that baseline explicit: at every decision step it evaluates
``GeneticAlgorithm.objective_function`` -- the same ``tau`` that GA and PSO
maximize (docs/E02_ARGMINMAX.md) -- for every feasible action and returns the
argmax.

Drop-in for ``GeneticAlgorithm``: it subclasses it and overrides *only* the
per-decision search, so the inherited ``run_genetic_algo()`` (full-episode
rollout used by ``perform_evaluate._evaluate_heuristic``) and the guided-training
hook (``dqn_from_demon_v1._train_guided_by_optimizer`` with
``optimizer_method='genetic_algorithm'``) both work unchanged.  Nothing in
``genetic_algo.py`` is modified.

Deterministic: given ``(state, action_list, next_inflow)`` the returned action is
a pure function -- no RNG is touched -- so two runs on the same scenario are
bit-identical.  Ties are broken by lowest action index (``np.argmax``), i.e. the
combination with the fewest / lowest-capacity pumps, matching the tie behaviour
the GA search converges to.
"""
import numpy as np

from genetic_algo import GeneticAlgorithm


class EnumSearch(GeneticAlgorithm):
    """Pick the feasible action maximizing the 1-step objective ``tau`` (exact)."""

    def score_all_actions(self, state, action_list, next_inflow=0.0):
        """Return the tau value of every feasible action, indexed by action id."""
        return np.array([
            self.objective_function(action, state, action_list, next_inflow)
            for action in range(len(self.Actions))
        ])

    def genetic_algorithm(self, state, action_list, next_inflow=0.0):
        """Override the GA search with an exact argmax over all feasible actions.

        Keeps the method name so the inherited ``run_genetic_algo`` and the
        generic guided-training hook call into this without any other change.
        """
        scores = self.score_all_actions(state, action_list, next_inflow)
        return int(np.argmax(scores))

    # Readable alias for scripts that want to say what they mean.
    def enumerate_best(self, state, action_list, next_inflow=0.0):
        return self.genetic_algorithm(state, action_list, next_inflow)


if __name__ == '__main__':
    import data_paths

    _, _, test_inps = data_paths.load_fixed_split()
    params = {'minutes': 2, 'weights': [0.40, 0.25, 0.25, 0.10], 'act_type': 0}
    enum = EnumSearch(**params)
    actions, states, rewards, infos = enum.run_genetic_algo(test_inps[0])
    print(f'{test_inps[0]}: {len(actions)} steps, '
          f'max_level={max(s[-1] for s in states):.3f}, '
          f'cum_reward={np.sum(rewards):.3f}, '
          f'action histogram={np.bincount(actions, minlength=len(enum.Actions)).tolist()}')
