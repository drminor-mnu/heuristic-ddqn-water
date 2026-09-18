#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5 Task A1 control: uniform-random action guide.

docs/REVISION_EXPERIMENT_PLAN.md E5, A1: "GA 대신 np.random.randint(0, 6)로
가이던스를 대체". Isolates whether the online-guidance performance gain comes
from GA's *intelligent* action selection or merely from the *increased
exploration diversity* injected whenever the guide fires.

Same drop-in convention as enum_baseline.EnumSearch: subclasses
GeneticAlgorithm purely for __init__ (Actions/weights/minutes), and
overrides only `genetic_algorithm()` so it plugs into
dqn_from_demon_v1._train_guided_by_optimizer's existing
optimizer_class/optimizer_method hook unchanged -- no other code is
modified. Consumes exactly one np.random draw per guided decision (GA/PSO
consume many per decision across population/generations, so the two runs'
random streams diverge after the first guided step -- inherent to what is
being compared, not a bug).
"""
import numpy as np

from genetic_algo import GeneticAlgorithm


class RandomGuide(GeneticAlgorithm):
    """Ignore the state/objective entirely; return a uniform-random feasible
    action. Same call signature as GeneticAlgorithm.genetic_algorithm() so
    `_train_guided_by_optimizer(optimizer_class=RandomGuide,
    optimizer_method='genetic_algorithm', ...)` needs no other change."""

    def genetic_algorithm(self, state, action_list, next_inflow=0.0):
        return int(np.random.randint(0, len(self.Actions)))


if __name__ == '__main__':
    import data_paths

    _, _, test_inps = data_paths.load_fixed_split()
    params = {'minutes': 2, 'weights': [0.40, 0.25, 0.25, 0.10], 'act_type': 0}
    guide = RandomGuide(**params)
    actions, states, rewards, infos = guide.run_genetic_algo(test_inps[0])
    print(f'{test_inps[0]}: {len(actions)} steps, '
          f'max_level={max(s[-1] for s in states):.3f}, '
          f'action histogram={np.bincount(actions, minlength=len(guide.Actions)).tolist()}')
