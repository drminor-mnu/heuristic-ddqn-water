#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local search optimization for the pump-control problem."""

import numpy as np

from genetic_algo import GeneticAlgorithm
from water_gym import WaterGym


class LocalSearch(GeneticAlgorithm):
    """Multi-start hill climbing over valid pump configurations."""

    def __init__(self, **params):
        super().__init__(**params)
        self.num_restarts = params.get("num_restarts", 10)
        self.max_iterations = params.get("max_iterations", 100)
        self.neighborhood_size = params.get("neighborhood_size", 1)
        self.min_improvement = params.get("min_improvement", 1e-6)

    def _neighbors(self, action):
        """Return valid actions within the configured Hamming distance."""
        actions = np.asarray(self.Actions, dtype=bool)
        current = actions[action]
        distances = np.count_nonzero(actions != current, axis=1)
        indices = np.flatnonzero(
            (distances > 0) & (distances <= self.neighborhood_size)
        )

        # Some constrained action sets have no one-bit neighbor. In that case,
        # use the closest valid configurations so the search can still move.
        if indices.size == 0:
            positive = distances[distances > 0]
            if positive.size:
                indices = np.flatnonzero(distances == positive.min())
        return indices

    def local_search(self, state, action_list):
        fitness = lambda action: self.objective_function(action, state, action_list)
        starts = [int(action_list[-1])]
        if self.num_restarts > 1:
            starts.extend(
                np.random.randint(0, len(self.Actions), size=self.num_restarts - 1).tolist()
            )

        best_action = starts[0]
        best_fitness = fitness(best_action)

        for start in starts:
            current_action = start
            current_fitness = fitness(current_action)

            for _ in range(self.max_iterations):
                neighbors = self._neighbors(current_action)
                if neighbors.size == 0:
                    break
                neighbor_fitness = np.array([fitness(int(action)) for action in neighbors])
                candidate_index = int(np.argmax(neighbor_fitness))
                candidate_action = int(neighbors[candidate_index])
                candidate_fitness = float(neighbor_fitness[candidate_index])

                if candidate_fitness <= current_fitness + self.min_improvement:
                    break
                current_action = candidate_action
                current_fitness = candidate_fitness

            if current_fitness > best_fitness:
                best_action = current_action
                best_fitness = current_fitness

        return best_action

    def run_local_search(self, test_inp):
        """Run local-search control over an entire WaterGym simulation."""
        gym = WaterGym(test_inp, minutes=self.minutes,
                       weights=self.weights, act_type=self.act_type)
        state, reward, done, info = gym.reset()

        action_list = [0]
        state_list = [state]
        reward_list = [reward]
        info_list = [info]

        while not done:
            action = self.local_search(state, action_list)
            action_list.append(action)
            state, reward, done, info = gym.step(action)
            state_list.append(state)
            reward_list.append(reward)
            info_list.append(info)

        return action_list, state_list, reward_list, info_list
