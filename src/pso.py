#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Particle swarm optimization for the pump-control problem."""

import numpy as np

from genetic_algo import GeneticAlgorithm
from water_gym import WaterGym


class ParticleSwarmOptimization(GeneticAlgorithm):
    """Discrete PSO using the same objective and action space as the GA."""

    def __init__(self, **params):
        super().__init__(**params)
        self.swarm_size = params.get("swarm_size", 30)
        self.num_iterations = params.get("num_iterations", 100)
        self.inertia = params.get("inertia", 0.7)
        self.cognitive = params.get("cognitive", 1.5)
        self.social = params.get("social", 1.5)
        self.patience = params.get("patience", 10)
        self.min_improvement = params.get("min_improvement", 1e-6)

    def _nearest_valid_action(self, position):
        """Project a continuous particle position onto a valid pump action."""
        actions = np.asarray(self.Actions, dtype=float)
        distances = np.sum((actions - position) ** 2, axis=1)
        return int(np.argmin(distances))

    def particle_swarm_optimization(self, state, action_list, next_inflow=0.0):
        dimensions = len(self.Actions[0])
        action_fitness = np.array([
            self.objective_function(action, state, action_list, next_inflow)
            for action in range(len(self.Actions))
        ])
        positions = np.random.uniform(0.0, 1.0, (self.swarm_size, dimensions))
        velocities = np.random.uniform(-1.0, 1.0, (self.swarm_size, dimensions))

        action_indices = np.array(
            [self._nearest_valid_action(position) for position in positions]
        )
        fitness = action_fitness[action_indices]

        personal_best_positions = positions.copy()
        personal_best_fitness = fitness.copy()
        best_particle = int(np.argmax(fitness))
        global_best_position = positions[best_particle].copy()
        global_best_action = int(action_indices[best_particle])
        global_best_fitness = float(fitness[best_particle])
        no_improve_count = 0

        for _ in range(self.num_iterations):
            r1 = np.random.random((self.swarm_size, dimensions))
            r2 = np.random.random((self.swarm_size, dimensions))
            velocities = (
                self.inertia * velocities
                + self.cognitive * r1 * (personal_best_positions - positions)
                + self.social * r2 * (global_best_position - positions)
            )
            positions = np.clip(positions + velocities, 0.0, 1.0)

            action_indices = np.array(
                [self._nearest_valid_action(position) for position in positions]
            )
            fitness = action_fitness[action_indices]

            improved = fitness > personal_best_fitness
            personal_best_positions[improved] = positions[improved]
            personal_best_fitness[improved] = fitness[improved]

            iteration_best = int(np.argmax(fitness))
            if fitness[iteration_best] > global_best_fitness + self.min_improvement:
                global_best_fitness = float(fitness[iteration_best])
                global_best_position = positions[iteration_best].copy()
                global_best_action = int(action_indices[iteration_best])
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.patience:
                break

        return global_best_action

    def run_pso(self, test_inp):
        """Run PSO control over an entire WaterGym simulation."""
        gym = WaterGym(test_inp, minutes=self.minutes,
                       weights=self.weights, act_type=self.act_type)
        state, reward, done, info = gym.reset()

        action_list = [0]
        state_list = [state]
        reward_list = [reward]
        info_list = [info]

        while not done:
            action = self.particle_swarm_optimization(
                state, action_list, next_inflow=gym.outfalls[gym.clock + 1]
            )
            action_list.append(action)
            state, reward, done, info = gym.step(action)
            state_list.append(state)
            reward_list.append(reward)
            info_list.append(info)

        return action_list, state_list, reward_list, info_list


# Short alias for convenient imports.
PSO = ParticleSwarmOptimization
