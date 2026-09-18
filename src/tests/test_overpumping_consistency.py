import ast
from pathlib import Path
import sys

import numpy as np


SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from genetic_algo import GeneticAlgorithm
from pso import ParticleSwarmOptimization
from water_gym import Actions0, Level, Volume, pumpq


def _volume_to_level(volume):
    if volume < 0:
        return Level[0]
    for index in range(len(Volume) - 1):
        if volume < Volume[index + 1]:
            break
    else:
        index += 1
    if index < 6:
        return Level[index] + (
            (Level[index + 1] - Level[index])
            / (Volume[index + 1] - Volume[index])
            * (volume - Volume[index])
        )
    return Level[index]


def test_all_active_evaluators_accumulate_overpumping():
    tree = ast.parse((SRC / "perform_evaluate.py").read_text(encoding="utf-8"))
    bad_assignments = []

    def contains_overpump_name(node):
        return any(
            isinstance(child, ast.Name) and child.id == "counts_overpump"
            for child in ast.walk(node)
        )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Subscript) and contains_overpump_name(target)
            for target in node.targets
        ):
            bad_assignments.append(node.lineno)
    assert bad_assignments == []


def test_ga_objective_matches_environment_next_step_reward():
    weights = [0.25, 0.4, 0.25, 0.1]
    ga = GeneticAlgorithm(minutes=2, weights=weights, act_type=0)
    state = [0.0, 0.0, 0.0, 150.0, Level[0]]
    next_inflow = 25.0
    action = 2
    previous_action = 1

    attempted_outflow = np.asarray(pumpq)[Actions0[action]].sum() * 2
    available_volume = state[3] + next_inflow
    overpump = float(attempted_outflow > available_volume)
    actual_outflow = min(attempted_outflow, available_volume)
    next_volume = max(available_volume - attempted_outflow, 0.0)
    next_level = _volume_to_level(next_volume)
    vol_reward = (actual_outflow / np.asarray(pumpq).sum()) * (next_level / Level[-1])
    act_reward = (
        np.asarray(Actions0[previous_action]) & np.asarray(Actions0[action])
    ).sum() / len(Actions0[action])
    energy_reward = 1.0 - np.asarray(pumpq)[Actions0[action]].sum() / np.asarray(pumpq).sum()
    expected = (
        weights[0] * vol_reward
        + weights[1] * act_reward
        + weights[2] * energy_reward
        - weights[3] * overpump
    )

    actual = ga.objective_function(action, state, [previous_action], next_inflow=next_inflow)
    assert actual == expected


def test_pso_forwards_next_inflow_to_objective():
    pso = ParticleSwarmOptimization(swarm_size=4, num_iterations=1, patience=1)
    received = []

    def objective(action, state, action_list, next_inflow=0.0):
        received.append(next_inflow)
        return float(action)

    pso.objective_function = objective
    pso.particle_swarm_optimization([0, 0, 0, 0, Level[0]], [0], next_inflow=37.0)
    assert received == [37.0] * len(pso.Actions)
