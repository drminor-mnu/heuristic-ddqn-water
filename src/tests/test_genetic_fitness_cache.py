import numpy as np

from genetic_algo import GeneticAlgorithm


def test_cached_fitness_matches_direct_objective_evaluation():
    algorithm = GeneticAlgorithm(
        minutes=2, weights=[0.25, 0.4, 0.25, 0.1], act_type=0
    )
    population = np.asarray(
        [algorithm.Actions[index] for index in (0, 4, 4, 5, 0, 2, 2, 4)]
    )
    state = np.asarray([0.0, 0.0, 0.0, 2500.0, 6.0])
    action_list = [0, 4]
    expected = np.asarray([
        algorithm.objective_function(
            algorithm.action_transform(individual),
            state,
            action_list,
            next_inflow=20.0,
        )
        for individual in population
    ])

    actual = algorithm.calculate_fitness(
        population, state, action_list, next_inflow=20.0
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
