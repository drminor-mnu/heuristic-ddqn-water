import numpy as np

from genetic_algo import GeneticAlgorithm
from water_gym import Actions0, WaterGym, switching_stability


def test_switching_stability_uses_changed_pump_count():
    assert switching_stability(Actions0[0], Actions0[0]) == 1.0
    assert switching_stability(Actions0[5], Actions0[5]) == 1.0
    assert switching_stability(Actions0[0], Actions0[1]) == 0.8
    assert switching_stability(Actions0[0], Actions0[5]) == 0.0


def test_environment_and_ga_objective_agree_on_switching_component():
    environment = WaterGym("unused.inp", minutes=2, weights=[0, 1, 0, 0])
    environment.acts = [0, 1]
    environment.outflow = [0.0, 0.0]
    environment.reservior_level = [4.7, 4.7]
    _, info = environment.reward(clock=1, over_pump=0)

    algorithm = GeneticAlgorithm(
        minutes=2, weights=[0, 1, 0, 0], act_type=0
    )
    objective = algorithm.objective_function(
        act=1,
        state=np.asarray([0.0, 0.0, 0.0, 1000.0, 5.8]),
        action_list=[0],
        next_inflow=0.0,
    )

    assert info[1] == 0.8
    assert objective == info[1]
