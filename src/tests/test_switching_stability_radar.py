import numpy as np

from plot_switching_stability_radar import (
    LABEL_RADIUS, relative_cost_scores,
)


def test_objective_labels_are_positioned_outside_unit_radar():
    assert LABEL_RADIUS > 1.0


def test_relative_cost_scores_reward_lower_operating_cost():
    scores = relative_cost_scores(
        np.asarray([[10.0, 20.0, 4.0], [5.0, 40.0, 2.0]])
    )
    np.testing.assert_allclose(
        scores, [[0.5, 1.0, 0.5], [1.0, 0.5, 1.0]]
    )
