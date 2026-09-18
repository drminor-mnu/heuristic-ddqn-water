import numpy as np

from plot_dqn_radar import relative_cost_scores


def test_relative_cost_scores_are_higher_for_lower_cost():
    scores = relative_cost_scores(np.array([[5.0, 10.0, 20.0], [10.0, 5.0, 20.0]]))
    np.testing.assert_allclose(scores[0], [1.0, 0.5, 1.0])
    np.testing.assert_allclose(scores[1], [0.5, 1.0, 1.0])
