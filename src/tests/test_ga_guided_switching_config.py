from run_ga_guided_switching_stability import (
    EXPERIMENT_NAME, PARAMS, SEED, TEST_RATE, TRAIN_RATE,
)


def test_switching_experiment_preserves_split_and_emphasizes_switching():
    assert PARAMS["weights"] == [0.15, 0.60, 0.15, 0.10]
    assert sum(PARAMS["weights"]) == 1.0
    assert PARAMS["ga_start"] == 0.6
    assert PARAMS["act_type"] == 0
    assert TRAIN_RATE == 0.6
    assert TEST_RATE == 0.4
    assert SEED == 20260726
    assert EXPERIMENT_NAME == (
        "dqn_guided_GA_w0_15_w0_6_w0_15_w0_1_"
        "ga0_6_min2_tr0_6_acttype0"
    )
