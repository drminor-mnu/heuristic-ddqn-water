import numpy as np

from benchmark_full_data_runtime import (
    FULL_SAMPLE_SIZES, RUN_SEEDS, canonicalize_raw_rows,
    load_full_population, summarize_runs,
)


def test_full_population_uses_all_source_entries_in_stratified_order():
    paths = load_full_population()

    assert len(paths) == len(set(paths)) == 7440
    assert FULL_SAMPLE_SIZES == tuple(range(500, 5001, 500))
    first_strata = {
        tuple(path.rsplit("/", 1)[-1].split("_")[:2])
        for path in paths[:60]
    }
    assert len(first_strata) == 60


def test_summary_uses_sample_standard_deviation_across_five_runs():
    raw = [
        {
            "method": "GA", "run": run, "seed": seed,
            "test_scenarios": 500, "elapsed_seconds": float(run),
            "seconds_per_scenario": run / 500,
            "scenarios_per_second": 500 / run,
        }
        for run, seed in enumerate(RUN_SEEDS, start=1)
    ]

    summary = summarize_runs(raw)

    assert len(summary) == 1
    assert summary[0]["runs"] == 5
    assert summary[0]["mean_elapsed_seconds"] == 3.0
    assert summary[0]["std_elapsed_seconds"] == np.std(
        [1, 2, 3, 4, 5], ddof=1
    )


def test_raw_values_are_canonicalized_to_persisted_nine_digit_precision():
    row = {
        "method": "GA", "run": 1, "seed": 101, "test_scenarios": 500,
        "elapsed_seconds": 1.1234567896,
        "seconds_per_scenario": 0.0022469135792,
        "scenarios_per_second": 444.9999999996,
    }

    canonical = canonicalize_raw_rows([row])[0]

    assert canonical["elapsed_seconds"] == 1.123456790
    assert canonical["seconds_per_scenario"] == 0.002246914
    assert canonical["scenarios_per_second"] == 445.0
