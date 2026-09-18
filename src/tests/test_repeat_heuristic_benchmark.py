from pathlib import Path

import numpy as np

from repeat_heuristic_benchmark import (
    DURATIONS, YEARS, LegacyResult, _aggregate_scenarios,
    append_mean_timing, average_results, parse_legacy_result,
    write_legacy_result,
)


def synthetic(offset):
    blocks = tuple(np.full((2, 3), offset + index, dtype=float) for index in range(13))
    return LegacyResult(blocks)


def test_legacy_results_are_averaged_and_keep_thirteen_blocks(tmp_path):
    mean = average_results([synthetic(0), synthetic(2)])
    assert len(mean.blocks) == 13
    np.testing.assert_allclose(mean.blocks[4], 5.0)
    path = tmp_path / "mean.csv"
    write_legacy_result(mean, path)
    append_mean_timing(path, 12.345678)
    parsed = parse_legacy_result(path)
    assert len(parsed.blocks) == 13
    np.testing.assert_allclose(parsed.blocks[4], 5.0)
    assert path.read_text().splitlines()[-1] == "mean_execution_time_seconds,12.345678"


def test_parallel_scenario_rows_are_aggregated_in_legacy_layout():
    rows = []
    for year in YEARS:
        for duration in DURATIONS:
            for _ in range(50):
                rows.append((
                    year, duration, 6.0, 4.0, 3.0, 2.0,
                    np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
                    0.5, 1.0,
                ))

    result = _aggregate_scenarios(rows)

    assert len(result.blocks) == 13
    np.testing.assert_allclose(result.blocks[0], 50.0)
    np.testing.assert_allclose(result.blocks[1], 6.0)
    np.testing.assert_allclose(result.blocks[2], 4.0)
    np.testing.assert_allclose(result.blocks[3], 3.0)
    np.testing.assert_allclose(result.blocks[4], 2.0)
    for year_index in range(len(YEARS)):
        np.testing.assert_allclose(
            result.blocks[5 + year_index],
            np.tile([1.0, 2.0, 3.0, 4.0, 5.0], (len(DURATIONS), 1)),
        )
    np.testing.assert_allclose(result.blocks[11], 0.5)
    np.testing.assert_allclose(result.blocks[12], 50.0)
