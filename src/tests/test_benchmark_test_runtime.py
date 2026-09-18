import matplotlib as mpl

import benchmark_test_runtime as runtime_benchmark
from benchmark_test_runtime import (
    METHOD_CONFIGS, SAMPLE_SIZES, cumulative_rows, scenario_seed,
    stratified_order,
)
from plot_style import STYLE


def test_stratified_order_is_deterministic_and_balances_prefixes():
    paths = [
        f"/x/inp_{year}_{duration}_{case}.inp"
        for year in ("10", "20")
        for duration in ("0060", "0120")
        for case in range(3)
    ]

    ordered = stratified_order(paths)

    assert ordered == stratified_order(reversed(paths))
    assert len(ordered) == len(set(ordered)) == 12
    assert {
        "_".join(path.rsplit("/", 1)[-1].split("_")[1:3])
        for path in ordered[:4]
    } == {"10_0060", "10_0120", "20_0060", "20_0120"}


def test_cumulative_rows_report_elapsed_per_case_and_throughput():
    rows = cumulative_rows(
        "GA", completion_times=[1.0, 1.8, 3.0, 4.0], sample_sizes=(2, 4)
    )

    assert rows == [
        {
            "method": "GA", "test_scenarios": 2, "elapsed_seconds": 1.8,
            "seconds_per_scenario": 0.9, "scenarios_per_second": 2 / 1.8,
        },
        {
            "method": "GA", "test_scenarios": 4, "elapsed_seconds": 4.0,
            "seconds_per_scenario": 1.0, "scenarios_per_second": 1.0,
        },
    ]


def test_all_five_methods_have_test_only_configs_and_requested_sizes():
    assert tuple(METHOD_CONFIGS) == (
        "Regular DQN", "GA-guided DQN", "PSO-guided DQN", "GA", "PSO"
    )
    assert SAMPLE_SIZES == (50, 100, 250, 500, 1000, 1500, 2000, 2700)
    assert all("train" not in config for config in METHOD_CONFIGS.values())
    assert METHOD_CONFIGS["Regular DQN"]["weights"] == (0.5, 0.5, 0.0, 0.0)


def test_scenario_seed_always_fits_numpy_uint32_range():
    seeds = [scenario_seed(index) for index in (0, 1, 2699)]
    assert all(0 <= seed < 2**32 for seed in seeds)
    assert len(set(seeds)) == 3


def test_scenario_seed_varies_by_independent_run_seed():
    seeds = [scenario_seed(4999, run_seed) for run_seed in (101, 202, 303, 404, 505)]
    assert all(0 <= seed < 2**32 for seed in seeds)
    assert len(set(seeds)) == 5


def test_runtime_scaling_plot_uses_shared_style_without_leaking_rcparams(tmp_path):
    rows = [
        {
            "method": method,
            "test_scenarios": 50,
            "elapsed_seconds": float(index + 1),
        }
        for index, method in enumerate(METHOD_CONFIGS)
    ]
    original_font_weight = mpl.rcParams["font.weight"]

    output = runtime_benchmark.plot_runtime_scaling(
        rows, tmp_path / "runtime.png"
    )

    assert output.is_file()
    assert output.stat().st_size > 0
    assert runtime_benchmark.RUNTIME_PLOT_STYLE == {
        **STYLE,
        "figure.titlesize": 20,
        "axes.titlesize": 20,
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
    assert mpl.rcParams["font.weight"] == original_font_weight


def test_runtime_legend_uses_ddqn_for_dqn_methods_only():
    assert runtime_benchmark.runtime_legend_label("Regular DQN") == "Regular DDQN"
    assert (
        runtime_benchmark.runtime_legend_label("GA-guided DQN")
        == "GA-guided DDQN"
    )
    assert runtime_benchmark.runtime_legend_label("GA") == "GA"


def test_runtime_detail_title_uses_ddqn():
    assert runtime_benchmark.DDQN_DETAIL_TITLE == "DDQN Inference Detail"
