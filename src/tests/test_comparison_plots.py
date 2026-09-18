import csv

from plot_comparison_figures import (
    COMPARISON_PLOT_STYLE,
    load_reward_series,
    plot_reward_comparison,
)


def test_load_reward_series_selects_train_or_test_columns(tmp_path):
    source = tmp_path / "rewards.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "regular_train_reward", "ga_guided_train_reward",
            "regular_test_reward", "ga_guided_test_reward",
        ))
        writer.writeheader()
        writer.writerow({
            "regular_train_reward": "1", "ga_guided_train_reward": "2",
            "regular_test_reward": "3", "ga_guided_test_reward": "4",
        })
        writer.writerow({
            "regular_train_reward": "5", "ga_guided_train_reward": "6",
            "regular_test_reward": "", "ga_guided_test_reward": "",
        })

    assert load_reward_series(source, "train") == ([1.0, 5.0], [2.0, 6.0])
    assert load_reward_series(source, "test") == ([3.0], [4.0])


def test_reward_comparison_uses_scenario_label_and_compact_font(tmp_path):
    assert {
        key: COMPARISON_PLOT_STYLE[key]
        for key in (
            "font.size", "axes.labelsize", "xtick.labelsize",
            "ytick.labelsize", "legend.fontsize",
        )
    } == {
        "font.size": 12,
        "axes.labelsize": 20,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
    }
    output = plot_reward_comparison(
        [1.0, 2.0], [2.0, 3.0], tmp_path / "comparison.png"
    )
    assert output.is_file()
    assert output.stat().st_size > 0
