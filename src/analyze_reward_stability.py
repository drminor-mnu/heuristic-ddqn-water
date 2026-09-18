#!/usr/bin/env python3
"""Calculate stability metrics for reward_comparison.csv."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_metrics(rewards, window=100, tail_ratio=0.25):
    """Return detrended, rolling-trend, and final-stage reward metrics."""
    rewards = pd.Series(rewards, dtype=float).dropna().reset_index(drop=True)
    if rewards.empty:
        raise ValueError("Reward data is empty.")
    if not 0.0 < tail_ratio <= 1.0:
        raise ValueError("tail_ratio must be in the range (0, 1].")

    effective_window = min(window, len(rewards))
    min_periods = max(2, effective_window // 2)
    trend = rewards.rolling(
        window=effective_window,
        center=True,
        min_periods=min_periods,
    ).mean()

    residuals = rewards - trend
    tail_size = max(1, int(np.ceil(len(rewards) * tail_ratio)))
    final_rewards = rewards.iloc[-tail_size:]
    final_mean = final_rewards.mean()
    final_std = final_rewards.std(ddof=0)

    return {
        "sample_count": len(rewards),
        "rolling_window": effective_window,
        "residual_std": residuals.std(ddof=0),
        "rolling_mean_std": trend.std(ddof=0),
        "final_25pct_mean": final_mean,
        "final_25pct_std": final_std,
        "final_25pct_cv": final_std / (abs(final_mean) + 1e-8),
    }


def analyze_csv(input_path, output_path=None, window=100, tail_ratio=0.25):
    """Analyze every reward column and optionally save the result as CSV."""
    data = pd.read_csv(input_path)
    results = {
        column: calculate_metrics(data[column], window, tail_ratio)
        for column in data.columns
    }
    result_frame = pd.DataFrame.from_dict(results, orient="index")
    result_frame.index.name = "reward_data"

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_frame.to_csv(output_path, float_format="%.6f")

    return result_frame


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "results" / "reward_comparison.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "results" / "reward_stability_metrics.csv",
    )
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--tail-ratio", type=float, default=0.25)
    args = parser.parse_args()

    result = analyze_csv(
        args.input,
        args.output,
        window=args.window,
        tail_ratio=args.tail_ratio,
    )
    print(result.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved metrics to: {args.output}")


if __name__ == "__main__":
    main()
