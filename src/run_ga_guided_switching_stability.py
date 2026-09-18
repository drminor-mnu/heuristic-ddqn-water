#!/usr/bin/env python3
"""Train and evaluate switching-focused GA-guided DDQN-GRU."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

import perform_evaluate
import data_paths


ROOT = Path(__file__).resolve().parent.parent
YEARS = ["10", "20", "30", "50", "80", "100"]
DURATIONS = [
    "0060", "0120", "0180", "0240", "0360",
    "0540", "0720", "1080", "1440",
]
TRAIN_RATE = 0.6
TEST_RATE = 0.4
SEED = 20260726
PARAMS = {
    "epochs": 1,
    "minutes": 2,
    "weights": [0.15, 0.60, 0.15, 0.10],
    "ga_start": 0.6,
    "ga_end": 0.0,
    "eps_start": 0.3,
    "eps_end": 0.05,
    "gamma": 0.1,
    "batch_size": 20,
    "learning_rate": 0.001,
    "sync_freq": 200,
    "act_type": 0,
}
EXPERIMENT_NAME = (
    "dqn_guided_GA_w0_15_w0_6_w0_15_w0_1_"
    "ga0_6_min2_tr0_6_acttype0"
)


def _hash_paths(paths) -> str:
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_paths, _, test_paths = data_paths.load_fixed_split()
    split_meta = data_paths.get_split_meta()
    print(f"split={split_meta['split_file']} seed={split_meta['split_seed']}")
    
    if len(train_paths) != 3996 or len(test_paths) != 2700:
        raise RuntimeError(
            f"expected 3,996 training and 2,700 test scenarios, got "
            f"{len(train_paths)} and {len(test_paths)}"
        )

    model_path = ROOT / "trained_models" / f"{EXPERIMENT_NAME}.pkl"
    result_path = ROOT / "results" / f"{EXPERIMENT_NAME}.csv"
    artifact_dir = ROOT / "results" / "ga_guided_switching_stability"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    losses, train_rewards, test_rewards, elapsed = (
        perform_evaluate.evaluate_dpn_guided_GA(
            train_paths, test_paths, YEARS, DURATIONS,
            train=True, model_path=str(model_path), **PARAMS,
        )
    )
    if not model_path.is_file() or not result_path.is_file():
        raise RuntimeError("training or evaluation artifact was not created")

    np.save(artifact_dir / "train_losses.npy", losses)
    np.save(artifact_dir / "train_rewards.npy", train_rewards)
    np.save(artifact_dir / "test_rewards.npy", test_rewards)
    metadata = {
        "experiment": EXPERIMENT_NAME,
        "seed": SEED,
        "train_count": len(train_paths),
        "test_count": len(test_paths),
        "train_paths_sha256": _hash_paths(train_paths),
        "test_paths_sha256": _hash_paths(test_paths),
        "parameters": PARAMS,
        "training_elapsed_seconds": float(elapsed),
        "model": str(model_path),
        "result": str(result_path),
        "split_file": split_meta["split_file"],
        "split_seed": split_meta["split_seed"],
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
