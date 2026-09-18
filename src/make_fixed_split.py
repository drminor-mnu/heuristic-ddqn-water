#!/usr/bin/env python3
"""Generate a fixed, seeded train/test split using perform_evaluate.py's own
stratified_split_data() logic (54 combinations x 124 samples, 74 train / 50 test
per combination), so all E3+ experiments can share one deterministic split
instead of perform_evaluate.py's random.seed(time.time()) at import time.

Run from src/:
    python make_fixed_split.py [split_seed]
"""
import sys
import os
import re
import json
import random
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perform_evaluate as pe  # noqa: E402  (reuses pe.stratified_split_data unmodified)


def stratum_key(path):
    nums = re.findall(r'\d+', path.split('/')[-1])
    return (nums[0], nums[1])


def main(split_seed: int):
    random.seed(split_seed)  # overrides perform_evaluate's random.seed(time.time()) done at import

    years = ['10', '20', '30', '50', '80', '100']
    durations = ['0060', '0120', '0180', '0240', '0360', '0540', '0720', '1080', '1440']
    trate = 0.6
    vrate = 1.0 - (0.4 + trate)

    inp_list, train_inps, valid_inps, test_inps = pe.stratified_split_data(
        years, durations, trate=trate, vrate=vrate, stratify=True)

    train_counts = Counter(stratum_key(f) for f in train_inps)
    valid_counts = Counter(stratum_key(f) for f in valid_inps)
    test_counts = Counter(stratum_key(f) for f in test_inps)

    n_strata = len(years) * len(durations)
    train_bad = {f"{k[0]}y_{k[1]}m": v for k, v in train_counts.items() if v != 74}
    test_bad = {f"{k[0]}y_{k[1]}m": v for k, v in test_counts.items() if v != 50}

    result = {
        "split_seed": split_seed,
        "source_function": "perform_evaluate.stratified_split_data (unmodified)",
        "source_list_file": "data/selected_inp.txt",
        "years": years,
        "durations": durations,
        "trate": trate,
        "vrate": vrate,
        "n_strata": n_strata,
        "n_total": len(inp_list),
        "n_train": len(train_inps),
        "n_valid": len(valid_inps),
        "n_test": len(test_inps),
        "train_per_stratum_ok": len(train_bad) == 0,
        "test_per_stratum_ok": len(test_bad) == 0,
        "train_per_stratum_anomalies": train_bad,
        "test_per_stratum_anomalies": test_bad,
        "train_inps": train_inps,
        "valid_inps": valid_inps,
        "test_inps": test_inps,
    }

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "splits")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"fixed_split_seed{split_seed}.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"split_seed={split_seed}")
    print(f"n_strata={n_strata} n_total={len(inp_list)} n_train={len(train_inps)} "
          f"n_valid={len(valid_inps)} n_test={len(test_inps)}")
    print(f"train_per_stratum_ok={result['train_per_stratum_ok']} "
          f"anomalies={train_bad if train_bad else 'none'}")
    print(f"test_per_stratum_ok={result['test_per_stratum_ok']} "
          f"anomalies={test_bad if test_bad else 'none'}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(seed)
