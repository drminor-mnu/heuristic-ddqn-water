#!/usr/bin/env python3
"""One-off verification: regenerate the seed=42 split from
data/selected_inp_relative.txt (via perform_evaluate.stratified_split_data,
unmodified, with its internal open() redirected to the relative file) and
confirm it selects the exact same scenarios, in the exact same order, as the
existing absolute-path data/splits/fixed_split_seed42.json.
"""
import builtins
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perform_evaluate as pe  # noqa: E402
import data_paths  # noqa: E402

ABS_PREFIX = "data/gasan/"
REL_PREFIX = "data/gasan/"

_real_open = builtins.open


def _patched_open(path, *args, **kwargs):
    if path == "../data/selected_inp.txt":
        path = "../data/selected_inp_relative.txt"
    return _real_open(path, *args, **kwargs)


def suffix(path, is_abs):
    return path[len(ABS_PREFIX):] if is_abs else path[len(REL_PREFIX):]


def main(split_seed: int, old_split_path: str):
    with open(old_split_path) as fh:
        old = json.load(fh)

    builtins.open = _patched_open
    try:
        random.seed(split_seed)
        years = ['10', '20', '30', '50', '80', '100']
        durations = ['0060', '0120', '0180', '0240', '0360', '0540', '0720', '1080', '1440']
        trate = 0.6
        vrate = 1.0 - (0.4 + trate)
        inp_list, train_inps, valid_inps, test_inps = pe.stratified_split_data(
            years, durations, trate=trate, vrate=vrate, stratify=True)
    finally:
        builtins.open = _real_open

    new = {"train_inps": train_inps, "valid_inps": valid_inps, "test_inps": test_inps}

    all_ok = True
    for key in ("train_inps", "valid_inps", "test_inps"):
        old_suffixes = [suffix(p, True) for p in old[key]]
        new_suffixes = [suffix(p, False) for p in new[key]]
        same = old_suffixes == new_suffixes
        all_ok &= same
        old_h = hashlib.md5("\n".join(old_suffixes).encode()).hexdigest()
        new_h = hashlib.md5("\n".join(new_suffixes).encode()).hexdigest()
        print(f"{key}: len_old={len(old_suffixes)} len_new={len(new_suffixes)} "
              f"order_and_composition_identical={same} md5_old={old_h} md5_new={new_h}")

    # confirm every relative path in the new split actually resolves to a real file
    missing = []
    for key in ("train_inps", "valid_inps", "test_inps"):
        for p in new[key]:
            resolved = data_paths.resolve_path(p)
            if not os.path.exists(resolved):
                missing.append(resolved)
    print(f"missing_files_after_resolve={len(missing)}")
    if missing:
        print("first missing:", missing[:5])

    print(f"ALL_OK={all_ok and not missing}")

    if all_ok and not missing:
        out = dict(old)
        out["source_list_file"] = "data/selected_inp_relative.txt"
        out["path_style"] = "repo-relative"
        out["train_inps"] = train_inps
        out["valid_inps"] = valid_inps
        out["test_inps"] = test_inps
        with open(old_split_path, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"overwrote {old_split_path} with repo-relative paths")


if __name__ == "__main__":
    main(42, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "splits", "fixed_split_seed42.json"))
