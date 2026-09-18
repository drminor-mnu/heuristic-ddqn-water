#!/usr/bin/env python3
"""Resolve repo-relative scenario paths (e.g. from data/selected_inp_relative.txt
or data/splits/fixed_split_seed*.json) to real filesystem paths.

Base directory resolution order:
1. WATER_DATA_ROOT environment variable, if set.
2. Repository root (parent of this src/ directory) — default.

A relative path like "data/gasan/10year/10yr_0010m_h051.inp" is joined onto
the base directory. An already-absolute path is returned unchanged (so old
absolute-path split files keep working).
"""
import json
import os

_REPO_ROOT_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)

DEFAULT_SPLIT_PATH = os.path.join(_REPO_ROOT_DEFAULT, "data", "splits", "fixed_split_seed42.json")


def get_data_root() -> str:
    return os.environ.get("WATER_DATA_ROOT", _REPO_ROOT_DEFAULT)


def resolve_path(rel_path: str, base_dir: str = None) -> str:
    if os.path.isabs(rel_path):
        return rel_path
    if base_dir is None:
        base_dir = get_data_root()
    return os.path.join(base_dir, rel_path)


def get_split_path(split_path: str = None) -> str:
    """Resolution order: explicit argument > WATER_SPLIT_FILE env var > seed-42 default."""
    if split_path is not None:
        return split_path
    return os.environ.get("WATER_SPLIT_FILE", DEFAULT_SPLIT_PATH)


def load_fixed_split(split_path: str = None, expect_counts=(3996, 0, 2700)):
    """Load (train_inps, valid_inps, test_inps) from a fixed_split_seed*.json
    file, so E3+ entry points share one deterministic scenario assignment
    instead of each calling stratified_split_data()/fixed_test_paths() with
    their own (possibly unseeded) randomness. Paths are returned exactly as
    stored (repo-relative for the seed-42 file); resolve with resolve_path()
    before opening.

    expect_counts=(n_train, n_valid, n_test): raises ValueError if the loaded
    split's counts don't match (catches pointing at the wrong/corrupt split
    file). Pass expect_counts=None to skip validation (e.g. E12's variable
    training-set-size experiments, which intentionally use non-3996 splits).
    """
    path = get_split_path(split_path)
    with open(path) as fh:
        document = json.load(fh)
    train_inps = document["train_inps"]
    valid_inps = document["valid_inps"]
    test_inps = document["test_inps"]
    if expect_counts is not None:
        actual = (len(train_inps), len(valid_inps), len(test_inps))
        if actual != tuple(expect_counts):
            raise ValueError(
                f"split file {path!r} has train/valid/test counts {actual}, "
                f"expected {tuple(expect_counts)}"
            )
    return train_inps, valid_inps, test_inps


def get_split_meta(split_path: str = None) -> dict:
    """Return {'split_file': <repo-relative path>, 'split_seed', 'n_train',
    'n_valid', 'n_test'} for the split file that would be loaded, to record
    in a run's run_meta.json 'data' section (plan section 2.2)."""
    path = get_split_path(split_path)
    with open(path) as fh:
        document = json.load(fh)
    repo_root = get_data_root()
    rel_path = os.path.relpath(os.path.abspath(path), os.path.abspath(repo_root))
    return {
        "split_file": rel_path,
        "split_seed": document.get("split_seed"),
        "n_train": len(document["train_inps"]),
        "n_valid": len(document["valid_inps"]),
        "n_test": len(document["test_inps"]),
    }


def write_split_meta(entrypoint_name: str, split_path: str = None, out_dir: str = None) -> str:
    """Write {'data': get_split_meta(...)} to <out_dir>/run_meta_<entrypoint_name>.json
    so an entry point records which split/seed it actually used.

    out_dir defaults to results/_log/, which is fine for a single one-off run
    per entrypoint_name. E3 runs each entrypoint repeatedly (5 models x 5
    seeds) — callers doing that MUST pass a per-run out_dir (e.g.
    results/E03_seeds/{model}/seed{n}/), or successive runs will overwrite
    each other's run_meta_<entrypoint_name>.json in results/_log/.
    """
    meta = get_split_meta(split_path)
    if out_dir is None:
        out_dir = os.path.join(get_data_root(), "results", "_log")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"run_meta_{entrypoint_name}.json")
    with open(out_path, "w") as fh:
        json.dump({"data": meta}, fh, indent=2)
    return out_path
