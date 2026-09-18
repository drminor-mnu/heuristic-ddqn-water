#!/usr/bin/env python3
"""Evaluate only local search and save results to local_search_test.csv."""

import time

from local_search import LocalSearch
from perform_evaluate import _evaluate_heuristic
import data_paths


YEARS = ['10', '20', '30', '50', '80', '100']
DURATIONS = ['0060', '0120', '0180', '0240', '0360', '0540', '0720', '1080', '1440']


def main():
    _, _, test_inps = data_paths.load_fixed_split()
    meta = data_paths.get_split_meta()
    print(f"split={meta['split_file']} seed={meta['split_seed']}")

    params = {
        'minutes': 2,
        'weights': [0.25, 0.4, 0.25, 0.1],
        'act_type': 0,
        'neighborhood_size': 3,
        'num_restarts': 1,
    }

    start = time.perf_counter()
    _evaluate_heuristic(
        test_inps,
        YEARS,
        DURATIONS,
        optimizer_class=LocalSearch,
        optimize_label='Local Search (neighborhood=3)',
        run_method='run_local_search',
        result_prefix='local_search',
        result_file='local_search_test',
        **params,
    )
    print(f'Elapsed time: {time.perf_counter() - start:.1f} seconds')


if __name__ == '__main__':
    main()
