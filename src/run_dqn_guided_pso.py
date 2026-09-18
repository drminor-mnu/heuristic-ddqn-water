#!/usr/bin/env python3
"""Run the fixed 60% training experiment for PSO-guided DDQN-GRU."""

from pathlib import Path

import perform_evaluate
import data_paths


YEARS = ['10', '20', '30', '50', '80', '100']
DURATIONS = ['0060', '0120', '0180', '0240', '0360', '0540',
             '0720', '1080', '1440']
PARAMS = {
    'epochs': 1,
    'minutes': 2,
    'weights': [0.25, 0.4, 0.25, 0.1],
    'ga_start': 0.6,
    'ga_end': 0.0,
    'eps_start': 0.3,
    'eps_end': 0.05,
    'gamma': 0.1,
    'batch_size': 20,
    'learning_rate': 0.001,
    'sync_freq': 200,
    'act_type': 0,
}
EXPERIMENT_NAME = 'dqn_guided_PSO_w0_25_w0_4_w0_25_w0_1_ga0_6_min2_tr0_6_acttype0'


def main():
    train_inps, _, test_inps = data_paths.load_fixed_split()
    meta = data_paths.get_split_meta()
    print(f"split={meta['split_file']} seed={meta['split_seed']}")
    root = Path(__file__).resolve().parent.parent
    model_path = root / 'trained_models' / f'{EXPERIMENT_NAME}.pkl'
    result_path = root / 'results' / f'{EXPERIMENT_NAME}.csv'

    perform_evaluate.evaluate_dpn_guided_PSO(
        train_inps, test_inps, YEARS, DURATIONS,
        train=True, model_path=str(model_path), **PARAMS,
    )
    if not result_path.is_file() or result_path.stat().st_size == 0:
        raise RuntimeError(f'Expected result was not created: {result_path}')
    print(f'Result saved to: {result_path}')


if __name__ == '__main__':
    main()
