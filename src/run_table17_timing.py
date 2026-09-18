#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Table 17 (new sec 3.5.1) 1-step decision-time measurement for ENUM,
on the SAME (c)-basis as Table 5 / Figure 10: (a) heuristic decision +
(b) preprocessing (n/a for these heuristics -- they act directly on the
raw state) + (c) environment-step overhead, SWMM excluded.

Context (see results/summary/TABLE17_TIMING_CHECK.md for the full
comparison): E4-a (results/E04_enum/FINDINGS.md) already reports 1-step
decision times for ENUM/GA/PSO (0.039/1.43/20.0 ms), but that
measurement times ONLY the per-decision search call
(EnumSearch.score_all_actions / GeneticAlgorithm.genetic_algorithm /
ParticleSwarmOptimization.particle_swarm_optimization) -- it excludes
the WaterGym.step() environment-step overhead that Table 5 / Figure
10's (c) basis includes. Figure 10's own GA-only/PSO-only values
(results/summary/E10_unified_c_summary.csv, from run_e10_unified_c.py)
are already measured on the identical (c) basis used here, so this
script reuses those values verbatim rather than re-measuring them --
this guarantees Table 17's GA/PSO numbers are byte-identical to Figure
10's, eliminating any possible future Table-vs-Figure discrepancy. Only
ENUM (not present in the Figure 10 measurement) is newly measured here,
on the exact same sample/seed/repeat protocol.

No existing evaluation function is modified: EnumSearch (src/enum_baseline.py)
already exists and is used unchanged; time_heuristic_c()/stratified_subsample()
are imported unchanged from run_e10_unified_c.py.

Run from src/, with the `torch` conda env active.
"""
import csv
import statistics
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

import sys
sys.path.insert(0, str(SRC))

from run_e10_unified_c import (  # noqa: E402  (reuse, not re-implement)
    time_heuristic_c, stratified_subsample, SEED, SAMPLE_PER_STRATUM,
    N_REPEATS, NEW_WEIGHTS_4C,
)


def measure_enum(scenarios):
    """Returns {repeat: ms_per_decision} for EnumSearch, (c) basis, SWMM excluded."""
    from enum_baseline import EnumSearch

    per_repeat = {}
    for repeat in range(1, N_REPEATS + 1):
        abc_sum, n_sum, swmm_sum = 0.0, 0, 0.0
        for j, inp in enumerate(scenarios):
            print(f'\rENUM repeat {repeat}/{N_REPEATS} {j + 1}/{len(scenarios)}',
                  end='', flush=True)
            instance = EnumSearch(minutes=2, weights=NEW_WEIGHTS_4C, act_type=0)
            t, n, swmm = time_heuristic_c(instance, 'run_genetic_algo', inp)
            abc_sum += t
            n_sum += n
            swmm_sum += swmm
        print()
        ms = abc_sum / n_sum * 1000
        per_repeat[repeat] = ms
        print(f'  repeat {repeat}: {ms:.4f} ms/decision (n={n_sum}), '
              f'swmm pooled {swmm_sum:.3f}s over {len(scenarios)} scenarios (excluded)')
    return per_repeat


def load_existing_c_values():
    """GA-only/PSO-only (c) values, reused verbatim from the Figure-10 measurement."""
    src = ROOT / 'results' / 'summary' / 'E10_unified_c_summary.csv'
    out = {}
    with open(src, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['condition'] in ('GA-only', 'PSO-only'):
                out[r['condition']] = {
                    'mean_ms_per_decision': float(r['mean_ms_per_decision']),
                    'sd_ms_per_decision': float(r['sd_ms_per_decision']),
                    'n_repeats': int(r['n_repeats']),
                    'values': r['values'],
                    'source': str(src),
                }
    return out


def main():
    import os
    os.chdir(str(SRC))
    import data_paths

    _, _, test_inps = data_paths.load_fixed_split()
    scenarios = stratified_subsample(test_inps, SAMPLE_PER_STRATUM)
    print(f'{len(scenarios)} scenarios (stratified, {SAMPLE_PER_STRATUM}/stratum), '
          f'seed{SEED}, {N_REPEATS} repeats -- identical protocol to run_e10_unified_c.py')

    enum_per_repeat = measure_enum(scenarios)
    enum_values = list(enum_per_repeat.values())
    enum_mean = statistics.mean(enum_values)
    enum_sd = statistics.stdev(enum_values) if len(enum_values) > 1 else 0.0

    existing = load_existing_c_values()

    out_dir = ROOT / 'results' / 'summary'
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            'model': 'ENUM',
            'mean_ms_per_decision': f'{enum_mean:.4f}',
            'sd_ms_per_decision': f'{enum_sd:.4f}',
            'n_repeats': N_REPEATS,
            'values': ';'.join(f'{v:.4f}' for v in enum_values),
            'source': 'src/run_table17_timing.py (this script, new measurement)',
        },
        {
            'model': 'PSO',
            'mean_ms_per_decision': f"{existing['PSO-only']['mean_ms_per_decision']:.4f}",
            'sd_ms_per_decision': f"{existing['PSO-only']['sd_ms_per_decision']:.4f}",
            'n_repeats': existing['PSO-only']['n_repeats'],
            'values': existing['PSO-only']['values'],
            'source': 'results/summary/E10_unified_c_summary.csv (PSO-only row, '
                      'reused verbatim -- same (c) basis as Figure 10)',
        },
        {
            'model': 'GA',
            'mean_ms_per_decision': f"{existing['GA-only']['mean_ms_per_decision']:.4f}",
            'sd_ms_per_decision': f"{existing['GA-only']['sd_ms_per_decision']:.4f}",
            'n_repeats': existing['GA-only']['n_repeats'],
            'values': existing['GA-only']['values'],
            'source': 'results/summary/E10_unified_c_summary.csv (GA-only row, '
                      'reused verbatim -- same (c) basis as Figure 10)',
        },
    ]

    csv_path = out_dir / 'table17.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'mean_ms_per_decision', 'sd_ms_per_decision',
                                           'n_repeats', 'values', 'source'])
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {csv_path}')

    txt_path = out_dir / 'table17.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('Table 17. One-step decision time of ENUM, PSO, and GA (mean +- sd, 3 repeats).\n')
        f.write('Measurement basis: state observation + decision search + environment-step\n')
        f.write('overhead, excluding the once-per-episode SWMM simulation (same basis as\n')
        f.write('Table 5 Decision time and Figure 10; see footnote).\n\n')
        f.write(f'{"Model":<8}{"ms/decision":>16}\n')
        for r in rows:
            f.write(f'{r["model"]:<8}{r["mean_ms_per_decision"]:>10} +- {r["sd_ms_per_decision"]}\n')
        f.write('\nNote: PSO and GA values are reused verbatim from the Figure 10\n')
        f.write('measurement (results/summary/E10_unified_c_summary.csv), which already\n')
        f.write('uses this same basis -- not re-measured here, to guarantee identical\n')
        f.write('values between Table 17 and Figure 10. ENUM was newly measured on the\n')
        f.write('identical protocol (54-scenario stratified subsample, seed1, 3 repeats).\n')
    print(f'wrote {txt_path}')

    for r in rows:
        print(f'{r["model"]}: {r["mean_ms_per_decision"]} +- {r["sd_ms_per_decision"]} ms/decision')


if __name__ == '__main__':
    main()
