# heuristic-ddqn-water

Stormwater pumping-station control via heuristic-guided (GA/PSO) Double DQN,
evaluated against SWMM-simulated inflow scenarios. Code for the Water
journal submission water-4498965 ("Automatic Operation Algorithm for
Stormwater Pumping Stations Based on Heuristic-Guided Reinforcement
Learning"), currently in Major Revision.

## Environment

See `results/_log/ENVIRONMENT.json` for the exact hardware/library versions
used to produce the current results (Python 3.9.16, torch 2.0.1+cu117,
pyswmm 2.0.0, SWMM engine 5.2.4). No `requirements.txt`/`environment.yml`
is maintained separately — install the versions listed there with
`pip`/`conda` as needed; there is no automated environment-setup script.

## Layout

- `src/` — all runnable code (models, GA/PSO heuristics, SWMM wrapper,
  training/evaluation drivers, per-experiment `run_e0*_*.py` scripts).
- `data/` — the SWMM base model (`sample_inp/sample_gasan.inp`), the fixed
  train/test split (`splits/fixed_split_seed42.json`, used by every
  experiment from E0 onward), and the scenario manifests. The generated
  scenario files are **not** stored here — see *Data availability* below.
- `results/` — current, authoritative results, organized as
  `results/E0{n}_{name}/` per experiment plus `results/summary/` for
  cross-experiment aggregates.
- Superseded result trees from earlier development (`results_old/`,
  `results_org/`, `early_162_60pct/` and similar) are not included in
  this repository; they are not used by any current script.
- `docs/` — experiment reports and findings (`docs/E0*_*.md`,
  `docs/CODEBASE_MAP.md`, `docs/NOMENCLATURE.md`,
  `docs/RESULTS_PROVENANCE.md`).

## Reproducing a result

Every experiment driver (`src/run_e0*_*.py`) is resumable and writes a
`run_meta.json` (hyperparameters, reward weights, git commit, data split,
hardware/library versions) alongside its output — see
`docs/NOMENCLATURE.md` for what each field means and
`results/summary/E13_repro_table.md` for a consolidated table of every
recorded run's parameters (regenerate with
`python scripts/make_repro_table.py`).

Long training runs (order of hours per seed) are **not** run inside an
interactive session — each driver script is written to be launched with
`nohup`/nohup-equivalent and to skip already-completed (seed, condition)
combinations on restart. See the specific experiment's report in `docs/`
for its exact launch command.

## Script → output mapping (experiments with dedicated docs)

| Experiment | Driver script(s) | Report |
|---|---|---|
| E0 (infra/audit) | various `diag_e0*_*.py` | `docs/CODEBASE_MAP.md`, `docs/RESULTS_PROVENANCE.md` |
| E2 (argmin/argmax) | (audit only, no driver) | `docs/E02_ARGMINMAX.md`, `docs/E02_OBJECTIVE_AUDIT.md` |
| E3 (multi-seed) | `src/run_e03_seeds.py` | `docs/E03_FINDINGS.md` |
| E4 (enumeration / lookahead) | `src/run_e04_enum.py`, `run_e04_horizon.py`, `run_e04e_task*.py`, `run_e04f_task*.py` | `docs/E04E_REPORT.md`, `docs/E04F_REPORT.md`, `docs/E04C_SCOPING.md` |
| E5 (guidance ablation) | `src/run_e05_ablation.py`, `run_e05_analyze.py`, `run_e05_temporal.py`, `run_e05_early_curve.py`, `run_e05_contrast.py` | `docs/E05_REPORT.md`, `docs/E05_SUMMARY_TABLE.md` |
| E6 (weight sensitivity) | `src/diag_e06_weights.py`, `diag_e06_refine.py` | `docs/E06_WEIGHT_SENSITIVITY.md` |
| E10 (timing breakdown) | `src/run_e10_timing.py`, `run_e10_unified_c.py` | `results/summary/E10_unified_table.md`, `E10_timing_breakdown.csv` |
| E11 (action space) | (documentation only) | `docs/ACTION_SPACE_JUSTIFICATION.md` |
| E13 (repro/nomenclature) | `scripts/make_repro_table.py` | `docs/NOMENCLATURE.md`, `results/summary/E13_repro_table.md` |

## Data availability

### Regenerating the scenario set

**The scenario files themselves are not stored in this repository.** The
7,440 SWMM `.inp` files and their 7,440 rainfall `.txt` files (about 33 GB
once SWMM has also written its `.out`/`.rpt` output alongside them) are
generated from the base model and the Huff-curve coefficients that *are*
included here. To recreate them under `data/gasan/` and `data/rainfall/`:

```bash
cd src
python data_generate.py          # ~7,440 .inp + 7,440 rainfall .txt
```

Generation is deterministic (`np.random.seed(1234)` at module level) and
reproduces the original files byte for byte — verified by regenerating the
`10year` group and comparing all 1,240 `.inp` and 1,240 `.txt` files. Note
that `n_vars=30` is required for this, not the function's default of 29;
`data_generate.py`'s `__main__` block already passes the correct value.

Inputs used: `data/sample_inp/sample_gasan.inp` (the SWMM base model) and
the Huff quartile weights plus the return-period rainfall depth table,
both hard-coded in `src/data_generate.py`. `data/selected_inp.txt` lists
the full scenario set and `data/splits/fixed_split_seed42.json` defines the
train/test split (`split_seed=42`) used by every experiment.

**All rainfall scenarios used in the study are synthetic** (Huff-curve
based; see `docs/DATA_GENERATION.md`). No observed rainfall record
contributed to any reported result. A set of observed 1-minute rainfall
records from a Korea Meteorological Administration AWS station was
examined during the work but was excluded from the study — the station is
unrelated to the target catchment — and those files are not distributed
here.

## License

MIT (see `LICENSE`). The SWMM engine and `pyswmm` are third-party software
distributed under their own licenses.
