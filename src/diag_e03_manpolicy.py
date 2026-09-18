"""E3 diagnostic step 1+2: run the deterministic rule-based man_policy on a
stratified subsample of the seed-42 test split, and record per-scenario
inflow / max-drainage / capacity numbers for the physical-feasibility check.

Pure diagnostic: calls man_policy.operation() directly, does NOT touch
perform_evaluate.save_* (avoids legacy-filename collisions). Writes:
    results/summary/E03_diag_manpolicy.csv
    results/summary/E03_diag_feasibility.csv   (worst overflow scenarios)
"""
import os
import re
import sys
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.chdir(SRC)  # perform_evaluate/sim_swmm use ../data, ../results relative paths

N_PER_STRATUM = 5
SEED = 12345  # only for choosing the subsample; man_policy itself is deterministic


def pick_subsample():
    import data_paths
    _, _, test_inps = data_paths.load_fixed_split()
    strata = defaultdict(list)
    for p in test_inps:
        n = re.findall(r"\d+", p.split("/")[-1])
        strata[(n[0], n[1])].append(p)
    rng = np.random.RandomState(SEED)
    picked = []
    for key in sorted(strata):
        lst = sorted(strata[key])
        idx = rng.choice(len(lst), size=min(N_PER_STRATUM, len(lst)), replace=False)
        picked.extend(lst[i] for i in idx)
    return picked


def run_one(rel_inp):
    import data_paths
    import water_gym
    import man_policy
    from perform_evaluate import count_changes, count_pumps

    abs_inp = data_paths.resolve_path(rel_inp)
    fn = os.path.basename(rel_inp)
    rp, dur = re.findall(r"\d+", fn)[:2]

    t0 = time.perf_counter()
    actions, states, rewards, infos = man_policy.operation(abs_inp, 2)
    wall = time.perf_counter() - t0

    levels = [s[-1] for s in states]
    highest = max(levels)
    inflows = [s[1] for s in states]      # cur_inflow per step (m^3 over the 2-min step)
    outflows = [s[2] for s in states]     # cur_outflow per step
    vols = [s[3] for s in states]
    n_switches = count_changes(actions, 0)
    p100, p170, _ = count_pumps(actions, 0)
    ainfos = np.asarray(infos, dtype=np.float32)
    n_dry = int(ainfos[:, -1].sum())
    overflow = int(highest >= water_gym.Level[-1])

    n_steps = len(actions)
    total_inflow = float(np.sum(inflows))
    total_outflow = float(np.sum(outflows))
    # max drainage if action 5 (all pumps) were held every decision step:
    max_drain_per_step = sum(water_gym.pumpq) * 2  # minutes=2
    max_possible_drain = max_drain_per_step * (n_steps - 1)
    peak_inflow_step = float(np.max(inflows)) if inflows else 0.0

    return dict(
        scenario_id=fn.replace(".inp", ""), return_period=int(rp), duration_min=int(dur),
        n_steps=n_steps, max_level_m=highest, overflow_flag=overflow,
        n_switches=n_switches, n_intervals_100=p100, n_intervals_170=p170,
        n_dryrun_proxy=n_dry, cum_reward=float(np.sum(rewards)),
        total_inflow_m3=total_inflow, total_outflow_m3=total_outflow,
        peak_inflow_per_step_m3=peak_inflow_step,
        max_possible_drain_m3=max_possible_drain,
        reservoir_capacity_m3=water_gym.Volume[-1],
        inflow_minus_maxdrain_m3=total_inflow - max_possible_drain,
        peak_vol_m3=float(np.max(vols)),
        wall_s=wall,
    )


def main():
    picked = pick_subsample()
    print(f"subsample: {len(picked)} scenarios "
          f"({N_PER_STRATUM}/stratum x {len(picked)//N_PER_STRATUM} strata)")
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_one, p): p for p in picked}
        for i, fut in enumerate(as_completed(futs)):
            rows.append(fut.result())
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(picked)} done", flush=True)
    print(f"elapsed {time.perf_counter()-t0:.1f}s")

    import pandas as pd
    df = pd.DataFrame(rows).sort_values(["duration_min", "return_period", "scenario_id"])
    outdir = os.path.join(ROOT, "results", "summary")
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "E03_diag_manpolicy.csv"), index=False)

    print("\n=== man_policy overflow summary ===")
    print(f"overflow fraction: {df.overflow_flag.mean():.4f} "
          f"({int(df.overflow_flag.sum())}/{len(df)})")
    print(f"max_level_m: mean={df.max_level_m.mean():.3f} "
          f"min={df.max_level_m.min():.3f} max={df.max_level_m.max():.3f}")
    print(f"n_switches: mean={df.n_switches.mean():.2f}  "
          f"n_intervals_100 mean={df.n_intervals_100.mean():.1f}  "
          f"n_intervals_170 mean={df.n_intervals_170.mean():.1f}")

    print("\n=== overflow fraction by duration ===")
    print(df.groupby("duration_min").agg(
        overflow=("overflow_flag", "mean"),
        mean_level=("max_level_m", "mean"),
        n=("overflow_flag", "size"),
    ).to_string())

    print("\n=== physical feasibility (all subsample scenarios) ===")
    feas = df[["scenario_id", "return_period", "duration_min", "overflow_flag",
               "total_inflow_m3", "max_possible_drain_m3", "inflow_minus_maxdrain_m3",
               "reservoir_capacity_m3", "peak_vol_m3", "max_level_m"]]
    n_infeasible = int((df.inflow_minus_maxdrain_m3 > df.reservoir_capacity_m3).sum())
    n_inflow_gt_drain = int((df.inflow_minus_maxdrain_m3 > 0).sum())
    print(f"scenarios where total_inflow > max_possible_drain: {n_inflow_gt_drain}/{len(df)}")
    print(f"scenarios where (inflow - maxdrain) > reservoir_capacity (physically "
          f"unpreventable overflow even at full pumping): {n_infeasible}/{len(df)}")
    feas.to_csv(os.path.join(outdir, "E03_diag_feasibility.csv"), index=False)

    print("\n=== 5 representative overflow scenarios ===")
    ov = df[df.overflow_flag == 1].sort_values("inflow_minus_maxdrain_m3", ascending=False)
    show = ov.head(5) if len(ov) else df.head(5)
    for _, r in show.iterrows():
        print(f"  {r.scenario_id:20s} inflow={r.total_inflow_m3:12.1f}  "
              f"maxdrain={r.max_possible_drain_m3:12.1f}  "
              f"inflow-maxdrain={r.inflow_minus_maxdrain_m3:12.1f}  "
              f"cap={r.reservoir_capacity_m3:.0f}  peakvol={r.peak_vol_m3:12.1f}  "
              f"maxlvl={r.max_level_m:.2f}")

    print("\nwrote results/summary/E03_diag_manpolicy.csv, E03_diag_feasibility.csv")


if __name__ == "__main__":
    main()
