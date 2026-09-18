#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-e Task C: does optimizing tau lead to the true goal (R1-4)?

Post-processes run_e04e_taskBD.py's per-(scenario, L) raw
(results/E04_enum/saturation_w2_scenarios.csv [+ _L78.csv]) -- no new
simulation except the step-level term-decomposition dump for a handful of
chosen divergence examples.

1. Monotonicity of cum_tau_L in L, per scenario (provably non-decreasing --
   see proof in docs/E04E_REPORT.md Task C; padding any L-sequence with
   action 0 adds a strictly positive increment gamma_h**L * tau(0) >= w3 > 0,
   so ENUM's L+1-optimum can only be >= the L-optimum's value plus that,
   hence cum_tau_L(L+1) > cum_tau_L(L) for every scenario).
2. Cross-sectional Spearman correlation between cum_tau_L and max_level,
   per L, over the scenario set.
3. Divergence enumeration: for each consecutive available L pair, the
   fraction of scenarios where max_level got worse while cum_tau_L (by
   construction) improved.
4. Term-decomposition dump for 3 example scenarios at their divergence step.

Run from src/.
"""
import csv
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E04_DIR = ROOT / 'results' / 'E04_enum'
NEW_WEIGHTS = [0.40, 0.25, 0.25, 0.10]


def load_rows():
    rows = []
    for name in ('saturation_w2_scenarios.csv', 'saturation_w2_scenarios_L78.csv'):
        p = E04_DIR / name
        if p.exists():
            with open(p) as f:
                rows.extend(csv.DictReader(f))
    return rows


def spearman(xs, ys):
    n = len(xs)
    rx = _rank(xs)
    ry = _rank(ys)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n ** 2 - 1)) if n > 1 else float('nan')


def _rank(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def main():
    os.chdir(str(SRC))
    rows = load_rows()
    by_scenario_L: dict[tuple, dict] = {}
    for r in rows:
        by_scenario_L[(r['scenario_id'], int(r['L']))] = r
    scenarios = sorted({sid for sid, _ in by_scenario_L})
    Ls_present = sorted({L for _, L in by_scenario_L})
    print(f'{len(scenarios)} scenarios, L present: {Ls_present}')

    # ---- 1. monotonicity of cum_tau_L ----
    n_scen_full = 0
    n_monotone = 0
    violations = []
    for sid in scenarios:
        seq = [(L, by_scenario_L[(sid, L)]) for L in Ls_present if (sid, L) in by_scenario_L]
        if len(seq) < 2:
            continue
        n_scen_full += 1
        taus = [float(r['cum_tau_L']) for _, r in seq]
        ok = all(taus[i] < taus[i + 1] for i in range(len(taus) - 1))
        if ok:
            n_monotone += 1
        else:
            violations.append((sid, list(zip([L for L, _ in seq], taus))))
    print(f'\n[monotonicity] cum_tau_L strictly increasing in L: '
          f'{n_monotone}/{n_scen_full} scenarios')
    if violations:
        print(f'  VIOLATIONS ({len(violations)}):')
        for sid, tv in violations[:10]:
            print(f'    {sid}: {tv}')

    # ---- 2. Spearman(cum_tau_L, max_level) per L ----
    print('\n[correlation] Spearman(cum_tau_L, max_level) per L (cross-sectional, over scenarios):')
    corr_rows = []
    for L in Ls_present:
        pairs = [(float(r['cum_tau_L']), float(r['max_level_m']))
                 for (sid, LL), r in by_scenario_L.items() if LL == L]
        if len(pairs) > 2:
            rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        else:
            rho = float('nan')
        corr_rows.append({'L': L, 'n': len(pairs), 'spearman_tau_vs_maxlevel': round(rho, 4)})
        print(f'  L={L}: n={len(pairs)} rho={rho:.4f}')

    # ---- 3. divergence: consecutive-L fraction where max_level worsens ----
    print('\n[divergence] fraction of scenarios where max_level worsens L_i -> L_j '
          '(cum_tau_L always improves by construction):')
    div_rows = []
    div_examples = []
    for i in range(len(Ls_present) - 1):
        L1, L2 = Ls_present[i], Ls_present[i + 1]
        worse = []
        by_duration = defaultdict(list)
        by_rp = defaultdict(list)
        n_pairs = 0
        for sid in scenarios:
            k1, k2 = (sid, L1), (sid, L2)
            if k1 not in by_scenario_L or k2 not in by_scenario_L:
                continue
            n_pairs += 1
            r1, r2 = by_scenario_L[k1], by_scenario_L[k2]
            ml1, ml2 = float(r1['max_level_m']), float(r2['max_level_m'])
            is_worse = ml2 > ml1
            by_duration[r1['duration_min']].append(is_worse)
            by_rp[r1['return_period']].append(is_worse)
            if is_worse:
                worse.append((sid, ml1, ml2, ml2 - ml1))
        frac = len(worse) / n_pairs if n_pairs else float('nan')
        div_rows.append({'L_from': L1, 'L_to': L2, 'n_pairs': n_pairs,
                         'frac_worse': round(frac, 4)})
        print(f'  L{L1}->L{L2}: {len(worse)}/{n_pairs} ({frac:.1%}) worsen')
        worse.sort(key=lambda t: -t[3])
        div_examples.append((L1, L2, worse))

    # duration/rp breakdown for the single most informative consecutive pair
    if div_examples:
        L1, L2, worse = max(div_examples, key=lambda t: len(t[2]))
        print(f'\n  [breakdown for L{L1}->L{L2}, the pair with the most worsening cases]')
        by_duration = defaultdict(list)
        by_rp = defaultdict(list)
        for sid in scenarios:
            k1, k2 = (sid, L1), (sid, L2)
            if k1 not in by_scenario_L or k2 not in by_scenario_L:
                continue
            r1, r2 = by_scenario_L[k1], by_scenario_L[k2]
            is_worse = float(r2['max_level_m']) > float(r1['max_level_m'])
            by_duration[r1['duration_min']].append(is_worse)
            by_rp[r1['return_period']].append(is_worse)
        print('  by duration:', {d: f'{sum(v)}/{len(v)}' for d, v in sorted(by_duration.items())})
        print('  by return period:', {d: f'{sum(v)}/{len(v)}' for d, v in sorted(by_rp.items())})

    _write(E04_DIR / 'tau_goal_divergence_w2.csv',
          ['L', 'n', 'spearman_tau_vs_maxlevel'], corr_rows)
    _write(E04_DIR / 'tau_goal_divergence_w2_pairs.csv',
          ['L_from', 'L_to', 'n_pairs', 'frac_worse'], div_rows)

    # ---- 4. example dump: worst 3 across all consecutive pairs ----
    all_examples = []
    for L1, L2, worse in div_examples:
        for sid, ml1, ml2, delta in worse:
            all_examples.append((delta, sid, L1, L2, ml1, ml2))
    all_examples.sort(key=lambda t: -t[0])
    top3 = all_examples[:3]
    print(f'\n[examples] top divergence cases: {[(e[1], e[2], e[3], round(e[4],3), round(e[5],3)) for e in top3]}')

    ex_fields = ['example_rank', 'scenario_id', 'L_from', 'L_to', 'max_level_from',
                'max_level_to', 'delta']
    ex_rows = [{'example_rank': i + 1, 'scenario_id': sid, 'L_from': L1, 'L_to': L2,
               'max_level_from': ml1, 'max_level_to': ml2, 'delta': round(delta, 4)}
              for i, (delta, sid, L1, L2, ml1, ml2) in enumerate(top3)]
    _write(E04_DIR / 'tau_goal_examples_w2.csv', ex_fields, ex_rows)

    print('\n[term decomposition for top examples]')
    for i, (delta, sid, L1, L2, ml1, ml2) in enumerate(top3):
        dump_example(sid, L1, L2)


def _rollout_trace(pl, outfalls, length, forecast='persistence'):
    import numpy as np
    from water_gym import Level

    vol, level, prev = 0.0, float(Level[0]), 0
    max_level = level
    trace = []
    n_dec = length - 1
    for step in range(n_dec):
        clock = step + 1
        fc = _bf(outfalls, clock, length, pl.L, forecast)
        first, det = pl.plan(vol, level, prev, fc, return_detail=True)
        trace.append({
            'step': step, 'vol': vol, 'level': level, 'prev': prev, 'action': first,
        })
        cur_inflow = outfalls[clock]
        new_vol = vol + cur_inflow - pl.pump_rate[first] * pl.minutes
        vol = max(new_vol, 0.0)
        level = float(np.interp(vol, pl.Volume, pl.Level))
        prev = first
        if level > max_level:
            max_level = level
    return trace, max_level


def _bf(outfalls, clock, length, L, mode):
    from lookahead_enum import build_forecast
    return build_forecast(outfalls, clock, length, L, mode)


def _term_breakdown(weights, pumpq_vals, prev_vec, act_vec, vol, level, inflow, minutes=2):
    from water_gym import switching_stability
    pump_rate = sum(q for q, on in zip(pumpq_vals, act_vec) if on)
    qsum = sum(pumpq_vals)
    attempted = pump_rate * minutes
    available = vol + inflow
    excess = 1.0 if attempted > available else 0.0
    actual = min(attempted, available)
    w1, w2, w3, w4 = weights
    vol_reward = (actual / qsum) * (level / 10.0)
    act_reward = switching_stability(prev_vec, act_vec)
    energy_reward = 1.0 - pump_rate / qsum
    return {
        'w1*vol_reward': round(w1 * vol_reward, 4),
        'w2*act_reward': round(w2 * act_reward, 4),
        'w3*energy_reward': round(w3 * energy_reward, 4),
        'w4*excess_penalty': round(w4 * excess * -1.0, 4),
        'tau': round(w1 * vol_reward + w2 * act_reward + w3 * energy_reward
                     + w4 * excess * -1.0, 4),
    }


def dump_example(scenario_id, L1, L2):
    import data_paths
    from sim_swmm import SimSwmm
    from lookahead_enum import LookaheadEnum
    from water_gym import Actions0, pumpq

    _, _, test = data_paths.load_fixed_split()
    inp = next(p for p in test if Path(p).stem == scenario_id)
    rains, outfalls, length = SimSwmm(inp, 2, 'gasan').execute()

    print(f'\n  --- {scenario_id}  L{L1} vs L{L2} ---')
    traces = {}
    max_levels = {}
    for L in (L1, L2):
        pl = LookaheadEnum(NEW_WEIGHTS, minutes=2, L=L, gamma_h=1.0)
        trace, mx = _rollout_trace(pl, outfalls, length)
        traces[L] = trace
        max_levels[L] = mx
        print(f'    L={L}: max_level={mx:.4f}')

    t1, t2 = traces[L1], traces[L2]
    n = min(len(t1), len(t2))
    fork_step = next((i for i in range(n) if t1[i]['action'] != t2[i]['action']), None)
    if fork_step is None:
        print('    (no action divergence found in the overlapping horizon)')
        return
    print(f'    trajectories first diverge at step {fork_step}')
    lo, hi = max(0, fork_step - 1), min(n, fork_step + 4)
    for L, t in ((L1, t1), (L2, t2)):
        print(f'    -- L={L} around the fork --')
        for i in range(lo, hi):
            row = t[i]
            act_vec = Actions0[row['action']]
            prev_vec = Actions0[row['prev']]
            inflow = float(outfalls[row['step'] + 1])
            terms = _term_breakdown(NEW_WEIGHTS, pumpq, prev_vec, act_vec,
                                    row['vol'], row['level'], inflow)
            marker = ' <-- fork' if i == fork_step else ''
            print(f"      step {row['step']:>4} vol={row['vol']:>9.1f} "
                  f"level={row['level']:.4f} prev={row['prev']} a*={row['action']} "
                  f"{terms}{marker}")


def _write(path, fields, rows):
    from atomic_io import atomic_write

    def _w(fd):
        w = csv.DictWriter(fd, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    atomic_write(path, _w, newline='')


if __name__ == '__main__':
    main()
