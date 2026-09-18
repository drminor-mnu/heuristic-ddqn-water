#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-check: is GA-guided's long-duration (1080/1440 min) weakness the
same phenomenon across E3 v2 (dry-running proxy, seed4) and E12 (n=160
reward diff tail)? No training -- reads existing E3 v2 / E12 outputs only.

Run from src/.
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
E03_DIR = ROOT / 'results' / 'E03_seeds'
E12_DIR = ROOT / 'results' / 'E12_datasize'
SEEDS = [1, 2, 3, 4, 5]
DURATIONS = ['0060', '0120', '0180', '0240', '0360', '0540', '0720', '1080', '1440']


def load_e3(model, seed):
    p = E03_DIR / model / f'seed{seed}' / 'test_metrics.csv'
    with open(p) as f:
        return list(csv.DictReader(f))


def cliffs_delta(x, y):
    """delta = P(x>y) - P(x<y). delta<0 means x tends to be lower than y
    (Cliff's delta < 0 indicates lower values for the first group)."""
    x = np.asarray(x); y = np.asarray(y)
    y_sorted = np.sort(y)
    more = less = 0
    for xi in x:
        lo = np.searchsorted(y_sorted, xi, side='left')   # count of y < xi -> x>y
        hi = np.searchsorted(y_sorted, xi, side='right')  # count of y <= xi
        more += lo                        # x > y pairs
        less += len(y_sorted) - hi        # x < y pairs
    return (more - less) / (len(x) * len(y))


def wilcoxon_safe(a, b):
    if all(x == y for x, y in zip(a, b)):
        return 'n/a (all diffs zero)'
    try:
        _, p = wilcoxon(a, b)
        return p
    except ValueError:
        return float('nan')


def main():
    out_lines = ['# 장시간 사상(1080·1440분) GA-guided 열세 — 통합 확인 (2026-09-02)\n']
    out_lines.append('학습 불필요. E3 v2·E12 기존 산출물만 사용.\n')

    # Load all E3 v2 GA-guided and Regular rows, all 5 seeds, keyed by (seed, scenario_id)
    ga_rows = {}
    reg_rows = {}
    for seed in SEEDS:
        for r in load_e3('GA-guided', seed):
            ga_rows[(seed, r['scenario_id'])] = r
        for r in load_e3('Regular', seed):
            reg_rows[(seed, r['scenario_id'])] = r

    common_keys = sorted(set(ga_rows) & set(reg_rows))
    print(f'common (seed,scenario_id) pairs: {len(common_keys)}')

    # ============================================================
    # 1. Per-duration GA-guided - Regular diffs, all 9 durations, 5-seed pooled
    # ============================================================
    out_lines.append('## 1. E3 v2 5시드 지속시간별 GA-guided − Regular 차이 (9개 지속시간 전부)\n')
    out_lines.append('짝=(시나리오,시드), 지속시간별 pooled (최대 5시드×150시나리오=750쌍/지속시간).\n')
    out_lines.append('| 지속시간(분) | n | max_level diff(mean) | Wilcoxon p | Cliff δ | n_dryrun diff(mean) | Wilcoxon p | Cliff δ | n_switches diff(mean) | Wilcoxon p | Cliff δ |')
    out_lines.append('|---|---|---|---|---|---|---|---|---|---|---|')

    by_dur_rows = defaultdict(list)
    for k in common_keys:
        seed, sid = k
        m = re.search(r'(\d+)m', sid)
        dur = m.group(1).zfill(4) if m else None
        by_dur_rows[dur].append(k)

    dur_summary = {}
    for dur in DURATIONS:
        keys = by_dur_rows.get(dur, [])
        if not keys:
            continue
        ga_ml = [float(ga_rows[k]['max_level_m']) for k in keys]
        reg_ml = [float(reg_rows[k]['max_level_m']) for k in keys]
        ga_dr = [float(ga_rows[k]['n_dryrun_proxy']) for k in keys]
        reg_dr = [float(reg_rows[k]['n_dryrun_proxy']) for k in keys]
        ga_sw = [float(ga_rows[k]['n_switches']) for k in keys]
        reg_sw = [float(reg_rows[k]['n_switches']) for k in keys]

        def stat_block(a, b):
            diff = float(np.mean(a) - np.mean(b))
            p = wilcoxon_safe(a, b)
            d = cliffs_delta(a, b)
            return diff, p, d

        ml_diff, ml_p, ml_d = stat_block(ga_ml, reg_ml)
        dr_diff, dr_p, dr_d = stat_block(ga_dr, reg_dr)
        sw_diff, sw_p, sw_d = stat_block(ga_sw, reg_sw)
        dur_summary[dur] = {'n': len(keys), 'ml_diff': ml_diff, 'dr_diff': dr_diff, 'sw_diff': sw_diff}

        def fmt_p(p):
            return f'{p:.3g}' if isinstance(p, float) else p

        out_lines.append(f'| {int(dur)} | {len(keys)} | {ml_diff:.4f} | {fmt_p(ml_p)} | {ml_d:.4f} | '
                         f'{dr_diff:.4f} | {fmt_p(dr_p)} | {dr_d:.4f} | '
                         f'{sw_diff:.4f} | {fmt_p(sw_p)} | {sw_d:.4f} |')
        print(f'{dur}: n={len(keys)} ml_diff={ml_diff:.4f} dr_diff={dr_diff:.4f} sw_diff={sw_diff:.4f}')

    # ============================================================
    # 2. <1080 vs >=1080 split
    # ============================================================
    out_lines.append('\n## 2. 1,080분 기준 분할 대응비교\n')
    short_keys = [k for dur in DURATIONS if int(dur) < 1080 for k in by_dur_rows.get(dur, [])]
    long_keys = [k for dur in DURATIONS if int(dur) >= 1080 for k in by_dur_rows.get(dur, [])]
    out_lines.append(f'짧은 지속시간(<1080분, 60~720분): n={len(short_keys)}. '
                     f'긴 지속시간(>=1080분, 1080·1440분): n={len(long_keys)}.\n')
    out_lines.append('| 그룹 | 지표 | mean diff(GA−Reg) | Wilcoxon p | Cliff δ |')
    out_lines.append('|---|---|---|---|---|')
    for label, keys in (('짧음(<1080분)', short_keys), ('긺(>=1080분)', long_keys)):
        for metric_key, metric_label in (('max_level_m', 'max_level'), ('n_dryrun_proxy', 'dry-running'), ('n_switches', 'switches')):
            a = [float(ga_rows[k][metric_key]) for k in keys]
            b = [float(reg_rows[k][metric_key]) for k in keys]
            diff = float(np.mean(a) - np.mean(b))
            p = wilcoxon_safe(a, b)
            d = cliffs_delta(a, b)
            pstr = f'{p:.3g}' if isinstance(p, float) else p
            out_lines.append(f'| {label} | {metric_label} | {diff:.4f} | {pstr} | {d:.4f} |')
            print(f'{label} {metric_label}: diff={diff:.4f} p={p} delta={d:.4f}')

    # ============================================================
    # 3. E12 bottom-5% scenario_ids -- do they also underperform in E3 v2?
    # ============================================================
    out_lines.append('\n## 3. E12 하위 5% 시나리오가 E3 v2에서도 열세인가 (scenario_id 교집합)\n')

    # Reconstruct E12 bottom-5% tail scenario_ids (per seed 1-3, n160 Regular/GA-guided)
    def load_e12(model, seed):
        p = E12_DIR / model / 'n160' / f'seed{seed}' / 'test_metrics.csv'
        with open(p) as f:
            return list(csv.DictReader(f))

    e12_rows = []
    for seed in (1, 2, 3):
        reg = {r['scenario_id']: float(r['cum_reward']) for r in load_e12('Regular', seed)}
        ga = {r['scenario_id']: float(r['cum_reward']) for r in load_e12('GA-guided', seed)}
        common = sorted(set(reg) & set(ga))
        for sid in common:
            e12_rows.append({'seed': seed, 'scenario_id': sid, 'diff': ga[sid] - reg[sid]})
    e12_sorted = sorted(e12_rows, key=lambda r: r['diff'])
    n_tail = int(round(len(e12_sorted) * 0.05))
    e12_tail = e12_sorted[:n_tail]
    e12_tail_sids = sorted(set(r['scenario_id'] for r in e12_tail))
    out_lines.append(f'E12(n=160) 하위 5% 시나리오: {len(e12_tail)}행, 고유 scenario_id {len(e12_tail_sids)}개.\n')

    # For these scenario_ids, check E3 v2 (full 3996-training, 5-seed) GA-guided vs Regular
    # max_level and dry-running diffs, pooled across the 5 E3 v2 seeds
    tail_set = set(e12_tail_sids)
    e3_tail_keys = [k for k in common_keys if k[1] in tail_set]
    e3_nontail_keys = [k for k in common_keys if k[1] not in tail_set]
    out_lines.append(f'이 scenario_id 중 E3 v2 test_metrics.csv에도 존재하는 것: '
                     f'{len(set(k[1] for k in e3_tail_keys))} / {len(e12_tail_sids)}개 '
                     f'(E3 v2 5시드 각각에 대응하므로 최대 {len(e12_tail_sids)*5}쌍 가능, 실제 {len(e3_tail_keys)}쌍).\n')

    for label, keys in (('E12 하위5% 시나리오 (E3 v2 기준)', e3_tail_keys),
                        ('나머지 시나리오 (E3 v2 기준)', e3_nontail_keys)):
        if not keys:
            out_lines.append(f'{label}: 데이터 없음\n')
            continue
        a_ml = [float(ga_rows[k]['max_level_m']) for k in keys]
        b_ml = [float(reg_rows[k]['max_level_m']) for k in keys]
        a_dr = [float(ga_rows[k]['n_dryrun_proxy']) for k in keys]
        b_dr = [float(reg_rows[k]['n_dryrun_proxy']) for k in keys]
        ml_diff = float(np.mean(a_ml) - np.mean(b_ml))
        dr_diff = float(np.mean(a_dr) - np.mean(b_dr))
        out_lines.append(f'- **{label}** (n={len(keys)}): max_level diff(GA-Reg)={ml_diff:.4f}, '
                         f'dry-running diff(GA-Reg)={dr_diff:.4f}')
        print(f'{label}: n={len(keys)} ml_diff={ml_diff:.4f} dr_diff={dr_diff:.4f}')

    # ============================================================
    # 4. seed4 by-duration behavior vs other seeds (GA-guided, E3 v2)
    # ============================================================
    out_lines.append('\n\n## 4. seed4의 지속시간별 거동 (GA-guided, E3 v2, 다른 시드와 비교)\n')
    out_lines.append('| 지속시간(분) | seed4 max_level | 형제(1,2,3,5) max_level 평균 | seed4 dry-running | 형제 dry-running 평균 |')
    out_lines.append('|---|---|---|---|---|')
    for dur in DURATIONS:
        s4_ml, s4_dr = [], []
        sib_ml, sib_dr = [], []
        for k in by_dur_rows.get(dur, []):
            seed, sid = k
            if seed == 4:
                s4_ml.append(float(ga_rows[k]['max_level_m']))
                s4_dr.append(float(ga_rows[k]['n_dryrun_proxy']))
            else:
                sib_ml.append(float(ga_rows[k]['max_level_m']))
                sib_dr.append(float(ga_rows[k]['n_dryrun_proxy']))
        if s4_ml and sib_ml:
            out_lines.append(f'| {int(dur)} | {np.mean(s4_ml):.4f} | {np.mean(sib_ml):.4f} | '
                             f'{np.mean(s4_dr):.4f} | {np.mean(sib_dr):.4f} |')
            print(f'{dur}: seed4 ml={np.mean(s4_ml):.4f} sib_ml={np.mean(sib_ml):.4f} '
                  f'seed4 dr={np.mean(s4_dr):.4f} sib_dr={np.mean(sib_dr):.4f}')

    (ROOT / 'results' / 'summary' / 'LONG_DURATION_ANALYSIS.md').write_text('\n'.join(out_lines) + '\n')
    print('\nwrote results/summary/LONG_DURATION_ANALYSIS.md')


if __name__ == '__main__':
    main()
