"""E3 aggregation + paired tests. Reads results/E03_seeds/{model}/seed{n}/test_metrics.csv.
Writes results/summary/E03_summary.csv, E03_by_duration.csv, E03_by_return_period.csv, E03_stats.md.
No refactoring of experiment code; pure post-hoc analysis."""
import glob
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "results/E03_seeds"
OUT = "results/summary"
MODELS = ["Regular", "GA-guided", "PSO-guided", "GA-only", "PSO-only"]
SEEDS = [1, 2, 3, 4, 5]
METRICS = ["max_level_m", "n_switches", "n_intervals_100", "n_intervals_170",
           "n_dryrun_proxy", "cum_reward", "overflow_flag"]
T_CRIT_DF4 = stats.t.ppf(0.975, 4)  # 2.7764


def load_all():
    frames = []
    for m in MODELS:
        for s in SEEDS:
            df = pd.read_csv(f"{ROOT}/{m}/seed{s}/test_metrics.csv", dtype={"duration_min": str})
            df["model"] = m
            df["seed"] = s
            df["duration_int"] = df["duration_min"].astype(int)
            df["is_action0_locked"] = ((df.n_switches == 0) & (df.n_intervals_100 == 0)
                                       & (df.n_intervals_170 == 0)).astype(int)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def ci_row(per_seed_means):
    """per_seed_means: array len 5 (one mean per seed). Return mean, std(ddof=1), CI half-width, lo, hi."""
    a = np.asarray(per_seed_means, float)
    mean = a.mean()
    sd = a.std(ddof=1)
    half = T_CRIT_DF4 * sd / np.sqrt(len(a))
    return mean, sd, half, mean - half, mean + half


def aggregate(df, groupcols):
    """Two-stage: per (group, seed) mean over scenarios, then across seeds mean+-std+95%CI."""
    rows = []
    stage1 = df.groupby(groupcols + ["model", "seed"])[METRICS].mean().reset_index()
    for keys, g in stage1.groupby(groupcols + ["model"]):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(groupcols + ["model"], keys))
        rec["n_seeds"] = g["seed"].nunique()
        for met in METRICS:
            mean, sd, half, lo, hi = ci_row(g[met].values)
            rec[f"{met}_mean"] = mean
            rec[f"{met}_std"] = sd
            rec[f"{met}_ci95_half"] = half
            rec[f"{met}_ci95_lo"] = lo
            rec[f"{met}_ci95_hi"] = hi
        rows.append(rec)
    return pd.DataFrame(rows)


def cliffs_delta(x, y):
    """P(x>y) - P(x<y) via rank method. x,y paired but treated as samples here."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    # efficient: use sorting
    allv = np.concatenate([x, y])
    order = stats.rankdata(allv)
    rx = order[:n].sum()
    # Mann-Whitney U for x
    u_x = rx - n * (n + 1) / 2.0
    delta = 2.0 * u_x / (n * len(y)) - 1.0
    return delta


def paired_test(df, model_a, model_b, metric):
    """Pair on (scenario_id, seed). Wilcoxon signed-rank on a-b."""
    a = df[df.model == model_a][["scenario_id", "seed", metric]].rename(columns={metric: "a"})
    b = df[df.model == model_b][["scenario_id", "seed", metric]].rename(columns={metric: "b"})
    m = a.merge(b, on=["scenario_id", "seed"], how="inner")
    diff = (m["a"] - m["b"]).values
    n_pairs = len(diff)
    n_nonzero = int((diff != 0).sum())
    mean_a, mean_b = m["a"].mean(), m["b"].mean()
    if n_nonzero == 0:
        return dict(model_a=model_a, model_b=model_b, metric=metric, n_pairs=n_pairs,
                    n_nonzero=0, mean_a=mean_a, mean_b=mean_b, mean_diff=0.0,
                    W=np.nan, p=1.0, rank_biserial=0.0, cliffs_delta=0.0)
    try:
        W, p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided", correction=False,
                              mode="approx")
    except ValueError:
        W, p = np.nan, 1.0
    # rank-biserial for signed-rank: r = W+ - W- normalised
    nz = diff[diff != 0]
    ranks = stats.rankdata(np.abs(nz))
    r_plus = ranks[nz > 0].sum()
    r_minus = ranks[nz < 0].sum()
    total = r_plus + r_minus
    rank_biserial = (r_plus - r_minus) / total if total else 0.0
    cd = cliffs_delta(m["a"].values, m["b"].values)
    return dict(model_a=model_a, model_b=model_b, metric=metric, n_pairs=n_pairs,
                n_nonzero=n_nonzero, mean_a=mean_a, mean_b=mean_b,
                mean_diff=float(np.mean(diff)), W=float(W) if W == W else np.nan, p=float(p),
                rank_biserial=float(rank_biserial), cliffs_delta=float(cd))


def holm(pvals):
    """Return Holm-adjusted p-values in original order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    df = load_all()
    print(f"loaded {len(df)} rows ({df.model.nunique()} models x {df.seed.nunique()} seeds)")

    # ---- overall summary ----
    df["_all"] = "all"
    summ = aggregate(df, ["_all"]).drop(columns=["_all"])
    summ["scope"] = "overall"
    by_dur = aggregate(df, ["duration_int"])
    by_dur["scope"] = "by_duration"
    by_rp = aggregate(df, ["return_period"])
    by_rp["scope"] = "by_return_period"

    summ.to_csv(f"{OUT}/E03_summary.csv", index=False)
    by_dur.sort_values(["duration_int", "model"]).to_csv(f"{OUT}/E03_by_duration.csv", index=False)
    by_rp.sort_values(["return_period", "model"]).to_csv(f"{OUT}/E03_by_return_period.csv", index=False)
    print("wrote E03_summary.csv, E03_by_duration.csv, E03_by_return_period.csv")

    # ---- overflow / action-0 lock-in table ----
    lock = df.groupby(["model", "seed"]).agg(
        overflow_frac=("overflow_flag", "mean"),
        action0_locked_frac=("is_action0_locked", "mean"),
        zero_switch_frac=("n_switches", lambda s: (s == 0).mean()),
    ).reset_index()
    lock_summary = lock.groupby("model").agg(
        overflow_frac_mean=("overflow_frac", "mean"),
        overflow_frac_min=("overflow_frac", "min"),
        overflow_frac_max=("overflow_frac", "max"),
        action0_locked_mean=("action0_locked_frac", "mean"),
        action0_locked_min=("action0_locked_frac", "min"),
        action0_locked_max=("action0_locked_frac", "max"),
    ).reset_index()
    lock.to_csv(f"{OUT}/E03_overflow_by_seed.csv", index=False)

    # ---- paired tests ----
    pairs = [("GA-guided", "Regular"), ("PSO-guided", "Regular"),
             ("GA-guided", "PSO-guided"), ("GA-only", "GA-guided")]
    test_metrics = ["max_level_m", "n_switches", "n_intervals_100", "n_intervals_170",
                    "n_dryrun_proxy", "cum_reward"]
    results = []
    for a, b in pairs:
        for met in test_metrics:
            results.append(paired_test(df, a, b, met))
    res = pd.DataFrame(results)
    res["p_holm"] = holm(res["p"].values)
    res["sig_holm_05"] = res["p_holm"] < 0.05
    res.to_csv(f"{OUT}/E03_paired_tests.csv", index=False)

    # ---- dry-running by duration: GA-guided vs Regular, per duration ----
    dur_rows = []
    for d in sorted(df.duration_int.unique()):
        sub = df[df.duration_int == d]
        r = paired_test(sub, "GA-guided", "Regular", "n_dryrun_proxy")
        r["duration_min"] = d
        # per-seed means
        s1 = sub[sub.model == "GA-guided"].groupby("seed").n_dryrun_proxy.mean()
        s2 = sub[sub.model == "Regular"].groupby("seed").n_dryrun_proxy.mean()
        r["GAguided_mean"] = s1.mean()
        r["GAguided_std"] = s1.std(ddof=1)
        r["Regular_mean"] = s2.mean()
        r["Regular_std"] = s2.std(ddof=1)
        dur_rows.append(r)
    dur_dry = pd.DataFrame(dur_rows)
    dur_dry["p_holm"] = holm(dur_dry["p"].values)
    dur_dry.to_csv(f"{OUT}/E03_dryrun_by_duration.csv", index=False)

    # ---- manuscript comparison ----
    ms = {
        "Regular":     dict(max_level_m=5.495, n_switches=85.46, n_intervals_100=308.25, n_intervals_170=19.79, n_dryrun_proxy=5.34),
        "GA-guided":   dict(max_level_m=5.607, n_switches=91.50, n_intervals_100=312.84, n_intervals_170=18.97, n_dryrun_proxy=4.46),
        "PSO-guided":  dict(max_level_m=5.637, n_switches=78.61, n_intervals_100=449.48, n_intervals_170=19.16, n_dryrun_proxy=0.80),
        "GA-only":     dict(max_level_m=5.783, n_switches=43.66, n_intervals_100=np.nan, n_intervals_170=np.nan, n_dryrun_proxy=0.05),
        "PSO-only":    dict(max_level_m=5.783, n_switches=43.70, n_intervals_100=np.nan, n_intervals_170=np.nan, n_dryrun_proxy=0.14),
    }
    cmp_rows = []
    for m in MODELS:
        row = summ[summ.model == m].iloc[0]
        for met in ["max_level_m", "n_switches", "n_intervals_100", "n_intervals_170", "n_dryrun_proxy"]:
            cmp_rows.append(dict(model=m, metric=met,
                                 manuscript=ms[m][met],
                                 e3_mean=row[f"{met}_mean"],
                                 e3_ci95_lo=row[f"{met}_ci95_lo"],
                                 e3_ci95_hi=row[f"{met}_ci95_hi"]))
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(f"{OUT}/E03_manuscript_comparison.csv", index=False)

    # derived pct-change claims
    def pct(a, b):
        return (b - a) / b * 100.0  # reduction of a relative to b (Regular)
    reg = summ[summ.model == "Regular"].iloc[0]
    gag = summ[summ.model == "GA-guided"].iloc[0]
    pso = summ[summ.model == "PSO-guided"].iloc[0]

    # ---- write stats.md ----
    with open(f"{OUT}/E03_stats.md", "w") as f:
        w = f.write
        w("# E3 다중 시드 통계 분석\n\n")
        w("생성: `src/analyze_e03.py` · 데이터: `results/E03_seeds/{model}/seed{1..5}/test_metrics.csv`\n\n")
        w("## 0. 무결성\n\n")
        w("- 25/25 (model,seed) 조합에 `DONE` 존재, 기록 `row_count`=2700, `test_metrics.csv` 데이터행=2700 (전부 일치)\n")
        w("- 25개 파일이 동일한 2,700개 `scenario_id` 집합을 공유 (set 완전 일치, 중복 없음, return_period/duration 매핑 동일)\n")
        w("- 25개 `run_meta.json`의 `split_seed` 전부 42\n")
        w("- 테스트 계층: 지속시간 9종(60~1440분) × 재현기간 6종(10~100년), 각 층 50개 = 2,700\n\n")

        w("## 1. 전체 평균 (seed 5개에 대한 mean ± std, 95% CI는 t-분포 df=4, t=2.776)\n\n")
        w("| model | max_level_m | n_switches | n_intervals_100 | n_intervals_170 | n_dryrun_proxy | overflow_frac |\n")
        w("|---|---|---|---|---|---|---|\n")
        for m in MODELS:
            r = summ[summ.model == m].iloc[0]
            lk = lock_summary[lock_summary.model == m].iloc[0]
            w(f"| {m} "
              f"| {r.max_level_m_mean:.3f} ± {r.max_level_m_std:.3f} "
              f"| {r.n_switches_mean:.2f} ± {r.n_switches_std:.2f} "
              f"| {r.n_intervals_100_mean:.2f} ± {r.n_intervals_100_std:.2f} "
              f"| {r.n_intervals_170_mean:.2f} ± {r.n_intervals_170_std:.2f} "
              f"| {r.n_dryrun_proxy_mean:.3f} ± {r.n_dryrun_proxy_std:.3f} "
              f"| {lk.overflow_frac_mean:.4f} |\n")
        w("\n95% CI (하한, 상한):\n\n")
        w("| model | max_level_m | n_switches | n_dryrun_proxy |\n|---|---|---|---|\n")
        for m in MODELS:
            r = summ[summ.model == m].iloc[0]
            w(f"| {m} | [{r.max_level_m_ci95_lo:.3f}, {r.max_level_m_ci95_hi:.3f}] "
              f"| [{r.n_switches_ci95_lo:.2f}, {r.n_switches_ci95_hi:.2f}] "
              f"| [{r.n_dryrun_proxy_ci95_lo:.3f}, {r.n_dryrun_proxy_ci95_hi:.3f}] |\n")

        w("\n## 2. 월류(overflow) 및 액션 0 고착\n\n")
        w("`action0_locked` = 한 시나리오에서 n_switches=0 AND n_intervals_100=0 AND n_intervals_170=0 (펌프를 한 번도 켜지 않음)\n\n")
        w("| model | overflow_frac (mean [min,max]) | action0_locked_frac (mean [min,max]) |\n|---|---|---|\n")
        for m in MODELS:
            lk = lock_summary[lock_summary.model == m].iloc[0]
            w(f"| {m} | {lk.overflow_frac_mean:.4f} [{lk.overflow_frac_min:.4f}, {lk.overflow_frac_max:.4f}] "
              f"| {lk.action0_locked_mean:.4f} [{lk.action0_locked_min:.4f}, {lk.action0_locked_max:.4f}] |\n")
        w("\n시드별 상세는 `E03_overflow_by_seed.csv`.\n")

        w("\n## 3. Paired Wilcoxon signed-rank (짝: 동일 scenario_id × 동일 seed, N=13,500 쌍)\n\n")
        w("효과크기: rank-biserial correlation (signed-rank 기준), Cliff's delta. Holm 보정은 24개 검정(4쌍 × 6지표) 전체에 적용.\n\n")
        w("| 비교 (A vs B) | 지표 | mean A | mean B | mean(A−B) | p (raw) | p (Holm) | 유의(.05) | rank-biserial | Cliff δ |\n")
        w("|---|---|---|---|---|---|---|---|---|---|\n")
        for _, r in res.iterrows():
            w(f"| {r.model_a} vs {r.model_b} | {r.metric} | {r.mean_a:.3f} | {r.mean_b:.3f} "
              f"| {r.mean_diff:.3f} | {r.p:.2e} | {r.p_holm:.2e} | {'✓' if r.sig_holm_05 else '—'} "
              f"| {r.rank_biserial:.3f} | {r.cliffs_delta:.3f} |\n")

        w("\n## 4. 핵심 질문\n\n")
        w("### (a) GA-guided의 dry-running proxy 개선은 유의한가? 장기 지속시간에서 역전되는가?\n\n")
        w(f"전체 평균: GA-guided {gag.n_dryrun_proxy_mean:.3f} vs Regular {reg.n_dryrun_proxy_mean:.3f} "
          f"(원고: 4.46 vs 5.34, −16.5%). ")
        rr = res[(res.model_a == 'GA-guided') & (res.model_b == 'Regular') & (res.metric == 'n_dryrun_proxy')].iloc[0]
        w(f"E3 재현: mean(A−B)={rr.mean_diff:.4f}, Holm p={rr.p_holm:.2e}, Cliff δ={rr.cliffs_delta:.3f}.\n\n")
        w("지속시간별 (GA-guided vs Regular, n_dryrun_proxy):\n\n")
        w("| duration_min | GA-guided mean±std | Regular mean±std | mean(GA−Reg) | p (raw) | p (Holm, 9검정) |\n|---|---|---|---|---|---|\n")
        for _, r in dur_dry.iterrows():
            w(f"| {int(r.duration_min)} | {r.GAguided_mean:.3f} ± {r.GAguided_std:.3f} "
              f"| {r.Regular_mean:.3f} ± {r.Regular_std:.3f} | {r.mean_diff:.3f} "
              f"| {r.p:.2e} | {r.p_holm:.2e} |\n")

        w("\n### (b) GA-only / PSO-only는 2,700 시나리오 전체에서도 액션 0에 고착되는가?\n\n")
        for m in ["GA-only", "PSO-only"]:
            lk = lock_summary[lock_summary.model == m].iloc[0]
            w(f"- **{m}**: 5개 시드 전부에서 action0_locked_frac = "
              f"{lk.action0_locked_min:.4f}~{lk.action0_locked_max:.4f}, overflow_frac = "
              f"{lk.overflow_frac_min:.4f}~{lk.overflow_frac_max:.4f}\n")
        w("\n### (c) 원고 보고 수치와의 비교\n\n")
        w("| 지표 | 원고 | E3 재현 (mean, 95% CI) |\n|---|---|---|\n")
        for m in MODELS:
            r = summ[summ.model == m].iloc[0]
            w(f"| **{m}** max_level_m | {ms[m]['max_level_m']} "
              f"| {r.max_level_m_mean:.3f} [{r.max_level_m_ci95_lo:.3f}, {r.max_level_m_ci95_hi:.3f}] |\n")
            w(f"| **{m}** n_switches | {ms[m]['n_switches']} "
              f"| {r.n_switches_mean:.2f} [{r.n_switches_ci95_lo:.2f}, {r.n_switches_ci95_hi:.2f}] |\n")
            w(f"| **{m}** n_dryrun_proxy | {ms[m]['n_dryrun_proxy']} "
              f"| {r.n_dryrun_proxy_mean:.3f} [{r.n_dryrun_proxy_ci95_lo:.3f}, {r.n_dryrun_proxy_ci95_hi:.3f}] |\n")
        w("\n원고 파생 주장:\n")
        w(f"- Regular max_level = 5.495 → E3 {reg.max_level_m_mean:.3f}\n")
        w(f"- Regular n_switches = 85.46 → E3 {reg.n_switches_mean:.2f}\n")
        w(f"- GA-guided dry-running −16.5% vs Regular → E3 { pct(gag.n_dryrun_proxy_mean, reg.n_dryrun_proxy_mean):+.1f}%\n")
        w(f"- PSO-guided n_switches −8.0% vs Regular → E3 { pct(pso.n_switches_mean, reg.n_switches_mean):+.1f}%\n")
        w(f"- PSO-guided dry-running −85.0% vs Regular → E3 { pct(pso.n_dryrun_proxy_mean, reg.n_dryrun_proxy_mean):+.1f}%\n")

    # console summary for the three questions
    print("\n=== (b) overflow / action0 lock-in ===")
    print(lock_summary.to_string(index=False))
    print("\n=== (c) manuscript comparison ===")
    print(cmp.to_string(index=False))
    print("\n=== (a) dryrun by duration GA-guided vs Regular ===")
    print(dur_dry[["duration_min", "GAguided_mean", "Regular_mean", "mean_diff", "p", "p_holm"]].to_string(index=False))
    print("\n=== paired tests ===")
    print(res[["model_a", "model_b", "metric", "mean_a", "mean_b", "mean_diff", "p", "p_holm", "sig_holm_05", "cliffs_delta"]].to_string(index=False))


if __name__ == "__main__":
    main()
