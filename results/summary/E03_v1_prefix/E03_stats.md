# E3 다중 시드 통계 분석

생성: `src/analyze_e03.py` · 데이터: `results/E03_seeds/{model}/seed{1..5}/test_metrics.csv`

## 0. 무결성

- 25/25 (model,seed) 조합에 `DONE` 존재, 기록 `row_count`=2700, `test_metrics.csv` 데이터행=2700 (전부 일치)
- 25개 파일이 동일한 2,700개 `scenario_id` 집합을 공유 (set 완전 일치, 중복 없음, return_period/duration 매핑 동일)
- 25개 `run_meta.json`의 `split_seed` 전부 42
- 테스트 계층: 지속시간 9종(60~1440분) × 재현기간 6종(10~100년), 각 층 50개 = 2,700

## 1. 전체 평균 (seed 5개에 대한 mean ± std, 95% CI는 t-분포 df=4, t=2.776)

| model | max_level_m | n_switches | n_intervals_100 | n_intervals_170 | n_dryrun_proxy | overflow_frac |
|---|---|---|---|---|---|---|
| Regular | 9.998 ± 0.003 | 1.75 ± 2.09 | 88.83 ± 61.83 | 4.63 ± 9.37 | 0.101 ± 0.199 | 0.9976 |
| GA-guided | 10.000 ± 0.000 | 0.04 ± 0.05 | 0.08 ± 0.08 | 0.01 ± 0.01 | 0.021 ± 0.034 | 0.9998 |
| PSO-guided | 10.000 ± 0.000 | 0.06 ± 0.12 | 6.45 ± 14.36 | 0.00 ± 0.00 | 0.004 ± 0.009 | 1.0000 |
| GA-only | 10.000 ± 0.000 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.000 ± 0.000 | 1.0000 |
| PSO-only | 10.000 ± 0.000 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.000 ± 0.000 | 1.0000 |

95% CI (하한, 상한):

| model | max_level_m | n_switches | n_dryrun_proxy |
|---|---|---|---|
| Regular | [9.994, 10.002] | [-0.85, 4.35] | [-0.147, 0.348] |
| GA-guided | [9.999, 10.000] | [-0.02, 0.10] | [-0.021, 0.064] |
| PSO-guided | [10.000, 10.000] | [-0.08, 0.21] | [-0.007, 0.015] |
| GA-only | [10.000, 10.000] | [0.00, 0.00] | [0.000, 0.000] |
| PSO-only | [10.000, 10.000] | [0.00, 0.00] | [0.000, 0.000] |

## 2. 월류(overflow) 및 액션 0 고착

`action0_locked` = 한 시나리오에서 n_switches=0 AND n_intervals_100=0 AND n_intervals_170=0 (펌프를 한 번도 켜지 않음)

| model | overflow_frac (mean [min,max]) | action0_locked_frac (mean [min,max]) |
|---|---|---|
| Regular | 0.9976 [0.9915, 1.0000] | 0.3959 [0.0119, 0.9726] |
| GA-guided | 0.9998 [0.9993, 1.0000] | 0.9825 [0.9433, 1.0000] |
| PSO-guided | 1.0000 [1.0000, 1.0000] | 0.9433 [0.7315, 1.0000] |
| GA-only | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |
| PSO-only | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |

시드별 상세는 `E03_overflow_by_seed.csv`.

## 3. Paired Wilcoxon signed-rank (짝: 동일 scenario_id × 동일 seed, N=13,500 쌍)

효과크기: rank-biserial correlation (signed-rank 기준), Cliff's delta. Holm 보정은 24개 검정(4쌍 × 6지표) 전체에 적용.

| 비교 (A vs B) | 지표 | mean A | mean B | mean(A−B) | p (raw) | p (Holm) | 유의(.05) | rank-biserial | Cliff δ |
|---|---|---|---|---|---|---|---|---|---|
| GA-guided vs Regular | max_level_m | 10.000 | 9.998 | 0.002 | 4.85e-04 | 2.43e-03 | ✓ | 0.676 | 0.002 |
| GA-guided vs Regular | n_switches | 0.038 | 1.751 | -1.713 | 0.00e+00 | 0.00e+00 | ✓ | -0.954 | -0.582 |
| GA-guided vs Regular | n_intervals_100 | 0.078 | 88.835 | -88.757 | 0.00e+00 | 0.00e+00 | ✓ | -0.998 | -0.597 |
| GA-guided vs Regular | n_intervals_170 | 0.006 | 4.632 | -4.626 | 5.22e-232 | 9.40e-231 | ✓ | -0.998 | -0.104 |
| GA-guided vs Regular | n_dryrun_proxy | 0.021 | 0.101 | -0.079 | 3.81e-66 | 4.96e-65 | ✓ | -0.621 | -0.038 |
| GA-guided vs Regular | cum_reward | 189.362 | 192.789 | -3.426 | 0.00e+00 | 0.00e+00 | ✓ | -0.998 | -0.067 |
| PSO-guided vs Regular | max_level_m | 10.000 | 9.998 | 0.002 | 7.95e-07 | 4.77e-06 | ✓ | 1.000 | 0.002 |
| PSO-guided vs Regular | n_switches | 0.063 | 1.751 | -1.689 | 0.00e+00 | 0.00e+00 | ✓ | -0.986 | -0.557 |
| PSO-guided vs Regular | n_intervals_100 | 6.451 | 88.835 | -82.384 | 0.00e+00 | 0.00e+00 | ✓ | -1.000 | -0.553 |
| PSO-guided vs Regular | n_intervals_170 | 0.000 | 4.632 | -4.632 | 7.60e-232 | 1.29e-230 | ✓ | -1.000 | -0.104 |
| PSO-guided vs Regular | n_dryrun_proxy | 0.004 | 0.101 | -0.097 | 1.13e-114 | 1.59e-113 | ✓ | -0.923 | -0.052 |
| PSO-guided vs Regular | cum_reward | 189.613 | 192.789 | -3.176 | 0.00e+00 | 0.00e+00 | ✓ | -1.000 | -0.063 |
| GA-guided vs PSO-guided | max_level_m | 10.000 | 10.000 | -0.000 | 1.09e-01 | 2.18e-01 | — | -1.000 | -0.000 |
| GA-guided vs PSO-guided | n_switches | 0.038 | 0.063 | -0.024 | 2.47e-09 | 1.73e-08 | ✓ | -0.206 | -0.038 |
| GA-guided vs PSO-guided | n_intervals_100 | 0.078 | 6.451 | -6.373 | 5.66e-123 | 8.50e-122 | ✓ | -0.860 | -0.040 |
| GA-guided vs PSO-guided | n_intervals_170 | 0.006 | 0.000 | 0.006 | 1.39e-02 | 5.55e-02 | — | 1.000 | 0.001 |
| GA-guided vs PSO-guided | n_dryrun_proxy | 0.021 | 0.004 | 0.017 | 2.04e-26 | 1.63e-25 | ✓ | 0.716 | 0.014 |
| GA-guided vs PSO-guided | cum_reward | 189.362 | 189.613 | -0.250 | 7.51e-153 | 1.20e-151 | ✓ | -0.961 | -0.007 |
| GA-only vs GA-guided | max_level_m | 10.000 | 10.000 | 0.000 | 1.09e-01 | 2.18e-01 | — | 1.000 | 0.000 |
| GA-only vs GA-guided | n_switches | 0.000 | 0.038 | -0.038 | 3.20e-50 | 3.85e-49 | ✓ | -1.000 | -0.017 |
| GA-only vs GA-guided | n_intervals_100 | 0.000 | 0.078 | -0.078 | 4.18e-44 | 4.59e-43 | ✓ | -1.000 | -0.017 |
| GA-only vs GA-guided | n_intervals_170 | 0.000 | 0.006 | -0.006 | 1.39e-02 | 5.55e-02 | — | -1.000 | -0.001 |
| GA-only vs GA-guided | n_dryrun_proxy | 0.000 | 0.021 | -0.021 | 2.40e-43 | 2.40e-42 | ✓ | -1.000 | -0.017 |
| GA-only vs GA-guided | cum_reward | 189.367 | 189.362 | 0.004 | 2.79e-27 | 2.51e-26 | ✓ | 0.802 | 0.002 |

## 4. 핵심 질문

### (a) GA-guided의 dry-running proxy 개선은 유의한가? 장기 지속시간에서 역전되는가?

전체 평균: GA-guided 0.021 vs Regular 0.101 (원고: 4.46 vs 5.34, −16.5%). E3 재현: mean(A−B)=-0.0793, Holm p=4.96e-65, Cliff δ=-0.038.

지속시간별 (GA-guided vs Regular, n_dryrun_proxy):

| duration_min | GA-guided mean±std | Regular mean±std | mean(GA−Reg) | p (raw) | p (Holm, 9검정) |
|---|---|---|---|---|---|
| 60 | 0.003 ± 0.006 | 0.016 ± 0.022 | -0.013 | 4.21e-03 | 1.68e-02 |
| 120 | 0.063 ± 0.096 | 0.054 ± 0.080 | 0.009 | 2.98e-01 | 2.98e-01 |
| 180 | 0.055 ± 0.122 | 0.030 ± 0.044 | 0.025 | 1.40e-02 | 4.21e-02 |
| 240 | 0.065 ± 0.141 | 0.023 ± 0.051 | 0.043 | 1.19e-03 | 5.97e-03 |
| 360 | 0.005 ± 0.012 | 0.017 ± 0.037 | -0.011 | 1.90e-02 | 4.21e-02 |
| 540 | 0.000 ± 0.000 | 0.015 ± 0.021 | -0.015 | 9.40e-05 | 5.64e-04 |
| 720 | 0.000 ± 0.000 | 0.111 ± 0.249 | -0.111 | 2.26e-20 | 1.58e-19 |
| 1080 | 0.000 ± 0.000 | 0.229 ± 0.511 | -0.229 | 5.05e-30 | 4.04e-29 |
| 1440 | 0.000 ± 0.000 | 0.410 ± 0.796 | -0.410 | 1.01e-51 | 9.10e-51 |

### (b) GA-only / PSO-only는 2,700 시나리오 전체에서도 액션 0에 고착되는가?

- **GA-only**: 5개 시드 전부에서 action0_locked_frac = 1.0000~1.0000, overflow_frac = 1.0000~1.0000
- **PSO-only**: 5개 시드 전부에서 action0_locked_frac = 1.0000~1.0000, overflow_frac = 1.0000~1.0000

### (c) 원고 보고 수치와의 비교

| 지표 | 원고 | E3 재현 (mean, 95% CI) |
|---|---|---|
| **Regular** max_level_m | 5.495 | 9.998 [9.994, 10.002] |
| **Regular** n_switches | 85.46 | 1.75 [-0.85, 4.35] |
| **Regular** n_dryrun_proxy | 5.34 | 0.101 [-0.147, 0.348] |
| **GA-guided** max_level_m | 5.607 | 10.000 [9.999, 10.000] |
| **GA-guided** n_switches | 91.5 | 0.04 [-0.02, 0.10] |
| **GA-guided** n_dryrun_proxy | 4.46 | 0.021 [-0.021, 0.064] |
| **PSO-guided** max_level_m | 5.637 | 10.000 [10.000, 10.000] |
| **PSO-guided** n_switches | 78.61 | 0.06 [-0.08, 0.21] |
| **PSO-guided** n_dryrun_proxy | 0.8 | 0.004 [-0.007, 0.015] |
| **GA-only** max_level_m | 5.783 | 10.000 [10.000, 10.000] |
| **GA-only** n_switches | 43.66 | 0.00 [0.00, 0.00] |
| **GA-only** n_dryrun_proxy | 0.05 | 0.000 [0.000, 0.000] |
| **PSO-only** max_level_m | 5.783 | 10.000 [10.000, 10.000] |
| **PSO-only** n_switches | 43.7 | 0.00 [0.00, 0.00] |
| **PSO-only** n_dryrun_proxy | 0.14 | 0.000 [0.000, 0.000] |

원고 파생 주장:
- Regular max_level = 5.495 → E3 9.998
- Regular n_switches = 85.46 → E3 1.75
- GA-guided dry-running −16.5% vs Regular → E3 +78.9%
- PSO-guided n_switches −8.0% vs Regular → E3 +96.4%
- PSO-guided dry-running −85.0% vs Regular → E3 +96.0%
