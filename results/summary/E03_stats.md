# E3 다중 시드 통계 분석

> **Cliff's δ 부호규약**: "Cliff's δ < 0 indicates lower values for the
> first group (model_a)." — `src/analyze_e03.py`의 `cliffs_delta(x,y)`는
> 이 규약을 처음부터 따르고 있었음(2026-09-02 재확인, 수정 없음).

생성: `src/analyze_e03.py` · 데이터: `results/E03_seeds/{model}/seed{1..5}/test_metrics.csv`

---

## ★ 확정 (2026-08-30) — 이 파일의 수치가 E3 전량 집계의 유일한 정본

E3 v2 집계에 대해 서로 다른 두 수치 집합이 유통되어 검산으로 확정함.

**정본 (이 파일 / `E03_summary.csv` / commit 98d303b 이후 재생성분)**
- Regular max_level 6.224 m, GA-guided 6.033 m, GA−Regular = −0.191 (Cliff δ −0.417)
- Regular n_dryrun_proxy 3.304, cum_reward 153.07 · GA-only n_switches 72.07

**검산 근거**
1. `analyze_e03.py` 재실행 시 출력이 커밋본과 **바이트 동일** (`git status --porcelain results/summary/` 빈 출력). 읽는 경로는 `ROOT="results/E03_seeds"` 한 곳.
2. `results/E03_seeds/Regular/seed1/test_metrics.csv`(2,700행)를 **스크립트 없이 단순 산술평균** → max_level 6.2163, cum_reward 152.60. 정본과 일치.
3. Regular n_dryrun_proxy 시드별 단순평균 = [7.77, 3.38, 0.79, 1.31, 3.26] → 시드평균의 평균 = **3.304** (2단계 집계와 정확히 일치). seed1이 이상치(7.77)일 뿐 계산 오류 아님.

**폐기된 수치의 출처 (혼입 경고)**
- 잘못 유통된 값: Regular max_level 6.140 / dry-running 0.372 / cum_reward 86.28 / GA-only n_switches 58.9 / GA−Regular +0.007.
- 출처: `results/summary/E03_validate2.csv` — **n_test=30, 시드 2개, 학습 500 에피소드**짜리 검증 프로브. 30개 테스트 시나리오가 지속시간 층화가 안 되어 단기 쪽으로 쏠림 → cum_reward가 절반(86 vs 153), dry-running이 1/9(0.37 vs 3.30)로 나온 것.
- **경고**: `E03_validate*.csv`, `E03_weight_sweep.csv`, `E06_*` 등 검증·스윕 프로브 수치는 표본 규모·구성·학습량이 전량 집계(25조합 × 2,700 × 5시드 = 67,500실행)와 다르다. **어떤 문서·응답서에서도 프로브 수치를 전량 집계와 같은 표/문장에 섞어 인용하지 말 것.** 원고 Table 6~16 교체·응답서 인용은 전부 이 파일과 `E03_summary.csv`/`E03_by_duration.csv`/`E03_paired_tests.csv`만을 근거로 한다.

서사 정리 및 원인 분석은 `docs/E03_FINDINGS.md`.

---

## 0. 무결성

- 25/25 (model,seed) 조합에 `DONE` 존재, 기록 `row_count`=2700, `test_metrics.csv` 데이터행=2700 (전부 일치)
- 25개 파일이 동일한 2,700개 `scenario_id` 집합을 공유 (set 완전 일치, 중복 없음, return_period/duration 매핑 동일)
- 25개 `run_meta.json`의 `split_seed` 전부 42
- 테스트 계층: 지속시간 9종(60~1440분) × 재현기간 6종(10~100년), 각 층 50개 = 2,700

## 1. 전체 평균 (seed 5개에 대한 mean ± std, 95% CI는 t-분포 df=4, t=2.776)

| model | max_level_m | n_switches | n_intervals_100 | n_intervals_170 | n_dryrun_proxy | overflow_frac |
|---|---|---|---|---|---|---|
| Regular | 6.224 ± 0.077 | 39.30 ± 6.41 | 307.18 ± 6.18 | 21.03 ± 1.99 | 3.304 ± 2.752 | 0.0000 |
| GA-guided | 6.033 ± 0.091 | 34.20 ± 4.76 | 312.28 ± 6.50 | 18.76 ± 0.99 | 4.873 ± 4.394 | 0.0000 |
| PSO-guided | 6.205 ± 0.165 | 32.54 ± 5.67 | 307.64 ± 3.26 | 19.33 ± 1.46 | 2.157 ± 1.090 | 0.0000 |
| GA-only | 6.338 ± 0.000 | 72.07 ± 0.00 | 257.63 ± 0.00 | 45.12 ± 0.00 | 0.331 ± 0.000 | 0.0000 |
| PSO-only | 6.338 ± 0.000 | 72.19 ± 0.05 | 257.59 ± 0.03 | 45.15 ± 0.02 | 0.336 ± 0.003 | 0.0000 |

95% CI (하한, 상한):

| model | max_level_m | n_switches | n_dryrun_proxy |
|---|---|---|---|
| Regular | [6.128, 6.320] | [31.34, 47.26] | [-0.113, 6.721] |
| GA-guided | [5.920, 6.146] | [28.29, 40.11] | [-0.583, 10.328] |
| PSO-guided | [6.000, 6.409] | [25.51, 39.58] | [0.804, 3.510] |
| GA-only | [6.338, 6.338] | [72.07, 72.07] | [0.331, 0.331] |
| PSO-only | [6.338, 6.338] | [72.13, 72.25] | [0.332, 0.340] |

## 2. 월류(overflow) 및 액션 0 고착

`action0_locked` = 한 시나리오에서 n_switches=0 AND n_intervals_100=0 AND n_intervals_170=0 (펌프를 한 번도 켜지 않음)

| model | overflow_frac (mean [min,max]) | action0_locked_frac (mean [min,max]) |
|---|---|---|
| Regular | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| GA-guided | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| PSO-guided | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| GA-only | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| PSO-only | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

시드별 상세는 `E03_overflow_by_seed.csv`.

## 3. Paired Wilcoxon signed-rank (짝: 동일 scenario_id × 동일 seed, N=13,500 쌍)

효과크기: rank-biserial correlation (signed-rank 기준), Cliff's delta. Holm 보정은 24개 검정(4쌍 × 6지표) 전체에 적용.

| 비교 (A vs B) | 지표 | mean A | mean B | mean(A−B) | p (raw) | p (Holm) | 유의(.05) | rank-biserial | Cliff δ |
|---|---|---|---|---|---|---|---|---|---|
| GA-guided vs Regular | max_level_m | 6.033 | 6.224 | -0.191 | 0.00e+00 | 0.00e+00 | ✓ | -0.649 | -0.417 |
| GA-guided vs Regular | n_switches | 34.195 | 39.298 | -5.102 | 0.00e+00 | 0.00e+00 | ✓ | -0.417 | -0.052 |
| GA-guided vs Regular | n_intervals_100 | 312.278 | 307.179 | 5.098 | 2.19e-255 | 2.19e-254 | ✓ | 0.348 | 0.015 |
| GA-guided vs Regular | n_intervals_170 | 18.763 | 21.032 | -2.269 | 0.00e+00 | 0.00e+00 | ✓ | -0.780 | -0.106 |
| GA-guided vs Regular | n_dryrun_proxy | 4.873 | 3.304 | 1.569 | 2.97e-11 | 5.94e-11 | ✓ | 0.079 | 0.113 |
| GA-guided vs Regular | cum_reward | 152.816 | 153.070 | -0.253 | 5.21e-51 | 2.08e-50 | ✓ | -0.149 | -0.007 |
| PSO-guided vs Regular | max_level_m | 6.205 | 6.224 | -0.019 | 7.29e-21 | 2.19e-20 | ✓ | -0.093 | -0.077 |
| PSO-guided vs Regular | n_switches | 32.543 | 39.298 | -6.754 | 0.00e+00 | 0.00e+00 | ✓ | -0.476 | -0.115 |
| PSO-guided vs Regular | n_intervals_100 | 307.638 | 307.179 | 0.458 | 2.39e-02 | 2.39e-02 | ✓ | 0.023 | 0.002 |
| PSO-guided vs Regular | n_intervals_170 | 19.332 | 21.032 | -1.700 | 0.00e+00 | 0.00e+00 | ✓ | -0.542 | -0.091 |
| PSO-guided vs Regular | n_dryrun_proxy | 2.157 | 3.304 | -1.147 | 2.58e-136 | 1.81e-135 | ✓ | -0.303 | -0.088 |
| PSO-guided vs Regular | cum_reward | 153.664 | 153.070 | 0.594 | 0.00e+00 | 0.00e+00 | ✓ | 0.428 | 0.015 |
| GA-guided vs PSO-guided | max_level_m | 6.033 | 6.205 | -0.172 | 0.00e+00 | 0.00e+00 | ✓ | -0.563 | -0.306 |
| GA-guided vs PSO-guided | n_switches | 34.195 | 32.543 | 1.652 | 2.34e-52 | 1.17e-51 | ✓ | 0.161 | 0.068 |
| GA-guided vs PSO-guided | n_intervals_100 | 312.278 | 307.638 | 4.640 | 6.52e-209 | 5.87e-208 | ✓ | 0.314 | 0.013 |
| GA-guided vs PSO-guided | n_intervals_170 | 18.763 | 19.332 | -0.569 | 2.86e-89 | 1.72e-88 | ✓ | -0.245 | -0.008 |
| GA-guided vs PSO-guided | n_dryrun_proxy | 4.873 | 2.157 | 2.715 | 6.55e-178 | 5.24e-177 | ✓ | 0.326 | 0.213 |
| GA-guided vs PSO-guided | cum_reward | 152.816 | 153.664 | -0.847 | 0.00e+00 | 0.00e+00 | ✓ | -0.492 | -0.023 |
| GA-only vs GA-guided | max_level_m | 6.338 | 6.033 | 0.305 | 0.00e+00 | 0.00e+00 | ✓ | 0.971 | 0.846 |
| GA-only vs GA-guided | n_switches | 72.066 | 34.195 | 37.871 | 0.00e+00 | 0.00e+00 | ✓ | 0.918 | 0.324 |
| GA-only vs GA-guided | n_intervals_100 | 257.628 | 312.278 | -54.650 | 0.00e+00 | 0.00e+00 | ✓ | -1.000 | -0.201 |
| GA-only vs GA-guided | n_intervals_170 | 45.124 | 18.763 | 26.361 | 0.00e+00 | 0.00e+00 | ✓ | 1.000 | 0.694 |
| GA-only vs GA-guided | n_dryrun_proxy | 0.331 | 4.873 | -4.541 | 0.00e+00 | 0.00e+00 | ✓ | -0.960 | -0.557 |
| GA-only vs GA-guided | cum_reward | 151.128 | 152.816 | -1.688 | 0.00e+00 | 0.00e+00 | ✓ | -0.810 | -0.051 |

## 4. 핵심 질문

### (a) GA-guided의 dry-running proxy 개선은 유의한가? 장기 지속시간에서 역전되는가?

전체 평균: GA-guided 4.873 vs Regular 3.304 (원고: 4.46 vs 5.34, −16.5%). E3 재현: mean(A−B)=1.5687, Holm p=5.94e-11, Cliff δ=0.113.

지속시간별 (GA-guided vs Regular, n_dryrun_proxy):

| duration_min | GA-guided mean±std | Regular mean±std | mean(GA−Reg) | p (raw) | p (Holm, 9검정) |
|---|---|---|---|---|---|
| 60 | 0.845 ± 0.908 | 0.271 ± 0.366 | 0.573 | 2.28e-49 | 2.06e-48 |
| 120 | 1.007 ± 1.244 | 0.574 ± 0.642 | 0.433 | 3.05e-31 | 2.44e-30 |
| 180 | 1.629 ± 1.349 | 1.205 ± 1.048 | 0.423 | 7.58e-23 | 5.30e-22 |
| 240 | 1.913 ± 1.553 | 1.990 ± 1.693 | -0.077 | 7.32e-05 | 4.39e-04 |
| 360 | 2.217 ± 1.623 | 3.113 ± 2.566 | -0.895 | 1.75e-01 | 3.49e-01 |
| 540 | 2.956 ± 1.394 | 3.571 ± 3.054 | -0.615 | 3.60e-02 | 1.44e-01 |
| 720 | 3.745 ± 1.751 | 4.189 ± 3.828 | -0.443 | 6.02e-02 | 1.81e-01 |
| 1080 | 8.597 ± 8.265 | 6.361 ± 5.368 | 2.237 | 9.81e-01 | 9.81e-01 |
| 1440 | 20.943 ± 30.900 | 8.461 ± 7.046 | 12.482 | 1.44e-04 | 7.21e-04 |

### (b) GA-only / PSO-only는 2,700 시나리오 전체에서도 액션 0에 고착되는가?

- **GA-only**: 5개 시드 전부에서 action0_locked_frac = 0.0000~0.0000, overflow_frac = 0.0000~0.0000
- **PSO-only**: 5개 시드 전부에서 action0_locked_frac = 0.0000~0.0000, overflow_frac = 0.0000~0.0000

### (c) 원고 보고 수치와의 비교

| 지표 | 원고 | E3 재현 (mean, 95% CI) |
|---|---|---|
| **Regular** max_level_m | 5.495 | 6.224 [6.128, 6.320] |
| **Regular** n_switches | 85.46 | 39.30 [31.34, 47.26] |
| **Regular** n_dryrun_proxy | 5.34 | 3.304 [-0.113, 6.721] |
| **GA-guided** max_level_m | 5.607 | 6.033 [5.920, 6.146] |
| **GA-guided** n_switches | 91.5 | 34.20 [28.29, 40.11] |
| **GA-guided** n_dryrun_proxy | 4.46 | 4.873 [-0.583, 10.328] |
| **PSO-guided** max_level_m | 5.637 | 6.205 [6.000, 6.409] |
| **PSO-guided** n_switches | 78.61 | 32.54 [25.51, 39.58] |
| **PSO-guided** n_dryrun_proxy | 0.8 | 2.157 [0.804, 3.510] |
| **GA-only** max_level_m | 5.783 | 6.338 [6.338, 6.338] |
| **GA-only** n_switches | 43.66 | 72.07 [72.07, 72.07] |
| **GA-only** n_dryrun_proxy | 0.05 | 0.331 [0.331, 0.331] |
| **PSO-only** max_level_m | 5.783 | 6.338 [6.338, 6.338] |
| **PSO-only** n_switches | 43.7 | 72.19 [72.13, 72.25] |
| **PSO-only** n_dryrun_proxy | 0.14 | 0.336 [0.332, 0.340] |

원고 파생 주장:
- Regular max_level = 5.495 → E3 6.224
- Regular n_switches = 85.46 → E3 39.30
- GA-guided dry-running −16.5% vs Regular → E3 -47.5%
- PSO-guided n_switches −8.0% vs Regular → E3 +17.2%
- PSO-guided dry-running −85.0% vs Regular → E3 +34.7%
