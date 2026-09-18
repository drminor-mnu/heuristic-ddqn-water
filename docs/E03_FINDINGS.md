# E3 — 다중 시드 결과 서사 및 원인 분석

> **Cliff's δ 부호규약**: "Cliff's δ < 0 indicates lower values for the
> first group." (예: "GA-guided, Regular" 순서면 δ<0은 GA-guided가
> Regular보다 낮음을 뜻함 — 이 문서·`E03_paired_tests.csv`는 이 규약을
> 처음부터 따르고 있었음, 2026-09-02 재확인.)

**작성일**: 2026-08-30 · **데이터**: E3 v2 (수정 코드 3-fix + weights [0.40, 0.25, 0.25, 0.10] + γ=0.7, 5모델 × 5시드 × 2,700 시나리오 = 67,500 실행)
**정본 수치 출처**: `results/summary/E03_summary.csv`, `E03_by_duration.csv`, `E03_by_return_period.csv`, `E03_paired_tests.csv`, `E03_stats.md`. 검산·정본 확정 근거는 `results/summary/E03_stats.md` 상단 「★ 확정」 절.

> **수치 혼입 경고**: `E03_validate*.csv`, `E03_weight_sweep.csv`, `E06_*` 등은 표본 30개 내외·시드 2개·학습 500 에피소드짜리 **검증 프로브**다. 이 문서와 응답서는 전량 집계(위 5개 파일)만 인용한다. 프로브 수치를 전량 집계와 같은 표·문장에 섞지 말 것.

---

## 1. 핵심 결론

### 1.1 요약

E3 v2에서 **5개 모델 전부 2,700 시나리오 × 5시드에서 월류 0, 액션 0 고착 0** — E3 v1의 붕괴(전 모델 ~100% 침수)는 해소됐다.

기능하는 모델 간 비교에서 **어느 모델도 전 지표를 지배하지 않는다.** GA-guided의 강점과 약점은 아래와 같다.

- **peak 수위 제어**: GA-guided 6.033 m vs Regular 6.224 m. paired Wilcoxon(짝 = scenario×seed, N=13,500) mean(GA−Reg) = −0.191 m, Holm p ≈ 0, **Cliff δ = −0.417 (중간 효과크기)**, rank-biserial −0.649. **9개 지속시간 구간 전부에서 GA-guided가 유의하게 낮다.**
- **on/off 스위칭**: 전체 평균 GA-guided 34.2 vs Regular 39.3 (mean diff −5.1, Holm p ≈ 0, 그러나 Cliff δ −0.052 로 효과크기는 무시할 수준). **단기(60~240분)는 사실상 동률, 장기로 갈수록 격차가 벌어진다** (1440분: 84.6 vs 101.7, diff −17.1).
- **dry-running risk-proxy**: 전체 평균(5시드) GA-guided 4.87 vs Regular 3.30 (mean diff +1.57, Holm p 5.9e-11, Cliff δ +0.113) — **5시드 평균으로는 GA-guided가 47% 열세**. 이 열세는 **1440분 구간에 집중**되고(GA 20.9 vs Reg 8.5, diff +12.5), 그 1440분 격차의 대부분은 **한 학습 시드(seed4)의 1440분 시나리오**에서 나온다(seed4 76.0 vs 나머지 4.4~12.0). seed4를 포함한 5시드 평균이 보고 수치이며, §3에서 이 구조를 분해한다 — 구조적 열세가 아니라 **학습 안정성 문제**로, 제안 기법이 특정 조건(저유입 24h)에서 시드에 민감함을 시사한다.
- **cum_reward**: GA-guided 152.82 vs Regular 153.07 (mean diff −0.25, Cliff δ −0.007) — **구분 불가**.

**즉, GA-guided는 "종합적으로 우월한 모델"이 아니라 목적 간 trade-off를 다르게 배치한 모델이다.** peak 수위를 낮게 고정하는 데에서 일관되고 중간 크기의 이점을 보이고, 스위칭에서 장기 이점을 보이며, dry-running proxy에서는 5시드 평균 기준 열세다(대부분 seed4의 학습 안정성 문제, §3).

### 1.2 지속시간별 상세 — max_level (Table 11 대응)

| duration(분) | GA-guided (m, mean±std [95% CI]) | Regular (m) | paired mean(GA−Reg) | Holm p | rank-biserial |
|---|---|---|---|---|---|
| 60 | 6.049 ± 0.061 [5.974, 6.124] | 6.448 ± 0.181 | −0.399 | 1.8e-228 | −0.962 |
| 120 | 6.075 ± 0.071 [5.986, 6.163] | 6.425 ± 0.169 | −0.350 | 1.6e-212 | −0.930 |
| 180 | 6.027 ± 0.055 [5.959, 6.096] | 6.321 ± 0.131 | −0.294 | 1.0e-170 | −0.832 |
| 240 | 6.038 ± 0.059 [5.965, 6.111] | 6.265 ± 0.083 | −0.226 | 2.0e-141 | −0.757 |
| 360 | 6.068 ± 0.076 [5.974, 6.163] | 6.165 ± 0.073 | −0.097 | 4.5e-54 | −0.462 |
| 540 | 6.044 ± 0.093 [5.929, 6.159] | 6.090 ± 0.125 | −0.047 | 1.1e-33 | −0.361 |
| 720 | 6.021 ± 0.129 [5.861, 6.180] | 6.098 ± 0.115 | −0.078 | 3.9e-56 | −0.471 |
| 1080 | 5.990 ± 0.214 [5.725, 6.255] | 6.101 ± 0.142 | −0.111 | 2.5e-41 | −0.402 |
| 1440 | 5.983 ± 0.274 [5.643, 6.324] | 6.101 ± 0.179 | −0.118 | 2.1e-18 | −0.261 |

**서술 정확성 주의 — "장기에서 격차가 벌어진다"는 max_level에는 해당하지 않는다.** max_level의 GA−Regular 격차는 **단기에서 가장 크고**(60분 −0.40 m, δ −0.96) **장기로 갈수록 줄어든다**(1440분 −0.12 m, δ −0.26). 일관된 것은 *격차의 확대*가 아니라 **GA-guided가 지속시간과 무관하게 ~6.0 m 천장을 유지**한다는 점이다. Regular는 단기에 6.45 m까지 치솟았다가 장기에 6.10 m로 내려와 GA-guided에 수렴한다. "장기에서 격차가 벌어진다"가 성립하는 지표는 **스위칭**이다(§1.1). 응답서·원고에는 두 지표를 구분해 서술한다.

### 1.3 5모델 전체 배치 (Table 16 대응, 전체 평균 mean±std [95% CI])

| model | max_level_m | n_switches | n_int_100 | n_int_170 | n_dryrun_proxy | cum_reward |
|---|---|---|---|---|---|---|
| Regular | 6.224 ± 0.077 [6.128, 6.320] | 39.30 ± 6.41 [31.34, 47.26] | 307.18 ± 6.18 | 21.03 ± 1.99 | 3.304 ± 2.752 [−0.113, 6.721] | 153.07 ± 0.42 |
| GA-guided | 6.033 ± 0.091 [5.920, 6.146] | 34.20 ± 4.76 [28.29, 40.11] | 312.28 ± 6.50 | 18.76 ± 0.99 | 4.873 ± 4.394 [−0.583, 10.328] | 152.82 ± 1.02 |
| PSO-guided | 6.205 ± 0.165 [6.000, 6.409] | 32.54 ± 5.67 [25.51, 39.58] | 307.64 ± 3.26 | 19.33 ± 1.46 | 2.157 ± 1.090 [0.804, 3.510] | 153.66 ± 0.73 |
| GA-only | 6.338 ± 0.000 | 72.07 ± 0.00 | 257.63 ± 0.00 | 45.12 ± 0.00 | 0.331 ± 0.000 | 151.13 ± 0.00 |
| PSO-only | 6.338 ± 0.000 | 72.19 ± 0.05 | 257.59 ± 0.03 | 45.15 ± 0.02 | 0.336 ± 0.003 | 151.12 ± 0.00 |

- GA-only/PSO-only는 5시드에서 **결정론적**(std ≈ 0) — 학습이 없어 시나리오가 같으면 출력이 같다. "5회 반복"이 분산을 만들지 않는다.
- **원고와 방향이 뒤집힌 항목**: 원고는 GA-only/PSO-only의 스위칭이 가장 적다(43.7)고 보고했으나, v2에서는 **가장 많다**(72.1). 원고 GA-only dry-running 0.05 → v2 0.33. (원고 GA-only 행의 재현성은 이미 `docs/RESULTS_PROVENANCE.md`에서 "코드 동일성 검증 불가"로 확정됨.)

---

## 2. 리뷰 대응 정리

### R1-6 (Reviewer 1, #6) — "우월성 주장 완화 + 판정 기준 명시"

> "…it is not clear that heuristic-guided DDQN can generally be considered superior. Rather, it appears to provide a computationally cheaper alternative with different operational trade-offs. …clearly define the criteria used to determine which method performs better."

**E3로 뒷받침되는 응답**:

1. **판정 기준을 명시적으로 선언한다.** ① 제1 제약: 전 테스트 시나리오 월류 0 (v2에서 5모델 전부 충족). ② 그 위에서, 실시간 추론 가능 조건 하에 나머지 지표(peak 수위, 스위칭, 펌프사용, dry-running proxy)를 비교. man_policy·원고 수치는 판정 기준에 넣지 않는다.
2. **어느 모델도 전 지표를 지배하지 않음을 §1.3 표로 제시한다.**
   - GA-guided: peak 수위 최저(6.03), 장기 스위칭 최소. dry-running proxy 열세(주로 시드 불안정, §3).
   - PSO-guided: 균형형 — 스위칭 최소 수준(32.5), guided 중 dry-running proxy 최저(2.16).
   - GA-only/PSO-only: dry-running proxy 최저(0.33)이나 스위칭 2배(72), 170 m³/min 가동구간 2배(45 vs ~19), peak 수위 최고(6.34). 결정론적.
   - Regular: 모든 지표에서 중간.
3. **제안 기법의 실질적 강점을 재배치한다** — 종합 성능 우위가 아니라 **추론 속도(온라인 실행 가능)와 데이터 효율**(`docs/CONTEXT.md` 3.4, Figure 10). GA/PSO 단독은 매 스텝 탐색이 필요해 실시간 운영에 부적합하다는 점이 guided 방식의 존재 이유(R1-1/R4-1, 계획서 E4와 연결).
4. Abstract·3.3·3.5·4절에서 "effectively addresses the multi-objective optimization problem" 류 표현을 제거하고 trade-off 서술로 통일.

### R4-7 (Reviewer 4, #7) — "dry-running proxy ≠ 실제 운영 안전성"

> "…clearly distinguish the simulation-based dry-running proxy from actual operational safety unless physical validation is available."

**E3로 뒷받침되는 응답**:

1. `n_dryrun_proxy`는 SWMM 시뮬레이션 상에서 **"시도 방류량 > 가용 저류량"이 발생한 스텝의 이진 카운트**다(`docs/E02_OBJECTIVE_AUDIT.md` §2~3). 펌프 흡입수위·캐비테이션·건식운전 손상 같은 물리 현상을 직접 계측한 값이 아니며, 현장 검증도 없다. 3.3·3.5절에서 "dry-running **risk proxy**"로 명칭을 고정하고 "safety improvement / risk reduction" 단정 표현을 제거한다.
2. **이 proxy는 학습 시드에 매우 민감하다** (§3). 동일 설정·다른 시드에서 GA-guided의 proxy가 시드별 [3.83, 3.16, 2.74, 12.64, 1.99]로 4배 차이 난다. 즉 proxy **비교 자체가 취약**하므로, 모델 간 proxy 차이를 안전성 결론으로 옮기는 것은 이중으로 부적절하다.
3. 물리 검증이 없는 현 단계에서는 proxy를 "탐색적 지표"로만 제시하고, 실제 안전성 평가는 향후 과제(현장 펌프 로그 대조)로 명시(3.5절).

### R4-8 (Reviewer 4, #8) — "일반화 주장 완화"

> "…moderate the generalization claims to reflect that the study was evaluated at a single pumping station using synthetically generated rainfall scenarios within the same modeling framework."

**E3로 뒷받침되는 응답**:

1. 다중 시드는 **시드 노이즈로 인한 과잉 일반화를 걷어냈을 뿐**, 공간적(단일 배수펌프장)·데이터적(동일 Huff 곡선 합성강우)·프레임워크적(동일 SWMM) 일반성을 확장하지 않는다. 이 한계를 3.5·4절에 명시.
2. **오히려 seed4의 국소최적 분기(§3.3~3.4)가 "결론이 학습 난수에 민감하다"는 직접 증거**다 — 동일 하이퍼파라미터·동일 데이터에서 한 시드가 정성적으로 다른 정책(공격적 저수위 유지)으로 수렴했고, 그 정책이 명세된 4기준 보상으로는 거의 페널티를 받지 않아(w4=0.10 과소) 학습 지표만으로는 드러나지 않았다. 이는 일반화·강건성 언어를 더 보수적으로 써야 할 근거이지, 방법의 강건성을 주장할 근거가 아니다.
3. 실측 강우·타 지점 일반화 검증은 계획서 E8의 대상임을 응답서에 적시(R1-5와 공동 대응).

---

## 3. dry-running proxy 열세의 원인 — 가설과 검증

### 3.1 사용자 제기 가설 (구조적)

> GA-guided가 수위를 더 낮게 유지하려면 더 공격적으로 펌핑해야 하고, 그것이 저류량 대비 과펌핑(dry-running proxy 이벤트)을 늘린다.

### 3.2 보고 방침 — seed4를 제외하지 않는다

다중 시드를 요구한 취지는 **변동성을 드러내라**는 것이다. 불리한 시드를 빼면 그 취지에 반하고 리뷰어가 즉시 지적한다. 따라서 **보고 수치는 seed4를 포함한 5시드 평균**(전체 +1.57 / 1440분 +12.48)이며, 아래처럼 분해해 설명한다.

1. **5시드 평균으로는 GA-guided의 dry-running proxy가 Regular보다 높다** (4.87 vs 3.30, +47%).
2. **이 초과분은 seed4의 1440분 시나리오에 집중된다** (seed4 1440분 76.0 vs 나머지 4시드 4.4~12.0; Regular 1440분 8.5).
3. **seed4를 제외하면 GA-guided 1440분 dry-running = 7.19 로 Regular 8.46보다 낮아, 차이가 사라진다.** (이는 "빼고 보고하라"가 아니라 초과분의 소재를 특정하기 위한 대조다.)
4. **즉 구조적 열세가 아니라 학습 안정성 문제다.** 제안 기법(GA-guided)이 특정 조건(저유입 24h 강우)에서 학습 시드에 민감하며, 5시드 중 1개가 정성적으로 다른 정책으로 수렴할 수 있음을 시사한다. → **R4-8(일반화 완화)의 직접 근거**: 동일 하이퍼파라미터·동일 데이터에서 시드만 바꿔 이런 분기가 나오므로, 일반화·강건성 주장은 보수적으로 서술해야 한다.

### 3.3 seed4의 정책이 다른 시드와 어떻게 다른가 (TEST)

**seed4 vs 형제 시드 평균(1·2·3·5), 지속시간별**:

| dur(분) | max_level (s4 / 형제) | dry-run (s4 / 형제) | n_int_100 (s4 / 형제) | n_switches (s4 / 형제) | cum_reward (s4 / 형제) |
|---|---|---|---|---|---|
| 60  | 5.95 / 6.07 | 0.4 / 1.0 | 96 / 95   | 11 / 12 | 32.5 / 32.5 |
| 120 | 5.96 / 6.10 | 0.4 / 1.2 | 148 / 147 | 12 / 14 | 49.3 / 49.4 |
| 180 | 5.94 / 6.05 | 0.9 / 1.8 | 191 / 191 | 16 / 16 | 65.3 / 65.3 |
| 240 | 5.96 / 6.06 | 1.1 / 2.1 | 233 / 234 | 19 / 19 | 81.3 / 81.2 |
| 360 | 6.01 / 6.08 | 2.0 / 2.3 | 308 / 309 | 24 / 24 | 112.7 / 112.8 |
| 540 | 6.01 / 6.05 | 3.7 / 2.8 | 394 / 393 | 36 / 31 | 158.4 / 158.9 |
| 720 | 5.94 / 6.04 | 6.2 / 3.1 | 438 / 434 | 51 / 40 | 202.4 / 203.7 |
| 1080| 5.78 / 6.04 | 23.1 / 5.0 | 502 / 478 | 83 / 60 | 288.6 / 293.1 |
| 1440| **5.64 / 6.07** | **76.0 / 7.2** | **603 / 504** | **108 / 79** | **369.0 / 382.4** |

**패턴**:
- **단기·중기(60~360분): seed4와 형제 시드가 사실상 구분되지 않는다.** max_level·n_int_100·n_switches·cum_reward 전부 동일 수준이고, dry-run은 오히려 seed4가 약간 *낮다*.
- **540분부터 갈라지고, 격차가 지속시간에 따라 단조 증가한다.** 1440분에서 seed4는 형제 대비 수위를 **0.4 m 더 낮게**(5.64 vs 6.07) 유지하고, 이를 위해 100 m³/min 펌프를 **약 100구간 더**(603 vs 504) 가동하며 on/off도 **30회 더**(108 vs 79) 한다. dry-running proxy = "가용 저류량 < 시도 방류량" 스텝의 카운트이므로, 저유입 24h 이벤트에서 이 저율 연속펌핑이 마른 저류지를 반복해 때려 이벤트가 폭증한다.
- **그런데 cum_reward는 seed4가 1440분에서도 형제 대비 −3.6%(369.0 vs 382.4)에 불과하다.** 4기준 보상함수가 seed4의 행동을 거의 페널티로 잡지 못한다 — w1(수위) 항은 수위를 낮게 유지하면 오히려 커지고, w4(dry-run) = 0.10 은 이 상충을 억제하기에 너무 작다.

**해석**: seed4는 **학습 발산(blow-up)이 아니라, 동일 보상함수의 다른 국소최적("공격적 저수위 유지")으로 수렴한 것**이다. 이 국소최적은 명세된 보상 기준으로는 형제 시드와 거의 대등하게 좋으므로(w4가 약해 dry-running을 벌하지 못함), 학습 지표만으로는 이상으로 보이지 않는다. 나머지 4시드는 더 온건한 국소최적에 안착했다. 근본 원인은 optimizer/학습 실행이 아니라 **보상 명세(w4 과소)에서 장시간 저유입 조건 시 복수 국소최적이 갈리는 것**이며, 시드 변이가 이를 드러냈다.

### 3.4 학습 로그(train_log.csv)에서의 seed4 징후

`train_log.csv`는 `episode, train_loss, train_reward`만 기록한다(정책 지표 없음). epochs=1 이므로 에피소드 i는 5시드 모두 같은 학습 시나리오다.

- **전역 지표는 정상**: seed4 전체 train_loss 평균 0.00347 (형제 0.0032~0.0035), train_reward 평균 143.5 (형제 143.5~143.7) — 구분 불가. loss 최대값 0.0534(ep4, 초기)로 seed1의 0.045와 동급, **발산·폭주 없음**. (seed3에 ep0 단일 NaN loss 1건 — 초기화 시점 로깅 아티팩트로 판단, 무의미.)
- **후반 학습에서 seed4가 덜 개선됨**: 마지막 200 에피소드 train_reward 평균 seed4 144.9 vs seed2/3/5 153~160 (seed1도 145.6으로 유사하게 정체).
- **장시간 학습 시나리오에서 간헐적 저득점**: seed4가 형제 대비 에피소드 보상이 200~290 낮은 장시간 에피소드가 학습 전 구간에 산재(ep231, 2326, 3680 …). 해당 에피소드의 loss는 정상(0.003~0.006) → **불안정 수렴이 아니라, 입력공간 일부에서 저보상 정책으로 안정적으로 수렴**.
- 단, 1440분 학습 에피소드 전체 평균 보상은 seed4가 145.8로 형제(133~148)보다 오히려 높다 — 장시간에서 seed4는 **대부분 잘하되 소수 저유입 시나리오에서 붕괴**하는 이분 패턴. 테스트(ε=0, 순수 argmax)에서는 학습 시 잔여 탐험(ε_end=0.05)이 깨주던 연속펌핑 패턴이 그대로 실행되어 dry-running이 학습 지표가 시사하는 것보다 크게 나온다(가설).
- **대조**: Regular seed4의 1440분 dry-run은 4.7(Regular 시드 중 최저)이고, Regular의 상방 이상치는 seed1(20.1)이다. 즉 "seed4 난수 자체"의 문제가 아니라 **GA-guided에서 이 국소최적이 도달 가능**하다는 것 — GA 시연(1-스텝 목적함수, 역시 w4 약함)이 이 최적으로의 경로를 열어줬을 가능성.

**특정 가능한 시나리오** (전부 seed4·1440분, 재현기간 10~100년에 고르게 분포, 공통적으로 n_int_100 ≈ 770~900 / max_level ≈ 5.55~5.70): `10yr_1440m_h1221`(proxy 269), `10yr_1440m_h1239`(265), `20yr_1440m_h1213`(264), `50yr_1440m_h1236`(264), `10yr_1440m_h1217`(262), `30yr_1440m_h1237`(262) … 1440분 1,500 시나리오 중 **상위 86개(5.7%, 전부 seed4)가 GA-guided 1440분 dry-running 총량의 50%**를 차지.

### 3.5 R2-7 대응 — "GA 16% vs PSO 85%의 큰 차이"

> Reviewer 2, #7: "large proxy reduction between GA-guided and PSO-guided: 16% vs. 85%. Such a large difference … sounds strange. Please find a scientific reason."

**v2 근거로 정리한 답** (보고 수치는 5시드 평균, 분해는 괄호):

| | 원고 주장 (vs Regular) | E3 v2 5시드 평균 (vs Regular) | 분해 / 시드 안정성 |
|---|---|---|---|
| PSO-guided dry-running proxy | −85% | **−35%** (지속시간별 −23%~−47%) | 시드별 [2.57, 2.23, 2.88, 0.26, 2.84] — 안정, 분해 불필요 |
| GA-guided dry-running proxy | −16% | **+47%** (5시드 평균, 열세) | 시드별 [3.83, 3.16, 2.74, **12.64**, 1.99]. 초과분은 seed4의 1440분에 집중(§3.2~3.4). seed4 포함 5시드 1440분 dry-run에서 seed4 제외 시 GA 7.19 < Reg 8.46 |

1. **PSO-guided의 감소는 실재한다.** 방향이 원고와 같고 전 지속시간에서 일관되며 시드 간 산포가 작다(5시드 [0.26~2.88]). 다만 크기는 −85%가 아니라 **약 −35%**다. (원고의 −85%는 E3 v1 붕괴 아티팩트 — guided 모델이 무동작에 가깝게 수렴해 proxy가 0에 근접했던 값. `docs/MANUSCRIPT_FIXES.md` MF-9.)
2. **GA-guided는 5시드 평균으로 감소하지 않는다(+47% 열세).** 이 열세는 §3.2~3.4에서 보이듯 seed4가 장시간 저유입 조건에서 다른 국소최적("공격적 저수위 유지")으로 수렴한 데서 나오며, 나머지 4시드만 보면 Regular와 대등하다. 즉 GA-guided의 dry-running 성능은 **학습 시드에 민감**하고, 원고의 −16%는 이 시드 산포(seed 간 24h dry-run 4.4~76.0) 안에 들어 통계적 지위가 없다.
3. **따라서 "16% vs 85%"라는 비대칭의 두 수치 모두 재현되지 않는다.** 재현되는 것은 방향뿐이다 — **PSO-guided는 dry-running proxy를 안정적으로 줄이고(≈ −35%, 시드 무관), GA-guided는 5시드 평균으로 줄이지 않으며 시드에 민감하다.** 두 모델이 저수위 유지를 위해 택한 펌핑 리듬이 다르고(PSO 간헐적 · GA 저율 연속), GA 쪽이 저유입 24h 시나리오에서 복수 국소최적으로 갈릴 여지가 크다는 것이 "큰 차이"의 실체다.
4. **응답서 문안 권고**: 절대 개선율(16%, 85%)을 인용하지 말고, *"Over five training seeds, PSO-guided DDQN reduces the dry-running risk proxy by ≈35% relative to Regular DDQN, consistently across durations and seeds. GA-guided DDQN shows no reduction on the five-seed mean and is sensitive to the training seed: four of five seeds are comparable to Regular DDQN, while one seed converged to an aggressive low-level policy that produces a large excess of proxy events, concentrated in 24 h low-inflow events. This seed sensitivity is reported as a limitation."* GA-guided 시드 민감성을 §3.2 4항대로 **R4-8(일반화 완화)의 근거**로 함께 명시.

---

## 4. 원고 Table 11~16 교체 형식

**교체 원칙** (`docs/REVISION_EXPERIMENT_PLAN.md` E1·E3, `docs/MANUSCRIPT_FIXES.md` MF-9):
- 수치는 스크립트 출력을 그대로 복사한다. 수작업 전사 금지 (Table 8 오류가 전사에서 발생).
- 모든 셀을 `mean ± std` 로 쓰고, **각 표 하단 또는 캡션에 5시드 기준·95% CI(t-분포, df=4, t=2.776)** 를 명시. Table 16(요약)은 셀마다 CI를 병기.
- `n = 5 seeds × 2,700 scenarios`. 검정: paired Wilcoxon signed-rank(짝 = scenario×seed), 효과크기 Cliff δ, Holm 보정(24검정). 유의성 별표는 캡션에 기준 명시.

**메트릭 ↔ 표 ↔ 소스 파일**:

| 원고 표 | 내용 | v2 소스 | 형식 |
|---|---|---|---|
| Table 11 | 5모델 max water level, 지속시간별 | `E03_by_duration.csv` (`max_level_m_*`) | 행 = 지속시간 9종(60~1440), 열 = 5모델, 셀 = `mean ± std` |
| Table 12 | 5모델 on/off 변경 횟수, 지속시간별 | `E03_by_duration.csv` (`n_switches_*`) | 동일 |
| Table 13 | 5모델 100 m³/min 가동구간 수, 지속시간별 | `E03_by_duration.csv` (`n_intervals_100_*`) | 동일 |
| Table 14 | 5모델 170 m³/min 가동구간 수, 지속시간별 | `E03_by_duration.csv` (`n_intervals_170_*`) | 동일 |
| Table 15 | 5모델 dry-running risk-proxy, 지속시간별 | `E03_by_duration.csv` (`n_dryrun_proxy_*`) + `E03_dryrun_by_duration.csv` (GA vs Reg 검정) | 동일. **캡션에 시드 민감성 각주**(§3.3) |
| Table 16 | 전체 평균 요약 (5모델 × 5지표) | `E03_summary.csv` | 셀 = `mean ± std [95% CI lo, hi]` |

**Table 16 교체본 (그대로 사용 가능)** — 위 §1.3 표. 캡션: *"Mean ± standard deviation over 5 training seeds; each seed evaluated on all 2,700 test scenarios. Values in brackets are 95% confidence intervals (Student's t, df = 4). GA-only and PSO-only are deterministic given the scenario set (std ≈ 0)."*

**Table 11 교체본 (지속시간 × max_level, GA-guided/Regular 발췌)** — 위 §1.2 표의 `GA-guided`·`Regular` 열이 그대로 들어가며, PSO-guided/GA-only/PSO-only 열은 `E03_by_duration.csv`에서 동일 형식으로 채운다. 예시 두 행:

| duration(분) | Regular | GA-guided | PSO-guided | GA-only | PSO-only |
|---|---|---|---|---|---|
| 60 | 6.448 ± 0.181 | 6.049 ± 0.061 | 6.378 ± 0.312 | 6.53 ± 0.00 | 6.53 ± 0.00 |
| 1440 | 6.101 ± 0.179 | 5.983 ± 0.274 | 5.897 ± 0.105 | 6.30 ± 0.00 | 6.30 ± 0.00 |

(나머지 7개 지속시간 행과 다른 표는 `E03_by_duration.csv`에서 동일 형식으로 채운다. 이 문서는 형식 지정용이며 최종 표는 `src/analyze_e03.py` 산출 CSV를 스크립트로 렌더한다.)

**Table 15 캡션 추가 문장 (필수)** — 표의 모든 셀은 5시드 전부를 포함한다. 각주는 seed4를 빼는 것이 아니라 분산의 소재를 밝힌다:
*"All values are means over the five training seeds. For GA-guided DDQN the dry-running risk proxy is seed-sensitive at long durations: the five per-seed means for 24 h events are 4.4 / 7.3 / 12.0 / 76.0 / 5.1, i.e. four seeds are comparable to Regular DDQN (8.5) while one seed converged to an aggressive low-level control policy (peak level ≈ 5.6 m vs ≈ 6.0 m; ~100 additional 100 m³/min operating intervals) that produces a large proxy excess. The five-seed mean reported here therefore reflects this training-stability variance rather than a systematic disadvantage of GA guidance. See docs/E03_FINDINGS.md §3."*

---

## 5. 완료 조건 대조 (계획서 E3)

- [x] 무결성: 25조합 2,700행, scenario_id 집합 동일, run_meta에 weights/gamma 신설정 기록 — `E03_stats.md` §0
- [x] 집계: mean±std, 95% CI(t, df=4), 지속시간별·재현기간별 분해 — `E03_summary.csv`, `E03_by_duration.csv`, `E03_by_return_period.csv`
- [x] paired Wilcoxon + Cliff δ + Holm — `E03_paired_tests.csv`, `E03_stats.md` §3
- [x] (a) 5모델 월류 해소 확인 — 전부 0
- [x] (c) GA-guided vs Regular 판정 — peak 수위 우세(δ −0.42), 스위칭 장기 우세, dry-running 5시드 평균 열세(seed4 학습 안정성 문제로 분해), 총보상 동률. **종합 우월 아님, trade-off.**
- [x] (d) 지속시간별 역전 — dry-running proxy에서 1440분 열세(seed4 1440분 주도, §3.3), max_level은 역전 없음
- [x] R2-7 원인 규명 — §3.5 (보상 명세 w4 과소 → 장시간 저유입 시 복수 국소최적, 시드 변이가 드러냄)
- [x] seed4 원인 규명 — §3.3~3.4 (발산 아님, 동일 보상함수의 다른 국소최적; train_log 전역 정상·후반 정체; R4-8 근거)
- [ ] 표 11~16 최종 렌더 스크립트(`src/analyze_e03.py` → 표 CSV) — 형식은 §4에 확정, 렌더는 원고 수정 단계에서
