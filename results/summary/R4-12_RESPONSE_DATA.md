# R4-12 대응 자료 정리 (2026-09-09)

R4-12는 원고 원본 Table 5의 "Testing Time" 값이 Figure 10(§3.4)이 서술하는
"DDQN 추론시간"과 상충한다고 지적했다. 이 문서는 그 대응에 필요한 자료를
한 곳에 모은 것이다. 원자료·상세 도출 과정은 각 절 말미에 인용한
출처 파일을 참고할 것. 해석·평가 문장은 넣지 않고 사실과 수치만 기록한다.
원고 docx는 이 문서 작성으로 수정하지 않았다.

## 1. 원고 원본 Table 5 "Testing Time"의 정체 — 확정 불가

원고 원본 Table 5의 값:

| Model | Training Time (s) | Testing Time (s) |
|---|---|---|
| Regular DDQN | 2,737.03 | 257,224.97 |
| GA-guided DDQN | 12,124.44 | 256,591.19 |

**Testing Time이 무엇을 측정한 값인지는 두 개의 독립적인 실측 경로로도
설명되지 않는다**:

- E10/E14 방법론(결정 1회를 (a) forward pass, (b) 전처리, (c) 환경 step,
  (d) SWMM 1회로 분해)으로 역산하면 시나리오당 95.03~95.27초에 해당하는데,
  실측 SWMM 1회 비용(0.157~0.161초/episode)의 약 590~608배이고, (a)+(b)+(c)+(d)를
  전부 합쳐도 결정당 1ms 미만이라 설명되지 않는다.
- E3 v2 자체의 `test_metrics.csv`에 기록된 시나리오별 실측 `total_time_s`
  합계(Regular seed1 1,053.35s, GA-guided seed1 1,500.29s)와 대조해도
  원고 값과의 배율이 Regular 244.2배, GA-guided 171.0배로 **모델마다
  다르다** — 공통 배율/상수를 곱하거나 더한 값이라는 가설도 성립하지
  않는다.

**결론(기존 조사 유지, 이번에 독립 경로 하나 추가로 확인)**: 원본
Testing Time의 산출 근거는 원본 로그 소실로 추적 불가능하다. 이번 리비전에서는
이 열 자체를 삭제하고 Decision(c)로 대체했다(§3 참고).

참고로 Training Time 열은 사정이 다르다 — 원고 값(Regular 2,737.03s)이
E14 실측 평균(2,960.68s)과 약 +8.17% 차이로, Testing Time의 배율 오차
(171~244배)보다 두 자릿수 이상 작다.

출처: `results/summary/TABLE5_COMPARABILITY.md` §9.

## 2. 재생성판에서 재발한 Table 5 ↔ Figure 10 불일치 — 세 원인과 처리

R4-12 대응으로 Table 5·Figure 10을 재생성하는 과정에서, **같은 종류의
불일치가 다시 발생했다**(Table 5 Decision(c)와 재생성 Figure 10의 값이
서로 달랐음). 원인을 세 가지로 분리해 확인했다.

### 2.1 원인 1 — 측정 범위 차이 (해소됨)

Table 5의 Decision(c) = (a)+(b)+(c)의 합, **SWMM 실행 시간(d)을 명시적으로
제외**한 값이었다. 반면 재생성 초판 Figure 10은 `total_time_s` =
(a)+(b)+(c)+(d) **전부를 포함**한 값을 썼다(SWMM 실행 시간 포함). 같은
모델(Regular, 4기준)로 직접 검증한 결과:

| 범위 | ms/decision |
|---|---|
| (c), SWMM 제외 | 0.278 |
| (c)+(d), SWMM 포함(에피소드당 1회를 결정수로 환산) | 0.835 |
| 재생성 초판 Figure 10이 실제로 쓴 `total_time_s` | 0.699 |

(c)와 (c)+(d)의 비율(약 3배)이 Table 5 대 재생성 초판 Figure 10의 비율
(2.8~4.2배)과 자릿수가 같아, **SWMM 포함 여부가 불일치의 주된 원인**임을
확인했다.

**처리**: 5모델 전부를 (a)+(b)+(c), SWMM 제외로 통일 재측정(§3).

### 2.2 원인 2 — "Regular"가 서로 다른 모델 (각주로 명시, 모델 자체는 통일하지 않음)

Table 5의 Regular(§3.1)는 **2기준 보상**(E14 체크포인트)이고, Figure 10의
Regular(§3.4)는 **4기준 보상**(E03 v2 체크포인트, GA-guided/PSO-guided와
동일 기준)이다 — 이름은 같지만 서로 다른 학습 실행이다. 각 절의 목적상
자연스러운 선택(§3.1은 2기준-vs-4기준 비교, §3.4는 5모델을 동일 4기준
아래 비교)이므로 모델 자체를 통일하지는 않고, **Table 5 각주에 두 Regular가
다른 모델임을 명시하는 문장을 추가**하는 것으로 처리했다.

같은 (c) 정의로 재측정한 뒤 두 Regular의 차이는 약 **1.6%**로 줄었다
(2기준 0.1581±0.0016ms vs 4기준 0.1556±0.0031ms) — 종전 측정범위가
다를 때 추정됐던 격차(약 19%)보다 훨씬 작다.

### 2.3 원인 3 — 측정 잡음 (3회 반복으로 정량화, 완전 제거는 불가능)

같은 모델·같은 범위 정의((c)+(d))를, 서로 다른 두 스크립트로 각각
1회씩 측정했을 때 이미 0.835 vs 0.699로 **약 16~19% 차이**가 있었다.
마이크로벤치마크(수 ms 단위 벽시계 시간) 특성상 시스템 잡음(스레드
스케줄링, GC 등)을 각주만으로 완전히 없앨 수는 없다는 점을 사실로
남긴다.

**처리**: 5모델 전부 **3회 반복 측정**, mean±sd 보고(§3). 이번 반복측정
(같은 스크립트 안에서의 반복)에서는 sd가 평균의 0.4~2% 수준으로 작게
나왔다 — §1에서 확인된 "별도 스크립트 간 16~19% 차이"보다 작은데, 이는
스크립트 간 잡음과 스크립트 내 반복 잡음이 다른 성격일 가능성을 시사하나
확정하지는 않는다.

**부수 관찰**: 이번 DDQN (c) 재측정값(0.155~0.158ms)이 기존
`E10_timing_detailed.csv`의 (c)값(Regular 0.278ms, GA-guided 0.239ms)보다
낮다. 기존 스크립트는 (a)/(b)/(c) 세 구간을 별도의 `perf_counter()` 호출
3쌍으로 나눠 측정했고, 이번 신규 스크립트는 한 결정 전체를 연속된 한
블록으로 측정했다 — 타이머 호출 자체의 오버헤드가 기존값을 부풀렸을
가능성이 있으나 확정 원인은 아니다.

출처: `results/summary/TABLE5_FIGURE10_CONSISTENCY.md` §1-§2, §6-1~6-3.

## 3. PSO-only 측정 버그 — 발견과 정정

Option C 재측정 구현 과정에서 **기존 PSO-only 계산시간 측정에 버그가
있음을 발견**했다.

- **원인**: `src/run_e10_timing.py`의 `time_heuristic()`이 GA-only·PSO-only
  양쪽 모두에 대해 `g.run_genetic_algo(test_inp)`를 호출한다.
  `ParticleSwarmOptimization`(`src/pso.py`)은 `GeneticAlgorithm`을
  상속하지만 `run_genetic_algo`/`genetic_algorithm`을 오버라이드하지
  않는다(PSO 고유 로직은 별도 이름의 `run_pso`/`particle_swarm_optimization`에
  있음). 따라서 **기존 "PSO-only" 측정은 실제로는 상속받은 GA 탐색
  알고리즘을 실행한 것**이었다.
- **증상과의 일치**: 이 버그로 기존 PSO-only(21.792ms)가 GA-only
  (21.398ms)와 거의 같은 값을 보였던 것이 설명된다.
- **정정**: 신규 스크립트(`src/run_e10_unified_c.py`)에서 `run_pso`를
  올바르게 명시적으로 디스패치하도록 구현해 재측정한 결과, **PSO-only는
  GA-only보다 약 14배 빠르다**(1.41ms vs 19.93ms).
- **영향 범위 확인**: `run_e10_timing.py`는 타이밍 측정 전용 스크립트다.
  원고의 실질적인 PSO-only 성능 지표(Table 11-16의 max_level, n_switches
  등)는 `perform_evaluate.evaluate_pso()`(`run_method='run_pso'`로 올바르게
  디스패치, `_evaluate_heuristic()` 경유)를 쓰는 **별도 경로**이므로 **이
  버그의 영향을 받지 않는다** — 코드 직접 대조로 확인 완료.
- **수정 범위**: `run_e10_timing.py` 자체는 리팩터링 금지 원칙에 따라
  수정하지 않았다. 버그는 신규 계측 스크립트에서 정정된 방식으로만
  반영되고, 기존 파일은 사실 기록만 남겼다.

이 발견에 따라 원고 전문에서 GA-only·PSO-only의 계산시간을 "유사하다"고
묶어 서술한 곳을 전수 확인했다 — 결과는
`docs/response/TIMING_STATEMENTS_RECHECK.md` 참조(Abstract 1건 수정
필요, Conclusions 1건 경계선 사례로 등록).

출처: `results/summary/TABLE5_FIGURE10_CONSISTENCY.md` §6-4,
`docs/response/MANUSCRIPT_EDITS.md` #69.

## 4. 최종 통일 값 (5모델 + 2기준 Regular, mean±sd, 3회 반복)

측정 범위: (a) 신경망 forward pass + (b) 상태 전처리(정규화+5스텝 시퀀스
구성) + (c) 환경 step 오버헤드(`WaterGym.step()`, SWMM 미호출). **SWMM
실행 시간은 제외.**

| 모델 | (c), ms/decision, mean±sd (3회 반복) |
|---|---|
| Regular(4기준, E03 v2 — Figure 10용) | 0.1556 ± 0.0031 |
| GA-guided(4기준) | 0.1546 ± 0.0009 |
| PSO-guided(4기준) | 0.1558 ± 0.0017 |
| Regular(2기준, E14 — Table 5용) | 0.1581 ± 0.0016 |
| GA-only | 19.9324 ± 0.1072 |
| PSO-only(버그 수정 후) | 1.4106 ± 0.0063 |

**적용처**:
- Table 5 Decision(c) 열 — Regular 0.158±0.002, GA-guided 0.155±0.001
  (반올림 표기). `docs/response/SECTION_3_1_PACKAGE.md` §2, 원고 Table 5
  실물 반영 확인(2026-09-09).
- Figure 10 — 5모델 전부, 오차막대(sd) 포함.
  `results/summary/figures_final/figure10.png`,
  `results/summary/figure_data/fig10.csv`. 원고 삽입 이미지와 바이트
  단위로 동일함 확인(2026-09-09).

**오차막대 표시 여부**: 표시함 — 3회 반복 측정을 잡음 정량화 목적으로
도입했으므로, 그 결과인 표준편차를 오차막대로 시각화하는 것이 반복측정을
요청한 취지에 부합한다고 판단했다.

원자료: `results/summary/E10_unified_c_summary.csv`(모델별 mean/sd),
`results/summary/E10_unified_c_repeats.csv`(3회 반복 개별값).

## 5. SWMM 비용 (별도 보고, 에피소드당 1회, 결정마다 아님)

| 모델 | SWMM 1회 비용 |
|---|---|
| GA-only | 152.9 ~ 153.7 ms/scenario (3회 반복 범위) |
| PSO-only | 153.0 ~ 153.7 ms/scenario (3회 반복 범위) |
| Regular(2기준, E14) | 0.157 s/episode ≈ 156.7 ms/episode |
| GA-guided(4기준) | 0.161 s/episode ≈ 161.4 ms/episode |

다섯 모델 모두 SWMM 1회 비용이 150~161ms 범위로 자릿수가 일치해 측정의
타당성을 뒷받침한다. Figure 10 캡션과 Table 5 각주(b)에 이 값을 참고용으로
병기했다("...excludes the SWMM simulation, which is executed once per
episode to generate the inflow hydrograph (0.157-0.161 s per episode) and
is not invoked at each decision step.").

## 6. 측정 조건

| 항목 | 값 |
|---|---|
| 하드웨어 | CPU: Intel i9-13900KF / GPU: NVIDIA RTX 3080 Ti (12 GB) / RAM: 94 GB |
| 표본 | E10과 동일한 54-시나리오 층화 부표본(54개 return period–duration 조합에서 1개씩), seed1 |
| 반복 횟수 | 3회 (조건당) |
| 측정 범위 | (a)+(b)+(c), SWMM 실행((d)) 제외 |
| 측정 스크립트 | `src/run_e10_unified_c.py`(신규, 기존 평가 함수 `genetic_algo.py`/`pso.py`/`water_gym.py`/`perform_evaluate.py` 미수정) |
| SWMM 분리 방법 | `WaterGym.reset`을 측정 함수 내부에서만 monkey-patch해 SWMM 소요시간을 별도 캡처, `finally` 블록에서 즉시 원복 |

## 7. R4-12 대응 상태 요약

- 원본 Testing Time의 산출 근거: 확정 불가(§1, 원본 로그 소실).
- 재생성판 재발 불일치의 세 원인: 측정범위(해소), 모델정체성(각주로 명시),
  측정잡음(3회 반복으로 정량화) — §2.
- 부수 발견: PSO-only 측정 버그, Table 11-16에는 영향 없음 확인 — §3.
- Table 5·Figure 10을 (a)+(b)+(c), SWMM 제외 기준으로 통일, mean±sd(3회
  반복) 보고 — §4.
- 남은 차이(2기준 vs 4기준 Regular, 약 1.6%)는 설명된 차이이지 불일치가
  아니다.

출처: `results/summary/TABLE5_FIGURE10_CONSISTENCY.md`(전체, 특히 §6),
`docs/response/MANUSCRIPT_EDITS.md` #26, #64, #65, #69,
`docs/response/TIMING_STATEMENTS_RECHECK.md`.
