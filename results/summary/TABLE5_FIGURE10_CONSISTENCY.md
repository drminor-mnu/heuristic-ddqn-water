# Table 5 ↔ Figure 10 측정 범위 정합 확인 (R4-12 재발 방지)

R4-12가 원고 원본에서 지적한 문제(Table 5의 테스트 시간이 Figure 10의
"DDQN 추론시간" 서술과 상충)와 **같은 종류의 불일치가 이번 재생성판에서도
발생했다** — Table 5의 Decision(c) 값과 이번에 재생성한 Figure 10의
값이 다르다. 임의로 하나를 고치지 않고 사실만 정리하고 선택지를
제시한다. 새 측정은 하지 않았다(기존 완료된 E10/E14 결과만 재조회).

## 0. 문제로 지적된 수치

| 모델 | Table 5 Decision(c), ms/decision | Figure 10(이번 재생성), ms/decision | 비율 |
|---|---|---|---|
| Regular | 0.1655 | 0.699 | 4.2배 |
| GA-guided | 0.2389 | 0.680 | 2.8배 |
| PSO-guided | (Table 5에 행 없음) | 0.680 | — |

두 표·그림 다 "결정 1회당 시간"을 말하고 있지만, **측정 범위와 사용된
모델 체크포인트가 둘 다 다르다.** 아래 §1-§2에서 각각을 분리해
확인한다.

## 1. 측정 범위 차이 (요청 1) — `total_time_s`는 무엇을 포함하는가

`src/run_e10_timing_detailed.py`(`E10_timing_detailed.csv` 생성)는
결정 1회를 4단계로 분해해 측정한다:

| 단계 | 정의 | 포함 여부 — Table 5 Decision(c) | 포함 여부 — 이번 Figure 10 |
|---|---|---|---|
| (a) | 신경망 forward pass만 | 포함 | 포함(간접) |
| (b) | (a) + 상태 전처리(정규화+5스텝 시퀀스 구성) | 포함 | 포함(간접) |
| (c) | (a)+(b) + 환경 step 오버헤드(`WaterGym.step()`, **SWMM 미호출**) | **포함(= Table 5 값 자체)** | 포함(간접) |
| (d) | SWMM 1회 실행(`WaterGym.reset()`, 에피소드당 1회, 결정마다 아님) | **제외** | **포함** |

**Table 5의 Decision(c)** = (a)+(b)+(c)만의 합, **SWMM 실행 시간을
명시적으로 제외**한 값이다(`results/summary/TABLE5_COMPARABILITY.md`
§10 각주: "SWMM 1회(에피소드당) 비용 (d)는 이 표에서 제외했으나
참고용으로 병기"). 결정 간격(2분=120,000ms)과 비교하기 위한
"추론+환경반영 자체의 순수 계산비용"을 보여주려는 의도로 읽힌다.

**이번 Figure 10**은 `results/summary/E10_timing_breakdown.csv`의
`total_time_s`를 썼다 — 이 열은 (a)+(b)+(c)+(d) **전부를 포함한
에피소드 전체 벽시계 시간**이다(코드 확인:
`src/run_e10_timing.py`가 `time.perf_counter()`로 `gym.reset()`부터
에피소드 종료까지를 통째로 잰다). **SWMM 실행 시간이 포함돼 있다.**

**포함 여부를 직접 검증**(같은 모델 Regular(E03 v2, 4기준)로 재계산):

| 범위 | ms/decision |
|---|---|
| (c) = a+b+c, SWMM 제외 | 0.278 |
| (c)+(d), SWMM 포함(에피소드당 1회를 결정수로 환산) | 0.835 |
| `E10_timing_breakdown.csv`의 `total_time_s`(별도 스크립트 재측정) | 0.699 |

(c)와 (c)+(d)의 차이(0.278 vs 0.835, 약 3배)가 **Figure 10과 Table 5의
비율(2.8~4.2배)과 같은 자릿수**다 — 즉 **불일치의 주된 원인은
SWMM 실행 시간의 포함 여부**임을 확인했다.

**GA-only/PSO-only는애초에 (c)만 따로 잴 수 없다** —
`run_genetic_algo()`가 `reset()`과 탐색 루프를 자체적으로 실행해
이 스크립트들이 그 경계에 계측점을 넣지 않았기 때문이다
(`swmm_time_s`/`inference_time_s` 열이 빈 값,
`E10_timing_breakdown.csv` 직접 확인). **현재 있는 데이터로는
GA-only/PSO-only의 SWMM-제외 순수 탐색시간을 분리할 수 없다** — 이
사실이 아래 §3의 선택지에 직접 영향을 준다.

## 2. 별개의 문제 — "Regular"가 서로 다른 모델이다 (측정 범위와 무관)

Table 5의 Regular 행(0.1655ms)은 **§3.1의 2기준 Regular DDQN**
(`results/E14_two_criteria/`, `src/run_e14_timing.py`로 측정)이다.
이번 Figure 10에서 쓴 Regular(0.699ms, 또는 재계산 0.835ms)는
**E03 v2의 4기준 Regular DDQN**(`results/E03_seeds/Regular/`)이다.
**같은 이름("Regular")이지만 서로 다른 학습 실행(가중치·기준 수가
다른 별개 체크포인트)이다.**

이것이 자연스러운 이유: Table 5(§3.1)는 "2기준 vs 4기준" 비교가
목적이라 2기준 Regular가 맞는 선택이고, Figure 10(§3.4)은 5모델(Regular/
GA-guided/PSO-guided/GA-only/PSO-only)을 **모두 같은 4기준 보상
아래** 비교하는 것이 목적이라(PSO-guided·GA-only·PSO-only는애초에
4기준 전용이라 2기준 버전 자체가 없음) 4기준 Regular를 쓰는 것이
Figure 10 자체의 내적 일관성에는 맞다.

**같은 (c)+(d) 범위로 두 Regular를 대조**하면 모델 차이만으로도
차이가 난다(측정범위를 똑같이 맞춰도 완전히 같아지지 않는다):

| Regular 버전 | (c)+(d), ms/decision |
|---|---|
| 2기준(E14, Table 5가 쓰는 모델) | 0.703 |
| 4기준(E03 v2, Figure 10 내적일관성상 필요한 모델) | 0.835 |

(참고로 GA-guided는 Table 5·Figure 10 둘 다 **같은 4기준 E03 v2
체크포인트**를 쓰므로 이 "모델 정체성" 문제가 없다 — GA-guided의
0.2389 vs 0.680(또는 재계산 0.793/0.680) 차이는 순수하게 §1의
측정범위 차이 때문이다.)

**측정 잡음도 있다**: 같은 모델(E03 v2 Regular)을 같은 개념
((c)+(d))으로 서로 다른 두 스크립트가 측정한 결과가 이미
0.835(`run_e10_timing_detailed.py` 기반 재계산) vs 0.699
(`run_e10_timing.py`의 `total_time_s`, 별도 실행) — **같은 모델·같은
범위인데도 별도 실행 간 약 16~19% 차이**가 난다. 마이크로벤치마크
특성상(수 ms 단위 벽시계 시간) 스레드 스케줄링·GC 등 시스템 잡음이
어느 쪽 값을 쓰든 남는다 — 이 캡션·각주로 완전히 없앨 수 있는
차이가 아니라는 점도 사실로 남긴다.

## 3. 선택지 (결정하지 않음 — 근거만 제시)

### 옵션 A — 통일하지 않고 각주로 범위를 명시한다

Table 5는 지금처럼 (c)(SWMM 제외)를 유지, Figure 10은 지금처럼
`total_time_s`(SWMM 포함, (c)+(d) 상당)를 유지하되, **두 표/그림
각주에 측정 범위를 명시적으로 대조 서술**한다. 예:
- Table 5 각주: "Decision time excludes the once-per-episode SWMM
  simulation cost, reported separately in Figure 10 as part of total
  execution time."
- Figure 10 캡션/각주: "Execution time includes the once-per-episode
  SWMM simulation cost; see Table 5 for SWMM-excluded per-decision
  inference cost."

**장점**: 새 측정·재구현 불필요, 가장 빠르고 안전. §1에서 확인한
"주된 원인=SWMM 포함 여부"를 독자에게 그대로 설명하면 R4-12가
지적했던 "설명 없는 불일치"라는 문제 자체는 해소된다(값 자체는
다르게 유지되지만 이유가 명시됨).
**단점**: 두 수치가 여전히 다르게 남아, 숫자만 보고 대조하는 독자는
헷갈릴 수 있다. Regular 모델 정체성 차이(§2)도 별도로 각주에 설명해야
완전해진다.

### 옵션 B — Figure 10의 DDQN 3종만 (c)로 바꾸고, GA-only/PSO-only는 total 유지

Figure 10의 Regular/GA-guided/PSO-guided 막대만 (c) 범위(SWMM 제외,
Table 5와 같은 정의)로 바꾸고, GA-only/PSO-only는 §1에서 확인했듯
(c) 단독 측정이 불가능하므로 `total_time_s`(SWMM 포함)를 그대로
쓴다.

**장점**: DDQN 3종에 한해 Table 5·Figure 10이 완전히 같은 정의를
공유하게 된다(단, Regular는 여전히 §2의 모델 정체성 문제가 남음 —
Table 5용 2기준 Regular를 쓸지, Figure 10 내적일관성을 위해 4기준
Regular를 쓸지는 별도 결정 필요).
**단점**: 한 그림 안에서 막대 5개가 서로 다른 측정 범위를 쓰게
된다 — DDQN 3종은 "SWMM 제외", GA-only/PSO-only는 "SWMM 포함"이라
같은 y축에서 직접 비교하면 오히려 오해를 부를 수 있다(GA-only/
PSO-only가 실제보다 상대적으로 더 불리해 보이지는 않지만, 애초에
"같은 조건 비교"라는 그림의 취지 자체가 옅어짐). 캡션에 이 비대칭을
반드시 명시해야 한다.

### 옵션 C — GA-only/PSO-only를 재계측해 (c) 상당값을 새로 만든다

`run_genetic_algo()` 호출부에 `gym.reset()`(SWMM)과 이후 탐색 루프
사이에 계측점을 새로 추가하는 **작은 신규 스크립트**를 작성해
GA-only/PSO-only도 SWMM 제외 값을 얻는다. 이러면 5모델 전부가 (c)
하나의 정의로 완전히 통일된다.

**장점**: 가장 깨끗한 해결 — Table 5·Figure 10이 완전히 같은 정의를
공유하고, Figure 10 내부 5개 막대도 전부 동일 조건이 된다.
**단점**: **새 계측 실행이 필요하다**(학습은 아니지만 재실행이며,
"학습 불필요" 원칙과는 별개로 "재측정 없음" 방침과는 배치된다 —
사용자 승인 필요). GA-only/PSO-only는 SWMM 없이도 탐색 자체가
수 ms~수십 ms 걸릴 수 있어(모집단·입자 평가 반복), (c) 상당값이 이미
DDQN보다 압도적으로 크다는 결론 자체는 바뀌지 않을 가능성이 높다
(§1에서 total 기준으로도 20ms대 vs 0.7ms대로 30배 차이).

### 옵션 D — Table 5에 두 번째 열(SWMM 포함 total)을 추가해 병기

Table 5의 Decision(c) 열은 그대로 두고, **Figure 10과 직접 대조 가능한
"Decision time, total (incl. SWMM)" 열을 하나 더 추가**한다(Regular는
2기준 모델 기준 0.703ms, GA-guided는 4기준 모델 기준 §1의 값 사용).
Figure 10 쪽은 그대로 둔다.

**장점**: 원하는 쪽 수치를 각 표·그림에서 바로 찾을 수 있어 교차
대조가 쉬워진다 — 옵션 A(각주 설명)보다 더 명시적이다.
**단점**: Table 5 열이 하나 늘어 §3.1 패키지의 표 형식(이미 7열로
확정, `docs/response/SECTION_3_1_PACKAGE.md`)을 다시 바꿔야 한다 —
이미 완료 처리된 §3.1 작업에 영향을 준다. Regular 모델 정체성 문제도
여전히 남는다(2기준 Regular의 total을 쓸지 병기할지 결정 필요).

## 4. 요약 표 (참고용, 원인별 정리)

| 원인 | 크기 | 해소 가능 여부 |
|---|---|---|
| 측정 범위(SWMM 포함 여부) | 약 3배(0.278 vs 0.835, 같은 모델) | 가능 — 옵션 A/B/C/D 어느 쪽으로도 해소 가능 |
| Regular 모델 정체성(2기준 vs 4기준) | 약 1.19배(0.703 vs 0.835, 같은 범위) | 완전 해소는 어려움 — Table 5·Figure 10 각각의 절 목적(§3.1 vs §3.4)이 서로 다른 모델을 요구함 |
| 측정 잡음(별도 스크립트 실행 간) | 약 1.16~1.19배(같은 모델·같은 범위인데도) | 해소 불가 — 반복 측정·평균으로 축소는 가능하나 이 세션 범위 밖 |

**세 원인이 곱해져서 Table 5 대 이번 Figure 10의 2.8~4.2배 차이가
난다** — 단일 원인이 아니라는 점이 가장 중요한 사실이다. 어느
옵션을 택하든 이 세 원인을 분리해서 설명해야 R4-12 같은 지적이
재발하지 않는다.

## 5. 출처

`results/summary/TABLE5_COMPARABILITY.md` §10-11(Table 5 Decision(c)
확정 경위), `results/summary/E10_unified_table.md`(세부 분해 표),
`results/summary/E10_timing_detailed.csv`(E03 v2 3모델, (a)-(d) 분해,
162행), `results/summary/E14_timing_detailed.csv`(E14 2기준 Regular,
동일 분해, 54행), `results/summary/E10_timing_breakdown.csv`(5모델
total_time_s, 이번 Figure 10 소스, 271행), `src/run_e14_timing.py`,
`src/run_e10_timing.py`, `src/run_e10_timing_detailed.py`(전부 기존
스크립트, 이번에 수정·재실행하지 않음 — 원시 CSV에서 직접
재계산만 수행).

---

## 6. 채택안 (2026-09-09, 선택지 C — 전면 통일)

**사용자 결정**: 선택지 C(GA-only/PSO-only 재계측으로 5모델 전부를
(c) 정의로 완전 통일). 신규 계측 실행함(`src/run_e10_unified_c.py`,
기존 평가 함수는 미수정 — DDQN은 기존 스크립트와 동일 로직을 새
파일에 복제, GA-only/PSO-only는 `WaterGym.reset`을 측정 함수
안에서만 일시적으로 monkey-patch해 SWMM 시간을 분리하고 `finally`
블록에서 즉시 원복).

### 6-1. 원인 1(측정 범위) 처리

**해소됨.** 5모델 전부를 (a)+(b)+(c)(SWMM 제외) 하나의 정의로
통일했다. GA-only/PSO-only도 이번에 SWMM 분리 계측을 새로 확보했다
(에피소드당 SWMM 시간: GA-only 3회 평균 152.9~153.7 ms/scenario,
PSO-only 153.0~153.7 ms/scenario — 기존 DDQN 쪽 SWMM 값(Regular
156.7ms, GA-guided 161.4ms/episode)과 자릿수가 같아 타당성 확인됨).

### 6-2. 원인 2(Regular 모델 정체성) 처리

**완전히 해소하지는 않되, 명시했다.** Table 5는 여전히 §3.1 목적상
2기준 E14 Regular를, Figure 10은 §3.4의 5모델 내적일관성상 4기준
E03 v2 Regular를 쓴다 — 이번 재측정으로도 **같은 (c) 정의인데도
두 Regular가 다른 값**을 낸다(2기준 0.1581±0.0016ms vs 4기준
0.1556±0.0031ms, §6-3 표 참고. 차이는 이제 매우 작다 — 약 1.6%,
종전 재측정 전 추정치였던 약 19%보다 훨씬 줄었다). Table 5 각주에
사용자 확정 문구("The Regular DDQN in this table uses the
two-criterion reward...")를 추가해 두 Regular가 다른 모델임을
명시하는 것으로 처리했다(모델을 통일하지는 않음 — 옵션 검토 결과
각 표·그림의 절 목적상 통일이 부적절하다는 §2 결론 유지).

### 6-3. 원인 3(측정 잡음) 처리

**3회 반복 측정으로 정량화했다(완전 제거는 원래 불가능).** 5모델
전부 표준편차를 보고한다 — DDQN 3종·2기준 Regular는 sd가 평균의
0.6~2%로 작지만, **GA-only는 sd 0.11ms(평균의 0.5%), PSO-only는
sd 0.0063ms(평균의 0.4%)로 모두 작다** — 이번 반복측정에서는
§1에서 우려했던 "별도 실행 간 16~19% 차이" 수준의 큰 잡음은 나타나지
않았다(같은 스크립트 안에서 반복했기 때문으로 추정 — 별도 스크립트
간 차이와 같은 스크립트 내 반복 간 차이는 다른 종류의 잡음일 수
있음, 확정하지 않음).

**부수 관찰(사실 기록)**: 이번 DDQN (c) 재측정값(0.155~0.158ms)이
기존 `E10_timing_detailed.csv`의 (c)값(Regular 0.278ms, GA-guided
0.239ms)보다 상당히 낮다. 코드 비교 결과, 기존 스크립트는 (a)/(b)/(c)
세 구간을 **별도의 `perf_counter()` 호출 3쌍**으로 나눠 측정했고,
이번 신규 스크립트는 한 결정 전체를 **연속된 한 블록**으로 측정했다
— 별도 타이머 호출 자체의 오버헤드가 기존값을 부풀렸을 가능성이
있다(확정 원인 아님, 사실만 기록). 즉 (c)라는 개념은 같지만 계측
방식(타이머 분할 여부)에 따라서도 절대값이 달라질 수 있다는 것을
이번에 추가로 확인했다.

### 6-4. 부수 발견 — PSO-only 측정 버그 (중요)

`src/run_e10_timing.py`의 `time_heuristic()`이 GA-only·PSO-only
양쪽 모두에 대해 `g.run_genetic_algo(test_inp)`를 호출한다.
`ParticleSwarmOptimization`은 `run_genetic_algo`도
`genetic_algorithm`도 오버라이드하지 않는다(`run_pso`/
`particle_swarm_optimization`이라는 별도 이름의 메서드만 있음,
`src/pso.py` 확인) — 따라서 **"PSO-only" 라벨이 붙은 기존 측정은
실제로는 상속받은 GA 알고리즘을 실행한 것**이다. 이것이 기존
PSO-only(21.792ms)와 GA-only(21.398ms)가 거의 같았던 이유로
설명된다. 이번에 올바른 메서드를 명시적으로 디스패치하도록
수정한 신규 스크립트로 재측정한 결과, **PSO-only는 GA-only보다
약 14배 빠르다**(1.41ms vs 19.93ms) — PSO가 매 스텝 위치 갱신만
하는 반면 GA는 매 스텝 population 전체를 재평가하기 때문으로
보인다(확정 원인 아님, 사실만 기록).

**영향 범위**: `run_e10_timing.py`는 **타이밍 측정 전용 스크립트**다.
Table 11-16 등 원고의 실질적인 PSO-only 성능 지표(max_level,
n_switches 등)는 `perform_evaluate.evaluate_pso()`(`run_method=
'run_pso'`로 올바르게 디스패치, `_evaluate_heuristic()` 경유)를
쓰는 **별도 경로**이므로 **이 버그의 영향을 받지 않는다** — 직접
확인 완료. `run_e10_timing.py` 자체는 리팩터링 금지 원칙에 따라
수정하지 않았다(사실 기록만).

### 6-5. 최종 채택 값

| 모델 | (c), ms/decision, mean±sd (3회 반복) |
|---|---|
| Regular(4기준, E03 v2 — Figure 10용) | 0.1556 ± 0.0031 |
| GA-guided(4기준) | 0.1546 ± 0.0009 |
| PSO-guided(4기준) | 0.1558 ± 0.0017 |
| Regular(2기준, E14 — Table 5용) | 0.1581 ± 0.0016 |
| GA-only | 19.9324 ± 0.1072 |
| PSO-only(버그 수정 후) | 1.4106 ± 0.0063 |

**적용처**: Table 5 Decision(c) 열 → `docs/response/SECTION_3_1_PACKAGE.md`
§2(2026-09-09 갱신, Regular 0.1581±0.0016 / GA-guided
0.1546±0.0009), 모델정체성 각주 추가. Figure 10 → 5모델 전부
`results/summary/figures_final/figure10.png`(오차막대 표시, 3회
반복 sd), `results/summary/figure_data/fig10.csv`.

**오차막대 표시 여부(요청 판단)**: **표시했다** — 3회 반복 측정을
바로 이 잡음(§1·§6-3)을 정량화하려고 도입했으므로, 그 결과인
표준편차를 오차막대로 시각화하는 것이 반복측정을 요청한 취지에
직접 부합한다고 판단했다.

**R4-12 대응 상태**: 세 원인 중 (1) 측정범위는 완전히 통일, (3) 측정
잡음은 반복측정으로 정량화·보고, (2) 모델 정체성은 각주로 명시 —
"설명 없는 불일치"라는 R4-12의 핵심 지적은 이제 해소됐다. 남은
차이(2기준 vs 4기준 Regular, 약 1.6%)는 설명된 차이이지 불일치가
아니다.

