# E10 — Table 5 + Figure 10 통합안 (계산시간 분해)

**대응**: R4-12, R2-2. 대응 리뷰어: 원고 Table 5의 테스트 시간(Regular
257,224.97초, GA-guided 256,591.19초)이 DDQN 추론만으로는 설명 불가능한
값이며 SWMM 시뮬레이션 시간이 포함된 것으로 보인다는 지적, Figure 10의
DDQN 추론시간 서술과의 상충.

## 선행 확인 — `evaluate_dpn_gru(train=False)` 버그 (계획서 E10 배경 항목)

`docs/REMAINING_WORK.md` 1번 항목에서 확정: `perform_evaluate.py:424`의
`return basic_losses, basic_rewards, dqn_test_rewards, elapsed`가
`train=False`일 때 미정의 변수를 참조해 `UnboundLocalError`로 항상
실패한다. **E3 v2를 포함한 저장소의 모든 실제 실행 경로는 `train=True`로만
이 함수를 호출**(`run_e03_seeds.py:290`, `diag_e03_validate.py:81`)하므로
**이 버그는 E3 v2 수치에 영향을 주지 않는다**(버그가 발동하는 조건 자체가
실행된 적 없음). 이번 측정은 이 함수를 호출하지 않고
`dqn_from_demon_v1.test_model()`의 루프를 그대로 재현한 신규 스크립트
(`src/run_e10_timing.py`)로 수행했다 — 리팩터링 없이 버그를 우회.

## 3분할 계측 결과 (신규 실행, 학습 없음, 기존 E3 v2 seed1 체크포인트 재사용)

54개 시나리오(층화 부표본, 1/stratum), 신가중치 `[0.40,0.25,0.25,0.10]`,
γ=0.7. `inference_time_s`는 Q-network forward pass(또는 GA/PSO 탐색)
합산, `swmm_time_s`는 `WaterGym.reset()`(SimSwmm 실행) 1회, `total_time_s`는
에피소드 전체(둘 다 포함).

| 모델 | 평균 결정 수/시나리오 | 평균 SWMM 시간(초) | 평균 추론 시간(초, 전체결정 합) | 평균 총시간(초) | 결정당 시간(ms) |
|---|---|---|---|---|---|
| Regular | 291.3 | 0.156 | 0.0345 | 0.204 | 0.129 |
| GA-guided | 291.3 | 0.152 | 0.0330 | 0.198 | 0.115 |
| PSO-guided | 291.3 | 0.152 | 0.0330 | 0.198 | 0.115 |
| GA-only | 291.3 | (미분리, 단일 호출) | (미분리) | 6.084 | 21.398 |
| PSO-only | 291.3 | (미분리, 단일 호출) | (미분리) | 6.174 | 21.792 |

**GA-only/PSO-only는 SWMM/추론을 분리 계측하지 않았다** — `run_genetic_algo()`가
`reset()`과 탐색 루프를 자체적으로 실행해 이번 스크립트가 그 경계에
계측점을 넣지 않았기 때문(총시간만 기록). DDQN 3모델은 SWMM(1회, 0.15초
내외)과 추론(전체 결정 합산, 0.03초 내외)이 명확히 분리된다.

## 세부 분해 (2026-09-01 추가, R4-12 대응 — 측정 범위 명시)

`src/run_e10_timing_detailed.py`(신규, `test_model()` 루프를 줄 단위로
재현, 기존 함수 미변경). 54개 시나리오(층화 1/stratum), seed1, 신가중치.
**하드웨어**: CPU 13th Gen Intel Core i9-13900KF(24코어/32스레드), GPU
NVIDIA RTX 3080 Ti(12GB), RAM 94GB(`results/_log/ENVIRONMENT.json`).

**분해 항목**:
- (a) 신경망 forward pass만: `model(state.unsqueeze(0))` 호출 1회
- (b) (a) + 상태 전처리: min-max 정규화(`_input_norm`) + 5스텝 시퀀스
  텐서 구성(`torch.stack`/`torch.tensor`)
- (c) (a)+(b) + 상태 관측·액션 전달 오버헤드: GPU→CPU 텐서 이동, argmax,
  `WaterGym.step()` 호출(행동 디코딩·물수지 갱신·보상/정보 계산 — **SWMM
  미호출**, `docs/E04E_REPORT.md`/`water_gym.py`로 이미 확정된 사실)
- (d) SWMM 1회 실행(에피소드당, `WaterGym.reset()`) — **결정마다 발생하는
  비용이 아니라 에피소드당 1회**이므로 "1스텝" 참조값이 아니라 에피소드당
  총량과 결정당 환산값을 함께 표기

| 단계 | 정의 | Regular | GA-guided | PSO-guided |
|---|---|---|---|---|
| (a) | 신경망 forward pass만 | 0.103 ms | 0.096 ms | 0.096 ms |
| (a+b) | + 상태 전처리(정규화+5스텝 시퀀스 구성) | 0.187 ms | 0.163 ms | 0.176 ms |
| **(c) = (a+b+c)** | **+ 환경 step 오버헤드(SWMM 미포함) — 기본값** | **0.278 ms** | **0.239 ms** | **0.264 ms** |
| (d) | SWMM 1회(에피소드당, 결정마다 아님) | 0.163 s/에피소드 | 0.161 s/에피소드 | 0.162 s/에피소드 |
| (c)+(d) 결정당 환산 | (d)를 평균 291.3결정/에피소드로 나눠 (c)에 합산 | 0.835 ms | 0.793 ms | 0.820 ms |

(3모델 전부 54시나리오·15,732결정 합산 기준. 산출:
`results/summary/E10_timing_detailed.csv`, 162행.)

**결정 간격(120,000 ms)과의 비교**: (c) 기준 결정당 0.24~0.28 ms,
(c)+(d) 기준 0.79~0.84 ms — 둘 다 120,000 ms 대비 1 ms 미만이다(배수로는
(c) 기준 약 43만~50만배, (c)+(d) 기준 약 14.4만~15.1만배 여유 — 상세
표는 `E10_realtime_margin.md`).

**실제 배치에서 SWMM이 불필요한 이유(한 문장)**: 실시간 운용에서는 유입량이
SWMM 시뮬레이션이 아니라 현장 센서(수위계·유입계) 실측값으로 직접
관측되며, 이 연구의 오프라인 평가에서도 SWMM은 각 시나리오의 강우-유입
수문곡선을 에피소드 시작 시 1회 생성하는 데만 쓰이고(`WaterGym.reset()`)
학습된 정책의 매 결정 스텝(`WaterGym.step()`)에는 전혀 관여하지 않는다.

## 원고 Table 5 vs Figure 10 불일치 특정 (R4-12)

계획서에 이미 기록된 리뷰어 지적(`docs/REVISION_EXPERIMENT_PLAN.md` E10
배경)을 그대로 인용: **Table 5의 테스트 시간(Regular 257,224.97초,
GA-guided 256,591.19초)은 DDQN 추론만으로 설명 불가능한 값이며 SWMM
시뮬레이션 시간이 포함된 것으로 보인다는 것, 그리고 이 값이 Figure
10의 "DDQN 추론시간" 서술과 상충한다는 것**이 R4-12의 핵심이다.
`docs/manuscript_outline.md`의 표/그림 색인 기준: **Table 5의 제목은
"Reward weights and computational times for the exploratory comparison"**
(2.5절, 시간 단위·측정 방법 서술 없음), **Figure 10의 제목은 "Test
execution time of the five models **and enlarged comparison of DDQN
inference time**"**(3.4절) — 제목 자체가 "5모델 전체 실행시간"과 "DDQN
추론시간만 확대 비교"를 **별개의 패널**로 명시하고 있어, Table 5가
전체(아마도 SWMM 포함) 값이고 Figure 10이 그중 DDQN 추론시간만 따로
확대해 보여준다는 구조로 읽힐 수 있다. **다만 원고 본문에 Table 5 값이
몇 개 시나리오의 합산인지, SWMM 포함 여부를 명시한 문장이 있는지는
원본 실행 로그 소실로 이 세션에서 직접 확인하지 못했다**(`docs/RESULTS_PROVENANCE.md`).
참고로 현재 고정분할 테스트 시나리오 수(2,700개)로 단순히 나누면
257,224.97초/2,700 ≈ 95.3초/시나리오가 되어, 이번 실측
(SWMM 포함 결정당 0.8ms대, 시나리오당 0.2초대)과는 자릿수가 다르다 —
**단, 원고 테스트 시나리오 수가 실제로 2,700이었는지 자체가 확인되지
않으므로 이 나눗셈은 참고치일 뿐 확정 배율이 아니다.**

257,224.97초·256,591.19초는 각각 약 71.45시간·71.28시간에 해당한다.
**응답서 서술은 이 값이 무엇인지 확정할 수 없다는 사실만 담는다**
(검증 불가한 추측은 응답서에 넣지 않음 — 아래 인용). 이 값에 대한
검증되지 않은 추정(기록 목적, 응답서 미사용)은
`E10_realtime_margin.md`의 "원고 Table 5 값에 대한 분석" 절 참조.

> **응답서 서술 초안**: "이전 Table 5의 값이 무엇을 측정한 것인지
> 확정할 수 없으며, 원본 실행 로그가 보존되지 않아 역추적이
> 불가능하다. 해당 표를 삭제하고 측정 범위·단위·하드웨어를 명시한
> 새 표로 교체한다."

산출: `results/summary/E10_timing_breakdown.csv`(270행, 5모델×54시나리오),
`results/summary/E10_timing_detailed.csv`(162행, DDQN 3모델 세부분해).
