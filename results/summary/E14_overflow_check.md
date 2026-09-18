# E14 월류(Overflow) 확인

> **Cliff's δ 부호규약**: 본 문서에는 δ 계산이 없다(해당 없음).

## 0. Table 6~10에 월류 지표 포함 여부

`src/run_e14_table6to10.py`의 `METRICS = ['max_level_m', 'n_switches',
'n_intervals_100', 'n_intervals_170', 'n_dryrun_proxy']` — **월류 지표는
Table 6~10(재생성판)에 포함되어 있지 않다.** 이번 문서에서 별도 산출한다.

## 1. 정의

`src/perform_evaluate.py`의 `overflow_flag = int(highest >=
water_gym.Level[-1])` (해당 코드가 5곳에 반복 등장: 378/539/716/810/963행,
모두 동일 정의). `highest`는 시나리오 진행 중 저수지 최고 수위
(`max_level_m`과 동일 개념), `water_gym.Level[-1]`은 10.0 m(원고 본문의
"10 m overflow level"). `test_metrics.csv`에 이미 `overflow_flag` 열로
기록되어 있어 재계산 없이 그대로 집계했다.

## 2. 전체 집계

데이터: E14 Regular(2기준, w=[0.5,0.5,0,0]) 5시드×2,700시나리오=13,500행,
E3 v2 GA-guided(4기준, 신가중치[0.40,0.25,0.25,0.10]) 5시드×2,700시나리오
=13,500행. 출처: `results/E14_two_criteria/Regular/seed{1..5}/test_metrics.csv`,
`results/E03_seeds/GA-guided/seed{1..5}/test_metrics.csv`.

| 모델 | n_overflow | n_total | 비율 |
|---|---|---|---|
| Regular(2-criterion) | 0 | 13,500 | 0.0000% |
| GA-guided(4-criterion) | 0 | 13,500 | 0.0000% |

시드별 분해(둘 다 전 시드 0):

| 모델 | seed1 | seed2 | seed3 | seed4 | seed5 |
|---|---|---|---|---|---|
| Regular(2-criterion) | 0 | 0 | 0 | 0 | 0 |
| GA-guided(4-criterion) | 0 | 0 | 0 | 0 | 0 |

## 3. max_level_m 분포 (참고, 10.0 m 여유폭 확인용)

| 모델 | min | max | mean |
|---|---|---|---|
| Regular(2-criterion) | 4.7000 | 6.5662 | 5.0225 |
| GA-guided(4-criterion) | 5.4723 | 6.8596 | 6.0328 |

전체 27,000행(두 모델 합산) 중 `max_level_m` 최댓값은
**6.859639296601662**(duration=120min, return_period=100 셀)로, 10.0 m
임계값까지 약 3.14 m 여유가 있다.

## 4. 지속시간·재현기간별 분포

모든 (duration_int, return_period_int) 54개 셀에서 두 모델 모두
`n_overflow=0`이었다(별도 표로 나열할 값이 없음 — 전 셀 0).

## 5. 사실 기록

- E14 Regular(2기준, w3=w4=0)와 E3 v2 GA-guided(4기준) 모두 월류 건수는
  **0건**이다.
- 2기준 설정(w3=w4=0, dry-running/펌프사용 가중치 없음)이 dry-running
  proxy·펌프 스위칭 등 다른 지표에서는 GA-guided 대비 큰 차이를 보이지만
  (`results/summary/E14_table_n_dryrun_proxy.md` 등 참조), 월류 자체는
  발생하지 않았다.
- 해석·판단은 하지 않는다.
