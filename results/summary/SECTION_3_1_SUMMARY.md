# §3.1 재생성판 요약표 — 2기준 Regular DDQN vs 4기준 GA-guided DDQN

해석·옹호 없이 사실·숫자만 기록한다.

> **Cliff's δ 부호규약**: 본 문서에는 δ 계산이 없다(해당 없음).

## 0. 조건

- **Regular(2-criterion)**: w=[0.5, 0.5, 0, 0], E14(신규 재학습),
  `results/E14_two_criteria/Regular/seed{1..5}/test_metrics.csv`
- **GA-guided(4-criterion)**: w=[0.40, 0.25, 0.25, 0.10](신가중치), E3 v2,
  `results/E03_seeds/GA-guided/seed{1..5}/test_metrics.csv`
- 공통: n=5시드×2,700시나리오=13,500행/모델, γ=0.7, split_seed=42
  (train_n=3996, test_n=2700 고정분할)
- 집계방법: `src/run_e14_table6to10.py`의 2단계 방식(시나리오→시드평균,
  9지속시간×6재현기간=54셀 각각의 시드평균, 그 54셀 평균을 전체평균으로
  재평균) — `analyze_e03.py`의 CI방법론과 동일한 2단계 구조.

## 1. 5지표 전체 평균 (54셀 평균)

| 지표 | Regular(2기준) | GA-guided(4기준) | 배수 |
|---|---|---|---|
| max_level_m | 5.0225 | 6.0328 | GA/Reg = 1.2012× (GA guided가 높음) |
| n_switches | 18.7021 | 34.1954 | GA/Reg = 1.8284× (GA-guided가 높음) |
| n_intervals_100 | 817.5785 | 312.2778 | Reg/GA = 2.6181× (Regular가 높음) |
| n_intervals_170 | 109.0788 | 18.7628 | Reg/GA = 5.8136× (Regular가 높음) |
| n_dryrun_proxy | 278.8391 | 4.8725 | Reg/GA = 57.2271× (Regular가 높음) |

(출처: `results/summary/E14_table_{max_level_m,n_switches,n_intervals_100,
n_intervals_170,n_dryrun_proxy}.csv`, `_mean` 열의 54행 단순평균.)

## 2. 월류(overflow) 건수

`results/summary/E14_overflow_check.md` 결과:

| 모델 | n_overflow | n_total |
|---|---|---|
| Regular(2기준) | 0 | 13,500 |
| GA-guided(4기준) | 0 | 13,500 |

두 모델 모두 월류 0건. 지속시간·재현기간별 54개 셀 전부 0건(분포 표
없음, 전 셀 동일).

## 3. 원고 이전 판(원본 Table 6~10) 수치와의 대조

원고 원본(docx, `docs/response/MANUSCRIPT_TEXT.md`로 변환된 원문)의
Table 6~10 값을 동일한 방식(9지속시간×6재현기간=54값 단순평균, 모델별)
으로 집계한 결과:

| 지표 | 원고 Regular(원본, 구가중치) | 원고 GA-guided(원본, 구가중치[0.25,0.40,0.25,0.10]) | 재생성 Regular(신규, w=[0.5,0.5,0,0]) | 재생성 GA-guided(신가중치) |
|---|---|---|---|---|
| max_level_m | 4.8428 | 5.4734 | 5.0225 | 6.0328 |
| n_switches | 7.8511 | 89.3926 | 18.7021 | 34.1954 |
| n_intervals_100 | 859.3448 | 312.6648 | 817.5785 | 312.2778 |
| n_intervals_170 | 577.8452 | 19.1296 | 109.0788 | 18.7628 |
| n_dryrun_proxy | 5.7641 | 0.0204 | 278.8391 | 4.8725 |

**주의**:
- 원고 Table 6~10의 "Regular" 열은 w=[0.5,0.5,0,0](2기준)으로 재생성판과
  가중치는 같으나, 코드 경로가 다르다(재현 불가로 재학습, 원본
  체크포인트 소실 — `docs/response/SECTION_3_1_REGENERATION.md` 참조).
- 원고 Table 6~10의 "GA-guided" 열은 **구가중치**
  [0.25,0.40,0.25,0.10]이며, 재생성판 GA-guided는 **신가중치**
  [0.40,0.25,0.25,0.10]다 — 두 열은 서로 다른 가중치의 모델이므로
  직접 대조는 참고용일 뿐이다(이미 `docs/response/
  SECTION_3_1_REGENERATION.md`에 명시된 사실, 재확인만).
- 원고 Table 8(n_intervals_100)은 3개 행이 완전히 동일한 알려진 오류를
  포함한 상태로 평균에 반영되어 있다(정정하지 않음, `results/summary/
  R4-11_table8_check.md` 참조) — 원고 Regular 열의 859.3448이라는
  값에는 이 오류가 그대로 들어가 있다.
- n_dryrun_proxy의 원고 값(Regular 5.7641, GA-guided 0.0204)과
  재생성판 값(Regular 278.8391, GA-guided 4.8725)의 격차가 매우 크다
  — 원인은 조사하지 않았다(코드 재현성/버전 차이 여부 미확인, 사실만
  병기).

## 4. 요약 (판단 없음)

- 재생성판에서 max_level_m·n_switches는 GA-guided가 Regular보다 높고,
  n_intervals_100·n_intervals_170·n_dryrun_proxy는 Regular가
  GA-guided보다 높다 — 방향은 원고 원본과 대체로 일치하나(단, n_switches
  원고에서도 GA-guided가 Regular보다 높음, 배수만 다름), 절대값과
  배수는 원고와 재생성판 사이에 상당한 차이가 있다.
- 월류는 두 모델 모두 0건으로 동일.
- 신구 가중치가 다른 GA-guided 열끼리는 직접 비교 대상이 아님을
  재확인.

## 5. §3.1 서술 초안용 확정 수치

### 5.1 월류 임계 대비 여유

- 월류 임계: **10.0 m** (`water_gym.Level[-1]`).
- 두 모델·54셀·13,500행씩(§2) 중 관측된 max_level_m 최댓값: **6.8596 m**
  (GA-guided, duration=120min·return_period=100년 셀 — `E14_overflow_check.md`
  §3).
- 임계 대비 여유: 10.0 − 6.8596 = **3.14 m**.

### 5.2 두 설정의 최고수위 차이가 이 여유 안에 있음

- Regular(2기준) 전체평균 max_level_m: **5.0225**
- GA-guided(4기준) 전체평균 max_level_m: **6.0328**
- 차이: 6.0328 − 5.0225 = **1.0103 m ≈ 1.01 m**
- 이 차이(1.01 m)는 §5.1의 여유(3.14 m)보다 작다 — 즉 두 설정 중 더
  높은 쪽(GA-guided)의 평균 수위와 임계(10.0 m) 사이의 격차가, 두
  설정 간 평균수위 차이 자체보다 약 3.1배 크다. 판단·해석은 하지
  않는다.

### 5.3 5지표 배수 (§1 표 재인용, 한 표로)

| 지표 | Regular(2기준) | GA-guided(4기준) | 배수 |
|---|---|---|---|
| max_level_m | 5.0225 | 6.0328 | GA/Reg = **1.20×** |
| n_switches | 18.7021 | 34.1954 | GA/Reg = **1.83×** |
| n_intervals_100 | 817.5785 | 312.2778 | Reg/GA = **2.62×** |
| n_intervals_170 | 109.0788 | 18.7628 | Reg/GA = **5.81×** |
| n_dryrun_proxy | 278.8391 | 4.8725 | Reg/GA = **57.23×** |
| overflow (월류) | 0/13,500 | 0/13,500 | 동일(둘 다 0) |
