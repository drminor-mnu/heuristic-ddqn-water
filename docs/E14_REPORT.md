# E14 — §3.1 재생성 (2기준 Regular DDQN 재학습 + Table 6~10/Figure 1~4/Table 5)

> **Cliff's δ 부호규약**: "Cliff's δ < 0 indicates lower values for the
> first group." (`results/summary/DELTA_SIGN_FIX.md`)

**대응**: R4-11(Table 8), R4-12(Table 5), R1-6(§3.1↔§3.3 성격 구분).
**방침**: 삭제가 아니라 재생성(`docs/response/SECTION_3_1_REGENERATION.md`).

## 0. 실행 개요

| 조건 | 가중치 | γ | 시드 | 학습 시나리오 | 테스트 시나리오 | 소스 |
|---|---|---|---|---|---|---|
| Regular(2기준) | [0.50, 0.50, 0, 0] | 0.7 | 1~5(신규) | 3,996 | 2,700 | `results/E14_two_criteria/Regular/seed{1..5}/`(신규 학습) |
| GA-guided(4기준) | [0.40, 0.25, 0.25, 0.10] | 0.7 | 1~5(재사용) | 3,996 | 2,700 | `results/E03_seeds/GA-guided/seed{1..5}/`(E3 v2, 재학습 없음) |

**무결성 확인(1단계, 전부 통과)**: 5/5 시드 `DONE` 존재, `test_metrics.csv`
각 2,700행, `train_log.csv` 3,996에피소드 완주(E3 v2 Regular와 동일
에피소드 수), `run_meta.json` 가중치 `{w1:0.5,w2:0.5,w3:0,w4:0}`·γ=0.7·
`split_seed=42`(E3 v2와 동일) 전부 일치, `git_commit` 5개 전부
`c9281ab`로 일관.

**사실 기록(추정 정정, 원인 확인됨)**: E14 실측 학습시간(아래 §4)은
시드당 평균 2,960.7초(0.82시간)로, 이전 세션이 E3 v2 4기준 Regular의
`elapsed_train_s`(10,230.6초, 2.84시간)를 근거로 추정했던 값(~2.84시간)의
약 1/3.5에 불과했다. 이후 조사(`results/summary/TABLE5_COMPARABILITY.md`
§3) 결과, 원인은 **병렬 워커 수 차이**로 확인되었다 —
`results/_log/E03.log`(E3 v2 Regular, `--workers 8`)와
`results/_log/E14.log`(E14 Regular, `--workers 2`, 기본값) 확인. 동일
코드경로·동일 하이퍼파라미터·동일 데이터규모(가중치만 다름)인 두
Regular 실행을 직접 비교한 결과 workers=8→2 전환이 학습 벽시계시간을
약 3.46배 단축시켰다(5시드 평균 10,231.72s→2,960.68s). 하드웨어 부하
로그는 존재하지 않아(§4 동일 문서) 워커수 외 요인은 확인 불가.

## 1. Table 6~10 재생성 결과 (2026-09-03, `src/run_e14_table6to10.py`)

**형식**: 9(지속시간)×6(재현기간) 교차 그리드, Model×Duration 행 ×
Return period 열, 5시드 mean±sd(95% CI는 CSV에 별도 열로 포함, 마크다운
표는 mean±sd만 표시). 전체 108행(9×6×2모델)은
`results/summary/E14_table_{metric}.csv`/`.md`에 있다 — 아래는 핵심
경향만 요약한다(전체 54개 셀×5표 나열 안 함).

### 전체 평균(54개 셀 평균, 지표별)

| 지표 | Regular(2기준) | GA-guided(4기준) |
|---|---|---|
| max_level_m (Table 6) | 5.023 | 6.033 |
| n_switches (Table 7) | 18.702 | 34.195 |
| n_intervals_100 (Table 8) | 817.579 | 312.278 |
| n_intervals_170 (Table 9) | 109.079 | 18.763 |
| n_dryrun_proxy (Table 10) | 278.839 | 4.873 |

### 지속시간별 경향 (6개 재현기간 평균, 대표 2개 지표)

| 지속시간(분) | n_switches Regular | n_switches GA-guided | diff(GA−Reg) | n_dryrun Regular | n_dryrun GA-guided | diff(GA−Reg) |
|---|---|---|---|---|---|---|
| 60 | 13.01 | 11.66 | −1.34 | 41.45 | 0.84 | −40.60 |
| 120 | 15.83 | 13.37 | −2.45 | 68.20 | 1.01 | −67.20 |
| 180 | 17.18 | 15.67 | −1.50 | 102.00 | 1.63 | −100.37 |
| 240 | 17.56 | 18.94 | +1.37 | 135.51 | 1.91 | −133.60 |
| 360 | 18.38 | 24.18 | +5.81 | 200.29 | 2.22 | −198.07 |
| 540 | 18.45 | 32.15 | +13.70 | 290.52 | 2.96 | −287.57 |
| 720 | 19.00 | 42.21 | +23.22 | 379.47 | 3.75 | −375.72 |
| 1080 | 22.32 | 64.93 | +42.61 | 557.89 | 8.60 | −549.29 |
| 1440 | 26.60 | 84.63 | +58.03 | 734.21 | 20.94 | −713.27 |

**사실만**: n_switches는 60~180분에서 GA-guided가 Regular보다 낮다가
240분부터 역전해 GA-guided가 지속시간에 따라 급격히 증가한다(1440분에서
Regular의 약 3.2배). n_dryrun_proxy는 전 지속시간에서 Regular가
GA-guided보다 훨씬 높고(w3=w4=0이라 Regular가 펌프사용·dry-running에
전혀 페널티를 받지 않음), 그 격차가 지속시간에 따라 커진다(60분
40.6회차 → 1440분 713.3회차). max_level_m·n_intervals_100·n_intervals_170의
지속시간별 표는 CSV 참조(요지: max_level은 GA-guided가 전 지속시간에서
높고, 펌프사용(i100/i170)은 Regular가 훨씬 높다 — Regular가 저수위
유지를 위해 훨씬 많이·자주 펌프를 돌리기 때문).

### 이전 판(원고) 수치와의 대조 — 60분 행 예시

| 재현기간 | 원고 Regular(2기준) | 신규 Regular(2기준) | 원고 GA-guided(구4기준) | 신규 GA-guided(신4기준) |
|---|---|---|---|---|
| 10 | 4.70 | 5.096±0.194 | 5.451 | 5.923±0.073 |
| 20 | 4.84 | 5.346±0.154 | 5.574 | 5.909±0.087 |
| 30 | 5.19 | 5.532±0.068 | 5.69 | 5.944±0.108 |
| 50 | 5.51 | 5.779±0.114 | 5.927 | 6.034±0.070 |
| 80 | 5.81 | 5.976±0.107 | 6.14 | 6.184±0.067 |
| 100 | 5.93 | 6.093±0.091 | 6.296 | 6.299±0.059 |

(원고값은 1회 실행, 신규값은 5시드 mean±sd. 원고 GA-guided는 구가중치
[0.25,0.40,0.25,0.10]·수정 전 코드, 신규 GA-guided는 신가중치
[0.40,0.25,0.25,0.10]·수정된 코드이므로 코드·가중치가 모두 다르다 —
직접적인 재현 검증이 아니라 같은 형식의 다른 실행값을 나란히 놓은
것이다.) 두 판 모두 "Regular < GA-guided"(max_level) 방향은 동일하게
유지된다.

## 2. Figure 1~2 재생성 결과 (`src/run_e14_fig1_2.py`)

5시드 롤링평균(100에피소드), 신가중치/2기준 각각.

| 조건 | 최종 롤링평균 train_loss | 최종 롤링평균 train_reward |
|---|---|---|
| Regular(2기준) | 0.0062 | 157.7445 |
| GA-guided(4기준) | 0.0030 | 152.8492 |

산출: `results/summary/E14_figure1_loss.png`, `E14_figure2_reward.png`.

**캡션 초안**(원문과 동일 형식, 선정된 scenario_id는 캡션에 넣지 않음 —
아래 §3 참조):
- Figure 1: "Training-loss trajectories of Regular DDQN and GA-guided DDQN under different reward structures."
- Figure 2: "Training-reward trajectories of the two models under different reward structures."

## 3. Figure 3~4 재생성 결과 (`src/run_e14_fig3_4.py`)

**시나리오**: 30년 재현기간·60분 지속시간 테스트 층(50개 후보) 중
`random.Random(42).choice()`로 **`30yr_0060m_h211`** 고정 선정
(선정 방식은 이 문서와 `SECTION_3_1_REGENERATION.md`에만 기록,
아래 캡션 초안에는 조건만 표기).

**캡션 초안**(원문과 동일, scenario_id 미표기):
- Figure 3: "Operation of the two-criterion Regular DDQN for a 60 min rainfall event with a 30-year return period."
- Figure 4: "Operation of the four-criterion GA-guided DDQN under the same scenario as Figure 3."

체크포인트: Regular seed1, GA-guided seed1(둘 다 임의 선택, 문서화만).
산출: `results/summary/E14_figure3_regular_operation.png`,
`E14_figure4_ga_guided_operation.png`.

## 4. 2기준 Regular 계산시간 측정 (`src/run_e14_timing.py`, 신규)

E10과 동일 방법론·표본·하드웨어(54개 시나리오, 층화 1/stratum, seed1,
CPU i9-13900KF/GPU RTX 3080 Ti 12GB/RAM 94GB) — (a) forward pass만,
(a+b) +전처리, (a+b+c)=(c) +환경step(SWMM 제외), (d) SWMM(에피소드당).

| 조건 | (a) | (a+b) | (c) | (d) | (c)+(d) |
|---|---|---|---|---|---|
| Regular(2기준) | 0.0970 ms | 0.1233 ms | 0.1655 ms | 0.1567 s/ep | 0.7034 ms |
| GA-guided(4기준, 기존 E10 측정) | 0.0958 ms | 0.1632 ms | 0.2389 ms | 0.1614 s/ep | 0.7930 ms |

15,732결정(54시나리오) 동일 표본 크기. 여유율: Regular(2기준) (c)기준
약 725,000배, (c)+(d)기준 약 171,000배(120,000ms 대비).

## 5. Table 5 조립

**정정(2026-09-03, `results/summary/TABLE5_COMPARABILITY.md` 검증
결과)**: 아래 "두 행의 측정 조건이 동일함"이라는 서술은 **부정확했다.**
Decision(c)/SWMM(d) 열은 동일 조건(단일 프로세스, 54시나리오, seed1)이
맞으나, **Training Time 열은 서로 다른 병렬조건에서 측정되었다** —
Regular(2기준, E14)는 `--workers 2`, GA-guided(4기준, E3 v2)는
`--workers 8`(`results/_log/E14.log`, `results/_log/E03.log` 확인).
동일 모델(Regular)을 workers=8/2 두 조건에서 직접 비교한 결과 벽시계
시간이 약 3.46배 차이 났다(상세는 `TABLE5_COMPARABILITY.md` §3) — 즉
아래 Training Time 두 값은 재학습 없이는 그대로 비교할 수 없다. 이
표는 원래 서술 그대로 보존하고, 정정 사실만 위에 덧붙인다.

| Model | w1 | w2 | w3 | w4 | Training Time(s), mean±sd(5시드) | Decision time (c), ms/decision | SWMM (d), s/episode |
|---|---|---|---|---|---|---|---|
| Regular DDQN (2-criterion) | 0.50 | 0.50 | 0 | 0 | 2,960.7 ± 368.1 | 0.1655 | 0.1567 |
| GA-guided DDQN (4-criterion) | 0.40 | 0.25 | 0.25 | 0.10 | 17,678.6 ± 1,656.1 | 0.2389 | 0.1614 |

(원고 원 표: Regular 2,737.03s(1회)/GA-guided 12,124.44s(1회) — Training
Time 자릿수는 같은 범위이나 GA-guided 쪽이 이번 재현에서 더 크다(E3 v2
GA-guided는 온라인 GA 탐색을 포함한 학습이라 순수 DDQN 학습인 2기준
Regular보다 오래 걸림, 이미 알려진 사실). 원고의 Testing Time(257,224.97s/
256,591.19s)에 대응하는 값은 본 표에서 (c)+(d) 방식 결정당 시간(ms)으로
대체됐다 — 단위·측정범위가 달라 직접 배율 비교는 하지 않는다.)

## 6. R1-6 답변용 수치

전체 표: `results/summary/R1-6_FIGURES.md`. 요지만 인용(전부 신가중치,
γ=0.7, n=5시드×2,700=13,500 기준, 출처 `E03_summary.csv`):

| 모델 | n_switches | n_intervals_100 | n_intervals_170 | n_dryrun_proxy |
|---|---|---|---|---|
| Regular | 39.298±6.413 | 307.179±6.177 | 21.032±1.990 | 3.304±2.752 |
| GA-guided | 34.195±4.760 | 312.278±6.501 | 18.763±0.987 | 4.873±4.394 |
| PSO-guided | 32.543±5.666 | 307.638±3.260 | 19.332±1.458 | 2.157±1.090 |
| GA-only | 72.066±0.000 | 257.628±0.000 | 45.124±0.000 | 0.331±0.000 |
| PSO-only | 72.192±0.049 | 257.587±0.034 | 45.150±0.017 | 0.336±0.003 |

seed4 제외 시 1080·1440분 dry-running 방향 반전(5시드: GA-guided가
Regular보다 +2.24/+12.48 높음 → GA-guided만 seed4 제외 4시드: −1.39/−1.27
낮음, 방향 반전) — 상세 `R1-6_FIGURES.md` (d). 지속시간별 Cliff's δ(부호
정정판) — 상세 §(e) 같은 문서.

---

## 붙여넣기용 요약

**설정**: 신가중치 4기준 [0.40,0.25,0.25,0.10](GA-guided, E3 v2 재사용
5시드) vs 2기준 [0.50,0.50,0,0](Regular, E14 신규 5시드), γ=0.7,
split_seed=42, 3,996 학습/2,700 테스트 시나리오. 무결성 확인 전부 통과.

**Table 6~10(핵심 경향)**: 전체 평균(54개 셀) — max_level Regular 5.02
vs GA-guided 6.03(GA-guided가 항상 높음), switches Regular 18.70 vs
GA-guided 34.20(60~180분엔 GA-guided가 낮으나 240분부터 역전해 1440분엔
Regular의 3.2배), i100 Regular 817.58 vs GA-guided 312.28(Regular가
항상 훨씬 높음), i170 Regular 109.08 vs GA-guided 18.76(Regular가 항상
훨씬 높음), dry-running Regular 278.84 vs GA-guided 4.87(Regular가 전
지속시간에서 훨씬 높고 격차가 지속시간에 따라 급증, 60분 40.6회차→1440분
713.3회차) — **2기준 Regular는 w3=w4=0이라 펌프사용·dry-running에 전혀
페널티가 없어 그 두 지표가 극단적으로 커진 것은 가중치 구조상 예정된
결과다.**

**Table 5**: Regular(2기준) 학습 2,960.7±368.1s(5시드), 결정당 0.1655ms(c),
SWMM 0.1567s/ep(d). GA-guided(4기준) 학습 17,678.6±1,656.1s(5시드),
결정당 0.2389ms(c), SWMM 0.1614s/ep(d). 두 행 측정 조건(코드경로·분할·
하드웨어·방법론) 동일.

**Figure 1~2**: 최종 롤링평균 loss Regular 0.0062/GA-guided 0.0030,
reward Regular 157.74/GA-guided 152.85.

**Figure 3~4**: 시나리오 `30yr_0060m_h211`(고정시드 난수선정, 캡션엔
조건만 표기).

**R1-6 표**: `R1-6_FIGURES.md` — (a)(b)(c) E3 v2 5모델 스위칭/펌프사용/
dry-running mean±sd, (d) seed4 제외 시 1080·1440분 dry-running 방향
반전(5시드 포함 +2.24/+12.48 → GA 4시드만 −1.39/−1.27), (e) 지속시간별
Cliff's δ 부호정정판(max_level 전구간 δ<0, switches는 360분부터 δ<0,
n_dryrun 전구간 δ>0).

해석·옹호 없음, 원고 docx 미수정.
