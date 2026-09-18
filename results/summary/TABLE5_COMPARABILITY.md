# Table 5 (재생성판) 비교가능성 검증 — 학습시간 병렬조건 확인

해석·옹호 없이 사실·숫자만 기록한다.

## 1. 확인 대상

`docs/E14_REPORT.md` §5에서 조립한 Table 5(재생성판)의 두 행:

| Model | Training Time (5시드 mean±sd) | 출처 |
|---|---|---|
| Regular(2기준) | 2,960.7±368.1 s | `results/E14_two_criteria/Regular/seed{1..5}/run_meta.json` (`src/run_e14_two_criteria.py`) |
| GA-guided(4기준) | 17,678.6±1,656.1 s | `results/E03_seeds/GA-guided/seed{1..5}/run_meta.json` (`src/run_e03_seeds.py`) |

## 2. 병렬 워커 수 — 실제 로그 확인

`results/_log/E14.log`:
```
[2026-09-02T17:10:20] 5 pending seed(s), weights=[0.5, 0.5, 0.0, 0.0], workers=2
```
E14 Regular 5시드는 **`--workers 2`**로 실행되었다(`src/run_e14_two_criteria.py`
의 `--workers` 기본값도 2).

`results/_log/E03.log`(E3 v2 GA-guided를 포함한 25-combo 배치, 두 차례
전체 재실행 기록이 모두 존재):
```
[2026-08-26T21:54:06] 25 total combos, 0 already done, 25 to run, workers=8
...
[2026-08-29T16:57:44] 25 total combos, 0 already done, 25 to run, workers=8
```
E3 v2 GA-guided(및 Regular 포함 5모델×5시드=25개 조합)는 **`--workers
8`**로 실행되었다(`src/run_e03_seeds.py`의 `--workers` 기본값도 8, 사용
예시 문자열도 `--workers 8`).

**→ Table 5의 두 행은 서로 다른 병렬 조건(workers=2 vs workers=8)에서
측정되었다.** 이는 확인된 사실이며 추정이 아니다.

## 3. 같은 모델(Regular)로 본 워커수 영향 — 직접 비교 가능한 자연실험

E3 v2 Regular(4기준, `results/E03_seeds/Regular/`)와 E14 Regular(2기준,
`results/E14_two_criteria/Regular/`)는 동일 코드 경로
(`dqn_from_demon_v1.train()`, `perform_evaluate.evaluate_dpn_gru`)를
공유하고, `run_meta.json` 확인 결과 하이퍼파라미터·데이터 규모가 전부
동일하다(lr=0.001, batch_size=20, target_update_C=10, gru_layers=3,
gru_hidden=32, seq_len=5, train_n=3996, test_n=2700, split_seed=42).
차이는 (a) reward weights(4기준 원가중치 vs w=[0.5,0.5,0,0])와 (b) 병렬
워커 수(8 vs 2)뿐이다.

| | E3 v2 Regular(workers=8) | E14 Regular(workers=2) | 비율 |
|---|---|---|---|
| seed1 elapsed_train_s | 10,230.64 | 3,128.10 | 3.27× |
| seed2 | 10,230.87 | 3,126.34 | 3.27× |
| seed3 | 10,232.58 | 3,121.79 | 3.28× |
| seed4 | 10,231.60 | 3,124.95 | 3.27× |
| seed5 | 10,232.92 | 2,302.20 | 4.44× |
| **5시드 평균** | **10,231.72** | **2,960.68** | **3.46×** |

E14 seed5만 다른 4개 시드(3,121.8~3,128.1s)보다 뚜렷하게 짧다
(2,302.2s). `run_e14_two_criteria.py`는 `ProcessPoolExecutor(max_workers=2)`
로 5개 시드 작업을 제출하므로, 5개 작업을 2슬롯에 순차 배정하면 마지막
1개(5번째)는 상당 구간 다른 작업과 경합 없이 단독 실행되는 구간이
생긴다 — seed5가 유독 빠른 것은 이 배정 구조와 일치한다(추가 로그로
동시실행 여부를 직접 확인하지는 않았음, 정황 일치만 기록).

**같은 모델(Regular)에서 workers=8 대비 workers=2가 학습 벽시계시간을
약 3.46배 단축시켰다** — 이 배율은 이전 턴에서 보고한 "예측(2.84h)
대비 실측(0.82h) 1/3.5" 오차와 동일한 현상이다(2.84h→10,230s,
0.82h→2,961s는 같은 값의 다른 단위 표기).

## 4. 하드웨어 부하 기록

`run_meta.json`(E3v2·E14 공통 스키마: `exp_id, model, seed, git_commit,
started_at, finished_at, elapsed_train_s, hyperparams, reward_weights,
heuristic_params, data, versions`)에는 CPU/GPU 부하 필드가 없다. 저장소
전체에서 CPU/GPU 사용률 로그 파일(`nvidia-smi`, `htop` 캡처 등)을
검색했으나 **존재하지 않는다.** 워커수 외의 시스템 부하 요인(다른
프로세스의 동시 점유 등)은 기록으로 확인할 방법이 없다.

## 5. 그 외 조건 차이

- **배치 크기·시퀀스 길이·GRU 구조**: E3v2 Regular = E14 Regular 동일
  (§3 표 참조). GA-guided는 `target_update_C`가 10이 아니라 **200**이고,
  `min_replay_size`/`ga_start`/`ga_end`/`eps_start`/`eps_end`(휴리스틱
  안내 탐색 스케줄) 필드가 추가로 존재 — 이는 GA-guided가 Regular와
  다른 학습 알고리즘(휴리스틱 유도 시연 재생)을 쓰기 때문에 존재하는
  **의도된 알고리즘 차이**이지 병렬조건 문제가 아니다.
- **데이터 규모**: train_n=3996, test_n=2700, split_seed=42로 세 조건
  (E3v2 Regular, E3v2 GA-guided, E14 Regular) 전부 동일.
- **heuristic_params**: GA-guided는 `population_size=100,
  num_generations=100`의 유전알고리즘 시연 데이터 생성 비용이 추가된다
  — Regular에는 없는 항목이며, 이 자체가 GA-guided의 학습시간이
  Regular보다 (병렬조건과 무관하게) 더 걸릴 것으로 기대되는 이유 중
  하나다. 즉 GA-guided의 절대 학습시간이 Regular보다 큰 것 자체는
  병렬조건 문제와 별개로 알고리즘 차이로도 설명 가능한 부분이 있다 —
  다만 **"몇 배 더 걸리는가"라는 정량값은 워커수가 다른 조건에서
  측정되었으므로 그대로 인용할 수 없다.**

## 6. 결론 (판단 아님, 조건 확인 결과)

Table 5(재생성판)의 Regular 행(workers=2)과 GA-guided 행(workers=8)은
**동일 조건에서 측정되지 않았다.** R4-12가 지적한 것이 정확히 이
계산시간 비교 문제이므로, 이 상태로 개정판에 실으면 동일한 문제를
반복하게 된다.

## 7. 제안 (실행하지 않음, 세 안 제시)

### (a) 조건이 동일했다면 그대로 사용
해당 없음 — §2에서 확인했듯 조건이 다르다(workers=2 vs 8).

### (b) 동일 조건 재측정

두 가지 방향이 가능하며, 워커수를 맞추는 것만으로는 **경합 강도까지
같아지지 않는다**는 점에 주의해야 한다: 원본 GA-guided 측정은 5모델×
5시드=25개 작업이 8슬롯을 연속으로 채우며 진행된 반면, Regular 5시드
단독 배치는 작업이 5개뿐이라 `--workers 8`로 재실행해도 최대 5개
프로세스만 동시 실행되어 원래의 8-슬롯 포화 경합 상태를 재현하지
못한다. 따라서 진짜 동일 조건을 만들려면 아래 두 방향 중 하나가
필요하다.

- **(b-1) GA-guided를 Regular와 같은 고립 조건(workers=2, 5개 작업만)
  으로 재측정**: §3에서 확인한 배율(workers=8→2 전환 시 약 3.46배
  단축)을 GA-guided에 그대로 적용하면(**주의: 이는 Regular에서 관찰된
  배율을 GA-guided로 외삽한 가정이며, GA-guided 자체로 확인된 값이
  아니다**) 시드당 약 17,678.6s/3.46 ≈ **5,109s(약 1.42h)**로 추정.
  5시드를 `--workers 2`로 돌리면 E14 Regular와 같은 작업배정 구조(2
  슬롯, 마지막 1개는 상대적으로 빠름) 상 총 벽시계시간은 대략 3라운드
  분(2+2+1) ≈ **5,109s×2.5~3 ≈ 3.5~4.3h** 로 거친 추정.
- **(b-2) Regular를 GA-guided와 같은 25-combo 8-worker 포화 배치로
  재측정**: `results/_log/E03.log`의 기존 25-combo 실행(약 12.5h 벽시계
  추정치, 실제로도 약 12.5~13.5h 소요된 기록 있음)을 5모델 전부 다시
  돌려야 동일 경합조건이 재현된다 — Regular 5시드만 다시 도는 것이
  아니라 **PSO-guided/GA-only/PSO-only까지 포함한 25개 조합 전체
  재학습**이 필요해, 총 소요시간은 기존 기록상 약 12.5h(또는 그 이상,
  로그상 실제 완료까지 08-26 21:54→08-27 10:21 = 약 12.4h, 08-29
  16:57→08-30 05:26 = 약 12.5h) 규모다. 학습 자체는 5개 조건(4모델
  25-9=16 남은 조합... 실제로는 GA-guided/PSO-guided/GA-only/PSO-only
  나머지 20개조합)이 이미 완료돼 있어 재사용 가능하다면, **Regular
  5시드만 별도로 이 20개 조합과 동시에 다시 돌리는 방식은 불가능**
  (기존 20개 조합은 이미 학습이 끝나 프로세스가 존재하지 않으므로
  경합을 인위적으로 재현하려면 실제로 그만큼의 더미 부하를 동시에
  걸거나 25개 전부를 다시 학습해야 함).
- (b-1)이 (b-2)보다 재학습 규모가 훨씬 작다(5회 vs 25회 재학습).

### (c) 재측정 불가 시 — 학습시간을 빼고 추론시간만 보고

`src/run_e14_timing.py`(E14 Regular용)와 `src/run_e10_timing_detailed.py`
(GA-guided용, 기존 E10)는 **둘 다 `ProcessPoolExecutor` 없이 단일
프로세스로 시나리오를 순차 처리**한다(코드 확인:
`run_e14_timing.py`는 `for inp in scenarios:` 단순 루프,
`run_e10_timing_detailed.py:126`도 동일 패턴) — 즉 **Decision(c)/SWMM(d)
추론시간 수치는 병렬조건의 영향을 받지 않으며, 두 모델 모두 동일하게
고립된 단일 프로세스·54시나리오·seed1 조건에서 측정되었다.** Table 5에서
"Training Time" 행을 제외하고 Decision(c)=Regular 0.1655ms/GA-guided
0.2389ms, SWMM(d) 관련 값만 보고하면 재학습 없이 비교가능성 문제를
피할 수 있다.

## 8. 요약

- Table 5 재생성판의 학습시간 행은 서로 다른 병렬조건(workers=2 vs 8)
  에서 측정되어 **현재 상태로는 비교 불가**.
- 같은 모델(Regular)에서 워커수 차이가 벽시계시간을 약 3.46배
  바꾼다는 것은 직접 측정된 사실이다(§3).
- 이 배율을 GA-guided에 그대로 적용한 재측정 소요시간 추정치는
  §7-(b-1)에 제시했으나, **가정에 근거한 외삽**이며 확정값이 아니다.
- 추론시간(Decision/SWMM) 수치는 애초에 병렬조건 confound가 없다(§7-(c)).
- 결정(재학습 여부·범위·Table 5 서술 방식)은 하지 않았다.

## 9. Table 5 열 구분 재확인 — Training Time vs Testing Time

원고 Table 5(`docs/response/MANUSCRIPT_TEXT.md` L540-546)의 정확한
값:

| Model | w1 | w2 | w3 | w4 | Training Time (s) | Testing Time (s) |
|---|---|---|---|---|---|---|
| Regular DDQN | 0.50 | 0.50 | 0 | 0 | 2,737.03 | 257,224.97 |
| GA-guided DDQN | 0.25 | 0.40 | 0.25 | 0.10 | 12,124.44 | 256,591.19 |

R4-12가 지적한 257,224.97s/256,591.19s는 **Testing Time** 열이다(Training
Time 열이 아님) — 이번 절에서 재확인.

### 9.1 Testing Time 추적 가능성

- 2,700 시나리오 기준 시나리오당 단순 나눗셈: Regular
  257,224.97/2700 = **95.269 s/시나리오**, GA-guided
  256,591.19/2700 = **95.034 s/시나리오**.
- E14/E10의 (a)-(d) 방법론 실측(단일프로세스, 54시나리오, seed1) —
  SWMM 1회(에피소드당) 비용: Regular(2기준) 0.1567 s/episode,
  GA-guided(4기준) 0.1614 s/episode(`docs/E14_REPORT.md` §5) — **95
  s/시나리오는 이 SWMM 1회 비용의 약 590~608배**이며, (a)+(b)+(c)+(d)
  전부 합쳐도 결정당 1ms 미만 수준이라 설명되지 않는다(기존 확인,
  재확인만).
- **독립적인 추가 확인(이번에 신규)**: E3 v2 자신의 `test_metrics.csv`에
  기록된 시나리오별 실측 `total_time_s`(평가 시 직접 측정된 값, E10과
  별개 소스) 합계 — Regular seed1: 2,700개 합 **1,053.35 s**(평균
  0.390 s/시나리오), GA-guided seed1: 합 **1,500.29 s**(평균 0.556
  s/시나리오). 원고 Testing Time과의 비율: Regular
  257,224.97/1,053.35 = **244.2배**, GA-guided
  256,591.19/1,500.29 = **171.0배**.
- **두 모델의 배율(244.2배 vs 171.0배)이 서로 다르다** — 즉 Testing
  Time이 실측 평가시간에 어떤 공통 상수를 곱하거나 더한 값이라는
  가설도 성립하지 않는다(공통 배율이라면 두 모델에서 같아야 함).
  E10 기반 확인(기존)과 test_metrics.csv 기반 확인(신규) **두 개의
  독립적인 실측 경로 모두 Testing Time 257k대 수치를 설명하지
  못한다.**
- **결론**: Testing Time이 무엇을 측정했는지는 이번 조사로도 추적되지
  않는다(원본 로그 소실, 기존 결론 유지). 다만 "설명 불가"라는
  판단을 뒷받침하는 독립 증거가 하나(E10) 에서 둘(E10 + 자체
  total_time_s 실측)로 늘었다.

### 9.2 Training Time 열의 근접성

- Regular: 원고 2,737.03 s vs E14 실측(2기준, 5시드평균) 2,960.68 s —
  차이 **+8.17%**(E14가 더 큼). Testing Time의 배율 오차(171~244배)에
  비하면 **두 자릿수 이상 작은 차이**다.
- 이 근접성이 뜻하는 바(사실 기록, 판단 아님): Training Time 열은
  Testing Time 열과 달리 실제 측정 가능한 벽시계 시간(모델 학습에
  걸린 시간)과 **같은 자릿수, 한 자릿수 % 수준의 오차**로 일치한다 —
  Testing Time처럼 자릿수가 다른 값은 아니다. 다만 §2-§3에서 확인한
  병렬조건 차이(원고가 어떤 workers 설정으로 측정했는지는 원본 로그
  소실로 불명)가 있어, 8.17% 차이 자체가 "거의 같다"는 뜻인지 "약간
  다른 조건에서 측정됐다"는 뜻인지는 이 비교만으로 결정할 수 없다.

### 9.3 E15 결과 (완료, 2026-09-03 갱신 — §9.3 사전진술을 실제값으로 교체)

- 원고 GA-guided Training Time: **12,124.44 s**.
- E3 v2 GA-guided(workers=8) 실측: 17,678.6±1,656.1 s(5시드 평균) —
  원고값보다 약 **+45.7%** 큼.
- **E15(GA-guided, workers=2) 실측**(`results/E15_timing/GA-guided/
  seed{1..5}/run_meta.json`): elapsed_train_s = [10,154.56, 10,199.12,
  10,156.92, 10,214.89, 9,630.16], **5시드 평균 10,071.13 s, sd
  247.90 s(ddof=1)** — 원고값(12,124.44s) 대비 **−16.94%**(E15가 더
  작음).
- **workers=8→2 배율은 GA-guided에서 1.76배**(17,678.6/10,071.13)였다
  — §3에서 확인한 Regular의 배율(3.46배)과 **다르다**. §7-(b-1)에서
  Regular의 배율을 GA-guided에 그대로 적용해 추정한 값(약 5,109
  s/시드)은 **실측(10,071.13 s/시드)과 약 2배 차이**로 빗나갔다 —
  워커수-배율이 모델(또는 알고리즘 구조)에 따라 다르다는 뜻이며,
  이번 외삽이 부정확했음을 그대로 기록한다.
- 원고값과의 근접도 비교: workers=8(45.7% 오차) → workers=2(−16.94%
  오차)로 **오차가 줄었다**(더 근접). 다만 정확히 일치하지는
  않는다. 이 근접도 변화가 "원고가 낮은 병렬조건에서 측정했을
  가능성"을 완전히 확인하지는 못하지만(§9.2와 같은 판단 유보),
  **workers를 낮추는 방향이 원고값에 더 가까워지는 방향과 일치한다는
  사실**은 기록한다.

## 10. Table 5 최종 구성 (확정)

E14 Regular(2기준)와 E15 GA-guided(4기준)를 **동일 병렬조건
(workers=2)**으로 확보했으므로, 두 Training Time 값은 이제 직접
비교 가능하다. Testing Time 열은 §9에서 확인한 대로 원본 측정
내용을 추적할 수 없어 삭제하고, 결정 1회당 추론시간((c) 방법론
기준)으로 대체한다.

| Model | w1 | w2 | w3 | w4 | Training Time (s), mean±sd (5시드)(a) | Decision time (c), ms/decision(b) |
|---|---|---|---|---|---|---|
| Regular DDQN (2-criterion) | 0.50 | 0.50 | 0 | 0 | 2,960.7 ± 368.1 | 0.1655 |
| GA-guided DDQN (4-criterion) | 0.40 | 0.25 | 0.25 | 0.10 | 10,071.1 ± 247.9 | 0.2389 |

**각주(각 값의 측정 조건)**

- (a) Training Time: 순수 학습루프 벽시계시간만 측정
  (`perform_evaluate.py`의 `_evaluate_dpn_guided`/`evaluate_dpn_gru`
  내부 `time.perf_counter()`로 `trainer(...)` 호출 구간만 측정, 데이터
  로딩·테스트평가 시간 제외). 병렬조건: **두 행 모두 `--workers 2`**
  (Regular: `results/E14_two_criteria/`, `src/run_e14_two_criteria.py`;
  GA-guided: `results/E15_timing/`, `src/run_e15_timing.py`). 하드웨어:
  CPU i9-13900KF, GPU RTX 3080 Ti 12GB, RAM 94GB(본 저장소 전체
  타이밍측정 공통 하드웨어). 데이터: train_n=3,996, test_n=2,700,
  split_seed=42, γ=0.7, 5시드(1-5) 평균±표준편차(ddof=1).
- (b) Decision time (c): 단일 프로세스(병렬 없음), 54개 시나리오
  (계층화 부표본, 1/stratum), seed1, (a)순수forward + (b)전처리(정규화
  +5스텝 시퀀스) + (c)환경step오버헤드(SWMM 미포함, `WaterGym.step()`만)
  까지 누적한 결정당 시간. Regular: `src/run_e14_timing.py`, GA-guided:
  `src/run_e10_timing_detailed.py`(기존 E10 측정 재사용, 재측정 안 함).
  SWMM 1회(에피소드당) 비용 (d)는 이 표에서 제외했으나 참고용으로
  병기: Regular 0.1567 s/episode, GA-guided 0.1614 s/episode.
- GA-guided의 가중치는 **신가중치**([0.40,0.25,0.25,0.10], E6 채택값)이며
  원고 원본 Table 5의 GA-guided 행 가중치([0.25,0.40,0.25,0.10], 구가중치)와
  **다르다** — 이미 알려진 사실(`docs/response/SECTION_3_1_REGENERATION.md`),
  Table 5 교체 시 원고에 반드시 명시해야 한다.
- Testing Time 열은 원 원고에 존재했으나(257,224.97s/256,591.19s) §9의
  확인 결과 추적 불가로 삭제했다 — 삭제 사유 자체도 각주 또는 본문에
  명시할 필요가 있다(무엇을 측정했는지 확정 불가, 원본 로그 소실로
  역추적 불가).

## 11. 요약 (판단 없음)

- Table 5 두 행(Training Time)은 이제 동일 병렬조건(workers=2)에서
  측정되어 §2-§3의 비교불가 문제가 해소됐다.
- 재측정된 GA-guided Training Time(10,071.1s)은 원고값(12,124.44s)에
  E3v2측정(17,678.6s)보다 가깝지만(오차 −16.94% vs +45.7%) 정확히
  일치하지는 않는다.
- workers=8→2 배율은 Regular(3.46배)와 GA-guided(1.76배)가 서로 달라
  §7-(b-1)의 외삽 추정치는 부정확했다(실측 대비 약 2배 차이).
- Testing Time은 이번에도(§9) 추적되지 않아 최종 표에서 삭제하고
  Decision(c) 값으로 대체했다.
