# 코드베이스 맵 — 중복 구현 및 위험 항목 인벤토리 (E0-2, 진행 중)

**작성일**: 2026-08-25
**상태**: 부분 작성 — `stratified_split_data` 3중 구현 전체 대조는 별도 진행 예정, 우선 아래 위험 항목부터 기록.

## 위험 항목 1 — `perform_evaluate.py:34`의 `random.seed(time.time())`가 import 시점에 실행됨

```python
# perform_evaluate.py:34 (모듈 최상위, 함수 밖)
random.seed(time.time())
```

- 이 줄은 `perform_evaluate` 모듈을 **import하는 순간** 실행된다. 즉 `import perform_evaluate`만 해도 전역 `random` 상태가 현재 시각으로 재시드된다.
- `stratified_split_data()`(같은 파일 L36-89)는 이 전역 `random` 상태를 그대로 사용해 `random.shuffle()`로 층화표집을 수행한다 — 시드가 실행마다(그리고 import 시점마다) 달라지므로 **같은 스크립트를 두 번 실행해도 train/test 시나리오 구성이 달라진다.**
- **E3의 시나리오 단위 paired 비교가 성립하려면 5모델(Regular/GA-guided/PSO-guided/GA-only/PSO-only) × 5시드 = 25회 실행이 전부 동일한 test 시나리오 집합을 대상으로 해야 한다.** 현재 구조로는 각 실행이 이 모듈을 import할 때마다 분할이 달라질 수 있어 이 전제가 깨진다.
- 실제로 이 문제 때문에 본편의 실제 2,700개 테스트셋을 고정 저장한 파일이 기존 저장소 어디에도 없었다 (`docs/DATA_SPLIT_AUDIT.md` 참조).

**대응 (이미 별도로 완료)**: `data/splits/fixed_split_seed42.json`을 생성해 고정 분할을 확보함(`docs/DATA_SPLIT_AUDIT.md`의 "경로 이식성" 절 참조). **단, 이 json 파일이 존재하는 것과 실제 실행 스크립트가 이걸 읽는 것은 별개다.**

**완료 (2026-08-25) — 진입점 통일**: 아래 실행 스크립트 전부가 각자 `stratified_split_data()`를 호출하던 것을 멈추고 `data/splits/fixed_split_seed42.json` 기준 고정 분할을 읽도록 통일됨. `perform_evaluate.py:34`의 `random.seed(time.time())`도 주석 처리되어 import 시점 재시딩이 더 이상 발생하지 않는다.

| 파일 | 처리 방식 |
|---|---|
| `perform_evaluate.py` | `data_paths.load_fixed_split()` 사용 |
| `run_local_search_test.py` | `data_paths.load_fixed_split()` 사용 |
| `run_dqn_guided_pso.py` | `data_paths.load_fixed_split()` 사용 |
| `run_ga_guided_switching_stability.py` | `data_paths.load_fixed_split()` 사용 |
| `benchmark_test_runtime.py` | `data_paths.load_fixed_split()` 사용 |
| `benchmark_full_data_runtime.py` | 전체 모집단(7,440개) 대상 런타임 측정이 목적이므로 분할을 쓰지 않음 — `selected_inp_relative.txt` + `resolve_path()`로 전수 접근 (의도적 미사용, 결함 아님) |
| `repeat_heuristic_benchmark.py` | 호출부만 `load_fixed_split()`로 교체. 기존 `fixed_test_paths()` 정의는 삭제하지 않고 미사용 상태로 보존 (지침 4항: 미사용 코드 삭제 금지, 표시만) |

**`results/repeated_heuristics/fixed_test_2700.json` 대조 — 완료**: `fixed_split_seed42.json`과 파일명 기준 대조 결과 겹침 1,111/2,700 = 41.1%로, 서로 다른 독립 표본임을 확정. 상세는 `docs/RESULTS_PROVENANCE.md` 참조.

## (예정) 중복 구현 인벤토리 — 별도 작업

아래는 지금까지 조사 과정에서 이미 발견된 항목이며, 전체 3중 대조표는 후속 작업에서 완성 예정:
- `stratified_split_data`: `perform_evaluate.py:36`, `cost_model_v2.py:111`, `early_162_60pct/src/fair_ga/split.py` — 3곳. 로직 차이(나머지 할당 vs 이중 floor)는 `docs/DATA_SPLIT_AUDIT.md`에 이미 기록됨.
- `_volume_to_level()`(`genetic_algo.py:53-69`) vs `vol2level()`(`water_gym.py:192-214`) — 독립 작성된 동일 로직의 중복. 현재는 수치상 일치 확인됨(`docs/E02_OBJECTIVE_AUDIT.md`).
- `cost_model_v2.py`의 물리 기반 dry-run/wear/repair 비용 모델 — GA/PSO/DQN 어디에서도 import되지 않는 미사용 코드 (`docs/E02_ARGMINMAX.md`).
- **`evaluate_genetic_algo()`(`perform_evaluate.py:553-657`) vs `_evaluate_heuristic()`(`perform_evaluate.py:659-716`)** — GA-only 평가 루프(`evaluate_genetic_algo`)는 PSO/local search가 공유하는 `_evaluate_heuristic()`을 거치지 않고, 동일한 (year×duration) 집계·`count_changes`/`count_pumps`/overpump 합산 로직을 별도로 반복 구현한 것. `evaluate_pso`/`evaluate_local_search`는 `_evaluate_heuristic(optimizer_class=..., run_method=...)`로 통일되어 있는데 `evaluate_genetic_algo`만 통일에서 빠져 있다. E3의 5모델(Regular/GA-guided/PSO-guided/GA-only/PSO-only) 중 GA-only가 바로 이 미통일 함수로 평가되므로, 시나리오 단위 CSV 저장(2.2절) 작업 시 `_evaluate_heuristic`과 별도로 이 함수에도 동일한 저장 로직을 넣어야 함 (2026-08-25 결정, 리팩터링 아님 — 통합하지 않고 각자에 추가만 함).

## 확인 완료 — `test_metrics.csv`(2.2절) `total_time_s`, `n_dryrun_proxy` 컬럼 정의

**total_time_s는 SWMM 시뮬레이션(WaterGym.reset)을 포함한다. inference_time_s와 swmm_time_s로의 분해는 E10에서 수행한다.**

근거(서브에이전트 조사, 2026-08-25): `total_time_s` 계측 대상인 4개 호출이 전부 `WaterGym(...)` 생성 + `.reset()`을 **함수 내부**에서 수행하며, `reset()`이 `SimSwmm.execute()`(`sim_swmm.py:100`, `pyswmm.Simulation`)를 실제로 실행한다(`water_gym.py:90-91`). `step()`(`water_gym.py:114-189`)은 `reset()`이 미리 계산한 배열을 인덱싱만 하고 SWMM을 호출하지 않는다.

| 대상 함수 | 계측 호출 | reset() 위치 |
|---|---|---|
| `evaluate_dpn_gru`, `_evaluate_dpn_guided` | `dqn_from_demon_v1.test_model()` | `dqn_from_demon_v1.py:896-897` |
| `evaluate_genetic_algo` | `GeneticAlgorithm.run_genetic_algo()` | `genetic_algo.py:302-304` |
| `_evaluate_heuristic`(PSO/local search) | `run_pso()` / `run_local_search()` | `pso.py:86-89`, `local_search.py:74-77` |
| `evaluate_man_policy` | `man_policy.operation()` | `man_policy.py:87-88` |

**n_dryrun_proxy는 이진 플래그의 합(=발생 횟수)이다.** `water_gym.py:162-167`의 `step()`은 `cur_vol < 0.0`일 때만 `over_pump = 1`(아니면 0)을 스텝마다 산출하고, `reward()`(`water_gym.py:267`)의 `info` 튜플 마지막 원소가 이 값을 그대로 담는다. 기존 집계 `counts_overpump[...] += np.sum(ainfos[:, -1])`와 신설 `n_dryrun_proxy`는 동일하게 "과다펌핑이 발생한 스텝 수"(정수 카운트)이며 비율이나 연속값이 아니다.

R4-12(Table 5 테스트 시간 257,224초가 Figure 10 추론시간 서술과 상충) 관련 시사점은 `docs/MANUSCRIPT_FIXES.md` 참조.

## 확인 완료 — `evaluate_dpn_gru(train=False)` 미검증 버그 (2026-08-26)

`save_scenario_metrics()` 통합 후 실제 체크포인트로 검증하는 과정에서 발견. `evaluate_dpn_gru`의 `train=False` 분기(`model = None`만 설정)는 `basic_losses`/`basic_rewards`를 정의하지 않는데, 함수 끝의 `return basic_losses, basic_rewards, dqn_test_rewards, elapsed`가 이를 참조해 `UnboundLocalError`로 항상 실패한다.

- `grep -rn "train=False" src/`로 저장소 전체를 확인한 결과 이 경로를 호출하는 현재 유효 코드는 없음(`obsolete/`의 주석 처리된 줄만 존재) — **논문 결과 생성에는 영향 없음**, 애초에 한 번도 실행되지 않은 미검증 경로로 보임.
- 리팩터링 금지 지침에 따라 수정하지 않고 표시만 한다. E3 등에서 사전학습 체크포인트 재사용(재학습 생략)을 시도할 계획이 생기면 이 지점을 먼저 고쳐야 한다. 현재 계획서(E3)는 5모델×5시드 전부 재학습이 원칙이므로 이 경로를 타지 않는다.
- `save_scenario_metrics()` 자체는 이 버그 이전 시점(루프 종료 후, return 직전)에 정상 실행되어 CSV가 올바르게 생성됨을 확인함(`train=False`, 시나리오 2건, `seed=999`, `model_label='smoke_test'`로 검증 — 두 컬럼 모두 CSV에 정확히 반영됨).
- 수정 필요 항목으로 `docs/REVISION_EXPERIMENT_PLAN.md` E10 절에도 기록함.

**부수 발견**: `results/`의 기존 CSV 다수가 `-r--r--r--`(444, 쓰기 금지)로 되어 있다. 검증 중 `evaluate_dpn_gru`가 기존 `dqn_gru_regular_..._acttype0.csv`를 덮어쓰려다 `PermissionError`로 실패한 뒤 발견했다. 의도적 보호로 보이며, 신규 실험이 레거시 결과와 같은 파일명(`{model_path 기반 str_w}.csv`)을 쓰면 `save_results()` 단계에서 즉시 실패한다 — E3 이후 신규 실행은 계획서 2.1/2.2절의 `results/E0X_.../` 하위 경로를 반드시 써서 파일명이 겹치지 않게 해야 한다. 상세 대응은 `docs/REVISION_EXPERIMENT_PLAN.md` 2.2절에 기록함.

## 확인 완료 — E0 시드/스모크 검증 중 발견한 2건 (2026-08-26, 미조사·표시만)

**(1) GA-guided·PSO-guided 학습 첫 에피소드 loss가 `nan`**. `dqn_from_demon_v1.py`의 `_train_guided_by_optimizer()`(GA·PSO 공용, `train_guided_byGA`/`train_guided_byPSO`가 호출)로 학습하면 첫 번째 학습 에피소드의 `train_losses[0]`가 `nan`으로 출력된다. 동일 시드 2회 재현 검증에서 두 optimizer(GA, PSO) 모두, 두 번의 반복 실행 모두에서 동일하게 재현됨(잡음이 아니라 결정적 버그). 원인은 조사하지 않음 — `min_replay_size`(기본 `batch_size*5`=100) 도달 직후 첫 미니배치 업데이트 시점의 경계 조건으로 추정되나 확정 아님. Regular DDQN(`dqn_from_demon_v1.train()`)에서는 재현되지 않았다(같은 조건에서 loss가 정상적으로 감소). 표시만 하고 수정하지 않음 — E3 학습 로그 집계 시 `nan`을 제외 처리할지 결정 필요.

**(2) GA-only/PSO-only가 소규모 스모크 테스트 3개 시나리오 전부에서 액션 0(전량 정지)만 선택**. `evaluate_genetic_algo()`(`GeneticAlgorithm.run_genetic_algo()`)와 `evaluate_pso()`(`ParticleSwarmOptimization.run_pso()`, `_evaluate_heuristic()` 경유)를 60분 지속시간 3개 시나리오(20yr/30yr/20yr, 서로 다른 강우)에 대해 실행한 결과, `n_switches=0`, `n_intervals_100=0`, `n_intervals_170=0`, `overflow_flag=1`, `cum_reward=37.699999999999996`(세 시나리오 모두 소수점까지 동일)로 나왔다.

- **원인 규명 완료(2026-08-26, 30개 시나리오로 확장 검증 + 실측 확인)**: 코드 버그가 아니라 `objective_function()`의 1-스텝 목적함수 구조가 가중치 (0.25,0.4,0.25,0.1)에서 "펌프 미가동"을 수학적으로 항상 우세하게 만드는 필연적 귀결이다. 액션 0은 `vol_reward=0`(유출량이 0이므로) 대신 `act_reward`·`energy_reward`가 최댓값(각 1)을 받아 상수 0.65를 얻는데, 다른 어떤 액션으로 전환해도 `vol_reward`의 이론적 최댓값(2.0 × w1=0.25=0.5) 조차 스위칭 비용을 상쇄하지 못해 0.65를 넘지 못한다 — 720스텝 침수 시나리오 실측에서도 액션 0이 시종일관 0.6500으로 1위, 나머지는 최대 0.6091. 상세 유도·수식·실측표는 `docs/E00_HEURISTIC_MYOPIA_FINDING.md` 참조. 이 결과는 `docs/RESULTS_PROVENANCE.md`가 "확인됨"으로 판정한 원고 Table 11 GA-only 값(최고수위 5.783m, 비침수)과 정면으로 충돌하며, 해당 문서에도 교차 기록함.
- 표시만 하고 수정하지 않음(리팩터링·결과 조정 금지 원칙). E3에서 GA-only/PSO-only를 현재 코드·동일 가중치로 재실행하면 원고와 판이한(침수 위주) 결과가 나올 가능성이 높다 — E4(완전열거도 같은 이유로 액션 0에 고착될 것으로 예상)·E9(보상-목적 정합성 위반의 극단 사례)에 직접적인 근거가 된다.

## 확인 불가 항목 1 — 강우 시나리오 파일명의 quartile(분위) 인덱스

`data_generate.py`의 `write_datafiles()`(L282-388)는 `(year, duration)` 조합마다 4개 raintype(Huff 1~4분위)을 순회하며, raintype별로 기본 1개 + `more_rains(n_vars=...)`로 만든 추가 샘플을 `h{n:03}` 형식의 전역 증가 카운터로 저장한다. `n`은 연도 루프에서만 초기화되고(L324 `n = 0`) duration 루프 안에서는 리셋되지 않으므로, `(year, duration)` 블록 크기가 raintype 4개 전부 동일하기만 하면 `n % block_size`로 raintype(=quartile) 위치를 역산할 수 있는 구조다.

**실측 확인**: `data/gasan/10year/`에서 `10yr_0010m_h000~h123`(124개), `10yr_0060m_h124~h247`(124개), `10yr_0120m_h248~h371`(124개)로 duration마다 정확히 124개씩 연속 배정됨을 파일명으로 확인. 124 = 4 raintype × 31이면 `quartile = ((n % 124) // 31) + 1`이 성립한다.

**확정 실패 — 코드로 재현 불가**: 124/4=31을 만들려면 `n_vars=30`(기본 1개 + more 30개)이어야 하는데, `write_datafiles`의 기본값은 `n_vars=29`(→ raintype당 30개, 블록 120)이고, 이 함수의 유일한 호출부(`__main__` L490-491)는 주석 처리되어 있으며 `n_vars`를 명시적으로 넘기지 않는다. 즉 실제 디스크의 파일을 만든 실행 인자가 현재 코드에 남아 있지 않다 — 코드 기본값(120/블록)과 실측 파일 수(124/블록)가 불일치한다.

**판정**: `n % 124 // 31 + 1` 공식은 실측 파일 카운트와는 정합하지만, 그 카운트를 만든 실제 생성 파라미터를 코드에서 확정할 수 없으므로 "확정"으로 표시하지 않는다. 임의로 채택하지 않고, `test_metrics.csv`의 `quartile` 컬럼은 당분간 빈 값으로 둔다(2026-08-25, 사용자 지시: "잘못된 값보다 빈 값이 낫다"). E3의 paired 검정 층화는 `return_period × duration_min`만으로 충분하며, quartile은 이 공식이 별도로 확정된 뒤 파일명에서 재계산 가능하다.
