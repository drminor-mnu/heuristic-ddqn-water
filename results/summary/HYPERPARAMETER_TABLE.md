# 하이퍼파라미터 통합 표 (원고 부록용)

`docs/response/NEW_CONTENT.md` §7에서 확인한 대로 통합 표가 저장소에
없어 신규 작성한다. 값은 전부 코드 상수 또는 `run_meta.json`에서
그대로 추출했다(추정 없음). 각 값 옆에 출처를 표기한다. 해석·옹호
없이 사실만 기록한다.

## 1. DDQN 공통

| 항목 | 값 | 출처 |
|---|---|---|
| 학습률(lr) | 1e-3(0.001) | `src/dqn_from_demon_v1.py:766`(Regular `train()`), `run_meta.json`(`hyperparams.lr`, 전 조건 공통) |
| 배치 크기(batch_size) | 20 | `src/dqn_from_demon_v1.py:772`(Regular), `run_e03_seeds.py`의 `GUIDED_EXTRA['batch_size']`(GA/PSO-guided도 20) |
| 리플레이 버퍼 크기(MemSize) | 3,000 | `src/dqn_from_demon_v1.py:64` `MemSize = 3000 # the size of replay buffer` |
| 최소 샘플링 시작 크기(리플레이 워밍업) | **Regular: `len(replay) > batch_size`(=20, 하드코딩) / GA·PSO-guided: `len(replay) >= min_replay_size`(=100, 기본값 batch_size×5)** — 2026-09-03 코드 직접 확인(아래 §11 상세) | `src/dqn_from_demon_v1.py:826`(Regular `train()`), `:468,617`(guided `_train_guided_by_optimizer()`) |
| 타깃망 동기화 주기(target_update_C) | Regular: 10 / GA-guided·PSO-guided: 200 | `src/dqn_from_demon_v1.py:775`(Regular `sync_freq=10`), `run_e03_seeds.py` `GUIDED_EXTRA['sync_freq']=200` |
| 할인계수(γ) | 0.7(전 조건 공통, E6에서 채택 — 코드 기본값은 0.1이었으나 미사용) | `run_meta.json`(`hyperparams.gamma`, 전 조건), `docs/E06_WEIGHT_SENSITIVITY.md` |
| 옵티마이저 | Adam | `src/dqn_from_demon_v1.py:766` `torch.optim.Adam(...)` |
| 손실함수 | MSELoss | `src/dqn_from_demon_v1.py:764` `torch.nn.MSELoss()` |
| ε 스케줄(Regular) | 시작 1.0 → 에피소드마다 `epsilon -= 1/num_samples`, 하한 0.2에서 감쇠 정지(선형에 가까운 계단감쇠) | `src/dqn_from_demon_v1.py:771,862-864`(`if epsilon > 0.2: epsilon -= 1./num_samples`) |
| ε 스케줄(GA-guided·PSO-guided) | 시작 0.3 → 종료 0.05, 선형감쇠(`eps_start-(eps_start-eps_end)*progress`) | `src/dqn_from_demon_v1.py:526`, `run_e03_seeds.py` `GUIDED_EXTRA['eps_start']=0.3,['eps_end']=0.05` |
| ε 적용 대상 | 휴리스틱 미사용(1-ga_prob) 분기에서만 ε-greedy 적용 — GA/PSO 제안 시엔 ε 미적용 | `src/dqn_from_demon_v1.py:577-579` |
| 에피소드 수(전체데이터) | 3,996(=train_n, epochs=1) | `run_meta.json`(`data.train_n`), 전 조건 `train_log.csv` 행수 3,996 확인(E3v2/E14/E15 공통) |
| 에피소드 수(한정데이터, E12) | 160 | `docs/E12_REPORT.md`(train_n_condition=160) — 원고 인용 "162"와의 불일치는 `MANUSCRIPT_FIXES.md`(MF-11-a) 별도 기록, 여기서는 재론하지 않음 |

## 2. GRU (Q-네트워크)

| 항목 | 값 | 출처 |
|---|---|---|
| 입력 차원(input_dim) | 5 | `src/dqn_from_demon_v1.py:281,483,759,933` `DqnGRU(input_dim=5, ...)` |
| 은닉 차원(hidden_dim) | 32 | `src/dqn_from_demon_v1.py:46` `hidden_dim = 32` |
| 층 수(num_layers) | 3 | `src/dqn_from_demon_v1.py:45` `num_layers = 3` |
| 시퀀스 길이(nstates) | 5(직전 4스텝 + 현재 1스텝) | `src/dqn_from_demon_v1.py:44` `nstates = 5` |

## 3. GA (유전 알고리즘, 시연 생성용)

| 항목 | 값 | 출처 |
|---|---|---|
| 개체수(population_size) | 100 | `run_meta.json`(`heuristic_params.population_size`, GA-guided·GA-only 공통) |
| 세대수(num_generations) | 100 | `run_meta.json`(`heuristic_params.num_generations`) |
| 교차확률(crossover_rate) | 0.8 | `run_meta.json`(`heuristic_params.crossover_rate`) |
| 돌연변이확률(mutation_rate) | 0.01 | `run_meta.json`(`heuristic_params.mutation_rate`) |
| 선택 방식 | 토너먼트 선택(tournament size=3) | `src/genetic_algo.py:156-180` `select_parents()`, `tournament_indices = np.random.choice(len(population), size=3, ...)` |

## 4. PSO (입자 군집 최적화, 시연 생성용)

| 항목 | 값 | 출처 |
|---|---|---|
| 입자수(swarm_size) | 30 | `run_meta.json`(`heuristic_params.swarm_size`, PSO-guided·PSO-only 공통) |
| 반복수(num_iterations) | 100 | `run_meta.json`(`heuristic_params.num_iterations`) |
| 관성계수(inertia) | 0.7 | `run_meta.json`(`heuristic_params.inertia`) |
| 인지계수(cognitive) | 1.5 | `run_meta.json`(`heuristic_params.cognitive`) |
| 사회계수(social) | 1.5 | `run_meta.json`(`heuristic_params.social`) |
| 조기종료 인내(patience) | 10 | `run_meta.json`(`heuristic_params.patience`) |
| 최소개선폭(min_improvement) | 1e-6 | `run_meta.json`(`heuristic_params.min_improvement`) |

## 5. 휴리스틱 가이던스(GA/PSO-guided 전용)

| 항목 | 값 | 출처 |
|---|---|---|
| ga_prob 시작값(ga_start) | 0.6 | `run_e03_seeds.py` `GUIDED_EXTRA['ga_start']=0.6`, `run_meta.json`(`hyperparams.ga_start`) |
| ga_prob 종료값(ga_end) | 0.0 | `run_e03_seeds.py` `GUIDED_EXTRA['ga_end']=0.0` |
| 감쇠 형태 | 선형(`ga_prob = ga_start-(ga_start-ga_end)*progress`, progress=i/(num_samples-1), 0→1) | `src/dqn_from_demon_v1.py:525` |
| 가이던스 적용 방식 | 매 결정 스텝마다 확률 ga_prob로 GA/PSO가 행동 제안, 그 외에는 DDQN ε-greedy — 별도 사전 채움(preload) 없음 | `src/dqn_from_demon_v1.py:568-580`, `results/summary/DELTA_SIGN_FIX.md`와 무관, `WARMSTART_OCCURRENCES.md` §0 |

## 6. 보상 가중치 (w1: 수위, w2: 스위칭, w3: 펌프사용, w4: dry-running)

| 모델 | w1 | w2 | w3 | w4 | 비고 |
|---|---|---|---|---|---|
| Regular DDQN(2기준, §3.1) | 0.50 | 0.50 | 0 | 0 | E14, `results/E14_two_criteria/` |
| Regular DDQN(4기준, §3.3, E3v2) | 0.40 | 0.25 | 0.25 | 0.10 | 신가중치(채택값) |
| GA-guided DDQN(§3.1, 구가중치) | 0.25 | 0.40 | 0.25 | 0.10 | 원고 원본(재현 불가, 코드로 재현 시 신가중치만 가능) |
| GA-guided/PSO-guided/GA-only/PSO-only(§3.3, E3v2) | 0.40 | 0.25 | 0.25 | 0.10 | 신가중치(채택값), E6 결과 |

**신가중치 채택 근거**: 구가중치(w2>w1)는 1-스텝 목적함수가 첫 펌프
기동을 정당화하지 못해 액션 0 고착·월류를 유발(`docs/E00_HEURISTIC_MYOPIA_FINDING.md`,
`results/E04_enum/HORIZON_ESCAPE.md`) — E6 가중치 스윕(29조합) 결과
채택값이 비지배해집합에 포함됨을 확인(`docs/E06_WEIGHT_SENSITIVITY.md`).

## 7. 데이터

| 항목 | 값 | 출처 |
|---|---|---|
| 전체 시나리오 수 | 6,696 | 원고 인용값, `MANUSCRIPT_FIXES.md`(MF-11-a/b 주의사항 별도) |
| 학습 시나리오 수(전체데이터) | 3,996(약 60%) | `run_meta.json`(`data.train_n`, 전 조건 공통) |
| 시험 시나리오 수(전체데이터) | 2,700(약 40%) | `run_meta.json`(`data.test_n`, 전 조건 공통) |
| 학습 시나리오 수(한정데이터, E12) | 160 | `docs/E12_REPORT.md` |
| 분할 시드(split_seed) | 42 | `run_meta.json`(`data.split_seed`), `data/splits/fixed_split_seed42.json` |
| 분할 파일 | `data/splits/fixed_split_seed42.json` | `run_meta.json`(`data.split_file`) |
| 재현기간 | 10, 20, 30, 50, 80, 100년(6종) | `run_e03_seeds.py` `YEARS` |
| 지속시간 | 60, 120, 180, 240, 360, 540, 720, 1080, 1440분(9종) | `run_e03_seeds.py` `DURATIONS` |

## 8. 환경(WaterGym)

| 항목 | 값 | 출처 |
|---|---|---|
| 결정 간격(minutes) | 2분 | `run_e03_seeds.py` `BASE_PARAMS['minutes']=2` |
| 상태 변수(5차원) | 강우량(rains), 유입량/유출부(outfalls), 유출량(outflow), 저류량(reservior_vol), 저류수위(reservior_level) | `src/water_gym.py:105-110` `reset()`의 반환 리스트 |
| 정규화 방식 | min-max 표준화(`_input_norm`, 코드가 `state/(state.sum()+ε)`에서 전환 완료) | `MANUSCRIPT_FIXES.md`(MF-7) |
| 초기 저류수위 | 4.7 m | `src/water_gym.py:97` `self.reservior_level = [4.7]` |
| 월류 임계 | 10.0 m(`Level[-1]`) | `src/perform_evaluate.py:378` 등, `results/summary/E14_overflow_check.md` |
| 펌프 구성 | 100 m³/min ×3대, 170 m³/min ×2대(Q_max=640 m³/min) | `src/perform_evaluate.py:200` `pumpq=(100,100,100,170,170)`, `results/summary/PUMP_USE_INDEX.md` §0 |
| 행동 공간 크기 | 6(act_type=0, `Actions0`) | `src/perform_evaluate.py` `Actions0`, `docs/ACTION_SPACE_JUSTIFICATION.md` |

## 9. 하드웨어·소프트웨어

| 항목 | 값 | 출처 |
|---|---|---|
| CPU | i9-13900KF | 본 저장소 전체 타이밍측정 공통 하드웨어(`docs/E14_REPORT.md`, `results/summary/E10_unified_table.md` 등 반복 명시) |
| GPU | RTX 3080 Ti 12GB | 위와 동일 |
| RAM | 94GB | 위와 동일 |
| Python | 3.9.16(conda-forge, GCC 11.3.0) | `run_meta.json`(`versions.python`, E15 예시) |
| PyTorch | 2.0.1+cu117 | `run_meta.json`(`versions.torch`) |
| 병렬 스레드(프로세스당) | OMP/MKL/OPENBLAS_NUM_THREADS=2, `torch.set_num_threads(2)` | `src/run_e03_seeds.py:242-251`, `src/run_e14_two_criteria.py`, `src/run_e15_timing.py` |
| 병렬 프로세스 수(workers) | 실행마다 다름 — Regular(2기준,E14) 2, GA-guided(4기준,E15) 2, E3v2 전체배치(5모델×5시드) 8 | `results/_log/{E03,E14,E15}.log`, `results/summary/TABLE5_COMPARABILITY.md` §2 |

## 10. 확인 불가 항목

**없음(2026-09-03, §11에서 해소).** 이전 판(2026-09-03 이전)에서
Regular의 min_replay_size를 "확인 불가"로 표기했으나, 소스 코드
직접 확인으로 해소했다 — 상세는 §11.

## 11. min_replay_size 확인 결과 (2026-09-03)

`run_meta.json`에는 GA/PSO-guided의 `hyperparams.min_replay_size`
필드만 기록되고(값 100), Regular 쪽 기록에는 이 필드 자체가 없다 —
이는 **기록 누락이 아니라 Regular의 학습 함수(`train()`)가
`min_replay_size`라는 변수를 아예 쓰지 않기 때문**임을 코드에서
확인했다(아래).

### 11.1 두 경로

| 경로 | 함수 | 게이트 조건 | 값 |
|---|---|---|---|
| Regular DDQN | `train()`(`src/dqn_from_demon_v1.py:727`) | `if len(replay) > batch_size:`(`:826`) | **20**(=batch_size, 하드코딩 — `batch_size = 20` 직접 대입, `:772`) |
| GA-guided/PSO-guided/ENUM-guided | `_train_guided_by_optimizer()`(`:431`, `train_guided_byGA`/`byPSO`/`byENUM`이 전부 이 함수로 위임, `:685-724`) | `if len(replay) >= min_replay_size:`(`:617`) | **100**(=batch_size×5, `min_replay_size = params.get('min_replay_size', batch_size*5)`, `:468` — 인자로 전달 가능하나 `run_e03_seeds.py`의 `GUIDED_EXTRA`에 `min_replay_size` 키가 없어 항상 기본값 100 적용) |

**하드코딩인지 인자인지**: Regular는 `batch_size` 자체가 함수 내부
하드코딩(`batch_size = 20`, 인자로 받지 않음) — 게이트도 이 하드코딩된
값을 직접 재사용. Guided 경로는 `min_replay_size`가 `params.get(...)`
로 받는 **인자**이며 기본값만 batch_size×5 — 즉 이론상 드라이버가
다른 값을 넘길 수 있으나, 실제로 `run_e03_seeds.py`/`run_e14_two_criteria.py`/
`run_e15_timing.py` 어디도 `min_replay_size`를 명시적으로 넘기지
않아 **항상 기본값 100이 적용됐다.**

### 11.2 두 경로가 다른 값을 쓰는 이유가 코드에 드러나는가

**드러나지 않는다.** `min_replay_size = params.get('min_replay_size',
batch_size * 5)`(`:468`) 자체에는 "왜 5배인지"를 설명하는 주석이
없다. Regular의 `len(replay) > batch_size`(`:826`)도 마찬가지로
"왜 min_replay_size 같은 별도 워밍업 상수를 안 쓰는지"에 대한 주석이
없다. **값의 차이(20 vs 100)는 코드로 확인되나, 그 설계 근거는
코드에 기록되어 있지 않다** — "해당 파라미터를 사용하지 않음"(Regular는
애초에 `min_replay_size` 개념이 없음)과 "값을 특정할 수 없음"(둘 다
아님, 둘 다 정확한 값이 존재·확인됨)을 구분하면, **Regular는 전자
(파라미터 자체가 없음, 대신 batch_size를 그대로 게이트로 씀), 값
자체의 불확실성은 없다.**

### 11.3 참고 — 미사용 레거시 경로

`train_guided_byGA_old()`(`:253-431`)라는 별도 함수가 존재하며 자체
`replay`/게이트(`if len(replay) > batch_size:`, `:371`)를 갖지만,
**저장소 전체에서 이 함수를 호출하는 코드가 없다**(`grep -rn
"train_guided_byGA_old" src/*.py` 결과 정의부 1줄만 매치) — `pre_train()`/
`demonstraion_replay()`와 같은 성격의 미사용 코드로 확인된다(MF-2와
동일 패턴, 이번에 신규 확인, MF-2 자체를 확장하지는 않음).
