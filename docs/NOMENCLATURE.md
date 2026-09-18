# 기호표 (Nomenclature) — 원고 Eq.(1)~(16) 대 코드 대조

**대응**: R3-4 (E13). 원고 `docs/water-4498965.docx`의 식(1)~(16) 색인은
`docs/manuscript_outline.md` §4에서 이미 추출됨 — 이 문서는 그 각 기호를
실제 코드 변수·파일:줄번호에 대응시킨다. 값 자체(가중치·γ 등)의 근거는
`docs/E02_OBJECTIVE_AUDIT.md`·`docs/E02_ARGMINMAX.md`·
`docs/E06_WEIGHT_SENSITIVITY.md`를 인용하며 재조사하지 않는다.

## 1. 정책·목적함수 (Eq. 1, 9~11)

| 기호 | 원고 의미 | 코드 대응 | 위치 |
|---|---|---|---|
| τ(aₜ, xₜ) | 휴리스틱(GA/PSO) 탐색 목적함수 | `objective_function()` 반환값 | `genetic_algo.py:71,104` |
| π*_T | Σ τ 최소화 정책(Eq.1의 argmin 프레이밍) | 코드에는 명시적 변수 없음 — 실제 행동선택은 항상 argmax(τ·R 둘 다 최대화 방향으로 구현, `docs/E02_ARGMINMAX.md`) | — |
| a* (DDQN, Eq.9) | argmax_a Qθ(x_{t+1}, a) | `acts_ = torch.argmax(Q1_next, dim=1, keepdim=True)` | `dqn_from_demon_v1.py:385` |
| yₜ (Eq.10) | rₜ + γ(1−doneₜ)Qθ⁻(x_{t+1}, a*) | `Q2_gru, _ = target_model(...)`; `tqvals = Q2_gru.gather(...)` | `dqn_from_demon_v1.py:388-389` |
| L(θ) (Eq.11) | TD loss = [Qθ(xₜ,aₜ) − yₜ]² | `loss = loss_fn(X, Y)` (`torch.nn.MSELoss()`) | `dqn_from_demon_v1.py:288,394` |
| **불일치 (E2 확정 사실)** | Eq.1·Algorithm 3 22행은 argmin, Eq.9는 argmax | 코드는 어디서도 argmin으로 행동을 고르지 않음(`argmin`은 `pso.py`의 좌표투영 Eq.8 한 곳뿐, 무관) — Algorithm 3 22행 오타로 확정, 코드 수정 없음 | `docs/E02_ARGMINMAX.md` |

## 2. 시나리오 수준 4개 목적 (Eq. 2~6)

| 기호 | 원고 의미 | 코드 대응 | 위치 |
|---|---|---|---|
| H(π) | 시나리오 내 최고 수위 | `highest` (스텝별 `state[-1]` 최댓값) → `max_level_m` 컬럼 | `perform_evaluate.py:371-375,391` |
| F(π) | 전체 펌프 on/off 전환 횟수 합 | `count_changes(actions, act_type)` → `n_switches` | `perform_evaluate.py:170,361` |
| U(π) | 정규화 펌프사용량 누적(pump-use proxy) | `count_pumps(actions, act_type)` → `n_intervals_100`/`n_intervals_170` (100/170 ㎥ 펌프 가동 인터벌 수) | `perform_evaluate.py:196,365-368` |
| dₜ (Eq.5) | dry-running 위험 이진 지표, Q(aₜ)·Δt > V_avail 시 1 | `over_pump`(`WaterGym.step()`, 이진) / `excess_pump`(`objective_function()`, 이진) — 두 구현이 판정 로직상 동일(`docs/E02_OBJECTIVE_AUDIT.md` §3) | `water_gym.py:162-165`, `genetic_algo.py:76` |
| D(π) (Eq.6) | dₜ 누적 합 | `n_dryrun_proxy = np.sum(ainfos[:, -1])` | `perform_evaluate.py:383-384` |

**주의(기존 확정 사실 인용)**: `over_pump`의 `reward()` docstring은 "횟수"처럼
서술하나 실제로는 0/1 이진 플래그다(`docs/E02_OBJECTIVE_AUDIT.md` 77행) —
docstring 부정확, 코드 동작은 명확.

## 3. Demonstration·PSO 투영 (Eq. 7~8)

| 기호 | 원고 의미 | 코드 대응 | 위치 |
|---|---|---|---|
| 𝒟_demo (Eq.7) | GA가 생성한 demonstration 궤적 집합(Algorithm 3 Phase 1) | **코드에 존재하지 않음** — `pre_train()`/`demonstraion_replay()`는 저장소 어디서도 호출되지 않는 미사용·미완성 함수(`demon_inps` 미정의 변수 참조로 호출 시 `NameError`) | `docs/E00_ALGORITHM3_MISMATCH.md` |
| Π_A(zᵢ) (Eq.8) | PSO 연속 위치를 최근접 실행가능 조합으로 투영(argmin, 유클리드 거리) | `_nearest_valid_action(position)` | `pso.py:23` 부근 |

## 4. 보상함수 (Eq. 12~16)

| 기호 | 원고 의미 | 코드 대응 | 위치 |
|---|---|---|---|
| Rₜ (Eq.12) | w1·R_level + w2·R_act + w3·R_use + w4·R_dry | `t_reward = w1*vol_reward + w2*act_reward + w3*energy_reward + w4*over_pump*excess_pump_penalty` (DDQN 보상, 행동 실행 **후** 상태 기준) / `objective_function()`의 `reward` (휴리스틱 τ, **현재** 관측 상태 기준 — Rₜ의 1-스텝 근사, 동일 값 아님) | `water_gym.py:72`(R), `genetic_algo.py:40-41`(τ) |
| R_level (Eq.13) | (Q(aₜ)/Qmax)·(hₜ/hmax) | `vol_reward` — **명목식과 다름**: 실제는 `min(Q(aₜ)·Δt, V_avail)/Q_max` (capping+Δt, 범위 [0,~2]), R은 `next_level`, τ는 `state[-1]`(현재 수위) 기준(시점 불일치, `docs/E02_OBJECTIVE_AUDIT.md` §0) | `water_gym.py:66`(R), `genetic_algo.py:33`(τ) |
| R_act (Eq.14) | (1/n)Σ I[uᵢ(t)=uᵢ(t−1)] | `act_reward = switching_stability(previous, current)` | `water_gym.py:48,69` |
| R_use (Eq.15) | 1 − Q(aₜ)/Qmax (pump-use proxy, "에너지"가 아님) | `energy_reward` | `water_gym.py:70` |
| R_dry (Eq.16) | −1(dry-run 시) 또는 0 | `w4 * over_pump * excess_pump_penalty`(-1.0) | `water_gym.py:72`, `genetic_algo.py:41` |
| Qmax | 정규화 상수 | `sum(pumpq) = 640` (=100+100+100+170+170) | `water_gym.py:18` |
| hmax | 정규화 상수 | `Level[-1] = 10` | `water_gym.py:20` |
| w1, w2, w3, w4 | 4개 보상항 가중치, `Σwᵢ=1` | `WaterGym.__init__(weights=...)` → `self.w`; `GeneticAlgorithm.__init__(**params)` → `self.weights`, `self.w1..w4` | `water_gym.py:65,241-244`; `genetic_algo.py:45-46` |
| γ | 할인율 | `dqn_from_demon_v1.py` 학습 루프의 `gamma` 인자(`run_meta.json`의 `hyperparams.gamma`) | `results/summary/E13_repro_table.md` |

## 5. GA/PSO 하이퍼파라미터 (Algorithm 3, R2-6 요구)

값은 실행별로 `run_meta.json`에 기록되며 전체는
`results/summary/E13_repro_table.md`(`scripts/make_repro_table.py`로 자동
생성) 참조. 아래는 **run_meta.json에 기록되지 않는(하드코딩된) 정적 값**만
별도 표기한다.

| 파라미터 | 값 | 위치 |
|---|---|---|
| GA 선택 방식 | 토너먼트(크기 3, 비복원) | `genetic_algo.py:select_parents()` |
| GA 조기종료 patience | 10세대 | `genetic_algo.py:genetic_algorithm()` |
| GA 조기종료 최소개선 ε | 1e-6 | 동일 위치 |
| PSO v_max | **미구현** — 속도 클램프 없음 | `pso.py` 생성자에 해당 파라미터 자체가 없음 |
| PSO 조기종료 patience | 10회(기본값, `params.get("patience", 10)`) | `pso.py` |
| PSO 조기종료 최소개선 ε | 1e-6(기본값) | `pso.py` |

## 6. 원고 기호 색인 원본

`docs/manuscript_outline.md` §4 (Eq. 1–16 표, pandoc 추출 원문 기반, 추정 없음).
