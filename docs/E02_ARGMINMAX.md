# E2 — argmin/argmax 및 목적함수-보상 관계 검증

**작성일**: 2026-08-25

## 배경 — 확인해야 했던 불일치

- Algorithm 3, 22행: `a* ← argmin_a Qθ(x_{t+1}, a)`
- 식(9): `a* = argmax_{a∈A} Qθ(x_{t+1}, a)`
- 식(1)은 τ 최소화, 식(12)는 보상 R 최대화

## 1. 코드는 일관되게 argmax — argmin은 어디에도 없음

**실제 논문 결과를 생성한 진입점**: `src/perform_evaluate.py`가 `dqn_gru_regular_w...`, `dqn_guided_GA_w...`, `dqn_guided_PSO_w...` 모델 경로/파일명을 그대로 하드코딩하고 있어 이 파일이 결과 CSV의 출처임이 확인됨 (`perform_evaluate.py:993, 1009, 1055`). 이 스크립트는 `src/dqn_from_demon_v1.py`를 사용한다 (`perform_evaluate.py:23`).

`dqn_from_demon_v1.py`의 행동선택/타깃 계산 지점 전부 argmax:
- `dqn_from_demon_v1.py:916` — `test_model()` 평가 시 행동선택: `action = np.argmax(q_val_)`
- `dqn_from_demon_v1.py:771` — 학습 루프 내 행동선택: `action = np.argmax(q_val_)`
- `dqn_from_demon_v1.py:368, 612, 798` — Double DQN 타깃(온라인망으로 다음 행동 선택): `torch.argmax(Q1_next, dim=1, keepdim=True)  # a* = argmax_a Q_online(s',a)`

저장소 내 다른 DQN 변종(`dqn_gru_torch.py`, `dqn_gru_torch_v2.py`, `dqn_gru_torch_v3.py`, `dqn_lstm_torch.py`, `dqn_dnn_torch.py`, `dqn_cnn_torch.py`, `hybrid_torch.py`, `hybrid_dqn_gru_genetic.py`, `dqn_from_demon.py`)도 전수 조사 결과 `argmax`만 사용. `argmin`은 저장소 전체에서 `src/pso.py:28`(입자 위치를 가장 가까운 유효 행동에 스냅하는 거리 계산용, DQN 행동선택과 무관) 단 한 곳뿐.

**결론**: 실행되는 코드에 argmin으로 행동을 선택하는 경로는 존재하지 않는다. **Algorithm 3의 argmin은 오타로 확정.** 식(9)의 argmax가 코드와 일치. 수정 불필요(코드), 원고 Algorithm 3만 정정.

## 2. τ와 R의 관계 — 계획서 가정과 다른 결과

> **2026-08-30 갱신 (아래 「§4. τ 정의 완화」 참조)**: 아래 "τ ≡ R" 서술은 이후 조사(`docs/MANUSCRIPT_FIXES.md` MF-5)로 **완화됨**. τ는 R의 **근사**이지 동일 값이 아니다 — (a) MF-5a로 τ는 현재 수위(`state[-1]`)를, 학습 보상 R(`WaterGym.reward()`)은 행동 후 수위를 쓰고, (b) 두 식 모두 R_level의 용량 항이 원고 식(13)과 다르다(MF-5b, capping + ×Δt). 이 절의 원 서술은 기록으로 남기되, 확정 서술은 §4를 따른다.

GA/PSO가 공유하는 목적함수: `src/genetic_algo.py` `objective_function()`. 원 docstring은 *"Predict the reward that WaterGym will return for the next step."* 였으나 (MF-5a 이후) 예측 대상과 상태 시점이 갈렸다.

- `genetic_algo.py` (GA), `pso.py`(PSO) 모두 이 값을 **argmax로 최대화**한다 (`genetic_algo.py:172,219,223`, `pso.py:46,71,72`, best_fitness는 `-inf`에서 시작해 개선분이 있을 때만 갱신 — 최대화 탐색 패턴).
- 항 구성이 `WaterGym.reward()`(water_gym.py)와 동일한 형태(vol_reward, act_reward=switching_stability, energy_reward, 가중합) — 상세 항별 대조는 `docs/E02_OBJECTIVE_AUDIT.md` 참조.

**결론 (2026-08-25, 이후 §4로 완화)**: 코드상 τ와 R은 같은 항 구성·같은 방향(**둘 다 최대화**)이다. 계획서 E2가 제안한 "τ ≡ −R (τ 최소화·R 최대화로 통일)"은 실제 코드와 맞지 않는다 — 코드에는 최소화되는 목적함수가 존재하지 않는다. 별도로 `src/cost_model_v2.py`에 물리 기반 dry-run/wear/repair/downtime 비용 모델이 정의되어 있으나, GA/PSO/DQN 학습·평가 경로 어디에서도 import되지 않는 **미사용 코드**로 확인됨 (삭제하지 않고 표시만 함).

## 완료 조건 대조

- [x] 구현이 argmin인지 argmax인지 확정, 근거 기록 — **argmax로 확정**
- [x] τ의 실제 정의(코드 근거) 파악 — R과 같은 항 구성·같은 방향(최대화). 단 §4로 "동일 값" 아님 확정.
- [x] τ의 전체 정규화 항 문서화 — `docs/E02_OBJECTIVE_AUDIT.md`

## 권장 응답서 방향

- Algorithm 3 22행의 argmin → argmax로 정정 (오타 인정, 코드 근거 인용).
- 식(1)의 "τ 최소화" 서술을 **최대화**로 통일. τ ≡ −R 프레이밍은 사용하지 않는다.
- τ와 R의 관계는 아래 §4의 독립 정의를 사용한다 ("1-스텝 예측치" 표현은 쓰지 않는다).

---

## 4. τ 정의 완화 (2026-08-30, 사용자 확정 · `docs/MANUSCRIPT_FIXES.md` MF-5)

MF-5a로 `objective_function`의 수위 항을 `next_level` → `state[-1]`(현재 수위)로 되돌리면서, τ와 학습 보상 R(`WaterGym.reward()`)의 **수위 평가 시점이 갈렸다**. R은 행동 실행 후 상태에서 계산되고, τ는 현재 관측 상태에서 평가된다. "근사"가 아니라 다른 값을 계산하므로 τ를 독립적으로 정의한다.

**확정 서술 (원고 2.2/2.3절 및 Algorithm 3 캡션/주석에 반영):**

> τ는 휴리스틱(GA/PSO)의 탐색 목적함수로, 식(13)~(16)과 동일한 항 구성을 따르되 **현재 관측 상태를 기준으로** 평가한다. DDQN의 보상 R은 **행동 실행 후 상태**에서 계산되므로, τ는 R의 근사이지 동일한 값이 아니다.

이에 따라 §2의 "함수 자체가 보상의 1-스텝 예측치" 및 「결론」의 "τ ≡ R" 서술은 위 독립 정의로 교체한다. `docs/E02_OBJECTIVE_AUDIT.md`도 동일하게 갱신됨.

**부수 사실 (MF-5b, 한계로 명시)**: τ와 R **둘 다** R_level의 용량 항이 원고 식(13)의 `Q(aₜ)/Q_max`와 다르다 — `min(Q(aₜ)·Δt, V_avail)/Q_max`(capping + ×Δt, V_min 미반영). 코드·식 모두 이번 라운드에 정정하지 않고 원고에 한계 문장으로 명시한다. 실효 가중치 영향은 `docs/E06_WEIGHT_SENSITIVITY.md` §8.
