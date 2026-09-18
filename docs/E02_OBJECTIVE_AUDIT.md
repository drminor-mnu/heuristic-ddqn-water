# E2 부속 — objective_function() vs WaterGym.reward() 항별 대조

**작성일**: 2026-08-25 · **갱신**: 2026-08-30 (아래 「0. 2026-08-30 갱신」)
**방법**: 코드 직접 대조. 추정 없음. 논문 식(13)~(16) 원문은 `docs/water-4498965.docx` 문단 35~74에서 추출해 인용.

---

## 0. 2026-08-30 갱신 — τ ≠ R (이후 조사 반영, `docs/MANUSCRIPT_FIXES.md` MF-5)

이 문서의 원래 결론(§5: "두 함수는 동일 입력에 대해 **수치적으로 동일한 결과**를 낸다", §6: 코드가 식(13)의 `Q(aₜ)/Qmax × hₜ/hmax`를 구현)은 **더 이상 성립하지 않는다**. 아래 원문은 2026-08-25 감사 기록으로 남기되, 확정 사실은 이 절을 따른다.

1. **수위 항 시점 (MF-5a, 커밋 bdcf4d7)**: `objective_function`의 `vol_reward`가 `next_level`(행동 후 수위) → `state[-1]`(**현재 수위**)로 정정됨. `WaterGym.reward()`는 여전히 `reservior_level[clock]`(**행동 후 수위**)를 쓴다. → τ와 R의 수위 평가 시점이 다르다. §5의 "수치적으로 동일" 서술 폐기.
2. **용량 항 (MF-5b, 미정정·한계 명시)**: τ·R **둘 다** R_level 1항이 `min(Q(aₜ)·Δt, V_avail)/Q_max` — 원고 식(13)의 명목 `Q(aₜ)/Q_max`와 다르다(capping + ×Δt로 범위 [0,~2], V_min 미반영). §6 표 (13) 행의 "코드가 식(13)을 구현"은 부정확.
3. **확정 τ 정의** (원고·`docs/E02_ARGMINMAX.md` §4와 동일):
   > τ는 휴리스틱(GA/PSO)의 탐색 목적함수로, 식(13)~(16)과 동일한 **항 구성**을 따르되 **현재 관측 상태를 기준으로** 평가한다. DDQN의 보상 R은 **행동 실행 후 상태**에서 계산되므로, τ는 R의 근사이지 동일한 값이 아니다.
4. 실효 가중치 영향(R_level 범위 [0,~2] → 명목 w1의 실효 배율): `docs/E06_WEIGHT_SENSITIVITY.md` §8.

---

## 1. `genetic_algo.py:71` `objective_function()` — 계산 항 전체

```python
# genetic_algo.py:71-106
def objective_function(self, act, state, action_list, next_inflow=0.0):
    pump_rate = np.array(pumpq)[self.Actions[act]].sum()
    attempted_outflow = pump_rate * self.minutes
    available_volume = state[3] + next_inflow
    excess_pump = 1.0 if attempted_outflow > available_volume else 0.0
    actual_outflow = min(attempted_outflow, available_volume)
    next_volume = max(available_volume - attempted_outflow, 0.0)
    next_level = self._volume_to_level(next_volume)

    vol_reward = (actual_outflow/np.array(pumpq).sum()) * (next_level/Level[-1])
    first = action_list[-1]
    second = act
    act_reward = switching_stability(self.Actions[first], self.Actions[second])
    energy_reward = 1.0 - pump_rate / np.array(pumpq).sum()
    excess_pump_penalty = -1.0

    reward = self.w1 * vol_reward + self.w2 * act_reward \
           + self.w3 * energy_reward + self.w4 * excess_pump * excess_pump_penalty
    return reward
```

5개 항: `vol_reward`, `act_reward`(switching_stability, `water_gym.py`에서 직접 import — 중복 정의 아님), `energy_reward`, `excess_pump`(dry-run 지시자), 이 넷의 가중합.

## 2. `water_gym.py:217` `WaterGym.reward()` + `step()`의 dry-run 판정 — 계산 항 전체

```python
# water_gym.py:160-169 (step() 내 dry-run 판정)
cur_outflow = np.array(pumpq)[self.Actions[action]].sum() * self.minutes
cur_vol = self.reservior_vol[self.clock-1] + cur_inflow - cur_outflow
over_pump = 0
if cur_vol < 0.0:
    cur_vol = 0.0
    over_pump = 1
    cur_outflow = self.reservior_vol[self.clock-1] + cur_inflow
    self.overpumping.append(1)
else:
    self.overpumping.append(0)
```

```python
# water_gym.py:241-267
w1, w2, w3, w4 = self.w
vol_reward = (self.outflow[clock]/np.array(pumpq).sum()) * (self.reservior_level[clock]/Level[-1])
first = self.Actions[self.acts[clock]]
second = self.Actions[self.acts[clock-1]]
act_reward = switching_stability(second, first)
energy_reward = 1.0 - np.array(pumpq)[self.Actions[self.acts[clock]]].sum() / np.array(pumpq).sum()
excess_pump_penalty = -1.0
t_reward = w1*vol_reward + w2*act_reward + w3*energy_reward + w4*over_pump*excess_pump_penalty
```

동일한 5개 항 구조: `vol_reward`, `act_reward`, `energy_reward`, `over_pump`(dry-run 지시자), 가중합.

**주의**: `over_pump`의 docstring(`reward()` 함수 L231)은 "The number of times dry running occurs"라고 되어 있어 카운트처럼 보이지만, 실제 `step()` 코드(L162-165)는 `over_pump`를 **0 또는 1의 이진 플래그**로만 설정한다(`if cur_vol < 0.0: over_pump = 1`, 그 외 경로 없음). 카운트가 아니라 이진값 — docstring이 부정확. `objective_function()`의 `excess_pump`(역시 이진)과 **카디널리티 동일**.

## 3. 핵심 질문 — dry-running 항이 objective_function()에 포함되는가

**포함된다.** `excess_pump` 변수(`genetic_algo.py:76`)가 그것이며, `w4 * excess_pump * excess_pump_penalty(-1.0)` 항으로 최종 reward에 가중 합산된다(`genetic_algo.py:104`). `WaterGym.step()`의 `over_pump`(L162-165)와 판정 로직이 구조적으로 동일:

| | 조건 | 반환 |
|---|---|---|
| GA `excess_pump` | `attempted_outflow > available_volume` (= `state[3] + next_inflow`) | 1.0 / 0.0 |
| WaterGym `over_pump` | `reservior_vol[clock-1] + cur_inflow - cur_outflow < 0` | 1 / 0 |

`available_volume`(GA) = `state[3] + next_inflow`이고 `state[3]`은 호출부에서 `reservior_vol[clock-1]`에 해당하는 값이 전달되므로(아래 5번 참조) 두 조건은 동일한 물리량을 비교한다.

**논문 식(16)과의 변수명 차이 — 저자 확인, 코드 결함 아님**: 식(16)의 정의(문단 39)는 "Vₜ is the detention-basin volume at time t, **Vmin is the minimum volume required to account for the pump suction level and dead storage**"이며, 코드에는 `Vmin`이라는 이름의 변수가 별도로 존재하지 않는다(`water_gym.py`, `genetic_algo.py`, `cost_model_v2.py`, `pso.py` 전체 검색 결과 없음). 저자가 직접 검토한 결과, **이는 논문 표기와 코드 변수명 사이의 차이일 뿐 판정 로직 자체에는 문제가 없음을 확인함**. `Volume`/`Level` 좌표계의 원점(`Volume[0]=0.0` ↔ `Level[0]=4.7m`)이 이미 펌프 흡입 한계·사수량을 제외한 기준으로 설정되어 있어, 코드에서 `cur_vol`/`available_volume`을 0과 비교하는 것이 곧 식(16)의 `max(Vₜ−Vmin,0)`과 동일한 판정이 된다는 것이 저자의 확인 내용이다. **재실행 불필요, 응답서에는 "구현 변수명이 Vmin을 명시하지 않지만 좌표계 원점에 반영되어 있다"는 설명만 추가하면 됨.**

## 4. 가중치 w1~w4 전달 경로

- `WaterGym.__init__(weights=[1.,1.,1.,0.])` → `self.w = weights`, `reward()`에서 `w1,w2,w3,w4 = self.w` (water_gym.py:65, 241-244)
- `GeneticAlgorithm.__init__(**params)` → `self.weights = params.get('weights', [1,1,1,0])`, `self.w1,w2,w3,w4 = self.weights` (genetic_algo.py:45-46)
- 실제 호출부(`perform_evaluate.py`, `dqn_from_demon_v1.py`)에서 두 클래스 모두 **동일한 weights 리스트**를 params로 전달(파일명 `w0_25_w0_4_w0_25_w0_1` = [0.25, 0.40, 0.25, 0.10]과 일치).
- **w4는 죽은 코드가 아니다.** 두 함수 모두에서 dry-run/excess_pump 항에 곱해져 최종 reward/objective 값에 실제로 반영된다(0.10이라는 값 자체가 `results/dqn_gru_regular_w0_25_..._acttype0.csv` 등 실제 사용된 결과 파일명에 나타남).

## 5. 두 함수는 동일한 입력에 대해 동일한 값을 내는가

**예 — 구조적으로 동일한 값을 낸다**, 단 두 가지 조건에서:
- 호출부 확인(`dqn_from_demon_v1.py:330-332, 558`): GA에 전달되는 `next_inflow`는 추정치가 아니라 **실제 다음 시점의 참값** `gym.outfalls[gym.clock + 1]`이다. 즉 GA/PSO는 "미래를 모른 채 추정"하는 게 아니라, 환경이 이미 알고 있는 다음 스텝의 실제 유입량을 그대로 받아 1-스텝을 정확히 재현한다.
- `_volume_to_level()`(genetic_algo.py:53-69)과 `vol2level()`(water_gym.py:192-214)은 **독립적으로 중복 작성된 코드**이지만, 라인 단위로 대조한 결과 분기 조건과 선형보간 수식이 완전히 동일하다(E0-2 중복 구현 인벤토리 항목으로 별도 기록 필요 — 현재는 값이 일치하지만 한쪽만 수정되면 어긋날 위험이 있는 구조).
- 따라서 `state[3]`(직전 시점 실제 reservoir volume)과 `next_inflow`(다음 시점 실제 inflow)가 `WaterGym.step()`이 실제로 사용한 값과 같다면, 같은 행동에 대해 `objective_function()`과 `reward()`는 **수치적으로 동일한 결과**를 낸다. 이것이 GA/PSO docstring의 "Predict the reward that WaterGym will return"이 문자 그대로 정확함을 뒷받침한다.

## 6. 논문 식(13)~(16) ↔ 코드 변수 대응표

| 논문 식 | 논문 정의 (원문 요약) | 코드 변수 | 정규화 항의 코드 값 |
|---|---|---|---|
| (13) 수위 보상 | `Q(aₜ)/Qmax × hₜ/hmax` | `vol_reward` | `Qmax = sum(pumpq) = 640`(=100+100+100+170+170), `hmax = Level[-1] = 10` (water_gym.py:18,20) |
| (14) 액션일관성 보상 | 이전/현재 펌프 on-off 상태 일치 비율 | `act_reward` = `switching_stability()` | 정규화 항 없음(0~1 비율 자체가 결과) — `previous.size`(5, 펌프 수)로 나눔 |
| (15) 펌프사용 보상 | `1 − Q(aₜ)/Qmax` | `energy_reward` | 동일 `Qmax = 640` |
| (16) dry-running 패널티 | `Q(aₜ)·Δt`와 `V_avail = max(Vₜ−Vmin,0)` 비교, 이진 이벤트 | `excess_pump`/`over_pump` | `Δt = self.minutes`; **Vmin에 해당하는 코드 값 미발견(위 3번 참조)** |

## 완료 조건 대조
- [x] τ의 전체 정규화 항(Qmax=640, hmax=10, Δt=minutes) 문서화
- [x] 식(16)의 Vmin 항 — 저자 확인 완료, 좌표계 원점에 반영된 변수명 차이일 뿐 코드 결함 아님
