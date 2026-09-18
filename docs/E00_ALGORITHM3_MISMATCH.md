# Algorithm 3(원고) vs `_train_guided_by_optimizer()`(코드) 대조

**작성일**: 2026-08-26
**결론(先)**: **원고 Algorithm 3이 기술하는 두 단계(Phase 1: 시연 데이터 프리로딩, Phase 2: 감쇠하는 온라인 가이던스) 중 Phase 1은 실제 학습 경로에 존재하지 않는다.** 실제로 실행되는 코드는 **(b) 학습 중 확률적 온라인 가이던스만** 구현하고 있으며, Phase 1에 해당하는 코드(`pre_train()`, `demonstraion_replay()`)는 저장소에 존재하지만 **어디에서도 호출되지 않는 미사용 코드**다. 코드는 수정하지 않았다.

---

## 1. `pre_train()`과 `demonstraion_replay()`의 전체 코드

`src/dqn_from_demon_v1.py:145-233` (원문 그대로, 함수명의 오타 `demonstraion`도 원문 그대로임).

```python
def demonstraion_replay(replay:deque, demo_inps:list, **params):
    """ToU (Time of Use)
    A replay buffer for pre-training is generated from the given demonstration data (inps)
    using a Genetic Algorithm.

    Parameters
    ----------
    replay : deque
        replay buffer
    train_inps : list
        DESCRIPTION.
    n_episodes : int, optional
        DESCRIPTION. The default is 20.
    **params : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    minutes = params.get('minutes', 2)
    n_episodes = len(demo_inps)

    # (state_prev, action, reward, state_post, done)
    for inp in demon_inps[:n_episodes]:
        GA = GeneticAlgorithm(**params)
        actions, states, rewards, infos = GA.run_genetic_algo(inp)
        input_q = deque([], nstates) # for build input vector
        # build initial input
        for k in range(nstates-1):
            input_q.append(np.array([0.0 for _ in range(len(states[0]))]))
        done = False
        for i in range(len(actions)-1):
            state_ = np.array(states[i]) / (np.array(states[i]).sum() + 0.00001) # input normalization
            input_q.append(state_)
            state_prev = torch.tensor(np.stack(input_q), dtype=torch.float32)

            state_ = np.array(states[i+1]) / (np.array(states[i+1]).sum() + 0.00001) # input normalization
            input_q.append(state_)
            state_post = torch.tensor(np.stack(input_q), dtype=torch.float32)

            "genetic algorithm의 경우 state[0] -> action[1] -> state[1] 순서로 진행됨"
            if i >= len(actions)-2:
                done = True
            exp =  (state_prev, actions[i+1], rewards[i+1], state_post, done)
            replay.append(exp)



def pre_train(train_inps, replay, rate:float=0.1, **params):
    """
    A subset of the training data is selected to generate demonstration data,
    which is then used to pre-train the network.

    Parameters
    ----------
    train_inps : TYPE
        DESCRIPTION.
    rate : float, optional
        DESCRIPTION. The default is 0.1.

    Returns
    -------
    None.

    """
    ## Selecting the demonstration dataset corresponing to the rate, r
    train_inps.sort()
    years = list(set([re.findall(r'\d+', f.split('/')[-1])[0] for f in train_inps]))
    durations = list(set([re.findall(r'\d+', f.split('/')[-1])[1] for f in train_inps]))
    demo_inps = []
    for y in years:
        for m in durations:
            data = [f for f in train_inps
                        if re.findall(r'\d+', f.split('/')[-1])[0] == y
                        and re.findall(r'\d+', f.split('/')[-1])[1] == m]
            random.shuffle(data)
            nsample = len(data) # the number of total sample
            n_rate = int(nsample * rate) # the number of training sample

            demo_inps.extend(data[:n_rate])

    random.shuffle(demo_inps)

    demonstraion_replay(replay, demo_inps, **params)

    return demo_inps
```

**두 함수의 실제 동작**:
- `pre_train()`은 `train_inps`를 (year, duration) 층별로 `rate` 비율만큼 뽑아 `demo_inps`를 구성하고(=Algorithm 3의 `D_demo`에 해당), `demonstraion_replay(replay, demo_inps, **params)`를 호출한다.
- `demonstraion_replay()`는 각 `demo_inps` 시나리오마다 `GeneticAlgorithm.run_genetic_algo(inp)`로 **GA 단독(순수 휴리스틱) 전체 에피소드**를 실행하고, 그 궤적의 (state_prev, action, reward, state_post, done) 튜플을 전부 외부에서 넘겨받은 `replay` 버퍼에 `append`한다.
- 즉 이 둘을 합치면 **Algorithm 3 Phase 1("D_demo의 각 시나리오에 대해 HOPT만으로 전체 에피소드를 실행하고 모든 transition을 B에 저장")과 정확히 같은 로직**이다.
- **다만 `pre_train()`은 그 자체로 gradient update를 전혀 수행하지 않는다.** `optimizer`, `loss_fn`, `policy_model` 등이 함수 안에 전혀 없다 — 이름과 달리 "버퍼를 채우기만" 하고 반환한다. 실제 학습(Phase 2)을 하려면 이 `replay`를 **외부에서 학습 함수에 넘겨야 하는데, 그런 학습 함수가 존재하지 않는다**(§4 참조).
- **버그**: `demonstraion_replay()` 171행이 매개변수명 `demo_inps` 대신 정의되지 않은 `demon_inps`를 참조한다(`for inp in demon_inps[:n_episodes]:`). 이 함수를 지금 호출하면 즉시 `NameError`가 난다. 한 번도 실행되지 않았다는 방증이다.

---

## 2. 저장소 전체 호출부 검색 — 미사용 확정

```
grep -rn "pre_train(\|demonstraion_replay(\|demon_inps" --include="*.py" .
```

검색 대상: 저장소 전수(`src/`, `src/obsolete/`, 과거 실험 디렉터리 및 구 결과 디렉터리 포함).

**결과**: 일치하는 줄은 `src/dqn_from_demon_v1.py` 안의 4곳뿐 — 두 함수의 **정의 자체**(145행, 196행), `demonstraion_replay()` 내부의 버그성 참조(171행), 그리고 `pre_train()`이 `demonstraion_replay()`를 호출하는 단 한 줄(231행). **`pre_train()`을 호출하는 코드는 저장소 어디에도 없다.** `.py` 외 파일(`.sh`, `.ipynb`, `.md`)에서도 두 함수명이 전혀 언급되지 않는다.

**판정: 두 함수는 확정적으로 미사용(dead code)이다.**

---

## 3. git 이력 추적 — 불가능, 사실만 기록

```
git log --follow --format="%H %ai %s" -- src/dqn_from_demon_v1.py
→ ca4e68a6bef9e73515190c40fb388a4450d55e24 2026-08-25 14:58:51 +0900 Snapshot: state at resubmission of water-4498965
```

이 저장소는 **2026-08-25의 재투고 시점 스냅샷 커밋이 `src/dqn_from_demon_v1.py`에 대한 유일한 커밋이다.** 그 이전 이력이 전혀 없어(단일 커밋으로 전체 파일이 처음 들어옴), `pre_train()`/`demonstraion_replay()`가 **언제 추가되었는지, 언제부터 호출부가 사라졌는지(애초에 호출된 적이 있었는지조차) git으로는 확인할 수 없다.** 이 사실 자체를 기록하는 것 외에 추가로 확정할 수 있는 것이 없다.

---

## 4. Algorithm 3 원문 vs `_train_guided_by_optimizer()` 줄 단위 대조

원고 Algorithm 3 전문(Table 3, `docs/manuscript_outline.md` 작성 시 pandoc으로 추출):

```
Algorithm 3. Heuristic-Guided Demonstration Replay Generation and DDQN Training

Input: Training rainfall scenarios D, Simulation environment E, Feasible action set A,
Heuristic optimizer HOPT (GA or PSO), Replay buffer B, Policy network Qθ,
Target network Qθ−, Discount factor γ, Batch size M, Target update frequency C,
Initial heuristic-action probability pH
Output: Trained DDQN policy Qθ

Phase 1: Heuristic Demonstration Generation
1: Initialize replay buffer B
2: for each demonstration scenario d ∈ D_demo do
3:   Initialize environment E(d) and observe x0
4:   Initialize previous action history La
5:   while episode is not terminated do
6:     Select heuristic action: at^H ← HOPT(xt, La, A)
7:     Execute at^H and observe rt, xt+1, and done
8:     Store (xt, at^H, rt, xt+1, done) in B
9:     Update La and xt
10:  end while
11: end for

Phase 2: DDQN Training with Decaying Heuristic Guidance
12: Initialize policy network Qθ and target network Qθ− ← Qθ
13: for each training episode e do
14:   Initialize environment and observe xt
15:   while episode is not terminated do
16:     With probability pH(e), select at ← HOPT(xt, La, A)
17:     Otherwise select at using the ε-greedy policy of Qθ
18:     Execute at and observe rt, xt+1, and done
19:     Store transition in B
20:     if |B| ≥ M then
21:       Sample a mini-batch from B
22:       Select a* ← argmin_a Qθ(xt+1, a)
23:       Compute yt ← rt + γ(1−done)·Qθ−(xt+1, a*)
24:       Update Qθ by minimizing [Qθ(xt,at)−yt]²
25:     end if
26:     Every C updates, set Qθ− ← Qθ
27:     Update xt and La
28:   end while
29:   Decrease pH(e) according to the predefined guidance schedule
30: end for
31: return trained policy Qθ
```

| Algorithm 3 단계 | 코드 대응(`_train_guided_by_optimizer`, `dqn_from_demon_v1.py:414-650`) | 판정 |
|---|---|---|
| **Phase 1 전체 (1~11행)** — D_demo의 각 시나리오를 HOPT 단독으로 완주시키며 모든 transition을 B에 사전 저장 | **없음.** 479행 `replay = deque(maxlen=MemSize)`로 매 학습 호출마다 **빈 버퍼**로 시작. `pre_train()`/`demonstraion_replay()` 호출이 전혀 없다(§2). | **불일치 — 코드에 미구현** |
| 12행: Qθ, Qθ⁻ 초기화 | 465-474행: `policy_model`, `target_model` 초기화, 정확히 대응 | 일치 |
| 13~15행: 에피소드 루프, 환경 초기화 | 489-520행: `for epoch`→`for i in range(num_samples)`(시나리오별 1에피소드), `WaterGym(...).reset()` | 일치(단 "epoch"이 사실상 시나리오 순회이며 논문의 "episode"와 1:1 대응 — 명명만 다름) |
| 16~17행: 확률 pH(e)로 HOPT, 아니면 ε-greedy | 555-565행: `if random.random() < ga_prob: HOPT액션 else: ε-greedy` — **정확히 일치** | 일치 |
| 18~19행: 실행 후 B에 저장 | 571-586행: `gym.step(action)` 후 `replay.append(exp)` — 매 스텝 저장 | 일치 |
| 20~21행: `\|B\|≥M`이면 미니배치 샘플링 | 593-594행: `if len(replay) >= min_replay_size:` 후 `random.sample(replay, batch_size)` | **부분 불일치.** Algorithm 3은 버퍼 최소 크기 임계값과 배치 크기를 같은 기호 M으로 표기해 "배치 하나 채우면 즉시 학습 시작"으로 읽히는데, 코드는 `min_replay_size = batch_size*5`(기본 100)로 **별도의 더 높은 워밍업 임계값**을 쓴다(E0 nan-loss 조사에서 확인, `docs/CODEBASE_MAP.md` 참조). Regular `train()`은 `len(replay) > batch_size`(=M 그대로)를 써서 오히려 Algorithm 3과 더 가깝다. |
| 22행: `a* ← argmin_a Qθ(xt+1,a)` | 612행: `torch.argmax(Q_online_next, ...)` | **불일치(이미 E2에서 확정)** — argmin은 원고 오타, 코드는 argmax. `docs/E02_ARGMINMAX.md` 참조. |
| 23~24행: 타깃 계산, TD loss 최소화 | 606-623행: DDQN 타깃(온라인망 행동선택 + 타깃망 평가) 계산 후 `loss.backward()`/`optimizer.step()` | 일치 |
| 26행: C 업데이트마다 타깃망 동기화 | 628-630행: `update_count % sync_freq == 0` | 일치 |
| 29행: 에피소드마다 pH(e) 감쇠 | 503-513행: `progress = i/(num_samples-1)`에 따라 `ga_prob`을 `ga_start`→`ga_end`로 선형 보간 | 일치(선형 보간이 "predefined guidance schedule"의 한 구현으로 타당) |

### 핵심 판정

**Algorithm 3이 기술하는 것은 (a) 학습 전 demonstration 프리로딩 + 이후 버퍼 샘플링(DQfD류) 구조다 — Phase 1과 Phase 2가 분리되어 있고, Phase 1에서 채운 transition이 Phase 2 내내 B에 남아 함께 샘플링된다.** 그러나 **실제로 실행되는 코드(`_train_guided_by_optimizer`, `evaluate_dpn_guided_GA`/`evaluate_dpn_guided_PSO` → `_evaluate_dpn_guided`가 호출하는 경로)는 (b) 학습 중 확률적 온라인 가이던스만 구현한다.** 매 스텝 `ga_prob` 확률로 휴리스틱이 그 순간의 액션 하나를 골라줄 뿐, "휴리스틱만으로 완주한 궤적을 학습 시작 전에 미리 쌓아두는" 절차 자체가 없다. 리플레이 버퍼는 오직 온라인 학습 중 실제로 실행된(휴리스틱이든 ε-greedy든) 스텝들로만 채워진다.

Algorithm 3 개행 22의 argmin 오타는 이미 E2에서 "원고만 정정 대상, 코드는 정상(argmax)"으로 판정된 바 있다. 이번 발견은 그와 성격이 다르다 — **원고가 기술하는 알고리즘 구조(2단계: 프리로딩 후 학습) 자체가 실행되는 코드와 다르다.** 오타 수준이 아니라 원고 Phase 1 전체가 코드에 대응물이 없는 것이다.

---

## 5. E3·E5 설계에 대한 함의 (판정만 제공, 결정은 사용자 몫)

- **E5(단일변수 ablation, 계획서 §E5)**는 "A2: GA demonstration 프리로딩만(온라인 가이던스 없음)"과 "A3: 온라인 가이던스만(프리로딩 없음)"을 서로 다른 조건으로 분리해 각각 학습하도록 설계되어 있다. 그런데 **현재 코드에 실존하는 학습 함수는 A3에 해당하는 것 하나뿐이다.** A2(프리로딩 단독)에 대응하는 완성된 학습 함수가 없고, 원고가 "GA-guided DDQN"이라고 부르며 실제 논문 결과를 냈다고 여겨지는 것(`evaluate_dpn_guided_GA`)도 실은 A3(온라인 가이던스 단독)에 해당한다 — **원고의 "제안 기법"이 스스로 기술한 Algorithm 3(A4: 결합)이 아니라 A3에 가깝다는 뜻이다.**
- E5를 계획대로 수행하려면 A2(순수 프리로딩) 학습 경로를 **새로 작성**해야 하는데, 그 경우 `pre_train()`/`demonstraion_replay()`를 고쳐 쓰는 것이 출발점이 될 수 있다(단, 171행의 `demon_inps` 버그부터 고쳐야 하고, `pre_train()`이 반환하는 `replay`를 실제로 소비할 학습 루프도 새로 필요하다). 이는 "리팩터링 금지"가 아니라 "신규 구현"의 영역이므로 별도로 승인받아야 한다.
- **E3(다중 시드)**는 현재 코드 그대로 "GA-guided/PSO-guided"를 재학습하면, 그 결과는 여전히 (b) 온라인 가이던스만 반영한다. 이는 지금까지의 모든 논의(계획서, 원고, 이전 결과)가 "GA-guided"라고 불러온 대상과 **동일한 코드**이므로 E3 자체를 막을 이유는 없지만, 응답서 작성 시 "제안 기법이 Algorithm 3(2단계 구조)을 구현한다"는 서술은 **원고 텍스트를 코드에 맞게 고쳐야 함**을 뜻한다 — 코드를 원고에 맞추는 방향(A2/A4 신규 구현)이 아니라, 원고 서술을 코드에 맞추는 방향(Algorithm 3에서 Phase 1을 빼거나 "온라인 가이던스"로 재서술)이 더 현실적인 선택지일 수 있다. **이 판단은 사용자가 한다.**

## 하지 않은 것

- `pre_train()`/`demonstraion_replay()`의 `demon_inps` 버그를 고치지 않았다.
- `_train_guided_by_optimizer()`의 `min_replay_size` 불일치나 Phase 1 부재를 코드로 메우지 않았다.
- 원고 Algorithm 3이나 본문 서술을 수정하지 않았다.
- PSO-guided(`train_guided_byPSO`)는 동일한 `_train_guided_by_optimizer()`를 optimizer_class만 바꿔 호출하므로 위 판정이 그대로 적용된다 — 별도 대조하지 않았다.
