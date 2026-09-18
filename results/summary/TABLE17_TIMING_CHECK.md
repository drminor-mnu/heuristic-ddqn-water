# Table 17 계산시간 정합 확인 (신설 §3.5.1, R4-12 재발 방지)

사용자 지적: Table 17 후보값(E4-a, ENUM/PSO/GA = 0.039/1.43/20.0 ms)과
Figure 10 재측정 GA-only(19.93ms)가 같은 논문에서 GA의 계산시간을 두
값으로 보고하는 셈이라 R4-12 재발 우려. 확인 결과를 아래에 기록한다.
해석·옹호 없이 사실·수치만 적는다. 원고 docx는 이 문서 작성으로
수정하지 않았다.

## 1. E4-a 값이 어떤 측정 범위인가

`results/E04_enum/FINDINGS.md` §C의 값은 `src/run_e04_enum.py`
`run_agreement()`에서 나왔다. 코드 확인
(`src/run_e04_enum.py:141-180`):

```python
t0 = time.perf_counter()
scores = enum.score_all_actions(state, action_list, next_inflow)
...
t_enum = time.perf_counter() - t0

t0 = time.perf_counter()
a_ga = int(ga.genetic_algorithm(state, action_list, next_inflow))
t_ga = time.perf_counter() - t0

t0 = time.perf_counter()
a_pso = int(pso.particle_swarm_optimization(state, action_list, next_inflow))
t_pso = time.perf_counter() - t0
...
state, _r, done, _i = gym.step(a_enum)   # <- 타이밍 블록 밖, 미포함
```

**E4-a는 탐색 함수 호출 자체만 잰다** — `WaterGym.step()`(환경 step
오버헤드)은 타이밍 블록 밖에서 별도로 호출되며 **측정에 포함되지
않는다.** 상태는 이미 계산되어 인자로 들어오므로 GA/PSO/ENUM에는
DDQN의 (b)(전처리)에 대응하는 별도 단계가 없다 — 세 방법 모두 탐색
함수 안에서 직접 원시 상태를 쓴다.

## 2. Figure 10의 (c) 기준과 같은가 다른가

**측정 범위가 다르다.** Figure 10(및 Table 5) 갱신판의 GA-only/PSO-only
값은 `src/run_e10_unified_c.py`의 `time_heuristic_c()`로 측정됐다 —
이 함수는 **`run_genetic_algo()`/`run_pso()` 전체 호출**(에피소드 전체를
자체적으로 돌리며, 매 스텝 탐색 + `WaterGym.step()` 호출을 전부 포함)의
벽시계 시간에서 `WaterGym.reset()`(SWMM, monkey-patch로 분리)만 뺀
값이다. 즉 **환경 step 오버헤드가 포함되어 있다** — E4-a와 달리.

요약:

| | 상태관측 | 탐색(결정) | 환경 step(`gym.step()`) | SWMM(`gym.reset()`) |
|---|---|---|---|---|
| E4-a(`run_e04_enum.py`) | 포함(외부에서 1회 계산, 재사용) | **포함** | **미포함** | 미포함(대상 아님) |
| Figure 10/Table 5 (c)-basis(`run_e10_unified_c.py`) | 포함 | **포함** | **포함** | 미포함 |

## 3. 재측정 — ENUM을 (c) 기준으로 신규 측정, GA/PSO는 재사용

범위 차이가 실제 값에 미치는 영향을 직접 확인하기 위해, **GA-only/PSO-only는
이미 Figure 10에서 (c) 기준으로 측정되어 있으므로 그 값을 그대로
재사용**(재측정하지 않음 — 같은 스크립트·같은 표본·같은 정의를 다시
돌려도 실행 잡음만 추가될 뿐이고, 재사용하면 Table 17과 Figure 10이
**같은 숫자**를 공유해 향후 불일치 가능성 자체를 없앨 수 있음).
**ENUM만 새로 측정**했다 — `results/E04_enum/`에는 ENUM의 (c)-기준
값이 없었기 때문(E4-a는 탐색-only 기준). 기존 평가 함수는 수정하지
않았다: `src/enum_baseline.py`(`EnumSearch`)는 원본 그대로 쓰고, 신규
스크립트 `src/run_table17_timing.py`가 `run_e10_unified_c.py`의
`time_heuristic_c()`/`stratified_subsample()`을 **재사용(import)**만
한다 — `EnumSearch`가 `run_genetic_algo()`를 상속(코드 확인:
`genetic_algo.py`의 `run_genetic_algo()`는 `self.genetic_algorithm(...)`을
호출하므로 `EnumSearch`가 오버라이드한 `genetic_algorithm()`으로 다형적으로
분기)하므로 `time_heuristic_c(instance, 'run_genetic_algo', test_inp)`를
그대로 쓸 수 있었다. 표본·시드·반복은 Figure 10과 동일(54-시나리오 층화
부표본, seed1, 3회 반복).

## 4. 결과 — E4-a vs (c)-기준 재측정, 자릿수 비교

| 모델 | E4-a(탐색-only), ms | (c)-기준(SWMM 제외, 환경step 포함), mean±sd, ms |
|---|---|---|
| ENUM | 0.039 (median 0.039) | **0.0420 ± 0.0011** |
| PSO | 1.43 (median 1.42) | **1.4106 ± 0.0063**(Figure 10과 동일값, 재사용) |
| GA | 20.0 (median 19.2) | **19.9324 ± 0.1072**(Figure 10과 동일값, 재사용) |

**세 모델 모두 두 측정 범위 간 차이가 1~8% 이내이며, GA·PSO는 사실상
동일값(오차범위 내)이다.** ENUM만 절대적으로는 8% 차이(0.039→0.042)가
있으나, 두 값 모두 1ms 미만으로 절대 크기가 작아 Table 17에 함께
실리는 PSO(1.4ms)·GA(19.9ms)에 비해 무시할 수 있는 수준이다.

**해석(사실 기록)**: 환경 step 오버헤드(`WaterGym.step()`)의 절대
비용이 매우 작다(DDQN 경로에서 (a)+(b)+(c) 전체가 약 0.155ms인 것에서도
간접 확인됨 — (a)+(b)만으로도 이보다 작을 것이므로 (c) 성분 자체가
1ms 미만) — 그래서 그 유무가 PSO(1.4ms)·GA(19.9ms) 같은 큰 값에는
거의 영향을 주지 않지만, ENUM(0.04ms 수준)처럼 절대값이 작을 때는
상대적으로 눈에 띄는 차이(8%)로 나타난다.

## 5. 채택안

- **Table 17은 (c) 기준(환경 step 오버헤드 포함, SWMM 제외) 값을
  쓴다** — Table 5·Figure 10과 동일 기준. GA·PSO는 Figure 10의
  `E10_unified_c_summary.csv` 값을 그대로 재사용(같은 숫자를
  공유하므로 불일치 가능성 자체가 없음), ENUM은 이번에 신규 측정한
  0.0420±0.0011ms를 쓴다.
- **E4-a의 기존 값(0.039/1.43/20.0)은 폐기하지 않는다** —
  `results/E04_enum/FINDINGS.md`는 탐색-only 비교(§1-1-1의 "메타휴리스틱이
  더 느리다"는 R1-1/R4-1 논증)가 목적이라 해당 문맥에서는 그대로 유효하다.
  다만 **Table 17(신설 §3.5.1)에는 Table 5·Figure 10과 같은 (c) 기준
  값을 쓴다** — 같은 논문 안에서 같은 이름("GA의 계산시간")으로 서로
  다른 기준의 두 숫자가 나란히 보고되는 R4-12류 재발을 막기 위함.
- Table 17 각주에 측정 기준을 Table 5·Figure 10과 동일하다고 명시할
  것(사용자 지시사항).

출력: `results/summary/table17.csv`, `results/summary/table17.txt`.
측정 스크립트: `src/run_table17_timing.py`(신규, 기존 평가 함수 미수정).
