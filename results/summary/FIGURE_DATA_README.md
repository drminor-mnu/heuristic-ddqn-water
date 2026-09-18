# Figure 1~4 재현용 데이터 (CSV)

Figure 1~4를 사용자가 직접 다시 그릴 수 있도록 원 데이터를
`results/summary/figure_data/`에 CSV로 출력했다. 원고 그림 파일
자체(`E14_figure*.png`)는 건드리지 않았다. 아래는 각 CSV의 생성
경로를 진입점부터 추적한 결과다.

---

## 1. Figure 1·2 (학습곡선)

**생성 스크립트**: `src/run_e14_fig1_2.py` (실제 `E14_figure1_loss.png`
/ `E14_figure2_reward.png`를 만든 스크립트, 코드 직접 확인).

- **사용한 train_log.csv 경로**:
  - Regular(2-criterion): `results/E14_two_criteria/Regular/seed{1..5}/train_log.csv`
  - GA-guided(4-criterion): `results/E03_seeds/GA-guided/seed{1..5}/train_log.csv`
  - 열: `episode, train_loss, train_reward`. 두 모델 다 각 시드
    3996 에피소드(파일 3997행 = 헤더+3996).
- **단일 시드 vs 5시드 평균**: **5시드 평균**이다
  (`run_e14_fig1_2.py:55-58`) — 5개 시드의 `train_loss`/`train_reward`를
  같은 에피소드 인덱스에서 평균낸 뒤(`arr.mean(axis=0)`) 평활화한다.
  단일 시드 원본이 아니다.
- **평활화 여부·윈도우**: **평활화함**. `ROLL = 100`
  (`run_e14_fig1_2.py:34`), `np.convolve(x, kernel, mode='valid')`로
  100-에피소드 이동평균, **5시드 평균을 낸 뒤에** 적용(시드별로
  먼저 평활화하지 않음). `mode='valid'`이므로 평활화된 배열 길이는
  `3996 - 100 + 1 = 3897`이고, 원 스크립트는 이 평활화된 배열의
  위치(0-index)를 그대로 x축(episode)으로 플롯한다 — 즉 그림의 x=k
  지점은 실제로는 원 에피소드 k~k+99 구간의 평균이다.
- **결측치 발견 사항(사실 기록, 임의수정 안 함)**: GA-guided
  episode 0의 5시드 평균 `train_loss`가 `nan`이다(`fig1_loss.csv`
  raw 열 1건) — 리플레이 버퍼가 아직 안 찼을 episode 0에서 5개 시드
  중 하나 이상이 loss를 계산하지 않은 것으로 보인다(원인 미검증,
  train_log.csv 자체에 존재하는 값을 그대로 읽었을 뿐). 이 nan은
  100-윈도우 평활화 시 `smoothed[0]` 1개 값(3897개 중 1개)에만
  전파된다. `train_reward`에는 nan 없음(fig2 raw 열 0건).

**출력**: `results/summary/figure_data/fig1_loss.csv`,
`fig2_reward.csv`. 열 구성(둘 다 동일 패턴):

| 열 | 의미 |
|---|---|
| `episode` | 0-index, 평활화된 그림의 x축 위치와 동일 |
| `regular_{loss|reward}_mean_5seed_raw` | Regular 5시드 평균(평활화 전) |
| `ga_guided_{loss|reward}_mean_5seed_raw` | GA-guided 5시드 평균(평활화 전) |
| `regular_{loss|reward}_mean_5seed_smoothed_roll100` | 위를 100-윈도우 이동평균(원 그림에 실제로 그려진 값). 배열 길이가 짧아 대응 원본이 없는 마지막 99행은 빈 값 |
| `ga_guided_{loss|reward}_mean_5seed_smoothed_roll100` | 상동, GA-guided |

raw 열은 참고용으로 추가한 것이며(평활화 없이 다시 그리고 싶을
경우), 원 그림과 동일한 곡선은 `*_smoothed_roll100` 두 열이다. 최종
smoothed 값(episode 3896): loss는 Regular 0.0062 / GA-guided 0.0030,
reward는 Regular 157.7445 / GA-guided 152.8492 — `run_e14_fig1_2.py`가
원래 콘솔에 출력하던 "final rolling-mean" 값과 같은 계산 경로로
재생성했다(원 실행 로그 자체는 남아있지 않아 직접 대조는 못 했고,
동일 코드·동일 데이터로 재계산해 일치를 담보했다).

---

## 2. Figure 3·4 (운전 궤적)

**생성 스크립트**: `src/run_e14_fig3_4.py` (실제
`E14_figure3_regular_operation.png` / `E14_figure4_ga_guided_operation.png`를
만든 스크립트).

- **시나리오**: `30yr_0060m_h211`
  (`data/gasan/30year/30yr_0060m_h211.inp`, 스크립트에 고정 상수로
  박혀 있음 — 원 원고 그림의 시나리오 ID 자체는 유실돼 재현 불가능,
  스크립트 주석에 그 경위가 적혀 있다).
- **Regular**: `results/E14_two_criteria/Regular/seed1/model.pkl`,
  가중치 `[0.50, 0.50, 0, 0]`(2-criterion).
- **GA-guided**: `results/E03_seeds/GA-guided/seed1/model.pkl`,
  가중치 `[0.40, 0.25, 0.25, 0.10]`(4-criterion, 신가중치).
- **모델 롤아웃**: `dqn_from_demon_v1.test_model()`을 직접 호출해
  실행 시점에 새로 시뮬레이션한다(사전 저장된 궤적 파일이 없음 —
  `run_e14_fig3_4.py`도 매번 재실행하는 구조).
- **state 구조**: `WaterGym.step()`/`reset()`이 반환하는
  `[rain, inflow, outflow, reservoir_vol, reservoir_level]`
  (`src/water_gym.py:106-111`, `184-189`로 확인) — `inflow`가
  2번째 원소, `outflow`가 3번째, `water_level`이 마지막(5번째,
  `state[-1]`) 원소다. `run_e14_fig3_4.py`도 `state[-1]`을 수위로
  사용하므로 이 CSV의 매핑과 동일하다.
- **inflow_volume/outflow_volume 단위**: 둘 다 **2분 결정간격당
  부피(m³)**다 — `inflow`는 SWMM outfall 노드의
  `total_inflow(m³/s) × time_unit(s)`로 이미 부피로 계산되어 있고
  (`src/sim_swmm.py` `swmm_execute()`), `outflow`는
  `pumpq.sum() × self.minutes`로 역시 부피다(`src/water_gym.py:160`).
  두 값의 물리 단위가 일치함을 코드 추적으로 확인했다(추정이 아님).
- **time_min**: 결정 간격 `self.minutes=2`(분)를 `rollout()`이
  `test_model(..., minutes=2)`로 명시 고정(`run_e14_fig3_4.py:46`,
  기본값도 2) — `time_min = 행 인덱스 × 2`로 계산했다.
- **action 값 범위(오프셋 건 해소, 2026-09-05 확인)**: `Actions0`
  (6개 펌프 조합, `src/water_gym.py:23-28`) 인덱스를 **이 CSV에는
  코드 내부값 그대로 0~5로 저장했다.** 원고 그림(Figure 3·4)이 실제로
  쓰는 1~6 표기는 `src/graph_draw.py:65`
  `plot_individual_case(state_list, action_list)`의
  `action = np.array(action_list) + 1`에서 나온다 — 그림을 그릴 때만
  +1 해서 표시하고, 시뮬레이션·코드 내부(`WaterGym`, `test_model()`
  반환값)에는 0~5가 그대로 쓰인다. 즉 **원고의 "액션 번호 1~6"
  표기가 맞다**(이전 판에서 "원고 실제 표기 미확인"으로 남겼던 항목,
  `graph_draw.py` 직접 대조로 이번에 확정). **이 CSV(`fig3_regular.csv`,
  `fig4_ga_guided.csv`)의 `action`열은 0~5(코드 내부값)이며 1~6이
  아니다** — 원고와 같은 1~6 표기로 그리려면 값에 +1 해야 한다.

**출력**: `results/summary/figure_data/fig3_regular.csv`,
`fig4_ga_guided.csv`. 열: `time_min, inflow_volume, outflow_volume,
water_level, action`. 각 59행(0~116분, 2분 간격, 시뮬레이션 종료
시점까지).

---

## 3. max_level 대조 확인

`test_metrics.csv`의 `30yr_0060m_h211` 행(seed1)과 CSV 내 `water_level`
최댓값을 대조했다.

| 모델 | test_metrics.csv max_level_m | CSV 내 water_level 최댓값 | 일치 |
|---|---|---|---|
| Regular(E14 seed1) | 5.45305243132749 | 5.453052 | **일치** |
| GA-guided(E03 v2 seed1) | 5.932338981368884 | 5.932339 | **일치** |

CSV는 소수 6자리로 반올림해 저장했고, `test_metrics.csv`의 전체
자릿수와 비교해 6자리까지 정확히 일치함을 확인했다. 롤아웃을 CSV
출력용으로 다시 실행했음에도(모델은 결정적 추론이므로) 기존
`test_metrics.csv` 생성 시점의 값과 동일하다 — 두 실행이 서로 다른
소스가 아님을 의미한다.

---

## 4. 원본 그림을 그린 함수의 출처(2026-09-05 조사)

`results_org/images/`의 세 원본 이미지(원고 그림에 쓰인 것으로
추정되는 저자 작업본, 읽기전용 참조)를 어느 함수가 그렸는지
`src/` 전수 조사한 결과.

| 원본 이미지 | 그린 함수 | 일치 여부 |
|---|---|---|
| `figure4.png` | `src/graph_draw.py:37` `plot_individual_case(state_list, action_list)` | **정확히 일치**(좌: inflow/outflow, 우: elevation+action 트윈축, action+1 오프셋까지 코드로 확인) |
| `rewards_training.png` | 없음 | **현재 `src/`의 어떤 함수도 정확히 일치하지 않음** |
| `loss_v2.png` | 없음 | **현재 `src/`의 어떤 함수도 정확히 일치하지 않음** |

`rewards_training.png`/`loss_v2.png`와 legend 문구("Regular DDQN"/
"GA guided DDQN")·xlabel("Episode")까지 일치하는 후보는
`src/perform_evaluate.py:254` `plot_reward()`(및 동일 사본
`perform_evaluate copy.py`)뿐이었으나, 이 함수는 `subplots(2,1)`로
서브플롯 2개를 만든 뒤 배열인 `ax`에 바로 `ax.set_xlabel(...)`을
호출해 **그대로 실행하면 AttributeError로 실패하는 버그**가 있다
(git 최초 커밋부터 이 형태 — 과거에 정상 작동한 버전 자체가 없음).
`dqn_from_demon_v1.py:907` `plot_losses()`/`:924` `plot_rewards()`도
xlabel이 "Scenario"이고 `plot_rewards`도 서브플롯 분리라 이미지와
다르다. 두 이미지의 실제 생성 경위는 코드로 추적 불가 — 대화형
환경(Spyder 등)에서 즉석 수정 후 소스에는 반영하지 않은 것으로
보이나 확인할 방법은 없다(추정, 사실 아님).

공통점: 위 세 파일(`graph_draw.py`, `perform_evaluate.py`,
`dqn_from_demon_v1.py`) 전부 import 시점에
`plot_style.set_plot_style()`을 호출한다 — 세 원본 이미지에 공통된
굵은 글씨체·격자(grid) 스타일은 이 함수에서 온다.

## 5. 최종 그림 재생성 결과 (2026-09-05, 최종 정리)

`src/run_figure3_4_final.py`(Figure 3·4, 원본 함수
`graph_draw.plot_individual_case()`를 **수정 없이 그대로 호출**),
`src/run_figure1_2_final.py`(Figure 1·2, 원본 함수가 없어 동일
스타일로 신규 작성)를 실행해 `results/summary/figures_final/
figure{1,2,3,4}.png`를 생성했다. `graph_draw.py`·`plot_style.py` 등
기존 함수는 이 과정에서 전혀 수정하지 않았다.

### 수정 경위(같은 날 진행된 시행착오 포함, 최종 상태만 유효)

| 시점 | 요청 | 조치 | 최종 상태 |
|---|---|---|---|
| 1차 | Figure 1·2 타이틀 제거 | `run_figure1_2_final.py`의 `ax.set_title(...)` 삭제 | **유지**(최종) |
| 1차 | Figure 3·4 action을 0부터 시작 | `graph_draw.plot_individual_case()`(내부에서 `action=action_list+1`로 0~5→1~6 표시)를 우회하려고, +1을 뺀 로컬 복사 함수 `plot_individual_case_action0()`를 신규 정의해 그걸 호출하도록 변경 | **취소됨** — 되돌림 |
| 2차(정정) | "액션 타입은 1부터 시작하는 게 맞다" | 로컬 복사 함수를 폐기하고 `run_figure3_4_final.py`를 다시 원본 함수 `graph_draw.plot_individual_case()`를 직접 호출하는 형태로 되돌림(로컬 재구현 코드 삭제, 원본 함수 그대로 사용) | **최종 확정** — Action Type 1~6 |
| 3차 | Figure 1·2의 x축 라벨을 "Episode"→"Scenario"로 변경 | `run_figure1_2_final.py`의 `ax.set_xlabel('Episode')`를 `ax.set_xlabel('Scenario')`로 변경 | **취소됨** — 되돌림 |
| 4차(정정) | "x축을 다시 Episode로" | `ax.set_xlabel('Scenario')`를 다시 `ax.set_xlabel('Episode')`로 되돌림 | **최종 확정** — xlabel "Episode" |

**최종적으로 유효한 변경은 "Figure 1·2 타이틀 제거" 1건뿐이다.**
xlabel은 "Episode"→"Scenario"→"Episode"로 되돌아와 원래 값과 같다
(재생성 결과 `figure1.png`/`figure2.png` 바이트 수가 "Scenario" 실험
직전과 정확히 동일함을 확인 — 98,511B/210,565B). Figure 3·4의 action 표시는
원래 상태(원본 함수의 +1 오프셋, 1~6 표시)로 되돌아갔다 — 처음
§1/§4에서 확인했던 "원고 1~6 표기가 맞다"는 결론이 최종적으로
유지된다.

### 파일 확인(최종본, 2026-09-05)

| 파일 | 크기(px) | DPI | 파일 크기(byte) | 비고 |
|---|---|---|---|---|
| `figure1.png` | 2160×864 | 72.009 | 98,511 | 타이틀 없음(제거됨), xlabel="Episode"(최종) |
| `figure2.png` | 2160×864 | 72.009 | 210,565 | 타이틀 없음(제거됨), xlabel="Episode"(최종) |
| `figure3.png` | 2160×864 | 72.009 | 123,747 | Action Type 1~6(원본 함수 그대로) |
| `figure4.png` | 2160×864 | 72.009 | 137,753 | Action Type 1~6(원본 함수 그대로) |

`figure3.png`/`figure4.png`의 바이트 수(123,747 / 137,753)가 이번
배치 최초 생성분과 정확히 동일하다 — 원본 함수를 그대로 다시
호출했으니 산출물도 동일해야 하고, 실제로 그렇다는 것을 바이트
단위로 확인했다(우회했다가 되돌리는 과정에서 다른 부작용이 남지
않았음을 뜻한다).

목표(원고, `results_org/images/figure4.png`)는 2131×834px @
72.009dpi다. `graph_draw.plot_individual_case()`가 내부에서
`fig.set_size_inches(30, 12)`를 고정하고 있어(이 함수를 수정하지
않기로 했으므로 크기 자체는 바꿀 수 없음) `savefig(dpi=72)`로
맞춘 결과 2160×864px가 나왔다 — 목표 대비 가로 +1.4%, 세로 +3.6%
근소한 차이(원본이 화면 캡처/저장 과정에서 UI 여백이 약간
잘렸을 가능성, 확인 불가). DPI는 72.009와 사실상 동일(72 지정).

### Figure 3·4 육안 대조 (vs `results_org/images/figure4.png`, 최종본)

| 항목 | 원본 | 재생성본 | 일치 |
|---|---|---|---|
| 축 라벨(좌: Time(min)/Volume(m³), 우: Time(min)/Elevation(m)/Action Type) | 동일 | 동일 | **일치**(동일 함수 사용) |
| 범례 문구(Inflow into the reservoir / Outflow from the reservoir / Reservoir Elevation / Pump Operation) | 위 4개 | 위 4개 | **일치** |
| 마커 모양·색(빨간 다이아몬드=inflow, 파란 원=outflow&level, 초록 다이아몬드=action) | 동일 | 동일 | **일치** |
| Action Type 축 범위 | 1~6 | **1~6**(원본 함수의 +1 오프셋 그대로) | **일치**(재확인 완료 — 원본 함수를 그대로 쓰므로 당연히 일치) |
| Volume/Elevation 축의 구체적 수치 범위 | 시나리오 고유값 | 시나리오 고유값 | **다름(예상됨)** — 원고 그림의 정확한 원본 시나리오·시드는 유실돼 재현 불가(`run_e14_fig3_4.py` 주석에 기록된 기존 사실). 같은 30yr_0060m_h211 시나리오·같은 지속시간(60분)이라는 조건만 맞춘 대체 재현이라 절대 수치(최고 수위 5.45m/5.93m 등)는 원본과 다를 수 있다 — 이 데이터로 만든 그림이라는 전제하에서는 유일하게 맞는 값(§3에서 test_metrics.csv와 6자리까지 일치 확인됨). |

### Figure 1·2 육안 대조 (vs `loss_v2.png`, `rewards_training.png`, 최종본)

| 항목 | 원본 | 재생성본 | 일치 |
|---|---|---|---|
| 타이틀 | 없음 | 없음(제거됨) | **일치** |
| xlabel "Episode" | O | O(Scenario로 바꿨다가 재정정으로 되돌림, 최종은 Episode) | **일치** |
| ylabel(Fig1="Loss", Fig2="Reward") | O | O | **일치**(원본 이미지에서 직접 확인한 라벨 그대로 사용) |
| 스타일(격자, 굵은 글씨, 범례 폰트) | `set_plot_style()`풍 | `set_plot_style()` 직접 호출 | **일치** |
| 범례 문구 | "Regular DDQN" / "GA guided DDQN"(하이픈 없음) | "Regular DDQN" / "GA-guided DDQN"(하이픈 있음) | **다름** — 사용자 지시로 하이픈 있는 표기를 명시적으로 사용, 원본 표기와 다르다는 점을 사실로 기록 |
| x축 범위(에피소드 수) | **약 0~650~700** (두 원본 이미지 모두 눈금이 600대에서 끝남) | **0~3896** | **크게 다름** — 원본 그림의 학습은 약 650~700 에피소드에서 종료된 실행으로 보이는데, 이번 재현에 쓴 E14/E03 v2 데이터는 시드당 3996 에피소드다. 같은 학습 실행이 아니라 **서로 다른 학습 실행(원본은 구버전 또는 다른 설정의 단일 실행, 재현본은 현재 확정된 5시드 실험)의 그림**이라는 뜻이다. 임의로 원본 스케일에 맞추지 않았다. |
| 곡선 형태 | 원시값(노이즈 매우 심함, 5시드 평균/평활화로 보이지 않음) | 5시드 평균 + 100-에피소드 이동평균(매끈함) | **다름(의도된 차이)** — 지시대로 평균·평활화 데이터를 사용했다. |
| y축 값 범위(Loss) | 0~0.016 | 0~0.016 | **유사**(초기 피크값 스케일은 비슷) |
| y축 값 범위(Reward) | 약 20~350(노이즈 폭 큼) | 약 110~165(평활화로 축소) | **다름(의도된 차이)** — 평활화·평균 처리로 극값이 줄었다. |

### 최종 결론

- **Figure 3·4**: 원본 함수(`graph_draw.plot_individual_case()`)를
  수정 없이 그대로 호출한 결과물이라 **형식이 원고와 완전히
  일치**한다(Action Type 1~6 포함). 시나리오 데이터 자체의 차이
  (재현 불가능한 원본 시나리오 대체)만 존재하며, 이는 이전부터
  알려진 한계다.
- **Figure 1·2**: 원본을 그린 함수 자체가 현재 `src/`에 없어(§4
  참조) 새로 작성한 스크립트의 산출물이다. 스타일(격자·굵은 글씨·
  xlabel/ylabel·타이틀 없음)은 원본과 맞췄다(xlabel을 한때
  "Scenario"로 바꿨다가 재정정으로 "Episode"로 되돌려, 최종적으로
  원본과 동일). 다만 **데이터 소스가 근본적으로 다르다**
  (원본은 ~650~700 에피소드의 단일/구버전 실행, 재현본은 3996
  에피소드 5시드 평균+평활화) — 곡선의 길이·형태·범위가 다르다는
  것을 감추지 않고 그대로 기록한다.

## 6. 참고

- 이 문서와 CSV 파일들은 원고 docx를 수정하지 않았다. Figure
  1~4를 다시 그리는 것은 사용자 몫이다.
- Figure 1·2용 5시드 데이터 중 결측치 1건(GA-guided episode 0
  loss)은 원본 `train_log.csv`에 이미 존재하던 값이며, 이 문서
  작성 과정에서 새로 발생한 것이 아니다.
- 최종 그림 생성 스크립트: `src/run_figure3_4_final.py`(기존
  `graph_draw.py`의 `plot_individual_case()`를 수정 없이 호출),
  `src/run_figure1_2_final.py`(신규 작성). 둘 다 `results/summary/
  figures_final/`에 출력하며, `src/run_e14_fig1_2.py`/
  `run_e14_fig3_4.py`(기존 §1·§2 대상, `results/summary/
  E14_figure*.png` 출력)와는 별개의 스크립트다 — 기존 스크립트를
  수정하지 않았다.
