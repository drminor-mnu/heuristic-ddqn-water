# 결과 출처 감사 (E0-1)

**작성일**: 2026-08-25
**방법**: `docs/water-4498965.docx`에서 Table 6~16 전체 수치를 추출하고, `results/`, `results_old/`, `results_org/`, `early_162_60pct/`, `fair_ga_comparison_3pct/`, `gru_size_grid_3pct/`의 실제 CSV 값과 직접 대조. 이름/날짜만으로 추정하지 않고 값 일치를 1차 근거로 삼았다.

## 요약

| 대상 | 상태 |
|---|---|
| Table 6~15의 **Regular** 행 | 전부 확인됨 |
| Table 11~15의 **GA / PSO (only, 학습無)** 행 | 확인됨(파일 값 일치) — **단 2026-08-26 재실행 불일치 발견, 아래 참조** |
| Table 6~15의 **GA-guided / PSO-guided** 행 | **전부 출처 불명** |
| Table 16 요약 (Regular / GA-only / PSO-only) | 확인됨 (5.495, 85.46 등 정확히 일치) |
| Table 16의 GA-guided / PSO-guided | 출처 불명 (파생 요약 파일에서만 간접 확인, 원본 raw 없음) |
| Table 8의 세 행 중복 오류 | 원인 규명됨 — 아래 참조 |
| 본문 수치 5.401 | 확인됨 (정확히 일치) |
| 본문 수치 16.5% / 8.0% / 85.0% | Regular 쪽 항만 확인됨, GA/PSO-guided 쪽 미확인 값에 의존 |

## 표별 상세

### Table 6~10 (2기준 Regular vs 4기준 GA-guided)
- **Regular 행**: `results/dqn_gru_regular_w0_5_w0_5_w0_0_w0_0_min2_tr0_6_acttype0.csv` — 모든 duration(60~1440)·재현기간(RP10~100) 값이 정확히 일치.
- **GA-guided 행**: 출처 불명. 확인한 후보: `results/dqn_guided_GA_..._tr0_6.csv`(전부 0), `..._tr0_03`, `..._tr0_2`, `results_org/*GA*`, `early_162_60pct/outputs*` — 어느 것도 논문 값과 일치하지 않음.

### Table 8 — 확인된 수치 오류의 원인
- Regular/60min 행은 `dqn_gru_regular_w0_5..._tr0_6.csv` block3 col1과 정확히 일치 (171.3 등).
- 같은 파일 block3 col2 (Regular/120min)의 **실제 값은 260.28~260.52**. 그러나 논문에는 GA-guided/60min, Regular/120min, GA-guided/120min 세 행 모두 `130.32/141.26/145.4/151.08/155.68/157.26`로 동일하게 인쇄됨.
- `docs/REVISION_EXPERIMENT_PLAN.md:238`의 저자 예상("Regular 120min은 약 260 수준이어야 함")과 정확히 일치 — **Regular/120min 행이 잘못된 값으로 대체된 것이 확정적 오류.**
- 130.32는 Table 13의 다른 셀과 우연히 같지만, 나머지 값(140.12/146.18/…)은 다르므로 단순히 Table 13에서 복사된 것도 아님. GA-guided 쪽 원본이 소실되어(아래 "핵심 발견" 참조) 복사 방향은 특정 불가 — **Regular/120min이 오염되었다는 사실만 확정, 원인 경로는 미확정.**

### Table 11~15 (5모델 비교)
- **Regular**: `results/dqn_gru_regular_w0_25_w0_4_w0_25_w0_1_min2_tr0_6.csv` — 전 구간 일치.
- **GA(only)**: `results/genetic_w0.25_w0.4_w0.25_w0.1_20260724.csv` — 일치. (주의: `_mean.csv` 파일은 반올림값이 달라 논문 값과 다름 — 실제 출처는 `_20260724` 파일)
- **PSO(only)**: `results/pso_w0.25_w0.4_w0.25_w0.1_20260724.csv` — 일치.

**불일치 발견 (2026-08-26)**: 현재 저장소의 `src/genetic_algo.py`(`objective_function()`)를 위와 동일한 가중치(0.25/0.4/0.25/0.1)로 지금 다시 실행하면, 위 두 파일의 최고수위 평균(5.783m 등, 침수 없음)이 **재현되지 않는다.** 대신 시험한 33개 시나리오(스모크 3개 + 확장 30개, 지속시간 60/360/1440분 전체) 전부에서 액션 0(전량 정지)에 고착되어 최고수위 10m(침수)가 나온다. 원인은 코드 버그가 아니라 `objective_function()`의 1-스텝 목적함수 구조가 이 가중치에서 "펌프 미가동"을 수학적으로 항상 우세하게 만들기 때문임을 확인했다(상세 유도·실측 근거는 `docs/E00_HEURISTIC_MYOPIA_FINDING.md`). 이 저장소는 2026-08-25 스냅샷 이전 git 이력이 없어(`git log --follow -- src/genetic_algo.py` 커밋 1개뿐) 2026-07-24 실행 당시 코드가 지금과 같았는지 버전관리로 확인할 수 없으며, 따라서 "확인됨" 판정은 **파일 값과 원고 수치의 일치 여부에 대해서만 유효**하고 그 파일을 낳은 코드가 현재 코드와 동일했는지는 검증 불가로 남는다. E3에서 GA-only/PSO-only를 재실행하면 원고 Table 11과 판이한(침수 위주) 결과가 나올 가능성이 높다는 점을 미리 인지해야 한다.
- **GA-guided / PSO-guided**: 전부 출처 불명 (아래 핵심 발견 참조).

### Table 16 (요약)
- Regular(5.495/85.46/308.25/19.79/5.34), GA-only(5.783/43.66/…/0.05), PSO-only(5.783/43.70/…/0.14) — 전부 정확히 일치 (위 세 파일의 54개 값 평균으로 재계산해 확인).
- **GA-guided(5.607/91.50/312.84/18.97/4.46)**: 간접 확인만 가능. `results/dqn_regular_vs_ga_guided_radar_summary.csv`에 5.607074/91.495185/4.455556이 남아있어 논문 값과 소수점 3자리까지 일치하지만, 이건 **파생 요약 파일**이며 이걸 만든 원본 raw 파일(`dqn_guided_GA_..._tr0_6.csv`)은 현재 전부 0으로 덮어써진 상태.
- **PSO-guided(5.637/78.61/449.48/19.16/0.80)**: 출처 불명. 유일한 후보 파일(`dqn_guided_PSO_..._tr0_6.csv`)의 실제 평균(switches=103.48, dryrun=3.46)이 논문 값과 다름 — 이 파일은 논문 결과를 만든 실행이 아님.

## 핵심 발견 — GA-guided/PSO-guided 원본 데이터 소실

`results/dqn_guided_GA_w0_25_w0_4_w0_25_w0_1_ga0_6_min2_tr0_6_acttype0.csv`가 Table 6~16 전체의 GA-guided 행의 공통 출처로 지목되는 파일이다(`src/plot_dqn_radar.py` 등이 이 경로를 하드코딩). 그런데:

- 이 파일은 **현재 전부 0 값**으로, 2026-07-27 18:15에 마지막으로 덮어써짐.
- 이 값을 사용해 만들어진 파생 결과물(`dqn_regular_vs_ga_guided_radar_summary.csv`, 2026-07-24 18:08 / `_switching_radar_summary.csv`, 2026-07-26 15:50)은 **덮어써지기 전**에 생성되어 살아남음.
- 즉 **2026-07-24~26 사이 실제 값이 있었던 실행 결과가, 2026-07-27의 후속 재실행(추정: 실패했거나 미완료)에 의해 원본 raw csv 위에서 조용히 소실**됨. 버전관리 없이 같은 경로에 덮어쓰는 구조라 git으로도 복구 불가 (현재 저장소는 재투고 시점 스냅샷 2개 커밋뿐이며, 둘 다 이미 오염된 상태를 담고 있음. 과거 `docs/_git_history_ga-guided-ddqn`도 이 실험보다 훨씬 이전 이력만 보존).
- PSO-guided의 유일한 후보 파일도 논문 값과 불일치 — 이쪽도 원본 없음.
- 결과적으로 **Table 6~16 전체 셀의 절반가량(모든 GA-guided/PSO-guided 항목)의 시나리오 단위 원본 데이터가 저장소 내 어디에도 존재하지 않는다.** Table 16의 GA-guided 3개 값만 파생 요약으로 간접 확인 가능하고, 나머지(Table 6~15의 duration/RP별 세부값, Table 16의 PSO-guided 전체)는 재현 불가능.

## 디렉터리 접미사 (config 근거)

- `_3pct` = `train_fraction: 0.03` (`fair_ga_comparison_3pct/config/experiment.json`, `gru_size_grid_3pct/config/experiment.json`)
- `_60pct` = `train_fraction: 0.60` (`early_162_60pct/config/experiment.json`)
- `early_162`: README 기준 "54개 층에서 3개씩 총 162개 시나리오" 선택. **논문 본문은 "160개"로 서술 — 실제는 162개로 불일치.** 테스트셋도 논문은 "2,700개"이나 실제 `fixed_test.json`은 **2,646개** — 역시 불일치.
- `fair_ga_comparison_3pct`: 3% 데이터에서의 표본효율성 확인 실험(별도 split_seed=20260721). 5.401 수치와는 무관 — 5.401은 `early_162_60pct` 소속.

## 본문 서술 수치

| 수치 | 상태 | 출처 |
|---|---|---|
| 5.401 | **확인됨** (정확 일치) | `early_162_60pct/outputs_full_schedule/summary.json` → `test_reward_difference_ga_minus_regular: 5.401066...` |
| 5.495 / 85.46 | 확인됨 | Regular 소스 파일 평균 |
| 16.5% | Regular 쪽(5.34)만 확인, GA-guided 쪽(4.46)은 파생 요약에만 존재 | 재계산 시 내적 일관성은 있으나 1차 출처 미확인 |
| 8.0%, 85.0% | 동일 — PSO-guided(78.61, 0.80) 원본 미확인 | 좌동 |

## 재실행 대상 (재현 불가로 판정된 것)

- GA-guided DDQN, PSO-guided DDQN 전체 재학습·재평가 필요 (Table 6~16 전 구간)
- `early_162`의 실제 시나리오 수(162 vs 논문 표기 160), 테스트셋 수(2,646 vs 논문 표기 2,700) 불일치 — 본문 수정 또는 재확인 필요

## `results/repeated_heuristics/` — seed42가 아닌 별도 분할 사용 (2026-08-25 추가)

`results/repeated_heuristics/`의 GA/PSO 5시드 반복 결과(`genetic_seed_*.csv`, `pso_seed_*.csv`, `genetic_w0.25_..._mean.csv`, `pso_w0.25_..._mean.csv`, `fixed_test_2700.json`)는 전부 `src/repeat_heuristic_benchmark.py`의 자체 `fixed_test_paths(split_seed=20260724)`로 생성된 테스트 2,700개를 사용했다. 이는 `data_paths.load_fixed_split()`이 읽는 `data/splits/fixed_split_seed42.json`(split_seed=42, 본편 학습·평가 전반이 쓰는 기준 분할)과 **다른 분할**이다.

- 파일명 기준 두 테스트셋의 겹침은 1,111/2,700 = **41.1%** — `docs/DATA_SPLIT_AUDIT.md`의 독립 표본 이론값(50/124 ≈ 40.3%)과 일치. 즉 완전히 다른 무작위 표본이며, 한쪽이 다른 쪽의 부분집합이거나 재현 오류인 관계가 아니다.
- 두 분할이 참조하는 시나리오 파일 자체(.inp)는 동일 데이터 트리에서 나온 것으로 확인됨(`docs/DATA_SPLIT_AUDIT.md`의 MD5 전수 대조 참조) — 데이터 자체가 다른 것이 아니라 **표본 추출(어느 2,700개를 test로 뽑았는가)**이 다르다.
- 2026-08-25 커밋에서 `src/benchmark_test_runtime.py`의 `_load_paths()`를 `fixed_test_2700.json` 읽기에서 `data_paths.load_fixed_split()`(seed42)로 교체했다. 이 교체 **이전**에 생성된 `results/repeated_heuristics/` 산출물과, 교체 **이후** `benchmark_test_runtime.py`/`benchmark_full_data_runtime.py`가 새로 만드는 산출물은 서로 다른 테스트 모집단에서 나온 것이므로 **직접 비교하면 안 된다.**
- `src/repeat_heuristic_benchmark.py`는 이번 라운드에서 호출부만 `data_paths.load_fixed_split()`로 바꾸고 `fixed_test_paths()` 함수 정의 자체는 삭제하지 않았다(코드 동결 원칙). 따라서 이 스크립트를 다시 실행하면 이제는 seed42 테스트셋을 쓰게 되어, 기존 `results/repeated_heuristics/` 산출물과는 재현되지 않는다 — 재실행 시 반드시 새 출력 경로로 구분해서 남길 것.
- **E10(계산시간 분해)에서 이 디렉터리의 수치를 참고 자료로 쓸 경우, 어느 분할에서 나온 값인지 표에 명시할 것.** seed42 기준 산출물과 섞어서 평균·비교하지 말 것.

## ENUM-guided DDQN — 이미 3시드 학습·비교 완료 (2026-09-11 추가)

외부 검토 (b) "§3.6이 '가이던스 신호를 완전열거로도 만들 수 있다'고 서술하나 확인하지 않았다"는
**사실과 다르다.** 진입점부터 추적한 결과:

- `src/enum_baseline.py`의 `EnumSearch`(`GeneticAlgorithm` 서브클래스, `objective_function` 상속,
  탐색만 6-way argmax로 교체, RNG 미사용·결정론적)가 이미 존재하고,
- `src/dqn_from_demon_v1.py:730` `train_guided_byENUM()`이 동일한 `_train_guided_by_optimizer`
  제네릭 훅에 `optimizer_class=EnumSearch`만 바꿔 넣어 이미 연결되어 있으며,
- `src/perform_evaluate.py:614` `evaluate_dpn_guided_ENUM()`도 존재하고,
- `src/run_e04_guided.py`가 `run_e03_seeds.py`의 설정(weights/γ/가이던스 스케줄/YEARS/DURATIONS)을
  그대로 import해 재사용하는 **이미 완성된 드라이버**로서, `results/E04_enum/ddqn_by_L/L{L}/seed{n}/`에
  resume 가능·원자적 쓰기로 출력한다.
- **`--L 1 --seeds 1,2,3`으로 이미 실행 완료**(2026-08-30, 사용자 터미널, 각 시드 ~73분,
  `git_commit=f071384...`). 3시드 전부 `DONE`(row_count=2700 일치), `results/E04_enum/FINDINGS.md`
  §E에 E3 v2의 Regular/GA-guided/PSO-guided(5시드)와의 비교표까지 이미 작성되어 있다.

**계획서 근거**: `docs/REVISION_EXPERIMENT_PLAN.md` E4-a 항목3, 완료조건 목록에 처음부터
"ENUM-guided DDQN 학습(3시드)"으로 명시되어 있었다(5시드 아님) — E3 v2의 5시드 표준과는
다른 시드 수로 설계된 것.

**따라서 이번 요청(Task 2, 2026-09-11)의 전제("확인 안 됨") 및 산출물 사양
(`src/enum_guide.py`/`src/run_e16_enum_guided.py` 신규 작성, `results/E16_enum_guided/` 신규 경로)은
기존 산출물과 충돌한다.** 신규 파일·신규 경로를 만들면 이미 존재하는 `enum_baseline.py`/
`run_e04_guided.py`/`results/E04_enum/ddqn_by_L/L1/seed{1,2,3}/`와 내용이 100% 중복되는
평행 구현이 생긴다. 임의로 조정하지 않고 이 사실만 기록한다 — 처리 방침은 사용자 보고 후 결정.
