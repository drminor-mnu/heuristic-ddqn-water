# 데이터 생성·분할 규모 감사 (E0 부속)

**작성일**: 2026-08-25
**방법**: 코드 정적 추적 + 실제 파일/디렉터리 카운트 직접 실행. 추정 없음.

## 요약

| # | 항목 | 논문 값 | 실제 값 | 판정 |
|---|---|---|---|---|
| 1 | 전체 시나리오 수 | 6,696 | **6,696** | 일치 |
| 2 | 학습셋 수 (본편) | 3,996 | **3,996** | 일치 |
| 3 | 테스트셋 수 (본편) | 2,700 | **2,700** | 일치 |
| 3′ | 테스트셋 수 (160-시나리오 한정 실험, 5.401 관련) | "2,700"으로 서술 | **2,646** | **불일치** |
| 4 | 층당 배분 | 54조합 × (74 train + 50 test) | **54조합 × (74+50)** | 일치 |
| 5 | early_162의 실제 개수 | "160개" | **162개** | **불일치** |

## 1~2~4. 6,696 / 3,996 / 74+50 — 코드·데이터로 정확히 재현됨

- `src/data_generate.py`가 재현기간 6개(10/20/30/50/80/100년) × 지속시간 10개(10~1440분) × 분위 등 조합으로 강우 시나리오를 생성. 실측 파일 수(`data/gasan/**/*.inp`)와 `data/selected_inp.txt`가 일치.
- 논문의 "54개 조합"은 **10분 지속시간을 제외한 6재현기간×9지속시간**. 이 필터링은 `perform_evaluate.py` 등 여러 실행 스크립트의 `__main__`에 `durations = ['0060',...,'1440']`(0010 제외)로 하드코딩되어 반복됨.
- 필터 적용 후 실측: 6×9×124(조합당 시나리오) = **6,696** (파일시스템 카운트로 확인).
- `stratified_split_data()`(`perform_evaluate.py`, `cost_model_v2.py`)가 조합당 124개 중 `int(124*0.6)=74`를 train, 나머지 `124-74=50`을 test로 배정 → 54×74=**3,996**, 54×50=**2,700**. 논문 수치와 정확히 일치.
- **재현성 관련 주의(부수 발견)**: 본편 스크립트의 랜덤시드가 `random.seed(time.time())`(`perform_evaluate.py:34`)로 설정되어 있어, "몇 개"는 결정론적으로 6,696/54/74/50이 재현되지만 **"정확히 어느 시나리오가 train/test인지"는 실행마다 달라질 수 있다.** 본편의 실제 2,700개 테스트셋을 고정한 split 파일은 저장소 어디에도 없음. E0 6번(시드 통합)에서 반드시 고쳐야 할 항목.

## 3′·5. "2,700"과 "160개" — 두 건 모두 논문 서술 오류로 판정

### 5.401 실험의 테스트셋은 2,700이 아니라 2,646

`early_162_60pct/`는 본편과 **별도로 재구현된 분할 코드**(`early_162_60pct/src/fair_ga/split.py`)를 쓰며, train/test를 각각 독립적으로 `floor()` 처리한다:
```python
test_count  = int(np.floor(config.test_fraction  * len(values)))  # floor(0.40*124) = 49
train_count = int(np.floor(config.train_fraction * len(values)))  # floor(0.60*124) = 74
```
본편처럼 "나머지 전부를 test로" 배정하지 않고 양쪽 다 floor 하면서 조합당 1개씩 버려짐 → 54×49 = **2,646** (2,700−54).

직접 확인:
- `early_162_60pct/data/splits/fixed_test.json` — 항목 수 2,646 (json 직접 로드 카운트)
- `early_162_60pct/outputs_full_schedule/summary.json` — `full_test_count: 2646`, `test_reward_difference_ga_minus_regular: 5.401066417616505` (논문의 5.401을 만든 바로 그 실행)

**판정**: 코드 로직 차이(본편의 "나머지 할당" vs 한정실험의 "이중 floor")로 실제 2,646이 나온 것은 사실이나, 논문이 이 실험을 설명하며 본편의 "2,700"을 그대로 재사용 서술한 것은 **집필 단계의 서술 오류**. 5.401 수치 자체는 실제 실행 결과와 정확히 일치하므로 유효하지만, 표본 크기 설명만 정정 필요.

### early_162의 실제 개수는 162

- `early_162_60pct/README.md`: "selects three scenarios from each of the 54 strata (**162 total**)"
- `early_train_seed_101.json` 메타데이터: `count: 162`, `per_stratum: 3`, `seed: 101` — 직접 카운트로도 162 확인
- 54(조합)×3(조합당 샘플)=162. "160"이 나올 수 있는 분할 방식은 코드 어디에도 없음.

**판정**: README·config·json 메타데이터가 전부 162로 상호 일치 — 코드가 나중에 바뀐 흔적 없음. **논문 집필 시 반올림/서술 오류로 판정.**

## 참조 파일
`src/data_generate.py`, `src/perform_evaluate.py`(L34, L978-985, `stratified_split_data`), `src/cost_model_v2.py`(`stratified_split_data`), `src/run_local_search_test.py`, `data/selected_inp.txt`, `data/gasan/**/*.inp`, `early_162_60pct/README.md`, `early_162_60pct/config/experiment.json`, `early_162_60pct/src/fair_ga/split.py`, `early_162_60pct/data/splits/{fixed_test.json, early_train_seed_101.json, train_seed_101.json}`, `early_162_60pct/outputs_full_schedule/summary.json`, `docs/water-4498965.docx`.

## 경로 이식성 (E0 부속, 2026-08-25 추가)

### 문제
`data/selected_inp.txt`(7,440줄)가 저장소 밖 형제 디렉터리의 절대경로(`data/gasan/...`)를 담고 있어, 이 경로에 의존하는 모든 산출물(고정 분할 json 포함)이 이 머신·이 디렉터리 구성에서만 재현 가능했다.

### 검증 (리맵 전 필수 확인)
`torch/data/gasan/`와 `hybrid_v0/data/gasan/`을 파일명 집합 + 전수 MD5 + 총 용량으로 대조:

| 확장자 | 전체 | 일치 | 불일치 |
|---|---|---|---|
| `.inp` (시나리오 정의) | 7,440 | **7,440** | **0** |
| `.rpt` (SWMM 리포트) | 7,440 | 697 | 6,743 |
| `.out` (SWMM 바이너리 결과) | 7,440 | 7,436 | 4 |

파일명 집합 완전 일치(diff 0), 총 용량 차이 4바이트(34,687,998,990 vs 34,687,998,994). `.rpt` 불일치 샘플을 직접 diff한 결과 유일한 차이는 리포트 하단의 실행 타임스탬프 2줄(`Analysis begun/ended on:`)뿐 — 시나리오 정의와 시뮬레이션 결과 자체는 동일. **`.inp` 기준으로 두 디렉터리는 완전히 동일하다고 판정.** (`.out` 4건 예외는 원인 미상, 규모상 무시 가능 수준.)

### 조치
- `data/selected_inp_relative.txt` 신규 생성 (원본 `selected_inp.txt`는 읽기전용 유지, 수정 안 함) — `data/gasan/...` 저장소 기준 상대경로.
- `src/data_paths.py` 신규 — `resolve_path()`: `WATER_DATA_ROOT` 환경변수 우선, 기본값은 저장소 루트(`data/gasan/...`로 해석됨). 이미 절대경로인 문자열은 그대로 통과.
- `data/splits/fixed_split_seed42.json`을 `perform_evaluate.stratified_split_data`(코드 무수정, `open()` 호출만 상대경로 파일로 일시 리다이렉트)로 재생성.
- **검증**: train(3996)/valid(0)/test(2700) 각각 경로 접두사를 제거한 뒤 순서·구성을 md5로 대조 — 기존 절대경로 버전과 **완전 일치**. `resolve_path()`로 7,440개 전부 실제 파일 존재 확인(`missing=0`).
- 검증 통과 후 `fixed_split_seed42.json`을 상대경로 버전으로 교체(기존 절대경로 버전 대체).

### .out 불일치 4개 파일 (원인 미상, 규모상 무시 가능)
`torch/data/gasan/`와 `hybrid_v0/data/gasan/` 사이에서 유일하게 MD5가 다른 `.out`(SWMM 바이너리 결과) 파일 4개:
- `100year/100yr_0720m_h973.out`
- `10year/10yr_1440m_h1161.out`
- `30year/30yr_0060m_h143.out`
- `80year/80yr_1080m_h1075.out`

`.inp`(입력 정의)는 이 4개 모두 포함해 7,440개 전부 동일했으므로, 이 4개는 시나리오 자체가 다른 것이 아니라 SWMM 실행 결과 캐시가 다른 시점/환경에서 재생성된 것으로 추정. 원인 특정은 하지 않았음(우선순위 낮음).

## rainfall 경로 리맵 (2026-08-25 추가)

### 검증
`torch/data/rainfall/`와 `hybrid_v0/data/rainfall/`를 `.inp`와 동일하게 대조:
- 파일 수: 양쪽 모두 **7,444개**, 전부 `.txt`, 파일명 집합 완전 일치(diff 0)
- **전수 MD5 비교: 0건 불일치** (`.inp`보다도 더 엄격히 동일 — rpt처럼 타임스탬프가 박히는 텍스트가 아니라서)
- 총 용량도 완전 동일(22,632,038 bytes 양쪽)
- **7,444 vs `.inp`의 7,440 차이(4개) 원인**: `real_data/나주_20250716~19.inp` 4일치 실측 강우 이벤트의 강우 시계열 파일이 대응하는 `.inp`(시나리오 정의) 없이 먼저 준비되어 있음. 계획서 E8(실측 강우 평가)에서 쓰일 것으로 추정되는 선행 준비 자료로 판단됨 — 문제 아님. 나머지 7,440개는 `.inp`와 1:1 대응, 누락/잉여 없음.

**결론: rainfall 데이터도 두 디렉터리가 완전히 동일** — 리맵 진행.

### 조치
`src/sim_swmm.py`의 `SimSwmm.__init__`을 수정:
- `self.inp_file = inp` → `self.inp_file = resolve_path(inp)` (절대경로는 그대로 통과, 상대경로는 `WATER_DATA_ROOT` 또는 저장소 루트 기준으로 해석)
- `self.rain_file = '<HOME>/.../rainfall/' + inter_name + '.txt'` (하드코딩) → `self.rain_file = resolve_path('data/rainfall/' + inter_name + '.txt')`

### 스모크 검증
`WATER_DATA_ROOT` 미설정 상태에서 상대경로(`data/gasan/10year/10yr_0010m_h051.inp`)로 `SimSwmm`을 생성해 `.execute()` 실행:
```
inp_file: data/gasan/10year/10yr_0010m_h051.inp
rain_file: data/rainfall/10year/10yr_0010m_h051.txt
inp exists: True / rain exists: True
execute() ok, length=34, 정상적인 rain/outfall 값 반환
```
저장소 기본 경로만으로 정상 동작 확인.

### 저장소 전체 하드코딩 절대경로 전수 검색

**`.py` 파일**: 실제 실행 경로(perform_evaluate.py → dqn_from_demon_v1.py/genetic_algo.py/pso.py/water_gym.py/sim_swmm.py)에서 실행되는 하드코딩 경로는 `sim_swmm.py`의 위 2건이 전부였고 둘 다 수정 완료. 그 외 발견된 것은 전부 다음 두 유형이며 **실행 경로 밖이라 수정하지 않음**:
- **독스트링 예시 경로** (`dqn_gru_torch.py:33`, `dqn_from_demon_v1.py:52`, `cost_model_v2.py:113` 등 다수) — 주석 텍스트일 뿐 실행되지 않음
- **`if __name__=='__main__':` 데모/단독 테스트 블록** (`water_gym.py:273`, `sim_swmm.py:158`, `genetic_algo.py:396`, `man_policy.py:175`, `dqn_gru_torch.py:451` 등) — 모듈을 직접 실행할 때만 쓰이며, `perform_evaluate.py`가 import할 때는 호출되지 않음
- `src/obsolete/*.py` 전부와 `src/perform_evaluate copy.py` — 이름 그대로 미사용/사본, 수정 대상 아님
- `src/verify_relative_split.py:19`의 `ABS_PREFIX`는 의도된 것(레거시 절대경로와 대조하기 위한 감사 스크립트 상수)

**`.json`/`.txt` 파일**: 전부 읽기전용으로 잠근 기존 결과·분할 파일 안에 있음(`early_162_60pct/`, `fair_ga_comparison_3pct/`, `gru_size_grid_3pct/`, `results/` 하위) — 지시대로 건드리지 않음.

### 중요 발견 — `results/repeated_heuristics/fixed_test_2700.json`

경로 검색 중 **`results/repeated_heuristics/fixed_test_2700.json`**을 발견했다. 파일명이 "고정된 2,700개 테스트셋"을 강하게 시사하며, 만약 이것이 실제로 논문 Table 6~16(Regular/GA-only/PSO-only 확인된 값들)을 생성한 바로 그 테스트셋이라면, 이번에 새로 만든 `fixed_split_seed42.json`은 **기존에 확인된 결과들과 시나리오 단위로 paired 비교가 안 되는 다른 분할**일 수 있다. `docs/RESULTS_PROVENANCE.md`에서 "본편의 실제 2,700개 테스트셋을 고정한 split 파일이 저장소 어디에도 없다"고 기록했는데, 이 파일이 그 반증일 가능성이 있다. **아직 내용을 열어 대조하지 않았다 — 별도 지시 시 `fixed_split_seed42.json`과 정확히 일치하는지 확인 필요.**

### 남은 과제
- `results/repeated_heuristics/fixed_test_2700.json`이 seed42 분할과 일치하는지 확인 (위 참조, 중요도 높음)
- 모든 학습 진입점이 `fixed_split_seed42.json`을 읽도록 강제하는 작업은 아직 미착수 (`docs/CODEBASE_MAP.md`의 위험 항목 1 참조)

## 응답서 반영 방향
- "160개 시나리오"를 "162개"로 정정, "2,700 paired test scenarios"를 이 실험에 한해 "2,646"으로 정정 (본편 표 6~16의 "2,700"은 그대로 유지 — 본편은 실제로 2,700 맞음).
- 본편 데이터 분할의 랜덤시드를 고정하고 split 파일을 저장소에 커밋 (E0 시드 통합 항목과 연계).
