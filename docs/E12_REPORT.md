# E12 — 한정 데이터 실험 재수행 (R2-1, R2-4, R4-4)

> **Cliff's δ 부호규약**: "Cliff's δ < 0 indicates lower values for the
> first group (GA-guided)." 이 문서는 원래 δ=+0.0082로 기재됐으나 계산
> 버그로 밝혀져 아래 **−0.0082로 정정**했다(2026-09-02,
> `results/summary/DELTA_SIGN_FIX.md` 참조).

## 실행 범위와 축소 사유

계획서 원안: 학습 시나리오 수 `{80, 160, 400, 1000, 3996}` ×
`{Regular, GA-guided, PSO-guided}` × 3시드 = 45회(3996 조건 9회는 E3 v2
재사용으로 절감 가능, 순수 신규 최대 36회).

**실제 실행**: `train_n=160` × `{Regular, GA-guided}` × `seed={1,2,3}` =
**6회만**. 축소 사유(2026-09-01 사용자 지시):
- Abstract가 실제로 인용한 정량 수치는 "160개 학습 시나리오, GA-guided
  vs Regular"뿐이다(§ `results/summary/E12_abstract_numbers.md`의
  Abstract 정량 수치 전수 목록 참조) — 다른 데이터 크기(80/400/1000)나
  PSO-guided는 Abstract에 구체적 수치로 인용되지 않는다.
- 3996-시나리오 조건은 E3 v2 5시드 결과로 이미 대체 가능해 애초에
  이 최소 세트에 포함하지 않았다.
- 36회 전체 신규 실행은 하지 않는다는 명시적 지시에 따름.

**테스트셋**: 고정분할의 전체 `test_inps` 2,700개(원고의 "2,700 paired
test scenarios"와 개수 일치) — 학습 시나리오만 160개로 절단.

**160개 선정 방식**: `data_paths.load_fixed_split()`의 `train_inps`
리스트 앞에서부터 160개(결정적, 고정 순서). 원고가 160개를 어떻게
뽑았는지는 서술이 없어 **동일한 선정 방식이라는 보장은 없다**.

## 실행 및 무결성 확인

`src/run_e12_datasize.py`(`run_e03_seeds.py` 헬퍼 재사용, 별도 출력
경로 `results/E12_datasize/`로 E3 v2와 분리). 사용자가 tmux 세션 `e12`에서
직접 실행(2026-09-01 15:45~16:32).

| 확인 항목 | 결과 |
|---|---|
| 6개 (모델,시드) 조합 존재 | 전부 존재 |
| `test_metrics.csv` 행 수 | 전부 2,700행 |
| `train_log.csv` 완주 | 전부 160에피소드(=train_n) 완주 |
| `run_meta.json` 가중치 | 전부 `w1=0.4, w2=0.25, w3=0.25, w4=0.1` |
| `run_meta.json` γ | 전부 0.7 |
| `run_meta.json` `train_n_condition` | 전부 160 |
| `run_meta.json` `data.train_n`/`data.test_n` | 전부 160 / 2700 |

무결성 이상 없음 — 분석 진행.

## 결과 (n=3시드, 신가중치, 학습시나리오 160개)

### 조건별 요약

| 조건 | 3시드 평균 cum_reward | mean±sd | seed1 | seed2 | seed3 |
|---|---|---|---|---|---|
| Regular(n160) | 153.166 | ±0.560 | 152.552 | 153.300 | 153.647 |
| GA-guided(n160) | 152.687 | ±1.144 | 153.195 | 151.377 | 153.490 |

### 대응비교

| 단위 | n | mean diff(GA−Regular) | Wilcoxon p | Cliff's δ |
|---|---|---|---|---|
| 시드평균 | 3 | −0.479 | 0.75(n=3 최소 가능 p=0.25보다 큼 — 유의성 판단 불가) | 계산 안 함(n=3 표본으로 δ 신뢰도 낮음) |
| 시나리오단위(scenario_id 대응) | 8,100(3시드×2,700) | −0.479 | 1.008e-34 | −0.0082(무시할 수준) |

## 원고 서술과의 대조

원고: "the mean reward over 2,700 paired test scenarios was 5.401
higher"(GA-guided > Regular), "demonstrating improved initial sample
efficiency."

이번 재현: GA-guided가 Regular보다 **−0.479 낮음**(3시드 평균) — 부호
반대, 크기도 원고값(+5.401)의 약 1/11. 시나리오단위 Cliff's δ=−0.0082는
효과크기가 사실상 없음을 나타낸다(부호는 mean diff와 일치). 세 시드
중 방향이 일관되지도 않는다(seed1 +0.642, seed2 −1.923, seed3 −0.157).

**이 세션에서 어느 granularity로도 원고의 "+5.401, 표본효율성 개선"
주장을 지지하는 근거를 얻지 못했다.** 원인은 조사하지 않았다(160개
시나리오 선정 방식이 원고와 다를 가능성, 원고가 단일 실행값이었을
가능성 등 여러 후보가 있으나 이 문서에서 판정하지 않는다).

## 산출 파일

- `results/E12_datasize/{Regular,GA-guided}/n160/seed{1,2,3}/{test_metrics.csv,train_log.csv,run_meta.json,DONE,model.pkl}`
- `results/summary/E12_abstract_numbers.md`("E12 확정 결과" 절에 동일 수치)
- `src/run_e12_datasize.py`(드라이버, 신규 실행 없음 — 이미 완료)
