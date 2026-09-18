# Abstract 정량 수치 출처 대조 (R2-1, R2-4, R4-4)

> **Cliff's δ 부호규약**: "Cliff's δ < 0 indicates lower values for the
> first group (GA-guided)." 이 문서의 δ=+0.0082 표기는 계산 버그로
> 확인되어 **−0.0082로 정정**했다(2026-09-02,
> `results/summary/DELTA_SIGN_FIX.md`).

원고 Abstract 전문(pandoc 추출, `docs/water-4498965.docx`, 스크래치에만
보관 — 저장소에 커밋하지 않음, 기존 관행과 동일)에서 정량적 수치·주장을
전부 추출했다.

## Abstract의 정량 수치·주장 전체 목록

| # | Abstract 원문(발췌) | 성격 | 현재 데이터로 대체 가능? |
|---|---|---|---|
| 1 | "6,696 rainfall scenarios for the Gasan Stormwater Pumping Station in Seoul" | 데이터 규모(고정값) | **이미 일치** — `data/splits/fixed_split_seed42.json`의 `n_total=6696`(train 3996+test 2700)과 정확히 같음. 재실행 불필요 |
| 2 | "GA-guided DDQN reduced the loss faster than Regular DDQN" | 정성적(학습곡선 비교) | **이미 대체 가능** — `docs/E05_REPORT.md` §5-3 학습곡선(A3=GA-guided/A0=Regular 유사 조건)로 답변 가능. 방향(더 빠른 손실 감소) 재확인은 학습곡선 그림 직접 판독 필요, 이 문서에서 수행하지 않음 |
| 3 | **"In a limited-data experiment using only 160 training scenarios, the mean reward over 2,700 paired test scenarios was 5.401 higher"** | **정량(단일 수치, 시드 미명시)** | **확정됨(2026-09-01, E12 완료)** — GA-guided(n160)의 평균 누적보상은 Regular(n160)보다 **−0.479 낮음**(+5.401과 부호·크기 모두 불일치). 상세는 "E12 확정 결과" 절 |
| 4 | "GA-guided DDQN reduced the dry-running risk proxy by approximately 16.5% relative to Regular DDQN, whereas PSO-guided DDQN reduced both pump switching and dry-running risk [by ~85.0%]" | 정량(Table 16/§3.3 인용, 전체 학습 기준) | **이미 대체 가능(실행 완료)** — `results/summary/E03_summary.csv`, `docs/RESULT_DELTA.md` §3. **단, 5시드 평균 기준 원고 수치와 불일치**(GA-guided는 개선이 아니라 열세, PSO-guided는 약 −35%) — `docs/E03_FINDINGS.md` §3.5(R2-7)에 원인 분해 기록됨. 재실행 불필요, 기존 불일치 그대로 보고 대상 |
| 5 | "the heuristic-only policies performed well for some operational indicators, their test-time computational cost increased substantially" | 정성적 | **이미 대체 가능** — 이번 배치 `results/summary/E10_realtime_margin.md`가 GA-only/PSO-only 결정당 시간(21ms대)이 DDQN(0.12ms대)의 약 165배임을 실측 확인. "substantially"의 정성적 방향과 일치 |
| 6 | "the trained DDQN policies retained low inference cost as the number of scenarios increased" | 정성적 | **이미 대체 가능** — 같은 E10 실측(DDQN 결정당 0.12~0.13ms, 시나리오 수 무관하게 일정)으로 답변 가능 |

## Abstract 수치 교체안 (원고값 / E3 v2 값 / 출처, 2026-09-01 추가)

| # | Abstract 원고값 | E3 v2(현재) 값 | 출처 파일 |
|---|---|---|---|
| 1 | 6,696 rainfall scenarios | 6,696(train 3,996 + test 2,700) | `data/splits/fixed_split_seed42.json` |
| 2 | "GA-guided reduced loss faster than Regular" (정성) | E5 §5-3 학습곡선상 A3(GA-guided)·A0(Regular) 비교로 판독 가능(수치 재확인 이 문서에서 안 함) | `docs/E05_REPORT.md` §5-3, `results/summary/E05_learning_curves.png` |
| 3 | 160개 학습→2,700개 테스트 평균보상 +5.401 | **−0.479**(부호·크기 불일치) | `results/E12_datasize/{Regular,GA-guided}/n160/seed{1,2,3}/test_metrics.csv` |
| 4a | GA-guided dry-running −16.5%(Regular 대비) | **+47.48%**(감소 아니라 증가 — 부호 반대) | `results/summary/E03_summary.csv`(`n_dryrun_proxy_mean`: GA-guided 4.8725, Regular 3.3039) |
| 4b | PSO-guided dry-running −85.0%(Regular 대비) | **−34.70%** | 동일 파일(PSO-guided 2.1573, Regular 3.3039) |
| 5 | "heuristic-only 계산비용 대폭 증가" (정성) | DDQN 대비 GA/PSO-only 결정당 시간 약 1,600~1,900배(SWMM제외 (c)기준) / 약 6~7배((c+d)기준) — 방향 일치 | `results/summary/E10_realtime_margin.md` |
| 6 | "trained DDQN은 낮은 추론비용 유지" (정성) | DDQN 결정당 0.24~0.28ms((c)기준), 시나리오 수와 무관하게 일정 | `results/summary/E10_unified_table.md` |

**4a·4b의 원고-현재 차이는 학습 시드 변이에 기인함이 이미 규명돼 있다**
— `docs/E03_FINDINGS.md` §3.5(R2-7 대응): GA-guided의 dry-running proxy는
5시드 중 seed4 하나(1440분 시나리오, 76.0 — 나머지 4시드 4.4~12.0)가
전체 평균을 밀어올리며, seed4를 제외하면 GA-guided 1440분 dry-running
(7.19)이 Regular(8.46)보다 오히려 낮아진다. 즉 원고의 "감소" 방향과
현재의 "증가" 방향이 갈리는 것은 **구조적 열세가 아니라 특정 학습
시드의 국소최적 분기**로 이미 분해돼 기록되어 있다(재조사 안 함, 인용만).

## E12 확정 결과 (2026-09-01, 학습 완료)

**실행**: `train_n=160` × `{Regular, GA-guided}` × `seed={1,2,3}` = 6회,
신가중치 `[0.40,0.25,0.25,0.10]`, γ=0.7, 전체 test_inps 2,700개.
무결성 확인(6/6 조합 DONE, `test_metrics.csv` 각 2,700행, `train_log.csv`
160에피소드 완주, `run_meta.json`에 가중치·γ·`train_n_condition=160` 정확
기록) 전부 통과.

### 시드평균 (n=3)

| 조건 | 3시드 평균 cum_reward | 시드별 값 |
|---|---|---|
| Regular(n160) | 153.166 ± 0.560(mean±sd) | seed1 152.552, seed2 153.300, seed3 153.647 |
| GA-guided(n160) | 152.687 ± 1.144(mean±sd) | seed1 153.195, seed2 151.377, seed3 153.490 |

**차이(GA-guided − Regular) = −0.479**(절대값), **−0.31%**(백분율).
시드별 차이: seed1 +0.642, seed2 −1.923, seed3 −0.157(방향 일관성 없음).
시드평균 기준 Wilcoxon(n=3) p=0.75(n=3 최소 가능 p=0.25보다 큼 — 유의성
판단 불가 수준).

### 시나리오 단위 (n=8,100 = 3시드 × 2,700시나리오, scenario_id로 대응)

| 지표 | 값 |
|---|---|
| mean diff(GA-guided − Regular, 대응 시나리오별) | −0.479 |
| Wilcoxon p | 1.008e-34 |
| Cliff's δ | **−0.0082**(거의 0, 무시할 수준; 2026-09-02 부호 정정 — 이전 +0.0082는 계산 버그, `results/summary/DELTA_SIGN_FIX.md`) |
| 방향 일치 쌍 수 | GA-guided가 더 높은 쌍 3,885 / 더 낮은 쌍 4,215(동률 0) |

**정정(2026-09-02)**: 대응쌍 평균차(−0.479)와 Cliff's δ(−0.0082, 전체
교차쌍 기준)는 이제 **부호가 일치한다**(둘 다 GA-guided가 낮은 방향).
이전에 보고했던 δ=+0.0082는 `src/run_e12_diff_analysis.py`의 계산
버그(부호 반전)였음이 확인됐다 — "두 통계량의 부호가 다르다"는 이전
서술은 철회한다. 상세: `results/summary/DELTA_SIGN_FIX.md`.

**분포 확인(2026-09-02, `results/summary/E12_diff_analysis.md`)**: 시나리오별
차이(GA-guided−Regular)의 **중앙값은 −0.040**로 평균(−0.479)보다 0에
훨씬 가깝다(p5=−4.733, p25=−0.976, p75=+0.493, p95=+1.870 — 음의
꼬리가 양의 꼬리보다 김). **하위 5%(405개) 중 397개(98.0%)가 지속시간
1080·1440분**(전체 8,100개 중 1080·1440분 비중은 22.2%)에 몰려 있다
— 재현기간별 비중은 3.4~6.2%로 균등에 가까워 치우침이 없다. 하위 5%
405행은 전부 서로 다른 scenario_id(같은 시나리오가 2개 이상 시드에서
동시에 하위 5%에 들지 않음). **하위 5% 제외 시 평균차는 −0.479 →
−0.171로 줄되 0이 되지는 않고**, Cliff's δ는 −0.0082 → −0.0019로
여전히 작은 음수를 유지한다(평균차와 부호가 계속 일치). 히스토그램:
`results/summary/E12_diff_dist.png`.

### 원고 값(+5.401)과의 대조

원고: GA-guided가 Regular보다 평균보상 +5.401 높음. 이번 재현(n=160,
3시드): GA-guided가 Regular보다 **−0.479 낮음**(부호 반대, 크기도
원고값의 약 1/11). 시드평균 Wilcoxon p=0.75(유의성 판단 불가, n=3),
시나리오단위 Cliff's δ=−0.0082(효과크기 무시할 수준, mean diff와 부호
일치) — **어느 granularity
에서도 원고의 "+5.401, sample efficiency 개선" 주장을 지지하는 근거가
나오지 않았다.** 160개 시나리오 선정 방식(고정분할 앞부분 결정적 160개,
원고의 선정 방식과 동일하다는 보장 없음)이 다를 가능성은 §"실행 명령"
절에 이미 기록됨.

산출: `results/E12_datasize/`. 기본 대응비교 재현 코드는 이 문서 작성
시 1회성 Python 스니펫으로 실행(별도 스크립트 파일 없음). 분포·꼬리
분석은 `src/run_e12_diff_analysis.py`(신규) → `results/summary/E12_diff_analysis.md`,
`E12_diff_dist.png` 참조.

## Abstract 교체 문장 초안 (2026-09-01)

원고 Abstract 원문(pandoc 추출, 문장 단위)과 교체안을 나란히 배치한다.
**교체안은 초안이며, 응답서·원고 반영 여부는 별도 판단 사항이다.**

### 문장 1 (데이터 규모) — 변경 없음

> **원문**: "Evaluation using 6,696 rainfall scenarios for the Gasan
> Stormwater Pumping Station in Seoul showed that..."

**교체안**: 없음 — 6,696과 현재 데이터가 정확히 일치(`data/splits/fixed_split_seed42.json`).

### 문장 2 (손실 감소 속도) — 확인됨, 유지 가능

> **원문**: "...under an identical reward setting, GA-guided DDQN reduced
> the loss faster than Regular DDQN."

**확인**: E3 v2 seed1 `train_log.csv`의 `train_loss` 열, 학습 초반
1/4구간(에피소드 0~999) 평균: Regular 0.0049, GA-guided 0.0038 —
GA-guided가 더 낮다(방향 일치). 전체 구간 평균도 Regular 0.0038,
GA-guided 0.0032로 같은 방향.

**교체안**: 원문 유지 가능. 필요시 수치 병기:
> "...GA-guided DDQN reduced the loss faster than Regular DDQN (mean
> training loss over the first quarter of episodes: 0.0038 vs 0.0049,
> seed 1)."

### 문장 3 (160개 시나리오, +5.401) — 확정, 재작성 필요

> **원문**: "In a limited-data experiment using only 160 training
> scenarios, the mean reward over 2,700 paired test scenarios was
> 5.401 higher, demonstrating improved initial sample efficiency."

**확인**: 재현 결과 GA-guided가 Regular보다 **−0.479 낮음**(3시드 평균,
n=160 학습). 원고의 "+5.401 higher"·"improved sample efficiency" 주장은
이번 재현에서 지지되지 않는다(부호 반대, 크기도 다름). 상세는 "E12 확정
결과" 절.

**교체안**:
> "In a limited-data experiment using only 160 training scenarios
> (3 seeds), the mean reward over 2,700 paired test scenarios did not
> differ meaningfully between GA-guided DDQN and Regular DDQN
> (153.19 vs 153.17 to 151.38 vs 153.30 across seeds; 3-seed mean
> 152.69 ± 1.14 vs 153.17 ± 0.56; scenario-level Cliff's δ = 0.008)."

"demonstrating improved initial sample efficiency"는 이번 재현 결과와
상충하므로 삭제가 필요하다(재작성 여부는 응답서 작성 시점의 판단).

### 문장 4 (dry-running, 전체 학습 기준) — 방향 반전, 재작성 필요

> **원문**: "With the full training set, GA-guided DDQN reduced the
> dry-running risk proxy by approximately 16.5% relative to Regular
> DDQN, whereas PSO-guided DDQN reduced both pump switching and
> dry-running risk."

**E3 v2 5시드 값**(`results/summary/E03_summary.csv`): Regular
3.30±2.75, GA-guided 4.87±4.39(**+47.5%, 감소 아니라 증가**), PSO-guided
2.16±1.09(**−34.7%**, 원문 방향과 일치하나 수치는 다름 — 원문 §3.3 body
text 값 −85.0%와 대조).

**교체안**:
> "With the full training set (5 seeds), GA-guided DDQN's mean
> dry-running risk-proxy count (4.87 ± 4.39) was not lower than Regular
> DDQN's (3.30 ± 2.75); PSO-guided DDQN reduced the mean dry-running
> risk-proxy count by approximately 34.7% relative to Regular DDQN
> (2.16 ± 1.09 vs 3.30 ± 2.75, 5-seed mean ± std)."

GA-guided 관련 "reduced"(개선) 주장은 원문에서 삭제해야 한다 — 부호가
반대다. 원인(seed4 국소최적, 재조사 안 함)은 위 표와
`docs/E03_FINDINGS.md` §3.5 참조.

### 문장 5-6 (계산비용) — 방향 일치, 정량화 가능

> **원문**: "Although the heuristic-only policies performed well for
> some operational indicators, their test-time computational cost
> increased substantially. In contrast, the trained DDQN policies
> retained low inference cost as the number of scenarios increased."

**교체안**(정량 병기, 방향은 원문과 일치하므로 삭제 불필요):
> "...their test-time computational cost increased substantially
> (GA-only/PSO-only: 21.4–21.8 ms per decision vs DDQN: 0.24–0.28 ms per
> decision, excluding the one-time per-episode SWMM cost). In contrast,
> the trained DDQN policies retained low inference cost (well under 1 ms
> per decision against a 120 s control interval) regardless of the
> number of scenarios evaluated."

## 신규 실행이 필요했던 항목 — #3 (완료, 아래는 실행 당시 기록)

- **원고 주장**: 160개 학습 시나리오, GA-guided vs Regular, 2,700개
  테스트 시나리오 평균 보상 차이 +5.401(시드 반복 여부 불명, 원고
  서술상 단일 실행으로 보임).
- **45회(계획서 원안) 대신 최소 조건만 지정**: `train_n=160` ×
  `{Regular, GA-guided}` × `seed={1,2,3}` = **6회 신규 학습**. 나머지
  데이터 크기(80/400/1000/3996)와 PSO-guided는 이번에 실행하지 않는다
  (`docs/REMAINING_WORK.md` 참조 — 36회 전체 실행은 하지 않는다는 지시에
  따름). 3996 조건은 E3 v2 5시드 결과로 이미 대체 가능하므로 애초에
  이 최소 세트에 포함하지 않았다.
- **테스트셋**: 고정분할의 전체 test_inps 2,700개 그대로(원고의 "2,700
  paired test scenarios"와 개수 일치) — 학습 시나리오만 160개로 절단.
- **160개 선정 방식**: `data_paths.load_fixed_split()`의 `train_inps`
  리스트 앞에서부터 160개(결정적, 고정 순서). 원고가 160개를 어떻게
  뽑았는지는 서술이 없어 **동일한 선정 방식이라는 보장은 없다** — 이
  가정을 `src/run_e12_datasize.py` 모듈 docstring에 명시했다.

## 실행 명령 (사용자가 실행 완료, tmux 세션 `e12`)

```bash
cd src
nohup python run_e12_datasize.py --workers 2 > ../results/_log/E12.out 2>&1 &
```

- 재개 가능(이미 완료된 (모델,시드) 조합은 `DONE` 마커로 자동 스킵).
- 산출 경로: `results/E12_datasize/{Regular,GA-guided}/n160/seed{1,2,3}/`(완료).
- 분석 결과는 위 "E12 확정 결과" 절.
