# Cliff's δ 부호 규약 통일 (2026-09-02)

**채택 규약**: "Cliff's δ < 0 indicates lower values for the first group."
(δ = P(첫 번째 그룹 > 두 번째 그룹) − P(첫 번째 그룹 < 두 번째 그룹).)

## 1. 산출 코드 5개 전수 확인

| 파일 | `cliffs_delta(x,y)` 공식 | 호출부 인자 순서(x=첫 번째 그룹인가) | 판정 |
|---|---|---|---|
| `src/analyze_e03.py`(→`E03_paired_tests.csv`, `E03_stats.md`, `E03_FINDINGS.md`) | Mann-Whitney U 기반, `δ=2U/(n·m)-1`(표준 공식과 수학적으로 동일) | `paired_test(df, model_a, model_b, metric)` → `cliffs_delta(a=model_a, b=model_b)` — model_a가 x | **정상** |
| `src/run_e04f_taskI_stats.py`(→`E04E_REPORT.md`, `E04F_REPORT.md`) | `more=Σ[xi>yi]`, `less=Σ[xi<yi]`, `δ=(more-less)/(nx·ny)`(표준 공식) | `add_row(label, n, seeds, x, y)`, 라벨이 "L1 vs L2"면 x=L1(첫 번째 명시 그룹) | **정상** |
| `src/run_e05_analyze.py`(→`E05_stage1_tests.csv`, `E05_REPORT.md`, `E05_SUMMARY_TABLE.md`) | 위와 동일(표준 공식) | `paired_test(cond_a, cond_b, label)` → `cliffs_delta(a=cond_a, b=cond_b)`, 라벨 "A3-A1"이면 A3가 x | **정상** |
| `src/run_e12_diff_analysis.py`(→`E12_diff_analysis.md`, `E12_abstract_numbers.md`, `docs/E12_REPORT.md`) | **버그**: `less`/`more` 누적 변수가 뒤바뀌어 있었음(아래 상세) → 실질적으로 `δ = -(P(x>y)-P(x<y))`, 표준 공식의 부호 반전 | 호출부는 x=GA-guided(첫 번째)로 정상 | **버그 확인·수정함** |
| `src/run_long_duration_analysis.py`(→`LONG_DURATION_ANALYSIS.md`) | 위와 동일한 버그(같은 코드를 복사해 작성) | 호출부는 x=GA-guided(첫 번째)로 정상 | **버그 확인·수정함** |

## 2. 버그 상세

`run_e12_diff_analysis.py`·`run_long_duration_analysis.py`의 원래 코드:

```python
def cliffs_delta(x, y):
    ...
    for xi in x:
        lo = np.searchsorted(y_sorted, xi, side='left')   # y 중 xi보다 작은 개수 = "x>y" 쌍 수
        hi = np.searchsorted(y_sorted, xi, side='right')  # y 중 xi 이하인 개수
        less += lo                        # 버그: lo는 "x>y" 쌍인데 less에 누적
        more += len(y_sorted) - hi        # 버그: 이 값은 "x<y" 쌍인데 more에 누적
    return (more - less) / (len(x) * len(y))
```

`lo`(y<xi인 개수)는 실제로 "x>y" 쌍의 수인데 변수명 `less`에 더해지고,
`len(y_sorted)-hi`(y>xi인 개수, "x<y" 쌍의 수)가 변수명 `more`에
더해졌다 — 즉 두 누적값이 이름과 반대로 뒤바뀌어 있었다. 결과적으로
`(more-less)/(nx·ny)`가 표준 공식 `(P(x>y)-P(x<y))`의 **부호 반전값**을
반환했다. 다른 세 파일은 `more`/`less`를 올바른 조건(`xi>yi`/`xi<yi`)에
직접 누적해 문제가 없었다.

**수정**: `more += lo`(x>y), `less += len(y_sorted)-hi`(x<y)로 누적
대상을 맞바꿔 표준 공식과 일치시켰다.

## 3. 변경된 값 목록 (파일, 항목, 이전값 → 새값)

| 파일 | 항목 | 이전값 | 새값 |
|---|---|---|---|
| `results/summary/E12_diff_analysis.md` §3 | 전체(n=8,100) Cliff's δ | +0.0082 | **−0.0082** |
| `results/summary/E12_diff_analysis.md` §3 | 하위5% 제외(n=7,695) Cliff's δ | +0.0019 | **−0.0019** |
| `docs/E12_REPORT.md` "시나리오 단위" 표 | Cliff's δ | +0.0082 | **−0.0082** |
| `docs/E12_REPORT.md` "원고 값과의 대조" | Cliff's δ | +0.0082 | **−0.0082** |
| `results/summary/E12_abstract_numbers.md` "시나리오 단위" 표 | Cliff's δ | +0.0082 | **−0.0082** |
| `results/summary/E12_abstract_numbers.md` "주의" 문단 | Cliff's δ(전체/하위5%제외) | +0.0082 / +0.0019 | **−0.0082 / −0.0019** — "부호가 다르다"는 서술 자체를 철회 |
| `results/summary/E12_abstract_numbers.md` "원고 값과의 대조" | Cliff's δ | +0.0082 | **−0.0082** |
| `results/summary/LONG_DURATION_ANALYSIS.md` §1(9개 지속시간) | max_level Cliff's δ(9행) | +0.6956 ~ +0.2838(전부 양수로 오기재) | **−0.6956 ~ −0.2838**(전부 음수) |
| `results/summary/LONG_DURATION_ANALYSIS.md` §1 | n_dryrun Cliff's δ(9행) | −0.1904 ~ −0.1024(전부 음수로 오기재) | **+0.1904 ~ +0.1024**(전부 양수) |
| `results/summary/LONG_DURATION_ANALYSIS.md` §1 | n_switches Cliff's δ(9행) | 부호 반전 상태 | **부호 정정**(예: 60분 −0.0897→**+0.0897**, 1440분 −0.3523→**+0.3523** 등, 표 전체) |
| `results/summary/LONG_DURATION_ANALYSIS.md` §2 | 짧음(<1080분) max_level/dry-running/switches Cliff's δ | +0.4473 / −0.1344 / +0.0525 | **−0.4473 / +0.1344 / −0.0525** |
| `results/summary/LONG_DURATION_ANALYSIS.md` §2 | 긺(≥1080분) max_level/dry-running/switches Cliff's δ | +0.3178 / −0.0762 / +0.3092 | **−0.3178 / +0.0762 / −0.3092** |

**영향받지 않은 값**(부호가 이미 정확했음, 재확인만 완료, 수정 없음):
`docs/E03_FINDINGS.md`, `results/summary/E03_stats.md`,
`results/summary/E03_paired_tests.csv`, `docs/E04E_REPORT.md`,
`docs/E04F_REPORT.md`, `docs/E05_REPORT.md`, `docs/E05_SUMMARY_TABLE.md`,
`results/summary/E05_stage1_tests.csv`. mean diff·Wilcoxon p·표본 크기 등
**Cliff's δ가 아닌 모든 수치는 이번 정정과 무관하며 변경되지 않았다**
(버그는 `cliffs_delta()` 함수에만 있었고, 다른 통계량은 별도 코드로
계산됨).

## 4. 사용자가 지적한 원 사례 재확인

- `docs/E03_FINDINGS.md`: GA-guided가 Regular보다 max_level 낮음(우세),
  δ=−0.417 — **정상 값이었음**(수정 없음).
- `results/summary/LONG_DURATION_ANALYSIS.md` §2, 짧은 구간(<1080분)
  max_level: mean diff=−0.2130(GA-guided 낮음, Regular보다 우세),
  δ는 **+0.4473(버그) → −0.4473(정정)**으로 이제 위 E03_FINDINGS 사례와
  **부호·방향이 일치한다**(둘 다 GA-guided가 낮을 때 δ<0).

## 5. 추가 점검 사례 (2026-09-03) — `docs/E04F_REPORT.md` Task H, δ=+0.352

응답서에 "완전예지 계획기(ENUM-L5)가 학습 정책(GA-guided)보다 높은
최고수위를 보였다"로 쓸 예정인 근거값. 이 값을 산출한 스크립트를
저장소에서 찾을 수 없었다(`run_e04f_taskH.py`/`run_e04e_taskE.py`는
데이터(`foresight_compare_w2.csv`)만 생성하고 `cliffs_delta`를
호출·정의하지 않음 — Task H의 δ 계산은 사전에 저장된 스크립트가
아니라 일회성 계산으로 산출된 것으로 보인다).

**원본 데이터로부터 직접 재계산**(`results/E04_enum/
foresight_compare_w2.csv`의 L=5·forecast=perfect 270행을
`results/E03_seeds/GA-guided/seed{1..5}/test_metrics.csv`의 시나리오별
5시드 평균과 scenario_id로 병합, §1의 "정상" 판정을 받은 표준 공식
`more=Σ[xi>yi], less=Σ[xi<yi], δ=(more-less)/(nx·ny)` 사용, 첫 번째
그룹 x=ENUM-L5(완전예지), 두 번째 그룹 y=GA-guided):

| | 재계산값 | 원 보고값(`E04F_REPORT.md`) |
|---|---|---|
| mean diff(ENUM−GA-guided) | +0.068275 | +0.068 |
| Wilcoxon p | 2.911×10⁻⁸ | 2.9×10⁻⁸ |
| Cliff's δ | **+0.352126** | **+0.352** |

**세 값 모두 정확히 일치한다.** 부호 확인: 첫 번째 그룹(ENUM-L5)의
mean이 GA-guided보다 높음(mean diff>0, "ENUM이 나쁨") → 규약("δ<0이면
첫 번째 그룹이 낮음")에 따라 δ는 양수여야 하며, 실제로 **δ=+0.352로
부호가 규약과 일치한다.** **버그 없음 — 재계산·수정 불필요.**
