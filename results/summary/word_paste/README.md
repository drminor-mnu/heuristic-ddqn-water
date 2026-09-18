# Word/Excel 삽입용 표 파일 안내

이 디렉터리는 원고에 들어갈 표 22개(Table 5, Table 6~10, Table 11~15,
Table 16, Appendix Table A1~A10)를 Word 원고에 붙여넣기 쉬운 형태로 내보낸
CSV/TXT 파일이다. 값은 `results/summary/table6_10_main/`,
`table6_10_sd/`, `table11_15_main/`, `table11_15_sd/`의 확정본과
동일하며(소수점까지 일치), 형식만 CSV/TSV로 바꾼 것이다. **원고 docx
자체는 수정하지 않았다** — 이 파일들을 참고해 사용자가 직접 Word에
붙여넣어야 한다. (별도로 `table16_full.csv/.txt`가 있는데, 이는
"22개"에 포함된 Table 16과 같은 표의 **이전 19열 시안**을 본문 서술
참조용으로 보존한 파일이다 — 원고에 들어갈 표로 세지 않는다.)

---

## 1. 파일 구성

각 표마다 `.csv`(쉼표 구분)와 `.txt`(탭 구분) 두 벌이 있다. 내용은
동일하고 구분자만 다르다 — Excel에서 열 때는 `.csv`, Word의
"텍스트를 표로 변환"에 쓸 때는 `.txt`를 권장한다.

| 파일명 | 표 |
|---|---|
| `table05.csv/.txt` | Table 5 (학습 요약: 가중치, 학습시간, 결정시간) |
| `table06.csv/.txt` ~ `table10.csv/.txt` | Table 6~10 (본문, 2모델) |
| `table11.csv/.txt` ~ `table15.csv/.txt` | Table 11~15 (본문, 5모델) |
| `table16.csv/.txt` | Table 16 (5모델 전체 요약, U(π) 포함, mean (sd) 셀 병기, 7열) |
| `table16_full.csv/.txt` | Table 16 이전 시안(19열, 평균·SD·95%CI 별도 열 — 본문 CI 참조용 보존) |
| `tableA1.csv/.txt` ~ `tableA5.csv/.txt` | Appendix Table A1~A5 (Table 6~10의 5시드 표준편차) |
| `tableA6.csv/.txt` ~ `tableA10.csv/.txt` | Appendix Table A6~A10 (Table 11~15의 5시드 표준편차) |

**인코딩**: 전 파일 UTF-8 with BOM(`utf-8-sig`)으로 저장했다. Excel은
BOM이 있어야 UTF-8을 자동 인식한다(없으면 `±` 같은 특수문자가
깨진다). 메모장이 아니라 Excel/Word로 직접 열면 문제없다.

---

## 2. 표 구조 공통 사항 (Table 6~15, A1~A10)

- 열: `Model, Duration (min), 10, 20, 30, 50, 80, 100` — 뒤 6개 열이
  재현기간(년) 6종.
- 행: 지속시간(min) 9종(60/120/180/240/360/540/720/1080/1440) ×
  모델(2종 또는 5종)을 반복. **원고 표에서는 Duration 열이
  지속시간별로 병합된 셀이지만, 이 CSV/TXT에는 병합 없이 매 행마다
  값이 반복돼 있다** — Excel/텍스트 파일은 병합 셀을 표현할 수
  없기 때문이다. Word에 붙여넣은 뒤 병합은 아래 "권장 작업 순서"를
  따르면 필요 없고(기존 표에 값만 붙여넣으므로), "대안 작업 순서"를
  따를 경우 3-4단계에서 수동으로 병합해야 한다.
- 모델 표기: 2모델 표(Table 6~10, A1~A5)는 `Regular, GA-guided` 순서
  18행. 5모델 표(Table 11~15, A6~A10)는
  `Regular, GA-guided, PSO-guided, GA, PSO` 순서 45행.
- 소수점 자리수: **원고 원본과 동일하게 지표별로 다르다.** 최고
  수위(max_level_m, Table 6/11, A1/A6)는 소수점 3자리, 나머지 4개
  지표(n_switches, n_intervals_100, n_intervals_170, n_dryrun_proxy —
  Table 7~10/12~15, A2~A5/A7~A10)는 소수점 2자리. 아래 3절 표에
  표별로 명시했다.

Table 5·Table 16은 위 구조와 다르다. Table 5는 지속시간·재현기간
그리드가 아니라 모델별 1행(2행)에 가중치 4개 + 학습시간(mean±sd) +
결정시간 열이다. Table 16은 5모델 전체(지속시간·재현기간 평균)
요약표로, **원본 지표 5개(Model 제외)에 신규 Pump-use index U(π)
1열만 추가한 7열**이며, 각 셀은 "평균 (표준편차)" 형식으로
병기한다(예: `6.224 (0.077)`) — 95% CI 열은 제외했다(사용자 결정,
2026-09-08: n=5에서 CI가 sd로부터 직접 계산돼 정보가 중복되고 폭이
넓어 오독 우려). 평균·SD·CI를 전부 별도 열로 분리했던 이전 19열
시안은 `table16_full.csv`로 보존했다. 값의 출처·의미는
`docs/response/SECTION_3_3_PACKAGE.md` §3(원고 원본 구조 확인 포함)
참조.

---

## 3. 표별 제원

| 표 | 지표 | 모델 수 | 값 영역 행 수 | 열 수 | 소수점 | CSV 경로 |
|---|---|---|---|---|---|---|
| Table 5 | 학습 요약(가중치·시간) | 2 | 2 | 7 | 시간 1자리, 결정시간 4자리 | `table05.csv` |
| Table 6 | max_level_m | 2 | 18 | 6 | 3 | `table06.csv` |
| Table 7 | n_switches | 2 | 18 | 6 | 2 | `table07.csv` |
| Table 8 | n_intervals_100 | 2 | 18 | 6 | 2 | `table08.csv` |
| Table 9 | n_intervals_170 | 2 | 18 | 6 | 2 | `table09.csv` |
| Table 10 | n_dryrun_proxy | 2 | 18 | 6 | 2 | `table10.csv` |
| Table 11 | max_level_m | 5 | 45 | 6 | 3 | `table11.csv` |
| Table 12 | n_switches | 5 | 45 | 6 | 2 | `table12.csv` |
| Table 13 | n_intervals_100 | 5 | 45 | 6 | 2 | `table13.csv` |
| Table 14 | n_intervals_170 | 5 | 45 | 6 | 2 | `table14.csv` |
| Table 15 | n_dryrun_proxy | 5 | 45 | 6 | 2 | `table15.csv` |
| Table 16 | 5개 지표+U(π), 전체요약(mean (sd) 셀 병기) | 5 | 5 | 7(원고 6+U(π) 1) | Water Level 3, 나머지 2(U 포함) | `table16.csv` |
| Table 16(보존용) | 상동, 평균·SD·CI 전부 별도열 | 5 | 5 | 19(Model 1+평균6+SD6+CI95 6) | 상동 | `table16_full.csv` |
| Appendix A1 | max_level_m (sd, Table 6) | 2 | 18 | 6 | 3 | `tableA1.csv` |
| Appendix A2 | n_switches (sd, Table 7) | 2 | 18 | 6 | 2 | `tableA2.csv` |
| Appendix A3 | n_intervals_100 (sd, Table 8) | 2 | 18 | 6 | 2 | `tableA3.csv` |
| Appendix A4 | n_intervals_170 (sd, Table 9) | 2 | 18 | 6 | 2 | `tableA4.csv` |
| Appendix A5 | n_dryrun_proxy (sd, Table 10) | 2 | 18 | 6 | 2 | `tableA5.csv` |
| Appendix A6 | max_level_m (sd, Table 11) | 5 | 45 | 6 | 3 | `tableA6.csv` |
| Appendix A7 | n_switches (sd, Table 12) | 5 | 45 | 6 | 2 | `tableA7.csv` |
| Appendix A8 | n_intervals_100 (sd, Table 13) | 5 | 45 | 6 | 2 | `tableA8.csv` |
| Appendix A9 | n_intervals_170 (sd, Table 14) | 5 | 45 | 6 | 2 | `tableA9.csv` |
| Appendix A10 | n_dryrun_proxy (sd, Table 15) | 5 | 45 | 6 | 2 | `tableA10.csv` |

"값 영역 행 수"는 헤더(모델명, Duration (min), 재현기간 6종)를 제외한
데이터 행 수, "열 수"는 재현기간 6종(값이 들어가는 열, 왼쪽 두 식별
열은 제외)이다. 예: Table 6은 18행 × 6열의 값 영역. **Table 5·16은
이 그리드 구조가 아니므로 행·열 수를 별도로 표기했다** — Table 16은
5행(모델 1개당 1행) × 19열(Model 열 포함, 원본 6개 지표+U(π)의
평균/SD/95%CI를 각각 별도 열로 표기)이다.

---

## 4. 권장 작업 순서 — 기존 Word 표에 값만 붙여넣기 (권장)

**이 절은 Table 6~15/A1~A10(지속시간×재현기간 그리드 구조)에만
해당한다. Table 5·Table 16은 구조가 달라 별도 안내(§6) 참조.**

원고에는 이미 Table 6~15/A1~A10 형태의 표 틀(제목행, Duration 병합
셀 포함)이 있다고 가정한다. 이 방법은 병합 셀을 건드리지 않는다.

1. `tableNN.csv`를 Excel에서 연다(더블클릭, 또는 Excel 실행 후
   파일 열기 — 인코딩 문제 없이 바로 열린다).
2. **값 영역만** 선택한다 — Model/Duration 열(왼쪽 두 열)은 제외하고,
   재현기간 6개 열 × 해당 지표의 값 영역 행 수만 드래그해 선택한다.
   (예: Table 6은 18행 × 6열의 숫자 영역만 선택 — 3절 표 참고.)
3. 복사(Ctrl+C).
4. 원고 Word 파일에서 해당 표의 첫 번째 값 셀(모델·Duration 다음,
   재현기간 "10년" 열의 첫 행)을 클릭한다.
5. 붙여넣기(Ctrl+V) 후 붙여넣기 옵션에서 **"대상 서식에 맞추기(Match
   Destination Formatting)"**를 선택한다 — 그래야 원고 표의 글꼴·
   테두리·정렬이 유지되고 Excel 서식이 섞여 들어가지 않는다.

이 방법은 Duration 병합 셀과 표 틀을 그대로 유지하므로 별도의 병합
작업이 필요 없다.

---

## 5. 대안 작업 순서 — 텍스트를 표로 변환 (표 틀이 없을 때)

원고에 표 틀 자체가 없거나 통째로 다시 만들어야 하는 경우:

1. `tableNN.txt`(탭 구분)를 텍스트 편집기 또는 Word에서 연다.
2. 전체 내용을 복사해 Word 문서에 붙여넣는다(순수 텍스트로,
   서식 없이 — "붙여넣기 옵션 → 텍스트만 유지").
3. 붙여넣은 텍스트 전체를 선택한 상태에서 **삽입 → 표 → 텍스트를
   표로 변환**을 실행하고, 구분 기준을 "탭"으로 지정한다. 이 표는
   Model, Duration (min), 10, 20, 30, 50, 80, 100의 8열짜리 표가
   되며, Duration 열은 병합되지 않고 매 행 반복된 상태다.
4. Duration 열 병합(선택 사항, 원고 형식과 맞추려면 필요): 같은
   지속시간 값을 가진 연속된 행들(2모델 표는 2행씩, 5모델 표는
   5행씩)의 Duration 셀을 각각 블록 선택한 뒤 **표 도구 → 레이아웃 →
   셀 병합**을 지속시간 9개 그룹마다 반복한다(2모델 표는 9회,
   5모델 표는 9회).
5. Model 열이 필요 없는 경우(원고가 Duration 병합 후 모델명만 다시
   쓰는 형식이라면) 삭제하거나, 필요하면 그대로 둔다 — 원고의 기존
   Table 6/11 서식을 기준으로 판단한다.

---

## 6. Table 5·Table 16 작업 안내(그리드 표와 다른 구조)

### Table 5

기존 원고 표(2행×7열: Model, w1-w4, Training Time, Decision Time)와
`table05.csv`의 열 구성이 그대로 대응한다. §4의 방법을 그대로 써도
되지만, Duration 병합이 없는 단순한 표이므로 전체(2행×7열)를 그대로
복사해 붙여넣어도 된다.

### Table 16 (2026-09-08 형식 확정 — 7열, mean (sd) 병기)

**형식 확정**: 95% CI 열은 제외하고(n=5에서 CI는 sd로부터 직접
계산돼 정보가 중복되고 폭이 넓어 오독을 부를 수 있다는 판단, 모델
간 비교는 본문의 대응검정·효과크기로 제시), 표준편차는 별도 열이
아니라 **각 셀 안에 "평균 (표준편차)" 형태로 병기**한다(예: `6.224
(0.077)`). 그 결과 `table16.csv`는 **원고 기존 표(6열)에 Pump-use
index U(π) 열 하나만 추가한 7열** 구조다 — Model, Maximum water
level, Number of on/off changes, 100 m³/min operating intervals,
170 m³/min operating intervals, Pump-use index U(π), Dry-running risk
proxy. 소수점은 최고수위 3자리, 나머지(U(π) 포함) 2자리.

**이전 19열 시안(평균·SD·95%CI를 전부 별도 열로 분리)은
`table16_full.csv`/`table16_full.txt`로 보존**했다 — 본문 서술에서
CI 수치가 필요할 때만 참조하고, Word 표 붙여넣기에는 쓰지 않는다.

**권장 절차(원고 표에 U(π) 열 하나만 삽입)**:
1. 원고 Word 파일에서 Table 16의 마지막 열(Dry-Running Proxy Count)
   오른쪽에 새 열을 하나 삽입한다(표 도구 → 레이아웃 → 오른쪽에
   삽입, 헤더 행 포함 총 6행).
2. 새 열 헤더 셀에 "Pump-use index U(π)"를 입력(또는 `table16.csv`
   헤더 6번째 열 값을 그대로 복사).
3. `table16.csv`를 Excel에서 열어 **U(π) 열(6번째 열)의 값 5개만**
   선택해 복사, 원고 표의 새로 만든 열 아래 5개 셀에 붙여넣는다
   ("대상 서식에 맞추기"로).
4. 나머지 5개 열(Maximum water level, Number of on/off changes,
   100/170 m³/min operating intervals, Dry-running risk proxy)은
   원고 표에 이미 있는 값을 `table16.csv`의 해당 열 값으로 **덮어쓸지
   그대로 둘지 확인** — `table16.csv`의 값은 mean (sd) 병기 형식인데
   원고 원본은 평균 단일값만 표기(§3 확인)이므로, 표준편차까지 표에
   넣을지는 사용자 판단이 필요하다(넣기로 했다면 원고의 기존 5개 열
   전체를 `table16.csv`의 해당 값으로 교체).
5. 대안으로 `table16.txt`를 "텍스트를 표로 변환"하면 7열짜리 표가
   통째로 생성된다 — 이 경우 §5 방식대로 Model 열 유지 여부만
   판단하면 된다.

이 방법은 열 삽입 1회 + 값 붙여넣기만 필요해 19열 시안보다 훨씬
간단하다.

---

## 7. 주의 사항

- 이 디렉터리의 CSV/TXT는 **결과 파일이며 원고 docx가 아니다**.
  Word에 실제로 반영하는 것은 사용자가 직접 수행한다.
- 값 자체(소수점 포함)는 `results/summary/table6_10_main/`,
  `table6_10_sd/`, `table11_15_main/`, `table11_15_sd/`의 `.md`
  파일과 완전히 동일하다 — 이 CSV/TXT를 다시 계산하지 않았고, 같은
  소스 CSV(`results/summary/E14_table_*.csv`,
  `results/summary/E03_table_*.csv`)에서 동일한 스크립트로
  생성했다.
- Table 8은 원고 현재본에서 세 행이 완전히 동일한 값 오류가
  있었던 표다(리비전 절차서 E1 대상) — 이 디렉터리의 `table08.csv`는
  재계산된 정상값이며, 원고의 오류값이 아니다.
