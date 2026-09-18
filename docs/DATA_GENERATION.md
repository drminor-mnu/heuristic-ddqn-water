# 강우 시나리오 생성·섭동(perturbation) 절차 (R1-5)

**대응**: R1-5 전반부(생성 절차 문서화), R4-8. 코드 추적만 수행, 재실행
없음. 소스: `src/data_generate.py`.

## 1. 기본 강우 곡선 — Huff 방법

`generate_rainfall(rainfall, duration, raintype)`(`data_generate.py:180`)이
Huff 4분위(quartile) 다항식 가중치(`weights[raintype-1]`, 6차 다항식
`x, x², ..., x⁶`)를 시간 비율 `x=(i+1)/duration*100`에 적용해 누적강우량을
만든다. `raintype ∈ {1,2,3,4}`(Huff 1~4분위). 음수·역감소 구간은
`confirm_valid()`로 감지해 0 또는 직전 값으로 보정(`data_generate.py:207-219`).

## 2. 재현기간·지속시간 격자

기준표(`write_datafiles()`, `data_generate.py:315-320`, "전주관측소
빈도강우량 참조"): 재현기간 6종(10/20/30/50/80/100년) × 지속시간 9~10종
(10~1440분) × 총강우량(mm) 조견표. 현재 고정분할(`fixed_split_seed42.json`)의
실제 사용 지속시간은 9종(**10분 제외**: `["0060","0120","0180","0240",
"0360","0540","0720","1080","1440"]`)이다.

## 3. 섭동(perturbation) — `more_rains()`

`more_rains(rainfall, n_vars, std_r=0.5)`(`data_generate.py:391-425`)가
기준표의 총강우량(mm)을 평균으로 하는 **정규분포** `N(rainfall, std_r)`에서
샘플을 뽑아(중복 제거 후 재추출, 최대 100회 반복) 추가 시나리오의
총강우량을 만든다. 이 샘플들은 `generate_rainfall()`에 다시 넣어 같은
Huff 분위(raintype)의 다른 크기 강우 곡선을 만든다.

**문서-코드 불일치(사실만 기록, 수정 안 함)**: `write_datafiles()`의
docstring은 "build more rainfall data by using normal distribution with
std rain \* 0.1 (10%)"라고 서술하나, 실제 호출부(`data_generate.py:365`,
`more_rains(rainfall[i][j], n_vars=n_vars)`)는 `std_r`을 넘기지 않아
`more_rains()`의 **기본값 0.5(절대 mm, 총강우량의 10%가 아님)**가 쓰인다.
예: 100년 빈도 1440분 총강우량 290.4mm에서 std=0.5mm은 총강우량의 약
0.17%에 불과해, docstring이 말하는 10%(≈29mm)와 크게 다르다.

## 4. 시드 관리

`data_generate.py:20`에 모듈 레벨로 **`np.random.seed(1234)`가 한 번만
고정**된다. 즉 전체 강우 코퍼스(6,696개 시나리오)가 **하나의 전역 시드
아래 순차 생성**되며, 시나리오별 개별 시드는 없다. `more_rains()`의 반복
샘플링(`n < 100`)도 이 동일한 전역 스트림을 소비한다.

## 5. 층화 추출 및 train/test 분할

강우 생성 자체는 재현기간×지속시간×분위 조합마다 "기준 1개 + 섭동
n_vars개"를 **모두** 파일로 남기는 방식이며, 생성 단계에서 train/test를
구분하지 않는다. 분할은 이후 별도 단계(`data_paths.load_fixed_split()`,
`data/splits/fixed_split_seed42.json`, split_seed=42)에서 이미 생성된
전체 6,696개 시나리오 풀을 재현기간×지속시간 층(stratum) 기준으로
train(3,996)/test(2,700)로 나눈다.

**train/test 시계열·시드 공유 여부**: **공유된다.** 학습·테스트가 동일한
Huff 곡선 생성 함수 + 동일한 전역 시드로 만든 **하나의 시나리오 풀**을
사후에 층화 분할한 것이다 — 별도 생성 절차나 독립 시드를 쓰지 않는다.
R1-5의 지적("테스트 세트가 미지의 조건이 아니라 같은 분포의 다른
표본")이 이 절차상 구조적으로 타당함을 코드가 확인해 준다. 정량적
검증은 `results/summary/E08_distribution_shift.md` 참조 — 실제로 거의
동일한 분포임이 확인됐다(KS 통계량 0.008~0.009).

## 6. 실측 강우 데이터 (참고, E8-3)

`data_generate.py:428-481`에 저자가 이미 작성한
`generate_rain_inp_realdata()`가 기상청 기상자료개방포털의 관측소 CSV를
`.inp`로 변환하는 경로가 존재한다(**관측소는 "나주"** — 계획서가 언급한
"서울"과 다름, 이 문서에서 확정해 기록). 상세는
`docs/OBSERVED_EVENTS.md` 참조.
