# 데이콘 - 운수종사자 인지적 특성 데이터를 활용한 교통사고 위험 예측 - Best Score 모델
https://dacon.io/competitions/official/236607/overview/description

운수종사자 자격검사(A: 신규자격, B: 자격유지) 과정에서 수집된 인지·반응 관련 세부 검사 데이터를 활용하여, 검사 결과 기준 교통사고 위험군에 속할 확률을 예측하는 AI 모델을 개발합니다.

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [데이터 구조](#데이터-구조)
- [EDA (Exploratory Data Analysis)](#eda-exploratory-data-analysis)
- [전처리 방법](#전처리-방법)
- [모델 구조](#모델-구조)
- [모델링 및 학습 전략](#5-모델링-및-학습-전략)
- [실행 방법](#실행-방법)
- [파일 구조](#파일-구조)

## 🎯 프로젝트 개요

### 목표
운수종사자의 교통사고 위험도를 예측하여 안전 운전 능력을 평가합니다.

운수종사자의 인지검사 결과(A/B 배터리)를 기반으로  
**향후 사고 위험군(위험군=1 / 정상군=0)** 을 예측하는 이진 분류 모델입니다.

- **입력 데이터**: A·B 두 종류의 인지검사 결과 + 기본 인구학 정보(나이, 검사일 등) + 과거 검사 이력
- **출력**: 각 검사(`Test_id`)별 위험군 확률 `P(Label=1)`
- **모델 구조**
  - A 세트: **단일 HGBM + 5-seed 앙상블 + OOF 검증**
  - B 세트: **검사 이력 기반 2-그룹(history-split) HGBM 앙상블 + OOF 

### 검사 유형
- **A 검사 (신규 자격 검사)**: 지각운동요인, 지적운동요인, 정서/행동 안정성 평가
- **B 검사 (자격 유지 검사)**: 시야각, 시각기억, 주의지속, 다중과제 등 실전 운전 능력 평가

### 평가 지표
- **AUC (Area Under ROC Curve)**: ROC 곡선 아래 면적 (50% 가중치)
- **Brier Score**: 예측 확률과 실제값 간의 평균 제곱 오차 (25% 가중치)
- **ECE (Expected Calibration Error)**: 예측 확률의 보정 오차 (25% 가중치)

**최종 점수 = 0.5 × (1 - AUC) + 0.25 × Brier + 0.25 × ECE**

### 2.2 원시 특성 파일

- `data/train/A.csv`, `data/train/B.csv`
- 각 행은 **한 번의 검사**(A 또는 B)를 의미하며,
  - `PrimaryKey`: 동일 피검자의 ID
  - `Age`: 나이 구간 (예: `55A`, `60B` 형태)
  - `TestDate`: 연월 (예: `201906`)
  - `A1-*, A2-*, ..., B10-*`:  
    반응 시간(Time), 정답/오답, 조건, 자극 유형 등이  
    **쉼표(,)로 연결된 시퀀스 문자열**로 제공됩니다.

예시)
- `A1-4`: `"612.5, 589.2, 700.1, ..."`  (반응시간 시퀀스)
- `A1-3`: `"1,0,1,..."` (정답 여부 시퀀스)

## 🔍 EDA (Exploratory Data Analysis)

EDA는 `eda/` 폴더의 노트북에서 수행되었습니다. 주요 분석 내용은 다음과 같습니다:

### 1. 기본 데이터 탐색 (`analyze_train.ipynb`)
- 데이터 로딩 및 구조 파악
- Label 분포 분석 (위험군 비율)
- Test 종류별 분포 (A/B 검사 비율)
- Age 분포 및 Age별 위험군 비율 분석

### 2. 종합 피처 엔지니어링 (`comprehensive_feature_engineering.ipynb`)
- A 검사별 피처 생성 (A1~A9)
- B 검사별 피처 생성 (B1~B10)
- 인지 프로파일 그룹화
- 상관관계 분석
- 피처 중요도 분석

### 3. 도메인 기반 분석 (`domain_based_analysis.ipynb`)
- 검사별 도메인 지식 기반 피처 설계
- 복합 점수(Composite Score) 생성
- 나이 정규화 점수(Age-normed Score) 생성

### 4. 피처 Cutpoint 분석 (`feature_cutpoint_analysis.ipynb`)
- 피처별 최적 임계값 분석
- 이진 피처 생성 가능성 검토

### 주요 발견사항
- A 검사와 B 검사의 Label 분포가 다름
- Age와 위험도 간 상관관계 존재
- 시계열 데이터 구조 (PrimaryKey 기준 여러 검사 추적 가능)
- CSV 형식으로 저장된 시퀀스 데이터 (예: "1,2,3,4") 처리 필요

## 🔧 전처리 방법

전처리는 6단계로 구성되며, 원시 시퀀스 데이터를 의미 있는 수치 피처로 변환합니다.

### 4.1 공통 유틸 함수

#### 기본 변환 함수

- **`convert_age(val)`**
  - `Age` 문자열(`55A`, `60B`)을 **숫자형 나이(`Age_num`)**로 변환
  - `"65a"` → 65, `"65b"` → 70 (5살 단위 보정)

- **`split_testdate(val)`**
  - `TestDate`에서 `Year`, `Month` 추출
  - `"202312"` → Year=2023, Month=12

#### 시퀀스 처리 함수

각 시퀀스 컬럼(`"1,0,1,..."`, `"612.5,580.0,..."`)에 대해:

- **`seq_mean(series)`**: numeric 시퀀스의 평균
  - 예: `"612.5,580.0,700.1"` → 630.87

- **`seq_std(series)`**: numeric 시퀀스의 표준편차
  - 반응시간의 변동성 측정

- **`seq_rate(series, target)`**: target(예: `"1"`) 비율
  - 예: `"1,0,1,1"` → 0.75 (정답률)

- **`masked_mean_from_csv_series(cond_series, val_series, mask_val)`**
  - **조건 시퀀스**(예: 왼/오, valid/invalid, 속도조건 등)에 따라 특정 subset에 대한 평균 계산
  - 예: 왼쪽 자극(`mask_val=1`)에 대한 반응시간만 추출하여 평균 계산

- **`masked_mean_in_set_series(cond_series, val_series, mask_set)`**
  - 여러 조건 값의 집합에 해당하는 경우의 평균 계산

- **`masked_rate_equals(cond_series, val_series, mask_set, positive)`**
  - 특정 조건 집합에서 positive 값의 비율 계산

---

### 4.2 A 세트: `preprocess_A`

**목적**: A1~A5, A6~A9의 원시 시퀀스를 **정리된 수치 피처**로 변환.

#### 주요 파생 피처

##### 과제별 정답률/반응시간

- **A1 (지각 반응속도)**:
  - `A1_resp_rate`: 응답 비율
  - `A1_rt_mean`, `A1_rt_std`: 반응시간 평균/표준편차
  - `A1_rt_left`, `A1_rt_right`: 좌/우 반응시간
  - `A1_rt_side_diff`: 좌/우 반응시간 차이
  - `A1_rt_slow`, `A1_rt_fast`: 느림/빠름 조건 반응시간
  - `A1_rt_speed_diff`: 속도 조건 간 반응시간 차이

- **A2 (선택 반응속도)**:
  - `A2_resp_rate`: 응답 비율
  - `A2_rt_mean`, `A2_rt_std`: 반응시간 평균/표준편차
  - `A2_rt_cond1_diff`: 조건1 간 반응시간 차이
  - `A2_rt_cond2_diff`: 조건2 간 반응시간 차이

- **A3 (선택 지각검사)**:
  - `A3_valid_ratio`: 유효 자극 비율
  - `A3_invalid_ratio`: 무효 자극 비율
  - `A3_correct_ratio`: 정답 비율
  - `A3_resp2_rate`: 응답2 비율
  - `A3_rt_mean`, `A3_rt_std`: 반응시간 평균/표준편차
  - `A3_rt_size_diff`: 크기 조건 간 반응시간 차이
  - `A3_rt_side_diff`: 측면 조건 간 반응시간 차이
  - `A3_rt_valid`, `A3_rt_invalid`: 유효/무효 자극 반응시간
  - `A3_rt_valid_invalid_gap`: 유효-무효 반응시간 차이

- **A4 (Stroop 검사)**:
  - `A4_acc_rate`: 정확도 비율
  - `A4_resp2_rate`: 응답2 비율
  - `A4_rt_mean`, `A4_rt_std`: 반응시간 평균/표준편차
  - `A4_stroop_diff`: 일치/불일치 조건 반응시간 차이 (Stroop 효과)
  - `A4_rt_color_diff`: 색상 조건 간 반응시간 차이
  - `A4_acc_con`, `A4_acc_incon`: 일치/불일치 조건 정확도
  - `A4_acc_gap_incon_con`: 불일치-일치 정확도 차이

- **A5 (변화 감지)**:
  - `A5_acc_rate`: 정확도 비율
  - `A5_resp2_rate`: 응답2 비율
  - `A5_acc_nonchange`: 변화 없음 조건 정확도
  - `A5_acc_change`: 변화 있음 조건 정확도

##### Cognitive Score

- **`CogScore_A`**: A6, A7, A9-4 등을 합산한 **인지 능력 종합 점수**
  - A6: 기억력 관련 점수
  - A7: 기억력 관련 점수
  - A9-4: 인지 관련 점수

##### 정서/타당도 관련

- **`A8_Validity_Score`**: 응답 왜곡/비일관성 지표
  - A8-1 + A8-2 합계
  - 높을수록 응답 신뢰도 저하

- **`A9_Emotional_Score`**: 정서/행동/스트레스 관련 종합 점수
  - A9-1 (정서안정성) + A9-2 (행동안정성) + A9-3 (현실판단력) + A9-5 (생활스트레스)
  - 높을수록 불안정/스트레스가 높음

**최종 처리**: 시퀀스 원본 컬럼(`A1-1~A5-3` 등)은 제거하고, **집계된 수치 피처 + 기본 인구학 정보**만 남깁니다.

---

### 4.3 B 세트: `preprocess_B`

**목적**: B1~B10 Task에서 **지속주의·주의전환·멀티태스킹 성능**을 나타내는 피처 생성.

#### 주요 파생 피처

##### 과제별 정답률/반응시간

- **B1, B2**: 듀얼 태스크
  - `B1_acc_task1`, `B1_rt_mean`, `B1_rt_std`, `B1_acc_task2`
  - `B2_acc_task1`, `B2_rt_mean`, `B2_rt_std`, `B2_acc_task2`

- **B3~B5**: 지속주의 검사
  - `B3_acc_rate`, `B3_rt_mean`, `B3_rt_std`
  - `B4_acc_rate`, `B4_rt_mean`, `B4_rt_std`
  - `B5_acc_rate`, `B5_rt_mean`, `B5_rt_std`

- **B6~B8**: 단순 정확도 검사
  - `B6_acc_rate`, `B7_acc_rate`, `B8_acc_rate`

##### B9 (청각/시각 Go/No-Go 형태)

- `B9_aud_hit_rate`: 청각 Hit Rate (15개 타겟 중)
- `B9_aud_miss_rate`: 청각 Miss Rate
- `B9_aud_fa_rate`: 청각 False Alarm Rate (35개 distracter 중)
- `B9_aud_cr_rate`: 청각 Correct Rejection Rate
- `B9_aud_overall_acc`: 청각 전체 정확도
- `B9_aud_sensitivity`: 청각 민감도 (Hit Rate - False Alarm Rate)
- `B9_vis_err_rate`: 시각 에러율 (32개 시도 중)

##### B10 (듀얼태스킹)

- `B10_aud_hit_rate`: 청각 Hit Rate (20개 타겟 중)
- `B10_aud_miss_rate`: 청각 Miss Rate
- `B10_aud_fa_rate`: 청각 False Alarm Rate (60개 distracter 중)
- `B10_aud_cr_rate`: 청각 Correct Rejection Rate
- `B10_aud_overall_acc`: 청각 전체 정확도
- `B10_aud_sensitivity`: 청각 민감도
- `B10_vis1_err_rate`: 시각1 에러율 (52개 시도 중)
- `B10_vis2_acc_rate`: 시각2 정확도 (20개 시도 중)

##### B9 ↔ B10 멀티태스킹 코스트

- **`B10_multitask_cost_aud`**: B10 청각 정답률 – B9 청각 정답률
  - 음수 값일수록 멀티태스킹 시 성능 저하가 큼

- **`B10_multitask_cost_vis`**: B10 시각 에러율 – B9 시각 에러율
  - 양수 값일수록 멀티태스킹 시 에러 증가

마찬가지로 원본 시퀀스 컬럼은 제거합니다.

---

### 4.4 고급 파생: `add_features_A`, `add_features_B`

#### (1) 공통 아이디어

##### Speed-Accuracy Tradeoff

- 반응이 느리면서 정답률까지 떨어지는 경우 위험 신호로 반영
- 예: `A1_speed_acc_tradeoff = A1_rt_mean / A1_resp_rate`
- A1, A2, A4에 적용

##### RT 변동성 (CV: Coefficient of Variation)

- `*_rt_cv = std / mean`
- 일정하지 않은 반응(변동성↑)은 집중력 저하로 해석
- A1, A2, A3, A4에 적용

##### 절대값 Gap

- 조건 간 차이의 절대값으로 변동성 측정
- 예: `A1_rt_side_gap_abs = |A1_rt_side_diff|`

##### Stroop × Error 상호작용

- `A4_stroop_x_err = A4_stroop_diff × (1 - A4_acc_rate)`
- Stroop 효과가 클수록, 오류율이 높을수록 위험 신호

#### (2) Composite Scores – A

##### `PerceptualSpeed_A`

- `A1_rt_mean`, `A4_rt_mean` 등 **지각/반응 속도 관련 RT**를
- Z-score 정규화 후 **(속도가 빠를수록 좋은 방향)** 으로 부호 조정해 결합
- RT는 낮을수록 좋으므로 부호 반전

##### `CognitiveAbility_A`

- `CogScore_A`를 Z-score로 변환한 **인지 능력 요약 점수**
- 원본 `CogScore_A`는 중복을 피하기 위해 drop

##### `EmotionalRisk_A`

- `A8_Validity_Score`, `A9_Emotional_Score`를 Z-score 후 합산
- 점수가 높을수록 **정서/응답 타당도 위험**이 크도록 설계
- 부호 반전하여 위험도가 높을수록 값이 커지도록 조정

#### (3) Composite Scores – B

##### `RiskScore_B_norm`

- `B3~B5` 정답률, RT CV, speed-accuracy tradeoff 등을 종합한 **위험도 점수**를
- Z-score 및 부호 조정을 통해 **높을수록 위험**이 되게 정규화

##### `MultitaskAbility_B`

- `B9_aud_overall_acc`, `B10_multitask_cost_aud` 등을 Z-score로 결합
- 멀티태스킹 시 성능 저하가 적을수록 높은 점수가 되도록 설계

---

### 4.5 나이 보정 스코어: `add_age_normed_composites_*`

나이가 인지검사에 미치는 영향을 보정하기 위해, **나이 구간별 z-score**를 만든 추가 피처를 사용합니다.

#### 나이 bin 구간
- `<50`, `50-59`, `60-69`, `70+`, `Unknown`

#### A용 Age-normed 피처
- `PerceptualSpeed_A_ageNorm`: 나이 구간 내 지각속도 정규화 점수
- `CognitiveAbility_A_ageNorm`: 나이 구간 내 인지능력 정규화 점수
- `EmotionalRisk_A_ageNorm`: 나이 구간 내 정서위험 정규화 점수

#### B용 Age-normed 피처
- `RiskScore_B_ageNorm`: 나이 구간 내 위험도 정규화 점수
- `MultitaskAbility_B_ageNorm`: 나이 구간 내 멀티태스킹 능력 정규화 점수

각 나이 bin 내에서 평균·표준편차를 기준으로 정규화하여 **동일 연령대 내 상대적 위치**를 반영합니다.

---

### 4.6 History Features: `add_history_features_A`, `add_history_features_B`

`PrimaryKey` + `YearMonthIndex`를 기준으로 **같은 피검자의 시간축 상의 패턴**을 반영합니다.

#### 공통 히스토리 피처

- **`n_tests_so_far`**: 해당 피검자가 **몇 번째 검사인지** (1, 2, 3, …)
  - B 검사의 경우: 1 = 첫 검사, >1 = 재검사

- **`months_since_first_test`**: 최초 검사 이후 경과 개월 수

- **`months_since_prev_test`**: 직전 검사 이후 경과 개월 수

#### 핵심 특징

- **누적 이력의 rolling mean + trend** 만 사용
- `_numpy_rolling_mean_shifted_safe`로 구현:
  - **항상 과거 값만 사용** (현재 행은 포함 안 함 → 데이터 누수 방지)
  - 그룹별(PrimaryKey별)로 window=10까지 고려
  - 각 PrimaryKey 내에서 YearMonthIndex 기준 정렬 후 계산

#### A 검사 히스토리 피처 예시

- `A4_acc_prev_mean`, `A4_stroop_prev_mean`: 과거 A4 정확도/Stroop 효과 평균
- `A1_rt_prev_mean`, `A1_rt_trend`: 과거 A1 반응시간 평균 및 트렌드
- `CognitiveAbility_A_prev_mean`, `CognitiveAbility_A_trend`: 과거 인지능력 평균 및 트렌드
- `A4_stroop_trend`: 현재 Stroop 효과와 과거 평균의 차이

#### B 검사 히스토리 피처 예시

- `B4_acc_prev_mean`, `B4_acc_trend`: 과거 B4 정확도 평균 및 트렌드
- `B10_multicost_prev_mean`, `B10_multicost_trend`: 과거 멀티태스킹 코스트 평균 및 트렌드
- `B3_rt_prev_mean`, `B3_rt_trend`: 과거 B3 반응시간 평균 및 트렌드
- `B4_rt_prev_mean`, `B4_rt_trend`, `B5_rt_prev_mean`, `B5_rt_trend`: B4, B5 반응시간 히스토리

**중요**: 첫 검사(`n_tests_so_far == 1`)의 경우 히스토리 피처는 대부분 NaN이 되며, 이는 B 검사의 history-split 서브모델링 전략의 핵심입니다.

---

### 4.7 Row-wise 피처

- **`NA_COUNT`**: 결측치 개수
- **`NA_RATIO`**: 결측치 비율 (전체 피처 중)

행 단위 결측치 정보를 추가하여 데이터 품질을 반영합니다.

---

### 4.8 전처리 파이프라인 (`build_preprocessor`)

- **수치형 피처**: 
  - `SimpleImputer` (median) → 정규화 없음 (트리 모델 사용)
- **범주형 피처**: 
  - `OrdinalEncoder` → `SimpleImputer` (most_frequent)

`ColumnTransformer`를 사용하여 수치형/범주형 피처를 자동으로 구분하여 처리합니다.

## 🤖 모델 구조

### 모델 아키텍처

#### A 검사 모델
- **단일 모델**: 나이 서브모델 없이 단일 HistGradientBoostingClassifier 사용
- **앙상블**: 5개의 서로 다른 random seed 모델의 확률 평균
- **OOF (Out-of-Fold) 검증**: 2-Fold StratifiedKFold

#### B 검사 모델
- **History-based Submodels**: `n_tests_so_far` 기준으로 분리
  - **First 모델** (`n_tests_so_far == 1`): 첫 검사용 모델 (히스토리 피처 부재)
  - **Repeat 모델** (`n_tests_so_far > 1`): 재검사용 모델 (히스토리 피처 활용)
- **앙상블**: 각 그룹마다 5개 seed 모델의 확률 평균
- **OOF 검증**: 2-Fold StratifiedKFold

### 모델: HistGradientBoostingClassifier

#### A 검사 하이퍼파라미터
```python
{
    "learning_rate": 0.05,
    "max_iter": 1500,
    "max_depth": None,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 30,
    "class_weight": "balanced",
    "random_state": seed  # 5개 seed: 42, 202, 777, 1001, 8888
}
```

#### B 검사 하이퍼파라미터
```python
{
    "learning_rate": 0.02,
    "max_iter": 1500,
    "max_depth": None,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 40,
    "class_weight": "balanced",
    "random_state": seed  # 5개 seed: 42, 202, 777, 1001, 8888
}
```

### 확률 보정 (Calibration)
- **CalibratedClassifierCV**: Isotonic 방법, 3-Fold CV
- 클래스 불균형 및 작은 클래스 수에 대한 안전장치 포함

### 앙상블 전략
- **AvgProbaEnsemble**: 5개 seed 모델의 예측 확률 평균
- 각 fold마다 독립적으로 학습하여 OOF 예측 생성

---

## 5. 모델링 및 학습 전략

### 5.1 공통 설정

#### 모델 선택
- **모델**: `HistGradientBoostingClassifier` (scikit-learn 내장)
- LightGBM 미사용 (호환성 및 간결성 고려)

#### 클래스 불균형 대응
- **`class_weight="balanced"`**: 클래스 불균형 자동 보정
- 양성 클래스(위험군)에 더 높은 가중치 부여

#### Early Stopping
- **`max_iter = 1500`**: 넉넉하게 설정하여 early stopping이 실제 iteration 결정
- **`early_stopping=True`**: 검증 성능이 개선되지 않으면 조기 종료
- **`validation_fraction=0.15`**: 학습 데이터의 15%를 내부 검증셋으로 사용
- **`n_iter_no_change`**:
  - A 검사: 30 iteration 동안 개선 없으면 종료
  - B 검사: 40 iteration 동안 개선 없으면 종료

#### Ensemble Strategy
- **`ENSEMBLE_SEEDS = (42, 202, 777, 1001, 8888)`**: 5개의 서로 다른 random seed
- 각 seed별로 HGBM 학습 후 **확률 평균 앙상블** (`AvgProbaEnsemble`)
- 앙상블을 통해 예측 분산을 줄이고 일반화 성능 향상

---

### 5.2 Calibration (확률 보정)

#### 목적
- 모델의 예측 확률을 실제 위험도에 맞게 보정
- ECE (Expected Calibration Error) 감소를 통한 최종 점수 개선

#### 방법
- **`CalibratedClassifierCV`**: Isotonic 방법, `cv=3`
- Isotonic regression을 사용하여 확률 분포를 보정

#### 두 가지 모드

##### (1) OOF 학습 시
- **`train_and_calibrate_with_split`**
  - 학습 데이터를 **train/calib로 나누고**
  - base model은 train 데이터로 학습
  - calibration은 calib 데이터로 수행
  - **과적합 방지** 및 일반화 성능 향상

##### (2) 최종 모델 학습 시
- 전체 데이터를 사용해 한 번 더 calibration
- 서버 제출 시 더 많은 데이터로 보정된 확률 사용

#### 안전장치
- 클래스가 1개만 있는 경우 calibration 불가 → base_clf 반환
- 클래스별 샘플 수가 `CALIB_CV`보다 적으면 fallback
- `ValueError`, `RuntimeError` 발생 시 base_clf로 fallback

---

### 5.3 A 세트: 단일 모델 + OOF (`fit_partition_A_oof_single`)

#### 전략
- 나이 서브모델 없이 **단일 HistGradientBoostingClassifier** 사용
- `Age_num`은 일반 피처로만 사용 (나이 분할 모델 제거)
- OOF 검증을 통한 신뢰할 수 있는 성능 평가

#### OOF 절차

1. **데이터 분할**
   - `StratifiedKFold(n_splits=2, shuffle=True, random_state=42)`
   - Label 비율을 유지하면서 2-fold로 분할

2. **각 Fold에서**:
   - 전처리(`ColumnTransformer`)를 **fold별로 새로 fit** → 데이터 누수 방지
   - 5-seed HGBM + calibration 앙상블 학습
     - 각 seed별로 base model 학습
     - calibration 수행 (train/calib split)
     - 5개 모델의 확률 평균
   - fold별 OOF 예측 기록 및 점수 출력
     - AUC, Brier, ECE, combined score

3. **전체 OOF 기준**:
   - 모든 fold의 OOF 예측값을 합쳐서 최종 성능 계산
   - AUC, Brier, ECE, combined score 출력

4. **최종 모델**:
   - 전체 데이터를 사용해 다시 5-seed HGBM + calibration 학습
   - `A_MODEL_PATH`, `A_PREPROC_PATH`에 저장 (pickle protocol 4)

---

### 5.4 B 세트: history-split 서브모델 + OOF (`fit_partition_B_submodels_oof`)

#### 핵심 아이디어

B 검사는 **재검사 특성**상 과거 검사 데이터가 중요합니다:

- **`n_tests_so_far == 1`** → **첫 검사**
  - history 피처 대부분 NaN
  - 과거 정보 부재
  - 초기 상태 평가

- **`n_tests_so_far > 1`** → **재검사**
  - history 정보 풍부
  - 과거 검사 패턴 활용 가능
  - 변화 추세 분석 가능

**두 그룹의 특성이 다르므로 서로 다른 모델을 학습**

#### OOF 절차

1. **데이터 분할**
   - `StratifiedKFold(n_splits=2, shuffle=True, random_state=42)`
   - 전체 B train을 2-fold로 분할

2. **각 Fold에서**:

   - **First 그룹 분리** (`n_tests_so_far == 1`)
     - 5-seed HGBM + calibration 앙상블 학습
     - 실제 사용된 iteration 수 기록
   
   - **Repeat 그룹 분리** (`n_tests_so_far > 1`)
     - 5-seed HGBM + calibration 앙상블 학습
     - 실제 사용된 iteration 수 기록

   - **Validation 예측**:
     - validation 데이터에서 `n_tests_so_far` 확인
     - 1이면 `ensemble_first` 사용
     - 2 이상이면 `ensemble_repeat` 사용
     - OOF 예측 기록 및 fold별 점수 출력

3. **전체 OOF 기준**:
   - 모든 fold의 OOF 예측값을 합쳐서 최종 성능 계산
   - AUC, Brier, ECE, combined score 출력

4. **최종 모델**:
   - 전체 데이터를 기준으로 다시 first / repeat 각각 5-seed 앙상블 학습
   - `B_MODEL_PATH`, `B_PREPROC_PATH`에 저장
   - 모델 번들 구조: `{mode: "history_split", ensemble_first: ..., ensemble_repeat: ...}`

#### 예측 시 (`predict_partition_B_submodels`)

- 입력 B 샘플에서 `n_tests_so_far`를 확인
- **1이면** `ensemble_first`를 사용하여 확률 예측
- **2 이상이면** `ensemble_repeat`를 사용하여 확률 예측
- `ens_repeat`가 None이면 (single fallback) `ens_first`만 사용

#### Fallback 메커니즘
- First 또는 Repeat 그룹의 샘플 수가 너무 적은 경우
- 단일 모델로 fallback (history-split 없이 전체 데이터로 학습)

---

### 5.5 최적 하이퍼파라미터 저장

#### `best_params.json` 구조
- `hgb_params_A`, `hgb_params_B`: 각 검사별 최적 하이퍼파라미터
- `ensemble_seeds`: 사용된 seed 리스트
- `use_calibration`, `calib_method`, `calib_cv`: calibration 설정
- `oof_score_A`, `oof_score_B`, `oof_score_total`: OOF 검증 점수
- `random_state`: 재현성을 위한 seed
- `sklearn_version`: 버전 정보

#### 목적
- 서버 제출용 스크립트(`script.py`)에서 동일한 하이퍼파라미터 사용
- 학습 시 찾은 최적 설정을 추론 시에도 동일하게 적용

---

## 🚀 실행 방법

### 1. 학습 (train.py)

```bash
cd best_score
python train.py
```

**주요 설정:**
- `USE_OOF = True`: OOF 검증 활성화
- `N_SPLITS = 2`: 2-Fold CV
- `ENSEMBLE_SEEDS = (42, 202, 777, 1001, 8888)`: 5개 seed 앙상블
- `USE_CALIBRATION = True`: 확률 보정 활성화

**출력:**
- `model_hgb/`: 학습된 모델 및 전처리 파이프라인
  - `hgb_A.pkl`: A 검사 모델
  - `hgb_B.pkl`: B 검사 모델 (first/repeat 포함)
  - `preproc_A.pkl`, `preproc_B.pkl`: 전처리 파이프라인
  - `meta.json`: 모델 메타데이터
  - `best_params.json`: 최적 하이퍼파라미터

### 2. 추론 (script.py)

서버 제출용 스크립트입니다. 모델을 저장하지 않고 즉시 재학습하여 추론합니다 (NumPy BitGenerator 에러 회피).

---

## 📝 주요 특징

### 1. History-based Submodels (B 검사)
B 검사는 재검사 특성상 과거 검사 데이터가 중요합니다. 첫 검사와 재검사를 분리하여 학습함으로써 더 정확한 예측을 달성했습니다.

### 2. Ensemble with Multiple Seeds
5개의 서로 다른 random seed로 학습한 모델을 앙상블하여 일반화 성능을 향상시켰습니다.

### 3. Feature Engineering
- 도메인 지식 기반 복합 점수 생성
- 나이 효과 보정 (Age-normed Score)
- 시계열 히스토리 피처 활용

### 4. Probability Calibration
CalibratedClassifierCV를 사용하여 예측 확률의 보정 오차를 줄였습니다.

### 5. Early Stopping
과적합 방지를 위해 early stopping을 활용하여 최적의 iteration 수를 자동으로 결정합니다.

## 🔗 참고 자료

- EDA 노트북: `../eda/` 폴더
  - `analyze_train.ipynb`: 기본 데이터 탐색
  - `comprehensive_feature_engineering.ipynb`: 종합 피처 엔지니어링
  - `domain_based_analysis.ipynb`: 도메인 기반 분석
  - `feature_cutpoint_analysis.ipynb`: 피처 Cutpoint 분석

