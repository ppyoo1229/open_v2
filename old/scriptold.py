#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
서버 제출용 추론 스크립트 (script.py)
- 평가 서버 환경: Ubuntu 22.04.5 LTS, CPU 3 vCPU, RAM 28GB, Python 3.10.12
- 실행 시간 제한: 30분 (추론 코드 실행)
- 데이터 경로: data/ (서버에서 자동 생성, 테스트 데이터 포함)
- 출력 경로: output/ (서버에서 자동 생성, submission.csv 저장)
- 모델 저장/로딩 없이 즉시 재학습 (NumPy BitGenerator 에러 회피)
- best_params.json에서 최적 하이퍼파라미터 로드
"""
import os
import time
import warnings
warnings.filterwarnings("ignore")

from typing import Tuple, List, Sequence
from collections import namedtuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn import __version__ as sklver

import pickle
import json  

# =============================================================================
# 경로 설정
# =============================================================================
# 서버 환경: data/ 디렉토리는 서버에서 자동 생성 (테스트 데이터 포함)
# 서버 환경: output/ 디렉토리는 서버에서 자동 생성 (submission.csv 저장)
# 로컬/서버 모두 동일한 경로 사용
DATA_DIR = "data"    # 서버에서 자동 생성되는 테스트 데이터 디렉토리
OUTPUT_DIR = "output"  # 서버에서 자동 생성되는 출력 디렉토리
MODEL_DIR  = "model"   # best_params.json 등 모델 관련 파일 저장 (제출 시 포함)

SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
BEST_PARAMS_PATH = os.path.join(MODEL_DIR, "best_params.json")

# 참고: 모델 저장/로딩은 사용하지 않음 (NumPy BitGenerator 에러 회피)
# 대신 best_params.json에서 하이퍼파라미터만 로드하고 모델은 재학습

# =============================================================================
# 하이퍼파라미터 로드 (best08.py에서 찾은 최적값 사용)
# =============================================================================
def load_best_params():
    """
    best08.py에서 저장한 최적 하이퍼파라미터를 로드.
    없으면 기본값 사용 (fallback).
    """
    if os.path.exists(BEST_PARAMS_PATH):
        try:
            with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
                params = json.load(f)
            print(f"Loaded best hyperparameters from {BEST_PARAMS_PATH}")
            return params
        except Exception as e:
            print(f" Failed to load best params: {e}")
            print(f" Using default hyperparameters...")
            return None
    else:
        print(f"Best params file not found: {BEST_PARAMS_PATH}")
        print(f" Using default hyperparameters...")
        return None

# 하이퍼파라미터 로드
best_params = load_best_params()

if best_params is not None:
    # best08.py에서 찾은 최적 하이퍼파라미터 사용
    BASE_HGB_PARAMS_A = best_params["hgb_params_A"].copy()
    BASE_HGB_PARAMS_B = best_params["hgb_params_B"].copy()
    ENSEMBLE_SEEDS = tuple(best_params["ensemble_seeds"])
    USE_CALIBRATION = best_params.get("use_calibration", True)
    CALIB_METHOD = best_params.get("calib_method", "isotonic")
    CALIB_CV = best_params.get("calib_cv", 3)
    AGE_SPLIT_A = best_params.get("age_split_A", 60.0)
    AGE_BUFFER_MIN_A = best_params.get("age_buffer_min_A", 55.0)
    AGE_BUFFER_MAX_A = best_params.get("age_buffer_max_A", 65.0)
    RANDOM_STATE = best_params.get("random_state", 42)
    print(f" Using best hyperparameters from best08.py OOF evaluation")
else:
    # Fallback: 기본 하이퍼파라미터
    RANDOM_STATE = 42
    AGE_SPLIT_A = 60.0
    AGE_BUFFER_MIN_A = 55.0
    AGE_BUFFER_MAX_A = 65.0
    USE_CALIBRATION = True
    CALIB_METHOD = "isotonic"
    CALIB_CV = 3
    ENSEMBLE_SEEDS: Sequence[int] = (42, 202, 777)
    BASE_HGB_PARAMS_A = dict(
        learning_rate=0.06,
        max_iter=300,
        max_depth=None,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=25,
        class_weight="balanced",
    )
    BASE_HGB_PARAMS_B = dict(
        learning_rate=0.03,
        max_iter=700,
        max_depth=None,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        class_weight="balanced",
    )
    print(f"  Using default hyperparameters (best_params.json not found)")

TEST_SIZE_FOR_LOG = 0.0  # 서버 제출용: 전체 데이터로 학습 (validation 없음, 시간 절약)

# =============================================================================
# 공통 유틸
# =============================================================================

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)

def read_index_files() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_idx = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_idx  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train_idx, test_idx

def read_raw_feature_files(split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    A_df = pd.read_csv(os.path.join(DATA_DIR, split, "A.csv"))
    B_df = pd.read_csv(os.path.join(DATA_DIR, split, "B.csv"))
    return A_df, B_df

def separate_num_cat(df: pd.DataFrame, drop_cols: List[str]) -> Tuple[List[str], List[str]]:
    cols = [c for c in df.columns if c not in drop_cols]
    cat_cols = [c for c in cols if str(df[c].dtype) in ("object", "category")]
    num_cols = [c for c in cols if c not in cat_cols]
    return num_cols, cat_cols

def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordenc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preproc

def _mk_calibrator(base_clf):
    try:
        major, minor, *_ = map(int, sklver.split(".")[:2])
    except Exception:
        major, minor = 1, 4
    kw = dict(method=CALIB_METHOD, cv=CALIB_CV)
    if (major, minor) >= (1, 4):
        return CalibratedClassifierCV(estimator=base_clf, **kw)
    else:
        return CalibratedClassifierCV(base_estimator=base_clf, **kw)

def maybe_calibrate(base_clf, X_train, y_train):
    if not USE_CALIBRATION:
        return base_clf
    calib = _mk_calibrator(base_clf)
    calib.fit(X_train, y_train)
    return calib

class AvgProbaEnsemble:
    def __init__(self, models: List):
        self.models = models

    def predict_proba(self, X):
        probs = [m.predict_proba(X) for m in self.models]
        return np.mean(probs, axis=0)

def add_rowwise_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols]
    na_count = X.isna().sum(axis=1).astype(np.int32)
    na_ratio = (na_count / (len(feature_cols) + 1e-9)).astype(np.float32)
    df2 = df.copy()
    df2["NA_COUNT"] = na_count
    df2["NA_RATIO"] = na_ratio
    return df2

def blend_by_age_with_buffer(
    age_array: np.ndarray,
    p_under: np.ndarray,
    p_over: np.ndarray,
    split: float = AGE_SPLIT_A,
    buf_min: float = AGE_BUFFER_MIN_A,
    buf_max: float = AGE_BUFFER_MAX_A,
) -> np.ndarray:
    """
    Age_num을 기준으로 under/over 두 모델의 확률을 버퍼 구간에서 부드럽게 블렌딩.
    - age <= buf_min  : 100% under 모델
    - age >= buf_max  : 100% over 모델
    - buf_min < age < buf_max : 선형 보간
    - age NaN         : 100% under 모델 (학습 시와 동일)
    """
    age = np.asarray(age_array, dtype=float)
    p_under = np.asarray(p_under, dtype=float)
    p_over  = np.asarray(p_over, dtype=float)

    # 기본: 전부 under로 두고 시작
    w_over = np.zeros_like(age, dtype=float)

    # NaN 처리: 학습 시와 동일하게 under 모델 사용
    mask_nan = np.isnan(age)
    w_over[mask_nan] = 0.0  # 100% under

    # 완충 바깥 구간
    mask_low  = (~mask_nan) & (age <= buf_min)
    mask_high = (~mask_nan) & (age >= buf_max)
    w_over[mask_low]  = 0.0   # 순수 under
    w_over[mask_high] = 1.0   # 순수 over

    # 버퍼 구간: 선형 보간
    mask_buf = (~mask_nan) & (age > buf_min) & (age < buf_max)
    w_over[mask_buf] = (age[mask_buf] - buf_min) / (buf_max - buf_min)

    w_under = 1.0 - w_over
    blended = w_under * p_under + w_over * p_over
    return blended

# =============================================================================
# 평가 지표
# =============================================================================

def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    bin_totals = np.histogram(
        y_prob, bins=np.linspace(0, 1, n_bins + 1), density=False
    )[0]
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals / len(y_prob)
    bin_weights = bin_weights[non_empty_bins]
    prob_true = prob_true[:len(bin_weights)]
    prob_pred = prob_pred[:len(bin_weights)]
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece

def auc_brier_ece(answer_df: pd.DataFrame, submission_df: pd.DataFrame):
    if submission_df.isnull().values.any():
        raise ValueError("The submission dataframe contains missing values.")

    if len(answer_df.columns) != len(submission_df.columns) or not all(
        answer_df.columns == submission_df.columns
    ):
        raise ValueError("The columns of the answer and submission dataframes do not match.")
        
    submission_df = submission_df[submission_df.iloc[:, 0].isin(answer_df.iloc[:, 0])]
    submission_df.index = range(submission_df.shape[0])
    
    auc_scores = []
    for column in answer_df.columns[1:]:
        y_true = answer_df[column]
        y_scores = submission_df[column]
        auc = roc_auc_score(y_true, y_scores)
        auc_scores.append(auc)
    mean_auc = np.mean(auc_scores)

    brier_scores = []
    ece_scores = []
    for column in answer_df.columns[1:]:
        y_true = answer_df[column].values
        y_prob = submission_df[column].values
        
        brier = mean_squared_error(y_true, y_prob)
        brier_scores.append(brier)
        
        ece = expected_calibration_error(y_true, y_prob)
        ece_scores.append(ece)
    
    mean_brier = np.mean(brier_scores)
    mean_ece   = np.mean(ece_scores)
    
    combined_score = 0.5 * (1 - mean_auc) + 0.25 * mean_brier + 0.25 * mean_ece
    return combined_score, mean_auc, mean_brier, mean_ece

Metrics = namedtuple("Metrics", ["auc", "brier", "ece", "score"])

def evaluate_score(y_true, y_proba):
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    y_proba = np.clip(y_proba, 1e-7, 1 - 1e-7)

    answer_df = pd.DataFrame({
        "Test_id": np.arange(len(y_true)),
        "Label": y_true,
    })
    submission_df = pd.DataFrame({
        "Test_id": np.arange(len(y_true)),
        "Label": y_proba,
    })

    combined_score, mean_auc, mean_brier, mean_ece = auc_brier_ece(
        answer_df, submission_df
    )
    return Metrics(
        auc=mean_auc,
        brier=mean_brier,
        ece=mean_ece,
        score=combined_score,
    )

# =============================================================================
# FE 유틸
# =============================================================================

def convert_age(val):
    if pd.isna(val):
        return np.nan
    try:
        base = int(str(val)[:-1])
        return base if str(val)[-1].lower() == "a" else base + 5
    except Exception:
        return np.nan

def split_testdate(val):
    try:
        v = int(val)
        return v // 100, v % 100
    except Exception:
        return np.nan, np.nan

def _safe_div(a, b, eps=1e-6):
    return a / (b + eps)

def _clip01(s: pd.Series) -> pd.Series:
    return s.clip(lower=0.0, upper=1.0)

def seq_mean(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    df = s.str.split(",", expand=True).replace("", np.nan).astype(float)
    arr = df.to_numpy()
    with np.errstate(invalid="ignore"):
        m = np.nanmean(arr, axis=1)
    return pd.Series(m, index=series.index)

def seq_std(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    df = s.str.split(",", expand=True).replace("", np.nan).astype(float)
    arr = df.to_numpy()
    with np.errstate(invalid="ignore"):
        std = np.nanstd(arr, axis=1)
    return pd.Series(std, index=series.index)

def seq_rate(series: pd.Series, target: str = "1") -> pd.Series:
    s = series.fillna("").astype(str)
    df = s.str.split(",", expand=True)
    arr = df.to_numpy()
    non_empty = (arr != "")
    denom = non_empty.sum(axis=1)
    mask = (arr == str(target))
    num = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = num / np.where(denom == 0, np.nan, denom)
    return pd.Series(rate, index=series.index)

def masked_mean_from_csv_series(cond_series, val_series, mask_val):
    cond_df = cond_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    val_df  = val_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    cond_arr = cond_df.to_numpy(dtype=float)
    val_arr  = val_df.to_numpy(dtype=float)
    mask = (cond_arr == mask_val)
    with np.errstate(invalid="ignore"):
        sums   = np.nansum(np.where(mask, val_arr, np.nan), axis=1)
        counts = np.sum(mask, axis=1)
        out = sums / np.where(counts == 0, np.nan, counts)
    return pd.Series(out, index=cond_series.index)

def masked_mean_in_set_series(cond_series, val_series, mask_set):
    cond_df = cond_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    val_df  = val_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    cond_arr = cond_df.to_numpy(dtype=float)
    val_arr  = val_df.to_numpy(dtype=float)
    mask = np.isin(cond_arr, list(mask_set))
    with np.errstate(invalid="ignore"):
        sums   = np.nansum(np.where(mask, val_arr, np.nan), axis=1)
        counts = np.sum(mask, axis=1)
        out = sums / np.where(counts == 0, np.nan, counts)
    return pd.Series(out, index=cond_series.index)

def masked_rate_equals(cond_series, val_series, mask_set, positive="1"):
    cond_df = cond_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    val_df  = val_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    cond_arr = cond_df.to_numpy(dtype=float)
    val_arr  = val_df.to_numpy()
    mask = np.isin(cond_arr, list(mask_set))
    pos  = (val_arr == str(positive))
    with np.errstate(invalid="ignore"):
        nums = np.sum(np.where(mask, pos, False), axis=1)
        dens = np.sum(mask, axis=1)
        out  = nums / np.where(dens == 0, np.nan, dens)
    return pd.Series(out, index=cond_series.index)

def _has(df, cols):
    return all(c in df.columns for c in cols)

# =============================================================================
# A/B 원시 → 기본 파생
# =============================================================================

def preprocess_A(df: pd.DataFrame) -> pd.DataFrame:
    """script04 version - simpler than script05"""
    df = df.copy()
    df["Age_num"] = df["Age"].map(convert_age)
    ym = df["TestDate"].map(split_testdate)
    df["Year"]  = [y for y, m in ym]
    df["Month"] = [m for y, m in ym]

    feats = pd.DataFrame(index=df.index)

    feats["A1_resp_rate"] = seq_rate(df["A1-3"], "1")
    feats["A1_rt_mean"]   = seq_mean(df["A1-4"])
    feats["A1_rt_std"]    = seq_std(df["A1-4"])
    feats["A1_rt_left"]   = masked_mean_from_csv_series(df["A1-1"], df["A1-4"], 1)
    feats["A1_rt_right"]  = masked_mean_from_csv_series(df["A1-1"], df["A1-4"], 2)
    feats["A1_rt_side_diff"] = feats["A1_rt_left"] - feats["A1_rt_right"]
    feats["A1_rt_slow"]   = masked_mean_from_csv_series(df["A1-2"], df["A1-4"], 1)
    feats["A1_rt_fast"]   = masked_mean_from_csv_series(df["A1-2"], df["A1-4"], 3)
    feats["A1_rt_speed_diff"] = feats["A1_rt_slow"] - feats["A1_rt_fast"]

    feats["A2_resp_rate"] = seq_rate(df["A2-3"], "1")
    feats["A2_rt_mean"]   = seq_mean(df["A2-4"])
    feats["A2_rt_std"]    = seq_std(df["A2-4"])
    feats["A2_rt_cond1_diff"] = (
        masked_mean_from_csv_series(df["A2-1"], df["A2-4"], 1) -
        masked_mean_from_csv_series(df["A2-1"], df["A2-4"], 3)
    )
    feats["A2_rt_cond2_diff"] = (
        masked_mean_from_csv_series(df["A2-2"], df["A2-4"], 1) -
        masked_mean_from_csv_series(df["A2-2"], df["A2-4"], 3)
    )

    s = df["A3-5"].fillna("")
    total   = s.apply(lambda x: len(x.split(",")) if x else 0)
    valid   = s.apply(lambda x: sum(v in {"1","2"} for v in x.split(",")) if x else 0)
    invalid = s.apply(lambda x: sum(v in {"3","4"} for v in x.split(",")) if x else 0)
    correct = s.apply(lambda x: sum(v in {"1","3"} for v in x.split(",")) if x else 0)

    feats["A3_valid_ratio"]   = (valid / total).replace([np.inf,-np.inf], np.nan)
    feats["A3_invalid_ratio"] = (invalid / total).replace([np.inf,-np.inf], np.nan)
    feats["A3_correct_ratio"] = (correct / total).replace([np.inf,-np.inf], np.nan)

    feats["A3_resp2_rate"] = seq_rate(df["A3-6"], "1")
    feats["A3_rt_mean"]    = seq_mean(df["A3-7"])
    feats["A3_rt_std"]     = seq_std(df["A3-7"])
    feats["A3_rt_size_diff"] = (
        masked_mean_from_csv_series(df["A3-1"], df["A3-7"], 1) -
        masked_mean_from_csv_series(df["A3-1"], df["A3-7"], 2)
    )
    feats["A3_rt_side_diff"] = (
        masked_mean_from_csv_series(df["A3-3"], df["A3-7"], 1) -
        masked_mean_from_csv_series(df["A3-3"], df["A3-7"], 2)
    )
    feats["A3_rt_valid"]   = masked_mean_in_set_series(df["A3-5"], df["A3-7"], {1,2})
    feats["A3_rt_invalid"] = masked_mean_in_set_series(df["A3-5"], df["A3-7"], {3,4})
    feats["A3_rt_valid_invalid_gap"] = feats["A3_rt_valid"] - feats["A3_rt_invalid"]

    feats["A4_acc_rate"]   = seq_rate(df["A4-3"], "1")
    feats["A4_resp2_rate"] = seq_rate(df["A4-4"], "1")
    feats["A4_rt_mean"]    = seq_mean(df["A4-5"])
    feats["A4_rt_std"]     = seq_std(df["A4-5"])
    feats["A4_stroop_diff"] = (
        masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 2) -
        masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 1)
    )
    feats["A4_rt_color_diff"] = (
        masked_mean_from_csv_series(df["A4-2"], df["A4-5"], 1) -
        masked_mean_from_csv_series(df["A4-2"], df["A4-5"], 2)
    )
    feats["A4_acc_con"]   = masked_rate_equals(df["A4-1"], df["A4-3"], {1}, positive="1")
    feats["A4_acc_incon"] = masked_rate_equals(df["A4-1"], df["A4-3"], {2}, positive="1")
    feats["A4_acc_gap_incon_con"] = feats["A4_acc_incon"] - feats["A4_acc_con"]

    feats["A5_acc_rate"]   = seq_rate(df["A5-2"], "1")
    feats["A5_resp2_rate"] = seq_rate(df["A5-3"], "1")
    feats["A5_acc_nonchange"] = masked_mean_from_csv_series(df["A5-1"], df["A5-2"], 1)
    feats["A5_acc_change"]    = masked_mean_in_set_series(df["A5-1"], df["A5-2"], {2,3,4})

    # A6, A7, A9 cognitive score
    a6_cols = [c for c in df.columns if c.startswith('A6-')]
    a7_cols = [c for c in df.columns if c.startswith('A7-')]

    feats["CogScore_A"] = 0.0
    if a6_cols:
        feats["CogScore_A"] += df[a6_cols].sum(axis=1)
    if a7_cols:
        feats["CogScore_A"] += df[a7_cols].sum(axis=1)
    if 'A9-4' in df.columns:
        feats["CogScore_A"] += df['A9-4'].fillna(0)

    # =====  [신규] A8/A9 피처 추가  =====
    # A8 - 타당도 (응답 왜곡, 비일관성)
    if _has(df, ["A8-1", "A8-2"]):
        feats["A8_Validity_Score"] = df["A8-1"].fillna(0) + df["A8-2"].fillna(0)
    
    # A9 - 정서/행동 점수 (A9-4는 이미 CogScore_A에 있으므로 제외)
    # (A9-1: 정서안정성, A9-2: 행동안정성, A9-3: 현실판단력, A9-5: 생활스트레스)
    a9_emo_cols = ["A9-1", "A9-2", "A9-3", "A9-5"]
    if all(c in df.columns for c in a9_emo_cols):
        # 점수가 높을수록 불안정/스트레스가 높다고 가정
        feats["A9_Emotional_Score"] = df[a9_emo_cols].fillna(0).sum(axis=1)

    seq_cols = [
        "A1-1","A1-2","A1-3","A1-4",
        "A2-1","A2-2","A2-3","A2-4",
        "A3-1","A3-2","A3-3","A3-4","A3-5","A3-6","A3-7",
        "A4-1","A4-2","A4-3","A4-4","A4-5",
        "A5-1","A5-2","A5-3",
    ]
    out = pd.concat([df.drop(columns=seq_cols, errors="ignore"), feats], axis=1)

    for col in out.columns:
        if col.endswith(("_rate", "_resp_rate", "_acc_rate")):
            out[col] = _clip01(out[col].astype(float))

    out.replace([np.inf,-np.inf], np.nan, inplace=True)
    return out

def preprocess_B(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age_num"] = df["Age"].map(convert_age)
    ym = df["TestDate"].map(split_testdate)
    df["Year"]  = [y for y, m in ym]
    df["Month"] = [m for y, m in ym]

    feats = pd.DataFrame(index=df.index)

    feats["B1_acc_task1"] = seq_rate(df["B1-1"], "1")
    feats["B1_rt_mean"]   = seq_mean(df["B1-2"])
    feats["B1_rt_std"]    = seq_std(df["B1-2"])
    feats["B1_acc_task2"] = seq_rate(df["B1-3"], "1")

    feats["B2_acc_task1"] = seq_rate(df["B2-1"], "1")
    feats["B2_rt_mean"]   = seq_mean(df["B2-2"])
    feats["B2_rt_std"]    = seq_std(df["B2-2"])
    feats["B2_acc_task2"] = seq_rate(df["B2-3"], "1")

    feats["B3_acc_rate"] = seq_rate(df["B3-1"], "1")
    feats["B3_rt_mean"]  = seq_mean(df["B3-2"])
    feats["B3_rt_std"]   = seq_std(df["B3-2"])

    feats["B4_acc_rate"] = seq_rate(df["B4-1"], "1")
    feats["B4_rt_mean"]  = seq_mean(df["B4-2"])
    feats["B4_rt_std"]   = seq_std(df["B4-2"])

    feats["B5_acc_rate"] = seq_rate(df["B5-1"], "1")
    feats["B5_rt_mean"]  = seq_mean(df["B5-2"])
    feats["B5_rt_std"]   = seq_std(df["B5-2"])

    feats["B6_acc_rate"] = seq_rate(df["B6"], "1")
    feats["B7_acc_rate"] = seq_rate(df["B7"], "1")
    feats["B8_acc_rate"] = seq_rate(df["B8"], "1")

    # B9
    if all(c in df.columns for c in ["B9-1","B9-2","B9-3","B9-4","B9-5"]):
        B9_AUD_TARGET = 15.0
        B9_AUD_DIST   = 35.0
        B9_VIS_TRIALS = 32.0

        b9_hit  = df["B9-1"].astype(float)
        b9_miss = df["B9-2"].astype(float)
        b9_fa   = df["B9-3"].astype(float)
        b9_cr   = df["B9-4"].astype(float)
        b9_err  = df["B9-5"].astype(float)

        feats["B9_aud_hit_rate"]  = _safe_div(b9_hit,  B9_AUD_TARGET)
        feats["B9_aud_miss_rate"] = _safe_div(b9_miss, B9_AUD_TARGET)
        feats["B9_aud_fa_rate"]   = _safe_div(b9_fa,   B9_AUD_DIST)
        feats["B9_aud_cr_rate"]   = _safe_div(b9_cr,   B9_AUD_DIST)

        feats["B9_aud_overall_acc"] = _safe_div(
            b9_hit + b9_cr, B9_AUD_TARGET + B9_AUD_DIST
        )
        feats["B9_aud_sensitivity"] = (
            feats["B9_aud_hit_rate"] - feats["B9_aud_fa_rate"]
        )

        feats["B9_vis_err_rate"] = _safe_div(b9_err, B9_VIS_TRIALS)

    # B10
    if all(c in df.columns for c in ["B10-1","B10-2","B10-3","B10-4","B10-5","B10-6"]):
        B10_AUD_TARGET  = 20.0
        B10_AUD_DIST    = 60.0
        B10_VIS1_TRIALS = 52.0
        B10_VIS2_TRIALS = 20.0

        b10_hit  = df["B10-1"].astype(float)
        b10_miss = df["B10-2"].astype(float)
        b10_fa   = df["B10-3"].astype(float)
        b10_cr   = df["B10-4"].astype(float)
        b10_err1 = df["B10-5"].astype(float)
        b10_ok2  = df["B10-6"].astype(float)

        feats["B10_aud_hit_rate"]  = _safe_div(b10_hit,  B10_AUD_TARGET)
        feats["B10_aud_miss_rate"] = _safe_div(b10_miss, B10_AUD_TARGET)
        feats["B10_aud_fa_rate"]   = _safe_div(b10_fa,   B10_AUD_DIST)
        feats["B10_aud_cr_rate"]   = _safe_div(b10_cr,   B10_AUD_DIST)

        feats["B10_aud_overall_acc"] = _safe_div(
            b10_hit + b10_cr, B10_AUD_TARGET + B10_AUD_DIST
        )
        feats["B10_aud_sensitivity"] = (
            feats["B10_aud_hit_rate"] - feats["B10_aud_fa_rate"]
        )

        feats["B10_vis1_err_rate"] = _safe_div(b10_err1, B10_VIS1_TRIALS)
        feats["B10_vis2_acc_rate"] = _safe_div(b10_ok2,  B10_VIS2_TRIALS)

    # B9 vs B10 multitask cost
    if "B9_aud_overall_acc" in feats.columns and "B10_aud_overall_acc" in feats.columns:
        feats["B10_multitask_cost_aud"] = (
            feats["B10_aud_overall_acc"] - feats["B9_aud_overall_acc"]
        )
    if "B9_vis_err_rate" in feats.columns and "B10_vis1_err_rate" in feats.columns:
        feats["B10_multitask_cost_vis"] = (
            feats["B10_vis1_err_rate"] - feats["B9_vis_err_rate"]
        )

    seq_cols = [
        "B1-1","B1-2","B1-3",
        "B2-1","B2-2","B2-3",
        "B3-1","B3-2",
        "B4-1","B4-2",
        "B5-1","B5-2",
        "B6","B7","B8",
        "B9-1","B9-2","B9-3","B9-4","B9-5",
        "B10-1","B10-2","B10-3","B10-4","B10-5","B10-6",
    ]
    out = pd.concat([df.drop(columns=seq_cols, errors="ignore"), feats], axis=1)

    for col in out.columns:
        if col.endswith("_rate"):
            out[col] = _clip01(out[col].astype(float))

    out.replace([np.inf,-np.inf], np.nan, inplace=True)
    return out

# =============================================================================
# 고급 파생 + 인디케이터 피처
# =============================================================================

def blend_by_age_with_buffer(
    age_array: np.ndarray,
    p_under: np.ndarray,
    p_over: np.ndarray,
    split: float = AGE_SPLIT_A,
    buf_min: float = AGE_BUFFER_MIN_A,
    buf_max: float = AGE_BUFFER_MAX_A,
) -> np.ndarray:
    """
    Age_num을 기준으로 under/over 두 모델의 확률을 버퍼 구간에서 부드럽게 블렌딩.
    - age <= buf_min  : 100% under 모델
    - age >= buf_max  : 100% over 모델
    - buf_min < age < buf_max : 선형 보간
    - age NaN         : 100% under 모델 (학습 시와 동일)
    """
    age = np.asarray(age_array, dtype=float)
    p_under = np.asarray(p_under, dtype=float)
    p_over = np.asarray(p_over, dtype=float)

    # 기본: 전부 under로 두고 시작
    w_over = np.zeros_like(age, dtype=float)

    # NaN 처리: 학습 시와 동일하게 under 모델 사용
    mask_nan = np.isnan(age)
    w_over[mask_nan] = 0.0  # 100% under

    # 완충 바깥 구간
    mask_low  = (~mask_nan) & (age <= buf_min)
    mask_high = (~mask_nan) & (age >= buf_max)
    w_over[mask_low]  = 0.0   # 순수 under
    w_over[mask_high] = 1.0   # 순수 over

    # 버퍼 구간: 선형 보간
    mask_buf = (~mask_nan) & (age > buf_min) & (age < buf_max)
    w_over[mask_buf] = (age[mask_buf] - buf_min) / (buf_max - buf_min)

    w_under = 1.0 - w_over
    blended = w_under * p_under + w_over * p_over
    return blended

def add_features_A(df: pd.DataFrame, train_age_mean=None, train_age_std=None, 
                   train_a8_validity_threshold=None, train_a9_emotional_threshold=None) -> pd.DataFrame:
    feats = df.copy(); eps = 1e-6

    # -----------------------
    # 기존 고급 파생
    # -----------------------
    if _has(feats, ["Year","Month"]):
        feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]

    # ===== 새로 추가: Age_z 및 Age bin 더미 =====
    if "Age_num" in feats.columns:
        # Age_z (표준화) - train 기준 mean/std 사용 (제공되면)
        if train_age_mean is not None and train_age_std is not None:
            age_mean = train_age_mean
            age_std = train_age_std
        else:
            age_mean = feats["Age_num"].mean()
            age_std = feats["Age_num"].std()
        
        if age_std > eps:
            feats["Age_z"] = ((feats["Age_num"] - age_mean) / age_std).astype(np.float32)
        else:
            feats["Age_z"] = 0.0
        
        # Age bin 더미 변수 (안전하게 처리)
        try:
            age_num_clean = feats["Age_num"].fillna(50)  # NaN은 중간값으로
            age_bin = pd.cut(
                age_num_clean,
                bins=[0, 50, 60, 100],
                labels=["<50", "50-59", "60+"],
                include_lowest=True,
                duplicates='drop'
            )
            age_dummies = pd.get_dummies(age_bin, prefix="Age_bin", dummy_na=False)
            # 모든 카테고리 확보 (누락된 카테고리도 0으로 채움)
            expected_cols = ["Age_bin_<50", "Age_bin_50-59", "Age_bin_60+"]
            for col in expected_cols:
                if col in age_dummies.columns:
                    feats[col] = age_dummies[col].astype(np.int8)
                else:
                    feats[col] = 0
        except Exception as e:
            # 실패 시 기본값
            for col in ["Age_bin_<50", "Age_bin_50-59", "Age_bin_60+"]:
                feats[col] = 0

    for k in ["A1","A2","A4"]:
        m = f"{k}_rt_mean"
        acc = "resp_rate" if k in ["A1","A2"] else "acc_rate"
        acc_col = f"{k}_{acc}"
        if _has(feats, [m, acc_col]):
            feats[f"{k}_speed_acc_tradeoff"] = _safe_div(feats[m], feats[acc_col], eps)

    for k in ["A1","A2","A3","A4"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if _has(feats, [m, s]):
            feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    for name, base in [
        ("A1_rt_side_gap_abs",  "A1_rt_side_diff"),
        ("A1_rt_speed_gap_abs", "A1_rt_speed_diff"),
        ("A2_rt_cond1_gap_abs", "A2_rt_cond1_diff"),
        ("A2_rt_cond2_gap_abs", "A2_rt_cond2_diff"),
        ("A4_stroop_gap_abs",   "A4_stroop_diff"),
        ("A4_color_gap_abs",    "A4_rt_color_diff"),
        ("A3_valid_invalid_gap_abs", "A3_rt_valid_invalid_gap"),
        ("A4_acc_gap_abs", "A4_acc_gap_incon_con"),
    ]:
        if base in feats.columns:
            feats[name] = feats[base].abs()

    if _has(feats, ["A3_valid_ratio","A3_invalid_ratio"]):
        feats["A3_valid_invalid_ratio_gap"] = (
            feats["A3_valid_ratio"] - feats["A3_invalid_ratio"]
        )
    if _has(feats, ["A3_correct_ratio","A3_invalid_ratio"]):
        feats["A3_correct_invalid_gap"] = (
            feats["A3_correct_ratio"] - feats["A3_invalid_ratio"]
        )
    if _has(feats, ["A5_acc_change","A5_acc_nonchange"]):
        feats["A5_change_nonchange_gap"] = (
            feats["A5_acc_change"] - feats["A5_acc_nonchange"]
        )

    if _has(feats, ["A4_stroop_diff","A4_acc_rate"]):
        feats["A4_stroop_x_err"] = feats["A4_stroop_diff"] * (
            1 - feats["A4_acc_rate"].fillna(0)
        )
    if _has(feats, ["Age_num","A1_rt_mean"]):
        feats["Age_x_A1_RT"] = feats["Age_num"] * feats["A1_rt_mean"]

    # ===== 새로 추가: 추가 Age 인터랙션 =====
    if _has(feats, ["Age_num", "A4_stroop_diff"]):
        feats["Age_x_A4_stroop"] = feats["Age_num"] * feats["A4_stroop_diff"]
    
    if _has(feats, ["Age_num", "A4_acc_rate"]):
        feats["Age_x_A4_acc_inv"] = feats["Age_num"] * (1 - feats["A4_acc_rate"].fillna(0))
    
    if _has(feats, ["Age_num", "A5_change_nonchange_gap"]):
        feats["Age_x_A5_gap"] = feats["Age_num"] * feats["A5_change_nonchange_gap"]
    
    if _has(feats, ["Age_num", "CogScore_A"]):
        feats["Age_x_CogScore"] = feats["Age_num"] * feats["CogScore_A"]

    # -----------------------
    # 인디케이터 피처 (cutpoint 분석 결과 반영)
    # -----------------------

    # (1) 나이 60세 이상
    if "Age_num" in feats.columns:
        feats["Age_60plus"] = (feats["Age_num"].fillna(0) >= 60).astype(np.int8)

    # (2) A4 정확도 매우 낮음 (A4_acc_rate <= 0.8)
    if "A4_acc_rate" in feats.columns:
        feats["A4_low_acc_flag"] = (feats["A4_acc_rate"].fillna(1.0) <= 0.8).astype(np.int8)

    # (3) CogScore_A 낮음 (고정 임계값 26)
    if "CogScore_A" in feats.columns:
        feats["A_low_cog_flag"] = (feats["CogScore_A"].fillna(100) <= 26).astype(np.int8)

    # ===== 새로 추가: 추가 Indicator 피처 =====
    # 주의: quantile threshold는 train 데이터 기준으로 계산되어야 함
    # 여기서는 기본값만 설정하고, 실제 threshold는 외부에서 주입 필요
    if "A4_stroop_diff" in feats.columns:
        # 임시로 상위 25% 기준 (실제로는 train 데이터 기준 threshold 사용 권장)
        try:
            stroop_data = feats["A4_stroop_diff"].dropna()
            if len(stroop_data) > 0:
                stroop_threshold = stroop_data.quantile(0.75)
                feats["A4_high_stroop_flag"] = (feats["A4_stroop_diff"].fillna(0) > stroop_threshold).astype(np.int8)
            else:
                feats["A4_high_stroop_flag"] = 0
        except Exception:
            feats["A4_high_stroop_flag"] = 0
    
    if "A1_rt_mean" in feats.columns:
        try:
            rt_data = feats["A1_rt_mean"].dropna()
            if len(rt_data) > 0:
                rt_threshold = rt_data.quantile(0.75)
                feats["A1_slow_rt_flag"] = (feats["A1_rt_mean"].fillna(0) > rt_threshold).astype(np.int8)
            else:
                feats["A1_slow_rt_flag"] = 0
        except Exception:
            feats["A1_slow_rt_flag"] = 0

    # (4) 고령 + A4 저정확도 결합
    if _has(feats, ["Age_num", "A4_acc_rate"]):
        feats["Old_and_low_A4"] = (
            (feats["Age_num"].fillna(0) >= 60) &
            (feats["A4_acc_rate"].fillna(1.0) <= 0.8)
        ).astype(np.int8)

    # =====  [신규] A8/A9 기반 Indicator 및 Interaction  =====
    # (5) A8 타당도 점수가 높은 그룹 (비일관적이거나 왜곡된 응답)
    # train 데이터 기준 threshold 사용 (과적합 방지)
    if "A8_Validity_Score" in feats.columns:
        try:
            if train_a8_validity_threshold is not None:
                validity_threshold = train_a8_validity_threshold
            else:
                # fallback: 현재 데이터 기준 (train 데이터만 있을 때)
                validity_data = feats["A8_Validity_Score"].dropna()
                if len(validity_data) > 0:
                    validity_threshold = validity_data.quantile(0.90)
                else:
                    validity_threshold = 0
            
            feats["A8_Invalid_Flag"] = (feats["A8_Validity_Score"].fillna(0) > validity_threshold).astype(np.int8)
        except Exception:
            feats["A8_Invalid_Flag"] = 0
    
    # (6) A9 정서/스트레스 점수가 높은 그룹
    # train 데이터 기준 threshold 사용 (과적합 방지)
    if "A9_Emotional_Score" in feats.columns:
        try:
            if train_a9_emotional_threshold is not None:
                emo_threshold = train_a9_emotional_threshold
            else:
                # fallback: 현재 데이터 기준 (train 데이터만 있을 때)
                emo_data = feats["A9_Emotional_Score"].dropna()
                if len(emo_data) > 0:
                    emo_threshold = emo_data.quantile(0.90)  # 상위 10%
                else:
                    emo_threshold = 0
            
            feats["A9_High_Stress_Flag"] = (feats["A9_Emotional_Score"].fillna(0) > emo_threshold).astype(np.int8)
        except Exception:
            feats["A9_High_Stress_Flag"] = 0
    
    # (7) Age x Emotional Interaction
    if _has(feats, ["Age_num", "A9_Emotional_Score"]):
        feats["Age_x_A9_Stress"] = feats["Age_num"] * feats["A9_Emotional_Score"]
    
    # (8) Old + High Stress 결합 (A-Submodel의 핵심)
    if _has(feats, ["Age_60plus", "A9_High_Stress_Flag"]):
        feats["Old_and_High_Stress"] = (
            (feats["Age_60plus"] == 1) & (feats["A9_High_Stress_Flag"] == 1)
        ).astype(np.int8)

    feats.replace([np.inf,-np.inf], np.nan, inplace=True)
    return feats

def add_features_B(df: pd.DataFrame, train_age_mean=None, train_age_std=None) -> pd.DataFrame:
    feats = df.copy(); eps = 1e-6

    # -----------------------
    # 기존 고급 파생 (RiskScore_B 포함)
    # -----------------------
    if _has(feats, ["Year","Month"]):
        feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]

    for k, acc_col, rt_col in [
        ("B1", "B1_acc_task1", "B1_rt_mean"),
        ("B2", "B2_acc_task1", "B2_rt_mean"),
        ("B3", "B3_acc_rate",  "B3_rt_mean"),
        ("B4", "B4_acc_rate",  "B4_rt_mean"),
        ("B5", "B5_acc_rate",  "B5_rt_mean"),
    ]:
        if _has(feats, [rt_col, acc_col]):
            feats[f"{k}_speed_acc_tradeoff"] = _safe_div(feats[rt_col], feats[acc_col], eps)

    for k in ["B1","B2","B3","B4","B5"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if _has(feats, [m, s]):
            feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    parts = []
    for k in ["B4","B5"]:
        if _has(feats, [f"{k}_rt_cv"]):
            parts.append(0.25 * feats[f"{k}_rt_cv"].fillna(0))
    for k in ["B3","B4","B5"]:
        acc = f"{k}_acc_rate"
        if acc in feats:
            parts.append(0.25 * (1 - feats[acc].fillna(0)))
    for k in ["B1","B2"]:
        tcol = f"{k}_speed_acc_tradeoff"
        if tcol in feats:
            parts.append(0.25 * feats[tcol].fillna(0))
    if parts:
        feats["RiskScore_B"] = sum(parts)

    rt_cols = [c for c in feats.columns if c.endswith("_rt_mean")]
    if rt_cols:
        feats["B_rt_mean_global"] = feats[rt_cols].mean(axis=1)
        feats["B_rt_std_global"]  = feats[rt_cols].std(axis=1)

    acc_cols = [c for c in feats.columns if c.endswith("_acc_rate")]
    if acc_cols:
        feats["B_acc_mean_global"] = feats[acc_cols].mean(axis=1)
        feats["B_acc_std_global"]  = feats[acc_cols].std(axis=1)

    if _has(feats, ["B4_acc_rate","B3_acc_rate"]):
        feats["B4_B3_acc_gap"] = feats["B4_acc_rate"] - feats["B3_acc_rate"]
    if _has(feats, ["B4_rt_mean","B3_rt_mean"]):
        feats["B4_B3_rt_gap"]  = feats["B4_rt_mean"] - feats["B3_rt_mean"]

    if _has(feats, ["B5_acc_rate","B1_acc_task1"]):
        feats["B5_B1_acc_gap"] = feats["B5_acc_rate"] - feats["B1_acc_task1"]
    if _has(feats, ["B5_rt_mean","B1_rt_mean"]):
        feats["B5_B1_rt_gap"]  = feats["B5_rt_mean"] - feats["B1_rt_mean"]

    # B10 vs B9 multitask ratio (멀티태스킹 비율)
    if _has(feats, ["B10_aud_overall_acc", "B9_aud_overall_acc"]):
        feats["B10_multitask_ratio_aud"] = _safe_div(
            feats["B10_aud_overall_acc"],
            feats["B9_aud_overall_acc"],
            eps
        )

    # -----------------------
    # 인디케이터 피처 (cutpoint 분석 결과 반영)
    # -----------------------

    # (1) 나이 60세 이상
    if "Age_num" in feats.columns:
        feats["Age_60plus"] = (feats["Age_num"].fillna(0) >= 60).astype(np.int8)

    # (1-1) 나이 80세 이상 (초고위험군)
    if "Age_num" in feats.columns:
        feats["Age_80plus"] = (feats["Age_num"].fillna(0) >= 80).astype(np.int8)

    # ===== 새로 추가: B검사 Indicator 피처 =====
    if "B4_acc_rate" in feats.columns:
        feats["B4_low_acc_flag"] = (feats["B4_acc_rate"].fillna(1.0) < 0.85).astype(np.int8)
    
    if "B4_rt_mean" in feats.columns:
        try:
            b4_rt_data = feats["B4_rt_mean"].dropna()
            if len(b4_rt_data) > 0:
                b4_rt_threshold = b4_rt_data.quantile(0.75)
                feats["B4_slow_rt_flag"] = (feats["B4_rt_mean"].fillna(0) > b4_rt_threshold).astype(np.int8)
            else:
                feats["B4_slow_rt_flag"] = 0
        except Exception:
            feats["B4_slow_rt_flag"] = 0
    
    if "RiskScore_B" in feats.columns:
        try:
            risk_data = feats["RiskScore_B"].dropna()
            if len(risk_data) > 0:
                risk_threshold = risk_data.quantile(0.75)
                feats["B_high_risk_flag"] = (feats["RiskScore_B"].fillna(0) > risk_threshold).astype(np.int8)
            else:
                feats["B_high_risk_flag"] = 0
        except Exception:
            feats["B_high_risk_flag"] = 0
    
    # 복합 조건: Old + Low B4
    if _has(feats, ["Age_num", "B4_low_acc_flag"]):
        feats["Old_and_low_B4"] = (
            (feats["Age_num"].fillna(0) >= 60) & (feats["B4_low_acc_flag"] == 1)
        ).astype(np.int8)

    # (2) RiskScore_B 상위 20% (high risk 그룹)
    if "RiskScore_B" in feats.columns:
        try:
            risk_data = feats["RiskScore_B"].dropna()
            if len(risk_data) > 0:
                thr = risk_data.quantile(0.8)
                feats["RiskB_top20"] = (feats["RiskScore_B"].fillna(0) >= thr).astype(np.int8)
            else:
                feats["RiskB_top20"] = 0
        except Exception:
            feats["RiskB_top20"] = 0

    # (3) B10 multitask cost의 양 극단 (very negative / very positive)
    if "B10_multitask_cost_aud" in feats.columns:
        feats["B10_multicost_very_neg"] = (
            feats["B10_multitask_cost_aud"].fillna(0) <= -0.0875
        ).astype(np.int8)
        feats["B10_multicost_very_pos"] = (
            feats["B10_multitask_cost_aud"].fillna(0) >= 0.04
        ).astype(np.int8)

    feats.replace([np.inf,-np.inf], np.nan, inplace=True)
    return feats

# =============================================================================
# History Features
# =============================================================================

def add_history_features_A(df: pd.DataFrame) -> pd.DataFrame:
    """
    A history features using PrimaryKey + YearMonthIndex
    최적화: expanding() -> rolling() 윈도우로 변경 (O(n²) -> O(n))
    """
    df = df.copy()

    if not _has(df, ["PrimaryKey", "YearMonthIndex"]):
        return df

    df = df.sort_values(["PrimaryKey", "YearMonthIndex"]).reset_index(drop=True)

    df["n_tests_so_far"] = df.groupby("PrimaryKey").cumcount() + 1

    df["months_since_first_test"] = (
        df.groupby("PrimaryKey")["YearMonthIndex"].transform(lambda x: x - x.iloc[0])
    )

    df["months_since_prev_test"] = (
        df.groupby("PrimaryKey")["YearMonthIndex"].diff().fillna(0)
    )

    # 최적화: rolling window 사용 (최근 10개 테스트만 고려)
    MAX_HISTORY_WINDOW = 10
    
    if "A4_acc_rate" in df.columns:
        df["A4_acc_prev_mean"] = (
            df.groupby("PrimaryKey")["A4_acc_rate"]
            .transform(lambda x: x.rolling(window=MAX_HISTORY_WINDOW, min_periods=1).mean().shift(1))
        )

    if "CogScore_A" in df.columns:
        df["CogScore_A_prev_mean"] = (
            df.groupby("PrimaryKey")["CogScore_A"]
            .transform(lambda x: x.rolling(window=MAX_HISTORY_WINDOW, min_periods=1).mean().shift(1))
        )
        df["CogScore_A_trend"] = (
            df["CogScore_A"] - df["CogScore_A_prev_mean"]
        )

    if "A1_rt_mean" in df.columns:
        df["A1_rt_prev_mean"] = (
            df.groupby("PrimaryKey")["A1_rt_mean"]
            .transform(lambda x: x.rolling(window=MAX_HISTORY_WINDOW, min_periods=1).mean().shift(1))
        )
        df["A1_rt_trend"] = (
            df["A1_rt_mean"] - df["A1_rt_prev_mean"]
        )

    if "A4_stroop_diff" in df.columns:
        df["A4_stroop_prev_mean"] = (
            df.groupby("PrimaryKey")["A4_stroop_diff"]
            .transform(lambda x: x.rolling(window=MAX_HISTORY_WINDOW, min_periods=1).mean().shift(1))
        )
        df["A4_stroop_trend"] = (
            df["A4_stroop_diff"] - df["A4_stroop_prev_mean"]
        )

    return df

def add_history_features_B(df: pd.DataFrame) -> pd.DataFrame:
    """
    B history features using PrimaryKey + YearMonthIndex
    최적화: rolling window 사용 (O(n²) -> O(n)), groupby 객체 재사용
    """
    df = df.copy()

    if not _has(df, ["PrimaryKey", "YearMonthIndex"]):
        return df

    df = df.sort_values(["PrimaryKey", "YearMonthIndex"]).reset_index(drop=True)

    # 기본 피처 생성
    grouped = df.groupby("PrimaryKey")  # 그룹 객체 재사용
    df["n_tests_so_far"] = grouped.cumcount() + 1

    df["months_since_first_test"] = (
        grouped["YearMonthIndex"].transform(lambda x: x - x.iloc[0])
    )

    df["months_since_prev_test"] = (
        grouped["YearMonthIndex"].diff().fillna(0)
    )

    # 최적화: rolling window 사용 (expanding 대신)
    MAX_HISTORY_WINDOW = 10
    MAX_VOLATILITY_WINDOW = 5  # 변동성 윈도우 축소로 성능 향상
    
    # Mean 피처 계산 (rolling window)
    if "B4_acc_rate" in df.columns:
        df["B4_acc_prev_mean"] = (
            grouped["B4_acc_rate"]
            .transform(lambda x: x.rolling(window=MAX_HISTORY_WINDOW, min_periods=1).mean().shift(1))
        )
        df["B4_acc_trend"] = df["B4_acc_rate"] - df["B4_acc_prev_mean"]
        df["B4_acc_trend_per_month"] = df["B4_acc_trend"] / (df["months_since_prev_test"] + 1e-6)
        df["B4_acc_trend_ratio"] = df["B4_acc_trend"] / (df["B4_acc_prev_mean"].abs() + 1e-6)

    if "B10_multitask_cost_aud" in df.columns:
        df["B10_multicost_prev_mean"] = (
            grouped["B10_multitask_cost_aud"]
            .transform(lambda x: x.rolling(window=MAX_HISTORY_WINDOW, min_periods=1).mean().shift(1))
        )
        df["B10_multicost_trend"] = df["B10_multitask_cost_aud"] - df["B10_multicost_prev_mean"]
        df["B10_multicost_trend_per_month"] = df["B10_multicost_trend"] / (df["months_since_prev_test"] + 1e-6)
        df["B10_multicost_trend_ratio"] = df["B10_multicost_trend"] / (df["B10_multicost_prev_mean"].abs() + 1e-6)

    for task in ["B3", "B4", "B5"]:
        rt_col = f"{task}_rt_mean"
        if rt_col in df.columns:
            df[f"{task}_rt_prev_mean"] = (
                grouped[rt_col]
                .transform(lambda x: x.rolling(window=MAX_HISTORY_WINDOW, min_periods=1).mean().shift(1))
            )
            df[f"{task}_rt_trend"] = df[rt_col] - df[f"{task}_rt_prev_mean"]
            df[f"{task}_rt_trend_per_month"] = df[f"{task}_rt_trend"] / (df["months_since_prev_test"] + 1e-6)
            df[f"{task}_rt_trend_ratio"] = df[f"{task}_rt_trend"] / (df[f"{task}_rt_prev_mean"].abs() + 1e-6)
    
    # 변동성 피처 (rolling window, 윈도우 크기 축소)
    if "B4_acc_rate" in df.columns:
        df["B4_acc_prev_std"] = (
            grouped["B4_acc_rate"]
            .transform(lambda x: x.rolling(window=MAX_VOLATILITY_WINDOW, min_periods=2).std().shift(1))
        )
    
    if "B10_multitask_cost_aud" in df.columns:
        df["B10_multicost_prev_std"] = (
            grouped["B10_multitask_cost_aud"]
            .transform(lambda x: x.rolling(window=MAX_VOLATILITY_WINDOW, min_periods=2).std().shift(1))
        )
    
    for task in ["B3", "B4", "B5"]:
        rt_col = f"{task}_rt_mean"
        if rt_col in df.columns:
            df[f"{task}_rt_prev_std"] = (
                grouped[rt_col]
                .transform(lambda x: x.rolling(window=MAX_VOLATILITY_WINDOW, min_periods=2).std().shift(1))
            )
    
    # Interaction 피처
    if _has(df, ["Age_num", "B4_acc_trend"]):
        df["Age_x_B4_acc_trend"] = df["Age_num"].fillna(0) * df["B4_acc_trend"].fillna(0)
    
    if _has(df, ["Age_num", "B10_multicost_trend"]):
        df["Age_x_B10_multicost_trend"] = df["Age_num"].fillna(0) * df["B10_multicost_trend"].fillna(0)

    return df

# =============================================================================
# 학습 함수 (서버 제출용 - OOF 없이 최종 모델만 학습, NumPy 에러 회피)
# =============================================================================
# 주의: 서버 환경에서는 모델 저장/로딩 대신 즉시 재학습하는 방식 사용
# 이유: NumPy BitGenerator 호환성 문제 회피 (numpy.random._pcg64.PCG64 에러)
# HistGradientBoostingClassifier는 내부적으로 병렬 처리를 사용하므로 n_jobs 제거 (안전성)

def build_model(seed: int, which: str) -> HistGradientBoostingClassifier:
    if which == "A":
        params = BASE_HGB_PARAMS_A.copy()
    else:
        params = BASE_HGB_PARAMS_B.copy()
    params["random_state"] = seed
    
    # n_jobs 파라미터 제거 (안전성을 위해)
    # HistGradientBoostingClassifier는 내부적으로 병렬 처리를 사용하므로 n_jobs가 없어도 됨
    params.pop("n_jobs", None)
    
    return HistGradientBoostingClassifier(**params)

def fit_partition_A_submodels(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    label_col: str = "Label",
):
    """
    A용: Age_num 기준으로 <60, >=60 두 개의 3-seed 앙상블(HGB)을 학습.
    서버 제출용: 모델 저장 없이 즉시 재학습 (NumPy 에러 회피)
    """
    which = "A"
    key = "Test_id"
    assert key in df_feat.columns, f"{which}: '{key}' not found in features"
    assert "Age_num" in df_feat.columns, f"{which}: 'Age_num' not found in features"

    # index와 feature merge
    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")

    drop_cols = [key, label_col] + (["Test"] if "Test" in df.columns else [])
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # row-wise NA 피처 추가
    df = add_rowwise_features(df, feature_cols)

    num_cols, cat_cols = separate_num_cat(df, drop_cols)
    preproc = build_preprocessor(num_cols, cat_cols)

    X_all = df.drop(columns=drop_cols)
    y_all = df[label_col].astype(int).values
    age_all = X_all["Age_num"].values

    # 서버 제출용: 전체 데이터로 학습 (validation 없음, 시간 절약)
    X_tr, y_tr, age_tr = X_all, y_all, age_all

    # 전처리 학습은 전체 train에 대해 1번만
    X_tr_t = preproc.fit_transform(X_tr)

    # under60 / over60 마스크
    mask_under_tr = np.isnan(age_tr) | (age_tr < AGE_SPLIT_A)
    mask_over_tr = (~np.isnan(age_tr)) & (age_tr >= AGE_SPLIT_A)

    # 혹시라도 under 또는 over 쪽에 데이터가 거의 없으면 방어
    if mask_under_tr.sum() == 0:
        print(f"[{which}] WARNING: no samples for Age < {AGE_SPLIT_A}, fallback to single model.")
        members = []
        for sd in ENSEMBLE_SEEDS:
            base = build_model(sd, which=which).fit(X_tr_t, y_tr)
            mdl = maybe_calibrate(base, X_tr_t, y_tr)
            members.append(mdl)
        ensemble = AvgProbaEnsemble(members)
        return preproc, ensemble, None

    if mask_over_tr.sum() == 0:
        print(f"[{which}] WARNING: no samples for Age >= {AGE_SPLIT_A}, fallback to single model.")
        members = []
        for sd in ENSEMBLE_SEEDS:
            base = build_model(sd, which=which).fit(X_tr_t, y_tr)
            mdl = maybe_calibrate(base, X_tr_t, y_tr)
            members.append(mdl)
        ensemble = AvgProbaEnsemble(members)
        return preproc, ensemble, None

    # ---------- under60 앙상블 (Age < 60) ----------
    X_tr_under = X_tr_t[mask_under_tr]
    y_tr_under = y_tr[mask_under_tr]

    members_under = []
    for sd in ENSEMBLE_SEEDS:
        base = build_model(sd, which=which).fit(X_tr_under, y_tr_under)
        mdl = maybe_calibrate(base, X_tr_under, y_tr_under)
        members_under.append(mdl)
    ensemble_under = AvgProbaEnsemble(members_under)

    # ---------- over60 앙상블 (Age >= 60) ----------
    X_tr_over = X_tr_t[mask_over_tr]
    y_tr_over = y_tr[mask_over_tr]

    members_over = []
    for sd in ENSEMBLE_SEEDS:
        base = build_model(sd, which=which).fit(X_tr_over, y_tr_over)
        mdl = maybe_calibrate(base, X_tr_over, y_tr_over)
        members_over.append(mdl)
    ensemble_over = AvgProbaEnsemble(members_over)

    return preproc, ensemble_under, ensemble_over

def fit_partition_B_submodels(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    label_col: str = "Label",
):
    """
    B용: n_tests_so_far 기준으로 first(==1), repeat(>1) 두 개의 3-seed 앙상블(HGB)을 학습.
    서버 제출용: 모델 저장 없이 즉시 재학습 (NumPy 에러 회피)
    """
    which = "B"
    key = "Test_id"
    assert key in df_feat.columns, f"{which}: '{key}' not found in features"
    assert "n_tests_so_far" in df_feat.columns, f"{which}: 'n_tests_so_far' not found in features"

    # index와 feature merge
    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")

    drop_cols = [key, label_col] + (["Test"] if "Test" in df.columns else [])
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # row-wise NA 피처 추가
    df = add_rowwise_features(df, feature_cols)

    num_cols, cat_cols = separate_num_cat(df, drop_cols)
    preproc = build_preprocessor(num_cols, cat_cols)

    X_all = df.drop(columns=drop_cols)
    y_all = df[label_col].astype(int).values
    n_tests_all = X_all["n_tests_so_far"].values

    # 서버 제출용: 전체 데이터로 학습 (validation 없음, 시간 절약)
    X_tr, y_tr, n_tests_tr = X_all, y_all, n_tests_all

    # 전처리 학습은 전체 train에 대해 1번만
    X_tr_t = preproc.fit_transform(X_tr)

    # first / repeat 마스크
    mask_first_tr = (n_tests_tr == 1)
    mask_repeat_tr = (n_tests_tr > 1)

    # 혹시라도 first 또는 repeat 쪽에 데이터가 거의 없으면 방어
    if mask_first_tr.sum() == 0:
        print(f"[{which}] WARNING: no samples for n_tests_so_far == 1, fallback to single model.")
        members = []
        for sd in ENSEMBLE_SEEDS:
            base = build_model(sd, which=which).fit(X_tr_t, y_tr)
            mdl = maybe_calibrate(base, X_tr_t, y_tr)
            members.append(mdl)
        ensemble = AvgProbaEnsemble(members)
        return preproc, ensemble, None

    if mask_repeat_tr.sum() == 0:
        print(f"[{which}] WARNING: no samples for n_tests_so_far > 1, fallback to single model.")
        members = []
        for sd in ENSEMBLE_SEEDS:
            base = build_model(sd, which=which).fit(X_tr_t, y_tr)
            mdl = maybe_calibrate(base, X_tr_t, y_tr)
            members.append(mdl)
        ensemble = AvgProbaEnsemble(members)
        return preproc, ensemble, None

    # ---------- first 앙상블 (n_tests_so_far == 1) ----------
    X_tr_first = X_tr_t[mask_first_tr]
    y_tr_first = y_tr[mask_first_tr]

    members_first = []
    for sd in ENSEMBLE_SEEDS:
        base = build_model(sd, which=which).fit(X_tr_first, y_tr_first)
        mdl = maybe_calibrate(base, X_tr_first, y_tr_first)
        members_first.append(mdl)
    ensemble_first = AvgProbaEnsemble(members_first)

    # ---------- repeat 앙상블 (n_tests_so_far > 1) ----------
    X_tr_repeat = X_tr_t[mask_repeat_tr]
    y_tr_repeat = y_tr[mask_repeat_tr]

    members_repeat = []
    for sd in ENSEMBLE_SEEDS:
        base = build_model(sd, which=which).fit(X_tr_repeat, y_tr_repeat)
        mdl = maybe_calibrate(base, X_tr_repeat, y_tr_repeat)
        members_repeat.append(mdl)
    ensemble_repeat = AvgProbaEnsemble(members_repeat)

    return preproc, ensemble_first, ensemble_repeat

def predict_partition(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    preproc,
    clf_or_ens,
    which: str,
) -> pd.DataFrame:
    key = "Test_id"
    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")
    drop_cols = [key] + (["Test"] if "Test" in df.columns else [])

    feature_cols = [c for c in df.columns if c not in drop_cols]
    df = add_rowwise_features(df, feature_cols)

    X = df.drop(columns=drop_cols, errors="ignore")
    X_t = preproc.transform(X)
    proba = np.clip(clf_or_ens.predict_proba(X_t)[:, 1], 1e-7, 1-1e-7)

    out = df_idx[[key]].copy()
    out["Label"] = proba
    out["__which__"] = which
    return out

def predict_partition_A_submodels(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    preproc,
    ens_under,
    ens_over,
) -> pd.DataFrame:
    """
    A용: 나이 기반 서브모델(<60, >=60) 두 개의 예측값을 버퍼 구간에서 블렌딩.
    ens_over가 None이면(single fallback) ens_under만 사용.
    """
    which = "A"
    key = "Test_id"
    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")

    drop_cols = [key] + (["Test"] if "Test" in df.columns else [])
    feature_cols = [c for c in df.columns if c not in drop_cols]

    df = add_rowwise_features(df, feature_cols)

    X = df.drop(columns=drop_cols, errors="ignore")

    if "Age_num" not in X.columns:
        raise ValueError(f"[{which}] Age_num column is required for age-based blending.")

    ages = X["Age_num"].values
    X_t = preproc.transform(X)

    if ens_over is None:
        # single model fallback
        proba = np.clip(ens_under.predict_proba(X_t)[:, 1], 1e-7, 1-1e-7)
    else:
        p_under = ens_under.predict_proba(X_t)[:, 1]
        p_over  = ens_over.predict_proba(X_t)[:, 1]
        blended = blend_by_age_with_buffer(ages, p_under, p_over)
        proba   = np.clip(blended, 1e-7, 1-1e-7)

    out = df_idx[[key]].copy()
    out["Label"] = proba
    out["__which__"] = which
    return out

def predict_partition_B_submodels(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    preproc,
    ens_first,
    ens_repeat,
) -> pd.DataFrame:
    """
    B용: n_tests_so_far 기준 서브모델(first/repeat) 두 개의 예측값을 분리 적용.
    ens_repeat가 None이면(single fallback) ens_first만 사용.
    """
    which = "B"
    key = "Test_id"
    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")

    drop_cols = [key] + (["Test"] if "Test" in df.columns else [])
    feature_cols = [c for c in df.columns if c not in drop_cols]

    df = add_rowwise_features(df, feature_cols)

    X = df.drop(columns=drop_cols, errors="ignore")

    if "n_tests_so_far" not in X.columns:
        raise ValueError(f"[{which}] n_tests_so_far column is required for history-based split.")

    n_tests = X["n_tests_so_far"].values
    X_t = preproc.transform(X)

    if ens_repeat is None:
        # single model fallback
        proba = np.clip(ens_first.predict_proba(X_t)[:, 1], 1e-7, 1-1e-7)
    else:
        # first와 repeat 각각 예측
        proba = np.zeros(len(n_tests), dtype=float)
        mask_first = (n_tests == 1)
        mask_repeat = (n_tests > 1)

        if mask_first.sum() > 0:
            proba[mask_first] = ens_first.predict_proba(X_t[mask_first])[:, 1]
        if mask_repeat.sum() > 0:
            proba[mask_repeat] = ens_repeat.predict_proba(X_t[mask_repeat])[:, 1]

        proba = np.clip(proba, 1e-7, 1-1e-7)

    out = df_idx[[key]].copy()
    out["Label"] = proba
    out["__which__"] = which
    return out

# =============================================================================
# 메인 (추론 전용)
# =============================================================================

def main():
    set_seed(RANDOM_STATE)
    t0 = time.time()
    ensure_dirs()

    print("Loading index files ...")
    train_idx, test_idx = read_index_files()  # train_idx도 필요 (재학습용)

    print("Loading raw A/B (train/test) ...")
    A_train_raw, B_train_raw = read_raw_feature_files("train")
    A_test_raw,  B_test_raw  = read_raw_feature_files("test")

    print("\nFE for A (with history features) ...")
    # Train 데이터 기준 통계량 계산 (Age_z, A8/A9 threshold용)
    A_train_prep = preprocess_A(A_train_raw)
    A_train_age_mean = A_train_prep["Age_num"].mean() if "Age_num" in A_train_prep.columns else None
    A_train_age_std = A_train_prep["Age_num"].std() if "Age_num" in A_train_prep.columns else None
    
    # Train 데이터만으로 A8/A9 threshold 계산 (과적합 방지)
    A_train_feat_base = add_features_A(A_train_prep, train_age_mean=A_train_age_mean, train_age_std=A_train_age_std)
    train_a8_validity_threshold = None
    train_a9_emotional_threshold = None
    if "A8_Validity_Score" in A_train_feat_base.columns:
        try:
            a8_data = A_train_feat_base["A8_Validity_Score"].dropna()
            if len(a8_data) > 0:
                train_a8_validity_threshold = a8_data.quantile(0.90)
        except Exception:
            pass
    if "A9_Emotional_Score" in A_train_feat_base.columns:
        try:
            a9_data = A_train_feat_base["A9_Emotional_Score"].dropna()
            if len(a9_data) > 0:
                train_a9_emotional_threshold = a9_data.quantile(0.90)
        except Exception:
            pass
    
    A_all_raw = pd.concat([A_train_raw, A_test_raw], axis=0, ignore_index=True)
    A_all = preprocess_A(A_all_raw)
    A_all = add_features_A(A_all, train_age_mean=A_train_age_mean, train_age_std=A_train_age_std,
                           train_a8_validity_threshold=train_a8_validity_threshold,
                           train_a9_emotional_threshold=train_a9_emotional_threshold)
    A_all = add_history_features_A(A_all)

    A_train_ids = set(A_train_raw["Test_id"])
    A_test_ids = set(A_test_raw["Test_id"])
    A_train_feat = A_all[A_all["Test_id"].isin(A_train_ids)].reset_index(drop=True)
    A_test_feat = A_all[A_all["Test_id"].isin(A_test_ids)].reset_index(drop=True)
    print(f"  A_train_feat: {A_train_feat.shape}, A_test_feat: {A_test_feat.shape}")

    print("\nFE for B (with history features) ...")
    # Train 데이터 기준 통계량 계산 (Age_z용)
    B_train_prep = preprocess_B(B_train_raw)
    B_train_age_mean = B_train_prep["Age_num"].mean() if "Age_num" in B_train_prep.columns else None
    B_train_age_std = B_train_prep["Age_num"].std() if "Age_num" in B_train_prep.columns else None
    
    B_all_raw = pd.concat([B_train_raw, B_test_raw], axis=0, ignore_index=True)
    B_all = preprocess_B(B_all_raw)
    B_all = add_features_B(B_all, train_age_mean=B_train_age_mean, train_age_std=B_train_age_std)
    B_all = add_history_features_B(B_all)

    B_train_ids = set(B_train_raw["Test_id"])
    B_test_ids = set(B_test_raw["Test_id"])
    B_train_feat = B_all[B_all["Test_id"].isin(B_train_ids)].reset_index(drop=True)
    B_test_feat = B_all[B_all["Test_id"].isin(B_test_ids)].reset_index(drop=True)
    print(f"  B_train_feat: {B_train_feat.shape}, B_test_feat: {B_test_feat.shape}")

    # ===== A 학습 (Age-based submodels) - OOF 없이 전체 데이터로 학습 =====
    A_train_idx = train_idx[train_idx["Test"] == "A"].copy()
    A_test_idx = test_idx[test_idx["Test"] == "A"].copy()
    print("\n[A] training with age-based submodels (no OOF, server optimized) ...")
    print(f"  Train samples: {len(A_train_idx)}, Test samples: {len(A_test_idx)}")
    t_a_start = time.time()
    preproc_A, clf_A_under, clf_A_over = fit_partition_A_submodels(A_train_feat, A_train_idx, "Label")
    t_a_elapsed = time.time() - t_a_start
    print(f"  [A] Training completed in {t_a_elapsed/60:.2f} min")
    
    # 서버 제출용: Train 평가 스킵 (시간 절약, 30분 제한 준수)

    # ===== B 학습 (History-based submodels) - OOF 없이 전체 데이터로 학습 =====
    B_train_idx = train_idx[train_idx["Test"] == "B"].copy()
    B_test_idx = test_idx[test_idx["Test"] == "B"].copy()
    print("\n[B] training with history-based submodels (no OOF, server optimized) ...")
    print(f"  Train samples: {len(B_train_idx)}, Test samples: {len(B_test_idx)}")
    t_b_start = time.time()
    preproc_B, clf_B_first, clf_B_repeat = fit_partition_B_submodels(B_train_feat, B_train_idx, "Label")
    t_b_elapsed = time.time() - t_b_start
    print(f"  [B] Training completed in {t_b_elapsed/60:.2f} min")
    
    # 서버 제출용: Train 평가 스킵 (시간 절약, 30분 제한 준수)

    # ===== 추론 =====
    A_test_idx = test_idx[test_idx["Test"] == "A"].copy()
    B_test_idx = test_idx[test_idx["Test"] == "B"].copy()
    print("\nInference on test ...")
    preds_A = (
        predict_partition_A_submodels(A_test_feat, A_test_idx, preproc_A, clf_A_under, clf_A_over)
        if len(A_test_idx) else None
    )
    preds_B = (
        predict_partition_B_submodels(B_test_feat, B_test_idx, preproc_B, clf_B_first, clf_B_repeat)
        if len(B_test_idx) else None
    )

    if preds_A is not None and preds_B is not None:
        sub = pd.concat([preds_A, preds_B], axis=0, ignore_index=True)
    elif preds_A is not None:
        sub = preds_A.copy()
    elif preds_B is not None:
        sub = preds_B.copy()
    else:
        sub = test_idx[["Test_id"]].copy()
        sub["Label"] = 0.001

    # sample_submission 순서 맞추기
    try:
        sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
        sub = sub.merge(sample[["Test_id"]], on="Test_id", how="right")
        sub = sub[["Test_id", "Label"]]
    except Exception:
        sub = sub[["Test_id", "Label"]]

    # submission.csv 저장 (서버 환경: output/ 디렉토리)
    sub.to_csv(SUBMISSION_PATH, index=False)

    dt = time.time() - t0
    print(f"\n submission saved -> {SUBMISSION_PATH} | elapsed: {dt/60:.2f} min")
    
    # 서버 환경: CPU 30분 제한 확인
    TIME_LIMIT = 1800  # 30분 = 1800초
    if dt > TIME_LIMIT:
        print(f"  경고: 실행 시간 30분 초과 ({dt/60:.2f} min)")
    else:
        print(f" 실행 시간 30분 이내 ({dt/60:.2f} min / {TIME_LIMIT/60:.0f} min)")
    print(f" models trained on-the-fly (no model loading, NumPy error avoided)")

if __name__ == "__main__":
    main()
