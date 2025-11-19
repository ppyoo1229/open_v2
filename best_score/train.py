#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import json
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn import __version__ as sklver

HAS_LIGHTGBM = False

import pickle  

# =============================================================================
# 경로 설정
# =============================================================================

DATA_DIR   = "data"
OUTPUT_DIR = "output"
MODEL_DIR  = "model"

SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
META_PATH       = os.path.join(MODEL_DIR, "meta.json")
BEST_PARAMS_PATH = os.path.join(MODEL_DIR, "best_params.json")

A_MODEL_PATH    = os.path.join(MODEL_DIR, "hgb_A.pkl")
B_MODEL_PATH    = os.path.join(MODEL_DIR, "hgb_B.pkl")
A_PREPROC_PATH  = os.path.join(MODEL_DIR, "preproc_A.pkl")
B_PREPROC_PATH  = os.path.join(MODEL_DIR, "preproc_B.pkl")

RANDOM_STATE = 42

# 실제 사용된 iteration 수 저장용 (전역 변수)
ACTUAL_ITERS_A = None
ACTUAL_ITERS_B = None

# =============================================================================
# History-based submodel 설정 (B 전용)
# =============================================================================
# B는 n_tests_so_far 기준으로 분리
# n_tests_so_far == 1: 첫 검사 (히스토리 피처 NaN)
# n_tests_so_far > 1: 재검사 (히스토리 피처 활용)
# =============================================================================
# 실행 옵션
# =============================================================================
USE_CALIBRATION = True
CALIB_METHOD   = "isotonic"
CALIB_CV       = 3
TEST_SIZE_FOR_LOG = 0.1

USE_OOF = True  
N_SPLITS = 2  
ENSEMBLE_SEEDS: Sequence[int] = (42, 202, 777, 1001, 8888)  # 3개 -> 5개

# 하이퍼파라미터 
BASE_HGB_PARAMS_A = dict(
    learning_rate=0.05,
    max_iter=1500,  
    max_depth=None,
    max_leaf_nodes=63,
    min_samples_leaf=20,
    l2_regularization=0.0,
    early_stopping=True,  # OOF와 서버 모두 켜기
    validation_fraction=0.15,
    n_iter_no_change=30,
    class_weight="balanced",
)

BASE_HGB_PARAMS_B = dict(
    learning_rate=0.02,
    max_iter=1500,  # 넉넉히 설정, early_stopping이 실제 iter 결정
    max_depth=None,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.0,
    early_stopping=True,  # OOF와 서버 모두 켜기
    validation_fraction=0.15,
    n_iter_no_change=40,
    class_weight="balanced",
)

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

def _mk_calibrator(base_clf, cv=None):
    """
    Calibrator 생성
    - cv: None이면 CALIB_CV 사용
    - 현재 구현은 항상 K-fold(CALIB_CV) 기반 calibration만 사용
      (prefit 모드는 사용하지 않음)
    """
    if cv is None:
        cv = CALIB_CV
    try:
        major, minor, *_ = map(int, sklver.split(".")[:2])
    except Exception:
        major, minor = 1, 4
    kw = dict(method=CALIB_METHOD, cv=cv)
    if (major, minor) >= (1, 4):
        return CalibratedClassifierCV(estimator=base_clf, **kw)
    else:
        return CalibratedClassifierCV(base_estimator=base_clf, **kw)

def maybe_calibrate(base_clf, X_train, y_train):
    """
    Calibration 수행 (전체 데이터 사용)
    - Final 모델 학습 시 사용 (전체 데이터로 calibration)
    """
    if not USE_CALIBRATION:
        return base_clf
    calib = _mk_calibrator(base_clf)
    calib.fit(X_train, y_train)
    return calib

def train_and_calibrate_with_split(X_train, y_train, seed, which, model_type="hgb", calib_test_size=0.2, min_samples=50):
    """
    Base model 학습 + Calibration (train/calib 분리)
    - OOF 학습 시 사용 (overfitting 방지)
    - base model 학습 데이터를 train/calib으로 분리
    - base model은 train으로 학습, calibration은 calib으로 수행
    - 최소 샘플 수 미만이면 분리하지 않음 (데이터 부족 시)
    - 클래스 수 부족 시 calibration 생략 (base model만 사용)
    
    Args:
        X_train: base model 학습에 사용할 전체 데이터
        y_train: base model 학습에 사용할 레이블
        seed: random seed
        which: "A" or "B"
        model_type: "hgb" (호환성)
        calib_test_size: calibration용으로 분리할 비율 (기본: 0.2)
        min_samples: 분리를 위한 최소 샘플 수 (기본: 50)
    
    Returns:
        tuple: (calibrated_classifier, base_clf)
        - calibrated_classifier: Calibrated classifier (또는 base_clf if USE_CALIBRATION=False or 클래스 수 부족)
        - base_clf: 학습된 base classifier (n_iter_ 확인용)
    """
    # 최소 샘플 수 미만이면 분리하지 않고 전체 사용
    if len(X_train) < min_samples:
        base_clf = build_model(seed, which, model_type, for_oof=True)
        base_clf.fit(X_train, y_train)
        if not USE_CALIBRATION:
            return base_clf, base_clf
        
        # 클래스 수 부족 체크 (calibration 불가능한 경우)
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            # 클래스가 1개만 있으면 calibration 불가 → base_clf 반환
            return base_clf, base_clf
        
        # 클래스별 샘플 수 체크 (CALIB_CV보다 적으면 에러 발생 가능)
        class_counts = np.bincount(y_train.astype(int))
        if class_counts.min() < CALIB_CV:
            # calibration 시 CV 에러 발생 가능 → base_clf 반환
            return base_clf, base_clf
        
        try:
            calib = _mk_calibrator(base_clf, cv=CALIB_CV)
            calib.fit(X_train, y_train)
            return calib, base_clf
        except (ValueError, RuntimeError) as e:
            # CalibratedClassifierCV에서 클래스 수 부족 등 에러 시 fallback
            return base_clf, base_clf
    
    # train/calib 분리 (base model 학습용 / calibration용)
    try:
        X_tr_base, X_calib, y_tr_base, y_calib = train_test_split(
            X_train, y_train,
            test_size=calib_test_size,
            random_state=seed,  # seed 사용하여 재현성 보장
            stratify=y_train if len(np.unique(y_train)) > 1 else None,
        )
    except ValueError:
        # stratification 실패 시 (예: 클래스가 하나만 있는 경우)
        X_tr_base, X_calib, y_tr_base, y_calib = train_test_split(
            X_train, y_train,
            test_size=calib_test_size,
            random_state=seed,
        )
    
    # base model을 train 부분으로 학습
    base_clf = build_model(seed, which, model_type, for_oof=True)
    base_clf.fit(X_tr_base, y_tr_base)
    
    # calibration은 calib 데이터로 수행
    if not USE_CALIBRATION:
        return base_clf, base_clf
    
    # y_calib 클래스 수 체크 (calibration 불가능한 경우 방어)
    unique_classes = np.unique(y_calib)
    if len(unique_classes) < 2:
        # 클래스가 1개만 있으면 calibration 불가 → base_clf 반환
        return base_clf, base_clf
    
    # 클래스별 샘플 수 체크 (CALIB_CV보다 적으면 CV 에러 발생 가능)
    class_counts = np.bincount(y_calib.astype(int))
    if class_counts.min() < CALIB_CV:
        # calibration 시 CV 에러 발생 가능 → base_clf 반환
        return base_clf, base_clf
    
    try:
        calib = _mk_calibrator(base_clf, cv=CALIB_CV)
        calib.fit(X_calib, y_calib)
        return calib, base_clf
    except (ValueError, RuntimeError) as e:
        # 마지막 안전장치: CalibratedClassifierCV에서 클래스 수 부족 등 에러 시 fallback
        return base_clf, base_clf

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
# 도메인 Composite Score 및 Age-normed Score 헬퍼 함수
# =============================================================================

def _normalize_features(df, features, method='zscore'):
    """
    여러 피처를 정규화하여 결합
    
    Parameters:
    -----------
    df : DataFrame
    features : list - 정규화할 피처명 리스트
    method : str - 'zscore' or 'minmax'
    
    Returns:
    --------
    DataFrame - 정규화된 피처들
    """
    result = pd.DataFrame(index=df.index)
    
    for feat in features:
        if feat in df.columns:
            values = df[feat].fillna(df[feat].median())
            if method == 'zscore':
                mean_val = values.mean()
                std_val = values.std()
                if std_val > 0:
                    result[feat] = (values - mean_val) / std_val
                else:
                    result[feat] = 0
            elif method == 'minmax':
                min_val = values.min()
                max_val = values.max()
                if max_val > min_val:
                    result[feat] = (values - min_val) / (max_val - min_val)
                else:
                    result[feat] = 0
    
    return result

def _create_age_bin(age_num):
    """나이를 bin으로 변환"""
    if pd.isna(age_num):
        return 'Unknown'
    elif age_num < 50:
        return '<50'
    elif age_num < 60:
        return '50-59'
    elif age_num < 70:
        return '60-69'
    else:
        return '70+'

def _create_age_normed_score(df, feature_name, age_col='Age_num'):
    """
    나이 bin별로 정규화된 점수 생성
    
    Parameters:
    -----------
    df : DataFrame
    feature_name : str - 정규화할 피처명
    age_col : str - 나이 컬럼명
    
    Returns:
    --------
    Series - age-normed z-score
    """
    if feature_name not in df.columns:
        return pd.Series(np.nan, index=df.index)
    
    df = df.copy()
    df['_Age_bin'] = df[age_col].apply(_create_age_bin)
    
    result = pd.Series(np.nan, index=df.index)
    
    for age_bin in df['_Age_bin'].unique():
        if age_bin == 'Unknown':
            continue
        mask = df['_Age_bin'] == age_bin
        values = df.loc[mask, feature_name].fillna(df.loc[mask, feature_name].median())
        
        mean_val = values.mean()
        std_val = values.std()
        
        if std_val > 0:
            result.loc[mask] = (df.loc[mask, feature_name].fillna(mean_val) - mean_val) / std_val
        else:
            result.loc[mask] = 0
    
    return result

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

    # A8/A9 피처 추가
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

def add_features_A(df: pd.DataFrame) -> pd.DataFrame:
    """
    A 고급 파생 (연속형 + composite 위주, hand-made 0/1 플래그 없음)
    """
    feats = df.copy(); eps = 1e-6

    # -----------------------
    # 시간 축 / 기본 파생
    # -----------------------
    if _has(feats, ["Year","Month"]):
        feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]

    # speed-accuracy tradeoff
    for k in ["A1","A2","A4"]:
        m = f"{k}_rt_mean"
        acc = "resp_rate" if k in ["A1","A2"] else "acc_rate"
        acc_col = f"{k}_{acc}"
        if _has(feats, [m, acc_col]):
            feats[f"{k}_speed_acc_tradeoff"] = _safe_div(feats[m], feats[acc_col], eps)

    # CV of RT
    for k in ["A1","A2","A3","A4"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if _has(feats, [m, s]):
            feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    # 절대값 gap들
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

    # 비율 기반 gap
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

    # Stroop × error
    if _has(feats, ["A4_stroop_diff","A4_acc_rate"]):
        feats["A4_stroop_x_err"] = feats["A4_stroop_diff"] * (
            1 - feats["A4_acc_rate"].fillna(0)
        )

    # Age interaction: 하나만 유지해서 과한 중복 방지
    if _has(feats, ["Age_num","A1_rt_mean"]):
        feats["Age_x_A1_RT"] = feats["Age_num"] * feats["A1_rt_mean"]

    # -----------------------
    # Composite Scores (연속형 요약)
    # -----------------------
    # 1. PerceptualSpeed_A: 지각/속도 (RT 낮을수록 좋음 → 부호 반전)
    perceptual_speed_features = ['A1_rt_mean', 'A4_rt_mean']
    if all(f in feats.columns for f in perceptual_speed_features):
        normalized = _normalize_features(feats, perceptual_speed_features, method='zscore')
        feats['PerceptualSpeed_A'] = -normalized[perceptual_speed_features].mean(axis=1)

    # 2. CognitiveAbility_A: CogScore_A z-score (CogScore_A는 drop하여 중복 제거)
    if 'CogScore_A' in feats.columns:
        cog_mean = feats['CogScore_A'].mean()
        cog_std = feats['CogScore_A'].std()
        if cog_std > 0:
            feats['CognitiveAbility_A'] = (feats['CogScore_A'] - cog_mean) / cog_std
        else:
            feats['CognitiveAbility_A'] = 0.0
        # 선형 변환 관계이므로 원본 CogScore_A는 drop
        feats = feats.drop(columns=['CogScore_A'], errors='ignore')

    # 3. EmotionalRisk_A: A8/A9 기반 정서/스트레스 요약 (원본 피처는 drop하여 중복 제거)
    emotional_features = []
    if 'A8_Validity_Score' in feats.columns:
        emotional_features.append('A8_Validity_Score')
    if 'A9_Emotional_Score' in feats.columns:
        emotional_features.append('A9_Emotional_Score')

    if len(emotional_features) > 0:
        normalized = _normalize_features(feats, emotional_features, method='zscore')
        feats['EmotionalRisk_A'] = -normalized[emotional_features].mean(axis=1)
        # 선형 결합 관계이므로 원본 피처들은 drop
        feats = feats.drop(columns=emotional_features, errors='ignore')

    # 최종 정리
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats

def add_features_B(df: pd.DataFrame) -> pd.DataFrame:
    """
    B 고급 파생 (연속형 + Risk/Multitask composite 위주, hand-made 0/1 플래그 없음)
    """
    feats = df.copy(); eps = 1e-6

    # -----------------------
    # 기본 파생 + RiskScore_B
    # -----------------------
    if _has(feats, ["Year","Month"]):
        feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]

    # speed-accuracy tradeoff
    for k, acc_col, rt_col in [
        ("B1", "B1_acc_task1", "B1_rt_mean"),
        ("B2", "B2_acc_task1", "B2_rt_mean"),
        ("B3", "B3_acc_rate",  "B3_rt_mean"),
        ("B4", "B4_acc_rate",  "B4_rt_mean"),
        ("B5", "B5_acc_rate",  "B5_rt_mean"),
    ]:
        if _has(feats, [rt_col, acc_col]):
            feats[f"{k}_speed_acc_tradeoff"] = _safe_div(feats[rt_col], feats[acc_col], eps)

    # CV of RT
    for k in ["B1","B2","B3","B4","B5"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if _has(feats, [m, s]):
            feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    # RiskScore_B: 속도/정확도 기반 risk summary
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

    # 전역 RT/ACC 통계
    rt_cols = [c for c in feats.columns if c.endswith("_rt_mean")]
    if rt_cols:
        feats["B_rt_mean_global"] = feats[rt_cols].mean(axis=1)
        feats["B_rt_std_global"]  = feats[rt_cols].std(axis=1)

    acc_cols = [c for c in feats.columns if c.endswith("_acc_rate")]
    if acc_cols:
        feats["B_acc_mean_global"] = feats[acc_cols].mean(axis=1)
        feats["B_acc_std_global"]  = feats[acc_cols].std(axis=1)

    # 과제 간 gap
    if _has(feats, ["B4_acc_rate","B3_acc_rate"]):
        feats["B4_B3_acc_gap"] = feats["B4_acc_rate"] - feats["B3_acc_rate"]
    if _has(feats, ["B4_rt_mean","B3_rt_mean"]):
        feats["B4_B3_rt_gap"]  = feats["B4_rt_mean"] - feats["B3_rt_mean"]

    if _has(feats, ["B5_acc_rate","B1_acc_task1"]):
        feats["B5_B1_acc_gap"] = feats["B5_acc_rate"] - feats["B1_acc_task1"]
    if _has(feats, ["B5_rt_mean","B1_rt_mean"]):
        feats["B5_B1_rt_gap"]  = feats["B5_rt_mean"] - feats["B1_rt_mean"]

    # B10 vs B9 multitask ratio
    if _has(feats, ["B10_aud_overall_acc", "B9_aud_overall_acc"]):
        feats["B10_multitask_ratio_aud"] = _safe_div(
            feats["B10_aud_overall_acc"],
            feats["B9_aud_overall_acc"],
            eps
        )

    # -----------------------
    # Composite Scores (연속형 요약)
    # -----------------------
    # 1. MultitaskAbility_B: 지속주의 + multitask cost
    multitask_features = []
    if 'B9_aud_overall_acc' in feats.columns:
        multitask_features.append('B9_aud_overall_acc')
    if 'B10_multitask_cost_aud' in feats.columns:
        multitask_features.append('B10_multitask_cost_aud')

    if len(multitask_features) > 0:
        normalized = _normalize_features(feats, multitask_features, method='zscore')
        if 'B10_multitask_cost_aud' in multitask_features and 'B10_multitask_cost_aud' in normalized.columns:
            normalized['B10_multitask_cost_aud'] = -normalized['B10_multitask_cost_aud']
        feats['MultitaskAbility_B'] = normalized[multitask_features].mean(axis=1)

    # 2. RiskScore_B_norm: z-normalized risk (높을수록 나쁨 → 부호 반전)
    # RiskScore_B는 drop하여 중복 제거 (선형 변환 관계)
    if 'RiskScore_B' in feats.columns:
        risk_mean = feats['RiskScore_B'].mean()
        risk_std = feats['RiskScore_B'].std()
        if risk_std > 0:
            feats['RiskScore_B_norm'] = -(feats['RiskScore_B'] - risk_mean) / risk_std
        else:
            feats['RiskScore_B_norm'] = 0.0
        # 선형 변환 관계이므로 원본 RiskScore_B는 drop
        feats = feats.drop(columns=['RiskScore_B'], errors='ignore')

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats

# =============================================================================
# Age-normed Composite Scores
# =============================================================================

def add_age_normed_composites_A(df: pd.DataFrame) -> pd.DataFrame:
    """
    A용 age-normed composite scores 추가
    나이 bin별로 정규화된 composite score 생성 (age-split 구조 보완)
    """
    df = df.copy()

    # PerceptualSpeed_A_ageNorm
    if 'PerceptualSpeed_A' in df.columns and 'Age_num' in df.columns:
        df['PerceptualSpeed_A_ageNorm'] = _create_age_normed_score(
            df, 'PerceptualSpeed_A', age_col='Age_num'
        )

    # CognitiveAbility_A_ageNorm
    if 'CognitiveAbility_A' in df.columns and 'Age_num' in df.columns:
        df['CognitiveAbility_A_ageNorm'] = _create_age_normed_score(
            df, 'CognitiveAbility_A', age_col='Age_num'
        )

    # EmotionalRisk_A_ageNorm
    if 'EmotionalRisk_A' in df.columns and 'Age_num' in df.columns:
        df['EmotionalRisk_A_ageNorm'] = _create_age_normed_score(
            df, 'EmotionalRisk_A', age_col='Age_num'
        )

    return df

def add_age_normed_composites_B(df: pd.DataFrame) -> pd.DataFrame:
    """
    B용 age-normed composite scores 추가
    나이 bin별로 정규화된 composite score 생성 (history-split 구조 보완)
    """
    df = df.copy()

    if 'RiskScore_B_norm' in df.columns and 'Age_num' in df.columns:
        df['RiskScore_B_ageNorm'] = _create_age_normed_score(
            df, 'RiskScore_B_norm', age_col='Age_num'
        )

    if 'MultitaskAbility_B' in df.columns and 'Age_num' in df.columns:
        df['MultitaskAbility_B_ageNorm'] = _create_age_normed_score(
            df, 'MultitaskAbility_B', age_col='Age_num'
        )

    return df

# =============================================================================
# History Features
# =============================================================================

def _numpy_rolling_mean_shifted_safe(arr, window, group_starts, group_ends, min_periods=1):
    """
    NumPy 기반 rolling mean with shift(1) - 데이터 누수 방지 보장
    핵심: 각 그룹 내에서만 계산, 현재 행의 값은 절대 포함하지 않음
    """
    result = np.full(len(arr), np.nan, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    
    for start, end in zip(group_starts, group_ends):
        if end - start < min_periods + 1:
            continue
        
        group_arr = arr[start:end]
        group_len = end - start
        
        for i in range(1, group_len):
            window_size = min(window, i)
            
            if window_size < min_periods:
                continue
            
            window_start_idx = i - window_size
            window_end_idx = i
            
            window_data = group_arr[window_start_idx:window_end_idx]
            
            valid_mask = ~np.isnan(window_data)
            valid_count = np.sum(valid_mask, dtype=np.int32)
            if valid_count >= min_periods:
                valid_data = window_data[valid_mask]
                if len(valid_data) > 0:
                    result[start + i] = np.nanmean(valid_data)
    
    return result

def add_history_features_A(df: pd.DataFrame) -> pd.DataFrame:
    """
    A history features using PrimaryKey + YearMonthIndex
    NumPy 기반 rolling mean + trend, 과도한 2차 파생 없음
    """
    df = df.copy()

    if not _has(df, ["PrimaryKey", "YearMonthIndex"]):
        return df

    df = df.sort_values(["PrimaryKey", "YearMonthIndex"]).reset_index(drop=True)

    new_features = {}

    primary_keys = df["PrimaryKey"].values
    year_month = df["YearMonthIndex"].values.astype(np.float32)

    _, group_starts = np.unique(primary_keys, return_index=True)
    group_ends = np.concatenate([group_starts[1:], [len(df)]])

    # n_tests_so_far
    n_tests = np.zeros(len(df), dtype=np.int32)
    for start, end in zip(group_starts, group_ends):
        n_tests[start:end] = np.arange(1, end - start + 1, dtype=np.int32)
    new_features["n_tests_so_far"] = n_tests

    # months_since_first_test
    months_since_first = np.zeros(len(df), dtype=np.float32)
    for start, end in zip(group_starts, group_ends):
        months_since_first[start:end] = year_month[start:end] - year_month[start]
    new_features["months_since_first_test"] = months_since_first

    # months_since_prev_test
    months_since_prev = np.zeros(len(df), dtype=np.float32)
    for start, end in zip(group_starts, group_ends):
        if end - start > 1:
            months_since_prev[start+1:end] = np.diff(year_month[start:end])
    new_features["months_since_prev_test"] = months_since_prev

    MAX_HISTORY_WINDOW = 10
    MIN_PERIODS_MEAN = 1

    if "A4_acc_rate" in df.columns:
        prev = _numpy_rolling_mean_shifted_safe(
            df["A4_acc_rate"].values, MAX_HISTORY_WINDOW, group_starts, group_ends,
            min_periods=MIN_PERIODS_MEAN
        )
        new_features["A4_acc_prev_mean"] = prev

    # CognitiveAbility_A 사용 (CogScore_A는 add_features_A에서 drop되므로)
    if "CognitiveAbility_A" in df.columns:
        prev = _numpy_rolling_mean_shifted_safe(
            df["CognitiveAbility_A"].values, MAX_HISTORY_WINDOW, group_starts, group_ends,
            min_periods=MIN_PERIODS_MEAN
        )
        new_features["CognitiveAbility_A_prev_mean"] = prev
        new_features["CognitiveAbility_A_trend"] = df["CognitiveAbility_A"].values - prev

    if "A1_rt_mean" in df.columns:
        prev = _numpy_rolling_mean_shifted_safe(
            df["A1_rt_mean"].values, MAX_HISTORY_WINDOW, group_starts, group_ends,
            min_periods=MIN_PERIODS_MEAN
        )
        new_features["A1_rt_prev_mean"] = prev
        new_features["A1_rt_trend"] = df["A1_rt_mean"].values - prev

    if "A4_stroop_diff" in df.columns:
        prev = _numpy_rolling_mean_shifted_safe(
            df["A4_stroop_diff"].values, MAX_HISTORY_WINDOW, group_starts, group_ends,
            min_periods=MIN_PERIODS_MEAN
        )
        new_features["A4_stroop_prev_mean"] = prev
        new_features["A4_stroop_trend"] = df["A4_stroop_diff"].values - prev

    if new_features:
        new_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_df], axis=1)

    return df

def add_history_features_B(df: pd.DataFrame) -> pd.DataFrame:
    """
    B history features using PrimaryKey + YearMonthIndex
    - 핵심 history 정보만 유지: prev_mean + trend 중심
    - per_month, ratio, std, Age_x_* 같은 2차 파생은 제거 (노이즈/과적합 감소)
    """
    df = df.copy()

    if not _has(df, ["PrimaryKey", "YearMonthIndex"]):
        return df

    df = df.sort_values(["PrimaryKey", "YearMonthIndex"]).reset_index(drop=True)

    new_features = {}

    primary_keys = df["PrimaryKey"].values
    year_month = df["YearMonthIndex"].values.astype(np.float32)

    _, group_starts = np.unique(primary_keys, return_index=True)
    group_ends = np.concatenate([group_starts[1:], [len(df)]])

    # n_tests_so_far (history-split용 핵심)
    n_tests = np.zeros(len(df), dtype=np.int32)
    for start, end in zip(group_starts, group_ends):
        n_tests[start:end] = np.arange(1, end - start + 1, dtype=np.int32)
    new_features["n_tests_so_far"] = n_tests

    # months_since_first_test
    months_since_first = np.zeros(len(df), dtype=np.float32)
    for start, end in zip(group_starts, group_ends):
        months_since_first[start:end] = year_month[start:end] - year_month[start]
    new_features["months_since_first_test"] = months_since_first

    # months_since_prev_test
    months_since_prev = np.zeros(len(df), dtype=np.float32)
    for start, end in zip(group_starts, group_ends):
        if end - start > 1:
            months_since_prev[start+1:end] = np.diff(year_month[start:end])
    new_features["months_since_prev_test"] = months_since_prev

    MAX_HISTORY_WINDOW = 10
    MIN_PERIODS_MEAN = 1

    # --- Accuracy history (B4) ---
    if "B4_acc_rate" in df.columns:
        b4_acc = df["B4_acc_rate"].values.astype(np.float32)
        prev = _numpy_rolling_mean_shifted_safe(
            b4_acc, MAX_HISTORY_WINDOW, group_starts, group_ends,
            min_periods=MIN_PERIODS_MEAN
        )
        new_features["B4_acc_prev_mean"] = prev
        new_features["B4_acc_trend"] = b4_acc - prev

    # --- Multitask cost history (B10) ---
    if "B10_multitask_cost_aud" in df.columns:
        b10_cost = df["B10_multitask_cost_aud"].values.astype(np.float32)
        prev = _numpy_rolling_mean_shifted_safe(
            b10_cost, MAX_HISTORY_WINDOW, group_starts, group_ends,
            min_periods=MIN_PERIODS_MEAN
        )
        new_features["B10_multicost_prev_mean"] = prev
        new_features["B10_multicost_trend"] = b10_cost - prev

    # --- RT history (B3/B4/B5) ---
    for task in ["B3", "B4", "B5"]:
        rt_col = f"{task}_rt_mean"
        if rt_col in df.columns:
            rt_values = df[rt_col].values.astype(np.float32)
            prev = _numpy_rolling_mean_shifted_safe(
                rt_values, MAX_HISTORY_WINDOW, group_starts, group_ends,
                min_periods=MIN_PERIODS_MEAN
            )
            new_features[f"{task}_rt_prev_mean"] = prev
            new_features[f"{task}_rt_trend"] = rt_values - prev

    # ⚠️ prev_std, *_trend_per_month, *_trend_ratio,
    #     Age_x_B4_acc_trend, Age_x_B10_multicost_trend 등은 과감히 삭제

    if new_features:
        new_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_df], axis=1)

    return df

# =============================================================================
# 학습 & 추론
# =============================================================================

def build_model(seed: int, which: str, model_type: str = "hgb", for_oof: bool = False):
    """
    모델 생성 (HGBM only)
    - model_type: "hgb" (호환성을 위해 유지, 실제로는 항상 HGBM 사용)
    - for_oof: 호환성을 위해 유지 (실제로는 사용하지 않음, early_stopping이 validation_fraction 사용)
    """
    if which == "A":
        params = BASE_HGB_PARAMS_A.copy()
    else:
        params = BASE_HGB_PARAMS_B.copy()
    params["random_state"] = seed
    
    # HGBM: early_stopping=True, validation_fraction으로 내부 검증셋 사용
    # OOF와 서버 모두 동일한 설정 사용
    return HistGradientBoostingClassifier(**params)

def fit_partition(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    label_col: str,
    which: str,
):
    key = "Test_id"
    assert key in df_feat.columns, f"{which}: '{key}' not found in features"

    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")
    drop_cols = [key, label_col] + (["Test"] if "Test" in df.columns else [])

    feature_cols = [c for c in df.columns if c not in drop_cols]
    df = add_rowwise_features(df, feature_cols)

    num_cols, cat_cols = separate_num_cat(df, drop_cols)
    preproc = build_preprocessor(num_cols, cat_cols)

    X = df.drop(columns=drop_cols)
    y = df[label_col].astype(int).values

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=TEST_SIZE_FOR_LOG, random_state=RANDOM_STATE, stratify=y
    )

    X_tr_t = preproc.fit_transform(X_tr)
    X_val_t = preproc.transform(X_val)

    members = []
    for sd in ENSEMBLE_SEEDS:
        base = build_model(sd, which=which).fit(X_tr_t, y_tr)
        mdl  = maybe_calibrate(base, X_tr_t, y_tr)
        members.append(mdl)
    ensemble = AvgProbaEnsemble(members)

    try:
        val_proba = ensemble.predict_proba(X_val_t)[:, 1]
        metrics = evaluate_score(y_val, val_proba)
        print(
            f"[{which}] Holdout - "
            f"AUC={metrics.auc:.5f}, "
            f"Brier={metrics.brier:.5f}, "
            f"ECE={metrics.ece:.5f}, "
            f"Score={metrics.score:.5f}"
        )
    except Exception as e:
        print(f"[{which}] validation logging skipped: {e}")

    # 학습된 모델/전처기 저장
    if which == "A":
        preproc_path = A_PREPROC_PATH
        model_path   = A_MODEL_PATH
    else:
        preproc_path = B_PREPROC_PATH
        model_path   = B_MODEL_PATH

    with open(preproc_path, "wb") as f:
        pickle.dump(preproc, f, protocol=4)
    with open(model_path, "wb") as f:
        pickle.dump(ensemble, f, protocol=4)

    print(f"[{which}] saved model & preproc -> {model_path}, {preproc_path}")

    return preproc, ensemble

def fit_partition_A_oof_single(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    label_col: str = "Label",
):
    """
    A용 (OOF): 나이 서브모델 없이 단일 HGBM 5-seed 앙상블 + OOF.
    Age_num / *_age_normed 등은 그냥 일반 피처로만 사용.
    """
    global ACTUAL_ITERS_A
    which = "A"
    key = "Test_id"
    assert key in df_feat.columns, f"{which}: '{key}' not found in features"

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

    oof_preds = np.zeros(len(df), dtype=float)
    all_iters = []

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
        print(f"\n--- [A] Fold {fold}/{N_SPLITS} (single model) ---")
        X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
        y_tr, y_val = y_all[tr_idx], y_all[val_idx]

        # 각 fold마다 preproc은 다시 fit (leakage 없음)
        X_tr_t = preproc.fit_transform(X_tr)
        X_val_t = preproc.transform(X_val)

        members = []
        for sd in ENSEMBLE_SEEDS:
            mdl, base = train_and_calibrate_with_split(
                X_tr_t, y_tr,
                seed=sd, which=which, model_type="hgb"
            )
            if hasattr(base, "n_iter_"):
                all_iters.append(base.n_iter_)
            members.append(mdl)
        ensemble = AvgProbaEnsemble(members)

        val_proba = ensemble.predict_proba(X_val_t)[:, 1]
        oof_preds[val_idx] = val_proba

        metrics_fold = evaluate_score(y_val, val_proba)
        print(f"  Fold {fold} Score: {metrics_fold.score:.5f} (AUC={metrics_fold.auc:.5f})")

    # 전체 OOF 기준 성능
    metrics = evaluate_score(y_all, oof_preds)
    print(
        f"\n[A] Overall OOF (single HGBM, {N_SPLITS}-fold, {len(ENSEMBLE_SEEDS)}-seed) - "
        f"AUC={metrics.auc:.5f}, Brier={metrics.brier:.5f}, "
        f"ECE={metrics.ece:.5f}, Score={metrics.score:.5f}"
    )

    # 전체 데이터로 최종 모델 한 번 더 학습
    print(f"\n--- [A] Training final single model on all data (HGBM {len(ENSEMBLE_SEEDS)}-seed ensemble) ---")
    X_all_t = preproc.fit_transform(X_all)

    final_members = []
    all_iters_final = []
    for sd in ENSEMBLE_SEEDS:
        base = build_model(sd, which=which, model_type="hgb", for_oof=False)
        base.fit(X_all_t, y_all)
        if hasattr(base, "n_iter_"):
            all_iters_final.append(base.n_iter_)
        mdl = maybe_calibrate(base, X_all_t, y_all)
        final_members.append(mdl)
    final_ens = AvgProbaEnsemble(final_members)

    # iteration 기록
    ACTUAL_ITERS_A = {
        "single": all_iters_final if all_iters_final else None,
    }
    if all_iters_final:
        print(f"[A] OOF Iters(HGBM, single): mean={np.mean(all_iters_final):.1f}")

    # 저장 (단일모델 모드로 묶어서)
    model_bundle = dict(
        mode="single",
        ensemble=final_ens,
    )

    with open(A_PREPROC_PATH, "wb") as f:
        pickle.dump(preproc, f, protocol=4)
    with open(A_MODEL_PATH, "wb") as f:
        pickle.dump(model_bundle, f, protocol=4)

    print(f"[A] saved single-model bundle & preproc -> {A_MODEL_PATH}, {A_PREPROC_PATH}")

    return preproc, final_ens, metrics

def fit_partition_B_submodels_oof(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    label_col: str = "Label",
):
    """
    B용 (OOF): n_tests_so_far 기준으로 first(==1), repeat(>1) 분리 학습
    - 각 그룹마다 HGBM 5-seed ensemble 학습 → 확률 평균 앙상블
    - N_SPLITS Fold OOF 예측값과 최종 모델(앙상블) 반환
    """
    which = "B"
    key = "Test_id"
    assert key in df_feat.columns, f"{which}: '{key}' not found in features"
    assert "n_tests_so_far" in df_feat.columns, f"{which}: 'n_tests_so_far' not found in features"

    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")
    drop_cols = [key, label_col] + (["Test"] if "Test" in df.columns else [])
    feature_cols = [c for c in df.columns if c not in drop_cols]
    df = add_rowwise_features(df, feature_cols)

    num_cols, cat_cols = separate_num_cat(df, drop_cols)
    preproc = build_preprocessor(num_cols, cat_cols)

    X_all = df.drop(columns=drop_cols)
    y_all = df[label_col].astype(int).values
    n_tests_all = X_all["n_tests_so_far"].values

    oof_preds = np.zeros(len(df), dtype=float)

    final_ensemble_first  = []
    final_ensemble_repeat = []

    all_iters_first, all_iters_repeat = [], []

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
        print(f"\n--- [B] Fold {fold}/{N_SPLITS} ---")
        X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
        y_tr, y_val = y_all[tr_idx], y_all[val_idx]
        n_tests_tr, n_tests_val = n_tests_all[tr_idx], n_tests_all[val_idx]

        X_tr_t = preproc.fit_transform(X_tr)
        X_val_t = preproc.transform(X_val)

        mask_first_tr  = (n_tests_tr == 1)
        mask_repeat_tr = (n_tests_tr > 1)
        mask_first_val  = (n_tests_val == 1)
        mask_repeat_val = (n_tests_val > 1)

        members_first_fold, members_repeat_fold = [], []

        for sd in ENSEMBLE_SEEDS:
            # ---------- first (n_tests_so_far == 1) ----------
            if mask_first_tr.sum() > 0 and mask_first_val.sum() > 0:
                # HGBM only (train/calib 분리로 overfitting 방지)
                mdl_f_hgb, base_f_hgb = train_and_calibrate_with_split(
                    X_tr_t[mask_first_tr], y_tr[mask_first_tr],
                    seed=sd, which="B", model_type="hgb"
                )
                if hasattr(base_f_hgb, 'n_iter_'):
                    all_iters_first.append(base_f_hgb.n_iter_)
                members_first_fold.append(mdl_f_hgb)

            # ---------- repeat (n_tests_so_far > 1) ----------
            if mask_repeat_tr.sum() > 0 and mask_repeat_val.sum() > 0:
                # HGBM only (train/calib 분리로 overfitting 방지)
                mdl_r_hgb, base_r_hgb = train_and_calibrate_with_split(
                    X_tr_t[mask_repeat_tr], y_tr[mask_repeat_tr],
                    seed=sd, which="B", model_type="hgb"
                )
                if hasattr(base_r_hgb, 'n_iter_'):
                    all_iters_repeat.append(base_r_hgb.n_iter_)
                members_repeat_fold.append(mdl_r_hgb)

        # 만약 어떤 fold에서 first 또는 repeat 쪽에 모델이 하나도 안 생기면 fallback
        if not members_first_fold or not members_repeat_fold:
            print(f"  [B][Fold {fold}] WARNING: first/repeat 중 일부가 비어 있어 single HGBM ensemble로 fallback.")
            mdl_single, base_single = train_and_calibrate_with_split(
                X_tr_t, y_tr,
                seed=RANDOM_STATE, which="B", model_type="hgb"
            )
            if hasattr(base_single, 'n_iter_'):
                all_iters_first.append(base_single.n_iter_)
            val_proba = mdl_single.predict_proba(X_val_t)[:, 1]
            oof_preds[val_idx] = val_proba
            continue

        ens_f = AvgProbaEnsemble(members_first_fold)
        ens_r = AvgProbaEnsemble(members_repeat_fold)

        val_proba = np.zeros(len(y_val), dtype=float)
        if mask_first_val.sum() > 0:
            val_proba[mask_first_val] = ens_f.predict_proba(X_val_t[mask_first_val])[:, 1]
        if mask_repeat_val.sum() > 0:
            val_proba[mask_repeat_val] = ens_r.predict_proba(X_val_t[mask_repeat_val])[:, 1]

        oof_preds[val_idx] = val_proba
        metrics = evaluate_score(y_val, val_proba)
        print(f"  Fold {fold} Score: {metrics.score:.5f} (AUC={metrics.auc:.5f})")

    # -------- 전체 데이터로 final 모델 학습 (HGBM 5-seed ensemble) --------
    print("\n--- [B] Training final history-split models on all data (HGBM 5-seed ensemble) ---")
    X_all_t = preproc.fit_transform(X_all)
    n_tests_all = X_all["n_tests_so_far"].values

    mask_first_all  = (n_tests_all == 1)
    mask_repeat_all = (n_tests_all > 1)

    for sd in ENSEMBLE_SEEDS:
        if mask_first_all.sum() > 0:
            base_f_hgb = build_model(sd, which="B", model_type="hgb", for_oof=False)
            base_f_hgb.fit(X_all_t[mask_first_all], y_all[mask_first_all])
            mdl_f_hgb = maybe_calibrate(base_f_hgb, X_all_t[mask_first_all], y_all[mask_first_all])
            final_ensemble_first.append(mdl_f_hgb)

        if mask_repeat_all.sum() > 0:
            base_r_hgb = build_model(sd, which="B", model_type="hgb", for_oof=False)
            base_r_hgb.fit(X_all_t[mask_repeat_all], y_all[mask_repeat_all])
            mdl_r_hgb = maybe_calibrate(base_r_hgb, X_all_t[mask_repeat_all], y_all[mask_repeat_all])
            final_ensemble_repeat.append(mdl_r_hgb)

    final_ens_f = AvgProbaEnsemble(final_ensemble_first)
    final_ens_r = AvgProbaEnsemble(final_ensemble_repeat)

    metrics = evaluate_score(y_all, oof_preds)
    print(
        f"\n[B] Overall OOF (history-split, HGBM 5-seed ensemble) - "
        f"AUC={metrics.auc:.5f}, "
        f"Brier={metrics.brier:.5f}, "
        f"ECE={metrics.ece:.5f}, "
        f"Score={metrics.score:.5f}"
    )

    model_bundle = dict(
        mode="history_split",
        ensemble_first=final_ens_f,
        ensemble_repeat=final_ens_r,
    )

    with open(B_PREPROC_PATH, "wb") as f:
        pickle.dump(preproc, f, protocol=4)
    with open(B_MODEL_PATH, "wb") as f:
        pickle.dump(model_bundle, f, protocol=4)
    print(f"[B] saved 2-group model bundle & preproc -> {B_MODEL_PATH}, {B_PREPROC_PATH}")

    global ACTUAL_ITERS_B
    ACTUAL_ITERS_B = {
        'first':  all_iters_first  if all_iters_first  else None,
        'repeat': all_iters_repeat if all_iters_repeat else None,
    }
    if all_iters_first or all_iters_repeat:
        first_str  = f"{np.mean(all_iters_first):.1f}"  if all_iters_first  else "N/A"
        repeat_str = f"{np.mean(all_iters_repeat):.1f}" if all_iters_repeat else "N/A"
        print(f"[B] OOF Iters(HGBM): First={first_str}, Repeat={repeat_str}")

    return preproc, final_ens_f, final_ens_r, metrics

def fit_partition_B_submodels(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    label_col: str = "Label",
):
    """
    B용: n_tests_so_far 기준으로 first(==1), repeat(>1) 두 개의 3-seed 앙상블(HGB)을 학습.
    - 전처리(preproc)는 전체 데이터 기준으로 하나만 fit.
    - first: 히스토리 피처가 거의 NaN이므로 별도 모델
    - repeat: 모든 1, 2차 히스토리 피처 활용
    """
    global ACTUAL_ITERS_B  # 함수 시작 부분에 global 선언
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

    # holdout split (n_tests_so_far도 같이 쪼갬)
    X_tr, X_val, y_tr, y_val, n_tests_tr, n_tests_val = train_test_split(
        X_all, y_all, n_tests_all,
        test_size=TEST_SIZE_FOR_LOG,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    # 전처리 학습은 전체 train에 대해 1번만
    X_tr_t  = preproc.fit_transform(X_tr)
    X_val_t = preproc.transform(X_val)

    # first / repeat 마스크
    mask_first_tr = (n_tests_tr == 1)
    mask_repeat_tr = (n_tests_tr > 1)

    # 혹시라도 first 또는 repeat 쪽에 데이터가 거의 없으면 방어
    if mask_first_tr.sum() == 0:
        print(f"[{which}] WARNING: no samples for n_tests_so_far == 1, fallback to single model.")
        # 그냥 기존 fit_partition 로직 비슷하게 한 개 모델만 학습
        members = []
        for sd in ENSEMBLE_SEEDS:
            base = build_model(sd, which=which, model_type="hgb", for_oof=False)
            base.fit(X_tr_t, y_tr)
            mdl  = maybe_calibrate(base, X_tr_t, y_tr)
            members.append(mdl)
        ensemble = AvgProbaEnsemble(members)

        # 검증
        try:
            val_proba = ensemble.predict_proba(X_val_t)[:, 1]
            metrics = evaluate_score(y_val, val_proba)
            print(
                f"[{which}] Holdout(single) - "
                f"AUC={metrics.auc:.5f}, "
                f"Brier={metrics.brier:.5f}, "
                f"ECE={metrics.ece:.5f}, "
                f"Score={metrics.score:.5f}"
            )
        except Exception as e:
            print(f"[{which}] validation logging skipped: {e}")

        # 저장은 한 모델만 묶어서
        model_bundle = dict(
            mode="single",
            ensemble=ensemble,
        )
    elif mask_repeat_tr.sum() == 0:
        print(f"[{which}] WARNING: no samples for n_tests_so_far > 1, fallback to single model.")
        # 그냥 기존 fit_partition 로직 비슷하게 한 개 모델만 학습
        members = []
        for sd in ENSEMBLE_SEEDS:
            base = build_model(sd, which=which, model_type="hgb", for_oof=False)
            base.fit(X_tr_t, y_tr)
            mdl  = maybe_calibrate(base, X_tr_t, y_tr)
            members.append(mdl)
        ensemble = AvgProbaEnsemble(members)

        # 검증
        try:
            val_proba = ensemble.predict_proba(X_val_t)[:, 1]
            metrics = evaluate_score(y_val, val_proba)
            print(
                f"[{which}] Holdout(single) - "
                f"AUC={metrics.auc:.5f}, "
                f"Brier={metrics.brier:.5f}, "
                f"ECE={metrics.ece:.5f}, "
                f"Score={metrics.score:.5f}"
            )
        except Exception as e:
            print(f"[{which}] validation logging skipped: {e}")

        # 저장은 한 모델만 묶어서
        model_bundle = dict(
            mode="single",
            ensemble=ensemble,
        )
    else:
        # ---------- first 앙상블 (n_tests_so_far == 1) ----------
        X_tr_first = X_tr_t[mask_first_tr]
        y_tr_first = y_tr[mask_first_tr]

        members_first = []
        actual_iters_first = []
        for sd in ENSEMBLE_SEEDS:
            base = build_model(sd, which=which, model_type="hgb", for_oof=False)
            base.fit(X_tr_first, y_tr_first)
            # 실제 사용된 iteration 수 확인 (early stopping으로 멈췄을 수 있음)
            if hasattr(base, 'n_iter_'):
                actual_iters_first.append(base.n_iter_)
            mdl  = maybe_calibrate(base, X_tr_first, y_tr_first)
            members_first.append(mdl)
        ensemble_first = AvgProbaEnsemble(members_first)
        if actual_iters_first:
            print(f"  [{which}] first 그룹 실제 사용된 iteration: {actual_iters_first} (평균: {np.mean(actual_iters_first):.1f})")

        # ---------- repeat 앙상블 (n_tests_so_far > 1) ----------
        X_tr_repeat = X_tr_t[mask_repeat_tr]
        y_tr_repeat = y_tr[mask_repeat_tr]

        members_repeat = []
        actual_iters_repeat = []
        for sd in ENSEMBLE_SEEDS:
            base = build_model(sd, which=which, model_type="hgb", for_oof=False)
            base.fit(X_tr_repeat, y_tr_repeat)
            # 실제 사용된 iteration 수 확인 (early stopping으로 멈췄을 수 있음)
            if hasattr(base, 'n_iter_'):
                actual_iters_repeat.append(base.n_iter_)
            mdl  = maybe_calibrate(base, X_tr_repeat, y_tr_repeat)
            members_repeat.append(mdl)
        ensemble_repeat = AvgProbaEnsemble(members_repeat)
        if actual_iters_repeat:
            print(f"  [{which}] repeat 그룹 실제 사용된 iteration: {actual_iters_repeat} (평균: {np.mean(actual_iters_repeat):.1f})")
        
        # 실제 사용된 iteration 수를 best_params.json에 저장하기 위해 전역 변수에 저장
        ACTUAL_ITERS_B = {
            'first': actual_iters_first if actual_iters_first else None,
            'repeat': actual_iters_repeat if actual_iters_repeat else None,
        }

        # ---------- holdout에서 적절한 모델로 평가 ----------
        try:
            mask_first_val = (n_tests_val == 1)
            mask_repeat_val = (n_tests_val > 1)

            # first와 repeat 각각 예측
            val_proba = np.zeros(len(y_val), dtype=float)
            if mask_first_val.sum() > 0:
                val_proba[mask_first_val] = ensemble_first.predict_proba(X_val_t[mask_first_val])[:, 1]
            if mask_repeat_val.sum() > 0:
                val_proba[mask_repeat_val] = ensemble_repeat.predict_proba(X_val_t[mask_repeat_val])[:, 1]

            metrics = evaluate_score(y_val, val_proba)
            print(
                f"[{which}] Holdout(history-split) - "
                f"AUC={metrics.auc:.5f}, "
                f"Brier={metrics.brier:.5f}, "
                f"ECE={metrics.ece:.5f}, "
                f"Score={metrics.score:.5f}"
            )
        except Exception as e:
            print(f"[{which}] validation logging skipped: {e}")

        model_bundle = dict(
            mode="history_split",
            ensemble_first=ensemble_first,
            ensemble_repeat=ensemble_repeat,
        )

    # ===== 저장 =====
    preproc_path = B_PREPROC_PATH
    model_path   = B_MODEL_PATH

    with open(preproc_path, "wb") as f:
        pickle.dump(preproc, f, protocol=4)
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f, protocol=4)

    print(f"[{which}] saved model bundle & preproc -> {model_path}, {preproc_path}")

    # 반환은 preproc + 서브모델들 (단일모델 fallback 포함)
    if model_bundle.get("mode") == "history_split":
        return preproc, model_bundle["ensemble_first"], model_bundle["ensemble_repeat"]
    else:
        # single model fallback일 때는 repeat 모델 자리에 None
        return preproc, model_bundle["ensemble"], None

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

def save_meta():
    meta = dict(
        model="HGBM (5-seed ensemble) + Single model(A) + History-based submodels(B) + OOF validation",
        hgb_params_A=BASE_HGB_PARAMS_A,
        hgb_params_B=BASE_HGB_PARAMS_B,
        ensemble_seeds=list(ENSEMBLE_SEEDS),
        use_calibration=USE_CALIBRATION,
        calib_method=CALIB_METHOD,
        calib_cv=CALIB_CV,
        sklearn_version=sklver,
        random_state=RANDOM_STATE,
        save_format="pickle_protocol_4",
        note_A="Single HGBM model (no age-based submodels, Age_num used as feature only)",
    )
    
    # 실제 사용된 iteration 수 저장 (메타 정보용, 분석 목적)
    if ACTUAL_ITERS_A is not None:
        meta["actual_iters_A"] = ACTUAL_ITERS_A
        # 통계 정보 계산 (참고용)
        all_iters_A = []
        if ACTUAL_ITERS_A.get('young'):
            all_iters_A.extend(ACTUAL_ITERS_A['young'])
        if ACTUAL_ITERS_A.get('mid'):
            all_iters_A.extend(ACTUAL_ITERS_A['mid'])
        if ACTUAL_ITERS_A.get('old'):
            all_iters_A.extend(ACTUAL_ITERS_A['old'])
        if ACTUAL_ITERS_A.get('under'):
            all_iters_A.extend(ACTUAL_ITERS_A['under'])
        if ACTUAL_ITERS_A.get('over'):
            all_iters_A.extend(ACTUAL_ITERS_A['over'])
        if ACTUAL_ITERS_A.get('single'):
            all_iters_A.extend(ACTUAL_ITERS_A['single'])
        if all_iters_A:
            print(f"  [A] 실제 사용된 iteration 평균: {np.mean(all_iters_A):.1f} (메타 정보)")
    
    if ACTUAL_ITERS_B is not None:
        meta["actual_iters_B"] = ACTUAL_ITERS_B
        # 통계 정보 계산 (참고용)
        all_iters_B = []
        if ACTUAL_ITERS_B.get('first'):
            all_iters_B.extend(ACTUAL_ITERS_B['first'])
        if ACTUAL_ITERS_B.get('repeat'):
            all_iters_B.extend(ACTUAL_ITERS_B['repeat'])
        if ACTUAL_ITERS_B.get('single'):
            all_iters_B.extend(ACTUAL_ITERS_B['single'])
        if all_iters_B:
            print(f"  [B] 실제 사용된 iteration 평균: {np.mean(all_iters_B):.1f} (메타 정보)")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def save_best_params(metrics_A: Metrics, metrics_B: Metrics, n_A: int, n_B: int):
    """
    OOF 검증 결과와 하이퍼파라미터를 best_params.json에 저장
    """
    total = n_A + n_B
    weight_A = n_A / (total + 1e-9)
    weight_B = n_B / (total + 1e-9)
    
    final_score = weight_A * metrics_A.score + weight_B * metrics_B.score
    final_auc = weight_A * metrics_A.auc + weight_B * metrics_B.auc
    
    # 파라미터는 그대로 사용 (early_stopping=True, max_iter=1500으로 넉넉히 설정)
    params_A_final = BASE_HGB_PARAMS_A.copy()
    params_B_final = BASE_HGB_PARAMS_B.copy()
    
    best_params_to_save = dict(
        hgb_params_A=params_A_final,
        hgb_params_B=params_B_final,
        ensemble_seeds=list(ENSEMBLE_SEEDS),
        use_calibration=USE_CALIBRATION,
        calib_method=CALIB_METHOD,
        calib_cv=CALIB_CV,
        random_state=RANDOM_STATE,
        sklearn_version=sklver,
        # OOF 스코어 기록
        oof_score_A=metrics_A.score,
        oof_score_B=metrics_B.score,
        oof_score_total=final_score,
        oof_auc_total=final_auc,
        note_A="Single HGBM model (no age-based submodels, Age_num used as feature only)",
    )
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(best_params_to_save, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Best parameters (OOF Score: {final_score:.6f}) saved to {BEST_PARAMS_PATH}")

# =============================================================================
# 메인
# =============================================================================

def main():
    set_seed(RANDOM_STATE)
    t0 = time.time()
    ensure_dirs()

    print("📂 Loading index files ...")
    train_idx, _ = read_index_files()  # OOF 검증이므로 test_idx는 필요 없음

    print("📂 Loading raw A/B (train only) ...")
    A_train_raw, B_train_raw = read_raw_feature_files("train")

    print("\n🔵 FE for A (with history features) ...")
    t_fe_a = time.time()
    A_train_feat = preprocess_A(A_train_raw)
    A_train_feat = add_features_A(A_train_feat)
    A_train_feat = add_age_normed_composites_A(A_train_feat)
    A_train_feat = add_history_features_A(A_train_feat)
    print(f"  A_train_feat: {A_train_feat.shape} (elapsed: {(time.time()-t_fe_a)/60:.2f} min)")

    print("\n🟢 FE for B (with history features) ...")
    t_fe_b = time.time()
    B_train_feat = preprocess_B(B_train_raw)
    B_train_feat = add_features_B(B_train_feat)
    B_train_feat = add_age_normed_composites_B(B_train_feat)
    B_train_feat = add_history_features_B(B_train_feat)
    print(f"  B_train_feat: {B_train_feat.shape} (elapsed: {(time.time()-t_fe_b)/60:.2f} min)")

    # ===== A 학습 (Single HGBM model - no age-based submodels) =====
    A_train_idx = train_idx[train_idx["Test"] == "A"].copy()
    if USE_OOF:
        print("\n[A] training single HGBM model (OOF) ...")
        preproc_A, clf_A, metrics_A = fit_partition_A_oof_single(A_train_feat, A_train_idx, "Label")
    else:
        print("\n[A] training single HGBM model (Holdout) ...")
        preproc_A, clf_A = fit_partition(A_train_feat, A_train_idx, "Label", which="A")
        metrics_A = Metrics(auc=0.0, brier=0.0, ece=0.0, score=0.0)  # 더미

    # ===== B 학습 (History-based submodels) =====
    B_train_idx = train_idx[train_idx["Test"] == "B"].copy()
    if USE_OOF:
        print("\n[B] training with history-based submodels (OOF) ...")
        (
            preproc_B,
            clf_B_first,
            clf_B_repeat,
            metrics_B
        ) = fit_partition_B_submodels_oof(B_train_feat, B_train_idx, "Label")
    else:
        print("\n[B] training with history-based submodels (Holdout) ...")
        preproc_B, clf_B_first, clf_B_repeat = fit_partition_B_submodels(B_train_feat, B_train_idx, "Label")
        # Holdout 평가는 fit_partition_B_submodels 내부에서 수행됨
        metrics_B = Metrics(auc=0.0, brier=0.0, ece=0.0, score=0.0)

    # ===== 최종 점수 계산 및 best_params.json 저장 =====
    n_A = len(A_train_idx)
    n_B = len(B_train_idx)
    total = n_A + n_B
    weight_A = n_A / total
    weight_B = n_B / total

    if USE_OOF:
        final_score = weight_A * metrics_A.score + weight_B * metrics_B.score
        final_auc = weight_A * metrics_A.auc + weight_B * metrics_B.auc
        final_brier = weight_A * metrics_A.brier + weight_B * metrics_B.brier
        final_ece = weight_A * metrics_A.ece + weight_B * metrics_B.ece

        print("\n" + "="*80)
        print(f"🚀 Overall OOF Score ({N_SPLITS}-Fold CV, {len(ENSEMBLE_SEEDS)}-seed ensemble):")
        print(f"  - A Score: {metrics_A.score:.6f} (AUC={metrics_A.auc:.6f})")
        print(f"  - B Score: {metrics_B.score:.6f} (AUC={metrics_B.auc:.6f})")
        print(f"  - Weights: A={weight_A:.2f}, B={weight_B:.2f}")
        print(f"  - Final Score : {final_score:.6f}")
        print(f"  - Final AUC   : {final_auc:.6f}")
        print(f"  - Final Brier : {final_brier:.6f}")
        print(f"  - Final ECE   : {final_ece:.6f}")
        print("="*80)
    else:
        print("\n" + "="*80)
        print(f"⚡ Holdout Evaluation ({len(ENSEMBLE_SEEDS)}-seed ensemble):")
        print(f"  ⚠️  빠른 평가 모드: OOF 없이 단순 holdout 사용")
        print(f"  ⚠️  성능 추정이 덜 정확할 수 있음 (LB 점수와 차이 가능)")
        print(f"  💡 정확한 평가를 원하면 USE_OOF=True로 설정")
        print("="*80)
        # Holdout 모드에서는 더미 값 사용
        final_score = 0.0
        final_auc = 0.0
        final_brier = 0.0
        final_ece = 0.0

    # iteration 추적 확인
    print(f"\n[Iteration Tracking]")
    print(f"  ACTUAL_ITERS_A: {ACTUAL_ITERS_A}")
    print(f"  ACTUAL_ITERS_B: {ACTUAL_ITERS_B}")

    # best_params.json 저장
    save_best_params(metrics_A, metrics_B, n_A, n_B)
    save_meta()

    dt = time.time() - t0
    print(f"\n✅ OOF validation completed | elapsed: {dt/60:.2f} min")
    print(f"✅ models saved to: {MODEL_DIR}")
    print(f"✅ Now run 'submit/script.py' to generate submission using these models.")

if __name__ == "__main__":
    main()
