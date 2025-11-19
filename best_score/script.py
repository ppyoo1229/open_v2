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

HAS_LIGHTGBM = False

import pickle
import json  

# =============================================================================
# 경로 설정
# =============================================================================
DATA_DIR = "data"    
OUTPUT_DIR = "output"  
MODEL_DIR  = "model"  

SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
BEST_PARAMS_PATH = os.path.join(MODEL_DIR, "best_params.json")

# =============================================================================
# 하이퍼파라미터 로드
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
    RANDOM_STATE = best_params.get("random_state", 42)
    
    print(f" Using best hyperparameters from best08.py OOF evaluation (HGBM only, early_stopping=True)")
else:
    # Fallback: 기본 하이퍼파라미터
    RANDOM_STATE = 42
    USE_CALIBRATION = True
    CALIB_METHOD = "isotonic"
    CALIB_CV = 3
    ENSEMBLE_SEEDS: Sequence[int] = (42, 202, 777, 1001, 8888)  # 5개 seed
    BASE_HGB_PARAMS_A = dict(
        learning_rate=0.05,
        max_iter=1500,  # 넉넉히 설정, early_stopping이 실제 iter 결정
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
        max_iter=1500,  
        max_depth=None,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True, 
        validation_fraction=0.15,
        n_iter_no_change=40,
        class_weight="balanced",
    )
    print(f"  Using default hyperparameters (best_params.json not found, HGBM only)")

TEST_SIZE_FOR_LOG = 0.0  

VALIDATE_NUMPY_IMPLEMENTATION = False  

# =============================================================================
# 서버 환경 감지 및 자동 최적화 (30분 제한 준수)
# =============================================================================
def detect_server_environment():
    import multiprocessing
    
    cpu_count = multiprocessing.cpu_count()
    
    # RAM 정보는 psutil이 있으면 사용, 없으면 CPU만으로 판단
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        # 평가 서버 사양: CPU 3 vCPU, RAM 28GB
        is_server = (cpu_count <= 4) and (ram_gb >= 20) and (ram_gb <= 32)
        if is_server:
            print(f"Server environment detected: CPU={cpu_count}, RAM={ram_gb:.1f}GB")
    except ImportError:
        # psutil이 없으면 CPU만으로 판단 (3 vCPU 이하)
        is_server = (cpu_count <= 4)
        if is_server:
            print(f"Server environment detected: CPU={cpu_count} (RAM info unavailable)")
    
    if is_server:
        print("  Applying server optimizations for 30-minute time limit...")
    
    return is_server

IS_SERVER_ENV = detect_server_environment()

# 서버 환경 최적화: Calibration CV를 3→2로 줄이기 (시간 약 30% 절약)
if IS_SERVER_ENV and CALIB_CV > 2:
    print(f"  Optimizing: CALIB_CV {CALIB_CV} -> 2 (time saving)")
    CALIB_CV = 2

# 서버 환경 최적화: B 모델 max_iter를 700→500으로 제한 (early_stopping이 있으므로 영향 제한적)
if IS_SERVER_ENV and best_params is None:
    # best_params가 없을 때만 기본값 조정
    if BASE_HGB_PARAMS_B.get("max_iter", 700) > 500:
        print(f"  Optimizing: B model max_iter {BASE_HGB_PARAMS_B.get('max_iter')} -> 500 (time saving)")
        BASE_HGB_PARAMS_B["max_iter"] = 500
elif IS_SERVER_ENV and best_params is not None:
    # best_params가 있을 때는 경고만 출력 (사용자가 설정한 값이므로)
    if BASE_HGB_PARAMS_B.get("max_iter", 700) > 500:
        print(f"  Warning: B model max_iter={BASE_HGB_PARAMS_B.get('max_iter')} may be slow on server")
        print(f"  Consider reducing to 500 if time limit is exceeded")

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
    """
    Index 파일 읽기 (메모리 효율적)
    """
    train_idx = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), low_memory=True)
    test_idx  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), low_memory=True)
    return train_idx, test_idx

def read_raw_feature_files(split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    메모리 효율적인 CSV 읽기 (OOM 방지)
    - C 엔진 먼저 시도 (빠름), OOM 발생 시 Python 엔진으로 fallback (메모리 효율적)
    - low_memory=True: 타입 추론을 청크 단위로 수행 (메모리 절약)
    """
    A_path = os.path.join(DATA_DIR, split, "A.csv")
    B_path = os.path.join(DATA_DIR, split, "B.csv")
    
    # 메모리 효율적인 읽기: C 엔진 먼저 시도 (빠름), OOM 발생 시 Python 엔진으로 fallback
    def read_csv_safe(path):
        try:
            # C 엔진 + low_memory=True (메모리 절약하면서 빠름)
            return pd.read_csv(path, engine="c", low_memory=True, sep=",")
        except (MemoryError, pd.errors.ParserError) as e:
            if "out of memory" in str(e).lower() or "memory" in str(e).lower():
                print(f"Warning: OOM with C engine for {path}, trying Python engine (slower but more memory-efficient)...")
                try:
                    # Python 엔진 (느리지만 메모리 효율적)
                    return pd.read_csv(path, engine="python", low_memory=True, sep=",")
                except Exception as e2:
                    print(f"Error: Python engine also failed: {e2}")
                    raise
            else:
                raise
    
    A_df = read_csv_safe(A_path)
    B_df = read_csv_safe(B_path)
    
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
        if col.endswith(("_rate", "_resp_rate", "_acc_rate")):
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
    
    로직:
    - i번째 행 (0-indexed)의 rolling mean은 0부터 i-1번째 행만 사용
    - 첫 번째 행 (i=0)은 항상 NaN (shift(1))
    - 그룹 경계를 넘지 않음 (group_starts, group_ends로 보장)
    - min_periods: 유효한 값이 최소 min_periods개 있어야 계산
    
    Args:
        arr: 입력 배열
        window: rolling window 크기
        group_starts: 각 그룹의 시작 인덱스 배열
        group_ends: 각 그룹의 끝 인덱스 배열
        min_periods: 최소 유효 값 개수 (기본값: 1)
    
    Returns:
        shift(1)이 적용된 rolling mean 결과 (데이터 누수 없음)
    """
    result = np.full(len(arr), np.nan, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    
    for start, end in zip(group_starts, group_ends):
        if end - start < min_periods + 1:
            continue  # min_periods + shift(1)을 위한 최소 크기
        
        group_arr = arr[start:end]
        group_len = end - start
        
        # 현재 행을 제외한 이전 행들만 사용 (shift(1) 보장)
        for i in range(1, group_len):  # i=0은 skip (shift(1) 후 NaN)
            # window 크기: min(window, i) - 현재까지의 행 개수 (현재 행 제외)
            window_size = min(window, i)
            
            if window_size < min_periods:
                continue  # min_periods 미만이면 계산하지 않음
            
            # 윈도우: [i-window_size, i) - 현재 행(i) 제외
            window_start_idx = i - window_size
            window_end_idx = i  # 현재 행 제외
            
            # 윈도우 데이터 추출 (현재 행 제외 보장)
            window_data = group_arr[window_start_idx:window_end_idx]
            
            # NaN 제외하고 평균 계산 (min_periods 체크)
            valid_mask = ~np.isnan(window_data)
            valid_count = np.sum(valid_mask, dtype=np.int32)
            if valid_count >= min_periods:
                valid_data = window_data[valid_mask]
                if len(valid_data) > 0:
                    result[start + i] = np.nanmean(valid_data)
            # valid_count < min_periods면 결과는 이미 NaN
    
    return result

def _numpy_rolling_std_shifted_safe(arr, window, group_starts, group_ends, min_periods=2):
    """
    NumPy 기반 rolling std with shift(1) - 데이터 누수 방지 보장
    핵심: 각 그룹 내에서만 계산, 현재 행의 값은 절대 포함하지 않음
    
    로직:
    - i번째 행 (0-indexed)의 rolling std는 0부터 i-1번째 행만 사용
    - 첫 번째 행 (i=0)은 항상 NaN (shift(1))
    - 그룹 경계를 넘지 않음 (group_starts, group_ends로 보장)
    - min_periods: 유효한 값이 최소 min_periods개 있어야 계산
    
    Args:
        arr: 입력 배열
        window: rolling window 크기
        group_starts: 각 그룹의 시작 인덱스 배열
        group_ends: 각 그룹의 끝 인덱스 배열
        min_periods: 최소 유효 값 개수 (기본값: 2, std는 최소 2개 필요)
    
    Returns:
        shift(1)이 적용된 rolling std 결과 (데이터 누수 없음)
    """
    result = np.full(len(arr), np.nan, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    
    for start, end in zip(group_starts, group_ends):
        if end - start < min_periods + 1:
            continue  # min_periods + shift(1)을 위한 최소 크기
        
        group_arr = arr[start:end]
        group_len = end - start
        
        for i in range(1, group_len):
            window_size = min(window, i)
            if window_size < min_periods:
                continue
            
            window_start_idx = i - window_size
            window_end_idx = i  # 현재 행 제외
            
            window_data = group_arr[window_start_idx:window_end_idx]
            
            # NaN 제외하고 std 계산
            valid_mask = ~np.isnan(window_data)
            valid_count = np.sum(valid_mask, dtype=np.int32)
            if valid_count >= min_periods:
                valid_data = window_data[valid_mask]
                if len(valid_data) > 0:
                    result[start + i] = np.std(valid_data, ddof=0, dtype=np.float32)
    
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
# 학습 함수 (서버 제출용 - OOF 없이 최종 모델만 학습, NumPy 에러 회피)
# =============================================================================
# 주의: 서버 환경에서는 모델 저장/로딩 대신 즉시 재학습하는 방식 사용
# NumPy BitGenerator 호환성 문제 회피 (numpy.random._pcg64.PCG64 에러)
# HistGradientBoostingClassifier는 내부적으로 병렬 처리를 사용하므로 n_jobs 제거 (안전성)

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
    """
    단일 HGBM 모델 학습 (서버 제출용 - OOF 없이 전체 데이터로 학습, 모델 저장 없음)
    - A검사용: 단일 모델 (나이 기반 서브모델 없음)
    - B검사용: 단일 모델 (히스토리 기반 서브모델은 fit_partition_B_submodels 사용)
    """
    key = "Test_id"
    assert key in df_feat.columns, f"{which}: '{key}' not found in features"

    # index와 feature merge
    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")
    drop_cols = [key, label_col] + (["Test"] if "Test" in df.columns else [])

    feature_cols = [c for c in df.columns if c not in drop_cols]
    df = add_rowwise_features(df, feature_cols)

    num_cols, cat_cols = separate_num_cat(df, drop_cols)
    preproc = build_preprocessor(num_cols, cat_cols)

    # 서버 제출용: 전체 데이터로 학습 (validation 없음, 시간 절약)
    X_all = df.drop(columns=drop_cols)
    y_all = df[label_col].astype(int).values

    X_all_t = preproc.fit_transform(X_all)

    # 5-seed 앙상블 학습
    members = []
    for sd in ENSEMBLE_SEEDS:
        base = build_model(sd, which=which).fit(X_all_t, y_all)
        mdl = maybe_calibrate(base, X_all_t, y_all)
        members.append(mdl)
    ensemble = AvgProbaEnsemble(members)

    # 서버 제출용: 모델 저장 없음, Train 평가 스킵 (시간 절약, 30분 제한 준수)

    return preproc, ensemble

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

    # 서버 제출용: 전체 데이터로 학습 (validation 없음)
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
            # HGBM only
            base_hgb = build_model(sd, which=which, model_type="hgb", for_oof=False)
            base_hgb.fit(X_tr_t, y_tr)
            mdl_hgb = maybe_calibrate(base_hgb, X_tr_t, y_tr)
            members.append(mdl_hgb)
        ensemble = AvgProbaEnsemble(members)
        return preproc, ensemble, None

    if mask_repeat_tr.sum() == 0:
        print(f"[{which}] WARNING: no samples for n_tests_so_far > 1, fallback to single model.")
        members = []
        for sd in ENSEMBLE_SEEDS:
            # HGBM only
            base_hgb = build_model(sd, which=which, model_type="hgb", for_oof=False)
            base_hgb.fit(X_tr_t, y_tr)
            mdl_hgb = maybe_calibrate(base_hgb, X_tr_t, y_tr)
            members.append(mdl_hgb)
        ensemble = AvgProbaEnsemble(members)
        return preproc, ensemble, None

    # ---------- first 앙상블 (n_tests_so_far == 1) ----------
    X_tr_first = X_tr_t[mask_first_tr]
    y_tr_first = y_tr[mask_first_tr]

    members_first = []
    for sd in ENSEMBLE_SEEDS:
        # HGBM only
        base_f_hgb = build_model(sd, which=which, model_type="hgb", for_oof=False)
        base_f_hgb.fit(X_tr_first, y_tr_first)
        mdl_f_hgb = maybe_calibrate(base_f_hgb, X_tr_first, y_tr_first)
        members_first.append(mdl_f_hgb)
    ensemble_first = AvgProbaEnsemble(members_first)

    # ---------- repeat 앙상블 (n_tests_so_far > 1) ----------
    X_tr_repeat = X_tr_t[mask_repeat_tr]
    y_tr_repeat = y_tr[mask_repeat_tr]

    members_repeat = []
    for sd in ENSEMBLE_SEEDS:
        # HGBM only
        base_r_hgb = build_model(sd, which=which, model_type="hgb", for_oof=False)
        base_r_hgb.fit(X_tr_repeat, y_tr_repeat)
        mdl_r_hgb = maybe_calibrate(base_r_hgb, X_tr_repeat, y_tr_repeat)
        members_repeat.append(mdl_r_hgb)
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

    print("\n🔵 FE for A (with history features) ...")
    t_fe_a = time.time()
    # Train과 Test를 합쳐서 history features 생성 (PrimaryKey 기준 정렬을 위해)
    A_all_raw = pd.concat([A_train_raw, A_test_raw], axis=0, ignore_index=True)
    A_all = preprocess_A(A_all_raw)
    A_all = add_features_A(A_all)
    A_all = add_age_normed_composites_A(A_all)
    A_all = add_history_features_A(A_all)

    A_train_ids = set(A_train_raw["Test_id"])
    A_test_ids = set(A_test_raw["Test_id"])
    A_train_feat = A_all[A_all["Test_id"].isin(A_train_ids)].reset_index(drop=True)
    A_test_feat = A_all[A_all["Test_id"].isin(A_test_ids)].reset_index(drop=True)
    print(f"  A_train_feat: {A_train_feat.shape}, A_test_feat: {A_test_feat.shape} (elapsed: {(time.time()-t_fe_a)/60:.2f} min)")
    
    print("\n🟢 FE for B (with history features) ...")
    t_fe_b = time.time()
    # Train과 Test를 합쳐서 history features 생성 (PrimaryKey 기준 정렬을 위해)
    B_all_raw = pd.concat([B_train_raw, B_test_raw], axis=0, ignore_index=True)
    B_all = preprocess_B(B_all_raw)
    B_all = add_features_B(B_all)
    B_all = add_age_normed_composites_B(B_all)
    B_all = add_history_features_B(B_all)

    B_train_ids = set(B_train_raw["Test_id"])
    B_test_ids = set(B_test_raw["Test_id"])
    B_train_feat = B_all[B_all["Test_id"].isin(B_train_ids)].reset_index(drop=True)
    B_test_feat = B_all[B_all["Test_id"].isin(B_test_ids)].reset_index(drop=True)
    print(f"  B_train_feat: {B_train_feat.shape}, B_test_feat: {B_test_feat.shape} (elapsed: {(time.time()-t_fe_b)/60:.2f} min)")

    # ===== A 학습 (Single HGBM model - no age-based submodels) =====
    A_train_idx = train_idx[train_idx["Test"] == "A"].copy()
    A_test_idx = test_idx[test_idx["Test"] == "A"].copy()
    print("\n[A] training single HGBM model (no OOF, server optimized) ...")
    print(f"  Train samples: {len(A_train_idx)}, Test samples: {len(A_test_idx)}")
    t_a_start = time.time()
    preproc_A, clf_A = fit_partition(A_train_feat, A_train_idx, "Label", which="A")
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
        predict_partition(A_test_feat, A_test_idx, preproc_A, clf_A, which="A")
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
