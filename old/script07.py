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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn import __version__ as sklver

import pickle  # 모델 저장용

# =============================================================================
# 경로 설정
# =============================================================================

DATA_DIR   = "data"
OUTPUT_DIR = "output"
MODEL_DIR  = "model"

SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
META_PATH       = os.path.join(MODEL_DIR, "meta.json")

# 모델/전처기 저장 경로
A_MODEL_PATH    = os.path.join(MODEL_DIR, "hgb_A.pkl")
B_MODEL_PATH    = os.path.join(MODEL_DIR, "hgb_B.pkl")
A_PREPROC_PATH  = os.path.join(MODEL_DIR, "preproc_A.pkl")
B_PREPROC_PATH  = os.path.join(MODEL_DIR, "preproc_B.pkl")

RANDOM_STATE = 42

# =============================================================================
# Age-based submodel 설정 (A 전용)
# =============================================================================
AGE_SPLIT_A       = 60.0   # under60 / over60 경계
AGE_BUFFER_MIN_A  = 55.0   # 버퍼 구간 시작 (<= 여기는 pure under60)
AGE_BUFFER_MAX_A  = 65.0   # 버퍼 구간 끝 (>= 여기는 pure over60)

# =============================================================================
# 실행 옵션
# =============================================================================
USE_CALIBRATION = True
CALIB_METHOD   = "isotonic"
CALIB_CV       = 3
TEST_SIZE_FOR_LOG = 0.1

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
    - age NaN         : 50% under + 50% over (보수적 기본값)
    """
    age = np.asarray(age_array, dtype=float)
    p_under = np.asarray(p_under, dtype=float)
    p_over  = np.asarray(p_over, dtype=float)

    # 기본: 전부 under로 두고 시작
    w_over = np.zeros_like(age, dtype=float)

    # NaN 처리: 중간값으로 0.5
    mask_nan = np.isnan(age)
    w_over[mask_nan] = 0.5

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
    feats = df.copy(); eps = 1e-6

    # -----------------------
    # 기존 고급 파생
    # -----------------------
    if _has(feats, ["Year","Month"]):
        feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]

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

    # -----------------------
    # 인디케이터 피처 (cutpoint 분석 결과 반영)
    # -----------------------

    # (1) 나이 60세 이상
    if "Age_num" in feats.columns:
        feats["Age_60plus"] = (feats["Age_num"] >= 60).astype(int)

    # (2) A4 정확도 매우 낮음 (A4_acc_rate <= 0.8)
    if "A4_acc_rate" in feats.columns:
        feats["A4_low_acc_flag"] = (feats["A4_acc_rate"] <= 0.8).astype(int)

    # (3) CogScore_A 낮음 (고정 임계값 26)
    if "CogScore_A" in feats.columns:
        feats["A_low_cog_flag"] = (feats["CogScore_A"] <= 26).astype(int)

    # (4) 고령 + A4 저정확도 결합
    if _has(feats, ["Age_num", "A4_acc_rate"]):
        feats["Old_and_low_A4"] = (
            (feats["Age_num"] >= 60) &
            (feats["A4_acc_rate"] <= 0.8)
        ).astype(int)

    feats.replace([np.inf,-np.inf], np.nan, inplace=True)
    return feats

def add_features_B(df: pd.DataFrame) -> pd.DataFrame:
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

    # -----------------------
    # 인디케이터 피처 (cutpoint 분석 결과 반영)
    # -----------------------

    # (1) 나이 60세 이상
    if "Age_num" in feats.columns:
        feats["Age_60plus"] = (feats["Age_num"] >= 60).astype(int)

    # (2) RiskScore_B 상위 20% (high risk 그룹)
    if "RiskScore_B" in feats.columns:
        try:
            thr = feats["RiskScore_B"].quantile(0.8)
            feats["RiskB_top20"] = (feats["RiskScore_B"] >= thr).astype(int)
        except Exception:
            feats["RiskB_top20"] = 0

    # (3) B10 multitask cost의 양 극단 (very negative / very positive)
    if "B10_multitask_cost_aud" in feats.columns:
        feats["B10_multicost_very_neg"] = (
            feats["B10_multitask_cost_aud"] <= -0.0875
        ).astype(int)
        feats["B10_multicost_very_pos"] = (
            feats["B10_multitask_cost_aud"] >= 0.04
        ).astype(int)

    feats.replace([np.inf,-np.inf], np.nan, inplace=True)
    return feats

# =============================================================================
# History Features
# =============================================================================

def add_history_features_A(df: pd.DataFrame) -> pd.DataFrame:
    """
    A history features using PrimaryKey + YearMonthIndex
    Uses expanding + shift(1) to avoid data leakage
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

    if "A4_acc_rate" in df.columns:
        df["A4_acc_prev_mean"] = (
            df.groupby("PrimaryKey")["A4_acc_rate"]
            .transform(lambda x: x.expanding().mean().shift(1))
        )

    if "CogScore_A" in df.columns:
        df["CogScore_A_prev_mean"] = (
            df.groupby("PrimaryKey")["CogScore_A"]
            .transform(lambda x: x.expanding().mean().shift(1))
        )
        df["CogScore_A_trend"] = (
            df["CogScore_A"] - df["CogScore_A_prev_mean"]
        )

    if "A1_rt_mean" in df.columns:
        df["A1_rt_prev_mean"] = (
            df.groupby("PrimaryKey")["A1_rt_mean"]
            .transform(lambda x: x.expanding().mean().shift(1))
        )
        df["A1_rt_trend"] = (
            df["A1_rt_mean"] - df["A1_rt_prev_mean"]
        )

    if "A4_stroop_diff" in df.columns:
        df["A4_stroop_prev_mean"] = (
            df.groupby("PrimaryKey")["A4_stroop_diff"]
            .transform(lambda x: x.expanding().mean().shift(1))
        )
        df["A4_stroop_trend"] = (
            df["A4_stroop_diff"] - df["A4_stroop_prev_mean"]
        )

    return df

def add_history_features_B(df: pd.DataFrame) -> pd.DataFrame:
    """
    B history features using PrimaryKey + YearMonthIndex
    Uses expanding + shift(1) to avoid data leakage
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

    if "B4_acc_rate" in df.columns:
        df["B4_acc_prev_mean"] = (
            df.groupby("PrimaryKey")["B4_acc_rate"]
            .transform(lambda x: x.expanding().mean().shift(1))
        )
        df["B4_acc_trend"] = (
            df["B4_acc_rate"] - df["B4_acc_prev_mean"]
        )

    if "B10_multitask_cost_aud" in df.columns:
        df["B10_multicost_prev_mean"] = (
            df.groupby("PrimaryKey")["B10_multitask_cost_aud"]
            .transform(lambda x: x.expanding().mean().shift(1))
        )
        df["B10_multicost_trend"] = (
            df["B10_multitask_cost_aud"] - df["B10_multicost_prev_mean"]
        )

    for task in ["B3", "B4", "B5"]:
        rt_col = f"{task}_rt_mean"
        if rt_col in df.columns:
            df[f"{task}_rt_prev_mean"] = (
                df.groupby("PrimaryKey")[rt_col]
                .transform(lambda x: x.expanding().mean().shift(1))
            )
            df[f"{task}_rt_trend"] = (
                df[rt_col] - df[f"{task}_rt_prev_mean"]
            )

    return df

# =============================================================================
# 학습 & 추론
# =============================================================================

def build_model(seed: int, which: str) -> HistGradientBoostingClassifier:
    if which == "A":
        params = BASE_HGB_PARAMS_A.copy()
    else:
        params = BASE_HGB_PARAMS_B.copy()
    params["random_state"] = seed
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

def fit_partition_A_submodels(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    label_col: str = "Label",
):
    """
    A용: Age_num 기준으로 <60, >=60 두 개의 3-seed 앙상블(HGB)을 학습.
    - 전처리(preproc)는 전체 데이터 기준으로 하나만 fit.
    - 검증 점수는 '블렌딩 후' 확률로 계산.
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

    # holdout split (Age도 같이 쪼갬)
    X_tr, X_val, y_tr, y_val, age_tr, age_val = train_test_split(
        X_all, y_all, age_all,
        test_size=TEST_SIZE_FOR_LOG,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    # 전처리 학습은 전체 train에 대해 1번만
    X_tr_t  = preproc.fit_transform(X_tr)
    X_val_t = preproc.transform(X_val)

    # under60 / over60 마스크
    mask_under_tr = np.isnan(age_tr) | (age_tr < AGE_SPLIT_A)
    mask_over_tr  = (~np.isnan(age_tr)) & (age_tr >= AGE_SPLIT_A)

    # 혹시라도 over 쪽에 데이터가 거의 없으면 방어
    if mask_over_tr.sum() == 0:
        print(f"[{which}] WARNING: no samples for Age >= {AGE_SPLIT_A}, fallback to single model.")
        # 그냥 기존 fit_partition 로직 비슷하게 한 개 모델만 학습
        members = []
        for sd in ENSEMBLE_SEEDS:
            base = build_model(sd, which=which).fit(X_tr_t, y_tr)
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
        # ---------- under60 앙상블 ----------
        X_tr_under = X_tr_t[mask_under_tr]
        y_tr_under = y_tr[mask_under_tr]

        members_under = []
        for sd in ENSEMBLE_SEEDS:
            base = build_model(sd, which=which).fit(X_tr_under, y_tr_under)
            mdl  = maybe_calibrate(base, X_tr_under, y_tr_under)
            members_under.append(mdl)
        ensemble_under = AvgProbaEnsemble(members_under)

        # ---------- over60 앙상블 ----------
        X_tr_over = X_tr_t[mask_over_tr]
        y_tr_over = y_tr[mask_over_tr]

        members_over = []
        for sd in ENSEMBLE_SEEDS:
            base = build_model(sd, which=which).fit(X_tr_over, y_tr_over)
            mdl  = maybe_calibrate(base, X_tr_over, y_tr_over)
            members_over.append(mdl)
        ensemble_over = AvgProbaEnsemble(members_over)

        # ---------- holdout에서 blended proba 평가 ----------
        try:
            p_under_val = ensemble_under.predict_proba(X_val_t)[:, 1]
            p_over_val  = ensemble_over.predict_proba(X_val_t)[:, 1]
            blended_val = blend_by_age_with_buffer(age_val, p_under_val, p_over_val)
            metrics = evaluate_score(y_val, blended_val)
            print(
                f"[{which}] Holdout(age-split) - "
                f"AUC={metrics.auc:.5f}, "
                f"Brier={metrics.brier:.5f}, "
                f"ECE={metrics.ece:.5f}, "
                f"Score={metrics.score:.5f}"
            )
        except Exception as e:
            print(f"[{which}] validation logging skipped: {e}")

        model_bundle = dict(
            mode="age_split",
            ensemble_under60=ensemble_under,
            ensemble_over60=ensemble_over,
            age_split=AGE_SPLIT_A,
            buffer_min=AGE_BUFFER_MIN_A,
            buffer_max=AGE_BUFFER_MAX_A,
        )

    # ===== 저장 =====
    preproc_path = A_PREPROC_PATH
    model_path   = A_MODEL_PATH

    with open(preproc_path, "wb") as f:
        pickle.dump(preproc, f, protocol=4)
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f, protocol=4)

    print(f"[{which}] saved model bundle & preproc -> {model_path}, {preproc_path}")

    # 반환은 preproc + 서브모델들 (단일모델 fallback 포함)
    if model_bundle.get("mode") == "age_split":
        return preproc, model_bundle["ensemble_under60"], model_bundle["ensemble_over60"]
    else:
        # single model fallback일 때는 over 모델 자리에 None
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

def save_meta():
    meta = dict(
        model="HGB(3-seed ensemble) + Age-based submodels(A) + script04_A FE + history features",
        hgb_params_A=BASE_HGB_PARAMS_A,
        hgb_params_B=BASE_HGB_PARAMS_B,
        ensemble_seeds=list(ENSEMBLE_SEEDS),
        use_calibration=USE_CALIBRATION,
        calib_method=CALIB_METHOD,
        calib_cv=CALIB_CV,
        age_split_A=AGE_SPLIT_A,
        age_buffer_min_A=AGE_BUFFER_MIN_A,
        age_buffer_max_A=AGE_BUFFER_MAX_A,
        sklearn_version=sklver,
        random_state=RANDOM_STATE,
        save_format="pickle_protocol_4",
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# =============================================================================
# 메인
# =============================================================================

def main():
    set_seed(RANDOM_STATE)
    t0 = time.time()
    ensure_dirs()

    print("📂 Loading index files ...")
    train_idx, test_idx = read_index_files()

    print("📂 Loading raw A/B (train/test) ...")
    A_train_raw, B_train_raw = read_raw_feature_files("train")
    A_test_raw,  B_test_raw  = read_raw_feature_files("test")

    print("\n🔵 FE for A (with history features) ...")
    A_all_raw = pd.concat([A_train_raw, A_test_raw], axis=0, ignore_index=True)
    A_all = preprocess_A(A_all_raw)
    A_all = add_features_A(A_all)
    A_all = add_history_features_A(A_all)

    A_train_ids = set(A_train_raw["Test_id"])
    A_test_ids = set(A_test_raw["Test_id"])
    A_train_feat = A_all[A_all["Test_id"].isin(A_train_ids)].reset_index(drop=True)
    A_test_feat = A_all[A_all["Test_id"].isin(A_test_ids)].reset_index(drop=True)
    print(f"  A_train_feat: {A_train_feat.shape}, A_test_feat: {A_test_feat.shape}")

    print("\n🟢 FE for B (with history features) ...")
    B_all_raw = pd.concat([B_train_raw, B_test_raw], axis=0, ignore_index=True)
    B_all = preprocess_B(B_all_raw)
    B_all = add_features_B(B_all)
    B_all = add_history_features_B(B_all)

    B_train_ids = set(B_train_raw["Test_id"])
    B_test_ids = set(B_test_raw["Test_id"])
    B_train_feat = B_all[B_all["Test_id"].isin(B_train_ids)].reset_index(drop=True)
    B_test_feat = B_all[B_all["Test_id"].isin(B_test_ids)].reset_index(drop=True)
    print(f"  B_train_feat: {B_train_feat.shape}, B_test_feat: {B_test_feat.shape}")

    # ===== A 학습 (Age-based submodels) =====
    A_train_idx = train_idx[train_idx["Test"] == "A"].copy()
    A_test_idx  = test_idx[test_idx["Test"] == "A"].copy()
    print("\n[A] training with age-based submodels ...")
    preproc_A, clf_A_under, clf_A_over = fit_partition_A_submodels(A_train_feat, A_train_idx, "Label")

    # ===== B 학습 =====
    B_train_idx = train_idx[train_idx["Test"] == "B"].copy()
    B_test_idx  = test_idx[test_idx["Test"] == "B"].copy()
    print("\n[B] training ...")
    preproc_B, clf_B = fit_partition(B_train_feat, B_train_idx, "Label", "B")

    # ===== 추론 =====
    print("\n🔮 Inference on test ...")
    preds_A = (
        predict_partition_A_submodels(A_test_feat, A_test_idx, preproc_A, clf_A_under, clf_A_over)
        if len(A_test_idx) else None
    )
    preds_B = (
        predict_partition(B_test_feat, B_test_idx, preproc_B, clf_B, "B")
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

    sub.to_csv(SUBMISSION_PATH, index=False)
    save_meta()

    dt = time.time() - t0
    print(f"\n✅ submission saved -> {SUBMISSION_PATH} | elapsed: {dt/60:.2f} min")
    print(f"✅ models saved to: {MODEL_DIR}")

if __name__ == "__main__":
    main()
