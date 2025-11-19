#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
공통 전처리 및 유틸리티 함수
- best08.py와 train_cnn.py에서 공통으로 사용하는 전처리 로직
"""
from typing import Tuple, List
from collections import namedtuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.calibration import calibration_curve

# =============================================================================
# 전처리 유틸
# =============================================================================

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

def _normalize_features(df, features, method='zscore', train_mask=None):
    """
    여러 피처를 정규화하여 결합
    
    Parameters:
    -----------
    df : DataFrame
    features : list - 정규화할 피처명 리스트
    method : str - 'zscore' or 'minmax'
    train_mask : Series, optional - train 데이터 마스크 (None이면 전체 데이터 사용)
                 train_mask가 제공되면 train 데이터만으로 통계 계산하여 전체에 적용
    
    Returns:
    --------
    DataFrame - 정규화된 피처들
    """
    result = pd.DataFrame(index=df.index)
    
    # 통계 계산용 df_train
    if train_mask is not None:
        df_train = df[train_mask]
    else:
        df_train = df
    
    for feat in features:
        if feat in df.columns:
            values_train = df_train[feat].fillna(df_train[feat].median())
            if method == 'zscore':
                mean_val = values_train.mean()
                std_val = values_train.std()
                if std_val > 0:
                    result[feat] = (df[feat].fillna(mean_val) - mean_val) / std_val
                else:
                    result[feat] = 0.0
            elif method == 'minmax':
                min_val = values_train.min()
                max_val = values_train.max()
                if max_val > min_val:
                    result[feat] = (df[feat] - min_val) / (max_val - min_val)
                else:
                    result[feat] = 0.0
    
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

def _create_age_normed_score(df, feature_name, age_col='Age_num', train_mask=None):
    """
    나이 bin별로 정규화된 점수 생성
    
    Parameters:
    -----------
    df : DataFrame
    feature_name : str - 정규화할 피처명
    age_col : str - 나이 컬럼명
    train_mask : Series, optional - train 데이터 마스크 (None이면 전체 데이터 사용)
                 train_mask가 제공되면 train 데이터만으로 통계 계산하여 전체에 적용
    
    Returns:
    --------
    Series - age-normed z-score
    """
    if feature_name not in df.columns:
        return pd.Series(np.nan, index=df.index)
    
    age_bins = df[age_col].apply(_create_age_bin)
    result = pd.Series(np.nan, index=df.index, dtype=float)
    
    for age_bin in age_bins.unique():
        if age_bin == 'Unknown':
            continue
        mask_bin = (age_bins == age_bin)
        
        # 통계 계산에 쓸 train subset
        if train_mask is not None:
            mask_train_bin = mask_bin & train_mask
            if mask_train_bin.sum() == 0:
                continue
            values_train = df.loc[mask_train_bin, feature_name]
        else:
            values_train = df.loc[mask_bin, feature_name]
        
        values_train = values_train.fillna(values_train.median())
        mean_val = values_train.mean()
        std_val = values_train.std()
        
        if std_val > 0:
            result.loc[mask_bin] = (
                df.loc[mask_bin, feature_name].fillna(mean_val) - mean_val
            ) / std_val
        else:
            result.loc[mask_bin] = 0.0
    
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

def add_features_A(df: pd.DataFrame, train_mask: pd.Series = None) -> pd.DataFrame:
    """
    A 고급 파생 (연속형 + composite 위주, hand-made 0/1 플래그 없음)
    
    Parameters:
    -----------
    df : DataFrame
    train_mask : Series, optional - train 데이터 마스크 (None이면 전체 데이터 사용)
                 train_mask가 제공되면 train 데이터만으로 통계 계산하여 전체에 적용
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
        normalized = _normalize_features(feats, perceptual_speed_features, method='zscore', train_mask=train_mask)
        feats['PerceptualSpeed_A'] = -normalized[perceptual_speed_features].mean(axis=1)

    # 2. CognitiveAbility_A: CogScore_A z-score (CogScore_A는 drop하여 중복 제거)
    if 'CogScore_A' in feats.columns:
        # train_mask가 있으면 train 데이터만으로 통계 계산
        if train_mask is not None:
            cog_train = feats.loc[train_mask, 'CogScore_A']
        else:
            cog_train = feats['CogScore_A']
        cog_mean = cog_train.mean()
        cog_std = cog_train.std()
        if cog_std > 0:
            feats['CognitiveAbility_A'] = (feats['CogScore_A'].fillna(cog_mean) - cog_mean) / cog_std
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
        normalized = _normalize_features(feats, emotional_features, method='zscore', train_mask=train_mask)
        feats['EmotionalRisk_A'] = -normalized[emotional_features].mean(axis=1)
        # 선형 결합 관계이므로 원본 피처들은 drop
        feats = feats.drop(columns=emotional_features, errors='ignore')

    # 최종 정리
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats

def add_features_B(df: pd.DataFrame, train_mask: pd.Series = None) -> pd.DataFrame:
    """
    B 고급 파생 (연속형 + Risk/Multitask composite 위주, hand-made 0/1 플래그 없음)
    
    Parameters:
    -----------
    df : DataFrame
    train_mask : Series, optional - train 데이터 마스크 (None이면 전체 데이터 사용)
                 train_mask가 제공되면 train 데이터만으로 통계 계산하여 전체에 적용
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
        normalized = _normalize_features(feats, multitask_features, method='zscore', train_mask=train_mask)
        if 'B10_multitask_cost_aud' in multitask_features and 'B10_multitask_cost_aud' in normalized.columns:
            normalized['B10_multitask_cost_aud'] = -normalized['B10_multitask_cost_aud']
        feats['MultitaskAbility_B'] = normalized[multitask_features].mean(axis=1)

    # 2. RiskScore_B_norm: z-normalized risk (높을수록 나쁨 → 부호 반전)
    # RiskScore_B는 drop하여 중복 제거 (선형 변환 관계)
    if 'RiskScore_B' in feats.columns:
        # train_mask가 있으면 train 데이터만으로 통계 계산
        if train_mask is not None:
            risk_train = feats.loc[train_mask, 'RiskScore_B']
        else:
            risk_train = feats['RiskScore_B']
        risk_mean = risk_train.mean()
        risk_std = risk_train.std()
        if risk_std > 0:
            feats['RiskScore_B_norm'] = -(feats['RiskScore_B'].fillna(risk_mean) - risk_mean) / risk_std
        else:
            feats['RiskScore_B_norm'] = 0.0
        # 선형 변환 관계이므로 원본 RiskScore_B는 drop
        feats = feats.drop(columns=['RiskScore_B'], errors='ignore')

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats

# =============================================================================
# Age-normed Composite Scores
# =============================================================================

def add_age_normed_composites_A(df: pd.DataFrame, train_mask: pd.Series = None) -> pd.DataFrame:
    """
    A용 age-normed composite scores 추가
    나이 bin별로 정규화된 composite score 생성 (age-split 구조 보완)
    
    Parameters:
    -----------
    df : DataFrame
    train_mask : Series, optional - train 데이터 마스크 (None이면 전체 데이터 사용)
                 train_mask가 제공되면 train 데이터만으로 통계 계산하여 전체에 적용
    """
    df = df.copy()

    # PerceptualSpeed_A_ageNorm
    if 'PerceptualSpeed_A' in df.columns and 'Age_num' in df.columns:
        df['PerceptualSpeed_A_ageNorm'] = _create_age_normed_score(
            df, 'PerceptualSpeed_A', age_col='Age_num', train_mask=train_mask
        )

    # CognitiveAbility_A_ageNorm
    if 'CognitiveAbility_A' in df.columns and 'Age_num' in df.columns:
        df['CognitiveAbility_A_ageNorm'] = _create_age_normed_score(
            df, 'CognitiveAbility_A', age_col='Age_num', train_mask=train_mask
        )

    # EmotionalRisk_A_ageNorm
    if 'EmotionalRisk_A' in df.columns and 'Age_num' in df.columns:
        df['EmotionalRisk_A_ageNorm'] = _create_age_normed_score(
            df, 'EmotionalRisk_A', age_col='Age_num', train_mask=train_mask
        )

    return df

def add_age_normed_composites_B(df: pd.DataFrame, train_mask: pd.Series = None) -> pd.DataFrame:
    """
    B용 age-normed composite scores 추가
    나이 bin별로 정규화된 composite score 생성 (history-split 구조 보완)
    
    Parameters:
    -----------
    df : DataFrame
    train_mask : Series, optional - train 데이터 마스크 (None이면 전체 데이터 사용)
                 train_mask가 제공되면 train 데이터만으로 통계 계산하여 전체에 적용
    """
    df = df.copy()

    if 'RiskScore_B_norm' in df.columns and 'Age_num' in df.columns:
        df['RiskScore_B_ageNorm'] = _create_age_normed_score(
            df, 'RiskScore_B_norm', age_col='Age_num', train_mask=train_mask
        )

    if 'MultitaskAbility_B' in df.columns and 'Age_num' in df.columns:
        df['MultitaskAbility_B_ageNorm'] = _create_age_normed_score(
            df, 'MultitaskAbility_B', age_col='Age_num', train_mask=train_mask
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
        trend = df["CognitiveAbility_A"].values - prev
        new_features["CognitiveAbility_A_trend"] = trend
        # Trend 방향성과 크기 분리 (tree 모델 split 최적화)
        new_features["CognitiveAbility_A_trend_direction"] = np.sign(trend)
        new_features["CognitiveAbility_A_trend_magnitude"] = np.abs(trend)

    if "A1_rt_mean" in df.columns:
        prev = _numpy_rolling_mean_shifted_safe(
            df["A1_rt_mean"].values, MAX_HISTORY_WINDOW, group_starts, group_ends,
            min_periods=MIN_PERIODS_MEAN
        )
        new_features["A1_rt_prev_mean"] = prev
        trend = df["A1_rt_mean"].values - prev
        new_features["A1_rt_trend"] = trend
        # Trend 방향성과 크기 분리 (tree 모델 split 최적화)
        new_features["A1_rt_trend_direction"] = np.sign(trend)
        new_features["A1_rt_trend_magnitude"] = np.abs(trend)

    if "A4_stroop_diff" in df.columns:
        prev = _numpy_rolling_mean_shifted_safe(
            df["A4_stroop_diff"].values, MAX_HISTORY_WINDOW, group_starts, group_ends,
            min_periods=MIN_PERIODS_MEAN
        )
        new_features["A4_stroop_prev_mean"] = prev
        trend = df["A4_stroop_diff"].values - prev
        new_features["A4_stroop_trend"] = trend
        # Trend 방향성과 크기 분리 (tree 모델 split 최적화)
        new_features["A4_stroop_trend_direction"] = np.sign(trend)
        new_features["A4_stroop_trend_magnitude"] = np.abs(trend)

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
        trend = b4_acc - prev
        new_features["B4_acc_trend"] = trend
        # Trend 방향성과 크기 분리 (tree 모델 split 최적화)
        new_features["B4_acc_trend_direction"] = np.sign(trend)
        new_features["B4_acc_trend_magnitude"] = np.abs(trend)

    # --- Multitask cost history (B10) ---
    if "B10_multitask_cost_aud" in df.columns:
        b10_cost = df["B10_multitask_cost_aud"].values.astype(np.float32)
        prev = _numpy_rolling_mean_shifted_safe(
            b10_cost, MAX_HISTORY_WINDOW, group_starts, group_ends,
            min_periods=MIN_PERIODS_MEAN
        )
        new_features["B10_multicost_prev_mean"] = prev
        trend = b10_cost - prev
        new_features["B10_multicost_trend"] = trend
        # Trend 방향성과 크기 분리 (tree 모델 split 최적화)
        new_features["B10_multicost_trend_direction"] = np.sign(trend)
        new_features["B10_multicost_trend_magnitude"] = np.abs(trend)

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
            trend = rt_values - prev
            new_features[f"{task}_rt_trend"] = trend
            # Trend 방향성과 크기 분리 (tree 모델 split 최적화)
            new_features[f"{task}_rt_trend_direction"] = np.sign(trend)
            new_features[f"{task}_rt_trend_magnitude"] = np.abs(trend)

    # ⚠️ prev_std, *_trend_per_month, *_trend_ratio,
    #     Age_x_B4_acc_trend, Age_x_B10_multicost_trend 등은 과감히 삭제

    if new_features:
        new_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_df], axis=1)

    return df

# =============================================================================
# Feature Block 적용 (블록 단위 드롭)
# =============================================================================

def apply_feature_blocks(df: pd.DataFrame, which: str, config: dict) -> pd.DataFrame:
    """
    which: 'A' or 'B'
    config: FEATURE_BLOCKS_A / FEATURE_BLOCKS_B
    """
    df = df.copy()
    cols_to_drop = []

    # 1) RT CV 제거
    if not config.get("USE_RT_CV", True):
        cols_to_drop.extend([
            c for c in df.columns
            if c.endswith("_rt_cv")
        ])

    # 2) ratio / gap 계열 제거
    if not config.get("USE_RATIO_GAP", True):
        cols_to_drop.extend([
            c for c in df.columns
            if (
                "speed_acc_tradeoff" in c
                or c.endswith("_gap")
                or c.endswith("_gap_abs")
                or c.endswith("_x_err")
            )
        ])

    # 3) Composite Scores 제거
    if not config.get("USE_COMPOSITE", True):
        composite_names = []
        if which == "A":
            composite_names = ["PerceptualSpeed_A", "CognitiveAbility_A", "EmotionalRisk_A"]
        else:  # B
            composite_names = ["MultitaskAbility_B", "RiskScore_B_norm"]
        cols_to_drop.extend([c for c in composite_names if c in df.columns])

    # 4) Global Stats 제거 (B 전용)
    if which == "B" and not config.get("USE_GLOBAL_STATS", True):
        cols_to_drop.extend([
            c for c in df.columns
            if c in ["B_rt_mean_global", "B_rt_std_global", "B_acc_mean_global", "B_acc_std_global"]
        ])

    # 5) Task Gaps 제거 (B 전용)
    if which == "B" and not config.get("USE_TASK_GAPS", True):
        cols_to_drop.extend([
            c for c in df.columns
            if c.startswith("B4_B3_") or c.startswith("B5_B1_")
        ])

    # 6) Multitask Ratio 제거 (B 전용)
    if which == "B" and not config.get("USE_MULTITASK_RATIO", True):
        cols_to_drop.extend([
            c for c in df.columns
            if "multitask_ratio" in c
        ])

    # 7) Age Interaction 제거
    if not config.get("USE_AGE_INTERACTION", True):
        cols_to_drop.extend([
            c for c in df.columns
            if c.startswith("Age_x_")
        ])

    # 8) age-normed composite 제거
    if not config.get("USE_AGE_NORMED", True):
        cols_to_drop.extend([
            c for c in df.columns
            if c.endswith("_ageNorm")
        ])

    # 9) History 기본 정보 제거
    if not config.get("USE_HISTORY_BASIC", True):
        cols_to_drop.extend([
            c for c in df.columns
            if c in ["n_tests_so_far", "months_since_first_test", "months_since_prev_test"]
        ])

    # 10) History 이전 평균 제거
    if not config.get("USE_HISTORY_PREV_MEAN", True):
        cols_to_drop.extend([
            c for c in df.columns
            if c.endswith("_prev_mean")
        ])

    # 11) History trend 제거
    if not config.get("USE_HISTORY_TREND", True):
        cols_to_drop.extend([
            c for c in df.columns
            if c.endswith("_trend") and not (c.endswith("_trend_direction") or c.endswith("_trend_magnitude"))
        ])

    # 12) trend 방향/크기 분해 피처 제거
    if not config.get("USE_TREND_DIRMAG", True):
        cols_to_drop.extend([
            c for c in df.columns
            if c.endswith("_trend_direction") or c.endswith("_trend_magnitude")
        ])

    # 13) 인디케이터 피처 제거 (0/1 플래그)
    if not config.get("USE_INDICATORS", True):
        indicator_patterns = []
        if which == "A":
            # A 전용 인디케이터
            indicator_patterns = [
                "Age_60plus",
                "A4_low_acc_flag",
                "A_low_cog_flag",
                "Old_and_low_A4",
                "A4_high_stroop_flag",
                "A1_slow_rt_flag",
            ]
        else:  # B
            # B 전용 인디케이터
            indicator_patterns = [
                "Age_60plus",
                "Age_80plus",
                "RiskB_top20",
                "B10_multicost_very_neg",
                "B10_multicost_very_pos",
                "B4_low_acc_flag",
                "B4_slow_rt_flag",
                "B_high_risk_flag",
                "Old_and_low_B4",
            ]
        
        # 정확한 이름 매칭 + 패턴 매칭 (flag로 끝나는 것들)
        for pattern in indicator_patterns:
            if pattern in df.columns:
                cols_to_drop.append(pattern)
        
        # _flag로 끝나는 모든 피처도 제거 (추가 안전장치)
        cols_to_drop.extend([
            c for c in df.columns
            if c.endswith("_flag") and c not in cols_to_drop
        ])

    # 중복 제거 후 드롭
    cols_to_drop = sorted(set(cols_to_drop))
    if cols_to_drop:
        print(f"[{which}] Dropping feature blocks: {len(cols_to_drop)} columns")
        # 필요하면 목록 찍어볼 수 있음:
        # for c in cols_to_drop: print("   -", c)
        df = df.drop(columns=cols_to_drop, errors="ignore")

    return df

