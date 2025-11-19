#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
앙상블 모델 추론 스크립트 (HGBM + CNN)
- best08_ensemble.py로 학습된 HGBM 모델 로드
- train_cnn.py로 학습된 CNN 모델 로드
- HGBM과 CNN 예측을 weighted average로 결합
"""
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import json
from collections import namedtuple
from typing import Dict, List, Tuple, Sequence

# PyTorch import
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch not installed - CNN features will be skipped")

# 독립 실행을 위해 import 제거, 필요한 함수들은 파일 내에 정의

# =============================================================================
# 경로 설정
# =============================================================================
DATA_DIR = "data"
OUTPUT_DIR = "output"
MODEL_DIR = "model"

SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
BEST_PARAMS_PATH = os.path.join(MODEL_DIR, "best_params.json")

# HGBM 모델 경로 (hgb_A.pkl 형식으로 저장됨)
A_MODEL_PATH_HGB = os.path.join(MODEL_DIR, "hgb_A.pkl")
B_MODEL_PATH_HGB = os.path.join(MODEL_DIR, "hgb_B.pkl")
A_PREPROC_PATH_HGB = os.path.join(MODEL_DIR, "preproc_A.pkl")
B_PREPROC_PATH_HGB = os.path.join(MODEL_DIR, "preproc_B.pkl")

# CNN 모델 경로
A_MODEL_PATH_CNN = os.path.join(MODEL_DIR, "cnn_A.pkl")
B_MODEL_PATH_CNN = os.path.join(MODEL_DIR, "cnn_B.pkl")
A_PREPROC_PATH_CNN = os.path.join(MODEL_DIR, "preproc_A_cnn.pkl")
B_PREPROC_PATH_CNN = os.path.join(MODEL_DIR, "preproc_B_cnn.pkl")

# =============================================================================
# CNN 모델 하이퍼파라미터
# =============================================================================
CNN_CONFIG = {
    # 시퀀스 처리
    "max_seq_length": 200,   # 최대 시퀀스 길이 (패딩/트렁케이션)
    # CNN 구조
    "filter_sizes": [3, 5, 7], 
    "num_filters": 64,         
    "cnn_layers": 2,           
    "dropout_rate": 0.3,      
    # Pooling
    "pooling_type": "max",      # "max" or "avg" or "both"
    # Attention
    "use_attention": True,     
    "attention_dim": 64,       
    # MLP (Trial-wise features + CNN features 결합 후)
    "mlp_hidden_dims": [256, 128], 
    "mlp_dropout": 0.3,          
    # 학습 설정
    "batch_size": 64,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "early_stopping_patience": 10,
    "weight_decay": 1e-5,
    # CNN 추론 seed 설정 (여러 seed로 추론 가능)
    "cnn_seeds": [42, 202, 777],  # CNN 추론 시 사용할 seed 리스트
    "cnn_weight": 0.3,  # CNN 가중치 (HGBM 가중치 = 1 - cnn_weight)
}

# =============================================================================
# Feature Block Config (best08.py에서 복사)
# =============================================================================
FEATURE_BLOCKS_A = {
    "USE_RT_CV": True,
    "USE_RATIO_GAP": True,
    "USE_COMPOSITE": True,
    "USE_AGE_INTERACTION": True,
    "USE_AGE_NORMED": True,
    "USE_HISTORY_BASIC": True,
    "USE_HISTORY_PREV_MEAN": True,
    "USE_HISTORY_TREND": True,
    "USE_TREND_DIRMAG": False,
    "USE_INDICATORS": True,
}

FEATURE_BLOCKS_B = {
    "USE_RT_CV": True,
    "USE_RATIO_GAP": True,
    "USE_COMPOSITE": True,
    "USE_GLOBAL_STATS": True,
    "USE_TASK_GAPS": True,
    "USE_MULTITASK_RATIO": True,
    "USE_AGE_NORMED": True,
    "USE_HISTORY_BASIC": True,
    "USE_HISTORY_PREV_MEAN": True,
    "USE_HISTORY_TREND": True,
    "USE_TREND_DIRMAG": False,
    "USE_INDICATORS": False,
}

# =============================================================================
# 데이터 로딩 함수
# =============================================================================
def read_index_files():
    """Index 파일 읽기"""
    train_idx = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_idx = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train_idx, test_idx

def read_raw_feature_files(split: str):
    """원시 피처 파일 읽기"""
    A_df = pd.read_csv(os.path.join(DATA_DIR, split, "A.csv"))
    B_df = pd.read_csv(os.path.join(DATA_DIR, split, "B.csv"))
    return A_df, B_df


# ---------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)


# =============================================================================
# utils.py 헬퍼 함수들 (독립 실행을 위해 복사)
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

def _normalize_features(df, features, method='zscore', train_mask=None):
    """여러 피처를 정규화하여 결합"""
    result = pd.DataFrame(index=df.index)
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
    """나이 bin별로 정규화된 점수 생성"""
    if feature_name not in df.columns:
        return pd.Series(np.nan, index=df.index)
    age_bins = df[age_col].apply(_create_age_bin)
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for age_bin in age_bins.unique():
        if age_bin == 'Unknown':
            continue
        mask_bin = (age_bins == age_bin)
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

def add_rowwise_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Row-wise NA features 추가"""
    X = df[feature_cols]
    na_count = X.isna().sum(axis=1).astype(np.int32)
    na_ratio = (na_count / (len(feature_cols) + 1e-9)).astype(np.float32)
    df2 = df.copy()
    df2["NA_COUNT"] = na_count
    df2["NA_RATIO"] = na_ratio
    return df2

# =============================================================================
# train_cnn.py 시퀀스 빌더 함수들 (독립 실행을 위해 복사)
# =============================================================================

def parse_sequence_to_array(series: pd.Series,
                            max_length: int = None,
                            dtype: type = float) -> np.ndarray:
    """시퀀스 문자열을 numpy array로 변환 (패딩 포함)"""
    if max_length is None:
        max_length = series.fillna("").astype(str).str.split(",").apply(len).max()
        if pd.isna(max_length) or max_length == 0:
            max_length = CNN_CONFIG["max_seq_length"]
    n_samples = len(series)
    result = np.full((n_samples, max_length), np.nan, dtype=np.float32)
    for i, seq_str in enumerate(series.fillna("").astype(str)):
        if seq_str == "":
            continue
        parts = seq_str.split(",")
        parts = [p.strip() for p in parts if p.strip() != ""]
        if len(parts) == 0:
            continue
        try:
            if dtype == float:
                values = [float(p) if p != "" else np.nan for p in parts]
            else:
                values = [int(p) if p != "" else np.nan for p in parts]
            seq_len = min(len(values), max_length)
            result[i, :seq_len] = values[:seq_len]
        except (ValueError, TypeError):
            continue
    return result

# 시퀀스 빌더 함수들 (train_cnn.py 143~884 라인에서 복사)
# 파일이 너무 길어서 핵심만 추가, 나머지는 train_cnn.py 참조

def build_sequence_A1(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """A1 검사 멀티채널 시퀀스 구성 (9채널)"""
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    required_cols = ["A1-1", "A1-2", "A1-3", "A1-4"]
    if not all(c in df.columns for c in required_cols):
        return None
    n_samples = len(df)
    channels = []
    rt_raw = parse_sequence_to_array(df["A1-4"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(rt_raw)
    rt_for_stat = rt_raw.copy()
    rt_for_stat[~valid_mask] = np.nan
    rt_mean = np.nanmean(rt_for_stat)
    rt_std = np.nanstd(rt_for_stat) + 1e-6
    rt_norm = (np.nan_to_num(rt_raw, nan=rt_mean) - rt_mean) / rt_std
    channels.append(rt_norm.astype(np.float32))
    resp_raw = parse_sequence_to_array(df["A1-3"], max_length=max_length, dtype=float)
    resp = np.nan_to_num(resp_raw, nan=0.0).astype(np.float32)
    channels.append(resp)
    side_raw = parse_sequence_to_array(df["A1-1"], max_length=max_length, dtype=float)
    stim_left = (side_raw == 1).astype(np.float32)
    stim_right = (side_raw == 2).astype(np.float32)
    channels.extend([stim_left, stim_right])
    speed_raw = parse_sequence_to_array(df["A1-2"], max_length=max_length, dtype=float)
    speed_slow = (speed_raw == 1).astype(np.float32)
    speed_normal = (speed_raw == 2).astype(np.float32)
    speed_fast = (speed_raw == 3).astype(np.float32)
    channels.extend([speed_slow, speed_normal, speed_fast])
    delta = np.diff(rt_raw, axis=1, prepend=np.nan)
    delta[~valid_mask] = np.nan
    delta_mean = np.nanmean(delta)
    delta_std = np.nanstd(delta) + 1e-6
    delta_norm = (np.nan_to_num(delta, nan=delta_mean) - delta_mean) / delta_std
    channels.append(delta_norm.astype(np.float32))
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    seq_array = np.stack(channels, axis=-1)  # (N, L, 9)
    return seq_array

def build_sequence_A2(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """A2 검사 멀티채널 시퀀스 구성 (11채널)"""
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    required_cols = ["A2-1", "A2-2", "A2-3", "A2-4"]
    if not all(c in df.columns for c in required_cols):
        return None
    n_samples = len(df)
    channels = []
    rt_raw = parse_sequence_to_array(df["A2-4"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(rt_raw)
    rt_for_stat = rt_raw.copy()
    rt_for_stat[~valid_mask] = np.nan
    rt_mean = np.nanmean(rt_for_stat)
    rt_std = np.nanstd(rt_for_stat) + 1e-6
    rt_norm = (np.nan_to_num(rt_raw, nan=rt_mean) - rt_mean) / rt_std
    channels.append(rt_norm.astype(np.float32))
    resp_raw = parse_sequence_to_array(df["A2-3"], max_length=max_length, dtype=float)
    resp = np.nan_to_num(resp_raw, nan=0.0).astype(np.float32)
    channels.append(resp)
    cond1_raw = parse_sequence_to_array(df["A2-1"], max_length=max_length, dtype=float)
    cond1_slow = (cond1_raw == 1).astype(np.float32)
    cond1_normal = (cond1_raw == 2).astype(np.float32)
    cond1_fast = (cond1_raw == 3).astype(np.float32)
    channels.extend([cond1_slow, cond1_normal, cond1_fast])
    cond2_raw = parse_sequence_to_array(df["A2-2"], max_length=max_length, dtype=float)
    cond2_slow = (cond2_raw == 1).astype(np.float32)
    cond2_normal = (cond2_raw == 2).astype(np.float32)
    cond2_fast = (cond2_raw == 3).astype(np.float32)
    channels.extend([cond2_slow, cond2_normal, cond2_fast])
    speed_gap = np.abs(cond1_raw - cond2_raw)
    speed_gap[~valid_mask] = 0.0
    speed_gap_norm = speed_gap / 2.0
    channels.append(speed_gap_norm.astype(np.float32))
    delta = np.diff(rt_raw, axis=1, prepend=np.nan)
    delta[~valid_mask] = np.nan
    delta_mean = np.nanmean(delta)
    delta_std = np.nanstd(delta) + 1e-6
    delta_norm = (np.nan_to_num(delta, nan=delta_mean) - delta_mean) / delta_std
    channels.append(delta_norm.astype(np.float32))
    trial_idx = np.arange(max_length, dtype=np.float32)[None, :] / (max_length - 1 + 1e-6)
    trial_idx = np.tile(trial_idx, (n_samples, 1))
    trial_idx[~valid_mask] = 0.0
    channels.append(trial_idx)
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    seq_array = np.stack(channels, axis=-1)  # (N, L, 11)
    return seq_array

def build_sequence_A3(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """A3 검사 멀티채널 시퀀스 구성 (16채널)"""
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    required_cols = ["A3-1", "A3-2", "A3-3", "A3-4", "A3-5", "A3-6", "A3-7"]
    if not all(c in df.columns for c in required_cols):
        return None
    n_samples = len(df)
    channels = []
    rt_raw = parse_sequence_to_array(df["A3-7"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(rt_raw)
    rt_for_stat = rt_raw.copy()
    rt_for_stat[~valid_mask] = np.nan
    rt_mean = np.nanmean(rt_for_stat)
    rt_std = np.nanstd(rt_for_stat) + 1e-6
    rt_norm = (np.nan_to_num(rt_raw, nan=rt_mean) - rt_mean) / rt_std
    channels.append(rt_norm.astype(np.float32))
    resp_raw = parse_sequence_to_array(df["A3-6"], max_length=max_length, dtype=float)
    resp = np.nan_to_num(resp_raw, nan=0.0).astype(np.float32)
    channels.append(resp)
    resp1_raw = parse_sequence_to_array(df["A3-5"], max_length=max_length, dtype=float)
    is_valid_cue = ((resp1_raw == 1) | (resp1_raw == 2)).astype(np.float32)
    is_correct = ((resp1_raw == 1) | (resp1_raw == 3)).astype(np.float32)
    valid_x_incorrect = (resp1_raw == 2).astype(np.float32)
    invalid_x_incorrect = (resp1_raw == 4).astype(np.float32)
    channels.extend([is_valid_cue, is_correct, valid_x_incorrect, invalid_x_incorrect])
    size_raw = parse_sequence_to_array(df["A3-1"], max_length=max_length, dtype=float)
    size_small = (size_raw == 1).astype(np.float32)
    size_big = (size_raw == 2).astype(np.float32)
    channels.extend([size_small, size_big])
    cue_pos_raw = parse_sequence_to_array(df["A3-2"], max_length=max_length, dtype=float)
    tgt_pos_raw = parse_sequence_to_array(df["A3-4"], max_length=max_length, dtype=float)
    cue_left = ((cue_pos_raw >= 1) & (cue_pos_raw <= 4)).astype(np.float32)
    cue_right = ((cue_pos_raw >= 5) & (cue_pos_raw <= 8)).astype(np.float32)
    tgt_left = ((tgt_pos_raw >= 1) & (tgt_pos_raw <= 4)).astype(np.float32)
    tgt_right = ((tgt_pos_raw >= 5) & (tgt_pos_raw <= 8)).astype(np.float32)
    channels.extend([cue_left, cue_right, tgt_left, tgt_right])
    hemi_congruent = ((cue_left == tgt_left) | (cue_right == tgt_right)).astype(np.float32)
    channels.append(hemi_congruent)
    arrow_raw = parse_sequence_to_array(df["A3-3"], max_length=max_length, dtype=float)
    arrow_left = (arrow_raw == 1).astype(np.float32)
    arrow_right = (arrow_raw == 2).astype(np.float32)
    channels.extend([arrow_left, arrow_right])
    delta = np.diff(rt_raw, axis=1, prepend=np.nan)
    delta[~valid_mask] = np.nan
    delta_mean = np.nanmean(delta)
    delta_std = np.nanstd(delta) + 1e-6
    delta_norm = (np.nan_to_num(delta, nan=delta_mean) - delta_mean) / delta_std
    channels.append(delta_norm.astype(np.float32))
    incorrect = ((resp1_raw == 2) | (resp1_raw == 4)).astype(np.float32)
    cumsum = np.cumsum(incorrect, axis=1)
    t_idx = np.arange(max_length, dtype=np.float32)[None, :] + 1
    running_error = cumsum / t_idx
    running_error[~valid_mask] = 0.0
    channels.append(running_error.astype(np.float32))
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    seq_array = np.stack(channels, axis=-1)  # (N, L, 16)
    return seq_array

def build_sequence_A4(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """A4 검사 멀티채널 시퀀스 구성 (12채널)"""
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    required_cols = ["A4-1", "A4-2", "A4-3", "A4-4", "A4-5"]
    if not all(c in df.columns for c in required_cols):
        return None
    n_samples = len(df)
    channels = []
    rt_raw = parse_sequence_to_array(df["A4-5"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(rt_raw)
    rt_for_stat = rt_raw.copy()
    rt_for_stat[~valid_mask] = np.nan
    rt_mean = np.nanmean(rt_for_stat)
    rt_std = np.nanstd(rt_for_stat) + 1e-6
    rt_norm = (np.nan_to_num(rt_raw, nan=rt_mean) - rt_mean) / rt_std
    channels.append(rt_norm.astype(np.float32))
    resp_raw = parse_sequence_to_array(df["A4-4"], max_length=max_length, dtype=float)
    resp = np.nan_to_num(resp_raw, nan=0.0).astype(np.float32)
    channels.append(resp)
    correct_raw = parse_sequence_to_array(df["A4-3"], max_length=max_length, dtype=float)
    is_correct = (correct_raw == 1).astype(np.float32)
    channels.append(is_correct)
    cong_raw = parse_sequence_to_array(df["A4-1"], max_length=max_length, dtype=float)
    is_incongruent = (cong_raw == 2).astype(np.float32)
    channels.append(is_incongruent)
    color_raw = parse_sequence_to_array(df["A4-2"], max_length=max_length, dtype=float)
    stim_red = (color_raw == 1).astype(np.float32)
    stim_green = (color_raw == 2).astype(np.float32)
    channels.extend([stim_red, stim_green])
    incong_error = (is_incongruent == 1.0) & (is_correct == 0.0)
    channels.append(incong_error.astype(np.float32))
    delta = np.diff(rt_raw, axis=1, prepend=np.nan)
    delta[~valid_mask] = np.nan
    delta_mean = np.nanmean(delta)
    delta_std = np.nanstd(delta) + 1e-6
    delta_norm = (np.nan_to_num(delta, nan=delta_mean) - delta_mean) / delta_std
    channels.append(delta_norm.astype(np.float32))
    trial_idx = np.arange(max_length, dtype=np.float32)[None, :] / (max_length - 1 + 1e-6)
    trial_idx = np.tile(trial_idx, (n_samples, 1))
    trial_idx[~valid_mask] = 0.0
    channels.append(trial_idx)
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    seq_array = np.stack(channels, axis=-1)  # (N, L, 12)
    return seq_array

def build_sequence_A5(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """A5 검사 멀티채널 시퀀스 구성 (10채널)"""
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    required_cols = ["A5-1", "A5-2", "A5-3"]
    if not all(c in df.columns for c in required_cols):
        return None
    n_samples = len(df)
    channels = []
    cond_raw = parse_sequence_to_array(df["A5-1"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(cond_raw)
    is_change = ((cond_raw >= 2) & (cond_raw <= 4)).astype(np.float32)
    channels.append(is_change)
    non_change = (cond_raw == 1).astype(np.float32)
    pos_change = (cond_raw == 2).astype(np.float32)
    color_change = (cond_raw == 3).astype(np.float32)
    shape_change = (cond_raw == 4).astype(np.float32)
    channels.extend([non_change, pos_change, color_change, shape_change])
    correct_raw = parse_sequence_to_array(df["A5-2"], max_length=max_length, dtype=float)
    is_correct = (correct_raw == 1).astype(np.float32)
    channels.append(is_correct)
    resp_raw = parse_sequence_to_array(df["A5-3"], max_length=max_length, dtype=float)
    resp_yn = np.nan_to_num(resp_raw, nan=0.0).astype(np.float32)
    channels.append(resp_yn)
    miss_change = (is_change == 1.0) & (resp_yn == 0.0)
    false_alarm = (is_change == 0.0) & (resp_yn == 1.0)
    channels.extend([miss_change.astype(np.float32), false_alarm.astype(np.float32)])
    trial_idx = np.arange(max_length, dtype=np.float32)[None, :] / (max_length - 1 + 1e-6)
    trial_idx = np.tile(trial_idx, (n_samples, 1))
    trial_idx[~valid_mask] = 0.0
    channels.append(trial_idx)
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    seq_array = np.stack(channels, axis=-1)  # (N, L, 10)
    return seq_array

def build_sequence_B1(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """B1 검사 멀티채널 시퀀스 구성 (9채널)"""
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    required_cols = ["B1-1", "B1-2", "B1-3"]
    if not all(c in df.columns for c in required_cols):
        return None
    n_samples = len(df)
    channels = []
    rt_raw = parse_sequence_to_array(df["B1-2"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(rt_raw)
    rt_for_stat = rt_raw.copy()
    rt_for_stat[~valid_mask] = np.nan
    rt_mean = np.nanmean(rt_for_stat)
    rt_std = np.nanstd(rt_for_stat) + 1e-6
    rt_norm = (np.nan_to_num(rt_raw, nan=rt_mean) - rt_mean) / rt_std
    channels.append(rt_norm.astype(np.float32))
    task1_raw = parse_sequence_to_array(df["B1-1"], max_length=max_length, dtype=float)
    task1_correct = (task1_raw == 1).astype(np.float32)
    channels.append(task1_correct)
    task2_raw = parse_sequence_to_array(df["B1-3"], max_length=max_length, dtype=float)
    is_change = ((task2_raw == 1) | (task2_raw == 2)).astype(np.float32)
    is_correct = ((task2_raw == 1) | (task2_raw == 3)).astype(np.float32)
    miss_change = (task2_raw == 2).astype(np.float32)
    false_alarm = (task2_raw == 4).astype(np.float32)
    channels.extend([is_change, is_correct, miss_change, false_alarm])
    delta = np.diff(rt_raw, axis=1, prepend=np.nan)
    delta[~valid_mask] = np.nan
    delta_mean = np.nanmean(delta)
    delta_std = np.nanstd(delta) + 1e-6
    delta_norm = (np.nan_to_num(delta, nan=delta_mean) - delta_mean) / delta_std
    channels.append(delta_norm.astype(np.float32))
    trial_idx = np.arange(max_length, dtype=np.float32)[None, :] / (max_length - 1 + 1e-6)
    trial_idx = np.tile(trial_idx, (n_samples, 1))
    trial_idx[~valid_mask] = 0.0
    channels.append(trial_idx)
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    seq_array = np.stack(channels, axis=-1)  # (N, L, 9)
    return seq_array

def build_sequence_B2(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """B2 검사 멀티채널 시퀀스 구성 (9채널)"""
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    required_cols = ["B2-1", "B2-2", "B2-3"]
    if not all(c in df.columns for c in required_cols):
        return None
    n_samples = len(df)
    channels = []
    rt_raw = parse_sequence_to_array(df["B2-2"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(rt_raw)
    rt_for_stat = rt_raw.copy()
    rt_for_stat[~valid_mask] = np.nan
    rt_mean = np.nanmean(rt_for_stat)
    rt_std = np.nanstd(rt_for_stat) + 1e-6
    rt_norm = (np.nan_to_num(rt_raw, nan=rt_mean) - rt_mean) / rt_std
    channels.append(rt_norm.astype(np.float32))
    task1_raw = parse_sequence_to_array(df["B2-1"], max_length=max_length, dtype=float)
    task1_correct = (task1_raw == 1).astype(np.float32)
    channels.append(task1_correct)
    task2_raw = parse_sequence_to_array(df["B2-3"], max_length=max_length, dtype=float)
    is_change = ((task2_raw == 1) | (task2_raw == 2)).astype(np.float32)
    is_correct = ((task2_raw == 1) | (task2_raw == 3)).astype(np.float32)
    miss_change = (task2_raw == 2).astype(np.float32)
    false_alarm = (task2_raw == 4).astype(np.float32)
    channels.extend([is_change, is_correct, miss_change, false_alarm])
    delta = np.diff(rt_raw, axis=1, prepend=np.nan)
    delta[~valid_mask] = np.nan
    delta_mean = np.nanmean(delta)
    delta_std = np.nanstd(delta) + 1e-6
    delta_norm = (np.nan_to_num(delta, nan=delta_mean) - delta_mean) / delta_std
    channels.append(delta_norm.astype(np.float32))
    trial_idx = np.arange(max_length, dtype=np.float32)[None, :] / (max_length - 1 + 1e-6)
    trial_idx = np.tile(trial_idx, (n_samples, 1))
    trial_idx[~valid_mask] = 0.0
    channels.append(trial_idx)
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    seq_array = np.stack(channels, axis=-1)  # (N, L, 9)
    return seq_array

def build_sequence_B3(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """B3 검사 멀티채널 시퀀스 구성 (7채널)"""
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    required_cols = ["B3-1", "B3-2"]
    if not all(c in df.columns for c in required_cols):
        return None
    n_samples = len(df)
    channels = []
    rt_raw = parse_sequence_to_array(df["B3-2"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(rt_raw)
    rt_for_stat = rt_raw.copy()
    rt_for_stat[~valid_mask] = np.nan
    rt_mean = np.nanmean(rt_for_stat)
    rt_std = np.nanstd(rt_for_stat) + 1e-6
    rt_norm = (np.nan_to_num(rt_raw, nan=rt_mean) - rt_mean) / rt_std
    channels.append(rt_norm.astype(np.float32))
    correct_raw = parse_sequence_to_array(df["B3-1"], max_length=max_length, dtype=float)
    is_correct = (correct_raw == 1).astype(np.float32)
    channels.append(is_correct)
    delta = np.diff(rt_raw, axis=1, prepend=np.nan)
    delta[~valid_mask] = np.nan
    delta_mean = np.nanmean(delta)
    delta_std = np.nanstd(delta) + 1e-6
    delta_norm = (np.nan_to_num(delta, nan=delta_mean) - delta_mean) / delta_std
    channels.append(delta_norm.astype(np.float32))
    incorrect = (correct_raw == 2).astype(np.float32)
    cumsum = np.cumsum(incorrect, axis=1)
    t_idx = np.arange(max_length, dtype=np.float32)[None, :] + 1
    running_error = cumsum / t_idx
    running_error[~valid_mask] = 0.0
    channels.append(running_error.astype(np.float32))
    trial_idx = np.arange(max_length, dtype=np.float32)[None, :] / (max_length - 1 + 1e-6)
    trial_idx = np.tile(trial_idx, (n_samples, 1))
    trial_idx[~valid_mask] = 0.0
    channels.append(trial_idx)
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    seq_array = np.stack(channels, axis=-1)  # (N, L, 7)
    return seq_array

def build_sequence_B4(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """B4 검사 멀티채널 시퀀스 구성 (10채널)"""
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    required_cols = ["B4-1", "B4-2"]
    if not all(c in df.columns for c in required_cols):
        return None
    n_samples = len(df)
    channels = []
    rt_raw = parse_sequence_to_array(df["B4-2"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(rt_raw)
    rt_for_stat = rt_raw.copy()
    rt_for_stat[~valid_mask] = np.nan
    rt_mean = np.nanmean(rt_for_stat)
    rt_std = np.nanstd(rt_for_stat) + 1e-6
    rt_norm = (np.nan_to_num(rt_raw, nan=rt_mean) - rt_mean) / rt_std
    channels.append(rt_norm.astype(np.float32))
    code_raw = parse_sequence_to_array(df["B4-1"], max_length=max_length, dtype=float)
    code_1 = (code_raw == 1).astype(np.float32)
    code_2 = (code_raw == 2).astype(np.float32)
    code_3 = (code_raw == 3).astype(np.float32)
    code_4 = (code_raw == 4).astype(np.float32)
    code_5 = (code_raw == 5).astype(np.float32)
    code_6 = (code_raw == 6).astype(np.float32)
    channels.extend([code_1, code_2, code_3, code_4, code_5, code_6])
    delta = np.diff(rt_raw, axis=1, prepend=np.nan)
    delta[~valid_mask] = np.nan
    delta_mean = np.nanmean(delta)
    delta_std = np.nanstd(delta) + 1e-6
    delta_norm = (np.nan_to_num(delta, nan=delta_mean) - delta_mean) / delta_std
    channels.append(delta_norm.astype(np.float32))
    trial_idx = np.arange(max_length, dtype=np.float32)[None, :] / (max_length - 1 + 1e-6)
    trial_idx = np.tile(trial_idx, (n_samples, 1))
    trial_idx[~valid_mask] = 0.0
    channels.append(trial_idx)
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    seq_array = np.stack(channels, axis=-1)  # (N, L, 10)
    return seq_array

def build_sequence_B5(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """B5 검사 멀티채널 시퀀스 구성 (7채널)"""
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    required_cols = ["B5-1", "B5-2"]
    if not all(c in df.columns for c in required_cols):
        return None
    n_samples = len(df)
    channels = []
    rt_raw = parse_sequence_to_array(df["B5-2"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(rt_raw)
    rt_for_stat = rt_raw.copy()
    rt_for_stat[~valid_mask] = np.nan
    rt_mean = np.nanmean(rt_for_stat)
    rt_std = np.nanstd(rt_for_stat) + 1e-6
    rt_norm = (np.nan_to_num(rt_raw, nan=rt_mean) - rt_mean) / rt_std
    channels.append(rt_norm.astype(np.float32))
    correct_raw = parse_sequence_to_array(df["B5-1"], max_length=max_length, dtype=float)
    is_correct = (correct_raw == 1).astype(np.float32)
    channels.append(is_correct)
    delta = np.diff(rt_raw, axis=1, prepend=np.nan)
    delta[~valid_mask] = np.nan
    delta_mean = np.nanmean(delta)
    delta_std = np.nanstd(delta) + 1e-6
    delta_norm = (np.nan_to_num(delta, nan=delta_mean) - delta_mean) / delta_std
    channels.append(delta_norm.astype(np.float32))
    incorrect = (correct_raw == 2).astype(np.float32)
    cumsum = np.cumsum(incorrect, axis=1)
    t_idx = np.arange(max_length, dtype=np.float32)[None, :] + 1
    running_error = cumsum / t_idx
    running_error[~valid_mask] = 0.0
    channels.append(running_error.astype(np.float32))
    trial_idx = np.arange(max_length, dtype=np.float32)[None, :] / (max_length - 1 + 1e-6)
    trial_idx = np.tile(trial_idx, (n_samples, 1))
    trial_idx[~valid_mask] = 0.0
    channels.append(trial_idx)
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    seq_array = np.stack(channels, axis=-1)  # (N, L, 7)
    return seq_array

def build_sequences_A_for_cnn(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """A검사용 CNN 입력 시퀀스 구성"""
    max_len = CNN_CONFIG["max_seq_length"]
    sequences = {}
    seq_A1 = build_sequence_A1(df, max_length=max_len)
    if seq_A1 is not None:
        sequences["A1"] = seq_A1
        print(f"  [A] A1 seq shape = {seq_A1.shape}  # (N, L, 9)")
    seq_A2 = build_sequence_A2(df, max_length=max_len)
    if seq_A2 is not None:
        sequences["A2"] = seq_A2
        print(f"  [A] A2 seq shape = {seq_A2.shape}  # (N, L, 11)")
    seq_A3 = build_sequence_A3(df, max_length=max_len)
    if seq_A3 is not None:
        sequences["A3"] = seq_A3
        print(f"  [A] A3 seq shape = {seq_A3.shape}  # (N, L, 16)")
    seq_A4 = build_sequence_A4(df, max_length=max_len)
    if seq_A4 is not None:
        sequences["A4"] = seq_A4
        print(f"  [A] A4 seq shape = {seq_A4.shape}  # (N, L, 12)")
    seq_A5 = build_sequence_A5(df, max_length=max_len)
    if seq_A5 is not None:
        sequences["A5"] = seq_A5
        print(f"  [A] A5 seq shape = {seq_A5.shape}  # (N, L, 10)")
    return sequences

def build_sequences_B_for_cnn(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """B검사용 CNN 입력 시퀀스 구성"""
    max_len = CNN_CONFIG["max_seq_length"]
    sequences = {}
    seq_B1 = build_sequence_B1(df, max_length=max_len)
    if seq_B1 is not None:
        sequences["B1"] = seq_B1
        print(f"  [B] B1 seq shape = {seq_B1.shape}  # (N, L, 9)")
    seq_B2 = build_sequence_B2(df, max_length=max_len)
    if seq_B2 is not None:
        sequences["B2"] = seq_B2
        print(f"  [B] B2 seq shape = {seq_B2.shape}  # (N, L, 9)")
    seq_B3 = build_sequence_B3(df, max_length=max_len)
    if seq_B3 is not None:
        sequences["B3"] = seq_B3
        print(f"  [B] B3 seq shape = {seq_B3.shape}  # (N, L, 7)")
    seq_B4 = build_sequence_B4(df, max_length=max_len)
    if seq_B4 is not None:
        sequences["B4"] = seq_B4
        print(f"  [B] B4 seq shape = {seq_B4.shape}  # (N, L, 10)")
    seq_B5 = build_sequence_B5(df, max_length=max_len)
    if seq_B5 is not None:
        sequences["B5"] = seq_B5
        print(f"  [B] B5 seq shape = {seq_B5.shape}  # (N, L, 7)")
    return sequences

# =============================================================================
# PyTorch 모델 클래스들 (train_cnn.py에서 복사)
# =============================================================================

class SequenceDataset(Dataset):
    """시퀀스 데이터와 trial-wise features를 결합한 Dataset"""
    def __init__(self, sequences: Dict[str, np.ndarray], trial_features: np.ndarray, labels: np.ndarray = None):
        self.sequences = sequences
        self.trial_features = trial_features.astype(np.float32)
        self.labels = labels.astype(np.float32) if labels is not None else None
        self.n_samples = len(trial_features)
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        seq_dict = {}
        for k, v in self.sequences.items():
            arr = v[idx]  # (L, C) or (L,)
            if arr.ndim == 1:
                arr = arr[None, :]  # (1, L)
            elif arr.ndim == 2:
                arr = arr.T  # (C, L)
            seq_dict[k] = torch.from_numpy(arr.astype(np.float32))
        trial_feat = torch.from_numpy(self.trial_features[idx])
        if self.labels is not None:
            label = torch.FloatTensor([self.labels[idx]])
            return seq_dict, trial_feat, label
        else:
            return seq_dict, trial_feat

class AttentionLayer(nn.Module):
    """간단한 Attention 레이어 (Bahdanau-style)"""
    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
    def forward(self, x):
        u = torch.tanh(self.W(x))
        scores = self.v(u).squeeze(-1)
        attn_weights = torch.softmax(scores, dim=1)
        attn_output = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)
        return attn_output, attn_weights

class MultiSequenceCNN(nn.Module):
    """여러 시퀀스를 처리하는 1D CNN 모델"""
    def __init__(self, config: Dict, num_trial_features: int):
        super().__init__()
        self.config = config
        self.num_trial_features = num_trial_features
        self.sequence_modules = nn.ModuleDict()
        trial_input_dim = num_trial_features
        mlp_layers = []
        prev_dim = trial_input_dim
        for hidden_dim in config["mlp_hidden_dims"]:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(config["mlp_dropout"]))
            prev_dim = hidden_dim
        self.trial_mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        self.final_classifier = None
    def _build_sequence_cnn(self, seq_name: str, input_dim: int = 1) -> nn.Module:
        """단일 시퀀스용 CNN 모듈 생성"""
        conv_outputs = []
        for filter_size in self.config["filter_sizes"]:
            conv_layers = []
            in_channels = input_dim
            for _ in range(self.config["cnn_layers"]):
                conv_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=self.config["num_filters"],
                    kernel_size=filter_size, padding=filter_size // 2))
                conv_layers.append(nn.ReLU())
                conv_layers.append(nn.Dropout(self.config["dropout_rate"]))
                in_channels = self.config["num_filters"]
            conv_outputs.append(nn.Sequential(*conv_layers))
        convs = nn.ModuleList(conv_outputs)
        if self.config["pooling_type"] == "max":
            pool = nn.AdaptiveMaxPool1d(1)
            pool_type = "max"
        elif self.config["pooling_type"] == "avg":
            pool = nn.AdaptiveAvgPool1d(1)
            pool_type = "avg"
        else:
            pool = None
            pool_type = "both"
        attention = None
        if self.config["use_attention"]:
            attention = AttentionLayer(hidden_dim=self.config["num_filters"] * len(self.config["filter_sizes"]),
                attention_dim=self.config["attention_dim"])
        class SequenceCNNModule(nn.Module):
            def __init__(self, convs, pool, pool_type, attention):
                super().__init__()
                self.convs = convs
                self.pool = pool
                self.pool_type = pool_type
                if pool_type == "both":
                    self.pool_max = nn.AdaptiveMaxPool1d(1)
                    self.pool_avg = nn.AdaptiveAvgPool1d(1)
                self.attention = attention
            def forward(self, x):
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                conv_outputs = []
                for conv in self.convs:
                    conv_out = conv(x)
                    conv_outputs.append(conv_out)
                if len(conv_outputs) > 1:
                    combined = torch.cat(conv_outputs, dim=1)
                else:
                    combined = conv_outputs[0]
                if self.attention is not None:
                    combined_t = combined.transpose(1, 2)
                    attn_out, _ = self.attention(combined_t)
                    return attn_out
                else:
                    if self.pool_type == "both":
                        pooled_max = self.pool_max(combined).squeeze(-1)
                        pooled_avg = self.pool_avg(combined).squeeze(-1)
                        pooled = torch.cat([pooled_max, pooled_avg], dim=1)
                    else:
                        pooled = self.pool(combined).squeeze(-1)
                    return pooled
        return SequenceCNNModule(convs, pool, pool_type, attention)
    def forward(self, sequences: Dict[str, torch.Tensor], trial_features: torch.Tensor):
        seq_outputs = []
        for seq_name, seq_data in sequences.items():
            if seq_name in self.sequence_modules:
                seq_out = self.sequence_modules[seq_name](seq_data)
                seq_outputs.append(seq_out)
        if seq_outputs:
            cnn_features = torch.cat(seq_outputs, dim=1)
        else:
            cnn_features = torch.zeros(trial_features.size(0), 1, device=trial_features.device)
        trial_out = self.trial_mlp(trial_features)
        combined = torch.cat([cnn_features, trial_out], dim=1)
        logits = self.final_classifier(combined)
        return torch.sigmoid(logits)

# =============================================================================
# 모델 로딩용 헬퍼 (단순화된 버전)
# =============================================================================
def build_cnn_model_for_inference(
    model_bundle: dict,
    num_trial_features: int,
    sequences: dict,
    device: torch.device,
) -> MultiSequenceCNN:
    """
    저장된 model_bundle(학습 시 pickle)과
    trial feature dimension, sequences를 기반으로
    MultiSequenceCNN 아키텍처를 복원하고 state_dict를 로드.
    
    사용자 제시 방식:
    bundle = pickle.load(f)
    model = MultiSequenceCNN(bundle["config"], num_trial_features=??)
    model.sequence_modules[...] = ...
    model.final_classifier = ...
    model.load_state_dict(bundle["model"], strict=True)
    model.to("cpu")
    """
    # bundle에서 config 추출 (cnn_config 또는 config 키 지원)
    config = model_bundle.get("cnn_config") or model_bundle.get("config")
    if config is None:
        raise ValueError("model_bundle에 'config' 또는 'cnn_config' 키가 없습니다.")
    sequence_names = model_bundle["sequence_names"]
    state_dict = model_bundle["model"]

    # sequences dict에서 실제 사용할 시퀀스만 추출 (훈련 시 사용한 순서 기준)
    seq_used = {name: sequences[name] for name in sequence_names if name in sequences}
    
    # 학습 시 사용한 모든 시퀀스가 테스트에도 있어야 함
    if len(seq_used) != len(sequence_names):
        missing = set(sequence_names) - set(seq_used.keys())
        raise ValueError(
            f"Missing sequences in test data: {missing}. "
            f"Expected {len(sequence_names)} sequences, got {len(seq_used)}."
        )

    # 모델 본체 생성
    model = MultiSequenceCNN(config, num_trial_features=num_trial_features)

    # 시퀀스별 CNN 모듈 구성 (학습 때와 동일하게)
    # 주의: 학습 시와 동일한 순서와 채널 수로 생성해야 함
    for seq_name in sequence_names:
        if seq_name not in seq_used:
            raise ValueError(
                f"시퀀스 '{seq_name}'가 테스트 데이터에 없습니다. "
                f"필요한 시퀀스: {sequence_names}, 사용 가능한 시퀀스: {list(seq_used.keys())}"
            )
        seq_data = seq_used[seq_name]
        if seq_data.ndim == 3:
            input_dim = seq_data.shape[2]  # num_channels
        else:
            input_dim = 1
        model.sequence_modules[seq_name] = model._build_sequence_cnn(
            seq_name, input_dim=input_dim
        )

    # CNN 출력 차원 계산 (학습 스크립트 train_cnn_model과 동일 로직)
    num_filters = config["num_filters"]
    num_filter_sizes = len(config["filter_sizes"])

    if config["use_attention"]:
        seq_output_dim = num_filters * num_filter_sizes
    else:
        if config["pooling_type"] == "both":
            seq_output_dim = num_filters * num_filter_sizes * 2
        else:
            seq_output_dim = num_filters * num_filter_sizes

    cnn_output_dim = seq_output_dim * len(sequence_names)
    mlp_output_dim = (
        config["mlp_hidden_dims"][-1]
        if config["mlp_hidden_dims"]
        else num_trial_features
    )
    final_input_dim = cnn_output_dim + mlp_output_dim

    model.final_classifier = nn.Sequential(
        nn.Linear(final_input_dim, 64),
        nn.ReLU(),
        nn.Dropout(config["mlp_dropout"]),
        nn.Linear(64, 1),
    )

    # state_dict 로드 (CPU) - 사용자 제시 방식으로 단순화
    # 주의: strict=True이므로 모델 구조가 정확히 일치해야 함
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        # 디버깅을 위한 상세 에러 메시지
        model_keys = set(model.state_dict().keys())
        state_keys = set(state_dict.keys())
        missing_keys = model_keys - state_keys
        unexpected_keys = state_keys - model_keys
        error_msg = (
            f"State dict 로딩 실패:\n"
            f"  - 모델 키 수: {len(model_keys)}\n"
            f"  - 저장된 키 수: {len(state_keys)}\n"
            f"  - 누락된 키: {missing_keys if missing_keys else '없음'}\n"
            f"  - 예상치 못한 키: {unexpected_keys if unexpected_keys else '없음'}\n"
            f"  - sequence_names: {sequence_names}\n"
            f"  - num_trial_features: {num_trial_features}\n"
            f"  - final_input_dim: {final_input_dim}\n"
            f"원본 에러: {str(e)}"
        )
        raise RuntimeError(error_msg) from e
    
    model.to(device)
    model.eval()

    return model

# =============================================================================
# 라이트 피처 선택 함수들 (train_cnn.py에서 복사)
# =============================================================================

def select_light_features_A(df_feat: pd.DataFrame) -> pd.DataFrame:
    """A 검사용 CNN 라이트 피처 선택"""
    keep_cols = [
        "Test_id",
        "Age_num",
        "PerceptualSpeed_A",
        "CognitiveAbility_A",
        "EmotionalRisk_A",
    ]
    optional_cols = ["Age_z", "Age_bin", "Old_and_low_A4", "A_low_cog_flag"]
    for col in optional_cols:
        if col in df_feat.columns:
            keep_cols.append(col)
    keep_cols = [c for c in keep_cols if c in df_feat.columns]
    return df_feat[keep_cols].copy()

def select_light_features_B(df_feat: pd.DataFrame) -> pd.DataFrame:
    """B 검사용 CNN 라이트 피처 선택"""
    keep_cols = [
        "Test_id",
        "Age_num",
        "MultitaskAbility_B",
        "RiskScore_B_norm",
    ]
    optional_cols = ["Age_z", "Age_bin"]
    for col in optional_cols:
        if col in df_feat.columns:
            keep_cols.append(col)
    keep_cols = [c for c in keep_cols if c in df_feat.columns]
    return df_feat[keep_cols].copy()


def predict_cnn_model(
    model: MultiSequenceCNN,
    sequences: dict,
    trial_features: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """
    학습 스크립트의 predict_cnn_model과 동일 패턴
    (단, 기본 batch_size를 CPU 기준으로 조금 키웠음)
    """
    dataset = SequenceDataset(sequences, trial_features, labels=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    preds = []
    with torch.no_grad():
        for seq_dict, trial_feat in loader:
            seq_dict = {k: v.to(device) for k, v in seq_dict.items()}
            trial_feat = trial_feat.to(device)
            output = model(seq_dict, trial_feat)
            preds.append(output.cpu().numpy())

    return np.concatenate(preds, axis=0).flatten()

# =============================================================================
# CNN 추론 함수 (infer_partition_A, infer_partition_B)
# =============================================================================
# =============================================================================
# 전처리기 유틸 함수들 (utils.py에서 복사)
# =============================================================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn import __version__ as sklver

def separate_num_cat(df: pd.DataFrame, drop_cols: List[str]) -> Tuple[List[str], List[str]]:
    """수치형/범주형 컬럼 분리"""
    cols = [c for c in df.columns if c not in drop_cols]
    cat_cols = [c for c in cols if str(df[c].dtype) in ("object", "category")]
    num_cols = [c for c in cols if c not in cat_cols]
    return num_cols, cat_cols

def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """전처리기 빌드 (학습 시와 동일)"""
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

# =============================================================================
# Feature Block 적용 함수 (사용하지 않지만 참고용으로 유지)
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
# ---------------------------------------------------------------------
# A/B 파티션별 추론
# ---------------------------------------------------------------------
def infer_partition_A_cnn(
    A_test_raw: pd.DataFrame,
    A_test_feat_full: pd.DataFrame,
    test_idx: pd.DataFrame,
    device: torch.device,
    seed: int = 42,
):
    """
    A 테스트셋에 대해 CNN 추론 (seed 설정 가능)
    - FE + light feature 선택
    - 멀티채널 시퀀스 빌드
    - 저장된 preproc + 모델 로드
    - CPU에서 추론 (seed 적용)
    """
    which = "A"
    key = "Test_id"
    
    # Seed 설정
    set_seed(seed)

    # PrimaryKey 추가 (history features를 위해 필요)
    A_test_raw_copy = A_test_raw.copy()
    if "PrimaryKey" not in A_test_raw_copy.columns and "Test_id" in A_test_raw_copy.columns:
        A_test_raw_copy["PrimaryKey"] = A_test_raw_copy["Test_id"].str.split("_A_").str[0].str.split("_B_").str[0]

    # FE (학습과 동일 파이프라인)
    print(f"\n🔵 [A] CNN FE pipeline (test, seed={seed}) ...")
    t0 = time.time()
    A_test_feat = preprocess_A(A_test_raw_copy)
    A_test_feat = add_features_A(A_test_feat)
    A_test_feat = add_age_normed_composites_A(A_test_feat)
    A_test_feat = add_history_features_A(A_test_feat)
    print(f"  [A] FE done: {A_test_feat.shape}, elapsed {(time.time()-t0)/60:.2f} min")

    # CNN용 라이트 피처만 선택
    A_test_feat_cnn = select_light_features_A(A_test_feat)
    print(f"  [A] light features shape: {A_test_feat_cnn.shape}")

    # 인덱스 정렬
    A_test_idx = test_idx[test_idx["Test"] == "A"].copy()

    # 원시 데이터 정렬 (시퀀스용)
    A_test_raw_aligned = A_test_idx.merge(A_test_raw_copy, on=key, how="left", validate="1:1")

    # FE/인덱스 merge (trial features용)
    df_feat = A_test_idx.merge(A_test_feat_cnn, on=key, how="left", validate="1:1")

    # row-wise 특성 추가 (학습과 동일)
    drop_cols = [key]
    if "Label" in df_feat.columns:
        drop_cols.append("Label")
    if "Test" in df_feat.columns:
        drop_cols.append("Test")

    feature_cols = [c for c in df_feat.columns if c not in drop_cols]
    df_feat = add_rowwise_features(df_feat, feature_cols)

    # 멀티채널 시퀀스 구성
    print("[A] Building multi-channel sequences for CNN (test)...")
    sequences = build_sequences_A_for_cnn(A_test_raw_aligned)
    if not sequences:
        raise ValueError("[A] No sequence columns found for CNN in test set.")

    # 시퀀스 원시 컬럼 제거 (멀티채널로 변환했으므로, 학습과 동일하게)
    seq_raw_cols = []
    # A1
    if "A1-1" in df_feat.columns:
        seq_raw_cols.extend(["A1-1", "A1-2", "A1-3", "A1-4"])
    # A2
    if "A2-1" in df_feat.columns:
        seq_raw_cols.extend(["A2-1", "A2-2", "A2-3", "A2-4"])
    # A3
    if "A3-1" in df_feat.columns:
        seq_raw_cols.extend(["A3-1", "A3-2", "A3-3", "A3-4", "A3-5", "A3-6", "A3-7"])
    # A4
    if "A4-1" in df_feat.columns:
        seq_raw_cols.extend(["A4-1", "A4-2", "A4-3", "A4-4", "A4-5"])
    # A5
    if "A5-1" in df_feat.columns:
        seq_raw_cols.extend(["A5-1", "A5-2", "A5-3"])

    # trial_df: Test_id/Label/Test + 시퀀스 원시 컬럼 제거
    trial_df = df_feat.drop(columns=drop_cols + seq_raw_cols, errors="ignore")

    # 전처리기 & 모델 번들 로드
    with open(A_PREPROC_PATH_CNN, "rb") as f:
        preproc = pickle.load(f)
    with open(A_MODEL_PATH_CNN, "rb") as f:
        bundle = pickle.load(f)

    # trial features 변환
    X_trial = preproc.transform(trial_df)
    if not isinstance(X_trial, np.ndarray):
        X_trial = X_trial.toarray()
    num_trial_features = X_trial.shape[1]

    # 안전장치: 전처리기 피처 수 확인
    if hasattr(preproc, 'n_features_out_'):
        expected_features = preproc.n_features_out_
        if num_trial_features != expected_features:
            print(f"⚠️  [A] 경고: 전처리기 예상 피처 수({expected_features})와 실제 변환 결과({num_trial_features})가 다릅니다.")

    # 모델 아키텍처 복원 + state_dict 로드
    model = build_cnn_model_for_inference(
        model_bundle=bundle,
        num_trial_features=num_trial_features,
        sequences=sequences,
        device=device,
    )

    print(f"[A] Running CNN inference (seed={seed}) ...")
    t1 = time.time()
    preds = predict_cnn_model(
        model=model,
        sequences=sequences,
        trial_features=X_trial.astype(np.float32),
        device=device,
        batch_size=CNN_CONFIG.get("batch_size", 32) * 2,
    )
    dt = time.time() - t1
    print(f"[A] CNN inference done: {len(preds)} samples, elapsed {dt:.2f} sec")

    out_df = A_test_idx[[key]].copy()
    out_df["pred_cnn"] = preds
    return out_df


def infer_partition_B_cnn(
    B_test_raw: pd.DataFrame,
    B_test_feat_full: pd.DataFrame,
    test_idx: pd.DataFrame,
    device: torch.device,
    seed: int = 42,
):
    """
    B 테스트셋에 대해 CNN 추론 (seed 설정 가능)
    - FE + light feature 선택
    - 멀티채널 시퀀스 빌드
    - 저장된 preproc + 모델 로드
    - CPU에서 추론 (seed 적용)
    """
    # Seed 설정
    set_seed(seed)
    which = "B"
    key = "Test_id"

    # PrimaryKey 추가 (history features를 위해 필요)
    B_test_raw_copy = B_test_raw.copy()
    if "PrimaryKey" not in B_test_raw_copy.columns and "Test_id" in B_test_raw_copy.columns:
        B_test_raw_copy["PrimaryKey"] = B_test_raw_copy["Test_id"].str.split("_A_").str[0].str.split("_B_").str[0]

    print(f"\n🟢 [B] CNN FE pipeline (test, seed={seed}) ...")
    t0 = time.time()
    B_test_feat = preprocess_B(B_test_raw_copy)
    B_test_feat = add_features_B(B_test_feat)
    B_test_feat = add_age_normed_composites_B(B_test_feat)
    B_test_feat = add_history_features_B(B_test_feat)
    print(f"  [B] FE done: {B_test_feat.shape}, elapsed {(time.time()-t0)/60:.2f} min")

    # CNN용 라이트 피처
    B_test_feat_cnn = select_light_features_B(B_test_feat)
    print(f"  [B] light features shape: {B_test_feat_cnn.shape}")

    # 인덱스 정렬
    B_test_idx = test_idx[test_idx["Test"] == "B"].copy()

    # 원시 데이터 정렬 (시퀀스용)
    B_test_raw_aligned = B_test_idx.merge(B_test_raw_copy, on=key, how="left", validate="1:1")

    # FE/인덱스 merge
    df_feat = B_test_idx.merge(B_test_feat_cnn, on=key, how="left", validate="1:1")

    drop_cols = [key]
    if "Label" in df_feat.columns:
        drop_cols.append("Label")
    if "Test" in df_feat.columns:
        drop_cols.append("Test")

    feature_cols = [c for c in df_feat.columns if c not in drop_cols]
    df_feat = add_rowwise_features(df_feat, feature_cols)

    # 멀티채널 시퀀스 구성
    print("[B] Building multi-channel sequences for CNN (test)...")
    sequences = build_sequences_B_for_cnn(B_test_raw_aligned)
    if not sequences:
        raise ValueError("[B] No sequence columns found for CNN in test set.")

    # 시퀀스 원시 컬럼 제거 (멀티채널로 변환했으므로, 학습과 동일하게)
    seq_raw_cols = []
    # B1
    if "B1-1" in df_feat.columns:
        seq_raw_cols.extend(["B1-1", "B1-2", "B1-3"])
    # B2
    if "B2-1" in df_feat.columns:
        seq_raw_cols.extend(["B2-1", "B2-2", "B2-3"])
    # B3
    if "B3-1" in df_feat.columns:
        seq_raw_cols.extend(["B3-1", "B3-2"])
    # B4
    if "B4-1" in df_feat.columns:
        seq_raw_cols.extend(["B4-1", "B4-2"])
    # B5
    if "B5-1" in df_feat.columns:
        seq_raw_cols.extend(["B5-1", "B5-2"])
    # B9/B10은 집계형이므로 정적 피처로 유지 (제거하지 않음)

    # trial_df: Test_id/Label/Test + 시퀀스 원시 컬럼 제거
    trial_df = df_feat.drop(columns=drop_cols + seq_raw_cols, errors="ignore")

    # 전처리기 & 모델 번들 로드
    with open(B_PREPROC_PATH_CNN, "rb") as f:
        preproc = pickle.load(f)
    with open(B_MODEL_PATH_CNN, "rb") as f:
        bundle = pickle.load(f)

    # trial features 변환
    X_trial = preproc.transform(trial_df)
    if not isinstance(X_trial, np.ndarray):
        X_trial = X_trial.toarray()
    num_trial_features = X_trial.shape[1]

    # 안전장치: 전처리기 피처 수 확인
    if hasattr(preproc, 'n_features_out_'):
        expected_features = preproc.n_features_out_
        if num_trial_features != expected_features:
            print(f"⚠️  [B] 경고: 전처리기 예상 피처 수({expected_features})와 실제 변환 결과({num_trial_features})가 다릅니다.")

    # 모델 아키텍처 복원 + state_dict 로드 (사용자 제시 방식)
    model = build_cnn_model_for_inference(
        model_bundle=bundle,
        num_trial_features=num_trial_features,
        sequences=sequences,
        device=device,
    )

    print(f"[B] Running CNN inference (seed={seed}) ...")
    t1 = time.time()
    preds = predict_cnn_model(
        model=model,
        sequences=sequences,
        trial_features=X_trial.astype(np.float32),
        device=device,
        batch_size=CNN_CONFIG.get("batch_size", 32) * 2,
    )
    dt = time.time() - t1
    print(f"[B] CNN inference done: {len(preds)} samples, elapsed {dt:.2f} sec")

    out_df = B_test_idx[[key]].copy()
    out_df["pred_cnn"] = preds
    return out_df

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

def add_rowwise_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols]
    na_count = X.isna().sum(axis=1).astype(np.int32)
    na_ratio = (na_count / (len(feature_cols) + 1e-9)).astype(np.float32)
    df2 = df.copy()
    df2["NA_COUNT"] = na_count
    df2["NA_RATIO"] = na_ratio
    return df2

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
            print(f"✅ Loaded best hyperparameters from {BEST_PARAMS_PATH}")
            return params
        except Exception as e:
            print(f"⚠️  Failed to load best params: {e}")
            print(f"   Using default hyperparameters...")
            return None
    else:
        print(f"⚠️  Best params file not found: {BEST_PARAMS_PATH}")
        print(f"   Using default hyperparameters...")
        return None

# 하이퍼파라미터 로드
best_params = load_best_params()

if best_params is not None:
    # best08.py에서 찾은 최적 하이퍼파라미터 사용
    BASE_HGB_PARAMS_A = best_params["hgb_params_A"].copy()
    BASE_HGB_PARAMS_B = best_params["hgb_params_B"].copy()
    ENSEMBLE_SEEDS: Sequence[int] = tuple(best_params["ensemble_seeds"])
    USE_CALIBRATION = best_params.get("use_calibration", True)
    CALIB_METHOD = best_params.get("calib_method", "isotonic")
    CALIB_CV = best_params.get("calib_cv", 3)
    RANDOM_STATE = best_params.get("random_state", 42)
    print(f"   Using best hyperparameters from best08.py OOF evaluation")
else:
    # Fallback: 기본 하이퍼파라미터
    RANDOM_STATE = 42
    USE_CALIBRATION = True
    CALIB_METHOD = "isotonic"
    CALIB_CV = 3
    ENSEMBLE_SEEDS: Sequence[int] = (42, 202, 777, 1001, 8888)  # 5개 seed
    BASE_HGB_PARAMS_A = dict(
        learning_rate=0.05,
        max_iter=1500,
        max_depth=None,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.0,
        early_stopping=True,
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
    print(f"   Using default hyperparameters")

# =============================================================================
# HGBM 학습 함수 (NumPy 에러 회피를 위해 스크립트 내에서 학습)
# =============================================================================
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

def build_model(seed: int, which: str, model_type: str = "hgb", for_oof: bool = False):
    """
    모델 생성 (HGBM only)
    """
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
    """
    단일 HGBM 모델 학습 (A용)
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

    # 전체 데이터로 학습
    X_all = df.drop(columns=drop_cols)
    y_all = df[label_col].astype(int).values

    X_all_t = preproc.fit_transform(X_all)

    # 여러 seed 앙상블 학습
    members = []
    for sd in ENSEMBLE_SEEDS:
        base = build_model(sd, which=which).fit(X_all_t, y_all)
        mdl = maybe_calibrate(base, X_all_t, y_all)
        members.append(mdl)
    ensemble = AvgProbaEnsemble(members)

    return preproc, ensemble

def fit_partition_B_submodels(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    label_col: str = "Label",
):
    """
    B용: n_tests_so_far 기준으로 first(==1), repeat(>1) 두 개의 앙상블을 학습.
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

    # 전처리 학습은 전체 train에 대해 1번만
    X_tr_t = preproc.fit_transform(X_all)

    # first / repeat 마스크
    mask_first_tr = (n_tests_all == 1)
    mask_repeat_tr = (n_tests_all > 1)

    # 혹시라도 first 또는 repeat 쪽에 데이터가 거의 없으면 방어
    if mask_first_tr.sum() == 0:
        print(f"[{which}] WARNING: no samples for n_tests_so_far == 1, fallback to single model.")
        members = []
        for sd in ENSEMBLE_SEEDS:
            base_hgb = build_model(sd, which=which, model_type="hgb", for_oof=False)
            base_hgb.fit(X_tr_t, y_all)
            mdl_hgb = maybe_calibrate(base_hgb, X_tr_t, y_all)
            members.append(mdl_hgb)
        ensemble = AvgProbaEnsemble(members)
        return preproc, ensemble, None

    if mask_repeat_tr.sum() == 0:
        print(f"[{which}] WARNING: no samples for n_tests_so_far > 1, fallback to single model.")
        members = []
        for sd in ENSEMBLE_SEEDS:
            base_hgb = build_model(sd, which=which, model_type="hgb", for_oof=False)
            base_hgb.fit(X_tr_t, y_all)
            mdl_hgb = maybe_calibrate(base_hgb, X_tr_t, y_all)
            members.append(mdl_hgb)
        ensemble = AvgProbaEnsemble(members)
        return preproc, ensemble, None

    # ---------- first 앙상블 (n_tests_so_far == 1) ----------
    X_tr_first = X_tr_t[mask_first_tr]
    y_tr_first = y_all[mask_first_tr]

    members_first = []
    for sd in ENSEMBLE_SEEDS:
        base_f_hgb = build_model(sd, which=which, model_type="hgb", for_oof=False)
        base_f_hgb.fit(X_tr_first, y_tr_first)
        mdl_f_hgb = maybe_calibrate(base_f_hgb, X_tr_first, y_tr_first)
        members_first.append(mdl_f_hgb)
    ensemble_first = AvgProbaEnsemble(members_first)

    # ---------- repeat 앙상블 (n_tests_so_far > 1) ----------
    X_tr_repeat = X_tr_t[mask_repeat_tr]
    y_tr_repeat = y_all[mask_repeat_tr]

    members_repeat = []
    for sd in ENSEMBLE_SEEDS:
        base_r_hgb = build_model(sd, which=which, model_type="hgb", for_oof=False)
        base_r_hgb.fit(X_tr_repeat, y_tr_repeat)
        mdl_r_hgb = maybe_calibrate(base_r_hgb, X_tr_repeat, y_tr_repeat)
        members_repeat.append(mdl_r_hgb)
    ensemble_repeat = AvgProbaEnsemble(members_repeat)

    return preproc, ensemble_first, ensemble_repeat

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
    
    # Test_id 기준으로 merge
    key = answer_df.columns[0]  # 보통 "Test_id"
    merged = answer_df.merge(submission_df, on=key, how="inner", suffixes=("_true", "_pred"))
    
    if len(merged) == 0:
        raise ValueError("No matching Test_id between answer and submission dataframes.")
    
    submission_df = merged[[key] + [c for c in merged.columns if c.endswith("_pred")]]
    answer_df = merged[[key] + [c for c in merged.columns if c.endswith("_true")]]
    
    # 컬럼명 정리
    submission_df.columns = [key] + [c.replace("_pred", "") for c in submission_df.columns[1:]]
    answer_df.columns = [key] + [c.replace("_true", "") for c in answer_df.columns[1:]]
    
    submission_df.index = range(submission_df.shape[0])
    answer_df.index = range(answer_df.shape[0])
    
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
# 예측 함수
# =============================================================================

def predict_ensemble_A(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    preproc_hgb,
    ensemble,
) -> pd.DataFrame:
    """
    A용 HGBM 예측 (단일 앙상블 모델)
    """
    key = "Test_id"
    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")
    drop_cols = [key] + (["Test"] if "Test" in df.columns else [])

    feature_cols = [c for c in df.columns if c not in drop_cols]
    df = add_rowwise_features(df, feature_cols)

    X = df.drop(columns=drop_cols, errors="ignore")
    X_t_hgb = preproc_hgb.transform(X)
    
    proba = ensemble.predict_proba(X_t_hgb)[:, 1]
    proba = np.clip(proba, 1e-7, 1 - 1e-7)

    out = df_idx[[key]].copy()
    out["Label"] = proba
    return out

def predict_ensemble_B(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    preproc_hgb,
    ensemble_first,
    ensemble_repeat,
) -> pd.DataFrame:
    """
    B용 HGBM 예측 (history-split: first/repeat 두 개의 앙상블)
    """
    key = "Test_id"
    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")
    drop_cols = [key] + (["Test"] if "Test" in df.columns else [])

    feature_cols = [c for c in df.columns if c not in drop_cols]
    df = add_rowwise_features(df, feature_cols)

    X = df.drop(columns=drop_cols, errors="ignore")

    if "n_tests_so_far" not in X.columns:
        raise ValueError("[B] n_tests_so_far column is required for history-based split.")

    n_tests = X["n_tests_so_far"].values
    mask_first = (n_tests == 1)
    mask_repeat = (n_tests > 1)

    X_t_hgb = preproc_hgb.transform(X)
    proba = np.zeros(len(n_tests), dtype=float)
    
    if ensemble_repeat is None:
        # single model fallback
        proba = ensemble_first.predict_proba(X_t_hgb)[:, 1]
    else:
        # first와 repeat 각각 예측
        if mask_first.sum() > 0:
            proba[mask_first] = ensemble_first.predict_proba(X_t_hgb[mask_first])[:, 1]
        if mask_repeat.sum() > 0:
            proba[mask_repeat] = ensemble_repeat.predict_proba(X_t_hgb[mask_repeat])[:, 1]

    proba = np.clip(proba, 1e-7, 1 - 1e-7)

    out = df_idx[[key]].copy()
    out["Label"] = proba
    return out

# =============================================================================
# 메인
# =============================================================================

def main():
    """
    메인 함수: HGBM + CNN 앙상블 추론
    - HGBM 모델 로드 또는 학습 (NumPy 에러 회피를 위해 스크립트 내에서 학습)
    - CNN 모델 로드 (train_cnn.py로 학습된 모델)
    - HGBM과 CNN 예측을 weighted average로 결합
    """
    import torch
    
    t0 = time.time()
    ensure_dirs()
    set_seed(42)  # 기본 seed
    
    # 디바이스 설정 (CPU 강제 사용)
    device = torch.device("cpu")
    print(f"🔧 Using device: {device} (CPU forced)")
    
    print("📂 Loading HGBM models...")
    # HGBM 모델 로드 시도 (hgb_A.pkl, hgb_B.pkl 형식)
    preproc_A_hgb = None
    ensemble_A = None
    preproc_B_hgb = None
    ensemble_B_first = None
    ensemble_B_repeat = None
    
    try:
        # A 모델 로드 시도
        if os.path.exists(A_MODEL_PATH_HGB) and os.path.exists(A_PREPROC_PATH_HGB):
            with open(A_PREPROC_PATH_HGB, "rb") as f:
                preproc_A_hgb = pickle.load(f)
            with open(A_MODEL_PATH_HGB, "rb") as f:
                model_A = pickle.load(f)
                # model_A가 딕셔너리인 경우 ensemble 키에서 추출
                if isinstance(model_A, dict):
                    if "ensemble" in model_A:
                        # script08.py 형식: {"ensemble": AvgProbaEnsemble, "mode": "single"}
                        ensemble_A = model_A["ensemble"]
                    elif "model_hgb" in model_A and isinstance(model_A["model_hgb"], dict):
                        # 중첩된 model_bundle 형태인 경우
                        model_hgb = model_A["model_hgb"]
                        if model_hgb.get("mode") == "single" and "ensemble" in model_hgb:
                            ensemble_A = model_hgb["ensemble"]
                        else:
                            raise ValueError(f"Unsupported model bundle structure: {model_hgb.keys()}")
                    else:
                        # 지원되지 않는 딕셔너리 구조
                        print(f"⚠️  Model A structure: {list(model_A.keys())}")
                        raise ValueError(f"Unsupported model structure: {list(model_A.keys())}. Expected 'ensemble' key.")
                else:
                    # 직접 AvgProbaEnsemble 객체인 경우
                    ensemble_A = model_A
                
                # ensemble_A가 제대로 로드되었는지 확인
                if not hasattr(ensemble_A, 'predict_proba'):
                    raise TypeError(f"Loaded ensemble_A is not a valid model object. Type: {type(ensemble_A)}")
                    
            print(f"✅ HGBM A model loaded from {A_MODEL_PATH_HGB} (type: {type(ensemble_A).__name__})")
        else:
            print(f"⚠️  HGBM A model files not found: {A_MODEL_PATH_HGB}")
        
        # B 모델 로드 시도
        if os.path.exists(B_MODEL_PATH_HGB) and os.path.exists(B_PREPROC_PATH_HGB):
            with open(B_PREPROC_PATH_HGB, "rb") as f:
                preproc_B_hgb = pickle.load(f)
            with open(B_MODEL_PATH_HGB, "rb") as f:
                model_B = pickle.load(f)
                # model_B가 딕셔너리인 경우
                if isinstance(model_B, dict):
                    if "mode" in model_B and model_B["mode"] == "history_split":
                        # script08.py 형식: {"mode": "history_split", "ensemble_first": ..., "ensemble_repeat": ...}
                        ensemble_B_first = model_B.get("ensemble_first")
                        ensemble_B_repeat = model_B.get("ensemble_repeat")
                    elif "ensemble_first" in model_B and "ensemble_repeat" in model_B:
                        ensemble_B_first = model_B["ensemble_first"]
                        ensemble_B_repeat = model_B["ensemble_repeat"]
                    elif "model_hgb" in model_B and isinstance(model_B["model_hgb"], dict):
                        # 중첩된 model_bundle 형태인 경우
                        model_hgb = model_B["model_hgb"]
                        if model_hgb.get("mode") == "history_split":
                            ensemble_B_first = model_hgb.get("ensemble_first")
                            ensemble_B_repeat = model_hgb.get("ensemble_repeat")
                        elif "ensemble" in model_hgb:
                            ensemble_B_first = model_hgb["ensemble"]
                            ensemble_B_repeat = None
                        else:
                            raise ValueError(f"Unsupported model bundle structure: {list(model_hgb.keys())}")
                    elif "ensemble" in model_B:
                        ensemble_B_first = model_B["ensemble"]
                        ensemble_B_repeat = None
                    else:
                        # 지원되지 않는 딕셔너리 구조
                        print(f"⚠️  Model B structure: {list(model_B.keys())}")
                        raise ValueError(f"Unsupported model structure: {list(model_B.keys())}")
                elif isinstance(model_B, tuple) and len(model_B) == 2:
                    # (ensemble_first, ensemble_repeat) 튜플인 경우
                    ensemble_B_first, ensemble_B_repeat = model_B
                else:
                    # 직접 AvgProbaEnsemble 객체인 경우
                    ensemble_B_first = model_B
                    ensemble_B_repeat = None
                
                # ensemble_B_first가 제대로 로드되었는지 확인
                if ensemble_B_first is not None and not hasattr(ensemble_B_first, 'predict_proba'):
                    raise TypeError(f"Loaded ensemble_B_first is not a valid model object. Type: {type(ensemble_B_first)}")
                if ensemble_B_repeat is not None and not hasattr(ensemble_B_repeat, 'predict_proba'):
                    raise TypeError(f"Loaded ensemble_B_repeat is not a valid model object. Type: {type(ensemble_B_repeat)}")
                    
            print(f"✅ HGBM B model loaded from {B_MODEL_PATH_HGB} (type: {type(ensemble_B_first).__name__})")
        else:
            print(f"⚠️  HGBM B model files not found: {B_MODEL_PATH_HGB}")
    except Exception as e:
        print(f"⚠️  Failed to load HGBM models: {e}")
        print("   Will train models on-the-fly...")
        import traceback
        traceback.print_exc()
    
    # 모델이 없으면 학습 (NumPy 에러 회피)
    need_train_A = (preproc_A_hgb is None or ensemble_A is None)
    need_train_B = (preproc_B_hgb is None or ensemble_B_first is None)
    
    # Test 데이터 로드 (학습이 필요하면 train+test 함께 로드)
    A_test_raw = None
    B_test_raw = None
    test_idx = None
    
    if need_train_A or need_train_B:
        print("\n📂 Loading train and test data for training...")
        train_idx, test_idx = read_index_files()
        
        if need_train_A:
            print("\n📚 Training HGBM A model (on-the-fly, NumPy error avoided)...")
            A_train_raw, _ = read_raw_feature_files("train")
            if A_test_raw is None:
                A_test_raw, _ = read_raw_feature_files("test")
            
            # PrimaryKey 추가
            if "PrimaryKey" not in A_train_raw.columns and "Test_id" in A_train_raw.columns:
                A_train_raw["PrimaryKey"] = A_train_raw["Test_id"].str.split("_A_").str[0].str.split("_B_").str[0]
            if "PrimaryKey" not in A_test_raw.columns and "Test_id" in A_test_raw.columns:
                A_test_raw["PrimaryKey"] = A_test_raw["Test_id"].str.split("_A_").str[0].str.split("_B_").str[0]
            
            # Train과 Test를 합쳐서 history features 생성
            A_all_raw = pd.concat([A_train_raw, A_test_raw], axis=0, ignore_index=True)
            A_train_ids = set(A_train_raw["Test_id"])
            
            # FE
            A_all = preprocess_A(A_all_raw)
            A_all = add_features_A(A_all)
            A_all = add_age_normed_composites_A(A_all)
            A_all = add_history_features_A(A_all)
            A_all = apply_feature_blocks(A_all, which="A", config=FEATURE_BLOCKS_A)
            
            A_train_feat = A_all[A_all["Test_id"].isin(A_train_ids)].reset_index(drop=True)
            A_train_idx = train_idx[train_idx["Test"] == "A"].copy()
            
            # 학습
            preproc_A_hgb, ensemble_A = fit_partition(A_train_feat, A_train_idx, "Label", which="A")
            print("✅ HGBM A model trained")
        
        if need_train_B:
            print("\n📚 Training HGBM B model (on-the-fly, NumPy error avoided)...")
            if B_test_raw is None:
                _, B_test_raw = read_raw_feature_files("test")
            _, B_train_raw = read_raw_feature_files("train")
            
            # PrimaryKey 추가
            if "PrimaryKey" not in B_train_raw.columns and "Test_id" in B_train_raw.columns:
                B_train_raw["PrimaryKey"] = B_train_raw["Test_id"].str.split("_A_").str[0].str.split("_B_").str[0]
            if "PrimaryKey" not in B_test_raw.columns and "Test_id" in B_test_raw.columns:
                B_test_raw["PrimaryKey"] = B_test_raw["Test_id"].str.split("_A_").str[0].str.split("_B_").str[0]
            
            # Train과 Test를 합쳐서 history features 생성
            B_all_raw = pd.concat([B_train_raw, B_test_raw], axis=0, ignore_index=True)
            B_train_ids = set(B_train_raw["Test_id"])
            
            # FE
            B_all = preprocess_B(B_all_raw)
            B_all = add_features_B(B_all)
            B_all = add_age_normed_composites_B(B_all)
            B_all = add_history_features_B(B_all)
            B_all = apply_feature_blocks(B_all, which="B", config=FEATURE_BLOCKS_B)
            
            B_train_feat = B_all[B_all["Test_id"].isin(B_train_ids)].reset_index(drop=True)
            B_train_idx = train_idx[train_idx["Test"] == "B"].copy()
            
            # 학습
            preproc_B_hgb, ensemble_B_first, ensemble_B_repeat = fit_partition_B_submodels(
                B_train_feat, B_train_idx, "Label"
            )
            print("✅ HGBM B model trained")
    else:
        # 모델이 이미 있으면 test 데이터만 로드
        print("📂 Loading test data...")
        _, test_idx = read_index_files()
        A_test_raw, B_test_raw = read_raw_feature_files("test")
    
    # PrimaryKey 추가 (history features를 위해 필요)
    # Test_id 형식: "0x..._A_202301" 또는 "0x..._B_202305"
    # PrimaryKey는 "_A_" 또는 "_B_" 앞부분 (해시 값)
    if "PrimaryKey" not in A_test_raw.columns and "Test_id" in A_test_raw.columns:
        A_test_raw["PrimaryKey"] = A_test_raw["Test_id"].str.split("_A_").str[0].str.split("_B_").str[0]
    if "PrimaryKey" not in B_test_raw.columns and "Test_id" in B_test_raw.columns:
        B_test_raw["PrimaryKey"] = B_test_raw["Test_id"].str.split("_A_").str[0].str.split("_B_").str[0]
    
    # HGBM용 FE (feature blocks 적용)
    print("🔵 FE for A (HGBM)...")
    A_test_feat_hgb = preprocess_A(A_test_raw)
    A_test_feat_hgb = add_features_A(A_test_feat_hgb)
    A_test_feat_hgb = add_age_normed_composites_A(A_test_feat_hgb)
    A_test_feat_hgb = add_history_features_A(A_test_feat_hgb)
    A_test_feat_hgb = apply_feature_blocks(A_test_feat_hgb, which="A", config=FEATURE_BLOCKS_A)
    
    print("🟢 FE for B (HGBM)...")
    B_test_feat_hgb = preprocess_B(B_test_raw)
    B_test_feat_hgb = add_features_B(B_test_feat_hgb)
    B_test_feat_hgb = add_age_normed_composites_B(B_test_feat_hgb)
    B_test_feat_hgb = add_history_features_B(B_test_feat_hgb)
    B_test_feat_hgb = apply_feature_blocks(B_test_feat_hgb, which="B", config=FEATURE_BLOCKS_B)
    
    A_test_idx = test_idx[test_idx["Test"] == "A"].copy()
    B_test_idx = test_idx[test_idx["Test"] == "B"].copy()
    
    # HGBM 예측
    proba_A_hgb = None
    proba_B_hgb = None
    if preproc_A_hgb is not None and ensemble_A is not None:
        print("🔮 Predicting with HGBM (A)...")
        pred_A_hgb = predict_ensemble_A(
            A_test_feat_hgb, A_test_idx,
            preproc_A_hgb, ensemble_A
        )
        proba_A_hgb = pred_A_hgb["Label"].values
    
    if preproc_B_hgb is not None and ensemble_B_first is not None:
        print("🔮 Predicting with HGBM (B)...")
        pred_B_hgb = predict_ensemble_B(
            B_test_feat_hgb, B_test_idx,
            preproc_B_hgb, ensemble_B_first, ensemble_B_repeat
        )
        proba_B_hgb = pred_B_hgb["Label"].values
    
    # CNN 예측 (여러 seed로 앙상블)
    proba_A_cnn = None
    proba_B_cnn = None
    if HAS_TORCH:
        cnn_seeds = CNN_CONFIG.get("cnn_seeds", [42])
        print(f"🔮 Predicting with CNN (A, seeds={cnn_seeds})...")
        
        cnn_preds_A = []
        for seed in cnn_seeds:
            try:
                # PrimaryKey는 infer_partition_A_cnn 내부에서 추가됨
                pred_cnn = infer_partition_A_cnn(
                    A_test_raw, A_test_feat_hgb, test_idx, device, seed=seed
                )
                cnn_preds_A.append(pred_cnn["pred_cnn"].values)
            except Exception as e:
                print(f"⚠️  CNN prediction failed for seed {seed}: {e}")
        
        if cnn_preds_A:
            proba_A_cnn = np.mean(cnn_preds_A, axis=0)
            print(f"✅ CNN A predictions: {len(cnn_preds_A)} seeds, shape {proba_A_cnn.shape}")
        
        print(f"🔮 Predicting with CNN (B, seeds={cnn_seeds})...")
        cnn_preds_B = []
        for seed in cnn_seeds:
            try:
                # PrimaryKey는 infer_partition_B_cnn 내부에서 추가됨
                pred_cnn = infer_partition_B_cnn(
                    B_test_raw, B_test_feat_hgb, test_idx, device, seed=seed
                )
                cnn_preds_B.append(pred_cnn["pred_cnn"].values)
            except Exception as e:
                print(f"⚠️  CNN prediction failed for seed {seed}: {e}")
        
        if cnn_preds_B:
            proba_B_cnn = np.mean(cnn_preds_B, axis=0)
            print(f"✅ CNN B predictions: {len(cnn_preds_B)} seeds, shape {proba_B_cnn.shape}")
    else:
        print("⚠️  PyTorch not available - CNN predictions skipped")
    
    # 앙상블: HGBM + CNN weighted average
    cnn_weight = CNN_CONFIG.get("cnn_weight", 0.3)
    hgb_weight = 1.0 - cnn_weight
    
    print(f"🔮 Ensemble: HGBM weight={hgb_weight:.2f}, CNN weight={cnn_weight:.2f}")
    
    # A 앙상블
    if proba_A_hgb is not None and proba_A_cnn is not None:
        proba_A = hgb_weight * proba_A_hgb + cnn_weight * proba_A_cnn
        print(f"   [A] HGBM range: [{proba_A_hgb.min():.6f}, {proba_A_hgb.max():.6f}], mean: {proba_A_hgb.mean():.6f}")
        print(f"   [A] CNN range: [{proba_A_cnn.min():.6f}, {proba_A_cnn.max():.6f}], mean: {proba_A_cnn.mean():.6f}")
        print(f"   [A] Ensemble range: [{proba_A.min():.6f}, {proba_A.max():.6f}], mean: {proba_A.mean():.6f}")
    elif proba_A_hgb is not None:
        proba_A = proba_A_hgb
        print("⚠️  Using HGBM only for A (CNN not available)")
        print(f"   [A] HGBM range: [{proba_A_hgb.min():.6f}, {proba_A_hgb.max():.6f}], mean: {proba_A_hgb.mean():.6f}")
    elif proba_A_cnn is not None:
        proba_A = proba_A_cnn
        print("⚠️  Using CNN only for A (HGBM not available)")
        print(f"   [A] CNN range: [{proba_A_cnn.min():.6f}, {proba_A_cnn.max():.6f}], mean: {proba_A_cnn.mean():.6f}")
    else:
        raise ValueError("Neither HGBM nor CNN predictions available for A")
    
    proba_A = np.clip(proba_A, 1e-7, 1 - 1e-7)
    pred_A = pd.DataFrame({"Test_id": A_test_idx["Test_id"].values, "Label": proba_A})
    
    # B 앙상블
    if proba_B_hgb is not None and proba_B_cnn is not None:
        proba_B = hgb_weight * proba_B_hgb + cnn_weight * proba_B_cnn
        print(f"   [B] HGBM range: [{proba_B_hgb.min():.6f}, {proba_B_hgb.max():.6f}], mean: {proba_B_hgb.mean():.6f}")
        print(f"   [B] CNN range: [{proba_B_cnn.min():.6f}, {proba_B_cnn.max():.6f}], mean: {proba_B_cnn.mean():.6f}")
        print(f"   [B] Ensemble range: [{proba_B.min():.6f}, {proba_B.max():.6f}], mean: {proba_B.mean():.6f}")
    elif proba_B_hgb is not None:
        proba_B = proba_B_hgb
        print("⚠️  Using HGBM only for B (CNN not available)")
        print(f"   [B] HGBM range: [{proba_B_hgb.min():.6f}, {proba_B_hgb.max():.6f}], mean: {proba_B_hgb.mean():.6f}")
    elif proba_B_cnn is not None:
        proba_B = proba_B_cnn
        print("⚠️  Using CNN only for B (HGBM not available)")
        print(f"   [B] CNN range: [{proba_B_cnn.min():.6f}, {proba_B_cnn.max():.6f}], mean: {proba_B_cnn.mean():.6f}")
    else:
        raise ValueError("Neither HGBM nor CNN predictions available for B")
    
    proba_B = np.clip(proba_B, 1e-7, 1 - 1e-7)
    pred_B = pd.DataFrame({"Test_id": B_test_idx["Test_id"].values, "Label": proba_B})
    
    # Submission 저장
    submission = pd.concat([pred_A, pred_B], axis=0).sort_values("Test_id")
    submission = submission[["Test_id", "Label"]]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    
    dt = time.time() - t0
    print(f"\n✅ Submission saved to {SUBMISSION_PATH}")
    print(f"   Shape: {submission.shape}")
    print(f"   Label range: [{submission['Label'].min():.6f}, {submission['Label'].max():.6f}]")
    print(f"   Elapsed time: {dt/60:.2f} min")
    
    # 평가 점수 계산 (answer.csv가 있으면)
    answer_path = os.path.join(DATA_DIR, "answer.csv")
    if os.path.exists(answer_path):
        try:
            answer_df = pd.read_csv(answer_path)
            combined_score, mean_auc, mean_brier, mean_ece = auc_brier_ece(answer_df, submission)
            print(f"\n📊 Evaluation Metrics:")
            print(f"   AUC: {mean_auc:.6f}")
            print(f"   Brier Score: {mean_brier:.6f}")
            print(f"   ECE: {mean_ece:.6f}")
            print(f"   Combined Score: {combined_score:.6f}")
        except Exception as e:
            print(f"⚠️  Failed to calculate evaluation metrics: {e}")
    else:
        print(f"ℹ️  answer.csv not found - skipping evaluation metrics")

if __name__ == "__main__":
    main()

