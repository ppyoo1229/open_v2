#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D CNN 추론 전용 스크립트 (CPU)
- 학습 스크립트(cnn_train.py)에서 저장한 모델/전처리기를 사용
- test A/B raw + FE를 다시 만들고, CPU에서만 추론 수행
- 출력: output/cnn_pred_A.csv, output/cnn_pred_B.csv
"""

import os
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle

# PyTorch (CPU 전용)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("❌ PyTorch가 필요합니다. `pip install torch`로 설치하세요.")

# =============================================================================
# 경로 설정
# =============================================================================

DATA_DIR   = "data"
OUTPUT_DIR = "output"
MODEL_DIR  = "model"

SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
META_PATH       = os.path.join(MODEL_DIR, "meta.json")
BEST_PARAMS_PATH = os.path.join(MODEL_DIR, "best_params.json")

A_MODEL_PATH      = os.path.join(MODEL_DIR, "cnn_A.pkl")
B_MODEL_PATH      = os.path.join(MODEL_DIR, "cnn_B.pkl")
A_PREPROC_PATH    = os.path.join(MODEL_DIR, "preproc_A_cnn.pkl")
B_PREPROC_PATH    = os.path.join(MODEL_DIR, "preproc_B_cnn.pkl")

RANDOM_STATE = 42

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
}

# Note: train_cnn.py에서는 피처 블록을 사용하지 않고 바로 select_light_features를 호출합니다.
# 따라서 apply_feature_blocks는 사용하지 않습니다.


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


def read_index_files():
    train_idx = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_idx = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train_idx, test_idx


def read_raw_feature_files(split: str):
    A_df = pd.read_csv(os.path.join(DATA_DIR, split, "A.csv"))
    B_df = pd.read_csv(os.path.join(DATA_DIR, split, "B.csv"))
    return A_df, B_df


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

from typing import Dict, List, Tuple

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
# 전처리기 유틸 함수들 (utils.py에서 복사)
# =============================================================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

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
def infer_partition_A(
    A_test_raw: pd.DataFrame,
    A_test_feat_full: pd.DataFrame,
    test_idx: pd.DataFrame,
    device: torch.device,
):
    """
    A 테스트셋에 대해:
    - FE + light feature 선택
    - 멀티채널 시퀀스 빌드
    - 저장된 preproc + 모델 로드
    - CPU에서 추론
    """
    which = "A"
    key = "Test_id"

    # FE (학습과 동일 파이프라인)
    # Note: train_mask=None 사용
    # - 학습 시: train 데이터만 있으므로 train 데이터로 통계 계산 (자동)
    # - 추론 시: test 데이터만 있으므로 test 데이터로 통계 계산
    # 엄밀한 의미의 leakage 방지는 아니지만, 학습/추론 모두 동일한 FE 함수를 사용하고,
    # preproc는 정석대로 train-fit/test-transform으로 처리하므로 실무적 타협 수준
    print("\n🔵 [A] FE pipeline (test) ...")
    t0 = time.time()
    A_test_feat = preprocess_A(A_test_raw)
    A_test_feat = add_features_A(A_test_feat, train_mask=None)  # 학습 시와 동일: train_mask=None
    A_test_feat = add_age_normed_composites_A(A_test_feat, train_mask=None)  # 학습 시와 동일
    A_test_feat = add_history_features_A(A_test_feat)
    # Note: train_cnn.py와 동일하게 피처 블록을 사용하지 않고 바로 라이트 피처 선택
    print(f"  [A] FE done: {A_test_feat.shape}, elapsed {(time.time()-t0)/60:.2f} min")

    # CNN용 라이트 피처만 선택
    A_test_feat_cnn = select_light_features_A(A_test_feat)
    print(f"  [A] light features shape: {A_test_feat_cnn.shape}")

    # 인덱스 정렬
    A_test_idx = test_idx[test_idx["Test"] == "A"].copy()

    # 원시 데이터 정렬 (시퀀스용)
    A_test_raw_aligned = A_test_idx.merge(A_test_raw, on=key, how="left", validate="1:1")

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
    with open(A_PREPROC_PATH, "rb") as f:
        preproc = pickle.load(f)
    with open(A_MODEL_PATH, "rb") as f:
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

    # 모델 아키텍처 복원 + state_dict 로드 (사용자 제시 방식)
    # model = MultiSequenceCNN(bundle["config"], num_trial_features=??)
    # model.sequence_modules[...] = ...
    # model.final_classifier = ...
    # model.load_state_dict(bundle["model"], strict=True)
    # model.to("cpu")
    model = build_cnn_model_for_inference(
        model_bundle=bundle,
        num_trial_features=num_trial_features,
        sequences=sequences,
        device=device,
    )

    print("[A] Running inference on CPU ...")
    t1 = time.time()
    preds = predict_cnn_model(
        model=model,
        sequences=sequences,
        trial_features=X_trial.astype(np.float32),
        device=device,
        batch_size=CNN_CONFIG.get("batch_size", 32) * 2,  # 추론은 배치 조금 키움
    )
    dt = time.time() - t1
    print(f"[A] Inference done: {len(preds)} samples, elapsed {dt:.2f} sec")

    # 저장
    out_df = A_test_idx[[key]].copy()
    out_df["pred_cnn"] = preds
    out_path = os.path.join(OUTPUT_DIR, "cnn_pred_A.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[A] Saved predictions → {out_path}")

    return out_df


def infer_partition_B(
    B_test_raw: pd.DataFrame,
    B_test_feat_full: pd.DataFrame,
    test_idx: pd.DataFrame,
    device: torch.device,
):
    """
    B 테스트셋에 대해:
    - FE + light feature 선택
    - 멀티채널 시퀀스 빌드
    - 저장된 preproc + 모델 로드
    - CPU에서 추론
    """
    which = "B"
    key = "Test_id"

    print("\n🟢 [B] FE pipeline (test) ...")
    # Note: train_mask=None 사용 (A와 동일한 이유)
    t0 = time.time()
    B_test_feat = preprocess_B(B_test_raw)
    B_test_feat = add_features_B(B_test_feat, train_mask=None)  # 학습 시와 동일: train_mask=None
    B_test_feat = add_age_normed_composites_B(B_test_feat, train_mask=None)  # 학습 시와 동일
    B_test_feat = add_history_features_B(B_test_feat)
    # Note: train_cnn.py와 동일하게 피처 블록을 사용하지 않고 바로 라이트 피처 선택
    print(f"  [B] FE done: {B_test_feat.shape}, elapsed {(time.time()-t0)/60:.2f} min")

    # CNN용 라이트 피처
    B_test_feat_cnn = select_light_features_B(B_test_feat)
    print(f"  [B] light features shape: {B_test_feat_cnn.shape}")

    # 인덱스 정렬
    B_test_idx = test_idx[test_idx["Test"] == "B"].copy()

    # 원시 데이터 정렬 (시퀀스용)
    B_test_raw_aligned = B_test_idx.merge(B_test_raw, on=key, how="left", validate="1:1")

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
    with open(B_PREPROC_PATH, "rb") as f:
        preproc = pickle.load(f)
    with open(B_MODEL_PATH, "rb") as f:
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

    print("[B] Running inference on CPU ...")
    t1 = time.time()
    preds = predict_cnn_model(
        model=model,
        sequences=sequences,
        trial_features=X_trial.astype(np.float32),
        device=device,
        batch_size=CNN_CONFIG.get("batch_size", 32) * 2,
    )
    dt = time.time() - t1
    print(f"[B] Inference done: {len(preds)} samples, elapsed {dt:.2f} sec")

    out_df = B_test_idx[[key]].copy()
    out_df["pred_cnn"] = preds
    out_path = os.path.join(OUTPUT_DIR, "cnn_pred_B.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[B] Saved predictions → {out_path}")

    return out_df


# ---------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------
def main():
    if not HAS_TORCH:
        print("❌ PyTorch가 필요합니다. `pip install torch`로 설치 후 다시 실행해주세요.")
        return

    set_seed(42)
    ensure_dirs()

    device = torch.device("cpu")
    print(f"🖥 Using device: {device} (CPU inference only)")

    print("📂 Loading index files ...")
    train_idx, test_idx = read_index_files()

    print("📂 Loading raw A/B (test only) ...")
    A_test_raw, B_test_raw = read_raw_feature_files("test")

    t0 = time.time()

    # A, B 파티션별 추론
    pred_A = infer_partition_A(
        A_test_raw=A_test_raw,
        A_test_feat_full=None,  # FE는 함수 안에서 다시 계산
        test_idx=test_idx,
        device=device,
    )

    pred_B = infer_partition_B(
        B_test_raw=B_test_raw,
        B_test_feat_full=None,
        test_idx=test_idx,
        device=device,
    )

    # 추론 시간 계산
    dt = time.time() - t0

    # ===== 제출물 생성 =====
    print("\n📝 Creating submission.csv ...")
    
    # A와 B 예측 합치기
    if pred_A is not None and pred_B is not None:
        # 컬럼명 통일 (pred_cnn -> Label)
        pred_A_sub = pred_A.rename(columns={"pred_cnn": "Label"})[["Test_id", "Label"]]
        pred_B_sub = pred_B.rename(columns={"pred_cnn": "Label"})[["Test_id", "Label"]]
        sub = pd.concat([pred_A_sub, pred_B_sub], axis=0, ignore_index=True)
    elif pred_A is not None:
        sub = pred_A.rename(columns={"pred_cnn": "Label"})[["Test_id", "Label"]]
    elif pred_B is not None:
        sub = pred_B.rename(columns={"pred_cnn": "Label"})[["Test_id", "Label"]]
    else:
        sub = test_idx[["Test_id"]].copy()
        sub["Label"] = 0.001
    
    # 확률 값 클리핑 (0~1 범위)
    sub["Label"] = sub["Label"].clip(lower=1e-7, upper=1-1e-7)
    
    # sample_submission 순서 맞추기
    try:
        sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
        sub = sub.merge(sample[["Test_id"]], on="Test_id", how="right")
        sub = sub[["Test_id", "Label"]]
    except Exception as e:
        print(f"⚠️  Warning: Could not load sample_submission.csv: {e}")
        sub = sub[["Test_id", "Label"]]
    
    # submission.csv 저장
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"✅ Submission saved → {SUBMISSION_PATH}")
    print(f"   - Total predictions: {len(sub)} rows")
    print(f"   - Label range: [{sub['Label'].min():.6f}, {sub['Label'].max():.6f}]")
    print(f"   - Label mean: {sub['Label'].mean():.6f}")
    
    # ===== 점수 로그 생성 =====
    print("\n📊 Creating score log ...")
    try:
        # 모델 메타 정보 로드 시도
        meta_info = {}
        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta_info = json.load(f)
        
        # best_params.json에서 평가 점수 로드 시도
        best_params_info = {}
        evaluation_scores = {}
        if os.path.exists(BEST_PARAMS_PATH):
            try:
                with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
                    best_params_info = json.load(f)
                
                # OOF 평가 점수 추출
                if "oof_score_A" in best_params_info:
                    evaluation_scores["oof_score_A"] = best_params_info["oof_score_A"]
                if "oof_score_B" in best_params_info:
                    evaluation_scores["oof_score_B"] = best_params_info["oof_score_B"]
                if "oof_score_total" in best_params_info:
                    evaluation_scores["oof_score_total"] = best_params_info["oof_score_total"]
                if "oof_auc_total" in best_params_info:
                    evaluation_scores["oof_auc_total"] = best_params_info["oof_auc_total"]
                
                if evaluation_scores:
                    print(f"  📈 Loaded evaluation scores from {BEST_PARAMS_PATH}")
                    for key, value in evaluation_scores.items():
                        print(f"     - {key}: {value:.6f}")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not load best_params.json: {e}")
        
        # 추론 메타 정보
        inference_meta = {
            "script": "script_cnn.py",
            "model_type": "1D CNN (MultiSequenceCNN)",
            "inference_device": "cpu",
            "inference_time_sec": dt,
            "inference_time_min": dt / 60,
            "num_predictions_A": len(pred_A) if pred_A is not None else 0,
            "num_predictions_B": len(pred_B) if pred_B is not None else 0,
            "total_predictions": len(sub),
            "label_stats": {
                "min": float(sub["Label"].min()),
                "max": float(sub["Label"].max()),
                "mean": float(sub["Label"].mean()),
                "std": float(sub["Label"].std()),
            },
            "cnn_config": CNN_CONFIG,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # 평가 점수 추가
        if evaluation_scores:
            inference_meta["evaluation_scores"] = evaluation_scores
        
        # 학습 메타 정보가 있으면 병합
        if meta_info:
            inference_meta["training_meta"] = meta_info
        
        # best_params 정보가 있으면 추가 (하이퍼파라미터 등)
        if best_params_info:
            # 민감한 정보 제외하고 주요 설정만 포함
            inference_meta["model_config"] = {
                "ensemble_seeds": best_params_info.get("ensemble_seeds"),
                "use_calibration": best_params_info.get("use_calibration"),
                "calib_method": best_params_info.get("calib_method"),
                "calib_cv": best_params_info.get("calib_cv"),
            }
        
        # 점수 로그 저장
        score_log_path = os.path.join(OUTPUT_DIR, "cnn_inference_log.json")
        with open(score_log_path, "w", encoding="utf-8") as f:
            json.dump(inference_meta, f, ensure_ascii=False, indent=2)
        print(f"✅ Score log saved → {score_log_path}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not create score log: {e}")
        import traceback
        traceback.print_exc()
    print("\n" + "=" * 80)
    print(f"✅ All inference done on CPU | elapsed: {dt/60:.2f} min")
    print(f"  - A preds: {len(pred_A)} rows → output/cnn_pred_A.csv")
    print(f"  - B preds: {len(pred_B)} rows → output/cnn_pred_B.csv")
    print(f"  - Submission: {len(sub)} rows → {SUBMISSION_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
