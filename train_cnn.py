#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D CNN 기반 모델 학습 스크립트
- utils.py의 FE 로직 재사용 (풀 FE 파이프라인)
- 풀 FE → select_light_features_*로 라이트 피처만 추출
- Raw Sequences (A1-4, A2-4, A3-7, A4-5 등)를 1D CNN으로 처리
- Trial-wise features와 결합하여 최종 예측
- Attention 메커니즘 지원

Note: GBM(best08.py)과 파이프라인 분리
- GBM: 풀 FE + apply_feature_blocks로 블록 실험/튜닝
- CNN: 풀 FE → 라이트 피처 선택 (블록 개념 미사용)
"""
import os
import time
import json
import warnings
warnings.filterwarnings("ignore")

from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedGroupKFold

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch not installed. Install with: pip install torch")

import pickle

# =============================================================================
# 경로 설정
# =============================================================================

DATA_DIR   = "data"
OUTPUT_DIR = "output"
MODEL_DIR  = "model"

META_PATH         = os.path.join(MODEL_DIR, "meta_cnn.json")

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

# =============================================================================
# 실행 옵션
# =============================================================================
USE_OOF = False
N_SPLITS = 2
# =============================================================================
# 시퀀스 관련 유틸
# =============================================================================

def parse_sequence_to_array(series: pd.Series,
                            max_length: int = None,
                            dtype: type = float) -> np.ndarray:
    """
    시퀀스 문자열을 numpy array로 변환 (패딩 포함)

    Parameters
    ----------
    series : pd.Series
        콤마로 구분된 시퀀스 문자열
    max_length : int
        최대 길이 (None이면 자동 계산)
    dtype : type
        데이터 타입 (float or int)

    Returns
    -------
    np.ndarray
        shape (n_samples, max_length)
    """
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


def build_sequence_A1(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """
    A1 검사 멀티채널 시퀀스 구성 (8~9채널)
    - RT_norm, is_response, stim_side one-hot(2), speed one-hot(3), ΔRT, mask
    """
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    
    required_cols = ["A1-1", "A1-2", "A1-3", "A1-4"]
    if not all(c in df.columns for c in required_cols):
        return None
    
    n_samples = len(df)
    channels = []
    
    # RT 파싱 및 정규화
    rt_raw = parse_sequence_to_array(df["A1-4"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(rt_raw)
    rt_for_stat = rt_raw.copy()
    rt_for_stat[~valid_mask] = np.nan
    rt_mean = np.nanmean(rt_for_stat)
    rt_std = np.nanstd(rt_for_stat) + 1e-6
    rt_norm = (np.nan_to_num(rt_raw, nan=rt_mean) - rt_mean) / rt_std
    channels.append(rt_norm.astype(np.float32))
    
    # is_response (A1-3: 0/1)
    resp_raw = parse_sequence_to_array(df["A1-3"], max_length=max_length, dtype=float)
    resp = np.nan_to_num(resp_raw, nan=0.0).astype(np.float32)
    channels.append(resp)
    
    # stim_side one-hot (A1-1: 1=left, 2=right)
    side_raw = parse_sequence_to_array(df["A1-1"], max_length=max_length, dtype=float)
    stim_left = (side_raw == 1).astype(np.float32)
    stim_right = (side_raw == 2).astype(np.float32)
    channels.extend([stim_left, stim_right])
    
    # speed one-hot (A1-2: 1=slow, 2=normal, 3=fast)
    speed_raw = parse_sequence_to_array(df["A1-2"], max_length=max_length, dtype=float)
    speed_slow = (speed_raw == 1).astype(np.float32)
    speed_normal = (speed_raw == 2).astype(np.float32)
    speed_fast = (speed_raw == 3).astype(np.float32)
    channels.extend([speed_slow, speed_normal, speed_fast])
    
    # ΔRT
    delta = np.diff(rt_raw, axis=1, prepend=np.nan)
    delta[~valid_mask] = np.nan
    delta_mean = np.nanmean(delta)
    delta_std = np.nanstd(delta) + 1e-6
    delta_norm = (np.nan_to_num(delta, nan=delta_mean) - delta_mean) / delta_std
    channels.append(delta_norm.astype(np.float32))
    
    # mask
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    
    seq_array = np.stack(channels, axis=-1)  # (N, L, 9)
    return seq_array


def build_sequence_A2(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """
    A2 검사 멀티채널 시퀀스 구성 (11채널)
    - RT_norm, is_response, cond1 one-hot(3), cond2 one-hot(3), speed_gap, ΔRT, trial_index_norm, mask
    """
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    
    required_cols = ["A2-1", "A2-2", "A2-3", "A2-4"]
    if not all(c in df.columns for c in required_cols):
        return None
    
    n_samples = len(df)
    channels = []
    
    # RT 파싱 및 정규화
    rt_raw = parse_sequence_to_array(df["A2-4"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(rt_raw)
    rt_for_stat = rt_raw.copy()
    rt_for_stat[~valid_mask] = np.nan
    rt_mean = np.nanmean(rt_for_stat)
    rt_std = np.nanstd(rt_for_stat) + 1e-6
    rt_norm = (np.nan_to_num(rt_raw, nan=rt_mean) - rt_mean) / rt_std
    channels.append(rt_norm.astype(np.float32))
    
    # is_response (A2-3: 0/1)
    resp_raw = parse_sequence_to_array(df["A2-3"], max_length=max_length, dtype=float)
    resp = np.nan_to_num(resp_raw, nan=0.0).astype(np.float32)
    channels.append(resp)
    
    # Condition1 one-hot (1=slow, 2=normal, 3=fast)
    cond1_raw = parse_sequence_to_array(df["A2-1"], max_length=max_length, dtype=float)
    cond1_slow = (cond1_raw == 1).astype(np.float32)
    cond1_normal = (cond1_raw == 2).astype(np.float32)
    cond1_fast = (cond1_raw == 3).astype(np.float32)
    channels.extend([cond1_slow, cond1_normal, cond1_fast])
    
    # Condition2 one-hot
    cond2_raw = parse_sequence_to_array(df["A2-2"], max_length=max_length, dtype=float)
    cond2_slow = (cond2_raw == 1).astype(np.float32)
    cond2_normal = (cond2_raw == 2).astype(np.float32)
    cond2_fast = (cond2_raw == 3).astype(np.float32)
    channels.extend([cond2_slow, cond2_normal, cond2_fast])
    
    # speed_gap
    speed_gap = np.abs(cond1_raw - cond2_raw)
    speed_gap[~valid_mask] = 0.0
    speed_gap_norm = speed_gap / 2.0  # 0,1,2 -> 0, 0.5, 1
    channels.append(speed_gap_norm.astype(np.float32))
    
    # ΔRT
    delta = np.diff(rt_raw, axis=1, prepend=np.nan)
    delta[~valid_mask] = np.nan
    delta_mean = np.nanmean(delta)
    delta_std = np.nanstd(delta) + 1e-6
    delta_norm = (np.nan_to_num(delta, nan=delta_mean) - delta_mean) / delta_std
    channels.append(delta_norm.astype(np.float32))
    
    # trial_index_norm
    trial_idx = np.arange(max_length, dtype=np.float32)[None, :] / (max_length - 1 + 1e-6)
    trial_idx = np.tile(trial_idx, (n_samples, 1))
    trial_idx[~valid_mask] = 0.0
    channels.append(trial_idx)
    
    # mask
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    
    seq_array = np.stack(channels, axis=-1)  # (N, L, 11)
    return seq_array


def build_sequence_A3(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """
    A3 검사 멀티채널 시퀀스 구성 (16채널)
    - RT_norm, is_response, is_valid_cue, is_correct, valid_x_incorrect, invalid_x_incorrect,
      target_size one-hot(2), cue/target hemisphere(4: left/right 각각), cue_target_congruency, arrow_direction(2),
      ΔRT, running_error_rate, mask
    """
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    
    required_cols = ["A3-1", "A3-2", "A3-3", "A3-4", "A3-5", "A3-6", "A3-7"]
    if not all(c in df.columns for c in required_cols):
        return None
    
    n_samples = len(df)
    channels = []
    
    # RT 파싱 및 정규화
    rt_raw = parse_sequence_to_array(df["A3-7"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(rt_raw)
    rt_for_stat = rt_raw.copy()
    rt_for_stat[~valid_mask] = np.nan
    rt_mean = np.nanmean(rt_for_stat)
    rt_std = np.nanstd(rt_for_stat) + 1e-6
    rt_norm = (np.nan_to_num(rt_raw, nan=rt_mean) - rt_mean) / rt_std
    channels.append(rt_norm.astype(np.float32))
    
    # is_response (A3-6: 0/1)
    resp_raw = parse_sequence_to_array(df["A3-6"], max_length=max_length, dtype=float)
    resp = np.nan_to_num(resp_raw, nan=0.0).astype(np.float32)
    channels.append(resp)
    
    # Response1 파싱 (A3-5: 1=valid-correct, 2=valid-incorrect, 3=invalid-correct, 4=invalid-incorrect)
    resp1_raw = parse_sequence_to_array(df["A3-5"], max_length=max_length, dtype=float)
    is_valid_cue = ((resp1_raw == 1) | (resp1_raw == 2)).astype(np.float32)
    is_correct = ((resp1_raw == 1) | (resp1_raw == 3)).astype(np.float32)
    valid_x_incorrect = (resp1_raw == 2).astype(np.float32)
    invalid_x_incorrect = (resp1_raw == 4).astype(np.float32)
    channels.extend([is_valid_cue, is_correct, valid_x_incorrect, invalid_x_incorrect])
    
    # target_size one-hot (A3-1: 1=small, 2=big)
    size_raw = parse_sequence_to_array(df["A3-1"], max_length=max_length, dtype=float)
    size_small = (size_raw == 1).astype(np.float32)
    size_big = (size_raw == 2).astype(np.float32)
    channels.extend([size_small, size_big])
    
    # cue/target hemisphere (A3-2, A3-4: 1~8, 1~4=left, 5~8=right)
    cue_pos_raw = parse_sequence_to_array(df["A3-2"], max_length=max_length, dtype=float)
    tgt_pos_raw = parse_sequence_to_array(df["A3-4"], max_length=max_length, dtype=float)
    cue_left = ((cue_pos_raw >= 1) & (cue_pos_raw <= 4)).astype(np.float32)
    cue_right = ((cue_pos_raw >= 5) & (cue_pos_raw <= 8)).astype(np.float32)
    tgt_left = ((tgt_pos_raw >= 1) & (tgt_pos_raw <= 4)).astype(np.float32)
    tgt_right = ((tgt_pos_raw >= 5) & (tgt_pos_raw <= 8)).astype(np.float32)
    channels.extend([cue_left, cue_right, tgt_left, tgt_right])
    
    # cue_target_congruency
    hemi_congruent = ((cue_left == tgt_left) | (cue_right == tgt_right)).astype(np.float32)
    channels.append(hemi_congruent)
    
    # arrow_direction (A3-3: 1=left, 2=right)
    arrow_raw = parse_sequence_to_array(df["A3-3"], max_length=max_length, dtype=float)
    arrow_left = (arrow_raw == 1).astype(np.float32)
    arrow_right = (arrow_raw == 2).astype(np.float32)
    channels.extend([arrow_left, arrow_right])
    
    # ΔRT
    delta = np.diff(rt_raw, axis=1, prepend=np.nan)
    delta[~valid_mask] = np.nan
    delta_mean = np.nanmean(delta)
    delta_std = np.nanstd(delta) + 1e-6
    delta_norm = (np.nan_to_num(delta, nan=delta_mean) - delta_mean) / delta_std
    channels.append(delta_norm.astype(np.float32))
    
    # running_error_rate (벡터화 버전)
    incorrect = ((resp1_raw == 2) | (resp1_raw == 4)).astype(np.float32)  # incorrect trials
    cumsum = np.cumsum(incorrect, axis=1)  # (N, L)
    t_idx = np.arange(max_length, dtype=np.float32)[None, :] + 1  # 1~L
    running_error = cumsum / t_idx
    running_error[~valid_mask] = 0.0
    channels.append(running_error.astype(np.float32))
    
    # mask
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    
    seq_array = np.stack(channels, axis=-1)  # (N, L, 16)
    return seq_array


def build_sequence_A5(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """
    A5 검사 멀티채널 시퀀스 구성 (10채널)
    - is_change, change_type one-hot(4), is_correct, is_response_YN, miss_change, false_alarm,
      trial_index_norm, mask
    """
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    
    required_cols = ["A5-1", "A5-2", "A5-3"]
    if not all(c in df.columns for c in required_cols):
        return None
    
    n_samples = len(df)
    channels = []
    
    # Condition 파싱 (A5-1: 1=non_change, 2=pos_change, 3=color_change, 4=shape_change)
    cond_raw = parse_sequence_to_array(df["A5-1"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(cond_raw)
    
    is_change = ((cond_raw >= 2) & (cond_raw <= 4)).astype(np.float32)
    channels.append(is_change)
    
    # change_type one-hot
    non_change = (cond_raw == 1).astype(np.float32)
    pos_change = (cond_raw == 2).astype(np.float32)
    color_change = (cond_raw == 3).astype(np.float32)
    shape_change = (cond_raw == 4).astype(np.float32)
    channels.extend([non_change, pos_change, color_change, shape_change])
    
    # is_correct (A5-2: 1=correct, 2=incorrect)
    correct_raw = parse_sequence_to_array(df["A5-2"], max_length=max_length, dtype=float)
    is_correct = (correct_raw == 1).astype(np.float32)
    channels.append(is_correct)
    
    # is_response_YN (A5-3: 0/1)
    resp_raw = parse_sequence_to_array(df["A5-3"], max_length=max_length, dtype=float)
    resp_yn = np.nan_to_num(resp_raw, nan=0.0).astype(np.float32)
    channels.append(resp_yn)
    
    # miss_change, false_alarm
    miss_change = (is_change == 1.0) & (resp_yn == 0.0)
    false_alarm = (is_change == 0.0) & (resp_yn == 1.0)
    channels.extend([miss_change.astype(np.float32), false_alarm.astype(np.float32)])
    
    # trial_index_norm (0~1)
    trial_idx = np.arange(max_length, dtype=np.float32)[None, :] / (max_length - 1 + 1e-6)
    trial_idx = np.tile(trial_idx, (n_samples, 1))
    trial_idx[~valid_mask] = 0.0
    channels.append(trial_idx)
    
    # mask
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    
    seq_array = np.stack(channels, axis=-1)  # (N, L, 10)
    return seq_array


def build_sequence_A4(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """
    A4 검사 멀티채널 시퀀스 구성 (12채널)
    - RT_norm, is_response, is_correct, is_incongruent, stim_color one-hot(2),
      incong_x_error, ΔRT, trial_index_norm, mask
    """
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    
    required_cols = ["A4-1", "A4-2", "A4-3", "A4-4", "A4-5"]
    if not all(c in df.columns for c in required_cols):
        return None
    
    n_samples = len(df)
    channels = []
    
    # RT 파싱 및 정규화
    rt_raw = parse_sequence_to_array(df["A4-5"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(rt_raw)
    rt_for_stat = rt_raw.copy()
    rt_for_stat[~valid_mask] = np.nan
    rt_mean = np.nanmean(rt_for_stat)
    rt_std = np.nanstd(rt_for_stat) + 1e-6
    rt_norm = (np.nan_to_num(rt_raw, nan=rt_mean) - rt_mean) / rt_std
    channels.append(rt_norm.astype(np.float32))
    
    # is_response (A4-4: 0/1)
    resp_raw = parse_sequence_to_array(df["A4-4"], max_length=max_length, dtype=float)
    resp = np.nan_to_num(resp_raw, nan=0.0).astype(np.float32)
    channels.append(resp)
    
    # is_correct (A4-3: 1=correct, 2=incorrect)
    correct_raw = parse_sequence_to_array(df["A4-3"], max_length=max_length, dtype=float)
    is_correct = (correct_raw == 1).astype(np.float32)
    channels.append(is_correct)
    
    # is_incongruent (A4-1: 1=congruent, 2=incongruent)
    cong_raw = parse_sequence_to_array(df["A4-1"], max_length=max_length, dtype=float)
    is_incongruent = (cong_raw == 2).astype(np.float32)
    channels.append(is_incongruent)
    
    # stim_color one-hot (A4-2: 1=red, 2=green)
    color_raw = parse_sequence_to_array(df["A4-2"], max_length=max_length, dtype=float)
    stim_red = (color_raw == 1).astype(np.float32)
    stim_green = (color_raw == 2).astype(np.float32)
    channels.extend([stim_red, stim_green])
    
    # incong_x_error flag
    incong_error = (is_incongruent == 1.0) & (is_correct == 0.0)
    channels.append(incong_error.astype(np.float32))
    
    # ΔRT
    delta = np.diff(rt_raw, axis=1, prepend=np.nan)
    delta[~valid_mask] = np.nan
    delta_mean = np.nanmean(delta)
    delta_std = np.nanstd(delta) + 1e-6
    delta_norm = (np.nan_to_num(delta, nan=delta_mean) - delta_mean) / delta_std
    channels.append(delta_norm.astype(np.float32))
    
    # trial_index_norm
    trial_idx = np.arange(max_length, dtype=np.float32)[None, :] / (max_length - 1 + 1e-6)
    trial_idx = np.tile(trial_idx, (n_samples, 1))
    trial_idx[~valid_mask] = 0.0
    channels.append(trial_idx)
    
    # mask
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    
    seq_array = np.stack(channels, axis=-1)  # (N, L, 12)
    return seq_array


def build_sequence_B1(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """
    B1 검사 멀티채널 시퀀스 구성 (9채널)
    - RT_norm, task1_correct, is_change, is_correct, miss_change, false_alarm, ΔRT, trial_index_norm, mask
    """
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    
    required_cols = ["B1-1", "B1-2", "B1-3"]
    if not all(c in df.columns for c in required_cols):
        return None
    
    n_samples = len(df)
    channels = []
    
    # RT 파싱 및 정규화 (B1-2)
    rt_raw = parse_sequence_to_array(df["B1-2"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(rt_raw)
    rt_for_stat = rt_raw.copy()
    rt_for_stat[~valid_mask] = np.nan
    rt_mean = np.nanmean(rt_for_stat)
    rt_std = np.nanstd(rt_for_stat) + 1e-6
    rt_norm = (np.nan_to_num(rt_raw, nan=rt_mean) - rt_mean) / rt_std
    channels.append(rt_norm.astype(np.float32))
    
    # task1_correct (B1-1: 1=correct, 2=incorrect)
    task1_raw = parse_sequence_to_array(df["B1-1"], max_length=max_length, dtype=float)
    task1_correct = (task1_raw == 1).astype(np.float32)
    channels.append(task1_correct)
    
    # task2 파싱 (B1-3: 1=change-correct, 2=change-incorrect, 3=nonchange-correct, 4=nonchange-incorrect)
    task2_raw = parse_sequence_to_array(df["B1-3"], max_length=max_length, dtype=float)
    is_change = ((task2_raw == 1) | (task2_raw == 2)).astype(np.float32)
    is_correct = ((task2_raw == 1) | (task2_raw == 3)).astype(np.float32)
    miss_change = (task2_raw == 2).astype(np.float32)
    false_alarm = (task2_raw == 4).astype(np.float32)
    channels.extend([is_change, is_correct, miss_change, false_alarm])
    
    # ΔRT
    delta = np.diff(rt_raw, axis=1, prepend=np.nan)
    delta[~valid_mask] = np.nan
    delta_mean = np.nanmean(delta)
    delta_std = np.nanstd(delta) + 1e-6
    delta_norm = (np.nan_to_num(delta, nan=delta_mean) - delta_mean) / delta_std
    channels.append(delta_norm.astype(np.float32))
    
    # trial_index_norm
    trial_idx = np.arange(max_length, dtype=np.float32)[None, :] / (max_length - 1 + 1e-6)
    trial_idx = np.tile(trial_idx, (n_samples, 1))
    trial_idx[~valid_mask] = 0.0
    channels.append(trial_idx)
    
    # mask
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    
    seq_array = np.stack(channels, axis=-1)  # (N, L, 9)
    return seq_array


def build_sequence_B2(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """
    B2 검사 멀티채널 시퀀스 구성 (B1과 동일, 9채널)
    """
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    
    required_cols = ["B2-1", "B2-2", "B2-3"]
    if not all(c in df.columns for c in required_cols):
        return None
    
    n_samples = len(df)
    channels = []
    
    # RT 파싱 및 정규화 (B2-2)
    rt_raw = parse_sequence_to_array(df["B2-2"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(rt_raw)
    rt_for_stat = rt_raw.copy()
    rt_for_stat[~valid_mask] = np.nan
    rt_mean = np.nanmean(rt_for_stat)
    rt_std = np.nanstd(rt_for_stat) + 1e-6
    rt_norm = (np.nan_to_num(rt_raw, nan=rt_mean) - rt_mean) / rt_std
    channels.append(rt_norm.astype(np.float32))
    
    # task1_correct (B2-1: 1=correct, 2=incorrect)
    task1_raw = parse_sequence_to_array(df["B2-1"], max_length=max_length, dtype=float)
    task1_correct = (task1_raw == 1).astype(np.float32)
    channels.append(task1_correct)
    
    # task2 파싱 (B2-3: 1~4)
    task2_raw = parse_sequence_to_array(df["B2-3"], max_length=max_length, dtype=float)
    is_change = ((task2_raw == 1) | (task2_raw == 2)).astype(np.float32)
    is_correct = ((task2_raw == 1) | (task2_raw == 3)).astype(np.float32)
    miss_change = (task2_raw == 2).astype(np.float32)
    false_alarm = (task2_raw == 4).astype(np.float32)
    channels.extend([is_change, is_correct, miss_change, false_alarm])
    
    # ΔRT
    delta = np.diff(rt_raw, axis=1, prepend=np.nan)
    delta[~valid_mask] = np.nan
    delta_mean = np.nanmean(delta)
    delta_std = np.nanstd(delta) + 1e-6
    delta_norm = (np.nan_to_num(delta, nan=delta_mean) - delta_mean) / delta_std
    channels.append(delta_norm.astype(np.float32))
    
    # trial_index_norm
    trial_idx = np.arange(max_length, dtype=np.float32)[None, :] / (max_length - 1 + 1e-6)
    trial_idx = np.tile(trial_idx, (n_samples, 1))
    trial_idx[~valid_mask] = 0.0
    channels.append(trial_idx)
    
    # mask
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    
    seq_array = np.stack(channels, axis=-1)  # (N, L, 9)
    return seq_array


def build_sequence_B3(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """
    B3 검사 멀티채널 시퀀스 구성 (7채널)
    - RT_norm, is_correct, ΔRT, running_error_rate, trial_index_norm, mask
    """
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    
    required_cols = ["B3-1", "B3-2"]
    if not all(c in df.columns for c in required_cols):
        return None
    
    n_samples = len(df)
    channels = []
    
    # RT 파싱 및 정규화
    rt_raw = parse_sequence_to_array(df["B3-2"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(rt_raw)
    rt_for_stat = rt_raw.copy()
    rt_for_stat[~valid_mask] = np.nan
    rt_mean = np.nanmean(rt_for_stat)
    rt_std = np.nanstd(rt_for_stat) + 1e-6
    rt_norm = (np.nan_to_num(rt_raw, nan=rt_mean) - rt_mean) / rt_std
    channels.append(rt_norm.astype(np.float32))
    
    # is_correct (B3-1: 1=correct, 2=incorrect)
    correct_raw = parse_sequence_to_array(df["B3-1"], max_length=max_length, dtype=float)
    is_correct = (correct_raw == 1).astype(np.float32)
    channels.append(is_correct)
    
    # ΔRT
    delta = np.diff(rt_raw, axis=1, prepend=np.nan)
    delta[~valid_mask] = np.nan
    delta_mean = np.nanmean(delta)
    delta_std = np.nanstd(delta) + 1e-6
    delta_norm = (np.nan_to_num(delta, nan=delta_mean) - delta_mean) / delta_std
    channels.append(delta_norm.astype(np.float32))
    
    # running_error_rate (벡터화 버전)
    incorrect = (correct_raw == 2).astype(np.float32)
    cumsum = np.cumsum(incorrect, axis=1)  # (N, L)
    t_idx = np.arange(max_length, dtype=np.float32)[None, :] + 1  # 1~L
    running_error = cumsum / t_idx
    running_error[~valid_mask] = 0.0
    channels.append(running_error.astype(np.float32))
    
    # trial_index_norm
    trial_idx = np.arange(max_length, dtype=np.float32)[None, :] / (max_length - 1 + 1e-6)
    trial_idx = np.tile(trial_idx, (n_samples, 1))
    trial_idx[~valid_mask] = 0.0
    channels.append(trial_idx)
    
    # mask
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    
    seq_array = np.stack(channels, axis=-1)  # (N, L, 7)
    return seq_array


def build_sequence_B4(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """
    B4 검사 멀티채널 시퀀스 구성 (10채널)
    - RT_norm, resp_code one-hot(6), ΔRT, trial_index_norm, mask
    - resp_code는 1~6까지의 카테고리 (나중에 mapping 확정되면 refinement 가능)
    """
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    
    required_cols = ["B4-1", "B4-2"]
    if not all(c in df.columns for c in required_cols):
        return None
    
    n_samples = len(df)
    channels = []
    
    # RT 파싱 및 정규화
    rt_raw = parse_sequence_to_array(df["B4-2"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(rt_raw)
    rt_for_stat = rt_raw.copy()
    rt_for_stat[~valid_mask] = np.nan
    rt_mean = np.nanmean(rt_for_stat)
    rt_std = np.nanstd(rt_for_stat) + 1e-6
    rt_norm = (np.nan_to_num(rt_raw, nan=rt_mean) - rt_mean) / rt_std
    channels.append(rt_norm.astype(np.float32))
    
    # resp_code one-hot (B4-1: 1~6)
    code_raw = parse_sequence_to_array(df["B4-1"], max_length=max_length, dtype=float)
    code_1 = (code_raw == 1).astype(np.float32)
    code_2 = (code_raw == 2).astype(np.float32)
    code_3 = (code_raw == 3).astype(np.float32)
    code_4 = (code_raw == 4).astype(np.float32)
    code_5 = (code_raw == 5).astype(np.float32)
    code_6 = (code_raw == 6).astype(np.float32)
    channels.extend([code_1, code_2, code_3, code_4, code_5, code_6])
    
    # ΔRT
    delta = np.diff(rt_raw, axis=1, prepend=np.nan)
    delta[~valid_mask] = np.nan
    delta_mean = np.nanmean(delta)
    delta_std = np.nanstd(delta) + 1e-6
    delta_norm = (np.nan_to_num(delta, nan=delta_mean) - delta_mean) / delta_std
    channels.append(delta_norm.astype(np.float32))
    
    # trial_index_norm
    trial_idx = np.arange(max_length, dtype=np.float32)[None, :] / (max_length - 1 + 1e-6)
    trial_idx = np.tile(trial_idx, (n_samples, 1))
    trial_idx[~valid_mask] = 0.0
    channels.append(trial_idx)
    
    # mask
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    
    seq_array = np.stack(channels, axis=-1)  # (N, L, 10)
    return seq_array


def build_sequence_B5(df: pd.DataFrame, max_length: int = None) -> np.ndarray:
    """
    B5 검사 멀티채널 시퀀스 구성 (7채널)
    - RT_norm, is_correct, ΔRT, running_error_rate, trial_index_norm, mask
    """
    if max_length is None:
        max_length = CNN_CONFIG["max_seq_length"]
    
    required_cols = ["B5-1", "B5-2"]
    if not all(c in df.columns for c in required_cols):
        return None
    
    n_samples = len(df)
    channels = []
    
    # RT 파싱 및 정규화
    rt_raw = parse_sequence_to_array(df["B5-2"], max_length=max_length, dtype=float)
    valid_mask = ~np.isnan(rt_raw)
    rt_for_stat = rt_raw.copy()
    rt_for_stat[~valid_mask] = np.nan
    rt_mean = np.nanmean(rt_for_stat)
    rt_std = np.nanstd(rt_for_stat) + 1e-6
    rt_norm = (np.nan_to_num(rt_raw, nan=rt_mean) - rt_mean) / rt_std
    channels.append(rt_norm.astype(np.float32))
    
    # is_correct (B5-1: 1=correct, 2=incorrect)
    correct_raw = parse_sequence_to_array(df["B5-1"], max_length=max_length, dtype=float)
    is_correct = (correct_raw == 1).astype(np.float32)
    channels.append(is_correct)
    
    # ΔRT
    delta = np.diff(rt_raw, axis=1, prepend=np.nan)
    delta[~valid_mask] = np.nan
    delta_mean = np.nanmean(delta)
    delta_std = np.nanstd(delta) + 1e-6
    delta_norm = (np.nan_to_num(delta, nan=delta_mean) - delta_mean) / delta_std
    channels.append(delta_norm.astype(np.float32))
    
    # running_error_rate (벡터화 버전)
    incorrect = (correct_raw == 2).astype(np.float32)
    cumsum = np.cumsum(incorrect, axis=1)  # (N, L)
    t_idx = np.arange(max_length, dtype=np.float32)[None, :] + 1  # 1~L
    running_error = cumsum / t_idx
    running_error[~valid_mask] = 0.0
    channels.append(running_error.astype(np.float32))
    
    # trial_index_norm
    trial_idx = np.arange(max_length, dtype=np.float32)[None, :] / (max_length - 1 + 1e-6)
    trial_idx = np.tile(trial_idx, (n_samples, 1))
    trial_idx[~valid_mask] = 0.0
    channels.append(trial_idx)
    
    # mask
    mask = valid_mask.astype(np.float32)
    channels.append(mask)
    
    seq_array = np.stack(channels, axis=-1)  # (N, L, 7)
    return seq_array


def build_sequences_A_for_cnn(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    A검사용 CNN 입력 시퀀스 구성
    - A1, A2, A3, A4, A5 모두 멀티채널로 구성
    
    Returns
    -------
    Dict[str, np.ndarray]
        {seq_name: (n_samples, seq_length, num_channels)}
    """
    max_len = CNN_CONFIG["max_seq_length"]
    sequences = {}
    
    # A1: 멀티채널 (9채널)
    seq_A1 = build_sequence_A1(df, max_length=max_len)
    if seq_A1 is not None:
        sequences["A1"] = seq_A1
        print(f"  [A] A1 seq shape = {seq_A1.shape}  # (N, L, 9)")
    
    # A2: 멀티채널 (10채널)
    seq_A2 = build_sequence_A2(df, max_length=max_len)
    if seq_A2 is not None:
        sequences["A2"] = seq_A2
        print(f"  [A] A2 seq shape = {seq_A2.shape}  # (N, L, 11)")
    
    # A3: 멀티채널 (16채널)
    seq_A3 = build_sequence_A3(df, max_length=max_len)
    if seq_A3 is not None:
        sequences["A3"] = seq_A3
        print(f"  [A] A3 seq shape = {seq_A3.shape}  # (N, L, 16)")
    
    # A4: Stroop 멀티채널 (12채널)
    seq_A4 = build_sequence_A4(df, max_length=max_len)
    if seq_A4 is not None:
        sequences["A4"] = seq_A4
        print(f"  [A] A4 seq shape = {seq_A4.shape}  # (N, L, 12)")
    
    # A5: 멀티채널 (10채널)
    seq_A5 = build_sequence_A5(df, max_length=max_len)
    if seq_A5 is not None:
        sequences["A5"] = seq_A5
        print(f"  [A] A5 seq shape = {seq_A5.shape}  # (N, L, 10)")
    
    return sequences


def build_sequences_B_for_cnn(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    B검사용 CNN 입력 시퀀스 구성
    - B1, B2, B3, B4를 멀티채널로 구성
    - B9/B10은 집계형이므로 시퀀스가 아닌 정적 피처로 처리 (여기서는 제외)
    
    Returns
    -------
    Dict[str, np.ndarray]
        {seq_name: (n_samples, seq_length, num_channels)}
    """
    max_len = CNN_CONFIG["max_seq_length"]
    sequences = {}
    
    # B1: 멀티채널 (9채널)
    seq_B1 = build_sequence_B1(df, max_length=max_len)
    if seq_B1 is not None:
        sequences["B1"] = seq_B1
        print(f"  [B] B1 seq shape = {seq_B1.shape}  # (N, L, 9)")
    
    # B2: 멀티채널 (9채널)
    seq_B2 = build_sequence_B2(df, max_length=max_len)
    if seq_B2 is not None:
        sequences["B2"] = seq_B2
        print(f"  [B] B2 seq shape = {seq_B2.shape}  # (N, L, 9)")
    
    # B3: 멀티채널 (7채널)
    seq_B3 = build_sequence_B3(df, max_length=max_len)
    if seq_B3 is not None:
        sequences["B3"] = seq_B3
        print(f"  [B] B3 seq shape = {seq_B3.shape}  # (N, L, 7)")
    
    # B4: 멀티채널 (10채널 - resp_code one-hot(6))
    seq_B4 = build_sequence_B4(df, max_length=max_len)
    if seq_B4 is not None:
        sequences["B4"] = seq_B4
        print(f"  [B] B4 seq shape = {seq_B4.shape}  # (N, L, 10)")
    
    # B5: 멀티채널 (7채널)
    seq_B5 = build_sequence_B5(df, max_length=max_len)
    if seq_B5 is not None:
        sequences["B5"] = seq_B5
        print(f"  [B] B5 seq shape = {seq_B5.shape}  # (N, L, 7)")
    
    # B9/B10은 집계형이므로 시퀀스가 아닌 정적 피처로 처리됨
    
    return sequences

# =============================================================================
# PyTorch Dataset & Model
# =============================================================================

class SequenceDataset(Dataset):
    """시퀀스 데이터와 trial-wise features를 결합한 Dataset"""

    def __init__(self,
                 sequences: Dict[str, np.ndarray],
                 trial_features: np.ndarray,
                 labels: np.ndarray = None):
        """
        Parameters
        ----------
        sequences : Dict[str, np.ndarray]
            - 각 value는 (n_samples, seq_length, num_channels) 또는 (n_samples, seq_length)
        trial_features : np.ndarray
            (n_samples, n_features)
        labels : np.ndarray
            (n_samples,) 또는 None
        """
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
                # (L,) → (1, L)
                arr = arr[None, :]          # (1, L)
            elif arr.ndim == 2:
                # (L, C) → (C, L)
                arr = arr.T                 # (C, L)
            else:
                # 이미 (C, L)이면 그대로 둠
                pass
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
        """
        x: (batch_size, seq_length, hidden_dim)
        Returns: (batch_size, hidden_dim), (batch_size, seq_length)
        """
        u = torch.tanh(self.W(x))              # (B, L, A)
        scores = self.v(u).squeeze(-1)         # (B, L)
        attn_weights = torch.softmax(scores, dim=1)
        attn_output = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)
        return attn_output, attn_weights


class MultiSequenceCNN(nn.Module):
    """여러 시퀀스를 처리하는 1D CNN 모델"""

    def __init__(self, config: Dict, num_trial_features: int):
        super().__init__()
        self.config = config
        self.num_trial_features = num_trial_features

        # 시퀀스별 CNN 모듈
        self.sequence_modules = nn.ModuleDict()

        # Trial-wise features용 MLP
        trial_input_dim = num_trial_features
        mlp_layers = []
        prev_dim = trial_input_dim
        for hidden_dim in config["mlp_hidden_dims"]:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(config["mlp_dropout"]))
            prev_dim = hidden_dim

        self.trial_mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()

        # 최종 분류기 (fit 시점에 실제 input_dim 셋업)
        self.final_classifier = None

    def _build_sequence_cnn(self, seq_name: str, input_dim: int = 1) -> nn.Module:
        """단일 시퀀스용 CNN 모듈 생성"""
        conv_outputs = []
        for filter_size in self.config["filter_sizes"]:
            conv_layers = []
            in_channels = input_dim
            for _ in range(self.config["cnn_layers"]):
                conv_layers.append(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=self.config["num_filters"],
                        kernel_size=filter_size,
                        padding=filter_size // 2,
                    )
                )
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
            attention = AttentionLayer(
                hidden_dim=self.config["num_filters"] * len(self.config["filter_sizes"]),
                attention_dim=self.config["attention_dim"],
            )

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
                # x: (batch_size, C, L) 또는 (batch_size, L)
                if x.dim() == 2:
                    # (B, L) → (B, 1, L)
                    x = x.unsqueeze(1)
                # 그 외에는 (B, C, L)이라고 가정

                conv_outputs = []
                for conv in self.convs:
                    conv_out = conv(x)  # (batch_size, num_filters, L)
                    conv_outputs.append(conv_out)

                if len(conv_outputs) > 1:
                    combined = torch.cat(conv_outputs, dim=1)  # (B, num_filters * len(filter_sizes), L)
                else:
                    combined = conv_outputs[0]

                if self.attention is not None:
                    # (B, C', L) → (B, L, C') for attention
                    combined_t = combined.transpose(1, 2)  # (B, L, C')
                    attn_out, _ = self.attention(combined_t)  # (B, C')
                    return attn_out
                else:
                    # Pooling
                    if self.pool_type == "both":
                        pooled_max = self.pool_max(combined).squeeze(-1)  # (B, C')
                        pooled_avg = self.pool_avg(combined).squeeze(-1)  # (B, C')
                        pooled = torch.cat([pooled_max, pooled_avg], dim=1)  # (B, 2*C')
                    else:
                        pooled = self.pool(combined).squeeze(-1)  # (B, C')
                    return pooled

        return SequenceCNNModule(convs, pool, pool_type, attention)

    def forward(self,
                sequences: Dict[str, torch.Tensor],
                trial_features: torch.Tensor):
        """
        sequences : {seq_name: (B, L)}
        trial_features : (B, num_trial_features)
        """
        seq_outputs = []
        for seq_name, seq_data in sequences.items():
            if seq_name in self.sequence_modules:
                seq_out = self.sequence_modules[seq_name](seq_data)
                seq_outputs.append(seq_out)

        if seq_outputs:
            cnn_features = torch.cat(seq_outputs, dim=1)
        else:
            cnn_features = torch.zeros(
                trial_features.size(0), 1, device=trial_features.device
            )

        trial_out = self.trial_mlp(trial_features)
        combined = torch.cat([cnn_features, trial_out], dim=1)
        logits = self.final_classifier(combined)  # (B, 1)
        return torch.sigmoid(logits)

# =============================================================================
# utils.py 유틸 import
# =============================================================================

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from utils import (
        preprocess_A, preprocess_B,
        add_features_A, add_features_B,
        add_age_normed_composites_A, add_age_normed_composites_B,
        add_history_features_A, add_history_features_B,
        evaluate_score, Metrics,
        separate_num_cat, build_preprocessor,
        add_rowwise_features,
    )
except ImportError:
    print("⚠️  Could not import from utils.py. Please ensure utils.py is in the same directory.")
    raise

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
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def read_index_files() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_idx = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_idx  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train_idx, test_idx


def read_raw_feature_files(split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    A_df = pd.read_csv(os.path.join(DATA_DIR, split, "A.csv"))
    B_df = pd.read_csv(os.path.join(DATA_DIR, split, "B.csv"))
    return A_df, B_df

# =============================================================================
# CNN용 정적 피처 라이트 버전
# =============================================================================

def select_light_features_A(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    A 검사용 CNN 라이트 피처 선택
    - Age 관련 기본 정보
    - 소수의 도메인 composite
    - 강한 indicator 몇 개
    
    Note: Label, PrimaryKey, Test는 df_idx(train_idx)에 있으므로 여기서 제외
    """
    keep_cols = [
        "Test_id",
        "Age_num",
        "PerceptualSpeed_A",
        "CognitiveAbility_A",
        "EmotionalRisk_A",
    ]
    
    # Age_z, Age_bin 등이 있으면 추가
    optional_cols = ["Age_z", "Age_bin", "Old_and_low_A4", "A_low_cog_flag"]
    for col in optional_cols:
        if col in df_feat.columns:
            keep_cols.append(col)
    
    keep_cols = [c for c in keep_cols if c in df_feat.columns]
    return df_feat[keep_cols].copy()


def select_light_features_B(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    B 검사용 CNN 라이트 피처 선택
    - Age 관련 기본 정보
    - 소수의 도메인 composite
    
    Note: Label, PrimaryKey, Test는 df_idx(train_idx)에 있으므로 여기서 제외
    """
    keep_cols = [
        "Test_id",
        "Age_num",
        "MultitaskAbility_B",
        "RiskScore_B_norm",
    ]
    
    # Age_z 등이 있으면 추가
    optional_cols = ["Age_z", "Age_bin"]
    for col in optional_cols:
        if col in df_feat.columns:
            keep_cols.append(col)
    
    keep_cols = [c for c in keep_cols if c in df_feat.columns]
    return df_feat[keep_cols].copy()

# =============================================================================
# 모델 학습 / 예측 함수
# =============================================================================

def train_cnn_model(
    sequences: Dict[str, np.ndarray],
    trial_features: np.ndarray,
    labels: np.ndarray,
    config: Dict,
    device: torch.device,
    validation_data: Tuple = None,
) -> nn.Module:
    """
    CNN 모델 학습
    """
    model = MultiSequenceCNN(config, num_trial_features=trial_features.shape[1])

    # 시퀀스별 CNN 모듈 생성
    for seq_name, seq_data in sequences.items():
        # seq_data: (N, L, C) 또는 (N, L)
        if seq_data.ndim == 3:
            input_dim = seq_data.shape[2]  # num_channels
        else:
            input_dim = 1
        model.sequence_modules[seq_name] = model._build_sequence_cnn(
            seq_name, input_dim=input_dim
        )

    # CNN 출력 차원 계산
    num_filters = config["num_filters"]
    num_filter_sizes = len(config["filter_sizes"])

    if config["use_attention"]:
        seq_output_dim = num_filters * num_filter_sizes
    else:
        if config["pooling_type"] == "both":
            seq_output_dim = num_filters * num_filter_sizes * 2
        else:
            seq_output_dim = num_filters * num_filter_sizes

    cnn_output_dim = seq_output_dim * len(sequences)
    mlp_output_dim = (
        config["mlp_hidden_dims"][-1]
        if config["mlp_hidden_dims"]
        else trial_features.shape[1]
    )
    final_input_dim = cnn_output_dim + mlp_output_dim

    model.final_classifier = nn.Sequential(
        nn.Linear(final_input_dim, 64),
        nn.ReLU(),
        nn.Dropout(config["mlp_dropout"]),
        nn.Linear(64, 1),
    )

    model = model.to(device)

    # Dataset & DataLoader
    train_dataset = SequenceDataset(sequences, trial_features, labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    if validation_data is not None:
        val_sequences, val_trial_features, val_labels = validation_data
        val_dataset = SequenceDataset(val_sequences, val_trial_features, val_labels)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0,
        )
    else:
        val_loader = None

    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0
        for seq_dict, trial_feat, label in train_loader:
            seq_dict = {k: v.to(device) for k, v in seq_dict.items()}
            trial_feat = trial_feat.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(seq_dict, trial_feat)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for seq_dict, trial_feat, label in val_loader:
                    seq_dict = {k: v.to(device) for k, v in seq_dict.items()}
                    trial_feat = trial_feat.to(device)
                    label = label.to(device)

                    output = model(seq_dict, trial_feat)
                    loss = criterion(output, label)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch+1}/{config['num_epochs']}: "
                    f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}"
                )

            # ★ Early stopping용 best model 저장 (깊은 복사)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= config["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch+1}")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
        else:
            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch+1}/{config['num_epochs']}: "
                    f"Train Loss={train_loss:.4f}"
                )

    return model


def predict_cnn_model(
    model: nn.Module,
    sequences: Dict[str, np.ndarray],
    trial_features: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """CNN 모델 예측"""
    model.eval()
    dataset = SequenceDataset(sequences, trial_features, labels=None)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    preds = []
    with torch.no_grad():
        for seq_dict, trial_feat in loader:
            seq_dict = {k: v.to(device) for k, v in seq_dict.items()}
            trial_feat = trial_feat.to(device)

            output = model(seq_dict, trial_feat)
            preds.append(output.cpu().numpy())

    return np.concatenate(preds, axis=0).flatten()

# =============================================================================
# A/B 파티션 학습
# =============================================================================

def fit_partition_A_cnn(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    df_raw: pd.DataFrame = None,
    label_col: str = "Label",
):
    """
    A 테스트용 CNN 모델 학습
    
    Parameters
    ----------
    df_feat : pd.DataFrame
        전처리된 피처 데이터프레임 (라이트 버전)
    df_idx : pd.DataFrame
        인덱스 데이터프레임
    df_raw : pd.DataFrame, optional
        원시 데이터프레임 (시퀀스 컬럼 포함). None이면 df_feat에서 찾음
    label_col : str
        레이블 컬럼명
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for CNN model. Install with: pip install torch"
        )

    which = "A"
    key = "Test_id"
    assert key in df_feat.columns, f"{which}: '{key}' not found in features"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{which}] Using device: {device}")

    # df_idx와 df_feat merge (Label, Test, PrimaryKey 등을 포함)
    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")
    
    # 원시 데이터 준비 (시퀀스 컬럼이 필요)
    if df_raw is None:
        # df_feat에 원시 컬럼이 남아있다고 가정
        df_raw = df_feat.copy()
    else:
        # 원시 데이터도 df_idx와 merge
        df_raw = df_idx.merge(df_raw, on=key, how="left", validate="1:1")

    # Label이 df_idx에 있어야 함 (merge 후 확인)
    if label_col not in df.columns:
        raise ValueError(
            f"[{which}] '{label_col}' not found in merged dataframe. "
            f"Please ensure '{label_col}' is in df_idx (train_idx). "
            f"Available columns in df_idx: {list(df_idx.columns)}, "
            f"Available columns in df after merge: {list(df.columns)}"
        )

    drop_cols = [key, label_col] + (["Test"] if "Test" in df.columns else [])
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Row-wise NA features (시퀀스 문자열 포함 상태지만 내부에서 numeric만 골라 쓰도록 되어 있음)
    df = add_rowwise_features(df, feature_cols)

    # 멀티채널 시퀀스 구성 (원시 데이터 사용)
    print(f"[{which}] Building multi-channel sequences for CNN (A)...")
    sequences = build_sequences_A_for_cnn(df_raw)
    if not sequences:
        raise ValueError("[A] No sequence columns found for CNN.")

    # 시퀀스 원시 컬럼 제거 (멀티채널로 변환했으므로)
    seq_raw_cols = []
    # A1
    if "A1-1" in df.columns:
        seq_raw_cols.extend(["A1-1", "A1-2", "A1-3", "A1-4"])
    # A2
    if "A2-1" in df.columns:
        seq_raw_cols.extend(["A2-1", "A2-2", "A2-3", "A2-4"])
    # A3
    if "A3-1" in df.columns:
        seq_raw_cols.extend(["A3-1", "A3-2", "A3-3", "A3-4", "A3-5", "A3-6", "A3-7"])
    # A4
    if "A4-1" in df.columns:
        seq_raw_cols.extend(["A4-1", "A4-2", "A4-3", "A4-4", "A4-5"])
    # A5
    if "A5-1" in df.columns:
        seq_raw_cols.extend(["A5-1", "A5-2", "A5-3"])
    
    trial_df = df.drop(
        columns=drop_cols + seq_raw_cols, errors="ignore"
    )

    num_cols, cat_cols = separate_num_cat(trial_df, drop_cols)
    preproc = build_preprocessor(num_cols, cat_cols)

    y_all = df[label_col].astype(int).values

    if USE_OOF:
        oof_preds = np.zeros(len(df), dtype=float)

        if "PrimaryKey" not in df.columns:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
            )
            # splitter는 인덱스만 필요하므로 더미 배열 사용
            splitter = skf.split(np.arange(len(df)), y_all)
        else:
            groups = df["PrimaryKey"].values
            gkf = StratifiedGroupKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
            )
            # splitter는 인덱스만 필요하므로 더미 배열 사용
            splitter = gkf.split(np.arange(len(df)), y_all, groups=groups)

        for fold, (tr_idx, val_idx) in enumerate(splitter, 1):
            print(f"\n--- [{which}] Fold {fold}/{N_SPLITS} ---")

            X_tr_trial = preproc.fit_transform(trial_df.iloc[tr_idx])
            X_val_trial = preproc.transform(trial_df.iloc[val_idx])

            seq_tr = {k: v[tr_idx] for k, v in sequences.items()}
            seq_val = {k: v[val_idx] for k, v in sequences.items()}

            model = train_cnn_model(
                sequences=seq_tr,
                trial_features=X_tr_trial,
                labels=y_all[tr_idx],
                config=CNN_CONFIG,
                device=device,
                validation_data=(seq_val, X_val_trial, y_all[val_idx]),
            )

            val_preds = predict_cnn_model(
                model, seq_val, X_val_trial, device
            )
            oof_preds[val_idx] = val_preds

            metrics_fold = evaluate_score(y_all[val_idx], val_preds)
            print(
                f"  Fold {fold} Score: {metrics_fold.score:.5f} "
                f"(AUC={metrics_fold.auc:.5f})"
            )

        metrics = evaluate_score(y_all, oof_preds)
        print(
            f"\n[{which}] Overall OOF Score: {metrics.score:.5f} "
            f"(AUC={metrics.auc:.5f})"
        )

        # ★ OOF 저장 → 스태킹/앙상블용
        oof_df = df[[key, label_col]].copy()
        oof_df["pred_cnn"] = oof_preds
        oof_path_csv = os.path.join(OUTPUT_DIR, f"cnn_oof_{which}.csv")
        oof_df.to_csv(oof_path_csv, index=False)
        np.save(os.path.join(OUTPUT_DIR, f"cnn_oof_{which}.npy"), oof_preds)
        print(f"[{which}] Saved OOF preds → {oof_path_csv}")

        print(f"\n--- [{which}] Training final model on all data ---")
        X_all_trial = preproc.fit_transform(trial_df)
        final_model = train_cnn_model(
            sequences=sequences,
            trial_features=X_all_trial,
            labels=y_all,
            config=CNN_CONFIG,
            device=device,
            validation_data=None,
        )
    else:
        # Holdout 분기
        X_trial = preproc.fit_transform(trial_df)
        idx_all = np.arange(len(X_trial))
        X_tr, X_val, y_tr, y_val, tr_idx, val_idx = train_test_split(
            X_trial,
            y_all,
            idx_all,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_all,
        )

        seq_tr = {k: v[tr_idx] for k, v in sequences.items()}
        seq_val = {k: v[val_idx] for k, v in sequences.items()}

        final_model = train_cnn_model(
            sequences=seq_tr,
            trial_features=X_tr,
            labels=y_tr,
            config=CNN_CONFIG,
            device=device,
            validation_data=(seq_val, X_val, y_val),
        )

        val_preds = predict_cnn_model(
            final_model, seq_val, X_val, device
        )
        metrics = evaluate_score(y_val, val_preds)
        print(
            f"[{which}] Holdout Score: {metrics.score:.5f} "
            f"(AUC={metrics.auc:.5f})"
        )

    # CPU로 옮겨서 저장 (CPU 전용 추론 환경 호환성)
    state_dict_cpu = {k: v.detach().cpu() for k, v in final_model.state_dict().items()}
    
    model_bundle = {
        "model": state_dict_cpu,
        "config": CNN_CONFIG,
        "sequence_names": list(sequences.keys()),
        "preprocessor": preproc,
    }

    with open(A_PREPROC_PATH, "wb") as f:
        pickle.dump(preproc, f, protocol=4)
    with open(A_MODEL_PATH, "wb") as f:
        pickle.dump(model_bundle, f, protocol=4)

    print(f"[{which}] Saved model & preproc -> {A_MODEL_PATH}, {A_PREPROC_PATH}")
    return preproc, final_model, metrics


def fit_partition_B_cnn(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    df_raw: pd.DataFrame = None,
    label_col: str = "Label",
):
    """
    B 테스트용 CNN 모델 학습
    
    Parameters
    ----------
    df_feat : pd.DataFrame
        전처리된 피처 데이터프레임 (라이트 버전)
    df_idx : pd.DataFrame
        인덱스 데이터프레임
    df_raw : pd.DataFrame, optional
        원시 데이터프레임 (시퀀스 컬럼 포함). None이면 df_feat에서 찾음
    label_col : str
        레이블 컬럼명
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for CNN model. Install with: pip install torch"
        )

    which = "B"
    key = "Test_id"
    assert key in df_feat.columns, f"{which}: '{key}' not found in features"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{which}] Using device: {device}")

    # df_idx와 df_feat merge (Label, Test, PrimaryKey 등을 포함)
    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")
    
    # 원시 데이터 준비 (시퀀스 컬럼이 필요)
    if df_raw is None:
        # df_feat에 원시 컬럼이 남아있다고 가정
        df_raw = df_feat.copy()
    else:
        # 원시 데이터도 df_idx와 merge
        df_raw = df_idx.merge(df_raw, on=key, how="left", validate="1:1")

    # Label이 df_idx에 있어야 함 (merge 후 확인)
    if label_col not in df.columns:
        raise ValueError(
            f"[{which}] '{label_col}' not found in merged dataframe. "
            f"Please ensure '{label_col}' is in df_idx (train_idx). "
            f"Available columns in df_idx: {list(df_idx.columns)}, "
            f"Available columns in df after merge: {list(df.columns)}"
        )

    drop_cols = [key, label_col] + (["Test"] if "Test" in df.columns else [])
    feature_cols = [c for c in df.columns if c not in drop_cols]

    df = add_rowwise_features(df, feature_cols)

    # 멀티채널 시퀀스 구성 (원시 데이터 사용)
    print(f"[{which}] Building multi-channel sequences for CNN (B)...")
    sequences = build_sequences_B_for_cnn(df_raw)
    if not sequences:
        raise ValueError("[B] No sequence columns found for CNN.")

    # 시퀀스 원시 컬럼 제거 (멀티채널로 변환했으므로)
    seq_raw_cols = []
    # B1
    if "B1-1" in df.columns:
        seq_raw_cols.extend(["B1-1", "B1-2", "B1-3"])
    # B2
    if "B2-1" in df.columns:
        seq_raw_cols.extend(["B2-1", "B2-2", "B2-3"])
    # B3
    if "B3-1" in df.columns:
        seq_raw_cols.extend(["B3-1", "B3-2"])
    # B4
    if "B4-1" in df.columns:
        seq_raw_cols.extend(["B4-1", "B4-2"])
    # B5
    if "B5-1" in df.columns:
        seq_raw_cols.extend(["B5-1", "B5-2"])
    # B9/B10은 집계형이므로 정적 피처로 유지 (제거하지 않음)
    
    trial_df = df.drop(
        columns=drop_cols + seq_raw_cols, errors="ignore"
    )

    num_cols, cat_cols = separate_num_cat(trial_df, drop_cols)
    preproc = build_preprocessor(num_cols, cat_cols)

    y_all = df[label_col].astype(int).values

    if USE_OOF:
        oof_preds = np.zeros(len(df), dtype=float)

        if "PrimaryKey" not in df.columns:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
            )
            # splitter는 인덱스만 필요하므로 더미 배열 사용
            splitter = skf.split(np.arange(len(df)), y_all)
        else:
            groups = df["PrimaryKey"].values
            gkf = StratifiedGroupKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
            )
            # splitter는 인덱스만 필요하므로 더미 배열 사용
            splitter = gkf.split(np.arange(len(df)), y_all, groups=groups)

        for fold, (tr_idx, val_idx) in enumerate(splitter, 1):
            print(f"\n--- [{which}] Fold {fold}/{N_SPLITS} ---")

            X_tr_trial = preproc.fit_transform(trial_df.iloc[tr_idx])
            X_val_trial = preproc.transform(trial_df.iloc[val_idx])

            seq_tr = {k: v[tr_idx] for k, v in sequences.items()}
            seq_val = {k: v[val_idx] for k, v in sequences.items()}

            model = train_cnn_model(
                sequences=seq_tr,
                trial_features=X_tr_trial,
                labels=y_all[tr_idx],
                config=CNN_CONFIG,
                device=device,
                validation_data=(seq_val, X_val_trial, y_all[val_idx]),
            )

            val_preds = predict_cnn_model(
                model, seq_val, X_val_trial, device
            )
            oof_preds[val_idx] = val_preds

            metrics_fold = evaluate_score(y_all[val_idx], val_preds)
            print(
                f"  Fold {fold} Score: {metrics_fold.score:.5f} "
                f"(AUC={metrics_fold.auc:.5f})"
            )

        metrics = evaluate_score(y_all, oof_preds)
        print(
            f"\n[{which}] Overall OOF Score: {metrics.score:.5f} "
            f"(AUC={metrics.auc:.5f})"
        )

        # ★ OOF 저장 → 스태킹/앙상블용
        oof_df = df[[key, label_col]].copy()
        oof_df["pred_cnn"] = oof_preds
        oof_path_csv = os.path.join(OUTPUT_DIR, f"cnn_oof_{which}.csv")
        oof_df.to_csv(oof_path_csv, index=False)
        np.save(os.path.join(OUTPUT_DIR, f"cnn_oof_{which}.npy"), oof_preds)
        print(f"[{which}] Saved OOF preds → {oof_path_csv}")

        print(f"\n--- [{which}] Training final model on all data ---")
        X_all_trial = preproc.fit_transform(trial_df)
        final_model = train_cnn_model(
            sequences=sequences,
            trial_features=X_all_trial,
            labels=y_all,
            config=CNN_CONFIG,
            device=device,
            validation_data=None,
        )
    else:
        # Holdout 분기
        X_trial = preproc.fit_transform(trial_df)
        idx_all = np.arange(len(X_trial))
        X_tr, X_val, y_tr, y_val, tr_idx, val_idx = train_test_split(
            X_trial,
            y_all,
            idx_all,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_all,
        )

        seq_tr = {k: v[tr_idx] for k, v in sequences.items()}
        seq_val = {k: v[val_idx] for k, v in sequences.items()}

        final_model = train_cnn_model(
            sequences=seq_tr,
            trial_features=X_tr,
            labels=y_tr,
            config=CNN_CONFIG,
            device=device,
            validation_data=(seq_val, X_val, y_val),
        )

        val_preds = predict_cnn_model(
            final_model, seq_val, X_val, device
        )
        metrics = evaluate_score(y_val, val_preds)
        print(
            f"[{which}] Holdout Score: {metrics.score:.5f} "
            f"(AUC={metrics.auc:.5f})"
        )

    # CPU로 옮겨서 저장 (CPU 전용 추론 환경 호환성)
    state_dict_cpu = {k: v.detach().cpu() for k, v in final_model.state_dict().items()}
    
    model_bundle = {
        "model": state_dict_cpu,
        "config": CNN_CONFIG,
        "sequence_names": list(sequences.keys()),
        "preprocessor": preproc,
    }

    with open(B_PREPROC_PATH, "wb") as f:
        pickle.dump(preproc, f, protocol=4)
    with open(B_MODEL_PATH, "wb") as f:
        pickle.dump(model_bundle, f, protocol=4)

    print(f"[{which}] Saved model & preproc -> {B_MODEL_PATH}, {B_PREPROC_PATH}")
    return preproc, final_model, metrics

# =============================================================================
# 메인 함수
# =============================================================================

def main():
    if not HAS_TORCH:
        print("❌ PyTorch is required. Install with: pip install torch")
        return

    set_seed(RANDOM_STATE)
    t0 = time.time()
    ensure_dirs()

    print("📂 Loading index files ...")
    train_idx, _ = read_index_files()

    print("📂 Loading raw A/B (train only) ...")
    A_train_raw, B_train_raw = read_raw_feature_files("train")

    # 풀 FE 파이프라인 (블록 제거 없이 모든 피처 생성)
    print("\n🔵 FE for A (full pipeline) ...")
    t_fe_a = time.time()
    A_train_feat = preprocess_A(A_train_raw)
    A_train_feat = add_features_A(A_train_feat)
    A_train_feat = add_age_normed_composites_A(A_train_feat)
    A_train_feat = add_history_features_A(A_train_feat)
    print(
        f"  A_train_feat: {A_train_feat.shape} "
        f"(elapsed: {(time.time()-t_fe_a)/60:.2f} min)"
    )

    print("\n🟢 FE for B (full pipeline) ...")
    t_fe_b = time.time()
    B_train_feat = preprocess_B(B_train_raw)
    B_train_feat = add_features_B(B_train_feat)
    B_train_feat = add_age_normed_composites_B(B_train_feat)
    B_train_feat = add_history_features_B(B_train_feat)
    print(
        f"  B_train_feat: {B_train_feat.shape} "
        f"(elapsed: {(time.time()-t_fe_b)/60:.2f} min)"
    )

    # CNN용 라이트 피처 선택 (블록 개념 없이 직접 선택)
    print("\n🔵 Selecting light features for CNN (A)...")
    A_train_feat_cnn = select_light_features_A(A_train_feat)
    print(f"  A_train_feat_cnn: {A_train_feat_cnn.shape} (light version)")
    
    print("\n🟢 Selecting light features for CNN (B)...")
    B_train_feat_cnn = select_light_features_B(B_train_feat)
    print(f"  B_train_feat_cnn: {B_train_feat_cnn.shape} (light version)")

    A_train_idx = train_idx[train_idx["Test"] == "A"].copy()
    print("\n[A] Training CNN model ...")
    preproc_A, model_A, metrics_A = fit_partition_A_cnn(
        A_train_feat_cnn, A_train_idx, df_raw=A_train_raw, label_col="Label"
    )

    B_train_idx = train_idx[train_idx["Test"] == "B"].copy()
    print("\n[B] Training CNN model ...")
    preproc_B, model_B, metrics_B = fit_partition_B_cnn(
        B_train_feat_cnn, B_train_idx, df_raw=B_train_raw, label_col="Label"
    )

    n_A = len(A_train_idx)
    n_B = len(B_train_idx)
    weight_A = n_A / (n_A + n_B)
    weight_B = n_B / (n_A + n_B)

    if USE_OOF:
        final_score = weight_A * metrics_A.score + weight_B * metrics_B.score
        final_auc   = weight_A * metrics_A.auc   + weight_B * metrics_B.auc

        print("\n" + "=" * 80)
        print(f"🚀 Overall OOF Score ({N_SPLITS}-Fold CV):")
        print(f"  - A Score: {metrics_A.score:.6f} (AUC={metrics_A.auc:.6f})")
        print(f"  - B Score: {metrics_B.score:.6f} (AUC={metrics_B.auc:.6f})")
        print(f"  - Final Score: {final_score:.6f}")
        print(f"  - Final AUC:   {final_auc:.6f}")
        print("=" * 80)

    meta = {
        "model": "1D CNN (Multi-sequence + Trial-wise features)",
        "validation_mode": "OOF" if USE_OOF else "Holdout",
        "use_oof": USE_OOF,
        "n_splits": N_SPLITS,
        "cnn_config": CNN_CONFIG,
        "random_state": RANDOM_STATE,
        "pytorch_version": torch.__version__ if HAS_TORCH else None,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    dt = time.time() - t0
    print(f"\n✅ Training completed | elapsed: {dt/60:.2f} min")
    print(f"✅ Models saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
