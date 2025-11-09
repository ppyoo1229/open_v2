#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
제출 추론 스크립트: GRU Step 1 + HGB Step 2 독립실행을 위한 수정본

목적:
  - 서버에서 제공한 data/train/*.csv, data/test/*.csv로 FE 수행
  - 5개 fold GRU 모델로 test 데이터 encoding (hidden 평균)
  - GRU features를 A_all에 merge
  - 학습된 HGB로 최종 예측
  - submission.csv 생성

주의사항:
  - 캐시된 parquet 파일 사용하지 않음 (매번 GRU encoding 수행)
  - RTX 3090 최적화 (배치 크기 증가)
  - test 데이터는 총 162,216개 샘플
"""

import os
import time
import json
import warnings
warnings.filterwarnings("ignore")

from typing import Tuple, List
import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import pickle

# =============================================================================
# 공통 유틸 (best08.py에서 복사 - 독립 실행을 위해)
# =============================================================================

def read_index_files() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_idx = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_idx  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train_idx, test_idx

def separate_num_cat(df: pd.DataFrame, drop_cols: List[str]) -> Tuple[List[str], List[str]]:
    cols = [c for c in df.columns if c not in drop_cols]
    cat_cols = [c for c in cols if str(df[c].dtype) in ("object", "category")]
    num_cols = [c for c in cols if c not in cat_cols]
    return num_cols, cat_cols

def add_rowwise_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols]
    na_count = X.isna().sum(axis=1).astype(np.int32)
    na_ratio = (na_count / (len(feature_cols) + 1e-9)).astype(np.float32)
    df2 = df.copy()
    df2["NA_COUNT"] = na_count
    df2["NA_RATIO"] = na_ratio
    return df2

# =============================================================================
# FE 유틸 (best08.py에서 복사)
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
# A/B 원시 → 기본 파생 (best08.py에서 복사)
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

    if "Age_num" in feats.columns:
        feats["Age_60plus"] = (feats["Age_num"] >= 60).astype(int)
    if "A4_acc_rate" in feats.columns:
        feats["A4_low_acc_flag"] = (feats["A4_acc_rate"] <= 0.8).astype(int)
    if "CogScore_A" in feats.columns:
        feats["A_low_cog_flag"] = (feats["CogScore_A"] <= 26).astype(int)
    if _has(feats, ["Age_num", "A4_acc_rate"]):
        feats["Old_and_low_A4"] = (
            (feats["Age_num"] >= 60) &
            (feats["A4_acc_rate"] <= 0.8)
        ).astype(int)

    feats.replace([np.inf,-np.inf], np.nan, inplace=True)
    return feats

def add_features_B(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy(); eps = 1e-6

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

    if _has(feats, ["B10_aud_overall_acc", "B9_aud_overall_acc"]):
        feats["B10_multitask_ratio_aud"] = _safe_div(
            feats["B10_aud_overall_acc"],
            feats["B9_aud_overall_acc"],
            eps
        )

    if "Age_num" in feats.columns:
        feats["Age_60plus"] = (feats["Age_num"] >= 60).astype(int)
    if "Age_num" in feats.columns:
        feats["Age_80plus"] = (feats["Age_num"] >= 80).astype(int)
    if "RiskScore_B" in feats.columns:
        try:
            thr = feats["RiskScore_B"].quantile(0.8)
            feats["RiskB_top20"] = (feats["RiskScore_B"] >= thr).astype(int)
        except Exception:
            feats["RiskB_top20"] = 0
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
    """A history features using PrimaryKey + YearMonthIndex"""
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
    """B history features using PrimaryKey + YearMonthIndex"""
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
        df["B4_acc_trend_per_month"] = df["B4_acc_trend"] / (df["months_since_prev_test"] + 1e-6)
        df["B4_acc_trend_ratio"] = df["B4_acc_trend"] / (df["B4_acc_prev_mean"].abs() + 1e-6)

    if "B10_multitask_cost_aud" in df.columns:
        df["B10_multicost_prev_mean"] = (
            df.groupby("PrimaryKey")["B10_multitask_cost_aud"]
            .transform(lambda x: x.expanding().mean().shift(1))
        )
        df["B10_multicost_trend"] = (
            df["B10_multitask_cost_aud"] - df["B10_multicost_prev_mean"]
        )
        df["B10_multicost_trend_per_month"] = df["B10_multicost_trend"] / (df["months_since_prev_test"] + 1e-6)
        df["B10_multicost_trend_ratio"] = df["B10_multicost_trend"] / (df["B10_multicost_prev_mean"].abs() + 1e-6)

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
            df[f"{task}_rt_trend_per_month"] = df[f"{task}_rt_trend"] / (df["months_since_prev_test"] + 1e-6)
            df[f"{task}_rt_trend_ratio"] = df[f"{task}_rt_trend"] / (df[f"{task}_rt_prev_mean"].abs() + 1e-6)

    return df

# =============================================================================
# 경로 설정
# =============================================================================

DATA_DIR   = "data"
OUTPUT_DIR = "output"
MODEL_DIR  = "model"

SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

A_GRU_SCALER_PATH = os.path.join(MODEL_DIR, "gru_step1_A_scaler.pkl")
A_GRU_MODEL_BASE  = os.path.join(MODEL_DIR, "gru_A_fold{}.pt")

A_MODEL_PATH    = os.path.join(MODEL_DIR, "hgb_A.pkl")
B_MODEL_PATH    = os.path.join(MODEL_DIR, "hgb_B.pkl")
A_PREPROC_PATH  = os.path.join(MODEL_DIR, "preproc_A.pkl")
B_PREPROC_PATH  = os.path.join(MODEL_DIR, "preproc_B.pkl")

RANDOM_STATE = 42
N_SPLITS = 5

# 디바이스 자동 감지 (CPU/GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CPU = (DEVICE.type == "cpu")
print(f"Using device: {DEVICE} ({'CPU' if IS_CPU else 'GPU'})")

# GRU 설정 
SEQ_FEATURES_A = [
    "Age_num",
    "YearMonthIndex",
    "n_tests_so_far",
    "months_since_first_test",
    "months_since_prev_test",
    "A4_acc_rate",
    "CogScore_A",
    "A1_rt_mean",
    "A4_stroop_diff",
    "A5_change_nonchange_gap",
    "A3_valid_invalid_ratio_gap",
    "A4_acc_prev_mean",
    "A4_stroop_prev_mean",
    "A1_rt_prev_mean",
    "CogScore_A_prev_mean",
    "A4_stroop_trend",
    "A1_rt_trend",
    "CogScore_A_trend",
]
GRU_CONFIG_A = {
    "SEQ_FEATURES": SEQ_FEATURES_A,
    "HIDDEN_DIM": 64,
    "NUM_LAYERS": 2,
}
if IS_CPU:
    GRU_BATCH_SIZE = 64 
    NUM_WORKERS = 0 
else:
    GRU_BATCH_SIZE = 512 
    NUM_WORKERS = 2  

# =============================================================================
# Age-based blending 
# =============================================================================
AGE_SPLIT_A       = 60.0
AGE_BUFFER_MIN_A  = 55.0
AGE_BUFFER_MAX_A  = 65.0

def blend_by_age_with_buffer(
    age_array: np.ndarray,
    p_under: np.ndarray,
    p_over: np.ndarray,
    split: float = AGE_SPLIT_A,
    buf_min: float = AGE_BUFFER_MIN_A,
    buf_max: float = AGE_BUFFER_MAX_A,
) -> np.ndarray:
    """best08.py의 blend_by_age_with_buffer"""
    age = np.asarray(age_array, dtype=float)
    p_under = np.asarray(p_under, dtype=float)
    p_over  = np.asarray(p_over, dtype=float)

    w_over = np.zeros_like(age, dtype=float)
    mask_nan = np.isnan(age)
    w_over[mask_nan] = 0.0

    mask_low  = (~mask_nan) & (age <= buf_min)
    mask_high = (~mask_nan) & (age >= buf_max)
    w_over[mask_low]  = 0.0
    w_over[mask_high] = 1.0

    mask_buf = (~mask_nan) & (age > buf_min) & (age < buf_max)
    w_over[mask_buf] = (age[mask_buf] - buf_min) / (buf_max - buf_min)

    w_under = 1.0 - w_over
    blended = w_under * p_under + w_over * p_over
    return blended

# =============================================================================
# GRU 모델 (train_dl_step1.py와 동일)
# =============================================================================

class HistoryGRUClassifier(nn.Module):
    """train_dl_step1.py와 동일한 모델 구조"""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out, hidden = self.gru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            gru_out, batch_first=True
        )
        batch_size, max_len, hidden_dim = gru_out.shape
        gru_out_flat = gru_out.reshape(-1, hidden_dim)
        logits_flat = self.mlp(gru_out_flat).squeeze(-1)
        logits = logits_flat.reshape(batch_size, max_len)
        return logits, gru_out

# =============================================================================
# GRU Encoding
# =============================================================================

def build_sequences_for_inference(
    df_all: pd.DataFrame,
    feat_cols: List[str],
) -> Tuple[List[np.ndarray], List[str], List[pd.DataFrame]]:
    """
    PrimaryKey별 시퀀스 생성 (추론용, Label 없음)
    
    Returns:
        sequences: List of (T_i, D) arrays
        primary_keys: List of PrimaryKey strings
        test_id_maps: List of DataFrames (각 시퀀스의 각 시점에 해당하는 Test_id)
    """
    feat_cols = [c for c in feat_cols if c in df_all.columns]
    if not feat_cols:
        raise ValueError("No valid sequence features found.")
    
    df = df_all.copy()
    df = df.sort_values(["PrimaryKey", "YearMonthIndex"]).reset_index(drop=True)
    
    sequences, primary_keys, test_id_maps = [], [], []
    
    for pk, g in df.groupby("PrimaryKey", sort=False):
        g = g.sort_values("YearMonthIndex").reset_index(drop=True)
        
        if len(g) == 0:
            continue
        
        X = g[feat_cols].to_numpy(dtype=np.float32)
        sequences.append(X)
        primary_keys.append(pk)
        
        # 각 시점의 Test_id 저장
        test_id_map = g[["Test_id"]].copy()
        test_id_maps.append(test_id_map)
    
    return sequences, primary_keys, test_id_maps

def encode_with_gru(
    df_all: pd.DataFrame,
    scaler: StandardScaler,
    model: nn.Module,
    config: dict,
    batch_size: int = 512,
) -> pd.DataFrame:
    print("  Encoding with GRU...")
    
    feat_cols = config["SEQ_FEATURES"]
    feat_cols = [c for c in feat_cols if c in df_all.columns]
    
    # 시퀀스 생성
    sequences, primary_keys, test_id_maps = build_sequences_for_inference(df_all, feat_cols)
    print(f"    Total sequences: {len(sequences)}")
    
    # Scaler 적용
    sequences_scaled = []
    for seq in sequences:
        seq_scaled = scaler.transform(seq)
        seq_scaled = np.nan_to_num(seq_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        sequences_scaled.append(seq_scaled)
    
    # Dataset & DataLoader
    class InferenceDataset(Dataset):
        def __init__(self, sequences):
            self.sequences = sequences
        def __len__(self):
            return len(self.sequences)
        def __getitem__(self, idx):
            return self.sequences[idx]
    
    def collate_fn_inference(batch):
        seqs = batch
        seq_lengths = [len(seq) for seq in seqs]
        max_len = max(seq_lengths) if seq_lengths else 1
        feat_dim = seqs[0].shape[1] if len(seqs) > 0 and len(seqs[0]) > 0 else 0
        
        batch_size = len(seqs)
        padded_seqs = np.zeros((batch_size, max_len, feat_dim), dtype=np.float32)
        
        for i, seq in enumerate(seqs):
            seq_len = len(seq)
            padded_seqs[i, :seq_len, :] = seq
        
        return (
            torch.from_numpy(padded_seqs),
            torch.tensor(seq_lengths, dtype=torch.long),
        )
    
    dataset = InferenceDataset(sequences_scaled)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_inference,
        num_workers=NUM_WORKERS,  
        pin_memory=(not IS_CPU),  
    )
    
    # Encoding
    model.eval()
    all_preds = {}
    all_hiddens = {}
    hidden_dim = config["HIDDEN_DIM"]
    
    print(f"    Total batches: {len(loader)} (batch_size={batch_size})")
    
    with torch.no_grad():
        seq_idx = 0
        for batch_idx, (x_padded, lengths) in enumerate(loader):
            if batch_idx % max(1, len(loader) // 10) == 0 or batch_idx == len(loader) - 1:
                print(f"    Processing batch {batch_idx + 1}/{len(loader)}...")
            
            if IS_CPU:
                x_padded = x_padded.to(DEVICE)
            else:
                x_padded = x_padded.to(DEVICE, non_blocking=True)
            
            logits, hiddens = model(x_padded, lengths)
            probs = torch.sigmoid(logits)
            
            for i in range(len(lengths)):
                seq_len = lengths[i].item()
                pk = primary_keys[seq_idx]
                test_id_map = test_id_maps[seq_idx]
                
                for t in range(seq_len):
                    tid = test_id_map.iloc[t]["Test_id"]
                    pred = probs[i, t].item()
                    hidden_vec = hiddens[i, t].cpu().numpy()
                    
                    all_preds[tid] = pred
                    all_hiddens[tid] = hidden_vec
                
                seq_idx += 1
    
    # DataFrame 생성
    result_df = pd.DataFrame({
        "Test_id": list(all_preds.keys()),
        "gru_A_pred": list(all_preds.values()),
    })
    
    # Hidden features 추가
    for i in range(hidden_dim):
        result_df[f"gru_A_h{i}"] = [all_hiddens[tid][i] for tid in all_preds.keys()]
    
    return result_df

def encode_with_gru_ensemble(
    df_all: pd.DataFrame,
    scaler: StandardScaler,
    config: dict,
) -> pd.DataFrame:
    """
    5개 fold GRU 모델로 encoding → hidden 평균
    """
    env_type = "CPU" if IS_CPU else "GPU"
    print(f"🔵 Encoding with {N_SPLITS}-fold GRU ensemble ({env_type}, batch_size={GRU_BATCH_SIZE})...")
    
    hidden_dim = config["HIDDEN_DIM"]
    feat_cols = config["SEQ_FEATURES"]
    feat_cols = [c for c in feat_cols if c in df_all.columns]
    input_dim = len(feat_cols)
    
    # 각 fold 모델 로드 및 encoding
    all_preds_list = []
    all_hiddens_list = []
    test_ids_list = None
    
    for fold in range(N_SPLITS):
        model_path = A_GRU_MODEL_BASE.format(fold)
        if not os.path.exists(model_path):
            print(f"  ⚠️  Warning: {model_path} not found. Skipping fold {fold}.")
            continue
        
        print(f"  Fold {fold + 1}/{N_SPLITS}...")
        model = HistoryGRUClassifier(
            input_dim=input_dim,
            hidden_dim=config["HIDDEN_DIM"],
            num_layers=config["NUM_LAYERS"],
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        
        # Encoding
        encoded_df = encode_with_gru(df_all, scaler, model, config, batch_size=GRU_BATCH_SIZE)
        
        if test_ids_list is None:
            test_ids_list = encoded_df["Test_id"].values
        
        all_preds_list.append(encoded_df["gru_A_pred"].values)
        all_hiddens_list.append(encoded_df[[f"gru_A_h{i}" for i in range(hidden_dim)]].values)
        
        # 메모리 정리
        del model
        if not IS_CPU:
            torch.cuda.empty_cache()
    
    if len(all_preds_list) == 0:
        raise ValueError("No GRU models found!")
    
    # 평균 계산
    print("  Averaging predictions and hidden features...")
    preds_avg = np.mean(all_preds_list, axis=0)
    hiddens_avg = np.mean(all_hiddens_list, axis=0)
    
    # 결과 DataFrame
    result_df = pd.DataFrame({
        "Test_id": test_ids_list,
        "gru_A_pred": preds_avg,
    })
    
    for i in range(hidden_dim):
        result_df[f"gru_A_h{i}"] = hiddens_avg[:, i]
    
    return result_df

# =============================================================================
# HGB 예측 함수 (best08.py와 동일)
# =============================================================================

def predict_partition_A_submodels(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    preproc,
    ens_under,
    ens_over,
) -> pd.DataFrame:
    """best08.py의 predict_partition_A_submodels"""
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
    """best08.py의 predict_partition_B_submodels"""
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
        proba = np.clip(ens_first.predict_proba(X_t)[:, 1], 1e-7, 1-1e-7)
    else:
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
# 메인
# =============================================================================

def main():
    print("=" * 80)
    print("제출 추론 스크립트: GRU Step 1 + HGB Step 2")
    print("=" * 80)
    print(f"Environment: {DEVICE} ({'CPU' if IS_CPU else 'GPU'})")
    print(f"GRU Batch Size: {GRU_BATCH_SIZE}")
    if IS_CPU:
        print("CPU 환경")

    print("=" * 80)
    
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 데이터 로드
    print("\n Loading data...")
    train_idx, test_idx = read_index_files()
    
    A_train_raw = pd.read_csv(os.path.join(DATA_DIR, "train", "A.csv"))
    A_test_raw  = pd.read_csv(os.path.join(DATA_DIR, "test", "A.csv"))
    B_train_raw = pd.read_csv(os.path.join(DATA_DIR, "train", "B.csv"))
    B_test_raw  = pd.read_csv(os.path.join(DATA_DIR, "test", "B.csv"))
    
    print(f"  A_train: {len(A_train_raw)}, A_test: {len(A_test_raw)}")
    print(f"  B_train: {len(B_train_raw)}, B_test: {len(B_test_raw)}")
    
    # 2. FE 수행
    t_fe_start = time.time()
    print("\n FE for A...")
    A_all_raw = pd.concat([A_train_raw, A_test_raw], axis=0, ignore_index=True)
    A_all = preprocess_A(A_all_raw)
    A_all = add_features_A(A_all)
    A_all = add_history_features_A(A_all)
    
    # A_all에 GRU features를 merge하기 전에 test_ids 저장
    A_train_ids = set(A_train_raw["Test_id"])
    A_test_ids = set(A_test_raw["Test_id"])
    
    print("\n FE for B...")
    B_all_raw = pd.concat([B_train_raw, B_test_raw], axis=0, ignore_index=True)
    B_all = preprocess_B(B_all_raw)
    B_all = add_features_B(B_all)
    B_all = add_history_features_B(B_all)
    
    B_train_ids = set(B_train_raw["Test_id"])
    B_test_ids = set(B_test_raw["Test_id"])
    B_test_feat = B_all[B_all["Test_id"].isin(B_test_ids)].reset_index(drop=True)
    print(f"  B_test_feat: {B_test_feat.shape}")
    t_fe_elapsed = time.time() - t_fe_start
    print(f" FE time: {t_fe_elapsed/60:.2f} min")
    
    # 3. GRU Scaler 로드
    print("\n Loading GRU scaler...")
    if not os.path.exists(A_GRU_SCALER_PATH):
        raise FileNotFoundError(f"GRU scaler not found: {A_GRU_SCALER_PATH}")
    scaler = joblib.load(A_GRU_SCALER_PATH)
    
    # 4. GRU Encoding (5개 fold 앙상블) - A_all 전체 encoding
    print("\n GRU Encoding for A_all (train+test concat)...")
    t_gru_start = time.time()
    
    # A_all 전체를 GRU encoding (train+test 모두 포함)
    gru_features_all = encode_with_gru_ensemble(A_all, scaler, GRU_CONFIG_A)
    t_gru_elapsed = time.time() - t_gru_start
    print(f" GRU features shape (all): {gru_features_all.shape}")
    print(f" GRU encoding time: {t_gru_elapsed/60:.2f} min")
    
    # 5. GRU features를 A_all에 merge
    print("\n  Merging GRU features into A_all...")
    A_all = A_all.merge(gru_features_all, on="Test_id", how="left", validate="1:1")
    print(f"  A_all shape after GRU merge: {A_all.shape}")
    
    # GRU features가 없는 경우 기본값
    gru_cols = [f"gru_A_h{i}" for i in range(64)] + ["gru_A_pred"]
    missing_gru = A_all["gru_A_pred"].isna().sum()
    if missing_gru > 0:
        print(f" Warning: {missing_gru} samples missing GRU features (filling with 0)")
        A_all[gru_cols] = A_all[gru_cols].fillna(0.0)
    
    # 6. Test 데이터만 필터링 (A_test_feat)
    A_test_feat = A_all[A_all["Test_id"].isin(A_test_ids)].reset_index(drop=True)
    print(f"  A_test_feat shape (extracted from A_all): {A_test_feat.shape}")
    
    # 7. HGB 모델 로드
    print("\n Loading HGB models...")
    if not os.path.exists(A_MODEL_PATH):
        raise FileNotFoundError(f"HGB A model not found: {A_MODEL_PATH}")
    if not os.path.exists(B_MODEL_PATH):
        raise FileNotFoundError(f"HGB B model not found: {B_MODEL_PATH}")
    if not os.path.exists(A_PREPROC_PATH):
        raise FileNotFoundError(f"HGB A preproc not found: {A_PREPROC_PATH}")
    if not os.path.exists(B_PREPROC_PATH):
        raise FileNotFoundError(f"HGB B preproc not found: {B_PREPROC_PATH}")
    
    with open(A_MODEL_PATH, "rb") as f:
        model_bundle_A = pickle.load(f)
    with open(B_MODEL_PATH, "rb") as f:
        model_bundle_B = pickle.load(f)
    with open(A_PREPROC_PATH, "rb") as f:
        preproc_A = pickle.load(f)
    with open(B_PREPROC_PATH, "rb") as f:
        preproc_B = pickle.load(f)
    
    # 8. 예측
    print("\n Predicting...")
    A_test_idx = test_idx[test_idx["Test"] == "A"].copy()
    B_test_idx = test_idx[test_idx["Test"] == "B"].copy()
    
    # A 예측
    if model_bundle_A.get("mode") == "age_split":
        ens_A_under = model_bundle_A["ensemble_under60"]
        ens_A_over = model_bundle_A["ensemble_over60"]
        preds_A = predict_partition_A_submodels(
            A_test_feat, A_test_idx, preproc_A, ens_A_under, ens_A_over
        ) if len(A_test_idx) else None
    else:
        ens_A = model_bundle_A["ensemble"]
        preds_A = predict_partition_A_submodels(
            A_test_feat, A_test_idx, preproc_A, ens_A, None
        ) if len(A_test_idx) else None
    
    # B 예측
    if model_bundle_B.get("mode") == "history_split":
        ens_B_first = model_bundle_B["ensemble_first"]
        ens_B_repeat = model_bundle_B["ensemble_repeat"]
        preds_B = predict_partition_B_submodels(
            B_test_feat, B_test_idx, preproc_B, ens_B_first, ens_B_repeat
        ) if len(B_test_idx) else None
    else:
        ens_B = model_bundle_B["ensemble"]
        preds_B = predict_partition_B_submodels(
            B_test_feat, B_test_idx, preproc_B, ens_B, None
        ) if len(B_test_idx) else None
    
    # 9. 제출 파일 생성
    print("\n Creating submission...")
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
    
    # 기본값 채우기 (혹시 모를 경우)
    sub["Label"] = sub["Label"].fillna(0.001)
    sub["Label"] = sub["Label"].clip(lower=1e-7, upper=1-1e-7)
    
    sub.to_csv(SUBMISSION_PATH, index=False)
    
    dt = time.time() - t0
    print(f"\n Submission saved: {SUBMISSION_PATH}")
    print(f"  Submission shape: {sub.shape}")
    print(f"  Label range: [{sub['Label'].min():.6f}, {sub['Label'].max():.6f}]")
    print(f"\n 시간 요약:")
    print(f"  Total elapsed: {dt/60:.2f} min ({dt:.1f} sec)")
    
    # 시간 제한 확인
    TIME_LIMIT = 1800  # 30분 = 1800초
    if IS_CPU:
        if dt > TIME_LIMIT:
            print(f"\n  경고: 실행 시간 30분 초과")
            print(f"   실제: {dt/60:.2f} min > 목표: {TIME_LIMIT/60:.0f} min")
            print(f"   초과: {(dt - TIME_LIMIT)/60:.2f} min")
        else:
            print(f"\n 실행 시간 30분 이내")
            print(f"   실제: {dt/60:.2f} min / 목표: {TIME_LIMIT/60:.0f} min")
            print(f"   여유: {(TIME_LIMIT - dt)/60:.2f} min")

if __name__ == "__main__":
    main()

