"""
src/analytics/feature_engineer.py

Builds the supervised feature matrix for ML/XGBoost models:
  - Lag features          (Y_{t-1} … Y_{t-p})
  - Rolling statistics    (mean, std, min, max over multiple windows)
  - Calendar features     (hour, day-of-week, is_weekend, is_business_hour)
  - Fourier features      (sin/cos pairs for daily & weekly cycles)
  - Target encoding       (shift to avoid leakage)

Also contains the train/val/test splitter that enforces chronological order.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import (
    DATETIME_COL,
    TARGET_COL,
    TRAIN_RATIO,
    VAL_RATIO,
    HORIZON,
)


# ── Lag features ──────────────────────────────────────────────────────

def add_lags(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Add lag columns Y_{t-k} for each k in lags."""
    if lags is None:
        # Lags covering: 5m, 10m, 30m, 1h, 2h, 6h, 12h, 24h, 48h
        lags = [1, 2, 6, 12, 24, 72, 144, 288, 576]
    for lag in lags:
        df[f"lag_{lag}"] = df[TARGET_COL].shift(lag)
    return df


# ── Rolling statistics ────────────────────────────────────────────────

def add_rolling(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Add rolling mean, std, min, max for each window size (in periods)."""
    if windows is None:
        windows = [6, 12, 24, 48, 288]   # 30m, 1h, 2h, 4h, 24h
    for w in windows:
        rolled = df[TARGET_COL].shift(1).rolling(w, min_periods=w // 2)
        df[f"roll_mean_{w}"] = rolled.mean()
        df[f"roll_std_{w}"]  = rolled.std()
        df[f"roll_min_{w}"]  = rolled.min()
        df[f"roll_max_{w}"]  = rolled.max()
    return df


# ── Exponential weighted features ────────────────────────────────────

def add_ewm(df: pd.DataFrame, spans: list[int] | None = None) -> pd.DataFrame:
    """Add EWM mean for decay spans (in periods)."""
    if spans is None:
        spans = [12, 48, 288]
    for span in spans:
        df[f"ewm_{span}"] = df[TARGET_COL].shift(1).ewm(span=span).mean()
    return df


# ── Calendar features ─────────────────────────────────────────────────

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Encode temporal structure."""
    dt = df[DATETIME_COL]
    df["hour"]             = dt.dt.hour
    df["minute"]           = dt.dt.minute
    df["day_of_week"]      = dt.dt.dayofweek          # Mon=0, Sun=6
    df["day_of_month"]     = dt.dt.day
    df["week_of_year"]     = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"]       = (dt.dt.dayofweek >= 5).astype(int)
    df["is_business_hour"] = (
        (dt.dt.hour >= 8) & (dt.dt.hour < 19) & (dt.dt.dayofweek < 5)
    ).astype(int)
    return df


# ── Fourier features ──────────────────────────────────────────────────

def add_fourier(
    df: pd.DataFrame,
    daily_terms: int = 3,
    weekly_terms: int = 2,
) -> pd.DataFrame:
    """
    Add sin/cos Fourier terms for:
      - Daily  cycle  (288 periods = 24 h at 5-min resolution)
      - Weekly cycle  (2016 periods = 7 days)
    """
    t = np.arange(len(df))
    for k in range(1, daily_terms + 1):
        df[f"sin_daily_{k}"]  = np.sin(2 * np.pi * k * t / 288)
        df[f"cos_daily_{k}"]  = np.cos(2 * np.pi * k * t / 288)
    for k in range(1, weekly_terms + 1):
        df[f"sin_weekly_{k}"] = np.sin(2 * np.pi * k * t / 2016)
        df[f"cos_weekly_{k}"] = np.cos(2 * np.pi * k * t / 2016)
    return df


# ── Multi-step target ─────────────────────────────────────────────────

def add_multistep_targets(
    df: pd.DataFrame, horizon: int = HORIZON
) -> tuple[pd.DataFrame, list[str]]:
    """
    Add columns y_h1, y_h2, … y_hH as direct multi-step targets.
    Returns (df_with_targets, target_cols).
    """
    target_cols = []
    for h in range(1, horizon + 1):
        col = f"y_h{h}"
        df[col] = df[TARGET_COL].shift(-h)
        target_cols.append(col)
    return df, target_cols


# ── Full feature matrix builder ───────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    horizon: int = HORIZON,
    add_targets: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Apply all feature engineering steps in sequence.

    Returns
    -------
    df_feat      : feature matrix (rows with NaN dropped)
    feature_cols : list of input feature column names
    target_cols  : list of target column names (y_h1 … y_hH)
    """
    df = df.copy().reset_index(drop=True)

    logger.info("Building feature matrix …")
    df = add_lags(df)
    df = add_rolling(df)
    df = add_ewm(df)
    df = add_calendar(df)
    df = add_fourier(df)

    if add_targets:
        df, target_cols = add_multistep_targets(df, horizon=horizon)
    else:
        target_cols = []

    # Drop rows with NaN (from lags at the start & targets at the end)
    df = df.dropna().reset_index(drop=True)

    exclude = {DATETIME_COL, TARGET_COL} | set(target_cols)
    feature_cols = [c for c in df.columns if c not in exclude]

    logger.info(
        f"Feature matrix: {len(df):,} rows × {len(feature_cols)} features  "
        f"| {len(target_cols)} targets"
    )
    return df, feature_cols, target_cols


# ── Chronological train/val/test split ───────────────────────────────

def time_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict chronological split — NO shuffling.
    Prevents any form of data leakage.
    """
    n = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = df.iloc[:n_train].copy()
    val   = df.iloc[n_train : n_train + n_val].copy()
    test  = df.iloc[n_train + n_val :].copy()

    logger.info(
        f"Split → train:{len(train):,} | val:{len(val):,} | test:{len(test):,}"
    )
    return train, val, test
