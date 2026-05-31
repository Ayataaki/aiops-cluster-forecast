"""
scripts/generate_synthetic.py

Generates a realistic synthetic CPU-usage time-series that mimics the
statistical properties of the Google Cluster trace (v3):
  - Daily seasonality (business-hours peak, night-time trough)
  - Weekly seasonality (weekday vs weekend)
  - Slow upward trend
  - Heavy-tailed noise (occasional burst spikes)

Output: data/instance_usage_sample.csv  (ready for load_instance_usage)

Run:
    python scripts/generate_synthetic.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_DIR, DATETIME_COL, TARGET_COL, FREQ

np.random.seed(42)

# ── Parameters ────────────────────────────────────────────────────────
N_DAYS = 30          # one month of 5-min data  → 30×288 = 8 640 rows
PERIODS_PER_DAY = 288


def make_series(n_days: int = N_DAYS) -> pd.DataFrame:
    n = n_days * PERIODS_PER_DAY
    t = np.arange(n)

    # ── Trend (slow ramp) ─────────────────────────────────────────────
    trend = 0.10 + t / n * 0.12        # from 10 % to 22 % NCU

    # ── Daily seasonality ─────────────────────────────────────────────
    daily = 0.15 * np.sin(2 * np.pi * t / PERIODS_PER_DAY - np.pi / 2)

    # ── Weekly seasonality (weekend dip) ─────────────────────────────
    day_of_week = (t // PERIODS_PER_DAY) % 7
    weekend_mask = (day_of_week >= 5).astype(float)
    weekly = -0.05 * weekend_mask

    # ── Gaussian noise ────────────────────────────────────────────────
    noise = np.random.normal(0, 0.02, n)

    # ── Random burst spikes (Poisson-driven) ─────────────────────────
    n_bursts = np.random.poisson(lam=n_days * 3)   # ~3 bursts/day avg
    burst_idx = np.random.choice(n, n_bursts, replace=False)
    burst_heights = np.random.exponential(scale=0.15, size=n_bursts)
    bursts = np.zeros(n)
    for idx, h in zip(burst_idx, burst_heights):
        # Each burst lasts 1-6 periods
        length = np.random.randint(1, 7)
        bursts[idx : idx + length] += h * np.linspace(1, 0.2, length)

    cpu = trend + daily + weekly + noise + bursts
    cpu = np.clip(cpu, 0.0, 1.0)   # NCU in [0, 1]

    # ── Build DatetimeIndex (arbitrary start = 2019-05-01 00:00 UTC) ──
    idx = pd.date_range("2019-05-01", periods=n, freq=FREQ, tz="UTC")

    df = pd.DataFrame(
        {
            DATETIME_COL: idx,
            TARGET_COL: cpu,
            "avg_mem": np.clip(cpu * 0.6 + np.random.normal(0, 0.02, n), 0, 1),
        }
    )

    # ── Simulate raw start_time in microseconds (trace format) ────────
    base_us = int(600e6)   # 600 seconds offset
    step_us = 300_000_000  # 5 min in microseconds
    df["start_time"] = base_us + np.arange(n) * step_us

    return df


if __name__ == "__main__":
    out_path = DATA_DIR / "instance_usage_sample.csv"
    df = make_series()
    df.to_csv(out_path, index=False)
    print(f"✅  Synthetic dataset saved → {out_path}")
    print(f"    Shape : {df.shape}")
    print(df.head(3).to_string())
