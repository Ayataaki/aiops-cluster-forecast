"""
src/analytics/diagnostics.py

Full statistical diagnostic pipeline for a univariate CPU-usage series:
  1. Time-series decomposition  (trend / seasonality / residual)
  2. Stationarity tests         (ADF, KPSS)
  3. Autocorrelation analysis   (ACF / PACF)
  4. Automatic differencing     (if non-stationary)
  5. HTML + PNG report saved to results/

Run standalone:
    python src/analytics/diagnostics.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering in Codespace
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import (
    ADF_PVALUE_THRESHOLD,
    KPSS_PVALUE_THRESHOLD,
    RESULTS_DIR,
    DATETIME_COL,
    TARGET_COL,
    HW_SEASONAL_PERIODS,
)
from ingestion.data_loader import load_instance_usage


# ── 1. Decomposition ──────────────────────────────────────────────────

def stl_decompose(series: pd.Series, period: int = 288) -> dict:
    """
    STL decomposition (Season-Trend decomposition using LOESS).
    Robust to outliers – better suited than classical additive for cluster data.
    """
    logger.info("Running STL decomposition …")
    stl = STL(series, period=period, robust=True)
    res = stl.fit()
    return {
        "trend":    res.trend,
        "seasonal": res.seasonal,
        "residual": res.resid,
        "strength_trend":    1 - res.resid.var() / (res.trend + res.resid).var(),
        "strength_seasonal": 1 - res.resid.var() / (res.seasonal + res.resid).var(),
    }


# ── 2. Stationarity tests ─────────────────────────────────────────────

def run_adf(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test (H0: unit root exists)."""
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "statistic": result[0],
        "pvalue":    result[1],
        "lags_used": result[2],
        "n_obs":     result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < ADF_PVALUE_THRESHOLD,
    }


def run_kpss(series: pd.Series) -> dict:
    """KPSS test (H0: series is stationary around a constant or trend)."""
    try:
        stat, pvalue, lags, crit = kpss(series.dropna(), regression="c", nlags="auto")
    except Exception:
        stat, pvalue, lags, crit = kpss(series.dropna(), regression="c")
    return {
        "statistic": stat,
        "pvalue":    pvalue,
        "lags_used": lags,
        "critical_values": crit,
        "is_stationary": pvalue > KPSS_PVALUE_THRESHOLD,
    }


def stationarity_report(series: pd.Series, label: str = "original") -> dict:
    adf  = run_adf(series)
    kpss_ = run_kpss(series)

    logger.info(
        f"[{label}] ADF  p={adf['pvalue']:.4f}  → "
        f"{'✅ stationary' if adf['is_stationary'] else '❌ unit root'}"
    )
    logger.info(
        f"[{label}] KPSS p={kpss_['pvalue']:.4f}  → "
        f"{'✅ stationary' if kpss_['is_stationary'] else '❌ non-stationary'}"
    )
    return {"adf": adf, "kpss": kpss_, "label": label}


def auto_difference(series: pd.Series, max_d: int = 2) -> tuple[pd.Series, int]:
    """
    Difference the series until ADF says it is stationary.
    Returns (differenced_series, d_order).
    """
    s = series.copy()
    d = 0
    while d < max_d:
        if run_adf(s)["is_stationary"]:
            break
        s = s.diff().dropna()
        d += 1
    logger.info(f"Stationarity achieved with d={d}")
    return s, d


# ── 3. Ljung-Box test ─────────────────────────────────────────────────

def ljung_box_test(series: pd.Series, lags: int = 20) -> pd.DataFrame:
    """Check for remaining autocorrelation in residuals."""
    return acorr_ljungbox(series.dropna(), lags=lags, return_df=True)


# ── 4. Plotting ───────────────────────────────────────────────────────

def plot_full_diagnostics(
    series: pd.Series,
    decomp: dict,
    stats: dict,
    save_path: Path | None = None,
) -> Path:
    """
    Build a 3×2 diagnostic figure:
      Row 0: raw series  |  STL decomposition
      Row 1: ACF         |  PACF
      Row 2: residual histogram  |  residual Q-Q plot
    """
    from scipy import stats as scipy_stats

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("AIOps – CPU Usage Diagnostic Dashboard", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Raw series ────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    series.plot(ax=ax0, linewidth=0.6, color="#2196F3", alpha=0.8)
    decomp["trend"].plot(ax=ax0, color="#F44336", linewidth=1.5, label="Trend")
    ax0.set_title("Raw Series + STL Trend")
    ax0.set_ylabel("CPU (NCU)")
    ax0.legend(fontsize=8)

    # ── Seasonal component ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    # Show 3 days of seasonality
    decomp["seasonal"].iloc[:864].plot(ax=ax1, linewidth=0.7, color="#9C27B0")
    ax1.set_title("Seasonal Component (first 3 days)")
    ax1.set_ylabel("Amplitude")

    # ── ACF ───────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    plot_acf(series.dropna(), ax=ax2, lags=100, alpha=0.05,
             color="#2196F3", vlines_kwargs={"colors": "#2196F3"})
    ax2.set_title("ACF (100 lags)")

    # ── PACF ──────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    plot_pacf(series.dropna(), ax=ax3, lags=50, alpha=0.05, method="ywm",
              color="#FF9800", vlines_kwargs={"colors": "#FF9800"})
    ax3.set_title("PACF (50 lags)")

    # ── Residual histogram ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    resid = decomp["residual"].dropna()
    ax4.hist(resid, bins=60, color="#4CAF50", edgecolor="white", alpha=0.85)
    ax4.axvline(resid.mean(), color="#F44336", linestyle="--", label=f"μ={resid.mean():.4f}")
    ax4.set_title("Residual Distribution")
    ax4.legend(fontsize=8)

    # ── Q-Q plot ──────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    scipy_stats.probplot(resid, dist="norm", plot=ax5)
    ax5.set_title("Residual Q-Q Plot")

    # ── ADF / KPSS annotation ─────────────────────────────────────────
    adf_txt = (
        f"ADF  stat={stats['adf']['statistic']:.3f}  p={stats['adf']['pvalue']:.4f}  "
        f"{'✅' if stats['adf']['is_stationary'] else '❌'}\n"
        f"KPSS stat={stats['kpss']['statistic']:.3f}  p={stats['kpss']['pvalue']:.4f}  "
        f"{'✅' if stats['kpss']['is_stationary'] else '❌'}"
    )
    fig.text(0.5, 0.005, adf_txt, ha="center", fontsize=9,
             bbox=dict(boxstyle="round", fc="#FFF9C4", ec="#FFC107"))

    save_path = save_path or RESULTS_DIR / "diagnostics.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Diagnostic plot saved → {save_path}")
    return save_path


# ── 5. Main ───────────────────────────────────────────────────────────

def run_full_diagnostics(df: pd.DataFrame | None = None) -> dict:
    """
    Complete diagnostic run. Returns a dict with all results.
    Also prints a console summary and saves a PNG.
    """
    if df is None:
        df = load_instance_usage(source="csv")

    series = df.set_index(DATETIME_COL)[TARGET_COL].squeeze()
    series.name = TARGET_COL

    # ── Decompose ─────────────────────────────────────────────────────
    period = min(HW_SEASONAL_PERIODS, len(series) // 2)
    decomp = stl_decompose(series, period=period)

    # ── Stationarity on original ──────────────────────────────────────
    stats_orig = stationarity_report(series, label="original")

    # ── Auto-difference if needed ─────────────────────────────────────
    stationary_series, d_order = auto_difference(series)
    stats_diff = (
        stationarity_report(stationary_series, label=f"diff(d={d_order})")
        if d_order > 0 else stats_orig
    )

    # ── ACF / PACF suggested orders ───────────────────────────────────
    logger.info(
        f"STL Trend strength    : {decomp['strength_trend']:.3f}\n"
        f"STL Seasonal strength : {decomp['strength_seasonal']:.3f}"
    )

    # ── Plot ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = plot_full_diagnostics(
        series, decomp, stats_orig, save_path=RESULTS_DIR / "diagnostics.png"
    )

    # ── Console summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"  Series length   : {len(series):,} observations")
    print(f"  Frequency       : 5 min")
    print(f"  Date range      : {series.index[0]}  →  {series.index[-1]}")
    print(f"  Mean CPU (NCU)  : {series.mean():.4f}")
    print(f"  Std  CPU (NCU)  : {series.std():.4f}")
    print(f"  Trend strength  : {decomp['strength_trend']:.3f}")
    print(f"  Season strength : {decomp['strength_seasonal']:.3f}")
    print(f"  ADF  p-value    : {stats_orig['adf']['pvalue']:.4f}  "
          f"({'stationary' if stats_orig['adf']['is_stationary'] else 'NON-stationary'})")
    print(f"  KPSS p-value    : {stats_orig['kpss']['pvalue']:.4f}  "
          f"({'stationary' if stats_orig['kpss']['is_stationary'] else 'NON-stationary'})")
    print(f"  Differencing d  : {d_order}")
    print(f"  Diagnostic plot : {plot_path}")
    print("=" * 60 + "\n")

    return {
        "series": series,
        "decomp": decomp,
        "stats_original": stats_orig,
        "stats_stationary": stats_diff,
        "d_order": d_order,
        "stationary_series": stationary_series,
        "plot_path": str(plot_path),
    }


if __name__ == "__main__":
    run_full_diagnostics()
