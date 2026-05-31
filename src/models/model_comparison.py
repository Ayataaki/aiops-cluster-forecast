"""
src/models/model_comparison.py

Runs all models (baselines + XGBoost + Transformers) and produces:
  - results/model_comparison.csv
  - results/model_comparison.png  (bar chart)

Run:
    python src/models/model_comparison.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import HORIZON, RESULTS_DIR, TARGET_COL, DATETIME_COL
from ingestion.data_loader import load_instance_usage
from analytics.feature_engineer import build_feature_matrix, time_split
from models.baselines import compare_baselines, HoltWintersForecaster
from models.xgboost_forecaster import XGBoostForecaster
from models.transformer_model import (
    TransformerForecaster,
    to_nixtla_format,
    chronological_split,
)


def run_comparison(include_transformers: bool = True) -> pd.DataFrame:
    logger.info("Loading data …")
    df = load_instance_usage(source="csv")

    # ── Raw series split (for baselines) ─────────────────────────────
    feat_df, feature_cols, target_cols = build_feature_matrix(df, horizon=HORIZON)
    train_df, val_df, test_df = time_split(feat_df)

    train_arr = df[TARGET_COL].values[: int(len(df) * 0.70)]
    test_arr  = df[TARGET_COL].values[int(len(df) * 0.85) :]

    # ── 1. Baselines ──────────────────────────────────────────────────
    logger.info("\n── BASELINES ────────────────────────────────────────────────")
    baseline_df = compare_baselines(train_arr, test_arr, horizon=HORIZON, n_windows=50)
    all_results = baseline_df.reset_index().to_dict("records")

    # ── 2. XGBoost ────────────────────────────────────────────────────
    logger.info("\n── XGBOOST ──────────────────────────────────────────────────")
    xgb = XGBoostForecaster(horizon=HORIZON)
    xgb.fit(train_df, val_df, feature_cols, target_cols)
    xgb_metrics = xgb.evaluate(test_df)
    xgb_row = xgb_metrics.loc["XGB (all H)"].to_dict()
    xgb_row["model"] = "XGBoost"
    all_results.append(xgb_row)
    xgb.save()

    # ── 3. Transformers (optional – slow on CPU) ──────────────────────
    if include_transformers:
        train_raw, val_raw, test_raw = time_split(df)
        train_nf = to_nixtla_format(train_raw)
        val_nf   = to_nixtla_format(val_raw)
        test_nf  = to_nixtla_format(test_raw)
        full_train_nf = pd.concat([train_nf, val_nf], ignore_index=True)

        for model_name in ["informer", "autoformer"]:
            logger.info(f"\n── {model_name.upper()} ─────────────────────────────────────────────")
            try:
                fc = TransformerForecaster(model_name=model_name)
                fc.fit(train_nf, val_nf)
                m = fc.evaluate(full_train_nf, test_nf)
                if m:
                    m["model"] = model_name.capitalize()
                    all_results.append(m)
                fc.save()
            except Exception as e:
                logger.warning(f"{model_name} failed: {e}")

    # ── Compile results table ─────────────────────────────────────────
    results_df = pd.DataFrame(all_results).set_index("model")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "model_comparison.csv"
    results_df.round(5).to_csv(csv_path)

    logger.info(f"\n{'='*60}")
    logger.info("FINAL MODEL COMPARISON")
    logger.info(f"{'='*60}")
    logger.info(f"\n{results_df.round(5).to_string()}\n")

    # ── Plot ──────────────────────────────────────────────────────────
    _plot_comparison(results_df)

    return results_df


def _plot_comparison(results_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Model Comparison — Horizon H={HORIZON} (30 min)",
        fontsize=13, fontweight="bold",
    )

    metrics = ["MAE", "RMSE", "MAPE"]
    colors  = plt.cm.Set2.colors

    for ax, metric in zip(axes, metrics):
        if metric not in results_df.columns:
            continue
        values = results_df[metric].sort_values()
        bars = ax.barh(values.index, values.values,
                       color=colors[:len(values)], edgecolor="white")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = RESULTS_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Comparison plot saved → {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-transformers", action="store_true",
        help="Skip deep learning models (fast baseline-only run)"
    )
    args = parser.parse_args()
    run_comparison(include_transformers=not args.no_transformers)
