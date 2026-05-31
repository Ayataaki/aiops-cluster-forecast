"""
src/models/xgboost_forecaster.py

XGBoost-based direct multi-step forecaster.
Each horizon step h has its own XGBoost estimator (DIRECT strategy).

Usage:
    python src/models/xgboost_forecaster.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import HORIZON, MODELS_DIR, XGBOOST_CFG, TARGET_COL, DATETIME_COL
from analytics.feature_engineer import (
    build_feature_matrix,
    time_split,
)
from models.baselines import evaluate


class XGBoostForecaster:
    """
    Direct multi-step XGBoost forecaster.
    One XGBRegressor per horizon step, trained on the same feature set.
    """

    def __init__(self, horizon: int = HORIZON, cfg: dict | None = None):
        self.horizon = horizon
        self.cfg = cfg or XGBOOST_CFG
        self.models: list[XGBRegressor] = []
        self.scaler = RobustScaler()
        self.feature_cols: list[str] = []
        self.target_cols: list[str] = []
        self._is_fitted = False

    # ── Fit ───────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list[str],
        target_cols: list[str],
    ) -> "XGBoostForecaster":
        self.feature_cols = feature_cols
        self.target_cols  = target_cols

        X_train = self.scaler.fit_transform(train_df[feature_cols])
        X_val   = self.scaler.transform(val_df[feature_cols])

        self.models = []
        for i, t_col in enumerate(target_cols):
            y_train = train_df[t_col].values
            y_val   = val_df[t_col].values

            model = XGBRegressor(
                **self.cfg,
                eval_metric="rmse",
                early_stopping_rounds=30,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            best = model.best_iteration
            logger.info(
                f"  h={i+1:02d}  best_iter={best:3d}  "
                f"val_rmse={model.evals_result()['validation_0']['rmse'][best]:.5f}"
            )
            self.models.append(model)

        self._is_fitted = True
        logger.info("XGBoost training complete.")
        return self

    # ── Predict ───────────────────────────────────────────────────────

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Returns array of shape (n_samples, horizon)."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        preds = np.column_stack([m.predict(X_scaled) for m in self.models])
        return np.clip(preds, 0.0, 1.0)

    # ── Evaluate ──────────────────────────────────────────────────────

    def evaluate(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns per-horizon and aggregate metrics on the test set.
        """
        X_test = test_df[self.feature_cols]
        preds  = self.predict(X_test)    # (N, H)
        rows   = []
        for i, t_col in enumerate(self.target_cols):
            y_true = test_df[t_col].values
            y_pred = preds[:, i]
            m = evaluate(y_true, y_pred, f"XGB h={i+1}")
            rows.append(m)

        # Aggregate over all horizons
        y_true_flat = np.concatenate([test_df[c].values for c in self.target_cols])
        y_pred_flat = preds.flatten()
        rows.append(evaluate(y_true_flat, y_pred_flat, "XGB (all H)"))

        return pd.DataFrame(rows).set_index("model")

    # ── Feature importance ────────────────────────────────────────────

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Average gain importance across all horizon models."""
        importance = np.zeros(len(self.feature_cols))
        for m in self.models:
            importance += m.feature_importances_
        importance /= len(self.models)
        df = pd.DataFrame(
            {"feature": self.feature_cols, "importance": importance}
        ).sort_values("importance", ascending=False)
        return df.head(top_n)

    # ── Persist ───────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        path = path or MODELS_DIR / "xgb_forecaster.pkl"
        joblib.dump(self, path)
        logger.info(f"Model saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "XGBoostForecaster":
        path = path or MODELS_DIR / "xgb_forecaster.pkl"
        model = joblib.load(path)
        logger.info(f"Model loaded ← {path}")
        return model


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ingestion.data_loader import load_instance_usage

    df = load_instance_usage(source="csv")
    feat_df, feature_cols, target_cols = build_feature_matrix(df, horizon=HORIZON)
    train_df, val_df, test_df = time_split(feat_df)

    xgb = XGBoostForecaster(horizon=HORIZON)
    xgb.fit(train_df, val_df, feature_cols, target_cols)

    metrics = xgb.evaluate(test_df)
    print("\nXGBoost Test Metrics:")
    print(metrics.round(5).to_string())

    fi = xgb.feature_importance(top_n=15)
    print("\nTop-15 Feature Importances:")
    print(fi.to_string(index=False))

    xgb.save()
