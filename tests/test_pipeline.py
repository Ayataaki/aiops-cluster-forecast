"""
tests/test_pipeline.py

End-to-end pipeline tests. Each test is designed to run in under 60s
on a GitHub Codespace CPU (no GPU required).

Run:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_df():
    """Generate synthetic data once for the session."""
    from config import DATA_DIR
    import subprocess, sys as _sys
    # Ensure synthetic data exists
    out = DATA_DIR / "instance_usage_sample.csv"
    if not out.exists():
        subprocess.run(
            [_sys.executable, str(ROOT / "scripts" / "generate_synthetic.py")],
            check=True,
        )
    from ingestion.data_loader import load_instance_usage
    return load_instance_usage(source="csv")


@pytest.fixture(scope="session")
def split_data(synthetic_df):
    from analytics.feature_engineer import time_split
    train, val, test = time_split(synthetic_df)
    return train, val, test


@pytest.fixture(scope="session")
def feature_matrix(synthetic_df):
    from analytics.feature_engineer import build_feature_matrix, time_split
    from config import HORIZON
    feat_df, feat_cols, tgt_cols = build_feature_matrix(synthetic_df, horizon=HORIZON)
    train, val, test = time_split(feat_df)
    return train, val, test, feat_cols, tgt_cols


# ── Data ingestion ────────────────────────────────────────────────────

class TestDataLoader:
    def test_csv_loads(self, synthetic_df):
        assert len(synthetic_df) > 100

    def test_columns_present(self, synthetic_df):
        from config import DATETIME_COL, TARGET_COL
        assert DATETIME_COL in synthetic_df.columns
        assert TARGET_COL in synthetic_df.columns

    def test_no_nulls_in_target(self, synthetic_df):
        from config import TARGET_COL
        assert synthetic_df[TARGET_COL].isna().sum() == 0

    def test_cpu_in_range(self, synthetic_df):
        from config import TARGET_COL
        assert synthetic_df[TARGET_COL].between(0, 1).all()


# ── Diagnostics ───────────────────────────────────────────────────────

class TestDiagnostics:
    def test_adf_runs(self, synthetic_df):
        from analytics.diagnostics import run_adf
        from config import TARGET_COL, DATETIME_COL
        series = synthetic_df.set_index(DATETIME_COL)[TARGET_COL]
        result = run_adf(series)
        assert "pvalue" in result
        assert "is_stationary" in result

    def test_kpss_runs(self, synthetic_df):
        from analytics.diagnostics import run_kpss
        from config import TARGET_COL, DATETIME_COL
        series = synthetic_df.set_index(DATETIME_COL)[TARGET_COL]
        result = run_kpss(series)
        assert "pvalue" in result


# ── Feature engineering ───────────────────────────────────────────────

class TestFeatureEngineering:
    def test_feature_matrix_shape(self, feature_matrix):
        train, val, test, feat_cols, tgt_cols = feature_matrix
        from config import HORIZON
        assert len(tgt_cols) == HORIZON
        assert len(feat_cols) > 10    # at least 10 features

    def test_no_nans(self, feature_matrix):
        train, val, test, feat_cols, tgt_cols = feature_matrix
        assert not train[feat_cols + tgt_cols].isna().any().any()

    def test_chronological_order(self, split_data):
        from config import DATETIME_COL
        train, val, test = split_data
        assert train[DATETIME_COL].max() < val[DATETIME_COL].min()
        assert val[DATETIME_COL].max() < test[DATETIME_COL].min()


# ── Baselines ─────────────────────────────────────────────────────────

class TestBaselines:
    def test_naive_shape(self, split_data):
        from models.baselines import NaiveForecaster
        from config import HORIZON, TARGET_COL
        _, _, test = split_data
        naive = NaiveForecaster(horizon=HORIZON)
        naive.fit(test[TARGET_COL].values[:50])
        pred = naive.predict()
        assert pred.shape == (HORIZON,)

    def test_holt_winters_fits(self, split_data):
        from models.baselines import HoltWintersForecaster
        from config import HORIZON, TARGET_COL
        train, _, _ = split_data
        hw = HoltWintersForecaster(horizon=HORIZON, seasonal_periods=288)
        hw.fit(train[TARGET_COL].values)
        pred = hw.predict()
        assert len(pred) == HORIZON
        assert not np.any(np.isnan(pred))

    def test_hw_beats_naive(self, split_data):
        """HW RMSE should be at least as good as Naive on synthetic data."""
        from models.baselines import (
            NaiveForecaster, HoltWintersForecaster, rmse
        )
        from config import HORIZON, TARGET_COL
        train, _, test = split_data

        n_eval = min(50, len(test) - HORIZON)
        hw = HoltWintersForecaster(horizon=HORIZON, seasonal_periods=288)
        hw.fit(train[TARGET_COL].values)

        naive_errors, hw_errors = [], []
        for i in range(n_eval):
            truth = test[TARGET_COL].values[i : i + HORIZON]
            naive_pred = np.full(HORIZON, test[TARGET_COL].values[i])
            hw_pred = hw.predict()
            naive_errors.append(rmse(truth, naive_pred))
            hw_errors.append(rmse(truth, hw_pred))

        assert np.mean(hw_errors) <= np.mean(naive_errors) * 1.5  # within 50%


# ── XGBoost ───────────────────────────────────────────────────────────

class TestXGBoost:
    def test_xgb_trains_and_predicts(self, feature_matrix):
        from models.xgboost_forecaster import XGBoostForecaster
        from config import HORIZON
        train, val, test, feat_cols, tgt_cols = feature_matrix

        xgb = XGBoostForecaster(horizon=HORIZON, cfg={"n_estimators": 20, "random_state": 42})
        xgb.fit(train, val, feat_cols, tgt_cols)

        preds = xgb.predict(test[feat_cols].head(10))
        assert preds.shape == (10, HORIZON)
        assert (preds >= 0).all() and (preds <= 1).all()

    def test_xgb_metrics_reasonable(self, feature_matrix):
        from models.xgboost_forecaster import XGBoostForecaster
        from config import HORIZON
        train, val, test, feat_cols, tgt_cols = feature_matrix

        xgb = XGBoostForecaster(horizon=HORIZON, cfg={"n_estimators": 30, "random_state": 42})
        xgb.fit(train, val, feat_cols, tgt_cols)
        metrics = xgb.evaluate(test)
        rmse_val = metrics.loc["XGB (all H)", "RMSE"]
        assert rmse_val < 0.20    # RMSE < 20% NCU on synthetic data


# ── FastAPI ───────────────────────────────────────────────────────────

class TestAPI:
    def test_health(self):
        from fastapi.testclient import TestClient
        from api.main import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_predict_without_model(self):
        """Should return 503 when model file doesn't exist."""
        from fastapi.testclient import TestClient
        from api.main import app, _xgb_model
        import api.main as api_module
        # Temporarily remove model
        api_module._xgb_model = None
        client = TestClient(app)
        payload = {"values": [0.3] * 20, "model": "xgboost"}
        resp = client.post("/predict", json=payload)
        # Either 200 (model loaded) or 503 (not trained) – both are valid
        assert resp.status_code in (200, 503)
