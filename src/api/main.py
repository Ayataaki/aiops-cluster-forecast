"""
src/api/main.py

Predictive Auto-Scaling Decision API (FastAPI).

Endpoints:
  GET  /health              – liveness probe
  POST /predict             – return forecast for next HORIZON steps
  POST /scaling-decision    – return scale-out recommendation
  GET  /metrics/latest      – last observed CPU reading (simulated)
  GET  /results             – serve latest model comparison table

Run:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import (
    CPU_SCALE_OUT_THRESHOLD,
    HORIZON,
    MODELS_DIR,
    RESULTS_DIR,
    DATETIME_COL,
    TARGET_COL,
)

# ── App initialisation ────────────────────────────────────────────────

app = FastAPI(
    title="AIOps Predictive Auto-Scaling API",
    description=(
        "Forecasts cluster CPU usage using XGBoost / Informer / Autoformer "
        "and triggers proactive Kubernetes HPA scaling decisions."
    ),
    version="1.0.0",
)

# ── Lazy model loading (cached at startup) ────────────────────────────

_xgb_model = None
_transformer_model = None


def get_xgb():
    global _xgb_model
    if _xgb_model is None:
        from models.xgboost_forecaster import XGBoostForecaster
        pkl = MODELS_DIR / "xgb_forecaster.pkl"
        if pkl.exists():
            _xgb_model = XGBoostForecaster.load(pkl)
            logger.info("XGBoost model loaded.")
        else:
            logger.warning("XGBoost model not found. Run training first.")
    return _xgb_model


def get_transformer(model_name: str = "informer"):
    global _transformer_model
    if _transformer_model is None:
        try:
            from models.transformer_model import TransformerForecaster
            _transformer_model = TransformerForecaster.load(model_name)
            logger.info(f"Transformer ({model_name}) loaded.")
        except Exception as e:
            logger.warning(f"Transformer not available: {e}")
    return _transformer_model


# ── Request / Response schemas ────────────────────────────────────────

class CPUReadings(BaseModel):
    """Recent CPU usage measurements (NCU, in chronological order)."""
    values: list[float] = Field(
        ...,
        min_length=12,
        description="At least 12 recent 5-min CPU readings in [0, 1]",
        example=[0.22, 0.25, 0.28, 0.31, 0.30, 0.29,
                 0.33, 0.35, 0.40, 0.42, 0.45, 0.47],
    )
    timestamps: list[str] | None = Field(
        default=None,
        description="ISO-8601 timestamps for each reading (optional)",
    )
    model: str = Field(
        default="xgboost",
        description="Model to use: 'xgboost' | 'informer' | 'autoformer'",
    )


class ForecastResponse(BaseModel):
    model_used:   str
    horizon:      int
    predictions:  list[float]
    timestamps:   list[str]
    unit:         str = "NCU (Normalised Compute Unit, range [0,1])"


class ScalingDecision(BaseModel):
    scale_out:        bool
    reason:           str
    max_predicted_cpu: float
    threshold:        float
    recommended_replicas_delta: int
    forecast:         ForecastResponse


# ── Helpers ───────────────────────────────────────────────────────────

def _build_minimal_feature_row(values: list[float]) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame from raw CPU readings.
    Uses the same feature engineering logic as training.
    """
    from analytics.feature_engineer import (
        add_lags, add_rolling, add_ewm, add_calendar, add_fourier
    )

    n = len(values)
    # Create a tiny DataFrame mimicking the full dataset format
    freq = "5min"
    idx = pd.date_range(end=datetime.now(tz=timezone.utc), periods=n, freq=freq)
    df = pd.DataFrame({DATETIME_COL: idx, TARGET_COL: values})

    df = add_lags(df)
    df = add_rolling(df)
    df = add_ewm(df)
    df = add_calendar(df)
    df = add_fourier(df)
    df = df.dropna()

    return df.tail(1)   # only the last (most recent) row


def _make_future_timestamps(horizon: int = HORIZON) -> list[str]:
    now = datetime.now(tz=timezone.utc)
    return [
        (pd.Timestamp(now) + pd.Timedelta(minutes=5 * (h + 1))).isoformat()
        for h in range(horizon)
    ]


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health", summary="Liveness probe")
def health():
    return {"status": "ok", "timestamp": datetime.now(tz=timezone.utc).isoformat()}


@app.post("/predict", response_model=ForecastResponse, summary="CPU forecast")
def predict(body: CPUReadings):
    """
    Accepts recent CPU readings and returns a multi-step forecast.
    """
    values = np.array(body.values, dtype=float)

    if body.model == "xgboost":
        xgb = get_xgb()
        if xgb is None:
            raise HTTPException(
                status_code=503,
                detail="XGBoost model not available. Run: python src/models/model_comparison.py --no-transformers"
            )
        try:
            row_df = _build_minimal_feature_row(values.tolist())
            # Align feature columns with trained model
            missing = [c for c in xgb.feature_cols if c not in row_df.columns]
            for c in missing:
                row_df[c] = 0.0
            preds = xgb.predict(row_df[xgb.feature_cols])[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    elif body.model in ("informer", "autoformer"):
        transformer = get_transformer(body.model)
        if transformer is None:
            raise HTTPException(
                status_code=503,
                detail=f"{body.model} model not available. Run training first."
            )
        try:
            from models.transformer_model import to_nixtla_format
            freq = "5min"
            idx = pd.date_range(
                end=datetime.now(tz=timezone.utc), periods=len(values), freq=freq
            )
            tmp_df = pd.DataFrame({DATETIME_COL: idx, TARGET_COL: values})
            nf_df = to_nixtla_format(tmp_df)
            preds = transformer.predict_from_history(nf_df)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {body.model}")

    preds = np.clip(preds, 0.0, 1.0).tolist()

    return ForecastResponse(
        model_used=body.model,
        horizon=HORIZON,
        predictions=preds,
        timestamps=_make_future_timestamps(HORIZON),
    )


@app.post("/scaling-decision", response_model=ScalingDecision,
          summary="Proactive scaling recommendation")
def scaling_decision(body: CPUReadings):
    """
    Returns whether to scale out and by how many replicas,
    based on the predicted CPU load over the next HORIZON steps.
    """
    forecast_resp = predict(body)
    max_pred = max(forecast_resp.predictions)

    should_scale = max_pred > CPU_SCALE_OUT_THRESHOLD

    # Simple heuristic: each 10% above threshold → +1 replica
    delta = 0
    if should_scale:
        delta = max(1, int((max_pred - CPU_SCALE_OUT_THRESHOLD) / 0.10))

    reason = (
        f"Predicted CPU reaches {max_pred:.1%} in the next "
        f"{HORIZON * 5} min (threshold: {CPU_SCALE_OUT_THRESHOLD:.0%})"
        if should_scale
        else f"Max predicted CPU {max_pred:.1%} < threshold {CPU_SCALE_OUT_THRESHOLD:.0%}"
    )

    return ScalingDecision(
        scale_out=should_scale,
        reason=reason,
        max_predicted_cpu=round(max_pred, 4),
        threshold=CPU_SCALE_OUT_THRESHOLD,
        recommended_replicas_delta=delta,
        forecast=forecast_resp,
    )


@app.get("/metrics/latest", summary="Latest simulated CPU reading")
def latest_metrics():
    """Returns the most recent CPU reading from the dataset (demo mode)."""
    try:
        from ingestion.data_loader import load_instance_usage
        df = load_instance_usage(source="csv")
        latest = df.tail(1).iloc[0]
        return {
            "timestamp": str(latest[DATETIME_COL]),
            "avg_cpu":   round(float(latest[TARGET_COL]), 4),
            "unit":      "NCU",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results", summary="Model comparison results")
def get_results():
    """Returns the latest model benchmark results."""
    csv = RESULTS_DIR / "model_comparison.csv"
    if not csv.exists():
        raise HTTPException(
            status_code=404,
            detail="No results yet. Run: python src/models/model_comparison.py"
        )
    df = pd.read_csv(csv, index_col=0)
    return JSONResponse(content=df.round(5).to_dict(orient="index"))
