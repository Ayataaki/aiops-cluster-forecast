"""
src/models/baselines.py

Baseline forecasting models:
  1. NaiveForecaster       – Y_{t+h} = Y_t  (persist last value)
  2. SeasonalNaive         – Y_{t+h} = Y_{t - season + h}
  3. HoltWinters           – Triple exponential smoothing (statsmodels)

These are the benchmark models that every more advanced model must beat.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import HORIZON, HW_SEASONAL_PERIODS


# ── Evaluation metrics ────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    # Only compute MAPE on non-near-zero actuals (avoids inflation)
    mask = np.abs(y_true) > 0.01
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str = "") -> dict:
    m = {
        "model": name,
        "MAE":   mae(y_true, y_pred),
        "RMSE":  rmse(y_true, y_pred),
        "MAPE":  mape(y_true, y_pred),
    }
    logger.info(f"  {name:<25}  MAE={m['MAE']:.5f}  RMSE={m['RMSE']:.5f}  MAPE={m['MAPE']:.2f}%")
    return m


# ── 1. Naive ──────────────────────────────────────────────────────────

class NaiveForecaster:
    """Persist last observed value for all horizons."""

    def __init__(self, horizon: int = HORIZON):
        self.horizon = horizon
        self.last_value: float | None = None

    def fit(self, series: np.ndarray) -> "NaiveForecaster":
        self.last_value = series[-1]
        return self

    def predict(self) -> np.ndarray:
        return np.full(self.horizon, self.last_value)

    def rolling_evaluate(
        self, series: np.ndarray, n_windows: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate rolling origin evaluation on the last n_windows windows.
        Returns (y_true, y_pred) both shape (n_windows, horizon).
        """
        n = len(series)
        start = n - n_windows - self.horizon
        y_true_all, y_pred_all = [], []
        for i in range(n_windows):
            origin = start + i
            self.fit(series[: origin + 1])
            y_pred_all.append(self.predict())
            y_true_all.append(series[origin + 1 : origin + 1 + self.horizon])
        return np.array(y_true_all), np.array(y_pred_all)


# ── 2. Seasonal Naive ─────────────────────────────────────────────────

class SeasonalNaiveForecaster:
    """
    Y_{t+h} = Y_{t - S + h}   where S = seasonal period.
    Uses the corresponding 5-min slot from 24 hours ago.
    """

    def __init__(self, horizon: int = HORIZON, season: int = 288):
        self.horizon = horizon
        self.season  = season
        self._history: np.ndarray | None = None

    def fit(self, series: np.ndarray) -> "SeasonalNaiveForecaster":
        self._history = np.array(series)
        return self

    def predict(self) -> np.ndarray:
        preds = []
        for h in range(1, self.horizon + 1):
            idx = len(self._history) - self.season + (h - 1)
            preds.append(self._history[idx % len(self._history)])
        return np.array(preds)

    def rolling_evaluate(
        self, series: np.ndarray, n_windows: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(series)
        start = n - n_windows - self.horizon
        y_true_all, y_pred_all = [], []
        for i in range(n_windows):
            origin = start + i
            self.fit(series[: origin + 1])
            y_pred_all.append(self.predict())
            y_true_all.append(series[origin + 1 : origin + 1 + self.horizon])
        return np.array(y_true_all), np.array(y_pred_all)


# ── 3. Holt-Winters ───────────────────────────────────────────────────

class HoltWintersForecaster:
    """
    Triple exponential smoothing with damped additive trend + additive seasonality.
    Fitted once on the training set; forecasts by iterative update on new data.
    """

    def __init__(
        self,
        horizon: int = HORIZON,
        seasonal_periods: int = HW_SEASONAL_PERIODS,
    ):
        self.horizon = horizon
        self.seasonal_periods = seasonal_periods
        self._model = None
        self._fitted = None

    def fit(self, series: np.ndarray) -> "HoltWintersForecaster":
        logger.info(
            f"Fitting Holt-Winters (T={len(series):,}, "
            f"m={self.seasonal_periods}) …"
        )
        # Reduce seasonal period if series is too short
        sp = min(self.seasonal_periods, len(series) // 2)
        self._model = ExponentialSmoothing(
            series,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=sp,
            initialization_method="estimated",
        )
        self._fitted = self._model.fit(optimized=True, remove_bias=True)
        logger.info(
            f"  α={self._fitted.params['smoothing_level']:.4f}  "
            f"β={self._fitted.params.get('smoothing_trend', float('nan')):.4f}  "
            f"γ={self._fitted.params.get('smoothing_seasonal', float('nan')):.4f}"
        )
        return self

    def predict(self, steps: int | None = None) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("Call fit() first.")
        h = steps or self.horizon
        return self._fitted.forecast(h)

    def rolling_evaluate(
        self,
        train: np.ndarray,
        test: np.ndarray,
        n_windows: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Refit HW on each origin (expensive!). Use a reasonable n_windows.
        """
        n_windows = min(n_windows, len(test) - self.horizon)
        y_true_all, y_pred_all = [], []
        for i in range(n_windows):
            combined = np.concatenate([train, test[:i]])
            try:
                self.fit(combined)
                pred = self.predict()
                y_true_all.append(test[i : i + self.horizon])
                y_pred_all.append(pred)
            except Exception as e:
                logger.warning(f"HW fit failed at window {i}: {e}")
        return np.array(y_true_all), np.array(y_pred_all)


# ── Comparison helper ─────────────────────────────────────────────────

def compare_baselines(
    train_series: np.ndarray,
    test_series: np.ndarray,
    horizon: int = HORIZON,
    n_windows: int = 50,
) -> pd.DataFrame:
    """
    Run rolling evaluation on all baseline models and return a comparison table.
    """
    logger.info("\n── Baseline comparison ──────────────────────────────────────")
    results = []

    # Naive
    naive = NaiveForecaster(horizon=horizon)
    yt, yp = naive.rolling_evaluate(
        np.concatenate([train_series, test_series]), n_windows
    )
    results.append(evaluate(yt.flatten(), yp.flatten(), "Naive"))

    # Seasonal Naive
    sn = SeasonalNaiveForecaster(horizon=horizon)
    yt, yp = sn.rolling_evaluate(
        np.concatenate([train_series, test_series]), n_windows
    )
    results.append(evaluate(yt.flatten(), yp.flatten(), "SeasonalNaive(288)"))

    # Holt-Winters
    hw = HoltWintersForecaster(horizon=horizon)
    yt, yp = hw.rolling_evaluate(train_series, test_series, n_windows)
    results.append(evaluate(yt.flatten(), yp.flatten(), "HoltWinters"))

    df = pd.DataFrame(results).set_index("model")
    logger.info(f"\n{df.round(5).to_string()}\n")
    return df


if __name__ == "__main__":
    from ingestion.data_loader import load_instance_usage
    from analytics.feature_engineer import time_split
    from config import TARGET_COL, DATETIME_COL

    df = load_instance_usage(source="csv")
    train_df, _, test_df = time_split(df)
    train_arr = train_df[TARGET_COL].values
    test_arr  = test_df[TARGET_COL].values

    compare_baselines(train_arr, test_arr)
