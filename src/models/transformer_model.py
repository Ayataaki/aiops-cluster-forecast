"""
src/models/transformer_model.py

Deep learning forecasters using the NeuralForecast library:
  - Informer    (ProbSparse attention – fast for long sequences)
  - Autoformer  (auto-correlation mechanism + seasonal decomposition)
  - PatchTST    (patch-based ViT for time series – strong general baseline)

The library handles:
  • gradient-based training (PyTorch Lightning under the hood)
  • multi-step direct forecasting
  • automatic GPU/CPU dispatch

Usage:
    python src/models/transformer_model.py --model informer --epochs 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import (
    DATETIME_COL,
    HORIZON,
    INPUT_SIZE,
    INFORMER_CFG,
    AUTOFORMER_CFG,
    MODELS_DIR,
    TARGET_COL,
    TRAIN_RATIO,
    VAL_RATIO,
)
from models.baselines import evaluate


# ── neuralforecast imports (lazy so the file can be imported without GPU) ──

def _import_nf():
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import Informer, Autoformer, PatchTST
        from neuralforecast.losses.pytorch import MAE as NF_MAE
        return NeuralForecast, Informer, Autoformer, PatchTST, NF_MAE
    except ImportError as e:
        raise ImportError(
            "neuralforecast is not installed. Run: pip install neuralforecast"
        ) from e


# ── Data formatter (NeuralForecast long format) ───────────────────────

def to_nixtla_format(df: pd.DataFrame, unique_id: str = "cluster_a") -> pd.DataFrame:
    """
    NeuralForecast expects columns: unique_id, ds, y
    ds must be a proper datetime (or integer index for non-date freq).
    """
    nf_df = pd.DataFrame({
        "unique_id": unique_id,
        "ds": df[DATETIME_COL].values,
        "y":  df[TARGET_COL].values,
    })
    return nf_df


def chronological_split(
    nf_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(nf_df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return (
        nf_df.iloc[:n_train].copy(),
        nf_df.iloc[n_train : n_train + n_val].copy(),
        nf_df.iloc[n_train + n_val :].copy(),
    )


# ── Model factory ─────────────────────────────────────────────────────

def build_model(name: str, overrides: dict | None = None):
    NeuralForecast, Informer, Autoformer, PatchTST, NF_MAE = _import_nf()

    name = name.lower()
    if name == "informer":
        cfg = {**INFORMER_CFG, **(overrides or {})}
        model = Informer(loss=NF_MAE(), **cfg)
    elif name == "autoformer":
        cfg = {**AUTOFORMER_CFG, **(overrides or {})}
        model = Autoformer(loss=NF_MAE(), **cfg)
    elif name == "patchtst":
        model = PatchTST(
            input_size=INPUT_SIZE,
            h=HORIZON,
            hidden_size=64,
            n_heads=4,
            patch_len=16,
            stride=8,
            max_steps=300,
            val_check_steps=50,
            early_stop_patience_steps=5,
            loss=NF_MAE(),
            **(overrides or {}),
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Choose: informer | autoformer | patchtst")

    return model


# ── Trainer wrapper ───────────────────────────────────────────────────

class TransformerForecaster:
    """
    Thin wrapper around NeuralForecast for a single model.
    Handles fit / predict / evaluate / save / load.
    """

    def __init__(self, model_name: str = "informer", overrides: dict | None = None):
        NeuralForecast, *_ = _import_nf()
        self.model_name = model_name
        self._nf_class  = NeuralForecast
        self.model      = build_model(model_name, overrides)
        self.nf         = None
        self._save_path = MODELS_DIR / f"nf_{model_name}"

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> "TransformerForecaster":
        """
        Parameters: DataFrames with columns (unique_id, ds, y).
        """
        NeuralForecast, *_ = _import_nf()
        # Rebuild NeuralForecast with fresh model (stateful – call once)
        self.nf = NeuralForecast(models=[self.model], freq="5min")
        # NeuralForecast fit expects full df; val_size is inferred from the end
        full = pd.concat([train_df, val_df], ignore_index=True)
        val_size = len(val_df)
        logger.info(f"Training {self.model_name} on {len(train_df):,} samples …")
        self.nf.fit(df=full, val_size=val_size)
        logger.info(f"{self.model_name} training complete.")
        return self

    def predict(self, futr_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Returns a DataFrame with columns (unique_id, ds, <model_name>).
        """
        if self.nf is None:
            raise RuntimeError("Call fit() first.")
        return self.nf.predict(futr_df=futr_df)

    def predict_from_history(
        self, history_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Given recent history in nixtla format, forecast next HORIZON steps.
        Returns 1D array of length HORIZON.
        """
        preds = self.nf.predict(df=history_df)
        model_col = [c for c in preds.columns if c not in ("unique_id", "ds")][0]
        return preds[model_col].values[:HORIZON]

    def evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict:
        """
        Rolling-origin evaluation over the test set (one shot per HORIZON window).
        Returns dict with MAE, RMSE, MAPE.
        """
        y_true_all, y_pred_all = [], []
        step = HORIZON   # non-overlapping windows

        for start in range(0, len(test_df) - HORIZON, step):
            history = pd.concat(
                [train_df, test_df.iloc[:start]], ignore_index=True
            )
            if len(history) < INPUT_SIZE + HORIZON:
                continue
            try:
                pred = self.predict_from_history(history)
                truth = test_df["y"].values[start : start + HORIZON]
                y_pred_all.append(pred)
                y_true_all.append(truth)
            except Exception as e:
                logger.warning(f"Prediction failed at window {start}: {e}")

        if not y_true_all:
            logger.warning("No evaluation windows generated.")
            return {}

        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        return evaluate(y_true, y_pred, self.model_name)

    def save(self) -> Path:
        if self.nf is None:
            raise RuntimeError("Nothing to save – fit first.")
        self._save_path.mkdir(parents=True, exist_ok=True)
        self.nf.save(str(self._save_path), overwrite=True)
        logger.info(f"Model saved → {self._save_path}")
        return self._save_path

    @classmethod
    def load(cls, model_name: str) -> "TransformerForecaster":
        NeuralForecast, *_ = _import_nf()
        obj = cls.__new__(cls)
        obj.model_name = model_name
        obj._save_path = MODELS_DIR / f"nf_{model_name}"
        obj.nf = NeuralForecast.load(str(obj._save_path))
        logger.info(f"Model loaded ← {obj._save_path}")
        return obj


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="informer",
                        choices=["informer", "autoformer", "patchtst"])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    from ingestion.data_loader import load_instance_usage
    from analytics.feature_engineer import time_split

    df = load_instance_usage(source="csv")
    train_raw, val_raw, test_raw = time_split(df)

    overrides = {}
    if args.epochs:
        overrides["max_steps"] = args.epochs

    train_nf = to_nixtla_format(train_raw)
    val_nf   = to_nixtla_format(val_raw)
    test_nf  = to_nixtla_format(test_raw)

    forecaster = TransformerForecaster(model_name=args.model, overrides=overrides)
    forecaster.fit(train_nf, val_nf)

    logger.info("Evaluating on test set …")
    metrics = forecaster.evaluate(
        pd.concat([train_nf, val_nf], ignore_index=True), test_nf
    )
    print("\nTest metrics:", metrics)

    forecaster.save()
