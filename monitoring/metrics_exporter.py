"""
Prometheus metrics exporter for AIOps forecasts.
Exposes /metrics endpoint scraped by Prometheus every 5s.
Run: python monitoring/metrics_exporter.py
"""
import sys, time, random
from pathlib import Path
from prometheus_client import start_http_server, Gauge, Counter
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import RAW_CSV, TARGET_COL, HORIZON, MODELS_DIR
from models.nbeats_forecaster import NBEATSForecaster

# ── Prometheus metrics ────────────────────────────────────────────────
cpu_actual       = Gauge('aiops_cpu_actual',      'Current CPU utilization')
cpu_predicted    = Gauge('aiops_cpu_predicted',   'Predicted CPU (30min ahead)', ['model'])
cpu_error        = Gauge('aiops_prediction_error','MAE between actual and predicted')
scale_out_alert  = Gauge('aiops_scale_out_alert', '1 if scale-out needed, 0 otherwise')
predictions_total = Counter('aiops_predictions_total', 'Total predictions made')

THRESHOLD = 0.75

def run():
    # Load data + model
    df = pd.read_csv(RAW_CSV)
    series = df[TARGET_COL].dropna().values.astype("float32")
    
    try:
        model = NBEATSForecaster()
        model.load(MODELS_DIR / "nbeats.pt")
        print("✓ N-BEATS model loaded")
    except Exception as e:
        print(f"Model load failed ({e}), using random walk")
        model = None

    INPUT_SIZE = 48
    idx = INPUT_SIZE

    start_http_server(8000)
    print("✓ Prometheus metrics server started on :8000/metrics")
    print("  Ctrl+C to stop")

    while True:
        if idx + HORIZON >= len(series):
            idx = INPUT_SIZE

        window = series[idx - INPUT_SIZE: idx]
        actual = float(series[idx])

        if model:
            pred = model.predict(window)
        else:
            pred = np.clip(window[-HORIZON:] + np.random.normal(0, 0.02, HORIZON), 0, 1)

        peak_pred = float(np.max(pred))
        error     = float(np.abs(actual - pred[0]))

        cpu_actual.set(actual)
        cpu_predicted.labels(model='nbeats').set(peak_pred)
        cpu_error.set(error)
        scale_out_alert.set(1 if peak_pred > THRESHOLD else 0)
        predictions_total.inc()

        print(f"  CPU={actual:.3f}  pred_peak={peak_pred:.3f}  alert={'🔴' if peak_pred > THRESHOLD else '🟢'}")
        idx += 1
        time.sleep(5)

if __name__ == "__main__":
    run()
