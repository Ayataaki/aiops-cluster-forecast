# AIOps Predictive Observability — Google Cluster Trace v3

> **Proactive auto-scaling for Kubernetes via time-series forecasting of CPU load**
> Built on the Google Borg cluster-usage traces (May 2019), deployable in a GitHub Codespace in one command.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Quick Start (Codespace)](#quick-start-codespace)
4. [Dataset](#dataset)
5. [Statistical Methodology](#statistical-methodology)
6. [Models](#models)
7. [API Reference](#api-reference)
8. [Project Structure](#project-structure)
9. [Results Interpretation](#results-interpretation)

---

## Project Overview

Reactive infrastructure management (waiting for CPU to hit 90% before scaling) causes
latency spikes and SLA violations. This project implements a **predictive AIOps pipeline**
that forecasts cluster CPU load 30 minutes ahead and triggers proactive horizontal
pod autoscaling (HPA) before saturation occurs.

**Key design principles:**
- No model is accepted without beating statistical baselines (Naive, Holt-Winters)
- Strict chronological train/val/test splits — zero data leakage
- All models run on CPU (no GPU required) in a GitHub Codespace

---

## Architecture

```
[ Google Cluster Trace v3 / BigQuery ]
             │
             ▼
   [ data_loader.py ]  ──── preprocessing, resampling to 5-min grid
             │
             ▼
   [ diagnostics.py ]  ──── ADF, KPSS, STL decomposition, ACF/PACF
             │
             ▼
   [ feature_engineer.py ]  ── lags, rolling stats, Fourier, calendar
             │
      ┌──────┴──────────────────┐
      ▼                         ▼
[ baselines.py ]        [ xgboost_forecaster.py ]
  Naive / HW                XGBoost (direct H-step)
                                    │
                        [ transformer_model.py ]
                          Informer / Autoformer (NeuralForecast)
             │
             ▼
   [ FastAPI – main.py ]
     POST /predict
     POST /scaling-decision  ──► Kubernetes HPA
```

---

## Quick Start (Codespace)

```bash
# 1. Install dependencies (runs automatically on Codespace creation)
pip install -r requirements.txt

# 2. Generate synthetic data (no GCP auth needed)
make data

# 3. Run statistical diagnostics
make diagnostics
# → results/diagnostics.png

# 4. Train baselines + XGBoost (~2 min on CPU)
make train
# → results/model_comparison.csv + model_comparison.png

# 5. Start the prediction API
make api
# → http://localhost:8000/docs

# 6. Run tests
make test
```

To train Transformer models (adds ~10 min):
```bash
make train-all
```

---

## Dataset

**Source:** Google Cluster Workload Traces v3 (May 2019)
[https://research.google/tools/datasets/google-cluster-workload-traces-2019/](https://research.google/tools/datasets/google-cluster-workload-traces-2019/)

The trace covers 8 Borg compute cells for the entire month of May 2019.
Key fields used in this project (from the `instance_usage` table):

| Field | Description |
|---|---|
| `start_time` | Measurement window start (microseconds) |
| `average_usage.cpus` | Mean CPU rate (NCU = Normalised Compute Unit ∈ [0,1]) |
| `average_usage.memory` | Mean memory fraction ∈ [0,1] |

**Accessing real data via BigQuery:**
```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
python scripts/download_bq_sample.py
```

**Offline development (no GCP account):**
```bash
python scripts/generate_synthetic.py
# Creates data/instance_usage_sample.csv with realistic synthetic data
```

---

## Statistical Methodology

### 1. Stationarity Diagnostics

Before any modelling, the CPU series undergoes rigorous statistical testing:

**ADF test** (H₀: unit root exists):
- Reject H₀ if p-value < 0.05 → series is stationary

**KPSS test** (H₀: series is stationary):
- Fail to reject H₀ if p-value > 0.05 → series is stationary

If both tests agree on stationarity, no differencing is applied.
If non-stationary, automatic differencing d=1 or d=2 is applied.

### 2. Decomposition

STL (Season-Trend using LOESS) is used over classical additive decomposition
because it is robust to outliers (CPU spikes) and handles varying seasonality.

Components:
- **Trend T_t** — slow drift in baseline utilisation
- **Seasonal S_t** — daily (288 periods) and weekly cycles
- **Residual ε_t** — stochastic component

### 3. Feature Engineering

| Category | Features |
|---|---|
| Lags | Y_{t-1}, Y_{t-2}, Y_{t-6}, Y_{t-12}, Y_{t-24}, Y_{t-72}, Y_{t-144}, Y_{t-288}, Y_{t-576} |
| Rolling | mean/std/min/max over 30m, 1h, 2h, 4h, 24h windows |
| EWM | Exponentially weighted mean (spans 1h, 4h, 24h) |
| Calendar | hour, day-of-week, is_weekend, is_business_hour |
| Fourier | sin/cos pairs for daily and weekly cycles (3+2 harmonics) |

### 4. Train / Val / Test Split

```
──────────────────────────────────────────────────────► time
│◄──── 70% train ────►│◄─ 15% val ─►│◄─ 15% test ─►│

IMPORTANT: No random shuffling. Test = always the LAST N observations.
```

---

## Models

### Baseline Models (must be beaten)

| Model | Description |
|---|---|
| Naive | Y_{t+h} = Y_t (persist last value) |
| Seasonal Naive | Y_{t+h} = Y_{t-288+h} (same slot yesterday) |
| Holt-Winters | Triple exponential smoothing, damped additive trend |

### XGBoost (direct multi-step)

One XGBRegressor per horizon step h ∈ {1…6}.
Features are scaled with RobustScaler (resistant to burst spikes).
Early stopping on validation RMSE (patience=30).

### Informer (NeuralForecast)

Transformer architecture with ProbSparse self-attention.
Reduces memory complexity from O(L²) to O(L log L) —
suitable for long input windows (INPUT_SIZE=48 periods = 4 hours).

### Autoformer

Auto-correlation mechanism + seasonal decomposition block.
Designed for long-sequence forecasting with series decomposition
embedded in the attention module.

---

## API Reference

Start the server: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload`

Then visit: `http://localhost:8000/docs` for the interactive Swagger UI.

### POST /predict

```json
{
  "values": [0.22, 0.25, ...],  // >= 12 recent CPU readings (NCU)
  "model": "xgboost"             // "xgboost" | "informer" | "autoformer"
}
```

Returns a 6-step (30-min) forecast.

### POST /scaling-decision

Same input as `/predict`. Returns:

```json
{
  "scale_out": true,
  "reason": "Predicted CPU reaches 82% in next 30 min",
  "recommended_replicas_delta": 2,
  "max_predicted_cpu": 0.82,
  "threshold": 0.75,
  "forecast": { ... }
}
```

### GET /results

Returns the model benchmark table from `results/model_comparison.csv`.

---

## Project Structure

```
aiops-cluster-forecast/
├── .devcontainer/
│   └── devcontainer.json          # GitHub Codespace config
├── data/
│   └── instance_usage_sample.csv  # Dataset (generated or downloaded)
├── notebooks/
│   └── 01_exploration.ipynb       # Interactive EDA
├── results/                       # Plots + CSVs (auto-generated)
├── models_saved/                  # Serialised model weights
├── scripts/
│   ├── generate_synthetic.py      # Offline data generator
│   └── download_bq_sample.py      # BigQuery downloader
├── src/
│   ├── config.py                  # Central configuration
│   ├── ingestion/
│   │   └── data_loader.py         # BigQuery + CSV ingestion
│   ├── analytics/
│   │   ├── diagnostics.py         # ADF, KPSS, STL, ACF/PACF
│   │   └── feature_engineer.py    # Feature matrix builder
│   ├── models/
│   │   ├── baselines.py           # Naive, Seasonal Naive, HW
│   │   ├── xgboost_forecaster.py  # XGBoost direct multi-step
│   │   ├── transformer_model.py   # Informer / Autoformer
│   │   └── model_comparison.py    # Benchmark runner
│   └── api/
│       └── main.py                # FastAPI scaling-decision API
├── tests/
│   └── test_pipeline.py           # Pytest end-to-end tests
├── Makefile                       # Convenience commands
└── requirements.txt
```

---

## Results Interpretation

After `make train`, open `results/model_comparison.csv`.

**Minimum success criteria:**
- XGBoost RMSE < Holt-Winters RMSE
- Informer RMSE < Holt-Winters RMSE × 0.85 (15% improvement)
- All models beat Naive RMSE

**Scale-out trigger logic:**
If the maximum predicted CPU over the next 30 minutes exceeds 75% NCU,
the API recommends scaling out. Each additional 10% above threshold
adds one replica. This gives the cluster manager a 30-minute head-start
before the actual load arrives.
