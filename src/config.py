"""
config.py – Central project configuration.
All paths, hyper-parameters, and thresholds live here so that
every module can import a single source of truth.
"""

from pathlib import Path

# ── Project root ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent   # repo root
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models_saved"
RESULTS_DIR = ROOT / "results"

for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Raw data file (place Google CSV here after download) ─────────────
# BigQuery export → data/instance_usage_sample.csv
RAW_CSV = DATA_DIR / "instance_usage_sample.csv"

# ── Target column and key columns ───────────────────────────────────
TARGET_COL = "avg_cpu"          # renamed from average_usage.cpus
DATETIME_COL = "start_time_dt"  # parsed datetime index

# ── Train / Validation / Test split ratios ───────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST = remaining 15%  (always the last N rows – NO random shuffle!)

# ── Forecasting horizon ──────────────────────────────────────────────
HORIZON = 6                     # steps ahead  (6 × 5 min = 30 min)
INPUT_SIZE = 48                 # look-back window for deep models (4 h)
FREQ = "5min"                   # pandas frequency string

# ── Stationarity thresholds ──────────────────────────────────────────
ADF_PVALUE_THRESHOLD  = 0.05
KPSS_PVALUE_THRESHOLD = 0.05

# ── Scaling alert threshold ──────────────────────────────────────────
CPU_SCALE_OUT_THRESHOLD = 0.75  # trigger scale-out when pred > 75% NCU

# ── Model hyper-parameters (lightweight defaults for Codespace) ──────
INFORMER_CFG = dict(
    input_size=INPUT_SIZE,
    h=HORIZON,
    hidden_size=64,
    n_head=4,
    e_layers=2,
    d_layers=1,
    max_steps=300,
    val_check_steps=50,
    early_stop_patience_steps=5,
)

AUTOFORMER_CFG = dict(
    input_size=INPUT_SIZE,
    h=HORIZON,
    hidden_size=64,
    n_head=4,
    e_layers=2,
    d_layers=1,
    moving_avg=25,
    max_steps=300,
    val_check_steps=50,
    early_stop_patience_steps=5,
)

XGBOOST_CFG = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

# ── Holt-Winters ──────────────────────────────────────────────────────
HW_SEASONAL_PERIODS = 288       # 24 h × 12 periods/h  (5-min data)
