"""
src/ingestion/data_loader.py

Handles two data sources:
  1. Google BigQuery  (requires GOOGLE_CLOUD_PROJECT env var + auth)
  2. Local CSV        (fast path for Codespace offline development)

Usage:
    from ingestion.data_loader import load_instance_usage
    df = load_instance_usage(source="csv")   # or source="bigquery"
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

try:
    from google.cloud import bigquery
    BQ_AVAILABLE = True
except ImportError:
    BQ_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_CSV, DATETIME_COL, TARGET_COL


# ── BigQuery query ────────────────────────────────────────────────────
BQ_QUERY = """
SELECT
    start_time,
    end_time,
    collection_id,
    instance_index,
    machine_id,
    average_usage.cpus          AS avg_cpu,
    average_usage.memory        AS avg_mem,
    maximum_usage.cpus          AS max_cpu,
    maximum_usage.memory        AS max_mem,
    assigned_memory,
    cycles_per_instruction      AS cpi,
    memory_accesses_per_instruction AS mai
FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_usage`
WHERE
    -- Focus on a manageable slice: first 7 days, top-level jobs only
    start_time BETWEEN 600000000 AND 604800000000   -- microseconds
    AND collection_type = 0                          -- jobs only
    AND average_usage.cpus IS NOT NULL
ORDER BY start_time
LIMIT 5000000
"""


def load_from_bigquery(project_id: str | None = None) -> pd.DataFrame:
    """Pull data from Google BigQuery and return a raw DataFrame."""
    if not BQ_AVAILABLE:
        raise ImportError("google-cloud-bigquery is not installed.")

    project = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise ValueError(
            "Set GOOGLE_CLOUD_PROJECT env var or pass project_id= explicitly."
        )

    logger.info(f"Connecting to BigQuery project: {project}")
    client = bigquery.Client(project=project)
    df = client.query(BQ_QUERY).to_dataframe()
    logger.info(f"Fetched {len(df):,} rows from BigQuery.")
    return df


def load_from_csv(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load local CSV (export from BigQuery or sample file)."""
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found at {path}.\n"
            "Either:\n"
            "  • Run scripts/download_bq_sample.py  (needs GCP auth)\n"
            "  • Run scripts/generate_synthetic.py  (no auth needed)\n"
        )
    logger.info(f"Loading CSV from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows.")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names, parse timestamps, and resample to 5-min grid.

    The trace uses microseconds since epoch-600s.  We convert to a proper
    UTC DatetimeIndex, then aggregate to 5-min mean (matching the native
    measurement window of the trace).
    """
    # ── Normalise column names (BigQuery vs CSV may differ) ──────────
    df.columns = [c.lower().replace(".", "_") for c in df.columns]

    # Handle nested BigQuery struct columns that become flat after export
    cpu_col = next(
        (c for c in df.columns if "cpu" in c and "avg" in c), None
    ) or next((c for c in df.columns if c == "avg_cpu"), None)

    if cpu_col is None:
        raise KeyError(f"Cannot find CPU usage column. Columns: {df.columns.tolist()}")

    df = df.rename(columns={cpu_col: TARGET_COL})

    # ── Parse timestamps ──────────────────────────────────────────────
    # Trace time = microseconds since (trace_start - 600s)
    # We treat the numeric value as microseconds and convert to relative time
    if "start_time" in df.columns:
        df[DATETIME_COL] = pd.to_datetime(
            df["start_time"].astype(float), unit="us", utc=True
        )
    else:
        raise KeyError("No 'start_time' column found.")

    # ── Keep only needed columns ──────────────────────────────────────
    keep = [DATETIME_COL, TARGET_COL]
    for extra in ("avg_mem", "max_cpu", "max_mem", "machine_id", "collection_id"):
        if extra in df.columns:
            keep.append(extra)
    df = df[keep].copy()

    # ── Drop nulls and sort ───────────────────────────────────────────
    df = df.dropna(subset=[TARGET_COL]).sort_values(DATETIME_COL)

    # ── Aggregate to cluster-level 5-min mean ─────────────────────────
    df = (
        df.set_index(DATETIME_COL)[TARGET_COL]
        .resample("5min")
        .mean()
        .interpolate(method="time")
        .reset_index()
        .rename(columns={"start_time_dt": DATETIME_COL})
    )

    logger.info(
        f"Preprocessed series: {len(df)} rows | "
        f"range {df[DATETIME_COL].min()} → {df[DATETIME_COL].max()}"
    )
    return df


def load_instance_usage(source: str = "csv", **kwargs) -> pd.DataFrame:
    """
    Main entry-point.

    Parameters
    ----------
    source : "csv" | "bigquery"
    **kwargs : passed to the underlying loader (e.g. project_id=)
    """
    if source == "bigquery":
        raw = load_from_bigquery(**kwargs)
    else:
        raw = load_from_csv(**kwargs)

    return preprocess(raw)
