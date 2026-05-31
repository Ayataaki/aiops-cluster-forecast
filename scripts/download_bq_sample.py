"""
scripts/download_bq_sample.py

Downloads a 7-day sample from the Google Cluster Trace v3 (cell 'a')
via BigQuery and saves it as data/instance_usage_sample.csv.

Prerequisites:
  1. A GCP project with BigQuery enabled
  2. `gcloud auth application-default login`  (or a service-account key)
  3. GOOGLE_CLOUD_PROJECT set to your billing project

Run:
    export GOOGLE_CLOUD_PROJECT=your-project-id
    python scripts/download_bq_sample.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ingestion.data_loader import BQ_QUERY, preprocess
from config import DATA_DIR

try:
    from google.cloud import bigquery
except ImportError:
    print("Install google-cloud-bigquery:  pip install google-cloud-bigquery")
    sys.exit(1)

project = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not project:
    print("Set GOOGLE_CLOUD_PROJECT environment variable.")
    sys.exit(1)

print(f"Downloading from BigQuery project: {project} …")
client = bigquery.Client(project=project)

print("Running query (this may take 1-3 min for the first run) …")
df_raw = client.query(BQ_QUERY).to_dataframe()
print(f"Fetched {len(df_raw):,} rows.")

df = preprocess(df_raw)
out = DATA_DIR / "instance_usage_sample.csv"
df.to_csv(out, index=False)
print(f"✅  Saved → {out}  ({len(df):,} rows)")
