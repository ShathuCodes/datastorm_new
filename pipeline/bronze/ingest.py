"""
pipeline/bronze/ingest.py
=========================
BRONZE LAYER — Raw ingestion.
Reads CSV files exactly as-is with NO transformations.
Saves a timestamped copy to data/bronze/ for full auditability.
"""

import os
import pandas as pd
from datetime import datetime
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def _ensure_dirs():
    os.makedirs(config.BRONZE_DIR, exist_ok=True)


def load_raw(name: str) -> pd.DataFrame:
    """
    Load one raw CSV file with zero transformation.
    'name' must be a key in config.RAW_FILES.
    """
    filename = config.RAW_FILES[name]
    path = os.path.join(config.RAW_DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[BRONZE] Raw file not found: {path}\n"
            f"  → Place '{filename}' inside  data/raw/"
        )
    df = pd.read_csv(path, dtype=str, low_memory=False)
    print(f"  [BRONZE] Loaded '{name}': {len(df):,} rows × {len(df.columns)} cols")
    return df


def save_bronze(df: pd.DataFrame, name: str) -> str:
    """Save raw dataframe to bronze layer and return the saved path."""
    _ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(config.BRONZE_DIR, f"{name}_bronze_{ts}.csv")
    df.to_csv(out_path, index=False)
    print(f"  [BRONZE] Saved  '{name}' → {out_path}")
    return out_path


def ingest_all() -> dict:
    """
    Ingest all raw files, save to bronze, and return a dict of DataFrames.
    This is the ONLY function called by main.py for the bronze layer.
    """
    print("\n" + "═" * 60)
    print("  BRONZE LAYER — Raw Ingestion")
    print("═" * 60)

    bronze = {}
    for name in config.RAW_FILES:
        try:
            df = load_raw(name)
            save_bronze(df, name)
            bronze[name] = df
        except FileNotFoundError as e:
            print(f"  [BRONZE] WARNING — {e}")
            bronze[name] = pd.DataFrame()   # empty placeholder

    print(f"\n  [BRONZE] Done. {len([v for v in bronze.values() if len(v) > 0])} datasets ingested.\n")
    return bronze
