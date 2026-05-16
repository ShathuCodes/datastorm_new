"""
pipeline/silver/clean.py
=========================
SILVER LAYER — Data cleaning logic for each dataset.
Applies reusable DQ checks from dq_checks.py.
Produces sanitised DataFrames saved to data/silver/.
"""

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from pipeline.silver.dq_checks import run_checks, quarantine


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _save_silver(df: pd.DataFrame, name: str) -> str:
    os.makedirs(config.SILVER_DIR, exist_ok=True)
    path = os.path.join(config.SILVER_DIR, f"{name}_silver.csv")
    df.to_csv(path, index=False)
    print(f"    [SILVER] Saved '{name}' → {path}  ({len(df):,} rows)")
    return path


def _col(dataset: str, field: str) -> str:
    """Return the configured column name for a field in a dataset."""
    return config.COLS[dataset].get(field, field)


def _rename_to_std(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Rename raw column names to standard internal names."""
    mapping = {v: k for k, v in config.COLS[dataset].items()}
    return df.rename(columns=mapping)


# ─────────────────────────────────────────────────────────────────
# 1. OUTLET MASTER
# ─────────────────────────────────────────────────────────────────

def clean_outlet_master(raw_df: pd.DataFrame) -> pd.DataFrame:
    print("\n  ── Cleaning: outlet_master ──")
    df = raw_df.copy()

    # Rename columns to standard names
    df = _rename_to_std(df, "outlet_master")

    # Strip whitespace from string columns
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype(str).str.strip()

    checks = [
        {"type": "duplicates",    "key_cols": ["outlet_id"]},
        {"type": "nulls",         "mandatory_cols": ["outlet_id", "outlet_type", "distributor_id", "province"]},
        {"type": "ref_integrity", "fk_col": "distributor_id", "ref_values": set(config.VALID_DISTRIBUTOR_IDS)},
        {"type": "ref_integrity", "fk_col": "province",       "ref_values": set(config.VALID_PROVINCES)},
    ]
    clean, _ = run_checks(df, checks, "outlet_master")

    # Standardise outlet_type to lower-case for consistent grouping
    clean["outlet_type"] = clean["outlet_type"].str.lower().str.strip()

    _save_silver(clean, "outlet_master")
    return clean


# ─────────────────────────────────────────────────────────────────
# 2. COORDINATES
# ─────────────────────────────────────────────────────────────────

def clean_coordinates(raw_df: pd.DataFrame, valid_outlet_ids: set) -> pd.DataFrame:
    print("\n  ── Cleaning: coordinates ──")
    df = raw_df.copy()
    df = _rename_to_std(df, "coordinates")

    for col in ["lat", "lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    checks = [
        {"type": "duplicates",    "key_cols": ["outlet_id"]},
        {"type": "nulls",         "mandatory_cols": ["outlet_id", "lat", "lon"]},
        {"type": "ref_integrity", "fk_col": "outlet_id", "ref_values": valid_outlet_ids},
        {"type": "range",         "col": "lat", "min": config.LAT_MIN, "max": config.LAT_MAX},
        {"type": "range",         "col": "lon", "min": config.LON_MIN, "max": config.LON_MAX},
    ]
    clean, _ = run_checks(df, checks, "coordinates")
    _save_silver(clean, "coordinates")
    return clean


# ─────────────────────────────────────────────────────────────────
# 3. TRANSACTIONS
# ─────────────────────────────────────────────────────────────────

def clean_transactions(raw_df: pd.DataFrame, valid_outlet_ids: set) -> pd.DataFrame:
    print("\n  ── Cleaning: transactions ──")
    df = raw_df.copy()
    df = _rename_to_std(df, "transactions")

    # ── Type coercions ────────────────────────────────────────────
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # Try multiple date formats (legacy systems use different formats)
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y%m%d"]:
        try:
            df["date"] = pd.to_datetime(df["date"], format=fmt, errors="coerce")
            if df["date"].notna().sum() > len(df) * 0.5:
                break
        except Exception:
            pass
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Strip whitespace
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype(str).str.strip()

    # ── DQ Checks ─────────────────────────────────────────────────
    checks = [
        {"type": "nulls",         "mandatory_cols": ["outlet_id", "date", "volume"]},
        {"type": "ref_integrity", "fk_col": "outlet_id", "ref_values": valid_outlet_ids},
        {"type": "ref_integrity", "fk_col": "distributor_id", "ref_values": set(config.VALID_DISTRIBUTOR_IDS)},
        {"type": "range",         "col": "volume", "min": config.VOLUME_MIN, "max": config.VOLUME_MAX},
    ]
    clean, _ = run_checks(df, checks, "transactions")

    # ── Detect and quarantine ghost/zero transactions ──────────────
    # Zero-volume records with a valid invoice number are suspicious
    # (likely automated ghost entries from SFA system)
    zero_mask = clean["volume"] == 0.0
    ghost_records = clean[zero_mask].copy()
    if not ghost_records.empty:
        quarantine(ghost_records, "zero_volume_ghost_entry", "transactions")
    clean = clean[~zero_mask].copy()
    print(f"    [SILVER] Removed {len(ghost_records):,} zero-volume ghost entries")

    # ── Add helper time columns ────────────────────────────────────
    clean["year"]       = clean["date"].dt.year
    clean["month"]      = clean["date"].dt.month
    clean["year_month"] = clean["date"].dt.to_period("M").astype(str)

    _save_silver(clean, "transactions")
    return clean


# ─────────────────────────────────────────────────────────────────
# 4. SEASONALITY
# ─────────────────────────────────────────────────────────────────

def clean_seasonality(raw_df: pd.DataFrame) -> pd.DataFrame:
    print("\n  ── Cleaning: seasonality ──")
    df = raw_df.copy()
    df = _rename_to_std(df, "seasonality")

    df["index"] = pd.to_numeric(df["index"], errors="coerce")

    # Month may be a name ("January") or a number ("1")
    month_name_map = {
        "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
        "july":7,"august":8,"september":9,"october":10,"november":11,"december":12
    }
    month_numeric = pd.to_numeric(df["month"], errors="coerce")
    month_named   = df["month"].astype(str).str.lower().str.strip().map(month_name_map)
    df["month"]   = month_numeric.combine_first(month_named).astype("Int64")

    checks = [
        {"type": "nulls", "mandatory_cols": ["distributor_id", "month", "index"]},
        {"type": "range", "col": "month", "min": 1, "max": 12},
        {"type": "range", "col": "index", "min": 0.1, "max": 5.0},
    ]
    clean, _ = run_checks(df, checks, "seasonality")
    _save_silver(clean, "seasonality")
    return clean


# ─────────────────────────────────────────────────────────────────
# 5. HOLIDAYS
# ─────────────────────────────────────────────────────────────────

def clean_holidays(raw_df: pd.DataFrame) -> pd.DataFrame:
    print("\n  ── Cleaning: holidays ──")
    df = raw_df.copy()
    df = _rename_to_std(df, "holidays")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    checks = [
        {"type": "nulls", "mandatory_cols": ["date", "name"]},
    ]
    clean, _ = run_checks(df, checks, "holidays")
    clean["month"] = clean["date"].dt.month
    clean["year"]  = clean["date"].dt.year
    _save_silver(clean, "holidays")
    return clean


# ─────────────────────────────────────────────────────────────────
# SILVER LAYER ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def clean_all(bronze: dict) -> dict:
    print("\n" + "═" * 60)
    print("  SILVER LAYER — Data Cleaning & Quality Checks")
    print("═" * 60)

    outlet_master = clean_outlet_master(bronze["outlet_master"])
    valid_ids     = set(outlet_master["outlet_id"].dropna().unique())

    coordinates   = clean_coordinates(bronze["coordinates"], valid_ids)
    transactions  = clean_transactions(bronze["transactions"], valid_ids)
    seasonality   = clean_seasonality(bronze["seasonality"])
    holidays      = clean_holidays(bronze["holidays"])

    silver = {
        "outlet_master": outlet_master,
        "coordinates"  : coordinates,
        "transactions" : transactions,
        "seasonality"  : seasonality,
        "holidays"     : holidays,
    }
    print("\n  [SILVER] All datasets cleaned.\n")
    return silver
