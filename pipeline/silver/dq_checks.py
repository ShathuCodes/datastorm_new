"""
pipeline/silver/dq_checks.py
=============================
SILVER LAYER — Reusable, parameterizable Data Quality checks.

Every check has the same contract:
    Input : df (DataFrame), metadata args, dataset_name (str)
    Output: (clean_df, rejected_df, report_dict)

'rejected_df' always contains an extra column 'dq_failure_reason'
so rejected records can be quarantined with a documented cause.
"""

import os
import re
import pandas as pd
from datetime import datetime
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


# ─────────────────────────────────────────────────────────────────
# QUARANTINE HELPER
# ─────────────────────────────────────────────────────────────────

def quarantine(rejected_df: pd.DataFrame, reason: str, dataset_name: str) -> None:
    """
    Write rejected records to the silver/rejected/ store with
    the failure reason documented.  Records are NEVER silently dropped.
    """
    if rejected_df.empty:
        return
    os.makedirs(config.SILVER_REJECTED_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_reason = re.sub(r"[^a-zA-Z0-9_]", "_", reason)[:40]
    path = os.path.join(
        config.SILVER_REJECTED_DIR,
        f"{dataset_name}_{safe_reason}_{ts}.csv"
    )
    rejected_df = rejected_df.copy()
    rejected_df["dq_failure_reason"] = reason
    rejected_df.to_csv(path, index=False)
    print(f"    [DQ] ✗ Quarantined {len(rejected_df):,} rows → {path}")


# ─────────────────────────────────────────────────────────────────
# CHECK 1 — DUPLICATE CHECK
# ─────────────────────────────────────────────────────────────────

def check_duplicates(
    df: pd.DataFrame,
    key_cols: list,
    dataset_name: str,
    keep: str = "first"
) -> tuple:
    """
    Detect exact duplicate records based on a configurable primary key
    (single column or composite).

    Parameters
    ----------
    key_cols : list of column names that form the primary / composite key
    keep     : 'first' keeps the first occurrence; duplicates are quarantined
    """
    mask_dup = df.duplicated(subset=key_cols, keep=keep)
    clean    = df[~mask_dup].copy()
    rejected = df[mask_dup].copy()

    report = {
        "check"       : "duplicate",
        "key_cols"    : key_cols,
        "total_rows"  : len(df),
        "clean_rows"  : len(clean),
        "rejected_rows": len(rejected),
    }
    print(f"    [DQ] Duplicate check on {key_cols}: "
          f"{len(rejected):,} duplicates found out of {len(df):,}")

    quarantine(rejected, f"duplicate_key_{'_'.join(key_cols)}", dataset_name)
    return clean, rejected, report


# ─────────────────────────────────────────────────────────────────
# CHECK 2 — NULL / MISSING CHECK
# ─────────────────────────────────────────────────────────────────

def check_nulls(
    df: pd.DataFrame,
    mandatory_cols: list,
    dataset_name: str
) -> tuple:
    """
    Flag records where any mandatory field is null, empty string, or
    whitespace-only.
    """
    df_work = df.copy()
    # treat empty strings as null
    for col in mandatory_cols:
        if col in df_work.columns:
            df_work[col] = df_work[col].replace(r"^\s*$", pd.NA, regex=True)

    mask_null = df_work[mandatory_cols].isnull().any(axis=1)
    clean     = df[~mask_null].copy()
    rejected  = df[mask_null].copy()

    report = {
        "check"        : "null",
        "mandatory_cols": mandatory_cols,
        "total_rows"   : len(df),
        "clean_rows"   : len(clean),
        "rejected_rows": len(rejected),
    }
    print(f"    [DQ] Null check on {mandatory_cols}: "
          f"{len(rejected):,} rows with missing mandatory fields")

    quarantine(rejected, f"null_in_mandatory_fields", dataset_name)
    return clean, rejected, report


# ─────────────────────────────────────────────────────────────────
# CHECK 3 — REFERENTIAL INTEGRITY CHECK
# ─────────────────────────────────────────────────────────────────

def check_referential_integrity(
    df: pd.DataFrame,
    fk_col: str,
    ref_values: set,
    dataset_name: str
) -> tuple:
    """
    Validate that every value in fk_col exists in ref_values.
    Orphan records (FK not in reference set) are quarantined.

    Parameters
    ----------
    fk_col     : column in df that is the foreign key
    ref_values : set of valid values (e.g. set of all Outlet_IDs)
    """
    if fk_col not in df.columns:
        print(f"    [DQ] WARN: column '{fk_col}' not found — skipping ref check")
        return df.copy(), pd.DataFrame(), {"check": "referential_integrity", "skipped": True}

    mask_invalid = ~df[fk_col].isin(ref_values)
    clean        = df[~mask_invalid].copy()
    rejected     = df[mask_invalid].copy()

    report = {
        "check"        : "referential_integrity",
        "fk_col"       : fk_col,
        "total_rows"   : len(df),
        "clean_rows"   : len(clean),
        "rejected_rows": len(rejected),
    }
    print(f"    [DQ] Ref-integrity check '{fk_col}': "
          f"{len(rejected):,} rows with unknown FK values")

    quarantine(rejected, f"unknown_fk_{fk_col}", dataset_name)
    return clean, rejected, report


# ─────────────────────────────────────────────────────────────────
# CHECK 4 — VALUE RANGE CHECK
# ─────────────────────────────────────────────────────────────────

def check_value_range(
    df: pd.DataFrame,
    col: str,
    min_val: float,
    max_val: float,
    dataset_name: str
) -> tuple:
    """
    Assert that a numeric column falls within [min_val, max_val].
    Rows outside this window are quarantined.
    """
    if col not in df.columns:
        print(f"    [DQ] WARN: column '{col}' not found — skipping range check")
        return df.copy(), pd.DataFrame(), {"check": "value_range", "skipped": True}

    numeric = pd.to_numeric(df[col], errors="coerce")
    mask_out = (numeric < min_val) | (numeric > max_val) | numeric.isna()
    clean    = df[~mask_out].copy()
    rejected = df[mask_out].copy()

    report = {
        "check"        : "value_range",
        "col"          : col,
        "min_val"      : min_val,
        "max_val"      : max_val,
        "total_rows"   : len(df),
        "clean_rows"   : len(clean),
        "rejected_rows": len(rejected),
    }
    print(f"    [DQ] Range check '{col}' [{min_val}, {max_val}]: "
          f"{len(rejected):,} out-of-range rows")

    quarantine(rejected, f"out_of_range_{col}", dataset_name)
    return clean, rejected, report


# ─────────────────────────────────────────────────────────────────
# CHECK 5 — FORMAT / TYPE CHECK
# ─────────────────────────────────────────────────────────────────

def check_format(
    df: pd.DataFrame,
    col: str,
    pattern: str,
    dataset_name: str
) -> tuple:
    """
    Validate that string values in 'col' match a given regex pattern.

    Examples
    --------
    Outlet IDs   : r'^OUT_\\d{5}$'
    Dates        : r'^\\d{4}-\\d{2}-\\d{2}$'
    Distributor  : r'^DIST_(W|C|NW|S)_0[12345]$'
    """
    if col not in df.columns:
        print(f"    [DQ] WARN: column '{col}' not found — skipping format check")
        return df.copy(), pd.DataFrame(), {"check": "format", "skipped": True}

    compiled  = re.compile(pattern)
    mask_bad  = ~df[col].astype(str).str.match(compiled)
    clean     = df[~mask_bad].copy()
    rejected  = df[mask_bad].copy()

    report = {
        "check"        : "format",
        "col"          : col,
        "pattern"      : pattern,
        "total_rows"   : len(df),
        "clean_rows"   : len(clean),
        "rejected_rows": len(rejected),
    }
    print(f"    [DQ] Format check '{col}' /{pattern}/: "
          f"{len(rejected):,} non-conforming rows")

    quarantine(rejected, f"bad_format_{col}", dataset_name)
    return clean, rejected, report


# ─────────────────────────────────────────────────────────────────
# ORCHESTRATOR — run a list of checks in sequence
# ─────────────────────────────────────────────────────────────────

def run_checks(df: pd.DataFrame, checks: list, dataset_name: str) -> tuple:
    """
    Run a list of check specifications against a DataFrame sequentially.
    Each check specification is a dict:
        {"type": "duplicates", "key_cols": [...]}
        {"type": "nulls",      "mandatory_cols": [...]}
        {"type": "ref_integrity", "fk_col": "...", "ref_values": set(...)}
        {"type": "range",      "col": "...", "min": 0, "max": 9999}
        {"type": "format",     "col": "...", "pattern": "..."}

    Returns (final_clean_df, full_report_list).
    """
    current   = df.copy()
    all_reports = []

    for spec in checks:
        ctype = spec["type"]

        if ctype == "duplicates":
            current, _, rpt = check_duplicates(
                current, spec["key_cols"], dataset_name,
                keep=spec.get("keep", "first")
            )
        elif ctype == "nulls":
            current, _, rpt = check_nulls(
                current, spec["mandatory_cols"], dataset_name
            )
        elif ctype == "ref_integrity":
            current, _, rpt = check_referential_integrity(
                current, spec["fk_col"], spec["ref_values"], dataset_name
            )
        elif ctype == "range":
            current, _, rpt = check_value_range(
                current, spec["col"], spec["min"], spec["max"], dataset_name
            )
        elif ctype == "format":
            current, _, rpt = check_format(
                current, spec["col"], spec["pattern"], dataset_name
            )
        else:
            print(f"    [DQ] Unknown check type '{ctype}' — skipped")
            rpt = {"check": ctype, "skipped": True}

        all_reports.append(rpt)

    print(f"    [DQ] '{dataset_name}' after all checks: {len(current):,} clean rows "
          f"(started with {len(df):,})")
    return current, all_reports
