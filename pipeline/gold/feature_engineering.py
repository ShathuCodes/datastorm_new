"""
pipeline/gold/feature_engineering.py
======================================
GOLD LAYER — Feature Engineering.
Builds the analytical feature table from all Silver datasets.
No sklearn — pure pandas + numpy + manual math.
"""

import os
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


# ─────────────────────────────────────────────────────────────────
# MATH UTILITIES  (no external stats libraries)
# ─────────────────────────────────────────────────────────────────

def _percentile(series: pd.Series, p: float) -> float:
    """
    Compute the p-th percentile of a pandas Series from scratch
    using linear interpolation (same as numpy default).
    """
    vals = sorted(series.dropna().tolist())
    n = len(vals)
    if n == 0:
        return 0.0
    if n == 1:
        return vals[0]
    idx   = (p / 100.0) * (n - 1)
    lo    = int(idx)
    hi    = min(lo + 1, n - 1)
    frac  = idx - lo
    return vals[lo] + frac * (vals[hi] - vals[lo])


def _ols_slope(y_values: list) -> float:
    """
    Ordinary Least Squares slope (β₁) for y ~ t, where t = 0,1,...,n-1.
    β₁ = [n·Σ(t·y) - Σt·Σy] / [n·Σt² - (Σt)²]
    Returns 0.0 if fewer than 2 data points.
    """
    n = len(y_values)
    if n < 2:
        return 0.0
    t    = list(range(n))
    sum_t  = sum(t)
    sum_y  = sum(y_values)
    sum_t2 = sum(ti * ti for ti in t)
    sum_ty = sum(t[i] * y_values[i] for i in range(n))
    denom  = n * sum_t2 - sum_t * sum_t
    if denom == 0:
        return 0.0
    return (n * sum_ty - sum_t * sum_y) / denom


def _coefficient_of_variation(series: pd.Series) -> float:
    """
    CV = std / mean.  High CV suggests supply-constrained months
    mixed with unconstrained months — a signal of latent demand.
    """
    vals = series.dropna()
    if len(vals) < 2 or vals.mean() == 0:
        return 0.0
    return float(vals.std() / vals.mean())


# ─────────────────────────────────────────────────────────────────
# STEP 1 — MONTHLY AGGREGATION
# ─────────────────────────────────────────────────────────────────

def compute_monthly_agg(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transactions to outlet × year_month level.
    This is the foundation for all outlet-level features.
    """
    agg = (
        transactions
        .groupby(["outlet_id", "year_month"], as_index=False)
        .agg(
            monthly_volume = ("volume", "sum"),
            txn_count      = ("volume", "count"),
            distributor_id = ("distributor_id", "first"),
            year           = ("year", "first"),
            month          = ("month", "first"),
        )
    )
    agg["year_month_dt"] = pd.to_datetime(agg["year_month"])
    agg = agg.sort_values(["outlet_id", "year_month_dt"])
    return agg


# ─────────────────────────────────────────────────────────────────
# STEP 2 — HISTORICAL FEATURES PER OUTLET
# ─────────────────────────────────────────────────────────────────

def compute_outlet_features(monthly_agg: pd.DataFrame,
                             holidays: pd.DataFrame) -> pd.DataFrame:
    """
    For each outlet compute:
      - active_months       : number of months with recorded sales
      - mean_volume         : mean monthly volume (all months)
      - median_volume       : median monthly volume
      - p85_volume          : 85th percentile monthly volume (self-ceiling)
      - p90_volume          : 90th percentile monthly volume
      - max_volume          : peak single-month volume
      - cv_volume           : coefficient of variation (supply constraint signal)
      - trend_slope         : OLS slope over last N months (growth signal)
      - trend_slope_norm    : trend_slope / mean_volume (dimensionless)
      - jan_avg             : historical average for January months only
      - holiday_density     : holidays per month (catchment proxy from calendar)
    """

    # Pre-compute January holiday count per year for context
    jan_holidays = 0
    if not holidays.empty and "month" in holidays.columns:
        jan_holidays = int((holidays["month"] == config.TARGET_MONTH_NUM).sum())

    records = []
    for outlet_id, grp in monthly_agg.groupby("outlet_id"):
        grp = grp.sort_values("year_month_dt")
        vols = grp["monthly_volume"].tolist()
        n    = len(vols)

        if n < config.MIN_ACTIVE_MONTHS:
            continue   # not enough data — will handle as new/sparse outlet later

        series_vol = grp["monthly_volume"]

        # ── Basic statistics ────────────────────────────────────────
        mean_v   = float(series_vol.mean())
        median_v = float(series_vol.median())
        p85_v    = _percentile(series_vol, config.SELF_PERCENTILE)
        p90_v    = _percentile(series_vol, config.PEER_PERCENTILE)
        max_v    = float(series_vol.max())
        cv_v     = _coefficient_of_variation(series_vol)

        # ── January-specific average ────────────────────────────────
        jan_rows = grp[grp["month"] == config.TARGET_MONTH_NUM]["monthly_volume"]
        jan_avg  = float(jan_rows.mean()) if len(jan_rows) > 0 else mean_v

        # ── OLS Trend (last N months) ───────────────────────────────
        trend_months = min(config.TREND_MONTHS, n)
        recent_vols  = vols[-trend_months:]
        slope        = _ols_slope(recent_vols)
        slope_norm   = slope / mean_v if mean_v > 0 else 0.0

        # ── Distributor & time meta ─────────────────────────────────
        dist_id = grp["distributor_id"].iloc[-1]
        last_ym = grp["year_month_dt"].iloc[-1]

        records.append({
            "outlet_id"      : outlet_id,
            "distributor_id" : dist_id,
            "active_months"  : n,
            "mean_volume"    : mean_v,
            "median_volume"  : median_v,
            "p85_volume"     : p85_v,
            "p90_volume"     : p90_v,
            "max_volume"     : max_v,
            "cv_volume"      : cv_v,
            "jan_avg"        : jan_avg,
            "trend_slope"    : slope,
            "trend_slope_norm": slope_norm,
            "last_active_ym" : str(last_ym.to_period("M")),
            "jan_holidays"   : jan_holidays,
        })

    feat = pd.DataFrame(records)
    print(f"    [GOLD] Outlet features computed: {len(feat):,} outlets")
    return feat


# ─────────────────────────────────────────────────────────────────
# STEP 3 — PEER GROUP STATISTICS
# ─────────────────────────────────────────────────────────────────

def add_peer_group_features(features: pd.DataFrame,
                             outlet_master: pd.DataFrame) -> pd.DataFrame:
    """
    Segment outlets into peer groups by (outlet_type × province).
    Within each group compute:
      - peer_p50   : median of p90_volume across the peer group
      - peer_p75   : 75th percentile of p90_volume
      - peer_p90   : 90th percentile of p90_volume  ← primary ceiling
      - peer_count : number of outlets in the group
    """
    # Merge outlet_type and province from master
    meta_cols = ["outlet_id", "outlet_type", "province"]
    master_sub = outlet_master[meta_cols].drop_duplicates("outlet_id")
    feat = features.merge(master_sub, on="outlet_id", how="left")

    # Fill missing outlet_type / province with "unknown"
    feat["outlet_type"] = feat["outlet_type"].fillna("unknown")
    feat["province"]    = feat["province"].fillna("unknown")

    # ── Compute peer group stats ─────────────────────────────────
    def peer_stats(grp):
        vals  = grp["p90_volume"]
        p50   = _percentile(vals, 50)
        p75   = _percentile(vals, 75)
        p90   = _percentile(vals, config.PEER_PERCENTILE)
        return pd.Series({
            "peer_p50"  : p50,
            "peer_p75"  : p75,
            "peer_p90"  : p90,
            "peer_count": len(grp),
        })

    group_stats = (
        feat.groupby(["outlet_type", "province"])
        .apply(peer_stats)
        .reset_index()
    )
    feat = feat.merge(group_stats, on=["outlet_type", "province"], how="left")

    print(f"    [GOLD] Peer groups: {feat.groupby(['outlet_type','province']).ngroups}")
    return feat


# ─────────────────────────────────────────────────────────────────
# STEP 4 — JANUARY SEASONALITY FACTOR
# ─────────────────────────────────────────────────────────────────

def add_seasonality(features: pd.DataFrame,
                    seasonality: pd.DataFrame) -> pd.DataFrame:
    """
    Join the distributor-specific January seasonality index.
    If missing, default to 1.0 (no adjustment).
    """
    jan_season = seasonality[
        seasonality["month"] == config.TARGET_MONTH_NUM
    ][["distributor_id", "index"]].rename(columns={"index": "jan_seasonality"})

    feat = features.merge(jan_season, on="distributor_id", how="left")
    feat["jan_seasonality"] = feat["jan_seasonality"].fillna(1.0)
    return feat


# ─────────────────────────────────────────────────────────────────
# GOLD ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def build_features(silver: dict) -> pd.DataFrame:
    print("\n" + "═" * 60)
    print("  GOLD LAYER — Feature Engineering")
    print("═" * 60)

    monthly_agg = compute_monthly_agg(silver["transactions"])
    feat        = compute_outlet_features(monthly_agg, silver["holidays"])
    feat        = add_peer_group_features(feat, silver["outlet_master"])
    feat        = add_seasonality(feat, silver["seasonality"])

    # Merge coordinates for POI scraping
    coords = silver["coordinates"][["outlet_id", "lat", "lon"]]
    feat   = feat.merge(coords, on="outlet_id", how="left")

    # Save intermediate gold features
    os.makedirs(config.GOLD_DIR, exist_ok=True)
    path = os.path.join(config.GOLD_DIR, "outlet_features_gold.csv")
    feat.to_csv(path, index=False)
    print(f"    [GOLD] Features saved → {path}")
    return feat
