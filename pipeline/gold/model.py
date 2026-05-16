"""
pipeline/gold/model.py
=======================
GOLD LAYER — Mathematical Potential Estimation Model.

╔══════════════════════════════════════════════════════════════════╗
║           LATENT DEMAND UNCAPPING FRAMEWORK                     ║
║                                                                  ║
║  Observed volume V_obs = min(D_true, C_supply)                  ║
║  where C_supply = min(credit_limit, stock, delivery_cap)        ║
║                                                                  ║
║  Since V_obs ≤ D_true, historical data is LEFT-CENSORED.       ║
║  We estimate D_true (potential) in three stages:                ║
║                                                                  ║
║  Stage 1 — SELF CEILING                                         ║
║    B_self = p85(outlet's own monthly volumes)                   ║
║    Represents the outlet's demonstrated peak (supply aligned)   ║
║                                                                  ║
║  Stage 2 — PEER CEILING (gravity toward segment potential)      ║
║    B_peer = p90(all p85_volumes in same outlet_type×province)   ║
║    B_blended = B_self + α × (B_peer − B_self)  , α = 0.40      ║
║    Pulls every outlet toward what similar outlets achieve,      ║
║    representing the segment's achievable ceiling.               ║
║    → If B_self > B_peer, outlet already exceeds peers: keep B_self║
║                                                                  ║
║  Stage 3 — ADJUSTMENTS                                          ║
║    × jan_seasonality_index   (distributor calendar effect)      ║
║    × trend_factor            (growth trajectory extrapolation)  ║
║    × poi_boost_factor        (catchment demand drivers)         ║
║    × holiday_factor          (January public holiday uplift)    ║
║                                                                  ║
║  Final:                                                          ║
║    Potential_i = B_blended_i                                    ║
║                  × jan_seasonality_i                            ║
║                  × trend_factor_i                               ║
║                  × poi_boost_factor_i                           ║
║                  × holiday_factor                               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import pandas as pd
import numpy as np
import math
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


# ─────────────────────────────────────────────────────────────────
# STAGE 1 — SELF CEILING
# ─────────────────────────────────────────────────────────────────

def _self_ceiling(row: pd.Series) -> float:
    """
    Use the outlet's own p85 volume as its demonstrated demand ceiling.
    If the outlet has January-specific history, blend with jan_avg
    (weight jan_avg more heavily since we're predicting January).
    """
    p85      = row["p85_volume"]
    jan_avg  = row["jan_avg"]
    n_active = row["active_months"]

    # If we have January history, give it 60% weight
    # Otherwise rely entirely on the overall p85
    if jan_avg > 0 and n_active >= 12:
        b_self = 0.4 * p85 + 0.6 * jan_avg
    else:
        b_self = p85

    return max(b_self, 0.0)


# ─────────────────────────────────────────────────────────────────
# STAGE 2 — PEER CEILING BLEND
# ─────────────────────────────────────────────────────────────────

def _peer_blend(b_self: float, peer_p75: float, peer_p90: float,
                peer_count: int, alpha: float = config.REGRESSION_TO_PEER) -> float:
    """
    Blend self-ceiling with peer ceiling.

    B_blended = B_self + α × (B_peer − B_self)

    Logic:
    • If outlet outperforms its peers (b_self > peer_p90), keep b_self.
    • If outlet is below peer median, pull strongly toward peer_p75
      (conservative — don't assume every outlet can hit p90).
    • α controls how aggressively we uncap.

    We use peer_p75 (not p90) as the target ceiling because p90 may
    reflect a single exceptional outlier in a small peer group.
    """
    if b_self >= peer_p90:
        # Outlet already outperforms peers — no uncapping needed
        return b_self

    target = peer_p75   # conservative peer benchmark

    # Reduce alpha for small peer groups (less reliable statistics)
    if peer_count < 10:
        alpha = alpha * 0.5

    blended = b_self + alpha * (target - b_self)

    # Hard cap: never inflate more than MAX_UNCAP_RATIO × self ceiling
    cap     = b_self * config.MAX_UNCAP_RATIO
    return min(blended, cap)


# ─────────────────────────────────────────────────────────────────
# STAGE 3A — TREND FACTOR
# ─────────────────────────────────────────────────────────────────

def _trend_factor(trend_slope_norm: float) -> float:
    """
    Converts the normalised OLS slope into a multiplicative factor.

    slope_norm = β₁ / mean_volume  (dimensionless growth rate per month)

    Months from last data point to January 2026:
    We assume the trend period used is ~6 months, so we project
    forward approximately 1–3 months.

    trend_factor = 1 + clip(slope_norm × 2, −0.20, +0.25)

    The clip prevents runaway extrapolation from noisy slopes.
    Minimum factor = 0.80 (at most 20% decline from trend).
    Maximum factor = 1.25 (at most 25% uplift from trend).
    """
    projection_periods = 2.0       # project ~2 months forward
    raw_uplift  = trend_slope_norm * projection_periods
    clipped     = max(-0.20, min(0.25, raw_uplift))
    return 1.0 + clipped


# ─────────────────────────────────────────────────────────────────
# STAGE 3B — HOLIDAY FACTOR
# ─────────────────────────────────────────────────────────────────

def _holiday_factor(jan_holidays: int) -> float:
    """
    Beverages typically sell more around public holidays
    (public gatherings, celebrations).
    Each holiday in January adds a small multiplicative boost.

    factor = 1 + 0.02 × jan_holidays   (2% per holiday, max 10%)
    """
    return min(1.0 + 0.02 * jan_holidays, 1.10)


# ─────────────────────────────────────────────────────────────────
# HIGH-VARIANCE SIGNAL (supply-constrained outlet detection)
# ─────────────────────────────────────────────────────────────────

def _constraint_signal(cv_volume: float) -> float:
    """
    High CV (coefficient of variation) suggests the outlet experienced
    erratic supply: some months much lower than capability (stockouts,
    credit cuts), interspersed with unconstrained months.

    We apply a small additional uncapping boost to high-CV outlets.

    boost = 1 + 0.05 × min(cv_volume, 2.0)
    → Maximum 10% additional boost for CV ≥ 2.0
    """
    return 1.0 + 0.05 * min(cv_volume, 2.0)


# ─────────────────────────────────────────────────────────────────
# SPARSE OUTLET FALLBACK (new / rarely transacting outlets)
# ─────────────────────────────────────────────────────────────────

def _sparse_outlet_potential(row: pd.Series,
                              peer_p50: float,
                              jan_seasonality: float) -> float:
    """
    For outlets with fewer than MIN_ACTIVE_MONTHS months of data,
    assign the peer group median as a conservative potential estimate.
    Apply January seasonality but no trend (insufficient data).
    """
    return peer_p50 * jan_seasonality


# ─────────────────────────────────────────────────────────────────
# MAIN ESTIMATION FUNCTION
# ─────────────────────────────────────────────────────────────────

def estimate_potential(features: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full latent demand uncapping framework to every outlet.
    Returns features DataFrame with added 'potential' column.
    """
    print("\n" + "═" * 60)
    print("  MODEL — Estimating Latent Demand Potential")
    print("═" * 60)

    potentials = []

    for _, row in features.iterrows():
        outlet_id     = row["outlet_id"]
        jan_season    = float(row.get("jan_seasonality", 1.0))
        jan_holidays  = int(row.get("jan_holidays", 0))
        poi_boost     = float(row.get("poi_boost_factor", 1.0))

        # ── Stage 1: Self Ceiling ─────────────────────────────────
        b_self = _self_ceiling(row)

        # ── Stage 2: Peer Blend ───────────────────────────────────
        peer_p75  = float(row.get("peer_p75",  b_self))
        peer_p90  = float(row.get("peer_p90",  b_self))
        peer_p50  = float(row.get("peer_p50",  b_self))
        peer_count= int(row.get("peer_count", 1))

        b_blended = _peer_blend(b_self, peer_p75, peer_p90, peer_count)

        # ── Stage 3: Adjustment Factors ───────────────────────────
        t_factor  = _trend_factor(float(row.get("trend_slope_norm", 0.0)))
        h_factor  = _holiday_factor(jan_holidays)
        cv_boost  = _constraint_signal(float(row.get("cv_volume", 0.0)))

        # ── Final Potential ───────────────────────────────────────
        potential = b_blended * jan_season * t_factor * h_factor * cv_boost * poi_boost

        # Floor: potential can never be less than the outlet's own
        # observed mean (we don't predict a decline from reality)
        floor     = float(row.get("mean_volume", 0.0)) * jan_season
        potential = max(potential, floor)

        potentials.append({
            "outlet_id"              : outlet_id,
            "b_self"                 : round(b_self,     2),
            "b_blended"              : round(b_blended,  2),
            "jan_seasonality"        : round(jan_season, 4),
            "trend_factor"           : round(t_factor,   4),
            "holiday_factor"         : round(h_factor,   4),
            "cv_boost"               : round(cv_boost,   4),
            "poi_boost_factor"       : round(poi_boost,  4),
            "Maximum_Monthly_Liters" : round(potential,  2),
        })

    result = pd.DataFrame(potentials)
    print(f"\n  [MODEL] Potential estimated for {len(result):,} outlets")
    print(f"  [MODEL] Summary:")
    print(f"           Min   = {result['Maximum_Monthly_Liters'].min():>10,.1f} L")
    print(f"           Median= {result['Maximum_Monthly_Liters'].median():>10,.1f} L")
    print(f"           Mean  = {result['Maximum_Monthly_Liters'].mean():>10,.1f} L")
    print(f"           Max   = {result['Maximum_Monthly_Liters'].max():>10,.1f} L")
    return result


# ─────────────────────────────────────────────────────────────────
# HANDLE OUTLETS NOT IN TRANSACTIONS (zero-activity outlets)
# ─────────────────────────────────────────────────────────────────

def fill_missing_outlets(predictions: pd.DataFrame,
                          outlet_master: pd.DataFrame,
                          global_median: float,
                          jan_seasonality_map: dict) -> pd.DataFrame:
    """
    Some outlets in outlet_master may have zero transactions in history
    (new, inactive, or missing data).  We assign them a conservative
    potential = 50% of global median × their distributor's Jan index.
    This avoids dropping them from the submission file.
    """
    predicted_ids = set(predictions["outlet_id"].astype(str).tolist())
    all_ids       = set(outlet_master["outlet_id"].astype(str).tolist())
    missing_ids   = all_ids - predicted_ids

    if not missing_ids:
        return predictions

    missing_rows = outlet_master[
        outlet_master["outlet_id"].astype(str).isin(missing_ids)
    ][["outlet_id", "distributor_id"]].copy()

    missing_rows["jan_seasonality"] = (
        missing_rows["distributor_id"]
        .map(jan_seasonality_map)
        .fillna(1.0)
    )
    missing_rows["Maximum_Monthly_Liters"] = (
        global_median * 0.5 * missing_rows["jan_seasonality"]
    ).round(2)

    missing_rows = missing_rows[["outlet_id", "Maximum_Monthly_Liters"]]
    print(f"  [MODEL] Assigned fallback potential to {len(missing_rows):,} "
          f"zero-history outlets")

    return pd.concat([predictions, missing_rows], ignore_index=True)


# ─────────────────────────────────────────────────────────────────
# SAVE FINAL PREDICTIONS
# ─────────────────────────────────────────────────────────────────

def save_predictions(predictions: pd.DataFrame,
                     detailed: pd.DataFrame = None) -> str:
    """
    Save the final submission CSV and an optional detailed breakdown.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ── Submission file ───────────────────────────────────────────
    submission = predictions[["outlet_id", "Maximum_Monthly_Liters"]].rename(
        columns={"outlet_id": "Outlet_ID"}
    )
    sub_path = os.path.join(config.OUTPUT_DIR, f"{config.TEAM_NAME}_predictions.csv")
    submission.to_csv(sub_path, index=False)
    print(f"\n  [OUTPUT] Submission saved → {sub_path}  ({len(submission):,} rows)")

    # ── Detailed breakdown (for internal review / report) ─────────
    if detailed is not None:
        det_path = os.path.join(config.OUTPUT_DIR, "predictions_detailed.csv")
        detailed.to_csv(det_path, index=False)
        print(f"  [OUTPUT] Detailed breakdown → {det_path}")

    return sub_path
