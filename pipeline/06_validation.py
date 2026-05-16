"""
OUT-OF-TIME VALIDATION - Chronological Holdout Framework
=========================================================
Implements a strict out-of-time validation protocol to eliminate
chronological data leakage from the potential estimation pipeline.

WHY THIS MATTERS
----------------
Random K-Fold cross-validation on time-indexed data leaks future state
attributes (e.g., Dec 2025 volumes) into past-period training windows,
producing artificially inflated holdout scores that collapse on the true
January 2026 target window.

APPROACH: TimeSeriesSplit + Out-of-Time Anchor Holdout
-------------------------------------------------------
1. Aggregate transactions to monthly outlet level.
2. Use sklearn TimeSeriesSplit (n_splits=6) to create sequential folds:
     Fold 1: train [Jan23..Apr23] -> validate [May23]
     Fold 2: train [Jan23..Jul23] -> validate [Aug23]
     ...
     Fold 6: train [Jan23..Sep25] -> validate [Oct25]
3. For each fold, build the same feature set used in the gold model
   (historical stats, censoring score, size/type factors) on the training
   window and predict volume for the validation month.
4. Final holdout: train up to Oct 2025, validate on Nov-Dec 2025 as
   a proxy for January 2026 generalization.
5. Report MAE, RMSE, MedAE and MAPE per fold and overall.

Metrics are saved to output/validation_report.csv.

Usage:
    python pipeline/06_validation.py
"""

import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent.parent
SILVER   = ROOT / "pipeline" / "silver"
GOLD_DIR = ROOT / "pipeline" / "gold"
OUTPUT   = ROOT / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SPLITS          = 6      # TimeSeriesSplit folds
MIN_TRAIN_MONTHS  = 6      # minimum months required in a training window
HOLDOUT_MONTHS    = [      # final out-of-time holdout (proxy for Jan 2026)
    (2025, 11), (2025, 12)
]

SIZE_POTENTIAL_FACTOR = {
    "Extra Large": 1.30, "Large": 1.15, "Medium": 1.00, "Small": 0.88,
}
TYPE_POTENTIAL_FACTOR = {
    "Grocery": 1.10, "Hotel": 1.20, "Pharmacy": 0.90,
    "Kiosk": 0.85, "Eatery": 1.05, "Bakery": 0.95, "SMMT": 1.25,
}
SEASONALITY_MULTIPLIER = {
    "Favorable": 1.15, "Moderate": 1.00, "Un-Favorable": 0.88,
}


# ---------------------------------------------------------------------------
# Feature builder (mirrors gold model logic on a given training window)
# ---------------------------------------------------------------------------

def build_features_from_window(monthly: pd.DataFrame,
                                outlet: pd.DataFrame,
                                season: pd.DataFrame,
                                train_mask: pd.Series) -> pd.DataFrame:
    """
    Build outlet-level features using only the rows in `train_mask`.

    Parameters
    ----------
    monthly    : Full monthly aggregated transactions DataFrame.
    outlet     : Outlet master (type, size).
    season     : Distributor seasonality.
    train_mask : Boolean mask selecting the training window rows.

    Returns
    -------
    Feature DataFrame indexed by Outlet_ID.
    """
    train = monthly[train_mask].copy()
    if train.empty:
        return pd.DataFrame()

    # Distributor-month median in training window
    dist_med = (
        train.groupby(["Distributor_ID", "Month"])["monthly_volume"]
        .median().rename("dist_month_median").reset_index()
    )
    train = train.merge(dist_med, on=["Distributor_ID", "Month"], how="left")

    def _features(grp):
        vols = grp["monthly_volume"].values
        if len(vols) < 1:
            return None
        mean_vol   = float(np.mean(vols))
        median_vol = float(np.median(vols))
        max_vol    = float(np.max(vols))
        std_vol    = float(np.std(vols))
        cv         = std_vol / (mean_vol + 1e-9)

        # Censoring score (simplified — 3 signals only for speed)
        plateau = 0.0
        if len(vols) >= 3:
            cvs = [np.std(vols[i:i+3]) / (np.mean(vols[i:i+3]) + 1e-9)
                   for i in range(len(vols) - 2)]
            plateau = float(np.mean(np.array(cvs) < 0.10))

        dist_med_vals = grp["dist_month_median"].values
        at_cap = float(np.mean(
            np.abs(vols - dist_med_vals) / (dist_med_vals + 1e-9) < 0.15
        ))
        cv_score      = float(max(0.0, (0.30 - cv) / 0.30))
        cens_score    = float(np.clip(0.40 * plateau + 0.35 * at_cap + 0.25 * cv_score, 0, 1))

        base_vol = float(np.percentile(vols, 90) if cens_score > 0.30
                         else np.percentile(vols, 75))
        uplift   = 1.0 + cens_score * 0.60

        return pd.Series({
            "hist_median_vol":  median_vol,
            "hist_max_vol":     max_vol,
            "hist_cv":          cv,
            "censoring_score":  cens_score,
            "base_vol":         base_vol,
            "cens_uplift":      uplift,
            "primary_dist":     grp.groupby("Distributor_ID")["monthly_volume"]
                                    .sum().idxmax(),
        })

    feats = train.groupby("Outlet_ID").apply(_features).dropna().reset_index()

    # Merge outlet type/size factors
    feats = feats.merge(outlet[["Outlet_ID", "Outlet_Type", "Outlet_Size"]],
                        on="Outlet_ID", how="left")
    feats["size_factor"] = feats["Outlet_Size"].map(SIZE_POTENTIAL_FACTOR).fillna(1.0)
    feats["type_factor"] = feats["Outlet_Type"].map(TYPE_POTENTIAL_FACTOR).fillna(1.0)

    # Seasonality (use current month's distributor seasonality)
    season_map = season.groupby(["Distributor_ID", "Month"])["Seasonality_Index"].first()
    feats["prediction"] = (
        feats["base_vol"]
        * feats["cens_uplift"]
        * feats["size_factor"]
        * feats["type_factor"]
    ).clip(lower=1.0)

    return feats[["Outlet_ID", "prediction", "censoring_score"]]


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute MAE, RMSE, MedAE, and MAPE."""
    errors   = predicted - actual
    abs_err  = np.abs(errors)
    mae      = float(np.mean(abs_err))
    rmse     = float(np.sqrt(np.mean(errors ** 2)))
    med_ae   = float(np.median(abs_err))
    mape     = float(np.mean(abs_err / (np.abs(actual) + 1e-9)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MedAE": med_ae, "MAPE_%": mape}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("OUT-OF-TIME VALIDATION - Chronological Holdout Framework")
    print("=" * 60)
    print(f"  TimeSeriesSplit folds: {N_SPLITS}")
    print(f"  Final holdout months:  {HOLDOUT_MONTHS}")

    # Load data
    print("\n[1/4] Loading Silver data...")
    tx     = pd.read_parquet(SILVER / "transactions.parquet")
    outlet = pd.read_parquet(SILVER / "outlet_master.parquet")
    season = pd.read_parquet(SILVER / "distributor_seasonality.parquet")

    # Monthly aggregation
    print("[2/4] Aggregating to monthly outlet level...")
    monthly = (
        tx.groupby(["Outlet_ID", "Year", "Month", "Distributor_ID"])
        .agg(monthly_volume=("Volume_Liters", "sum"))
        .reset_index()
    )

    # Create a sortable period index
    monthly["period_idx"] = monthly["Year"] * 12 + monthly["Month"]
    all_periods           = sorted(monthly["period_idx"].unique())
    n_periods             = len(all_periods)
    print(f"  Total periods: {n_periods}  "
          f"(from {min(all_periods)//12}-{min(all_periods)%12:02d} "
          f"to {max(all_periods)//12}-{max(all_periods)%12:02d})")

    # Map period index to position for TimeSeriesSplit
    period_to_pos = {p: i for i, p in enumerate(all_periods)}
    monthly["period_pos"] = monthly["period_idx"].map(period_to_pos)

    # -----------------------------------------------------------------------
    # TimeSeriesSplit rolling-window validation
    # -----------------------------------------------------------------------
    print(f"\n[3/4] Running {N_SPLITS}-fold out-of-time validation...")
    tss      = TimeSeriesSplit(n_splits=N_SPLITS, gap=0)
    positions = monthly["period_pos"].values

    # We operate on the period-level (not row-level) for the split
    period_positions = np.arange(n_periods)
    fold_results     = []

    for fold_idx, (train_pos_idx, val_pos_idx) in enumerate(tss.split(period_positions)):
        if len(train_pos_idx) < MIN_TRAIN_MONTHS:
            continue

        train_periods = set(all_periods[i] for i in train_pos_idx)
        val_periods   = set(all_periods[i] for i in val_pos_idx)

        train_mask = monthly["period_idx"].isin(train_periods)
        val_mask   = monthly["period_idx"].isin(val_periods)

        # Describe this fold
        t_min  = min(train_periods)
        t_max  = max(train_periods)
        v_min  = min(val_periods)
        v_max  = max(val_periods)
        label  = (f"  Fold {fold_idx+1}: train "
                  f"{t_min//12}-{t_min%12:02d} -> {t_max//12}-{t_max%12:02d} | "
                  f"validate {v_min//12}-{v_min%12:02d} -> {v_max//12}-{v_max%12:02d}")
        print(label)

        # Build features from training window
        feats = build_features_from_window(monthly, outlet, season, train_mask)
        if feats.empty:
            print("    [SKIP] Insufficient training data.")
            continue

        # Actual validation volumes
        actuals = (
            monthly[val_mask]
            .groupby("Outlet_ID")["monthly_volume"]
            .mean()
            .reset_index()
            .rename(columns={"monthly_volume": "actual"})
        )

        merged = feats.merge(actuals, on="Outlet_ID", how="inner")
        if merged.empty:
            continue

        metrics = compute_metrics(merged["actual"].values, merged["prediction"].values)
        metrics.update({
            "fold":           fold_idx + 1,
            "train_periods":  len(train_periods),
            "val_periods":    len(val_periods),
            "outlets_scored": len(merged),
            "train_end":      f"{t_max//12}-{t_max%12:02d}",
            "val_window":     f"{v_min//12}-{v_min%12:02d} -> {v_max//12}-{v_max%12:02d}",
        })
        fold_results.append(metrics)
        print(f"    MAE={metrics['MAE']:,.1f} L  |  RMSE={metrics['RMSE']:,.1f} L  |  "
              f"MedAE={metrics['MedAE']:,.1f} L  |  MAPE={metrics['MAPE_%']:.1f}%")

    # -----------------------------------------------------------------------
    # Final out-of-time holdout (Nov-Dec 2025 as Jan 2026 proxy)
    # -----------------------------------------------------------------------
    print(f"\n[4/4] Final holdout: train up to Oct 2025, validate Nov-Dec 2025...")
    holdout_period_idxs = set(y * 12 + m for y, m in HOLDOUT_MONTHS)
    cutoff_period       = min(holdout_period_idxs) - 1  # Oct 2025

    final_train_mask = monthly["period_idx"] <= cutoff_period
    final_val_mask   = monthly["period_idx"].isin(holdout_period_idxs)

    feats_final  = build_features_from_window(monthly, outlet, season, final_train_mask)
    actuals_final = (
        monthly[final_val_mask]
        .groupby("Outlet_ID")["monthly_volume"]
        .mean()
        .reset_index()
        .rename(columns={"monthly_volume": "actual"})
    )
    merged_final = feats_final.merge(actuals_final, on="Outlet_ID", how="inner")

    final_metrics = {}
    if not merged_final.empty:
        final_metrics = compute_metrics(
            merged_final["actual"].values,
            merged_final["prediction"].values
        )
        final_metrics.update({
            "fold":           "FINAL_HOLDOUT",
            "train_periods":  int(final_train_mask.sum()),
            "val_periods":    int(final_val_mask.sum()),
            "outlets_scored": len(merged_final),
            "train_end":      "2025-10",
            "val_window":     "2025-11 -> 2025-12",
        })
        fold_results.append(final_metrics)
        print(f"  FINAL HOLDOUT:  MAE={final_metrics['MAE']:,.1f} L  |  "
              f"RMSE={final_metrics['RMSE']:,.1f} L  |  "
              f"MedAE={final_metrics['MedAE']:,.1f} L  |  "
              f"MAPE={final_metrics['MAPE_%']:.1f}%")

    # -----------------------------------------------------------------------
    # Save and summarize
    # -----------------------------------------------------------------------
    results_df = pd.DataFrame(fold_results)
    out_path   = OUTPUT / "validation_report.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\n[SUMMARY] Cross-fold averages (excluding final holdout):")
    cv_rows = results_df[results_df["fold"] != "FINAL_HOLDOUT"]
    if not cv_rows.empty:
        print(f"  Avg MAE:   {cv_rows['MAE'].mean():,.1f} L")
        print(f"  Avg RMSE:  {cv_rows['RMSE'].mean():,.1f} L")
        print(f"  Avg MedAE: {cv_rows['MedAE'].mean():,.1f} L")
        print(f"  Avg MAPE:  {cv_rows['MAPE_%'].mean():.1f}%")

    print(f"\n  Validation report saved -> {out_path}")
    print("\n[OK]  Out-of-time validation complete.\n")


if __name__ == "__main__":
    main()
