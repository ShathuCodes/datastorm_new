"""
main.py
=======
Pipeline Orchestrator — Data Storm v7.0
Runs: Bronze → Silver → Gold → Predictions

Usage
-----
    python main.py                    # full pipeline including POI scraping
    python main.py --skip-poi         # skip POI (uses cached or default boost=1)
    python main.py --skip-poi --eda   # also print EDA summary

Edit config.py to adjust column names, thresholds, and team name.
"""

import os
import sys
import argparse
import pandas as pd

# ── Make sure project root is on path ───────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from pipeline.bronze.ingest        import ingest_all
from pipeline.silver.clean         import clean_all
from pipeline.gold.feature_engineering import build_features
from pipeline.gold.poi_scraper     import scrape_poi_scores
from pipeline.gold.model           import (
    estimate_potential,
    fill_missing_outlets,
    save_predictions,
)


# ─────────────────────────────────────────────────────────────────
# OPTIONAL EDA SUMMARY
# ─────────────────────────────────────────────────────────────────

def print_eda(silver: dict):
    print("\n" + "═" * 60)
    print("  EDA — Quick Summary")
    print("═" * 60)
    txn = silver["transactions"]
    om  = silver["outlet_master"]
    print(f"  Transactions : {len(txn):,} rows")
    print(f"  Date range   : {txn['date'].min().date()}  →  {txn['date'].max().date()}")
    print(f"  Outlets (txn): {txn['outlet_id'].nunique():,}")
    print(f"  Outlets (master): {om['outlet_id'].nunique():,}")
    print(f"  Volume stats (liters):")
    print(f"    min    = {txn['volume'].min():>10,.1f}")
    print(f"    median = {txn['volume'].median():>10,.1f}")
    print(f"    mean   = {txn['volume'].mean():>10,.1f}")
    print(f"    max    = {txn['volume'].max():>10,.1f}")
    print(f"  Outlet types : {om['outlet_type'].value_counts().to_dict()}")
    print(f"  Province dist: {om['province'].value_counts().to_dict()}")
    print()


# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────

def run_pipeline(skip_poi: bool = False, eda: bool = False):

    print("\n" + "█" * 60)
    print("  DATA STORM v7.0 — LATENT POTENTIAL PIPELINE")
    print("  Team:", config.TEAM_NAME)
    print("  Target: Maximum Monthly Liters for January 2026")
    print("█" * 60)

    # ── BRONZE: Raw Ingestion ─────────────────────────────────────
    bronze = ingest_all()

    # ── SILVER: Clean & Quality Checks ───────────────────────────
    silver = clean_all(bronze)

    if eda:
        print_eda(silver)

    # ── GOLD: Feature Engineering ─────────────────────────────────
    features = build_features(silver)

    # ── GOLD: POI Enrichment ──────────────────────────────────────
    if skip_poi:
        print("\n  [POI] Skipping POI scraping (--skip-poi flag set).")
        print("  [POI] Setting poi_boost_factor = 1.0 for all outlets.")
        features["poi_boost_factor"] = 1.0
        features["poi_raw_score"]    = 0.0
    else:
        print("\n" + "═" * 60)
        print("  GOLD LAYER — POI Enrichment")
        print("═" * 60)
        features = scrape_poi_scores(features)

    # Save enriched gold table
    gold_path = os.path.join(config.GOLD_DIR, "outlet_features_gold_enriched.csv")
    features.to_csv(gold_path, index=False)
    print(f"  [GOLD] Enriched features saved → {gold_path}")

    # ── MODEL: Estimate Latent Potential ──────────────────────────
    predictions_detailed = estimate_potential(features)

    # ── Handle outlets with no transaction history ─────────────────
    jan_season_map = {}
    if not silver["seasonality"].empty:
        jan_s = silver["seasonality"][
            silver["seasonality"]["month"] == config.TARGET_MONTH_NUM
        ]
        jan_season_map = dict(
            zip(jan_s["distributor_id"], jan_s["index"].astype(float))
        )

    global_median = float(predictions_detailed["Maximum_Monthly_Liters"].median())

    predictions_full = fill_missing_outlets(
        predictions_detailed,
        silver["outlet_master"],
        global_median,
        jan_season_map,
    )

    # ── Save Outputs ──────────────────────────────────────────────
    sub_path = save_predictions(predictions_full, predictions_detailed)

    print("\n" + "█" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Submission file : {sub_path}")
    print("█" * 60 + "\n")
    return predictions_full


# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Storm v7.0 Pipeline")
    parser.add_argument("--skip-poi", action="store_true",
                        help="Skip POI scraping (faster, uses default boost)")
    parser.add_argument("--eda", action="store_true",
                        help="Print EDA summary after silver layer")
    args = parser.parse_args()

    run_pipeline(skip_poi=args.skip_poi, eda=args.eda)
