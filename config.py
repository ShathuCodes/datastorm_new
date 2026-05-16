"""
config.py
=========
Central configuration for the entire pipeline.
Adjust column names here if your CSVs use different headers.
"""

import os

# ─────────────────────────────────────────────
# DIRECTORY STRUCTURE
# ─────────────────────────────────────────────
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR        = os.path.join(BASE_DIR, "data", "raw")
BRONZE_DIR          = os.path.join(BASE_DIR, "data", "bronze")
SILVER_DIR          = os.path.join(BASE_DIR, "data", "silver")
SILVER_REJECTED_DIR = os.path.join(BASE_DIR, "data", "silver", "rejected")
GOLD_DIR            = os.path.join(BASE_DIR, "data", "gold")
OUTPUT_DIR          = os.path.join(BASE_DIR, "output")

# ─────────────────────────────────────────────
# RAW FILE NAMES  (place these in data/raw/)
# ─────────────────────────────────────────────
RAW_FILES = {
    "transactions" : "transactions_history_final.csv",
    "outlet_master": "outlet_master.csv",
    "coordinates"  : "outlet_coordinates.csv",
    "seasonality"  : "distributor_seasonality_details.csv",
    "holidays"     : "holiday_list.csv",
}

# ─────────────────────────────────────────────
# COLUMN NAME MAPPING
# Adjust these if your CSV headers differ.
# ─────────────────────────────────────────────
COLS = {
    "transactions": {
        "outlet_id"     : "Outlet_ID",
        "date"          : "Invoice_Date",       # parse as datetime
        "volume"        : "Volume_Liters",      # numeric, liters
        "distributor_id": "Distributor_ID",
        "invoice_no"    : "Invoice_No",         # optional
        "sku"           : "SKU_Code",           # optional
    },
    "outlet_master": {
        "outlet_id"     : "Outlet_ID",
        "outlet_name"   : "Outlet_Name",
        "outlet_type"   : "Outlet_Type",        # kade / grocery / eatery / pharmacy
        "distributor_id": "Distributor_ID",
        "province"      : "Province",
        "channel"       : "Channel",            # optional
        "credit_limit"  : "Credit_Limit_LKR",   # optional
    },
    "coordinates": {
        "outlet_id": "Outlet_ID",
        "lat"      : "Latitude",
        "lon"      : "Longitude",
    },
    "seasonality": {
        "distributor_id": "Distributor_ID",
        "month"         : "Month",              # integer 1–12  OR  month name
        "index"         : "Seasonality_Index",  # float, e.g. 1.12 means +12%
    },
    "holidays": {
        "date": "Holiday_Date",
        "name": "Holiday_Name",
    },
}

# ─────────────────────────────────────────────
# VALID REFERENCE VALUES
# ─────────────────────────────────────────────
VALID_DISTRIBUTOR_IDS = [
    "DIST_W_01", "DIST_W_02", "DIST_W_03",
    "DIST_C_01", "DIST_C_02", "DIST_C_03",
    "DIST_NW_01", "DIST_NW_02",
    "DIST_S_01",  "DIST_S_02",
]

VALID_PROVINCES = ["Western", "Central", "North-Western", "Southern"]

# ─────────────────────────────────────────────
# DATA QUALITY THRESHOLDS
# ─────────────────────────────────────────────
VOLUME_MIN   = 0.0       # liters – negative is impossible
VOLUME_MAX   = 50000.0   # liters – flag extreme values for review
LAT_MIN, LAT_MAX = 5.9,   9.9    # Sri Lanka bounding box
LON_MIN, LON_MAX = 79.5, 82.0

# ─────────────────────────────────────────────
# MODEL PARAMETERS
# ─────────────────────────────────────────────
TARGET_MONTH_NUM    = 1          # January
TARGET_YEAR         = 2026
PEER_PERCENTILE     = 90         # percentile used for peer ceiling (p90)
SELF_PERCENTILE     = 85         # percentile used for outlet's own ceiling
MAX_UNCAP_RATIO     = 3.5        # never inflate an outlet more than 3.5x its own peak
REGRESSION_TO_PEER  = 0.40       # how strongly to pull toward peer ceiling (0–1)
TREND_MONTHS        = 6          # use last N months for trend calculation
MIN_ACTIVE_MONTHS   = 2          # outlet must have at least this many months of data

# ─────────────────────────────────────────────
# POI SCRAPING
# ─────────────────────────────────────────────
POI_RADIUS_METERS = 500          # catchment radius around each outlet
POI_BATCH_SIZE    = 50           # outlets per Overpass API call
POI_SLEEP_SECONDS = 1.5          # polite delay between API calls

# Weights: how much each POI category contributes to demand potential
POI_WEIGHTS = {
    "bus_stop"       : 4,
    "bus_station"    : 5,
    "school"         : 3,
    "university"     : 3,
    "hospital"       : 2,
    "clinic"         : 2,
    "marketplace"    : 5,
    "supermarket"    : 2,
    "restaurant"     : 1,
    "place_of_worship": 2,
    "hotel"          : 3,
    "tourism"        : 3,
    "office"         : 2,
    "factory"        : 2,
}

# Maximum POI boost allowed (30 % above base)
POI_MAX_BOOST = 0.30

# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────
TEAM_NAME = "your_team_name"   # <-- change this before submitting
