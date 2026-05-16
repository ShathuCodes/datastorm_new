"""
pipeline/gold/poi_scraper.py
=============================
GOLD LAYER — External POI enrichment via OpenStreetMap Overpass API.
No external libraries beyond 'requests'.

Strategy
--------
For each outlet coordinate we query a radius of POI_RADIUS_METERS for
amenity tags that indicate footfall-generating places (bus stops, schools,
hospitals, markets, mosques, etc.).  A weighted POI score is calculated
and normalised to a boost factor [1.0, 1 + POI_MAX_BOOST].

To stay within Overpass rate limits we:
  1. Batch outlets into groups
  2. Use a UNION query (one HTTP call per batch)
  3. Sleep between batches
"""

import os
import time
import json
import math
import requests
import pandas as pd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# ─────────────────────────────────────────────────────────────────
# OVERPASS QUERY BUILDER
# ─────────────────────────────────────────────────────────────────

# Maps Overpass tags → our POI category → weight
POI_TAG_MAP = [
    # (osm_key, osm_value, category)
    ("amenity",  "bus_station",   "bus_station"),
    ("highway",  "bus_stop",      "bus_stop"),
    ("amenity",  "school",        "school"),
    ("amenity",  "university",    "university"),
    ("amenity",  "college",       "school"),
    ("amenity",  "hospital",      "hospital"),
    ("amenity",  "clinic",        "clinic"),
    ("amenity",  "marketplace",   "marketplace"),
    ("shop",     "supermarket",   "supermarket"),
    ("amenity",  "restaurant",    "restaurant"),
    ("amenity",  "place_of_worship", "place_of_worship"),
    ("tourism",  "hotel",         "hotel"),
    ("tourism",  "attraction",    "tourism"),
    ("office",   "government",    "office"),
    ("landuse",  "industrial",    "factory"),
]


def _build_overpass_query(lat: float, lon: float, radius: int) -> str:
    """
    Build an Overpass QL query for one location.
    Returns all nodes/ways within radius metres that match any POI tag.
    """
    radius_str = str(radius)
    ll         = f"{lat},{lon}"
    tag_blocks = "\n".join(
        f'  node["{k}"="{v}"](around:{radius_str},{ll});'
        for k, v, _ in POI_TAG_MAP
    )
    query = f"""
[out:json][timeout:25];
(
{tag_blocks}
);
out count;
"""
    return query


def _build_batch_query(outlets_subset: pd.DataFrame, radius: int) -> str:
    """
    Build a single Overpass query that counts POIs for multiple outlets.
    Each outlet gets its own named 'area' result so we can parse per-outlet.
    Unfortunately Overpass doesn't easily support per-outlet counts in one
    call, so we loop per-outlet but cache results.
    """
    # (see scrape_poi_scores below — we query one outlet at a time
    #  but sleep between them)
    pass


# ─────────────────────────────────────────────────────────────────
# POI SCORE COMPUTATION
# ─────────────────────────────────────────────────────────────────

def _compute_raw_poi_score(count_by_category: dict) -> float:
    """
    Weighted sum of POI counts.
    score = Σ weight_c × count_c
    """
    score = 0.0
    for category, weight in config.POI_WEIGHTS.items():
        score += weight * count_by_category.get(category, 0)
    return score


def _normalise_boost(score: float, max_score: float) -> float:
    """
    Normalise raw score to a boost factor ∈ [1.0, 1 + POI_MAX_BOOST].
    boost = 1 + POI_MAX_BOOST × (score / max_score)
    If max_score == 0, return 1.0 (no boost).
    """
    if max_score == 0:
        return 1.0
    return 1.0 + config.POI_MAX_BOOST * min(score / max_score, 1.0)


# ─────────────────────────────────────────────────────────────────
# PER-OUTLET QUERY
# ─────────────────────────────────────────────────────────────────

def _query_single_outlet(lat: float, lon: float, outlet_id: str) -> dict:
    """
    Hit Overpass API for one outlet and return category counts.
    Returns empty dict on any failure (POI score defaults to 0).
    """
    counts = {cat: 0 for _, _, cat in POI_TAG_MAP}
    radius = config.POI_RADIUS_METERS

    for osm_key, osm_val, category in POI_TAG_MAP:
        query = f"""
[out:json][timeout:15];
(
  node["{osm_key}"="{osm_val}"](around:{radius},{lat},{lon});
  way["{osm_key}"="{osm_val}"](around:{radius},{lat},{lon});
);
out count;
"""
        try:
            r = requests.post(OVERPASS_URL, data={"data": query}, timeout=20)
            if r.status_code == 200:
                data   = r.json()
                total  = data.get("elements", [{}])[0].get("tags", {}).get("total", 0)
                counts[category] += int(total)
            time.sleep(0.3)  # micro-delay between tag queries
        except Exception:
            pass  # network error — leave count at 0

    return counts


# ─────────────────────────────────────────────────────────────────
# HAVERSINE DISTANCE (utility, used for spatial checks)
# ─────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R   = 6371.0
    φ1  = math.radians(lat1)
    φ2  = math.radians(lat2)
    Δφ  = math.radians(lat2 - lat1)
    Δλ  = math.radians(lon2 - lon1)
    a   = math.sin(Δφ/2)**2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─────────────────────────────────────────────────────────────────
# MAIN SCRAPING FUNCTION
# ─────────────────────────────────────────────────────────────────

def scrape_poi_scores(features: pd.DataFrame) -> pd.DataFrame:
    """
    For every outlet with valid lat/lon, query Overpass and compute
    a poi_boost_factor.

    To respect rate limits and time constraints, we:
      - Skip outlets with invalid / missing coordinates
      - Cache results to gold/poi_cache.csv so re-runs don't re-query
      - Sleep POI_SLEEP_SECONDS between outlet queries
    """
    cache_path = os.path.join(config.GOLD_DIR, "poi_cache.csv")

    # Load cache if exists
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path)
        print(f"  [POI] Loaded cache: {len(cache):,} outlets already scraped")
    else:
        cache = pd.DataFrame(columns=["outlet_id", "poi_raw_score"])

    cached_ids = set(cache["outlet_id"].astype(str).tolist())

    # Outlets that have valid coordinates and are not cached
    has_coords  = features["lat"].notna() & features["lon"].notna()
    to_scrape   = features[has_coords & ~features["outlet_id"].astype(str).isin(cached_ids)].copy()

    print(f"  [POI] Outlets to scrape: {len(to_scrape):,} "
          f"(cached: {len(cached_ids):,})")

    new_rows = []
    for i, (_, row) in enumerate(to_scrape.iterrows(), 1):
        outlet_id = str(row["outlet_id"])
        lat       = float(row["lat"])
        lon       = float(row["lon"])

        counts    = _query_single_outlet(lat, lon, outlet_id)
        raw_score = _compute_raw_poi_score(counts)
        new_rows.append({"outlet_id": outlet_id, "poi_raw_score": raw_score})

        if i % 10 == 0:
            print(f"    [POI] {i}/{len(to_scrape)} scraped...")

        time.sleep(config.POI_SLEEP_SECONDS)

        # Save cache incrementally (safe against crashes)
        if i % 50 == 0:
            _save_cache(cache, new_rows, cache_path)

    _save_cache(cache, new_rows, cache_path)
    full_cache = pd.read_csv(cache_path)

    # ── Normalise raw scores to boost factors ─────────────────────
    max_score = float(full_cache["poi_raw_score"].max()) if len(full_cache) > 0 else 1.0
    full_cache["poi_boost_factor"] = full_cache["poi_raw_score"].apply(
        lambda s: _normalise_boost(s, max_score)
    )

    # ── Merge into features ───────────────────────────────────────
    full_cache["outlet_id"] = full_cache["outlet_id"].astype(str)
    features["outlet_id"]   = features["outlet_id"].astype(str)
    enriched = features.merge(
        full_cache[["outlet_id", "poi_raw_score", "poi_boost_factor"]],
        on="outlet_id", how="left"
    )
    # Default: no POI data → no boost
    enriched["poi_raw_score"]    = enriched["poi_raw_score"].fillna(0.0)
    enriched["poi_boost_factor"] = enriched["poi_boost_factor"].fillna(1.0)

    print(f"  [POI] Enrichment done. Mean boost = "
          f"{enriched['poi_boost_factor'].mean():.3f}")
    return enriched


def _save_cache(existing: pd.DataFrame, new_rows: list, path: str):
    if new_rows:
        combined = pd.concat(
            [existing, pd.DataFrame(new_rows)], ignore_index=True
        ).drop_duplicates("outlet_id")
        combined.to_csv(path, index=False)
