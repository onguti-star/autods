"""
geo.py — name-based geographic matching for choropleth maps.

The existing choropleth feature only works if you upload your own .geojson
file. Most people instead have a plain CSV/Excel file with a column of
country or US state *names* (e.g. "Egypt", "Kenya", "Texas") and a numeric
column they want to color a map by — with no boundary shapes at all.

This module bridges that gap: it ships small bundled boundary files for
world countries and US states (backend/geo_data/), and matches a column of
free-text names against them so a choropleth can be built without any
GeoJSON upload.

Adding a new level (e.g. US counties) later just means dropping a new
<name>.geojson file (with a "name" property per feature) into geo_data/,
registering it in _DATASETS below, and optionally adding an alias table.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

_DIR = os.path.join(os.path.dirname(__file__), "geo_data")

# level -> path to a GeoJSON FeatureCollection whose features each have a
# "name" property. Add new levels here (e.g. "us_counties") as new files
# are added to geo_data/.
_DATASETS = {
    "world": os.path.join(_DIR, "world_countries.geojson"),
    "us_states": os.path.join(_DIR, "us_states.geojson"),
}

# Human-readable labels for the UI.
LEVEL_LABELS = {
    "world": "country",
    "us_states": "US state",
}

_geojson_cache: dict[str, dict] = {}
_lookup_cache: dict[str, dict] = {}


def load_geojson(level: str) -> dict:
    """Load (and cache) the bundled GeoJSON FeatureCollection for a level."""
    if level not in _DATASETS:
        raise ValueError(f"Unknown geo level '{level}'. Options: {list(_DATASETS)}")
    if level not in _geojson_cache:
        with open(_DATASETS[level], encoding="utf-8") as f:
            _geojson_cache[level] = json.load(f)
    return _geojson_cache[level]


def _norm(value) -> str:
    """Normalize a name for matching: lowercase, strip punctuation/spacing."""
    s = str(value).strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Common alternate names/abbreviations that don't literally match the
# bundled boundary file's "name" property -> the canonical name that does.
_COUNTRY_ALIASES = {
    "usa": "United States of America", "us": "United States of America",
    "united states": "United States of America", "america": "United States of America",
    "uk": "United Kingdom", "great britain": "United Kingdom", "britain": "United Kingdom",
    "korea south": "South Korea", "republic of korea": "South Korea",
    "korea north": "North Korea", "dprk": "North Korea",
    "ivory coast": "Ivory Coast", "cote d ivoire": "Ivory Coast", "cote divoire": "Ivory Coast",
    "drc": "Democratic Republic of the Congo", "dr congo": "Democratic Republic of the Congo",
    "congo kinshasa": "Democratic Republic of the Congo", "congo dr": "Democratic Republic of the Congo",
    "congo brazzaville": "Republic of the Congo", "congo republic": "Republic of the Congo",
    "czechia": "Czech Republic", "north macedonia": "Macedonia", "macedonia fyrom": "Macedonia",
    "eswatini": "Swaziland", "burma": "Myanmar", "myanmar burma": "Myanmar",
    "tanzania": "United Republic of Tanzania",
    "uae": "United Arab Emirates", "emirates": "United Arab Emirates",
    "syria arab republic": "Syria", "brunei darussalam": "Brunei",
    "timor leste": "East Timor", "bosnia": "Bosnia and Herzegovina",
    "bahamas": "The Bahamas", "cape verde": "Cape Verde", "cabo verde": "Cape Verde",
    "russian federation": "Russia", "viet nam": "Vietnam", "laos pdr": "Laos",
    "moldova republic of": "Moldova", "iran islamic republic of": "Iran",
    "venezuela bolivarian republic of": "Venezuela", "bolivia plurinational state of": "Bolivia",
    "guinea bissau": "Guinea Bissau",
    "west bank": "Palestine", "palestinian territories": "Palestine", "gaza": "Palestine",
}

_US_STATE_ABBR = {
    "al": "Alabama", "ak": "Alaska", "az": "Arizona", "ar": "Arkansas", "ca": "California",
    "co": "Colorado", "ct": "Connecticut", "de": "Delaware", "fl": "Florida", "ga": "Georgia",
    "hi": "Hawaii", "id": "Idaho", "il": "Illinois", "in": "Indiana", "ia": "Iowa",
    "ks": "Kansas", "ky": "Kentucky", "la": "Louisiana", "me": "Maine", "md": "Maryland",
    "ma": "Massachusetts", "mi": "Michigan", "mn": "Minnesota", "ms": "Mississippi",
    "mo": "Missouri", "mt": "Montana", "ne": "Nebraska", "nv": "Nevada", "nh": "New Hampshire",
    "nj": "New Jersey", "nm": "New Mexico", "ny": "New York", "nc": "North Carolina",
    "nd": "North Dakota", "oh": "Ohio", "ok": "Oklahoma", "or": "Oregon", "pa": "Pennsylvania",
    "ri": "Rhode Island", "sc": "South Carolina", "sd": "South Dakota", "tn": "Tennessee",
    "tx": "Texas", "ut": "Utah", "vt": "Vermont", "va": "Virginia", "wa": "Washington",
    "wv": "West Virginia", "wi": "Wisconsin", "wy": "Wyoming", "dc": "District of Columbia",
}

_ALIASES = {"world": _COUNTRY_ALIASES, "us_states": {}}
_ABBR = {"world": {}, "us_states": _US_STATE_ABBR}


def _build_lookup(level: str) -> dict:
    """normalized name/alias/abbreviation -> feature index in the level's GeoJSON."""
    geojson = load_geojson(level)
    lookup: dict[str, int] = {}
    for idx, feature in enumerate(geojson.get("features", [])):
        name = (feature.get("properties") or {}).get("name")
        if name:
            lookup[_norm(name)] = idx

    for alias, canonical in _ALIASES.get(level, {}).items():
        target = lookup.get(_norm(canonical))
        if target is not None:
            lookup.setdefault(_norm(alias), target)

    for abbr, full in _ABBR.get(level, {}).items():
        target = lookup.get(_norm(full))
        if target is not None:
            lookup.setdefault(_norm(abbr), target)

    return lookup


def _lookup(level: str) -> dict:
    if level not in _lookup_cache:
        _lookup_cache[level] = _build_lookup(level)
    return _lookup_cache[level]


def match_names(values, level: str) -> tuple[dict, float]:
    """
    Match raw name values against a level's bundled boundaries.

    Returns (matched, match_rate) where `matched` maps each original raw
    value (that matched) to its feature index, and `match_rate` is the
    fraction of non-null input values that matched.
    """
    lookup = _lookup(level)
    non_null = [v for v in values if v is not None and str(v).strip() != ""]
    matched = {}
    for v in non_null:
        idx = lookup.get(_norm(v))
        if idx is not None:
            matched[v] = idx
    rate = (len(matched) / len(non_null)) if non_null else 0.0
    return matched, rate


def best_level_for_values(values) -> tuple[str | None, float, dict]:
    """Try every supported level and return the best-matching one."""
    best_level, best_rate, best_matched = None, 0.0, {}
    for level in _DATASETS:
        matched, rate = match_names(values, level)
        if rate > best_rate:
            best_level, best_rate, best_matched = level, rate, matched
    return best_level, best_rate, best_matched


def find_name_column(df: pd.DataFrame, min_match_rate: float = 0.6):
    """
    Scan a DataFrame's non-numeric columns for one that looks like it holds
    country or US state names, by actually matching its values against the
    bundled boundary data.

    Returns (col, level, matched) for the best candidate, or
    (None, None, {}) if nothing clears `min_match_rate`.
    """
    candidates = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    best_col, best_level, best_rate, best_matched = None, None, 0.0, {}
    for col in candidates:
        values = df[col].dropna().astype(str).tolist()
        if len(values) < 2:
            continue
        level, rate, matched = best_level_for_values(values)
        if level and rate > best_rate:
            best_col, best_level, best_rate, best_matched = col, level, rate, matched
    if best_rate >= min_match_rate:
        return best_col, best_level, best_matched
    return None, None, {}
