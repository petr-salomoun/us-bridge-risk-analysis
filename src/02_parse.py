#!/usr/bin/env python3
"""
Step 2: Parse NBI delimited file into a clean pandas DataFrame.

The 2024 NBI file is comma-delimited with single-quote text qualifier.
Column names are derived from the NBI record format specification.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# NBI column names (from record format specification)
# Position in the delimited file (0-indexed) maps to item numbers.
# The delimited CSV has a header row with item numbers.
# We rename to friendly names here.

RENAME_MAP = {
    "STATE_CODE_001": "STATE_CODE",
    "STRUCTURE_NUMBER_008": "STRUCTURE_NUMBER",
    "RECORD_TYPE_005A": "RECORD_TYPE",
    "ROUTE_PREFIX_005B": "ROUTE_PREFIX",
    "SERVICE_LEVEL_005C": "SERVICE_LEVEL",
    "ROUTE_NUMBER_005D": "ROUTE_NUMBER",
    "DIRECTION_005E": "DIRECTION_SUFFIX",
    "HIGHWAY_DIST_002": "HIGHWAY_DISTRICT",
    "COUNTY_CODE_003": "COUNTY_CODE",
    "PLACE_CODE_004": "PLACE_CODE",
    "FEATURES_DESC_006A": "FEATURES_INTERSECTED",
    "CRITICAL_FACILITY_006B": "CRITICAL_FACILITY",
    "FACILITY_CARRIED_007": "FACILITY_CARRIED",
    "LOCATION_009": "LOCATION",
    "MIN_VERT_CLR_010": "MIN_VERT_CLR_010",
    "KILOPOINT_011": "KILOPOINT",
    "BASE_HWY_NETWORK_012": "BASE_HWY_NETWORK",
    "LRS_INV_ROUTE_013A": "LRS_INV_ROUTE",
    "SUBROUTE_NO_013B": "SUBROUTE_NO",
    "LAT_016": "LATITUDE",
    "LONG_017": "LONGITUDE",
    "DETOUR_KILOS_019": "DETOUR_KILOS",
    "TOLL_020": "TOLL",
    "MAINTENANCE_021": "MAINTENANCE_RESP",
    "OWNER_022": "OWNER",
    "FUNCTIONAL_CLASS_026": "FUNCTIONAL_CLASS",
    "YEAR_BUILT_027": "YEAR_BUILT",
    "TRAFFIC_LANES_ON_028A": "LANES_ON",
    "TRAFFIC_LANES_UND_028B": "LANES_UNDER",
    "ADT_029": "ADT",
    "YEAR_ADT_030": "YEAR_ADT",
    "DESIGN_LOAD_031": "DESIGN_LOAD",
    "APPR_WIDTH_MT_032": "APPR_WIDTH",
    "MEDIAN_CODE_033": "MEDIAN_CODE",
    "DEGREES_SKEW_034": "DEGREES_SKEW",
    "STRUCTURE_FLARED_035": "STRUCTURE_FLARED",
    "RAILINGS_036A": "RAILINGS",
    "TRANSITIONS_036B": "TRANSITIONS",
    "APPR_RAIL_036C": "APPR_RAIL",
    "APPR_RAIL_END_036D": "APPR_RAIL_END",
    "HISTORY_037": "HISTORICAL_SIGNIF",
    "NAVIGATION_038": "NAVIGATION_CTRL",
    "NAV_VERT_CLR_MT_039": "NAV_VERT_CLR",
    "NAV_HORR_CLR_MT_040": "NAV_HORIZ_CLR",
    "OPEN_CLOSED_POSTED_041": "OPEN_CLOSED_POSTED",
    "SERVICE_ON_042A": "SERVICE_ON",
    "SERVICE_UND_042B": "SERVICE_UNDER",
    "STRUCTURE_KIND_043A": "MATERIAL_KIND",
    "STRUCTURE_TYPE_043B": "DESIGN_TYPE",
    "APPR_KIND_044A": "APPR_MATERIAL",
    "APPR_TYPE_044B": "APPR_DESIGN",
    "MAIN_UNIT_SPANS_045": "MAIN_SPANS",
    "APPR_SPANS_046": "APPR_SPANS",
    "HORR_CLR_MT_047": "HORZ_CLR",
    "MAX_SPAN_LEN_MT_048": "MAX_SPAN_LEN",
    "STRUCTURE_LEN_MT_049": "STRUCTURE_LEN",
    "LEFT_CURB_MT_050A": "LEFT_CURB",
    "RIGHT_CURB_MT_050B": "RIGHT_CURB",
    "ROADWAY_WIDTH_MT_051": "ROADWAY_WIDTH",
    "DECK_WIDTH_MT_052": "DECK_WIDTH",
    "VERT_CLR_OVER_MT_053": "VERT_CLR_OVER",
    "VERT_CLR_UND_REF_054A": "VERT_CLR_UND_REF",
    "VERT_CLR_UND_054B": "VERT_CLR_UND",
    "LAT_UND_REF_055A": "LAT_UND_REF",
    "LAT_UND_MT_055B": "LAT_UND",
    "LEFT_LAT_UND_MT_056": "LEFT_LAT_UND",
    "DECK_COND_058": "DECK_COND",
    "SUPERSTRUCTURE_COND_059": "SUPERSTR_COND",
    "SUBSTRUCTURE_COND_060": "SUBSTR_COND",
    "CHANNEL_COND_061": "CHANNEL_COND",
    "CULVERT_COND_062": "CULVERT_COND",
    "OPR_RATING_METH_063": "OPR_RATING_METH",
    "OPERATING_RATING_064": "OPERATING_RATING",
    "INV_RATING_METH_065": "INV_RATING_METH",
    "INVENTORY_RATING_066": "INVENTORY_RATING",
    "STRUCTURAL_EVAL_067": "STRUCT_EVAL",
    "DECK_GEOMETRY_EVAL_068": "DECK_GEOM_EVAL",
    "UNDCLRENCE_EVAL_069": "UNDCLR_EVAL",
    "POSTING_EVAL_070": "BRIDGE_POSTING",
    "WATERWAY_EVAL_071": "WATERWAY_EVAL",
    "APPR_ROAD_EVAL_072": "APPR_ROAD_EVAL",
    "WORK_PROPOSED_075A": "WORK_PROPOSED",
    "WORK_DONE_BY_075B": "WORK_DONE_BY",
    "IMP_LEN_MT_076": "IMP_LEN",
    "DATE_OF_INSPECT_090": "INSPECT_DATE",
    "INSPECT_FREQ_MONTHS_091": "INSPECT_FREQ",
    "FRACTURE_092A": "FRACTURE",
    "UNDWATER_LOOK_SEE_092B": "UNDERWATER_INSP",
    "SPEC_INSPECT_092C": "SPEC_INSPECT",
    "FRACTURE_LAST_DATE_093A": "FRACTURE_DATE",
    "UNDWATER_LAST_DATE_093B": "UNDERWATER_DATE",
    "SPEC_LAST_DATE_093C": "SPEC_DATE",
    "BRIDGE_IMP_COST_094": "BRIDGE_IMP_COST",
    "ROADWAY_IMP_COST_095": "ROADWAY_IMP_COST",
    "TOTAL_IMP_COST_096": "TOTAL_IMP_COST",
    "YEAR_OF_IMP_097": "YEAR_OF_IMP",
    "OTHER_STATE_CODE_098A": "BORDER_STATE",
    "OTHER_STATE_PCNT_098B": "BORDER_PCT",
    "OTHR_STATE_STRUC_NO_099": "BORDER_STRUC_NO",
    "STRAHNET_HIGHWAY_100": "STRAHNET_HWY",
    "PARALLEL_STRUCTURE_101": "PARALLEL_STRUCT",
    "TRAFFIC_DIRECTION_102": "TRAFFIC_DIR",
    "TEMP_STRUCTURE_103": "TEMP_STRUCTURE",
    "HIGHWAY_SYSTEM_104": "HWY_SYSTEM",
    "FEDERAL_LANDS_105": "FED_LANDS",
    "YEAR_RECONSTRUCTED_106": "YEAR_RECONSTRUCTED",
    "DECK_STRUCTURE_TYPE_107": "DECK_STRUCT_TYPE",
    "SURFACE_TYPE_108A": "SURFACE_TYPE",
    "MEMBRANE_TYPE_108B": "MEMBRANE_TYPE",
    "DECK_PROTECTION_108C": "DECK_PROTECTION",
    "PERCENT_ADT_TRUCK_109": "PCT_TRUCK",
    "NATIONAL_NETWORK_110": "NAT_NETWORK",
    "PIER_PROTECTION_111": "PIER_PROTECTION",
    "BRIDGE_LEN_IND_112": "BRIDGE_LEN_IND",
    "SCOUR_CRITICAL_113": "SCOUR_CRITICAL",
    "FUTURE_ADT_114": "FUTURE_ADT",
    "YEAR_OF_FUTURE_ADT_115": "YEAR_FUTURE_ADT",
    "MIN_NAV_CLR_MT_116": "MIN_NAV_CLR",
    "FED_AGENCY": "FED_AGENCY",
    "SUBMITTED_BY": "SUBMITTED_BY",
    "BRIDGE_CONDITION": "BRIDGE_CONDITION",
    "LOWEST_RATING": "LOWEST_RATING",
    "DECK_AREA": "DECK_AREA",
}

# Condition code columns
CONDITION_COLS = ["DECK_COND", "SUPERSTR_COND", "SUBSTR_COND", "CHANNEL_COND", "CULVERT_COND"]


def find_nbi_file(raw_dir: Path) -> Path:
    """Find the extracted NBI text/csv file."""
    candidates = sorted(raw_dir.glob("*.txt")) + sorted(raw_dir.glob("*.csv"))
    # Filter out tiny files
    candidates = [f for f in candidates if f.stat().st_size > 10_000_000]
    if not candidates:
        raise FileNotFoundError(
            f"No NBI data file found in {raw_dir}. "
            "Run 01_download.py first."
        )
    # Prefer the largest file
    best = max(candidates, key=lambda f: f.stat().st_size)
    log.info("Using NBI file: %s (%.1f MB)", best, best.stat().st_size / 1e6)
    return best


def load_nbi(path: Path) -> pd.DataFrame:
    """Load the NBI delimited file."""
    log.info("Reading %s ...", path)
    # NBI delimited files: comma-separated, single-quote text qualifier, has header
    df = pd.read_csv(
        path,
        dtype=str,
        na_values=["", " "],
        quotechar="'",
        on_bad_lines="skip",
        low_memory=False,
    )
    log.info("Loaded %d rows × %d cols", len(df), len(df.columns))

    # Rename columns
    col_map = {}
    for col in df.columns:
        col_strip = col.strip()
        if col_strip in RENAME_MAP:
            col_map[col] = RENAME_MAP[col_strip]
        else:
            col_map[col] = col_strip
    df.rename(columns=col_map, inplace=True)

    return df


def clean_nbi(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Cleaning NBI data ...")

    # Keep only highway bridges (RECORD_TYPE == '1')
    if "RECORD_TYPE" in df.columns:
        before = len(df)
        df = df[df["RECORD_TYPE"].str.strip() == "1"].copy()
        log.info("Filtered to highway bridges: %d -> %d rows", before, len(df))

    # Parse lat/lon: stored as DDMMSSXX (8-digit) in degrees*1e6? 
    # Actually NBI lat is stored as DDMMSSHH (8 digits, e.g. 38123456 = 38°12'34.56")
    # Longitude as DDDMMSSHH (9 digits)
    if "LATITUDE" in df.columns:
        def parse_lat(v):
            try:
                v = str(v).strip().zfill(8)
                deg = int(v[0:2])
                mn = int(v[2:4])
                sec = float(v[4:8]) / 100
                return deg + mn / 60 + sec / 3600
            except Exception:
                return np.nan

        def parse_lon(v):
            try:
                v = str(v).strip().zfill(9)
                deg = int(v[0:3])
                mn = int(v[3:5])
                sec = float(v[5:9]) / 100
                return -abs(deg + mn / 60 + sec / 3600)  # always west in US
            except Exception:
                return np.nan

        df["LAT"] = df["LATITUDE"].apply(parse_lat)
        df["LON"] = df["LONGITUDE"].apply(parse_lon)

        # Sanity-check: continental US + territories
        valid_geo = (
            (df["LAT"].between(17.0, 72.0)) &
            (df["LON"].between(-180.0, -60.0))
        )
        n_invalid = (~valid_geo).sum()
        if n_invalid > 0:
            log.warning("Dropping %d rows with invalid lat/lon", n_invalid)
        df = df[valid_geo].copy()

    # Parse numeric columns
    numeric_cols = [
        "YEAR_BUILT", "ADT", "LANES_ON", "FUTURE_ADT",
        "STRUCTURE_LEN", "MAX_SPAN_LEN", "OPERATING_RATING", "INVENTORY_RATING",
        "PCT_TRUCK",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse condition codes (0-9 scale, 'N' = not applicable)
    for col in CONDITION_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].str.strip(), errors="coerce")

    # Other rating fields
    for col in ["STRUCT_EVAL", "BRIDGE_POSTING", "SCOUR_CRITICAL",
                "WATERWAY_EVAL", "LOWEST_RATING"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].str.strip(), errors="coerce")

    # BRIDGE_CONDITION: 'G'/'F'/'P'/NaN
    if "BRIDGE_CONDITION" in df.columns:
        df["BRIDGE_CONDITION"] = df["BRIDGE_CONDITION"].str.strip().str.upper()

    # Year built sanity
    if "YEAR_BUILT" in df.columns:
        df.loc[df["YEAR_BUILT"] < 1800, "YEAR_BUILT"] = np.nan
        df.loc[df["YEAR_BUILT"] > 2025, "YEAR_BUILT"] = np.nan

    log.info("Clean data: %d rows × %d cols", len(df), len(df.columns))
    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    nbi_path = find_nbi_file(RAW_DIR)
    df = load_nbi(nbi_path)
    df = clean_nbi(df)
    out = PROCESSED_DIR / "nbi_clean.parquet"
    df.to_parquet(out, index=False)
    log.info("Saved clean data -> %s (%d rows)", out, len(df))
    return df


if __name__ == "__main__":
    main()
