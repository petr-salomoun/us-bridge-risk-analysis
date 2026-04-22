#!/usr/bin/env python3
"""
Step 5: Rank bridges by risk score and export results.
"""
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

STATE_CODES = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
    "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine",
    "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska",
    "32": "Nevada", "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico",
    "36": "New York", "37": "North Carolina", "38": "North Dakota", "39": "Ohio",
    "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
    "45": "South Carolina", "46": "South Dakota", "47": "Tennessee", "48": "Texas",
    "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming",
    "60": "American Samoa", "66": "Guam", "72": "Puerto Rico", "78": "U.S. Virgin Islands",
}


def assign_severity(score: float) -> str:
    if score >= 0.75:
        return "Critical"
    elif score >= 0.50:
        return "High"
    elif score >= 0.25:
        return "Medium"
    else:
        return "Low"


def format_output(df: pd.DataFrame) -> pd.DataFrame:
    """Select and format columns for output CSV."""
    out_cols = {
        "STATE_CODE": "state_code",
        "STRUCTURE_NUMBER": "structure_number",
        "FACILITY_CARRIED": "facility_carried",
        "FEATURES_INTERSECTED": "features_intersected",
        "LOCATION": "location",
        "COUNTY_CODE": "county_code",
        "LAT": "lat",
        "LON": "lon",
        "YEAR_BUILT": "year_built",
        "BRIDGE_AGE": "bridge_age",
        "ADT": "adt",
        "PCT_TRUCK": "pct_truck",
        "DECK_COND": "deck_condition",
        "SUPERSTR_COND": "superstr_condition",
        "SUBSTR_COND": "substr_condition",
        "CULVERT_COND": "culvert_condition",
        "CHANNEL_COND": "channel_condition",
        "MIN_STRUCT_COND": "min_struct_condition",
        "BRIDGE_CONDITION": "bridge_condition_category",
        "STRUCT_EVAL": "structural_eval",
        "SCOUR_CRITICAL": "scour_critical",
        "BRIDGE_POSTING": "bridge_posting",
        "OPEN_CLOSED_POSTED": "open_closed_posted",
        "INSPECT_DATE": "last_inspection",
        "STRUCTURE_LEN": "structure_len_m",
        "MAX_SPAN_LEN": "max_span_len_m",
        "LANES_ON": "lanes_on",
        "SDI": "sdi_score",
        "ML_PROBA": "ml_risk_proba",
        "RISK_SCORE": "risk_score",
        "SEVERITY": "severity",
        "SEVERITY_RANK": "severity_rank",
    }

    # Add state name
    df["STATE_NAME"] = df["STATE_CODE"].str.strip().str.zfill(2).map(STATE_CODES).fillna("Unknown")
    out_cols["STATE_NAME"] = "state_name"

    keep = [k for k in out_cols.keys() if k in df.columns]
    out = df[keep].rename(columns={k: out_cols[k] for k in keep})
    return out


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROCESSED_DIR / "nbi_scored.parquet")
    log.info("Loaded %d scored bridges", len(df))

    # Sort by risk score descending
    df = df.sort_values("RISK_SCORE", ascending=False).reset_index(drop=True)
    df["SEVERITY"] = df["RISK_SCORE"].apply(assign_severity)
    df["SEVERITY_RANK"] = df.index + 1  # 1 = most at risk

    # Severity distribution
    dist = df["SEVERITY"].value_counts()
    log.info("Severity distribution:\n%s", dist.to_string())

    # Export full ranked CSV
    out_df = format_output(df)
    csv_path = OUTPUTS_DIR / "bridges_ranked.csv"
    out_df.to_csv(csv_path, index=False)
    log.info("Saved ranked bridges -> %s (%d rows)", csv_path, len(out_df))

    # Export top-1000 critical bridges summary
    top1000 = out_df[out_df["severity"] == "Critical"].head(1000)
    top1000_path = OUTPUTS_DIR / "bridges_critical_top1000.csv"
    top1000.to_csv(top1000_path, index=False)
    log.info("Saved top 1000 critical bridges -> %s", top1000_path)

    # Per-state summary
    state_summary = (
        df.groupby("STATE_CODE")
        .agg(
            total_bridges=("RISK_SCORE", "count"),
            critical=("SEVERITY", lambda x: (x == "Critical").sum()),
            high=("SEVERITY", lambda x: (x == "High").sum()),
            medium=("SEVERITY", lambda x: (x == "Medium").sum()),
            low=("SEVERITY", lambda x: (x == "Low").sum()),
            avg_risk=("RISK_SCORE", "mean"),
            poor_condition=("IS_POOR", "sum") if "IS_POOR" in df.columns else ("RISK_SCORE", "count"),
        )
        .reset_index()
    )
    state_summary["state_name"] = (
        state_summary["STATE_CODE"].str.strip().str.zfill(2).map(STATE_CODES).fillna("Unknown")
    )
    state_summary = state_summary.sort_values("avg_risk", ascending=False)
    state_csv = OUTPUTS_DIR / "state_summary.csv"
    state_summary.to_csv(state_csv, index=False)
    log.info("Saved state summary -> %s", state_csv)

    # Save map-ready subset (all bridges with lat/lon, key fields only)
    map_cols = [c for c in [
        "state_code", "state_name", "structure_number", "facility_carried",
        "features_intersected", "location", "county_code",
        "lat", "lon", "year_built", "bridge_age", "adt", "pct_truck",
        "min_struct_condition", "bridge_condition_category",
        "deck_condition", "superstr_condition", "substr_condition",
        "scour_critical", "bridge_posting", "open_closed_posted",
        "sdi_score", "ml_risk_proba", "risk_score", "severity", "severity_rank",
    ] if c in out_df.columns]
    map_df = out_df[map_cols].dropna(subset=["lat", "lon"])
    map_path = PROCESSED_DIR / "bridges_map_data.parquet"
    map_df.to_parquet(map_path, index=False)
    log.info("Saved map data -> %s (%d rows with geo)", map_path, len(map_df))

    return out_df


if __name__ == "__main__":
    main()
