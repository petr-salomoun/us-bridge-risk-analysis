#!/usr/bin/env python3
"""
Step 3: Feature engineering for bridge risk model.

Applies structural engineering knowledge to derive meaningful risk features:
  - NBI Item 58 (deck), 59 (superstructure), 60 (substructure) weighted by
    collapse criticality: foundation failure > superstructure failure > deck failure.
  - NBI Item 62 (culvert): entire structure in one rating — treated independently.
  - Fracture critical members (Item 92A): single-point-of-failure — elevated risk.
  - Scour × substructure interaction: scour attacking a degraded foundation is
    multiplicative in collapse probability.
  - Design load (Item 31): lower design standards = structural reserve capacity deficit.
  - Inspection recency (Item 90): overdue inspection = undetected deterioration risk.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CURRENT_YEAR = 2024

# ── NBI Item 113: Scour Criticality ──────────────────────────────────────────
# Scour is the #1 cause of bridge collapses in the US (FHWA data).
# Ratings 4, 7, 9 = scour critical; 5, 8 = critical with countermeasures/monitoring.
SCOUR_RISK_MAP = {
    0: 0.4,   # unknown — moderate default (not inspected underwater)
    1: 0.05,  # tidal foundation — stable by definition
    2: 0.05,  # well-founded, not scour critical
    3: 0.55,  # unknown foundation — may be scour critical
    4: 0.90,  # unknown foundation — is scour critical
    5: 0.75,  # stable stream, scour critical (countermeasures applied)
    6: 0.65,  # scour critical — countermeasures applied
    7: 1.00,  # bridge closed due to scour
    8: 0.80,  # monitoring program (not yet closed)
    9: 1.00,  # scour critical — unknown if countermeasures applied
}

# ── NBI Item 31: Design Load ──────────────────────────────────────────────────
# Higher codes = higher design standard = more reserve capacity.
# Obsolete/low-load bridges operating under heavy modern truck traffic.
# Note: 0 = not rated/unknown → moderate default.
DESIGN_LOAD_RISK_MAP = {
    "0": 0.50,  # unknown — can't assess reserve capacity
    "1": 0.90,  # H10 — very obsolete (built pre-1940s typically)
    "2": 0.80,  # H15
    "3": 0.75,  # HS15 — common pre-1960s standard
    "4": 0.60,  # H20
    "5": 0.25,  # HS20 — current standard (most common)
    "6": 0.20,  # HS20+ modified
    "7": 0.10,  # HS25 — enhanced standard
    "8": 0.35,  # other specification (unknown adequacy)
    "9": 0.05,  # HL93 — current AASHTO LRFD standard
    "A": 0.30,  # notional rating
    "B": 0.30,
    "C": 0.30,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Engineering structural-engineering-informed features ...")
    df = df.copy()

    # ── Age ──────────────────────────────────────────────────────────────────
    df["BRIDGE_AGE"] = CURRENT_YEAR - df["YEAR_BUILT"]
    df.loc[df["BRIDGE_AGE"] < 0, "BRIDGE_AGE"] = np.nan
    # Non-linear age risk: accelerates after 50-60 years (corrosion, fatigue)
    age_clip = df["BRIDGE_AGE"].clip(0, 120).fillna(60)
    df["AGE_NORM"] = (age_clip / 120) ** 0.7  # concave-up: penalizes very old more

    # ── Structural component conditions with collapse-criticality weights ─────
    # NBI 0-9 scale. Collapse criticality (Paulson & Kulicki structural precedent):
    #   Substructure (60): foundation/pier failure → total collapse risk ~0.45
    #   Superstructure (59): primary load-carrying → partial/total collapse ~0.35
    #   Deck (58): riding surface → closure but rarely sudden total collapse ~0.20
    #
    # For culverts (Item 62): single rating covers entire structure.
    # Treat culvert condition as equivalent to superstructure condition.

    def cond_to_risk(series, fill=5.0):
        """Convert NBI 0-9 condition to [0,1] risk. Non-linear: below 4 = poor threshold."""
        c = pd.to_numeric(series, errors="coerce").fillna(fill)
        # Piecewise: 9→0, 5→0.3, 4→0.6, 3→0.8, 2→0.9, 0→1.0
        # Nonlinear scaling — deterioration below threshold accelerates collapse risk
        risk = 1.0 - (c / 9.0)
        # Extra penalty for ratings ≤ 4 (poor/serious/critical/failed)
        below_threshold = c <= 4
        risk = np.where(below_threshold, risk + 0.2 * (1.0 - c / 4.0), risk)
        return pd.Series(np.clip(risk, 0, 1), index=series.index)

    has_culvert = df["CULVERT_COND"].notna()
    has_bridge_cond = df["DECK_COND"].notna() | df["SUPERSTR_COND"].notna() | df["SUBSTR_COND"].notna()

    # Component risks for bridge structures
    deck_risk    = cond_to_risk(df["DECK_COND"],     fill=5.0)
    super_risk   = cond_to_risk(df["SUPERSTR_COND"], fill=5.0)
    substr_risk  = cond_to_risk(df["SUBSTR_COND"],   fill=5.0)
    culvert_risk = cond_to_risk(df["CULVERT_COND"],  fill=5.0)

    # Weighted composite for bridges: substructure highest weight
    df["BRIDGE_STRUCT_RISK"] = (
        0.45 * substr_risk +
        0.35 * super_risk +
        0.20 * deck_risk
    )
    # For culverts, entire rating covers the structure
    df["CULVERT_STRUCT_RISK"] = culvert_risk

    # Combined: use culvert rating when it's the primary element, bridge otherwise
    df["COND_RISK"] = np.where(
        has_culvert & ~has_bridge_cond,
        df["CULVERT_STRUCT_RISK"],
        np.where(
            has_culvert,
            # Both exist: blend (culvert often is the same structure as substructure)
            0.5 * df["BRIDGE_STRUCT_RISK"] + 0.5 * df["CULVERT_STRUCT_RISK"],
            df["BRIDGE_STRUCT_RISK"]
        )
    )
    df["COND_RISK"] = df["COND_RISK"].fillna(0.45)  # unknown → moderate risk

    # Expose individual component risks for SDI and ML
    df["SUBSTR_RISK"] = substr_risk.fillna(0.45)
    df["SUPER_RISK"]  = super_risk.fillna(0.45)
    df["DECK_RISK"]   = deck_risk.fillna(0.45)

    # ── Structurally deficient flag ───────────────────────────────────────────
    df["IS_POOR"] = (df.get("BRIDGE_CONDITION", pd.Series(dtype=str)) == "P").astype(int)
    df["IS_COND_POOR"] = (
        df[["DECK_COND", "SUPERSTR_COND", "SUBSTR_COND", "CULVERT_COND"]]
        .min(axis=1, skipna=True) <= 4
    ).astype(int)

    # ── Scour risk (Item 113) ─────────────────────────────────────────────────
    # Primary collapse cause for river bridges. Amplified by poor substructure.
    if "SCOUR_CRITICAL" in df.columns:
        base_scour = df["SCOUR_CRITICAL"].map(SCOUR_RISK_MAP).fillna(0.35)
    else:
        base_scour = pd.Series(0.35, index=df.index)

    # Scour × substructure interaction: poor foundation + scour = elevated collapse risk
    df["SCOUR_RISK"] = (base_scour * (1.0 + 0.5 * df["SUBSTR_RISK"])).clip(0, 1)

    # ── Fracture critical members (Item 92A) ──────────────────────────────────
    # NBI Y = yes (fracture critical = no structural redundancy, single point of failure)
    if "FRACTURE" in df.columns:
        df["FRACTURE_CRITICAL"] = df["FRACTURE"].astype(str).str.startswith("Y").astype(float)
    else:
        df["FRACTURE_CRITICAL"] = 0.0

    # ── Design load adequacy (Item 31) ────────────────────────────────────────
    # Bridges designed for lighter loads operating under heavy modern truck traffic.
    if "DESIGN_LOAD" in df.columns:
        dl = df["DESIGN_LOAD"].astype(str).str.strip()
        df["DESIGN_LOAD_RISK"] = dl.map(DESIGN_LOAD_RISK_MAP).fillna(0.40)
    else:
        df["DESIGN_LOAD_RISK"] = 0.40

    # ── Traffic stress ────────────────────────────────────────────────────────
    adt = df["ADT"].clip(0, 200_000).fillna(0)
    df["ADT_LOG_NORM"] = np.log1p(adt) / np.log1p(200_000)
    # Age × traffic × design load: old bridge + high traffic + obsolete design = stress
    df["ADT_AGE_STRESS"] = df["ADT_LOG_NORM"] * df["AGE_NORM"] * (1.0 + df["DESIGN_LOAD_RISK"])

    # ── Truck percentage ──────────────────────────────────────────────────────
    df["TRUCK_NORM"] = (df.get("PCT_TRUCK", pd.Series(0.0, index=df.index))
                        .fillna(0).clip(0, 100) / 100)
    # Truck × design load: heavy trucks on under-designed bridge
    df["TRUCK_LOAD_STRESS"] = df["TRUCK_NORM"] * df["DESIGN_LOAD_RISK"]

    # ── Load posting ──────────────────────────────────────────────────────────
    # Item 70: 5=unrestricted, ≤4 = some restriction, 2=posted, 1=below posting, 0=closed
    if "BRIDGE_POSTING" in df.columns:
        bp = pd.to_numeric(df["BRIDGE_POSTING"], errors="coerce").fillna(5)
        # Graduated: fully closed=1.0, heavily posted=0.7, minor restriction=0.3
        df["LOAD_POSTED"] = ((5 - bp.clip(0, 5)) / 5).clip(0, 1)
    else:
        df["LOAD_POSTED"] = 0.0

    # ── Channel condition (Item 61) ───────────────────────────────────────────
    if "CHANNEL_COND" in df.columns:
        ch = pd.to_numeric(df["CHANNEL_COND"], errors="coerce").fillna(6)
        df["CHANNEL_RISK"] = cond_to_risk(pd.to_numeric(df["CHANNEL_COND"], errors="coerce"), fill=6.0)
    else:
        df["CHANNEL_RISK"] = 0.0

    # ── Structural evaluation appraisal (Item 67) ────────────────────────────
    # Engineer's holistic assessment of overall structural adequacy.
    # Codes ≤2 = critical/imminent failure, 3 = serious, 4 = poor/tolerable
    if "STRUCT_EVAL" in df.columns:
        se = pd.to_numeric(df["STRUCT_EVAL"], errors="coerce").fillna(5)
        df["STRUCT_EVAL_RISK"] = cond_to_risk(se, fill=5.0)
    else:
        df["STRUCT_EVAL_RISK"] = 0.0

    # ── Waterway adequacy (Item 71) ───────────────────────────────────────────
    if "WATERWAY_EVAL" in df.columns:
        ww = pd.to_numeric(df["WATERWAY_EVAL"], errors="coerce").fillna(5)
        df["WATERWAY_RISK"] = cond_to_risk(ww, fill=5.0)
    else:
        df["WATERWAY_RISK"] = 0.0

    # ── Last inspection recency (Item 90) ────────────────────────────────────
    # FHWA requires biennial inspection. Overdue → undetected deterioration.
    if "INSPECT_DATE" in df.columns:
        def parse_inspect_year(v):
            try:
                s = str(v).strip().zfill(4)
                yy = int(s[2:4])
                return 2000 + yy if yy <= 30 else 1900 + yy
            except Exception:
                return np.nan
        df["INSPECT_YEAR"] = df["INSPECT_DATE"].apply(parse_inspect_year)
        df["YEARS_SINCE_INSPECT"] = CURRENT_YEAR - df["INSPECT_YEAR"]
        # Risk rises sharply after 2 years (standard cycle), cap at 5 years
        df["INSPECT_RISK"] = (df["YEARS_SINCE_INSPECT"].clip(0, 5) / 5).fillna(0.5)
    else:
        df["INSPECT_RISK"] = 0.3

    # ── Open/closed/posted status ─────────────────────────────────────────────
    if "OPEN_CLOSED_POSTED" in df.columns:
        df["IS_CLOSED"] = (df["OPEN_CLOSED_POSTED"].str.strip().isin(["K", "P"])).astype(int)
    else:
        df["IS_CLOSED"] = 0

    log.info("Features engineered. Shape: %s", df.shape)
    return df


FEATURE_COLS = [
    "COND_RISK",
    "SUBSTR_RISK",
    "SUPER_RISK",
    "DECK_RISK",
    "AGE_NORM",
    "ADT_LOG_NORM",
    "ADT_AGE_STRESS",
    "LOAD_POSTED",
    "SCOUR_RISK",
    "FRACTURE_CRITICAL",
    "DESIGN_LOAD_RISK",
    "CHANNEL_RISK",
    "STRUCT_EVAL_RISK",
    "WATERWAY_RISK",
    "TRUCK_NORM",
    "TRUCK_LOAD_STRESS",
    "INSPECT_RISK",
    "IS_CLOSED",
]


def main():
    inp = PROCESSED_DIR / "nbi_clean.parquet"
    df = pd.read_parquet(inp)
    df = engineer_features(df)
    out = PROCESSED_DIR / "nbi_features.parquet"
    df.to_parquet(out, index=False)
    log.info("Saved features -> %s (%d rows)", out, len(df))
    return df


if __name__ == "__main__":
    main()
