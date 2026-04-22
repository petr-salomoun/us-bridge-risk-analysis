#!/usr/bin/env python3
"""
Step 9: Collapse probability, rush-hour occupancy, and casualty differentiation.

1-Year Collapse Probability Model
-----------------------------------
Base rate: ~1.3e-5 per bridge per year (≈8 structural collapses/yr across 622k bridges).
Exponential scaling by risk_score: Critical bridges ~200× more likely than Low.

Rush-Hour Occupancy (Little's Law)
------------------------------------
N_persons = (ADT × K_peak) × (bridge_length / speed) × occupancy_factor

Fatality vs. Injury Differentiation
--------------------------------------
Not everyone on a collapsing bridge dies. Survival depends on:

1. Crossing type (water vs. land):
   - Water crossing + drowning/submersion = high fatality rate
   - Land crossing = vehicles may survive fall depending on height

2. Fall height proxy (max_span_len_m as geometric proxy):
   - Short spans (<20m): low fall, some survivable → lower fatality rate
   - Medium spans (20-50m): significant fall, moderate survival
   - Long spans (>50m): high fall, very low survival over water

3. Truck fraction:
   - Trucks: minimal crumple zone protection, high fatality rate when cabin crushed

Historical calibration:
  - I-35W (2007): 13 dead, ~145 injured, ~111 people on bridge at collapse
    → fatality rate ~12%, injury rate ~56%
  - Mianus River (1983): 3 dead, 3 injured, ~10 on bridge
    → fatality rate ~30%, injury rate ~30%
  - Fern Hollow (2022): 0 dead, 10 injured, ~5 on bridge
    → fatality rate 0%, injury rate ~100% (low span, no water)
  - Silver Bridge (1967): 46 dead, 9 injured, ~75 on bridge
    → fatality rate ~61%, injury rate ~12% (high fall into river, winter)

Model: fatality_rate = f(water_crossing, span_length, truck_fraction)
       injury_rate   = g(water_crossing, span_length, truck_fraction)
       with fatality + injury ≤ 1.0 (some may escape unharmed from low spans)
"""
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
CHARTS_DIR  = OUTPUTS_DIR / "charts"

# ── Collapse probability parameters ──────────────────────────────────────────
BASE_COLLAPSE_RATE    = 1.3e-5   # per bridge per year (~8 events/yr nationally)
P_COLLAPSE_AT_RISK_1  = 2.5e-3   # at risk_score=1.0: ~1 in 400/yr
K_EXPONENT            = math.log(P_COLLAPSE_AT_RISK_1 / BASE_COLLAPSE_RATE)

# ── Occupancy parameters ──────────────────────────────────────────────────────
SPEED_FREE_MS    = 80_000 / 3600   # 80 km/h
SPEED_POSTED_MS  = 40_000 / 3600   # 40 km/h (posted bridge)
PEAK_FRACTION    = 0.10            # HCM K-factor (10% of ADT in peak hour)
OCC_CAR          = 1.67            # persons/car (NHTS 2022)
OCC_TRUCK        = 1.10            # persons/truck
DEFAULT_LEN_M    = 30.0
DEFAULT_LANES    = 2

# ── Water crossing keyword detection ─────────────────────────────────────────
WATER_KEYWORDS = re.compile(
    r"\b(RIVER|CREEK|LAKE|BAY|OCEAN|HARBOR|CHANNEL|STREAM|BROOK|RUN|"
    r"BRANCH|FORK|POND|RESERVOIR|INLET|SOUND|STRAIT|ESTUARY|BAYOU|"
    r"SLOUGH|CANAL|WATERWAY|SEA|GULF|COVE|LAGOON)\b",
    re.IGNORECASE,
)

# ── Fatality / injury rate parameters ────────────────────────────────────────
# Base fatality rates calibrated to historical events
# Water crossing increases fatality rate substantially (drowning, submersion)
BASE_FATALITY_LAND_SHORT  = 0.08   # short span (<20m), land: survivable fall
BASE_FATALITY_LAND_MEDIUM = 0.20   # medium span (20-50m), land
BASE_FATALITY_LAND_LONG   = 0.40   # long span (>50m), land: severe fall
BASE_FATALITY_WATER_SHORT = 0.30   # water + short span: drowning risk even shallow
BASE_FATALITY_WATER_MEDIUM= 0.50   # water + medium span: high drowning + impact
BASE_FATALITY_WATER_LONG  = 0.70   # water + long span: I-35W / Silver Bridge type

# Injury rates (of those who don't die; injuries = survivable harm)
BASE_INJURY_LAND_SHORT    = 0.70   # most survivors are injured
BASE_INJURY_LAND_MEDIUM   = 0.60
BASE_INJURY_LAND_LONG     = 0.45
BASE_INJURY_WATER_SHORT   = 0.45
BASE_INJURY_WATER_MEDIUM  = 0.35
BASE_INJURY_WATER_LONG    = 0.22

# Truck fatality premium: trucks have higher fatality rate than cars
TRUCK_FATALITY_PREMIUM  = 0.15   # additional fatality rate for truck occupants


def is_water_crossing(features: pd.Series) -> pd.Series:
    """Return True if the bridge crosses a water body."""
    return features.fillna("").astype(str).str.upper().str.contains(
        r"RIVER|CREEK|LAKE|BAY|OCEAN|HARBOR|CHANNEL|STREAM|BROOK|"
        r"BRANCH|FORK|POND|RESERVOIR|INLET|SOUND|STRAIT|ESTUARY|BAYOU|"
        r"SLOUGH|CANAL|WATERWAY|SEA|GULF|COVE|LAGOON",
        regex=True, na=False
    )


def compute_collapse_prob(risk_score: pd.Series) -> pd.Series:
    return BASE_COLLAPSE_RATE * np.exp(K_EXPONENT * risk_score.clip(0, 1))


def compute_span_category(span_m: pd.Series) -> pd.Series:
    """Classify span length: 'short' (<20m), 'medium' (20-50m), 'long' (>50m)."""
    span = span_m.fillna(20.0)
    return pd.cut(span, bins=[0, 20, 50, 9999],
                  labels=["short", "medium", "long"], right=False)


def compute_fatality_injury_rates(df: pd.DataFrame):
    """
    Returns (fatality_rate, injury_rate) arrays for each bridge.
    Fatality rate: fraction of persons on bridge expected to die.
    Injury rate:   fraction expected to suffer non-fatal injury.
    Unharmed:      1 - fatality_rate - injury_rate.
    """
    water = is_water_crossing(df.get("features_intersected", pd.Series("", index=df.index)))
    span_cat = compute_span_category(df.get("max_span_len_m", pd.Series(20.0, index=df.index)))
    truck_frac = df.get("pct_truck", pd.Series(10.0, index=df.index)).fillna(10.0).clip(0, 100) / 100.0

    fat_rate = pd.Series(0.0, index=df.index)
    inj_rate = pd.Series(0.0, index=df.index)

    for is_w in [False, True]:
        for span in ["short", "medium", "long"]:
            mask = (water == is_w) & (span_cat == span)
            if is_w:
                fr = {"short": BASE_FATALITY_WATER_SHORT,
                      "medium": BASE_FATALITY_WATER_MEDIUM,
                      "long": BASE_FATALITY_WATER_LONG}[span]
                ir = {"short": BASE_INJURY_WATER_SHORT,
                      "medium": BASE_INJURY_WATER_MEDIUM,
                      "long": BASE_INJURY_WATER_LONG}[span]
            else:
                fr = {"short": BASE_FATALITY_LAND_SHORT,
                      "medium": BASE_FATALITY_LAND_MEDIUM,
                      "long": BASE_FATALITY_LAND_LONG}[span]
                ir = {"short": BASE_INJURY_LAND_SHORT,
                      "medium": BASE_INJURY_LAND_MEDIUM,
                      "long": BASE_INJURY_LAND_LONG}[span]
            fat_rate[mask] = fr
            inj_rate[mask] = ir

    # Truck premium: blend car and truck fatality rates
    fat_rate_adj = fat_rate * (1 - truck_frac) + (fat_rate + TRUCK_FATALITY_PREMIUM).clip(0, 1) * truck_frac
    # Ensure fatality + injury ≤ 1.0
    inj_rate_adj = np.minimum(inj_rate, 1.0 - fat_rate_adj)

    return fat_rate_adj, inj_rate_adj


def compute_rush_hour_occupancy(df: pd.DataFrame) -> pd.Series:
    length    = df["structure_len_m"].fillna(DEFAULT_LEN_M).clip(5, 5000)
    posted    = df["bridge_posting"].fillna(5)
    speed     = np.where(posted <= 3, SPEED_POSTED_MS, SPEED_FREE_MS)
    travel_h  = (length / speed) / 3600.0
    adt       = df["adt"].fillna(0).clip(0)
    peak_flow = adt * PEAK_FRACTION
    n_veh     = peak_flow * travel_h
    truck_frac= df["pct_truck"].fillna(10.0).clip(0, 100) / 100.0
    n_persons = n_veh * ((1 - truck_frac) * OCC_CAR + truck_frac * OCC_TRUCK)
    return n_persons.clip(0)


def run(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing collapse probabilities, occupancy, fatality/injury rates ...")
    df = df.copy()

    df["p_collapse_1yr"]   = compute_collapse_prob(df["risk_score"])
    df["rush_hour_persons"] = compute_rush_hour_occupancy(df)
    df["is_water_crossing"] = is_water_crossing(
        df.get("features_intersected", pd.Series("", index=df.index))
    )

    fat_rate, inj_rate = compute_fatality_injury_rates(df)
    df["fatality_rate"]   = fat_rate.round(3)
    df["injury_rate"]     = inj_rate.round(3)
    df["unharmed_rate"]   = (1.0 - fat_rate - inj_rate).clip(0).round(3)

    # Absolute expected counts per person × occupancy
    df["expected_fatalities"]  = (df["p_collapse_1yr"] * df["rush_hour_persons"] * df["fatality_rate"]).round(5)
    df["expected_injuries"]    = (df["p_collapse_1yr"] * df["rush_hour_persons"] * df["injury_rate"]).round(5)
    df["expected_affected"]    = (df["p_collapse_1yr"] * df["rush_hour_persons"]).round(5)  # total affected

    log.info("Collapse prob range: %.2e – %.2e", df["p_collapse_1yr"].min(), df["p_collapse_1yr"].max())
    log.info("Rush-hour persons range: %.1f – %.1f", df["rush_hour_persons"].min(), df["rush_hour_persons"].max())
    log.info("Water crossings: %d / %d", df["is_water_crossing"].sum(), len(df))
    log.info("Expected annual fatalities: %.3f", df["expected_fatalities"].sum())
    log.info("Expected annual injuries:   %.3f", df["expected_injuries"].sum())
    log.info("Expected annual affected:   %.3f", df["expected_affected"].sum())

    return df


def generate_charts(df: pd.DataFrame):
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Chart 10: Collapse probability distribution + risk vs. exposure scatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"Critical": "#d32f2f", "High": "#f57c00", "Medium": "#fbc02d", "Low": "#388e3c"}

    ax = axes[0]
    for sev in ["Critical", "High", "Medium", "Low"]:
        sub = df[df["severity"] == sev]["p_collapse_1yr"]
        if len(sub) == 0:
            continue
        ax.hist(np.log10(sub.clip(1e-7)), bins=40, alpha=0.6,
                color=colors[sev], label=f"{sev} (n={len(sub):,})", density=True)
    ax.set_xlabel("log₁₀(Annual Collapse Probability)")
    ax.set_ylabel("Density")
    ax.set_title("1-Year Collapse Probability by Severity Tier")
    ax.legend(fontsize=9)
    # Trim x-axis to where data actually lives (p range ~1e-6 to ~3e-3)
    ax.set_xlim(-6.2, -2.3)
    ax.set_xticks([-6, -5, -4, -3])
    ax.set_xticklabels(["1 in 1M", "1 in 100k", "1 in 10k", "1 in 1k"])

    ax2 = axes[1]
    sample = df[df["severity"].isin(["Critical", "High"]) & (df["rush_hour_persons"] > 0)].copy()
    water_colors = np.where(sample["is_water_crossing"],
                            [{"Critical": "#b71c1c", "High": "#e65100"}[s] for s in sample["severity"]],
                            [{"Critical": "#ef9a9a", "High": "#ffcc80"}[s] for s in sample["severity"]])
    ax2.scatter(
        np.log10(sample["p_collapse_1yr"].clip(1e-7)),
        np.log10(sample["rush_hour_persons"].clip(0.01)),
        c=water_colors, s=8, alpha=0.6, rasterized=True
    )
    ax2.set_xlabel("log₁₀(Annual Collapse Probability)")
    ax2.set_ylabel("log₁₀(Rush-Hour Persons on Bridge)")
    ax2.set_title("Critical & High: Risk vs. Exposure\n(dark=water crossing, light=land)")
    ax2.set_xticks([-5, -4, -3, -2])
    ax2.set_xticklabels(["1 in 100k", "1 in 10k", "1 in 1k", "1 in 100"])
    ax2.legend(handles=[
        Patch(color="#b71c1c", label="Critical / water"),
        Patch(color="#ef9a9a", label="Critical / land"),
        Patch(color="#e65100", label="High / water"),
        Patch(color="#ffcc80", label="High / land"),
    ], fontsize=8)
    fig.tight_layout()
    path = CHARTS_DIR / "10_collapse_risk_occupancy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)

    # Chart 11: Top 20 by expected fatalities — stacked fatalities + injuries
    top20 = df.nlargest(20, "expected_fatalities")[
        ["facility_carried", "features_intersected", "state_name", "adt",
         "rush_hour_persons", "p_collapse_1yr", "fatality_rate", "injury_rate",
         "expected_fatalities", "expected_injuries", "expected_affected",
         "severity", "is_water_crossing"]
    ].copy()
    top20["label"] = (top20["facility_carried"].str.strip().str[:22]
                      + "\n" + top20["state_name"].str[:12])

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_colors = [{"Critical": "#d32f2f", "High": "#f57c00", "Medium": "#fbc02d"}.get(s, "#888")
                  for s in top20["severity"]]
    ax.barh(range(len(top20)), top20["expected_fatalities"],
            color=bar_colors, edgecolor="white", linewidth=0.5, label="Expected Fatalities")
    ax.barh(range(len(top20)), top20["expected_injuries"],
            left=top20["expected_fatalities"],
            color=bar_colors, alpha=0.4, edgecolor="white", linewidth=0.5,
            label="Expected Injuries")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Expected Annual Casualties  (P(collapse) × occupancy × rate)")
    ax.set_title("Top 20 Bridges by Expected Annual Fatalities\n"
                 "(stacked: fatalities + injuries; rush-hour scenario)")
    for i, row in enumerate(top20.itertuples()):
        w_icon = "[water]" if row.is_water_crossing else "[land]"
        ax.text(row.expected_fatalities + row.expected_injuries + 0.001, i,
                f"  {row.rush_hour_persons:.0f} prs  "
                f"fatal:{row.fatality_rate:.0%}  inj:{row.injury_rate:.0%}  {w_icon}",
                va="center", fontsize=7, color="#333")
    ax.legend(fontsize=9)
    ax.set_xlim(0, top20[["expected_fatalities", "expected_injuries"]].sum(axis=1).max() * 1.6)
    fig.tight_layout()
    path2 = CHARTS_DIR / "11_top_casualty_exposure.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path2)

    return top20


def generate_report(df: pd.DataFrame, top20: pd.DataFrame) -> dict:
    stats = {}
    stats["base_collapse_rate_per_bridge"] = BASE_COLLAPSE_RATE
    stats["p_collapse_risk1_bridge"] = P_COLLAPSE_AT_RISK_1
    stats["total_expected_annual_collapses"] = round(float(df["p_collapse_1yr"].sum()), 2)
    stats["total_expected_annual_affected"]  = round(float(df["expected_affected"].sum()), 3)
    stats["total_expected_annual_fatalities"]= round(float(df["expected_fatalities"].sum()), 3)
    stats["total_expected_annual_injuries"]  = round(float(df["expected_injuries"].sum()), 3)
    stats["water_crossings_total"]           = int(df["is_water_crossing"].sum())
    stats["water_crossings_pct"]             = round(100 * df["is_water_crossing"].mean(), 1)

    for sev in ["Critical", "High", "Medium", "Low"]:
        sub = df[df["severity"] == sev]
        stats[f"{sev.lower()}_median_p_collapse"]    = float(sub["p_collapse_1yr"].median())
        stats[f"{sev.lower()}_median_rush_persons"]  = round(float(sub["rush_hour_persons"].median()), 1)
        stats[f"{sev.lower()}_total_exp_fatalities"] = round(float(sub["expected_fatalities"].sum()), 4)
        stats[f"{sev.lower()}_total_exp_injuries"]   = round(float(sub["expected_injuries"].sum()), 4)

    top5 = df.nlargest(5, "expected_fatalities")[
        ["facility_carried", "state_name", "adt", "rush_hour_persons", "is_water_crossing",
         "fatality_rate", "injury_rate", "p_collapse_1yr",
         "expected_fatalities", "expected_injuries", "expected_affected", "severity"]
    ]
    stats["top5_by_fatalities"] = top5.to_dict(orient="records")

    out = OUTPUTS_DIR / "collapse_exposure_report.json"
    with open(out, "w") as f:
        def default(o):
            if hasattr(o, "item"):
                return o.item()
            if isinstance(o, bool):
                return bool(o)
            return str(o)
        json.dump(stats, f, indent=2, default=default)
    log.info("Saved collapse exposure report -> %s", out)
    return stats


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(OUTPUTS_DIR / "bridges_ranked.csv", low_memory=False)
    log.info("Loaded %d bridges", len(df))

    df = run(df)
    top20 = generate_charts(df)
    stats = generate_report(df, top20)

    log.info("=== COLLAPSE EXPOSURE SUMMARY ===")
    log.info("Expected annual collapses: %.1f", stats["total_expected_annual_collapses"])
    log.info("Expected annual fatalities: %.2f", stats["total_expected_annual_fatalities"])
    log.info("Expected annual injuries:   %.2f", stats["total_expected_annual_injuries"])
    log.info("Expected annual affected:   %.2f", stats["total_expected_annual_affected"])

    # Save enriched top-1000 CSV
    top1000 = df.nlargest(1000, "risk_score")[[
        "severity_rank", "facility_carried", "features_intersected", "state_name",
        "adt", "lanes_on", "structure_len_m", "max_span_len_m",
        "risk_score", "severity", "p_collapse_1yr",
        "rush_hour_persons", "is_water_crossing",
        "fatality_rate", "injury_rate", "unharmed_rate",
        "expected_fatalities", "expected_injuries", "expected_affected",
        "bridge_condition_category", "bridge_age", "last_inspection"
    ]]
    top1000.to_csv(OUTPUTS_DIR / "bridges_top1000_collapse_exposure.csv", index=False)
    log.info("Saved top 1000 collapse exposure CSV")

    return df, stats


if __name__ == "__main__":
    main()
