#!/usr/bin/env python3
"""
Generate CONUS state-level risk heatmap as a PNG chart and
embed it in the analysis outputs.
"""
import json
import logging
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
DATA_DIR = Path(__file__).parent.parent / "data"

# FIPS to state abbreviation
FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}

CONUS_SKIP = {"02", "15", "72", "78", "66", "60"}  # AK, HI, territories


def download_geojson():
    """Download US states GeoJSON if not cached."""
    geo_path = DATA_DIR / "us_states.geojson"
    if geo_path.exists():
        return geo_path
    url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    log.info("Downloading US states GeoJSON from %s ...", url)
    urllib.request.urlretrieve(url, geo_path)
    log.info("Saved -> %s", geo_path)
    return geo_path


def build_choropleth(state_df: pd.DataFrame, geo_path: Path):
    """Build CONUS state choropleth — two side-by-side maps."""
    with open(geo_path) as f:
        geo = json.load(f)

    # Normalize state codes
    state_df = state_df.copy()
    state_df["fips"] = state_df["STATE_CODE"].astype(str).str.strip().str.zfill(2)
    state_df["pct_medium_plus"] = 100 * (state_df["critical"] + state_df["high"] + state_df["medium"]) / state_df["total_bridges"]
    state_df["pct_high_plus"] = 100 * (state_df["critical"] + state_df["high"]) / state_df["total_bridges"]

    medium_plus_by_fips = dict(zip(state_df["fips"], state_df["pct_medium_plus"]))
    high_plus_by_fips = dict(zip(state_df["fips"], state_df["pct_high_plus"]))

    cmap = plt.get_cmap("RdYlGn_r")

    panels = [
        (medium_plus_by_fips, 0, 35, "% Medium + High + Critical Risk Bridges"),
        (high_plus_by_fips,  0, 10, "% High + Critical Risk Bridges"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(22, 7))

    for ax, (risk_by_fips, vmin, vmax, cbar_label) in zip(axes, panels):
        ax.set_aspect("equal")
        ax.axis("off")

        for feature in geo["features"]:
            props = feature["properties"]
            state_name = props.get("name", "")
            fips_match = None
            for fips, abbr in FIPS_TO_ABBR.items():
                if state_df[state_df["fips"] == fips]["state_name"].values.__len__() > 0:
                    sn = state_df[state_df["fips"] == fips]["state_name"].values[0]
                    if sn == state_name:
                        fips_match = fips
                        break
            if fips_match is None or fips_match in CONUS_SKIP:
                continue

            risk = risk_by_fips.get(fips_match, (vmin + vmax) / 2)
            color = cmap((risk - vmin) / (vmax - vmin + 1e-9))

            geom = feature["geometry"]
            coords_list = []
            if geom["type"] == "Polygon":
                coords_list = [geom["coordinates"][0]]
            elif geom["type"] == "MultiPolygon":
                coords_list = [c[0] for c in geom["coordinates"]]

            centroid_x, centroid_y = [], []
            for coords in coords_list:
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                centroid_x.extend(xs)
                centroid_y.extend(ys)
                poly = plt.Polygon(list(zip(xs, ys)), closed=True,
                                   facecolor=color, edgecolor="white", linewidth=0.6)
                ax.add_patch(poly)

            if centroid_x and centroid_y:
                cx = np.mean(centroid_x)
                cy = np.mean(centroid_y)
                abbr = FIPS_TO_ABBR.get(fips_match, "")
                pct = risk_by_fips.get(fips_match, 0)
                ax.text(cx, cy, f"{abbr}\n{pct:.1f}%", ha="center", va="center",
                        fontsize=6.5, color="black", fontweight="bold",
                        bbox=dict(facecolor="white", alpha=0.35, edgecolor="none", pad=0.5))

        ax.set_xlim(-130, -65)
        ax.set_ylim(23, 52)

        sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.03, pad=0.02)
        cbar.set_label(cbar_label, fontsize=9)

    axes[0].set_title("Medium + High + Critical Risk\n(% of state bridge inventory)", fontsize=11, pad=10)
    axes[1].set_title("High + Critical Risk Only\n(% of state bridge inventory)", fontsize=11, pad=10)

    fig.suptitle("CONUS Bridge Risk Heatmap by State — 2024 NBI", fontsize=13, y=1.01)
    fig.tight_layout()
    path = CHARTS_DIR / "09_conus_state_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)
    return path


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    state_df = pd.read_csv(OUTPUTS_DIR / "state_summary.csv")
    geo_path = download_geojson()
    build_choropleth(state_df, geo_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
