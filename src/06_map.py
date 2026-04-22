#!/usr/bin/env python3
"""
Step 6: Generate interactive offline HTML map of bridge risk scores.

Shows all Critical, High, and Medium bridges as separate GeoJSON layers.
Only Critical layer is shown by default.

Markers are bridge-shaped SVG icons:
  - Color = severity tier
  - Size  = proportional to log(ADT+1) (traffic volume)
  - Tooltip includes ADT, risk score, and key bridge details
"""
import json
import logging
import math
from pathlib import Path

import pandas as pd
import folium
from folium import GeoJson, LayerControl
from folium.plugins import Fullscreen, MiniMap
from folium.utilities import JsCode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

SEVERITY_COLORS = {
    "Critical": "#d32f2f",
    "High": "#f57c00",
    "Medium": "#fbc02d",
}

# log(ADT+1) normalization: log(0+1)=0, log(999999+1)≈13.8
# Map to icon size range [8, 32] pixels (half-width of SVG)
ADT_LOG_MIN = 0.0
ADT_LOG_MAX = math.log(1_000_000)
ICON_SIZE_MIN = 6
ICON_SIZE_MAX = 30

CONDITION_LABELS = {
    9: "Excellent", 8: "Very Good", 7: "Good", 6: "Satisfactory",
    5: "Fair", 4: "Poor", 3: "Serious", 2: "Critical",
    1: "Imminent Failure", 0: "Failed/Closed",
}


def fmt(v, digits=0):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:,.{digits}f}"


def adt_to_icon_size(adt):
    """Map ADT to icon half-size in pixels via log^2 scale (faster drop for low traffic)."""
    if adt is None or (isinstance(adt, float) and math.isnan(adt)):
        adt = 0
    log_val = math.log(float(adt) + 1)
    norm = (log_val - ADT_LOG_MIN) / (ADT_LOG_MAX - ADT_LOG_MIN)
    norm = max(0.0, min(1.0, norm))
    norm_scaled = norm ** 2  # quadratic: keeps max, drops smaller values faster
    return ICON_SIZE_MIN + norm_scaled * (ICON_SIZE_MAX - ICON_SIZE_MIN)


def build_feature(row: pd.Series) -> dict:
    """Build a GeoJSON Feature for one bridge."""
    sev = row.get("severity", "Low")
    color = SEVERITY_COLORS.get(sev, "#888")
    risk = float(row.get("risk_score", 0))
    rank = int(row.get("severity_rank", 0))
    facility = str(row.get("facility_carried", "")).strip() or "Unknown"
    features_text = str(row.get("features_intersected", "")).strip() or "—"
    location = str(row.get("location", "")).strip() or "—"
    state = str(row.get("state_name", "")).strip() or "Unknown"
    struct_no = str(row.get("structure_number", "")).strip()
    year = row.get("year_built")
    age = row.get("bridge_age")
    adt = row.get("adt")
    min_cond = row.get("min_struct_condition")
    bc = str(row.get("bridge_condition_category", "")).strip()
    bc_full = {"G": "Good", "F": "Fair", "P": "Poor"}.get(bc, "N/A")
    p_col = row.get("p_collapse_1yr")
    rush = row.get("rush_hour_persons")
    fat_rate = row.get("fatality_rate")
    inj_rate = row.get("injury_rate")
    is_water = row.get("is_water_crossing", False)

    try:
        cond_label = CONDITION_LABELS.get(int(float(min_cond)), "N/A") if min_cond and not math.isnan(float(min_cond)) else "N/A"
    except Exception:
        cond_label = "N/A"

    icon_size = round(adt_to_icon_size(adt), 1)

    p_col_str = f"1 in {int(round(1/p_col)):,}" if p_col and p_col > 0 else "N/A"
    rush_str = f"{rush:.0f}" if rush is not None and not (isinstance(rush, float) and math.isnan(rush)) else "N/A"
    fat_str = f"{fat_rate:.0%}" if fat_rate is not None else "N/A"
    inj_str = f"{inj_rate:.0%}" if inj_rate is not None else "N/A"
    water_str = "Yes (water)" if is_water else "Land"

    popup_html = f"""
<div style="font-family:Arial,sans-serif;min-width:250px;max-width:320px;font-size:12px">
  <div style="background:{color};color:white;padding:7px 10px;border-radius:4px 4px 0 0;margin:-8px -8px 8px -8px">
    <b>{sev} Risk — Rank #{rank:,}</b>
  </div>
  <b>{facility}</b><br>
  Crosses: {features_text}<br>
  {location}, {state}<br>
  Structure: {struct_no}<br>
  <hr style="margin:5px 0;border-color:#eee">
  <b style="color:{color}">Risk Score: {risk:.3f}</b><br>
  NBI Condition: <b>{bc_full}</b><br>
  Built: {fmt(year)} ({fmt(age)} yrs old)<br>
  Daily Traffic: <b>{fmt(adt)} vehicles/day</b><br>
  <hr style="margin:5px 0;border-color:#eee">
  <span style="color:#c62828">&#9888; 1-yr Collapse Prob: <b>{p_col_str}</b></span><br>
  Rush-hour persons on bridge: <b>{rush_str}</b><br>
  Crossing type: {water_str}<br>
  If collapse: fatality rate <b>{fat_str}</b> / injury rate <b>{inj_str}</b><br>
  <div style="margin-top:5px;font-size:10px;color:#aaa">NBI 2024 · FHWA · Icon size &#8733; log(traffic)</div>
</div>"""

    adt_display = fmt(adt) if adt is not None else "N/A"

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [float(row["lon"]), float(row["lat"])],
        },
        "properties": {
            "severity": sev,
            "color": color,
            "risk_score": round(risk, 4),
            "rank": rank,
            "facility": facility[:40],
            "crosses": features_text[:40],
            "state": state,
            "adt": adt_display,
            "year_built": fmt(year),
            "condition": bc_full,
            "collapse_prob": p_col_str,
            "rush_persons": rush_str,
            "icon_size": icon_size,
            "popup_html": popup_html,
        },
    }


def build_map(df: pd.DataFrame) -> folium.Map:
    log.info("Building GeoJSON map for %d bridges ...", len(df))

    m = folium.Map(
        location=[38.5, -96.5],
        zoom_start=5,
        tiles=None,  # No default tile; add manually so CartoDB Positron is last (active)
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)
    folium.TileLayer("CartoDB DarkMatter", name="Dark Map").add_to(m)
    folium.TileLayer("CartoDB Positron", name="Light Map (default)").add_to(m)
    Fullscreen().add_to(m)
    MiniMap(toggle_display=True).add_to(m)

    # Bridge SVG icon via pointToLayer JS (must be JsCode to avoid JSON-encoding)
    # Deck at bottom, arches rising upward = correct orientation
    point_to_layer_js = JsCode("""
function(feature, latlng) {
    var props = feature.properties;
    var s = props.icon_size;
    var c = props.color;
    var w = s * 2;
    var h = s * 1.2;
    // SVG: arch bridge — deck at bottom (y=52), arches curve upward to y=5
    var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="' + w + '" height="' + h + '" viewBox="0 0 100 60">'
            + '<rect x="0" y="42" width="100" height="8" fill="' + c + '" opacity="0.9"/>'
            + '<path d="M5,42 Q25,5 50,42" stroke="' + c + '" stroke-width="7" fill="none" opacity="0.9"/>'
            + '<path d="M50,42 Q75,5 95,42" stroke="' + c + '" stroke-width="7" fill="none" opacity="0.9"/>'
            + '<line x1="25" y1="42" x2="25" y2="18" stroke="' + c + '" stroke-width="3" opacity="0.7"/>'
            + '<line x1="50" y1="42" x2="50" y2="5" stroke="' + c + '" stroke-width="3" opacity="0.7"/>'
            + '<line x1="75" y1="42" x2="75" y2="18" stroke="' + c + '" stroke-width="3" opacity="0.7"/>'
            + '</svg>';
    var icon = L.divIcon({
        html: svg,
        className: '',
        iconSize: [w, h],
        iconAnchor: [w/2, h/2]
    });
    return L.marker(latlng, {icon: icon});
}
""")

    for sev in ["Critical", "High"]:
        subset = df[df["severity"] == sev]
        log.info("  Building GeoJSON for %s: %d bridges", sev, len(subset))

        features = []
        for _, row in subset.iterrows():
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
                if math.isnan(lat) or math.isnan(lon):
                    continue
                features.append(build_feature(row))
            except Exception:
                continue

        geojson_data = {"type": "FeatureCollection", "features": features}
        show = sev == "Critical"

        layer = GeoJson(
            geojson_data,
            name=f"{sev} Bridges ({len(features):,})",
            show=show,
            point_to_layer=point_to_layer_js,
            tooltip=folium.GeoJsonTooltip(
                fields=["facility", "crosses", "state", "risk_score", "adt", "condition", "collapse_prob", "rush_persons"],
                aliases=["Facility:", "Crosses:", "State:", "Risk Score:", "Daily Traffic:", "Condition:", "Collapse P/yr:", "Rush-hr Persons:"],
                localize=True,
                sticky=False,
            ),
            popup=folium.GeoJsonPopup(
                fields=["popup_html"],
                aliases=[""],
                parse_html=True,
                max_width=340,
            ),
        )
        layer.add_to(m)
        log.info("    Added %s layer (%d features, show=%s)", sev, len(features), show)

    # Legend
    legend_html = """
<div style="position:fixed;bottom:50px;right:10px;z-index:1000;background:white;
     padding:12px 16px;border-radius:8px;border:1px solid #ccc;font-family:Arial,sans-serif;
     box-shadow:2px 2px 6px rgba(0,0,0,.2)">
  <b style="font-size:13px">Bridge Risk Severity</b><br><br>
  <div style="display:flex;align-items:center;gap:8px;margin:4px 0">
    <svg width="28" height="17" viewBox="0 0 100 60">
      <rect x="0" y="42" width="100" height="8" fill="#d32f2f"/>
      <path d="M5,42 Q25,5 50,42" stroke="#d32f2f" stroke-width="7" fill="none"/>
      <path d="M50,42 Q75,5 95,42" stroke="#d32f2f" stroke-width="7" fill="none"/>
    </svg>
    <span style="font-size:12px">Critical (risk &ge; 0.75)</span>
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin:4px 0">
    <svg width="28" height="17" viewBox="0 0 100 60">
      <rect x="0" y="42" width="100" height="8" fill="#f57c00"/>
      <path d="M5,42 Q25,5 50,42" stroke="#f57c00" stroke-width="7" fill="none"/>
      <path d="M50,42 Q75,5 95,42" stroke="#f57c00" stroke-width="7" fill="none"/>
    </svg>
    <span style="font-size:12px">High (0.50 – 0.75)</span>
  </div>
  <hr style="margin:8px 0;border-color:#eee">
  <span style="font-size:10px;color:#888">
    Icon size &prop; log(daily traffic)<br>
    Medium/Low bridges omitted<br>
    Toggle layers via control (top-right)
  </span>
</div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    # Title
    n_crit = len(df[df["severity"] == "Critical"])
    n_high = len(df[df["severity"] == "High"])
    title_html = f"""
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:1001;
     background:white;padding:10px 20px;border-radius:8px;border:1px solid #ccc;
     font-family:Arial,sans-serif;box-shadow:2px 2px 6px rgba(0,0,0,.15);text-align:center">
  <b style="font-size:16px;color:#333">US Bridge Risk Analysis</b><br>
  <span style="font-size:11px;color:#777">
    <span style="color:#d32f2f">&#9650;</span> {n_crit:,} Critical &nbsp;
    <span style="color:#f57c00">&#9650;</span> {n_high:,} High &nbsp;&middot;&nbsp; NBI 2024 &middot; FHWA
    &nbsp;&middot;&nbsp; Icon size &prop; log(traffic)
  </span>
</div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    LayerControl(collapsed=False).add_to(m)
    return m


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED_DIR / "bridges_map_data.parquet")
    log.info("Loaded %d bridges", len(df))

    df = df[df["severity"].isin(["Critical", "High"])].copy()
    df = df.sort_values("risk_score", ascending=False)
    log.info("After filtering Low: %d bridges to map", len(df))

    m = build_map(df)
    out = OUTPUTS_DIR / "bridge_risk_map.html"
    m.save(str(out))
    size_mb = out.stat().st_size / 1e6
    log.info("Saved interactive map -> %s (%.1f MB)", out, size_mb)
    return out


if __name__ == "__main__":
    main()
