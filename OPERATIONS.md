# Operations Guide — NBI Bridge Risk Analysis

## Overview

This pipeline downloads the full US National Bridge Inventory (NBI), engineers risk features, trains a machine learning model, ranks all 623k+ bridges by risk severity, and generates an interactive map.

**Total runtime:** ~15–20 minutes (first run, including download)  
**Disk space:** ~1 GB (raw + processed data + outputs)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (downloads data automatically)
bash run_pipeline.sh

# 3. Open the map
open outputs/bridge_risk_map.html   # macOS
xdg-open outputs/bridge_risk_map.html  # Linux
```

---

## Step-by-Step

Each step can be run independently:

| Script | Description | Input | Output |
|---|---|---|---|
| `src/01_download.py` | Download NBI 2024 zip from FHWA | — | `data/raw/nbi2024.zip` |
| `src/02_parse.py` | Parse CSV, decode lat/lon, clean | `data/raw/*.txt` | `data/processed/nbi_clean.parquet` |
| `src/03_features.py` | Engineer risk features | `nbi_clean.parquet` | `nbi_features.parquet` |
| `src/04_model.py` | Train GBT model, score bridges | `nbi_features.parquet` | `nbi_scored.parquet`, `gbt_model.pkl` |
| `src/05_rank.py` | Sort by risk, assign severity tiers | `nbi_scored.parquet` | `bridges_ranked.csv`, `state_summary.csv` |
| `src/06_map.py` | Generate interactive HTML map | `bridges_map_data.parquet` | `bridge_risk_map.html` |
| `src/07_analysis.py` | Statistical analysis + charts | `bridges_ranked.csv` | `outputs/charts/01–09_*.png` |
| `src/08_heatmap.py` | State-level CONUS choropleth | `state_summary.csv` | `outputs/charts/09_conus_state_heatmap.png` |
| `src/09_collapse.py` | Collapse probability + casualty exposure | `bridges_map_data.parquet` | `charts/10–11_*.png`, `collapse_exposure_report.json`, `bridges_top1000_collapse_exposure.csv` |
| `src/10_export.py` | Package publish-ready snapshot | All outputs | `export/` directory |

### Run individual steps

```bash
python3 src/01_download.py           # add --force to re-download
python3 src/02_parse.py
python3 src/03_features.py
python3 src/04_model.py
python3 src/05_rank.py
python3 src/06_map.py
python3 src/07_analysis.py
python3 src/08_heatmap.py
python3 src/09_collapse.py
python3 src/10_export.py             # package export/ for publication
```

### Pipeline options

```bash
bash run_pipeline.sh                 # full run
bash run_pipeline.sh --skip-download # skip download (data already present)
bash run_pipeline.sh --force-download # force re-download
```

---

## Output Files

| File | Size (approx) | Description |
|---|---|---|
| `outputs/bridges_ranked.csv` | ~150 MB | All 622k+ bridges, sorted by risk_score descending |
| `outputs/bridges_top1000_collapse_exposure.csv` | ~1 MB | Top 1000 bridges with collapse probability and casualty exposure |
| `outputs/state_summary.csv` | ~5 KB | Per-state risk statistics |
| `outputs/collapse_exposure_report.json` | ~3 KB | Aggregate collapse/casualty statistics |
| `outputs/bridge_risk_map.html` | ~40 MB | Self-contained interactive map (Critical + High bridges) |
| `outputs/charts/` | ~10 MB | 11 statistical charts (PNG) |
| `outputs/model_report.json` | ~1 KB | ML model CV metrics |
| `outputs/pipeline_YYYYMMDD_HHMMSS.log` | ~50 KB | Pipeline execution log |
| `data/processed/gbt_model.pkl` | ~5 MB | Trained GBT model (sklearn pickle) |

---

## Columns in bridges_ranked.csv

| Column | Description |
|---|---|
| `state_code` | 2-digit FIPS state code |
| `state_name` | State name |
| `structure_number` | Unique NBI bridge identifier |
| `facility_carried` | Road/highway carried by bridge |
| `features_intersected` | What the bridge crosses |
| `location` | Text description of location |
| `county_code` | 3-digit county FIPS code |
| `lat`, `lon` | Decimal degree coordinates |
| `year_built` | Year of original construction |
| `bridge_age` | Age in years (2024 − year_built) |
| `adt` | Average daily traffic (vehicles/day) |
| `pct_truck` | Percentage truck traffic |
| `deck_condition` | NBI deck rating 0–9 |
| `superstr_condition` | NBI superstructure rating 0–9 |
| `substr_condition` | NBI substructure rating 0–9 |
| `culvert_condition` | NBI culvert rating 0–9 (if applicable) |
| `min_struct_condition` | Minimum of the above condition codes |
| `bridge_condition_category` | G=Good / F=Fair / P=Poor (per 23 CFR 490) |
| `structural_eval` | NBI Item 67 appraisal rating |
| `scour_critical` | NBI Item 113 scour criticality |
| `bridge_posting` | Load posting status |
| `open_closed_posted` | Open / Posted / Closed status |
| `last_inspection` | NBI inspection date (MMYY format) |
| `structure_len_m` | Bridge length in meters |
| `max_span_len_m` | Maximum span length in meters |
| `lanes_on` | Number of traffic lanes on bridge |
| `sdi_score` | Rule-based Structural Deficiency Index [0–1] |
| `ml_risk_proba` | ML model probability of poor condition [0–1] |
| `risk_score` | Composite risk score = 0.5×SDI + 0.5×ML [0–1] |
| `severity` | Critical / High / Medium / Low |
| `severity_rank` | National rank (1 = highest risk) |

---

## Tests

```bash
python3 -m pytest tests/ -v
```

All 9 tests should pass. Tests cover:
- SDI score range validation
- Feature engineering correctness
- Severity tier boundaries
- Output CSV structure and counts
- Model AUC (> 0.70)
- Map file size
- State summary completeness

---

## Updating to a New NBI Year

1. Find new year URL at https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm
2. Update `DIRECT_ZIP_URL` in `src/01_download.py`
3. Update `CURRENT_YEAR` in `src/03_features.py`
4. Run: `bash run_pipeline.sh --force-download`

---

## System Requirements

- Python 3.11+
- ~1 GB disk space
- ~8 GB RAM recommended (GBT training on 622k rows)
- Internet connection for first run

---

## Troubleshooting

**Download fails:** FHWA site requires a Referer header (already set). If still failing, download manually from https://www.fhwa.dot.gov/bridge/nbi/ascii2024.cfm and place in `data/raw/`.

**Memory errors in step 4:** Reduce `n_estimators` in `src/04_model.py` to 50.

**Map loads slowly:** The default map renders 50,000 highest-risk bridges. Reduce `MAX_MARKERS` in `src/06_map.py` for faster loading.

**Parser warnings about bad lines:** Normal — ~100 rows have malformed quoting and are skipped.

---

## Publishing to GitHub

### What to include

`bridge_risk_map.html` is ~40 MB — well within GitHub's 100 MB per-file limit, so it is committed directly. The only files excluded are the large CSVs (~150 MB) and raw/processed data.

`.gitignore` (already created) excludes:
```
data/raw/
data/processed/
outputs/bridges_ranked.csv
outputs/bridges_top1000_collapse_exposure.csv
outputs/*.gz
```

Committed to the repository:
- `src/` — all 10 pipeline scripts
- `README.md`, `DETAILS.md`, `OPERATIONS.md`, `.gitignore`
- `requirements.txt`, `run_pipeline.sh`
- `outputs/charts/*.png` — all 11 charts
- `outputs/bridge_risk_map.html` — interactive map (~40 MB, direct download link in README)
- `outputs/state_summary.csv`, `collapse_exposure_report.json`, `model_report.json`

### Step-by-step GitHub publication

```bash
# 0. Run the export script to create a clean publish snapshot
python3 src/10_export.py
# Creates export/ with all intended files; large CSVs are compressed (.gz) for reference

# 1. Create a new GitHub repository at https://github.com/new
#    Set to Public; do NOT initialize with README

# 2. Initialize git in the export directory
cd export/
git init
git add .
git commit -m "Initial release: NBI bridge risk analysis pipeline and results"

# 3. Push to GitHub
git remote add origin https://github.com/petr-salomoun/us-bridge-risk-analysis.git
git branch -M main
git push -u origin main
```

> **Note:** The first push will upload ~42 MB (map + charts). This may take a minute on a slow connection.

### Hosting the full dataset

`bridges_ranked.csv` (~150 MB) exceeds GitHub's soft recommended limit. Options:
- **GitHub Releases** — attach `bridges_ranked.csv.gz` (58 MB) as a release asset:
  ```bash
  gh release create v1.0 outputs/bridges_ranked.csv.gz \
    --title "Full dataset v1.0" \
    --notes "All 622,566 bridges with risk scores (gzip-compressed CSV)"
  ```
- **Zenodo** (https://zenodo.org) — free DOI-assigned academic data hosting
- **Kaggle Datasets** — public datasets with version control

### After publishing

1. Share via:
   - Reddit: r/dataisbeautiful, r/MapPorn, r/civilengineering
   - Hacker News (Show HN)
   - LinkedIn
