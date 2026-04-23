# Technical Details — NBI Bridge Risk Analysis

## 1. Data Specification

### Source Format
The NBI 2024 delimited file (`2024HwyBridgesDelimitedAllStates.txt`) is a CSV with:
- **Header row:** item codes (e.g. `STATE_CODE_001`, `LAT_016`)
- **Delimiter:** comma
- **Text qualifier:** single quote `'`
- **Records:** 623,217 (all highway bridges submitted by states)
- **Columns:** 123 data fields + 3 computed fields (CAT10, CAT23, CAT29)
- **File size:** 255.8 MB uncompressed

### Parsing Notes
- Approximately 100 rows have malformed quoting (comma inside unquoted field) and are silently skipped via `on_bad_lines='skip'`
- Latitude stored as 8-digit DDMMSSHH (e.g. `38123456` = 38°12'34.56"N)
- Longitude stored as 9-digit DDDMMSSHH (always west; negated after decode)
- 651 bridges with invalid coordinates (out-of-range lat/lon) are excluded
- Final clean dataset: **622,566 bridges**

---

## 2. Feature Engineering Details

### Bridge Age
```
BRIDGE_AGE = 2024 − YEAR_BUILT
AGE_NORM = clip(BRIDGE_AGE, 0, 120) / 120
```
Year built values < 1800 or > 2024 are treated as missing.

### Minimum Structural Condition (MIN_STRUCT_COND)
NBI Items 58 (deck), 59 (superstructure), 60 (substructure), 62 (culvert).  
Condition scale: 9=Excellent → 0=Failed.

For non-culvert bridges, min of items 58/59/60.  
For culvert bridges, min of all four.

```
COND_RISK = (9 − MIN_STRUCT_COND) / 9
```

Matches the CAT23 (lowest rating) field published by FHWA.  
The official "Poor" threshold is MIN_STRUCT_COND ≤ 4.

> **Note:** In the actual SDI computation (see `src/03_features.py`), COND_RISK uses a **collapse-criticality-weighted** composite rather than a simple minimum: substructure (0.45) > superstructure (0.35) > deck (0.20), with a non-linear penalty for ratings ≤ 4. This weighted approach better reflects structural collapse mechanics (foundation failure is more catastrophic than deck degradation). The simple-minimum formula above describes the FHWA classification logic; the SDI uses the weighted version.

### Structural Evaluation Appraisal (STRUCT_EVAL_RISK)
NBI Item 67. Scale 0–9, where:
- 9 = Superior to present desirable criteria
- 4 = Meets minimum tolerable limits
- 2 = Critical (immediate action required)
- 0 = Basically intolerable

```
STRUCT_EVAL_RISK = (9 − STRUCT_EVAL) / 9
```

### Scour Risk Encoding
NBI Item 113 maps to risk probability:

| Code | Meaning | Risk |
|---|---|---|
| 0 | Unknown foundation (not inspected underwater) | 0.40 |
| 1 | Tidal foundation — stable by definition | 0.05 |
| 2 | Well-founded, not scour critical | 0.05 |
| 3 | Unknown foundation — may be scour critical | 0.55 |
| 4 | Unknown foundation — is scour critical | 0.90 |
| 5 | Stable stream, scour critical (countermeasures applied) | 0.75 |
| 6 | Scour critical — countermeasures applied | 0.65 |
| 7 | Bridge closed due to scour | 1.00 |
| 8 | Monitoring program (not yet closed) | 0.80 |
| 9 | Scour critical — unknown if countermeasures applied | 1.00 |

> **Note:** These risk values are further amplified by a scour-substructure interaction term: `SCOUR_RISK = base_scour × (1 + 0.5 × SUBSTR_RISK)`, reflecting that scour attacking a degraded foundation is multiplicative in collapse probability.

### Traffic Stress
```
ADT_LOG_NORM = log(1 + ADT) / log(1 + 200,000)
ADT_AGE_STRESS = ADT_LOG_NORM × AGE_NORM
```

ADT is capped at 200,000 vehicles/day before log transform.

### Load Posting
NBI Item 70 (Bridge Posting):
- 5 = Unrestricted → LOAD_POSTED = 0
- 0–4 = Some restriction → LOAD_POSTED = 1

### Inspection Recency
NBI Item 90 stores inspection date as MMYY (4 characters).  
Years ≤ 30 are treated as 2000+, otherwise 1900+.

```
YEARS_SINCE_INSPECT = 2024 − INSPECT_YEAR
INSPECT_RISK = clip(YEARS_SINCE_INSPECT, 0, 5) / 5
```

---

## 3. Rule-Based SDI

Structural Deficiency Index (SDI) is a weighted linear combination:

| Feature | Weight | Rationale |
|---|---|---|
| COND_RISK | 0.30 | Weighted structural composite (sub 0.45 + super 0.35 + deck 0.20) |
| STRUCT_EVAL_RISK | 0.18 | Engineer's structural adequacy appraisal |
| SCOUR_RISK | 0.14 | #1 collapse cause; already amplified by substructure interaction |
| FRACTURE_CRITICAL | 0.07 | Single-point-of-failure structural form |
| AGE_NORM | 0.07 | Proxy for unobserved time-dependent deterioration |
| DESIGN_LOAD_RISK | 0.06 | Load reserve capacity deficit |
| LOAD_POSTED | 0.06 | Formal load restriction = structural admission |
| ADT_AGE_STRESS | 0.04 | Traffic demand x age x design-load interaction |
| TRUCK_LOAD_STRESS | 0.03 | Heavy commercial load on under-designed bridge |
| CHANNEL_RISK | 0.02 | Channel instability (scour precursor) |
| WATERWAY_RISK | 0.02 | Hydraulic adequacy |
| INSPECT_RISK | 0.01 | Overdue inspection flag |
| IS_CLOSED | 0.00 | Already closed — risk realized, not predictive |

Weights sum to 1.0. Chosen heuristically, consistent with FHWA bridge management guidance (PONTIS/AASHTOWare).

---

## 4. ML Model

### Target Variable
`IS_POOR = 1 if BRIDGE_CONDITION == 'P'` (FHWA CAT10 field, per 23 CFR 490 Subpart D)

Prevalence: **6.76%** (42,057 poor bridges out of 622,566)

### Feature Set (ML only)
Deliberately excludes condition ratings to avoid circular prediction:
- AGE_NORM
- ADT_LOG_NORM
- ADT_AGE_STRESS
- TRUCK_NORM
- TRUCK_LOAD_STRESS
- SCOUR_RISK
- LOAD_POSTED
- WATERWAY_RISK
- FRACTURE_CRITICAL
- DESIGN_LOAD_RISK
- INSPECT_RISK
- IS_CLOSED

This ensures the ML component adds signal beyond what the condition ratings already encode.

### Model Architecture
`GradientBoostingClassifier` (scikit-learn 1.3):
- n_estimators: 100
- max_depth: 4
- learning_rate: 0.1
- subsample: 0.8
- Preprocessing: `SimpleImputer(strategy='median')`

### Validation
3-fold stratified cross-validation (final model after feature engineering improvements):
- **CV AUC-ROC: 0.909**
- **CV Average Precision: 0.644** (vs 0.068 baseline = ~9.5× lift)

AUC of 0.909 is strong for a model that does **not** use condition ratings. This confirms substantial predictive signal from structural age, traffic patterns, scour vulnerability, and load restrictions — independent of inspection-reported condition.

The average precision of 0.644 (vs. 0.068 baseline) indicates ~9.5× better precision-recall tradeoff than random.

### Composite Score
```
RISK_SCORE = 0.5 × SDI + 0.5 × ML_PROBA
```

Blending the rule-based SDI (which leverages actual condition ratings) with the ML probability (which captures structural/operational risk factors independently) produces a more robust composite than either alone.

---

## 5. Severity Tiers

| Tier | Range | Count | Description |
|---|---|---|---|
| Critical | ≥ 0.75 | 3,829 | Immediate review warranted |
| High | 0.50 – 0.75 | 18,413 | Elevated risk, prioritize inspection |
| Medium | 0.25 – 0.50 | 87,684 | Monitor; likely deteriorating |
| Low | < 0.25 | 512,640 | Generally adequate condition |

Thresholds are set at evenly-spaced breakpoints on the [0, 1] risk score scale (0.25, 0.50, 0.75), not data quantiles. The resulting distribution is heavily right-skewed: 82% of bridges fall in the Low tier, reflecting that most bridges are in adequate condition.

---

## 6. Map Implementation

### Technology
- **Folium 0.14+** with Leaflet.js
- `MarkerCluster` plugin for 50k-marker performance
- **Tile layers:** CartoDB Positron (default), OpenStreetMap, CartoDB Dark Matter
- **Extras:** Fullscreen control, MiniMap

### Performance Design
- Cap: 50,000 markers (highest-risk bridges prioritized)
- `CircleMarker` (not full `Marker` icon) for lower DOM overhead
- Critical/High bridges rendered larger (radius 5 vs 3)
- Severity groups implemented as separate `MarkerCluster` layers, togglable

### Popup Data
Each marker popup shows: severity, national rank, facility name, crossing name, location, state, structure number, risk score, NBI condition category, minimum condition code + text label, year built, age, daily traffic, scour code, load posting.

### Offline Operation
The output `bridge_risk_map.html` includes all JavaScript inline. Tile images require internet access (served by CartoDB/OSM CDN). For fully offline use, swap to a local tile server or use the `tiles=None` option with a GeoJSON base layer.

---

## 7. Limitations and Known Issues

### Data Quality
- NBI condition ratings are self-reported by state DOTs; inspection quality varies
- Some states under-report or round condition codes
- ~100 rows skipped due to CSV quoting errors (< 0.02% of data)
- 651 bridges excluded for out-of-range coordinates

### Model Validity
- ML model trained and tested on 2024 data only; temporal out-of-sample validation not performed
- Gradient Boosting may be sensitive to feature scaling relative to the SDI component
- No survival analysis performed on historical deterioration rates (data collection for multi-year trend analysis is a planned extension)

### Score Interpretation
- RISK_SCORE is an **ordinal ranking tool**, not an absolute probability of failure
- Two bridges with similar risk scores may have very different underlying failure modes
- Critical-tier bridges have very high ML probabilities AND high SDI — but final ground truth requires on-site engineering inspection

### Geographic Accuracy
- NBI lat/lon reflects bridge center point; precision varies by state (some use 4-second granularity)
- Coordinate decode from DDMMSSHH format may introduce up to ~5m error

---

## 8. Reproducibility

```
Python 3.11.7
pandas 2.2.x
numpy 1.26.x
scikit-learn 1.5.x
folium 0.14.x
xgboost 2.0.x
pyarrow 15.x
```

All random seeds set to 42. Pipeline is deterministic given same input data.

Model AUC may vary by ±0.002 due to stratified fold assignment variation if sklearn version changes.
