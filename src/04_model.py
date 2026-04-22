#!/usr/bin/env python3
"""
Step 4: Train risk model and generate risk scores.

Two models:
  A. Rule-based Structural Deficiency Index (SDI) — fully deterministic
  B. GradientBoostingClassifier (sklearn) — predicts P(poor condition)

Final risk_score = 0.5 * SDI + 0.5 * ml_proba
"""
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

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

# ML model uses only features that don't directly encode condition ratings
# (to avoid circular prediction since IS_POOR is derived from condition codes).
# Structural/geometric/traffic/administrative features only.
ML_FEATURE_COLS = [
    "AGE_NORM",
    "ADT_LOG_NORM",
    "ADT_AGE_STRESS",
    "TRUCK_NORM",
    "TRUCK_LOAD_STRESS",
    "SCOUR_RISK",
    "LOAD_POSTED",
    "WATERWAY_RISK",
    "FRACTURE_CRITICAL",
    "DESIGN_LOAD_RISK",
    "INSPECT_RISK",
    "IS_CLOSED",
]

# Rule-based SDI weights — structurally-informed (sum to 1.0)
# Rationale:
#   COND_RISK (composite weighted 0.45 sub + 0.35 super + 0.20 deck):
#     primary indicator of current structural state — highest weight
#   STRUCT_EVAL_RISK: engineer's holistic appraisal — strong corroborating signal
#   SCOUR_RISK: #1 cause of US bridge collapses (FHWA) — high weight;
#     already amplified by substructure condition in feature engineering
#   FRACTURE_CRITICAL: no structural redundancy = catastrophic failure mode
#   DESIGN_LOAD_RISK: reserve capacity deficit relative to current traffic loads
#   LOAD_POSTED: formal declaration of structural inadequacy
#   AGE_NORM: proxy for unobserved deterioration
#   CHANNEL_RISK / WATERWAY_RISK: hydraulic hazard context
#   ADT_AGE_STRESS / TRUCK_LOAD_STRESS: traffic demand vs. structural capacity
#   INSPECT_RISK / IS_CLOSED: operational status indicators
SDI_WEIGHTS = {
    "COND_RISK":           0.30,  # weighted composite (sub>super>deck)
    "STRUCT_EVAL_RISK":    0.18,  # engineer's structural adequacy appraisal
    "SCOUR_RISK":          0.14,  # #1 collapse cause; already × substr interaction
    "FRACTURE_CRITICAL":   0.07,  # single-point-of-failure structural form
    "DESIGN_LOAD_RISK":    0.06,  # load reserve capacity deficit
    "LOAD_POSTED":         0.06,  # formal load restriction = structural admission
    "AGE_NORM":            0.07,  # proxy for unobserved time-dependent deterioration
    "ADT_AGE_STRESS":      0.04,  # demand × age × design-load interaction
    "TRUCK_LOAD_STRESS":   0.03,  # heavy commercial load on under-designed bridge
    "CHANNEL_RISK":        0.02,  # channel instability (scour precursor)
    "WATERWAY_RISK":       0.02,  # hydraulic adequacy
    "INSPECT_RISK":        0.01,  # overdue inspection = undetected deterioration
    "IS_CLOSED":           0.00,  # already closed — risk realized, not predictive
}


def compute_sdi(df: pd.DataFrame) -> pd.Series:
    """Rule-based Structural Deficiency Index [0, 1]."""
    sdi = pd.Series(0.0, index=df.index)
    for col, w in SDI_WEIGHTS.items():
        if col in df.columns:
            sdi += w * df[col].fillna(0)
    return sdi.clip(0, 1)


def train_ml_model(X: pd.DataFrame, y: pd.Series):
    """Train GradientBoosting classifier. Returns fitted model + CV probabilities."""
    log.info("Training GradientBoostingClassifier (n_estimators=100) ...")

    gb_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )),
    ])

    log.info("  Running 3-fold cross-validation ...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_proba = cross_val_predict(gb_pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    auc = roc_auc_score(y, cv_proba)
    ap = average_precision_score(y, cv_proba)
    log.info("  GBT  CV-AUC=%.4f  CV-AP=%.4f", auc, ap)

    # Fit final model on all data
    gb_pipe.fit(X, y)

    report = {
        "gbt_cv_auc": round(float(auc), 4),
        "gbt_cv_ap": round(float(ap), 4),
        "n_samples": int(len(y)),
        "n_poor": int(y.sum()),
        "prevalence": round(float(y.mean()), 4),
    }

    return gb_pipe, cv_proba, report


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROCESSED_DIR / "nbi_features.parquet")
    log.info("Loaded %d bridges", len(df))

    # ── Rule-based SDI ────────────────────────────────────────────────────
    df["SDI"] = compute_sdi(df)

    # ── ML model ──────────────────────────────────────────────────────────
    # Target: IS_POOR (BRIDGE_CONDITION == 'P')
    if "IS_POOR" in df.columns and df["IS_POOR"].sum() > 100:
        X = df[ML_FEATURE_COLS].copy()
        y = df["IS_POOR"].fillna(0).astype(int)

        model, cv_proba, report = train_ml_model(X, y)

        # Save model
        model_path = PROCESSED_DIR / "gbt_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        log.info("Model saved -> %s", model_path)

        df["ML_PROBA"] = cv_proba
    else:
        log.warning("Insufficient poor-condition labels; using SDI only for ML component.")
        df["ML_PROBA"] = df["SDI"]
        report = {"note": "ML not trained due to insufficient labels"}

    # ── Composite risk score ──────────────────────────────────────────────
    df["RISK_SCORE"] = (0.5 * df["SDI"] + 0.5 * df["ML_PROBA"]).clip(0, 1)

    # Save model performance report
    report_path = OUTPUTS_DIR / "model_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Model report -> %s", report_path)

    # Save back to features file with risk scores
    out = PROCESSED_DIR / "nbi_scored.parquet"
    df.to_parquet(out, index=False)
    log.info("Saved scored bridges -> %s (%d rows)", out, len(df))

    return df, report


if __name__ == "__main__":
    main()
