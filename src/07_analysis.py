#!/usr/bin/env python3
"""
Step 7: Statistical analysis and chart generation.

Produces charts saved to outputs/charts/ as PNG files.
"""
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"

SEVERITY_COLORS = {
    "Critical": "#d32f2f",
    "High": "#f57c00",
    "Medium": "#fbc02d",
    "Low": "#388e3c",
}
PALETTE = [SEVERITY_COLORS[s] for s in ["Critical", "High", "Medium", "Low"]]

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def load_data():
    df = pd.read_csv(OUTPUTS_DIR / "bridges_ranked.csv", low_memory=False)
    state_df = pd.read_csv(OUTPUTS_DIR / "state_summary.csv")
    return df, state_df


# ── Chart 1: Severity distribution pie + bar ─────────────────────────────────
def chart_severity_distribution(df, out_dir):
    counts = df["severity"].value_counts().reindex(["Critical", "High", "Medium", "Low"])
    colors = [SEVERITY_COLORS[s] for s in counts.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie
    wedges, texts, autotexts = ax1.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(p/100*counts.sum()):,})",
        startangle=140,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax1.set_title("Bridge Risk Severity Distribution\n(622,566 US Highway Bridges, 2024)", fontsize=11)

    # Bar
    bars = ax2.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    ax2.set_ylabel("Number of Bridges")
    ax2.set_title("Count by Severity Tier")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    for bar, val in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1500,
                 f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    path = out_dir / "01_severity_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)
    return path


# ── Chart 2: Risk score histogram ────────────────────────────────────────────
def chart_risk_histogram(df, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    n, bins, patches = ax.hist(df["risk_score"].dropna(), bins=80, color="#1565c0", alpha=0.8, edgecolor="none")

    # Color by severity — use bin left edge to assign color
    # Bins are evenly spaced from ~0 to ~1; width ≈ 0.0125
    thresholds = [0.0, 0.25, 0.50, 0.75, 1.01]
    tier_colors = ["#388e3c", "#fbc02d", "#f57c00", "#d32f2f"]
    bin_width = bins[1] - bins[0]
    for patch, left in zip(patches, bins[:-1]):
        mid = left + bin_width / 2  # use bin midpoint for color assignment
        for i, (lo, hi) in enumerate(zip(thresholds[:-1], thresholds[1:])):
            if lo <= mid < hi:
                patch.set_facecolor(tier_colors[i])
                break

    ax.axvline(0.25, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0.50, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0.75, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    legend_patches = [
        mpatches.Patch(color=c, label=l)
        for c, l in zip(tier_colors, ["Low (<0.25)", "Medium (0.25–0.50)", "High (0.50–0.75)", "Critical (≥0.75)"])
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

    ax.set_xlabel("Composite Risk Score")
    ax.set_ylabel("Number of Bridges")
    ax.set_title("Distribution of Bridge Risk Scores — US Highway Bridges 2024")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.tight_layout()
    path = out_dir / "02_risk_score_histogram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)
    return path


# ── Chart 3: Age distribution by severity ────────────────────────────────────
def chart_age_by_severity(df, out_dir):
    df2 = df[df["bridge_age"].between(0, 150)].copy()
    fig, ax = plt.subplots(figsize=(10, 5))

    for sev, color in SEVERITY_COLORS.items():
        subset = df2[df2["severity"] == sev]["bridge_age"]
        subset.hist(bins=50, ax=ax, alpha=0.55, label=sev, color=color, density=True)

    ax.set_xlabel("Bridge Age (years)")
    ax.set_ylabel("Density")
    ax.set_title("Age Distribution by Risk Severity")
    ax.legend(fontsize=10)

    fig.tight_layout()
    path = out_dir / "03_age_by_severity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)
    return path


# ── Chart 4: Top 20 states by % poor/critical bridges ────────────────────────
def chart_state_risk(state_df, out_dir):
    state_df = state_df.copy()
    state_df["pct_at_risk"] = 100 * (state_df["critical"] + state_df["high"]) / state_df["total_bridges"]
    state_df["pct_poor"] = 100 * state_df.get("poor_condition", 0) / state_df["total_bridges"]

    top20 = state_df.nlargest(20, "pct_at_risk").sort_values("pct_at_risk")

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(top20["state_name"], top20["pct_at_risk"], color="#e53935", alpha=0.85, label="Critical+High %")
    ax.set_xlabel("% Bridges in Critical or High Risk Tier")
    ax.set_title("Top 20 States: Share of Critical+High Risk Bridges")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    for i, (val, row) in enumerate(zip(top20["pct_at_risk"], top20.itertuples())):
        n = int(row.critical + row.high)
        ax.text(val + 0.05, i, f" {n:,} bridges", va="center", fontsize=8)

    fig.tight_layout()
    path = out_dir / "04_state_risk_ranking.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)
    return path


# ── Chart 5: Condition code distribution (deck/super/sub) ────────────────────
def chart_condition_distribution(df, out_dir):
    conds = {
        "Deck": "deck_condition",
        "Superstructure": "superstr_condition",
        "Substructure": "substr_condition",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

    condition_labels = {
        0: "Failed", 1: "Imminent\nFailure", 2: "Critical", 3: "Serious",
        4: "Poor", 5: "Fair", 6: "Satisfactory", 7: "Good",
        8: "Very Good", 9: "Excellent",
    }

    for ax, (name, col) in zip(axes, conds.items()):
        if col not in df.columns:
            continue
        vals = df[col].dropna().astype(int)
        counts = vals.value_counts().sort_index()
        bar_colors = []
        for v in counts.index:
            if v <= 2:
                bar_colors.append("#d32f2f")
            elif v <= 4:
                bar_colors.append("#f57c00")
            elif v <= 6:
                bar_colors.append("#fbc02d")
            else:
                bar_colors.append("#388e3c")
        ax.bar(counts.index, counts.values, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(0, 10))
        ax.set_xticklabels(
            [f"{i}\n{condition_labels[i]}" for i in range(10)],
            fontsize=6.5, rotation=30, ha="right"
        )
        ax.set_title(f"{name} Condition Ratings")
        ax.set_ylabel("Number of Bridges" if ax == axes[0] else "")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    fig.suptitle("NBI Condition Rating Distributions — US Highway Bridges 2024", y=1.02, fontsize=12)
    fig.tight_layout()
    path = out_dir / "05_condition_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)
    return path


# ── Chart 6: Risk score vs. bridge age scatter (sampled) ─────────────────────
def chart_risk_vs_age(df, out_dir):
    sample = df[df["bridge_age"].between(0, 150)].sample(n=min(30000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))

    severity_order = ["Low", "Medium", "High", "Critical"]
    for sev in severity_order:
        sub = sample[sample["severity"] == sev]
        ax.scatter(sub["bridge_age"], sub["risk_score"],
                   c=SEVERITY_COLORS[sev], alpha=0.35, s=4, label=sev, rasterized=True)

    # Rolling mean
    binned = sample.copy()
    binned["age_bin"] = (binned["bridge_age"] // 5 * 5).astype(int)
    means = binned.groupby("age_bin")["risk_score"].mean()
    ax.plot(means.index + 2.5, means.values, "k-", linewidth=2, label="Mean risk (5yr bins)")

    ax.set_xlabel("Bridge Age (years)")
    ax.set_ylabel("Composite Risk Score")
    ax.set_title("Bridge Risk Score vs. Age (30,000 random sample)")
    ax.legend(markerscale=3, fontsize=9)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    path = out_dir / "06_risk_vs_age_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)
    return path


# ── Chart 7: Decade of construction breakdown ────────────────────────────────
def chart_construction_decade(df, out_dir):
    df2 = df[df["year_built"].between(1880, 2025)].copy()
    df2["decade"] = (df2["year_built"] // 10 * 10).astype(int)

    agg = df2.groupby(["decade", "severity"]).size().unstack(fill_value=0)
    for s in ["Low", "Medium", "High", "Critical"]:
        if s not in agg.columns:
            agg[s] = 0
    agg = agg[["Critical", "High", "Medium", "Low"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(agg))
    for sev, color in [("Critical", "#d32f2f"), ("High", "#f57c00"),
                       ("Medium", "#fbc02d"), ("Low", "#388e3c")]:
        ax.bar(agg.index, agg[sev], bottom=bottom, color=color,
               label=sev, width=8, edgecolor="none")
        bottom += agg[sev].values

    ax.set_xlabel("Decade Built")
    ax.set_ylabel("Number of Bridges")
    ax.set_title("US Bridges by Decade of Construction and Risk Severity")
    ax.legend(loc="upper left", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.tight_layout()
    path = out_dir / "07_construction_decade.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)
    return path


# ── Chart 8: Traffic volume vs risk ──────────────────────────────────────────
def chart_adt_vs_risk(df, out_dir):
    sample = df[df["adt"].between(1, 500_000)].sample(n=min(20000, len(df)), random_state=7)
    fig, ax = plt.subplots(figsize=(10, 5))

    for sev in ["Low", "Medium", "High", "Critical"]:
        sub = sample[sample["severity"] == sev]
        ax.scatter(np.log10(sub["adt"]), sub["risk_score"],
                   c=SEVERITY_COLORS[sev], alpha=0.35, s=4, label=sev, rasterized=True)

    ax.set_xlabel("Average Daily Traffic (log₁₀ scale)")
    ax.set_ylabel("Composite Risk Score")
    ax.set_title("Risk Score vs. Daily Traffic Volume (20k sample)")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["10", "100", "1,000", "10,000", "100,000"])
    ax.legend(markerscale=3, fontsize=9)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    path = out_dir / "08_adt_vs_risk.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)
    return path


def generate_stats_report(df, state_df) -> dict:
    """Compute key statistics for embedding in reports."""
    stats = {}
    stats["total_bridges"] = len(df)
    stats["with_geo"] = df[["lat", "lon"]].dropna().shape[0]
    stats["pct_poor_nbi"] = round(100 * (df.get("bridge_condition_category", df["severity"]) == "P").sum() / len(df), 2)

    # Severity counts
    for sev in ["Critical", "High", "Medium", "Low"]:
        n = (df["severity"] == sev).sum()
        stats[f"n_{sev.lower()}"] = int(n)
        stats[f"pct_{sev.lower()}"] = round(100 * n / len(df), 2)

    # Age stats
    age = df["bridge_age"].dropna()
    stats["median_age"] = round(float(age.median()), 1)
    stats["mean_age"] = round(float(age.mean()), 1)
    stats["pct_over_50"] = round(100 * (age > 50).sum() / len(age), 1)
    stats["pct_over_75"] = round(100 * (age > 75).sum() / len(age), 1)
    stats["oldest_bridge"] = int(df["year_built"].dropna().min())

    # Condition stats — use minimum across available component columns
    cond_cols = [c for c in ["deck_condition", "superstr_condition", "substr_condition", "culvert_condition"]
                 if c in df.columns]
    if cond_cols:
        mc = df[cond_cols].min(axis=1).dropna()
    else:
        mc = pd.Series(dtype=float)
    stats["pct_min_cond_poor"] = round(100 * (mc <= 4).sum() / len(mc), 2)
    stats["pct_min_cond_serious"] = round(100 * (mc <= 3).sum() / len(mc), 2)
    stats["pct_min_cond_critical"] = round(100 * (mc <= 2).sum() / len(mc), 2)

    # Traffic
    adt = df["adt"].dropna()
    stats["total_daily_crossings"] = int(adt.sum())
    high_risk_adt = df[df["severity"].isin(["Critical", "High"])]["adt"].dropna()
    stats["daily_crossings_high_risk"] = int(high_risk_adt.sum())

    # Top state
    top_state = state_df.sort_values("avg_risk", ascending=False).iloc[0]
    stats["riskiest_state"] = str(top_state["state_name"])
    stats["riskiest_state_avg_risk"] = round(float(top_state["avg_risk"]), 4)

    import json
    out = OUTPUTS_DIR / "statistics.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Statistics saved -> %s", out)
    return stats


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    df, state_df = load_data()
    log.info("Generating charts for %d bridges ...", len(df))

    chart_severity_distribution(df, CHARTS_DIR)
    chart_risk_histogram(df, CHARTS_DIR)
    chart_age_by_severity(df, CHARTS_DIR)
    chart_state_risk(state_df, CHARTS_DIR)
    chart_condition_distribution(df, CHARTS_DIR)
    chart_risk_vs_age(df, CHARTS_DIR)
    chart_construction_decade(df, CHARTS_DIR)
    chart_adt_vs_risk(df, CHARTS_DIR)

    stats = generate_stats_report(df, state_df)
    log.info("Key stats: %s", stats)

    log.info("All charts saved to %s", CHARTS_DIR)
    return stats


if __name__ == "__main__":
    main()
