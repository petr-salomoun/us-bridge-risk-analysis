#!/usr/bin/env bash
# run_pipeline.sh — Full NBI bridge risk analysis pipeline
# Usage: bash run_pipeline.sh [--force-download] [--skip-download]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/src"
LOG_DIR="$SCRIPT_DIR/outputs"
mkdir -p "$LOG_DIR"

FORCE_DL=""
SKIP_DL=0

for arg in "$@"; do
  case $arg in
    --force-download) FORCE_DL="--force" ;;
    --skip-download)  SKIP_DL=1 ;;
  esac
done

TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/pipeline_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "  NBI Bridge Risk Analysis Pipeline"
echo "  $(date)"
echo "============================================================"
echo ""

run_step() {
  local step="$1"
  local script="$2"
  shift 2
  echo "──────────────────────────────────────────"
  echo "  STEP $step: $script"
  echo "──────────────────────────────────────────"
  python3 "$SRC/$script" "$@"
  echo "  ✓ Step $step complete"
  echo ""
}

if [[ $SKIP_DL -eq 0 ]]; then
  run_step 1 "01_download.py" $FORCE_DL
else
  echo "  STEP 1: Skipping download (--skip-download)"
fi

run_step 2 "02_parse.py"
run_step 3 "03_features.py"
run_step 4 "04_model.py"
run_step 5 "05_rank.py"
run_step 6 "06_map.py"
run_step 7 "07_analysis.py"

echo "============================================================"
echo "  Pipeline complete."
echo ""
echo "  Outputs:"
echo "    outputs/bridges_ranked.csv         — All bridges by risk"
echo "    outputs/bridges_critical_top1000.csv — Top 1000 critical"
echo "    outputs/state_summary.csv          — Per-state statistics"
echo "    outputs/bridge_risk_map.html       — Interactive map
    outputs/charts/                    — Statistical charts (8 PNG files)
    outputs/statistics.json            — Key statistics"
echo "    outputs/model_report.json          — ML performance"
echo "    $LOG  — This log"
echo "============================================================"
