#!/usr/bin/env python3
"""
Step 10: Export — prepare a publish-ready snapshot of the project.

Creates an `export/` directory containing:
  - All source code (src/)
  - Documentation (README.md, DETAILS.md, OPERATIONS.md)
  - Infrastructure (requirements.txt, run_pipeline.sh, .gitignore)
  - All charts (outputs/charts/*.png)
  - Small data files (state_summary.csv, collapse_exposure_report.json, model_report.json)
  - Compressed large outputs (bridges_ranked.csv.gz, bridges_top1000_collapse_exposure.csv.gz)

The interactive map (bridge_risk_map.html) is copied uncompressed because browsers
need to open it directly; a compressed version is also produced for distribution.

Usage:
    python3 src/10_export.py [--export-dir path/to/export]
"""
import argparse
import gzip
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    size = dst.stat().st_size
    log.info("  Copied  %-60s  %s", str(dst.relative_to(dst.parents[len(dst.parts) - len(ROOT.parts) - 1])), _fmt_size(size))


def compress_file(src: Path, dst: Path):
    """Gzip-compress src -> dst.gz"""
    dst_gz = Path(str(dst) + ".gz")
    dst_gz.parent.mkdir(parents=True, exist_ok=True)
    chunk = 1 << 20  # 1 MB
    with open(src, "rb") as fin, gzip.open(dst_gz, "wb", compresslevel=6) as fout:
        while True:
            data = fin.read(chunk)
            if not data:
                break
            fout.write(data)
    in_size = src.stat().st_size
    out_size = dst_gz.stat().st_size
    ratio = (1 - out_size / max(in_size, 1)) * 100
    log.info("  Packed  %-60s  %s -> %s (%.0f%% smaller)",
             str(dst_gz.relative_to(dst_gz.parents[len(dst_gz.parts) - len(ROOT.parts) - 1])),
             _fmt_size(in_size), _fmt_size(out_size), ratio)
    return dst_gz


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="Export publish-ready snapshot")
    parser.add_argument("--export-dir", default=str(ROOT / "export"),
                        help="Destination directory (default: export/)")
    parser.add_argument("--no-compress", action="store_true",
                        help="Skip compression of large files (faster, larger output)")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    if export_dir.exists():
        log.info("Removing existing export directory: %s", export_dir)
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    log.info("Export directory: %s", export_dir)

    # ── Source code ────────────────────────────────────────────────────────────
    log.info("=== Source code ===")
    src_dir = ROOT / "src"
    for py in sorted(src_dir.glob("*.py")):
        copy_file(py, export_dir / "src" / py.name)

    # ── Documentation ──────────────────────────────────────────────────────────
    log.info("=== Documentation ===")
    for doc in ["README.md", "DETAILS.md", "OPERATIONS.md"]:
        f = ROOT / doc
        if f.exists():
            copy_file(f, export_dir / doc)
        else:
            log.warning("  Missing: %s", doc)

    # ── Infrastructure ─────────────────────────────────────────────────────────
    log.info("=== Infrastructure ===")
    for inf in ["requirements.txt", "run_pipeline.sh", ".gitignore"]:
        f = ROOT / inf
        if f.exists():
            copy_file(f, export_dir / inf)
        else:
            log.warning("  Missing: %s", inf)

    # ── Charts ─────────────────────────────────────────────────────────────────
    log.info("=== Charts ===")
    charts_src = ROOT / "outputs" / "charts"
    for png in sorted(charts_src.glob("*.png")):
        copy_file(png, export_dir / "outputs" / "charts" / png.name)

    # ── Small data files ───────────────────────────────────────────────────────
    log.info("=== Small data files ===")
    small_files = [
        ROOT / "outputs" / "state_summary.csv",
        ROOT / "outputs" / "collapse_exposure_report.json",
        ROOT / "outputs" / "model_report.json",
        ROOT / "outputs" / "statistics.json",
    ]
    for f in small_files:
        if f.exists():
            copy_file(f, export_dir / "outputs" / f.name)
        else:
            log.warning("  Missing: %s", f.name)

    # ── Large output files (compressed) ────────────────────────────────────────
    log.info("=== Large outputs (will be compressed) ===")
    large_files = [
        ROOT / "outputs" / "bridges_ranked.csv",
        ROOT / "outputs" / "bridges_top1000_collapse_exposure.csv",
    ]
    for f in large_files:
        if not f.exists():
            log.warning("  Missing: %s", f.name)
            continue
        dst = export_dir / "outputs" / f.name
        if args.no_compress:
            copy_file(f, dst)
        else:
            compress_file(f, dst)

    # ── Interactive map ─────────────────────────────────────────────────────────
    log.info("=== Interactive map ===")
    map_src = ROOT / "outputs" / "bridge_risk_map.html"
    if map_src.exists():
        # Copy uncompressed (for direct browser use)
        copy_file(map_src, export_dir / "outputs" / map_src.name)
        # Also compress for download/distribution
        if not args.no_compress:
            compress_file(map_src, export_dir / "outputs" / map_src.name)
    else:
        log.warning("  Missing: bridge_risk_map.html (run src/06_map.py first)")

    # ── Summary ────────────────────────────────────────────────────────────────
    total_size = sum(f.stat().st_size for f in export_dir.rglob("*") if f.is_file())
    log.info("=== Export complete ===")
    log.info("Destination : %s", export_dir)
    log.info("Total size  : %s", _fmt_size(total_size))
    log.info("")
    log.info("Next steps:")
    log.info("  1. Review export/ contents")
    log.info("  2. git init in export/ (or copy to a clean git repo)")
    log.info("  3. Follow OPERATIONS.md § 'Publishing to GitHub'")


if __name__ == "__main__":
    main()
