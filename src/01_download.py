#!/usr/bin/env python3
"""
Step 1: Download 2024 NBI highway bridge data (single CSV file, all states).
Downloads from FHWA and caches locally.
"""
import os
import sys
import zipfile
import hashlib
import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
DISCLAIMER_URL = "https://www.fhwa.dot.gov/bridge/nbi/disclaim.cfm?nbiYear=2024hwybronefiledel&nbiZip=zip"
DIRECT_ZIP_URL = "https://www.fhwa.dot.gov/bridge/nbi/2024hwybronefiledel.zip"
ZIP_PATH = RAW_DIR / "nbi2024.zip"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; NBI-Research/1.0; "
        "+https://github.com/bridges-risk-analysis)"
    ),
    "Referer": DISCLAIMER_URL,
}


def download_nbi(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists() and not force:
        log.info("NBI zip already present at %s — skipping download.", ZIP_PATH)
        return ZIP_PATH

    log.info("Downloading NBI 2024 data from %s", DIRECT_ZIP_URL)
    with requests.get(DIRECT_ZIP_URL, headers=HEADERS, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(ZIP_PATH, "wb") as fout:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                fout.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(f"\r  {downloaded/1e6:.1f} / {total/1e6:.1f} MB  ({pct:.0f}%)", end="", flush=True)
        print()
    log.info("Downloaded %.1f MB to %s", ZIP_PATH.stat().st_size / 1e6, ZIP_PATH)
    return ZIP_PATH


def extract_nbi(zip_path: Path, force: bool = False) -> list[Path]:
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        log.info("Zip contains: %s", names)
        for name in names:
            dest = RAW_DIR / name
            if dest.exists() and not force:
                log.info("Already extracted: %s", dest)
            else:
                log.info("Extracting %s -> %s", name, dest)
                zf.extract(name, RAW_DIR)
            extracted.append(dest)
    return extracted


def main(force: bool = False):
    zip_path = download_nbi(force=force)
    files = extract_nbi(zip_path, force=force)
    log.info("Ready: %s", [str(f) for f in files])
    return files


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    main(force=force_flag)
