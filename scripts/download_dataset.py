#!/usr/bin/env python3
"""
Download the UCI Individual Household Electric Power Consumption dataset.

Usage:
    python scripts/download_dataset.py

If automatic download fails, manual instructions are printed.
"""

from __future__ import annotations

import sys
import zipfile
from io import BytesIO
from pathlib import Path

import requests
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.utils import logger

DATASET_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
ALTERNATIVE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"


def download_file(url: str, dest: Path) -> bool:
    """Attempt to download and extract the dataset."""
    try:
        logger.info("Downloading from %s", url)
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        buffer = BytesIO()
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                buffer.write(chunk)
                pbar.update(len(chunk))

        buffer.seek(0)
        with zipfile.ZipFile(buffer) as zf:
            for name in zf.namelist():
                if name.endswith(".txt"):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with open(dest, "wb") as out:
                        out.write(zf.read(name))
                    logger.info("Extracted to %s", dest)
                    return True

        logger.error("No .txt file found in archive")
        return False

    except Exception as e:
        logger.error("Download failed: %s", e)
        return False


def print_manual_instructions(dest: Path) -> None:
    """Print manual download instructions."""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("\n1. Visit the UCI dataset page:")
    print("   https://archive.ics.uci.edu/dataset/235/")
    print("   individual+household+electric+power+consumption")
    print("\n2. Click 'Download' to get the ZIP file.")
    print("\n3. Extract 'household_power_consumption.txt'")
    print(f"\n4. Place the file at:\n   {dest}")
    print("\n5. Re-run the pipeline:")
    print("   python scripts/run_pipeline.py")
    print("=" * 60)


def main() -> int:
    config = get_config()
    dest = config.RAW_FILE

    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info("Dataset already exists at %s (%.1f MB)", dest, size_mb)
        return 0

    logger.info("Dataset not found. Attempting download...")

    for url in [DATASET_URL, ALTERNATIVE_URL]:
        if download_file(url, dest):
            size_mb = dest.stat().st_size / (1024 * 1024)
            logger.info("Download complete: %.1f MB", size_mb)
            return 0

    print_manual_instructions(dest)
    return 1


if __name__ == "__main__":
    sys.exit(main())
