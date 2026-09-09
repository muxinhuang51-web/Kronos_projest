"""Process raw data: apply adjustments, build split manifest, create dataset manifest."""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "post-train"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from post_train.data.adjustments import process_all_stocks
from post_train.data.splits import SplitConfig, build_split_manifest
from post_train.data.manifest import create_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("process_data")

RAW_ROOT = "data/a_share_daily/raw"
PROCESSED_DIR = "data/a_share_daily/processed"

def main():
    # Step 1: Apply adjustments
    logger.info("=== Step 1: Applying backward adjustments ===")
    process_all_stocks(RAW_ROOT, PROCESSED_DIR)

    # Step 2: Build split manifest
    logger.info("=== Step 2: Building split manifest ===")
    calendar_df = pd.read_parquet(f"{RAW_ROOT}/calendar/calendar.parquet", columns=["cal_date", "is_open"])
    split_cfg = SplitConfig(
        train_start="20160101", train_end="20231231",
        val_start="20240101", val_end="20241231",
        test_start="20250101", test_end=None,
    )
    build_split_manifest(
        PROCESSED_DIR,
        calendar_df,
        split_cfg,
        "data/a_share_daily/manifests/split_manifest.json",
    )

    # Step 3: Create dataset manifest
    logger.info("=== Step 3: Creating dataset manifest ===")
    create_manifest(
        PROCESSED_DIR,
        f"{RAW_ROOT}/calendar/calendar.parquet",
        split_cfg,
        "data/a_share_daily/manifests/dataset_manifest.json",
    )

    logger.info("All processing complete!")

if __name__ == "__main__":
    main()
