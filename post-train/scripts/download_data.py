"""Standalone data download script for A-share daily data."""

import logging
import os
import sys
from pathlib import Path

# Ensure post_train is importable
_Kronos_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_Kronos_ROOT / "post-train"))
sys.path.insert(0, str(_Kronos_ROOT))

from post_train.data.download import AShareDownloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("download")

TOKEN = os.environ.get("TUSHARE_TOKEN", "")
if not TOKEN:
    raise RuntimeError("TUSHARE_TOKEN environment variable not set.")

RAW_ROOT = "data/a_share_daily/raw"
START = "20160101"
END = "20260731"

def main():
    downloader = AShareDownloader(token=TOKEN, raw_root=RAW_ROOT)
    result = downloader.download_all(start=START, end=END)
    logger.info("Download complete: %s", result)

if __name__ == "__main__":
    main()
