"""Global target-date split definitions and split-manifest generation.

The implementation splits by prediction target date rather than randomly
splitting overlapping windows (D-012).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    train_start: str
    train_end: str     # e.g. "2023-12-31"
    val_start: str     # e.g. "2024-01-01"
    val_end: str       # e.g. "2024-12-31"
    test_start: str    # e.g. "2025-01-01"
    test_end: str | None = None  # None → latest available


def _normalize_date(val) -> str:
    """Convert any date-like value to YYYYMMDD string."""
    if isinstance(val, str):
        return val.replace("-", "")[:8]
    if hasattr(val, "strftime"):
        return val.strftime("%Y%m%d")
    return str(val).replace("-", "")[:8]


def assign_split(trade_date: str, config: SplitConfig) -> str:
    """Return ``'train'``, ``'val'``, or ``'test'`` based on *trade_date*."""
    td = _normalize_date(trade_date)
    if _normalize_date(config.train_start) <= td <= _normalize_date(config.train_end):
        return "train"
    if _normalize_date(config.val_start) <= td <= _normalize_date(config.val_end):
        return "val"
    # test: from test_start to config.test_end (or infinity)
    if config.test_end is None:
        if td >= _normalize_date(config.test_start):
            return "test"
    else:
        if _normalize_date(config.test_start) <= td <= _normalize_date(config.test_end):
            return "test"
    return "excluded"


def build_split_manifest(
    processed_dir: str,
    calendar_df: pd.DataFrame,
    config: SplitConfig,
    output_path: str,
) -> dict:
    """Build a per-split list of valid ``(ts_code, target_date)`` pairs.

    For each stock parquet file, read its trade dates and assign a split
    label.  Only trading days (as per *calendar_df*) are considered.

    Returns a dict with keys ``'train'``, ``'val'``, ``'test'`` mapping to
    lists of ``[ts_code, trade_date]`` pairs.
    """
    processed_dir = Path(processed_dir)
    trading_dates = {_normalize_date(d) for d in calendar_df[calendar_df["is_open"] == 1]["cal_date"]}

    manifest: dict[str, list] = {"train": [], "val": [], "test": []}
    stock_count = 0

    for fpath in sorted(processed_dir.glob("*.parquet")):
        ts_code = fpath.stem
        df = pd.read_parquet(fpath, columns=["trade_date"])
        stock_count += 1

        for td in df["trade_date"]:
            td_str = _normalize_date(td)
            if td_str not in trading_dates:
                continue
            split = assign_split(td_str, config)
            if split in manifest:
                manifest[split].append([ts_code, td_str])

    # Write to disk
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f)

    total = sum(len(v) for v in manifest.values())
    logger.info(
        "Split manifest written to %s: train=%d, val=%d, test=%d (total=%d, stocks=%d)",
        output_path,
        len(manifest["train"]),
        len(manifest["val"]),
        len(manifest["test"]),
        total,
        stock_count,
    )
    return manifest
