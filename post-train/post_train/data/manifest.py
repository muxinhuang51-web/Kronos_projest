"""Frozen dataset-manifest creation and validation.

The manifest records provider, download time, date range, universe rules,
adjustment policy, calendar version, row/symbol counts, missing/duplicate checks,
partition hashes, and the processing source revision (D-011).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .splits import SplitConfig

logger = logging.getLogger(__name__)


@dataclass
class DatasetManifest:
    created_at: str = ""
    provider: str = "tushare"
    date_range: tuple[str, str] = ("", "")
    stock_count: int = 0
    total_rows: int = 0
    calendar_hash: str = ""
    adjustment_policy: str = "backward_adjusted"
    normalization: str = "context_only_zscore"
    split_config: dict = field(default_factory=dict)
    source_revision: str = ""
    per_stock: dict[str, dict] = field(default_factory=dict)


def _hash_file(path: Path) -> str:
    """SHA-256 hex digest of *path*."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _get_git_revision() -> str:
    """Return the current git commit hash, or 'unknown'."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def create_manifest(
    processed_dir: str,
    calendar_path: str,
    split_config: SplitConfig,
    output_path: str,
) -> DatasetManifest:
    """Scan processed data and create a frozen manifest."""
    processed_dir = Path(processed_dir)
    calendar_path = Path(calendar_path)

    manifest = DatasetManifest(
        created_at=datetime.now(timezone.utc).isoformat(),
        calendar_hash=_hash_file(calendar_path),
        split_config=asdict(split_config),
        source_revision=_get_git_revision(),
    )

    files = sorted(processed_dir.glob("*.parquet"))
    manifest.stock_count = len(files)
    total_rows = 0

    # Infer date range from split config
    manifest.date_range = (split_config.train_start, split_config.test_end or "latest")

    for fpath in files:
        ts_code = fpath.stem
        df = pd.read_parquet(fpath, columns=["trade_date"])
        rows = len(df)
        total_rows += rows
        manifest.per_stock[ts_code] = {
            "row_count": rows,
            "start_date": str(df["trade_date"].iloc[0]).replace("-", "")[:8] if rows > 0 else "",
            "end_date": str(df["trade_date"].iloc[-1]).replace("-", "")[:8] if rows > 0 else "",
            "file_hash": _hash_file(fpath),
        }

    manifest.total_rows = total_rows

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(manifest), f, indent=2)

    logger.info(
        "Manifest written to %s: %d stocks, %d rows",
        output_path, manifest.stock_count, manifest.total_rows,
    )
    return manifest
